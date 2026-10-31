"""The scale token (depth_cond ScaleTokenCfg) and the windowed training
forward (train.window), on a tiny StreamVGGT on CPU.

    PYTHONPATH=src LD_LIBRARY_PATH=~/.local/lib \\
      ~/.pyenv/versions/3.11.13/envs/metrics/bin/python -c "import sys; \\
      sys.path.append('$HOME/.pyenv/versions/3.11.13/lib/python3.11/site-packages'); \\
      import pytest; sys.exit(pytest.main(['-q', 'tests/test_scale_token.py']))"
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import streamvggt.depth_cond.model as model_module
from eval.streaming_window import stream_depth
from streamvggt.depth_cond.config import (
    DepthCondCfg,
    EncoderType,
    InjectionType,
    LoRACfg,
    MetricCfg,
    ScaleReadout,
    ScaleTokenCfg,
    TrainCondCfg,
)
from streamvggt.depth_cond.model import MetricStreamVGGT
from streamvggt.depth_cond.scale_token import (
    running_log_scale,
    scale_readout_loss,
    withhold_depth_tail,
)
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.models.aggregator import Aggregator
from streamvggt.models.streamvggt import ScaleReadoutHead, StreamVGGT

H, W, DIM = 42, 56, 32


def tiny(scale_token=False, recurrent=False, readout=False, factorised=False, seed=0):
    """StreamVGGT's structure at toy width (24 blocks: the DPT heads read
    layers 4 / 11 / 17 / 23). Special tokens forced apart: at their 1e-6 init
    they are interchangeable and a mis-placed token would go unnoticed."""
    torch.manual_seed(seed)
    m = StreamVGGT.__new__(StreamVGGT)
    nn.Module.__init__(m)
    m.aggregator = Aggregator(
        img_size=W,
        patch_size=14,
        embed_dim=DIM,
        depth=24,
        num_heads=2,
        mlp_ratio=1.0,
        patch_embed="conv",
        scale_token=scale_token,
        scale_recurrent=recurrent,
    )
    dpt = dict(features=16, out_channels=[16, 16, 16, 16], conf_activation="expp1")
    m.depth_head = DPTHead(dim_in=2 * DIM, output_dim=2, activation="exp", **dpt)
    m.point_head = DPTHead(dim_in=2 * DIM, output_dim=4, activation="inv_log", **dpt)
    m.camera_head = CameraHead(dim_in=2 * DIM, trunk_depth=1, num_heads=2)
    m.track_head = None
    m.scale_head = ScaleReadoutHead(2 * DIM, hidden=8) if readout else None
    m.scale_factorised = factorised
    with torch.no_grad():
        agg = m.aggregator
        for p in (agg.camera_token, agg.register_token, agg.scale_token):
            if p is not None:
                p.copy_(torch.randn(p.shape))
    return m.eval()


def frames(S=5, B=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{"img": torch.rand(B, 3, H, W, generator=g)} for _ in range(S)]


def depth_of(out):
    return torch.stack([r["depth"] for r in out.ress], 1)  # [B, S, H, W, 1]


@torch.no_grad()
def test_unbounded_one_frame_path_equals_the_parallel_forward():
    m = tiny()
    fs = frames()
    par = depth_of(m(fs))
    seq = depth_of(m.inference(fs))
    win = depth_of(m.inference(fs, window=len(fs)))
    assert torch.allclose(par, seq, rtol=1e-4, atol=1e-5)
    assert torch.equal(seq, win)  # a window >= the clip never evicts


@torch.no_grad()
def test_one_frame_path_streams_a_batch_of_clips():
    m = tiny(scale_token=True, recurrent=True, readout=True, factorised=True)
    m.aggregator.scale_carry.proj.weight.normal_(std=0.1)
    fs = frames(S=4, B=2)
    both = depth_of(m.inference(fs, window=2))
    for b in range(2):
        one = depth_of(
            m.inference([{"img": f["img"][b : b + 1]} for f in fs], window=2)
        )
        assert torch.allclose(both[b : b + 1], one, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_window_keeps_the_last_frames_only():
    m = tiny()
    fs = frames()
    full = depth_of(m.inference(fs))
    w1 = depth_of(m.inference(fs, window=1))
    # window 1: frame 1 still sees frame 0 (cut after each step), frame 2 not
    assert torch.equal(full[:, :2], w1[:, :2])
    # at toy width (layer scale 0.01) the evicted frame moves frame 2 only
    # slightly, but it does move it
    assert (full[:, 2] - w1[:, 2]).abs().max() > 0


@torch.no_grad()
def test_scale_token_layout_and_parallel_path():
    m = tiny(scale_token=True)
    agg = m.aggregator
    assert agg.scale_token_idx == 5 and agg.patch_start_idx == 6
    fs = frames(S=3)
    par = depth_of(m(fs))
    seq = depth_of(m.inference(fs))
    assert torch.allclose(par, seq, rtol=1e-4, atol=1e-5)
    # the token changes the model: it is attended to
    assert not torch.allclose(par, depth_of(tiny()(fs)))


@torch.no_grad()
def test_recurrence_starts_as_a_no_op_and_needs_the_one_frame_path():
    plain, rec = tiny(scale_token=True), tiny(scale_token=True, recurrent=True)
    rec.load_state_dict(plain.state_dict(), strict=False)  # same weights
    fs = frames()
    assert torch.equal(
        depth_of(plain.inference(fs, window=2)), depth_of(rec.inference(fs, window=2))
    )
    with pytest.raises(ValueError):
        rec(fs)  # the parallel forward cannot carry a state
    with torch.no_grad():
        rec.aggregator.scale_carry.proj.weight.normal_()
    # the carried state now reaches frames whose own cache never held frame 0
    assert not torch.allclose(
        depth_of(plain.inference(fs, window=2))[:, 3:],
        depth_of(rec.inference(fs, window=2))[:, 3:],
    )


@torch.no_grad()
def test_factorised_readout_scales_depth_by_exp_readout():
    plain = tiny(scale_token=True)
    fact = tiny(scale_token=True, readout=True, factorised=True)
    aux = tiny(scale_token=True, readout=True)
    for m in (fact, aux):
        m.load_state_dict(plain.state_dict(), strict=False)
    fs = frames(S=3)
    ref = depth_of(plain(fs))
    assert torch.equal(depth_of(fact(fs)), ref)  # zero-init readout: exp(0)
    for m in (fact, aux):
        m.scale_head.net[-1].bias.fill_(math.log(2.0))
    assert torch.allclose(depth_of(fact(fs)), 2 * ref, rtol=1e-5)
    assert torch.equal(depth_of(aux(fs)), ref)  # aux never touches the output
    s = torch.stack([r["scale_log"] for r in aux(fs).ress], 1)
    assert torch.allclose(s, torch.full_like(s, math.log(2.0)))


@torch.no_grad()
def test_stream_depth_carries_the_token_like_inference():
    m = tiny(scale_token=True, recurrent=True, readout=True, factorised=True)
    m.aggregator.scale_carry.proj.weight.normal_(std=0.1)
    m.scale_head.net[-1].weight.normal_(std=0.1)
    fs = frames()
    ref = depth_of(m.inference(fs, window=2))[0, :, :, :, 0].numpy()
    got = stream_depth(m, fs, window=2, amp=False)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_gradient_reaches_the_carry_through_eviction():
    m = tiny(scale_token=True, recurrent=True, readout=True, factorised=True).train()
    m.scale_head.net[-1].weight.data.normal_(std=0.1)
    fs = frames()
    out = m.inference(fs, window=1)
    out.ress[-1]["depth"].mean().backward()  # frame 4: frames 0-2 long evicted
    assert m.aggregator.scale_carry.proj.weight.grad.abs().sum() > 0


def test_checkpointed_one_frame_path_gives_the_same_gradients():
    grads = []
    for ckpt in (False, True):
        m = tiny(
            scale_token=True, recurrent=True, readout=True, factorised=True
        ).train()
        m.scale_head.net[-1].weight.data.normal_(std=0.1)
        m.aggregator.grad_checkpointing = ckpt
        out = m.inference(frames(), window=2)
        torch.stack([r["depth"] for r in out.ress]).mean().backward()
        grads.append(
            torch.cat(
                [
                    m.aggregator.scale_carry.proj.weight.grad.flatten(),
                    m.aggregator.scale_token.grad.flatten(),
                ]
            )
        )
    assert grads[0].abs().sum() > 0
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-8)


# ---------------------------------------------------------------------------
# MetricStreamVGGT: config, pretrained loading, freezing, windowed forward
# ---------------------------------------------------------------------------
def metric_model(monkeypatch, scale: ScaleTokenCfg, window=None):
    def factory(
        img_size,
        patch_size,
        embed_dim,
        scale_token,
        scale_recurrent,
        scale_readout,
        scale_factorised,
    ):
        return tiny(scale_token, scale_recurrent, scale_readout, scale_factorised)

    monkeypatch.setattr(model_module, "StreamVGGT", factory)
    cfg = MetricCfg(
        depth_cond=DepthCondCfg(
            encoder=EncoderType.IDENTITY, injection=InjectionType.TOKEN
        ),
        lora=LoRACfg(enabled=False),
        train=TrainCondCfg(window=window, grad_checkpoint=False),
        scale_token=scale,
    )
    return MetricStreamVGGT(cfg, img_size=W, patch_size=14, embed_dim=DIM)


def fed_frames(S=4):
    fs = frames(S)
    for f in fs:
        f["sparse_depth"] = torch.full((1, H, W), 2.0)
        f["sparse_depth_mask"] = torch.ones((1, H, W), dtype=torch.bool)
    return fs


def test_config_refuses_untrainable_combinations():
    with pytest.raises(ValueError):  # recurrence needs the windowed forward
        MetricCfg(scale_token=ScaleTokenCfg(enabled=True, recurrent=True)).validate()
    with pytest.raises(ValueError):  # a readout nothing trains
        ScaleTokenCfg(
            enabled=True, readout=ScaleReadout.AUX, readout_weight=0.0
        ).validate()
    with pytest.raises(ValueError):
        TrainCondCfg(window=0).validate()
    with pytest.raises(ValueError):
        TrainCondCfg(depth_dropout_p=1.5).validate()
    MetricCfg(
        scale_token=ScaleTokenCfg(enabled=True, recurrent=True),
        train=TrainCondCfg(window=4),
    ).validate()
    MetricCfg().validate()  # the default: no token, parallel forward


def test_pretrained_load_initialises_the_token_from_a_register(monkeypatch, tmp_path):
    sd = tiny().state_dict()
    path = tmp_path / "pretrained.pth"
    torch.save(sd, path)
    m = metric_model(monkeypatch, ScaleTokenCfg(enabled=True, recurrent=True), window=2)
    m.load_pretrained(str(path))
    agg = m.model.aggregator
    assert torch.equal(agg.scale_token, agg.register_token[:, :, :1])
    torch.save({**sd, "bogus.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError):
        m.load_pretrained(str(path))


def test_freeze_trains_the_scale_modules(monkeypatch):
    m = metric_model(monkeypatch, ScaleTokenCfg(enabled=True, recurrent=True), window=2)
    m.freeze_for_finetune()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    for want in (
        "model.aggregator.scale_token",
        "model.aggregator.scale_carry.proj.weight",
        "model.scale_head.net.3.weight",
    ):
        assert want in names
    assert "model.aggregator.camera_token" not in names


def test_windowed_forward_matches_windowed_inference(monkeypatch):
    m = metric_model(
        monkeypatch, ScaleTokenCfg(enabled=True, recurrent=True), window=2
    ).eval()
    m.model.aggregator.scale_carry.proj.weight.data.normal_(std=0.1)
    with torch.no_grad():
        m.conditioner.token_proj.weight.normal_(std=0.1)  # zero-init: make it count
        fs = fed_frames()
        train_path = depth_of(m(fs))
        conditioning = m._route_conditioning(fs, torch.stack([f["img"] for f in fs], 1))
        feats = conditioning["depth_token_feats"]
        assert feats.abs().sum() > 0
        ref = depth_of(
            m.model.inference(
                fs,
                depth_token_feats_list=[feats[:, s] for s in range(len(fs))],
                window=2,
            )
        )
    assert torch.equal(train_path, ref)  # depth unchanged by skipping heads
    keys = set(m(fs).ress[0])
    assert "depth" in keys and not keys & {
        "camera_pose",
        "pts3d_in_other_view",
        "track",
    }


# ---------------------------------------------------------------------------
# training-side helpers
# ---------------------------------------------------------------------------
def test_readout_target_is_the_running_mean_depth():
    views = [
        {"depthmap": torch.tensor([[[2.0, 0.0]]])},
        {
            "depthmap": torch.tensor([[[4.0, 6.0]]]),
            "valid_mask": torch.tensor([[[True, False]]]),
        },
    ]
    # frame 0 sees only its own 2.0; frame 1 the mean of 2.0 and 4.0
    want = torch.tensor([[math.log(2.0), math.log(3.0)]], dtype=torch.float64)
    assert torch.allclose(running_log_scale(views), want)
    preds = [
        {"scale_log": torch.tensor([math.log(2.0)])},
        {"scale_log": torch.tensor([math.log(3.0) + 1])},
    ]
    assert torch.isclose(scale_readout_loss(preds, views), torch.tensor(0.5))
    # no valid pixel yet: that frame has no target and leaves the mean
    late = [{"depthmap": torch.zeros(1, 1, 2)}, views[1]]
    assert torch.isnan(running_log_scale(late)[0, 0])
    # frame 1 is then the first frame with GT: its target is log 4 alone
    want_late = torch.tensor(1 + math.log(3.0) - math.log(4.0))
    assert torch.isclose(scale_readout_loss(preds, late), want_late)
    assert scale_readout_loss(preds[:1], late[:1]) == 0  # no target anywhere


def test_depth_dropout_keeps_frame_0_and_cuts_a_tail():
    def views():
        return [
            {
                "sparse_depth": torch.ones(2, 3, 3),
                "sparse_depth_mask": torch.ones(2, 3, 3, dtype=torch.bool),
            }
            for _ in range(6)
        ]

    v = views()
    assert withhold_depth_tail(v, 0.0) == 0
    assert all(x["sparse_depth_mask"].all() for x in v)
    torch.manual_seed(0)
    v = views()
    assert withhold_depth_tail(v, 1.0) == 2
    for b in range(2):
        fed = [bool(x["sparse_depth_mask"][b].all()) for x in v]
        assert fed[0]
        start = fed.index(False)  # fed up to `start`, never after
        assert not any(fed[start:]) and all(fed[:start])
        assert all(float(x["sparse_depth"][b].abs().sum()) == 0 for x in v[start:])
