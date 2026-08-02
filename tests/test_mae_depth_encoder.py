"""CPU behavioral verification for the MAE depth-conditioning encoder."""

from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from streamvggt.depth_cond.conditioner import DepthConditioner, dpt_fusion_sizes
from streamvggt.depth_cond.config import (
    DepthCondCfg,
    EncoderType,
    InjectionType,
    MetricCfg,
)
from streamvggt.depth_cond.model import MetricStreamVGGT
import streamvggt.depth_cond.model as model_module


PATCH_SIZE = 14
MAE_DIM = 8


def mae_cfg(injection: InjectionType = InjectionType.TOKEN) -> DepthCondCfg:
    cfg = DepthCondCfg(
        encoder=EncoderType.MAE,
        injection=injection,
        mae_dim=MAE_DIM,
        mae_depth=1,
        mae_num_heads=2,
        mae_decoder_dim=8,
        mae_decoder_depth=1,
        mae_recon_weight=0.1,
    )
    cfg.validate()
    return cfg


def make_target(depth: torch.Tensor, valid: torch.Tensor) -> list[dict]:
    return [
        {"depthmap": depth[:, s], "valid_mask": valid[:, s]}
        for s in range(depth.shape[1])
    ]


class TinyHead(nn.Module):
    def __init__(self, features: int = 6) -> None:
        super().__init__()
        self.scratch = SimpleNamespace(layer1_rn=nn.Conv2d(1, features, 1))
        self.intermediate_layer_idx = [0, 1, 2, 3]


class TinyBackbone(nn.Module):
    """Frozen-backbone stand-in; conditioning modules remain entirely real."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.depth_head = TinyHead()
        self.point_head = TinyHead()
        self.aggregator = nn.Module()
        self.aggregator.grad_checkpointing = False

    @staticmethod
    def _images(views: list[dict]) -> torch.Tensor:
        return torch.stack([view["img"] for view in views], dim=1).sum(dim=2)

    @staticmethod
    def _conditioning_sum(token_feats, head_residuals, shape) -> torch.Tensor:
        value = torch.zeros(shape.shape[:2], device=shape.device, dtype=shape.dtype)
        if token_feats is not None:
            value = value + token_feats.sum(dim=(2, 3))
        if head_residuals is not None:
            for residuals in head_residuals.values():
                if residuals is not None:
                    value = value + sum(r.sum(dim=(2, 3, 4)) for r in residuals)
        return value[:, :, None, None]

    def forward(
        self,
        views,
        _query_points=None,
        patch_tokens=None,
        depth_token_feats=None,
        depth_head_residuals=None,
    ):
        del patch_tokens
        images = self._images(views)
        return images + self._conditioning_sum(
            depth_token_feats, depth_head_residuals, images
        )

    def inference(
        self,
        frames,
        _query_points=None,
        depth_token_feats_list=None,
        depth_head_residuals_list=None,
        frame_times_ms=None,
    ):
        del frame_times_ms
        images = self._images(frames)
        for s in range(images.shape[1]):
            token = (
                None if depth_token_feats_list is None else depth_token_feats_list[s]
            )
            residual = (
                None
                if depth_head_residuals_list is None
                else depth_head_residuals_list[s]
            )
            images[:, s : s + 1] = images[:, s : s + 1] + self._conditioning_sum(
                token, residual, images[:, s : s + 1]
            )
        return images


def make_frames(sequence: int = 2) -> list[dict]:
    frames = []
    for s in range(sequence):
        frames.append(
            {
                "img": torch.full((1, 3, 28, 28), 0.1 * (s + 1)),
                "sparse_depth": torch.full((1, 28, 28), 2.0 + s),
                "sparse_depth_mask": torch.ones(1, 28, 28, dtype=torch.bool),
            }
        )
    return frames


def make_tiny_model(injection: InjectionType) -> MetricStreamVGGT:
    cfg = MetricCfg(depth_cond=mae_cfg(injection))
    cfg.lora.enabled = False
    cfg.train.grad_checkpoint = False
    with mock.patch.object(model_module, "StreamVGGT", TinyBackbone):
        return MetricStreamVGGT(cfg, img_size=28, patch_size=PATCH_SIZE, embed_dim=8)


def test_contract_for_single_and_multi_frame() -> None:
    for sequence in (1, 3):
        depth = torch.ones(2, sequence, 28, 42)
        mask = torch.ones_like(depth, dtype=torch.bool)

        token = DepthConditioner(
            mae_cfg(InjectionType.TOKEN), {"token_dim": 10}, PATCH_SIZE
        )
        token.eval()
        token_out = token(depth, mask)
        assert token_out.shape == (2, sequence, 6, 10)

        sizes = dpt_fusion_sizes(28, 42, PATCH_SIZE)
        head = DepthConditioner(
            mae_cfg(InjectionType.HEAD),
            {"features": 7, "num_scales": 4, "heads": list(mae_cfg().heads)},
            PATCH_SIZE,
        )
        head.eval()
        head_out = head(depth, mask, out_hw_list=sizes)
        for name in ("depth", "point"):
            assert [r.shape for r in head_out[name]] == [
                (2, sequence, 7, h, w) for h, w in sizes
            ]
        print(
            f"[mae-contract] S={sequence}: token {tuple(token_out.shape)}, head {sizes}"
        )


def test_exact_init_noop_and_projection_gradients() -> None:
    frames = make_frames()
    reference = TinyBackbone().eval()(frames)
    for injection in (InjectionType.TOKEN, InjectionType.HEAD):
        torch.manual_seed(4)
        model = make_tiny_model(injection).train()
        output = model(frames)
        difference = (output - reference).abs().max().item()
        print(f"[mae-noop] {injection.value} max abs difference: {difference}")
        assert difference == 0.0
        output.sum().backward()
        if injection is InjectionType.TOKEN:
            grads = [model.conditioner.token_proj.weight.grad]
            label = "token_proj"
        else:
            grads = [
                conv.weight.grad
                for convs in model.conditioner.head_convs.values()
                for conv in convs
            ]
            label = "head_convs"
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        grad_sum = sum(g.abs().sum().item() for g in grads)
        assert grad_sum > 0.0
        print(f"[mae-noop] {label} grad abs sum: {grad_sum:.6f}")


def test_decoder_is_dropped_at_inference_and_parameter_invariant() -> None:
    torch.manual_seed(7)
    model = make_tiny_model(InjectionType.TOKEN).eval()
    frames = make_frames()
    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = model.conditioner.mae_decoder.register_forward_hook(count_call)
    with torch.no_grad():
        before = model.inference(frames).clone()
        assert model.conditioner.pop_aux_loss() is None
        for parameter in model.conditioner.mae_decoder.parameters():
            parameter.copy_(torch.randn_like(parameter))
        after = model.inference(frames).clone()
    handle.remove()
    difference = (after - before).abs().max().item()
    print(f"[mae-inference] decoder calls: {calls}; max abs difference: {difference}")
    assert calls == 0
    assert difference == 0.0
    assert model.conditioner.pop_aux_loss() is None


def test_encoder_and_decoder_both_train() -> None:
    torch.manual_seed(11)
    conditioner = DepthConditioner(mae_cfg(), {"token_dim": 5}, PATCH_SIZE).train()
    assert all(
        parameter.requires_grad for parameter in conditioner.encoder.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in conditioner.mae_decoder.parameters()
    )
    depth = torch.linspace(1.0, 3.0, 28 * 28).reshape(1, 1, 28, 28)
    valid = torch.ones_like(depth, dtype=torch.bool)
    visible = torch.zeros_like(valid)
    visible[..., ::4, ::4] = True
    sparse = depth * visible
    optimizer = torch.optim.Adam(conditioner.parameters(), lr=1e-2)
    before_encoder = [p.detach().clone() for p in conditioner.encoder.parameters()]
    before_decoder = [p.detach().clone() for p in conditioner.mae_decoder.parameters()]
    conditioner.set_reconstruction_target(make_target(depth, valid))
    conditioner(sparse, visible)
    loss = conditioner.pop_aux_loss()
    assert loss is not None
    loss.backward()
    encoder_grads = [p.grad for p in conditioner.encoder.parameters()]
    decoder_grads = [p.grad for p in conditioner.mae_decoder.parameters()]
    assert all(
        g is None or torch.isfinite(g).all() for g in encoder_grads + decoder_grads
    )
    assert any(g is not None and g.abs().sum() > 0 for g in encoder_grads)
    assert any(g is not None and g.abs().sum() > 0 for g in decoder_grads)
    optimizer.step()
    encoder_moved = any(
        not torch.equal(old, new)
        for old, new in zip(before_encoder, conditioner.encoder.parameters())
    )
    decoder_moved = any(
        not torch.equal(old, new)
        for old, new in zip(before_decoder, conditioner.mae_decoder.parameters())
    )
    assert encoder_moved and decoder_moved
    print(
        f"[mae-train] loss={loss.item():.6f}; encoder moved={encoder_moved}; decoder moved={decoder_moved}"
    )


def test_reconstruction_objective_overfits_one_pair() -> None:
    torch.manual_seed(13)
    conditioner = DepthConditioner(mae_cfg(), {"token_dim": 5}, PATCH_SIZE).train()
    yy, xx = torch.meshgrid(torch.arange(14), torch.arange(14), indexing="ij")
    depth = (1.0 + (xx + 2 * yy).float() / 30.0)[None, None]
    valid = torch.ones_like(depth, dtype=torch.bool)
    visible = torch.zeros_like(valid)
    visible[..., 0, 0] = True
    sparse = depth * visible
    optimizer = torch.optim.Adam(conditioner.parameters(), lr=2e-2)
    losses = []
    for _ in range(48):
        optimizer.zero_grad()
        conditioner.set_reconstruction_target(make_target(depth, valid))
        conditioner(sparse, visible)
        loss = conditioner.pop_aux_loss()
        assert loss is not None
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    factor = losses[-1] / losses[0]
    print(f"[mae-overfit] {losses[0]:.6f} -> {losses[-1]:.6f}; factor={factor:.4f}")
    assert factor < 0.5


def test_auxiliary_loss_masking_and_empty_withheld() -> None:
    torch.manual_seed(17)
    conditioner = DepthConditioner(mae_cfg(), {"token_dim": 5}, PATCH_SIZE).train()
    for parameter in conditioner.mae_decoder.parameters():
        nn.init.zeros_(parameter)
    depth = torch.full((1, 1, 14, 14), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    visible = torch.zeros_like(valid)
    valid[..., 0, 0] = False
    visible[..., 0, 1] = True
    sparse = depth * visible
    conditioner.set_reconstruction_target(make_target(depth, valid))
    conditioner(sparse, visible)
    reference = conditioner.pop_aux_loss()
    assert reference is not None

    changed = depth.clone()
    changed[..., 0, 0] = 1000.0
    changed[..., 0, 1] = 0.01
    conditioner.set_reconstruction_target(make_target(changed, valid))
    conditioner(sparse, visible)
    ignored = conditioner.pop_aux_loss()
    assert ignored is not None and torch.equal(reference, ignored)

    conditioner.set_reconstruction_target(make_target(depth, valid))
    conditioner(depth, valid)
    empty = conditioner.pop_aux_loss()
    assert empty is not None and empty.item() == 0.0 and torch.isfinite(empty)
    print(f"[mae-mask] reference={reference.item():.6f}; empty-withheld={empty.item()}")


def test_validation_mode_produces_no_auxiliary_term() -> None:
    conditioner = DepthConditioner(mae_cfg(), {"token_dim": 5}, PATCH_SIZE).eval()
    depth = torch.ones(1, 1, 14, 14)
    valid = torch.ones_like(depth, dtype=torch.bool)
    conditioner.set_reconstruction_target(make_target(depth, valid))
    conditioner(depth, torch.zeros_like(valid))
    assert conditioner.pop_aux_loss() is None
    print("[mae-val] eval forward produced no auxiliary term")


def test_config_rejects_every_malformed_mae_knob() -> None:
    cases = [
        ({"mae_dim": 0}, "mae_dim must be positive, got 0"),
        ({"mae_depth": 0}, "mae_depth must be positive, got 0"),
        ({"mae_num_heads": 0}, "mae_num_heads must be positive, got 0"),
        (
            {"mae_dim": 10, "mae_num_heads": 3},
            "mae_num_heads must divide mae_dim, got mae_num_heads=3, mae_dim=10",
        ),
        ({"mae_decoder_dim": 0}, "mae_decoder_dim must be positive, got 0"),
        ({"mae_decoder_depth": 0}, "mae_decoder_depth must be positive, got 0"),
        (
            {"mae_recon_weight": -0.1},
            "mae_recon_weight must be finite and non-negative, got -0.1",
        ),
    ]
    for changes, message in cases:
        cfg = mae_cfg()
        for name, value in changes.items():
            setattr(cfg, name, value)
        try:
            cfg.validate()
        except ValueError as error:
            assert str(error) == message
        else:
            raise AssertionError(f"expected ValueError: {message}")
        print(f"[mae-config] rejected: {message}")


def main() -> None:
    test_contract_for_single_and_multi_frame()
    test_exact_init_noop_and_projection_gradients()
    test_decoder_is_dropped_at_inference_and_parameter_invariant()
    test_encoder_and_decoder_both_train()
    test_reconstruction_objective_overfits_one_pair()
    test_auxiliary_loss_masking_and_empty_withheld()
    test_validation_mode_produces_no_auxiliary_term()
    test_config_rejects_every_malformed_mae_knob()
    print("MAE DEPTH ENCODER TESTS PASS")


if __name__ == "__main__":
    main()
