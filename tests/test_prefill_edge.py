"""The prefill conv encoder, the teacher edge loss and the uptake / boundary
monitors (streamvggt.depth_cond.prefill / conditioner, streamvggt.loss.
gradient_loss, eval.boundary)."""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from eval.boundary import boundary_f1, fed_pixel_metrics
from streamvggt.depth_cond.conditioner import DepthConditioner, prefill_channels
from streamvggt.depth_cond.config import DepthCondCfg, EncoderType, InjectionType
from streamvggt.depth_cond.prefill import nearest_fill
from streamvggt.loss.depth_train_loss import DepthTrainLoss
from streamvggt.loss.gradient_loss import prealign_scale, teacher_edge_loss
from streamvggt.loss.types import LossConfig


def _random_masks(n: int, h: int, w: int, p: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, h, w, generator=g) < p


def test_nearest_fill_matches_exact_edt() -> None:
    for h, w, p in ((61, 83, 0.03), (40, 40, 0.3), (29, 71, 0.001)):
        mask = _random_masks(2, h, w, p, seed=h * w)
        # unique value per pixel, so the filled value names its source pixel
        values = torch.arange(h * w, dtype=torch.float32).view(1, h, w).expand(2, h, w)
        filled, dist = nearest_fill(values.contiguous(), mask)
        for n in range(2):
            ref = distance_transform_edt(~mask[n].numpy())
            assert np.allclose(dist[n].numpy(), ref, atol=1e-3), (h, w, p, n)
            src = filled[n].long()
            sy, sx = src // w, src % w
            ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            # the source is a measured pixel at exactly the nearest distance
            assert bool(mask[n][sy, sx].all())
            d_src = ((sy - ys) ** 2 + (sx - xs) ** 2).float().sqrt()
            assert torch.allclose(d_src, dist[n], atol=1e-3)
        assert bool((dist[mask] == 0).all())


def test_nearest_fill_empty_frame_is_flagged() -> None:
    mask = torch.zeros(1, 20, 30, dtype=torch.bool)
    filled, dist = nearest_fill(torch.rand(1, 20, 30), mask)
    assert float(filled.abs().max()) == 0.0
    assert float(dist.min()) == 50.0


def test_prefill_channels_layout() -> None:
    x = torch.zeros(1, 2, 2, 28, 42)
    x[0, :, 1, 7, 7] = 1.0  # one measurement per frame
    x[0, :, 0, 7, 7] = 3.0
    out = prefill_channels(x, patch_size=14)
    assert out.shape == (1, 2, 3, 28, 42)
    assert bool((out[..., 0, :, :] == 3.0).all())  # filled everywhere
    assert float(out[0, 0, 2, 7, 7]) == 0.0  # distance 0 at the reading
    assert float(out[0, 0, 2, 7, 21]) > float(out[0, 0, 2, 7, 8])


def test_prefill_conv_token_arm_is_noop_at_init() -> None:
    cfg = DepthCondCfg(
        enabled=True, encoder=EncoderType.PREFILL_CONV, injection=InjectionType.TOKEN
    )
    cfg.validate()
    cond = DepthConditioner(cfg, {"token_dim": 32}, patch_size=14)
    depth = torch.rand(1, 3, 28, 42) * 5 + 0.5
    mask = _random_masks(3, 28, 42, 0.05, seed=1).unsqueeze(0)
    out = cond(depth, mask)
    assert out.shape == (1, 3, 2 * 3, 32)
    assert float(out.abs().max()) == 0.0
    # and after the zero projection is perturbed, the output depends on the hint
    torch.nn.init.normal_(cond.token_proj.weight)
    a = cond(depth, mask)
    b = cond(depth * 2, mask)
    assert not torch.allclose(a, b)


def test_prefill_conv_head_arm_shapes() -> None:
    cfg = DepthCondCfg(
        enabled=True, encoder=EncoderType.PREFILL_CONV, injection=InjectionType.HEAD
    )
    cfg.validate()
    cond = DepthConditioner(
        cfg, {"features": 8, "num_scales": 4, "heads": cfg.heads}, patch_size=14
    )
    depth = torch.rand(1, 2, 28, 42) * 5 + 0.5
    mask = torch.ones(1, 2, 28, 42, dtype=torch.bool)
    out = cond(depth, mask, out_hw_list=[(8, 12), (4, 6), (2, 3), (1, 2)])
    assert [tuple(r.shape) for r in out["depth"]] == [
        (1, 2, 8, 8, 12),
        (1, 2, 8, 4, 6),
        (1, 2, 8, 2, 3),
        (1, 2, 8, 1, 2),
    ]


def _scene(step_at: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    """A depth map with a step edge after column `step_at` and a slope, and
    its validity."""
    y, x = torch.meshgrid(torch.arange(24.0), torch.arange(32.0), indexing="ij")
    depth = 2.0 + 0.01 * x + 1.5 * (x > step_at).float() + 0.02 * y
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[3:5, 3:9] = False
    return depth.unsqueeze(0), valid.unsqueeze(0)


def test_edge_loss_zero_for_rescaled_teacher_with_prealign() -> None:
    depth, valid = _scene()
    teacher = depth * 0.37  # the teacher's scale is arbitrary
    loss = teacher_edge_loss(depth, teacher, depth, valid, prealign=True)
    assert float(loss) < 1e-6
    s = prealign_scale(depth, teacher, valid)
    assert torch.allclose(s, torch.tensor([1 / 0.37]), rtol=1e-4)
    # without the prealign the scale mismatch is charged as edge error
    assert float(teacher_edge_loss(depth, teacher, depth, valid, prealign=False)) > 0.01
    # log space is scale-free on its own
    assert (
        float(
            teacher_edge_loss(
                depth, teacher, depth, valid, prealign=False, log_space=True
            )
        )
        < 1e-6
    )


def test_edge_loss_charges_a_missing_edge_and_respects_the_mask() -> None:
    depth, valid = _scene()
    flat = torch.full_like(depth, 3.0)
    full = teacher_edge_loss(flat, depth, depth, valid)
    assert float(full) > 0.0
    # the edge column is the only place the gradients differ by 1.5 m; masking
    # it out leaves only the slope (0.01 / 0.02 per pixel)
    masked = valid.clone()
    masked[:, :, 15:17] = False
    part = teacher_edge_loss(flat, depth, depth, masked)
    assert float(part) < float(full)
    assert float(part) < 0.03
    assert float(teacher_edge_loss(flat, depth, depth, torch.zeros_like(valid))) == 0.0


def test_depth_train_loss_edge_term_needs_teacher_depth() -> None:
    depth, valid = _scene()
    gts = [{"depthmap": depth, "valid_mask": valid}]
    preds = [{"depth": depth.unsqueeze(-1), "depth_conf": torch.ones_like(depth)}]
    crit = LossConfig(edge_weight=0.5).build()
    assert isinstance(crit, DepthTrainLoss)
    try:
        crit(gts, preds)
    except ValueError as error:
        assert "teacher_depth" in str(error)
    else:
        raise AssertionError("edge term ran without a teacher depth")
    gts[0]["teacher_depth"] = depth * 2
    total, details = crit(gts, preds)
    assert "Ledge" in details and float(details["Ledge"]) < 1e-6
    _, details_off = LossConfig().build()(gts, preds)
    assert "Ledge" not in details_off


def test_loss_config_rejects_bad_edge_weight() -> None:
    for bad in (-1.0, float("nan")):
        try:
            LossConfig(edge_weight=bad)
        except ValueError as error:
            assert "edge_weight" in str(error)
        else:
            raise AssertionError(f"edge_weight={bad} accepted")


def test_boundary_f1_tolerance() -> None:
    depth, valid = _scene()
    assert float(boundary_f1(depth, depth, valid)[0]) == 1.0
    shifted, _ = _scene(step_at=16)  # edge one pixel over
    assert float(boundary_f1(shifted, depth, valid, tolerance=1)[0]) == 1.0
    # both pixels of a pair are marked, so a 1-px shift still overlaps by half
    assert abs(float(boundary_f1(shifted, depth, valid, tolerance=0)[0]) - 0.5) < 1e-6
    far, _ = _scene(step_at=19)
    assert float(boundary_f1(far, depth, valid, tolerance=1)[0]) == 0.0
    flat = torch.full_like(depth, 3.0)
    assert float(boundary_f1(flat, depth, valid)[0]) == 0.0  # no edge predicted
    assert torch.isnan(boundary_f1(depth, flat, valid)[0])  # GT has no edge


def test_fed_pixel_metrics() -> None:
    depth, valid = _scene()
    fed = _random_masks(1, 24, 32, 0.2, seed=3) & valid
    rel, log_s = fed_pixel_metrics(depth * 1.1, depth, fed)
    assert abs(float(rel) - 0.1) < 1e-5
    assert abs(float(log_s) - abs(np.log(1 / 1.1))) < 1e-5
    rel0, log0 = fed_pixel_metrics(depth, depth, torch.zeros_like(fed))
    assert torch.isnan(rel0) and torch.isnan(log0)


def test_edge_loss_dense_teacher_mask_covers_gt_holes() -> None:
    depth, valid = _scene()
    flat = torch.full_like(depth, 3.0)
    # punch the GT out along the edge: with the GT mask the edge is unseen,
    # with the (dense) teacher mask it is charged
    holed = valid.clone()
    holed[:, :, 14:18] = False
    gt_masked = teacher_edge_loss(flat, depth, depth, holed)
    dense = teacher_edge_loss(
        flat, depth, depth, holed, teacher_valid=torch.ones_like(valid)
    )
    # one 1.5 m step over 24 rows among ~1500 pairs adds ~0.024 to the mean
    assert float(dense) > 2 * float(gt_masked)
    # the teacher's sky (teacher_valid False) is left out even with GT there
    sky = torch.ones_like(valid)
    sky[:, :, 14:18] = False
    assert float(teacher_edge_loss(flat, depth, depth, valid, teacher_valid=sky)) < 0.03


def test_edge_term_requires_metric_data_term() -> None:
    try:
        LossConfig(edge_weight=0.5, depth_metric=False).build()
    except ValueError as error:
        assert "metric" in str(error)
    else:
        raise AssertionError("edge term accepted a scale-aligned data term")


def test_manifest_hashes_every_new_knob() -> None:
    import finetune_depth as fd

    m = fd.build_manifest(fd.FinetuneDepthCfg())
    for key in (
        "teacher.model",
        "teacher.res_scale",
        "teacher.chunk",
        "loss.edge_weight",
        "train_dataset.scannet_official_split",
        "val_dataset.scannet_official_split",
    ):
        assert key in m, key
    assert m["teacher.model"] == "none" and m["loss.edge_weight"] == 0.0


def test_edge_term_follows_depth_log_space() -> None:
    depth, valid = _scene()
    gts = [{"depthmap": depth, "valid_mask": valid, "teacher_depth": depth * 0.4}]
    preds = [{"depth": depth.unsqueeze(-1), "depth_conf": torch.ones_like(depth)}]
    # log space: the teacher's scale drops out, so a rescaled teacher costs nothing
    _, d = LossConfig(edge_weight=0.5, depth_log_space=True).build()(gts, preds)
    assert float(d["Ledge"]) < 1e-6
