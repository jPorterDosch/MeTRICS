"""src/slit_view.py: the CPU pieces of the slit-window view."""

import json

import numpy as np
import pytest

import slit_view as SV


def test_parse_model() -> None:
    m = SV.parse_model("Ours=ckpt:/runs/a:b")
    assert (m.label, m.kind, m.source) == ("Ours", "ckpt", "/runs/a:b")
    for bad in (
        "Ours",
        "Ours=/runs/a",
        "Ours=lora:/runs/a",
        "=arm:vda",
        "a/b=arm:vda",
        "scene=arm:vda",
    ):
        with pytest.raises(ValueError):
            SV.parse_model(bad)


def test_columns_put_time_on_the_horizontal_axis() -> None:
    S, H, W = 5, 4, 6
    arr = np.arange(S * H * W).reshape(S, H, W)
    cols = SV.columns(arr, [0, 3])
    assert cols.shape == (2, H, S)
    # [k, row, t] is frame t's pixel (row, x_k)
    assert cols[1, 2, 4] == arr[4, 2, 3]
    rgb = np.zeros((S, H, W, 3))
    assert SV.columns(rgb, [1]).shape == (1, H, S, 3)
    assert SV.column_px([0.0, 0.5, 1.0], 641) == [0, 320, 640]


def test_align_published_is_the_scored_fit() -> None:
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 5.0, size=(4, 8, 8)).astype(np.float32)
    # an affine disparity of the GT: the per-video fit recovers it exactly
    disp = 3.0 / gt + 0.1
    aligned, abs_rel = SV.align(disp, "disparity", gt, "published", 10.0)
    assert np.allclose(aligned, gt, rtol=1e-4) and abs_rel < 1e-4
    # a depth output with one global scale error: published fits it away,
    # metric shows it
    aligned, abs_rel = SV.align(2.0 * gt, "depth", gt, "published", 10.0)
    assert abs_rel < 1e-4
    raw, abs_rel = SV.align(2.0 * gt, "depth", gt, "metric", 20.0)
    assert np.allclose(raw, 2.0 * gt) and abs(abs_rel - 1.0) < 1e-6
    # metric as score_sequence scores it: fed pixels held out, a non-finite
    # pixel drawn at the floor (an error), not as a hole
    pred = gt.copy()
    pred[0, 0, 0] = np.inf
    held = np.zeros_like(gt, dtype=bool)
    held[0, 0, 0] = True
    drawn, abs_rel = SV.align(pred, "depth", gt, "metric", 10.0, held_out=held)
    assert drawn[0, 0, 0] == pytest.approx(SV.P.DEPTH_FLOOR) and abs_rel < 1e-6
    with pytest.raises(ValueError):
        SV.align(disp, "disparity", gt, "metric", 10.0)


def test_display_range_is_robust_to_outliers() -> None:
    gt = np.full((8, 100), 2.0)
    gt[0, 0] = 0.01  # one near outlier
    valid = np.ones_like(gt, dtype=bool)
    lo, hi = SV.display_range(gt, valid)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    with pytest.raises(ValueError):
        SV.display_range(gt, np.zeros_like(valid))


def _write_scene(d, xs, models, T=12, H=10, W=16):
    rng = np.random.default_rng(1)
    gt = rng.uniform(1.0, 4.0, size=(len(xs), H, T)).astype(np.float32)
    d.mkdir(parents=True)
    np.savez_compressed(
        d / "scene.npz",
        xs=np.array(xs),
        rgb=rng.integers(0, 255, size=(len(xs), H, T, 3), dtype=np.uint8),
        gt=gt,
        valid=gt > 1.5,
        ref=rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8),
        ref_frame=np.array(0),
    )
    for i, (label, meta) in enumerate(models):
        np.savez_compressed(d / f"{label}.npz", pred=gt * (1 + 0.1 * i))
        base = {
            "label": label,
            "protocol": "published",
            "density": 0.05,
            "seed": 42,
            "patch_size": 14,
            "image_size": 518,
            "tae": True,
            "abs_rel": 0.1 * i,
            "xs": xs,
            "written": f"2026-10-23T22:0{i}:00",
        }
        with open(d / f"{label}.json", "w") as f:
            json.dump({**base, **meta}, f)


def test_load_scene_orders_by_extraction_and_rejects_mixtures(tmp_path) -> None:
    d = tmp_path / "scannet_tae_s0"
    _write_scene(d, [3, 8], [("Zed", {}), ("Alpha", {})])
    _, models = SV.load_scene(d)
    assert [m["label"] for m, _ in models] == ["Zed", "Alpha"]  # not alphabetical
    for label, meta in (
        ("B", {"protocol": "metric"}),
        ("C", {"xs": [3, 9]}),
        ("D", {"seed": 7}),
        ("E", {"image_size": 1036}),
    ):
        bad = tmp_path / f"bad_{label}"
        _write_scene(bad, [3, 8], [("A", {}), (label, meta)])
        with pytest.raises(ValueError):
            SV.load_scene(bad)


def test_render_writes_one_figure_per_column(tmp_path) -> None:
    d = tmp_path / "scannet_tae_s0"
    _write_scene(d, [3, 8], [("Ours", {}), ("VDA", {})])
    paths = SV.render_scene(d, tmp_path, px_per_frame=2, error=True, cmap="inferno")
    assert [p.name for p in paths] == ["scannet_tae_s0_x3.png", "scannet_tae_s0_x8.png"]
    assert all(p.stat().st_size > 0 for p in paths)
