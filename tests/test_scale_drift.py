"""src/scale_drift.py: the CPU pieces of the depth-dropout experiment."""

import numpy as np
import torch

import scale_drift as SD


def test_parse_curve() -> None:
    c = SD.parse_curve("StreamVGGT=base")
    assert (c.label, c.source, c.drop, c.is_base) == ("StreamVGGT", "base", None, True)
    c = SD.parse_curve("Ours (dropout @20)=/runs/abc@20")
    assert (c.label, c.source, c.drop) == ("Ours (dropout @20)", "/runs/abc", 20)
    c = SD.parse_curve("Ours=/runs/abc")
    assert c.drop is None and not c.is_base
    for bad in ("noequals", "=base", "x=", "x=/runs/abc@twenty"):
        try:
            SD.parse_curve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")


def test_withhold_depth_zeroes_depth_and_mask_from_drop() -> None:
    views = [
        {
            "sparse_depth": torch.full((1, 4, 4), 2.0),
            "sparse_depth_mask": torch.ones((1, 4, 4), dtype=torch.bool),
        }
        for _ in range(5)
    ]
    SD.withhold_depth(views, 3)
    assert all(bool(v["sparse_depth_mask"].all()) for v in views[:3])
    assert not any(bool(v["sparse_depth_mask"].any()) for v in views[3:])
    assert all(float(v["sparse_depth"].abs().max()) == 0.0 for v in views[3:])
    SD.withhold_depth(views, None)  # fed throughout: untouched
    assert bool(views[0]["sparse_depth_mask"].all())
    # replaces tensors, never writes into them: a shallow copy per curve
    # (as main does) leaves the original draw intact for the next curve
    full = {
        "sparse_depth": torch.ones(1, 2, 2),
        "sparse_depth_mask": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    copies = [dict(full)]
    SD.withhold_depth(copies, 0)
    assert bool(full["sparse_depth_mask"].all())


def test_frame_scales_and_drift() -> None:
    gt = np.full((4, 3, 3), 2.0)
    scored = np.ones_like(gt, dtype=bool)
    # frames scaled 1.0, 1.0, 1.1, 0.5 of GT
    pred = gt * np.array([1.0, 1.0, 1.1, 0.5])[:, None, None]
    s = SD.frame_scales(pred, gt, scored)
    assert np.allclose(s, [1.0, 1.0, 1 / 1.1, 2.0])
    rel, absol = SD.drift_rows(s, ref_frames=2)
    assert np.allclose(rel, np.abs(np.log(s)))  # reference scale is 1
    assert np.allclose(absol, np.abs(np.log(s)))
    # a model off by a constant factor has absolute error but no drift
    s2 = SD.frame_scales(pred * 3.0, gt, scored)
    rel2, abs2 = SD.drift_rows(s2, ref_frames=2)
    assert np.allclose(rel2, rel) and abs2[0] > 1.0
    # a frame with nothing to score is NaN, not 0
    scored[1] = False
    assert np.isnan(SD.frame_scales(pred, gt, scored)[1])


def test_truncate_cache_keeps_last_frames() -> None:
    from eval.streaming_window import truncate_cache

    k = torch.arange(2 * 3 * 25).reshape(1, 2, 25, 3, 1).float()
    out = truncate_cache([(k, k), None], window=20)
    assert out[1] is None
    assert out[0][0].shape[2] == 20
    assert torch.equal(out[0][0], k[:, :, 5:])


def test_band_ignores_nan_clips() -> None:
    v = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 4.0]])
    med, lo, hi = SD.band(v)
    assert np.allclose(med, [3.0, 3.0]) and lo[0] == 2.0 and hi[0] == 4.0


def test_scale_fed_scales_values_not_mask() -> None:
    views = [
        {
            "sparse_depth": torch.full((1, 2, 2), 2.0),
            "sparse_depth_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        }
        for _ in range(3)
    ]
    out = SD.scale_fed(views, np.array([1.0, 2.0, 0.5]))
    assert [float(v["sparse_depth"].max()) for v in out] == [2.0, 4.0, 1.0]
    assert all(bool(v["sparse_depth_mask"].all()) for v in out)
    assert float(views[1]["sparse_depth"].max()) == 2.0  # originals untouched


def test_response_separates_scale_from_structure() -> None:
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 5, size=(3, 8, 8))
    scored = np.ones_like(base, dtype=bool)
    scaled = base * np.array([1.0, 1.25, 1.25])[:, None, None]
    r, q = SD.response(scaled, base, scored)
    assert np.allclose(r, np.log([1.0, 1.25, 1.25])) and np.allclose(q, 0.0)
    warped = scaled.copy()
    warped[2, :4] *= 1.5  # a structure change on top of the scale
    _, q2 = SD.response(warped, base, scored)
    assert q2[2] > 0.1 and q2[1] == 0.0


def test_sine_fit_recovers_gain_and_lag() -> None:
    t = np.arange(192)
    a, period = 0.2, 16.0
    r = 0.5 * a * np.sin(2 * np.pi * (t - 3) / period)  # half gain, 3-frame lag
    gain, phase = SD.sine_fit(r, period, a, start=20)
    assert abs(gain - 0.5) < 1e-6
    assert abs(SD.circular_lag([phase], period) - 3) < 1e-6
    # a lead is a negative lag, not P - lead
    r_lead = a * np.sin(2 * np.pi * (t + 2) / period)
    assert (
        abs(SD.circular_lag([SD.sine_fit(r_lead, period, a, 20)[1]], period) + 2) < 1e-6
    )
    # a negative amplitude (factor < 1) still reads a perfect follower as gain 1, lag 0
    gain, phase = SD.sine_fit(-a * np.sin(2 * np.pi * t / period), period, -a, 20)
    assert abs(gain - 1) < 1e-6 and abs(phase) < 1e-6


def test_circular_lag_does_not_split_noise_around_zero() -> None:
    # true lag 0: clip phases scattered on both sides of 0 must not average to P/2
    rng = np.random.default_rng(0)
    phases = rng.normal(0.0, 0.2, size=100)
    assert abs(SD.circular_lag(phases, 16.0)) < 0.2


def test_step_stats() -> None:
    tracking = np.concatenate([np.zeros(40), [0.3, 0.5, 0.7, 0.9], np.ones(148)])
    t63, final = SD.step_stats(tracking, 40)
    assert t63 == 2 and final == 1.0
    assert SD.step_stats(np.zeros(192), 40)[0] is None


def test_jitter_stats() -> None:
    rng = np.random.default_rng(2)
    u = 0.1 * rng.standard_normal(192)
    gain, passthrough = SD.jitter_stats(0.4 * u, u, start=20)
    assert abs(gain - 0.4) < 1e-9 and abs(passthrough - 0.4) < 1e-9


def test_relock_frames_is_relative_to_the_control() -> None:
    cut = 90
    control = np.full(180, 0.05)
    control[cut : cut + 5] = 0.2  # the scene cut alone disturbs scale for 5 frames
    err = control.copy()
    err[cut : cut + 12] = 0.4  # with the x2 world: 12 frames
    assert SD.relock_frames(err, control, cut) == 12
    assert SD.relock_frames(np.full(180, 1.0), control, cut) is None


def test_validate_rejects_degenerate_settings() -> None:
    import argparse

    base = dict(
        experiment="step",
        window=20,
        ref_frames=20,
        step_at=40,
        step_factor=1.25,
        sine_amplitude=1.2,
        sine_periods=[8.0],
        jitter_sigma=0.1,
        splice_scale=2.0,
    )
    ours = [SD.parse_curve("Ours=/runs/x")]
    SD.validate(argparse.Namespace(**base), ours, 192)
    for change, curves in (
        ({"step_at": 10}, ours),  # inside the first window
        ({"step_at": 200}, ours),  # past the clip
        ({"step_factor": 1.0}, ours),
        ({"experiment": "sine", "sine_amplitude": 0.8}, ours),
        ({"experiment": "sine", "sine_periods": [2.0]}, ours),
        ({"experiment": "jitter", "jitter_sigma": 0.0}, ours),
        ({}, [SD.parse_curve("StreamVGGT=base")]),  # nothing to perturb
        ({"experiment": "jitter"}, [SD.parse_curve("Ours=/runs/x@20")]),
    ):
        try:
            SD.validate(argparse.Namespace(**{**base, **change}), curves, 192)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {change} {curves}")


def test_median_scaled_absrel_ignores_global_scale() -> None:
    rng = np.random.default_rng(1)
    gt = rng.uniform(1, 5, size=(2, 6, 6))
    scored = np.ones_like(gt, dtype=bool)
    assert np.allclose(SD.median_scaled_absrel(gt * 3.0, gt, scored), 0.0)
