"""src/scale_drift.py: the CPU pieces of the depth-dropout experiment."""

import numpy as np
import pytest
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
    assert abs(SD.lags_across_periods([period], [phase])[0] - 3) < 1e-6
    # a lead is a negative lag, not P - lead
    r_lead = a * np.sin(2 * np.pi * (t + 2) / period)
    lead = SD.lags_across_periods([period], [SD.sine_fit(r_lead, period, a, 20)[1]])
    assert abs(lead[0] + 2) < 1e-6
    # a negative amplitude (factor < 1) still reads a perfect follower as gain 1, lag 0
    gain, phase = SD.sine_fit(-a * np.sin(2 * np.pi * t / period), period, -a, 20)
    assert abs(gain - 1) < 1e-6 and abs(phase) < 1e-6


def test_mean_response_does_not_split_noise_around_zero() -> None:
    # true lag 0: clip phases scattered on both sides of 0 must not average to pi
    rng = np.random.default_rng(0)
    fits = [(0.5, ph) for ph in rng.normal(0.0, 0.2, size=100)]
    gain, phase = SD.mean_response(fits)
    assert abs(phase) < 0.05 and 0.45 < gain <= 0.5


def test_mean_response_gain_is_not_lifted_by_noise() -> None:
    # a filtered-out period: per-clip fits are pure noise with random phase;
    # their median magnitude is the noise floor, the complex mean is ~0
    rng = np.random.default_rng(1)
    fits = [(0.1, ph) for ph in rng.uniform(-np.pi, np.pi, size=400)]
    assert SD.mean_response(fits)[0] < 0.02


def test_lags_unwrap_across_periods() -> None:
    # a fixed 6-frame delay: at P=8 the phase wraps to -pi/2 (a 2-frame lead)
    periods = [64.0, 32.0, 16.0, 8.0]
    wrapped = [np.angle(np.exp(1j * 2 * np.pi * 6 / p)) for p in periods]
    assert np.allclose(SD.lags_across_periods(periods, wrapped), 6.0)


def test_step_stats() -> None:
    tracking = np.concatenate([np.zeros(40), [0.3, 0.5, 0.7, 0.9], np.ones(148)])
    tau, settled = SD.step_stats(tracking, 40)
    assert tau == 2 and settled == 1.0
    # the time constant is relative to where the curve settles, not to 1
    tau, settled = SD.step_stats(0.5 * tracking, 40)
    assert tau == 2 and settled == 0.5
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
        dropout_at=None,
        step_at=40,
        step_factor=[0.8, 1.25, 2.0],
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
        ({"step_at": 180}, ours),  # no settled tail after it
        ({"step_factor": [1.0]}, ours),
        ({"step_factor": [1.5, 1.5]}, ours),  # one run each: no duplicates
        ({"step_factor": [0.0]}, ours),
        ({"experiment": "sine", "sine_amplitude": 0.8}, ours),
        ({"experiment": "sine", "sine_periods": [2.0]}, ours),
        ({"experiment": "jitter", "jitter_sigma": 0.0}, ours),
        ({}, [SD.parse_curve("StreamVGGT=base")]),  # nothing to perturb
        ({"experiment": "jitter"}, [SD.parse_curve("Ours=/runs/x@20")]),
        ({"experiment": "splice", "splice_scale": 1.0}, ours),  # that is the control
        ({"experiment": "dropout"}, [SD.parse_curve("Ours=/runs/x@192")]),
        ({"experiment": "dropout", "ref_frames": 30}, [SD.parse_curve("O=/r@20")]),
        (
            {"experiment": "dropout", "dropout_at": 40},
            [SD.parse_curve("O=/r@20")],
        ),
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


def _ours_and_base():
    return [SD.parse_curve("StreamVGGT=base"), SD.parse_curve("Ours=/runs/x@20")]


def test_rows_round_trip_through_the_csv(tmp_path) -> None:
    import csv

    header = SD.header_for("step")
    rows = [("Ours", "k=2", "c0", i, 0.0, 0.1 * i, 0.0, float("nan")) for i in range(3)]
    with open(tmp_path / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    h, back = SD.read_rows(tmp_path / "per_frame.csv")
    assert h == header and back[2][:6] == rows[2][:6] and np.isnan(back[2][7])
    series = SD.group(h, back, ("curve", "perturbation", "clip"))
    assert np.allclose(series[("Ours", "k=2", "c0")]["r"], [0.0, 0.1, 0.2])
    with pytest.raises(ValueError):  # frames out of order
        SD.group(h, back[::-1], ("curve", "perturbation", "clip"))


def test_dropout_summary_includes_base_absolute_error(tmp_path) -> None:
    import argparse

    args = argparse.Namespace(ref_frames=2, dropout_at=4, window=3)
    header = SD.header_for("dropout")
    rows = []
    for label, a in (("StreamVGGT", 0.8), ("Ours (dropout @20)", 0.05)):
        for clip in ("c0", "c1"):
            rows += [(label, clip, i, np.exp(a), 0.0, a) for i in range(8)]
    summary = SD.summarize_dropout(
        args,
        _ours_and_base()[:1] + [SD.parse_curve("Ours (dropout @20)=/runs/x@4")],
        header,
        rows,
        tmp_path,
    )
    base, ours = summary["curves"]
    assert np.allclose(base["abs_median"], 0.8) and base["metric"] is False
    assert np.allclose(ours["abs_median"], 0.05) and ours["metric"] is True
    assert (tmp_path / "scale_drift.png").stat().st_size > 0


def test_step_summary_per_factor_and_fed_split(tmp_path) -> None:
    import argparse

    S, step_at = 80, 40
    args = argparse.Namespace(
        experiment="step", step_factor=[2.0, 0.8], step_at=step_at, window=20
    )
    t = np.arange(S)
    rows = []
    for k, gain in ((0.8, 0.9), (2.0, 0.5)):  # the bigger step followed less
        u = np.where(t >= step_at, np.log(k), 0.0)
        for clip in ("c0", "c1"):
            # fed pixels copy the input; held-out pixels follow `gain` of it
            rows += [
                ("Ours", f"k={k:g}", clip, i, u[i], gain * u[i], 0.0, u[i])
                for i in range(S)
            ]
    summary = SD.summarize_response(
        args, [SD.parse_curve("Ours=/runs/x")], SD.header_for("step"), rows, tmp_path
    )
    (entry,) = summary["curves"]
    settled = [f["tracking_settled"] for f in entry["factors"]]
    fed = [f["tracking_fed_settled"] for f in entry["factors"]]
    # ascending factors whatever order they were given in
    assert [f["factor"] for f in entry["factors"]] == [0.8, 2.0]
    assert np.allclose(settled, [0.9, 0.5]) and np.allclose(fed, [1.0, 1.0])
    assert (tmp_path / "scale_step_gain.png").stat().st_size > 0
    # a run from before r_fed: same summary, the fed part empty
    old = [r[:7] for r in rows]
    summary = SD.summarize_response(
        args, [SD.parse_curve("Ours=/runs/x")], SD.header_for("step")[:7], old, tmp_path
    )
    assert np.isnan(summary["curves"][0]["factors"][0]["tracking_fed_settled"])


def test_jitter_summary_skips_a_clip_without_scored_pixels(tmp_path) -> None:
    import argparse

    S = 60
    args = argparse.Namespace(experiment="jitter", jitter_sigma=0.1, window=20)
    rng = np.random.default_rng(3)
    rows = []
    for clip, gain in (("c0", 0.5), ("c1", 0.5), ("c2", float("nan"))):
        u = 0.1 * rng.standard_normal(S)
        rows += [
            ("Ours", "sigma=0.1", clip, i, u[i], gain * u[i], 0.0, u[i])
            for i in range(S)
        ]
    summary = SD.summarize_response(
        args, [SD.parse_curve("Ours=/runs/x")], SD.header_for("jitter"), rows, tmp_path
    )
    (entry,) = summary["curves"]
    assert abs(entry["gain_median"] - 0.5) < 1e-9
    assert abs(entry["gain_median_fed"] - 1.0) < 1e-9


def test_cut_bins_and_pair_selection() -> None:
    assert SD.cut_bin(1.1) == 0 and SD.cut_bin(1 / 1.1) == 0
    assert SD.cut_bin(1.4) is None  # the gap keeps the control apart
    assert SD.cut_bin(1.8) == 1 and SD.cut_bin(1 / 2.2) == 2 and SD.cut_bin(4.0) == 3
    depths = [1.0, 1.1, 1.7, 2.2, 3.0]
    pairs = SD.select_cut_pairs(depths, per_bin=1, seed=0)
    cells = {(SD.cut_bin(r), r >= 1) for _, _, r in pairs}
    assert len(cells) == len(pairs)  # at most one per (bin, direction)
    for i, j, r in pairs:
        assert i != j and abs(r - depths[j] / depths[i]) < 1e-12
    assert pairs == SD.select_cut_pairs(depths, per_bin=1, seed=0)  # seeded
    # the cap: one tiny room cannot lead every "into larger" pair of a bin
    depths = [0.5] + [2.0 + 0.01 * k for k in range(10)]
    pairs = SD.select_cut_pairs(depths, per_bin=8, seed=0, cap=2)
    big_up = [p for p in pairs if SD.cut_bin(p[2]) == 3 and p[2] >= 1]
    assert big_up and sum(1 for i, _, _ in big_up if i == 0) <= 2
    assert SD.matched_lead([1.0, 1.9, 2.05, 3.0], 1, exclude=set()) == 2


def test_frames_until_within_is_shared_by_relock_and_recovery() -> None:
    err = np.array([0.0, 0.0, 0.5, 0.3, 0.045, 0.04])
    assert SD.frames_until_within(err, np.full(6, 0.03), start=2) == 2
    assert SD.relock_frames(err, np.full(6, 0.03), 2) == 2


def test_recovery_needs_a_held_stretch() -> None:
    ex = np.array([0.5, 0.3, 0.01, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3])
    assert SD.recovery_frames(ex, tol=0.02, hold=5) == 4  # not the lone frame 2
    assert SD.recovery_frames(np.full(10, 0.1), tol=0.02, hold=5) is None


def test_cut_summary_measures_excess_over_clip_b_alone(tmp_path) -> None:
    import argparse

    cut, T = 10, 10
    args = argparse.Namespace(experiment="cut", window=5)
    rows = []
    for pair, ratio in (("a+b", 2.2), ("b+a", 1 / 2.2), ("c+d", 1.1)):
        # spliced: clip b starts 0.3 off and decays to b's own 0.05 by frame 4
        b_err = np.r_[0.3, 0.2, 0.1, 0.06, np.full(T - 4, 0.05)]
        spliced = np.r_[np.full(cut, 0.05), b_err]
        rows += [
            ("Ours", pair, ratio, "spliced", t, np.exp(spliced[t]), 0.02)
            for t in range(cut + T)
        ]
        # the matched-depth reference into the same clip b: 0.05 throughout
        b = pair.split("+")[1]
        rows += [
            ("Ours", f"m+{b}", 1.0, f"ref:{b}", t, np.exp(0.05), 0.02)
            for t in range(cut + T)
        ]
    summary = SD.summarize_cut(
        args,
        [SD.parse_curve("Ours=/runs/x")],
        SD.header_for("cut"),
        rows,
        tmp_path,
        cut,
    )
    bins = summary["curves"][0]["bins"]
    assert sorted((tuple(b["ratio"]), b["into"], b["pairs"]) for b in bins) == [
        ((1.0, 1.25), "larger", 1),
        ((2.0, 2.5), "larger", 1),
        ((2.0, 2.5), "smaller", 1),
    ]
    assert all(b["recovery_frames_median"] == 3 for b in bins)
    assert (tmp_path / "scale_cut.png").exists()
    assert (tmp_path / "scale_cut_recovery.png").exists()


def test_hold_depth_repeats_the_last_measured_frame() -> None:
    depth = np.arange(5, dtype=np.float32)[:, None, None] * np.ones(
        (5, 2, 2), np.float32
    )
    mask = np.ones((5, 2, 2), bool)
    d, m = SD.hold_depth(depth, mask, 2)
    assert np.allclose(d[:, 0, 0], [0, 1, 1, 1, 1]) and m.all()
    assert np.allclose(depth[:, 0, 0], [0, 1, 2, 3, 4])  # input untouched
    d, _ = SD.hold_depth(depth, mask, None)
    assert d is depth


def test_promptda_curves_are_dropout_only() -> None:
    import argparse

    base = dict(
        experiment="dropout",
        window=20,
        ref_frames=20,
        dropout_at=20,
        step_at=40,
        step_factor=[1.25],
        sine_amplitude=1.2,
        sine_periods=[8.0],
        jitter_sigma=0.1,
        splice_scale=2.0,
        cut_pairs_per_bin=1,
    )
    ok = [SD.parse_curve("P=promptda"), SD.parse_curve("H=promptda-hold@20")]
    SD.validate(argparse.Namespace(**base), ok, 192)
    for change, curves in (
        ({"experiment": "step", "dropout_at": None}, [SD.parse_curve("P=promptda")]),
        ({}, [SD.parse_curve("H=promptda-hold")]),  # hold needs a drop frame
    ):
        with pytest.raises(ValueError):
            SD.validate(argparse.Namespace(**{**base, **change}), curves, 192)


def test_parse_curve_ema() -> None:
    c = SD.parse_curve("Base+EMA=base@20+ema0.9")
    assert (c.source, c.drop, c.ema, c.takes_depth) == ("base", 20, 0.9, True)
    c = SD.parse_curve("Ours=/runs/abc+ema0")
    assert (c.source, c.drop, c.ema) == ("/runs/abc", None, 0.0)
    assert not SD.parse_curve("B=base").takes_depth
    for bad in ("x=base+ema", "x=base+ema1", "x=base+emahalf"):
        with pytest.raises(ValueError):
            SD.parse_curve(bad)


def _fed(values, fed_frames, hw=(2, 3)):
    views = []
    for t, v in enumerate(values):
        mask = torch.zeros((1, *hw), dtype=torch.bool)
        if t in fed_frames:
            mask[0, 0, 0] = True
        views.append({"sparse_depth": torch.full((1, *hw), v) * mask, "sparse_depth_mask": mask})
    return views


def test_ema_correct_tracks_then_holds_the_fed_scale() -> None:
    pred = np.ones((5, 2, 3), np.float32)
    views = _fed([2.0, 2.0, 2.0, 8.0, 8.0], fed_frames={0, 1, 2})
    out = SD.ema_correct(pred, views, alpha=0.0)  # no memory: follows each frame
    np.testing.assert_allclose(out[:3], 2.0, rtol=1e-6)
    np.testing.assert_allclose(out[3:], 2.0, rtol=1e-6)  # unfed frames hold
    views = _fed([2.0, 2.0, 8.0], fed_frames={0, 1, 2})
    out = SD.ema_correct(np.ones((3, 2, 3), np.float32), views, alpha=0.5)
    # log-space EMA: frame 2 is halfway between log 2 and log 8
    np.testing.assert_allclose(out[2], 4.0, rtol=1e-6)
    out = SD.ema_correct(pred, _fed([1.0] * 5, fed_frames={2}), alpha=0.5)
    np.testing.assert_allclose(out[:2], 1.0)  # before the first measurement: as is


def test_ema_makes_base_a_depth_taking_curve() -> None:
    import argparse

    args = argparse.Namespace(
        experiment="step",
        window=20,
        ref_frames=20,
        dropout_at=None,
        step_at=40,
        step_factor=[1.25],
        sine_amplitude=1.2,
        sine_periods=[8.0],
        jitter_sigma=0.1,
        splice_scale=2.0,
        cut_pairs_per_bin=1,
    )
    SD.validate(args, [SD.parse_curve("B=base+ema0.9")], 192)
    with pytest.raises(ValueError):
        SD.validate(args, [SD.parse_curve("B=base")], 192)
    args.experiment = "dropout"
    with pytest.raises(ValueError):
        SD.validate(args, [SD.parse_curve("P=promptda+ema0.9")], 192)
