"""Scale under perturbed sparse depth: how the model's metric scale follows,
holds or filters what it is fed (--experiment; default `dropout`).

  dropout  depth withheld from a frame on: how long the KV cache carries
           absolute scale once the sparse depth stops (below)
  step     fed depth multiplied by --step-factor from --step-at on
  sine     fed depth multiplied by exp(a sin(2 pi t / P)) for each period P
  jitter   fed depth multiplied by exp(sigma eps_t), eps_t i.i.d. normal
  splice   two clips back to back, the second one's GT AND fed depth
           multiplied by --splice-scale (a world --splice-scale times larger:
           the images cannot tell, the sparse depth can)

step / sine / jitter measure the model's response against its own
unperturbed run on the same clip, so neither GT nor its errors enter:
  u_t = log m_t                       the input, m_t the factor on the fed depth
  r_t = median(log pred_p - log pred_0)  the output scale response
  q_t = median |log pred_p - log pred_0 - r_t|  the change in structure (scale-free)
A persistent step is a real change of scale that the output should follow
(tracking r_t / log k, time to 63%); per-frame jitter is noise the output
should not pass (passthrough std r / std u, regression gain); the sine sweep
gives the gain and lag at each period, a Bode plot of the scale channel.

Dropout and splice score against GT, as below.

Each curve streams the ScanNet TAE clips of the benchmark (100 scenes x 192
consecutive frames, the longest sequences it has) frame by frame through a
sliding KV cache of `--window` frames, the deployment setting. A curve is
`LABEL=SOURCE[@DROP]`:

  SOURCE  `base` for the pretrained StreamVGGT (RGB only), or a
          finetune_depth.py run directory (its checkpoint-last)
  DROP    the frame index from which the sparse depth is withheld (an
          all-zero validity mask, the model's "no measurement"); omitted =
          fed on every frame. Ignored for `base`, which takes no depth.

    python src/scale_drift.py --out <dir> --dropout-at 20 \\
        --curve "StreamVGGT=base" \\
        --curve "Ours (dropout @20)=<run>@20" \\
        --curve "Ours (fed)=<run>"

Sparse depth is the benchmark's own 5% TUBE_MASK draw for each clip (same
seed and tag as its TAE scoring), so a clip gets the same fed pixels in every
curve. Per frame, on the GT grid, over GT-valid pixels not fed to the model:

  s_t      = median(GT / pred), the scale the frame is off by
  relative = |log(s_t / s_ref)|, s_ref the median of s_t over the clip's
             first --ref-frames frames: drift from the model's own early
             scale, defined for any model, metric or not
  absolute = |log s_t|, the metric scale error (metric models only)

Writes <out>/per_frame.csv, <out>/summary.json (curves, settings and the
code that ran) and <out>/scale_drift.png: two panels, relative and absolute,
the median over clips with the interquartile band, vertical lines at the
dropout frame and at dropout + window (the first frame whose cache holds no
conditioned frame).
"""

from __future__ import annotations

# first: eval.provenance's audit hook records every repository file run from
# here on, with its content at that moment (results' code_files)
import eval.provenance  # noqa: E402,F401,I001

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_eval import (  # noqa: E402
    BenchmarkCfg,
    _prepare_sequence,
    _sparse_arrays,
    attach_sparse_depth,
)
from eval import provenance as PROV  # noqa: E402
from eval.protocols import gt_valid_mask  # noqa: E402
from eval.streaming_window import stream_depth  # noqa: E402
from eval.vda_benchmark import SPECS, load_manifest, model_to_gt  # noqa: E402
from model_io import (  # noqa: E402
    build_model,
    load_base_streamvggt,
    load_saved_args,
    rebuild_metric_cfg,
    resolve_checkpoint,
)

DENSITY = 0.05
DATASET = "scannet"
DEFAULT_BASE = "/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth"
EXPERIMENTS = ("dropout", "step", "sine", "jitter", "splice")
_COLORS = ["#2563eb", "#e8590c", "#16a34a", "#9333ea", "#6b7280"]


@dataclass
class Curve:
    label: str
    source: str  # "base" or a run directory
    drop: int | None  # first frame without sparse depth; None = always fed

    @property
    def is_base(self) -> bool:
        return self.source == "base"


def parse_curve(spec: str) -> Curve:
    """LABEL=SOURCE[@DROP]."""
    label, sep, rest = spec.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"--curve wants LABEL=SOURCE[@DROP], got {spec!r}")
    source, at, drop = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if at and not drop.isdigit():
        raise ValueError(f"--curve {spec!r}: DROP must be a frame index")
    return Curve(label, source, int(drop) if at else None)


def load_curve_model(curve: Curve, base_ckpt: str, device: torch.device):
    if curve.is_base:
        return load_base_streamvggt(base_ckpt).to(device)
    ckpt_path = resolve_checkpoint(curve.source, "last")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    raw = load_saved_args(ckpt)
    # as bench_checkpoint: the run's training-time sparse simulation is not
    # used here (the benchmark's draw is), and a pixel_freq run's freq-map
    # path would otherwise have to exist on this machine for validate()
    raw["depth_cond"] = dict(raw["depth_cond"])
    raw["depth_cond"]["sim_mode"] = "none"
    raw["depth_cond"]["sim_freq_map_path"] = ""
    raw["depth_cond"]["sim_freq_map_sha256"] = ""
    mcfg = rebuild_metric_cfg(raw)
    # the streaming loop calls the aggregator directly, never with a cache key
    mcfg.encoder_cache.enabled = False
    model, _ = build_model(mcfg, device, load_pretrained=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    del ckpt, state_dict
    return model.eval()


def withhold_depth(views: list[dict], drop: int | None) -> None:
    """Frames from `drop` on get no sparse depth: zero depth AND a zero
    validity mask, which the model reads as "no measurement" (in place)."""
    if drop is None:
        return
    for view in views[drop:]:
        if "sparse_depth" in view:
            view["sparse_depth"] = torch.zeros_like(view["sparse_depth"])
            view["sparse_depth_mask"] = torch.zeros_like(view["sparse_depth_mask"])


def frame_scales(pred: np.ndarray, gt: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """[S] median(GT / pred) per frame over the scored pixels; NaN for a
    frame with none or with a non-positive prediction everywhere."""
    out = np.full(len(gt), np.nan)
    for i in range(len(gt)):
        m = scored[i] & np.isfinite(pred[i]) & (pred[i] > 0)
        if m.any():
            out[i] = float(np.median(gt[i][m] / pred[i][m]))
    return out


def drift_rows(scales: np.ndarray, ref_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """(relative, absolute) per frame: |log(s_t / s_ref)| with s_ref the
    median scale over the first ref_frames frames, and |log s_t|."""
    log_s = np.log(scales)
    ref = np.nanmedian(log_s[:ref_frames])
    return np.abs(log_s - ref), np.abs(log_s)


def band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile range over clips (axis 0), NaN-aware."""
    return (
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 25, axis=0),
        np.nanpercentile(values, 75, axis=0),
    )


def scale_fed(views: list[dict], factors: np.ndarray) -> list[dict]:
    """Per-view copies whose fed depth is multiplied by factors[t] (the
    validity mask is unchanged: the same pixels, other values)."""
    out = []
    for view, f in zip(views, factors, strict=True):
        v = dict(view)
        if "sparse_depth" in v:
            v["sparse_depth"] = v["sparse_depth"] * float(f)
        out.append(v)
    return out


def response(
    pred_p: np.ndarray, pred_0: np.ndarray, scored: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(r_t, q_t) per frame over the scored pixels: the median log-ratio of
    the perturbed to the unperturbed prediction (the scale response), and
    the median absolute log-ratio left after removing it (structure change)."""
    S = len(pred_0)
    r, q = np.full(S, np.nan), np.full(S, np.nan)
    for i in range(S):
        m = scored[i] & (pred_p[i] > 0) & (pred_0[i] > 0)
        m &= np.isfinite(pred_p[i]) & np.isfinite(pred_0[i])
        if m.any():
            d = np.log(pred_p[i][m]) - np.log(pred_0[i][m])
            r[i] = float(np.median(d))
            q[i] = float(np.median(np.abs(d - r[i])))
    return r, q


def sine_fit(
    r: np.ndarray, period: float, amplitude: float, start: int
) -> tuple[float, float]:
    """(gain, phase) of r_t against u_t = amplitude sin(2 pi t / P), least
    squares over t >= start: r ~ A sin + B cos + c. With r = g amplitude
    sin(w t - phase), g = |(A, B)| / |amplitude| and phase in (-pi, pi],
    positive when the output lags. Average phases across clips with
    circular_lag, never linearly."""
    t = np.arange(len(r))
    ok = (t >= start) & np.isfinite(r)
    w = 2 * np.pi / period
    X = np.stack([np.sin(w * t[ok]), np.cos(w * t[ok]), np.ones(ok.sum())], 1)
    (A, B, _), *_ = np.linalg.lstsq(X, r[ok], rcond=None)
    sign = 1.0 if amplitude > 0 else -1.0
    return math.hypot(A, B) / abs(amplitude), math.atan2(-sign * B, sign * A)


def circular_lag(phases, period: float) -> float:
    """Lag in frames of the circular mean of per-clip phases: a lag near 0
    with clip noise on both sides stays near 0 instead of averaging 0 and P."""
    z = np.mean(np.exp(1j * np.asarray(phases, dtype=float)))
    return float(np.angle(z)) * period / (2 * np.pi)


def step_stats(tracking: np.ndarray, step_at: int) -> tuple[int | None, float]:
    """(frames from the step until the tracking curve first reaches 0.63 --
    None if it never does -- and its mean over the last 20 frames)."""
    t = np.arange(len(tracking))
    hit = np.where((t >= step_at) & (tracking >= 0.63))[0]
    final = float(np.nanmean(tracking[-20:]))
    return (None if not len(hit) else int(hit[0] - step_at)), final


def jitter_stats(r: np.ndarray, u: np.ndarray, start: int) -> tuple[float, float]:
    """(regression gain of r on u, passthrough std(r) / std(u)) over
    t >= start and finite r."""
    ok = np.isfinite(r) & (np.arange(len(r)) >= start)
    gain = float(np.polyfit(u[ok], r[ok], 1)[0])
    return gain, float(np.std(r[ok]) / np.std(u[ok]))


def relock_frames(
    err: np.ndarray, control: np.ndarray, cut: int, tol: float = 0.02
) -> int | None:
    """Frames after the cut until the scale error is back within `tol` of the
    same pair spliced at x1: a scene cut alone also disturbs scale, and the
    control removes that part. None if it never is."""
    t = np.arange(len(err))
    back = np.where((t >= cut) & (err <= control + tol))[0]
    return None if not len(back) else int(back[0] - cut)


def median_scaled_absrel(
    pred: np.ndarray, gt: np.ndarray, scored: np.ndarray
) -> np.ndarray:
    """[S] AbsRel after a per-frame median scale: structure error alone."""
    out = np.full(len(gt), np.nan)
    for i in range(len(gt)):
        m = scored[i] & np.isfinite(pred[i]) & (pred[i] > 0)
        if m.any():
            p = pred[i][m] * np.median(gt[i][m] / pred[i][m])
            out[i] = float(np.mean(np.abs(p - gt[i][m]) / gt[i][m]))
    return out


def _style(ax) -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def _band_line(ax, x, values, color, label) -> None:
    med, lo, hi = band(values)
    ax.plot(x, med, color=color, linewidth=2, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _clip(seq, spec, bench, device, tae: bool = True):
    """(gt, views with the benchmark's sparse draw, scored mask, box), built
    exactly as the benchmark scores the clip."""
    gt, views = _prepare_sequence(spec, seq, bench.image_size, device)
    # the benchmark's own draw for this clip (tag and seed as its scoring), so
    # every curve sees the same fed pixels
    tag = f"{spec.name}/tae/{seq.name}" if tae else f"{spec.name}/{seq.name}"
    attach_sparse_depth(views, DENSITY, tag, bench.patch_size, bench.seed, device)
    _, fed = _sparse_arrays(views)
    box = views[0].get("gt_box")  # whole-frame input, as bench_eval reads it
    # every curve is scored on the same pixels: GT-valid and never in the draw
    # (the fed ones would measure copying), on every frame
    scored = gt_valid_mask(gt, spec.max_depth) & ~model_to_gt(
        fed, gt.shape[1:], box, mask=True
    )
    return gt, views, scored, box


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def run_dropout(args, curves, models, seqs, spec, bench, device):
    rel = {c.label: [] for c in curves}
    absol = {c.label: [] for c in curves}
    rows = []
    for n, seq in enumerate(seqs):
        gt, views, scored, box = _clip(seq, spec, bench, device)
        for curve in curves:
            # shallow per-view copies: withhold_depth replaces tensors, never
            # writes into them, so the next curve still sees the full draw
            curve_views = [dict(v) for v in views]
            if not curve.is_base:
                withhold_depth(curve_views, curve.drop)
            pred = stream_depth(models[curve.label], curve_views, args.window)
            s = frame_scales(model_to_gt(pred, gt.shape[1:], box), gt, scored)
            r, a = drift_rows(s, args.ref_frames)
            rel[curve.label].append(r)
            absol[curve.label].append(a)
            rows += [
                (curve.label, seq.name, i, s[i], r[i], a[i]) for i in range(len(s))
            ]
        print(f"[dropout] clip {n + 1}/{len(seqs)} {seq.name}", flush=True)
        del views
    rel = {k: np.stack(v) for k, v in rel.items()}
    absol = {k: np.stack(v) for k, v in absol.items()}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    for k, c in enumerate(curves):
        x = np.arange(rel[c.label].shape[1])
        _band_line(ax1, x, rel[c.label], _COLORS[k % 5], c.label)
        if not c.is_base:
            _band_line(ax2, x, absol[c.label], _COLORS[k % 5], c.label)
    ax1.set_ylabel("|log(s_t / s_ref)|  (drift from own early scale)")
    ax2.set_ylabel("|log s_t|  (metric scale error)")
    ax2.set_xlabel("frame index")
    drop = args.dropout_at
    for ax in (ax1, ax2):
        if drop is not None:
            ax.axvline(drop - 0.5, color="#374151", linestyle="--", linewidth=1)
            ax.axvline(
                drop + args.window - 0.5, color="#9ca3af", linestyle=":", linewidth=1
            )
        _style(ax)
    title = f"Scale drift, ScanNet TAE clips, KV window {args.window}"
    if drop is not None:
        title += (
            f": depth withheld from frame {drop} (dashed); no conditioned frame in "
            f"cache from {drop + args.window} (dotted)"
        )
    ax1.set_title(title, fontsize=9)
    fig.savefig(args.out / "scale_drift.png", dpi=150)
    plt.close(fig)

    summary = {
        "ref_frames": args.ref_frames,
        "curves": [
            {
                "label": c.label,
                "source": c.source,
                "drop": c.drop,
                "rel_median": band(rel[c.label])[0],
                "abs_median": None if c.is_base else band(absol[c.label])[0],
            }
            for c in curves
        ],
    }
    return (
        ["curve", "clip", "frame", "scale", "rel_log_drift", "abs_log_error"],
        rows,
        summary,
    )


def _inputs(args, S: int, clip_no: int, bench) -> dict[str, np.ndarray]:
    """The log factors u_t applied to the fed depth, per perturbation."""
    t = np.arange(S)
    if args.experiment == "step":
        u = np.where(t >= args.step_at, math.log(args.step_factor), 0.0)
        return {f"k={args.step_factor:g}": u}
    if args.experiment == "sine":
        a = math.log(args.sine_amplitude)
        return {f"P={p:g}": a * np.sin(2 * np.pi * t / p) for p in args.sine_periods}
    # jitter: seeded per clip, so every curve gets the same draw on a clip
    rng = np.random.default_rng([bench.seed, clip_no])
    return {f"sigma={args.jitter_sigma:g}": args.jitter_sigma * rng.standard_normal(S)}


def run_response(args, curves, models, seqs, spec, bench, device):
    """step / sine / jitter: each perturbed run against the same model's
    unperturbed run on the same clip and draw."""
    exp = args.experiment
    out = {c.label: {} for c in curves}  # label -> perturbation -> [(r, q, u)]
    rows = []
    S = None
    for n, seq in enumerate(seqs):
        gt, views, scored, box = _clip(seq, spec, bench, device)
        S = len(views)
        inputs = _inputs(args, S, n, bench)
        for c in curves:
            model = models[c.label]
            pred_0 = model_to_gt(
                stream_depth(model, views, args.window), gt.shape[1:], box
            )
            for key, u in inputs.items():
                pred = stream_depth(model, scale_fed(views, np.exp(u)), args.window)
                r, q = response(model_to_gt(pred, gt.shape[1:], box), pred_0, scored)
                out[c.label].setdefault(key, []).append((r, q, u))
                rows += [
                    (c.label, key, seq.name, i, u[i], r[i], q[i]) for i in range(S)
                ]
        print(f"[{exp}] clip {n + 1}/{len(seqs)} {seq.name}", flush=True)
        del views
    header = ["curve", "perturbation", "clip", "frame", "u", "r", "q"]
    summary = {"curves": []}
    x = np.arange(S)
    if exp == "step":
        log_k = math.log(args.step_factor)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, constrained_layout=True
        )
        for k, c in enumerate(curves):
            (runs,) = out[c.label].values()
            tracking = np.stack([r for r, _, _ in runs]) / log_k
            q = np.stack([q for _, q, _ in runs])
            t63, final = step_stats(band(tracking)[0], args.step_at)
            summary["curves"].append(
                {
                    "label": c.label,
                    "tracking_median": band(tracking)[0],
                    "frames_to_63pct": t63,
                    "tracking_final": final,
                    "structure_change_median": band(q)[0],
                }
            )
            _band_line(ax1, x, tracking, _COLORS[k % 5], c.label)
            _band_line(ax2, x, q, _COLORS[k % 5], c.label)
        ax1.axhline(1.0, color="#9ca3af", linewidth=1)
        ax1.set_ylabel("r_t / log k  (fraction of the step followed)")
        ax2.set_ylabel("q_t  (structure change, scale-free)")
        ax2.set_xlabel("frame index")
        for ax in (ax1, ax2):
            ax.axvline(args.step_at - 0.5, color="#374151", linestyle="--", linewidth=1)
            _style(ax)
        ax1.set_title(
            f"Step response: fed depth x{args.step_factor:g} from frame {args.step_at} "
            f"(KV window {args.window})",
            fontsize=9,
        )
        fig.savefig(args.out / "scale_step.png", dpi=150)
    elif exp == "sine":
        a = math.log(args.sine_amplitude)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
        )
        for k, c in enumerate(curves):
            gains, lags = [], []
            for p in args.sine_periods:
                fits = [
                    sine_fit(r, p, a, args.window)
                    for r, _, _ in out[c.label][f"P={p:g}"]
                ]
                gains.append(float(np.median([g for g, _ in fits])))
                lags.append(circular_lag([ph for _, ph in fits], p))
            summary["curves"].append(
                {
                    "label": c.label,
                    "periods": args.sine_periods,
                    "gain": gains,
                    "lag_frames": lags,
                }
            )
            ax1.plot(
                args.sine_periods,
                gains,
                "o-",
                color=_COLORS[k % 5],
                linewidth=2,
                label=c.label,
            )
            ax2.plot(
                args.sine_periods,
                lags,
                "o-",
                color=_COLORS[k % 5],
                linewidth=2,
                label=c.label,
            )
        ax1.axhline(1.0, color="#9ca3af", linewidth=1)
        ax2.axhline(0.0, color="#9ca3af", linewidth=1)
        ax1.set_ylabel("gain  |r| / |u|  (clip median)")
        ax2.set_ylabel("lag (frames, circular mean)")
        ax2.set_xlabel("period of the fed-depth modulation (frames)")
        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            _style(ax)
        ax1.set_title(
            f"Frequency response: fed depth x exp({a:.3f} sin(2 pi t / P)), "
            f"fitted from frame {args.window}",
            fontsize=9,
        )
        fig.savefig(args.out / "scale_sine.png", dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        first = next(iter(out[curves[0].label].values()))
        ax.plot(first[0][2], color="#9ca3af", linewidth=1, label="input u_t (clip 1)")
        for k, c in enumerate(curves):
            (runs,) = out[c.label].values()
            stats = [jitter_stats(r, u, args.window) for r, _, u in runs]
            gain = float(np.median([g for g, _ in stats]))
            passthrough = float(np.median([p for _, p in stats]))
            summary["curves"].append(
                {
                    "label": c.label,
                    "gain_median": gain,
                    "passthrough_median": passthrough,
                }
            )
            ax.plot(
                runs[0][0],
                color=_COLORS[k % 5],
                linewidth=1.5,
                label=f"{c.label}: gain {gain:.2f}, passthrough {passthrough:.2f}",
            )
        ax.set_xlabel("frame index")
        ax.set_ylabel("log scale")
        _style(ax)
        ax.set_title(
            f"Jitter: fed depth x exp({args.jitter_sigma:g} eps_t); gain and passthrough "
            f"are clip medians from frame {args.window}",
            fontsize=9,
        )
        fig.savefig(args.out / "scale_jitter.png", dpi=150)
    plt.close(fig)
    return header, rows, summary


def run_splice(args, curves, models, seqs, spec, bench, device):
    """Clip n followed by clip n+1 (main manifest, 90 + 90 frames), the second
    one's GT and fed depth times --splice-scale; and the same pair at x1, the
    control that separates a scene cut's own disturbance from re-locking."""
    k = args.splice_scale
    err = {c.label: {k: [], 1.0: []} for c in curves}
    struct = {c.label: [] for c in curves}
    rows, cut = [], None
    for n in range(len(seqs)):
        a, b = seqs[n], seqs[(n + 1) % len(seqs)]
        gt_a, views_a, scored_a, box = _clip(a, spec, bench, device, tae=False)
        gt_b, views_b, scored_b, _ = _clip(b, spec, bench, device, tae=False)
        cut = len(views_a)
        scored = np.concatenate([scored_a, scored_b])
        for factor in (k, 1.0):
            gt = np.concatenate([gt_a, gt_b * factor])
            views = views_a + scale_fed(views_b, np.full(len(views_b), factor))
            for c in curves:
                if c.is_base and factor != k:
                    continue  # no fed depth: the control is the same run
                pred = stream_depth(models[c.label], views, args.window)
                pred = model_to_gt(pred, gt.shape[1:], box)
                s = frame_scales(pred, gt, scored)
                err[c.label][factor].append(np.abs(np.log(s)))
                if factor == k:
                    st = median_scaled_absrel(pred, gt, scored)
                    struct[c.label].append(st)
                    rows += [
                        (c.label, f"{a.name}+{b.name}", i, s[i], st[i])
                        for i in range(len(s))
                    ]
        print(f"[splice] pair {n + 1}/{len(seqs)} {a.name}+{b.name}", flush=True)
        del views_a, views_b
    header = ["curve", "pair", "frame", "scale", "median_scaled_absrel"]
    summary = {"splice_scale": k, "cut_frame": cut, "curves": []}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, constrained_layout=True
    )
    for i, c in enumerate(curves):
        q = np.stack(struct[c.label])
        x = np.arange(q.shape[1])
        entry = {"label": c.label, "structure_median": band(q)[0]}
        if not c.is_base:
            e, e1 = np.stack(err[c.label][k]), np.stack(err[c.label][1.0])
            entry |= {
                "abs_log_error_median": band(e)[0],
                "abs_log_error_median_x1": band(e1)[0],
                "relock_frames": relock_frames(band(e)[0], band(e1)[0], cut),
            }
            _band_line(ax1, x, e, _COLORS[i % 5], f"{c.label} (x{k:g})")
            ax1.plot(
                x,
                band(e1)[0],
                color=_COLORS[i % 5],
                linewidth=1,
                linestyle=":",
                label=f"{c.label} (x1 control)",
            )
        summary["curves"].append(entry)
        _band_line(ax2, x, q, _COLORS[i % 5], c.label)
    ax1.set_ylabel("|log s_t|  (metric scale error)")
    ax2.set_ylabel("median-scaled AbsRel  (structure)")
    ax2.set_xlabel("frame index")
    for ax in (ax1, ax2):
        ax.axvline(cut - 0.5, color="#374151", linestyle="--", linewidth=1)
        _style(ax)
    ax1.set_title(
        f"Splice: second clip's world x{k:g} (GT and fed depth) from frame {cut}, "
        f"KV window {args.window}",
        fontsize=9,
    )
    fig.savefig(args.out / "scale_splice.png", dpi=150)
    plt.close(fig)
    return header, rows, summary


def validate(args, curves, clip_len: int) -> None:
    """Every check that needs no GPU, before any model is loaded."""
    labels = [c.label for c in curves]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate curve labels: {labels}")
    if args.window < 1 or args.ref_frames < 1:
        raise ValueError("--window and --ref-frames must be >= 1")
    if args.experiment != "dropout" and any(c.drop is not None for c in curves):
        raise ValueError("@DROP only applies to --experiment dropout")
    if args.experiment in ("step", "sine", "jitter") and any(c.is_base for c in curves):
        raise ValueError(
            f"--experiment {args.experiment}: `base` takes no depth to perturb"
        )
    if args.experiment == "step" and not (
        args.window <= args.step_at < clip_len
        and args.step_factor > 0
        and args.step_factor != 1
    ):
        raise ValueError(
            f"step needs window ({args.window}) <= step_at ({args.step_at}) < clip length "
            f"({clip_len}) and a factor > 0, != 1"
        )
    if args.experiment == "sine" and not (
        args.sine_amplitude > 1 and all(p > 2 for p in args.sine_periods)
    ):
        raise ValueError("sine needs --sine-amplitude > 1 and every period > 2 frames")
    if args.experiment == "jitter" and not args.jitter_sigma > 0:
        raise ValueError("jitter needs --jitter-sigma > 0")
    if args.experiment == "splice" and not args.splice_scale > 0:
        raise ValueError("splice needs --splice-scale > 0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=EXPERIMENTS, default="dropout")
    ap.add_argument(
        "--curve", action="append", required=True, help="LABEL=SOURCE[@DROP]"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=20, help="KV cache window (frames)")
    ap.add_argument(
        "--ref-frames", type=int, default=20, help="dropout: reference frames for s_ref"
    )
    ap.add_argument(
        "--dropout-at", type=int, default=None, help="dropout: plot lines only"
    )
    ap.add_argument("--step-factor", type=float, default=1.25)
    ap.add_argument("--step-at", type=int, default=40, help="needs >= --window")
    ap.add_argument("--sine-amplitude", type=float, default=1.2, help="peak factor > 1")
    ap.add_argument("--sine-periods", type=float, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument(
        "--jitter-sigma", type=float, default=0.1, help="std of the log factor"
    )
    ap.add_argument("--splice-scale", type=float, default=2.0)
    ap.add_argument("--base-ckpt", default=DEFAULT_BASE)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    curves = [parse_curve(c) for c in args.curve]

    bench = BenchmarkCfg()
    spec = SPECS[DATASET]
    # dropout / step / sine / jitter: the 192-frame TAE clips; splice: the
    # 90-frame main clips, two to a stream
    seqs = load_manifest(
        bench.root, spec, tae=args.experiment != "splice", tae_slice=False
    )
    if args.max_clips:
        seqs = seqs[: args.max_clips]
    if args.experiment == "splice" and len(seqs) < 2:
        raise ValueError("splice needs at least two clips")
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1:
        raise ValueError(
            f"clips of different lengths {sorted(lengths)}: cannot band them"
        )
    validate(args, curves, lengths.pop())
    if not torch.cuda.is_available():
        raise RuntimeError("scale_drift needs a CUDA device")
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    # all models resident (~5 GB each on a 48 GB A40): each clip is read,
    # resized and drawn once, and every curve streams the same views
    models = {c.label: load_curve_model(c, args.base_ckpt, device) for c in curves}
    runner = {"dropout": run_dropout, "splice": run_splice}.get(
        args.experiment, run_response
    )
    header, rows, summary = runner(args, curves, models, seqs, spec, bench, device)

    with open(args.out / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    summary = {
        "experiment": args.experiment,
        "dataset": DATASET,
        "clips": len(seqs),
        "window": args.window,
        "density": DENSITY,
        "settings": {k: v for k, v in vars(args).items() if k not in ("curve", "out")},
        **summary,
        "curve_specs": [vars(c) for c in curves],
        "commit": PROV.code_fingerprint(),
        "code_files": PROV.loaded_code(),
        "job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)
    print(f"[{args.experiment}] wrote {args.out}")


if __name__ == "__main__":
    main()
