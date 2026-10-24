"""Scale under perturbed sparse depth: how the model's metric scale follows,
holds or filters what it is fed (--experiment; default `dropout`).

  dropout  depth withheld from a frame on: how long the KV cache carries
           absolute scale once the sparse depth stops (below)
  step     fed depth multiplied by each --step-factor from --step-at on (one
           run per factor: does the settled gain fall as the step grows?)
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
r_t is also measured on the fed pixels themselves (r_fed): a model that
copies the fed values where it gets them but does not spread the scale to
the rest of the frame shows r_fed ~ u_t with r_t well below it.
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
conditioned frame). StreamVGGT's absolute error is drawn dashed, "not
metric": the gap to ours is what the metric conditioning buys.

Summary and figures are computed from per_frame.csv's rows, so a finished
run is redrawn without a GPU (after a change to a statistic or a plot):

    python src/scale_drift.py --replot <run dir> --out <run dir>/replot
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
from model_io import load_base_streamvggt, load_run_model  # noqa: E402

DENSITY = 0.05
DATASET = "scannet"
DEFAULT_BASE = "/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth"
EXPERIMENTS = ("dropout", "step", "sine", "jitter", "splice")
_COLORS = [
    "#2563eb",
    "#e8590c",
    "#16a34a",
    "#9333ea",
    "#6b7280",
    "#db2777",
    "#0891b2",
    "#ca8a04",
]


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
    """(model, amp): the run's checkpoint-last and its own saved precision, as
    bench_checkpoint scores it; the pretrained StreamVGGT under bf16 autocast."""
    if curve.is_base:
        return load_base_streamvggt(base_ckpt).to(device), True
    # the streaming loop calls the aggregator directly, never with a cache key
    run = load_run_model(curve.source, "last", device, encoder_cache=False)
    return run.model, bool(run.raw.get("amp", 1))


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
    mean_response, never linearly."""
    t = np.arange(len(r))
    ok = (t >= start) & np.isfinite(r)
    w = 2 * np.pi / period
    X = np.stack([np.sin(w * t[ok]), np.cos(w * t[ok]), np.ones(ok.sum())], 1)
    (A, B, _), *_ = np.linalg.lstsq(X, r[ok], rcond=None)
    sign = 1.0 if amplitude > 0 else -1.0
    return math.hypot(A, B) / abs(amplitude), math.atan2(-sign * B, sign * A)


def mean_response(fits) -> tuple[float, float]:
    """(gain, phase) of the mean complex response gain * exp(-i phase) over
    clips. Averaging the complex values, not per-clip magnitudes, keeps a
    filtered-out period's gain from sitting on the fit-noise floor, and gives
    one phase for both (a phase near 0 with clip noise stays near 0)."""
    z = np.mean([g * np.exp(-1j * ph) for g, ph in fits])
    return float(abs(z)), float(-np.angle(z))


def lags_across_periods(periods, phases) -> list[float]:
    """Lag in frames per period, the phase unwrapped from the longest period
    (smallest phase) down: a delay longer than half a short period is not
    folded into a small lag or a lead. Returned in the order given."""
    order = np.argsort(periods)[::-1]
    unwrapped = np.unwrap(np.asarray(phases, dtype=float)[order])
    lags = np.empty(len(periods))
    lags[order] = unwrapped * np.asarray(periods, dtype=float)[order] / (2 * np.pi)
    return [float(x) for x in lags]


SETTLE_TAIL = 20  # step: frames at the clip end averaged as the settled value


def step_stats(
    tracking: np.ndarray, step_at: int, tail: int = SETTLE_TAIL
) -> tuple[int | None, float]:
    """(time constant: frames from the step until the tracking curve first
    reaches 63% of the value it settles at -- None if it never does or
    settles at or below 0 -- and that settled value, the mean of the last
    `tail` frames). Speed and how much of the step is followed are reported
    apart: a curve settling at 0.6 still has a time constant."""
    settled = float(np.nanmean(tracking[-tail:]))
    t = np.arange(len(tracking))
    hit = np.where((t >= step_at) & (tracking >= 0.63 * settled))[0]
    if settled <= 0 or not len(hit):
        return None, settled
    return int(hit[0] - step_at), settled


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


def _band_line(ax, x, values, color, label, linestyle="-") -> None:
    med, lo, hi = band(values)
    ax.plot(x, med, color=color, linewidth=2, label=label, linestyle=linestyle)
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
    """(gt, views with the benchmark's sparse draw, scored mask on the GT
    grid, fed mask on the MODEL grid -- the exact pixels the model was given,
    box), built exactly as the benchmark scores the clip."""
    gt, views = _prepare_sequence(spec, seq, bench.image_size, device)
    # the benchmark's own draw for this clip (tag and seed as its scoring), so
    # every curve sees the same fed pixels
    tag = f"{spec.name}/tae/{seq.name}" if tae else f"{spec.name}/{seq.name}"
    attach_sparse_depth(views, DENSITY, tag, bench.patch_size, bench.seed, device)
    _, fed = _sparse_arrays(views)
    box = views[0].get("gt_box")  # whole-frame input, as bench_eval reads it
    # every curve is scored on the same pixels: GT-valid and never in the draw
    # (the fed ones would measure copying), on every frame; sparse_mask_to_gt
    # is deliberately generous, so this holds out every GT pixel fed
    scored = gt_valid_mask(gt, spec.max_depth) & ~model_to_gt(
        fed, gt.shape[1:], box, mask=True
    )
    return gt, views, scored, fed, box


# ---------------------------------------------------------------------------
# per_frame.csv: written by the runs, read back by the summaries (and by
# --replot, which redraws a finished run without a GPU)
# ---------------------------------------------------------------------------
HEADERS = {
    "dropout": ["curve", "clip", "frame", "scale", "rel_log_drift", "abs_log_error"],
    "response": ["curve", "perturbation", "clip", "frame", "u", "r", "q", "r_fed"],
    "splice": ["curve", "factor", "pair", "frame", "scale", "median_scaled_absrel"],
}
_TEXT = {"curve", "clip", "perturbation", "pair"}


def header_for(experiment: str) -> list[str]:
    return HEADERS.get(experiment, HEADERS["response"])


def read_rows(path: Path) -> tuple[list[str], list[tuple]]:
    """per_frame.csv back as typed tuples: names as text, frame as int,
    everything else float (empty = NaN)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for raw in reader:
            row = []
            for name, v in zip(header, raw, strict=True):
                if name in _TEXT:
                    row.append(v)
                elif name == "frame":
                    row.append(int(v))
                else:
                    row.append(float(v) if v not in ("", "nan", "None") else math.nan)
            rows.append(tuple(row))
    return header, rows


def group(header: list[str], rows: list[tuple], by: tuple[str, ...]) -> dict:
    """{(values of `by`): {column: [S] array}} in first-seen order, every
    series checked to hold frames 0..S-1 in order. A column the CSV lacks
    (r_fed before 2026-10-23) is all NaN."""
    idx = {name: i for i, name in enumerate(header)}
    out: dict = {}
    for row in rows:
        out.setdefault(tuple(row[idx[k]] for k in by), []).append(row)
    series = {}
    for key, rs in out.items():
        frames = [r[idx["frame"]] for r in rs]
        if frames != list(range(len(rs))):
            raise ValueError(f"{key}: frames not 0..{len(rs) - 1} in order")
        series[key] = {name: np.array([r[i] for r in rs]) for name, i in idx.items()}
    return series


def _column(series: dict, name: str) -> np.ndarray:
    """One series' column, all NaN if the CSV predates it."""
    if name in series:
        return series[name]
    return np.full(len(series["frame"]), np.nan)


# ---------------------------------------------------------------------------
# experiments: each run_* streams the clips and returns per_frame.csv's rows;
# summarize_* turns rows into summary.json's curves and the figure
# ---------------------------------------------------------------------------
def run_dropout(args, curves, models, seqs, spec, bench, device):
    rows = []
    for n, seq in enumerate(seqs):
        gt, views, scored, _, box = _clip(seq, spec, bench, device)
        for curve in curves:
            # shallow per-view copies: withhold_depth replaces tensors, never
            # writes into them, so the next curve still sees the full draw
            curve_views = [dict(v) for v in views]
            if not curve.is_base:
                withhold_depth(curve_views, curve.drop)
            model, amp = models[curve.label]
            pred = stream_depth(model, curve_views, args.window, amp)
            s = frame_scales(model_to_gt(pred, gt.shape[1:], box), gt, scored)
            r, a = drift_rows(s, args.ref_frames)
            rows += [
                (curve.label, seq.name, i, s[i], r[i], a[i]) for i in range(len(s))
            ]
        print(f"[dropout] clip {n + 1}/{len(seqs)} {seq.name}", flush=True)
        del views
    return rows


def summarize_dropout(args, curves, header, rows, out: Path) -> dict:
    series = group(header, rows, ("curve", "clip"))

    def stack(label, col):
        return np.stack([v[col] for (lab, _), v in series.items() if lab == label])

    rel = {c.label: stack(c.label, "rel_log_drift") for c in curves}
    absol = {c.label: stack(c.label, "abs_log_error") for c in curves}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    for k, c in enumerate(curves):
        x = np.arange(rel[c.label].shape[1])
        _band_line(ax1, x, rel[c.label], _COLORS[k % len(_COLORS)], c.label)
        if c.is_base:
            # no metric scale to hold: its |log s_t| is how far off an RGB-only
            # model starts, the gap that metric conditioning closes
            ax2.plot(
                x,
                band(absol[c.label])[0],
                color=_COLORS[k % len(_COLORS)],
                linewidth=1.5,
                linestyle="--",
                label=f"{c.label} (not metric)",
            )
        else:
            _band_line(ax2, x, absol[c.label], _COLORS[k % len(_COLORS)], c.label)
    ax1.set_ylabel("|log(s_t / s_ref)|  (drift from own early scale)")
    ax2.set_ylabel("|log s_t|  (metric scale error)")
    ax2.set_xlabel("frame index")
    drop = args.dropout_at
    for ax in (ax1, ax2):
        if drop is not None:
            # solid: the dashed style is StreamVGGT's absolute error
            ax.axvline(drop - 0.5, color="#374151", linestyle="-", linewidth=0.8)
            ax.axvline(
                drop + args.window - 0.5, color="#9ca3af", linestyle=":", linewidth=1
            )
        _style(ax)
    title = f"Scale drift, ScanNet TAE clips, KV window {args.window}"
    if drop is not None:
        title += (
            f": depth withheld from frame {drop} (solid line); no conditioned frame in "
            f"cache from {drop + args.window} (dotted)"
        )
    ax1.set_title(title, fontsize=9)
    fig.savefig(out / "scale_drift.png", dpi=150)
    plt.close(fig)
    return {
        "ref_frames": args.ref_frames,
        "curves": [
            {
                "label": c.label,
                "source": c.source,
                "drop": c.drop,
                "rel_median": band(rel[c.label])[0],
                # base: not a metric model, so not an error it could avoid
                "abs_median": band(absol[c.label])[0],
                "metric": not c.is_base,
            }
            for c in curves
        ],
    }


def _inputs(args, S: int, clip_no: int, bench) -> dict[str, np.ndarray]:
    """The log factors u_t applied to the fed depth, per perturbation."""
    t = np.arange(S)
    if args.experiment == "step":
        return {
            f"k={k:g}": np.where(t >= args.step_at, math.log(k), 0.0)
            for k in args.step_factor
        }
    if args.experiment == "sine":
        a = math.log(args.sine_amplitude)
        return {f"P={p:g}": a * np.sin(2 * np.pi * t / p) for p in args.sine_periods}
    # jitter: seeded per clip, so every curve gets the same draw on a clip
    rng = np.random.default_rng([bench.seed, clip_no])
    return {f"sigma={args.jitter_sigma:g}": args.jitter_sigma * rng.standard_normal(S)}


def run_response(args, curves, models, seqs, spec, bench, device):
    """step / sine / jitter: each perturbed run against the same model's
    unperturbed run on the same clip and draw. r is measured twice: on the
    held-out pixels (r, the scored ones, GT grid) and on the fed pixels
    themselves (r_fed, on the model grid: exactly the pixels given, no
    resampling blending in their neighbours). r_fed near u with r well below
    it is a model that copies the fed values where it is given them without
    moving the frame's scale."""
    rows = []
    for n, seq in enumerate(seqs):
        gt, views, scored, fed, box = _clip(seq, spec, bench, device)
        S = len(views)
        inputs = _inputs(args, S, n, bench)
        for c in curves:
            model, amp = models[c.label]
            native_0 = stream_depth(model, views, args.window, amp)
            pred_0 = model_to_gt(native_0, gt.shape[1:], box)
            for key, u in inputs.items():
                native = stream_depth(
                    model, scale_fed(views, np.exp(u)), args.window, amp
                )
                pred = model_to_gt(native, gt.shape[1:], box)
                r, q = response(pred, pred_0, scored)
                r_fed, _ = response(native, native_0, fed)
                rows += [
                    (c.label, key, seq.name, i, u[i], r[i], q[i], r_fed[i])
                    for i in range(S)
                ]
        print(f"[{args.experiment}] clip {n + 1}/{len(seqs)} {seq.name}", flush=True)
        del views
    return rows


def summarize_response(args, curves, header, rows, out: Path) -> dict:
    exp = args.experiment
    series = group(header, rows, ("curve", "perturbation", "clip"))

    def runs(label, key):
        return [v for (lab, pk, _), v in series.items() if lab == label and pk == key]

    # r_fed: in every CSV from 2026-10-23 on; older runs summarise without it
    cols = ("r", "r_fed") if "r_fed" in header else ("r",)
    summary = {"curves": []}
    S = len(next(iter(series.values()))["frame"])
    x = np.arange(S)

    def fittable(values: list[np.ndarray]) -> list[np.ndarray]:
        """The clips with a finite value from frame `window` on: a clip with
        none (no scored pixel) cannot be fitted and must not count as 0."""
        return [v for v in values if np.isfinite(v[args.window :]).any()]

    if exp == "step":
        # ascending, so the gain-against-size line reads left to right
        factors = sorted(args.step_factor)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, constrained_layout=True
        )
        fig_g, axg = plt.subplots(figsize=(6, 4), constrained_layout=True)
        dashes = ["-", "-.", ":"]
        for ci, c in enumerate(curves):
            per_factor = []
            settled_held, settled_fed = [], []
            for k in factors:
                rs = runs(c.label, f"k={k:g}")
                log_k = math.log(k)
                tracking = np.stack([v["r"] for v in rs]) / log_k
                fed = np.stack([_column(v, "r_fed") for v in rs]) / log_k
                q = np.stack([v["q"] for v in rs])
                track_med = band(tracking)[0]
                tau, settled = step_stats(track_med, args.step_at)
                has_fed = "r_fed" in cols
                fed_med = band(fed)[0] if has_fed else np.full(S, np.nan)
                tau_fed, settled_f = (
                    step_stats(fed_med, args.step_at) if has_fed else (None, math.nan)
                )
                settled_held.append(settled)
                settled_fed.append(settled_f)
                per_factor.append(
                    {
                        "factor": k,
                        "tracking_median": track_med,
                        "time_constant_frames": tau,
                        "tracking_settled": settled,
                        "tracking_fed_median": fed_med,
                        "time_constant_fed_frames": tau_fed,
                        "tracking_fed_settled": settled_f,
                        "structure_change_median": band(q)[0],
                    }
                )
                # colour per factor, line style per curve
                color = _COLORS[factors.index(k) % len(_COLORS)]
                name = c.label if len(curves) > 1 else ""
                _band_line(
                    ax1,
                    x,
                    tracking,
                    color,
                    f"{name} x{k:g} held-out".strip(),
                    dashes[ci % len(dashes)],
                )
                if has_fed:
                    ax1.plot(
                        x,
                        fed_med,
                        color=color,
                        linewidth=1,
                        linestyle="--",
                        label=f"{name} x{k:g} fed pixels".strip(),
                    )
                _band_line(
                    ax2, x, q, color, f"{name} x{k:g}".strip(), dashes[ci % len(dashes)]
                )
            summary["curves"].append({"label": c.label, "factors": per_factor})
            axg.plot(
                factors, settled_held, "o-", linewidth=2, label=f"{c.label} held-out"
            )
            if "r_fed" in cols:
                axg.plot(
                    factors, settled_fed, "o--", linewidth=1.5, label=f"{c.label} fed"
                )
        ax1.axhline(1.0, color="#9ca3af", linewidth=1)
        ax1.set_ylabel("r_t / log k  (fraction of the step followed)")
        ax2.set_ylabel("q_t  (structure change, scale-free)")
        ax2.set_xlabel("frame index")
        for ax in (ax1, ax2):
            ax.axvline(args.step_at - 0.5, color="#374151", linestyle="--", linewidth=1)
            _style(ax)
        ax1.set_title(
            f"Step response: fed depth x k from frame {args.step_at} "
            f"(KV window {args.window}); dashed: on the fed pixels themselves",
            fontsize=9,
        )
        fig.savefig(out / "scale_step.png", dpi=150)
        axg.axhline(1.0, color="#9ca3af", linewidth=1)
        axg.set_xscale("log", base=2)
        axg.set_xticks(factors, [f"x{k:g}" for k in factors])
        axg.set_xlabel("step factor k")
        axg.set_ylabel(f"settled r / log k  (last {SETTLE_TAIL} frames)")
        _style(axg)
        axg.set_title("Settled gain against step size", fontsize=9)
        fig_g.savefig(out / "scale_step_gain.png", dpi=150)
        plt.close(fig_g)
    elif exp == "sine":
        a = math.log(args.sine_amplitude)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
        )
        for k, c in enumerate(curves):
            entry = {"label": c.label, "periods": args.sine_periods}
            for col in cols:
                suffix, style = ("", "o-") if col == "r" else ("_fed", "o--")
                gains, phases = [], []
                for p in args.sine_periods:
                    rs = fittable([v[col] for v in runs(c.label, f"P={p:g}")])
                    gain, phase = mean_response(
                        [sine_fit(r, p, a, args.window) for r in rs]
                    )
                    gains.append(gain)
                    phases.append(phase)
                lags = lags_across_periods(args.sine_periods, phases)
                entry |= {
                    f"gain{suffix}": gains,
                    f"phase_deg{suffix}": [math.degrees(ph) for ph in phases],
                    f"lag_frames{suffix}": lags,
                }
                where = "fed pixels" if suffix else "held-out"
                for ax, ys in ((ax1, gains), (ax2, lags)):
                    ax.plot(
                        args.sine_periods,
                        ys,
                        style,
                        color=_COLORS[k % len(_COLORS)],
                        linewidth=2 if not suffix else 1.5,
                        label=f"{c.label} {where}",
                    )
            summary["curves"].append(entry)
        ax1.axhline(1.0, color="#9ca3af", linewidth=1)
        ax2.axhline(0.0, color="#9ca3af", linewidth=1)
        ax1.set_ylabel("gain  |mean complex response|")
        ax2.set_ylabel("lag (frames, phase unwrapped from long periods)")
        ax2.set_xlabel("period of the fed-depth modulation (frames)")
        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            _style(ax)
        ax1.set_title(
            f"Frequency response: fed depth x exp({a:.3f} sin(2 pi t / P)), "
            f"fitted from frame {args.window}",
            fontsize=9,
        )
        fig.savefig(out / "scale_sine.png", dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        key = f"sigma={args.jitter_sigma:g}"
        first = runs(curves[0].label, key)[0]
        ax.plot(first["u"], color="#9ca3af", linewidth=1, label="input u_t (clip 1)")
        for k, c in enumerate(curves):
            rs = runs(c.label, key)
            entry = {"label": c.label}
            for col in cols:
                suffix = "" if col == "r" else "_fed"
                ok = [v for v in rs if np.isfinite(v[col][args.window :]).any()]
                stats = [jitter_stats(v[col], v["u"], args.window) for v in ok]
                entry[f"gain_median{suffix}"] = float(np.median([g for g, _ in stats]))
                entry[f"passthrough_median{suffix}"] = float(
                    np.median([p for _, p in stats])
                )
            summary["curves"].append(entry)
            label = (
                f"{c.label}: gain {entry['gain_median']:.2f}, "
                f"passthrough {entry['passthrough_median']:.2f}"
            )
            if "passthrough_median_fed" in entry:
                label += f" (fed pixels {entry['passthrough_median_fed']:.2f})"
            ax.plot(
                rs[0]["r"], color=_COLORS[k % len(_COLORS)], linewidth=1.5, label=label
            )
        ax.set_xlabel("frame index")
        ax.set_ylabel("log scale")
        _style(ax)
        ax.set_title(
            f"Jitter: fed depth x exp({args.jitter_sigma:g} eps_t); gain and passthrough "
            f"are clip medians from frame {args.window}",
            fontsize=9,
        )
        fig.savefig(out / "scale_jitter.png", dpi=150)
    plt.close(fig)
    return summary


def run_splice(args, curves, models, seqs, spec, bench, device):
    """Clip n followed by clip n+1 (main manifest, 90 + 90 frames), the second
    one's GT and fed depth times --splice-scale; and the same pair at x1, the
    control that separates a scene cut's own disturbance from re-locking."""
    k = args.splice_scale
    rows = []
    for n in range(len(seqs)):
        a, b = seqs[n], seqs[(n + 1) % len(seqs)]
        gt_a, views_a, scored_a, _, box = _clip(a, spec, bench, device, tae=False)
        gt_b, views_b, scored_b, _, _ = _clip(b, spec, bench, device, tae=False)
        scored = np.concatenate([scored_a, scored_b])
        for factor in (k, 1.0):
            gt = np.concatenate([gt_a, gt_b * factor])
            views = views_a + scale_fed(views_b, np.full(len(views_b), factor))
            for c in curves:
                if c.is_base and factor != k:
                    continue  # no fed depth: the control is the same run
                model, amp = models[c.label]
                pred = stream_depth(model, views, args.window, amp)
                pred = model_to_gt(pred, gt.shape[1:], box)
                s = frame_scales(pred, gt, scored)
                st = median_scaled_absrel(pred, gt, scored)
                rows += [
                    (c.label, factor, f"{a.name}+{b.name}", i, s[i], st[i])
                    for i in range(len(s))
                ]
        print(f"[splice] pair {n + 1}/{len(seqs)} {a.name}+{b.name}", flush=True)
        del views_a, views_b
    return rows


def summarize_splice(args, curves, header, rows, out: Path, cut: int) -> dict:
    if "factor" not in header:
        raise ValueError("per_frame.csv without a factor column (before the x1 rows)")
    k = args.splice_scale
    series = group(header, rows, ("curve", "factor", "pair"))

    def stack(label, factor, col):
        return np.stack(
            [
                v[col]
                for (lab, f, _), v in series.items()
                if lab == label and f == factor
            ]
        )

    summary = {"splice_scale": k, "cut_frame": cut, "curves": []}
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, constrained_layout=True
    )
    for i, c in enumerate(curves):
        q = stack(c.label, k, "median_scaled_absrel")
        x = np.arange(q.shape[1])
        entry = {"label": c.label, "structure_median": band(q)[0]}
        if not c.is_base:
            e = np.abs(np.log(stack(c.label, k, "scale")))
            e1 = np.abs(np.log(stack(c.label, 1.0, "scale")))
            entry |= {
                "abs_log_error_median": band(e)[0],
                "abs_log_error_median_x1": band(e1)[0],
                "relock_frames": relock_frames(band(e)[0], band(e1)[0], cut),
            }
            _band_line(ax1, x, e, _COLORS[i % len(_COLORS)], f"{c.label} (x{k:g})")
            ax1.plot(
                x,
                band(e1)[0],
                color=_COLORS[i % len(_COLORS)],
                linewidth=1,
                linestyle=":",
                label=f"{c.label} (x1 control)",
            )
        summary["curves"].append(entry)
        _band_line(ax2, x, q, _COLORS[i % len(_COLORS)], c.label)
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
    fig.savefig(out / "scale_splice.png", dpi=150)
    plt.close(fig)
    return summary


def summarize(args, curves, header, rows, out: Path, cut: int | None) -> dict:
    if args.experiment == "dropout":
        return summarize_dropout(args, curves, header, rows, out)
    if args.experiment == "splice":
        return summarize_splice(args, curves, header, rows, out, cut)
    return summarize_response(args, curves, header, rows, out)


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
    if args.experiment == "dropout":
        drops = {c.drop for c in curves if not c.is_base and c.drop is not None}
        if any(not 0 < d < clip_len for d in drops):
            raise ValueError(f"@DROP must be in (0, {clip_len}), got {sorted(drops)}")
        if drops and args.ref_frames > min(drops):
            raise ValueError(
                f"--ref-frames {args.ref_frames} reaches past the earliest drop "
                f"{min(drops)}: s_ref would include frames without depth"
            )
        if args.dropout_at is not None and drops != {args.dropout_at}:
            raise ValueError(
                f"--dropout-at {args.dropout_at} (plot lines) must be the curves' "
                f"drop frame, got {sorted(drops)}"
            )
    if args.experiment == "step" and not (
        args.window <= args.step_at <= clip_len - SETTLE_TAIL
        and all(k > 0 and k != 1 for k in args.step_factor)
        and len({f"{k:g}" for k in args.step_factor}) == len(args.step_factor)
    ):
        raise ValueError(
            f"step needs window ({args.window}) <= step_at ({args.step_at}) and "
            f"{SETTLE_TAIL} settled frames after it in clips of {clip_len}, and "
            "distinct factors > 0, != 1"
        )
    if args.experiment == "sine" and not (
        args.sine_amplitude > 1 and all(p > 2 for p in args.sine_periods)
    ):
        raise ValueError("sine needs --sine-amplitude > 1 and every period > 2 frames")
    if args.experiment == "jitter" and not args.jitter_sigma > 0:
        raise ValueError("jitter needs --jitter-sigma > 0")
    if args.experiment == "splice" and not (
        args.splice_scale > 0 and args.splice_scale != 1
    ):
        raise ValueError("splice needs --splice-scale > 0 and != 1 (x1 is its control)")


def replot(args) -> None:
    """Summary and figures of a finished run, from its per_frame.csv and
    summary.json alone (no GPU): into --out, the source run untouched."""
    src = args.replot
    with open(src / "summary.json") as f:
        old = json.load(f)
    settings = dict(old["settings"])
    if isinstance(settings.get("step_factor"), (int, float)):
        settings["step_factor"] = [settings["step_factor"]]  # one-factor runs
    run_args = argparse.Namespace(**settings)
    curves = [Curve(**c) for c in old["curve_specs"]]
    header, rows = read_rows(src / "per_frame.csv")
    if header != header_for(run_args.experiment)[: len(header)]:
        raise ValueError(f"{src}/per_frame.csv: unexpected columns {header}")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = summarize(run_args, curves, header, rows, args.out, old.get("cut_frame"))
    summary = {
        **{k: v for k, v in old.items() if k not in ("curves", "code_files")},
        **summary,
        "replot_of": str(src),
        # the code that streamed the clips, and the code that summarised them
        "run_code_files": old.get("code_files"),
        "code_files": PROV.loaded_code(),
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)
    print(f"[replot] {run_args.experiment}: {src} -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=EXPERIMENTS, default="dropout")
    ap.add_argument("--curve", action="append", default=[], help="LABEL=SOURCE[@DROP]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--replot",
        type=Path,
        default=None,
        help="a finished run's directory: redraw it into --out (CPU, no --curve)",
    )
    ap.add_argument("--window", type=int, default=20, help="KV cache window (frames)")
    ap.add_argument(
        "--ref-frames", type=int, default=20, help="dropout: reference frames for s_ref"
    )
    ap.add_argument(
        "--dropout-at", type=int, default=None, help="dropout: plot lines only"
    )
    ap.add_argument(
        "--step-factor",
        type=float,
        nargs="+",
        default=[1.25],
        help="one step run per factor, against one unperturbed run",
    )
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
    if args.replot is not None:
        if args.curve:
            raise ValueError("--replot takes the curves from the run's summary.json")
        replot(args)
        return
    if not args.curve:
        raise ValueError("--curve is required (or --replot)")
    curves = [parse_curve(c) for c in args.curve]
    args.step_factor = sorted(args.step_factor)

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
    clip_len = lengths.pop()
    validate(args, curves, clip_len)
    if not torch.cuda.is_available():
        raise RuntimeError("scale_drift needs a CUDA device")
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    # all models resident (~5 GB each on a 48 GB A40): each clip is read,
    # resized and drawn once, and every curve streams the same views
    # one copy per checkpoint: two curves of one run share it
    loaded = {}
    for c in curves:
        if c.source not in loaded:
            loaded[c.source] = load_curve_model(c, args.base_ckpt, device)
    models = {c.label: loaded[c.source] for c in curves}
    runner = {"dropout": run_dropout, "splice": run_splice}.get(
        args.experiment, run_response
    )
    rows = runner(args, curves, models, seqs, spec, bench, device)
    header = header_for(args.experiment)
    with open(args.out / "per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    record = {
        "experiment": args.experiment,
        "dataset": DATASET,
        "clips": len(seqs),
        "window": args.window,
        "density": DENSITY,
        "settings": {
            k: v for k, v in vars(args).items() if k not in ("curve", "out", "replot")
        },
        "cut_frame": clip_len if args.experiment == "splice" else None,
        "curve_specs": [vars(c) for c in curves],
        "commit": PROV.code_fingerprint(),
        "code_files": PROV.loaded_code(),
        "job_id": os.environ.get("SLURM_JOB_ID", "none"),
    }

    def write_summary(extra: dict) -> None:
        with open(args.out / "summary.json", "w") as f:
            json.dump(_jsonable({**record, **extra}), f, indent=2, default=str)

    # first without the curves: if summarising fails, the streamed rows are
    # still recoverable with --replot, which needs only this record
    write_summary({})
    # the splice cut is the first clip's length
    write_summary(summarize(args, curves, header, rows, args.out, clip_len))
    print(f"[{args.experiment}] wrote {args.out}")


if __name__ == "__main__":
    main()
