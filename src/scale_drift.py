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
  cut      two clips back to back, both at their TRUE scale, picked so their
           median GT depths differ (a small room into a large one, and the
           reverse): how long until the model is as good on the second clip
           as after a cut into it from a scene of matching depth

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

Dropout, splice and cut score against GT, as below.

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
from eval.vda_benchmark import (  # noqa: E402
    SPECS,
    gt_stack,
    load_manifest,
    model_to_gt,
)
from model_io import load_base_streamvggt, load_run_model  # noqa: E402

DENSITY = 0.05
DATASET = "scannet"
DEFAULT_BASE = "/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth"
EXPERIMENTS = ("dropout", "step", "sine", "jitter", "splice", "cut")
# cut: pairs binned by the ratio of their median GT depths, |log| edges; the
# first bin (< x1.25) is a same-scale control, the gap to x1.5 keeps it apart
CUT_BINS = ((1.0, 1.25), (1.5, 2.0), (2.0, 2.5), (2.5, math.inf))
RECOVER_HOLD = 5  # cut: frames it must stay back, so one lucky frame does not count
CUT_ROLE_CAP = 2  # cut: times one clip may lead (or follow) within one bin
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
    source: str  # "base", "promptda", "promptda-hold" or a run directory
    drop: int | None  # first frame without sparse depth; None = always fed

    @property
    def is_base(self) -> bool:
        return self.source == "base"

    @property
    def is_promptda(self) -> bool:
        return self.source in PROMPTDA_SOURCES


def parse_curve(spec: str) -> Curve:
    """LABEL=SOURCE[@DROP]."""
    label, sep, rest = spec.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"--curve wants LABEL=SOURCE[@DROP], got {spec!r}")
    source, at, drop = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if at and not drop.isdigit():
        raise ValueError(f"--curve {spec!r}: DROP must be a frame index")
    return Curve(label, source, int(drop) if at else None)


# PromptDA (dropout only): a per-frame model that needs depth on every frame.
# `promptda` gets nothing once the depth stops (no prediction: a NaN frame);
# `promptda-hold` keeps being handed the last depth frame it had, the fallback
# a deployment of such a model would use.
PROMPTDA_SOURCES = ("promptda", "promptda-hold")


def load_curve_model(curve: Curve, base_ckpt: str, device: torch.device):
    """(model, amp): the run's checkpoint-last and its own saved precision, as
    bench_checkpoint scores it; the pretrained StreamVGGT under bf16 autocast;
    PromptDA's arm (its own precision, amp unused)."""
    if curve.is_promptda:
        from eval.baselines import arms as A
        from eval.baselines import record as R

        # through the reproduction gate, as bench_baselines builds it: PromptDA
        # is recorded close, so it runs under --allow-close semantics, and its
        # computed status goes into summary.json (promptda_status)
        record = R.load()
        arm = A.build_arm(
            "promptda", device, [("scannet", "published")], record, allow_close=True
        )
        arm.parity_status = R.computed_status(
            R.arm_entry(record, "promptda"), record["tolerance"]
        )
        return arm, None
    if curve.is_base:
        return load_base_streamvggt(base_ckpt).to(device), True
    # the streaming loop calls the aggregator directly, never with a cache key
    run = load_run_model(curve.source, "last", device, encoder_cache=False)
    return run.model, bool(run.raw.get("amp", 1))


def hold_depth(depth: np.ndarray, mask: np.ndarray, drop: int | None):
    """(depth, mask) with every frame from `drop` on replaced by frame
    drop - 1's: the last measurement, re-used while the camera moves on."""
    if drop is None:
        return depth, mask
    depth, mask = depth.copy(), mask.copy()
    depth[drop:] = depth[drop - 1]
    mask[drop:] = mask[drop - 1]
    return depth, mask


def promptda_depth(
    arm, seq, spec, views: list[dict], gt_hw, hold_from: int | None
) -> np.ndarray:
    """[S, H, W] PromptDA depth on the GT grid, exactly as bench_baselines
    benchmarks it: the manifest's native frames (read_frames; the arm does its
    own upstream sizing), prompted with each view's sparse depth at OUR model
    resolution (or the held one), mapped to the GT grid by register_to_gt, and
    TF32 off for its call (the arms were verified without it; bench_eval's
    croco import turns it on process-wide). A frame with no prompt pixel has
    no prediction (NaN, the arm's rule)."""
    import bench_baselines as BB

    frames = BB.read_frames(seq)
    depth, mask = _sparse_arrays(views)
    depth, mask = hold_depth(depth, mask, hold_from)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        raw = arm.predict(frames, (depth, mask))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return BB.register_to_gt(raw, seq, spec, frames.shape[1:3], tuple(gt_hw))


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


RECOVER_TOL = 0.02  # |log s| above the reference: "back"


def frames_until_within(
    err: np.ndarray,
    reference: np.ndarray,
    start: int = 0,
    tol: float = RECOVER_TOL,
    hold: int = 1,
) -> int | None:
    """Frames from `start` until err is within `tol` of `reference` for `hold`
    frames in a row; None if it never is. The one definition of "recovered"
    behind both splice's relock and cut's recovery."""
    ok = np.nan_to_num(err - reference, nan=np.inf) <= tol
    for t in range(start, len(ok) - hold + 1):
        if ok[t : t + hold].all():
            return t - start
    return None


def relock_frames(
    err: np.ndarray, control: np.ndarray, cut: int, tol: float = RECOVER_TOL
) -> int | None:
    """Frames after the cut until the scale error is back within `tol` of the
    same pair spliced at x1: a scene cut alone also disturbs scale, and the
    control removes that part. None if it never is."""
    return frames_until_within(err, control, cut, tol)


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
    "cut": [
        "curve",
        "pair",
        "ratio",
        "run",
        "frame",
        "scale",
        "median_scaled_absrel",
    ],
}
_TEXT = {"curve", "clip", "perturbation", "pair", "run"}


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
            model, amp = models[curve.label]
            if curve.is_promptda:
                # already on the GT grid (register_to_gt, as benchmarked)
                hold = curve.drop if curve.source == "promptda-hold" else None
                if curve.source == "promptda":
                    withhold_depth(curve_views, curve.drop)
                pred_gt = promptda_depth(
                    model, seq, spec, curve_views, gt.shape[1:], hold
                )
            else:
                if not curve.is_base:
                    withhold_depth(curve_views, curve.drop)
                pred = stream_depth(model, curve_views, args.window, amp)
                pred_gt = model_to_gt(pred, gt.shape[1:], box)
            s = frame_scales(pred_gt, gt, scored)
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
        if any(c.is_promptda for c in curves):
            title += (
                f"\nPromptDA has no cache (per frame); its hold curve keeps receiving "
                f"frame {drop - 1}'s depth"
            )
    ax1.set_title(title, fontsize=9)
    fig.savefig(out / "scale_drift.png", dpi=150)
    plt.close(fig)

    # the curves that keep their input all clip long: on the shared axes they
    # sit on top of each other, so they get a figure of their own, each panel
    # on its own scale (same colours as above)
    kept = [(k, c) for k, c in enumerate(curves) if c.drop is None and not c.is_base]
    if len(kept) > 1:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
        )
        for k, c in kept:
            x = np.arange(rel[c.label].shape[1])
            color = _COLORS[k % len(_COLORS)]
            _band_line(ax1, x, rel[c.label], color, c.label)
            _band_line(ax2, x, absol[c.label], color, c.label)
        ax1.set_ylabel("|log(s_t / s_ref)|  (drift from own early scale)")
        ax2.set_ylabel("|log s_t|  (metric scale error)")
        ax2.set_xlabel("frame index")
        for ax in (ax1, ax2):
            _style(ax)
        ax1.set_title(
            "Scale drift, ScanNet TAE clips, zoomed: every curve without a "
            "dropout (the pretrained StreamVGGT left out)",
            fontsize=9,
        )
        fig.savefig(out / "scale_drift_kept.png", dpi=150)
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
            f"(KV window {args.window})"
            + ("; dashed: on the fed pixels themselves" if "r_fed" in cols else ""),
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


def clip_depth(gt: np.ndarray, max_depth: float) -> float:
    """A clip's scale: the median GT depth over its valid pixels."""
    return float(np.median(gt[gt_valid_mask(gt, max_depth)]))


def cut_bin(ratio: float) -> int | None:
    """Index into CUT_BINS of a depth ratio (either direction), or None in a gap."""
    r = math.exp(abs(math.log(ratio)))
    for i, (lo, hi) in enumerate(CUT_BINS):
        if lo <= r < hi:
            return i
    return None


def select_cut_pairs(
    depths: list[float], per_bin: int, seed: int, cap: int = CUT_ROLE_CAP
) -> list[tuple[int, int, float]]:
    """(a, b, depth_b / depth_a) clip index pairs: up to per_bin per
    (CUT_BINS bin, direction), in a seeded random order over every ordered
    pair of distinct clips, each clip leading at most `cap` and following at
    most `cap` pairs of a bin -- the extreme bins are built from the few
    smallest and largest rooms, which would otherwise fill them. Direction
    matters (into a larger scene, into a smaller one), so both are sampled."""
    rng = np.random.default_rng(seed)
    cells: dict = {}
    for i, da in enumerate(depths):
        for j, db in enumerate(depths):
            if i == j:
                continue
            b = cut_bin(db / da)
            if b is not None:
                cells.setdefault((b, db >= da), []).append((i, j, db / da))
    out = []
    for key in sorted(cells):
        cand = cells[key]
        lead: dict = {}
        follow: dict = {}
        picked = []
        for k in rng.permutation(len(cand)):
            i, j, r = cand[k]
            if lead.get(i, 0) < cap and follow.get(j, 0) < cap:
                picked.append(cand[k])
                lead[i] = lead.get(i, 0) + 1
                follow[j] = follow.get(j, 0) + 1
                if len(picked) == per_bin:
                    break
        out += sorted(picked)
    return out


def matched_lead(depths: list[float], b: int, exclude: set[int]) -> int:
    """The clip closest in median depth to clip b (not b, not `exclude`):
    leading b with it is a scene cut without a scale change, the reference
    each spliced pair is measured against."""
    order = sorted(
        (abs(math.log(d / depths[b])), i)
        for i, d in enumerate(depths)
        if i != b and i not in exclude
    )
    return order[0][1]


def recovery_frames(
    excess: np.ndarray, tol: float = RECOVER_TOL, hold: int = RECOVER_HOLD
) -> int | None:
    """Frames after the cut until the excess error (over the matched-depth
    reference) stays within `tol` for `hold` frames; None if never."""
    return frames_until_within(excess, np.zeros_like(excess), 0, tol, hold)


def run_cut(args, curves, models, seqs, spec, bench, device):
    """Clip a then clip b, both at their true scale, for pairs whose median
    GT depths differ (select_cut_pairs); and, once per clip b, the reference
    run m then b, m the clip closest to b in depth (matched_lead). Both runs
    cut into b at the same stream position with the same cache size, so the
    difference on b's frames is clip a's scale alone, not the cut itself or
    b's frames being off the stream's first (anchor) frame."""
    (c,) = curves
    model, amp = models[c.label]
    depths = [clip_depth(gt_stack(seq, spec), spec.max_depth) for seq in seqs]
    pairs = select_cut_pairs(depths, args.cut_pairs_per_bin, bench.seed)
    print(f"[cut] {len(pairs)} pairs from {len(seqs)} clips", flush=True)

    loaded: dict = {}  # the last few clips: a pair's b is reused by its reference

    def clip(k):
        if k not in loaded:
            if len(loaded) >= 3:
                loaded.pop(next(iter(loaded)))
            loaded[k] = _clip(seqs[k], spec, bench, device, tae=False)
        return loaded[k]

    def splice(i, j):
        """(scale, structure) per frame of clip i then clip j, each clip's
        prediction mapped onto its own GT grid."""
        gt_i, views_i, scored_i, _, box_i = clip(i)
        gt_j, views_j, scored_j, _, box_j = clip(j)
        pred = stream_depth(model, views_i + views_j, args.window, amp)
        n = len(views_i)
        p_i = model_to_gt(pred[:n], gt_i.shape[1:], box_i)
        p_j = model_to_gt(pred[n:], gt_j.shape[1:], box_j)
        s = np.concatenate(
            [frame_scales(p_i, gt_i, scored_i), frame_scales(p_j, gt_j, scored_j)]
        )
        st = np.concatenate(
            [
                median_scaled_absrel(p_i, gt_i, scored_i),
                median_scaled_absrel(p_j, gt_j, scored_j),
            ]
        )
        return s, st

    rows = []
    done_ref = set()
    for n, (i, j, ratio) in enumerate(pairs):
        name = f"{seqs[i].name}+{seqs[j].name}"
        s, st = splice(i, j)
        rows += [
            (c.label, name, ratio, "spliced", t, s[t], st[t]) for t in range(len(s))
        ]
        if j not in done_ref:
            # never one of the leads cut into j: the reference must be a
            # different stream from every spliced run it is compared against
            m = matched_lead(depths, j, exclude={a for a, b, _ in pairs if b == j})
            s, st = splice(m, j)
            ref = f"{seqs[m].name}+{seqs[j].name}"
            rows += [
                (
                    c.label,
                    ref,
                    depths[j] / depths[m],
                    f"ref:{seqs[j].name}",
                    t,
                    s[t],
                    st[t],
                )
                for t in range(len(s))
            ]
            done_ref.add(j)
        print(f"[cut] pair {n + 1}/{len(pairs)} {name} depth x{ratio:.2f}", flush=True)
    return rows


def summarize_cut(args, curves, header, rows, out: Path, cut: int) -> dict:
    series = group(header, rows, ("curve", "pair", "run"))
    summary = {
        "cut_frame": cut,
        # JSON has no infinity: the open top bin's upper edge is null
        "bins": [[lo, hi if math.isfinite(hi) else None] for lo, hi in CUT_BINS],
        "recover_tol": RECOVER_TOL,
        "recover_hold": RECOVER_HOLD,
        "curves": [],
    }
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.5), sharex=True, sharey="row", constrained_layout=True
    )
    fig_r, axr = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for c in curves:
        pairs = []  # (ratio, scale excess [T], structure excess [T], recovery)
        refs = {
            run[len("ref:") :]: v
            for (lab, _, run), v in series.items()
            if lab == c.label and run.startswith("ref:")
        }
        for (lab, pair, run), v in series.items():
            if lab != c.label or run != "spliced":
                continue
            # clip b's frames in both runs: same stream position, same cache
            ref = refs[pair.split("+", 1)[1]]
            ex = np.abs(np.log(v["scale"][cut:])) - np.abs(np.log(ref["scale"][cut:]))
            ex_st = v["median_scaled_absrel"][cut:] - ref["median_scaled_absrel"][cut:]
            pairs.append((float(v["ratio"][0]), ex, ex_st, recovery_frames(ex)))
        entry = {"label": c.label, "bins": []}
        for bi, (lo, hi) in enumerate(CUT_BINS):
            for up in (True, False):
                sel = [p for p in pairs if cut_bin(p[0]) == bi and (p[0] >= 1) == up]
                if not sel:
                    continue
                ex = np.stack([p[1] for p in sel])
                ex_st = np.stack([p[2] for p in sel])
                rec = [p[3] for p in sel]
                done = sorted(r for r in rec if r is not None)
                entry["bins"].append(
                    {
                        "ratio": [lo, hi if math.isfinite(hi) else None],
                        "into": "larger" if up else "smaller",
                        "pairs": len(sel),
                        "recovered": len(done),
                        # median over pairs, a never-recovered pair counted as
                        # longer than any: None once half never recover
                        "recovery_frames_median": (
                            done[(len(sel) - 1) // 2]
                            if len(done) > (len(sel) - 1) // 2
                            else None
                        ),
                        "scale_excess_median": band(ex)[0],
                        "structure_excess_median": band(ex_st)[0],
                    }
                )
                x = np.arange(ex.shape[1])
                col = 0 if up else 1
                label = f"x{lo:g}-{hi:g}" if math.isfinite(hi) else f">= x{lo:g}"
                color = _COLORS[bi % len(_COLORS)]
                _band_line(axes[0, col], x, ex, color, f"{label} (n={len(sel)})")
                _band_line(axes[1, col], x, ex_st, color, label)
        summary["curves"].append(entry)
        for up, marker in ((True, "o"), (False, "s")):
            sel = [p for p in pairs if (p[0] >= 1) == up]
            xs = [math.log2(p[0]) for p in sel]
            T = len(sel[0][1]) if sel else 0
            ys = [T if p[3] is None else p[3] for p in sel]
            face = ["none" if p[3] is None else None for p in sel]
            for xv, yv, f in zip(xs, ys, face, strict=True):
                axr.scatter(
                    xv,
                    yv,
                    marker=marker,
                    facecolors=f or ("#2563eb" if up else "#e8590c"),
                    edgecolors="#2563eb" if up else "#e8590c",
                    s=22,
                )
        axr.scatter([], [], marker="o", color="#2563eb", label="into a larger scene")
        axr.scatter([], [], marker="s", color="#e8590c", label="into a smaller scene")
    for row, name in (
        (0, "|log s_t| above the matched cut"),
        (1, "structure above the matched cut"),
    ):
        axes[row, 0].set_ylabel(name)
        for col in (0, 1):
            axes[row, col].axhline(0.0, color="#9ca3af", linewidth=1)
            _style(axes[row, col])
    axes[0, 0].set_title("into a larger scene (depth_b > depth_a)", fontsize=9)
    axes[0, 1].set_title("into a smaller scene", fontsize=9)
    for col in (0, 1):
        axes[1, col].set_xlabel("frames since the cut")
    fig.suptitle(
        f"Scene cut at true scale, KV window {args.window}: excess error on the "
        "second clip over the same clip cut into from a scene of matching depth",
        fontsize=10,
    )
    fig.savefig(out / "scale_cut.png", dpi=150)
    plt.close(fig)
    axr.set_xlabel("log2(depth_b / depth_a)")
    axr.set_ylabel(
        f"frames to recover (within {RECOVER_TOL} for {RECOVER_HOLD}; open: never)"
    )
    _style(axr)
    fig_r.savefig(out / "scale_cut_recovery.png", dpi=150)
    plt.close(fig_r)
    return summary


def summarize(args, curves, header, rows, out: Path, cut: int | None) -> dict:
    if args.experiment == "dropout":
        return summarize_dropout(args, curves, header, rows, out)
    if args.experiment == "splice":
        return summarize_splice(args, curves, header, rows, out, cut)
    if args.experiment == "cut":
        return summarize_cut(args, curves, header, rows, out, cut)
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
    if args.experiment != "dropout" and any(c.is_promptda for c in curves):
        # its output is network(prompt / range) * range: a scaled prompt gives
        # an exactly scaled output, so step / sine / jitter only restate that
        raise ValueError("PromptDA curves only apply to --experiment dropout")
    if any(c.source == "promptda-hold" and c.drop is None for c in curves):
        raise ValueError("promptda-hold needs @DROP: the frame it starts holding from")
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
    if args.experiment == "cut":
        if len(curves) != 1:
            raise ValueError("cut takes one --curve (its figures are per model)")
        if any(c.is_base for c in curves):
            raise ValueError("cut scores metric scale: `base` has none")
        if args.cut_pairs_per_bin < 1:
            raise ValueError("--cut-pairs-per-bin must be >= 1")


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
    ap.add_argument(
        "--cut-pairs-per-bin",
        type=int,
        default=12,
        help="cut: pairs per (depth-ratio bin, direction)",
    )
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
        bench.root, spec, tae=args.experiment not in ("splice", "cut"), tae_slice=False
    )
    if args.max_clips:
        seqs = seqs[: args.max_clips]
    if args.experiment in ("splice", "cut") and len(seqs) < 2:
        raise ValueError(f"{args.experiment} needs at least two clips")
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
        # one copy per model: both PromptDA sources share one arm
        key = "promptda" if c.is_promptda else c.source
        if key not in loaded:
            loaded[key] = load_curve_model(c, args.base_ckpt, device)
    models = {
        c.label: loaded["promptda" if c.is_promptda else c.source] for c in curves
    }
    runner = {"dropout": run_dropout, "splice": run_splice, "cut": run_cut}.get(
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
        "cut_frame": clip_len if args.experiment in ("splice", "cut") else None,
        "curve_specs": [vars(c) for c in curves],
        "commit": PROV.code_fingerprint(),
        "code_files": PROV.loaded_code(),
        "job_id": os.environ.get("SLURM_JOB_ID", "none"),
        # PromptDA runs under --allow-close semantics: its parity status, as
        # bench_baselines stamps it into a benchmark's results
        "promptda_status": getattr(
            loaded.get("promptda", (None,))[0], "parity_status", None
        ),
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
