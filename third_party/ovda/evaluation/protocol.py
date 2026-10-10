"""oVDA's paper protocol: inverse-depth alignment and pixel-weighted metrics.

Inputs are NumPy arrays at the ground-truth resolution, with GT in metres.
No model, prediction files, or CSV logs are required by these functions.
"""

import numpy as np


def fit_alignment(inverse_depth, ground_truth, valid):
    """Fit a*p+b to 1/GT and return the original (scale, shift) convention.

    The fitting mask includes valid GT beyond 80 m. Keep the input dtype:
    historical predictions and GT were float32. Parameters are fitted once
    per sequence, using either frame 0 or all frames.
    """
    if not np.any(valid):
        raise ValueError("No valid pixels for alignment")
    values = inverse_depth[valid]
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite predictions in alignment mask")
    design = np.column_stack((values, np.ones(values.size, dtype=values.dtype)))
    a, b = np.linalg.lstsq(design, 1.0 / ground_truth[valid], rcond=None)[0]
    if a == 0:
        return float("inf"), 0.0
    return float(1.0 / a), float(-b / a)


def aligned_depth(inverse_depth, scale, shift):
    """Apply alignment, invert, then clip predicted depth to [0, 80] metres.

    Only exactly zero inverse depths become 1e-4. Negative inverse depths
    become zero metric depth after clipping and remain errors in scoring.
    """
    inverse = (inverse_depth - shift) / scale
    inverse = np.where(inverse == 0, 1e-4, inverse)
    return np.clip(1.0 / inverse, 0.0, 80.0)


def evaluate_sequence(inverse_depth, ground_truth, dataset, alignment="first"):
    """Return metric totals for one ordered sequence of shape (frames, H, W).

    ``dataset`` is 'sintel', 'kitti', or 'bonn'. ``alignment='first'`` fits
    frame 0; 'all' fits the whole sequence. Both score frames 1 onward.
    Predictions must already match GT resolution. Pass returned totals to
    ``summarize`` to obtain pixel-weighted dataset metrics.

    This small reference implementation holds a sequence in memory; it is
    intended to explain the protocol, rather than manage model inference.
    """
    if inverse_depth.shape != ground_truth.shape or ground_truth.ndim != 3:
        raise ValueError("Prediction and GT must have matching (frames, H, W) shapes")
    if alignment not in ("first", "all"):
        raise ValueError("Alignment must be 'first' or 'all'")
    limit = {"sintel": 10000.0, "kitti": 255.9, "bonn": 100.0}[dataset]
    valid = (ground_truth > 0) & (ground_truth < limit)
    fit_frames = slice(0, 1) if alignment == "first" else slice(None)
    scale, shift = fit_alignment(
        inverse_depth[fit_frames], ground_truth[fit_frames], valid[fit_frames]
    )

    totals = {"pixels": 0, "abs_rel_sum": 0.0, "delta1_outliers": 0}
    for frame in range(1, len(ground_truth)):
        mask = valid[frame] & (ground_truth[frame] < 80.0)
        prediction = aligned_depth(inverse_depth[frame], scale, shift)[mask]
        target = ground_truth[frame][mask]
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Non-finite predictions at scoring pixels")
        with np.errstate(divide="ignore"):
            ratio = np.maximum(prediction / target, target / prediction)
        totals["pixels"] += int(target.size)
        totals["abs_rel_sum"] += float(
            np.sum(np.abs(prediction - target) / target, dtype=np.float64)
        )
        totals["delta1_outliers"] += int(np.count_nonzero(ratio > 1.25))
    return totals


def summarize(sequence_totals):
    """Combine totals across sequences; every valid pixel has equal weight.

    Delta1 includes the <= 1.25 boundary, matching the historical evaluator.
    Float64 error accumulation can differ slightly from old float32 sums.
    """
    totals = list(sequence_totals)
    pixels = sum(row["pixels"] for row in totals)
    if pixels == 0:
        raise ValueError("No valid scoring pixels")
    return {
        "AbsRel": sum(row["abs_rel_sum"] for row in totals) / pixels,
        "delta1": 1.0 - sum(row["delta1_outliers"] for row in totals) / pixels,
    }
