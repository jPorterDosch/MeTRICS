"""Scoring protocols for the video-depth benchmark.

Pure functions on numpy arrays -- no model, no file I/O -- so every protocol
is unit-testable on synthetic sequences and every arm (ours, VDA, PromptDA)
is scored by the same code path. Depths are [S,H,W] float32 in metres at GT
resolution unless a docstring says otherwise.

Three protocols, kept apart because they answer different questions:

  published       Video Depth Anything's scoring, verbatim: one least-squares
                  scale+shift per VIDEO fitted in DISPARITY space against the
                  dense GT, then AbsRel / delta1 / RMSE per frame, averaged
                  over frames. The only protocol under which numbers from
                  their tables are comparable to ours. The prediction enters
                  as disparity (VDA's native output); a depth model passes
                  1/depth -- see published_metrics.
  sparse_aligned  One scale+shift per FRAME fitted on the PROMPT pixels only
                  (the sparse depth a sensor would give), scored against the
                  dense GT with the prompt pixels held out. Causal, and
                  symmetric across arms: a model that consumes the prompt and
                  one that only sees it post hoc get the same pixels.
  metric          No alignment at all: raw metric depth against GT.

Plus two temporal metrics on the published-aligned depth:
  tae_vda         VDA's TAE (bidirectional reprojection with GT K/poses,
                  relative error, x100), vendored.
  tae_ours        the repo's own TAE (eval.temporal_consistency.metrics.tae),
                  GT-masked, not scaled.

The metric functions (AbsRel, RMSE, delta1) are VDA's own, loaded from the
vendored third_party/video_depth_anything/benchmark/eval/metric.py so the
per-frame reduction is theirs and not a re-implementation of it.
"""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass

import numpy as np
import torch

from eval.temporal_consistency.metrics import tae as tae_repo

_VDA_EVAL = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "video_depth_anything"
    / "benchmark"
    / "eval"
)

# Both are loaded by path rather than via sys.path: the vendored directory
# holds an eval.py, which would shadow THIS package (src/eval) for any later
# `import eval.*`.
_METRIC_MODULE = None
_TAE_MODULE = None


def _load_by_path(name: str, path: pathlib.Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"vendored VDA benchmark file missing: {path} (see third_party/"
            "video_depth_anything/README_VENDORED.md)"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vda_metric():
    global _METRIC_MODULE
    if _METRIC_MODULE is None:
        _METRIC_MODULE = _load_by_path("vda_benchmark_metric", _VDA_EVAL / "metric.py")
    return _METRIC_MODULE


def vda_tae_module():
    global _TAE_MODULE
    if _TAE_MODULE is None:
        _TAE_MODULE = _load_by_path("vda_benchmark_eval_tae", _VDA_EVAL / "eval_tae.py")
    return _TAE_MODULE


# Every protocol clips the scored prediction to this range, as VDA's eval.py
# does (a_min=1e-3, a_max=dataset max depth).
DEPTH_FLOOR = 1e-3


def _finite_or_floor(pred: np.ndarray) -> np.ndarray:
    """A non-finite prediction pixel (a bf16 head can emit one on a long
    stream) must neither crash the least-squares fit nor vanish: it is
    excluded from every fit and scored as DEPTH_FLOOR, i.e. as an error."""
    return np.where(np.isfinite(pred), pred, DEPTH_FLOOR)

METRIC_NAMES = ("abs_rel", "rmse", "delta1")


@dataclass(frozen=True)
class FrameMetrics:
    abs_rel: float
    rmse: float
    delta1: float
    frames: int  # frames that had at least one scored pixel

    def as_dict(self) -> dict[str, float]:
        return {
            "abs_rel": self.abs_rel,
            "rmse": self.rmse,
            "delta1": self.delta1,
            "frames": float(self.frames),
        }


_EMPTY = FrameMetrics(float("nan"), float("nan"), float("nan"), 0)


def gt_valid_mask(gt: np.ndarray, max_depth: float) -> np.ndarray:
    """VDA's validity: 1e-3 < gt < max_depth (their eval.py, verbatim)."""
    return np.logical_and(gt > 1e-3, gt < max_depth)


def _check_seq(name: str, arr: np.ndarray, like: np.ndarray | None = None) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must be [S,H,W], got shape {arr.shape}")
    if like is not None and arr.shape != like.shape:
        raise ValueError(f"{name} shape {arr.shape} != gt shape {like.shape}")


def vda_frame_metrics(
    pred_depth: np.ndarray, gt: np.ndarray, valid: np.ndarray
) -> FrameMetrics:
    """AbsRel / RMSE / delta1 with VDA's metric.py: each metric is a per-frame
    mean over the valid pixels, then a mean over the frames that have any.
    pred_depth must already be clipped/aligned by the caller."""
    _check_seq("pred_depth", pred_depth, gt)
    _check_seq("valid", valid, gt)
    n = valid.sum((-1, -2))
    keep = n > 0
    if not keep.any():
        return _EMPTY
    m = vda_metric()
    pred_t = torch.from_numpy(np.ascontiguousarray(pred_depth[keep])).double()
    gt_t = torch.from_numpy(np.ascontiguousarray(gt[keep])).double()
    valid_t = torch.from_numpy(np.ascontiguousarray(valid[keep]))
    # their functions divide by GT inside the masked-out region too, so an
    # invalid GT pixel of 0 would produce inf*0 = nan before the mask zeroes
    # it; substitute 1 there (the value is then discarded by the mask)
    gt_safe = torch.where(valid_t, gt_t, torch.ones_like(gt_t))
    return FrameMetrics(
        abs_rel=float(m.abs_relative_difference(pred_t, gt_safe, valid_t)),
        rmse=float(m.rmse_linear(pred_t, gt_safe, valid_t)),
        delta1=float(m.delta1_acc(pred_t, gt_safe, valid_t)),
        frames=int(keep.sum()),
    )


def vda_align_disparity(
    pred_disp: np.ndarray, gt: np.ndarray, max_depth: float
) -> np.ndarray:
    """VDA's per-video alignment, verbatim from eval.py::eval_depthcrafter:
    one (scale, shift) by least squares in disparity space over every valid
    pixel of the sequence, then back to depth, clipped to [1e-3, max_depth].
    Returns the aligned DEPTH [S,H,W]."""
    _check_seq("pred_disp", pred_disp, gt)
    fit = gt_valid_mask(gt, max_depth) & np.isfinite(pred_disp)
    if fit.sum() < 2:
        return np.full_like(gt, DEPTH_FLOOR, dtype=np.float32)
    gt_disp_masked = 1.0 / (gt[fit].reshape((-1, 1)).astype(np.float64) + 1e-8)
    infs = np.clip(_finite_or_floor(pred_disp), a_min=DEPTH_FLOOR, a_max=None)
    pred_disp_masked = infs[fit].reshape((-1, 1)).astype(np.float64)
    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    scale, shift = X
    aligned_pred = scale * infs + shift
    aligned_pred = np.clip(aligned_pred, a_min=DEPTH_FLOOR, a_max=None)
    pred_depth = np.zeros_like(aligned_pred)
    pos = aligned_pred > 0
    pred_depth[pos] = 1.0 / aligned_pred[pos]
    return np.clip(pred_depth, a_min=DEPTH_FLOOR, a_max=max_depth).astype(np.float32)


def published_metrics(
    pred_disp: np.ndarray, gt: np.ndarray, max_depth: float
) -> tuple[FrameMetrics, np.ndarray]:
    """The `published` protocol. pred_disp is the prediction AS DISPARITY --
    VDA's native output; for a depth model pass depth_to_disparity(depth).
    Returns (metrics, aligned depth) -- the aligned depth is what both TAEs
    are computed on."""
    aligned = vda_align_disparity(pred_disp, gt, max_depth)
    return vda_frame_metrics(aligned, gt, gt_valid_mask(gt, max_depth)), aligned


def depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    """1/depth with the same floor VDA applies to its own disparity."""
    return 1.0 / np.clip(depth.astype(np.float64), DEPTH_FLOOR, None)


def metric_metrics(pred_depth: np.ndarray, gt: np.ndarray, max_depth: float) -> FrameMetrics:
    """The `metric` protocol: no alignment, raw metric depth against GT."""
    _check_seq("pred_depth", pred_depth, gt)
    clipped = np.clip(_finite_or_floor(pred_depth), DEPTH_FLOOR, max_depth).astype(np.float32)
    return vda_frame_metrics(clipped, gt, gt_valid_mask(gt, max_depth))


def affine_fit(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares (s, t) with s*pred + t ~ target over 1-D arrays.
    A constant prediction has no identifiable scale: s=1, t = mean offset."""
    p = pred.astype(np.float64).reshape(-1)
    g = target.astype(np.float64).reshape(-1)
    ok = np.isfinite(p) & np.isfinite(g)
    p, g = p[ok], g[ok]
    if p.size < 2:
        raise ValueError(f"affine_fit needs at least 2 finite samples, got {p.size}")
    A = np.stack([p, np.ones_like(p)], axis=-1)
    if np.ptp(p) < 1e-12:
        return 1.0, float(np.mean(g - p))
    s, t = np.linalg.lstsq(A, g, rcond=None)[0]
    return float(s), float(t)


def sparse_aligned_metrics(
    pred_native: np.ndarray,
    prompt_depth: np.ndarray,
    prompt_mask: np.ndarray,
    pred_gt_res: np.ndarray,
    prompt_mask_gt_res: np.ndarray,
    gt: np.ndarray,
    max_depth: float,
    min_prompt_pixels: int = 2,
) -> FrameMetrics:
    """The `sparse_aligned` protocol.

    Per frame, (s, t) is fitted on the prompt pixels at the model's own
    resolution (pred_native / prompt_depth / prompt_mask, [S,h,w]) and applied
    to the prediction resized to GT resolution (pred_gt_res, [S,H,W]); the
    frame is scored on GT-valid pixels EXCLUDING the prompt (prompt_mask_gt_res,
    the prompt mask carried to GT resolution). Holding the prompt out matters
    where the "dense" GT is itself sparse (KITTI LiDAR): scoring the pixels
    the model was handed would measure copying, not completion.

    A frame with fewer than min_prompt_pixels prompt pixels cannot be aligned
    and is dropped (counted out of `frames`)."""
    _check_seq("pred_native", pred_native, prompt_depth)
    _check_seq("prompt_mask", prompt_mask, prompt_depth)
    _check_seq("pred_gt_res", pred_gt_res, gt)
    _check_seq("prompt_mask_gt_res", prompt_mask_gt_res, gt)
    S = gt.shape[0]
    aligned = np.full_like(gt, DEPTH_FLOOR, dtype=np.float32)
    valid = gt_valid_mask(gt, max_depth) & ~prompt_mask_gt_res
    for i in range(S):
        m = prompt_mask[i] & np.isfinite(pred_native[i])
        if int(m.sum()) < min_prompt_pixels:
            valid[i] = False
            continue
        # design decision: the fit lives in each model's NATIVE output space
        # (depth here); a disparity model fits 1/depth against 1/prompt
        s, t = affine_fit(pred_native[i][m], prompt_depth[i][m])
        aligned[i] = np.clip(s * _finite_or_floor(pred_gt_res[i]) + t, DEPTH_FLOOR, max_depth)
    return vda_frame_metrics(aligned, gt, valid)


def _relative_pose(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T_b_a = inv(pose_b) @ pose_a for cam2world poses, as (R, t)."""
    T = np.linalg.inv(pose_b) @ pose_a
    return T[:3, :3], T[:3, 3]


def tae_vda(
    depth: np.ndarray,
    Ks: list[np.ndarray],
    poses: list[np.ndarray],
    device: torch.device | str | None = None,
) -> float:
    """VDA's TAE over a sequence, verbatim from eval_tae.py::eval_TAE minus
    the file I/O: for every adjacent pair, project frame i's depth into frame
    i+1 with the GT relative pose and intrinsics and take the mean absolute
    relative error against frame i+1's prediction, both directions, averaged,
    x100. `depth` is the published-aligned depth (their pipeline aligns before
    the TAE, so this takes the output of published_metrics). Poses are
    cam2world 4x4; K is the 3x3 of frame i, as in their code (no per-pair
    intrinsics change is modelled). NOTE their tae_torch returns 0, not NaN,
    for a pair with no overlap; kept. Runs on `device` (default: cuda when
    available -- 170 frames at 464x618 is minutes on a CPU)."""
    _check_seq("depth", depth)
    S = depth.shape[0]
    if S < 2:
        return float("nan")
    if len(Ks) != S or len(poses) != S:
        raise ValueError(f"tae_vda: {S} frames but {len(Ks)} Ks / {len(poses)} poses")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = vda_tae_module()
    depth_t = torch.from_numpy(np.ascontiguousarray(depth)).double().to(device)
    ones = torch.ones(depth.shape[1:], dtype=torch.bool, device=device)
    error_sum = 0.0
    for i in range(S - 1):
        d1, d2 = depth_t[i], depth_t[i + 1]
        R_2_1, t_2_1 = _relative_pose(poses[i], poses[i + 1])
        R_1_2, t_1_2 = _relative_pose(poses[i + 1], poses[i])
        K = np.asarray(Ks[i], dtype=np.float64)
        e1 = mod.tae_torch(d1, d2, torch.from_numpy(R_2_1).double().to(device), t_2_1, K, ones)
        e2 = mod.tae_torch(d2, d1, torch.from_numpy(R_1_2).double().to(device), t_1_2, K, ones)
        error_sum += float(e1) + float(e2)
    return error_sum / (2 * (S - 1)) * 100.0


def tae_ours(
    depth: np.ndarray,
    valid: np.ndarray,
    Ks: list[np.ndarray],
    poses: list[np.ndarray],
) -> tuple[float, float]:
    """The repo's TAE (eval.temporal_consistency.metrics.tae) over a sequence:
    symmetric relative reprojection error between adjacent frames, restricted
    to GT-valid pixels, returned as (mean-abs, mean-sq) over the finite pairs.
    Same aligned depth as tae_vda so the two differ only in definition."""
    _check_seq("depth", depth)
    _check_seq("valid", valid, depth)
    S = depth.shape[0]
    if S < 2:
        return float("nan"), float("nan")
    if len(Ks) != S or len(poses) != S:
        raise ValueError(f"tae_ours: {S} frames but {len(Ks)} Ks / {len(poses)} poses")
    i2l = []
    for K, pose in zip(Ks, poses):
        k4 = np.eye(4, dtype=np.float32)
        k4[:3, :3] = np.asarray(K, dtype=np.float32)[:3, :3]
        i2l.append(np.asarray(pose, dtype=np.float32) @ np.linalg.inv(k4))
    abs_errs, sq_errs = [], []
    for i in range(S - 1):
        a, sq = tae_repo(
            depth[i].astype(np.float32),
            valid[i],
            i2l[i],
            depth[i + 1].astype(np.float32),
            valid[i + 1],
            i2l[i + 1],
        )
        if np.isfinite(a):
            abs_errs.append(float(a))
        if np.isfinite(sq):
            sq_errs.append(float(sq))
    return (
        float(np.mean(abs_errs)) if abs_errs else float("nan"),
        float(np.mean(sq_errs)) if sq_errs else float("nan"),
    )
