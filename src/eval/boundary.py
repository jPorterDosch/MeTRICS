"""Depth-boundary F1: are the prediction's depth discontinuities where the
GT's are? An edge pixel is one whose log-depth differs from a 4-neighbour by
more than `threshold` (0.1 = a 10% depth jump), both pixels valid. Precision
and recall are matched within `tolerance` pixels (a max-pool dilation of the
other map), as in the depth boundary error of Koch et al. (iBims-1) but
symmetric and without the distance weighting. Returns NaN for a frame whose
GT has no edge at all.

Also the training-time uptake monitors for a model fed sparse depth:
fed_abs_rel, the relative error at the fed pixels themselves, and
fed_log_scale, |log s| for the per-frame scale s = median(GT / pred) over the
fed pixels. A model that honours its hint has both near 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from streamvggt.loss.gradient_loss import prealign_scale


def _edges(
    depth: torch.Tensor, valid: torch.Tensor, threshold: float, log_eps: float
) -> torch.Tensor:
    """[N,H,W] depth, bool valid -> bool edge map: a pixel with a valid
    right or lower neighbour whose log-depth differs by > threshold (both
    pixels of the pair are marked)."""
    ld = torch.log(depth.clamp(min=log_eps))
    edge = torch.zeros_like(valid)
    dx = (ld[:, :, 1:] - ld[:, :, :-1]).abs() > threshold
    dx &= valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= dx
    edge[:, :, :-1] |= dx
    dy = (ld[:, 1:, :] - ld[:, :-1, :]).abs() > threshold
    dy &= valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, 1:, :] |= dy
    edge[:, :-1, :] |= dy
    return edge


def _dilate(edge: torch.Tensor, tolerance: int) -> torch.Tensor:
    if tolerance <= 0:
        return edge
    k = 2 * tolerance + 1
    return (
        F.max_pool2d(edge.float().unsqueeze(1), k, stride=1, padding=tolerance)[:, 0]
        > 0
    )


def boundary_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.1,
    tolerance: int = 1,
    log_eps: float = 1e-3,
) -> torch.Tensor:
    """pred, gt: [N,H,W] depth; valid: [N,H,W] bool. Returns [N] F1 per
    frame, NaN where the GT has no edge."""
    if pred.shape != gt.shape or pred.shape != valid.shape:
        raise ValueError(
            f"boundary_f1 wants matching [N,H,W], got {tuple(pred.shape)}, "
            f"{tuple(gt.shape)}, {tuple(valid.shape)}"
        )
    pe = _edges(pred, valid, threshold, log_eps)
    ge = _edges(gt, valid, threshold, log_eps)
    gt_count = ge.flatten(1).sum(1).float()
    pred_count = pe.flatten(1).sum(1).float()
    hit_p = (pe & _dilate(ge, tolerance)).flatten(1).sum(1).float()
    hit_g = (ge & _dilate(pe, tolerance)).flatten(1).sum(1).float()
    precision = torch.where(pred_count > 0, hit_p / pred_count.clamp(min=1), 0.0)
    recall = hit_g / gt_count.clamp(min=1)
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp(min=1e-12),
        0.0,
    )
    return torch.where(gt_count > 0, f1, torch.nan)


def fed_pixel_metrics(
    pred: torch.Tensor, gt: torch.Tensor, fed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """pred, gt: [N,H,W]; fed: [N,H,W] bool, the pixels whose GT the model
    was given. Returns (fed_abs_rel [N], fed_log_scale [N]), NaN for a frame
    with no fed pixel."""
    use = fed & (gt > 0) & (pred > 0)
    n = use.flatten(1).sum(1)
    rel = torch.where(use, (pred - gt).abs() / gt.clamp(min=1e-6), 0.0)
    abs_rel = rel.flatten(1).sum(1) / n.clamp(min=1)
    # the same per-frame median-ratio fit the edge loss pre-aligns with
    log_scale = torch.log(prealign_scale(gt, pred, use).clamp(min=1e-6)).abs()
    nan = torch.full_like(abs_rel, torch.nan)
    return torch.where(n > 0, abs_rel, nan), torch.where(n > 0, log_scale, nan)
