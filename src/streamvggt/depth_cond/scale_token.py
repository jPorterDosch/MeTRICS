"""Training-side pieces of the scale token (config.ScaleTokenCfg): the readout
loss and the depth dropout that gives the token something to carry.

Kept out of sparse.py: the benchmark imports that module, and its results
record every file they import (eval.provenance.loaded_code).
"""

from __future__ import annotations

import torch


def _gt_depth(view: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """([B, H, W] GT depth, [B, H, W] bool validity) of one view, the rule
    sparse.simulate_sparse_depth uses (depth > 0 and the view's valid_mask)."""
    depth = view["depthmap"]
    if depth.dim() == 4:  # [B, H, W, 1]
        depth = depth[..., 0]
    valid = depth > 0
    if "valid_mask" in view:
        valid = valid & view["valid_mask"].to(dtype=torch.bool, device=depth.device)
    return depth, valid


def running_log_scale(views: list[dict]) -> torch.Tensor:
    """[B, S] log of the mean valid GT depth over frames 0..t: the scale of
    the scene the stream has shown so far (causal: frame t's target never
    reads a later frame). It settles as the clip goes on, the time-invariant
    part as opposed to the per-frame structure. NaN until a clip's first
    valid pixel."""
    total, count, out = 0.0, 0, []
    for view in views:
        depth, valid = _gt_depth(view)
        total = (
            total
            + torch.where(valid, depth, torch.zeros_like(depth))
            .flatten(1)
            .sum(1)
            .double()
        )
        count = count + valid.flatten(1).sum(1)
        mean = total / count.clamp(min=1)
        out.append(
            torch.where(count > 0, mean.log(), torch.full_like(mean, float("nan")))
        )
    return torch.stack(out, dim=1)


def scale_readout_loss(preds: list[dict], views: list[dict]) -> torch.Tensor:
    """Mean |readout_t - running_log_scale_t| over the frames and clips with a
    target (zero loss if none has one)."""
    s = torch.stack([p["scale_log"] for p in preds], dim=1)  # [B, S]
    target = running_log_scale(views).to(device=s.device, dtype=s.dtype)
    keep = torch.isfinite(target)
    if not keep.any():
        return s.sum() * 0.0
    return (s[keep] - target[keep]).abs().mean()


def withhold_depth_tail(views: list[dict], p: float) -> int:
    """With probability `p` per clip, remove its sparse depth from a frame
    drawn uniformly in [1, S-1] on: zero depth and a zero validity mask, the
    model's "no measurement" (as scale_drift.py's dropout). Frame 0 always
    keeps its depth, so the scale has something to start from. In place;
    returns the number of clips cut. Uses the global torch RNG, like the
    sparse simulation."""
    S = len(views)
    if p <= 0 or S < 2 or "sparse_depth" not in views[0]:
        return 0
    B = views[0]["sparse_depth"].shape[0]
    device = views[0]["sparse_depth"].device
    cut = torch.rand(B, device=device) < p
    start = torch.randint(1, S, (B,), device=device)
    for t in range(1, S):
        drop = cut & (start <= t)  # [B]
        if not drop.any():
            continue
        view = views[t]
        keep = ~drop[:, None, None]
        view["sparse_depth"] = view["sparse_depth"] * keep
        view["sparse_depth_mask"] = view["sparse_depth_mask"] & keep
    return int(cut.sum())
