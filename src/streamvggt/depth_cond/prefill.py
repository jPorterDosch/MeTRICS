"""Nearest-measurement prefill of a sparse map, on the GPU.

The conditioner's prefill encoder (EncoderType.PREFILL_CONV) wants every pixel
to carry a value before the conv stem sees the map: the value of the nearest
measurement, plus how far away that measurement is, so the stem can tell a
reading from a long extrapolation. Done with the jump flooding algorithm
(Rong & Tan 2006): log2(max(H, W)) passes, each pulling the best seed from the
8 neighbours at the current step, all tensor ops, no host sync. JFA is not
exact in every pixel (a seed can be missed along a diagonal), but the error
is rare and at most a pixel or two of distance; the unit test bounds it
against scipy's exact EDT.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def _shift(t: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    """t[..., y - dy, x - dx] with `fill` outside: each pixel looks at the
    neighbour `step` pixels away in direction (dy, dx)."""
    H, W = t.shape[-2:]
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # left, right, top, bottom
    p = F.pad(t, pad, value=fill)
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return p[..., y0 : y0 + H, x0 : x0 + W]


def nearest_fill(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """values, mask: [N, H, W] (mask bool, True = measured). Returns
    (filled, dist): filled[n, y, x] is values at the nearest measured pixel of
    frame n, dist its Euclidean distance in pixels (0 at measured pixels). A
    frame with no measurement at all gets filled = 0 and dist = H + W, so the
    stem sees "nothing anywhere" rather than a stale number."""
    if values.shape != mask.shape or values.dim() != 3:
        raise ValueError(
            f"nearest_fill wants matching [N,H,W] tensors, got {tuple(values.shape)} "
            f"and {tuple(mask.shape)}"
        )
    N, H, W = mask.shape
    device = mask.device
    far = float(H + W)  # a coordinate this far off-grid loses every comparison
    ys = (
        torch.arange(H, device=device, dtype=torch.float32)
        .view(1, H, 1)
        .expand(N, H, W)
    )
    xs = (
        torch.arange(W, device=device, dtype=torch.float32)
        .view(1, 1, W)
        .expand(N, H, W)
    )
    none = torch.full((N, H, W), -far, device=device, dtype=torch.float32)
    seed_y = torch.where(mask, ys, none)
    seed_x = torch.where(mask, xs, none)
    best = (seed_y - ys) ** 2 + (seed_x - xs) ** 2

    step = 1 << max(int(math.ceil(math.log2(max(H, W)))) - 1, 0)
    steps = []
    while step >= 1:
        steps.append(step)
        step //= 2
    steps.append(1)  # JFA+1: one extra unit pass catches most of the misses
    for step in steps:
        for dy, dx in _NEIGHBOURS:
            cy = _shift(seed_y, dy * step, dx * step, -far)
            cx = _shift(seed_x, dy * step, dx * step, -far)
            d2 = (cy - ys) ** 2 + (cx - xs) ** 2
            better = d2 < best
            seed_y = torch.where(better, cy, seed_y)
            seed_x = torch.where(better, cx, seed_x)
            best = torch.where(better, d2, best)

    has_seed = mask.flatten(1).any(dim=1).view(N, 1, 1)
    idx = (seed_y.clamp(min=0) * W + seed_x.clamp(min=0)).long().flatten(1)
    filled = values.flatten(1).gather(1, idx).view(N, H, W)
    filled = torch.where(has_seed, filled, torch.zeros_like(filled))
    dist = torch.where(has_seed, best.sqrt(), torch.full_like(best, far))
    return filled, dist
