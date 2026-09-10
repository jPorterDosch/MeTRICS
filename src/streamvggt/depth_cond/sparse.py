"""Simulate sparse metric depth from dense GT depthmaps via patch masking.

Real deployments feed sparse metric depth from robot-mounted sensors; for
training on RGB-D datasets we reveal a random subset of patches of the GT
depth (MAE-style), controlled by SparseSimMode:

  NONE      -- no masking: the full dense GT depth is passed as conditioning.
  RANDOM    -- random visible patches, resampled independently per frame.
  TUBE_MASK -- one patch mask sampled per clip and shared by every frame
               (a "tube" through time, matching video-MAE terminology).
  PIXEL_FREQ -- per-pixel Bernoulli draws from an empirical validity-frequency map.

This is training wiring, not dataset generation: it runs on already-loaded
batches. Determinism is controlled by the global torch seed.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .config import SparseSimMode


_FREQ_RAW: dict[str, torch.Tensor] = {}
_FREQ_RESAMPLED: dict[tuple[str, int, int, str], torch.Tensor] = {}


def _check_freq_density(
    freq: torch.Tensor, path: str, mask_ratio: float, tol: float
) -> None:
    mean_invalid = 1.0 - freq.mean().item()
    if abs(mask_ratio - mean_invalid) > tol:
        raise ValueError(
            f"sim_mask_ratio={mask_ratio:.3f} but freq map {path} has mean "
            f"invalid fraction {mean_invalid:.3f}; PIXEL_FREQ matches the map "
            "as-is -- set --depth-cond.sim-mask-ratio "
            f"{round(mean_invalid, 2):.2f}"
        )


def load_freq_map(path: str, mask_ratio: float, tol: float = 0.02) -> torch.Tensor:
    """Load and validate an empirical validity-frequency map on CPU."""
    with np.load(path) as artifact:
        if "freq" not in artifact:
            raise ValueError(f"freq map {path} must contain key 'freq'")
        array = np.asarray(artifact["freq"])
    if array.ndim != 2:
        raise ValueError(f"freq map {path} must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"freq map {path} must contain only finite values")
    if not ((array >= 0.0) & (array <= 1.0)).all():
        raise ValueError(f"freq map {path} values must be in [0, 1]")
    freq = torch.from_numpy(array.astype(np.float32, copy=True))
    _check_freq_density(freq, path, mask_ratio, tol)
    return freq


def _freq_for(
    path: str,
    mask_ratio: float,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    if path not in _FREQ_RAW:
        _FREQ_RAW[path] = load_freq_map(path, mask_ratio)
    else:
        _check_freq_density(_FREQ_RAW[path], path, mask_ratio, 0.02)

    key = (path, H, W, str(device))
    if key not in _FREQ_RESAMPLED:
        m = _FREQ_RAW[path]
        if H > W:
            m = torch.rot90(m, k=-1)
        # The source and target aspect ratios are close; plain resize distortion
        # is negligible, so no crop or padding is needed.
        m = F.interpolate(
            m[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        _FREQ_RESAMPLED[key] = m.clamp(0.0, 1.0).to(device=device)
    return _FREQ_RESAMPLED[key]


def _patch_mask(
    B: int, H: int, W: int, patch_size: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Random per-sample patch visibility mask [B,H,W] (True = visible).
    Patches are patch_size x patch_size cells; ceil(num_patches*(1-ratio))
    patches are kept visible per sample."""
    gh = (H + patch_size - 1) // patch_size
    gw = (W + patch_size - 1) // patch_size
    n_patches = gh * gw
    n_visible = max(1, round(n_patches * (1.0 - mask_ratio)))
    scores = torch.rand(B, n_patches, device=device)
    keep = scores.argsort(dim=1)[:, :n_visible]
    grid = torch.zeros(B, n_patches, dtype=torch.bool, device=device)
    grid.scatter_(1, keep, True)
    grid = grid.reshape(B, gh, gw)
    mask = grid.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )
    return mask[:, :H, :W]


def simulate_sparse_depth(
    views: list[dict],
    mode: SparseSimMode,
    patch_size: int,
    mask_ratio: float,
    freq_map_path: str = "",
) -> list[dict]:
    """For each view dict with a dense 'depthmap' [B,H,W] and no 'sparse_depth',
    add in place:
      view['sparse_depth']      [B,H,W]  (0 where masked or GT-invalid)
      view['sparse_depth_mask'] [B,H,W]  bool
    Only pixels that are GT-valid AND patch-visible count as measurements."""
    mode = SparseSimMode(mode)
    tube_mask: torch.Tensor | None = None
    for view in views:
        if "sparse_depth" in view or "depthmap" not in view:
            continue
        depth = view["depthmap"]
        if depth.dim() == 4:  # [B,H,W,1]
            depth = depth[..., 0]
        B, H, W = depth.shape
        valid = depth > 0
        if "valid_mask" in view:
            valid = valid & view["valid_mask"].to(dtype=torch.bool, device=depth.device)

        match mode:
            case SparseSimMode.NONE:
                visible = torch.ones_like(valid)
            case SparseSimMode.RANDOM:
                visible = _patch_mask(B, H, W, patch_size, mask_ratio, depth.device)
            case SparseSimMode.TUBE_MASK:
                if tube_mask is None or tube_mask.shape != valid.shape:
                    tube_mask = _patch_mask(
                        B, H, W, patch_size, mask_ratio, depth.device
                    )
                visible = tube_mask
            case SparseSimMode.PIXEL_FREQ:
                if not freq_map_path:
                    raise ValueError(
                        "freq_map_path is required for SparseSimMode.PIXEL_FREQ"
                    )
                # Naive per-pixel Bernoulli: matches the marginal p(x,y) but real SPOT
                # holes are spatially clumped, so simulated masks are finer-grained than
                # deployment ones. Future upgrade (rejected for v1): threshold blurred
                # Gaussian noise at Phi^-1(p(x,y)) -- exact marginals with autocorrelation
                # length fit from real masks, optional AR(1) rho over time.
                p = _freq_for(freq_map_path, mask_ratio, H, W, depth.device)
                visible = torch.rand(B, H, W, device=depth.device) < p
            case _:
                raise ValueError(f"unknown sparse simulation mode: {mode!r}")

        mask = valid & visible
        view["sparse_depth"] = depth * mask
        view["sparse_depth_mask"] = mask
    return views
