"""Per-frame causal streaming with an optional sliding KV cache, depth head
only: the loop the ONNX export wrapper (streamvggt/export/wrapper.py) bakes
in, shared by the KV-window validation (tests/kv_cache_window_validation.py)
and the depth-dropout experiment (src/scale_drift.py).

Matches MetricStreamVGGT.inference as the benchmark runs it: the
conditioning and the aggregator under bf16 autocast (dust3r's
loss_of_one_batch wraps the whole inference in it), the depth head in fp32.
"""

from __future__ import annotations

import numpy as np
import torch


def truncate_cache(past_kv: list, window: int) -> list:
    """Keep the last `window` frame slices of every block's (K, V)
    ([B, heads, n_frames, P, head_dim]; dim 2 is the frame axis), the slice
    the export wrapper applies to its cache outputs.

    Frame 0's K/V leave the cache once the stream exceeds `window`; VGGT-family
    camera and point heads anchor to frame 0, depth is per-frame and the most
    robust to that eviction (tests/kv_cache_window_validation.py measures it).
    """
    return [
        None if kv is None else (kv[0][:, :, -window:], kv[1][:, :, -window:])
        for kv in past_kv
    ]


@torch.no_grad()
def stream_depth(
    model, views: list[dict], window: int | None, amp: bool = True
) -> np.ndarray:
    """[S, h, w] float32 depth, one frame at a time through the model's own
    aggregator and depth head, the cache cut to the last `window` frames after
    every step (None: unbounded, as StreamVGGT.inference). `model` is a
    StreamVGGT or a MetricStreamVGGT; each view's "img" is [1, 3, h, w] in
    [0, 1], and a MetricStreamVGGT reads its sparse depth from the view."""
    from streamvggt.depth_cond import MetricStreamVGGT

    metric = isinstance(model, MetricStreamVGGT)
    inner = model.model if metric else model
    past_kv = [None] * inner.aggregator.depth
    out = []
    for i, view in enumerate(views):
        images = view["img"][:, None]  # [1, 1, 3, h, w]
        token_feats, depth_residuals = None, None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if metric and model.conditioner is not None:
                cond = model._route_conditioning([view], images)
                token_feats = cond["depth_token_feats"]
                if cond["depth_head_residuals"] is not None:
                    depth_residuals = cond["depth_head_residuals"]["depth"]
            tokens, patch_start_idx, past_kv = inner.aggregator(
                images,
                past_key_values=past_kv,
                use_cache=True,
                past_frame_idx=i,
                injected_patch_feats=token_feats,
            )
        if depth_residuals is not None:
            depth_residuals = [r.float() for r in depth_residuals]
        depth, _ = inner.depth_head(
            [t.float() for t in tokens],
            images=images,
            patch_start_idx=patch_start_idx,
            depth_residuals=depth_residuals,
        )
        out.append(depth[0, 0, :, :, 0].float().cpu().numpy())
        if window is not None:
            past_kv = truncate_cache(past_kv, window)
    return np.stack(out)
