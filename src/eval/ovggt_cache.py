"""Our trained model streamed through OVGGT's KV-cache policy
(third_party/ovggt, arXiv:2603.05959): a training-free, constant-size cache
with importance-scored eviction, the first frame kept as a global anchor and
history anchors chosen by view coverage. Asks whether that cache policy, on
our weights, holds the metric scale that our sliding window loses (the
frame-sink arm of the scale-state ablation), and shows the two compose.

How the model is assembled:
  * the run's MetricStreamVGGT, loaded as it is benchmarked
    (model_io.load_run_model, checkpoint-last);
  * its LoRA adapters merged into the base attention weights
    (streamvggt.export.lora_merge.merge_lora, as the ONNX export does), which
    leaves a plain StreamVGGT state dict under the pretrained key names --
    OVGGT holds the same parameters and loads it strictly, trained depth head
    included;
  * our conditioner run per frame, its token features handed to OVGGT's
    aggregator before the attention blocks, so before the cache (the hooks
    marked `# MeTRICS:` in third_party/ovggt).
Precision as eval.streaming_window.stream_depth: conditioning and aggregator
under bf16 autocast when the run trained with amp; OVGGT runs its heads in
fp32 itself.

One upstream mechanism reads our outputs differently: the coverage test for
history anchors back-projects the predicted depth (metric, ours) with the
predicted poses (the camera head is StreamVGGT's, inherited untrained, in its
normalised units). The two scales disagree, so anchors may be placed at other
frames than on StreamVGGT's own up-to-scale depth. Kept as upstream does it.

Not supported: head injection (its residuals enter OVGGT's DPT heads, which
have no such input).
"""

from __future__ import annotations

import numpy as np
import torch

from eval.baselines import arms as A
from model_io import load_run_model
from streamvggt.depth_cond import InjectionType
from streamvggt.export.lora_merge import merge_lora


def load(run_dir: str, device: torch.device, total_budget: int | None = 200_000):
    """(OVGGT carrying the run's weights, the run's MetricStreamVGGT reduced
    to its conditioner, amp). total_budget: OVGGT's cache size in tokens, its
    model default 200,000 (split across the 24 global layers by its own
    rule); None disables eviction (a parity check against the full cache)."""
    run = load_run_model(run_dir, "last", device, encoder_cache=False)
    dc = run.mcfg.depth_cond
    if dc.enabled and dc.injection is not InjectionType.TOKEN:
        raise NotImplementedError("OVGGT streaming supports token injection only")
    if getattr(run.model.model.aggregator, "scale_token", None) is not None:
        raise NotImplementedError("OVGGT streaming does not carry a scale token")
    cls, *_ = A._fork_parts(A.OVGGTArm.ROOT, A.OVGGTArm.MODULE, A.OVGGTArm.CLASS)
    ovggt = cls(total_budget=total_budget)
    merge_lora(run.model.model.aggregator)
    ovggt.load_state_dict(run.model.model.state_dict(), strict=True)
    ovggt = ovggt.to(device).eval()
    metric = run.model
    metric.model = None  # the backbone now lives in `ovggt`; keep only the conditioner
    return ovggt, metric, bool(run.raw.get("amp", 1))


@torch.no_grad()
def stream_depth(ovggt, metric, views: list[dict], amp: bool = True) -> np.ndarray:
    """[S, h, w] float32 depth at the model resolution, frame by frame
    through OVGGT's own streaming loop and cache, our conditioning added per
    frame. Each view's "img" is [1, 3, h, w] in [0, 1]; the sparse depth is
    read from the view, as eval.streaming_window.stream_depth does."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        feats = None
        if metric.conditioner is not None:
            feats = [
                metric._route_conditioning([view], view["img"][:, None])["depth_token_feats"]
                for view in views
            ]
        out = ovggt.inference(views, depth_token_feats_list=feats)
    return np.stack([r["depth"][0, :, :, 0].float().numpy() for r in out.ress])
