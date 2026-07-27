"""Export-friendly streaming wrappers around StreamVGGT / MetricStreamVGGT.

One ONNX call == one frame of causal streaming. The wrapper reproduces the
per-frame loop of StreamVGGT.inference() (aggregator with use_cache=True,
then the depth head) while replacing everything the ONNX exporter cannot
carry: list-of-dict I/O, HF ModelOutput, autocast contexts, and python-side
cache management. Camera / point / track heads are dropped -- depth only.

I/O contract (all variants share it; ONE graph serves every frame):
  inputs : rgb   [1, 3, H, W]  float32 in [0, 1] -- NOT the [-1, 1] ImgNorm
                               tensor the datasets and visualize_spot.py's
                               load_spot_views build. That is the dataset
                               format; finetune_depth._prepare_batch rescales
                               it with (img + 1) / 2 before the model, and
                               Aggregator._encode then applies the ImageNet
                               mean/std itself (its docstring: "images ... in
                               range [0, 1]"). Feed pixel/255, not pixel/255
                               * 2 - 1.
           sparse_depth [1, 1, H, W]  float32 sparse METRIC depth (metres);
                               invalid pixels are anything outside
                               [depth_min, depth_max] -- the validity mask is
                               derived IN-GRAPH as (d >= depth_min) &
                               (d <= depth_max), matching the consumer's
                               convention, so no third input exists. NOT
                               named `depth`: ONNX value names share one
                               namespace with the outputs and the exporter
                               silently renames a collision (`depth` ->
                               `depth.1`). The base variant accepts and
                               ignores its VALUES but still exposes the input
                               (see StreamingDepthExport._zero_from -- an
                               untied input is pruned by the tracer).
           past_k_00..past_v_23: 48 cache tensors [1, 16, n_frames, P, 64],
                               dynamic dim 2. Frame 0 feeds n_frames == 0
                               (see StreamingDepthExport.empty_cache). The
                               aggregator's frame-0 vs later-frame camera/
                               register token choice is derived from the
                               cache's shape, which the tracer records
                               SYMBOLICALLY, so the selection stays dynamic
                               in-graph and no separate init graph exists --
                               asserted per-export by export_onnx.probe_frame0
                               and pinned by the divergent-token smoke model
                               in tests/export_onnx_parity.py.
  outputs: depth [1, H', W'], depth_conf [1, H', W'] where (H', W') is the
           NETWORK orientation -- (W, H) with rotate=True, (H, W) otherwise
           (rotated outputs are not rotated back) -- plus new_k_00..new_v_23
           (cache carried to the next call, already window-sliced).

Rotation: with rotate=True the graph rotates rgb+depth 90 degrees clockwise
before the network; the depth/conf outputs REMAIN in that rotated
orientation ([1, W, H] for [H, W] inputs) -- the consumer wants the rotated
maps, so no inverse rotation is applied. Implemented as transpose+flip
(torch.rot90 has no ONNX symbolic in the TorchScript exporter).

Sliding window: cache outputs are sliced to the last `window` frame slices
in-graph, so steady-state memory is bounded at window * per-frame-KV.

# NOTE (frame-0 eviction risk): once the stream exceeds `window`, frame 0's
# K/V leave the cache. VGGT-family models anchor their world frame to frame
# 0, so the camera/point heads could drift under a pure sliding window; a
# StreamingLLM-style sink (always retain kv[:, :, :1]) is the cheap fix if a
# future export keeps those heads. Depth is per-frame and empirically the
# most robust to eviction; the measured cost on this depth-only export is a
# slow global-scale drift on long sequences (see
# tests/kv_cache_window_validation.py) -- documented and accepted.

# TODO(token arm): validated export of the token-injection variant is
# deferred until the zero-init token run lands (the gated token checkpoints
# are deprecated by the gate removal on fix/improve-gate-init; their
# conditioning is near-inert). The wrapper supports the arm; only the
# validation is pending -- see the conditioning isolation ablation plan.
"""

import torch
import torch.nn as nn

import streamvggt.heads.utils as _head_utils
from streamvggt.depth_cond.conditioner import dpt_fusion_sizes
from streamvggt.depth_cond.config import InjectionType
from streamvggt.depth_cond.model import MetricStreamVGGT
from streamvggt.models.streamvggt import StreamVGGT

from .cache import NUM_GLOBAL_BLOCKS, assert_cache_layout, unflatten_cache


def _make_sincos_pos_embed_f32(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """Float32 drop-in for heads.utils.make_sincos_pos_embed. The original
    builds omega in float64 and eager PyTorch type-promotes the einsum (it
    casts back to float32 on return, so nothing downstream sees float64), but
    the exported ONNX Einsum keeps the mixed float/double inputs and
    onnxruntime rejects the graph.

    Parity with the unpatched model: `omega` does not depend on `pos`, so it
    is still built in float64 and only cast down afterwards -- the baked
    frequencies are then the correctly rounded float32 of the original's, and
    only the outer product and sin/cos run at lower precision. The return
    dtype matches the original's `emb.float()` unconditionally (pinned by
    check_sincos_patch_cost across input dtypes). Measured
    residual is 3.4e-6 on the embedding, which the DPT head scales by
    ratio=0.1 before adding it to the features -- ~3e-7, four orders below
    the 2e-3 parity threshold (pinned at 1e-5 by check_sincos_patch_cost in
    tests/export_onnx_parity.py). Graph-vs-eager parity is unaffected either
    way: the eager reference runs this same patched function."""
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = (1.0 / omega_0**omega).float()
    # float32 in, float32 out, whatever `pos` is: the original computes in
    # float64 and ends with `return emb.float()`, so a caller under bf16/fp16
    # autocast gets float32 from it. Following pos.dtype instead would hand
    # the DPT head a half-precision positional embedding -- a divergence the
    # export's own fp32 tracing would never reveal.
    out = torch.einsum("m,d->md", pos.reshape(-1).float(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


class _ConstantPositions:
    """Drop-in replacement for rope.PositionGetter at a FIXED resolution.

    The original caches per (h, w) python ints, but under ONNX tracing the
    grid dims arrive as traced 0-dim Tensors, so the cache key never matches
    and the cold branch re-runs torch.cartesian_prod -- which has no ONNX
    symbolic and kills the export. At a fixed export resolution the positions
    are a constant, so we precompute them eagerly once and always return
    them (they trace into the graph as an initializer).

    Otherwise it matches PositionGetter exactly -- batch-expanded and cloned
    -- and refuses a grid it was not built for, which is the failure mode of
    baking a resolution into a getter the model keeps (see
    StreamingDepthExport.__init__). Under tracing h/w arrive as Tensors and
    the guard is skipped; the export resolution is fixed by construction
    there, while eager (the parity reference) is where a second wrapper at a
    different resolution could silently reuse the wrong grid."""

    def __init__(self, positions: torch.Tensor, hw: tuple[int, int]) -> None:
        self.positions = positions  # [1, h*w, 2], int64
        self.hw = hw

    def __call__(self, batch, h, w, device) -> torch.Tensor:
        if isinstance(h, int) and isinstance(w, int) and (h, w) != self.hw:
            raise AssertionError(
                f"position getter was baked for grid {self.hw}, called with "
                f"{(h, w)} -- the model is shared by wrappers of different "
                "resolutions; rebuild the wrapper for this resolution"
            )
        pos = self.positions.to(device)
        # PositionGetter hands back a batch-expanded clone. `batch` can arrive
        # as a traced Tensor (which expand would choke on) and the export is
        # always B*S == 1, so only expand for a real python int > 1.
        if isinstance(batch, int) and batch > 1:
            pos = pos.expand(batch, -1, -1)
        return pos.clone()


def rotate_cw(x: torch.Tensor) -> torch.Tensor:
    """90 degrees clockwise over the last two dims == torch.rot90(x, k=-1)."""
    return x.transpose(-2, -1).flip(-1)


class StreamingDepthExport(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        image_hw: tuple[int, int],
        window: int = 20,
        rotate: bool = False,
        depth_min: float = 0.01,
        depth_max: float = 100.0,
    ) -> None:
        super().__init__()
        if isinstance(model, MetricStreamVGGT):
            inner: StreamVGGT = model.model
            self.conditioner = model.conditioner
            self.injection = (
                model.cfg.depth_cond.injection if self.conditioner else None
            )
        elif isinstance(model, StreamVGGT):
            inner = model
            self.conditioner = None
            self.injection = None
        else:
            raise TypeError(f"unsupported model type: {type(model).__name__}")

        assert_cache_layout(inner.aggregator)
        self.aggregator = inner.aggregator
        self.depth_head = inner.depth_head
        self.window = int(window)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.rotate = bool(rotate)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)

        # network-facing dims are POST-rotation; image_hw is what the GRAPH
        # takes (pre-rotation), which is what a consumer must feed
        h, w = image_hw
        self.image_hw = (h, w)
        self.net_hw = (w, h) if self.rotate else (h, w)
        ps = self.aggregator.patch_size
        if self.net_hw[0] % ps or self.net_hw[1] % ps:
            raise ValueError(
                f"network input {self.net_hw} must be a multiple of the "
                f"patch size {ps} (pre-rotation input was {image_hw})"
            )
        # head-arm residual scales are computed python-side; fixed-resolution
        # export makes them constants, which is exactly what we want in-graph
        self.out_hw_list = dpt_fusion_sizes(self.net_hw[0], self.net_hw[1], ps)
        self.n_tokens = self.aggregator.patch_start_idx + (self.net_hw[0] // ps) * (
            self.net_hw[1] // ps
        )
        # cache geometry from the live model, not constants (a small smoke
        # model has different head counts than the 1B checkpoint)
        attn = self.aggregator.global_blocks[0].attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim

        # Both patches below are DELIBERATE and both are visible outside this
        # object: the wrapper takes ownership of `model` for export purposes
        # (it is not a copy -- deep-copying the 1B checkpoint to keep the
        # caller's model pristine is not worth 5 GB), and the sincos patch is
        # process-wide. Construct the wrapper on a model you are exporting,
        # not on one a training loop is still using. Both replacements are
        # equivalent to the originals at the export resolution / in float32
        # and are pinned by tests/export_onnx_parity.py --cpu-unit.
        #
        # fixed-resolution export: swap the RoPE position getter for a
        # precomputed constant (see _ConstantPositions -- the original's
        # cache misses under tracing and re-enters aten::cartesian_prod,
        # which has no ONNX symbolic). The getter guards its own grid, so a
        # second wrapper at a different resolution fails loudly.
        if self.aggregator.position_getter is not None:
            ph, pw = self.net_hw[0] // ps, self.net_hw[1] // ps
            pos = torch.cartesian_prod(torch.arange(ph), torch.arange(pw))
            self.aggregator.position_getter = _ConstantPositions(
                pos.view(1, ph * pw, 2), (ph, pw)
            )
        # export-process-wide: float32 sincos embedding (see the drop-in's
        # docstring -- the original's float64 omega breaks the exported Einsum)
        _head_utils.make_sincos_pos_embed = _make_sincos_pos_embed_f32

        self.eval()
        self.requires_grad_(False)

    def empty_cache(self) -> list[torch.Tensor]:
        """48 zero-length cache tensors matching THIS model's geometry."""
        from .cache import empty_cache

        return empty_cache(self.n_tokens, self.num_heads, self.head_dim)

    @property
    def variant(self) -> str:
        if self.injection is None:
            return "base"
        return "token" if self.injection == InjectionType.TOKEN else "head"

    def set_fused_attention(self, enabled: bool) -> None:
        """Runtime toggle of F.scaled_dot_product_attention in every attention
        block (no source edits): the unfused matmul/softmax path is the
        fallback if SDPA fails to export at the chosen opset."""
        for blocks in (self.aggregator.frame_blocks, self.aggregator.global_blocks):
            for block in blocks:
                block.attn.fused_attn = enabled

    def _zero_from(self, depth: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """An exactly-zero scalar that DEPENDS on `depth`, for the base variant.

        The base arm ignores depth's values, but the I/O contract says every
        variant exposes the same 50 inputs -- and the ONNX tracer prunes any
        input no op consumes, so an untied `depth` silently vanishes from the
        graph (asserted against by export_onnx.assert_graph_io).

        The tied-in value is a BOOLEAN, deliberately: it is 0 or 1 in every
        dtype the graph can be converted to, so `x * 0.0` is exactly 0.0. An
        earlier version tied in the valid-pixel COUNT, which is fine in fp32
        but overflows fp16 -- a dense depth map counts past 65504, the fp16
        conversion casts that to inf, and inf * 0.0 = NaN poisons the entire
        fp16 graph (export_onnx.check_fp16_numerics now also catches that
        class of bug empirically). The predicate is the same validity mask
        the conditioned arms derive, and it is finite for ANY input -- NaN
        and Inf both compare False."""
        mask = (depth >= self.depth_min) & (depth <= self.depth_max)
        return (mask.sum() > 0).to(dtype) * 0.0

    def _conditioning(self, depth: torch.Tensor):
        """depth: [1, 1, H', W'] interpreted as [B, S, H, W]. Returns
        (injected_patch_feats, depth_head_residuals) for the active arm."""
        if self.conditioner is None:
            return None, None
        mask = (depth >= self.depth_min) & (depth <= self.depth_max)
        if self.injection == InjectionType.TOKEN:
            return self.conditioner(depth, mask), None
        residuals = self.conditioner(depth, mask, out_hw_list=self.out_hw_list)
        return None, residuals["depth"]

    def _step(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        past_kv: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if self.rotate:
            rgb = rotate_cw(rgb)
            depth = rotate_cw(depth)

        images = rgb.unsqueeze(1)  # [1, 1, 3, H', W']
        feats, residuals = self._conditioning(depth)
        if self.conditioner is None:
            images = images + self._zero_from(depth, images.dtype)

        # fresh list every call -- the aggregator mutates it in place
        pkv = (
            unflatten_cache(past_kv)
            if past_kv is not None
            else [None] * NUM_GLOBAL_BLOCKS
        )
        tokens, patch_start_idx, new_pkv = self.aggregator(
            images,
            past_key_values=pkv,
            use_cache=True,
            past_frame_idx=0,  # confirmed inert; RoPE is spatial-only
            injected_patch_feats=feats,
        )
        # heads run fp32 (StreamVGGT.forward disables autocast around them)
        tokens = [t.float() for t in tokens]
        depth_pred, conf = self.depth_head(
            tokens,
            images=images,
            patch_start_idx=patch_start_idx,
            depth_residuals=residuals,
        )
        depth_pred = depth_pred[:, 0, :, :, 0]  # [1, H', W']
        conf = conf[:, 0]  # [1, H', W']

        out_cache: list[torch.Tensor] = []
        for k, v in new_pkv:
            # sliding window, baked: no-op until the cache exceeds `window`
            # (negative Slice clamps), then keeps the most recent W frames.
            out_cache += [k[:, :, -self.window :], v[:, :, -self.window :]]

        # outputs deliberately stay in the rotated (network) orientation --
        # the consumer wants rotated depth maps, so no rotate_ccw here
        return depth_pred, conf, out_cache


class StepWrapper(nn.Module):
    """The single exported graph: 48 cache tensors in (zero-length at frame
    0), 48 (window-sliced) out."""

    def __init__(self, core: StreamingDepthExport) -> None:
        super().__init__()
        self.core = core

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, *past_kv):
        d, conf, cache = self.core._step(rgb, depth, list(past_kv))
        return (d, conf, *cache)
