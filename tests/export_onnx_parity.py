"""Parity checks for the streaming ONNX export.

Two modes:

--cpu-unit (MINUTES, login node, no checkpoint): unit checks of the pure-math
    export helpers -- LoRA merge equals the wrapped forward, cache
    flatten/unflatten round-trips, the transpose+flip rotation equals
    torch.rot90, the float32 sincos patch stays within 1e-5 of the model it
    replaces, and the in-graph window slice equals the reference truncation
    semantics from the KV-window validation experiment -- followed by a full
    export smoke on a tiny random-weight model: two exports (rotation off /
    on) at a non-square resolution, each checked for the 50-in / 50-out graph
    I/O contract, the declared geometry, the frame-0 probe, a 6-frame ORT
    rollout, and the output orientation -- plus one fp16 conversion that must
    still produce finite output, and a third export at B=2 whose graph must
    keep the two streams independent (check_batch_independence).

    Budget 18-26 min on ONE core (two runs on a shared login-node core
    measured 735 s and 1030 s when the smoke ran TWO exports; the
    torch.onnx.export calls were 349/357 s and 425/434 s respectively -- the
    spread is machine contention, not the code -- and the B=2 export adds a
    third of the same order). Everything that is not an export is small: the
    five unit checks finish in under 4 s, and the probes / rollouts /
    orientation checks are seconds. The export IS the cost: the model is tiny
    in width only, keeping the real 24-block layout the cache contract
    asserts, so the traced graph is large and torch's shape inference /
    constant folding dominate. Two avoidable multipliers are already handled
    -- thread pools sized from sched_getaffinity (runtime_utils.n_cpus:
    os.cpu_count() reports 48 on this node while the cpuset grants 1, so
    onnxruntime would otherwise spawn 48 threads to fight over one core) and
    a cached ORT session (~20 s to build; the probe / rollout / orientation
    checks all run the same file). Without them the post-export checks alone
    cost ~60 s per export instead of seconds.

full (checkpoint + exported graphs): N-frame rollout of random rgb/sparse
    depth through (1) the eager StreamingDepthExport wrapper (the PyTorch
    reference -- identical `_step` semantics to the graph by construction)
    and (2) the onnxruntime session(s), both on CPU so any difference is
    exporter fidelity rather than device numerics. Reports per-frame
    max-abs-diff for depth / conf / every cache tensor; FAILS if the depth
    diff exceeds --threshold.

    --window / --rotate / --height / --width MUST match the values the graph
    was exported with -- they are baked in, and the defaults here are the
    same as export_onnx.py's so the two do not silently drift. They are also
    checked, not merely documented: the resolution against the graph's
    declared input shapes, the orientation against the depth it actually
    returns (the graph declares those dims symbolically), and the window
    against where the graph's cache clamps, which needs --frames > --window
    to be observable. One gap remains by nature: at a SQUARE resolution a
    --rotate mismatch is a silent transpose that no shape check can see, and
    only the parity diff exposes it.

Known & accepted: under the baked sliding window with near-inert
conditioning, long sequences drift in global scale (nothing re-anchors it);
measured in tests/kv_cache_window_validation.py. This test checks
graph-vs-eager fidelity, not that behavior.

Examples (--window matches the export; --frames exceeds it so the clamp is
exercised):
  python tests/export_onnx_parity.py --cpu-unit
  # graph exported with the default --window 20
  python tests/export_onnx_parity.py --onnx-prefix onnx/base \
      --ckpt ckpt/checkpoints.pth --frames 22
  # graph exported with --window 4 --rotate
  python tests/export_onnx_parity.py --onnx-prefix onnx/head_18075bb5 \
      --weights /oscar/scratch/jdosch/metrics/checkpoints/18075bb53f48343c \
      --checkpoint best --frames 6 --window 4 --rotate
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Everything below needs src/ on the path, hence the E402s. The ONE import
# this file still defers is onnxconverter-common (see check_fp16_smoke):
# requirements.txt deliberately does not install it, so importing it here
# would break --cpu-unit for everyone who has not hand-installed it.
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import streamvggt.heads.utils as head_utils  # noqa: E402
from export_onnx import (  # noqa: E402
    assert_graph_geometry,
    assert_graph_io,
    cache_input_names,
    cache_output_names,
    check_fp16_numerics,
    convert_onnx_to_fp16,
    export_step,
    load_base_model,
    load_finetuned_model,
    merge_lora,
    ort_session,
    probe_frame0,
    run_by_name,
    set_providers,
    validate_rollout,
)
from runtime_utils import set_torch_threads  # noqa: E402
from streamvggt.depth_cond.conditioner import (  # noqa: E402
    dpt_fusion_sizes,
    masked_downsample,
)
from streamvggt.depth_cond.lora import LoRALinear, LoRAQKV  # noqa: E402
from streamvggt.export import StreamingDepthExport  # noqa: E402
from streamvggt.export.cache import (  # noqa: E402
    NUM_GLOBAL_BLOCKS,
    empty_cache,
    flatten_cache,
    unflatten_cache,
)
from streamvggt.export.wrapper import (  # noqa: E402
    _EXPORT_POOL,
    _make_sincos_pos_embed_f32,
    _masked_downsample_export,
    rotate_cw,
)
from streamvggt.heads.dpt_head import DPTHead  # noqa: E402
from streamvggt.layers.vision_transformer import DinoVisionTransformer  # noqa: E402
from streamvggt.models.aggregator import Aggregator  # noqa: E402
from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from visualize_spot import load_spot_views  # noqa: E402

# Fixed so that two constructions inside one run produce identical weights --
# check_baked_pos_encoding builds the same tiny ViT twice and compares the
# baked encoding against the original. Not a magic number to tune: the DATA
# seeds all come from --seed.
MODEL_SEED = 4


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cpu-unit",
        action="store_true",
        help="run only the checkpoint-free unit checks",
    )
    ap.add_argument(
        "--onnx-prefix",
        default=None,
        help="prefix used at export time (<prefix>_step.onnx etc.)",
    )
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt", "checkpoints.pth"))
    ap.add_argument("--weights", default=None)
    ap.add_argument(
        "--checkpoint", default="auto", choices=["auto", "best", "last", "final"]
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=8,
        help="frames to roll; use > --window to exercise the cache clamp",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=20,
        help="MUST match the window the graph was exported with (same default "
        "as export_onnx.py --window); verified against the graph once the "
        "rollout passes --window frames",
    )
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="MUST match the export-time --batch-size. B is baked into the "
        "graph (it is not a dynamic axis), so a graph exported at 2 can only "
        "be parity-checked at 2; verified against the graph's declared input "
        "shapes by assert_graph_geometry",
    )
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="MUST match the export-time --rotate; verified against the "
        "graph's declared output shape",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=2e-3,
        help="max allowed |depth diff| (use ~1e-1 for fp16 graphs)",
    )
    ap.add_argument(
        "--cuda",
        action="store_true",
        help="run the ONNX side on CUDAExecutionProvider. The eager side "
        "stays on CPU, so the reported diffs then include device numerics, "
        "not just exporter fidelity -- raise --threshold accordingly. Needs "
        "onnxruntime-gpu, plus cuDNN 9 (torch bundles 8) -- on Oscar, "
        "`module load cudnn/9.8.0.87-12-y7fu`",
    )
    ap.add_argument(
        "--tf32",
        action="store_true",
        help="let the CUDA provider use TF32 for fp32 matmuls (its own "
        "default, which set_providers turns OFF -- see its docstring). Costs "
        "~5e-4 relative per product against the CPU fp32 eager reference, "
        "which showed up as ~1e-2 on the cache and ~3e-4 on depth; use "
        "--threshold 1e-2 with it",
    )
    ap.add_argument("--seed", type=int, default=0)
    spot = ap.add_argument_group(
        "real data (optional)",
        "Roll REAL Spot frames instead of random noise. Random frames prove "
        "graph == eager; only real imagery can catch a wrong convention "
        "(rotation direction, [0,1] vs [-1,1], depth units), because those "
        "move eager and graph together.",
    )
    spot.add_argument(
        "--spot-dir",
        default=None,
        help="Spot sequence dir with color/ and depth/ "
        "(e.g. /oscar/data/jtompki1/cli277/new_spot_data/0)",
    )
    spot.add_argument("--spot-start", type=int, default=0)
    spot.add_argument("--spot-stride", type=int, default=1)
    spot.add_argument(
        "--dump-dir",
        default=None,
        help="write per-frame [input rgb | predicted depth] PNGs here; the two "
        "panels should sit 90 degrees apart when --rotate is baked in",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# --cpu-unit checks
# ---------------------------------------------------------------------------
def check_lora_merge() -> None:
    dim = 64

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FakeAttn()

    class FakeAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_blocks = nn.ModuleList([FakeBlock()])
            self.global_blocks = nn.ModuleList([FakeBlock()])

    agg = FakeAggregator().eval()
    for blocks in (agg.frame_blocks, agg.global_blocks):
        for block in blocks:
            block.attn.qkv = LoRAQKV(
                block.attn.qkv, ["q", "k", "v"], rank=4, alpha=8.0, dropout=0.0
            )
            block.attn.proj = LoRALinear(
                block.attn.proj, rank=4, alpha=8.0, dropout=0.0
            )
            # lora_B is zero-init; randomize so the merge moves real mass
            with torch.no_grad():
                for t in block.attn.qkv.targets:
                    nn.init.normal_(block.attn.qkv.adapters[t].lora_B, std=0.1)
                nn.init.normal_(block.attn.proj.adapter.lora_B, std=0.1)

    x = torch.randn(2, 7, dim)
    wrapped = [
        (b.attn.qkv(x).clone(), b.attn.proj(x).clone())
        for blocks in (agg.frame_blocks, agg.global_blocks)
        for b in blocks
    ]
    n = merge_lora(agg)
    if n != 4:
        raise AssertionError(f"expected 4 merged linears, got {n}")
    merged = [
        (b.attn.qkv(x), b.attn.proj(x))
        for blocks in (agg.frame_blocks, agg.global_blocks)
        for b in blocks
    ]
    for (wq, wp), (mq, mp) in zip(wrapped, merged):
        dq = (wq - mq).abs().max().item()
        dp = (wp - mp).abs().max().item()
        if dq >= 1e-5 or dp >= 1e-5:
            raise AssertionError(f"merge mismatch: qkv {dq:.2e} proj {dp:.2e}")
    if merge_lora(agg) != 0:
        raise AssertionError("merge must be a no-op the second time")
    print(
        f"[cpu-unit] lora merge == wrapped forward "
        f"(max diff {max(dq, dp):.2e}) and is idempotent"
    )


def check_cache_roundtrip() -> None:
    pairs = [
        (torch.randn(1, 16, 3, 10, 64), torch.randn(1, 16, 3, 10, 64))
        for _ in range(NUM_GLOBAL_BLOCKS)
    ]
    flat = flatten_cache(pairs)
    if len(flat) != 2 * NUM_GLOBAL_BLOCKS:
        raise AssertionError(
            f"expected {2 * NUM_GLOBAL_BLOCKS} tensors, got {len(flat)}"
        )
    back = unflatten_cache(flat)
    if back is pairs or not all(
        a[0] is b[0] and a[1] is b[1] for a, b in zip(pairs, back)
    ):
        raise AssertionError("round-trip must preserve tensor identity in a fresh list")
    if not (len(cache_input_names()) == len(set(cache_input_names())) == 48):
        raise AssertionError("cache input names must be 48 and unique")
    if len(cache_output_names()) != 48:
        raise AssertionError("cache output names must be 48")
    if not all(t.shape[2] == 0 for t in empty_cache(10)):
        raise AssertionError("empty_cache tensors must have zero-length frame axis")
    print("[cpu-unit] cache flatten/unflatten round-trip OK")


def check_rotation() -> None:
    # outputs stay in the rotated orientation by design (no inverse in the
    # pipeline), so only the input rotation is contract
    for shape in ((1, 3, 5, 7), (1, 5, 7)):
        x = torch.randn(*shape)
        if not torch.equal(rotate_cw(x), torch.rot90(x, k=-1, dims=(-2, -1))):
            raise AssertionError(f"rotate_cw != rot90(k=-1) for shape {shape}")
    print("[cpu-unit] transpose+flip rotation == torch.rot90(k=-1)")


def check_sincos_patch_cost() -> None:
    """Bound the ONE numerical liberty the export takes with the base model.

    StreamingDepthExport replaces heads.utils.make_sincos_pos_embed
    process-wide with a float32 version (the original's float64 omega
    promotes the Einsum, which onnxruntime rejects). The parity test cannot
    see this -- its eager reference runs the patched function too -- so
    compare the two implementations directly, against the original, and pin
    the deviation. The DPT head scales the embedding by ratio=0.1 before
    adding it to the features, so the reported number is an upper bound on
    what reaches the depth output."""
    if head_utils.make_sincos_pos_embed is _make_sincos_pos_embed_f32:
        raise AssertionError(
            "run this check before any StreamingDepthExport is constructed -- "
            "the constructor installs the patch process-wide"
        )
    # non-fp32 `pos` is the case the export itself never traces: the heads run
    # fp32 there, but the patch is process-wide, so anything running the model
    # under bf16/fp16 autocast hits it too. The original always ends in
    # `emb.float()`; a drop-in that followed pos.dtype would quietly hand the
    # DPT head a half-precision embedding.
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        pos = torch.arange(64, dtype=dtype)  # covers the largest grid axis
        for embed_dim in (64, 256):
            ref = head_utils.make_sincos_pos_embed(embed_dim, pos)
            got = _make_sincos_pos_embed_f32(embed_dim, pos)
            if got.dtype != torch.float32 or got.dtype != ref.dtype:
                raise AssertionError(
                    f"sincos drop-in returns {got.dtype} for {dtype} pos; "
                    f"original returns {ref.dtype} (float32 is the contract)"
                )
            if got.shape != ref.shape:
                raise AssertionError(
                    f"sincos drop-in returns {tuple(got.shape)}, original "
                    f"{tuple(ref.shape)}"
                )
            diff = (ref.double() - got.double()).abs().max().item()
            if diff > 1e-5:
                raise AssertionError(
                    f"float32 sincos embedding deviates by {diff:.2e} at "
                    f"embed_dim={embed_dim}, pos dtype {dtype} -- too far "
                    "from the base model"
                )
    print(
        f"[cpu-unit] float32 sincos patch within {diff:.2e} of the original "
        "(x0.1 head ratio before it reaches the features)"
    )


def check_baked_pos_encoding() -> None:
    """The DINOv2 patch embed's interpolate_pos_encoding is the one op that
    cannot be exported at all: an antialiased bicubic resize
    (aten::_upsample_bicubic2d_aa) with no ONNX symbolic at any opset.
    StreamingDepthExport bakes it to a constant; this checks that the
    constant is EQUAL to what the original computes, that it no longer
    depends on the interpolation, and that it refuses another resolution.

    The export smoke cannot cover this -- build_tiny_model uses the conv
    patch embed, which has no such method, so the tiny graph exports happily
    while the real checkpoint dies. That is exactly how this reached a real
    export. A tiny DinoVisionTransformer stands in for the 1B one here."""
    h, w = SMOKE_HW  # non-square: forces the interpolation branch
    ps = 14

    def tiny_vit():
        torch.manual_seed(MODEL_SEED)
        return DinoVisionTransformer(
            img_size=h,
            patch_size=ps,
            embed_dim=64,  # must match the tiny aggregator's width
            depth=1,
            num_heads=1,
            mlp_ratio=1.0,
            num_register_tokens=4,
            interpolate_antialias=True,  # as Aggregator builds it
            interpolate_offset=0.0,
        ).eval()

    class _PatchEmbedOnly(nn.Module):
        """Just the pos-encoding path -- a whole-model export costs minutes,
        and the op either lowers here or it does not."""

        def __init__(self, vit):
            super().__init__()
            self.vit = vit

        def forward(self, img):
            return self.vit.prepare_tokens_with_masks(img)

    def export(vit, path):
        torch.onnx.export(
            _PatchEmbedOnly(vit), (torch.rand(1, 3, h, w),), path, opset_version=17
        )
        return {n.op_type for n in onnx.load(path).graph.node}

    with tempfile.TemporaryDirectory() as td:
        # control: unmodified, this is the failure a real export hits
        try:
            export(tiny_vit(), os.path.join(td, "control.onnx"))
        except Exception as e:
            if "bicubic" not in str(e):
                raise AssertionError(
                    f"expected the antialiased-bicubic export failure, got "
                    f"[{type(e).__name__}] {e}"
                )
        else:
            raise AssertionError(
                "the unbaked DINOv2 pos encoding exported -- this check no "
                "longer proves the bake is what makes the export work"
            )

        # the real path: the wrapper's constructor bakes it
        model = build_tiny_model()
        model.aggregator.patch_embed = tiny_vit()
        wrapper = StreamingDepthExport(model, image_hw=(h, w), window=3)
        vit = wrapper.aggregator.patch_embed
        ref = tiny_vit().interpolate_pos_encoding(
            torch.zeros(1, (h // ps) * (w // ps) + 1, 64), h, w
        )
        got = vit.interpolate_pos_encoding(torch.zeros_like(ref), h, w)
        if got.shape != ref.shape or not torch.equal(got, ref):
            raise AssertionError(
                f"baked pos encoding {tuple(got.shape)} differs from the "
                f"original {tuple(ref.shape)}"
            )
        try:
            vit.interpolate_pos_encoding(ref, w, h)  # swapped == wrong res
        except AssertionError:
            pass
        else:
            raise AssertionError("baked pos encoding accepted the wrong resolution")
        ops = export(vit, os.path.join(td, "baked.onnx"))
        if "Resize" in ops:
            raise AssertionError(f"interpolation survived the bake: {sorted(ops)}")
    print(
        f"[cpu-unit] DINOv2 pos encoding: unbaked fails to export (bicubic), "
        f"baked equals it exactly and exports clean at {h}x{w}"
    )


def check_exportable_adaptive_pool() -> None:
    """The conditioned arms pool the sparse map to the DPT fusion sizes with
    F.adaptive_avg_pool2d, which the exporter lowers ONLY when the output size
    divides the input. At the real 518x392 two of the four fusion sizes are
    non-integral ratios (518/148 = 3.5, 518/19 = 27.3), so the head arm cannot
    export at all -- it dies with SymbolicValueError before writing anything.

    The wrapper swaps in a separable two-matmul equivalent. This checks both
    halves: that the original really is unexportable at these sizes (so the
    swap is load-bearing, not decoration) and that the replacement matches it
    numerically on a sparse map with realistic validity.

    The base arm never reaches this code, which is why the base export
    succeeded on the real checkpoint while the head arm crashed."""
    h, w = 518, 392  # the real network resolution, where the ratios go bad
    sizes = dpt_fusion_sizes(h, w, 14)
    x = torch.rand(1, 1, h, w)
    mask = (torch.rand(1, 1, h, w) < 0.427).float()  # real Spot validity share

    worst = 0.0
    for out_hw in sizes:
        ref = F.adaptive_avg_pool2d(x, out_hw)
        got = _EXPORT_POOL(x, out_hw)
        if got.shape != ref.shape:
            raise AssertionError(f"{out_hw}: shape {got.shape} != {ref.shape}")
        pa, fa = masked_downsample(x, mask, out_hw)
        pb, fb = _masked_downsample_export(x, mask, out_hw)
        worst = max(
            worst,
            (ref - got).abs().max().item(),
            (pa - pb).abs().max().item(),
            (fa - fb).abs().max().item(),
        )
    if worst > 1e-5:
        raise AssertionError(f"exportable pooling deviates by {worst:.2e}")

    class _Pool(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, t):
            return self.fn(t, sizes[0])  # (148, 112): the 3.5x ratio

    with tempfile.TemporaryDirectory() as td:
        try:
            torch.onnx.export(
                _Pool(F.adaptive_avg_pool2d),
                (x,),
                os.path.join(td, "control.onnx"),
                opset_version=17,
            )
        except Exception as e:
            if "adaptive_avg_pool2d" not in str(e):
                raise AssertionError(f"unexpected control failure: {e}")
        else:
            raise AssertionError(
                "F.adaptive_avg_pool2d exported at a non-integral ratio -- this "
                "check no longer proves the replacement is needed"
            )
        torch.onnx.export(
            _Pool(_EXPORT_POOL), (x,), os.path.join(td, "pool.onnx"), opset_version=17
        )
    print(
        f"[cpu-unit] adaptive pooling: original unexportable at {sizes[0]} "
        f"(3.5x), matmul form exports and matches within {worst:.2e}"
    )


def check_window_slice() -> None:
    """The in-graph slice kv[:, :, -W:] must equal the reference truncation
    semantics validated in the KV-window experiment (keep the LAST W frame
    slices; no-op while n_frames <= W)."""
    W = 4
    for n_frames in (1, W, W + 3):
        k = torch.randn(1, 16, n_frames, 10, 64)
        sliced = k[:, :, -W:]
        expect = k[:, :, max(0, n_frames - W) :]
        if not torch.equal(sliced, expect):
            raise AssertionError(f"window slice mismatch at n_frames={n_frames}")
        if sliced.shape[2] != min(n_frames, W):
            raise AssertionError(
                f"window slice kept {sliced.shape[2]} frames at "
                f"n_frames={n_frames}, expected {min(n_frames, W)}"
            )
    print("[cpu-unit] window slice matches reference truncation semantics")


def build_tiny_model():
    """Random-weight StreamVGGT with the REAL block/head code but tiny dims
    (conv patch embed, width 64, 4 heads) -- keeps the full 24-block layout
    the cache contract asserts, yet exports on a login-node CPU in a few
    minutes. Exercises everything the exporter must survive: the
    cached-attention concat, dynamic frame axis, DPT interpolations, window
    Slice, rotation.

    The camera / register tokens are deliberately forced APART (default init
    leaves them at std 1e-6, numerically interchangeable). That is what makes
    every export below able to see mis-tokening: the aggregator picks the
    frame-0 vs later-frame token slot from the cache's shape, and probe_frame0
    compares the cache output, where those token rows land verbatim -- so a
    slot the tracer constant-folded shows up at full magnitude. With
    interchangeable tokens the probe would pass regardless."""
    torch.manual_seed(MODEL_SEED)
    model = StreamVGGT.__new__(StreamVGGT)
    nn.Module.__init__(model)
    model.aggregator = Aggregator(
        img_size=140,
        patch_size=14,
        embed_dim=64,
        depth=24,
        num_heads=4,
        mlp_ratio=1.0,
        patch_embed="conv",
    )
    model.depth_head = DPTHead(
        dim_in=128,
        output_dim=2,
        activation="exp",
        conf_activation="expp1",
        features=32,
        out_channels=[32, 48, 64, 64],
    )
    model.camera_head = None
    model.point_head = None
    model.track_head = None
    g = torch.Generator().manual_seed(MODEL_SEED)
    with torch.no_grad():
        agg = model.aggregator
        agg.camera_token.copy_(torch.randn(agg.camera_token.shape, generator=g))
        agg.register_token.copy_(torch.randn(agg.register_token.shape, generator=g))
    return model.eval()


# non-square, both dims a multiple of the patch size: at 140x140 the
# rotation is invisible in the output shape, so an inverse rotation
# re-introduced at the end of the pipeline would pass every shape check
SMOKE_HW = (140, 168)


def run_export_smoke(args) -> None:
    """Full export -> frame-0 probe -> graph I/O contract -> ORT rollout ->
    output orientation, on the tiny model. Catches exporter/opset/runtime
    issues without a checkpoint or GPU.

    Two exports, one per rotation setting, at a NON-SQUARE resolution so the
    baked rotation is observable in the output shape. Both run on the
    divergent-token model (see build_tiny_model), so probe_frame0 also pins
    the dynamic frame-0 token selection the single-graph contract rests on --
    each export is the expensive part here, so that check rides along with
    these two rather than paying for a third."""
    model = build_tiny_model()
    h, w = SMOKE_HW
    for rotate in (False, True):
        wrapper = StreamingDepthExport(model, image_hw=(h, w), window=3, rotate=rotate)
        expect_hw = (w, h) if rotate else (h, w)
        if wrapper.net_hw != expect_hw:
            raise AssertionError(
                f"rotate={rotate}: net_hw {wrapper.net_hw}, expected {expect_hw}"
            )
        ns = SimpleNamespace(height=h, width=w, window=3, opset=17)
        with tempfile.TemporaryDirectory() as td:
            step = export_step(wrapper, ns, os.path.join(td, "tiny_step.onnx"))
            probe_frame0(step, wrapper, ns)
            validate_rollout(step, wrapper, ns, n_frames=6)
            check_graph_orientation(step, wrapper, args.seed)
            if not rotate:  # one conversion is enough; rotation is orthogonal
                check_fp16_smoke(step, wrapper, ns)
        print(
            f"[export-smoke] rotate={rotate}: OK (single graph, {h}x{w} in -> "
            f"{expect_hw[0]}x{expect_hw[1]} out, 50 in / 50 out, divergent "
            "frame-0 token selection stayed dynamic)"
        )
    check_no_inverse_rotation(model, args.seed)
    check_batch_independence(model, args.seed)
    print("export smoke PASSED")


@torch.no_grad()
def check_batch_independence(model, seed: int, batch: int = 2) -> None:
    """A batched graph must be N INDEPENDENT streams, not one joint one.

    Batching is only a deployment win if element i's depth depends on element
    i's frames and nothing else -- two synchronized cameras share a graph, not
    a scene. Nothing in the architecture should couple them (attention never
    crosses dim 0, and each element carries its own 48 cache tensors), but the
    cached path selects the per-frame camera/register token by slicing a
    B*S-flattened tensor, which silently took ONE row for the whole batch
    until it was unflattened first -- wrong-but-plausible at B>1 rather than
    loud, and invisible to any check that feeds every element the same frame.

    So: roll B distinct streams through one batched wrapper, roll each one
    through a B=1 wrapper alone, and require they agree. Distinct inputs per
    element are the point -- identical ones would pass under any amount of
    cross-talk.

    Run against the EXPORTED graph as well as the eager wrapper, which costs a
    third export. Eager alone cannot cover this: `_ConstantPositions.__call__`
    takes a real python int there and never reads the baked `self.batch`, and
    the aggregator's B*S unflatten sees concrete shapes -- exactly the two
    places B>1 breaks, both traced-only. The frame-0 probe rides along for the
    same reason (zero-length caches at B=2 are their own contract)."""
    h, w = SMOKE_HW
    frames = [
        [dummy_frame(h, w, seed=seed + 100 * b + i) for i in range(3)]
        for b in range(batch)
    ]

    batched = StreamingDepthExport(model, image_hw=(h, w), window=3, batch_size=batch)
    cache = None
    batched_out = []
    for i in range(3):
        rgb = torch.cat([frames[b][i][0] for b in range(batch)], dim=0)
        d = torch.cat([frames[b][i][1] for b in range(batch)], dim=0)
        depth, _, cache = batched._step(rgb, d, cache)
        batched_out.append(depth)

    # export BEFORE the solo wrappers below rebake the shared model's position
    # getter for batch 1 -- the traced graph must carry the B=batch bake
    ns = SimpleNamespace(height=h, width=w, window=3, opset=17)
    with tempfile.TemporaryDirectory() as td:
        step = export_step(batched, ns, os.path.join(td, "tiny_batch_step.onnx"))
        probe_frame0(step, batched, ns)
        sess = ort_session(step)
        ort_cache, graph_out = None, []
        for i in range(3):
            rgb = torch.cat([frames[b][i][0] for b in range(batch)], dim=0)
            d = torch.cat([frames[b][i][1] for b in range(batch)], dim=0)
            feed = {"rgb": rgb.numpy(), "sparse_depth": d.numpy()}
            feed.update(
                dict(
                    zip(
                        cache_input_names(),
                        ort_cache
                        if ort_cache is not None
                        else [t.numpy() for t in batched.empty_cache()],
                    )
                )
            )
            out = run_by_name(sess, feed)
            ort_cache = [out[n] for n in cache_output_names()]
            graph_out.append(out["depth"])

    # the B=1 wrapper rebakes the shared model's position getter for batch 1,
    # so the solo rollouts must run AFTER the batched one
    worst, worst_graph = 0.0, 0.0
    for b in range(batch):
        solo = StreamingDepthExport(model, image_hw=(h, w), window=3, batch_size=1)
        cache = None
        for i in range(3):
            rgb, d = frames[b][i]
            depth, _, cache = solo._step(rgb, d, cache)
            worst = max(worst, float((depth[0] - batched_out[i][b]).abs().max()))
            worst_graph = max(
                worst_graph, float(np.abs(graph_out[i][b] - depth[0].numpy()).max())
            )
    if worst >= 1e-5:
        raise AssertionError(
            f"batched stream != solo stream (max|diff| {worst:.3e}): batch "
            "element outputs depend on the other elements' frames"
        )
    # 1e-3, the frame-0 probe's gate: this is a per-element identity, not an
    # exporter-fidelity diff, so a graph that mixes elements fails by orders of
    # magnitude and a loose gate costs no coverage
    if worst_graph >= 1e-3:
        raise AssertionError(
            f"EXPORTED graph at B={batch} != solo eager streams (max|diff| "
            f"{worst_graph:.3e}): the traced batch path (baked RoPE positions "
            "or the B*S token unflatten) does not keep the streams independent"
        )
    print(
        f"[export-smoke] batch independence: B={batch} matches {batch} solo "
        f"rollouts, eager max|diff| {worst:.2e}, exported graph "
        f"{worst_graph:.2e} -- streams do not interact"
    )


@torch.no_grad()
def check_graph_orientation(step_path, wrapper, seed: int) -> None:
    """The EXPORTED graph emits the network orientation: [1, W, H] when the
    rotation is baked in, [1, H, W] when it is not. Asserted against the
    contract, not against the eager wrapper -- eager and graph share `_step`,
    so a rotation change moves both and their agreement proves nothing."""
    h, w = SMOKE_HW
    rgb, d = dummy_frame(h, w, seed=seed)
    feed = {"rgb": rgb.numpy(), "sparse_depth": d.numpy()}
    feed.update(
        dict(zip(cache_input_names(), (t.numpy() for t in wrapper.empty_cache())))
    )
    out = run_by_name(ort_session(step_path), feed)
    want = (1, *wrapper.net_hw)
    for name in ("depth", "depth_conf"):
        if out[name].shape != want:
            raise AssertionError(
                f"rotate={wrapper.rotate}: graph {name} shape "
                f"{out[name].shape}, expected {want} (outputs must stay in "
                "the network orientation -- no inverse rotation)"
            )


def check_fp16_smoke(step_path, wrapper, ns) -> None:
    """fp16 has a failure mode of its own: a structurally valid graph that
    outputs NaN because some value overflowed fp16 and inf reached an
    arithmetic op. onnx.checker cannot see it -- only running it can. That is
    not hypothetical: the base arm's tied-in `sparse_depth` term used to be a
    valid-pixel COUNT, which passes fp32 parity and turns the whole fp16 graph
    into NaN at 65505 valid pixels (see wrapper._zero_from). The tiny model
    converts in seconds, so the smoke pays for the coverage."""
    try:
        import onnxconverter_common  # noqa: F401
    except ImportError:
        print("[export-smoke] fp16: onnxconverter-common missing, SKIPPED")
        return
    fp16_path = step_path.replace(".onnx", "_fp16.onnx")
    convert_onnx_to_fp16(step_path, fp16_path)
    if not check_fp16_numerics(fp16_path, wrapper, ns):
        raise AssertionError(
            "fp16 graph produced non-finite output -- something in the graph "
            "overflows fp16 (check_fp16_numerics removed the file)"
        )
    print("[export-smoke] fp16 conversion runs and stays finite")


@torch.no_grad()
def check_no_inverse_rotation(model, seed: int) -> None:
    """Pin the no-inverse-rotation contract by construction: rotating in-graph
    must be EQUIVALENT to pre-rotating the inputs and not rotating at all.
    Any rotate_ccw at the end of `_step` breaks this equality -- and, since
    the test resolution is non-square, even the shapes."""
    h, w = SMOKE_HW
    rgb, d = dummy_frame(h, w, seed=seed)
    # both wrappers have net_hw == (w, h), so they agree on the constant
    # position getter this constructor bakes into the shared model
    baked = StreamingDepthExport(model, image_hw=(h, w), window=3, rotate=True)
    a_depth, a_conf, _ = baked._step(rgb, d, None)
    pre = StreamingDepthExport(model, image_hw=(w, h), window=3, rotate=False)
    b_depth, b_conf, _ = pre._step(rotate_cw(rgb), rotate_cw(d), None)

    if tuple(a_depth.shape) != (1, w, h):
        raise AssertionError(
            f"rotated depth shape {tuple(a_depth.shape)}, expected {(1, w, h)}"
        )
    dd = (a_depth - b_depth).abs().max().item()
    dc = (a_conf - b_conf).abs().max().item()
    if dd or dc:
        raise AssertionError(
            f"baked rotation != pre-rotated input (depth {dd:.2e}, conf "
            f"{dc:.2e}): the pipeline is rotating the outputs back"
        )
    print(
        "[export-smoke] baked rotation == pre-rotated input, outputs stay "
        f"rotated ({h}x{w} -> {w}x{h})"
    )


def run_cpu_unit(args) -> None:
    # one seed for the whole run, from --seed: the checks below draw their
    # data from the global RNG rather than each pinning a magic number
    torch.manual_seed(args.seed)
    check_lora_merge()
    check_cache_roundtrip()
    check_rotation()
    check_sincos_patch_cost()  # MUST precede the first wrapper construction
    check_baked_pos_encoding()
    check_exportable_adaptive_pool()
    check_window_slice()
    run_export_smoke(args)
    print("cpu-unit checks PASSED")


# ---------------------------------------------------------------------------
# full parity mode
# ---------------------------------------------------------------------------
def dummy_frame(h: int, w: int, seed: int, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(batch, 3, h, w, generator=g)
    d = torch.rand(batch, 1, h, w, generator=g) * 4.0 + 0.5
    keep = (torch.rand(batch, 1, h, w, generator=g) < 0.05).float()
    return rgb, d * keep


def spot_frames(args) -> list:
    """Real Spot frames, in the GRAPH's input convention.

    Random frames prove graph == eager but can say NOTHING about whether the
    convention is right: a wrong rotation direction, a [-1,1] vs [0,1] mixup,
    or millimetre depth all move eager and graph together, so every existing
    check still passes. Only real imagery exposes those.

    Conversion, both halves of which are the contract this pins down:
      * load_spot_views rotates with PIL BEFORE its resize/crop; the exported
        graph bakes the rotation instead and wants PRE-rotation frames. So ask
        for rotate="none" -- 392x518 landscape -- and let the graph rotate.
      * it returns ImgNorm [-1,1] (the dataset format); the model wants [0,1],
        which is what finetune_depth._prepare_batch does with (img + 1) / 2.
    """
    views = load_spot_views(
        Path(args.spot_dir), args.spot_start, args.frames, args.spot_stride, "none"
    )
    frames = []
    for v in views:
        rgb = (v["img"] + 1.0) / 2.0  # [-1,1] dataset format -> [0,1] model input
        depth = v["sparse_depth"][:, None].float()  # [1,1,H,W], metres
        frames.append((rgb.float(), depth))
    return frames


def _to_pil(chw_or_hw, is_depth: bool):
    a = np.asarray(chw_or_hw).squeeze()
    if is_depth:
        finite = np.isfinite(a) & (a > 0)
        lo, hi = (a[finite].min(), a[finite].max()) if finite.any() else (0.0, 1.0)
        a = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8))


def dump_depth(path: str, depth, rgb=None, rotate: bool = False) -> None:
    """Write a panel for the one check no assertion can make: looking at it.

    Orientation errors are invisible numerically -- eager and graph agree
    perfectly on an upside-down map -- so this is the only place they surface.

    With a baked rotation the panels are
        [ graph input | input rotated as the graph rotates it | graph output ]
    and the check is that the RIGHT TWO match in orientation and structure.
    Comparing the output against the raw input only proves that *a* rotation
    happened; the middle panel is what distinguishes clockwise from
    counter-clockwise, which is the error that would otherwise ship."""
    panels = []
    if rgb is not None:
        panels.append(_to_pil(rgb, is_depth=False))
        if rotate:
            panels.append(_to_pil(rotate_cw(torch.as_tensor(rgb)), is_depth=False))
    panels.append(_to_pil(depth, is_depth=True))
    w, h = sum(p.width for p in panels), max(p.height for p in panels)
    sheet = Image.new("RGB", (w, h))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width
    sheet.save(path)


@torch.no_grad()
def run_parity(args) -> None:
    if args.onnx_prefix is None:
        raise SystemExit("full mode needs --onnx-prefix (run export first)")
    step_path = f"{args.onnx_prefix}_step.onnx"
    if not os.path.isfile(step_path):
        raise SystemExit(f"missing {step_path}")
    step_sess = ort_session(step_path)
    # a graph exported before the I/O contract was pinned (pruned or renamed
    # `sparse_depth`) must fail here, not silently parity-check 49 inputs
    assert_graph_io(step_sess, step_path)
    print(f"graph: {step_path} (single graph, all frames)")

    if args.weights:
        model = load_finetuned_model(args.weights, args.checkpoint)
        merge_lora(model.model.aggregator)
    else:
        model = load_base_model(args.ckpt)
    wrapper = StreamingDepthExport(
        model,
        image_hw=(args.height, args.width),
        window=args.window,
        rotate=args.rotate,
        batch_size=args.batch_size,
    )
    # --rotate / --height / --width are export-time constants baked into the
    # graph; this is where a mismatch with the flags given here is caught
    assert_graph_geometry(step_sess, wrapper, step_path)

    frames = None
    if args.spot_dir:
        if args.batch_size != 1:
            raise SystemExit(
                "--spot-dir yields one stream, so it cannot feed a "
                f"--batch-size {args.batch_size} graph; use random frames"
            )
        frames = spot_frames(args)
        got_hw = tuple(frames[0][0].shape[-2:])
        if got_hw != (args.height, args.width):
            raise SystemExit(
                f"Spot frames are {got_hw}, graph takes "
                f"{(args.height, args.width)} -- pass matching --height/--width"
            )
        print(f"frames: {len(frames)} real Spot frames from {args.spot_dir}")
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    eager_cache, ort_cache = None, None
    worst_depth, worst_conf, worst_cache = 0.0, 0.0, 0.0
    for i in range(args.frames):
        if frames is not None:
            rgb, d = frames[i]
        else:
            rgb, d = dummy_frame(
                args.height, args.width, args.seed * 1000 + i, batch=args.batch_size
            )
        ref_depth, ref_conf, eager_cache = wrapper._step(rgb, d, eager_cache)
        feed = {"rgb": rgb.numpy(), "sparse_depth": d.numpy()}
        cache = (
            ort_cache
            if ort_cache is not None
            else [t.numpy() for t in wrapper.empty_cache()]
        )
        feed.update(dict(zip(cache_input_names(), cache)))
        out = run_by_name(step_sess, feed)
        ort_cache = [out[n] for n in cache_output_names()]

        # the graph declares its depth/conf dims symbolically (the DPT head
        # interpolates to traced sizes), so the orientation contract can only
        # be checked on a real output -- this is what catches a --rotate that
        # disagrees with the export, before numpy broadcasts the difference
        if out["depth"].shape != tuple(ref_depth.shape):
            raise SystemExit(
                f"frame {i}: graph depth {out['depth'].shape}, eager "
                f"{tuple(ref_depth.shape)} -- --rotate/--height/--width do "
                "not match the values the graph was exported with"
            )
        d_depth = float(np.abs(out["depth"] - ref_depth.numpy()).max())
        d_conf = float(np.abs(out["depth_conf"] - ref_conf.numpy()).max())
        # _step returns the cache FLAT (48 tensors, block-major k,v) -- the
        # same order as cache_output_names(), so compare element-wise. The
        # LENGTHS are both 48 by construction; the frame AXIS is the one that
        # diverges when the graph's baked --window differs from the one given
        # here, and numpy would broadcast that into a confusing error (or,
        # worse, silently) a few frames in.
        if len(ort_cache) != len(eager_cache):
            raise SystemExit(
                f"cache length mismatch: ORT {len(ort_cache)} vs eager "
                f"{len(eager_cache)}"
            )
        got, want = ort_cache[0].shape[2], eager_cache[0].shape[2]
        if got != want:
            raise SystemExit(
                f"frame {i}: graph cache holds {got} frames, eager holds "
                f"{want} -- the graph was exported with a --window other than "
                f"the {args.window} given here (it clamped at {got})"
            )
        d_cache = max(
            float(np.abs(o - e.numpy()).max()) for o, e in zip(ort_cache, eager_cache)
        )
        worst_depth = max(worst_depth, d_depth)
        worst_conf = max(worst_conf, d_conf)
        worst_cache = max(worst_cache, d_cache)
        if args.dump_dir:
            # rgb is PRE-rotation, depth is POST: they SHOULD be 90 degrees
            # apart. The middle panel (rgb rotated the way the graph rotates
            # it) is the one to compare the depth against -- it is what tells
            # clockwise from counter-clockwise.
            # element 0 only: the panel exists to eyeball ORIENTATION, which is
            # baked and therefore identical across batch elements, and _to_pil
            # takes a single frame
            dump_depth(
                os.path.join(args.dump_dir, f"frame_{i:03d}_graph.png"),
                out["depth"][:1],
                rgb=rgb.numpy()[:1],
                rotate=args.rotate,
            )
        print(
            f"  frame {i}: |d depth| {d_depth:.3e}  |d conf| {d_conf:.3e}  "
            f"|d cache| {d_cache:.3e}"
        )

    if args.frames > args.window:
        print(f"window verified: graph cache clamps at {args.window} frames")
    else:
        print(
            f"NOTE: --frames {args.frames} <= --window {args.window}, so the "
            "cache never reached the clamp -- the graph's baked window is "
            f"unverified. Re-run with --frames {args.window + 2} to check it."
        )
    print(
        f"\nmax over {args.frames} frames: depth {worst_depth:.3e}, "
        f"conf {worst_conf:.3e}, cache {worst_cache:.3e} "
        f"(threshold {args.threshold:g})"
    )
    if worst_depth > args.threshold:
        raise SystemExit(f"FAIL: depth parity {worst_depth:.3e} > {args.threshold:g}")
    print("parity PASSED")


def main():
    args = parse_args()
    # torch (like onnxruntime) sizes its pool from the machine's cores, not
    # from our cpuset -- on a busy login node that is pure contention
    set_torch_threads()
    if args.cuda:
        set_providers("CUDAExecutionProvider", "CPUExecutionProvider", tf32=args.tf32)
    elif args.tf32:
        raise SystemExit("--tf32 only means anything with --cuda")
    if args.cpu_unit:
        run_cpu_unit(args)
    else:
        run_parity(args)


if __name__ == "__main__":
    main()
