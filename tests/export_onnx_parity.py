"""Parity checks for the streaming ONNX export.

Two modes:

--cpu-unit (seconds, login node, no checkpoint): unit checks of the pure-math
    export helpers -- LoRA merge equals the wrapped forward, cache
    flatten/unflatten round-trips, the transpose+flip rotation equals
    torch.rot90, and the in-graph window slice equals the reference
    truncation semantics from the KV-window validation experiment.

full (checkpoint + exported graphs): N-frame rollout of random rgb/sparse
    depth through (1) the eager StreamingDepthExport wrapper (the PyTorch
    reference -- identical `_step` semantics to the graph by construction)
    and (2) the onnxruntime session(s), both on CPU so any difference is
    exporter fidelity rather than device numerics. Reports per-frame
    max-abs-diff for depth / conf / every cache tensor; FAILS if the depth
    diff exceeds --threshold.

Known & accepted: under the baked sliding window with near-inert
conditioning, long sequences drift in global scale (nothing re-anchors it);
measured in tests/kv_cache_window_validation.py. This test checks
graph-vs-eager fidelity, not that behavior.

Examples:
  python tests/export_onnx_parity.py --cpu-unit
  python tests/export_onnx_parity.py --onnx-prefix onnx/base \
      --ckpt ckpt/checkpoints.pth --frames 8 --window 4
  python tests/export_onnx_parity.py --onnx-prefix onnx/head_18075bb5 \
      --weights /oscar/scratch/jdosch/metrics/checkpoints/18075bb53f48343c \
      --checkpoint best --frames 8 --window 4
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


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
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument(
        "--window",
        type=int,
        default=4,
        help="MUST match the window the graphs were exported with",
    )
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument(
        "--rotate", action="store_true", help="MUST match the export-time --rotate"
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=2e-3,
        help="max allowed |depth diff| (use ~1e-1 for fp16 graphs)",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# --cpu-unit checks
# ---------------------------------------------------------------------------
def check_lora_merge() -> None:
    from streamvggt.export.lora_merge import merge_lora
    from streamvggt.depth_cond.lora import LoRALinear, LoRAQKV

    torch.manual_seed(0)
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
    from streamvggt.export.cache import (
        NUM_GLOBAL_BLOCKS,
        cache_input_names,
        cache_output_names,
        empty_cache,
        flatten_cache,
        unflatten_cache,
    )

    torch.manual_seed(1)
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
    from streamvggt.export.wrapper import rotate_ccw, rotate_cw

    torch.manual_seed(2)
    for shape in ((1, 3, 5, 7), (1, 5, 7)):
        x = torch.randn(*shape)
        if not torch.equal(rotate_cw(x), torch.rot90(x, k=-1, dims=(-2, -1))):
            raise AssertionError(f"rotate_cw != rot90(k=-1) for shape {shape}")
        if not torch.equal(rotate_ccw(x), torch.rot90(x, k=1, dims=(-2, -1))):
            raise AssertionError(f"rotate_ccw != rot90(k=1) for shape {shape}")
        if not torch.equal(rotate_ccw(rotate_cw(x)), x):
            raise AssertionError(f"rotate_ccw(rotate_cw(x)) != x for shape {shape}")
    print("[cpu-unit] transpose+flip rotation == torch.rot90 (and inverts)")


def check_window_slice() -> None:
    """The in-graph slice kv[:, :, -W:] must equal the reference truncation
    semantics validated in the KV-window experiment (keep the LAST W frame
    slices; no-op while n_frames <= W)."""
    torch.manual_seed(3)
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
    the cache contract asserts, yet exports on a login-node CPU in ~a minute.
    Exercises everything the exporter must survive: the cached-attention
    concat, dynamic frame axis, DPT interpolations, window Slice, rotation."""
    from streamvggt.heads.dpt_head import DPTHead
    from streamvggt.models.aggregator import Aggregator
    from streamvggt.models.streamvggt import StreamVGGT

    torch.manual_seed(4)
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
    return model.eval()


def run_export_smoke(args) -> None:
    """Full export -> frame-0 probe -> ORT rollout on the tiny model. Catches
    exporter/opset/runtime issues without a checkpoint or GPU."""
    import tempfile
    from types import SimpleNamespace

    from export_onnx import export_step, probe_frame0, validate_rollout
    from streamvggt.export import StreamingDepthExport

    model = build_tiny_model()
    for rotate in (False, True):
        wrapper = StreamingDepthExport(
            model, image_hw=(140, 140), window=3, rotate=rotate
        )
        ns = SimpleNamespace(height=140, width=140, window=3, opset=17)
        with tempfile.TemporaryDirectory() as td:
            step = export_step(wrapper, ns, os.path.join(td, "tiny_step.onnx"))
            probe_frame0(step, wrapper, ns)
            validate_rollout(step, wrapper, ns, n_frames=6)
        print(f"[export-smoke] rotate={rotate}: OK (single graph)")
    check_dynamic_token_select(model)
    print("export smoke PASSED")


def check_dynamic_token_select(model) -> None:
    """Pins the tracer behavior the single-graph contract rests on: the
    aggregator picks the frame-0 vs later-frame camera/register token slot
    from the cache's shape, and the tracer records that shape read
    symbolically, so the selection stays dynamic in the exported graph. The
    two slots are forced apart first (as in a trained checkpoint --
    build_tiny_model's random init leaves them at std 1e-6, numerically
    interchangeable, so the plain smoke above cannot see mis-tokening).
    probe_frame0's cache comparison then catches a baked slot at full
    magnitude (the token K/V rows land in the cache verbatim), and the
    rollout confirms frames >= 1 still get the later-frame slot. If a torch
    upgrade ever constant-folds the selection, this fails."""
    import tempfile
    from types import SimpleNamespace

    from export_onnx import export_step, probe_frame0, validate_rollout
    from streamvggt.export import StreamingDepthExport

    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        agg = model.aggregator
        agg.camera_token.copy_(torch.randn(agg.camera_token.shape, generator=g))
        agg.register_token.copy_(torch.randn(agg.register_token.shape, generator=g))

    wrapper = StreamingDepthExport(model, image_hw=(140, 140), window=3)
    ns = SimpleNamespace(height=140, width=140, window=3, opset=17)
    with tempfile.TemporaryDirectory() as td:
        step = export_step(wrapper, ns, os.path.join(td, "divergent_step.onnx"))
        probe_frame0(step, wrapper, ns)
        validate_rollout(step, wrapper, ns, n_frames=6)
    print("[export-smoke] divergent-token frame-0 selection stays dynamic: OK")


def run_cpu_unit(args) -> None:
    check_lora_merge()
    check_cache_roundtrip()
    check_rotation()
    check_window_slice()
    run_export_smoke(args)
    print("cpu-unit checks PASSED")


# ---------------------------------------------------------------------------
# full parity mode
# ---------------------------------------------------------------------------
def dummy_frame(h: int, w: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(1, 3, h, w, generator=g)
    d = torch.rand(1, 1, h, w, generator=g) * 4.0 + 0.5
    keep = (torch.rand(1, 1, h, w, generator=g) < 0.05).float()
    return rgb, d * keep


@torch.no_grad()
def run_parity(args) -> None:
    from export_onnx import (
        cache_input_names,
        cache_output_names,
        load_base_model,
        load_finetuned_model,
        merge_lora,
        ort_session,
        run_by_name,
    )
    from streamvggt.export import StreamingDepthExport

    if args.onnx_prefix is None:
        raise SystemExit("full mode needs --onnx-prefix (run export first)")
    step_path = f"{args.onnx_prefix}_step.onnx"
    if not os.path.isfile(step_path):
        raise SystemExit(f"missing {step_path}")
    step_sess = ort_session(step_path)
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
    )

    eager_cache, ort_cache = None, None
    worst_depth, worst_conf, worst_cache = 0.0, 0.0, 0.0
    for i in range(args.frames):
        rgb, d = dummy_frame(args.height, args.width, args.seed * 1000 + i)
        ref_depth, ref_conf, eager_cache = wrapper._step(rgb, d, eager_cache)
        feed = {"rgb": rgb.numpy(), "depth": d.numpy()}
        cache = (
            ort_cache
            if ort_cache is not None
            else [t.numpy() for t in wrapper.empty_cache()]
        )
        feed.update(dict(zip(cache_input_names(), cache)))
        out = run_by_name(step_sess, feed)
        ort_cache = [out[n] for n in cache_output_names()]

        d_depth = float(np.abs(out["depth"] - ref_depth.numpy()).max())
        d_conf = float(np.abs(out["depth_conf"] - ref_conf.numpy()).max())
        d_cache = max(
            float(np.abs(o - e.numpy()).max())
            for o, e in zip(ort_cache, [t for kv in eager_cache for t in kv])
        )
        worst_depth = max(worst_depth, d_depth)
        worst_conf = max(worst_conf, d_conf)
        worst_cache = max(worst_cache, d_cache)
        print(
            f"  frame {i}: |d depth| {d_depth:.3e}  |d conf| {d_conf:.3e}  "
            f"|d cache| {d_cache:.3e}"
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
    if args.cpu_unit:
        run_cpu_unit(args)
    else:
        run_parity(args)


if __name__ == "__main__":
    main()
