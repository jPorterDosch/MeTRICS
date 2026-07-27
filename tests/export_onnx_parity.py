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
    still produce finite output.

    Budget 12-17 min on ONE core (two runs on a shared login-node core
    measured 735 s and 1030 s; the two torch.onnx.export calls were 349/357 s
    and 425/434 s respectively -- the spread is machine contention, not the
    code). Everything that is not an export is small: the five unit checks
    finish in under 4 s, and the probes / rollouts / orientation checks are
    seconds. The export IS the cost: the model is tiny
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
    # outputs stay in the rotated orientation by design (no inverse in the
    # pipeline), so only the input rotation is contract
    from streamvggt.export.wrapper import rotate_cw

    torch.manual_seed(2)
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
    import streamvggt.heads.utils as head_utils
    from streamvggt.export.wrapper import _make_sincos_pos_embed_f32

    if head_utils.make_sincos_pos_embed is _make_sincos_pos_embed_f32:
        raise AssertionError(
            "run this check before any StreamingDepthExport is constructed -- "
            "the constructor installs the patch process-wide"
        )
    pos = torch.arange(64, dtype=torch.float32)  # covers the largest grid axis
    for embed_dim in (64, 256):
        ref = head_utils.make_sincos_pos_embed(embed_dim, pos)
        got = _make_sincos_pos_embed_f32(embed_dim, pos)
        if got.dtype != ref.dtype or got.shape != ref.shape:
            raise AssertionError(
                f"sincos drop-in returns {got.dtype}{tuple(got.shape)}, "
                f"original {ref.dtype}{tuple(ref.shape)}"
            )
        diff = (ref.double() - got.double()).abs().max().item()
        if diff > 1e-5:
            raise AssertionError(
                f"float32 sincos embedding deviates by {diff:.2e} at "
                f"embed_dim={embed_dim} -- too far from the base model"
            )
    print(
        f"[cpu-unit] float32 sincos patch within {diff:.2e} of the original "
        "(x0.1 head ratio before it reaches the features)"
    )


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
    g = torch.Generator().manual_seed(7)
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
    import tempfile
    from types import SimpleNamespace

    from export_onnx import export_step, probe_frame0, validate_rollout
    from streamvggt.export import StreamingDepthExport

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
            check_graph_orientation(step, wrapper)
            if not rotate:  # one conversion is enough; rotation is orthogonal
                check_fp16_smoke(step, wrapper, ns)
        print(
            f"[export-smoke] rotate={rotate}: OK (single graph, {h}x{w} in -> "
            f"{expect_hw[0]}x{expect_hw[1]} out, 50 in / 50 out, divergent "
            "frame-0 token selection stayed dynamic)"
        )
    check_no_inverse_rotation(model)
    print("export smoke PASSED")


@torch.no_grad()
def check_graph_orientation(step_path, wrapper) -> None:
    """The EXPORTED graph emits the network orientation: [1, W, H] when the
    rotation is baked in, [1, H, W] when it is not. Asserted against the
    contract, not against the eager wrapper -- eager and graph share `_step`,
    so a rotation change moves both and their agreement proves nothing."""
    from export_onnx import ort_session, run_by_name
    from streamvggt.export import cache_input_names

    h, w = SMOKE_HW
    rgb, d = dummy_frame(h, w, seed=31)
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
    from export_onnx import check_fp16_numerics, convert_onnx_to_fp16

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
def check_no_inverse_rotation(model) -> None:
    """Pin the no-inverse-rotation contract by construction: rotating in-graph
    must be EQUIVALENT to pre-rotating the inputs and not rotating at all.
    Any rotate_ccw at the end of `_step` breaks this equality -- and, since
    the test resolution is non-square, even the shapes."""
    from streamvggt.export import StreamingDepthExport
    from streamvggt.export.wrapper import rotate_cw

    h, w = SMOKE_HW
    rgb, d = dummy_frame(h, w, seed=41)
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
    check_lora_merge()
    check_cache_roundtrip()
    check_rotation()
    check_sincos_patch_cost()  # MUST precede the first wrapper construction
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
        assert_graph_geometry,
        assert_graph_io,
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
    )
    # --rotate / --height / --width are export-time constants baked into the
    # graph; this is where a mismatch with the flags given here is caught
    assert_graph_geometry(step_sess, wrapper, step_path)

    eager_cache, ort_cache = None, None
    worst_depth, worst_conf, worst_cache = 0.0, 0.0, 0.0
    for i in range(args.frames):
        rgb, d = dummy_frame(args.height, args.width, args.seed * 1000 + i)
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
    from runtime_utils import set_torch_threads

    # torch (like onnxruntime) sizes its pool from the machine's cores, not
    # from our cpuset -- on a busy login node that is pure contention
    set_torch_threads()
    if args.cpu_unit:
        run_cpu_unit(args)
    else:
        run_parity(args)


if __name__ == "__main__":
    main()
