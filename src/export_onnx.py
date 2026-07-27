"""Export StreamVGGT / MetricStreamVGGT streaming depth to ONNX.

One ONNX call == one frame. The exported graph(s) take `rgb [1,3,H,W]` (in
[0,1]) and `sparse_depth [1,1,H,W]` (sparse METRIC depth; the validity mask
is derived in-graph as 0.01 m <= d <= 100 m) plus 48 KV-cache tensors
[1,16,n_frames,P,64] with a dynamic frame axis, and return `depth` /
`depth_conf` in the NETWORK orientation ([1,W,H] with --rotate, [1,H,W]
otherwise -- rotated outputs are not rotated back) and the 48 updated cache
tensors, already sliced to the last `--window` frames. See
streamvggt/export/wrapper.py for the full I/O contract, the baked rotation,
and the documented frame-0-eviction / scale-drift caveats.

Every variant exports the SAME 50 inputs / 50 outputs -- including the base
arm, which ignores `sparse_depth`'s values but must still expose the input
(the tracer prunes inputs nothing consumes, so the wrapper ties it in with an
exactly-zero term). The input is NOT called `depth`: that name belongs to the
predicted output, and the exporter silently renames a collision (`depth` ->
`depth.1`). assert_graph_io re-checks the contract against the loaded graph
on every export, and run_by_name refuses any feed that does not match it
name-for-name.

Single graph, all frames: frame 0 is served by feeding 48 zero-length
caches. The aggregator's frame-0 vs later-frame camera/register token choice
is derived from the cache's shape, which the tracer records symbolically, so
the selection stays dynamic in the exported graph (verified empirically: a
graph exported with deliberately divergent token slots matches eager at both
frame 0 and later frames -- the smoke model in tests/export_onnx_parity.py
forces those slots apart). Because that dynamism is a tracer behavior,
not a language guarantee, every export runs a frame-0 probe against the
eager wrapper that ASSERTS it held (depth AND cache outputs, gate 1e-3; the
cache contains the token rows verbatim, so a mis-tokened frame 0 cannot
hide) -- the export aborts rather than ship a graph that mis-tokens frame 0.

Examples (compute node; the base checkpoint is ~5 GB in fp32):
  python src/export_onnx.py --ckpt ckpt/checkpoints.pth --variant base \
      --out onnx/base
  python src/export_onnx.py --weights /path/to/run_dir --checkpoint best \
      --rotate --out onnx/head_run
"""

import argparse
import functools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamvggt.depth_cond import MetricStreamVGGT  # noqa: E402
from streamvggt.export import (  # noqa: E402
    StepWrapper,
    StreamingDepthExport,
    cache_input_names,
    cache_output_names,
    merge_lora,
)
from runtime_utils import n_cpus, set_torch_threads  # noqa: E402
from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from visualize_depth import (  # noqa: E402
    load_saved_args,
    rebuild_metric_cfg,
    resolve_checkpoint,
)

PROBE_GATE = 1e-3  # frame-0 probe: max|diff| must stay below this (assertion)


# ---------------------------------------------------------------------------
# checkpoint loading (same rules as tests/kv_cache_window_validation.py)
# ---------------------------------------------------------------------------
def load_base_model(ckpt_path: str) -> StreamVGGT:
    model = StreamVGGT()
    sd = torch.load(ckpt_path, map_location="cpu")
    # raw state_dict, or {"model": state_dict} wrapper -- same unwrap rule as
    # MetricStreamVGGT.load_pretrained
    if (
        isinstance(sd, dict)
        and "model" in sd
        and not any(k.startswith("aggregator.") for k in sd)
    ):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    return model.eval()


def load_finetuned_model(weights: str, which: str) -> MetricStreamVGGT:
    """Rebuild the run's MetricStreamVGGT from its saved config and load the
    finetuned weights. LoRA must be applied BEFORE the load (wrapping renames
    keys); the encoder feature cache is dead weight here and would mkdir the
    training run's cache dir, so it is disabled."""
    ckpt_path = resolve_checkpoint(weights, which)
    print(f"loading finetuned run: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = rebuild_metric_cfg(load_saved_args(ckpt))
    mcfg.encoder_cache.enabled = False
    model = MetricStreamVGGT(mcfg)
    model.apply_lora_adapters()
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def build_wrapper(args) -> StreamingDepthExport:
    if args.weights:
        model = load_finetuned_model(args.weights, args.checkpoint)
        merged = merge_lora(model.model.aggregator)
        print(f"merged {merged} LoRA-wrapped linears into base weights")
    else:
        model = load_base_model(args.ckpt)
    wrapper = StreamingDepthExport(
        model,
        image_hw=(args.height, args.width),
        window=args.window,
        rotate=args.rotate,
    )
    if args.variant and wrapper.variant != args.variant:
        source = "the run's config" if args.weights else "--ckpt (base implied)"
        raise SystemExit(
            f"--variant {args.variant} does not match {source} "
            f"({wrapper.variant}); drop --variant to infer it"
        )
    print(
        f"wrapper variant: {wrapper.variant} | net input HxW: {wrapper.net_hw} "
        f"| window: {wrapper.window} | rotate: {wrapper.rotate}"
    )
    return wrapper


# ---------------------------------------------------------------------------
# dummy inputs
# ---------------------------------------------------------------------------
def dummy_frame(args, seed: int = 0):
    """rgb in [0,1]; sparse depth: ~5% of pixels carry 0.5-4.5 m readings,
    the rest are 0 (invalid under the in-graph 0.01-100 m mask)."""
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(1, 3, args.height, args.width, generator=g)
    d = torch.rand(1, 1, args.height, args.width, generator=g) * 4.0 + 0.5
    keep = (torch.rand(1, 1, args.height, args.width, generator=g) < 0.05).float()
    return rgb, d * keep


@torch.no_grad()
def real_cache(wrapper: StreamingDepthExport, args, n_frames: int = 2):
    """Roll the eager wrapper for n_frames to get a genuine cache -- traced
    shapes and values come from the real model, not hand-built zeros. n>=2 so
    slice_expand_and_flatten(...)[-1:] bakes the generic later-frame
    camera/register token, not frame 0's distinct one."""
    cache = None
    for i in range(n_frames):
        rgb, d = dummy_frame(args, seed=i)
        _, _, cache = wrapper._step(rgb, d, cache)
    return cache


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export_step(wrapper: StreamingDepthExport, args, path: str) -> str:
    rgb, d = dummy_frame(args, seed=10)
    cache = real_cache(wrapper, args, n_frames=2)
    dyn = {n: {2: "n_frames"} for n in cache_input_names()}
    dyn.update({n: {2: "n_frames_out"} for n in cache_output_names()})
    print(f"exporting step graph -> {path} (opset {args.opset})")
    input_names, output_names = graph_io_names()
    torch.onnx.export(
        StepWrapper(wrapper),
        (rgb, d, *cache),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    ort_session.cache_clear()  # never serve a session built from the old file
    return path


# ---------------------------------------------------------------------------
# onnxruntime helpers
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def ort_session(path: str):
    """Cached because building a session over this graph is expensive (~20 s
    for the tiny smoke model) and the probe / rollout / orientation checks all
    run the SAME file; export_step clears the cache, so a re-export to the
    same path can never be served a stale session.

    Threads are sized from the affinity mask: left alone, onnxruntime builds
    its pool from the machine's core count (48 here) and then tries to pin one
    thread per core -- on a restricted node that is a wall of
    pthread_setaffinity_np errors followed by 48 threads fighting over the one
    core we were given."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_cpus()
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def graph_io_names() -> tuple[list[str], list[str]]:
    """The I/O contract every exported graph must satisfy, in one place --
    also the names handed to torch.onnx.export, so the two cannot drift.

    The sparse-depth INPUT is not called `depth`: ONNX value names share one
    namespace, and the exporter silently renames a collision (input `depth`
    became `depth.1` once it stopped being pruned). `depth` is the predicted
    output; `sparse_depth` is the measured input."""
    return (
        ["rgb", "sparse_depth", *cache_input_names()],
        ["depth", "depth_conf", *cache_output_names()],
    )


def assert_graph_io(sess, path: str) -> None:
    """Fail loudly if the exported graph's I/O deviates from the contract.

    Two failure modes, both silent and both observed: the tracer PRUNES an
    input no op consumes (the base arm ignores the sparse depth, so its graph
    shipped with 49 inputs), and it RENAMES an input whose name collides with
    an output (`depth` -> `depth.1`). Either way the consumer feeds a tensor
    the graph does not have. Checked on every export, before any parity
    number is reported."""
    want_in, want_out = graph_io_names()
    collide = sorted(set(want_in) & set(want_out))
    if collide:
        raise SystemExit(
            f"I/O contract is ambiguous: {collide} used as both input and "
            "output name; the exporter will rename one of them"
        )
    got_in = [i.name for i in sess.get_inputs()]
    got_out = [o.name for o in sess.get_outputs()]
    problems = []
    for kind, want, got in (("input", want_in, got_in), ("output", want_out, got_out)):
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        if missing:
            problems.append(f"{kind}s dropped by the exporter: {missing}")
        if extra:
            problems.append(f"unexpected {kind}s: {extra}")
    if problems:
        raise SystemExit(
            f"{path}: graph I/O contract violated ({len(got_in)} inputs / "
            f"{len(got_out)} outputs, expected {len(want_in)}/{len(want_out)})"
            + "".join(f"\n  - {p}" for p in problems)
        )


def assert_graph_geometry(sess, wrapper: StreamingDepthExport, path: str) -> None:
    """Check the graph's DECLARED shapes against the wrapper's -- this is
    where a --height / --width mismatch with the export is caught, at load
    time, instead of as a confusing ORT error mid-rollout.

    Only dims the graph states as integers are compared. The INPUTS are fully
    static; the depth/conf OUTPUTS are not -- the DPT head sizes its
    interpolations from traced tensor shapes, so the exporter declares them
    symbolically ('Gatherdepth_dim_1') even though they are constant in
    practice. The output orientation is therefore checked on a real run
    instead (validate_rollout here, run_parity in the parity test), which
    also covers --rotate: `rgb` is PRE-rotation, so its shape is identical
    with and without the baked rotation and cannot tell them apart. At a
    SQUARE resolution nothing can -- a rotation mismatch is a silent
    transpose there, and only the parity diff exposes it."""
    want = {
        "rgb": [1, 3, *wrapper.image_hw],
        "sparse_depth": [1, 1, *wrapper.image_hw],
        "depth": [1, *wrapper.net_hw],
        "depth_conf": [1, *wrapper.net_hw],
    }
    got = {v.name: v.shape for v in (*sess.get_inputs(), *sess.get_outputs())}
    bad = []
    for name, wanted in want.items():
        if name not in got:
            continue
        stated = got[name]
        if len(stated) != len(wanted) or any(
            isinstance(g, int) and g != w for g, w in zip(stated, wanted)
        ):
            bad.append(f"{name}: graph {stated}, expected {wanted}")
    if bad:
        raise SystemExit(
            f"{path}: graph geometry does not match this wrapper "
            f"(rotate={wrapper.rotate}, input HxW={wrapper.image_hw})"
            + "".join(f"\n  - {b}" for b in bad)
        )


def run_by_name(sess, feed: dict) -> dict:
    """Feed strictly by input name (never positional) and return outputs by
    name -- the graph's own I/O names are the contract. Extra keys are an
    ERROR, not something to filter out: quietly dropping them is exactly what
    hides an input the exporter pruned."""
    names = {i.name for i in sess.get_inputs()}
    missing = sorted(names - feed.keys())
    extra = sorted(feed.keys() - names)
    if missing or extra:
        raise KeyError(
            f"ONNX feed does not match the graph: missing {missing}, "
            f"not in graph {extra}"
        )
    outs = sess.run(None, feed)
    return dict(zip([o.name for o in sess.get_outputs()], outs))


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def probe_frame0(step_path: str, wrapper: StreamingDepthExport, args) -> float:
    """Assert the graph serves frame 0 (zero-length caches): compares the ORT
    depth AND cache outputs against the eager wrapper's true frame-0 step.
    Guards the two ways single-graph mode can break: ORT rejecting a
    zero-sized Concat input, and the tracer constant-folding the frame-0 vs
    later-frame token selection. The cache check is the decisive one -- the
    K/V rows of the camera/register tokens land in the cache verbatim, so a
    baked later-frame token shows up there at full magnitude even when the
    depth output barely depends on it."""
    rgb, d = dummy_frame(args, seed=20)
    ref_depth, _, ref_cache = wrapper._step(rgb, d, None)
    sess = ort_session(step_path)
    # contract first: a pruned input or a geometry mismatch would otherwise
    # surface as a confusing runtime failure below
    assert_graph_io(sess, step_path)
    assert_graph_geometry(sess, wrapper, step_path)
    feed = {"rgb": _np(rgb), "sparse_depth": _np(d)}
    for name, t in zip(cache_input_names(), wrapper.empty_cache()):
        feed[name] = _np(t)
    try:
        out = run_by_name(sess, feed)
    except Exception as e:
        # The interesting failure is ORT refusing a zero-length input (empty
        # Concat / Slice on an empty axis). Anything else -- a name mismatch,
        # a shape rule, an unimplemented op -- is a DIFFERENT bug, so report
        # the exception type instead of asserting a diagnosis.
        raise SystemExit(
            f"frame-0 probe: ORT failed on zero-length caches [{type(e).__name__}] {e}"
        )
    d_depth = float(np.abs(out["depth"] - _np(ref_depth)).max())
    d_cache = max(
        float(np.abs(out[n] - _np(t)).max())
        for n, t in zip(cache_output_names(), ref_cache)
    )
    print(
        f"frame-0 probe: max diff vs eager frame-0: depth {d_depth:.3e}, "
        f"cache {d_cache:.3e} (gate {PROBE_GATE:g})"
    )
    worst = max(d_depth, d_cache)
    if worst >= PROBE_GATE:
        raise SystemExit(
            f"frame-0 probe FAILED: depth {d_depth:.3e} / cache {d_cache:.3e} "
            f">= {PROBE_GATE:g} -- the token selection did not stay dynamic "
            "through export; the graph would mis-token frame 0"
        )
    return worst


@torch.no_grad()
def validate_rollout(
    step_path: str,
    wrapper: StreamingDepthExport,
    args,
    n_frames=None,
) -> None:
    """ORT rollout: shapes, dtypes, cache growth 1,2,...,clamp-at-window, and
    a reported (not asserted -- the parity test owns thresholds) max diff vs
    the eager wrapper running the same frames."""
    n_frames = n_frames or args.window + 2
    step_sess = ort_session(step_path)
    eager_cache = None
    ort_cache = None
    worst = 0.0
    for i in range(n_frames):
        rgb, d = dummy_frame(args, seed=100 + i)
        ref_depth, ref_conf, eager_cache = wrapper._step(rgb, d, eager_cache)
        feed = {"rgb": _np(rgb), "sparse_depth": _np(d)}
        cache = (
            ort_cache
            if ort_cache is not None
            else [_np(t) for t in wrapper.empty_cache()]
        )
        feed.update(dict(zip(cache_input_names(), cache)))
        out = run_by_name(step_sess, feed)
        ort_cache = [out[n] for n in cache_output_names()]

        expect_frames = min(i + 1, wrapper.window)
        got = ort_cache[0].shape[2]
        if got != expect_frames:
            raise RuntimeError(
                f"frame {i}: cache n_frames {got}, expected {expect_frames}"
            )
        if out["depth"].shape != tuple(ref_depth.shape):
            raise RuntimeError(
                f"frame {i}: depth shape {out['depth'].shape}, "
                f"expected {tuple(ref_depth.shape)}"
            )
        if out["depth_conf"].shape != tuple(ref_conf.shape):
            raise RuntimeError(
                f"frame {i}: depth_conf shape {out['depth_conf'].shape}, "
                f"expected {tuple(ref_conf.shape)}"
            )
        if not np.isfinite(out["depth"]).all():
            raise RuntimeError(f"frame {i}: non-finite depth")
        worst = max(worst, float(np.abs(out["depth"] - _np(ref_depth)).max()))
    print(
        f"rollout OK: {n_frames} frames, cache clamps at window="
        f"{wrapper.window}; max|depth diff| vs eager = {worst:.3e}"
    )


# ---------------------------------------------------------------------------
# fp16 (best-effort; ported from the PromptDA script, IO kept fp32)
# ---------------------------------------------------------------------------
# onnxconverter-common builds the fp16 model as ONE protobuf, which caps at
# 2 GB. The 1B checkpoint is ~5 GB fp32 / ~2.4 GB fp16, so --fp16 CANNOT
# succeed for it; it is useful for small models only. Intended behavior --
# warned about up front and per-file, never fatal.
FP16_PROTOBUF_LIMIT = 2 * 1024**3


def weight_bytes(path: str) -> int:
    """Bytes of initializer data in an ONNX model, EXTERNAL data included.

    os.path.getsize is useless here: past 2 GB torch writes the weights to
    sibling files and the .onnx keeps only the graph skeleton, so the 5 GB
    model whose fp16 form cannot fit in a protobuf looks like a few MB on
    disk. External tensors carry their size in an `external_data` entry keyed
    "length"; internal ones are measured from the proto."""
    import onnx

    model = onnx.load(path, load_external_data=False)
    total = 0
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            total += sum(
                int(kv.value) for kv in init.external_data if kv.key == "length"
            )
        else:
            total += len(init.raw_data) or init.ByteSize()
    return total


def warn_fp16(fp32_paths: list) -> None:
    print(
        "WARNING: --fp16 is best-effort. onnxconverter-common serializes the "
        "converted model as a single protobuf (2 GB hard limit), so any graph "
        "whose fp16 weights exceed that -- the 1B checkpoint at ~2.4 GB does "
        "-- will fail here. The fp32 export above is unaffected."
    )
    for p in fp32_paths:
        if not os.path.isfile(p):
            continue
        try:
            fp32 = weight_bytes(p)
        except Exception as e:  # size warning must never break the export
            print(f"fp16: could not size {p} [{type(e).__name__}] {e}")
            continue
        # fp16 halves the fp32 weights; IO stays fp32 but is negligible
        if fp32 / 2 > FP16_PROTOBUF_LIMIT:
            print(
                f"WARNING: {p} holds {fp32 / 1024**3:.1f} GB of fp32 weights, "
                f"so its fp16 form (~{fp32 / 2 / 1024**3:.1f} GB) exceeds the "
                "2 GB protobuf limit -- the conversion below is expected to "
                "fail."
            )


def convert_onnx_to_fp16(fp32_path: str, fp16_path: str) -> str:
    import onnx
    from onnxconverter_common import float16 as onnx_float16

    model = onnx.load(fp32_path)
    model_fp16 = onnx_float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, fp16_path)
    return fp16_path


@torch.no_grad()
def check_fp16_numerics(fp16_path: str, wrapper: StreamingDepthExport, args) -> bool:
    """Run one frame through the fp16 graph and refuse to ship a broken one.

    A structurally valid fp16 graph can still be numerically dead: any value
    that overflows fp16 becomes inf, and inf reaching a multiply or subtract
    turns the whole map into NaN (that is exactly how a valid-pixel COUNT tied
    into the base graph poisoned it -- see wrapper._zero_from). check_model
    cannot see this; only running it can. Non-finite output DELETES the file
    rather than leave a graph that looks exportable lying next to the good
    one. The reported diff is informational -- fp16 legitimately differs from
    fp32 by ~1e-1, which is why the parity test takes a --threshold."""
    rgb, d = dummy_frame(args, seed=30)
    ref_depth, _, _ = wrapper._step(rgb, d, None)
    sess = ort_session(fp16_path)
    assert_graph_io(sess, fp16_path)
    assert_graph_geometry(sess, wrapper, fp16_path)
    feed = {"rgb": _np(rgb), "sparse_depth": _np(d)}
    feed.update(dict(zip(cache_input_names(), (_np(t) for t in wrapper.empty_cache()))))
    out = run_by_name(sess, feed)
    finite = {n: bool(np.isfinite(out[n]).all()) for n in ("depth", "depth_conf")}
    if not all(finite.values()):
        ort_session.cache_clear()  # release the file before unlinking it
        os.remove(fp16_path)
        print(
            f"ERROR: {fp16_path} produces non-finite output "
            f"({', '.join(n for n, ok in finite.items() if not ok)}) -- an "
            "fp16 overflow somewhere in the graph. Removed it; the fp32 graph "
            "is unaffected."
        )
        return False
    diff = float(np.abs(out["depth"] - _np(ref_depth)).max())
    print(f"fp16: frame-0 output finite, max|depth diff| vs fp32 eager = {diff:.3e}")
    return True


def try_fp16(paths: list, wrapper: StreamingDepthExport, args) -> None:
    warn_fp16(paths)
    for p in paths:
        fp16_path = p.replace(".onnx", "_fp16.onnx")
        try:
            convert_onnx_to_fp16(p, fp16_path)
            print(f"fp16: wrote {fp16_path}")
        except Exception as e:
            print(
                f"WARNING: fp16 conversion of {p} failed (best-effort, not "
                f"fatal; the fp32 graph is still valid) [{type(e).__name__}] {e}"
            )
            continue
        try:
            check_fp16_numerics(fp16_path, wrapper, args)
        except Exception as e:
            print(
                f"WARNING: could not validate {fp16_path} "
                f"[{type(e).__name__}] {e} -- treat it as unverified"
            )


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("model source (pick one)")
    src.add_argument(
        "--ckpt",
        default="ckpt/checkpoints.pth",
        help="base pretrained StreamVGGT checkpoint",
    )
    src.add_argument(
        "--weights",
        default=None,
        help="finetune_depth.py run dir (or checkpoint file) -> "
        "MetricStreamVGGT with the run's conditioning arm",
    )
    src.add_argument(
        "--checkpoint", default="auto", choices=["auto", "best", "last", "final"]
    )
    ap.add_argument(
        "--variant",
        default=None,
        choices=["base", "token", "head"],
        help="sanity check against the run's config (inferred from --weights; "
        "'base' implied without it)",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=392,
        help="input height BEFORE rotation (multiple of 14 after rotation)",
    )
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument(
        "--window",
        type=int,
        default=20,
        help="KV-cache sliding window, baked into the graph",
    )
    ap.add_argument(
        "--rotate",
        action="store_true",
        help="bake a 90-degree-clockwise input rotation into the graph; "
        "outputs stay in the rotated orientation",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="also write _fp16 variants (best-effort; fails above the 2 GB "
        "protobuf limit, i.e. for the 1B checkpoint -- warns, never fatal)",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--out",
        required=True,
        help="output prefix; writes <out>_step.onnx (one graph serves every "
        "frame -- feed zero-length caches at frame 0)",
    )
    ap.add_argument(
        "--no-fallback-sdpa",
        action="store_true",
        help="do not retry with unfused attention if the SDPA op fails to export",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    from torch.onnx import _constants

    max_opset = _constants.ONNX_MAX_OPSET
    if args.opset > max_opset:
        raise SystemExit(f"--opset {args.opset} > torch max {max_opset}")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    # same reason as ort_session's thread sizing: torch also defaults to the
    # machine's core count, which oversubscribes a cpuset-restricted node
    set_torch_threads()

    wrapper = build_wrapper(args)
    step_path = f"{args.out}_step.onnx"
    try:
        export_step(wrapper, args, step_path)
    except Exception as e:
        if args.no_fallback_sdpa:
            raise
        print(f"step export failed ({e}); retrying with unfused attention")
        wrapper.set_fused_attention(False)
        export_step(wrapper, args, step_path)

    probe_frame0(step_path, wrapper, args)
    validate_rollout(step_path, wrapper, args)
    if args.fp16:
        try_fp16([step_path], wrapper, args)
    print("done:", step_path)


if __name__ == "__main__":
    main()
