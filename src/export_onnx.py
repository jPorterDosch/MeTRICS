"""Export StreamVGGT / MetricStreamVGGT streaming depth to ONNX.

One ONNX call == one frame. The exported graph(s) take `rgb [1,3,H,W]` (in
[0,1]) and `depth [1,1,H,W]` (sparse METRIC depth; the validity mask is
derived in-graph as 0.01 m <= d <= 100 m) plus 48 KV-cache tensors
[1,16,n_frames,P,64] with a dynamic frame axis, and return `depth [1,H,W]`,
`depth_conf [1,H,W]` and the 48 updated cache tensors, already sliced to the
last `--window` frames. See streamvggt/export/wrapper.py for the full I/O
contract, the baked rotation, and the documented frame-0-eviction /
scale-drift caveats.

Single graph, all frames: frame 0 is served by feeding 48 zero-length
caches. The aggregator's frame-0 vs later-frame camera/register token choice
is derived from the cache's shape, which the tracer records symbolically, so
the selection stays dynamic in the exported graph (verified empirically: a
graph exported with deliberately divergent token slots matches eager at both
frame 0 and later frames -- see the divergent-token case in
tests/export_onnx_parity.py). Because that dynamism is a tracer behavior,
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
    if args.weights and args.variant and wrapper.variant != args.variant:
        raise SystemExit(
            f"--variant {args.variant} does not match the run's config "
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
    torch.onnx.export(
        StepWrapper(wrapper),
        (rgb, d, *cache),
        path,
        input_names=["rgb", "depth", *cache_input_names()],
        output_names=["depth", "depth_conf", *cache_output_names()],
        dynamic_axes=dyn,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    return path


# ---------------------------------------------------------------------------
# onnxruntime helpers
# ---------------------------------------------------------------------------
def ort_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def run_by_name(sess, feed: dict) -> dict:
    """Feed strictly by input name (never positional) and return outputs by
    name -- the graph's own I/O names are the contract."""
    names = {i.name for i in sess.get_inputs()}
    missing = names - feed.keys()
    if missing:
        raise KeyError(f"missing ONNX inputs: {sorted(missing)}")
    outs = sess.run(None, {k: v for k, v in feed.items() if k in names})
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
    feed = {"rgb": _np(rgb), "depth": _np(d)}
    for name, t in zip(cache_input_names(), wrapper.empty_cache()):
        feed[name] = _np(t)
    try:
        out = run_by_name(sess, feed)
    except Exception as e:  # zero-length concat unsupported, shape errors, ...
        raise SystemExit(f"frame-0 probe: graph rejects empty cache ({e})")
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
        feed = {"rgb": _np(rgb), "depth": _np(d)}
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
def convert_onnx_to_fp16(fp32_path: str, fp16_path: str) -> str:
    import onnx
    from onnxconverter_common import float16 as onnx_float16

    model = onnx.load(fp32_path)
    model_fp16 = onnx_float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, fp16_path)
    return fp16_path


def try_fp16(paths: list) -> None:
    for p in paths:
        fp16_path = p.replace(".onnx", "_fp16.onnx")
        try:
            convert_onnx_to_fp16(p, fp16_path)
            print(f"fp16: wrote {fp16_path}")
        except Exception as e:
            print(f"fp16: conversion of {p} failed (best-effort, not fatal): {e}")


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
        help="bake a 90-degree-clockwise input rotation (and the inverse on "
        "outputs) into the graph",
    )
    ap.add_argument(
        "--fp16", action="store_true", help="also write _fp16 variants (best-effort)"
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
        try_fp16([step_path])
    print("done:", step_path)


if __name__ == "__main__":
    main()
