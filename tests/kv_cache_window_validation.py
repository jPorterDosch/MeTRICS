"""Validate KV-cache sliding-window streaming against full-cache streaming.

The ONNX export wrapper will bake a fixed-size sliding window into the
streaming graph (the cache otherwise grows ~O(frames) without bound; see
layers/attention.py -- the K/V concat has no eviction). The model was never
trained with truncated context, so before trusting the exported graph this
script measures what truncation actually costs, on real data, with the exact
per-frame loop the wrapper will replicate:

  aggregator(use_cache=True) -> truncate cache to last W frames -> depth head

Two arms per clip, same frames, same weights:
  full    -- cache grows unbounded (reference; today's inference() behavior)
  window  -- cache truncated to the last W frame slices after every step

Reported per frame index, averaged over clips:
  AbsRel / delta<1.25 vs GT (per-frame median-scaled: the base checkpoint is
  scale-ambiguous, and windowing questions are about consistency, not scale)
  divergence -- mean |d_win - d_full| / d_full over GT-valid pixels; scale-free
  and the most direct read on what truncation changes. Exactly 0 until the
  window first slides (a built-in sanity check on the loop).

Uses ScanNet test split (long ordered sequences, stride 1, real GT depth).
Depth head only: the export target drops camera/point/track heads, and depth
needs no pose.

Model selection:
  default            -- base pretrained StreamVGGT (ckpt/checkpoints.pth)
  --weights RUN_DIR  -- a finetune_depth.py run (MetricStreamVGGT: rebuilds
                        the arch from the checkpoint's saved config, applies
                        LoRA before loading, feeds per-frame sparse-depth
                        conditioning simulated with the run's own sim
                        settings; the SAME sparse input is shared by both
                        arms so the comparison stays about the cache)

Requires a GPU. Examples (from the repo root, on a GPU node):
  python tests/kv_cache_window_validation.py --out-dir viz/kv_window_validation
  python tests/kv_cache_window_validation.py \
      --weights /oscar/scratch/jdosch/metrics/<run_id> --checkpoint best \
      --out-dir viz/kv_window_validation
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402
from streamvggt.datasets import *  # noqa: E402,F403 (registers ScanNet_Multi)
from streamvggt.depth_cond import MetricStreamVGGT  # noqa: E402
from streamvggt.depth_cond.sparse import simulate_sparse_depth  # noqa: E402
from visualize_depth import (  # noqa: E402
    load_saved_args,
    rebuild_metric_cfg,
    resolve_checkpoint,
)

RESOLUTION = (518, 392)  # (W, H), multiple of the 14px patch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/gpfs/data/jtompki1/cli277/metric/processed_scannet",
        help="processed ScanNet root (test split is used)",
    )
    ap.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "ckpt", "checkpoints.pth"),
        help="base pretrained StreamVGGT checkpoint (raw state_dict); "
        "ignored when --weights is given",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="finetune_depth.py run dir (or checkpoint file) -> load the "
        "finetuned MetricStreamVGGT instead of the base model",
    )
    ap.add_argument(
        "--checkpoint",
        default="auto",
        choices=["auto", "best", "last", "final"],
        help="which checkpoint of --weights to load",
    )
    ap.add_argument("--window", type=int, default=20, help="cache window W (frames)")
    ap.add_argument(
        "--num-views",
        type=int,
        default=60,
        help="frames per clip; must exceed --window to exercise truncation "
        "(default 3W so the windowed phase dominates)",
    )
    ap.add_argument("--clips", type=int, default=2, help="number of clips")
    ap.add_argument("--seed", type=int, default=0, help="clip selection seed")
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="run the aggregator in fp32 instead of bf16 autocast (2x cache "
        "memory; use if bf16 noise is suspected of masking small divergence)",
    )
    ap.add_argument(
        "--dump-heatmaps",
        action="store_true",
        help="also write per-frame depth heatmaps (full/window, turbo) and "
        "relative-difference maps (inferno, saturating at 10%%) to "
        "<out-dir>/heatmaps, named for src/heatmaps_to_gif.py "
        "(e.g. --compare full_clip0 window_clip0)",
    )
    ap.add_argument(
        "--out-dir", default=os.path.join(ROOT, "viz", "kv_window_validation")
    )
    return ap.parse_args()


def load_base_model(ckpt_path: str, device: torch.device) -> StreamVGGT:
    model = StreamVGGT()
    sd = torch.load(ckpt_path, map_location="cpu")
    # same unwrap rule as MetricStreamVGGT.load_pretrained
    if (
        isinstance(sd, dict)
        and "model" in sd
        and not any(k.startswith("aggregator.") for k in sd)
    ):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def load_finetuned_model(
    weights: str, which: str, device: torch.device
) -> MetricStreamVGGT:
    """Rebuild the run's MetricStreamVGGT and load its weights, mirroring
    visualize_depth.py: architecture from the checkpoint's saved config, LoRA
    applied BEFORE the load (wrapping renames keys), full finetuned state_dict
    on top (so no separate pretrained load is needed)."""
    ckpt_path = resolve_checkpoint(weights, which)
    print(f"Loading finetuned run: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = rebuild_metric_cfg(load_saved_args(ckpt))
    # the loop below calls the aggregator directly and never passes cache_key,
    # so the encoder feature cache is dead weight -- and constructing it would
    # mkdir the training run's cache dir from here
    mcfg.encoder_cache.enabled = False
    model = MetricStreamVGGT(mcfg)
    model.apply_lora_adapters()
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def truncate_cache(past_kv: list, window: int) -> list:
    """Keep only the last `window` frame slices of every block's (K, V).

    K/V are [B, heads, n_frames, P, head_dim]; dim 2 is the frame axis the
    cached attention concatenates on. This slice is the exact op the ONNX
    wrapper will apply to its cache outputs.

    NOTE (frame-0 eviction risk): once n_frames exceeds `window`, frame 0's
    K/V leave the cache. VGGT-family models anchor their world frame to frame
    0, so the camera/point heads could drift under a pure sliding window; a
    StreamingLLM-style sink (always retain kv[:, :, :1]) is the cheap fix if
    this experiment -- or a future export that keeps those heads -- shows
    drift. Depth is per-frame and expected to be the most robust to eviction.
    """
    out = []
    for kv in past_kv:
        if kv is None:
            out.append(None)
        else:
            k, v = kv
            out.append((k[:, :, -window:], v[:, :, -window:]))
    return out


@torch.no_grad()
def run_stream(
    model,
    views: list[dict],
    device: torch.device,
    window: int | None,
    amp: bool,
) -> list[np.ndarray]:
    """Per-frame causal streaming, depth head only. Mirrors the inference()
    paths (StreamVGGT / MetricStreamVGGT) but owns the cache so it can
    truncate between steps. Returns the [H, W] fp32 depth prediction per
    frame."""
    metric = isinstance(model, MetricStreamVGGT)
    inner = model.model if metric else model
    past_kv = [None] * inner.aggregator.depth
    preds = []
    for i, view in enumerate(views):
        # dataset img is ImgNorm [-1,1]; the model wants [0,1] (same rescale
        # as finetune_depth._prepare_batch)
        images = ((view["img"] + 1.0) / 2.0)[None, None].to(device)  # [1,1,3,H,W]

        # per-frame conditioning, exactly as MetricStreamVGGT.inference: token
        # feats enter the aggregator (before the cache), head residuals enter
        # the DPT head (after it). No-op for the base model / disabled arm.
        token_feats, depth_residuals = None, None
        if metric and model.conditioner is not None:
            cond = model._route_conditioning([view], images)
            token_feats = cond["depth_token_feats"]
            if cond["depth_head_residuals"] is not None:
                depth_residuals = cond["depth_head_residuals"]["depth"]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            tokens, patch_start_idx, past_kv = inner.aggregator(
                images,
                past_key_values=past_kv,
                use_cache=True,
                past_frame_idx=i,
                injected_patch_feats=token_feats,
            )
        # heads run fp32 (model.forward disables autocast around them)
        tokens32 = [t.float() for t in tokens]
        depth, _conf = inner.depth_head(
            tokens32,
            images=images,
            patch_start_idx=patch_start_idx,
            depth_residuals=depth_residuals,
        )
        preds.append(depth[0, 0, :, :, 0].float().cpu().numpy())
        if window is not None:
            past_kv = truncate_cache(past_kv, window)
    return preds


# same encodings as visualize_depth._export_heatmaps, so these drop into the
# existing viewer/GIF workflow: turbo depth on a per-clip robust range shared
# by both arms, inferno relative maps saturating at 10%
_REL_VMAX = 0.10


def dump_heatmaps(
    hm_dir: str, clip_no: int, preds: dict[str, list[np.ndarray]]
) -> None:
    """Write {full,window}_clip<N>_depth_FFF.png plus wdiff_clip<N>_depth_FFF
    (|window-full|/full). heatmaps_to_gif.py groups on the trailing _FFF."""
    os.makedirs(hm_dir, exist_ok=True)
    turbo = matplotlib.colormaps["turbo"]
    inferno = matplotlib.colormaps["inferno"]
    stack = np.stack(preds["full"])
    vmin, vmax = np.percentile(stack, 2), np.percentile(stack, 98)
    span = max(vmax - vmin, 1e-6)
    for f, (df, dw) in enumerate(zip(preds["full"], preds["window"])):
        for arm, d in (("full", df), ("window", dw)):
            rgba = turbo(np.clip((d - vmin) / span, 0.0, 1.0))
            plt.imsave(
                os.path.join(hm_dir, f"{arm}_clip{clip_no}_depth_{f:03d}.png"), rgba
            )
        rel = np.abs(dw - df) / np.maximum(df, 1e-8)
        rgba = inferno(np.clip(rel / _REL_VMAX, 0.0, 1.0))
        plt.imsave(os.path.join(hm_dir, f"wdiff_clip{clip_no}_depth_{f:03d}.png"), rgba)


def plot_curves(rows: list, window: int, num_views: int, out_path: str, desc: str):
    """Two-panel per-frame-index summary (mean over clips): does quality move
    when the window starts sliding? Divergence is exactly 0 until frame W+1 by
    construction, so the interesting signal is everything right of the line."""

    def series(arm, col):
        return [
            np.mean([r[col] for r in rows if r[2] == arm and r[1] == f])
            for f in range(num_views)
        ]

    x = np.arange(num_views)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True, constrained_layout=True
    )
    colors = {"full": "#2563eb", "window": "#e8590c"}  # CVD-safe blue/orange
    for arm in ("full", "window"):
        ax1.plot(x, series(arm, 3), color=colors[arm], linewidth=2, label=arm)
    ax1.set_ylabel("AbsRel (median-scaled)")
    ax1.legend(frameon=False)
    ax2.plot(x, series("window", 5), color=colors["window"], linewidth=2)
    ax2.set_ylabel("divergence vs full")
    # median scaling hides global scale drift from AbsRel; this panel shows it
    # directly -- the window arm's scale peeling away from the full arm's is
    # truncation drifting the model's global scale estimate
    for arm in ("full", "window"):
        ax3.plot(x, series(arm, 6), color=colors[arm], linewidth=2, label=arm)
    ax3.set_ylabel("median scale (GT/pred)")
    ax3.set_xlabel("frame index")
    for ax in (ax1, ax2, ax3):
        ax.axvline(window + 0.5, color="#9ca3af", linestyle="--", linewidth=1)
        ax.grid(True, color="#e5e7eb", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.annotate(
        f"window slides (W={window})",
        xy=(window + 0.5, ax1.get_ylim()[1]),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color="#6b7280",
    )
    ax1.set_title(f"KV-cache window validation -- {desc}", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def frame_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple:
    """Median-scaled (AbsRel, delta<1.25, scale) on GT-valid pixels.

    `scale` is the median GT/pred ratio the alignment applied. Median scaling
    deliberately hides global scale drift from AbsRel -- so scale itself is
    reported: if the window arm's scale trends away from the full arm's over
    frame index, truncation is drifting the model's global scale estimate
    (expected failure mode: the metric scale lives in the RGB/context pathway,
    not in the near-inert sparse conditioning, so nothing re-anchors it)."""
    p, g = pred[valid], gt[valid]
    s = np.median(g) / max(np.median(p), 1e-8)
    p = p * s
    absrel = float(np.mean(np.abs(p - g) / g))
    delta1 = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    return absrel, delta1, float(s)


def main():
    args = parse_args()
    assert args.num_views > args.window, "--num-views must exceed --window"
    assert torch.cuda.is_available(), "needs a CUDA device"
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.weights:
        model = load_finetuned_model(args.weights, args.checkpoint, device)
    else:
        print(f"Loading base model: {args.ckpt}")
        model = load_base_model(args.ckpt, device)
    conditioned = isinstance(model, MetricStreamVGGT) and model.conditioner is not None

    # TEST split enforces stride_range=(1,1): ordered consecutive frames,
    # which is what the streaming deployment sees.
    ds = eval(
        f"ScanNet_Multi(split='test', ROOT='{args.root}', "
        f"resolution=[{RESOLUTION}], num_views={args.num_views}, n_corres=0, "
        f"aug_crop=16, seed=777, stride_range=(1, 1))"
    )
    rng = np.random.default_rng(args.seed)
    clip_ids = sorted(rng.choice(len(ds), size=args.clips, replace=False))

    rows = []  # (clip, frame, arm, absrel, delta1, divergence)
    for clip_no, idx in enumerate(clip_ids):
        views = ds[(int(idx), 0, args.num_views)]
        scene = views[0]["label"].rsplit("_", 1)[0]
        print(
            f"\nclip {clip_no}: dataset idx {idx}, scene {scene}, {len(views)} frames"
        )

        if conditioned:
            # batch the GT depth ([H,W] numpy -> [1,H,W] tensor) so
            # simulate_sparse_depth and _gather_sparse_depth see the training
            # contract. Seeded, and run ONCE per clip: both arms must receive
            # identical sparse measurements or the comparison measures the
            # sim's randomness instead of the cache policy.
            torch.manual_seed(args.seed + clip_no)
            for view in views:
                view["depthmap"] = torch.from_numpy(view["depthmap"])[None]
                view["valid_mask"] = torch.from_numpy(view["valid_mask"])[None]
            mcfg = model.cfg
            simulate_sparse_depth(
                views,
                mode=mcfg.depth_cond.sim_mode,
                patch_size=mcfg.depth_cond.sim_patch_size,
                mask_ratio=mcfg.depth_cond.sim_mask_ratio,
            )
            for view in views:  # metrics code below expects numpy again
                view["depthmap"] = view["depthmap"][0].numpy()
                view["valid_mask"] = view["valid_mask"][0].numpy()

        preds = {}
        for arm, window in (("full", None), ("window", args.window)):
            preds[arm] = run_stream(model, views, device, window, not args.no_amp)
            print(f"  {arm}: done")

        if args.dump_heatmaps:
            dump_heatmaps(os.path.join(args.out_dir, "heatmaps"), clip_no, preds)

        for f, view in enumerate(views):
            gt = view["depthmap"]
            valid = view["valid_mask"] & (gt > 0)
            df, dw = preds["full"][f], preds["window"][f]
            div = float(
                np.mean(np.abs(dw[valid] - df[valid]) / np.maximum(df[valid], 1e-8))
            )
            for arm, d in (("full", df), ("window", dw)):
                absrel, delta1, scale = frame_metrics(d, gt, valid)
                rows.append((clip_no, f, arm, absrel, delta1, div, scale))

    csv_path = os.path.join(args.out_dir, "per_frame_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "clip",
                "frame",
                "arm",
                "absrel",
                "delta1",
                "divergence_vs_full",
                "median_scale",
            ]
        )
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ---- summary ----------------------------------------------------------
    W = args.window

    def agg(arm, lo, hi, col):
        vals = [r[col] for r in rows if r[2] == arm and lo <= r[1] < hi]
        return float(np.mean(vals)) if vals else float("nan")

    n = args.num_views
    if args.weights:
        arm = model.cfg.depth_cond.injection.value if conditioned else "disabled"
        desc = f"finetuned {os.path.basename(args.weights.rstrip('/'))} ({arm})"
    else:
        desc = "base"
    plot_curves(rows, W, n, os.path.join(args.out_dir, "metrics_curves.png"), desc)
    print(
        f"\n=== {desc} | window W={W}, {args.clips} clips x {n} frames, "
        f"{'fp32' if args.no_amp else 'bf16'} aggregator ==="
    )
    print(f"{'phase':<22}{'arm':<9}{'AbsRel':>9}{'d1.25':>9}{'div':>10}")
    for lo, hi, name in (
        (0, W, f"pre-window  [0,{W})"),
        (W, n, f"post-window [{W},{n})"),
    ):
        for arm in ("full", "window"):
            print(
                f"{name:<22}{arm:<9}"
                f"{agg(arm, lo, hi, 3):>9.4f}"
                f"{agg(arm, lo, hi, 4):>9.4f}"
                f"{agg(arm, lo, hi, 5):>10.5f}"
            )

    pre_div = agg("window", 0, W, 5)
    if pre_div > 1e-6:
        print(
            f"\nWARNING: pre-window divergence {pre_div:.2e} != 0 -- the "
            "truncation loop is touching frames it shouldn't; do not trust "
            "the comparison until this is explained."
        )
    else:
        print(
            "\nSanity OK: window arm is bit-identical to full until the "
            "window first slides."
        )


if __name__ == "__main__":
    main()
