"""Dump the exact frames eval_all.sh visualizes into the on-disk layout that
upstream PromptDA's own scripts read.

This is the ONLY custom code in the upstream-PromptDA comparison. It touches no
model: it runs our loaders, takes the frames and the sparse prompt the promptda
arm would have been fed, and writes them out as files. Everything downstream --
image loading, resizing, normalization, the forward, the depth writing, the
video panels -- is upstream's own script, run unmodified from a clean checkout
(see experiments/promptda_upstream_gif.sh).

Layout written, matching what promptda/scripts/infer_stray_scan.py globs:

    <out-dir>/rgb/%06d.jpg      uint8 RGB, JPEG quality 100, no chroma
                                subsampling (their loader globs rgb/*.jpg)
    <out-dir>/depth/%06d.png    uint16 MILLIMETRES -- their load_depth divides
                                by 1000, so this round-trips to metres at 1 mm
                                quantization (our sources are uint16-mm PNGs to
                                begin with; SPOT's float32 tops out near 6 m)
    <out-dir>/manifest.json     what was dumped, and the prompt statistics

--prompt-mode picks WHAT goes in depth/, and the two modes answer different
questions:

  arkit (default for hammer/scannet) -- the prompt PromptDA was TRAINED on: the
    dense GT depth downsampled to 192x256, "exactly the depth resolution of
    iPhone ARKit Depth" (Prompt Depth Anything, arXiv 2412.14015 §3.3). Their
    prompt is a complete low-resolution depth image; nothing in that paper
    masks it to a percentage of pixels. Use this to see PromptDA in
    distribution.

  patch-infill (default for spot, since SPOT has no dense GT) -- what our
    promptda arm actually feeds: the run's simulated patch-masked prompt (or
    SPOT's real sensor), densified with upstream's own nearest-neighbor gather
    from run_inference.py (validity `(d > 0) & (d < 1000)`, then a
    distance_transform_edt gather). Reproduces the in-repo arm.

The gap between the two is large and is the point: at sim_mask_ratio 0.95 /
patch 14 on a 518x392 frame, patch-infill leaves ~52 surviving 14x14 blocks
that a Voronoi fill extrapolates to dense, redrawn INDEPENDENTLY per frame,
whereas arkit is ~49k filled measurements laid out identically every frame.

--no-infill dumps the raw sparse map (patch-infill mode only), which PromptDA
has no way to interpret -- it treats every pixel of the prompt as a
measurement.

Frame selection is held identical to eval_all.sh so the output is
frame-for-frame comparable to the promptda_clip0 series an eval_all run wrote:
same checkpoint-derived val config, clip 0, --num-views 32, --sparse-seed 0,
and for SPOT the same window/stride/rotation/crop.

The checkpoint is a val-config carrier only -- no weights are loaded from it,
and no model is built, so this runs on CPU (login node is fine).

usage:
    python tests/dump_promptda_frames.py --stage hammer \
        --weights checkpoints/hs_conf_off_sweep/5338c5bfb9be9414 \
        --data-root /oscar/scratch/jdosch/data/processed_hammer \
        --out-dir viz/promptda_upstream/frames/hammer
    python tests/dump_promptda_frames.py --stage hammer --prompt-mode patch-infill \
        --weights ... --data-root ... --out-dir .../hammer_patch

    python tests/dump_promptda_frames.py --stage spot --start 998 \
        --weights checkpoints/hs_conf_off_sweep/5338c5bfb9be9414 \
        --out-dir viz/promptda_upstream/frames/spot_dynamic_998
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from finetune_depth import (  # noqa: E402
    FinetuneDepthCfg,
    _prepare_batch,
    _set_data_epoch,
    build_train_loader,
)
from streamvggt.datasets import Split  # noqa: E402
from visualize_depth import (  # noqa: E402
    _ARKIT_HW,
    arkit_prompt,
    load_saved_args,
    rebuild_metric_cfg,
    rebuild_val_dataset,
    resolve_checkpoint,
)
from visualize_spot import load_spot_views, parse_anchor, target_dims  # noqa: E402

# Upstream's own far bound, from run_inference.py. Deliberately NOT our
# _PROMPT_DEPTH_MAX_M (100 m): this dump feeds upstream's script, so it uses
# upstream's validity test. Nothing in these datasets reaches either bound.
_UPSTREAM_DEPTH_MAX_M = 1000.0

# uint16 millimetres saturate at 65.535 m. Anything above is already garbage by
# the validity test above, but clip explicitly rather than let it wrap.
_UINT16_MAX_MM = 65535

# arkit_prompt / _ARKIT_HW live in visualize_depth so the upstream dump and the
# in-repo arms build the dense end of the sparsity axis from ONE implementation.


def upstream_infill(depth: np.ndarray) -> np.ndarray:
    """Nearest-valid-neighbor fill, verbatim in form from upstream's
    run_inference.py::run_inference. Returns a dense map."""
    out = depth.astype(np.float32).copy()
    valid = (out > 0) & (out < _UPSTREAM_DEPTH_MAX_M)
    if not valid.any():
        raise ValueError("prompt frame has no valid depth; nothing to infill from")
    if (~valid).any():
        _, idx = distance_transform_edt(~valid, return_indices=True)
        out[~valid] = out[idx[0][~valid], idx[1][~valid]]
    return out


def write_frame(out_dir: Path, i: int, rgb: np.ndarray, prompt: np.ndarray) -> None:
    """rgb [3,H,W] float in [0,1]; prompt [H,W] float metres."""
    img = np.clip(rgb.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    # quality 100 + subsampling 0: their loader globs *.jpg, and chroma
    # subsampling would inject frame-to-frame colour noise into a run whose
    # whole point is measuring frame-to-frame stability
    Image.fromarray(img).save(
        out_dir / "rgb" / f"{i:06d}.jpg", quality=100, subsampling=0
    )
    mm = np.clip(np.rint(prompt * 1000.0), 0, _UINT16_MAX_MM).astype(np.uint16)
    Image.fromarray(mm).save(out_dir / "depth" / f"{i:06d}.png")


def views_from_dataset(args) -> list[dict]:
    """Clip 0 of the TEST split the run validates on -- the same clip, frames
    and simulated sparse prompt visualize_depth.py's promptda arm receives."""
    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    print(f"Loading checkpoint (val config only): {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = load_saved_args(ckpt)
    mcfg = rebuild_metric_cfg(raw)
    val_ds = rebuild_val_dataset(raw, args.data_root, args.stage)
    val_ds.num_views = args.num_views
    del ckpt

    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        val_dataset=val_ds,
        pretrained="",
        resume=None,
        num_workers=args.num_workers,
        batch_size=1,
        fixed_length=bool(raw.get("fixed_length", True)),
        output_dir=str(args.out_dir),
    )
    accelerator = Accelerator()
    loader = build_train_loader(cfg, Split.TEST, accelerator, batch_size=1)
    loader = accelerator.prepare(loader)
    _set_data_epoch(loader, 0)  # deterministic clip set/order, as in val_loop

    for batch in loader:
        # Reseeded immediately before the draw and with the same seed
        # visualize_depth uses for clip 0, so the sparse pattern is the one the
        # in-repo promptda arm conditioned on.
        torch.manual_seed(args.sparse_seed)
        _prepare_batch(batch, mcfg)
        return batch
    raise SystemExit("the TEST loader yielded no clips")


def views_from_spot(args) -> list[dict]:
    """The SPOT window, with its REAL sensor prompt (no simulation)."""
    if args.landscape_crop and args.rotate == "none":
        raise SystemExit("--landscape-crop needs --rotate cw|ccw")
    anchor = parse_anchor(args.crop_anchor) if args.landscape_crop else None
    tw, th = target_dims(args.rotate, args.landscape_crop)
    print(f"spot geometry: start={args.start} stride={args.stride} wh={tw}x{th}")

    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = rebuild_metric_cfg(load_saved_args(ckpt))
    del ckpt

    views = load_spot_views(
        Path(args.seq_dir), args.start, args.num_views, args.stride, args.rotate, anchor
    )
    # rescales img to [0,1]; the sparse simulation skips these views because a
    # real sparse_depth is already attached
    _prepare_batch(views, mcfg)
    return views


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=("hammer", "scannet", "spot"))
    ap.add_argument("--weights", required=True, help="run dir or checkpoint .pth")
    ap.add_argument(
        "--checkpoint", default="best", choices=("auto", "final", "best", "last")
    )
    ap.add_argument("--data-root", default=None, help="processed tree (hammer/scannet)")
    ap.add_argument("--out-dir", required=True, help="scene dir to write rgb/ + depth/")
    ap.add_argument("--num-views", type=int, default=32)
    ap.add_argument("--sparse-seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--prompt-mode",
        choices=("arkit", "patch-infill"),
        default=None,
        help="what to write into depth/. 'arkit' (default for hammer/scannet): "
        "dense GT downsampled to 192x256, the prompt PromptDA was trained on. "
        "'patch-infill' (default and only option for spot, which has no dense "
        "GT): the run's simulated/real sparse prompt densified with upstream's "
        "nearest-neighbor gather, i.e. what our promptda arm feeds. See the "
        "module docstring.",
    )
    ap.add_argument(
        "--infill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="patch-infill mode only: densify with upstream's nearest-neighbor "
        "gather (default). --no-infill writes the raw sparse map, which "
        "PromptDA cannot interpret -- it reads every prompt pixel as a "
        "measurement.",
    )
    # SPOT geometry, defaults matching eval_all.sh's spot_pair
    ap.add_argument("--seq-dir", default="/oscar/data/jtompki1/cli277/new_spot_data/0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rotate", default="cw", choices=("none", "cw", "ccw"))
    ap.add_argument(
        "--landscape-crop", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--crop-anchor", default="top")
    args = ap.parse_args()

    if args.prompt_mode is None:
        args.prompt_mode = "patch-infill" if args.stage == "spot" else "arkit"
    if args.stage == "spot" and args.prompt_mode == "arkit":
        raise SystemExit(
            "--prompt-mode arkit needs dense GT depth to downsample; SPOT has "
            "only its real sparse sensor prompt. Use patch-infill."
        )

    out_dir = Path(args.out_dir)
    if "." in out_dir.name:
        # upstream's load_data derives the scene name as
        # basename(input_path).split('.')[0], so a dot truncates the directory
        # it looks in and the run reads the wrong (or no) frames
        raise SystemExit(f"--out-dir basename must not contain '.': {out_dir.name}")
    (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)

    views = views_from_spot(args) if args.stage == "spot" else views_from_dataset(args)

    densities, mins, maxes = [], [], []
    for i, view in enumerate(views):
        rgb = view["img"][0].detach().float().cpu().numpy()  # [3,H,W] in [0,1]
        if args.prompt_mode == "arkit":
            if "depthmap" not in view:
                raise SystemExit(
                    "--prompt-mode arkit needs the dataset's dense GT under "
                    "'depthmap'; this view has none"
                )
            gt = view["depthmap"][0].detach().float().cpu().numpy()
            if gt.ndim == 3:  # [H,W,1]
                gt = gt[..., 0]
            gt_valid = gt > 0
            if "valid_mask" in view:
                gt_valid &= view["valid_mask"][0].detach().cpu().numpy().astype(bool)
            prompt, coverage = arkit_prompt(gt, gt_valid)
            densities.append(coverage)
        else:
            sparse = view["sparse_depth"][0].detach().float().cpu().numpy()  # [H,W] m
            mask = view["sparse_depth_mask"][0].detach().cpu().numpy().astype(bool)
            densities.append(float(mask.mean()))
            sparse = sparse * mask
            prompt = upstream_infill(sparse) if args.infill else sparse
        # Extremes of the map as WRITTEN -- these are the values PromptDA's
        # per-frame normalize/denormalize keys off, so their frame-to-frame
        # movement is a direct global rescale of the prediction.
        mins.append(float(prompt[prompt > 0].min()) if (prompt > 0).any() else 0.0)
        maxes.append(float(prompt.max()))
        write_frame(out_dir, i, rgb, prompt)

    h, w = views[0]["img"].shape[-2:]
    manifest = {
        "stage": args.stage,
        "frames": len(views),
        "height": int(h),
        "width": int(w),
        "prompt_mode": args.prompt_mode,
        # arkit writes a 192x256 prompt against a full-resolution RGB, exactly
        # as upstream does (their prompt is lower-res than their image)
        "prompt_hw": list(_ARKIT_HW)
        if args.prompt_mode == "arkit"
        else [int(h), int(w)],
        "infilled": bool(args.infill) if args.prompt_mode == "patch-infill" else True,
        "sparse_seed": args.sparse_seed if args.stage != "spot" else None,
        "prompt_density_mean": float(np.mean(densities)) if densities else 0.0,
        "prompt_min_m": min(mins) if mins else None,
        "prompt_max_m": max(maxes) if maxes else None,
        # Per-frame spread of the prompt extremes. PromptDA normalizes each
        # frame by its OWN prompt min/max and denormalizes the output by the
        # same pair, so movement in these columns is a direct per-frame rescale
        # of the prediction -- the first thing to check against visible flicker.
        "prompt_min_per_frame_spread_m": (max(mins) - min(mins)) if mins else None,
        "prompt_max_per_frame_spread_m": (max(maxes) - min(maxes)) if maxes else None,
    }
    if args.stage == "spot":
        manifest["spot"] = {
            "seq_dir": args.seq_dir,
            "start": args.start,
            "stride": args.stride,
            "rotate": args.rotate,
            "landscape_crop": args.landscape_crop,
            "crop_anchor": args.crop_anchor,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ph, pw = manifest["prompt_hw"]
    print(f"wrote {len(views)} frames (rgb {w}x{h}, prompt {pw}x{ph}) to {out_dir}")
    print(
        f"  mode {args.prompt_mode}: coverage {manifest['prompt_density_mean']:.2%}, "
        f"range [{manifest['prompt_min_m']:.2f}, {manifest['prompt_max_m']:.2f}] m, "
        f"per-frame min/max spread "
        f"{manifest['prompt_min_per_frame_spread_m']:.2f}/"
        f"{manifest['prompt_max_per_frame_spread_m']:.2f} m"
    )


if __name__ == "__main__":
    main()
