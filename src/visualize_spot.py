#!/usr/bin/env python
# --------------------------------------------------------
# Run a (base or finetuned) depth-conditioned StreamVGGT on raw SPOT captures
# and export the same artifacts as visualize_depth.py: per-frame point-cloud
# GLBs (serve_glb.py-ready) and optional 2D heatmap series (--heatmaps).
#
# SPOT data (/oscar/data/jtompki1/cli277/new_spot_data/<seq>) has no GT depth,
# no GT poses and no calibrated intrinsics for the raw 640x480 color stream --
# but it has the REAL sparse metric depth (the other stereo camera's point
# cloud projected into this view), which becomes the conditioning input
# directly: simulate_sparse_depth skips views that already carry
# 'sparse_depth', so for the first time the model is conditioned on genuine
# sensor sparsity instead of simulated patch masking.
#
# Framing modes (SPOT's cameras are mounted sideways, so raw frames are
# landscape but gravity points sideways):
#   default              518x392 landscape, gravity WRONG
#   --rotate cw          392x518 portrait, gravity right, aspect out of
#                        distribution (the model trained on ~4:3 landscape)
#   --rotate cw --landscape-crop
#                        518x392 landscape, gravity right AND aspect in
#                        distribution: crop a 4:3 window out of the rotated
#                        frame, discarding top/bottom rather than sides.
# The last two have identical patch counts (1036), so an A/B between them
# isolates framing rather than model capacity.
#
# Cameras: the model predicts its own (pose_enc -> extrinsics + intrinsics).
# To keep a base-vs-finetuned A/B attributable to DEPTH, the pose track is a
# shared reference: the --base run caches its predicted K/pose to
# <out-dir>/pose_cache.npz, and every later run unprojects with the CACHED
# track (its own prediction is only compared against the cache and reported
# as a divergence, which doubles as the "did finetuning move the cameras?"
# diagnostic). Run --base first.
#
# Example (GPU):
#   cd src
#   python visualize_spot.py --weights ../checkpoints/hammer_sweep/b536d87d26e297e1 \
#       --checkpoint best --base --num-views 32 --heatmaps --out-dir ../viz/spot_seq0
#   python visualize_spot.py --weights ../checkpoints/hammer_sweep/b536d87d26e297e1 \
#       --checkpoint best --num-views 32 --heatmaps --out-dir ../viz/spot_seq0
# --------------------------------------------------------
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from finetune_depth import (
    FinetuneDepthCfg,
    _clip_predictions,
    _prepare_batch,
    build_model,
)
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from visualize_depth import (
    _CONF_VMAX,
    _CONF_VMIN,
    _REL_VMAX,
    _export_heatmaps,
    _format_frame_timing,
    _per_frame_scene,
    _run_streaming_inference,
    _stack_depth_conf,
    load_saved_args,
    rebuild_metric_cfg,
    resolve_checkpoint,
)

# model input resolutions (multiples of the 14px patch, same as the HAMMER
# training list, which contains both landscape 518x392 and portrait 392x518)
RAW_W, RAW_H = 640, 480
_ROTATIONS = {"none": None, "cw": Image.ROTATE_270, "ccw": Image.ROTATE_90}


def target_dims(rotate: str, landscape_crop: bool = False) -> tuple[int, int]:
    """(width, height) of the model input. SPOT's cameras are mounted sideways,
    so rotating to upright turns the raw landscape frames portrait -- unless
    --landscape-crop re-crops a 4:3 window out of that tall frame, which puts
    the model back at 518x392, the primary training resolution."""
    return (392, 518) if rotate != "none" and not landscape_crop else (518, 392)


def read_spot_depth(path: Path) -> np.ndarray:
    """SPOT depth binary: int32 pixel-count header, then HxW float32 metres
    (see new_spot_data/render_depth.py::read_and_process_depth)."""
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if n != RAW_W * RAW_H:
            raise ValueError(f"{path}: header {n}, expected {RAW_W * RAW_H}")
        d = np.fromfile(f, dtype=np.float32, count=n)
    return d.reshape(RAW_H, RAW_W)


def _resize_crop(img: Image.Image, tw: int, th: int, resample) -> Image.Image:
    """Scale to cover (tw, th), then center-crop the overshoot (~5px here).
    Depth/mask must use NEAREST (no blending across holes); color BILINEAR."""
    s = max(tw / img.width, th / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    # Image.crop pads with zeros instead of raising when the box overruns the
    # source, so an undershoot here would be a silent black edge on every frame
    if nw < tw or nh < th:
        raise ValueError(f"resize {nw}x{nh} does not cover {tw}x{th}")
    img = img.resize((nw, nh), resample)
    left, top = (img.width - tw) // 2, (img.height - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _landscape_crop(img: Image.Image, anchor: float) -> Image.Image:
    """Crop the widest RAW_W:RAW_H (4:3) window out of an upright frame.

    SPOT frames are 4:3, but the cameras are mounted sideways, so --rotate
    turns them PORTRAIT -- an aspect the model essentially never trained on.
    Discarding the top/bottom of the rotated frame (rather than the sides)
    restores the training aspect while keeping gravity correct.

    The crop aspect is RAW_W/RAW_H and deliberately NOT tw/th: 4:3 is what a
    HAMMER/ScanNet frame arrives as, so this puts SPOT through the identical
    downstream chain instead of a subtly different one.

    `anchor` places the window vertically as a fraction of the DISCARDED band
    (0=top, 0.5=center, 1=bottom). No-op on a frame that is already >= 4:3, so
    this degrades safely under --rotate none.
    """
    a = RAW_W / RAW_H
    if img.width / img.height >= a:
        cw, ch = round(img.height * a), img.height
    else:
        cw, ch = img.width, round(img.width / a)
    left = (img.width - cw) // 2
    top = min(max(round(anchor * (img.height - ch)), 0), img.height - ch)
    return img.crop((left, top, left + cw, top + ch))


def _prep(img: Image.Image, rot, anchor: float | None, tw: int, th: int, resample):
    """rotate -> optional 4:3 landscape crop -> scale-to-cover + center-crop.

    Colour and depth MUST both go through this one function: the shared
    rotation and the shared integer crop boxes are the whole reason they stay
    pixel-aligned. `resample` is the ONLY thing allowed to differ between them
    (BILINEAR for colour, NEAREST for depth so holes never blend).
    """
    if rot is not None:
        img = img.transpose(rot)
    if anchor is not None:
        img = _landscape_crop(img, anchor)
    return _resize_crop(img, tw, th, resample)


_ANCHORS = {"top": 0.0, "center": 0.5, "bottom": 1.0}


def parse_anchor(s: str) -> float:
    """--crop-anchor: a named position or a raw float, both in [0,1]."""
    v = _ANCHORS.get(s)
    if v is None:
        try:
            v = float(s)
        except ValueError:
            raise SystemExit(
                f"--crop-anchor {s!r} is neither {'/'.join(_ANCHORS)} nor a float"
            )
    if not 0.0 <= v <= 1.0:
        raise SystemExit(f"--crop-anchor {s!r} outside [0,1]")
    return v


def geometry_key(args) -> str:
    """Everything that changes WHAT the model saw, and therefore which image
    frame the cached K/poses live in. Two runs whose keys differ cannot be
    paired -- one fingerprint beats a growing chain of per-key checks, where
    any key missing from an older cache silently degrades to 'matches'."""
    tw, th = target_dims(args.rotate, args.landscape_crop)
    a = parse_anchor(args.crop_anchor) if args.landscape_crop else None
    return (
        f"start={args.start}|stride={args.stride}|num_views={args.num_views}"
        f"|rotate={args.rotate}|anchor={a}|wh={tw}x{th}"
    )


def load_spot_views(
    seq_dir: Path,
    start: int,
    num_views: int,
    stride: int,
    rotate: str,
    anchor: float | None = None,
) -> list[dict]:
    """Build the [S]-list of single-view dicts the streaming path consumes.
    img is ImgNorm-style [-1,1] (matching dataset output; _prepare_batch
    rescales to [0,1]); real sensor sparse depth rides along, so
    simulate_sparse_depth will skip these views.

    `anchor` is None for the portrait path and a float in [0,1] to enable the
    4:3 landscape crop (see _landscape_crop)."""
    tw, th = target_dims(rotate, anchor is not None)
    rot = _ROTATIONS[rotate]
    views = []
    for i in range(num_views):
        idx = start + i * stride
        rgb = _prep(
            Image.open(seq_dir / "color" / f"{idx}.png").convert("RGB"),
            rot,
            anchor,
            tw,
            th,
            Image.BILINEAR,
        )
        img = torch.from_numpy(np.asarray(rgb).copy()).float().permute(2, 0, 1) / 255.0
        img = img * 2.0 - 1.0  # ImgNorm mean=std=0.5

        depth_im = Image.fromarray(
            read_spot_depth(seq_dir / "depth" / str(idx)), mode="F"
        )
        depth = np.asarray(_prep(depth_im, rot, anchor, tw, th, Image.NEAREST))
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        mask = depth > 0

        views.append(
            {
                "img": img[None],  # [1,3,H,W]
                "sparse_depth": torch.from_numpy(depth.copy())[None],
                "sparse_depth_mask": torch.from_numpy(mask.copy())[None],
                "idx": idx,
                "instance": str(idx),
                "true_shape": torch.tensor([[th, tw]]),
            }
        )
    return views


def predicted_cameras(
    preds: list[dict], hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """pose_enc rows -> (w2c [S,3,4], K [S,3,3]) via the model's own decoder."""
    enc = torch.stack([p["camera_pose"].detach().float().cpu() for p in preds], dim=1)
    extri, intri = pose_encoding_to_extri_intri(enc, hw)
    return extri[0], intri[0]


def to_c2w(w2c: torch.Tensor) -> torch.Tensor:
    """[S,3,4] world->cam  ->  [S,4,4] cam->world."""
    S = w2c.shape[0]
    m = torch.eye(4).repeat(S, 1, 1)
    m[:, :3, :4] = w2c
    return torch.linalg.inv(m)


def pose_divergence(w2c_a: torch.Tensor, w2c_b: torch.Tensor) -> str:
    Ra, Rb = w2c_a[:, :3, :3], w2c_b[:, :3, :3]
    cosang = ((Ra @ Rb.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(1) - 1) / 2
    deg = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
    dt = (w2c_a[:, :3, 3] - w2c_b[:, :3, 3]).norm(dim=1)
    return (
        f"rot mean/max = {deg.mean():.3f}/{deg.max():.3f} deg | "
        f"trans mean/max = {dt.mean():.4f}/{dt.max():.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        required=True,
        help="run dir or .pth (architecture + finetuned weights)",
    )
    ap.add_argument(
        "--checkpoint", choices=["auto", "final", "best", "last"], default="auto"
    )
    ap.add_argument(
        "--base",
        action="store_true",
        help="pretrained weights, conditioning at zero-init; also WRITES the pose cache",
    )
    ap.add_argument(
        "--pretrained",
        default=None,
        help="base checkpoint override (as in visualize_depth)",
    )
    ap.add_argument("--seq-dir", default="/oscar/data/jtompki1/cli277/new_spot_data/0")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--num-views", type=int, default=32)
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="frame stride (SPOT walks fast; try 2-3 if motion is large)",
    )
    ap.add_argument(
        "--rotate",
        choices=["none", "cw", "ccw"],
        default="none",
        help="rotate the raw frames to upright BEFORE the model (SPOT cameras "
        "are mounted sideways). Applied to color and depth alike; the model "
        "then runs at the portrait 392x518 resolution from the training list "
        "(unless --landscape-crop), and everything downstream (GLB, heatmaps, "
        "poses) is upright.",
    )
    ap.add_argument(
        "--landscape-crop",
        action="store_true",
        help="after --rotate, crop a 4:3 LANDSCAPE window out of the tall "
        "portrait frame and run at 518x392 -- the primary TRAINING resolution "
        "-- instead of the portrait 392x518. Gravity stays correct AND the "
        "aspect matches the training distribution. Patch count is identical "
        "either way (1036), so a with/without A/B isolates FRAMING, not model "
        "capacity. Requires --rotate cw|ccw.",
    )
    ap.add_argument(
        "--crop-anchor",
        default="0.75",
        help="vertical placement of the --landscape-crop window as a fraction "
        "of the DISCARDED band (280px): 'top'/'center'/'bottom' or a float in "
        "[0,1]. Default 0.75 -> top row 210, keeping rows 210-569: slightly "
        "below center, biased toward the ground SPOT walks on.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heatmaps", action="store_true")
    ap.add_argument(
        "--timing",
        action="store_true",
        help="measure per-frame inference time (CUDA events) and add a frame_ms "
        "column to the summary CSV. SPOT is the honest place to read this: real "
        "sensor sparsity means the conditioning encoder sees real input rather "
        "than a simulated mask.",
    )
    ap.add_argument(
        "--hm-scale",
        type=int,
        default=1,
        help="integer upscale for heatmap PNGs (default 1 = native). The panels "
        "are written at the model's working resolution (518x392), i.e. ~1.7in "
        "at 300dpi -- too small to print. 4 gives 2072x1568 (~7in at 300dpi). "
        "NEAREST block replication: it enlarges without inventing structure and "
        "adds NO information, it only stops a poster pipeline from resampling "
        "for you.",
    )
    ap.add_argument(
        "--rel-vmax",
        type=float,
        default=_REL_VMAX,
        help="relative error that saturates the GTERR colormap (default "
        f"{_REL_VMAX}). On SPOT 'gterr' is deviation from the SENSOR's sparse "
        "depth rather than true GT, which can run well above the in-domain "
        "default; if the summary CSV's saturated_frac columns are near 1 the "
        "panels are flat and carry no comparable structure. Keep it identical "
        "across the base and finetuned runs you intend to pair.",
    )
    ap.add_argument(
        "--tcons-vmax",
        type=float,
        default=None,
        help="separate ceiling for the warp self-consistency maps (default: "
        "same as --rel-vmax). On SPOT the two series sit an order of magnitude "
        "apart -- gterr runs ~0.15 while tcons stays ~0.01 -- so one shared "
        "scale renders one of them useless. Try --rel-vmax 0.3 "
        "--tcons-vmax 0.05.",
    )
    ap.add_argument(
        "--conf-vmin",
        type=float,
        default=_CONF_VMIN,
        help=f"bottom of the predicted-confidence colormap (default {_CONF_VMIN}, "
        "the expp1 floor). Must match across the runs you pair.",
    )
    ap.add_argument(
        "--conf-vmax",
        type=float,
        default=_CONF_VMAX,
        help=f"confidence that saturates the conf colormap (default {_CONF_VMAX}). "
        "SPOT is the furthest out of domain of anything here, so its confidences "
        "sit lower than in-domain -- check conf_saturated_frac in the summary CSV "
        "and lower this if every panel is dark. Keep it identical across the pair.",
    )
    args = ap.parse_args()

    # --rotate defaults to "none", on which the 4:3 crop is a silent no-op --
    # so without this the operator gets the OLD path while believing they got
    # the new one, and the geometry fingerprint below would happily accept it
    if args.landscape_crop and args.rotate == "none":
        raise SystemExit(
            "--landscape-crop needs --rotate cw|ccw; the raw frame is already 4:3"
        )
    anchor = parse_anchor(args.crop_anchor) if args.landscape_crop else None
    print(f"geometry: {geometry_key(args)}")

    ckpt_path = resolve_checkpoint(args.weights, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = load_saved_args(ckpt)
    mcfg = rebuild_metric_cfg(raw)

    pretrained_path = ""
    if args.base:
        pretrained_path = args.pretrained or raw.get("pretrained") or ""
        if not pretrained_path or not os.path.exists(pretrained_path):
            raise SystemExit(
                f"--base needs a valid pretrained checkpoint (saved: {raw.get('pretrained')!r})"
            )

    cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        pretrained=pretrained_path,
        resume=None,
        output_dir=args.out_dir,
    )
    accelerator = Accelerator()
    device = accelerator.device
    if args.base:
        print(f"BASE model: loading pretrained weights {pretrained_path}")
        model, _ = build_model(cfg, mcfg, device, load_pretrained=True)
    else:
        model, _ = build_model(cfg, mcfg, device, load_pretrained=False)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_dir = Path(args.seq_dir)
    views = load_spot_views(
        seq_dir, args.start, args.num_views, args.stride, args.rotate, anchor
    )
    tw, th = target_dims(args.rotate, args.landscape_crop)
    sensor = torch.stack([v["sparse_depth"][0] for v in views])  # [S,H,W], pre-device
    sensor_mask = torch.stack([v["sparse_depth_mask"][0] for v in views])
    dens = sensor_mask.float().mean().item()
    dvals = sensor[sensor_mask]
    if dvals.numel():
        print(
            f"sensor sparse depth: {dens:.2%} of pixels | "
            f"range [{dvals.min():.2f}, {dvals.max():.2f}] m, median {dvals.median():.2f}"
        )
    else:
        print("sensor sparse depth: 0.00% of pixels | 0 valid sensor pixels")
    for v in views:
        for k, t in v.items():
            if torch.is_tensor(t):
                v[k] = t.to(device)

    _prepare_batch(views, mcfg)  # rescales img; skips sparse sim (real sparse present)
    frame_times_ms = [] if args.timing else None
    with torch.no_grad():
        result = _run_streaming_inference(model, views, frame_times_ms)
    preds = result["pred"]
    if frame_times_ms and not args.heatmaps:
        # _export_heatmaps is what normally reports these; without it the
        # measurement would be taken and silently dropped
        print(f"per-frame inference: {_format_frame_timing(frame_times_ms)}")
    pred_depth = torch.stack([p["depth"].detach() for p in preds], dim=1)
    pred_depth = pred_depth.squeeze(-1).float().cpu()[0]  # [S,H,W]
    print(f"pred depth: range [{pred_depth.min():.2f}, {pred_depth.max():.2f}] m")
    pred_conf = _stack_depth_conf(preds)[0]  # [S,H,W], the model's own uncertainty
    print(
        f"pred conf: range [{pred_conf.min():.2f}, {pred_conf.max():.2f}], "
        f"mean {pred_conf.mean():.2f}"
    )

    # ---- shared pose track (see header) ----
    own_w2c, own_K = predicted_cameras(preds, (th, tw))
    mode = "base" if args.base else "finetuned"
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "pose_cache.npz")
    if args.base:
        np.savez(
            cache_path,
            w2c=own_w2c.numpy(),
            K=own_K.numpy(),
            start=args.start,
            stride=args.stride,
            num_views=args.num_views,
            rotate=args.rotate,
            landscape_crop=args.landscape_crop,
            crop_anchor=args.crop_anchor,
            geometry=geometry_key(args),
        )
        print(f"pose cache written: {cache_path}")
        w2c, K = own_w2c, own_K
    else:
        if not os.path.exists(cache_path):
            raise SystemExit(
                f"no pose cache at {cache_path}; run --base into this --out-dir first"
            )
        cached = np.load(cache_path)
        want = geometry_key(args)
        if "geometry" in cached:
            if str(cached["geometry"]) != want:
                raise SystemExit(
                    f"pose cache geometry {str(cached['geometry'])!r} != {want!r}; "
                    "the cached K/poses live in a different image frame. Re-run "
                    "--base into this --out-dir, or use a fresh one."
                )
        elif args.landscape_crop:
            # a pre-fingerprint cache is portrait by construction; letting a
            # landscape run attach to it would compare mismatched geometry
            raise SystemExit(
                f"{cache_path} predates --landscape-crop and cannot be paired "
                "with it; re-run --base into a fresh --out-dir."
            )
        else:
            # legacy cache, legacy mode: the original per-key checks still hold
            for k in ("start", "stride", "num_views"):
                if int(cached[k]) != getattr(args, k):
                    raise SystemExit(
                        f"pose cache {k}={int(cached[k])} != {getattr(args, k)}; frames must match"
                    )
            if "rotate" in cached and str(cached["rotate"]) != args.rotate:
                raise SystemExit(
                    f"pose cache rotate={cached['rotate']} != {args.rotate}; frames must match"
                )
        w2c, K = torch.from_numpy(cached["w2c"]), torch.from_numpy(cached["K"])
        print(f"pose divergence vs cached base track: {pose_divergence(own_w2c, w2c)}")

    c2w = to_c2w(w2c)
    imgs = torch.stack([v["img"][0].float().cpu() for v in views])  # [S,3,H,W] in [0,1]

    # no GT on SPOT: valid = the model's own finite/positive predictions
    predictions = _clip_predictions(
        imgs,
        pred_depth,
        torch.ones_like(pred_depth, dtype=torch.bool),
        K,
        c2w,
        mask_to_gt=False,
    )
    glb_dir = os.path.join(args.out_dir, "glb")
    os.makedirs(glb_dir, exist_ok=True)
    scene = _per_frame_scene(predictions)
    out_glb = os.path.join(glb_dir, f"{mode}_clip0.glb")
    scene.export(out_glb)
    print(f"wrote {out_glb}")

    if args.heatmaps:
        # 'gterr' here = deviation from the SENSOR's sparse metric depth at its
        # ~valid pixels (dotted maps); tcons and conf are dense and GT-free as
        # always. NOTE the conf_err_corr column is computed against that sparse
        # sensor depth, so on SPOT it is a correlation over ~a few percent of
        # pixels, not the whole frame.
        n = _export_heatmaps(
            os.path.join(args.out_dir, "heatmaps"),
            f"{mode}_clip0",
            pred_depth,
            sensor,
            sensor_mask,
            K,
            c2w,
            rel_vmax=args.rel_vmax,
            tcons_vmax=args.tcons_vmax,
            scale=args.hm_scale,
            conf=pred_conf,
            conf_vmin=args.conf_vmin,
            conf_vmax=args.conf_vmax,
            frame_times_ms=frame_times_ms,
        )
        print(f"wrote {n} heatmap PNGs")


if __name__ == "__main__":
    main()
