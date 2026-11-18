"""Side-by-side GIF of the depth-dropout experiment (scale_drift.py) on a few
ScanNet TAE clips: what each arm's streamed depth does once the sparse depth
stops. Two rows per frame:

  top     RGB | GT depth | each curve's metric prediction, all on the GT's
          depth range (so a drifting scale shows as the whole frame
          brightening or darkening)
  bottom  fed sparse pixels (white = given this frame) | GT-valid mask |
          each curve's log2(pred / GT) in [-1, 1] (blue = too near, red =
          too far): a scale drift is a uniform tint, a structure error a
          pattern. The caption carries the frame's median scale error.

Curves are scale_drift.py's `LABEL=SOURCE[@DROP][+emaALPHA]`, streamed the
same way (KV window --window, dropout from frame DROP, the benchmark's 5%
draw). Needs a GPU; one clip takes about a minute per curve.

    python drift_gif.py --window 8 --clip scene0791_00 --clip scene0766_00 \\
        --curve "Window ctrl (dropout @20)=<run>@20" --curve "Token rec aux (dropout @20)=<run>@20" \\
        --out /lustre/.../scale_drift/scaletok_sweep_drop20_w8/gifs
"""

from __future__ import annotations

import eval.provenance  # noqa: E402,F401,I001

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench_eval import BenchmarkCfg
from eval.spot_gif import _colorize
from eval.vda_benchmark import SPECS, load_manifest, model_to_gt
from heatmaps_to_gif import compose_grid, write_gif
from scale_drift import (
    DATASET,
    _clip,
    _sparse_arrays,
    curve_depth,
    load_curve_model,
    parse_curve,
    withhold_depth,
)


def _rgb_frames(views: list[dict], hw: tuple[int, int]) -> list[np.ndarray]:
    """The model-grid RGB of every view, [-1, 1] -> uint8, resized to the GT grid."""
    out = []
    for v in views:
        img = v["img"][0].detach().float().cpu().numpy()  # [3, h, w]
        img = ((img.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out.append(
            np.asarray(Image.fromarray(img).resize((hw[1], hw[0]), Image.BILINEAR))
        )
    return out


def _to_image(arr: np.ndarray, scale: float) -> Image.Image:
    im = Image.fromarray(arr)
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    return im


def render_clip(seq, spec, bench, curves, models, args, device, out_dir: Path) -> None:
    gt, views, scored, fed, box = _clip(seq, spec, bench, device)
    S, H, W = gt.shape
    rgb = _rgb_frames(views, (H, W))
    gt_ok = np.isfinite(gt) & (gt > 0)
    lo, hi = (
        float(np.nanpercentile(gt[gt_ok], 2)),
        float(np.nanpercentile(gt[gt_ok], 98)),
    )

    preds = {}
    for curve in curves:
        curve_views = [dict(v) for v in views]
        model, amp = models[curve.label]
        if curve.takes_depth:
            withhold_depth(curve_views, curve.drop)
        pred = curve_depth(curve, model, curve_views, args.window, amp)
        preds[curve.label] = model_to_gt(pred, (H, W), box)
        # the fed mask after this curve's dropout, on the GT grid, for the bottom row
        _, fed_c = _sparse_arrays(curve_views)
        preds[curve.label + "/fed"] = model_to_gt(fed_c, (H, W), box, mask=True)
        print(f"[gif] {seq.name}: {curve.label} done", flush=True)

    frames = []
    for t in range(S):
        top, top_labels, bottom, bottom_labels = [], [], [], []
        top.append(_to_image(rgb[t], args.scale))
        top_labels.append(f"{seq.name} frame {t}")
        top.append(_to_image(_colorize(gt[t], gt_ok[t], lo, hi, "turbo"), args.scale))
        top_labels.append(f"GT depth [{lo:.1f}, {hi:.1f}] m")
        bottom.append(
            _to_image(
                _colorize(
                    gt_ok[t].astype(np.float32), np.ones_like(gt_ok[t]), 0, 1, "gray"
                ),
                args.scale,
            )
        )
        bottom_labels.append("GT valid")
        for curve in curves:
            p = preds[curve.label][t]
            ok = gt_ok[t] & np.isfinite(p) & (p > 0)
            top.append(_to_image(_colorize(p, ok, lo, hi, "turbo"), args.scale))
            ratio = np.log2(np.where(ok, p / np.where(gt_ok[t], gt[t], 1.0), 1.0))
            med = (
                float(np.median(np.log(p[ok & scored[t]] / gt[t][ok & scored[t]])))
                if (ok & scored[t]).any()
                else float("nan")
            )
            fed_t = preds[curve.label + "/fed"][t]
            state = f"fed {int(fed_t.sum())} px" if fed_t.any() else "NO DEPTH"
            top_labels.append(f"{curve.label}: {state}")
            bottom.append(
                _to_image(_colorize(ratio, ok, -1.0, 1.0, "coolwarm"), args.scale)
            )
            bottom_labels.append(f"log2(pred/GT); median log s {med:+.3f}")
        # the bottom row's first panel is the GT-valid mask; pad to the top row's width
        bottom.insert(
            0,
            _to_image(
                _colorize(
                    preds[curves[0].label + "/fed"][t].astype(np.float32),
                    np.ones((H, W), bool),
                    0,
                    1,
                    "gray",
                ),
                args.scale,
            ),
        )
        bottom_labels.insert(0, f"fed pixels ({curves[0].label.split(' (')[0]})")
        frames.append(compose_grid(top + bottom, top_labels + bottom_labels, rows=2))
    out = out_dir / f"{seq.name}_drop{args.dropout_at}_w{args.window}.gif"
    write_gif(frames, out, args.fps)
    print(f"[gif] wrote {out} ({S} frames)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--curve", action="append", required=True, help="LABEL=SOURCE[@DROP][+emaALPHA]"
    )
    ap.add_argument(
        "--clip",
        action="append",
        required=True,
        help="TAE clip name, e.g. scene0791_00",
    )
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument(
        "--dropout-at",
        type=int,
        default=20,
        help="label only; DROP lives in each --curve",
    )
    ap.add_argument(
        "--base-ckpt", default="/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth"
    )
    ap.add_argument(
        "--scale", type=float, default=0.4, help="panel scale on the GT grid"
    )
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    curves = [parse_curve(c) for c in args.curve]
    bench = BenchmarkCfg()
    spec = SPECS[DATASET]
    seqs = {
        s.name: s for s in load_manifest(bench.root, spec, tae=True, tae_slice=False)
    }
    missing = [c for c in args.clip if c not in seqs]
    if missing:
        raise ValueError(f"clips not in the TAE manifest: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("drift_gif needs a CUDA device")
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for c in curves:
        if c.source not in loaded:
            loaded[c.source] = load_curve_model(c, args.base_ckpt, device)
    models = {c.label: loaded[c.source] for c in curves}
    for name in args.clip:
        render_clip(seqs[name], spec, bench, curves, models, args, device, args.out)


if __name__ == "__main__":
    main()
