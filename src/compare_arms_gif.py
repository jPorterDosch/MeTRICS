"""Side-by-side GIF of one benchmark scene across several trained arms, next
to the reference depth. CPU only: it reads what the benchmark already wrote.

Two sources, picked by the scene name:
  spot_<seq>_<start>   the per-window SPOT GIFs (<run>/bench_gifs/), whose
                       prediction and error columns are cropped per arm and
                       set beside one copy of RGB + sensor depth. Every arm's
                       GIF shares the sensor's depth range and a fixed 0-0.3
                       error range, so the colours compare directly.
  anything else        a cloud snapshot (<run>/bench_clouds/<scene>.npz, the
                       first bench.cloud_frames frames): RGB | GT | each arm's
                       raw metric prediction on the GT's depth range, and a
                       second row of log2(pred / GT) in [-1, 1] (blue = too
                       near, red = too far).

    python compare_arms_gif.py --scene spot_seq_0_998 \
        --run RANDOM=<C>/e9263a9af33e8195/bench_viz --run TUBE_MASK=... \
        --out ../report/spot_seq_0_998_masking.gif

Every --run must hold the same scene, and for snapshots the same frames
(checked). SPOT has no dense GT: its reference column is the sensor depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from eval.spot_gif import _colorize
from heatmaps_to_gif import compose_grid, write_gif

SPOT_COLUMNS = 5  # RGB | sensor | prediction | rel. error | confidence
SPOT_SCALE = 0.75  # spot_gif.write_spot_gif's default panel scale


def _spot_panel_height(frame_h: int) -> int:
    """Panel height inside a compose_grid strip of height frame_h, whose
    caption bar is max(16, panel_h // 14) tall."""
    for ph in range(frame_h, 0, -1):
        if ph + max(16, ph // 14) == frame_h:
            return ph
    raise ValueError(f"cannot split a {frame_h}px strip into bar + panel")


def spot_frames(scene: str, runs: list[tuple[str, Path]]) -> list[Image.Image]:
    gifs = [Image.open(run / "bench_gifs" / f"{scene}.gif") for _, run in runs]
    n = min(g.n_frames for g in gifs)
    if len({g.n_frames for g in gifs}) != 1:
        print(f"[warn] frame counts differ {[g.n_frames for g in gifs]}; using {n}")
    W, H = gifs[0].size
    pw = (W - 4 * (SPOT_COLUMNS - 1)) // SPOT_COLUMNS
    ph = _spot_panel_height(H)
    bar = H - ph

    def col(im: Image.Image, k: int) -> Image.Image:
        x = k * (pw + 4)
        return im.crop((x, bar, x + pw, H))

    frames = []
    blank = Image.new("RGB", (pw, ph), "white")
    for i in range(n):
        ims = []
        for g in gifs:
            g.seek(i)
            ims.append(g.convert("RGB"))
        top = [col(ims[0], 0), col(ims[0], 1)] + [col(im, 2) for im in ims]
        bottom = [blank, blank] + [col(im, 3) for im in ims]
        labels_top = [f"{scene}  {i + 1}/{n}", "sensor depth"] + [
            f"{name} pred" for name, _ in runs
        ]
        labels_bottom = ["", ""] + [f"{name} rel. err 0-0.3" for name, _ in runs]
        frames.append(compose_grid(top + bottom, labels_top + labels_bottom, rows=2))
    return frames


def snapshot_frames(
    scene: str, runs: list[tuple[str, Path]], scale: float
) -> list[Image.Image]:
    z = [np.load(run / "bench_clouds" / f"{scene}.npz") for _, run in runs]
    for (name, _), a in zip(runs[1:], z[1:]):
        if (
            a["frames"].shape != z[0]["frames"].shape
            or not (a["frames"] == z[0]["frames"]).all()
        ):
            raise ValueError(f"{scene}: {name} snapshots different frames")
    gt, ok = z[0]["gt_depth"], z[0]["gt_valid"]
    if not ok.any():
        raise ValueError(f"{scene}: no valid GT in the snapshot")
    lo, hi = (float(v) for v in np.percentile(gt[ok], [2, 98]))
    S = gt.shape[0]
    frames = []
    for i in range(S):
        top = [z[0]["rgb"][i], _colorize(gt[i], ok[i], lo, hi, "turbo")]
        bottom = [np.full_like(z[0]["rgb"][i], 255)] * 2
        labels_top = [f"{scene}  {i + 1}/{S}", f"GT ({lo:.1f}-{hi:.1f} m)"]
        labels_bottom = ["", ""]
        for (name, _), a in zip(runs, z):
            p = a["pred_depth"][i]
            m = ok[i] & np.isfinite(p)
            g = np.where(ok[i], gt[i], 1.0)
            absrel = (
                float(np.mean(np.abs(p[m] - g[m]) / g[m])) if m.any() else float("nan")
            )
            lr = np.log2(np.clip(p, 1e-3, None) / g)
            top.append(_colorize(p, np.isfinite(p), lo, hi, "turbo"))
            bottom.append(_colorize(lr, m, -1.0, 1.0, "coolwarm"))
            labels_top.append(f"{name} pred (metric)")
            labels_bottom.append(f"{name} log2(pred/GT)  AbsRel {absrel:.3f}")
        panels = [Image.fromarray(x) for x in top + bottom]
        if scale != 1.0:
            size = (round(panels[0].width * scale), round(panels[0].height * scale))
            panels = [q.resize(size, Image.BILINEAR) for q in panels]
        frames.append(compose_grid(panels, labels_top + labels_bottom, rows=2))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--scene", required=True, help="spot_<seq>_<start> or a snapshot stem"
    )
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="benchmark output dir (holding bench_gifs/ and bench_clouds/); repeat per arm",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--fps", type=float, default=None, help="default 10 SPOT, 6 snapshots"
    )
    ap.add_argument("--scale", type=float, default=0.5, help="snapshot panel scale")
    args = ap.parse_args()

    runs = []
    for r in args.run:
        if "=" not in r:
            raise ValueError(f"--run must be LABEL=DIR, got {r!r}")
        label, d = r.split("=", 1)
        runs.append((label, Path(d)))
    spot = args.scene.startswith("spot_")
    frames = (
        spot_frames(args.scene, runs)
        if spot
        else snapshot_frames(args.scene, runs, args.scale)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(frames, out, args.fps or (10.0 if spot else 6.0))


if __name__ == "__main__":
    main()
