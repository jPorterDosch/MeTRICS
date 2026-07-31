#!/usr/bin/env python
"""Assemble the PNG series that visualize_depth.py --heatmaps writes into GIFs.

Groups files by everything before the trailing frame number -- e.g.
base_clip0_depth_000.png .. _031.png -> base_clip0_depth.gif -- one GIF per
(tag, series). With --pair, additionally writes side-by-side GIFs (left|right)
for every series the two tags share: pair_depth.gif, pair_tcons.gif, ...

Those pair names carry no clip index, so pairing a second clip into the same
directory would overwrite the first clip's GIFs -- pass --pair-name when
looping over the clips of a --num-clips N export.

CPU only, PIL only. Examples:
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps \\
        --pair base_clip0 finetuned_clip0 --fps 10
    python heatmaps_to_gif.py --hm-dir ../viz/perscene/heatmaps \\
        --pair base_clip3 finetuned_clip3 --pair-name pair_clip3
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

_FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d{3})\.png$")


def collect_series(hm_dir: Path) -> dict[str, list[Path]]:
    series: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in hm_dir.glob("*.png"):
        m = _FRAME_RE.match(p.name)
        if m:
            series[m.group("prefix")].append((int(m.group("idx")), p))
    return {k: [p for _, p in sorted(v)] for k, v in series.items()}


def _positive_fps(s: str) -> float:
    """argparse type for --fps: must be > 0 (write_gif divides 1000/fps, and a
    huge value would truncate the frame duration to 0 ms). Rejects at parse
    time with a clean argparse error instead of a mid-run ZeroDivisionError."""
    fps = float(s)
    if not (0 < fps <= 1000):
        raise argparse.ArgumentTypeError(f"--fps must be in (0, 1000], got {fps}")
    return fps


def write_gif(frames: list[Image.Image], out: Path, fps: float) -> None:
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,  # loop forever
        disposal=2,
    )
    print(f"  {out.name}: {len(frames)} frames @ {fps:g} fps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hm-dir", required=True, help="the heatmaps/ directory")
    ap.add_argument("--fps", type=_positive_fps, default=10.0)
    ap.add_argument(
        "--pair",
        nargs=2,
        metavar=("LEFT_TAG", "RIGHT_TAG"),
        help="also write side-by-side GIFs for series both tags share, e.g. "
        "--pair base_clip0 finetuned_clip0",
    )
    ap.add_argument(
        "--pair-name",
        default="pair",
        help="filename stem for the --pair GIFs (default 'pair', giving "
        "pair_conf.gif etc). One clip's pair GIFs overwrite another's without "
        "this, since the tags are not part of the name.",
    )
    args = ap.parse_args()

    hm_dir = Path(args.hm_dir)
    series = collect_series(hm_dir)
    if not series:
        raise SystemExit(f"no *_NNN.png series found in {hm_dir}")

    print(f"{len(series)} series in {hm_dir}:")
    for prefix, paths in series.items():
        write_gif(
            [Image.open(p).convert("RGB") for p in paths],
            hm_dir / f"{prefix}.gif",
            args.fps,
        )

    if args.pair:
        left_tag, right_tag = args.pair
        # series name = <tag>_<kind>; match kinds across the two tags
        kinds = {
            p.removeprefix(left_tag + "_")
            for p in series
            if p.startswith(left_tag + "_")
        } & {
            p.removeprefix(right_tag + "_")
            for p in series
            if p.startswith(right_tag + "_")
        }
        if not kinds:
            raise SystemExit(f"no shared series between {left_tag!r} and {right_tag!r}")
        for kind in sorted(kinds):
            ls, rs = series[f"{left_tag}_{kind}"], series[f"{right_tag}_{kind}"]
            n = min(len(ls), len(rs))
            frames = []
            for lp, rp in zip(ls[:n], rs[:n]):
                li, ri = Image.open(lp).convert("RGB"), Image.open(rp).convert("RGB")
                if li.size != ri.size:
                    ri = ri.resize(li.size)
                canvas = Image.new("RGB", (li.width + ri.width + 4, li.height), "white")
                canvas.paste(li, (0, 0))
                canvas.paste(ri, (li.width + 4, 0))
                frames.append(canvas)
            write_gif(frames, hm_dir / f"{args.pair_name}_{kind}.gif", args.fps)


if __name__ == "__main__":
    main()
