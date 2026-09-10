"""Glob an image series into one GIF.

heatmaps_to_gif.py only understands this repo's `<tag>_<kind>_NNN.png` naming.
Upstream PromptDA's scripts write `%06d.png` (uint16 metric depth),
`%06d_depth.jpg` (their colormapped view) and `%06d_smooth.jpg` (their
rgb|pred|prompt video panel), so their output needs a naming-agnostic writer.

usage:
    python tests/frames_to_gif.py --glob 'out/*_smooth.jpg' --out out/smooth.gif
"""

import argparse
import sys
from glob import glob
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from heatmaps_to_gif import write_gif  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", required=True, help="shell glob, quoted")
    ap.add_argument("--out", required=True, help="output .gif path")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="downscale frames wider than this (0 = no limit). Upstream's "
        "_smooth panels are three frames wide, so a full-resolution GIF of a "
        "long clip gets large fast.",
    )
    args = ap.parse_args()

    paths = sorted(glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match {args.glob!r}")

    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if args.max_width and im.width > args.max_width:
            h = round(im.height * args.max_width / im.width)
            im = im.resize((args.max_width, h), Image.LANCZOS)
        frames.append(im)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(frames, out, args.fps)


if __name__ == "__main__":
    main()
