#!/usr/bin/env python
"""Assemble the PNG series that visualize_depth.py --heatmaps writes into GIFs.

Groups files by everything before the trailing frame number -- e.g.
base_clip0_depth_000.png .. _031.png -> base_clip0_depth.gif -- one GIF per
(tag, series). With --compare, additionally writes side-by-side GIFs (one
column per tag, in the order given, any number of arms >= 2) for every series
ALL tags share: compare_depth.gif, compare_tcons.gif, ...

Those compare names carry no clip index, so comparing a second clip into the
same directory would overwrite the first clip's GIFs -- pass --compare-name
when looping over the clips of a --num-clips N export.

CPU only, PIL only. Examples:
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps \\
        --compare base_clip0 promptda_clip0 finetuned_clip0 --fps 10
    python heatmaps_to_gif.py --hm-dir ../viz/perscene/heatmaps \\
        --compare base_clip3 finetuned_clip3 --compare-name compare_clip3
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
        "--compare",
        nargs="+",
        metavar="TAG",
        help="also write side-by-side GIFs (one column per tag, left to right "
        "in the order given) for series ALL tags share, e.g. "
        "--compare base_clip0 promptda_clip0 finetuned_clip0",
    )
    ap.add_argument(
        "--compare-name",
        default="compare",
        help="filename stem for the --compare GIFs (default 'compare', giving "
        "compare_conf.gif etc). One clip's compare GIFs overwrite another's "
        "without this, since the tags are not part of the name.",
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

    if args.compare:
        tags = args.compare
        if len(tags) < 2:
            raise SystemExit("--compare needs at least two tags")
        # series name = <tag>_<kind>; keep only kinds ALL tags share
        kinds = set.intersection(
            *(
                {p.removeprefix(t + "_") for p in series if p.startswith(t + "_")}
                for t in tags
            )
        )
        if not kinds:
            raise SystemExit(f"no shared series between {', '.join(map(repr, tags))}")
        for kind in sorted(kinds):
            # strict: arms of one comparison must have the same frame count;
            # silently truncating to the shortest would hide a partial export
            seqs = [series[f"{t}_{kind}"] for t in tags]
            frames = []
            for paths in zip(*seqs, strict=True):
                imgs = [Image.open(p).convert("RGB") for p in paths]
                # normalize followers to the first column's size (as before)
                imgs = [
                    im if im.size == imgs[0].size else im.resize(imgs[0].size)
                    for im in imgs
                ]
                w = sum(im.width for im in imgs) + 4 * (len(imgs) - 1)
                canvas = Image.new("RGB", (w, imgs[0].height), "white")
                x = 0
                for im in imgs:  # 4px white gutter between columns
                    canvas.paste(im, (x, 0))
                    x += im.width + 4
                frames.append(canvas)
            write_gif(frames, hm_dir / f"{args.compare_name}_{kind}.gif", args.fps)


if __name__ == "__main__":
    main()
