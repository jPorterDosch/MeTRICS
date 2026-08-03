#!/usr/bin/env python
"""Assemble the PNG series that visualize_depth.py --heatmaps writes into GIFs.

Groups files by everything before the trailing frame number -- e.g.
base_clip0_depth_000.png .. _031.png -> base_clip0_depth.gif -- one GIF per
(tag, series). With --compare, additionally writes side-by-side GIFs (one
column per tag, in the order given, any number of arms >= 2) for every series
ANY tag has -- compare_depth.gif, compare_tcons.gif, ... An arm missing a
series (PromptDA returns no confidence) gets a captioned placeholder column
rather than sinking the whole GIF.

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

from PIL import Image, ImageDraw

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


def placeholder_panel(size: tuple[int, int], caption: str) -> Image.Image:
    """A neutral panel standing in for a series an arm does not produce (e.g.
    PromptDA has no confidence head), so the side-by-side GIF keeps one column
    per arm instead of silently dropping the whole series."""
    im = Image.new("RGB", size, (64, 64, 64))
    draw = ImageDraw.Draw(im)
    left, top, right, bottom = draw.textbbox((0, 0), caption)
    draw.text(
        ((size[0] - (right - left)) // 2, (size[1] - (bottom - top)) // 2),
        caption,
        fill=(210, 210, 210),
    )
    return im


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
        # series name = <tag>_<kind>; a kind is comparable when ANY tag has it.
        # Tags lacking it (PromptDA writes no conf series) get a captioned
        # placeholder column, so e.g. compare_conf.gif survives a three-arm run
        # instead of silently disappearing.
        kinds = set.union(
            *(
                {p.removeprefix(t + "_") for p in series if p.startswith(t + "_")}
                for t in tags
            )
        )
        if not kinds:
            raise SystemExit(f"no series for any of {', '.join(map(repr, tags))}")
        for kind in sorted(kinds):
            # strict: arms that have this series must agree on frame count;
            # silently truncating to the shortest would hide a partial export
            have = [t for t in tags if f"{t}_{kind}" in series]
            seqs = [series[f"{t}_{kind}"] for t in have]
            frames = []
            missing_panels: dict[str, Image.Image] = {}
            for paths in zip(*seqs, strict=True):
                by_tag = {t: Image.open(p).convert("RGB") for t, p in zip(have, paths)}
                size = by_tag[have[0]].size
                cols = []
                for t in tags:
                    if t in by_tag:
                        im = by_tag[t]
                        # normalize followers to the first column's size
                        cols.append(im if im.size == size else im.resize(size))
                    else:
                        if missing_panels.get(t) is None or (
                            missing_panels[t].size != size
                        ):
                            missing_panels[t] = placeholder_panel(
                                size, f"{t}: no {kind} output"
                            )
                        cols.append(missing_panels[t])
                w = sum(im.width for im in cols) + 4 * (len(cols) - 1)
                canvas = Image.new("RGB", (w, cols[0].height), "white")
                x = 0
                for im in cols:  # 4px white gutter between columns
                    canvas.paste(im, (x, 0))
                    x += im.width + 4
                frames.append(canvas)
            write_gif(frames, hm_dir / f"{args.compare_name}_{kind}.gif", args.fps)


if __name__ == "__main__":
    main()
