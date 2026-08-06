#!/usr/bin/env python
"""Assemble the PNG series that visualize_depth.py --heatmaps writes into GIFs.

Groups files by everything before the trailing frame number -- e.g.
base_clip0_depth_000.png .. _031.png -> base_clip0_depth.gif -- one GIF per
(tag, series). With --compare, additionally writes side-by-side GIFs (one
column per tag, in the order given, any number of arms >= 2) for every series
ANY tag has -- compare_depth.gif, compare_tcons.gif, ... An arm missing a
series (PromptDA returns no confidence) gets a captioned placeholder column
rather than sinking the whole GIF.

With --gt-tag, every compare GIF opens with the arm-independent reference
columns -- input RGB then GT depth, e.g.
RGB -> GT depth -> PromptDA -> StreamVGGT -> Ours.

Those compare names carry no clip index, so comparing a second clip into the
same directory would overwrite the first clip's GIFs -- pass --compare-name
when looping over the clips of a --num-clips N export.

--prune deletes the per-frame PNGs whose GIF now exists (keeping the _depth
frames with 'non-depth'), which is the difference between a directory of a few
GIFs and one of several thousand PNGs.

CPU only, PIL only. Examples:
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps
    python heatmaps_to_gif.py --hm-dir ../viz/token_lora_seq/heatmaps \\
        --compare base_clip0 promptda_clip0 finetuned_clip0 --fps 10
    python heatmaps_to_gif.py --hm-dir ../viz/perscene/heatmaps \\
        --compare base_clip3 finetuned_clip3 --compare-name compare_clip3
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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


def _caption_font(px: int) -> ImageFont.ImageFont:
    # sized default font needs PIL >= 10; fall back to the fixed bitmap font
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def caption_bar(width: int, height: int, text: str) -> Image.Image:
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    font = _caption_font(int(height * 0.7))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((width - (right - left)) // 2, (height - (bottom - top)) // 2 - top),
        text,
        fill=(0, 0, 0),
        font=font,
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


def compose_grid(
    panels: list[Image.Image], labels: list[str], rows: int = 1
) -> Image.Image:
    """Lay captioned panels out in `rows` rows, filling left to right.

    rows=1 is the classic single strip. Above ~4 columns a strip gets too wide
    to read on a page, so e.g. 6 panels at rows=2 become a 2x3 grid. Panels are
    assumed pre-normalized to one size (the caller resizes followers to the
    first column); a short final row is left-aligned rather than stretched.
    """
    if rows < 1:
        raise ValueError(f"rows must be >= 1, got {rows}")
    per_row = math.ceil(len(panels) / rows)
    chunks = [panels[i : i + per_row] for i in range(0, len(panels), per_row)]
    lchunks = [labels[i : i + per_row] for i in range(0, len(labels), per_row)]

    bar_h = max(16, panels[0].height // 14)
    cell_h = panels[0].height + bar_h
    width = max(sum(p.width for p in ch) + 4 * (len(ch) - 1) for ch in chunks)
    height = cell_h * len(chunks) + 4 * (len(chunks) - 1)  # 4px gutters

    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for ch, lch in zip(chunks, lchunks, strict=True):
        x = 0
        for im, label in zip(ch, lch, strict=True):
            canvas.paste(caption_bar(im.width, bar_h, label), (x, y))
            canvas.paste(im, (x, y + bar_h))
            x += im.width + 4
        y += cell_h + 4
    return canvas


def prune_frames(series: dict[str, list[Path]], mode: str) -> int:
    """DELETE per-frame PNGs whose GIF now exists. Returns the count removed.

    Only ever runs over series whose animation is already on disk: a frame PNG
    is redundant only once the thing that replaced it exists. That check is
    also what makes the deferred (--prune-only) pass safe -- a series whose GIF
    step failed keeps its frames.
    """
    removed = 0
    for prefix, paths in series.items():
        hm_dir = paths[0].parent
        if not (hm_dir / f"{prefix}.gif").is_file():
            continue
        if mode == "non-depth" and prefix.endswith("_depth"):
            continue
        for p in paths:
            p.unlink()
            removed += 1
    kept = "kept _depth frames" if mode == "non-depth" else "kept no frames"
    print(f"pruned {removed} frame PNGs ({kept})")
    return removed


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
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="column subtitles for the --compare GIFs, parallel to the "
        "--compare tags (e.g. --labels PromptDA StreamVGGT Ours). Default: "
        "the raw tags.",
    )
    ap.add_argument(
        "--gt-tag",
        help="tag of the arm-independent reference series (e.g. gt_clip0). When "
        "set, its _rgb and _depth frames are prepended as 'RGB' and 'GT depth' "
        "columns to EVERY compare GIF -- the same reference columns regardless "
        "of the kind being compared. Both series must exist.",
    )
    ap.add_argument(
        "--prune",
        choices=("none", "non-depth", "all"),
        default="none",
        help="DELETE per-frame PNGs once their GIFs are written: 'non-depth' "
        "keeps the _depth_NNN.png frames (the ones worth opening one at a "
        "time) and drops gterr/tcons/conf/rgb; 'all' drops every frame PNG. "
        "Summary PNG/CSV and the GIFs are never touched. Default 'none'. "
        "One-way: the compare GIFs cannot be rebuilt from a pruned directory, "
        "so re-render the clip if you need different labels or fps.",
    )
    ap.add_argument(
        "--grid-rows",
        type=int,
        default=1,
        help="wrap the --compare panels into this many rows instead of one "
        "strip (e.g. 6 tags at --grid-rows 2 gives a 2x3 grid). Reference "
        "columns from --gt-tag count as panels and are placed first.",
    )
    ap.add_argument(
        "--prune-only",
        action="store_true",
        help="do the --prune deletion and NOTHING else -- write no GIFs. For "
        "deferring cleanup to the end of a multi-stage run, so every frame PNG "
        "stays available to later cross-stage compare GIFs. Needs --prune.",
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

    if args.prune_only:
        if args.prune == "none":
            raise SystemExit("--prune-only needs --prune non-depth|all")
        if args.compare:
            raise SystemExit("--prune-only writes no GIFs, so --compare is unused")
        prune_frames(series, args.prune)
        return

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
        labels = args.labels or tags
        if len(labels) != len(tags):
            raise SystemExit(
                f"--labels count ({len(labels)}) must match --compare ({len(tags)})"
            )
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
        # Arm-independent reference columns, leftmost, in this order: the input
        # frames then GT depth. BOTH are required -- visualize_depth.py writes
        # them together, so a missing one means a stale export (written before
        # the _rgb series existed) or a half-finished one, and dropping the
        # column silently would hide that.
        refs: list[tuple[str, str, list[Path]]] = []
        for kind, label in (("rgb", "RGB"), ("depth", "GT depth")):
            paths = series.get(f"{args.gt_tag}_{kind}") if args.gt_tag else None
            if args.gt_tag and paths is None:
                raise SystemExit(
                    f"--gt-tag {args.gt_tag}: no {args.gt_tag}_{kind} series in "
                    f"{hm_dir}. Re-render the clip."
                )
            if paths is not None:
                refs.append((f"\0{kind}", label, paths))
        col_labels = [lbl for _, lbl, _ in refs] + labels
        ref_tags = [t for t, _, _ in refs]
        # Arm-independent reference columns, leftmost, in this order: the input
        # frames then GT depth. BOTH are required -- visualize_depth.py writes
        # them together, so a missing one means a stale export (written before
        # the _rgb series existed) or a half-finished one, and dropping the
        # column silently would hide that.
        refs: list[tuple[str, str, list[Path]]] = []
        for kind, label in (("rgb", "RGB"), ("depth", "GT depth")):
            paths = series.get(f"{args.gt_tag}_{kind}") if args.gt_tag else None
            if args.gt_tag and paths is None:
                raise SystemExit(
                    f"--gt-tag {args.gt_tag}: no {args.gt_tag}_{kind} series in "
                    f"{hm_dir}. Re-render the clip -- exports predating the "
                    "reference RGB column have no _rgb frames."
                )
            if paths is not None:
                refs.append((f"\0{kind}", label, paths))
        col_labels = [lbl for _, lbl, _ in refs] + labels
        ref_tags = [t for t, _, _ in refs]
        for kind in sorted(kinds):
            # strict: arms that have this series must agree on frame count;
            # silently truncating to the shortest would hide a partial export
            # (the reference columns zip under the same rule)
            have = [t for t in tags if f"{t}_{kind}" in series]
            seqs = [series[f"{t}_{kind}"] for t in have]
            if refs:
                # tcons is a PER-PAIR series: frame i is the (i, i+1) warp
                # rendered on frame i+1's grid, so it is one shorter than the
                # per-frame reference columns and aligns with them from frame 1
                # on. Trim each reference from the FRONT by the difference; a
                # reference that is shorter still hits the strict zip below.
                n = len(seqs[0])
                have = ref_tags + have
                seqs = [p[len(p) - n :] if len(p) > n else p for _, _, p in refs] + seqs
            frames = []
            missing_panels: dict[str, Image.Image] = {}
            for paths in zip(*seqs, strict=True):
                by_tag = {t: Image.open(p).convert("RGB") for t, p in zip(have, paths)}
                size = by_tag[have[0]].size
                cols = []
                col_tags = ref_tags + tags
                for t in col_tags:
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
                frames.append(compose_grid(cols, col_labels, args.grid_rows))
            write_gif(frames, hm_dir / f"{args.compare_name}_{kind}.gif", args.fps)

    if args.prune != "none":
        prune_frames(series, args.prune)


if __name__ == "__main__":
    main()
