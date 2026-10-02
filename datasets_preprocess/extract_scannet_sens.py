#!/usr/bin/env python
"""Batch-extract raw ScanNet ``.sens`` files -- the one stage the repo's
ScanNet pipeline was missing.

The stock ScanNet reader (``scannet_reader.py`` / ``SensorData``) handles a
single ``.sens`` at a time. This walks the scene tree and runs the same four
exports for every scene, writing *in place* into each scene dir:

    <raw_root>/<split>/<scene>/
        <scene>.sens          # input, left untouched
        frames.zip            # ONE uncompressed archive holding
            color/{i}.jpg     #   export_color_images
            depth/{i}.png     #   export_depth_images
            pose/{i}.txt      #   export_poses
            intrinsic/intrinsic_{color,depth}.txt + extrinsic_*

The archive is the default because the loose layout costs one inode per frame:
ScanNet's ~2.5M frames would be ~7.5M inodes here and as many again after
preprocessing, which overruns a normal per-user Lustre inode quota. Members are
STORED (no compression), so readers seek straight to the bytes and the pixel
data is identical either way -- ``--extracted`` restores the loose layout when
inodes are not a concern.

That layout is exactly what ``preprocess_scannet.py`` consumes next, so the full
pipeline is three explicit stages, each its own script (no duplicated logic):

    1. python extract_scannet_sens.py  --raw-root <raw>            # this script
    2. python preprocess_scannet.py    --scannet_dir <raw> --output_dir <proc>
    3. python generate_set_scannet.py  --root <proc> --splits scans_train scans_test ...

Resumable: a scene with a complete ``frames.zip`` is skipped (the writer renames
``.tmp`` -> final only on success, so a final-named archive is always complete);
in ``--extracted`` mode the old check applies -- ``intrinsic/intrinsic_depth.txt``
exists and ``color`` frame count matches ``depth``. Shardable for a SLURM array
via ``--shard i --num-shards N`` (scenes assigned round-robin).

Run with the StreamVGGT env python (needs imageio + pypng, which SensorData uses).
"""

import argparse
import os
import os.path as osp
import shutil
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from dust3r.utils.zipio import SceneZipWriter  # noqa: E402

from scannet_sensor import SensorData  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        default="/lustre/isaac24/proj/UTK0516/metrics_data/scannet",
        help="dir containing the split subdirs (e.g. scans_train, scans_test)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["scans_train", "scans_test"],
        help="split subdirs under --raw-root to process",
    )
    p.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    p.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be done, extract nothing",
    )
    p.add_argument(
        "--extracted",
        action="store_true",
        help="write loose per-frame files instead of one frames.zip per scene "
        "(the original layout; ~one inode per frame)",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="write <out-root>/<split>/<scene>/ instead of in place next to the "
        ".sens -- for a raw tree you cannot write into (the shared scans_test "
        "is owned by another user), or for a partial export that must not be "
        "mistaken for a complete one by preprocess_scannet.py",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="export only the first N frames of each scan (default: all). The "
        "video-depth benchmark reads a scan's first 510 frames; exporting the "
        "other ~90%% of a 100-scene split costs ~100 GB and ~600k inodes for "
        "nothing. Requires --out-root: a truncated in-place export would be "
        "picked up by preprocess_scannet.py as the whole scan",
    )
    p.add_argument(
        "--color-at-depth-res",
        action="store_true",
        help="export colour resized (INTER_AREA) to the depth image size "
        "(640x480) instead of the native 1296x968. The video-depth benchmark "
        "applies one pixel crop ([8:-8, 11:-11]) to both, so they must share "
        "a size for the crop to mean the same field of view; the model runs "
        "at 518 wide either way. Requires --out-root and --extracted",
    )
    return p


# Written next to a loose export to record how it was cut; an export with a
# different cap (or colour size) is not "complete" for a run asking otherwise.
EXPORT_MARKER = ".export_options"


def export_options(max_frames, color_at_depth_res):
    return f"max_frames={max_frames if max_frames is not None else 'all'} color_at_depth_res={color_at_depth_res}"


def already_extracted(scene_dir, as_zip=True, options=None):
    """True if this scene already has a complete-looking extraction (skip it).
    `options` is export_options(...) of the current run: a loose export
    whose marker records different options (a 510-frame cut for a run that
    wants every frame) does not count. A marker-less loose tree predates the
    marker and was always a full-frame native-colour export."""
    if as_zip:
        # SceneZipWriter renames .tmp -> final only on clean exit, so the
        # mere existence of the final name means the archive is complete
        return osp.isfile(osp.join(scene_dir, "frames.zip"))
    marker = osp.join(scene_dir, EXPORT_MARKER)
    recorded = (
        open(marker).read().strip()
        if osp.isfile(marker)
        else export_options(None, False)
    )
    if options is not None and recorded != options:
        return False
    intr = osp.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
    color, depth, pose = (osp.join(scene_dir, d) for d in ("color", "depth", "pose"))
    if not (osp.isfile(intr) and osp.isdir(color) and osp.isdir(depth)):
        return False
    nc = len(os.listdir(color))
    # poses are exported after color and depth, so a run killed between the
    # two would otherwise pass as complete with an empty pose/ dir
    npose = len(os.listdir(pose)) if osp.isdir(pose) else 0
    return nc > 0 and nc == len(os.listdir(depth)) == npose


def extract_scene(
    scene_dir, sens_path, as_zip=True, max_frames=None, color_at_depth_res=False
):
    """Export color/depth/pose/intrinsic from one .sens into scene_dir (the
    .sens's own directory unless --out-root redirected it)."""
    sd = SensorData(sens_path)
    if as_zip:
        if max_frames is not None or color_at_depth_res:
            raise ValueError("--max-frames / --color-at-depth-res need --extracted")
        with SceneZipWriter(osp.join(scene_dir, "frames.zip")) as writer:
            sd.export_all_to_zip(writer)
        return
    # a previous export into this dir with other options (a larger cap, native
    # colour) would otherwise leave its extra / differently sized frames next
    # to the new ones and pass the count check as one consistent scene
    for sub in ("color", "depth", "pose", "intrinsic"):
        shutil.rmtree(osp.join(scene_dir, sub), ignore_errors=True)
    marker = osp.join(scene_dir, EXPORT_MARKER)
    if osp.isfile(marker):
        os.remove(marker)
    color_size = (sd.depth_height, sd.depth_width) if color_at_depth_res else None
    sd.export_color_images(
        osp.join(scene_dir, "color"), image_size=color_size, max_frames=max_frames
    )
    sd.export_depth_images(osp.join(scene_dir, "depth"), max_frames=max_frames)
    sd.export_poses(osp.join(scene_dir, "pose"), max_frames=max_frames)
    sd.export_intrinsics(osp.join(scene_dir, "intrinsic"))
    # last, so a run killed mid-export leaves no marker and is redone
    with open(osp.join(scene_dir, EXPORT_MARKER), "w") as f:
        f.write(export_options(max_frames, color_at_depth_res) + "\n")


def main():
    args = get_parser().parse_args()
    if args.max_frames is not None and (args.out_root is None or not args.extracted):
        raise SystemExit("--max-frames needs --out-root and --extracted (see --help)")
    if args.color_at_depth_res and (args.out_root is None or not args.extracted):
        raise SystemExit(
            "--color-at-depth-res needs --out-root and --extracted (see --help)"
        )
    options = (
        export_options(args.max_frames, args.color_at_depth_res)
        if args.extracted
        else None
    )

    # enumerate every scene that actually has a .sens; the export target is the
    # scene's own dir unless --out-root redirects it
    jobs = []
    for split in args.splits:
        split_dir = osp.join(args.raw_root, split)
        if not osp.isdir(split_dir):
            print(f"WARN: split dir missing, skipping: {split_dir}", file=sys.stderr)
            continue
        for scene in sorted(os.listdir(split_dir)):
            scene_dir = osp.join(split_dir, scene)
            sens = osp.join(scene_dir, f"{scene}.sens")
            if osp.isdir(scene_dir) and osp.isfile(sens):
                out_dir = (
                    scene_dir
                    if args.out_root is None
                    else osp.join(args.out_root, split, scene)
                )
                jobs.append((split, scene, out_dir, sens))

    # A shard id at or above the shard count selects nothing (i % n < n
    # always), which would silently extract zero scenes and exit 0.
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(
            f"--shard {args.shard} is out of range for --num-shards "
            f"{args.num_shards}; shard must be in [0, {args.num_shards})"
        )
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(jobs)} scenes assigned",
        flush=True,
    )

    as_zip = not args.extracted
    done = skipped = failed = 0
    for split, scene, scene_dir, sens in jobs:
        if already_extracted(scene_dir, as_zip, options):
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD EXTRACT {split}/{scene} -> {scene_dir}")
            continue
        try:
            print(f"[{done + failed + 1}] extract {split}/{scene}", flush=True)
            os.makedirs(scene_dir, exist_ok=True)
            extract_scene(
                scene_dir, sens, as_zip, args.max_frames, args.color_at_depth_res
            )
            done += 1
        except Exception as e:  # one bad .sens shouldn't kill the shard
            failed += 1
            print(f"FAILED {split}/{scene}: {e}", file=sys.stderr, flush=True)

    print(
        f"[shard {args.shard}] extracted={done} skipped={skipped} failed={failed}",
        flush=True,
    )
    # Per-scene failures are caught above so one bad .sens cannot cost the
    # shard its remaining scenes -- but the TASK still has to fail, or Slurm
    # reports success for an array element that produced nothing. Nothing runs
    # verify_scannet.py at this stage, so an exit status of 0 here is the only
    # signal there is, and the gap would otherwise surface much later as a
    # scene with no frames.zip during preprocessing.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
