#!/usr/bin/env python
"""Batch-extract raw ScanNet ``.sens`` files -- the one stage the repo's
ScanNet pipeline was missing.

The stock ScanNet reader (``scannet_reader.py`` / ``SensorData``) handles a
single ``.sens`` at a time. This walks the scene tree and runs the same four
exports for every scene, writing *in place* into each scene dir:

    <raw_root>/<split>/<scene>/
        <scene>.sens          # input, left untouched
        color/{i}.jpg         # export_color_images
        depth/{i}.png         # export_depth_images
        pose/{i}.txt          # export_poses
        intrinsic/intrinsic_{color,depth}.txt + extrinsic_*   # export_intrinsics

That layout is exactly what ``preprocess_scannet.py`` consumes next, so the full
pipeline is three explicit stages, each its own script (no duplicated logic):

    1. python extract_scannet_sens.py  --raw-root <raw>            # this script
    2. python preprocess_scannet.py    --scannet_dir <raw> --output_dir <proc>
    3. python generate_set_scannet.py  --root <proc> --splits scans_train scans_test ...

Resumable: a scene whose ``intrinsic/intrinsic_depth.txt`` exists and whose
``color`` frame count matches ``depth`` is skipped. Shardable for a SLURM array
via ``--shard i --num-shards N`` (scenes assigned round-robin).

Run with the StreamVGGT env python (needs imageio + pypng, which SensorData uses).
"""

import argparse
import os
import os.path as osp
import sys

from scannet_sensor import SensorData


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        default="/gpfs/data/jtompki1/cli277/metric/scannet",
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
    return p


def already_extracted(scene_dir):
    """True if this scene already has a complete-looking extraction (skip it)."""
    intr = osp.join(scene_dir, "intrinsic", "intrinsic_depth.txt")
    color, depth = osp.join(scene_dir, "color"), osp.join(scene_dir, "depth")
    if not (osp.isfile(intr) and osp.isdir(color) and osp.isdir(depth)):
        return False
    nc = len(os.listdir(color))
    return nc > 0 and nc == len(os.listdir(depth))


def extract_scene(scene_dir, sens_path):
    """Export color/depth/pose/intrinsic from one .sens, in place."""
    sd = SensorData(sens_path)
    sd.export_color_images(osp.join(scene_dir, "color"))
    sd.export_depth_images(osp.join(scene_dir, "depth"))
    sd.export_poses(osp.join(scene_dir, "pose"))
    sd.export_intrinsics(osp.join(scene_dir, "intrinsic"))


def main():
    args = get_parser().parse_args()

    # enumerate every scene that actually has a .sens
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
                jobs.append((split, scene, scene_dir, sens))

    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(jobs)} scenes assigned",
        flush=True,
    )

    done = skipped = failed = 0
    for split, scene, scene_dir, sens in jobs:
        if already_extracted(scene_dir):
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD EXTRACT {split}/{scene}")
            continue
        try:
            print(f"[{done + failed + 1}] extract {split}/{scene}", flush=True)
            extract_scene(scene_dir, sens)
            done += 1
        except Exception as e:  # one bad .sens shouldn't kill the shard
            failed += 1
            print(f"FAILED {split}/{scene}: {e}", file=sys.stderr, flush=True)

    print(
        f"[shard {args.shard}] extracted={done} skipped={skipped} failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
