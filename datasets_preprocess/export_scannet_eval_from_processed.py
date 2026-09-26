#!/usr/bin/env python
"""Write the loose ScanNet layout VDA's extractor reads, from the repo's
already-processed ScanNet tree instead of the raw .sens files.

    python datasets_preprocess/export_scannet_eval_from_processed.py \\
        --processed /lustre/isaac24/proj/UTK0516/metrics_data/processed/processed_scannet/scans_test \\
        --out-root $EVAL_RAW/scannet --max-frames 510

The raw scans_test .sens are owned by another user with mode 0600, so
extract_scannet_sens.py cannot read them. The processed tree
(preprocess_scannet.py output, world-readable) already holds exactly what
the benchmark needs, per scene in frames.zip:
    color/<i>.jpg   colour resized to the 640x480 depth size (LANCZOS) --
                    what --color-at-depth-res would have produced
    depth/<i>.png   uint16 millimetres, untouched
    cam/<i>.npz     intrinsics (the depth camera's 3x3) + cam2world pose
NOT frame-exact with VDA. Frames whose pose was non-finite were dropped by
preprocess_scannet.py, and VDA keeps them, so "the first 510 files" lands on
later frames wherever a scene has holes: 16 of the 100 scans_test scenes
(scene0791_00 shifted by 1011 frames, scene0794_00 by 100, scene0737_00 by
47, ten more by 1-30) and scene0718_00 has only 326 valid frames. Use the
.sens path (extract_scannet_sens.py) for numbers meant to sit next to VDA's
table; this is the fallback when the .sens are unreadable.

Output per scene: <out-root>/scans_test/<scene>/{color,depth,pose,intrinsic}
in the .sens-export naming (pose/<i>.txt as 4x4 text, intrinsic/
intrinsic_depth.txt as 4x4), plus the same .export_options marker
extract_scannet_sens.py writes, so the two exporters are interchangeable
and re-runs skip complete scenes.
"""

from __future__ import annotations

import argparse
import io
import os
import os.path as osp
import sys
import zipfile

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from extract_scannet_sens import EXPORT_MARKER, already_extracted, export_options  # noqa: E402


def export_scene(scene_dir: str, out_dir: str, max_frames: int | None) -> int:
    meta = np.load(osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True)
    names = sorted(str(n) for n in meta["images"])
    if max_frames is not None:
        names = names[:max_frames]
    if not names:
        raise ValueError(f"{scene_dir}: no frames listed in new_scene_metadata.npz")
    for sub in ("color", "depth", "pose", "intrinsic"):
        os.makedirs(osp.join(out_dir, sub), exist_ok=True)
    intrinsic_written = False
    with zipfile.ZipFile(osp.join(scene_dir, "frames.zip")) as z:
        for name in names:
            with open(osp.join(out_dir, "color", f"{name}.jpg"), "wb") as f:
                f.write(z.read(f"color/{name}.jpg"))
            with open(osp.join(out_dir, "depth", f"{name}.png"), "wb") as f:
                f.write(z.read(f"depth/{name}.png"))
            cam = np.load(io.BytesIO(z.read(f"cam/{name}.npz")))
            pose = np.asarray(cam["pose"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"{scene_dir}: bad pose for frame {name}: {pose}")
            np.savetxt(osp.join(out_dir, "pose", f"{name}.txt"), pose, fmt="%f")
            if not intrinsic_written:
                K = np.eye(4)
                K[:3, :3] = np.asarray(cam["intrinsics"], dtype=np.float64)[:3, :3]
                np.savetxt(
                    osp.join(out_dir, "intrinsic", "intrinsic_depth.txt"), K, fmt="%f"
                )
                intrinsic_written = True
    with open(osp.join(out_dir, EXPORT_MARKER), "w") as f:
        f.write(export_options(max_frames, True) + "\n")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--processed",
        default="/lustre/isaac24/proj/UTK0516/metrics_data/processed/processed_scannet/scans_test",
        help="preprocess_scannet.py output split dir (scene dirs with frames.zip)",
    )
    ap.add_argument(
        "--out-root", required=True, help="writes <out-root>/scans_test/<scene>/"
    )
    ap.add_argument("--max-frames", type=int, default=510)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all (sorted)")
    args = ap.parse_args()
    if args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    scenes = sorted(
        d
        for d in os.listdir(args.processed)
        if osp.isfile(osp.join(args.processed, d, "frames.zip"))
    )
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise SystemExit(f"no scenes with frames.zip under {args.processed}")
    options = export_options(args.max_frames, True)
    done = skipped = failed = 0
    for scene in scenes:
        out_dir = osp.join(args.out_root, "scans_test", scene)
        if already_extracted(out_dir, as_zip=False, options=options):
            skipped += 1
            continue
        try:
            n = export_scene(osp.join(args.processed, scene), out_dir, args.max_frames)
            done += 1
            print(f"[{done}] {scene}: {n} frames", flush=True)
        except Exception as e:  # one bad scene must not cost the rest
            failed += 1
            print(f"FAILED {scene}: {e}", file=sys.stderr, flush=True)
    print(f"exported={done} skipped={skipped} failed={failed}", flush=True)
    print(
        "NOTE: frame selection differs from VDA's on scenes with dropped "
        "non-finite-pose frames (see module docstring); .sens export for parity",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
