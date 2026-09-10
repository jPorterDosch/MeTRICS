import argparse
import io
import os
import os.path as osp
import sys

from PIL import Image
import numpy as np
import cv2
import multiprocessing
from tqdm import tqdm
import path_to_root  # noqa
import datasets_preprocess.utils.cropping as cropping  # noqa

# same sibling-import workaround preprocess_arkitscenes.py uses: path_to_root
# puts the repo root on sys.path, but the packages live under src/
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from dust3r.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    frames_root,
    listdir as zlistdir,
    read_bytes,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_dir", default="data/data_scannet")
    parser.add_argument("--output_dir", default="data/dust3r_data/processed_scannet")
    parser.add_argument(
        "--extracted",
        action="store_true",
        help="write loose per-frame files instead of one frames.zip per scene "
        "(the original layout; ~one inode per frame)",
    )
    return parser


def process_scene(args):
    """Convert one scene. Returns (scene, error-or-None) -- never raises.

    A worker exception would propagate out of pool.map and abort main()
    partway through 1613 scenes, so a single scene left without a frames.zip
    by a timed-out extract task would cost the whole 12 h job. Reporting the
    failure instead lets the other scenes finish and lists what to re-run.
    """
    rootdir, outdir, split, scene, as_zip = args
    try:
        _process_scene(rootdir, outdir, split, scene, as_zip)
    except Exception as e:
        return f"{split}/{scene}", repr(e)
    return f"{split}/{scene}", None


def _process_scene(rootdir, outdir, split, scene, as_zip):
    # Resume: a final-named frames.zip is complete (SceneZipWriter renames
    # only on success), so a re-run after a walltime kill picks up where it
    # stopped instead of redoing every scene it already converted.
    out_scene_dir = osp.join(outdir, split, scene)
    if as_zip and osp.isfile(osp.join(out_scene_dir, "frames.zip")):
        return
    # input frames come from either layout: loose files in the scene dir, or
    # members of the frames.zip written by extract_scannet_sens.py
    frame_dir = frames_root(osp.join(rootdir, split, scene))
    rgb_dir = osp.join(frame_dir, "color")
    depth_dir = osp.join(frame_dir, "depth")
    pose_dir = osp.join(frame_dir, "pose")
    depth_intrinsic = np.loadtxt(
        io.StringIO(
            read_bytes(osp.join(frame_dir, "intrinsic", "intrinsic_depth.txt")).decode()
        )
    )[:3, :3].astype(np.float32)
    color_intrinsic = np.loadtxt(
        io.StringIO(
            read_bytes(osp.join(frame_dir, "intrinsic", "intrinsic_color.txt")).decode()
        )
    )[:3, :3].astype(np.float32)
    if not np.isfinite(depth_intrinsic).all() or not np.isfinite(color_intrinsic).all():
        return
    os.makedirs(out_scene_dir, exist_ok=True)
    frame_num = len(zlistdir(rgb_dir))
    n_depth, n_pose = len(zlistdir(depth_dir)), len(zlistdir(pose_dir))
    if not (frame_num == n_depth == n_pose):
        raise ValueError(
            f"{split}/{scene}: stream lengths disagree -- color {frame_num}, "
            f"depth {n_depth}, pose {n_pose}; the extraction is incomplete"
        )

    if as_zip:
        # all converted frames go into ONE uncompressed zip per scene
        # (inode-safe layout); the write is atomic (.tmp -> rename), so a
        # final-named frames.zip is always complete
        with SceneZipWriter(osp.join(out_scene_dir, "frames.zip")) as writer:
            _convert_frames(
                frame_num,
                rgb_dir,
                depth_dir,
                pose_dir,
                depth_intrinsic,
                writer.writestr,
            )
        return

    out_rgb_dir = osp.join(out_scene_dir, "color")
    out_depth_dir = osp.join(out_scene_dir, "depth")
    out_cam_dir = osp.join(out_scene_dir, "cam")
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    def emit(member_name, data):
        with open(osp.join(out_scene_dir, member_name), "wb") as f:
            f.write(data)

    _convert_frames(frame_num, rgb_dir, depth_dir, pose_dir, depth_intrinsic, emit)


def _convert_frames(frame_num, rgb_dir, depth_dir, pose_dir, depth_intrinsic, emit):
    """Convert every frame and hand each output to `emit(member_name, bytes)`.

    Shared by both layouts so the pixel data and the per-frame skip rules are
    identical; only the sink differs (zip member vs loose file). The rgb is
    resized to the depth resolution and re-encoded as JPEG exactly as the
    original loose-file version did via PIL's Image.save.
    """
    for i in tqdm(range(frame_num)):
        rgb = Image.open(io.BytesIO(read_bytes(osp.join(rgb_dir, f"{i}.jpg"))))
        depth = cv2.imdecode(
            np.frombuffer(read_bytes(osp.join(depth_dir, f"{i}.png")), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        rgb = rgb.resize(depth.shape[::-1], resample=Image.Resampling.LANCZOS)
        pose = (
            np.loadtxt(io.StringIO(read_bytes(osp.join(pose_dir, f"{i}.txt")).decode()))
            .reshape(4, 4)
            .astype(np.float32)
        )
        if not np.isfinite(pose).all():
            continue

        cam_buf = io.BytesIO()
        np.savez(cam_buf, intrinsics=depth_intrinsic, pose=pose)
        emit(f"cam/{i:05d}.npz", cam_buf.getvalue())

        rgb_buf = io.BytesIO()
        rgb.save(rgb_buf, format="JPEG")
        emit(f"color/{i:05d}.jpg", rgb_buf.getvalue())

        ok, enc = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError(f"failed to encode depth frame {i}")
        emit(f"depth/{i:05d}.png", enc.tobytes())


def main(rootdir, outdir, as_zip=True):
    os.makedirs(outdir, exist_ok=True)
    splits = ["scans_test", "scans_train"]
    failures = []
    # sched_getaffinity respects the slurm/cgroup CPU allocation;
    # cpu_count() would oversubscribe a shared batch node
    pool = multiprocessing.Pool(processes=len(os.sched_getaffinity(0)))

    for split in splits:
        scenes = [
            f
            for f in os.listdir(os.path.join(rootdir, split))
            if os.path.isdir(osp.join(rootdir, split, f))
        ]
        results = pool.map(
            process_scene,
            [(rootdir, outdir, split, scene, as_zip) for scene in scenes],
        )
        failures.extend((name, err) for name, err in results if err)
    pool.close()
    pool.join()

    if failures:
        print(f"{len(failures)} scene(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        print(
            "re-run this script to retry them; converted scenes are skipped",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    sys.exit(main(args.scannet_dir, args.output_dir, as_zip=not args.extracted))
