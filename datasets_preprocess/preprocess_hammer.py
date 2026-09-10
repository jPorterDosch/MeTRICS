#!/usr/bin/env python3
"""
Preprocess Script for the HAMMER RGB-D Dataset

HAMMER (https://github.com/Junggy/HAMMER-dataset) ships 64 sequences named
sceneX_trajY[_naked]_Z. This script converts the polarization camera (the RGB
sensor) of every sequence into the layout the CUT3R/DUSt3R loaders expect
(same shape as processed ScanNet):

    <output_dir>/<split>/<sequence>/
        frames.zip              # ONE uncompressed archive holding
            rgb/XXXXXX.png      #   copied as-is from polarization/rgb
            depth/XXXXXX.png    #   copied as-is from polarization/_gt (uint16, mm)
            cam/XXXXXX.npz      #   {intrinsics: (3,3), pose: (4,4) cam2world} f32
        scene_metadata.npz      # {images: sorted frame basenames}, a real file

The archive keeps this to 3 inodes per sequence instead of ~2000; members are
STORED, so the rgb/depth bytes stay bit-exact copies of the raw input and the
layout is interchangeable with --extracted (loose files, the original layout).

Raw per-sequence inputs are read from either layout: loose files under
<sequence>/polarization/, or members of the <sequence>/frames.zip that
download_hammer.py writes by default.

Raw per-sequence inputs (all under <sequence>/polarization/):
    intrinsics.txt  3x3 pinhole K (single camera per sequence)
    rgb/XXXXXX.png  uint8 RGB
    _gt/XXXXXX.png  uint16 ground-truth depth in millimeters
    _pose/XXXXXX.txt  4x4 cam2world pose. The convention was verified
        empirically: warping GT depth from frame 000000 into frame 000100 of
        scene2_traj1_1 with relative pose inv(P1) @ P0 gives 0.4 mm median
        error vs 61 mm for the world2cam hypothesis.

The official split (dataset README) is train: scene2-11, test: scene12-14.

This script fails fast: any missing file, frame-count mismatch, non-contiguous
frame numbering, unreadable image, non-finite/non-rigid pose, or implausible
depth/intrinsics raises immediately instead of skipping.

Usage:
    python preprocess_hammer.py \
        --hammer_dir ~/scratch/data/hammer \
        --output_dir ~/scratch/data/processed_hammer
"""

import argparse
import io
import os
import os.path as osp
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

# same sibling-import workaround preprocess_arkitscenes.py uses
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from dust3r.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    exists as zexists,
    frames_root,
    listdir as zlistdir,
    read_bytes,
)

TRAIN_SCENES = {f"scene{i}" for i in range(2, 12)}
TEST_SCENES = {"scene12", "scene13", "scene14"}
# Counts read off the official _dataset_processed.zip central directory; a
# mismatch means an incomplete download or a changed upstream archive.
EXPECTED_NUM_SEQUENCES = {"train": 46, "test": 18}
SEQUENCE_RE = re.compile(r"^scene(\d+)_traj\d+(?:_naked)?_\d+$")
MAX_PLAUSIBLE_DEPTH_MM = 30000  # HAMMER is a tabletop/indoor setup
MIN_GT_VALID_FRACTION = 0.3  # _gt is rendered from the scene mesh, mostly dense


def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess HAMMER dataset.")
    parser.add_argument(
        "--hammer_dir",
        default=os.path.expanduser("~/scratch/data/hammer"),
        help="Directory containing the extracted sceneX_trajY_Z sequences.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/scratch/data/processed_hammer"),
        help="Directory where the processed train/ and test/ splits are written.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel worker processes (one sequence per task).",
    )
    parser.add_argument(
        "--extracted",
        action="store_true",
        help="write loose per-frame files instead of one frames.zip per "
        "sequence (the original layout; ~3 inodes per frame)",
    )
    return parser


def split_of_sequence(seq):
    match = SEQUENCE_RE.match(seq)
    if match is None:
        raise ValueError(f"Unrecognized HAMMER sequence name: {seq!r}")
    scene = f"scene{match.group(1)}"
    if scene in TRAIN_SCENES:
        return "train"
    if scene in TEST_SCENES:
        return "test"
    raise ValueError(f"{seq!r}: scene {scene!r} is in neither the train nor test split")


def load_intrinsics(pol_dir, seq):
    intrinsics_path = osp.join(pol_dir, "intrinsics.txt")
    K = np.loadtxt(io.StringIO(read_bytes(intrinsics_path).decode()))
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"{seq}: bad intrinsics matrix in {intrinsics_path}:\n{K}")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{seq}: non-positive focal length in {intrinsics_path}")
    return K.astype(np.float32)


def validate_intrinsics_against_image(K, W, H, seq):
    cx, cy = K[0, 2], K[1, 2]
    # mirror the principal-point margin assert in
    # BaseMultiViewDataset._crop_resize_if_necessary so a bad K fails here
    # instead of mid-training
    if min(cx, W - cx) <= W / 5 or min(cy, H - cy) <= H / 5:
        raise ValueError(
            f"{seq}: principal point ({cx:.1f}, {cy:.1f}) too close to the "
            f"border of a {W}x{H} image"
        )


def load_pose(pose_path, seq):
    pose = np.loadtxt(io.StringIO(read_bytes(pose_path).decode()))
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{seq}: bad pose in {pose_path}:\n{pose}")
    R = pose[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        raise ValueError(f"{seq}: non-orthonormal rotation in {pose_path}")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
        raise ValueError(f"{seq}: rotation determinant != 1 in {pose_path}")
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError(f"{seq}: last pose row is not [0, 0, 0, 1] in {pose_path}")
    return pose.astype(np.float32)


def list_frames(dirpath, ext, seq):
    basenames = sorted(f[: -len(ext)] for f in zlistdir(dirpath) if f.endswith(ext))
    if not basenames:
        raise ValueError(f"{seq}: no *{ext} files in {dirpath}")
    expected = [f"{i:06d}" for i in range(len(basenames))]
    if basenames != expected:
        raise ValueError(
            f"{seq}: frames in {dirpath} are not contiguous 000000..{len(basenames) - 1:06d}"
        )
    return basenames


def process_sequence(seq, input_dir, output_dir, as_zip=True):
    # raw frames come from either layout: loose files under <seq>/polarization,
    # or members of the <seq>/frames.zip that download_hammer.py writes
    pol_dir = osp.join(frames_root(osp.join(input_dir, seq)), "polarization")
    rgb_dir = osp.join(pol_dir, "rgb")
    gt_dir = osp.join(pol_dir, "_gt")
    pose_dir = osp.join(pol_dir, "_pose")
    for d in (rgb_dir, gt_dir, pose_dir):
        if not zexists(d):
            raise FileNotFoundError(f"{seq}: missing directory {d}")

    basenames = list_frames(rgb_dir, ".png", seq)
    if list_frames(gt_dir, ".png", seq) != basenames:
        raise ValueError(f"{seq}: _gt frames do not match rgb frames")
    if list_frames(pose_dir, ".txt", seq) != basenames:
        raise ValueError(f"{seq}: _pose frames do not match rgb frames")

    K = load_intrinsics(pol_dir, seq)

    seq_out = osp.join(output_dir, split_of_sequence(seq), seq)
    os.makedirs(seq_out, exist_ok=True)

    if as_zip:
        writer = SceneZipWriter(osp.join(seq_out, "frames.zip"))
        emit = writer.writestr
    else:
        writer = None
        for sub in ("rgb", "depth", "cam"):
            os.makedirs(osp.join(seq_out, sub), exist_ok=True)

        def emit(member_name, data):
            with open(osp.join(seq_out, member_name), "wb") as f:
                f.write(data)

    try:
        num = _convert_frames(seq, basenames, rgb_dir, gt_dir, pose_dir, K, emit)
    except BaseException:
        if writer is not None:
            writer.__exit__(RuntimeError, RuntimeError("aborted"), None)
        raise
    if writer is not None:
        writer.__exit__(None, None, None)  # closes + renames .tmp -> final

    # metadata stays a real file in the scene dir in both layouts, so the
    # loader can read it without opening the archive
    np.savez(osp.join(seq_out, "scene_metadata.npz"), images=basenames)
    return seq, num


def _convert_frames(seq, basenames, rgb_dir, gt_dir, pose_dir, K, emit):
    """Validate and convert every frame, handing outputs to emit(name, bytes).

    Every check the original performed is unchanged; only the sink differs
    (zip member vs loose file). rgb and depth are passed through as raw bytes
    rather than re-encoded, so they stay bit-exact just as the original
    shutil.copyfile did.
    """
    shape = None
    for basename in basenames:
        rgb_path = osp.join(rgb_dir, f"{basename}.png")
        depth_path = osp.join(gt_dir, f"{basename}.png")
        pose_path = osp.join(pose_dir, f"{basename}.txt")

        rgb_raw = read_bytes(rgb_path)
        depth_raw = read_bytes(depth_path)
        rgb = cv2.imdecode(np.frombuffer(rgb_raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{seq}: unreadable or non-RGB image {rgb_path}")
        depth = cv2.imdecode(np.frombuffer(depth_raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError(f"{seq}: unreadable or non-uint16 depth {depth_path}")
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"{seq}: depth shape {depth.shape} != rgb shape {rgb.shape[:2]} "
                f"for frame {basename}"
            )
        if depth.max() >= MAX_PLAUSIBLE_DEPTH_MM:
            raise ValueError(
                f"{seq}: max depth {depth.max()} mm in {depth_path} exceeds "
                f"{MAX_PLAUSIBLE_DEPTH_MM} mm; wrong file or wrong unit?"
            )
        if (depth > 0).mean() < MIN_GT_VALID_FRACTION:
            raise ValueError(
                f"{seq}: only {(depth > 0).mean():.1%} valid GT depth in {depth_path}"
            )
        if shape is None:
            shape = depth.shape
            H, W = shape
            validate_intrinsics_against_image(K, W, H, seq)
        elif depth.shape != shape:
            raise ValueError(
                f"{seq}: inconsistent image shapes within sequence "
                f"({depth.shape} vs {shape})"
            )

        pose = load_pose(pose_path, seq)

        cam_buf = io.BytesIO()
        np.savez(cam_buf, intrinsics=K, pose=pose)
        emit(f"cam/{basename}.npz", cam_buf.getvalue())
        # pass the raw bytes through instead of re-encoding: keeps RGB and GT
        # depth bit-exact, as the original shutil.copyfile did
        emit(f"rgb/{basename}.png", rgb_raw)
        emit(f"depth/{basename}.png", depth_raw)

    return len(basenames)


def main(input_dir, output_dir, num_workers, as_zip=True):
    if not osp.isdir(input_dir):
        raise FileNotFoundError(f"HAMMER directory not found: {input_dir}")

    sequences = sorted(
        d
        for d in os.listdir(input_dir)
        if osp.isdir(osp.join(input_dir, d)) and d.startswith("scene")
    )
    if not sequences:
        raise ValueError(f"No sceneX_trajY_Z sequences found in {input_dir}")

    split_counts = {"train": 0, "test": 0}
    for seq in sequences:
        split_counts[split_of_sequence(seq)] += 1
    if split_counts != EXPECTED_NUM_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_SEQUENCES} sequences per split, found "
            f"{split_counts}. Incomplete download or changed upstream archive?"
        )

    os.makedirs(output_dir, exist_ok=True)

    num_frames = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_sequence, seq, input_dir, output_dir, as_zip): seq
            for seq in sequences
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sequences"):
            seq, n = future.result()  # re-raises the first worker error
            num_frames[seq] = n

    total = sum(num_frames.values())
    for split, expected in EXPECTED_NUM_SEQUENCES.items():
        produced = sorted(os.listdir(osp.join(output_dir, split)))
        if len(produced) != expected:
            raise RuntimeError(
                f"{split}: wrote {len(produced)} sequences, expected {expected}"
            )
    print(
        f"HAMMER preprocessing successful: {len(sequences)} sequences "
        f"({split_counts['train']} train / {split_counts['test']} test), "
        f"{total} frames -> {output_dir}"
    )


if __name__ == "__main__":
    parser = get_parser()
    cli_args = parser.parse_args()
    main(
        cli_args.hammer_dir,
        cli_args.output_dir,
        cli_args.num_workers,
        as_zip=not cli_args.extracted,
    )
