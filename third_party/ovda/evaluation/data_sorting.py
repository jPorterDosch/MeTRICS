"""Pair and order the RGB/GT frames used in the oVDA paper.

These functions return {sequence_name: [(rgb_path, depth_path), ...]}.
Only paired RGB frames should be passed to the model, in the returned order.
The reference sequence lists are documentation, not runtime dependencies.
"""

from bisect import bisect_left, bisect_right
from pathlib import Path
import re


def natural_key(path):
    """Sort numeric components numerically, as in the original loaders."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def sintel_pairs(root):
    """Use training/final RGB and training/depth GT with matching frame IDs."""
    root = Path(root)
    sequences = {}
    for scene in sorted((root / "training/final").iterdir(), key=natural_key):
        if not scene.is_dir():
            continue
        rgb = sorted(scene.glob("frame_*.png"), key=natural_key)
        depth = sorted(
            (root / "training/depth" / scene.name).glob("frame_*.dpt"), key=natural_key
        )
        if [path.stem for path in rgb] != [path.stem for path in depth]:
            raise ValueError(f"Mismatched Sintel frame IDs: {scene.name}")
        sequences[scene.name] = list(zip(rgb, depth))
    return sequences


def kitti_pairs(root):
    """Use annotated/train drives; image_03 then image_02 are separate sequences.

    ``root`` contains kitti_depth/ and kitti_raw/. Pair by filename and feed
    only RGB frames with GT to the model. No Eigen/Garg crop is applied.
    """
    root = Path(root)
    annotation = root / "kitti_depth/data_depth_annotated/train"
    sequences = {}
    drives = sorted(
        (path for path in annotation.iterdir() if "_drive_" in path.name), key=natural_key
    )
    for drive in drives:
        date = drive.name.split("_drive_")[0]
        for camera in ("image_03", "image_02"):
            depth = sorted(
                (drive / "proj_depth/groundtruth" / camera).glob("*.png"), key=natural_key
            )
            rgb_dir = root / "kitti_raw" / date / drive.name / camera / "data"
            pairs = [(rgb_dir / path.name, path) for path in depth]
            if any(not rgb.is_file() for rgb, _ in pairs):
                raise ValueError(f"Missing KITTI RGB: {drive.name}/{camera}")
            if pairs:
                sequences[f"{drive.name}-{camera}"] = pairs
    return sequences


def read_timestamps(path):
    """Read timestamp/path text; duplicate timestamps retain the last entry."""
    entries = {}
    for line in Path(path).read_text().replace(",", " ").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) > 1:
            entries[float(fields[0])] = fields[1]
    return entries


def associate_timestamps(rgb, depth):
    """Match unused timestamps greedily by smallest difference, strictly <20 ms.

    Offset is zero. Ties are resolved by RGB then depth timestamp. Searching
    a local timestamp window avoids comparing every RGB to every depth.
    The final paired frames are sorted by RGB path, matching the old loader.
    """
    depth_times = sorted(depth)
    candidates = []
    for rgb_time in rgb:
        start = bisect_left(depth_times, rgb_time - 0.02)
        stop = bisect_right(depth_times, rgb_time + 0.02)
        for depth_time in depth_times[start:stop]:
            difference = abs(rgb_time - depth_time)
            if difference < 0.02:
                candidates.append((difference, rgb_time, depth_time))
    used_rgb, used_depth, pairs = set(), set(), []
    for _, rgb_time, depth_time in sorted(candidates):
        if rgb_time not in used_rgb and depth_time not in used_depth:
            used_rgb.add(rgb_time)
            used_depth.add(depth_time)
            pairs.append((rgb[rgb_time], depth[depth_time]))
    return sorted(pairs)


def bonn_pairs(root):
    """Pair all Bonn sequences from rgb.txt/depth.txt, excluding missing files.

    Scene order follows directory iteration, as in the original loader.
    The supplied sequence list records the historical order; using sequence
    names instead of numeric scene indices avoids filesystem-order ambiguity.
    """
    sequences = {}
    for scene in Path(root).iterdir():
        if not scene.is_dir() or "rgbd_bonn" not in scene.name:
            continue
        rgb = read_timestamps(scene / "rgb.txt")
        depth = read_timestamps(scene / "depth.txt")
        pairs = associate_timestamps(rgb, depth)
        sequences[scene.name] = [
            (scene / a, scene / b)
            for a, b in pairs
            if (scene / a).is_file() and (scene / b).is_file()
        ]
    return sequences
