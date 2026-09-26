"""GT cameras for the video-depth benchmark manifests.

VDA's gen_json writes no cameras (only its ScanNet TAE manifest has K/pose),
yet every video dataset ships them. This module reads each dataset's native
camera format and attach_cameras() rewrites a manifest with, per frame,
    "K":    3x3 pixel intrinsics of the frame the GT depth is stored in
            (UNCROPPED -- the eval crop is applied downstream by
            vda_benchmark.scaled_intrinsics, as for VDA's own ScanNet K)
    "pose": 4x4 cam2world, metric
Consumers: TAE (reprojection between adjacent frames) and the point-cloud
snapshots (unprojection), both in src/bench_eval.py. Nothing in the depth
protocols reads them.

Sources:
  scannet  <scene>/pose/<i>.txt + <scene>/intrinsic/intrinsic_depth.txt,
           copied into the tree by VDA's extractor. Non-finite poses
           (tracking failures, -inf) are kept verbatim, as in VDA's TAE
           manifest; the TAE code handles them per pair.
  sintel   training/camdata_left/<seq>/frame_%04d.cam: float tag, then M
           (3x3 intrinsics, float64) and N (3x4 world->cam, float64);
           pose = inv([N; 0 0 0 1]), the MonST3R convention.
  bonn     rgbd_bonn_<seq>/groundtruth.txt (TUM: t tx ty tz qx qy qz qw,
           cam2world), nearest timestamp to the RGB filename; the dataset's
           published RGB intrinsics.
  kitti    oxts/data/%010d.txt + the per-date calib files, composed as in
           pykitti: T_w_cam2 = T_w_imu @ inv(T_cam2_velo @ T_velo_imu), with
           T_cam2_velo = T2 @ R_rect_00 @ T_cam0_velo; K = P_rect_02[:3,:3].
  nyuv2    stills: identity pose, the standard NYU-D v2 RGB intrinsics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAG_FLOAT = 202021.25

# Bonn RGB-D Dynamic Dataset (Palazzolo et al. 2019), RGB camera, 640x480;
# depth is registered to RGB so the same K serves both.
BONN_K = np.array(
    [[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]]
)
# NYU Depth v2 RGB camera (toolbox camera_params, the values every depth
# paper uses), 640x480 before the 45:471, 41:601 crop.
NYU_K = np.array(
    [[518.857901, 0.0, 325.582449], [0.0, 519.469611, 253.736166], [0.0, 0.0, 1.0]]
)


def _finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w -- TUM order) -> 3x3 rotation."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def tum_pose(row: np.ndarray) -> np.ndarray:
    """One TUM groundtruth row (t tx ty tz qx qy qz qw) -> 4x4 cam2world."""
    if row.shape[0] < 8:
        raise ValueError(f"TUM row needs 8 values, got {row.shape[0]}")
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*row[4:8])
    T[:3, 3] = row[1:4]
    return T


def read_tum_trajectory(path: Path) -> np.ndarray:
    """[N,8] rows of a TUM groundtruth.txt (comment lines skipped)."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] < 8:
        raise ValueError(f"{path}: expected >= 8 columns, got {rows.shape[1]}")
    return rows


def nearest_tum_pose(
    traj: np.ndarray, timestamp: float, max_dt: float = 0.05
) -> np.ndarray:
    """cam2world of the trajectory row nearest `timestamp` (Bonn RGB and GT
    run at ~30 Hz, so a match further than max_dt is a real gap)."""
    i = int(np.argmin(np.abs(traj[:, 0] - timestamp)))
    dt = abs(traj[i, 0] - timestamp)
    if dt > max_dt:
        raise ValueError(
            f"no groundtruth pose within {max_dt}s of t={timestamp} (nearest {dt:.3f}s)"
        )
    return tum_pose(traj[i])


def _bonn_pose_or_none(traj: np.ndarray, t: float) -> np.ndarray | None:
    """A frame with no mocap pose within tolerance (balloon2 has one 54 ms gap,
    outside the protocol's frames 30-140) gets no pose: that sequence is then
    skipped for TAE in the manifest it appears in, instead of failing the
    whole dataset."""
    try:
        return nearest_tum_pose(traj, t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Sintel
# ---------------------------------------------------------------------------
def sintel_cam_read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(K 3x3, N 3x4 world->cam) from a Sintel .cam file."""
    with open(path, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != TAG_FLOAT:
            raise ValueError(f"{path}: bad .cam tag {check} (expected {TAG_FLOAT})")
        M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        N = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    return M, N


def sintel_pose(N: np.ndarray) -> np.ndarray:
    """3x4 world->cam -> 4x4 cam2world."""
    T = np.eye(4)
    T[:3, :4] = N
    return np.linalg.inv(T)


# ---------------------------------------------------------------------------
# KITTI (pykitti's composition, re-stated so no extra dependency is needed)
# ---------------------------------------------------------------------------
def read_kitti_calib(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                out[key.strip()] = np.array([float(x) for x in val.split()])
            except ValueError:
                continue  # calib_time etc.
    return out


def _rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = t.reshape(3)
    return T


def kitti_cam2_calibration(date_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(K_cam2 3x3, T_cam2_imu 4x4) from the three per-date calib files."""
    c2c = read_kitti_calib(date_dir / "calib_cam_to_cam.txt")
    v2c = read_kitti_calib(date_dir / "calib_velo_to_cam.txt")
    i2v = read_kitti_calib(date_dir / "calib_imu_to_velo.txt")
    for key, src in (("P_rect_02", c2c), ("R_rect_00", c2c), ("R", v2c), ("T", v2c)):
        if key not in src:
            raise KeyError(f"{date_dir}: calib lacks {key}")
    P_rect_02 = c2c["P_rect_02"].reshape(3, 4)
    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    T_cam0_velo = _rt(v2c["R"], v2c["T"])
    T_velo_imu = _rt(i2v["R"], i2v["T"])
    T2 = np.eye(4)
    T2[0, 3] = P_rect_02[0, 3] / P_rect_02[0, 0]
    T_cam2_velo = T2 @ R_rect_00 @ T_cam0_velo
    T_cam2_imu = T_cam2_velo @ T_velo_imu
    return P_rect_02[:3, :3].copy(), T_cam2_imu


def kitti_oxts_pose(packet: np.ndarray, scale: float) -> np.ndarray:
    """T_w_imu from one oxts row (lat lon alt roll pitch yaw ...), Mercator
    projection at `scale` = cos(lat0)."""
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    er = 6378137.0
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    tz = alt
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return _rt(Rz @ Ry @ Rx, np.array([tx, ty, tz]))


def kitti_cam2_poses(
    drive_dir: Path, frame_ids: list[str], T_cam2_imu: np.ndarray
) -> list[np.ndarray]:
    """cam2world of cam2 for the given oxts frame ids (%010d), world = the
    first frame's Mercator frame (pykitti's convention, not re-centred)."""
    packets = []
    for fid in frame_ids:
        p = drive_dir / "oxts" / "data" / f"{fid}.txt"
        if not p.is_file():
            raise FileNotFoundError(f"oxts packet missing: {p}")
        packets.append(np.loadtxt(p))
    scale = np.cos(packets[0][0] * np.pi / 180.0)
    T_imu_cam2 = np.linalg.inv(T_cam2_imu)
    return [kitti_oxts_pose(pk, scale) @ T_imu_cam2 for pk in packets]


# ---------------------------------------------------------------------------
# manifest rewrite
# ---------------------------------------------------------------------------
def _scannet_cameras(
    bench_ds: Path, seq: str, frame: dict
) -> tuple[np.ndarray, np.ndarray | None]:
    K = np.loadtxt(bench_ds / seq / "intrinsic" / "intrinsic_depth.txt")[:3, :3]
    stem = Path(frame["image"]).stem
    pose = np.loadtxt(bench_ds / seq / "pose" / f"{stem}.txt")
    if pose.shape != (4, 4):
        raise ValueError(f"{seq}/pose/{stem}.txt: expected 4x4, got {pose.shape}")
    # kept verbatim even when non-finite (ScanNet writes -inf on tracking
    # failure), as VDA's own TAE manifest does: tae_vda must see what theirs
    # sees, and tae_ours drops the affected pairs itself
    return K, pose


def attach_cameras(
    out: Path, name: str, manifest_rel: str, raw: Path
) -> dict[str, int]:
    """Rewrite <out>/<manifest_rel> with K/pose on every frame that has one.
    Returns counts: frames, with_pose, sequences_without_full_pose."""
    path = out / manifest_rel
    with open(path) as f:
        data = json.load(f)
    bench_ds = out / name
    stats = {"frames": 0, "with_pose": 0, "sequences_without_full_pose": 0}
    for entry in data[name]:
        ((seq, frames),) = entry.items()
        cams = _sequence_cameras(name, bench_ds, raw, seq, frames)
        missing = 0
        for fr, (K, pose) in zip(frames, cams):
            fr["K"] = np.asarray(K, dtype=np.float64).tolist()
            if pose is None:
                fr.pop("pose", None)
                missing += 1
            else:
                fr["pose"] = np.asarray(pose, dtype=np.float64).tolist()
                if _finite(np.asarray(pose)):
                    stats["with_pose"] += 1
                else:
                    missing += 1
            stats["frames"] += 1
        if missing:
            stats["sequences_without_full_pose"] += 1
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return stats


def _sequence_cameras(
    name: str, bench_ds: Path, raw: Path, seq: str, frames: list[dict]
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    if name == "scannet":
        return [_scannet_cameras(bench_ds, seq, fr) for fr in frames]
    if name == "sintel":
        cam_dir = raw / "sintel" / "training" / "camdata_left" / seq
        out = []
        for fr in frames:
            M, N = sintel_cam_read(cam_dir / (Path(fr["image"]).stem + ".cam"))
            out.append((M, sintel_pose(N)))
        return out
    if name == "bonn":
        traj = read_tum_trajectory(
            raw / "bonn" / "rgbd_bonn_dataset" / seq / "groundtruth.txt"
        )
        return [
            (BONN_K, _bonn_pose_or_none(traj, float(Path(fr["image"]).stem)))
            for fr in frames
        ]
    if name == "kitti":
        date = seq[:10]
        K, T_cam2_imu = kitti_cam2_calibration(raw / "kitti" / date)
        ids = [Path(fr["image"]).stem for fr in frames]
        poses = kitti_cam2_poses(raw / "kitti" / date / seq, ids, T_cam2_imu)
        return [(K, p) for p in poses]
    if name == "nyuv2":
        return [(NYU_K, np.eye(4)) for _ in frames]
    raise ValueError(f"no camera source for dataset {name!r}")


__all__ = [
    "BONN_K",
    "NYU_K",
    "attach_cameras",
    "kitti_cam2_calibration",
    "kitti_cam2_poses",
    "kitti_oxts_pose",
    "nearest_tum_pose",
    "quat_to_rot",
    "read_tum_trajectory",
    "sintel_cam_read",
    "sintel_pose",
    "tum_pose",
]
