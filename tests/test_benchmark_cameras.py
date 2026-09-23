"""GT camera readers for the benchmark manifests (numpy only).

python -m unittest tests/test_benchmark_cameras.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "datasets_preprocess"))

import benchmark_cameras as C  # noqa: E402


class QuaternionTest(unittest.TestCase):
    def test_identity_and_90deg_about_z(self):
        self.assertTrue(np.allclose(C.quat_to_rot(0, 0, 0, 1), np.eye(3)))
        s = np.sqrt(0.5)
        R = C.quat_to_rot(0, 0, s, s)  # 90 deg about z
        self.assertTrue(np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9))

    def test_tum_row_to_cam2world(self):
        T = C.tum_pose(np.array([1.5, 1.0, 2.0, 3.0, 0, 0, 0, 1]))
        self.assertTrue(np.allclose(T[:3, 3], [1, 2, 3]))
        self.assertTrue(np.allclose(T[:3, :3], np.eye(3)))

    def test_nearest_timestamp_and_gap_error(self):
        traj = np.array([[10.00, 0, 0, 0, 0, 0, 0, 1], [10.033, 1, 0, 0, 0, 0, 0, 1]])
        self.assertAlmostEqual(C.nearest_tum_pose(traj, 10.03)[0, 3], 1.0)
        with self.assertRaises(ValueError):
            C.nearest_tum_pose(traj, 11.0)


class SintelTest(unittest.TestCase):
    def test_cam_round_trip(self):
        M = np.array([[1120.0, 0, 511.5], [0, 1120.0, 217.5], [0, 0, 1]])
        N = np.hstack([np.eye(3), np.array([[0.5], [0.0], [2.0]])])  # world->cam
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "frame_0001.cam"
            with open(p, "wb") as f:
                np.array([C.TAG_FLOAT], np.float32).tofile(f)
                M.astype(np.float64).tofile(f)
                N.astype(np.float64).tofile(f)
            M2, N2 = C.sintel_cam_read(p)
        self.assertTrue(np.allclose(M2, M))
        pose = C.sintel_pose(N2)
        self.assertTrue(
            np.allclose(pose[:3, 3], [-0.5, 0.0, -2.0])
        )  # cam2world = inverse


class KittiTest(unittest.TestCase):
    def test_oxts_translation_east_and_yaw(self):
        lat0 = 49.0
        scale = np.cos(lat0 * np.pi / 180.0)
        a = C.kitti_oxts_pose(np.array([lat0, 8.0, 100.0, 0, 0, 0]), scale)
        # ~10 m east: 1 deg lon = pi*er/180 * scale metres
        dlon = 10.0 / (np.pi * 6378137.0 / 180.0 * scale)
        b = C.kitti_oxts_pose(
            np.array([lat0, 8.0 + dlon, 100.0, 0, 0, np.pi / 2]), scale
        )
        d = b[:3, 3] - a[:3, 3]
        self.assertAlmostEqual(d[0], 10.0, places=3)
        self.assertAlmostEqual(d[1], 0.0, places=3)
        self.assertTrue(
            np.allclose(b[:3, :3] @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)
        )

    def test_calibration_composition(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "calib_cam_to_cam.txt").write_text(
                "calib_time: x\nP_rect_02: 700 0 600 -40 0 700 180 0 0 0 1 0\n"
                "R_rect_00: 1 0 0 0 1 0 0 0 1\n"
            )
            (d / "calib_velo_to_cam.txt").write_text(
                "R: 0 -1 0 0 0 -1 1 0 0\nT: 0 0 0\n"
            )
            (d / "calib_imu_to_velo.txt").write_text(
                "R: 1 0 0 0 1 0 0 0 1\nT: -1 0 0\n"
            )
            K, T_cam2_imu = C.kitti_cam2_calibration(d)
        self.assertTrue(np.allclose(K, [[700, 0, 600], [0, 700, 180], [0, 0, 1]]))
        # T2 shifts cam2 by P_rect_02[0,3]/fx = -40/700 along x, after the velo->cam rotation
        self.assertAlmostEqual(T_cam2_imu[0, 3], -40 / 700 + 0.0, places=9)
        self.assertEqual(T_cam2_imu.shape, (4, 4))


class AttachTest(unittest.TestCase):
    def test_nyu_and_bonn_manifests_get_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, raw = Path(tmp) / "bench", Path(tmp) / "raw"
            (out / "nyuv2").mkdir(parents=True)
            with open(out / "nyuv2" / "nyuv2_test.json", "w") as f:
                json.dump(
                    {
                        "nyuv2": [
                            {
                                "test_0000": [
                                    {
                                        "image": "test_0000/rgb/0.png",
                                        "gt_depth": "x",
                                        "factor": 6000.0,
                                    }
                                ]
                            }
                        ]
                    },
                    f,
                )
            stats = C.attach_cameras(out, "nyuv2", "nyuv2/nyuv2_test.json", raw)
            self.assertEqual(
                stats, {"frames": 1, "with_pose": 1, "sequences_without_full_pose": 0}
            )
            fr = json.load(open(out / "nyuv2" / "nyuv2_test.json"))["nyuv2"][0][
                "test_0000"
            ][0]
            self.assertTrue(np.allclose(fr["K"], C.NYU_K))
            self.assertTrue(np.allclose(fr["pose"], np.eye(4)))

            seq = "rgbd_bonn_balloon2"
            (raw / "bonn" / "rgbd_bonn_dataset" / seq).mkdir(parents=True)
            np.savetxt(
                raw / "bonn" / "rgbd_bonn_dataset" / seq / "groundtruth.txt",
                np.array(
                    [[100.0, 0, 0, 0, 0, 0, 0, 1], [100.033, 0, 1, 0, 0, 0, 0, 1]]
                ),
            )
            (out / "bonn").mkdir()
            with open(out / "bonn" / "bonn_video.json", "w") as f:
                json.dump(
                    {
                        "bonn": [
                            {
                                seq: [
                                    {
                                        "image": f"{seq}/rgb/100.0300.png",
                                        "gt_depth": "x",
                                        "factor": 5000.0,
                                    }
                                ]
                            }
                        ]
                    },
                    f,
                )
            stats = C.attach_cameras(out, "bonn", "bonn/bonn_video.json", raw)
            self.assertEqual(stats["with_pose"], 1)
            fr = json.load(open(out / "bonn" / "bonn_video.json"))["bonn"][0][seq][0]
            self.assertAlmostEqual(fr["pose"][1][3], 1.0)

    def test_scannet_nonfinite_pose_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench"
            sc = out / "scannet" / "scene0000_00"
            (sc / "pose").mkdir(parents=True)
            (sc / "intrinsic").mkdir()
            np.savetxt(sc / "intrinsic" / "intrinsic_depth.txt", np.eye(4) * 2)
            np.savetxt(sc / "pose" / "0.txt", np.eye(4))
            bad = np.eye(4)
            bad[0, 3] = -np.inf
            np.savetxt(sc / "pose" / "1.txt", bad)
            with open(out / "scannet" / "scannet_video.json", "w") as f:
                json.dump(
                    {
                        "scannet": [
                            {
                                "scene0000_00": [
                                    {
                                        "image": "scene0000_00/color/0.jpg",
                                        "gt_depth": "x",
                                        "factor": 1000.0,
                                    },
                                    {
                                        "image": "scene0000_00/color/1.jpg",
                                        "gt_depth": "x",
                                        "factor": 1000.0,
                                    },
                                ]
                            }
                        ]
                    },
                    f,
                )
            stats = C.attach_cameras(
                out, "scannet", "scannet/scannet_video.json", Path(tmp)
            )
            self.assertEqual(
                stats, {"frames": 2, "with_pose": 1, "sequences_without_full_pose": 1}
            )
            frames = json.load(open(out / "scannet" / "scannet_video.json"))["scannet"][
                0
            ]["scene0000_00"]
            self.assertIn("pose", frames[0])
            self.assertNotIn("pose", frames[1])
            self.assertEqual(np.asarray(frames[1]["K"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
