"""Convert raw TartanAir into the processed per-trajectory layout.

Reads either layout:

  * zip (default, what download_tartanair.sh produces): the downloaded
    archives sit flat in --tartanair_dir, named <env>_<difficulty>_<asset>.zip
    with members <env>/<difficulty>/<traj>/<asset_dir>/<file>. They are never
    unzipped -- 144 inodes instead of the millions the extracted tree costs --
    and members are read by random access straight out of the archives.
  * extracted: a plain <root>/<env>/<difficulty>/<traj>/... tree, as produced
    by unzipping them by hand.

Writes one uncompressed frames.zip per trajectory by default (again one inode
instead of ~5 per frame); --extracted writes the original loose files. The
per-frame payloads are byte-identical between the two: rgb/depth/flow/mask are
copied verbatim out of the source and the cam npz is built the same way, so
the layouts are interchangeable for the loader.
"""

import argparse
import io
import os
import os.path as osp
import sys
import zipfile

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# same sibling-import workaround preprocess_arkitscenes.py uses
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from dust3r.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    listdir as zlistdir,
    read_bytes,
)

# subdirectory each asset's frames live under, inside its archive/tree
ASSET_DIRS = {
    "image_left": "image_left",
    "depth_left": "depth_left",
    "flow_flow": "flow",
    "flow_mask": "flow",
}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tartanair_dir",
        default="data/tartanair",
    )
    parser.add_argument(
        "--output_dir",
        default="data/mast3r_data/processed_tartanair",
    )
    parser.add_argument(
        "--extracted",
        action="store_true",
        help="write loose per-frame files instead of one frames.zip per "
        "trajectory (the original layout; ~5 inodes per frame)",
    )
    return parser


class RawSource:
    """Resolve raw member paths for one (env, difficulty), in either layout.

    Zip layout: each asset lives in its own archive, so the asset a file
    belongs to determines which archive to open -- an explicit mapping, no
    searching. Extracted layout: everything is under one directory tree.
    """

    def __init__(self, root, env, difficulty, zip_stem=None):
        self.root, self.env, self.difficulty = root, env, difficulty
        self.zip_stem = zip_stem  # None => extracted layout

    def _path(self, asset, traj, relname):
        sub = ASSET_DIRS[asset]
        if self.zip_stem is None:
            return osp.join(self.root, self.env, self.difficulty, traj, sub, relname)
        archive = osp.join(self.root, f"{self.zip_stem}_{asset}.zip")
        member = f"{self.env}/{self.difficulty}/{traj}/{sub}/{relname}"
        return f"{archive}/{member}"

    def rgb(self, traj, i):
        return self._path("image_left", traj, f"{i:06d}_left.png")

    def depth(self, traj, i):
        return self._path("depth_left", traj, f"{i:06d}_left_depth.npy")

    def flow(self, traj, i):
        return self._path("flow_flow", traj, f"{i:06d}_{i + 1:06d}_flow.npy")

    def mask(self, traj, i):
        return self._path("flow_mask", traj, f"{i:06d}_{i + 1:06d}_mask.npy")

    def pose_txt(self, traj):
        # pose_left.txt sits at the trajectory root, not under an asset dir,
        # and ships inside the image_left archive
        if self.zip_stem is None:
            return osp.join(self.root, self.env, self.difficulty, traj, "pose_left.txt")
        archive = osp.join(self.root, f"{self.zip_stem}_image_left.zip")
        return f"{archive}/{self.env}/{self.difficulty}/{traj}/pose_left.txt"

    def count(self, asset, traj):
        """Number of frame files for one asset -- the assert inputs.

        Goes through zipio.listdir so it shares the cached archive handle: the
        assert calls this four times per trajectory and there are ~1000
        trajectories, so opening the archive here would mean thousands of full
        central-directory parses of multi-GB zips over Lustre."""
        sub = ASSET_DIRS[asset]
        if self.zip_stem is None:
            return len(
                os.listdir(osp.join(self.root, self.env, self.difficulty, traj, sub))
            )
        archive = osp.join(self.root, f"{self.zip_stem}_{asset}.zip")
        return len(zlistdir(f"{archive}/{self.env}/{self.difficulty}/{traj}/{sub}"))

    def flow_count(self, traj):
        """Every file in the trajectory's flow directory -- flow and mask
        together, which is what the original counted with a single listdir.

        The two layouts need different arithmetic: extracted keeps flow and
        mask side by side in one flow/ directory (count it once, or they get
        counted twice), while the zip layout splits them across flow_flow.zip
        and flow_mask.zip (sum the two archives to get the same total)."""
        if self.zip_stem is None:
            return len(
                os.listdir(osp.join(self.root, self.env, self.difficulty, traj, "flow"))
            )
        return self.count("flow_flow", traj) + self.count("flow_mask", traj)


def discover(root):
    """Yield (RawSource, [trajectories]) for every env/difficulty found.

    Zip layout is detected by the presence of *_image_left.zip; the env and
    difficulty are read from the archive's member paths rather than parsed out
    of the filename, because env names themselves contain underscores
    (abandonedfactory_night).
    """
    image_zips = sorted(f for f in os.listdir(root) if f.endswith("_image_left.zip"))
    if image_zips:
        for fname in image_zips:
            stem = fname[: -len("_image_left.zip")]
            with zipfile.ZipFile(osp.join(root, fname)) as zf:
                names = zf.namelist()
            env = difficulty = None
            trajs = set()
            for n in names:
                parts = n.strip("/").split("/")
                if len(parts) >= 3:
                    env, difficulty = parts[0], parts[1]
                    trajs.add(parts[2])
            if env is None:
                continue
            yield RawSource(root, env, difficulty, stem), sorted(trajs)
        return

    envs = [f for f in sorted(os.listdir(root)) if osp.isdir(osp.join(root, f))]
    for env in envs:
        for difficulty in ["Easy", "Hard"]:
            d = osp.join(root, env, difficulty)
            if not osp.isdir(d):
                continue
            trajs = sorted(f for f in os.listdir(d) if osp.isdir(osp.join(d, f)))
            yield RawSource(root, env, difficulty), trajs


def frame_count(src, traj):
    """Number of frames in a trajectory, from its pose file (a few KB)."""
    return len(np.loadtxt(io.StringIO(read_bytes(src.pose_txt(traj)).decode())))


def convert_trajectory(src, traj, emit):
    """Convert one trajectory, handing each output to emit(name, bytes)."""
    intrinsics = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
    ).astype(np.float32)
    poses = np.loadtxt(io.StringIO(read_bytes(src.pose_txt(traj)).decode()))
    frame_num = len(poses)
    assert (
        src.count("image_left", traj)
        == src.count("depth_left", traj)
        == src.flow_count(traj) // 2 + 1
        == frame_num
    )
    for i in tqdm(range(frame_num), leave=False):
        pose = poses[i]
        x, y, z, qx, qy, qz, qw = pose
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = [x, y, z]
        w2c = np.linalg.inv(c2w)
        w2c = w2c[[1, 2, 0, 3]]
        c2w = np.linalg.inv(w2c)
        K = intrinsics

        # rgb/depth/flow/mask are copied verbatim (the original used
        # shutil.copy), so the processed bytes equal the raw bytes
        emit(f"{i:06d}_rgb.png", read_bytes(src.rgb(traj, i)))
        emit(f"{i:06d}_depth.npy", read_bytes(src.depth(traj, i)))
        if i < frame_num - 1:
            emit(f"{i:06d}_flow.npy", read_bytes(src.flow(traj, i)))
            emit(f"{i:06d}_mask.npy", read_bytes(src.mask(traj, i)))

        cam_buf = io.BytesIO()
        np.savez(
            cam_buf,
            camera_pose=c2w.astype(np.float32),
            camera_intrinsics=K.astype(np.float32),
        )
        emit(f"{i:06d}_cam.npz", cam_buf.getvalue())


def main(rootdir, outdir, as_zip=True):
    os.makedirs(outdir, exist_ok=True)
    for src, trajs in discover(rootdir):
        for traj in tqdm(trajs, desc=f"{src.env}/{src.difficulty}"):
            out_traj_dir = osp.join(outdir, src.env, src.difficulty, traj)
            os.makedirs(out_traj_dir, exist_ok=True)
            if as_zip:
                zip_path = osp.join(out_traj_dir, "frames.zip")
                if osp.isfile(zip_path):
                    continue  # complete (writer renames only on success)
                with SceneZipWriter(zip_path) as writer:
                    convert_trajectory(src, traj, writer.writestr)
            else:
                # mirror the zip branch's resume guard without writing a marker
                # file: that would be an extra output the zip layout does not
                # have, breaking byte-parity between the two. convert_trajectory
                # writes each frame's cam npz last, so the final frame's cam npz
                # existing means the trajectory finished.
                n = frame_count(src, traj)
                if osp.isfile(osp.join(out_traj_dir, f"{n - 1:06d}_cam.npz")):
                    continue

                def emit(name, data, _d=out_traj_dir):
                    with open(osp.join(_d, name), "wb") as f:
                        f.write(data)

                convert_trajectory(src, traj, emit)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.tartanair_dir, args.output_dir, as_zip=not args.extracted)
