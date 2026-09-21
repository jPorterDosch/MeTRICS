import io
import os
import json
import os.path as osp
import decimal
import argparse
import math
import sys
from bisect import bisect_left
from PIL import Image
import numpy as np
import quaternion
from scipy import interpolate
import cv2
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))
from streamvggt.datasets.utils.zipio import (  # noqa: E402
    SceneZipWriter,
    asset_root,
    listdir as zlistdir,
    read_bytes,
)
from streamvggt.datasets.base.segments import split_by_rate  # noqa: E402


# Frame selection. ARKitScenes is captured as video -- vga_wide runs at 30 fps
# and lowres_depth at 60 fps, over the same span -- and every vga_wide basename
# has an exact lowres_depth twin (100% on every scene sampled), so the two
# assets need no timestamp matching: their shared basenames ARE the stream.
#
# The previous selection came from DUSt3R's selected_pairs.npz, which kept
# whichever frames its covisibility sampler liked. That left a median of 8
# disconnected fragments per scene with a median length of 11 frames, so
# sampling a 32-frame clip from a scene produced a clip spanning seconds-long
# breaks in the capture. Here a scene contributes ONE contiguous run instead.
#
# TARGET_FPS is the rate the run is resampled to and MAX_FRAMES_PER_SCENE caps
# its length. Both are protocol parameters: they set what a clip of N frames
# spans, hence every temporal metric computed over one.
#
# 30 fps is the capture's own rate, and the store is kept at it because stride
# sampling can subsample but cannot invent frames: at 30 fps a clip at stride
# 1/2/3 is 30/15/10 fps, where a 10 fps store could only ever be 10. It also
# matches ScanNet's native rate, which is the rate DepthCrafter and Video Depth
# Anything evaluate video depth at, so temporal numbers are comparable.
#
# The 400-frame cap is what the quota allows, not a judgement: scenes hold a
# median 1,974 vga_wide frames (66 s), and at ~118 KB per frame the Training
# split costs ~218 GB at this cap against ~633 GB at 1,200 frames and ~1.2 TB
# uncapped. The cost is 13.3 s of video per scene rather than the full 66.
TARGET_FPS = 30.0
MAX_FRAMES_PER_SCENE = 400
# A run must be long enough to be worth keeping at all: 32 frames is the clip
# length used for evaluation, so anything shorter can never be evaluated.
MIN_RUN_FRAMES = 32
# Continuity threshold on the RAW stream, as a multiple of a run's own median
# step. vga_wide timestamps alternate 0.033/0.034 s, so a pure equality test
# would split every other frame. RAW_MAX_GAP_SECONDS caps it, which also sets
# the slowest capture still treated as video (see segments.max_gap_from_rate);
# the raw stream is 30 fps, so 0.5 s is far above anything a real capture hits.
RAW_GAP_FACTOR = 1.5
RAW_MAX_GAP_SECONDS = 0.5


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arkitscenes_dir",
        default=os.path.expanduser("~/scratch/data/dust3r_data/data_arkitscenes/raw"),
    )
    parser.add_argument(
        "--precomputed_pairs",
        default=os.path.expanduser(
            "~/scratch/data/dust3r_data/data_arkitscenes/arkitscenes_pairs"
        ),
        help="DUSt3R's arkitscenes_pairs dir. Only its per-split "
        "scene_list.json is read, for the Training/Test assignment -- keeping "
        "it means a rebuild does not silently move scenes between splits. The "
        "per-scene selected_pairs.npz is no longer used: frames are chosen "
        "from the raw capture (see select_video_run).",
    )
    parser.add_argument(
        "--output_dir",
        default="~/scratch/data/dust3r_data/processed_arkitscenes",
    )
    return parser


def value_to_decimal(value, decimal_places):
    decimal.getcontext().rounding = decimal.ROUND_HALF_UP  # define rounding method
    return decimal.Decimal(str(float(value))).quantize(
        decimal.Decimal("1e-{}".format(decimal_places))
    )


def closest(value, sorted_list):
    index = bisect_left(sorted_list, value)
    if index == 0:
        return sorted_list[0]
    elif index == len(sorted_list):
        return sorted_list[-1]
    else:
        value_before = sorted_list[index - 1]
        value_after = sorted_list[index]
        if value_after - value < value - value_before:
            return value_after
        else:
            return value_before


def frame_timestamp(basename):
    """Capture time in seconds from a "<scene>_<seconds>.png" frame name."""
    return float(basename.rsplit("_", 1)[-1][: -len(".png")])


def select_video_run(rgb_dir, depth_dir, traj_timestamps):
    """Pick one contiguous run of frames from a scene's raw assets.

    A frame is usable only if it has RGB, depth and a pose: vga_wide and
    lowres_depth share basenames exactly, and the pose comes from interpolating
    lowres_wide.traj, which is defined only between its first and last
    timestamp. The longest surviving run is then resampled to TARGET_FPS and
    truncated to MAX_FRAMES_PER_SCENE.

    args:
        rgb_dir: resolved vga_wide asset root.
        depth_dir: resolved lowres_depth asset root.
        traj_timestamps: ascending pose timestamps from lowres_wide.traj.

    returns:
        list of frame basenames ("<scene>_<seconds>.png"), ascending in time,
        or [] when the scene has no run of at least MIN_RUN_FRAMES.
    """
    shared = set(zlistdir(rgb_dir)) & set(zlistdir(depth_dir))
    names = sorted(
        (name for name in shared if name.endswith(".png")), key=frame_timestamp
    )
    if len(names) < MIN_RUN_FRAMES:
        return []

    timestamps = np.array([frame_timestamp(name) for name in names])
    # interp1d refuses to extrapolate, so frames outside the trajectory span
    # cannot be posed at all (~2% of a scene, at its head and tail)
    in_range = (timestamps >= traj_timestamps[0]) & (timestamps <= traj_timestamps[-1])
    names = [name for name, keep in zip(names, in_range) if keep]
    timestamps = timestamps[in_range]
    if len(names) < MIN_RUN_FRAMES:
        return []

    # the same rule the loader applies (streamvggt/datasets/base/segments.py):
    # a threshold derived per run from its OWN rate, refined until no run holds
    # a step above it, and capped so that a scene which is mostly fragments
    # cannot have its median step -- itself a gap -- merge every fragment into
    # one run. A median-only rule here is what produced the fragmented
    # selection this function replaces.
    runs = split_by_rate(timestamps, RAW_GAP_FACTOR, RAW_MAX_GAP_SECONDS)
    longest = max(runs, key=len)
    if len(longest) < MIN_RUN_FRAMES:
        return []

    # resample by taking every stride-th frame: the run stays contiguous in
    # time, just at a lower rate
    native_step = float(np.median(np.diff(timestamps[longest])))
    stride = max(1, int(round((1.0 / TARGET_FPS) / native_step)))
    kept = longest[::stride][:MAX_FRAMES_PER_SCENE]
    if len(kept) < MIN_RUN_FRAMES:
        return []
    return [names[i] for i in kept]


def get_up_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[0.0], [-1.0], [0.0], [0.0]]))


def get_right_vectors(pose_device_to_world):
    return np.matmul(pose_device_to_world, np.array([[1.0], [0.0], [0.0], [0.0]]))


def read_traj(traj_path):
    quaternions = []
    poses = []
    timestamps = []
    poses_p_to_w = []
    with open(traj_path) as f:
        traj_lines = f.readlines()
        for line in traj_lines:
            tokens = line.split()
            if len(tokens) != 7:
                raise ValueError(
                    f"{traj_path}: expected 7 fields per pose line "
                    f"(timestamp + 3 angle-axis + 3 translation), got "
                    f"{len(tokens)}: {line[:120]!r}"
                )
            traj_timestamp = float(tokens[0])

            timestamps_decimal_value = value_to_decimal(traj_timestamp, 3)
            timestamps.append(
                float(timestamps_decimal_value)
            )  # for spline interpolation

            angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
            r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))
            t_w_to_p = np.asarray(
                [float(tokens[4]), float(tokens[5]), float(tokens[6])]
            )

            pose_w_to_p = np.eye(4)
            pose_w_to_p[:3, :3] = r_w_to_p
            pose_w_to_p[:3, 3] = t_w_to_p

            pose_p_to_w = np.linalg.inv(pose_w_to_p)

            r_p_to_w_as_quat = quaternion.from_rotation_matrix(pose_p_to_w[:3, :3])
            t_p_to_w = pose_p_to_w[:3, 3]
            poses_p_to_w.append(pose_p_to_w)
            poses.append(t_p_to_w)
            quaternions.append(r_p_to_w_as_quat)
    return timestamps, poses, quaternions, poses_p_to_w


def main(rootdir, pairsdir, outdir):
    failed_total = []
    os.makedirs(outdir, exist_ok=True)

    subdirs = ["Test", "Training"]
    for subdir in subdirs:
        # STEP 1: list all scenes
        outsubdir = osp.join(outdir, subdir)
        os.makedirs(outsubdir, exist_ok=True)
        listfile = osp.join(pairsdir, subdir, "scene_list.json")
        with open(listfile, "r") as f:
            scene_dirs = json.load(f)

        valid_scenes = []
        failed = []
        for scene_subdir in tqdm(scene_dirs):
            if not os.path.isdir(osp.join(rootdir, "Test", scene_subdir)):
                if not os.path.isdir(osp.join(rootdir, "Training", scene_subdir)):
                    continue
                else:
                    root_subdir = "Training"
            else:
                root_subdir = "Test"
            out_scene_subdir = osp.join(outsubdir, scene_subdir)
            # the scene dir is created only once the scene is known to be
            # usable: creating it up front left 989 empty Training dirs behind
            # for scenes the old selection rejected

            scene_dir = osp.join(rootdir, root_subdir, scene_subdir)
            # asset_root resolves each raw asset for both layouts: an
            # extracted <asset>/ directory, or the downloaded <asset>.zip
            # left unextracted (inode-safe layout)
            depth_dir = asset_root(scene_dir, "lowres_depth")
            rgb_dir = asset_root(scene_dir, "vga_wide")
            intrinsics_dir = asset_root(scene_dir, "vga_wide_intrinsics")
            traj_path = osp.join(scene_dir, "lowres_wide.traj")
            if (
                depth_dir is None
                or rgb_dir is None
                or intrinsics_dir is None
                or not osp.isfile(traj_path)
            ):
                continue

            # STEP 2: a scene already converted is valid by construction (the
            # npz is written last, after frames.zip is renamed into place)
            scene_metadata_path = osp.join(out_scene_subdir, "scene_metadata.npz")
            if osp.isfile(scene_metadata_path):
                valid_scenes.append(scene_subdir)
                continue

            print(f"parsing {scene_subdir}")
            try:
                written = convert_scene(
                    scene_subdir,
                    rgb_dir,
                    depth_dir,
                    intrinsics_dir,
                    traj_path,
                    out_scene_subdir,
                    scene_metadata_path,
                )
            except Exception as e:
                # One bad scene costs itself, not the pass. This is a single
                # ~46 h process, and the resume check skips only scenes that
                # already wrote scene_metadata.npz -- so a scene that raises
                # would otherwise kill the run AND kill every retry at the
                # same place. Failures are counted and the exit status is
                # non-zero, so a lossy pass cannot look successful.
                failed.append(scene_subdir)
                print(f"FAILED {scene_subdir}: {e}", file=sys.stderr, flush=True)
                continue
            if written:
                valid_scenes.append(scene_subdir)

        if failed:
            failed_total.extend(failed)
            print(
                f"{len(failed)} scene(s) failed in {subdir}: "
                + ", ".join(failed[:10])
                + (", ..." if len(failed) > 10 else ""),
                file=sys.stderr,
            )
        outlistfile = osp.join(outsubdir, "scene_list.json")
        # (filter with a comprehension: the upstream remove-while-iterating
        # skipped the element after each removal, letting npz-less scenes
        # survive into the STEP-5 concat and crash it)
        valid_scenes = [
            scene_subdir
            for scene_subdir in valid_scenes
            if osp.isfile(osp.join(outsubdir, scene_subdir, "scene_metadata.npz"))
        ]
        with open(outlistfile, "w") as f:
            json.dump(valid_scenes, f)

        if not valid_scenes:
            print(f"No valid scenes for {subdir}; skipping all_metadata.npz")
            continue

        # STEP 5: concat all scene_metadata.npz into a single file
        scene_data = {}
        for scene_subdir in valid_scenes:
            scene_metadata_path = osp.join(
                outsubdir, scene_subdir, "scene_metadata.npz"
            )
            with np.load(scene_metadata_path) as data:
                trajectories = data["trajectories"]
                intrinsics = data["intrinsics"]
                images = data["images"]
                pairs = data["pairs"]
            scene_data[scene_subdir] = {
                "trajectories": trajectories,
                "intrinsics": intrinsics,
                "images": images,
                "pairs": pairs,
            }
        offset = 0
        counts = []
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        pairs = []
        for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
            num_imgs = data["images"].shape[0]
            img_pairs = data["pairs"]

            scenes.append(scene_subdir)
            sceneids.extend([scene_idx] * num_imgs)

            images.append(data["images"])

            K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
            K[:, 0, 0] = [fx for _, _, fx, _, _, _ in data["intrinsics"]]
            K[:, 1, 1] = [fy for _, _, _, fy, _, _ in data["intrinsics"]]
            K[:, 0, 2] = [hw for _, _, _, _, hw, _ in data["intrinsics"]]
            K[:, 1, 2] = [hh for _, _, _, _, _, hh in data["intrinsics"]]

            intrinsics.append(K)
            trajectories.append(data["trajectories"])

            # offset pairs
            img_pairs[:, 0:2] += offset
            pairs.append(img_pairs)
            counts.append(offset)

            offset += num_imgs

        images = np.concatenate(images, axis=0)
        intrinsics = np.concatenate(intrinsics, axis=0)
        trajectories = np.concatenate(trajectories, axis=0)
        pairs = np.concatenate(pairs, axis=0)
        np.savez(
            osp.join(outsubdir, "all_metadata.npz"),
            counts=counts,
            scenes=scenes,
            sceneids=sceneids,
            images=images,
            intrinsics=intrinsics,
            trajectories=trajectories,
            pairs=pairs,
        )

    # Exit non-zero so Slurm marks the job FAILED, the same contract
    # preprocess_scannetpp.py uses. The splits above are already written, so a
    # rerun picks up only the scenes that failed; printing a count and exiting
    # 0 would leave a lossy pass looking successful.
    if failed_total:
        raise SystemExit(
            f"{len(failed_total)} scene(s) failed to convert; rerun to retry "
            "them (converted scenes are skipped)"
        )


def convert_scene_metadata(
    scene_subdir,
    intrinsics_dir,
    timestamps,
    quaternions,
    poses,
    poses_cam_to_world,
    selected_images,
    timestamps_selected,
):
    # find scene orientation
    sky_direction_scene, rotated_to_cam = find_scene_orientation(poses_cam_to_world)

    # find/compute pose for selected timestamps
    # most images have a valid timestamp / exact pose associated
    timestamps_selected = np.array(timestamps_selected)
    spline = interpolate.interp1d(timestamps, poses, kind="linear", axis=0)
    interpolated_rotations = quaternion.squad(
        quaternions, timestamps, timestamps_selected
    )
    interpolated_positions = spline(timestamps_selected)

    trajectories = []
    intrinsics = []
    images = []
    # membership in the listing replaces per-probe existence checks: same
    # semantics in both layouts, and O(1) per probe instead of a stat / a
    # namelist scan
    intrinsic_names = set(zlistdir(intrinsics_dir))
    missing_intrinsics = 0
    for i, (basename, frame_id) in enumerate(selected_images):
        intrinsic_name = f"{scene_subdir}_{frame_id}.pincam"
        if intrinsic_name not in intrinsic_names:
            intrinsic_name = f"{scene_subdir}_{float(frame_id) - 0.001:.3f}.pincam"
        if intrinsic_name not in intrinsic_names:
            intrinsic_name = f"{scene_subdir}_{float(frame_id) + 0.001:.3f}.pincam"
        if intrinsic_name not in intrinsic_names:
            # Skip the frame, as the high-res script does. Frames now come from
            # the raw vga_wide listing rather than DUSt3R's curated selection,
            # so a frame with no .pincam is reachable; raising would kill a
            # ~46 h single-process pass at that scene, and since the resume
            # check skips only scenes that already wrote scene_metadata.npz,
            # every retry would die in the same place. A scene that loses ALL
            # its frames this way is caught after the loop.
            missing_intrinsics += 1
            continue
        w, h, fx, fy, hw, hh = np.loadtxt(
            io.BytesIO(read_bytes(osp.join(intrinsics_dir, intrinsic_name)))
        )  # PINHOLE

        pose = np.eye(4)
        pose[:3, :3] = quaternion.as_rotation_matrix(interpolated_rotations[i])
        pose[:3, 3] = interpolated_positions[i]

        images.append(basename)
        if sky_direction_scene == "RIGHT" or sky_direction_scene == "LEFT":
            intrinsics.append([h, w, fy, fx, hh, hw])  # swapped intrinsics
        else:
            intrinsics.append([w, h, fx, fy, hw, hh])
        trajectories.append(
            pose @ rotated_to_cam
        )  # pose_cam_to_world @ rotated_to_cam = rotated(cam) to world

    if missing_intrinsics:
        print(
            f"{scene_subdir}: {missing_intrinsics} of {len(selected_images)} "
            "frames had no .pincam within 1 ms and were skipped"
        )
    if selected_images and not images:
        raise FileNotFoundError(
            f"{scene_subdir}: none of the {len(selected_images)} selected "
            "frames has a .pincam in vga_wide_intrinsics. The intrinsics are "
            "named on a different clock than vga_wide here, which a per-frame "
            "skip would quietly turn into an empty scene."
        )
    return sky_direction_scene, trajectories, intrinsics, images


def find_scene_orientation(poses_cam_to_world):
    if len(poses_cam_to_world) > 0:
        up_vector = sum(get_up_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        right_vector = sum(get_right_vectors(p) for p in poses_cam_to_world) / len(
            poses_cam_to_world
        )
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])
    else:
        up_vector = np.array([[0.0], [-1.0], [0.0], [0.0]])
        right_vector = np.array([[1.0], [0.0], [0.0], [0.0]])
        up_world = np.array([[0.0], [0.0], [1.0], [0.0]])

    # value between 0, 180
    device_up_to_world_up_angle = (
        np.arccos(np.clip(np.dot(np.transpose(up_world), up_vector), -1.0, 1.0)).item()
        * 180.0
        / np.pi
    )
    device_right_to_world_up_angle = (
        np.arccos(
            np.clip(np.dot(np.transpose(up_world), right_vector), -1.0, 1.0)
        ).item()
        * 180.0
        / np.pi
    )

    up_closest_to_90 = abs(device_up_to_world_up_angle - 90.0) < abs(
        device_right_to_world_up_angle - 90.0
    )
    if up_closest_to_90:
        if abs(device_up_to_world_up_angle - 90.0) >= 45.0:
            raise ValueError(
                f"device up vector is {device_up_to_world_up_angle:.1f} deg "
                "from world up; expected within 45 deg of 90"
            )
        # LEFT
        if device_right_to_world_up_angle > 90.0:
            sky_direction_scene = "LEFT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, math.pi / 2.0]
            )
        else:
            # note that in metadata.csv RIGHT does not exist, but again it's not accurate...
            # well, turns out there are scenes oriented like this
            # for example Training/41124801
            sky_direction_scene = "RIGHT"
            cam_to_rotated_q = quaternion.from_rotation_vector(
                [0.0, 0.0, -math.pi / 2.0]
            )
    else:
        # right is close to 90
        if abs(device_right_to_world_up_angle - 90.0) >= 45.0:
            raise ValueError(
                f"device right vector is {device_right_to_world_up_angle:.1f} "
                "deg from world up; expected within 45 deg of 90"
            )
        if device_up_to_world_up_angle > 90.0:
            sky_direction_scene = "DOWN"
            cam_to_rotated_q = quaternion.from_rotation_vector([0.0, 0.0, math.pi])
        else:
            sky_direction_scene = "UP"
            cam_to_rotated_q = quaternion.quaternion(1, 0, 0, 0)
    cam_to_rotated = np.eye(4)
    cam_to_rotated[:3, :3] = quaternion.as_rotation_matrix(cam_to_rotated_q)
    rotated_to_cam = np.linalg.inv(cam_to_rotated)
    return sky_direction_scene, rotated_to_cam


def convert_scene(
    scene_subdir,
    rgb_dir,
    depth_dir,
    intrinsics_dir,
    traj_path,
    out_scene_subdir,
    scene_metadata_path,
):
    """Convert one scene: pick its run, write frames.zip, then the metadata.

    returns True when the scene was written, False when it holds no run long
    enough to keep. Raises for anything else, which main() isolates per scene.
    """
    # loads traj
    timestamps, poses, quaternions, poses_cam_to_world = read_traj(traj_path)

    poses = np.array(poses)
    quaternions = np.array(quaternions, dtype=np.quaternion)
    quaternions = quaternion.unflip_rotors(quaternions)
    timestamps = np.array(timestamps)

    # STEP 3: pick ONE contiguous run of the capture. The old selection
    # came from DUSt3R's selected_pairs.npz, whose covisibility sampler
    # returned scattered windows; `pairs` went unused by the streamvggt
    # loaders, and the scenes it rejected outright (988 of them, for an
    # empty pair list) are usable video.
    selection = select_video_run(rgb_dir, depth_dir, timestamps)
    if not selection:
        return False
    os.makedirs(out_scene_subdir, exist_ok=True)

    selected_images = [
        (basename, basename.split(".png")[0].split("_")[1]) for basename in selection
    ]
    timestamps_selected = [float(frame_id) for _, frame_id in selected_images]

    sky_direction_scene, trajectories, intrinsics, images = convert_scene_metadata(
        scene_subdir,
        intrinsics_dir,
        timestamps,
        quaternions,
        poses,
        poses_cam_to_world,
        selected_images,
        timestamps_selected,
    )
    if not isinstance(sky_direction_scene, str):
        raise TypeError(
            f"{scene_subdir}: sky_direction_scene must be a str, got "
            f"{type(sky_direction_scene).__name__}"
        )
    # every selected frame exists in both assets by construction
    # (select_video_run intersects the two listings), so the old
    # membership re-check is gone

    # all converted frames go into ONE uncompressed zip per scene
    # (inode-safe layout); the write is atomic (.tmp -> rename)
    # and the npz below is saved only after the zip completes, so
    # the skip-on-npz check above implies a complete frames.zip
    with SceneZipWriter(osp.join(out_scene_subdir, "frames.zip")) as writer:
        for basename in images:
            img = Image.open(io.BytesIO(read_bytes(osp.join(rgb_dir, basename))))
            depth = cv2.imdecode(
                np.frombuffer(read_bytes(osp.join(depth_dir, basename)), np.uint8),
                cv2.IMREAD_UNCHANGED,
            )

            # rotate the image
            if sky_direction_scene == "RIGHT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_90)
                except Exception:
                    img = img.transpose(Image.ROTATE_90)
                depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif sky_direction_scene == "LEFT":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_270)
                except Exception:
                    img = img.transpose(Image.ROTATE_270)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            elif sky_direction_scene == "DOWN":
                try:
                    img = img.transpose(Image.Transpose.ROTATE_180)
                except Exception:
                    img = img.transpose(Image.ROTATE_180)
                depth = cv2.rotate(depth, cv2.ROTATE_180)

            W, H = img.size
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            writer.writestr(
                "vga_wide/" + basename.replace(".png", ".jpg"),
                buf.getvalue(),
            )

            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
            ok, enc = cv2.imencode(".png", depth)
            if not ok:
                raise RuntimeError(f"png encode failed for {basename}")
            writer.writestr("lowres_depth/" + basename, enc.tobytes())

    # save at the end. `pairs` is written empty: the covisibility pairs
    # were DUSt3R's frame-selection input, nothing downstream reads
    # them (generate_set_arkitscenes builds an image_collection the
    # streamvggt loaders never touch), and they cannot be re-indexed
    # against a selection they did not produce.
    np.savez(
        scene_metadata_path,
        trajectories=trajectories,
        intrinsics=intrinsics,
        images=images,
        pairs=np.zeros((0, 3), dtype=np.float64),
    )
    return True


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.arkitscenes_dir, args.precomputed_pairs, args.output_dir)
