import os
import os.path as osp

import cv2
import numpy as np

from .base.base_multiview_dataset import (
    BaseMultiViewDataset,
    EmptyDatasetError,
    intrinsics_rows_to_K,
)
from .base.segments import segment_frame_ids_by_rate
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root

# preserves the original DUSt3R ARKitScenes stride cap; override via the
# constructor or the DatasetConfig CLI rather than editing this constant.
DEFAULT_STRIDE_RANGE = (1, 8)

# Continuity threshold for splitting a scene into runs, on the frame timestamps
# carried in the filenames ("<scene>_<seconds>.png").
#
# The capture rate is a property of the scene, not of the dataset: the current
# processed tree sits at 10 fps (0.1 s steps) and raw vga_wide runs at 30 fps,
# so the threshold is derived per scene from the median step rather than
# hard-coded. GAP_FACTOR admits the jitter in ARKit timestamps (steps of
# 0.033/0.034 s alternate) while rejecting a dropped frame.
#
# MAX_GAP_SECONDS is the ceiling: a scene whose frames are MOSTLY fragments has
# a median step that is itself a gap, and a pure factor rule would then merge
# every fragment into one run. Both are protocol parameters -- they decide
# which clips exist -- so they are stated here rather than inferred silently.
GAP_FACTOR = 1.5
MAX_GAP_SECONDS = 0.5


class ARKitScenes_Multi(BaseMultiViewDataset):
    """ARKitScenes RGB-D video sequences with metric (lowres) depth and
    per-frame trajectories, preprocessed into:
        ROOT/<Training|Test>/<scene>/{vga_wide,lowres_depth}/... +
        new_scene_metadata.npz (and a sibling ROOT_highres/ tree whose scenes
        are excluded here so the high-res variant can own them)."""

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        regular_stride=True,
        is_metric=True,
        highres_root=None,
        **kwargs,
    ):
        if is_metric is not True:
            raise ValueError(f"ARKitScenes is_metric must be True, got {is_metric!r}")
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        # explicit root of the highres sibling tree whose scenes this loader
        # must exclude; None falls back to the original DUSt3R convention of
        # deriving ROOT + "_highres" (silently skipped when absent)
        self.highres_root = highres_root
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        match self.split:
            case Split.TRAIN:
                self.split_dir = "Training"
            case Split.TEST:
                self.split_dir = "Test"
            case _:
                raise ValueError(
                    f"ARKitScenes split must be Split.TRAIN or Split.TEST, "
                    f"got {self.split!r}"
                )

        self.loaded_data = self._load_data(self.split_dir)

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, split, "all_metadata.npz")) as data:
            self.scenes: np.ndarray = data["scenes"]
            high_res_list = np.array([])
            # the highres tree uses Training/Validation (not Training/Test),
            # so resolve its subdir from the Split enum instead of reusing the
            # lowres directory name in `split`
            highres_dir = "Training" if self.split == Split.TRAIN else "Validation"
            if self.highres_root is not None:
                # explicit exclusion root (from the config, which knows the
                # real highres path): must exist -- a typo here would silently
                # double-count the highres scenes in both variants
                highres_split_dir = os.path.join(str(self.highres_root), highres_dir)
                if not os.path.isdir(highres_split_dir):
                    raise FileNotFoundError(
                        f"ARKitScenes highres_root was given explicitly but "
                        f"{highres_split_dir} does not exist"
                    )
                high_res_list = np.array(os.listdir(highres_split_dir))
            else:
                # original DUSt3R convention: sibling tree named ROOT_highres;
                # silently skipped when absent (lowres-only setups)
                highres_split_dir = os.path.join(
                    self.ROOT.rstrip("/") + "_highres",
                    highres_dir,
                )
                if os.path.isdir(highres_split_dir):
                    high_res_list = np.array(os.listdir(highres_split_dir))

            self.scenes = np.setdiff1d(self.scenes, high_res_list)
        # start-id sampling, identical to the other four loaders (ScanNet,
        # HAMMER, ARKitScenesHighRes): idx picks a start frame, and
        # get_seq_from_start_id walks forward from it under the stride policy.
        # The old dust3r layout kept a second `image_collection` sampler here
        # (co-visible groups, permuted) selected by a per-sample coin flip; it
        # emitted out-of-order clips a causal/streaming model never sees at
        # deployment, so it was removed. The `image_collection` metadata is now
        # unused, and scenes are kept on frame count alone -- the same rule the
        # sibling loaders apply.
        offset = 0
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        start_img_ids = []
        seq_img_list = []
        seqids = []
        j = 0
        for scene in self.scenes:
            scene_dir = osp.join(self.ROOT, split, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                num_imgs = imgs.shape[0]
                cut_off = self.min_views()
                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue

                img_ids = list(np.arange(num_imgs) + offset)

                # "<scene>_<seconds>.png" -> the frame's capture time. The
                # processed tree keeps whichever windows the upstream frame
                # selection happened to pick, so a scene is typically several
                # separate runs with seconds of missing capture between them.
                timestamps = np.array(
                    [
                        float(str(name).rsplit("_", 1)[-1][: -len(".png")])
                        for name in imgs
                    ]
                )
                sequences = segment_frame_ids_by_rate(
                    img_ids, timestamps, GAP_FACTOR, MAX_GAP_SECONDS, cut_off
                )
                if not sequences:
                    print(f"Skipping {scene}: no run of {cut_off} consecutive frames")
                    continue

                for img_ids_seq in sequences:
                    seq_img_list.append(img_ids_seq)
                    start_img_ids.extend(img_ids_seq[: len(img_ids_seq) - cut_off + 1])
                    # seqids is indexed by GLOBAL image id
                    for gid in img_ids_seq:
                        seqids.append((gid, len(seq_img_list) - 1))

                scenes.append(scene)
                sceneids.extend([j] * num_imgs)
                images.extend(imgs)
                intrinsics.extend(list(intrinsics_rows_to_K(intrins)))
                trajectories.extend(list(traj))

                offset += num_imgs
                j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"ARKitScenes found no usable scenes under {osp.join(self.ROOT, split)}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.intrinsics = intrinsics
        self.trajectories = trajectories
        # one entry per contiguous run, NOT per scene: indexed by
        # seqids[start_id], never by sceneids[start_id] (the scene-path lookup)
        self.seq_img_list = seq_img_list
        self.seqids = np.full(offset, -1, dtype=np.int64)
        for gid, seq_idx in seqids:
            self.seqids[gid] = seq_idx
        self.start_img_ids = start_img_ids

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.seq_img_list[self.seqids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,  # stride/order policy: self.stride_range + base defaults
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            # frames live either in the scene dir (extracted layout) or in
            # its frames.zip (inode-safe layout); metadata npz reads above
            # are unaffected (always real files in the scene dir)
            scene_dir = frames_root(
                osp.join(self.ROOT, self.split_dir, self.scenes[scene_id])
            )

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]
            if basename[:8] != self.scenes[scene_id]:
                raise RuntimeError(
                    f"ARKitScenes frame/scene mismatch: basename {basename!r} "
                    f"does not belong to scene {self.scenes[scene_id]!r}"
                )
            # Load RGB image
            rgb_image = imread_cv2(
                osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
            )
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "lowres_depth", basename), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="arkitscenes",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        if len(views) != num_views:
            raise RuntimeError(
                f"ARKitScenes produced {len(views)} views but {num_views} were requested"
            )
        return views
