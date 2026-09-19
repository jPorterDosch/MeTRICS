import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from .base.base_multiview_dataset import BaseMultiViewDataset, EmptyDatasetError
from .base.segments import segment_frame_ids_by_rate
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root, listdir as zlistdir

# preserves the original DUSt3R ScanNet++ stride cap (max_interval=3); override
# via the constructor or the DatasetConfig CLI rather than editing this
# constant.
#
# NOTE what a stride now BUYS here. The cap was set when a scene held ~143
# frames, every 50th of the 60 fps capture (~1.2 fps), so strides 1-3 spanned
# 0.8-2.5 s. preprocess_scannetpp.py now keeps every registered frame (~637
# per scene, every 10th, ~6 fps), so the same strides span 0.17-0.5 s and a
# clip covers ~5x less camera motion than it used to. The number is unchanged
# because it is the inherited default and the training script pins
# stride_range to (1, 1) anyway; raising it is how to recover the old
# parallax, and that is a training decision, not a loader one.
DEFAULT_STRIDE_RANGE = (1, 3)

# Continuity threshold for splitting a scene's iPhone run, in video frame
# numbers. The kept-frame step is set by preprocessing (see _load_data), so the
# threshold scales with the median step; MAX_FRAME_GAP caps it for a scene
# whose kept frames are mostly fragments. Both decide which clips exist, so
# they are protocol parameters.
GAP_FACTOR = 1.5
MAX_FRAME_GAP = 120


class ScanNetpp_Multi(BaseMultiViewDataset):
    """ScanNet++ iPhone frames with metric depth ray-cast from the scene mesh,
    preprocessed into:
        ROOT/<scene>/{images,depth}/... + new_scene_metadata.npz,
        plus ROOT/all_metadata.npz listing the scenes.

    Sampling differs from the DUSt3R loader in two ways, both deliberate:

    1. Start-id sampling only. The DUSt3R version chose per sample between a
       video walk and an `image_collection` co-visibility group, and permuted
       the latter 25% of the time -- out-of-order clips a causal/streaming model
       never sees at deployment. The ARKitScenes port dropped that sampler for
       the same reason and this follows it, so `image_collection` is unused
       here and scenes are kept on frame count alone.

    2. iPhone frames only, split into contiguous runs. The DSLR run is a set of
       stills (a median 20 cm of camera motion between consecutive DSC numbers,
       up to 9.7 m), so a clip drawn from it is not a video at any stride and
       its temporal metrics say nothing; it is dropped rather than sampled. The
       DUSt3R loader instead paired the two cameras and asserted
       `max(dslr_ids) < min(iphone_ids)`, which raised on an empty list.

    Images listed in the metadata but absent from disk (registered in the
    release's COLMAP reconstruction but never rendered -- 1077 of 64923 in the
    current release, carrying NaN poses) are filtered out of the sampling lists
    while their metadata rows stay in place, so intrinsics/trajectories indexing
    is unaffected and a NaN pose can never be drawn.
    """

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        regular_stride=True,
        is_metric=True,
        **kwargs,
    ):
        if is_metric is not True:
            raise ValueError(f"ScanNet++ is_metric must be True, got {is_metric!r}")
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        # The preprocessed ScanNet++ tree is train-only: preprocess_scannetpp.py
        # renders every scene in DUSt3R's scene list and writes no split.
        if self.split is not Split.TRAIN:
            raise ValueError(
                f"ScanNet++ is preprocessed train-only; split must be "
                f"Split.TRAIN, got {self.split!r}"
            )
        self.loaded_data = self._load_data()

    def _load_data(self):
        meta = osp.join(self.ROOT, "all_metadata.npz")
        if not osp.isfile(meta):
            raise FileNotFoundError(
                f"ScanNet++ all_metadata.npz not found: {meta} -- run "
                f"preprocess_scannetpp.py --finalize"
            )
        with np.load(meta) as data:
            all_scenes = data["scenes"]

        offset = 0
        scenes = []
        sceneids = []
        images = []
        intrinsics = []
        trajectories = []
        seqids = []
        seq_img_list = []
        start_img_ids = []
        j = 0
        self.image_num = 0

        for scene in tqdm(all_scenes):
            scene_dir = osp.join(self.ROOT, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]

            num_imgs = len(imgs)
            imgs_on_disk = {
                name[:-4]
                for name in zlistdir(osp.join(frames_root(scene_dir), "images"))
                if name.endswith(".jpg")
            }

            # iPhone frames only (named frame_%06d). The DSLR run is a set of
            # stills, not a capture: consecutive DSC numbers are a median 20 cm
            # of camera motion apart (up to 9.7 m), so a "clip" of them is not
            # video at any stride and its temporal metrics are meaningless.
            # Their metadata rows stay in place so intrinsics/trajectories
            # indexing by global id is unaffected.
            iphone_ids = []
            frame_numbers = []
            for i in range(num_imgs):
                if imgs[i] not in imgs_on_disk or not imgs[i].startswith("frame"):
                    continue
                iphone_ids.append(i + offset)
                frame_numbers.append(int(str(imgs[i]).split("_")[1]))

            cut_off = self.min_views()
            # frame_%06d is the index in the 60 fps iPhone video, so the step
            # between kept frames is the decimation of the capture. It is a
            # property of the preprocessing run (currently ~50, i.e. every 50th
            # frame), hence derived per scene rather than hard-coded.
            frame_numbers = np.array(frame_numbers)
            sequences = (
                segment_frame_ids_by_rate(
                    iphone_ids, frame_numbers, GAP_FACTOR, MAX_FRAME_GAP, cut_off
                )
                if len(iphone_ids) >= cut_off
                else []
            )
            if not sequences:
                print(f"Skipping {scene}")
                continue

            for img_ids in sequences:
                seq_img_list.append(img_ids)
                start_img_ids.extend(img_ids[: len(img_ids) - cut_off + 1])
                # seqids is indexed by GLOBAL image id, so every frame of this
                # run points at the sequence just appended
                for gid in img_ids:
                    seqids.append((gid, len(seq_img_list) - 1))

            scenes.append(scene)
            sceneids.extend([j] * num_imgs)
            images.extend(imgs)
            intrinsics.append(intrins)
            trajectories.append(traj)
            self.image_num += num_imgs
            offset += num_imgs
            j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"ScanNet++ found no usable scenes under {self.ROOT} "
                f"(need at least {self.min_views()} frames from one camera)"
            )

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.intrinsics = np.concatenate(intrinsics, axis=0)
        self.trajectories = np.concatenate(trajectories, axis=0)
        self.seq_img_list = seq_img_list
        self.start_img_ids = start_img_ids
        # dense global-id -> sequence-index lookup; unfiltered ids (absent from
        # disk) keep the -1 sentinel and are never reachable, since every
        # start_img_id comes from a sequence list
        self.seqids = np.full(offset, -1, dtype=np.int64)
        for gid, seq_idx in seqids:
            self.seqids[gid] = seq_idx

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return self.image_num

    def get_stats(self):
        return f"{len(self)} groups of views"

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
            # frames live either in the scene dir (extracted layout) or in its
            # frames.zip (inode-safe layout); the metadata npz reads above are
            # unaffected (always real files in the scene dir)
            scene_dir = frames_root(osp.join(self.ROOT, self.scenes[scene_id]))

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000
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
                    dataset="ScanNet++",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.99, dtype=np.float32),
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
                f"ScanNet++ produced {len(views)} views but {num_views} were requested"
            )
        return views
