import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from .base.base_multiview_dataset import BaseMultiViewDataset, EmptyDatasetError
from .base.segments import segment_frame_ids
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root, np_load

# preserves the original DUSt3R ScanNet stride cap; override via the constructor
# or the DatasetConfig CLI rather than editing this constant.
DEFAULT_STRIDE_RANGE = (1, 30)

# Largest frame-index step still treated as one continuous capture. ScanNet is
# extracted frame by frame from the .sens stream, so steps are 1 apart for
# 99.95% of frames; the rest are dropped frames (largest observed jump: 141
# frames, i.e. ~4.7 s at 30 fps). A clip spanning one of those is not video.
MAX_FRAME_GAP = 1

# ScanNet v2's official scene lists, vendored with their source and hashes in
# assets/scannet_splits/README.md. scans_train holds the train and val scenes;
# scans_test holds the 100 test scenes, which are the video-depth benchmark's.
OFFICIAL_SPLIT_DIR = osp.join(
    osp.dirname(__file__), "..", "..", "..", "assets", "scannet_splits"
)
OFFICIAL_SPLITS = ("train", "val")


def official_scenes(name: str) -> list[str]:
    """Scene ids of ScanNet v2's official `name` split ("train" or "val")."""
    if name not in OFFICIAL_SPLITS:
        raise ValueError(
            f"ScanNet official_split must be one of {OFFICIAL_SPLITS}, got {name!r}"
        )
    with open(osp.join(OFFICIAL_SPLIT_DIR, f"scannetv2_{name}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


class ScanNet_Multi(BaseMultiViewDataset):
    """ScanNet RGB-D video sequences with metric depth and per-frame cam2world
    poses, preprocessed into:
        ROOT/{scans_train,scans_test}/<scene>/{color,depth,cam}/... +
        new_scene_metadata.npz."""

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        regular_stride=True,
        is_metric=True,
        official_split=None,
        **kwargs,
    ):
        """`official_split` ("train" or "val") reads exactly the scenes of
        ScanNet v2's official list from scans_train, whatever `split` is;
        `split` still sets the sampling (TEST: seeded, regular strides), so a
        validation set is official_split="val" with Split.TEST. None keeps the
        original behaviour: every scene of scans_train for TRAIN, of
        scans_test for TEST."""
        if is_metric is not True:
            raise ValueError(f"ScanNet is_metric must be True, got {is_metric!r}")
        if official_split is not None and official_split not in OFFICIAL_SPLITS:
            raise ValueError(
                f"ScanNet official_split must be one of {OFFICIAL_SPLITS}, "
                f"got {official_split!r}"
            )
        self.ROOT = ROOT
        self.official_split = official_split
        self.video = True
        self.is_metric = is_metric
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        match self.split:
            case Split.TRAIN:
                subdir = "scans_train"
            case Split.TEST:
                subdir = "scans_test"
            case _:
                raise ValueError(
                    f"ScanNet split must be Split.TRAIN or Split.TEST, got {self.split!r}"
                )

        if official_split is not None:
            subdir = "scans_train"
        self.loaded_data = self._load_data(subdir)

    def _load_data(self, subdir):
        self.scene_root = osp.join(self.ROOT, subdir)
        if not osp.isdir(self.scene_root):
            raise FileNotFoundError(
                f"ScanNet split directory not found: {self.scene_root}"
            )
        if self.official_split is not None:
            wanted = official_scenes(self.official_split)
            missing = sorted(
                s for s in wanted if not osp.isdir(osp.join(self.scene_root, s))
            )
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(wanted)} official ScanNet "
                    f"{self.official_split} scenes missing under {self.scene_root}, "
                    f"e.g. {missing[:3]}"
                )
            self.scenes = sorted(wanted)
        else:
            self.scenes = [
                scene
                for scene in os.listdir(self.scene_root)
                if scene.startswith("scene")
            ]
        if not self.scenes:
            raise ValueError(f"No ScanNet sequences found in {self.scene_root}")

        offset = 0
        scenes = []
        sceneids = []
        seq_img_list = []
        seqids = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.scene_root, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                basenames = data["images"]
                num_imgs = len(basenames)
                img_ids = list(np.arange(num_imgs) + offset)
                cut_off = self.min_views()

                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue

                # basenames are the zero-padded .sens frame index ("00000"),
                # i.e. the frame's position on the 30 fps capture timeline.
                # preprocess_scannet.py drops frames with a non-finite pose, so
                # the indices have holes and adjacent entries are not always
                # adjacent in time.
                frame_idx = np.array([int(str(name)) for name in basenames])
                sequences = segment_frame_ids(
                    img_ids, frame_idx, MAX_FRAME_GAP, cut_off
                )
                if not sequences:
                    print(f"Skipping {scene}: no run of {cut_off} consecutive frames")
                    continue

                for img_ids_seq in sequences:
                    seq_img_list.append(img_ids_seq)
                    start_img_ids.extend(img_ids_seq[: len(img_ids_seq) - cut_off + 1])
                    # seqids is indexed by GLOBAL image id, so every frame of
                    # this run points at the sequence just appended
                    for gid in img_ids_seq:
                        seqids.append((gid, len(seq_img_list) - 1))

                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
                scenes.append(scene)

                # offset groups
                offset += num_imgs
                j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"ScanNet found no usable scenes under {self.scene_root}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        # one entry per contiguous run, NOT per scene (a scene with dropped
        # frames yields several): indexed by seqids[start_id], never by
        # sceneids[start_id], which stays the scene-path lookup
        self.seq_img_list = seq_img_list
        # dense global-id -> sequence-index lookup, matching ScanNetpp_Multi;
        # frames in a run too short for one clip keep the -1 sentinel and are
        # never reachable, since every start_img_id comes from a sequence list
        self.seqids = np.full(offset, -1, dtype=np.int64)
        for gid, seq_idx in seqids:
            self.seqids[gid] = seq_idx

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
            # frames live either in the scene dir (extracted layout) or in its
            # frames.zip (inode-safe layout); the metadata npz read above is
            # unaffected (always a real file in the scene dir)
            scene_dir = frames_root(osp.join(self.scene_root, self.scenes[scene_id]))
            rgb_dir = osp.join(scene_dir, "color")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(depth_dir, basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam = np_load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
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
                    dataset="ScanNet",
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
                f"ScanNet produced {len(views)} views but {num_views} were requested"
            )
        return views
