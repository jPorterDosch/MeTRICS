import os
import os.path as osp

import cv2
import numpy as np

from .base.base_multiview_dataset import BaseMultiViewDataset, EmptyDatasetError
from .types import Split
from .utils.image import imread_cv2

# preserves the original DUSt3R HyperSim stride cap; override via the constructor
# or the DatasetConfig CLI rather than editing this constant.
DEFAULT_STRIDE_RANGE = (1, 4)


class HyperSim_Multi(BaseMultiViewDataset):
    """Hypersim photorealistic synthetic indoor scenes with perfect metric
    depth and cam2world poses, preprocessed (preprocess_hypersim.py) into:
        ROOT/<scene>/<cam_XX>/{NNNNNN_rgb.png, NNNNNN_depth.npy, NNNNNN_cam.npz}

    Keeps the original DUSt3R contract: sub-scenes are enumerated from the
    directory tree (no split subdirs, no pairs, no metadata npz); sequences
    are sampled from index-ordered frames, and the dataset length is inflated
    10x so each anchor is revisited with different view samplings per epoch.
    The processed tree has no test split -- only Split.TRAIN is served."""

    def __init__(
        self,
        *args,
        ROOT,
        stride_range=DEFAULT_STRIDE_RANGE,
        regular_stride=True,
        is_metric=True,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        if self.split is not Split.TRAIN:
            raise ValueError(
                f"HyperSim has no test split (the processed tree is unsplit); "
                f"got {self.split!r}"
            )

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        j = 0
        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            if not rgb_paths:
                raise RuntimeError(f"HyperSim sub-scene {scene_dir} is empty")
            num_imgs = len(rgb_paths)
            cut_off = self.min_views()
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"HyperSim found no usable sub-scenes under {self.ROOT}"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.start_img_ids = start_img_ids

    def __len__(self):
        return len(self.start_img_ids) * 10

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        idx = idx // 10
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
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
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            rgb_path = self.images[view_idx]
            depth_path = rgb_path.replace("rgb.png", "depth.npy")
            cam_path = rgb_path.replace("rgb.png", "cam.npz")

            rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, depth_path)).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, cam_path))
            intrinsics = cam_file["intrinsics"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

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
                    dataset="hypersim",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
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
                f"HyperSim produced {len(views)} views but {num_views} were requested"
            )
        return views
