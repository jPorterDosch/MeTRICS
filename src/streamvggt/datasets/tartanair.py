import os
import os.path as osp

import numpy as np

from .base.base_multiview_dataset import BaseMultiViewDataset, EmptyDatasetError
from .base.segments import segment_frame_ids
from .types import Split
from .utils.image import imread_cv2
from .utils.zipio import frames_root, listdir as zlistdir, np_load

# preserves the original DUSt3R TartanAir stride cap (max_interval=20); override
# via the constructor or the DatasetConfig CLI rather than editing this constant.
# It is wide because TartanAir is rendered at a high, constant frame rate, so
# adjacent frames are nearly redundant.
DEFAULT_STRIDE_RANGE = (1, 20)

# Largest frame-index step still treated as one continuous capture. TartanAir
# renders every frame of a trajectory, so no trajectory splits today; this only
# guards a future preprocessing change.
MAX_FRAME_GAP = 1


class TartanAir_Multi(BaseMultiViewDataset):
    """TartanAir synthetic drone trajectories with metric depth and per-frame
    cam2world poses, preprocessed into:
        ROOT/<env>/<difficulty>/<trajectory>/<frame>_{rgb.png,depth.npy,cam.npz}

    Frames are flat at the trajectory root in both layouts -- loose files, or
    members of that directory's frames.zip. There is no metadata npz: the tree
    itself is the index, which is why this loader walks it at construction.

    Depth carries rendered sky at ~1e4 m and occasional far outliers, so
    _get_views drops sky and clips past the 98th percentile per frame. Those
    pixels become 0 and the base class turns them into an invalid mask, which
    is why outdoor environments legitimately report a low valid fraction.
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
            raise ValueError(f"TartanAir is_metric must be True, got {is_metric!r}")
        self.ROOT = ROOT
        self.video = True
        self.is_metric = is_metric
        super().__init__(
            *args, stride_range=stride_range, regular_stride=regular_stride, **kwargs
        )
        # TartanAir ships no train/test partition -- the preprocessed tree is a
        # flat set of trajectories. The DUSt3R loader spelled that `split=None`;
        # here Split is a required enum, so TRAIN is the only accepted value and
        # TEST fails fast rather than silently returning the training frames as
        # a held-out set.
        if self.split is not Split.TRAIN:
            raise ValueError(
                f"TartanAir has no test partition; split must be Split.TRAIN, "
                f"got {self.split!r}"
            )
        self._load_data()

    def _load_data(self):
        if not osp.isdir(self.ROOT):
            raise FileNotFoundError(f"TartanAir root not found: {self.ROOT}")
        scene_dirs = sorted(
            d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))
        )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        seq_img_list = []
        seqids = []
        start_img_ids = []
        j = 0

        for scene in scene_dirs:
            for mode in ["Easy", "Hard"]:
                mode_dir = osp.join(self.ROOT, scene, mode)
                # a partially downloaded tree can hold an env with only one
                # difficulty; that is a gap, not a failure, so skip it here and
                # let the EmptyDatasetError below fire if nothing is usable
                if not osp.isdir(mode_dir):
                    continue
                seq_dirs = sorted(
                    osp.join(mode_dir, d)
                    for d in os.listdir(mode_dir)
                    if osp.isdir(osp.join(mode_dir, d))
                )
                for seq_dir in seq_dirs:
                    # frames are flat at the trajectory root in both layouts:
                    # loose files in seq_dir, or members of seq_dir/frames.zip.
                    # Match _rgb.png specifically rather than any .png -- the
                    # basename is the name minus that exact 8-char suffix, so a
                    # stray .png would yield a truncated, unloadable basename.
                    basenames = sorted(
                        f[: -len("_rgb.png")]
                        for f in zlistdir(frames_root(seq_dir))
                        if f.endswith("_rgb.png")
                    )
                    num_imgs = len(basenames)
                    cut_off = self.min_views()

                    if num_imgs < cut_off:
                        print(f"Skipping {seq_dir}")
                        continue
                    img_ids = list(np.arange(num_imgs) + offset)

                    # preprocess_tartanair.py renders every frame of a
                    # trajectory, so this is one run per trajectory today; it
                    # is here so a future gap cannot silently produce clips
                    # that jump across it
                    frame_idx = np.array([int(name) for name in basenames])
                    sequences = segment_frame_ids(
                        img_ids, frame_idx, MAX_FRAME_GAP, cut_off
                    )
                    if not sequences:
                        print(
                            f"Skipping {seq_dir}: no run of {cut_off} "
                            f"consecutive frames"
                        )
                        continue

                    for img_ids_seq in sequences:
                        seq_img_list.append(img_ids_seq)
                        start_img_ids.extend(
                            img_ids_seq[: len(img_ids_seq) - cut_off + 1]
                        )
                        # seqids is indexed by GLOBAL image id
                        for gid in img_ids_seq:
                            seqids.append((gid, len(seq_img_list) - 1))

                    scenes.append(seq_dir)
                    sceneids.extend([j] * num_imgs)
                    images.extend(basenames)
                    offset += num_imgs
                    j += 1

        if not scenes:
            raise EmptyDatasetError(
                f"TartanAir found no usable trajectories under {self.ROOT} "
                f"(need at least {self.min_views()} frames each)"
            )
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        # one entry per contiguous run, NOT per trajectory: indexed by
        # seqids[start_id], never by sceneids[start_id] (the path lookup)
        self.seq_img_list = seq_img_list
        self.seqids = np.full(offset, -1, dtype=np.int64)
        for gid, seq_idx in seqids:
            self.seqids[gid] = seq_idx

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

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
            # frames live either loose in the trajectory dir (extracted layout)
            # or in its frames.zip (inode-safe layout)
            scene_dir = frames_root(self.scenes[scene_id])
            basename = self.images[view_idx]

            img = basename + "_rgb.png"
            image = imread_cv2(osp.join(scene_dir, img))
            depthmap = np_load(osp.join(scene_dir, basename + "_depth.npy"))
            camera_params = np_load(osp.join(scene_dir, basename + "_cam.npz"))

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= 1000
            depthmap[sky_mask] = -1.0  # sky
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="TartanAir",
                    label=scene_dir,
                    is_metric=self.is_metric,
                    instance=scene_dir + "_" + img,
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
                f"TartanAir produced {len(views)} views but {num_views} were requested"
            )
        return views
