"""The video-depth benchmark tree (Video Depth Anything's protocol) as data:
per-dataset protocol constants, manifest reading, GT loading and the
conversion of a sequence into the [S]-list of view dicts the model consumes.

The tree is built by datasets_preprocess/prepare_vda_benchmark.py from the
raw downloads, with VDA's own extraction code, into
    <root>/<dataset>/<sequence>/{rgb|clean|color, depth}/...
    <root>/<dataset>/<dataset>_video.json      (+ scannet_video_tae.json)
The json is VDA's: {dataset: [{sequence: [{image, gt_depth, factor[, K, pose]}]}]}.

The constants below are copied from VDA's eval.py / eval_tae.py argument
blocks, one per dataset; they decide the valid-depth range, the crop the GT
is scored under and how many frames of a sequence count. Changing one
changes the protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from dust3r.utils.image import ImgNorm, _resize_pil_image, exif_transpose
from PIL import Image


@dataclass(frozen=True)
class BenchSpec:
    name: str
    json: str  # manifest, relative to the benchmark root
    max_depth: float
    crop: tuple[int, int, int, int]  # (a, b, c, d): GT[a:b, c:d], VDA's crop
    max_len: int  # frames of a sequence that are scored
    video: bool  # False: stills, each a one-frame sequence (no TAE, no clip)
    tae_json: str | None = None  # manifest with K/pose (ScanNet only)
    tae_scenes: int = 0  # VDA: eval_scenes_num
    tae_range: tuple[int, int] = (0, 0)  # VDA: start_idx, end_idx
    base: str | None = (
        None  # tree dir + json key when they differ from name (the *_500 variants)
    )

    @property
    def dirname(self) -> str:
        return self.base or self.name


SPECS: dict[str, BenchSpec] = {
    "sintel": BenchSpec(
        "sintel", "sintel/sintel_video.json", 70.0, (0, 436, 0, 1024), 100, True
    ),
    "scannet": BenchSpec(
        "scannet",
        "scannet/scannet_video.json",
        10.0,
        (8, -8, 11, -11),
        90,
        True,
        tae_json="scannet/scannet_video_tae.json",
        tae_scenes=20,
        tae_range=(10, 180),
    ),
    "kitti": BenchSpec(
        "kitti", "kitti/kitti_video.json", 80.0, (0, 374, 0, 1242), 110, True
    ),
    "bonn": BenchSpec(
        "bonn", "bonn/bonn_video.json", 10.0, (0, 480, 0, 640), 110, True
    ),
    # not a VDA dataset block: their eval has only the 8-scene 500-frame
    # nyuv2 video split. This is the standard 654-still test split, each
    # still its own one-frame "sequence", with VDA's NYU crop and factor.
    "nyuv2": BenchSpec(
        "nyuv2", "nyuv2/nyuv2_test.json", 10.0, (45, 471, 41, 601), 1, False
    ),
    # VDA's headline (Table 1) protocol: up to 500 frames per video, from the
    # *_video_500.json manifests their extractor writes alongside the short
    # ones (ScanNet at stride 1 here, not the 90-frame split's stride 3).
    # Sintel is 50 frames either way and NYU's 500-frame split is an 8-scene
    # video set we do not build. Opt-in: ~4.5x the frames of the short
    # protocol, ScanNet alone 100 x 500.
    "scannet_500": BenchSpec(
        "scannet_500",
        "scannet/scannet_video_500.json",
        10.0,
        (8, -8, 11, -11),
        500,
        True,
        base="scannet",
    ),
    "kitti_500": BenchSpec(
        "kitti_500",
        "kitti/kitti_video_500.json",
        80.0,
        (0, 374, 0, 1242),
        500,
        True,
        base="kitti",
    ),
    "bonn_500": BenchSpec(
        "bonn_500",
        "bonn/bonn_video_500.json",
        10.0,
        (0, 480, 0, 640),
        500,
        True,
        base="bonn",
    ),
}


@dataclass
class Frame:
    image: Path
    depth: Path
    factor: float
    K: np.ndarray | None = None  # 3x3 (VDA writes the 4x4 intrinsic_depth.txt)
    pose: np.ndarray | None = None  # 4x4 cam2world


@dataclass
class Sequence:
    dataset: str
    name: str
    frames: list[Frame]
    # True for VDA's ScanNet TAE manifest, which lists the UNCROPPED
    # color_origin frames: build_views then applies the protocol crop to the
    # RGB itself, so the prediction is registered to the cropped GT
    rgb_uncropped: bool = False

    def __len__(self) -> int:
        return len(self.frames)


def _as_K3(K) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3 or 4x4, got {K.shape}")
    return K


def load_manifest(root: Path, spec: BenchSpec, tae: bool = False) -> list[Sequence]:
    """Every sequence of the manifest, frames truncated to the protocol's
    count (spec.max_len, or the TAE frame range for the TAE manifest --
    VDA's eval slices [start_idx:end_idx] after loading)."""
    rel = spec.tae_json if tae else spec.json
    if rel is None:
        raise ValueError(f"{spec.name} has no TAE manifest")
    path = Path(root) / rel
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark manifest missing: {path} -- run "
            "datasets_preprocess/prepare_vda_benchmark.py"
        )
    with open(path) as f:
        data = json.load(f)
    if spec.dirname not in data:
        raise KeyError(f"{path} has keys {list(data)}, expected {spec.dirname!r}")
    ds_root = Path(root) / spec.dirname
    sequences = []
    for entry in data[spec.dirname]:
        if len(entry) != 1:
            raise ValueError(
                f"{path}: one sequence per entry expected, got {list(entry)}"
            )
        ((name, frames),) = entry.items()
        if tae:
            lo, hi = spec.tae_range
            frames = frames[lo:hi]
        else:
            frames = frames[: spec.max_len]
        seq = Sequence(
            spec.name,
            name,
            rgb_uncropped=tae,
            frames=[
                Frame(
                    ds_root / fr["image"],
                    ds_root / fr["gt_depth"],
                    float(fr["factor"]),
                    _as_K3(fr["K"]) if "K" in fr else None,
                    np.asarray(fr["pose"], dtype=np.float64) if "pose" in fr else None,
                )
                for fr in frames
            ],
        )
        if len(seq.frames) == 0:
            continue
        sequences.append(seq)
    if tae:
        sequences = sequences[: spec.tae_scenes]
    if not sequences:
        raise ValueError(f"{path} lists no sequences")
    return sequences


def crop_slices(crop: tuple[int, int, int, int]) -> tuple[slice, slice]:
    a, b, c, d = crop
    return slice(a, b), slice(c, d)


def read_gt(frame: Frame, crop: tuple[int, int, int, int]) -> np.ndarray:
    """GT depth in metres, cropped as VDA's eval crops it; 0 where the sensor
    has no value (VDA's get_gt maps 0 -> -1, and its valid mask is gt > 1e-3
    either way)."""
    raw = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read GT depth {frame.depth}")
    if raw.ndim != 2:
        raise ValueError(
            f"{frame.depth}: expected a single-channel depth png, got {raw.shape}"
        )
    depth = raw.astype(np.float64) / frame.factor
    ys, xs = crop_slices(crop)
    depth = depth[ys, xs]
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def gt_stack(seq: Sequence, spec: BenchSpec) -> np.ndarray:
    """[S,H,W] GT for a sequence, every frame at the same (cropped) shape."""
    gts = [read_gt(fr, spec.crop) for fr in seq.frames]
    shape = gts[0].shape
    for fr, g in zip(seq.frames, gts):
        if g.shape != shape:
            raise ValueError(
                f"{seq.dataset}/{seq.name}: GT shape {g.shape} at {fr.depth.name} "
                f"differs from {shape}"
            )
    return np.stack(gts, axis=0)


def scaled_intrinsics(
    K: np.ndarray,
    crop: tuple[int, int, int, int],
    from_hw: tuple[int, int],
    to_hw: tuple[int, int],
) -> np.ndarray:
    """K for the model-resolution frame: principal point shifted by the crop
    origin, then scaled from the cropped GT size to the model size."""
    a, _, c, _ = crop
    K = _as_K3(K).copy()
    K[0, 2] -= c
    K[1, 2] -= a
    sy = to_hw[0] / from_hw[0]
    sx = to_hw[1] / from_hw[1]
    K[0] *= sx
    K[1] *= sy
    return K


def build_views(
    seq: Sequence,
    spec: BenchSpec,
    gt: np.ndarray,
    size: int,
    device: torch.device,
) -> list[dict]:
    """The [S]-list of single-view dicts for one sequence, model-side.

    RGB goes through dust3r's load_images_for_eval(size, crop=False), the
    same resize the vendored StreamVGGT video-depth launcher uses (long side
    -> size, both sides to a multiple of 14 by resizing, no crop), so the
    model sees the whole frame. img is ImgNorm [-1,1] as the datasets emit
    it; the caller rescales to [0,1] exactly like _prepare_batch does.

    depthmap is the cropped GT nearest-resized to the model resolution: it is
    only the SOURCE of the simulated sparse depth (simulate_sparse_depth
    reads view['depthmap'] and view['valid_mask']). Scoring uses the GT at
    its own resolution (gt_stack), never this copy.
    """
    if len(seq.frames) != gt.shape[0]:
        raise ValueError(
            f"{seq.name}: {len(seq.frames)} frames but gt has {gt.shape[0]}"
        )
    views = []
    H, W = gt.shape[1:]
    crop = spec.crop if seq.rgb_uncropped else None
    for i, fr in enumerate(seq.frames):
        img = load_rgb(fr.image, size, crop, (H, W))  # [1,3,h,w] in [-1,1]
        h, w = img.shape[-2:]
        depth = cv2.resize(gt[i], (w, h), interpolation=cv2.INTER_NEAREST)
        valid = (depth > 0) & np.isfinite(depth)
        view = {
            "img": img.to(device),
            "depthmap": torch.from_numpy(depth)[None].to(device),
            "valid_mask": torch.from_numpy(valid)[None].to(device),
            "true_shape": torch.tensor([[h, w]], device=device),
            "idx": i,
            "instance": str(fr.image),
            "dataset": spec.name,
        }
        if fr.K is not None:
            view["camera_intrinsics"] = torch.from_numpy(
                scaled_intrinsics(fr.K, spec.crop, (H, W), (h, w)).astype(np.float32)
            )[None].to(device)
        if fr.pose is not None:
            view["camera_pose"] = torch.from_numpy(fr.pose.astype(np.float32))[None].to(
                device
            )
        views.append(view)
    return views


def load_rgb(
    path: Path,
    size: int,
    crop: tuple[int, int, int, int] | None,
    gt_hw: tuple[int, int],
) -> torch.Tensor:
    """One RGB as the model input [1,3,h,w] in [-1,1]: dust3r's
    load_images_for_eval(size, crop=False) math (long side -> size, both
    sides to a multiple of 14 by resizing). With `crop`, the protocol crop is
    applied to the RGB first -- for a manifest that lists uncropped frames --
    and the RGB must then be the size of the uncropped GT, so the pixel crop
    means the same thing on both."""
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark image {path}")
    img = exif_transpose(Image.open(path)).convert("RGB")
    if crop is not None:
        a, b, c, d = crop
        Wr, Hr = img.size
        ys, xs = crop_slices(crop)
        rows, cols = range(Hr)[ys], range(Wr)[xs]
        if (len(rows), len(cols)) != gt_hw:
            raise ValueError(
                f"{path}: RGB {Wr}x{Hr} cropped by {crop} gives {len(cols)}x{len(rows)}, "
                f"but the cropped GT is {gt_hw[1]}x{gt_hw[0]}; export ScanNet colour at "
                "the depth resolution (extract_scannet_sens.py --color-at-depth-res)"
            )
        img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    img = _resize_pil_image(img, size)
    Wi, Hi = img.size
    cx, cy = Wi // 2, Hi // 2
    halfw, halfh = ((2 * cx) // 14) * 7, ((2 * cy) // 14) * 7
    img = img.resize((2 * halfw, 2 * halfh), Image.LANCZOS)
    return ImgNorm(img)[None]


def resize_to_gt(
    pred: np.ndarray, hw: tuple[int, int], nearest: bool = False
) -> np.ndarray:
    """[S,h,w] -> [S,H,W], bilinear like VDA's get_infer (cv2 default), or
    nearest for masks."""
    H, W = hw
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = np.stack(
        [cv2.resize(p.astype(np.float32), (W, H), interpolation=interp) for p in pred],
        axis=0,
    )
    return out.astype(bool) if nearest and pred.dtype == bool else out


def bench_root_ok(
    root: Path, datasets: tuple[str, ...], tae_datasets: tuple[str, ...] = ()
) -> list[str]:
    """Manifests missing under root, as "<dataset>" / "<dataset>/tae" names:
    the main manifest of every requested dataset, plus the TAE manifest of
    every TAE dataset that has one (ScanNet)."""
    missing = []
    for name in datasets:
        spec = SPECS[name]
        if not (Path(root) / spec.json).is_file():
            missing.append(name)
    for name in tae_datasets:
        spec = SPECS[name]
        if spec.tae_json and not (Path(root) / spec.tae_json).is_file():
            missing.append(f"{name}/tae")
    return missing


__all__ = [
    "SPECS",
    "BenchSpec",
    "Frame",
    "Sequence",
    "load_manifest",
    "read_gt",
    "gt_stack",
    "build_views",
    "resize_to_gt",
    "load_rgb",
    "scaled_intrinsics",
    "bench_root_ok",
]
