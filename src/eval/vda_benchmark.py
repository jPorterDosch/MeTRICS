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

from dust3r.utils.image import ImgNorm, exif_transpose
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
    # the manifest lists UNCROPPED frames (see Sequence.rgb_uncropped)
    rgb_uncropped: bool = False

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
        # VDA's eval_scenes_num is 20, taken from a manifest whose order is an
        # unsorted glob -- an arbitrary, unrepeatable 20 of the 100 scenes, and
        # per-scene TAE ranges 0.22-1.20, so a 20-scene mean moves by +-0.04
        # with the draw. All 100 are scored instead.
        tae_scenes=100,
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
    # All 26 Bonn sequences: what VDA's Bonn numbers are over (its extractor
    # takes every directory of the dataset; the 5 of `bonn` are
    # DepthCrafter's list). 110- and 500-frame protocols as for `bonn`.
    "bonn_all": BenchSpec(
        "bonn_all", "bonn_all/bonn_all_video.json", 10.0, (0, 480, 0, 640), 110, True
    ),
    "bonn_all_500": BenchSpec(
        "bonn_all_500",
        "bonn_all/bonn_all_video_500.json",
        10.0,
        (0, 480, 0, 640),
        500,
        True,
        base="bonn_all",
    ),
    # Sintel with R and B swapped on disk, as VDA's shipped extractor writes
    # its colour files. Only to reproduce VDA's published Sintel number; the
    # benchmark's `sintel` has true colours.
    "sintel_bgr": BenchSpec(
        "sintel_bgr",
        "sintel_bgr/sintel_bgr_video.json",
        70.0,
        (0, 436, 0, 1024),
        100,
        True,
    ),
    # the same 654 stills with the whole 480x640 frame as input and the same
    # crop for scoring: the single-image protocol (Marigold's) that Depth Any
    # Video reports. Only used to reproduce that table.
    "nyuv2_full": BenchSpec(
        "nyuv2_full",
        "nyuv2/nyuv2_test_full.json",
        10.0,
        (45, 471, 41, 601),
        1,
        False,
        base="nyuv2",
        rgb_uncropped=True,
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


def load_manifest(
    root: Path, spec: BenchSpec, tae: bool = False, tae_slice: bool = True
) -> list[Sequence]:
    """Every sequence of the manifest, frames truncated to the protocol's
    count (spec.max_len, or the TAE frame range for the TAE manifest --
    VDA's eval slices [start_idx:end_idx] after loading). tae_slice=False
    keeps every listed TAE frame, for a caller that -- like VDA's own
    infer.py -- runs the model on all of them and slices the prediction."""
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
            if tae_slice:
                lo, hi = spec.tae_range
                frames = frames[lo:hi]
        else:
            frames = frames[: spec.max_len]
        seq = Sequence(
            spec.name,
            name,
            rgb_uncropped=tae or spec.rgb_uncropped,
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


def full_frame_box(
    seq: Sequence, spec: BenchSpec, gt_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """(top, left, H_frame, W_frame) of the cropped GT inside an uncropped
    frame. The frame must be the uncropped GT's size, so the protocol crop
    means the same pixels on both."""
    img = exif_transpose(Image.open(seq.frames[0].image))
    Wf, Hf = img.size
    ys, xs = crop_slices(spec.crop)
    rows, cols = range(Hf)[ys], range(Wf)[xs]
    if (len(rows), len(cols)) != tuple(gt_hw):
        raise ValueError(
            f"{seq.frames[0].image}: frame {Wf}x{Hf} cropped by {spec.crop} gives "
            f"{len(cols)}x{len(rows)}, but the cropped GT is {gt_hw[1]}x{gt_hw[0]}"
        )
    return rows[0], cols[0], Hf, Wf


def model_to_gt(
    arr: np.ndarray, gt_hw: tuple[int, int], box=None, mask: bool = False
) -> np.ndarray:
    """A model-resolution [S,h,w] map onto the [S,H,W] GT grid: resized
    straight to it, or, for a whole-frame input (`box` from full_frame_box),
    resized to the frame and cropped to the GT region. `mask` maps a sparse
    fed-pixel mask with sparse_mask_to_gt's exact index map."""
    if box is None:
        return sparse_mask_to_gt(arr, gt_hw) if mask else resize_to_gt(arr, gt_hw)
    top, left, Hf, Wf = box
    full = sparse_mask_to_gt(arr, (Hf, Wf)) if mask else resize_to_gt(arr, (Hf, Wf))
    H, W = gt_hw
    return np.ascontiguousarray(full[:, top : top + H, left : left + W])


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

    A manifest that lists UNCROPPED frames (ScanNet's TAE clips) is fed the
    whole frame, as VDA's protocol does (infer.py on color_origin, then
    eval_tae.py --hard_crop on the prediction): the cropped GT is placed back
    at its offset in a zero canvas of the frame's size, so sparse depth is
    still drawn only inside the scored region, and every view carries
    `gt_box` = (top, left, H_frame, W_frame) for model_to_gt to crop the
    prediction (and the fed-pixel mask) back onto the GT grid.
    """
    if len(seq.frames) != gt.shape[0]:
        raise ValueError(
            f"{seq.name}: {len(seq.frames)} frames but gt has {gt.shape[0]}"
        )
    views = []
    H, W = gt.shape[1:]
    box = full_frame_box(seq, spec, (H, W)) if seq.rgb_uncropped else None
    for i, fr in enumerate(seq.frames):
        img = load_rgb(fr.image, size, None, (H, W))  # [1,3,h,w] in [-1,1]
        h, w = img.shape[-2:]
        src = gt[i]
        if box is not None:
            top, left, Hf, Wf = box
            src = np.zeros((Hf, Wf), dtype=gt.dtype)
            src[top : top + H, left : left + W] = gt[i]
        # NEAREST_EXACT: the source pixel nearest each target pixel's centre.
        # Plain INTER_NEAREST takes floor(x * scale), which shifts every fed
        # sample ~0.5 * (scale - 1) source pixels up-left of the RGB pixel it
        # conditions (0.7 px on KITTI) and into the sparse_aligned fit target
        depth = cv2.resize(src, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        # the protocol's valid range (VDA: 1e-3 < gt < max_depth), not just
        # > 0: Sintel's sky is stored at up to 650 m, and a sparse-depth
        # sample from it conditions the model on a depth the protocol never
        # scores and dominates the per-frame sparse_aligned fit
        valid = np.isfinite(depth) & (depth > 1e-3) & (depth < spec.max_depth)
        view = {
            "img": img.to(device),
            "depthmap": torch.from_numpy(depth)[None].to(device),
            "valid_mask": torch.from_numpy(valid)[None].to(device),
            "true_shape": torch.tensor([[h, w]], device=device),
            "idx": i,
            "instance": str(fr.image),
            "dataset": spec.name,
        }
        if box is not None:
            view["gt_box"] = box
        if fr.K is not None:
            K = (
                scaled_intrinsics(fr.K, (0, 0, 0, 0), box[2:], (h, w))
                if box is not None
                else scaled_intrinsics(fr.K, spec.crop, (H, W), (h, w))
            )
            view["camera_intrinsics"] = torch.from_numpy(K.astype(np.float32))[None].to(
                device
            )
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
    """One RGB as the model input [1,3,h,w] in [-1,1]: long side -> size,
    then each side resized to the NEAREST multiple of 14 (no crop). dust3r's
    load_images_for_eval(size, crop=False), which StreamVGGT's own launcher
    uses, FLOORS instead: 640x480 -> 518x378, a 2.7% squash (Sintel 4.8%).
    Rounding gives 518x392 for 4:3 frames, the training / validation
    resolution, with at most a 0.9% aspect error there (Sintel 1.6% at
    518x224, a grid training did not use; KITTI 1.3%; decided 2026-10-14).
    With `crop`, the protocol crop is
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
    # ONE resample, straight to the rounded size (dust3r's _resize_pil_image
    # sizes and filter: LANCZOS when shrinking, BICUBIC when enlarging)
    W0, H0 = img.size
    S = max(W0, H0)
    target = tuple(_round14(int(round(x * size / S))) for x in (W0, H0))
    img = img.resize(target, Image.LANCZOS if S > size else Image.BICUBIC)
    return ImgNorm(img)[None]


def _round14(x: int) -> int:
    """Nearest multiple of 14 (halves up), at least 14."""
    return max(14, int(x / 14 + 0.5) * 14)


def resize_to_gt(
    pred: np.ndarray, hw: tuple[int, int], nearest: bool = False
) -> np.ndarray:
    """[S,h,w] -> [S,H,W], bilinear like VDA's get_infer (cv2 default), or
    nearest for masks."""
    H, W = hw
    interp = cv2.INTER_NEAREST_EXACT if nearest else cv2.INTER_LINEAR
    out = np.stack(
        [cv2.resize(p.astype(np.float32), (W, H), interpolation=interp) for p in pred],
        axis=0,
    )
    return out.astype(bool) if nearest and pred.dtype == bool else out


def sparse_mask_to_gt(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """[S,h,w] model-resolution sparse mask -> [S,H,W] GT-resolution mask of
    every GT pixel the model was fed, for holding them out of the score.

    build_views samples the sparse depth from the GT nearest-DOWNsampled to
    (h, w); nearest-upsampling the mask back is not its inverse, so it misses
    some fed source pixels. The
    exact set is the downsample's own source-index map, read off cv2 by
    resizing an index image the same way; the upsampled patches are kept on
    top so whole fed patches stay excluded as before."""
    if mask.ndim != 3:
        raise ValueError(f"mask must be [S,h,w], got shape {mask.shape}")
    H, W = hw
    h, w = mask.shape[1:]
    if H * W >= 1 << 24:
        raise ValueError(f"GT {H}x{W} too large for a float32 index map")
    idx = np.arange(H * W, dtype=np.float32).reshape(H, W)
    src = cv2.resize(idx, (w, h), interpolation=cv2.INTER_NEAREST_EXACT).astype(
        np.int64
    )
    out = resize_to_gt(mask.astype(np.float32), hw, nearest=True) > 0.5
    flat = out.reshape(out.shape[0], -1)
    for i in range(mask.shape[0]):
        flat[i, src[mask[i]]] = True
    return out


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
        if name not in datasets:  # its TAE pass only runs alongside its main pass
            continue
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
    "sparse_mask_to_gt",
    "load_rgb",
    "scaled_intrinsics",
    "bench_root_ok",
]
