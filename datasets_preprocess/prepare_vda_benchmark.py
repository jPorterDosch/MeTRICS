#!/usr/bin/env python
"""Build the video-depth benchmark tree (Video Depth Anything's protocol)
from the raw downloads, with VDA's own extraction code where it works.

    python datasets_preprocess/prepare_vda_benchmark.py [sintel kitti bonn scannet nyuv2]

Reads  $EVAL_RAW  (datasets_download/download_eval_benchmarks.sbatch)
Writes $EVAL_OUT/<dataset>/<sequence>/{rgb|clean|color,depth}/... and the
manifests $EVAL_OUT/<dataset>/<dataset>_video.json (+ scannet_video_tae.json,
nyuv2_test.json) that src/eval/vda_benchmark.py reads.

ScanNet first needs the .sens frames exported (the raw scans_test is owned
by another user, so it goes to $EVAL_RAW/scannet; 510 frames is what VDA's
extractor reads):
    python datasets_preprocess/extract_scannet_sens.py --splits scans_test \\
        --extracted --out-root $EVAL_RAW/scannet --max-frames 510 --color-at-depth-res
(colour at the 640x480 depth size: VDA crops colour and depth by the same
[8:-8, 11:-11] pixels, which is the same field of view only if they share a
size -- at the native 1296x968 colour the two crops differ by ~1%)

The vendored scripts (third_party/video_depth_anything/benchmark/
dataset_extract, commit 4f5ae23) are used as-is for KITTI, Bonn and
ScanNet, with two shims applied from here rather than by editing them:
  * colour: they read with PIL (RGB) and write with cv2.imwrite (BGR), so
    every colour image they emit is channel-swapped; our loaders read with
    PIL, so imwrite is wrapped to swap back. Depth (2-D) is untouched.
  * bonn: their script calls get_sorted_files(root=...) but the helper's
    parameter is root_path -- a TypeError as shipped.
Two datasets are done here instead:
  * sintel: their script writes depth.astype(uint16) -- integer metres --
    while gen_json's factor (65535/650) expects depth * 65535/650, and it
    writes a {clean,depth}/<seq> layout that its own gen_json cannot read.
    Intended scale, gen_json's layout; frames are otherwise theirs.
  * nyuv2: their script covers only the 8-scene 500-frame video split (and
    calls a function it does not define). The 654-still test split is
    extracted from nyu_depth_v2_labeled.mat + splits.mat with VDA's NYU
    crop [45:471, 41:601] on RGB and factor 6000 on depth.
Bonn is restricted to the five sequences of DepthCrafter's meta_bonn.csv (the
published 5 x 110 protocol; MonST3R's list differs by one sequence); VDA's
extractor takes every directory it sees.

After extraction every manifest gets GT cameras (K + cam2world pose per
frame, see benchmark_cameras.py) -- VDA's gen_json writes none. `--cameras`
re-attaches them to manifests that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_cameras import attach_cameras  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "third_party" / "video_depth_anything" / "benchmark" / "dataset_extract"
if not VENDORED.is_dir():
    raise SystemExit(f"vendored VDA benchmark missing: {VENDORED}")
sys.path.insert(0, str(VENDORED))
import eval_utils  # noqa: E402
import dataset_extract_bonn  # noqa: E402
import dataset_extract_kitti  # noqa: E402
import dataset_extract_scannet  # noqa: E402
import dataset_extract_sintel  # noqa: E402

DEFAULT_RAW = "/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw"
DEFAULT_OUT = "/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench"

# DepthCrafter's benchmark/csv/meta_bonn.csv, the 5 x 110 protocol VDA's
# tables follow. person_tracking, NOT the person_tracking2 of MonST3R's
# prepare_bonn.py / src/eval/video_depth/metadata.py -- a different capture.
BONN_SEQUENCES = ("balloon2", "crowd2", "crowd3", "person_tracking", "synchronous")
SINTEL_FACTOR = 65535 / 650  # VDA gen_json's sintel factor
NYU_FACTOR = 6000.0  # VDA gen_json's nyuv2 factor
NYU_CROP = (45, 471, 41, 601)  # VDA copy_crop_files / eval.py nyuv2

MANIFEST = {
    "sintel": "sintel/sintel_video.json",
    "kitti": "kitti/kitti_video.json",
    "bonn": "bonn/bonn_video.json",
    "scannet": "scannet/scannet_video.json",
    "nyuv2": "nyuv2/nyuv2_test.json",
}
# VDA's 500-frame manifests, written by the same extractors; cameras are
# attached to these too so the *_500 benchmark variants can score TAE
MANIFEST_500 = {
    "kitti": "kitti/kitti_video_500.json",
    "bonn": "bonn/bonn_video_500.json",
    "scannet": "scannet/scannet_video_500.json",
}


class _RGBSafeCV2:
    """cv2 with an imwrite that takes RGB arrays (see module docstring)."""

    def __getattr__(self, name):
        return getattr(cv2, name)

    @staticmethod
    def imwrite(path, img, *args, **kwargs):
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.ascontiguousarray(img[..., ::-1])
        return cv2.imwrite(path, img, *args, **kwargs)


def install_shims() -> None:
    shim = _RGBSafeCV2()
    for mod in (eval_utils, dataset_extract_scannet, dataset_extract_sintel, dataset_extract_kitti, dataset_extract_bonn):
        mod.cv2 = shim
    dataset_extract_bonn.get_sorted_files = lambda root, suffix: eval_utils.get_sorted_files(root, suffix)


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


# ---------------------------------------------------------------------------
def prepare_sintel(raw: Path, out: Path) -> None:
    """Their extract_sintel writes <out>/sintel/{clean,depth}/<seq>/, but the
    gen_json it then calls globs <out>/sintel/<seq>/{clean,depth}/ -- so the
    manifest it produces lists two empty "sequences" named clean and depth.
    Written here in the layout gen_json reads, with the intended depth scale."""
    root = _require(raw / "sintel" / "training" / "clean", "Sintel clean pass")
    depth_root = _require(raw / "sintel" / "training" / "depth", "Sintel depth")
    name = "sintel"
    for seq_name in sorted(os.listdir(root)):
        names = eval_utils.get_sorted_files(str(root / seq_name), suffix=".png")
        for fn in names:
            depth = dataset_extract_sintel.depth_read(str(depth_root / seq_name / (fn[:-3] + "dpt")))
            img = Image.open(root / seq_name / fn).convert("RGB")
            out_img = out / name / seq_name / "clean" / fn
            out_depth = out / name / seq_name / "depth" / (fn[:-3] + "png")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_depth.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_img)
            # their script: depth.astype(np.uint16). Intended (per the factor
            # gen_json assigns): metres * 65535/650, saturating at 650 m --
            # sky is ~1e4 m and outside the 70 m protocol range either way.
            scaled = np.clip(np.round(depth.astype(np.float64) * SINTEL_FACTOR), 0, 65535)
            cv2.imwrite(str(out_depth), scaled.astype(np.uint16))
    eval_utils.gen_json(
        root_path=str(out / name), dataset=name, start_id=0, end_id=100, step=1,
        save_path=str(out / MANIFEST[name]),
    )


def prepare_kitti(raw: Path, out: Path) -> None:
    root = _require(raw / "kitti", "KITTI raw drives")
    depth_root = _require(raw / "kitti" / "val", "KITTI data_depth_annotated val")
    dataset_extract_kitti.extract_kitti(
        root=str(root), depth_root=str(depth_root), sample_len=-1,
        saved_dir=str(out) + "/", datatset_name="kitti",
    )


def prepare_bonn(raw: Path, out: Path) -> None:
    src = _require(raw / "bonn" / "rgbd_bonn_dataset", "Bonn rgbd_bonn_dataset")
    # stage only the five protocol sequences: the extractor takes every dir
    selected = raw / "bonn" / "selected"
    selected.mkdir(exist_ok=True)
    for seq in BONN_SEQUENCES:
        target = _require(src / f"rgbd_bonn_{seq}", f"Bonn sequence {seq}")
        link = selected / f"rgbd_bonn_{seq}"
        if not link.exists():
            os.symlink(target, link)
    dataset_extract_bonn.extract_bonn(
        root=str(selected), depth_root=str(selected), saved_dir=str(out) + "/",
        sample_len=-1, datatset_name="bonn",
    )


def prepare_scannet(raw: Path, out: Path) -> None:
    root = _require(raw / "scannet" / "scans_test", "ScanNet scans_test export (extract_scannet_sens.py --out-root)")
    scenes = sorted(os.listdir(root))
    incomplete = [s for s in scenes if not (root / s / "intrinsic" / "intrinsic_depth.txt").is_file()]
    if incomplete:
        raise FileNotFoundError(f"{len(incomplete)} ScanNet scenes lack an export under {root}: {incomplete[:5]}")
    dataset_extract_scannet.extract_scannet(
        root=str(root), sample_len=-1, datatset_name="scannet", saved_dir=str(out) + "/",
    )


def prepare_nyuv2(raw: Path, out: Path, expect: int = 654) -> None:
    import h5py  # matlab v7.3 files are HDF5
    from scipy.io import loadmat

    labeled = _require(raw / "nyu" / "nyu_depth_v2_labeled.mat", "NYU labeled .mat")
    splits = _require(raw / "nyu" / "splits.mat", "NYU splits.mat")
    test_idx = loadmat(str(splits))["testNdxs"].reshape(-1).astype(int) - 1  # 1-based
    if len(test_idx) != expect:
        raise ValueError(f"expected {expect} NYU test indices, got {len(test_idx)}")
    name = "nyuv2"
    a, b, c, d = NYU_CROP
    entries = []
    with h5py.File(str(labeled), "r") as f:
        images = f["images"]  # (N, 3, 640, 480) as stored
        depths = f["depths"]  # (N, 640, 480)
        for i in sorted(test_idx.tolist()):
            rgb = np.asarray(images[i]).transpose(2, 1, 0)  # -> (480, 640, 3)
            depth = np.asarray(depths[i]).transpose(1, 0)  # -> (480, 640) metres
            seq = f"test_{i:04d}"
            out_img = out / name / seq / "rgb" / f"{i:04d}.png"
            out_depth = out / name / seq / "depth" / f"{i:04d}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_depth.parent.mkdir(parents=True, exist_ok=True)
            # RGB pre-cropped like copy_crop_files; depth full-size, the eval
            # crops it (VDA's eval.py a/b/c/d), exactly as for their nyuv2
            Image.fromarray(rgb[a:b, c:d]).save(out_img)
            scaled = np.clip(np.round(depth.astype(np.float64) * NYU_FACTOR), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out_depth), scaled)
            rel_img = str(out_img.relative_to(out / name))
            rel_depth = str(out_depth.relative_to(out / name))
            entries.append({seq: [{"image": rel_img, "gt_depth": rel_depth, "factor": NYU_FACTOR}]})
    with open(out / MANIFEST[name], "w") as f:
        json.dump({name: entries}, f, indent=4)


PREPARE = {
    "sintel": prepare_sintel,
    "kitti": prepare_kitti,
    "bonn": prepare_bonn,
    "scannet": prepare_scannet,
    "nyuv2": prepare_nyuv2,
}


def cameras_for(out: Path, name: str, raw: Path) -> dict[str, int]:
    """GT cameras onto every manifest of a dataset -- the short one and, where
    VDA's extractor wrote it, the 500-frame one (the ScanNet TAE manifest
    already carries VDA's own; it is left alone)."""
    stats = attach_cameras(out, name, MANIFEST[name], raw)
    if name in MANIFEST_500 and (out / MANIFEST_500[name]).is_file():
        long = attach_cameras(out, name, MANIFEST_500[name], raw)
        stats = {k: stats[k] + long[k] for k in stats}
    return stats


def check_manifest(out: Path, name: str) -> int:
    """Sequences in the manifest and that every referenced file exists."""
    path = out / MANIFEST[name]
    with open(path) as f:
        data = json.load(f)
    seqs = data[name]
    missing = 0
    for entry in seqs:
        for seq_name, frames in entry.items():
            if not frames:
                raise ValueError(f"{path}: sequence {seq_name!r} lists no frames (wrong tree layout?)")
            for fr in frames:
                for key in ("image", "gt_depth"):
                    if not (out / name / fr[key]).is_file():
                        missing += 1
    if missing:
        raise FileNotFoundError(f"{path}: {missing} referenced files missing")
    return len(seqs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # no `choices`: argparse validates the empty default of nargs="*" against
    # them and rejects it, so the check is done by hand below
    ap.add_argument("datasets", nargs="*", help=f"any of {list(PREPARE)}; default: all")
    ap.add_argument("--raw", default=os.environ.get("EVAL_RAW", DEFAULT_RAW))
    ap.add_argument("--out", default=os.environ.get("EVAL_OUT", DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="rebuild a dataset whose manifest exists")
    ap.add_argument(
        "--cameras", action="store_true",
        help="only (re)attach GT cameras to the existing manifests; no extraction",
    )
    ap.add_argument(
        "--nyu-expect", type=int, default=654,
        help="NYU test-split size the .mat must yield (smoke tests on a stub .mat lower it)",
    )
    args = ap.parse_args()
    if not args.datasets:
        args.datasets = list(PREPARE)
    unknown = [d for d in args.datasets if d not in PREPARE]
    if unknown:
        ap.error(f"unknown datasets {unknown}; choose from {list(PREPARE)}")
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    install_shims()
    failed = []
    for name in args.datasets:
        manifest = out / MANIFEST[name]
        if args.cameras:
            if not manifest.is_file():
                print(f"== {name}: no manifest to attach cameras to ({manifest})", file=sys.stderr)
                failed.append(name)
                continue
            try:
                print(f"== {name}: cameras {cameras_for(out, name, raw)}", flush=True)
            except Exception as e:
                failed.append(name)
                print(f"== {name}: cameras FAILED: {e!r}", file=sys.stderr, flush=True)
            continue
        if manifest.is_file() and not args.force:
            print(f"== {name}: manifest exists ({check_manifest(out, name)} sequences), skipping")
            continue
        print(f"== {name}: extracting into {out / name}", flush=True)
        try:
            if name == "nyuv2":
                prepare_nyuv2(raw, out, expect=args.nyu_expect)
            else:
                PREPARE[name](raw, out)
            print(f"== {name}: {check_manifest(out, name)} sequences -> {manifest}", flush=True)
            print(f"== {name}: cameras {cameras_for(out, name, raw)}", flush=True)
        except Exception as e:  # keep going: one dataset's raw layout problem should not cost the others
            failed.append(name)
            print(f"== {name}: FAILED: {e!r}", file=sys.stderr, flush=True)
    if failed:
        print(f"failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
