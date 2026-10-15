# Depth Anything V2 -- vendored metric-depth evaluation

Source: https://github.com/DepthAnything/Depth-Anything-V2 at commit
`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` (2026-03-24), fetched 2026-10-10.
License: Apache-2.0 (`LICENSE`, copied from the repository root).

What is used, and from where:

| file | used by | for |
|---|---|---|
| `metric_depth/util/metric.py` | `src/eval/da_metric_paper.py` | `eval_depth`: AbsRel, delta1, ... over the valid pixels of one image |
| `metric_depth/dataset/splits/kitti/val.txt` | `src/eval/da_metric_paper.py` | the 652 KITTI Eigen test frames with annotated GT (same frames and order as ZoeDepth's `kitti_eigen_test_files_with_gt.txt`) |
| `metric_depth/train.py` (eval block, argparse defaults) | reference only | re-stated in `da_metric_paper.py`: prediction bilinear (align_corners) to GT, valid = GT > 0 and min_depth <= GT <= max_depth (0.001 / 80, the README's outdoor setting), no clip, images with < 10 valid pixels skipped, per-image metrics averaged |

Nothing here is edited. `val.txt` lists the authors' absolute paths
(`/mnt/bn/liheyang/Kitti/...`); `da_metric_paper.py` maps that prefix to our
KITTI root. Not vendored: the training code, the Hypersim / VKITTI2 loaders and
`depth_anything_v2/` (DA3 is the model here).
