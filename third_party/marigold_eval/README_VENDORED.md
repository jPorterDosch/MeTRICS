# Marigold -- vendored single-image evaluation

Source: https://github.com/prs-eth/Marigold at commit
`2bfbdeae5d10a50b71f1ba20c865d46e480f6010` (2026-09-06), fetched 2026-10-13.
License: Apache-2.0 (`LICENSE.txt`, copied from the repository root). Every
file is byte-identical to that commit.

What is used, and from where:

| file | used by | for |
|---|---|---|
| `src/util/alignment.py` | `src/eval/marigold_paper.py` | `align_depth_least_square`, `depth2disparity`, `disparity2depth` |
| `src/util/metric.py` | `src/eval/marigold_paper.py` | `abs_relative_difference`, `delta1_acc` |
| `data_split/kitti_depth/eigen_test_files_with_gt.txt` | `src/eval/marigold_paper.py` | KITTI Eigen test list (697 lines, 652 with GT) |
| `data_split/scannet_depth/scannet_val_sampled_list_800_1.txt` | `src/eval/marigold_paper.py` | the fixed 800-frame ScanNet validation sample |
| `data_split/nyu_depth/labeled/filename_list_test.txt` | `src/eval/marigold_paper.py` | the 654 NYUv2 test frames (rgb, raw depth, filled depth) |
| `script/depth/eval.py` | reference only | the evaluation loop `marigold_paper.evaluate` re-states (theirs reads prediction files and a DataLoader) |
| `src/dataset/{base_depth_dataset,kitti_dataset,scannet_dataset,nyu_dataset}.py`, `config/dataset_depth/*.yaml` | reference only | depth ranges, raw-vs-filled GT, kb crop and Eigen masks, re-stated in `marigold_paper.py` |

Nothing here is edited. The modules are loaded by path: their package is
called `src`, which would shadow ours. The data is Marigold's evaluation tars
(README "Evaluation on test datasets"), extracted under
`/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw/marigold_eval/`, apart
from the benchmark trees. Checked on 2026-10-13: the KITTI tar's images and GT
equal our raw KITTI files on every 25th listed frame; all 800 ScanNet entries
and all 652 KITTI entries exist in the tars; all 654 NYU test entries exist in
the NYU tar, their RGB equals nyu_depth_v2_labeled.mat's test images exactly,
and their raw depth equals its rawDepths to within the tar's 1 mm step.
