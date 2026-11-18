# TTT3R long-sequence preparation: vendored for reference

**Source:** https://github.com/Inception3D/TTT3R at commit `edd6d8c000aaf2ef0f588403e1b3bd3300a54cc4`, files `datasets_preprocess/long_prepare_bonn.py` and `long_prepare_kitti.py`, fetched verbatim on 2026-10-26.
**License:** `LICENSE`: CUT3R's, Creative Commons BY-NC-SA 4.0.

InfiniteVGGT and OVGGT evaluate on the long Bonn and KITTI sets these scripts write: `long_bonn_s1` and `long_kitti_s1`.

`src/eval/cut3r_video_paper.py` restates their frame selection on our raw trees instead of copying files:
- **Bonn:** RGB and depth each sorted, `[30 : 30 + n]`, n capped by what exists.
- **KITTI:** the first N annotated `val` frames of each drive, with the raw `image_02` frames of the same names.

These files are not executed; they are kept so the restatement can be checked against them.
