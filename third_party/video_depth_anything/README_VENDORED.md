# Video Depth Anything -- vendored benchmark and model code

Source: https://github.com/DepthAnything/Video-Depth-Anything at commit
`4f5ae23172ba60fd7bc11ef671cca678842c7072` (2025-10-07): the `benchmark/`
subtree fetched verbatim on 2026-09-22, the model code (`video_depth_anything/`
without `video_depth_stream.py`, and `utils/util.py`) on 2026-10-03 from the
same commit. License: Apache-2.0 (`LICENSE`, copied from the repository
root). The released weights are CC-BY-NC-4.0 and are not vendored.

What is used, and from where:

| file | used by | for |
|---|---|---|
| `benchmark/eval/metric.py` | `src/eval/protocols.py` | AbsRel / RMSE / delta1 with VDA's per-frame reduction |
| `benchmark/eval/eval_tae.py` | `src/eval/protocols.py` | `tae_torch`, VDA's reprojection TAE |
| `benchmark/eval/eval.py` | reference only | the per-video disparity alignment is re-stated in `protocols.vda_align_disparity` (their function reads files; ours takes arrays) and the per-dataset constants in `src/eval/vda_benchmark.py::SPECS` |
| `benchmark/dataset_extract/*.py` | `datasets_preprocess/prepare_vda_benchmark.py` | building the benchmark tree from the raw datasets |
| `video_depth_anything/**` | `src/eval/baselines/arms.py::VDAArm` | the model and its own `infer_video_depth` (32-frame windows, key-frame overlap, window alignment), for VDA-L and Metric-VDA-L |
| `utils/util.py` | imported by `video_depth_anything/video_depth.py` | `compute_scale_and_shift`, `get_interpolate_frames` |
| `benchmark/infer/infer.py` | reference only | the settings the arm repeats: `input_size=518`, `fp32=True`, `target_fps=1` |

Nothing in this directory is edited. The extraction scripts have defects as
shipped (BGR colour writes, a wrong keyword in the Bonn script, unscaled
Sintel depth, an undefined function in the NYU script); the preparer works
around them from the outside and documents each one in its module docstring.
`eval_tae.py` is loaded by path, not imported: its directory also holds an
`eval.py` that would shadow the `src/eval` package.

Model code imports: `video_depth.py` does `from utils.util import ...`, a
top-level name too generic to leave importable, so the arm puts this
directory on `sys.path` only for the import and drops `utils` from
`sys.modules` afterwards (`arms.py::_vendored`). `dpt_temporal.py` needs
`easydict`, which the `metrics` venv lacks (see the `run-baselines` skill).
Not vendored: `run.py`, `run_streaming.py`, `video_depth_stream.py` (the
experimental training-free streaming mode -- the online baseline is oVDA,
`third_party/ovda`), `app.py`, `loss/`, `utils/dc_utils.py`.
