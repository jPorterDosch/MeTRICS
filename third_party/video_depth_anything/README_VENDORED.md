# Video Depth Anything -- vendored benchmark code

Source: https://github.com/DepthAnything/Video-Depth-Anything, `benchmark/`
subtree at commit `4f5ae23172ba60fd7bc11ef671cca678842c7072` (2025-10-07),
fetched verbatim on 2026-09-22. License: Apache-2.0 (`LICENSE`, copied from
the repository root).

What is used, and from where:

| file | used by | for |
|---|---|---|
| `benchmark/eval/metric.py` | `src/eval/protocols.py` | AbsRel / RMSE / delta1 with VDA's per-frame reduction |
| `benchmark/eval/eval_tae.py` | `src/eval/protocols.py` | `tae_torch`, VDA's reprojection TAE |
| `benchmark/eval/eval.py` | reference only | the per-video disparity alignment is re-stated in `protocols.vda_align_disparity` (their function reads files; ours takes arrays) and the per-dataset constants in `src/eval/vda_benchmark.py::SPECS` |
| `benchmark/dataset_extract/*.py` | `datasets_preprocess/prepare_vda_benchmark.py` | building the benchmark tree from the raw datasets |

Nothing in this directory is edited. The extraction scripts have defects as
shipped (BGR colour writes, a wrong keyword in the Bonn script, unscaled
Sintel depth, an undefined function in the NYU script); the preparer works
around them from the outside and documents each one in its module docstring.
`eval_tae.py` is loaded by path, not imported: its directory also holds an
`eval.py` that would shadow the `src/eval` package.
