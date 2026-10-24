# Online Video Depth Anything (oVDA) -- vendored inference code

Source: https://github.com/FriedFeid/OnlineVideoDepthAnything at commit
`4173c24e22f528062a0cf8ae235c58a5d4376a74` (2026-03-09), fetched verbatim on
2026-10-03. Paper: arXiv:2510.09182.

License: mixed (`LICENSE`, copied from the repository root). Files marked as
derived from Video Depth Anything are Apache-2.0; everything else is the
**Non-Commercial Software License -- Heidelberg University Version 1.0
(NC-SA-UHDV1.0)**: free for non-commercial research with attribution to
Heidelberg University and the authors, commercial use prohibited, and
modifications must carry the same license. Nothing here is modified, and this
directory must stay that way; the license does not extend to the rest of
MeTRICS, but it does restrict what this directory may be used for.

What is used, and from where:

| file | used by | for |
|---|---|---|
| `src/models/**` | `src/eval/baselines/arms.py::OVDAArm` | the model and its own `infer_video_depth` (per-frame cache loop, resize policy) |
| `configs/oVDA_c16.yaml` | `OVDAArm` | the network arguments of the paper's context-16 model |
| `evaluation/protocol.py` | `src/eval/ovda_paper.py` | the authors' scoring: per-sequence inverse-depth fit (first frame or all), [0, 80] m clip, GT < 80 m scored from frame 1, pixel-weighted totals |
| `evaluation/data_sorting.py` | `src/eval/ovda_paper.py` | the authors' frame pairing: Sintel final pass, KITTI 138 train drives x 2 cameras, Bonn timestamp association |
| `evaluation/README.md`, `sequences.csv`, `bonn_pairs.json` | reference, and `datasets_download/download_kitti_ovda.sh` (drive list) | the protocol in prose, the exact sequence lists and order, every Bonn pair |

`src/utils/align_utils.py` was vendored earlier and removed again (2026-10-08): the
evaluation package below superseded our re-implementation of the paper's alignment
around it.

Not vendored: `run.py`, the notebooks, `src/utils/loading_utils.py` (video and
TIFF I/O, needs `tifffile`), `src/build_ONNX/` and `configs/oVDA_c8.yaml`.

Weights are NOT vendored: `oVDA_c16.pth` from https://huggingface.co/FriedFeid/oVDA
(same NC-SA-UHDV1.0 license), fetched to Lustre by
`python src/bench_baselines.py fetch --arm ovda`.

Imports: the model package is `src/models` and uses relative imports only, so
the arm puts `src/` on `sys.path` just for the import and then drops the
generic top-level name `models` from `sys.modules` again
(`arms.py::_vendored`). It needs `easydict`, which the `metrics` venv lacks
(see the `run-baselines` skill).

Known upstream behaviour handled from outside:

- `infer_video_depth(fp32=False)` calls `self.half()` and leaves the model in
  half precision; the arm always passes `fp32=True`.
- The paper lists `329x924` as the Sintel processing resolution, which is not
  a multiple of 14; the code produces `392x924` at the default
  `input_size=518`, and that is what runs.

`evaluation/` was sent by the first author on 2026-10-08 as the reference for
the paper's numbers (`ovda-evaluation-handover`); it carries the same mixed
NC-SA-UHDV1.0 / Apache-2.0 licence as this repository (`LICENSE`, identical
file) and is checked in verbatim. Their protocol reproduces the paper's
Sintel tables on our saved predictions (0.3799 / 0.5477 first-frame, 0.2941 /
0.6035 global vs 0.380 / 0.548 and 0.294 / 0.604).
