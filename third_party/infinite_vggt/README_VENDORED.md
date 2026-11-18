# InfiniteVGGT: vendored inference and evaluation code

**Source:** https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT at commit `7f9a5a26c2e2f9603151d1dcc61fd402ef3f6a27` (2026-04-19), fetched verbatim on 2026-10-26.
**Paper:** arXiv:2601.02281.
**License:** Apache-2.0 (`LICENSE.txt`, copied from the repository root).
**Weights:** none of its own. It is training-free on StreamVGGT's released `checkpoints.pth` (Hugging Face `lch01/StreamVGGT`).

Only `src/` is vendored, plus `README.md` and `LICENSE.txt`. Nothing is modified.

| File | Used by | For |
|---|---|---|
| `src/streamvggt/**` | `src/eval/baselines/arms.py::InfiniteVGGTArm` | the model: StreamVGGT with a rolling KV memory pruned by key diversity (`layers/attention.py`, `models/aggregator.py`); default `total_budget` is 1,200,000 tokens; the first frame is kept as anchor |
| `src/dust3r/inference.py::loss_of_one_batch` | `InfiniteVGGTArm` | its streaming loop (fp16 autocast) |
| `src/dust3r/utils/image.py` (`_resize_pil_image`, `ImgNorm`) | `InfiniteVGGTArm` | `load_images_for_eval`'s sizing |
| `src/eval/video_depth/**` | reference | the evaluation `src/eval/cut3r_video_paper.py` follows (byte-identical to OVGGT's `tools.py` and `eval_depth.py` apart from paths) |

**Imports:** the package names `dust3r`, `croco`, `streamvggt` and `vggt` collide with ours. `arms._fork_parts` imports them inside `arms._vendored`, which hides them again afterwards.

**Not used:** `demo_viser.py`, `run_inference.py`, `assets/` (88 MB of demo media), `lib/`, and `datasets_preprocess/` (training data).
