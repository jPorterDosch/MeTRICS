# OVGGT: vendored inference and evaluation code

**Source:** https://github.com/VAISR/OVGGT at commit `43304e42ba823fc1622799154fc07597b66a1389` (2026-09-02), fetched verbatim on 2026-10-26.
**Paper:** arXiv:2603.05959.
**License:** `LICENSE.txt`, copied from the repository root. It states StreamVGGT's Creative Commons BY-NC-SA 4.0: non-commercial use, attribution, share-alike. Nothing here is modified, and the license restricts what this directory may be used for.
**Weights:** none of its own. It is training-free on StreamVGGT's released `checkpoints.pth` (Hugging Face `lch01/StreamVGGT`).

Only `src/` is vendored, plus `README.md` and `LICENSE.txt`.

| File | Used by | For |
|---|---|---|
| `src/ovggt/**` | `src/eval/baselines/arms.py::OVGGTArm` | the model: a constant-cost KV cache with importance-scored eviction and first-frame and history anchors; default `total_budget` is 200,000 tokens; `inference` defaults are `anchor_interval` 250, `min_anchor_interval` 100, `max_anchors` 3 |
| `src/dust3r/inference.py::loss_of_one_batch` | `OVGGTArm` | its streaming loop: bf16 autocast on compute capability >= 8, else fp16; the `eviction` keyword stays at its default, False, as `launch.py` leaves it |
| `src/dust3r/utils/image.py` (`_resize_pil_image`, `ImgNorm`) | `OVGGTArm` | `load_images_for_eval`'s sizing |
| `src/eval/video_depth/tools.py::depth_evaluation` | `src/eval/cut3r_video_paper.py` | the scoring: per-sequence scale-only IRLS fit and metrics |

**Imports:** the package names `dust3r`, `croco` and `vggt` collide with ours; `arms._fork_parts` hides them as for InfiniteVGGT.
