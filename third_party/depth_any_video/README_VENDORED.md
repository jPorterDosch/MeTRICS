# Depth Any Video -- vendored inference code

Source: https://github.com/Nightmare-n/DepthAnyVideo at commit
`3592f8c5e427327a31cf1f3775e69cc3f2f26309` (2024-12-04), fetched verbatim on
2026-10-03. Paper: arXiv:2410.10815.

License: **CC BY-NC 4.0** (`LICENSE`, copied from the repository root):
non-commercial use, with attribution. Nothing here is modified.

What is used, and from where:

| file | used by | for |
|---|---|---|
| `dav/pipelines/dav_pipeline.py` | `src/eval/baselines/arms.py::DAVArm` | `DAVPipeline`, the flow-matching sampler and VAE decode |
| `dav/models/**` | `DAVArm` | `UNetSpatioTemporalRopeConditionModel` |
| `dav/utils/img_utils.py` | `DAVArm` | `imresize_max` / `imcrop_multi`, upstream's own input sizing |

Not vendored: `run_infer.py`, `app.py`, `predict.py`, `cog.yaml`, the demo
media and `dav/utils/eval_utils.py` (their metric and TAE definitions; ours
are in `src/eval/protocols.py`).

Weights are NOT vendored: https://huggingface.co/hhyangcs/depth-any-video
(diffusers layout: `unet`, `unet_interp`, `vae`, `scheduler`), fetched to
Lustre by `python src/bench_baselines.py fetch --arm dav`.

Environment: upstream pins `torch==2.4.0 diffusers==0.30.3
transformers==4.43.2 huggingface-hub==0.24.2`, none of which match the
`metrics` venv (no `diffusers`, `transformers` 5.0, `huggingface_hub` 1.22),
so this arm runs from its own venv (see the `run-baselines` skill).

Known upstream behaviour handled from outside:

- No ensembling is released, although the paper's Table 2 uses an ensemble
  of 20. The arm runs one pass with a fixed seed; the reproduction record
  says so next to the targets.
- `imcrop_multi` centre-crops each side to a multiple of 32, and the
  long-video path drops the first and last key frames, so the output is not
  always registered to the input frame set. The arm refuses both cases
  instead of returning a mis-registered prediction.
- `DAVPipeline.__call__` casts the VAE to float16 in place: GPU only.
