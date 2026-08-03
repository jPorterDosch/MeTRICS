# Vendored PromptDA (Prompt Depth Anything)

Copied verbatim from `/oscar/home/jdosch/PromptDA` (branch `inference-script`,
commit `d289d43`), itself derived from
https://github.com/DepthAnything/PromptDA. Only the subset needed for
inference is vendored:

- `promptda/promptda.py` — model + `predict()` API
- `promptda/model/{blocks,config,dpt}.py` — DPT head
- `promptda/utils/logger.py`
- `torchhub/facebookresearch_dinov2_main` — vendored DINOv2 backbone code
  (loaded with `torch.hub.load(source='local', pretrained=False)`;
  `promptda/promptda.py` resolves it relative to this directory, so the
  `promptda/` and `torchhub/` siblings must stay together)

Do not edit these files here; update the source repo and re-copy.
Used by `src/visualize_depth.py` / `src/visualize_spot.py` via the
`--promptda` flag (this directory is inserted on `sys.path`; `promptda` is a
namespace package, no `__init__.py`).

Weights are NOT vendored. Default checkpoint resolves via
`hf_hub_download("depth-anything/prompt-depth-anything-vitl", "model.ckpt")`
(pre-download on a login node) or pass `--promptda-ckpt /path/to/model.ckpt`.
