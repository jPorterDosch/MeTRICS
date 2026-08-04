# Vendored PromptDA (Prompt Depth Anything)

Copied verbatim from `https://github.com/fmz/PromptDA` (branch `inference-script`,
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

Every vendored file is byte-identical to upstream — keep it that way, so a
`diff -r` against a fresh checkout stays the parity check.

Weights are NOT vendored. Default checkpoint resolves via
`hf_hub_download("depth-anything/prompt-depth-anything-vitl", "model.ckpt")`
(pre-download on a login node) or pass `--promptda-ckpt /path/to/model.ckpt`.
A fine-tuned checkpoint (upstream's `run_inference.py -m/--model`: safetensors
or torch, loaded non-strict) goes in via `--promptda-local-ckpt`.

`tests/promptda_parity.py` guards the match: our `load_local_checkpoint` and
sparse-prompt infill against verbatim copies of upstream's, the inputs the arm
actually hands the model (`--wiring`), plus (with `--forward --upstream
<checkout>`) a bitwise-identical vitl forward.

Known, deliberate differences from upstream's inference scripts:

- Input resolution: upstream's demo loader caps the long side at 1008 px
  (multiple of 14, `INTER_AREA`); our arm feeds the dataset's native
  518x392 frames so all three comparison arms see identical pixels.
- Upstream's `load_local_checkpoint` only warns when nothing matched; ours
  makes a zero-key-overlap checkpoint fatal.
- Prompt validity: upstream derives it as `(d > 0) & (d < 1000)`; we intersect
  the dataset/sensor mask with a 100 m cutoff (`_PROMPT_DEPTH_MAX_M`, matching
  the ONNX export graph's `depth_max`). Neither bound is reachable by real
  data here — the loaders read uint16-millimetre PNGs (65.535 m max) and
  SPOT's float32 depth tops out near 6 m — so this only catches garbage.
