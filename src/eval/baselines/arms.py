"""Baseline arms of the video-depth benchmark: one small interface, one adapter
per vendored model (third_party/*/README_VENDORED.md).

    arm = build_arm("vda", device, uses=[("sintel", "published")])
    pred = arm.predict(frames)            # [S,H,W,3] uint8 RGB -> [S,h,w] float32

Every adapter calls the model's OWN inference entry point with the settings
its paper evaluates at, so resizing, windowing and caching are upstream's, not
ours. `predict` returns the model's native output (arm.info.output: "depth"
in metres or affine-invariant "disparity"), registered to the full input
frame at whatever resolution the model produced; the caller resizes to GT.

RGB-only arms never see sparse depth: `prompt` is accepted by the prompted
arm alone, and passing one to any other arm raises.

build_arm is the only way to get an arm, and it goes through the reproduction
gate (record.require_allowed): an unverified arm is built only for the
(dataset, protocol) pairs its own paper reports.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from scipy.ndimage import distance_transform_edt

from eval.baselines import record as R

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
# Released weights live on Lustre, never in the NFS home (quota) and never in
# the default HF cache there.
WEIGHTS_ROOT = Path(
    os.environ.get(
        "BASELINE_WEIGHTS",
        "/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd/baselines",
    )
)

Prompt = tuple[np.ndarray, np.ndarray]  # (depth [S,h,w] metres, valid [S,h,w] bool)


@dataclass(frozen=True)
class ArmInfo:
    name: str
    output: str  # "depth" (metres) | "disparity" (affine-invariant inverse depth)
    causal: bool  # frame t depends on frames <= t only
    prompted: bool  # consumes sparse/low-res metric depth


@contextlib.contextmanager
def _vendored(root: Path, *shadowed: str):
    """Import from a vendored repo without leaving it importable.

    `root` is on sys.path only inside the block. Upstream top-level names too
    generic to keep (`utils`, `models`) are passed as `shadowed`: whatever the
    process had under those names is restored afterwards and the vendored
    modules are dropped from sys.modules -- the objects imported inside the
    block keep working, because Python modules hold their imports by
    reference. Only safe for packages that import eagerly, which the vendored
    ones do (their only lazy imports are relative, inside non-shadowed
    packages)."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"vendored code missing: {root} (see {root.name}/README_VENDORED.md)"
        )

    def _mine(name: str) -> bool:
        return name.split(".")[0] in shadowed

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _mine(k)}
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if _mine(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def weights_dir(arm: str) -> Path:
    return WEIGHTS_ROOT / arm


def weights_file(arm: str, record: dict | None = None) -> Path:
    """Local path of the arm's released checkpoint (a directory for a
    diffusers-layout repo). Raises with the fetch command when absent."""
    entry = R.arm_entry(record or R.load(), arm)
    name = entry["weights"]["file"]
    path = weights_dir(arm) if name.startswith("(") else weights_dir(arm) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{arm} weights missing: {path}. On a login node run "
            f"`python src/bench_baselines.py fetch --arm {arm}`"
        )
    return path


def fetch_weights(arm: str, record: dict | None = None) -> Path:
    """Download the arm's released weights to Lustre (login node: compute
    nodes have no outbound network on most partitions)."""
    entry = R.arm_entry(record or R.load(), arm)
    repo, name = entry["weights"]["hf_repo"], entry["weights"]["file"]
    dst = weights_dir(arm)
    dst.mkdir(parents=True, exist_ok=True)
    if name.startswith("("):  # whole repo (diffusers layout)
        snapshot_download(repo_id=repo, local_dir=dst)
    else:
        hf_hub_download(repo_id=repo, filename=name, local_dir=dst)
    return weights_file(arm, record)


def _check_frames(frames: np.ndarray) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"frames must be [S,H,W,3] uint8 RGB, got {frames.shape} {frames.dtype}"
        )


def _no_prompt(info: ArmInfo, prompt: Prompt | None) -> None:
    if prompt is not None:
        raise ValueError(
            f"{info.name} is RGB-only: it must never be handed sparse depth"
        )


# ---------------------------------------------------------------------------
# Video Depth Anything (offline, 32-frame windows) and its metric variant
# ---------------------------------------------------------------------------
_VDA_VITL = {
    "encoder": "vitl",
    "features": 256,
    "out_channels": [256, 512, 1024, 1024],
}


def _vda_class():
    with _vendored(THIRD_PARTY / "video_depth_anything", "utils"):
        from video_depth_anything.video_depth import VideoDepthAnything
    return VideoDepthAnything


class VDAArm:
    """benchmark/infer/infer.py's call, verbatim: input_size 518, fp32,
    target_fps 1 (unused by the model). The relative model returns disparity
    aligned across windows; the metric one returns metres."""

    def __init__(self, device: torch.device, metric: bool = False):
        name = "vda_metric" if metric else "vda"
        self.info = ArmInfo(
            name, "depth" if metric else "disparity", causal=False, prompted=False
        )
        model = _vda_class()(**_VDA_VITL, metric=metric)
        model.load_state_dict(
            torch.load(weights_file(name), map_location="cpu"), strict=True
        )
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, frames: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        out, _ = self.model.infer_video_depth(
            frames, 1, input_size=518, device=self.device.type, fp32=True
        )
        return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Online Video Depth Anything (causal, per-frame cache)
# ---------------------------------------------------------------------------
_OVDA_ROOT = THIRD_PARTY / "ovda"


def _ovda_class():
    with _vendored(_OVDA_ROOT / "src", "models"):
        from models.video_depth import onlineVideoDepthAnything
    return onlineVideoDepthAnything


class OVDAArm:
    """The paper's context-16 model through its own infer_video_depth: one
    frame per forward, the cache carried inside. fp32 (see README_VENDORED)."""

    def __init__(self, device: torch.device):
        import yaml

        self.info = ArmInfo("ovda", "disparity", causal=True, prompted=False)
        with open(_OVDA_ROOT / "configs" / "oVDA_c16.yaml") as f:
            net = yaml.safe_load(f)["net"]
        model = _ovda_class()(**net)
        model.load_state_dict(torch.load(weights_file("ovda"), map_location="cpu"))
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, frames: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        out = self.model.infer_video_depth(
            frames.astype(np.float32) / 255.0,
            device=str(self.device),
            preprocess_device="cpu",
            input_size=518,
            fp32=True,
        )
        return np.asarray(out[0], dtype=np.float32)  # [1,S,H,W] -> [S,H,W]


# ---------------------------------------------------------------------------
# Depth Any Video (diffusion; its own venv, GPU only)
# ---------------------------------------------------------------------------
class DAVArm:
    """run_infer.py's pipeline call at its defaults (3 denoising steps,
    max_resolution 1024), one pass, seed fixed per sequence. Refuses the two
    cases where upstream's output is not registered to the input frames
    (README_VENDORED): a centre crop to a multiple of 32, and clips longer
    than one window, whose key frames the long-video path drops."""

    NUM_FRAMES = 32
    MAX_RESOLUTION = 1024

    def __init__(self, device: torch.device):
        from diffusers import (
            AutoencoderKLTemporalDecoder,
            FlowMatchEulerDiscreteScheduler,
        )

        self.info = ArmInfo("dav", "disparity", causal=False, prompted=False)
        with _vendored(THIRD_PARTY / "depth_any_video"):
            from dav.models import UNetSpatioTemporalRopeConditionModel
            from dav.pipelines import DAVPipeline
            from dav.utils import img_utils
        base = str(weights_file("dav"))
        self.pipe = DAVPipeline(
            vae=AutoencoderKLTemporalDecoder.from_pretrained(base, subfolder="vae"),
            unet=UNetSpatioTemporalRopeConditionModel.from_pretrained(
                base, subfolder="unet"
            ),
            unet_interp=UNetSpatioTemporalRopeConditionModel.from_pretrained(
                base, subfolder="unet_interp"
            ),
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                base, subfolder="scheduler"
            ),
        ).to(device)
        self.img_utils = img_utils
        self.device = device

    def predict(self, frames: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        if len(frames) > self.NUM_FRAMES:
            raise ValueError(
                f"dav: {len(frames)} frames > one {self.NUM_FRAMES}-frame window; the "
                "long-video path drops key frames and is not wired up"
            )
        resized = self.img_utils.imresize_max(list(frames), self.MAX_RESOLUTION)
        cropped = self.img_utils.imcrop_multi(resized)
        if cropped[0].shape != resized[0].shape:
            raise ValueError(
                f"dav: {resized[0].shape[:2]} is not a multiple of 32 on both sides; "
                "upstream would centre-crop and the output would not cover the frame"
            )
        image = torch.from_numpy(
            np.ascontiguousarray([f.transpose(2, 0, 1) / 255.0 for f in cropped])
        ).to(self.device)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16):
            out = self.pipe(
                image,
                num_frames=self.NUM_FRAMES,
                num_overlap_frames=6,
                num_interp_frames=16,
                decode_chunk_size=16,
                num_inference_steps=3,
            )
        return np.asarray(out.disparity, dtype=np.float32)


# ---------------------------------------------------------------------------
# Prompt Depth Anything (per frame, prompted, metric)
# ---------------------------------------------------------------------------
PROMPTDA_REPO = os.environ.get("PROMPTDA_REPO", str(THIRD_PARTY / "promptda"))
PROMPTDA_CKPT_DEFAULT = os.environ.get(
    "PROMPTDA_CKPT", "depth-anything/prompt-depth-anything-vitl"
)

# Far cutoff for a prompt measurement, matching the ONNX export graph's
# depth_max (streamvggt/export/wrapper.py) so both deployment paths call the
# same pixels valid. Upstream PromptDA uses 1000 m; nothing in these datasets
# reaches either bound (uint16-millimetre PNGs cap at 65.535 m, SPOT's float32
# depth tops out near 6 m), so this only fires on garbage.
PROMPT_DEPTH_MAX_M = 100.0


def load_local_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Overlay a local fine-tuned checkpoint onto an already-built model.

    Ported from PromptDA's run_inference.py (`load_local_checkpoint`): accepts
    .safetensors or a torch file, unwraps a "model_state"/"state_dict" wrapper,
    strips DataParallel's "module." prefix, and loads non-strict so a partial
    fine-tune (e.g. depth head only) applies cleanly.

    Deviation from upstream, deliberate: upstream only WARNS when every key is
    missing, which silently keeps the base weights (or random ones) and looks
    like a successful load. That case is fatal here.
    """
    print(f"Attempting to load checkpoint from {checkpoint_path} with strict={strict}")
    if checkpoint_path.endswith(".safetensors"):
        model_sd = load_file(checkpoint_path, device=str(device))
    else:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state" in ckpt:
            model_sd = ckpt["model_state"]
        elif "state_dict" in ckpt:
            model_sd = ckpt["state_dict"]
        else:
            model_sd = ckpt  # assume a bare state_dict
    model_sd = {k[7:] if k.startswith("module.") else k: v for k, v in model_sd.items()}

    missing, unexpected = model.load_state_dict(model_sd, strict=strict)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
    if len(missing) == len(model.state_dict()):
        raise SystemExit(
            f"{checkpoint_path!r} shares no parameter names with PromptDA; "
            "nothing was loaded"
        )
    print("Checkpoint loaded.")


def load_promptda(ckpt: str, device, local_ckpt: str | None = None) -> torch.nn.Module:
    """Build the vendored PromptDA in eval mode, mirroring PromptDA's own
    run_inference.py: `from_pretrained` for the base weights, then an optional
    non-strict overlay of a local fine-tune.

    The base checkpoint is resolved here (local path, else HF hub download) and
    its existence asserted BEFORE construction: PromptDA.load_checkpoint only
    warns on a missing file and would silently run with random depth-head
    weights. Post-resolution, from_pretrained takes its local-path branch, so
    the loaded weights are exactly what upstream loads.
    """
    # promptda is a namespace package (no __init__.py) that resolves its
    # torchhub sibling relative to itself, so the vendor DIR goes on sys.path
    # for the import; the name is specific enough to stay in sys.modules.
    with _vendored(Path(PROMPTDA_REPO)):
        from promptda.promptda import PromptDA

    if os.path.exists(ckpt):
        resolved = ckpt
    else:
        try:
            resolved = hf_hub_download(
                repo_id=ckpt, repo_type="model", filename="model.ckpt"
            )
        except Exception as exc:
            raise SystemExit(
                f"could not resolve PromptDA checkpoint {ckpt!r}: {exc}\n"
                "pre-download on a login node with\n"
                '  python -c "from huggingface_hub import hf_hub_download; '
                f"hf_hub_download('{ckpt}', 'model.ckpt')\"\n"
                "or pass --promptda-ckpt /abs/path/to/model.ckpt"
            ) from exc
    if not os.path.exists(resolved):
        raise SystemExit(f"PromptDA checkpoint {resolved!r} does not exist")

    print(f"PROMPTDA model: loading weights {resolved}")
    model = PromptDA.from_pretrained(resolved).to(device)
    if local_ckpt is not None:
        if not os.path.exists(local_ckpt):
            raise SystemExit(f"PromptDA local checkpoint {local_ckpt!r} does not exist")
        load_local_checkpoint(model, local_ckpt, device, strict=False)
    return model.eval()


def infill_sparse_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor infill of a sparse depth map ([H,W] meters, [H,W] bool).

    PromptDA normalizes the prompt by its min/max over the WHOLE map, and the
    fusion blocks bilinearly resample it, so sparse zeros both wreck the
    normalization range and bleed into valid measurements. PromptDA's own
    pipelines densify first with exactly this distance-transform gather.

    Upstream derives validity as (d > 0) & (d < 1000) rather than from a sensor
    mask; the far cutoff is folded in here (at PROMPT_DEPTH_MAX_M, inclusive
    like the export graph's) so one garbage far pixel cannot set the
    normalization max for the whole frame."""
    mask = mask & (depth <= PROMPT_DEPTH_MAX_M)
    if not mask.any():
        raise SystemExit(
            "frame has 0 valid sparse-depth pixels; PromptDA has no prompt "
            "to normalize against"
        )
    if mask.all():
        return depth
    _, idx = distance_transform_edt(~mask, return_indices=True)
    out = depth.copy()
    out[~mask] = depth[idx[0][~mask], idx[1][~mask]]
    return out


class PromptDAArm:
    """Per-frame PromptDA with upstream's input sizing (io_wrapper.load_image
    at its default max_size): the long side capped at 1008, each side floored
    to a multiple of 14, INTER_AREA. The prompt is densified by
    infill_sparse_depth and passed at its own resolution (the DPT head
    resamples it per scale). Returns metric depth at the model's resolution;
    the caller resizes it to whatever it scores at."""

    MAX_SIZE = 1008

    def __init__(self, device: torch.device):
        self.info = ArmInfo("promptda", "depth", causal=True, prompted=True)
        self.model = load_promptda(str(weights_file("promptda")), device)
        self.device = device

    def predict(self, frames: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        _check_frames(frames)
        if prompt is None:
            raise ValueError("promptda needs a prompt (depth, valid)")
        depth, valid = prompt
        if len(depth) != len(frames) or depth.shape != valid.shape:
            raise ValueError(
                f"prompt {depth.shape} / {valid.shape} does not match {len(frames)} frames"
            )
        H, W = frames.shape[1:3]
        scale = min(1.0, self.MAX_SIZE / max(H, W))
        h, w = int(H * scale // 14 * 14), int(W * scale // 14 * 14)
        out = []
        for i, frame in enumerate(frames):
            img = frame.astype(np.float32) / 255.0
            if (h, w) != (H, W):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            dense = infill_sparse_depth(depth[i].astype(np.float32), valid[i])
            pred = self.model.predict(
                torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device),
                torch.from_numpy(dense)[None, None].to(self.device),
            )
            out.append(pred[0, 0].float().cpu().numpy())
        return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------------------
# registry + gate
# ---------------------------------------------------------------------------
ARMS = {
    "vda": lambda device: VDAArm(device),
    "vda_metric": lambda device: VDAArm(device, metric=True),
    "dav": DAVArm,
    "ovda": OVDAArm,
    "promptda": PromptDAArm,
}


def build_arm(
    name: str,
    device: torch.device,
    uses: list[tuple[str, str]],
    record: dict | None = None,
):
    """The arm, built only if the reproduction record opens every (dataset,
    protocol) in `uses` to it (record.require_allowed)."""
    if name not in ARMS:
        raise KeyError(f"unknown baseline arm {name!r}; known: {sorted(ARMS)}")
    if not uses:
        raise ValueError("build_arm needs the (dataset, protocol) pairs it will run")
    R.require_allowed(record or R.load(), name, uses)
    return ARMS[name](device)
