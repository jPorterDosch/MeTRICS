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
    before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        yield
    except BaseException:
        # a failed import leaves half-initialised modules behind, and a retry
        # would import them as done: drop everything the block added
        for k in set(sys.modules) - before:
            del sys.modules[k]
        raise
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


def unshrunk_input_size(height: int, width: int, target: int = 518) -> int:
    """The input_size that infer_video_depth's aspect shrink maps back to
    `target`. For frames wider than 16:9 it shrinks the short side to
    input_size * 1.777 / ratio ("due to memory limitation"); KITTI's 374x1242
    goes to 280, and the metric model then sees objects too small and predicts
    depth ~1.3x too large. Raises if no size lands exactly on `target`."""
    ratio = max(height, width) / min(height, width)
    if ratio <= 1.78:  # their threshold: no shrink
        return target
    for size in range(target, int(target * ratio / 1.777) + 14 * int(ratio + 1)):
        if round(int(size * 1.777 / ratio) / 14) * 14 == target:
            return size
    raise ValueError(f"no input_size maps {height}x{width} back to {target}")


def _sdpa_attention(module: torch.nn.Module) -> None:
    """Replace one DINOv2 attention module's forward, on this instance only,
    by torch's scaled_dot_product_attention: the same softmax(q k^T / sqrt(d)) v
    without materialising the attention matrix. Without xFormers, upstream's
    MemEffAttention falls back to plain attention, which needs 39.5 GiB for
    one tensor at 518x1722 x 32 frames."""

    def forward(x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        if attn_bias is not None:
            raise ValueError("attention bias is not supported")
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

    module.forward = forward


class VDAArm:
    """benchmark/infer/infer.py's call, verbatim: input_size 518, fp32,
    target_fps 1 (unused by the model). The relative model returns disparity
    aligned across windows; the metric one returns metres.

    The metric model is run without infer_video_depth's aspect shrink
    (unshrunk_input_size): its absolute scale depends on the input
    resolution, and the README's KITTI number is only reached unshrunk
    (delta1 0.9074 unshrunk vs 0.3424 shrunk, KITTI-500, job 6373139). Its
    DINOv2 attention goes through SDPA (_sdpa_attention) so the unshrunk KITTI
    frames fit in memory. The relative model is left exactly as verified."""

    def __init__(
        self, device: torch.device, metric: bool = False, record: dict | None = None
    ):
        name = "vda_metric" if metric else "vda"
        self.info = ArmInfo(
            name, "depth" if metric else "disparity", causal=False, prompted=False
        )
        model = _vda_class()(**_VDA_VITL, metric=metric)
        model.load_state_dict(
            torch.load(weights_file(name, record), map_location="cpu"), strict=True
        )
        if metric:
            for m in model.pretrained.modules():
                if type(m).__name__ in ("Attention", "MemEffAttention"):
                    _sdpa_attention(m)
        self.model = model.to(device).eval()
        self.device = device
        self.metric = metric

    def predict(self, frames: np.ndarray, prompt: Prompt | None = None) -> np.ndarray:
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        size = unshrunk_input_size(*frames.shape[1:3]) if self.metric else 518
        # infer_video_depth takes the bare device TYPE (its autocast needs it)
        # and moves inputs to the current device of that type: pin it for the
        # call only
        with (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        ):
            out, _ = self.model.infer_video_depth(
                frames, 1, input_size=size, device=self.device.type, fp32=True
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

    def __init__(self, device: torch.device, record: dict | None = None):
        import yaml

        self.info = ArmInfo("ovda", "disparity", causal=True, prompted=False)
        with open(_OVDA_ROOT / "configs" / "oVDA_c16.yaml") as f:
            net = yaml.safe_load(f)["net"]
        model = _ovda_class()(**net)
        model.load_state_dict(
            torch.load(weights_file("ovda", record), map_location="cpu")
        )
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
    max_resolution 1024), run ENSEMBLE times with seeds 0..ENSEMBLE-1 and the
    disparities averaged. The paper evaluates with an ensemble of 20
    "following Ke et al." (Marigold); no ensembling code is released. Mean
    over members is used: Marigold's ensemble_depth (per-member affine
    alignment, then median) scored within 0.03 AbsRel of it on the first 200
    NYUv2 images (5.19 vs 5.16) at ~1 min per image of alignment.

    Two input-side paddings make the output cover exactly the input frames,
    where upstream's run_infer would not (README_VENDORED):
      * spatial: upstream centre-crops each side to a multiple of 32; the arm
        reflect-pads to the next multiple instead and crops the output back.
      * temporal: clips longer than one window go through upstream's
        long-video path (key-frame pairs 17 apart every 12 frames, predicted
        together, then 16 interpolated frames per pair), which returns only
        the frames strictly between the first and last key frame. The arm
        prepends one copy of the first frame and appends copies of the last
        until the last key frame falls after the clip (temporal_padding),
        then drops the padded frames."""

    NUM_FRAMES = 32
    MAX_RESOLUTION = 1024
    ENSEMBLE = 20

    def __init__(self, device: torch.device, record: dict | None = None):
        from diffusers import (
            AutoencoderKLTemporalDecoder,
            FlowMatchEulerDiscreteScheduler,
        )

        self.info = ArmInfo("dav", "disparity", causal=False, prompted=False)
        with _vendored(THIRD_PARTY / "depth_any_video"):
            from dav.models import UNetSpatioTemporalRopeConditionModel
            from dav.pipelines import DAVPipeline
            from dav.utils import img_utils
        base = str(weights_file("dav", record))
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

    @classmethod
    def temporal_padding(cls, n: int) -> tuple[int, int]:
        """(front, back) frame copies so that upstream's long-video path
        returns all `n` frames: its key-frame loop (dav_pipeline.py) restated,
        with num_interp_frames 16, num_overlap_frames 6."""
        if n <= cls.NUM_FRAMES:
            return 0, 0
        interp, overlap = 16, 6
        for back in range(0, 2 * (interp + 2)):
            m = n + 1 + back
            keys: list[int] = []
            for i in range(0, m, interp + 2 - overlap):
                if i + interp + 1 >= m or len(keys) >= cls.NUM_FRAMES:
                    break
                keys += [i, i + interp + 1]
            # output = padded frames keys[0]+1 .. keys[-1]-1 = original 0 .. n-1
            if keys and keys[0] == 0 and keys[-1] >= n + 1:
                return 1, back
        raise ValueError(
            f"dav: no padding makes the long-video path cover {n} frames "
            f"(at most {cls.NUM_FRAMES} key frames)"
        )

    def predict(
        self,
        frames: np.ndarray,
        prompt: Prompt | None = None,
        max_resolution: int | None = None,
    ) -> np.ndarray:
        """`max_resolution` is run_infer.py's --max_resolution (default 1024):
        Marigold's KITTI images (352x1216) are run at their own size.
        Returns disparity at the model resolution (after upstream's
        imresize_max), one map per input frame."""
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        # upstream's input sizing: the long side capped at run_infer's
        # --max_resolution (1024 unless the caller asks for another cap)
        if max_resolution is None:
            max_resolution = self.MAX_RESOLUTION
        resized = self.img_utils.imresize_max(list(frames), max_resolution)
        h, w = resized[0].shape[:2]
        ph, pw = -h % 32, -w % 32
        top, left = ph // 2, pw // 2
        padded = [
            np.pad(f, ((top, ph - top), (left, pw - left), (0, 0)), mode="reflect")
            for f in resized
        ]
        front, back = self.temporal_padding(len(padded))
        padded = [padded[0]] * front + padded + [padded[-1]] * back
        image = torch.from_numpy(
            np.ascontiguousarray([f.transpose(2, 0, 1) / 255.0 for f in padded])
        ).to(self.device)
        # fixed seeds per call, without touching the process RNG state
        devices = [self.device] if self.device.type == "cuda" else []
        members = []
        with (
            torch.random.fork_rng(devices=devices),
            torch.no_grad(),
            torch.autocast(self.device.type, dtype=torch.float16),
        ):
            for seed in range(self.ENSEMBLE):
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed_all(seed)
                out = self.pipe(
                    image,
                    num_frames=self.NUM_FRAMES,
                    num_overlap_frames=6,
                    num_interp_frames=16,
                    decode_chunk_size=16,
                    num_inference_steps=3,
                )
                members.append(np.asarray(out.disparity, dtype=np.float32))
        disp = np.mean(members, axis=0)
        # the long path already returns from the first padded frame + 1
        disp = disp[: len(frames)]
        if len(disp) != len(frames):
            raise ValueError(f"dav: {len(disp)} outputs for {len(frames)} frames")
        return np.ascontiguousarray(disp[:, top : top + h, left : left + w]).astype(
            np.float32
        )


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

    def __init__(self, device: torch.device, record: dict | None = None):
        self.info = ArmInfo("promptda", "depth", causal=True, prompted=True)
        self.model = load_promptda(str(weights_file("promptda", record)), device)
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
            if not (valid[i] & (depth[i] <= PROMPT_DEPTH_MAX_M)).any():
                # no prompt to normalise against (e.g. 1% patches that all miss
                # KITTI's LiDAR): no prediction. NaN is scored as an error by
                # every protocol and dropped by sparse_aligned's own rule.
                out.append(np.full((h, w), np.nan, np.float32))
                continue
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
# Depth Anything 3, monocular metric model (DA3METRIC-LARGE)
# ---------------------------------------------------------------------------
def load_da3(path: str):
    """DepthAnything3.from_pretrained on a local snapshot. Built inside the
    import block: the config names its classes by module path
    (cfg.create_object imports them on demand). api.py imports
    depth_anything_3.utils.export at module level, whose dependencies
    (moviepy, plyfile, ...) the metrics venv lacks; inference never exports,
    so an empty module stands in for it."""
    import types

    name = "depth_anything_3.utils.export"
    stub = sys.modules.setdefault(name, types.ModuleType(name))
    if not hasattr(stub, "export"):
        stub.export = None
    with _vendored(THIRD_PARTY / "depth_anything_3" / "src"):
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3.from_pretrained(path)


class DA3MetricArm:
    """DepthAnything3.inference with the short side at 504 (process_res=504,
    lower_bound_resize, each side then rounded to a multiple of 14): the
    paper's base resolution for the metric model (Sec. 4.4); the API's default
    upper_bound_resize puts the LONG side at 504 (KITTI AbsRel 0.097 against
    0.083, probe round 2). One image per call, with DA3's own sky
    post-process. The network output is canonical depth; metric depth =
    output * mean(fx, fy) / 300 with the intrinsics scaled to the processed
    resolution (README; apply_metric_scaling in utils/alignment.py), so
    `predict` needs the focal length of the input frames. Returns metres at
    the processed resolution.

    Sensitive to input encoding: 4:2:0 chroma subsampling of KITTI PNGs (as
    JPEG does) lowers its KITTI AbsRel from 0.083 to 0.073 (probe round 12,
    reproduction.json), so frames are fed exactly as the dataset stores them.
    Trained at aspect ratios up to 1:2 only: wider frames are the caller's to
    crop (eval/da_metric_paper.py does for the paper's table)."""

    PROCESS_RES = 504
    METHOD = "lower_bound_resize"

    def __init__(self, device: torch.device, record: dict | None = None):
        self.info = ArmInfo("da3_metric", "depth", causal=True, prompted=False)
        model = load_da3(str(weights_file("da3_metric", record)))
        self.model = model.to(device).eval()
        self.device = device

    def predict(
        self,
        frames: np.ndarray,
        prompt: Prompt | None = None,
        focal: tuple[float, float] | None = None,
    ) -> np.ndarray:
        _check_frames(frames)
        _no_prompt(self.info, prompt)
        if focal is None:
            raise ValueError(
                "da3_metric needs the frames' focal length (fx, fy) in pixels: "
                "its output is metric only once scaled by it"
            )
        fx, fy = focal
        H, W = frames.shape[1:3]
        out = []
        for frame in frames:
            raw = self.model.inference(
                [frame], process_res=self.PROCESS_RES, process_res_method=self.METHOD
            ).depth[0]
            sx, sy = raw.shape[1] / W, raw.shape[0] / H
            out.append(raw * 0.5 * (fx * sx + fy * sy) / 300.0)
        return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------------------
# registry + gate
# ---------------------------------------------------------------------------
ARMS = {
    "vda": lambda device, record: VDAArm(device, record=record),
    "vda_metric": lambda device, record: VDAArm(device, metric=True, record=record),
    "dav": DAVArm,
    "ovda": OVDAArm,
    "promptda": PromptDAArm,
    "da3_metric": DA3MetricArm,
}


def build_arm(
    name: str,
    device: torch.device,
    uses: list[tuple[str, str]],
    record: dict | None = None,
    allow_unverified: bool = False,
):
    """The arm, built only if the reproduction record opens every (dataset,
    protocol) in `uses` to it (record.require_allowed). The same record
    names the weights the arm loads. `allow_unverified` is the explicit,
    caller-recorded override for comparing an arm whose reproduction is
    close but not verified (bench_baselines benchmark --allow-close stamps
    the arm's computed status into its results)."""
    if name not in ARMS:
        raise KeyError(f"unknown baseline arm {name!r}; known: {sorted(ARMS)}")
    if not uses:
        raise ValueError("build_arm needs the (dataset, protocol) pairs it will run")
    record = record or R.load()
    if not allow_unverified:
        R.require_allowed(record, name, uses)
    return ARMS[name](device, record)
