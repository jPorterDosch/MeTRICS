"""Build a MetricStreamVGGT and load its checkpoint config: the one piece of
model construction that training (finetune_depth), the benchmark
(bench_checkpoint) and the visualizers share.

Kept apart from finetune_depth so a benchmark run imports none of the
training code: the benchmark's results record the code it ran file by file
(eval.provenance.loaded_code), and a change to a training script must not
make an earlier benchmark row look like it ran different code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import torch

from streamvggt.depth_cond import (
    DepthCondCfg,
    EncoderCacheCfg,
    LoRACfg,
    MetricCfg,
    MetricStreamVGGT,
    ScaleTokenCfg,
    TrainCondCfg,
)

# Checkpoint basenames finetune_depth.py writes; --checkpoint auto takes the
# first that exists (last: best was selected on validation).
CKPT_NAMES = {"last": "checkpoint-last.pth", "best": "checkpoint-best.pth"}
AUTO_ORDER = ("last", "best")


def build_model(
    mcfg: MetricCfg,
    device: torch.device,
    pretrained: str | None = None,
    resume: str | None = None,
    load_pretrained: bool = True,
) -> tuple[MetricStreamVGGT, dict]:
    """The model for `mcfg`, LoRA-wrapped and frozen for fine-tuning, on
    `device`. With load_pretrained, the StreamVGGT weights come from
    `pretrained` on a fresh run, or the encoder cache is initialised from
    `resume` on a resumed one."""
    model = MetricStreamVGGT(mcfg)
    if load_pretrained and pretrained and not resume:
        print(f"Loading pretrained StreamVGGT: {pretrained}")
        print(model.load_pretrained(pretrained))
    elif load_pretrained and resume:
        model.initialize_encoder_cache(resume)
    n = model.apply_lora_adapters()
    stats = model.freeze_for_finetune()
    model.to(device)
    print(
        f"LoRA: wrapped {n} attention modules (targets={mcfg.lora.targets}, rank={mcfg.lora.rank})"
    )
    print(
        f"Params: total {stats['total_params']:,} | trainable {stats['trainable_params']:,} "
        f"({stats['trainable_pct']:.3f}%) | base attention frozen: {stats['base_attention_frozen']}"
    )
    if not stats["base_attention_frozen"]:
        raise RuntimeError("base attention projections must stay frozen")
    return model, stats


class LoadedRun(NamedTuple):
    model: MetricStreamVGGT
    raw: dict
    """the run's saved config snapshot (sparse simulation switched off)"""
    mcfg: MetricCfg
    ckpt_path: Path


def load_run_model(
    weights: str,
    which: str,
    device: torch.device,
    base: bool = False,
    pretrained: str | None = None,
    encoder_cache: bool = True,
) -> LoadedRun:
    """A finetune_depth.py run's model in eval mode on `device`, with its saved
    config and the checkpoint file used. Its training-time sparse simulation
    is not rebuilt (callers draw their own sparse depth, and a pixel_freq
    run's map path would have to exist on this machine). With `base`, the run
    supplies only the architecture and the weights are the pretrained
    StreamVGGT (`pretrained`, default the path the run recorded; zero-init
    conditioning makes it the raw backbone). encoder_cache=False skips the
    run's encoder feature cache, for callers that never pass cache keys
    (constructing it creates the run's cache directory)."""
    ckpt_path = resolve_checkpoint(weights, which)
    # mmap: the state dict is ~5 GB fp32 and build_model makes a second copy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    raw = load_saved_args(ckpt)
    raw["depth_cond"] = dict(raw["depth_cond"])
    raw["depth_cond"]["sim_mode"] = "none"
    raw["depth_cond"]["sim_freq_map_path"] = ""
    raw["depth_cond"]["sim_freq_map_sha256"] = ""
    mcfg = rebuild_metric_cfg(raw)
    if not encoder_cache:
        mcfg.encoder_cache.enabled = False
    if base:
        # the raw backbone has no scale token: the token (a register copy), its
        # carry and readout would be extra modules, whatever their zero-init
        mcfg.scale_token = ScaleTokenCfg()
        pretrained = pretrained or raw.get("pretrained") or ""
        if not pretrained or not os.path.exists(pretrained):
            raise FileNotFoundError(
                f"base weights not found (saved path: {raw.get('pretrained')!r})"
            )
    model, _ = build_model(
        mcfg, device, pretrained=pretrained or "", load_pretrained=base
    )
    if not base:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
        del state_dict
    del ckpt
    return LoadedRun(model.eval(), raw, mcfg, Path(ckpt_path))


def load_base_streamvggt(ckpt_path: str):
    """The pretrained StreamVGGT (RGB only), in eval mode on CPU: a raw
    state_dict or a {"model": state_dict} wrapper, the same unwrap rule as
    MetricStreamVGGT.load_pretrained."""
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT()
    sd = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(sd, dict)
        and "model" in sd
        and not any(k.startswith("aggregator.") for k in sd)
    ):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    return model.eval()


def resolve_checkpoint(weights: str, which: str) -> Path:
    """Resolve --weights (+ --checkpoint) to a concrete .pth. `weights` may be
    the checkpoint file itself or the run directory that contains it."""
    p = Path(weights)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"--weights is neither a file nor a directory: {p}")
    if which != "auto":
        cand = p / CKPT_NAMES[which]
        if not cand.is_file():
            raise FileNotFoundError(f"No {cand.name} in {p}")
        return cand
    for key in AUTO_ORDER:
        cand = p / CKPT_NAMES[key]
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No checkpoint ({', '.join(CKPT_NAMES.values())}) found in {p}"
    )


def load_saved_args(ckpt: dict) -> dict:
    """The primitive config snapshot picklable_args() embedded at save time.
    Everything is builtin types (enums as strings) -- the config dataclasses'
    validate() coerces them back."""
    if "args" not in ckpt:
        raise KeyError(
            "checkpoint has no 'args' snapshot; cannot reconstruct the model "
            "architecture. Was it written by finetune_depth.py?"
        )
    saved = ckpt["args"]
    return dict(vars(saved)) if not isinstance(saved, dict) else dict(saved)


def rebuild_metric_cfg(raw: dict) -> MetricCfg:
    """Reconstruct the architecture config from the primitive snapshot. Each
    nested dataclass's validate() turns the string/list primitives back into
    enum members, so the built model matches the checkpoint's key layout."""
    return MetricCfg(
        depth_cond=DepthCondCfg(**raw["depth_cond"]),
        lora=LoRACfg(**raw["lora"]),
        encoder_cache=EncoderCacheCfg(**raw["encoder_cache"]),
        train=TrainCondCfg(**raw["train"]),
        # runs from before the scale token have no such block: off
        scale_token=ScaleTokenCfg(**raw.get("scale_token", {})),
    ).validate()
