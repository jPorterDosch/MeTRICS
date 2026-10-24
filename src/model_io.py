"""Build a MetricStreamVGGT and load its checkpoint config: the one piece of
model construction that training (finetune_depth), the benchmark
(bench_checkpoint) and the visualizers share.

Kept apart from finetune_depth so a benchmark run imports none of the
training code: the benchmark's results record the code it ran file by file
(eval.provenance.loaded_code), and a change to a training script must not
make an earlier benchmark row look like it ran different code.
"""

from __future__ import annotations

from pathlib import Path

import torch

from streamvggt.depth_cond import (
    DepthCondCfg,
    EncoderCacheCfg,
    LoRACfg,
    MetricCfg,
    MetricStreamVGGT,
    TrainCondCfg,
)

# Checkpoint basenames finetune_depth.py writes, best -> worst preference when
# --checkpoint is left at "auto". The pipeline no longer writes a separate
# checkpoint-final.pth; the end-of-training weights live in checkpoint-last.pth,
# so the "final" alias now resolves there (kept for backward compatibility).
CKPT_NAMES = {
    "final": "checkpoint-last.pth",
    "best": "checkpoint-best.pth",
    "last": "checkpoint-last.pth",
}
AUTO_ORDER = ("final", "best", "last")


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
    ).validate()
