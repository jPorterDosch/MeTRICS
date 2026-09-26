#!/usr/bin/env python
"""Run the video-depth benchmark (bench_eval.py) on an already trained
checkpoint, or on the pretrained backbone, without a training run.

    cd src
    python bench_checkpoint.py --weights ../checkpoints/<group>/<run_id> --checkpoint best
    python bench_checkpoint.py --weights <run_dir> --bench.datasets scannet bonn --bench.max-sequences 2
    python bench_checkpoint.py --weights <run_dir> --base --pretrained ../ckpt/checkpoints.pth   # baseline arm

The model is rebuilt from the config snapshot the checkpoint carries
(load_saved_args / rebuild_metric_cfg, as visualize_depth.py does), so no
training CLI has to be replayed; finetune_depth.py --epochs 0 --resume would
demand the identical arguments. Results go to <out-dir>/bench_results.json
and <out-dir>/bench_clouds/, default <weights dir>/bench_<checkpoint>/.
Nothing is logged to wandb here; bench_results.json is the record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro
from accelerate import Accelerator

from bench_eval import BenchmarkCfg, run_benchmark
from finetune_depth import FinetuneDepthCfg, build_model
from visualize_depth import load_saved_args, rebuild_metric_cfg, resolve_checkpoint


@dataclass
class BenchCheckpointCfg:
    weights: str
    """Run directory holding checkpoint-<name>.pth, or a .pth file."""
    checkpoint: str = "best"
    """auto | final | best | last (visualize_depth.py's resolution)."""
    base: bool = False
    """Score the pretrained backbone with zero-init conditioning instead of
    the checkpoint's weights (the checkpoint still supplies the
    architecture config)."""
    pretrained: str | None = None
    """Pretrained StreamVGGT weights for --base; default: the path the
    checkpoint recorded."""
    out_dir: str | None = None
    seed: int = 42
    bench: BenchmarkCfg = field(default_factory=lambda: BenchmarkCfg(enabled=True))


def main(cfg: BenchCheckpointCfg) -> None:
    ckpt_path = resolve_checkpoint(cfg.weights, cfg.checkpoint)
    print(f"checkpoint: {ckpt_path}")
    # mmap: the state dict is ~5 GB fp32 and build_model makes a second copy
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    raw = load_saved_args(ckpt)
    # the benchmark draws its own TUBE_MASK sparse depth; the checkpoint's
    # training-time simulation (and any per-machine freq-map path its
    # validate() would insist on) is not used
    raw["depth_cond"] = dict(raw["depth_cond"])
    raw["depth_cond"]["sim_mode"] = "none"
    raw["depth_cond"]["sim_freq_map_path"] = ""
    raw["depth_cond"]["sim_freq_map_sha256"] = ""
    mcfg = rebuild_metric_cfg(raw)

    pretrained = ""
    if cfg.base:
        pretrained = cfg.pretrained or raw.get("pretrained") or ""
        if not pretrained or not os.path.exists(pretrained):
            raise FileNotFoundError(
                f"--base needs pretrained weights (saved path: {raw.get('pretrained')!r})"
            )
    out_dir = cfg.out_dir or str(
        Path(ckpt_path).parent / f"bench_{'base' if cfg.base else cfg.checkpoint}"
    )
    train_cfg = FinetuneDepthCfg(
        depth_cond=mcfg.depth_cond,
        lora=mcfg.lora,
        encoder_cache=mcfg.encoder_cache,
        train=mcfg.train,
        pretrained=pretrained,
        resume=None,
        output_dir=out_dir,
    )
    accelerator = Accelerator()
    model, _ = build_model(
        train_cfg, mcfg, accelerator.device, load_pretrained=cfg.base
    )
    if not cfg.base:
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state_dict, strict=True)
        del state_dict
    del ckpt
    model.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg.bench.enabled = True
    run_benchmark(model, accelerator, cfg.bench, mcfg, out_dir, step=0, seed=cfg.seed)
    print(f"done -> {out_dir}/bench_results.json")


if __name__ == "__main__":
    main(tyro.cli(BenchCheckpointCfg))
