#!/usr/bin/env python
"""Run the video-depth benchmark (bench_eval.py) on an already trained
checkpoint, or on the pretrained backbone, without a training run.

    cd src
    python bench_checkpoint.py --weights ../checkpoints/<group>/<run_id> --checkpoint best
    python bench_checkpoint.py --weights <run_dir> --bench.datasets scannet bonn --bench.max-sequences 2
    python bench_checkpoint.py --weights <run_dir> --base --pretrained ../ckpt/checkpoints.pth   # baseline arm

The model is rebuilt from the config snapshot the checkpoint carries
(model_io.load_run_model), so no
training CLI has to be replayed; finetune_depth.py --epochs 0 --resume would
demand the identical arguments. Results go to <out-dir>/bench_results.json
and <out-dir>/bench_clouds/, default <weights dir>/bench_<checkpoint>/.
Nothing is logged to wandb here; bench_results.json is the record.
"""

from __future__ import annotations

# first: eval.provenance's audit hook records every repository file run from
# here on, with its content at that moment (results' code_files)
import eval.provenance  # noqa: E402,F401,I001

from dataclasses import dataclass, field
from pathlib import Path

import tyro
from accelerate import Accelerator

from bench_eval import BenchmarkCfg, run_benchmark

# model_io, not finetune_depth / visualize_depth: the results record every
# file this process imports, and training code must not be among them
from model_io import load_run_model


@dataclass
class BenchCheckpointCfg:
    weights: str
    """Run directory holding checkpoint-<name>.pth, or a .pth file."""
    checkpoint: str = "last"
    """auto | best | last (visualize_depth.py's resolution). Default
    last: "best" is selected on validation AbsRel, and the validation set
    includes ScanNet scans_test, the benchmark's own ScanNet scenes."""
    base: bool = False
    """Score the pretrained backbone with zero-init conditioning instead of
    the checkpoint's weights (the checkpoint still supplies the
    architecture config)."""
    pretrained: str | None = None
    """Pretrained StreamVGGT weights for --base; default: the path the
    checkpoint recorded."""
    out_dir: str | None = None
    bench: BenchmarkCfg = field(default_factory=lambda: BenchmarkCfg(enabled=True))


def main(cfg: BenchCheckpointCfg) -> None:
    accelerator = Accelerator()
    # the benchmark draws its own TUBE_MASK sparse depth; load_run_model drops
    # the checkpoint's training-time simulation
    model, raw, mcfg, ckpt_path = load_run_model(
        cfg.weights,
        cfg.checkpoint,
        accelerator.device,
        base=cfg.base,
        pretrained=cfg.pretrained,
    )
    print(f"checkpoint: {ckpt_path}")
    # named after the file actually loaded: --weights may name a .pth, which
    # wins over --checkpoint, and "auto" is not a checkpoint
    label = ckpt_path.stem.removeprefix("checkpoint-")
    out_dir = cfg.out_dir or str(
        ckpt_path.parent / f"bench_{'base' if cfg.base else label}"
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg.bench.enabled = True
    # the run's own precision; sparse draws come from cfg.bench.seed
    run_benchmark(
        model,
        accelerator,
        cfg.bench,
        mcfg,
        out_dir,
        step=0,
        amp=bool(raw.get("amp", 1)),
        weights="pretrained backbone (--base)" if cfg.base else str(ckpt_path),
    )
    print(f"done -> {out_dir}/bench_results.json")


if __name__ == "__main__":
    main(tyro.cli(BenchCheckpointCfg))
