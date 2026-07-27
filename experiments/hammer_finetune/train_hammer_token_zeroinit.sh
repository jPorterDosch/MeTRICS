#!/bin/bash
#SBATCH --job-name=token_zeroinit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=l40s
#SBATCH --time=48:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/logs/hammer_finetune/%j.out

# =============================================================================
# train_hammer_token_zeroinit.sh -- the GATE-INIT branch of the conditioning
# isolation ablation: token injection + LoRA on HAMMER with the ZERO_CONV
# projection init, single-variable against control b536d87d26e297e1
# (hammer_sweep, token+LoRA, GATED init).
#
# What changed and why: the control's gated scheme (random projection x
# scalar zero gate) has a measured soft gradient deadlock -- grad(token_proj)
# is scaled by the gate, so at gate=0 the projection never trains and the
# gate has nothing useful to grow for (b536d87d gate=-0.003, fcfc883f
# gate=-0.011 after 5 epochs; token_proj still at random init). This branch
# REMOVED the gate entirely (2026-07-25): the token projection is now
# zero-init with no gate (LoRA-B pattern) -- exact no-op at init, gradient
# flows from step 0. The ONE varied factor vs the control is therefore the
# branch's conditioner code, not a CLI flag; the config below is otherwise
# the control's. NOTE: the init scheme is no longer a config knob, so nothing
# in the manifest encodes this change -- the run id (5978e1b0..., differs
# from the control's only via incidental config-field refactors) does NOT
# advertise it; the exp-group and the code snapshot finetune_depth saves
# with the run are what identify this arm. Old gated checkpoints are NOT
# loadable on this branch.
#
# WHAT THIS RUN CAN AND CANNOT SHOW (agreed 2026-07-25): it is the cheap
# MECHANISM screen. Success = (a) token_proj moves off zero during training,
# and (b) the step-0 ablation delta (tests/conditioning_ablation_eval.py on
# this run's checkpoint-best) clearly exceeds the ~+0.003..+0.005 baseline
# the broken-gate arms already reach. A null delta here does NOT rule out
# conditioning -- HAMMER's tabletop data gives measurements little to add
# (the head arms trained their convs and still ignore them), and the dataset
# hypothesis is tested by experiments/scannet_finetune/train_scannet.sh, not
# by this run.
#
# Config parity with the control: every knob below matches b536d87d's saved
# args (verified from checkpoint-best.pth 2026-07-25): token injection, heads
# DEPTH, LoRA defaults (qkvo r16 a32), identity encoder / no temporal / fixed
# 10 m norm / log-depth, sim RANDOM patch14 ratio 0.95, 5 epochs, lr 1e-5,
# bs 1, seed 42, epoch sizes 4500/1000. One drift caveat: the control predates
# the max_interval->stride_range sampler refactor (saved max_interval=20);
# stride-range 1 20 is the documented equivalent under the current sampler.
#
# How to execute (from anywhere):
#   sbatch /oscar/home/jdosch/MeTRIC/experiments/hammer_finetune/train_hammer_token_zeroinit.sh
#
# Outputs: checkpoints + manifest.json under
#   ${REPO}/checkpoints/hammer_token_zeroinit/<run_id>/
# Metrics stream to wandb project "MeTRIC"; export WANDB_MODE=offline first
# if the node has no internet.
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
DATA=/oscar/scratch/jdosch/data/processed_hammer

# --- environment: the StreamVGGT conda env has torch/accelerate/tyro etc. ---
export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH
# expandable_segments: avoids fragmentation-class OOMs on L40S; every arm sets
# it for parity (the token head-only arm OOM'd exactly this way).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# absolute path so the script works when sbatch'd from any CWD
source "$REPO/.secrets/wandb-personal.env"

mkdir -p "$REPO/logs/hammer_finetune"

# finetune_depth.py resolves relative paths against the CWD, so run from src/
cd "$REPO/src"

python finetune_depth.py \
    --exp-group hammer_token_zeroinit \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$REPO/ckpt/checkpoints.pth" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: TOKEN + LoRA ----------------------------------` \
    `# the zero-init (ungated) token projection is baked into this branch's`  \
    `# conditioner code -- THE one varied factor vs control b536d87d. LoRA`   \
    `# stays at its defaults (enabled, qkvo, rank 16, alpha 32) to match the` \
    `# control; encoder/temporal/norm stay at their defaults (identity,`      \
    `# none, fixed 10 m, log-depth), same as the control.`                    \
    --depth-cond.injection TOKEN \
    --depth-cond.heads DEPTH \
    --train.train-heads DEPTH \
    \
    `# --- train data: HAMMER, matching the control ------------------------` \
    --train-dataset.root "$DATA" \
    --train-dataset.dataset HAMMER \
    --train-dataset.stride-range 1 20 \
    --train-dataset.epoch-size 4500 \
    --train-dataset.highres-root None \
    \
    `# --- val data: HAMMER test split, consecutive frames -----------------` \
    --val-dataset.root "$DATA" \
    --val-dataset.dataset HAMMER \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
    --val-dataset.highres-root None \
    \
    `# --- optimization: byte-for-byte the control run's settings ----------` \
    --batch-size 1 \
    --accum-iter 1 \
    --epochs 5 \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --warmup-epochs 0.5 \
    --weight-decay 0.05 \
    --amp 1 \
    --seed 42 \
    \
    `# --- cadence ---------------------------------------------------------` \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 8 \
    --print-freq 10
