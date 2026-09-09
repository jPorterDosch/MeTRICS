#!/bin/bash
#SBATCH --job-name=all_datasets_finetune
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_datasets_finetune_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_datasets_finetune_%j.out

# =============================================================================
# train_all_datasets.sh -- full fine-tuning run of the depth-conditioned
# StreamVGGT (MeTRIC) over every dataset preprocessed on ISAAC: ScanNet++,
# TartanAir and ScanNet. Entrypoint is src/finetune_depth.py (tyro CLI), the
# streamvggt stack -- NOT src/train.py / src/finetune.py, which are the older
# hydra + dust3r.datasets path and whose configs point at 13 roots that do not
# exist here.
#
# How to execute:
#   sbatch experiments/all_datasets_finetune/train_all_datasets.sh
# or, on a node that already holds a GPU allocation:
#   bash   experiments/all_datasets_finetune/train_all_datasets.sh
#
# ACCOUNT NOTE. Runs on ai-tenn (8 nodes x 4 H100), billed to isaac-utk0256 --
# ai-tenn is QOS-gated and ONLY that association carries the ai-tenn QOS, so
# the partition and the account move together. isaac-utk0516, which owns the
# data directory, cannot reach this partition. That is fine: filesystem access
# is group-based (isaac2858), not Slurm-account-based, so billing 0256 while
# reading /lustre/isaac24/proj/UTK0516 works.
#
# ai-tenn is contended (30/32 GPUs allocated, 30 jobs queued on 2026-09-07) and
# shares a 28-job-per-user cap with any other work on the same account. To fall
# back to the A40 node instead -- less compute, but a 24 h QOS cap:
#   --account=isaac-utk0516 --partition=campus-gpu-bigmem --qos=campus-gpu \
#   --gres=gpu:a40:1 --time=1-00:00:00
#
# WALLTIME is capped by the QOS, not the partition. ai-tenn allows
# 3-00:00:00; the partition's MaxTime of 30 days is irrelevant and anything
# longer is rejected at submission with QOSMaxWallDurationPerJobLimit --
# verify any change with `sbatch --test-only`. Other ceilings: campus-gpu
# 1-00:00:00 (A40), long-gpu 6-00:00:00 (V100S, 32 GB, falls back to fp16
# since Volta has no bf16), ai-tenn-debug 3 h (useful for a step-rate probe).
#
# BUDGET. 1,500 x 3 = 4,500 samples/epoch x 15 epochs at batch-size 1 =
# 67,500 optimizer steps. Sized against the prior HAMMER+ScanNet runs, which
# converged after 5 epochs of 3,500 samples = 17,500 steps: this is 3.9x that,
# margin for the third domain (TartanAir is outdoor synthetic, new to the mix).
#
# --epochs is the COSINE DENOMINATOR, not a stopping criterion -- lr decays as
# cos(pi * (epoch - warmup) / (epochs - warmup)). Setting it far above the real
# budget means the run never anneals: at --epochs 20 the lr is still 88% of
# peak where these runs converge, versus 1% at a correctly sized schedule. So
# size it to the budget you intend and let checkpoint-best pick the epoch.
#
# The 4,500-sample epoch also keeps val cadence near the prior runs' (--val-freq
# 1 validates per epoch): 15 val passes over the run instead of 5, so
# checkpoint-best has usable resolution.
#
# 67,500 steps should fit the 3-day window on an H100 at any plausible step
# rate (72 h allows up to ~3.8 s/step), so this is expected to complete in one
# submission. If it does not, the run chunks cleanly:
# --save-freq 0.1 checkpoints every 10% of an epoch and the next job continues
# with --resume <checkpoint>. Steps per job is walltime x speed, so the H100
# window (3 d) buys far more progress per submission than the A40 one (1 d)
# even though it is shorter in wall-clock terms.
#
# Outputs: checkpoints + manifest.json under
#   ${REPO}/checkpoints/${EXP_GROUP}/<experiment-id>/
# finetune_depth names the run dir by a SHA over the config and fails fast if
# it already exists, so a finished experiment is never silently re-run.
# Metrics stream to wandb project "MeTRIC" (entity sparse_representation_learning).
# =============================================================================

set -euo pipefail

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
DATA=/lustre/isaac24/proj/UTK0516/metrics_data/processed
EXP_GROUP=metric_all_datasets
# ARKitScenes is NOT under $DATA: metrics_data/processed is hqi-owned with
# mode drwxr-sr-x, so this uid cannot mkdir in it and the preprocess job was
# run with METRICS_PROCESSED_ROOT pointed at this sibling instead.
ARKIT_OUT="${ARKIT_OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/processed_jd}"
# StreamVGGT backbone weights: the official release (Zhuo & Zheng et al.,
# arXiv 2507.11539), staged on the project filesystem 2026-09-07 and verified
# against the model -- a raw state_dict of 1,797 fp32 tensors / 1.26 B params
# keyed aggregator.* + {camera,point,depth,track}_head.*, matching
# StreamVGGT.state_dict() exactly with no missing or unexpected keys.
# load_pretrained feeds it to load_state_dict(strict=True), so a checkpoint
# with any other key layout fails loudly rather than silently training from
# scratch. Override with PRETRAINED=/path/to/other.pth.
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}

# --- environment -------------------------------------------------------------
# The pyenv 3.11.13 interpreter, not a conda env (Oscar's StreamVGGT env has no
# counterpart here).
export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/bin:$PATH
# pyenv 3.11.13 was built against libbz2.so.1.0 but RHEL8 ships libbz2.so.1, so
# `import bz2` -- and therefore torchvision -- fails without this shim
# (~/.local/lib/libbz2.so.1.0 -> /usr/lib64/libbz2.so.1). Same reason the
# metrics-torch Jupyter kernel sets it in its own env block.
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}
# expandable_segments: avoids fragmentation-class OOMs where a small alloc
# fails with memory free but non-contiguous. Kept from the HAMMER arms for
# parity; a frozen backbone is not immunity.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# thread oversubscription across 12 dataloader workers otherwise thrashes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# wandb credentials live outside the repo (.secrets/ is gitignored and does not
# exist on ISAAC yet). Without them wandb falls back to offline and the run
# still completes; sync later with `wandb sync`.
if [ -f "$REPO/.secrets/wandb-personal.env" ]; then
    # shellcheck disable=SC1091
    source "$REPO/.secrets/wandb-personal.env"
else
    echo "[warn] no .secrets/wandb-personal.env -- running wandb offline"
    export WANDB_MODE=offline
fi

mkdir -p "$REPO/logs"

# --- preflight: fail here, not 20 minutes into dataset construction ----------
if [ ! -f "$PRETRAINED" ]; then
    echo "[fatal] pretrained StreamVGGT weights not found: $PRETRAINED"
    echo "        copy the checkpoint over, or set PRETRAINED=/path/to/ckpt.pth"
    exit 1
fi
for d in processed_scannetpp processed_tartanair processed_scannet; do
    [ -d "$DATA/$d" ] || { echo "[fatal] missing dataset root: $DATA/$d"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# finetune_depth.py resolves relative paths (and save_current_code's ".")
# against the CWD, so run from src/ like the other entrypoints
cd "$REPO/src"

python finetune_depth.py \
    --exp-group "$EXP_GROUP" \
    \
    `# --- model / checkpointing -------------------------------------------` \
    --pretrained "$PRETRAINED" \
    --save-dir "$REPO/checkpoints" \
    \
    `# --- conditioning arm: TOKEN injection, LoRA --------------------------` \
    `# The full-corpus run uses the token arm (sparse depth residual-added`    \
    `# into the RGB patch tokens) with LoRA on the attention projections, so`  \
    `# the backbone adapts to three domains instead of only the depth head`    \
    `# fitting one. Swap to the HAMMER control arm with:`                      \
    `#   --depth-cond.injection HEAD --depth-cond.heads DEPTH`      \
    `#   --lora.no-enabled --train.train-heads DEPTH`                          \
    --depth-cond.injection TOKEN \
    \
    `# --- train data: ScanNet++ + TartanAir + ScanNet ----------------------` \
    `# These are already the FinetuneDepthCfg defaults, but an experiment`     \
    `# script should pin what it trained on rather than inherit it -- the`     \
    `# defaults are shared and will move. Parallel tuples: entry i of every`   \
    `# --train-dataset.* flag describes dataset i.`                            \
    `#`                                                                        \
    `# stride-range is 1 1 for ALL THREE -- deliberately NOT each loader's`    \
    `# own default (ScanNet++ 3 / TartanAir 20 / ScanNet 8). The objective is` \
    `# frame-by-frame stability for robotics deployment: the model sees`       \
    `# consecutive frames and metric depth is supplied by the sensor, so`      \
    `# training on consecutive frames matches deployment -- and matches the`   \
    `# TEST split, which DatasetConfig.validate() pins to (1, 1).`             \
    `#`                                                                        \
    `# The cost is inter-view baseline. Median over a 10-view clip, (1,1) vs`  \
    `# the loader default: ScanNet 0.064 m vs 0.278 m, TartanAir 1.487 m vs`   \
    `# 9.109 m, ScanNet++ 0.753 m vs 1.169 m. ScanNet gives ~9 mm between`     \
    `# adjacent views, so multi-view parallax is near-nil and depth leans on`  \
    `# the conditioning input rather than triangulation. Intended, not an`     \
    `# oversight -- do not "restore" the defaults without re-reading this.`    \
    `#`                                                                        \
    `# ARKitScenes is ONE domain across TWO directories (lowres LiDAR +`      \
    `# highres laser GT partition the same scenes), so it gets one domain's`   \
    `# share, 750/375 on the DUSt3R 2:1, not a full slice per directory.`      \
    `# highres-root is passed explicitly so a wrong path RAISES; left at None` \
    `# the lowres loader falls back to a sibling-name convention that is`      \
    `# silently skipped when absent, leaving highres scenes in both sets.`     \
    `#`                                                                        \
    `# epoch-size is the "N @" weight: equal slices, not natural lengths --`   \
    `# ScanNet has 2.4M start frames against ScanNet++'s 60k and TartanAir's`  \
    `# 303k, and the mixture is for domain spread (indoor real / indoor`       \
    `# rendered / outdoor synthetic), not volume. 13,500 samples per epoch.`   \
    `#`                                                                        \
    `# highres-root is NOT passed: it only applies to ARKitScenes lowres and`  \
    `# the default no longer carries ARKitScenes entries to clear.`            \
    --train-dataset.root "$DATA/processed_scannetpp" \
                         "$DATA/processed_tartanair" \
                         "$DATA/processed_scannet" \
                         "$ARKIT_OUT/processed_arkitscenes" \
                         "$ARKIT_OUT/processed_arkitscenes_highres" \
    --train-dataset.dataset SCANNETPP TARTANAIR SCANNET \
                            ARKITSCENES_LOWRES ARKITSCENES_HIGHRES \
    --train-dataset.stride-range 1 1 1 1 1 1 1 1 1 1 \
    --train-dataset.epoch-size 1125 1125 1125 750 375 \
    --train-dataset.highres-root None None None \
                                 "$ARKIT_OUT/processed_arkitscenes_highres" None \
    \
    `# --- val data: ScanNet scans_test -------------------------------------` \
    `# Real-sensor depth, and the only one of the three with a genuine test`   \
    `# partition -- ScanNet++ and TartanAir are preprocessed train-only and`   \
    `# their loaders reject Split.TEST rather than hand back training frames.` \
    `# stride-range 1 1 (consecutive) is required for TEST and enforced by`    \
    `# DatasetConfig.validate(). Other val defaults: num_views 4, single`      \
    `# (518, 392) resolution, seed 42 -> the same clip set every epoch.`       \
    --val-dataset.root "$DATA/processed_scannet" \
    --val-dataset.dataset SCANNET \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
    \
    `# --- optimization ------------------------------------------------------` \
    `# batch-size 1 on an 80GB H100 (the HAMMER arms ran 1 on 48GB L40S and`  \
    `# were never memory-profiled). NOTE: batch size changes`  \
    `# make train/sparse/ratio_max incomparable across runs -- read`           \
    `# train/sparse/starved_frac instead, which is batch-size invariant.`      \
    --batch-size 1 \
    --accum-iter 1 \
    --epochs 15 \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --warmup-epochs 0.5 \
    --weight-decay 0.05 \
    --amp 1 \
    --seed 42 \
    \
    `# --- cadence -----------------------------------------------------------` \
    --val-freq 1 \
    --save-freq 0.1 \
    --num-workers 12 \
    --print-freq 10
