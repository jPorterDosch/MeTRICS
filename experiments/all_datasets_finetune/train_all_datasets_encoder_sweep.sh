#!/bin/bash
#SBATCH --job-name=all_ds_encoder
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_encoder_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_encoder_%j.out
# =============================================================================
# Depth-encoder / edge-loss ablation on the five-dataset mixture, after the
# comparison-table runs (2026-10-16) showed the model treating its fed depth
# as a weak hint: 1-3% error at the fed pixels indoors, 0.2-0.5x GT on KITTI,
# per-frame scale wobble up to 20%.
#
#   arm  encoder       edge loss                  role
#   ---  ------------  -------------------------  -----------------------------
#    0   identity      none                       control: the table's `tube`
#                                                 arm (token + LoRA, tube-mask
#                                                 sparsity) on the official
#                                                 ScanNet split
#    1   prefill_conv  none                       encoder effect
#    2   prefill_conv  DA3 gradient, teacher 1x   edge-loss effect
#    3   prefill_conv  DA3 gradient, teacher 2x   edge targets from a 2x denser
#                                                 teacher prediction
#    4   (none)        none                       no sparse depth at all
#                                                 (--depth-cond.no-enabled):
#                                                 LoRA + heads trained on the
#                                                 same data and schedule, RGB
#                                                 only
#
# prefill_conv (depth_cond/conditioner.py): the sparse map is filled from the
# nearest measurement, a distance-to-measurement channel is added, and a 3x3
# conv stem runs at full resolution before the stride-14 patch embed, so
# sub-patch structure reaches the token. The identity encoder averages each
# 14x14 patch to one number.
#
# Edge loss (loss/gradient_loss.py teacher_edge_loss): Prompt Depth Anything's
# Eq. 2, |d(D_pred - D_teacher)/dx| + |d/dy|, lambda 0.5, on LOG depth like
# the accuracy term (--loss.depth-log-space), so the teacher's scale drops
# out and the two terms share a relative scale. Teacher: Depth
# Anything 3 (the verified da3_metric arm's weights), run frozen on every
# frame during training; PromptDA used a per-scene Zip-NeRF, which 1513
# ScanNet scenes rule out. Arm 3 upsamples the frame 2x for the teacher and
# average-pools its depth back (--teacher.res-scale 2).
#
# Contrasts: 0 vs 1 = encoder; 1 vs 2 = edge loss; 2 vs 3 = teacher
# resolution; 4 vs 0 = how much of the gain is depth injection rather than
# the extra training and LoRA adaptation on this data (arm 4 still sees the
# metric depth loss, so it learns metric scale from RGB alone). Every arm: TOKEN injection, LoRA, TUBE_MASK training sparsity
# (the table's `tube` run, 50e636145fa9afbb; train_all_datasets.sh itself
# uses RANDOM masks). Everything else is train_all_datasets.sh verbatim,
# including the official ScanNet split (train 1201 / val 312; scans_test is
# the benchmark's). Arm 0 is NOT the old `tube` run: the split differs, so
# its hash and run directory are new. Val logs the uptake monitors fed_absrel / fed_logscale and
# boundary_f1 next to the usual metrics.
#
# Run as a job array, one arm per task:
#
#   sbatch --array=0-4 experiments/all_datasets_finetune/train_all_datasets_encoder_sweep.sh
#   sbatch --array=2   experiments/all_datasets_finetune/train_all_datasets_encoder_sweep.sh
#
# Cost: arm 0/1 as train_all_datasets.sh (~23 h on an H100); arms 2/3 add a
# DA3-L forward per frame (estimate +10% at 1x, +30% at 2x). BENCH=0 skips
# the end-of-run benchmark.
# =============================================================================

set -euo pipefail

ARM_NAMES=(identity_noedge prefill_noedge prefill_edge1x prefill_edge2x lora_nodepth)
ARM_FLAGS=(
    "--depth-cond.encoder IDENTITY"
    "--depth-cond.encoder PREFILL_CONV"
    "--depth-cond.encoder PREFILL_CONV --loss.edge-weight 0.5 --teacher.model DA3 --teacher.res-scale 1"
    "--depth-cond.encoder PREFILL_CONV --loss.edge-weight 0.5 --teacher.model DA3 --teacher.res-scale 2"
    "--depth-cond.no-enabled"
)

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
DATA=/lustre/isaac24/proj/UTK0516/metrics_data/processed
# The datasets rebuilt as video are NOT under $DATA: metrics_data/processed is
# hqi-owned with mode drwxr-sr-x, so this uid cannot mkdir in it and the
# preprocess jobs ran with METRICS_PROCESSED_ROOT pointed at this sibling.
#
# $DATA still holds the PRE-REBUILD ScanNet++ (~143 thinned iPhone frames per
# scene plus DSLR stills, against ~637 contiguous iPhone frames here). Both
# load, so reading ScanNet++ from $DATA silently trains on the old selection.
REBUILT="${REBUILT:-/lustre/isaac24/proj/UTK0516/metrics_data/processed_jd}"
ARKIT_OUT="${ARKIT_OUT:-$REBUILT}"
# its own group: four cells, one ladder
EXP_GROUP=metric_encoder_sweep
CKPT_DIR="${CKPT_DIR:-/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd}"
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
BENCH=${BENCH:-1}
BENCH_ROOT=${BENCH_ROOT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}
BENCH_ARGS=()
[ "$BENCH" = 1 ] && BENCH_ARGS=(--bench.enabled --bench.root "$BENCH_ROOT")
mkdir -p "$CKPT_DIR" "$REPO/logs"

# environment identical to train_all_datasets.sh (see its comments)
export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [ -f "$REPO/.secrets/wandb-personal.env" ]; then
    # shellcheck disable=SC1091
    source "$REPO/.secrets/wandb-personal.env"
else
    echo "[warn] no .secrets/wandb-personal.env -- running wandb offline"
    export WANDB_MODE=offline
fi

[ -f "$PRETRAINED" ] || { echo "[fatal] missing $PRETRAINED"; exit 1; }
DA3_WEIGHTS=${BASELINE_WEIGHTS:-/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd/baselines}/da3_metric
[ -f "$DA3_WEIGHTS/model.safetensors" ] \
    || { echo "[fatal] teacher weights missing: $DA3_WEIGHTS (python src/bench_baselines.py fetch --arm da3_metric)"; exit 1; }
for d in "$REBUILT/processed_scannetpp" "$DATA/processed_tartanair" "$DATA/processed_scannet" \
         "$ARKIT_OUT/processed_arkitscenes" "$ARKIT_OUT/processed_arkitscenes_highres"; do
    [ -d "$d" ] || { echo "[fatal] missing dataset root: $d"; exit 1; }
done

if [ "$BENCH" = 1 ]; then
    for d in sintel scannet kitti bonn_all nyuv2; do
        ls "$BENCH_ROOT/$d"/*.json >/dev/null 2>&1 \
            || { echo "[fatal] no benchmark manifest under $BENCH_ROOT/$d (BENCH=0 to skip)"; exit 1; }
    done
fi

ARMS=("${!ARM_NAMES[@]}")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    [ "$SLURM_ARRAY_TASK_ID" -lt "${#ARM_NAMES[@]}" ] \
        || { echo "[fatal] array task $SLURM_ARRAY_TASK_ID has no arm (0-$((${#ARM_NAMES[@]} - 1)))"; exit 1; }
    ARMS=("$SLURM_ARRAY_TASK_ID")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"

for i in "${ARMS[@]}"; do
    echo "=== arm $i: ${ARM_NAMES[$i]}  (${ARM_FLAGS[$i]}) ==="
    # ARM_FLAGS[$i] is deliberately unquoted: it word-splits into separate args
    # shellcheck disable=SC2086
    python finetune_depth.py \
        --exp-group "$EXP_GROUP" \
        --pretrained "$PRETRAINED" \
        --save-dir "$CKPT_DIR" \
        ${ARM_FLAGS[$i]} \
        --depth-cond.injection TOKEN \
        --lora.enabled \
        --depth-cond.sim-mode TUBE_MASK \
        --depth-cond.heads DEPTH \
        --train.train-heads DEPTH \
        --loss.depth-log-space \
        --loss.depth-alpha 0.02 \
        --train-dataset.root "$REBUILT/processed_scannetpp" \
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
        --train-dataset.scannet-official-split train \
        --val-dataset.root "$DATA/processed_scannet" \
                           "$ARKIT_OUT/processed_arkitscenes" \
                           "$ARKIT_OUT/processed_arkitscenes_highres" \
                           "$DATA/processed_hammer" \
        --val-dataset.dataset SCANNET ARKITSCENES_LOWRES ARKITSCENES_HIGHRES HAMMER \
        --val-dataset.stride-range 1 1 1 1 1 1 1 1 \
        --val-dataset.epoch-size 1000 1000 1000 1000 \
        --val-dataset.highres-root None "$ARKIT_OUT/processed_arkitscenes_highres" None None \
        --val-dataset.scannet-official-split val \
        --batch-size 1 \
        --accum-iter 1 \
        --epochs 15 \
        --lr 1e-5 \
        --min-lr 1e-7 \
        --warmup-epochs 0.5 \
        --weight-decay 0.05 \
        --amp 1 \
        --seed 42 \
        --val-freq 1 \
        --save-freq 0.1 \
        --num-workers 12 \
        --print-freq 10 \
        "${BENCH_ARGS[@]}" \
        || echo "=== arm $i did not run to completion (exists or failed); continuing ==="
done

echo "=== sweep loop finished ==="
