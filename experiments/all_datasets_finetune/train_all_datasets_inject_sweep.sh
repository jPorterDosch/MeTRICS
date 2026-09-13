#!/bin/bash
#SBATCH --job-name=all_ds_inject
#SBATCH --array=0-3
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_inject_%A_%a.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_inject_%A_%a.out

# =============================================================================
# Injection-site ablation over the full five-dataset mixture: the 2x2 ladder
# from experiments/hammer_finetune/train_hammer_{head,token}inject_{headonly,lora}.sh,
# run on train_all_datasets.sh's recipe instead of HAMMER.
#
#   arm  injection  trainable                      HAMMER ladder
#   ---  ---------  -----------------------------  -------------------------
#    0   HEAD       heads + conditioner (no LoRA)  1/4 baseline
#    1   HEAD       + LoRA on decoder attention    2/4 bridge
#    2   TOKEN      + LoRA on decoder attention    3/4 proposed
#    3   TOKEN      heads + conditioner (no LoRA)  4/4 negative control
#
# Contrasts: 0 vs 1 = pure LoRA effect at HEAD; 1 vs 2 = pure injection-site
# effect (the key one); 3 vs 2 = does TOKEN need decoder plasticity.
#
# ONLY injection and LoRA vary. Everything else is train_all_datasets.sh
# verbatim -- data, stride (1,1), epoch sizes, 15 epochs, lr schedule -- with
# the HAMMER ladder's loss: --loss.depth-log-space --loss.depth-alpha 0.02.
#
# The loss change gives every arm a new experiment hash, so none of them
# matches the earlier linear-loss run metric_all_datasets/be510a2bdb9e7035
# and all four train.
#
# Heads pinned to DEPTH (--depth-cond.heads DEPTH --train.train-heads DEPTH),
# as in the HAMMER ladder. Under depth_train the point head gets no gradient
# anyway (be510a2bdb9e7035: point_head 0/62 tensors changed), so this trains
# the same weights while keeping the trainable count and HEAD injection honest.
#
#   sbatch experiments/all_datasets_finetune/train_all_datasets_inject_sweep.sh
#   sbatch --array=0 ...                  # one arm
#   sbatch --array=0-3%1 ...              # one GPU at a time
#   ARM=1 bash ...                        # outside an array, on an allocated GPU
#
# Cost: TOKEN + LoRA took 22h49m on an H100 (linear-loss run). LoRA arms cost
# about the same; head-only arms skip the decoder backward and should be faster. All fit the 3-day cap.
# =============================================================================

set -euo pipefail

ARM="${SLURM_ARRAY_TASK_ID:-${ARM:-}}"
case "$ARM" in
    0) ARM_NAME="head_headonly";  ARM_FLAGS=(--depth-cond.injection HEAD  --lora.no-enabled) ;;
    1) ARM_NAME="head_lora";      ARM_FLAGS=(--depth-cond.injection HEAD  --lora.enabled) ;;
    2) ARM_NAME="token_lora";     ARM_FLAGS=(--depth-cond.injection TOKEN --lora.enabled) ;;
    3) ARM_NAME="token_headonly"; ARM_FLAGS=(--depth-cond.injection TOKEN --lora.no-enabled) ;;
    *) echo "ARM must be 0-3 (got '${ARM}'); set by --array or ARM=" >&2; exit 1 ;;
esac

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
DATA=/lustre/isaac24/proj/UTK0516/metrics_data/processed
ARKIT_OUT="${ARKIT_OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/processed_jd}"
# same exp_group as train_all_datasets.sh, so the four cells sit side by side
EXP_GROUP=metric_all_datasets
CKPT_DIR="${CKPT_DIR:-/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd}"
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
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
for d in "$DATA/processed_scannetpp" "$DATA/processed_tartanair" "$DATA/processed_scannet" \
         "$ARKIT_OUT/processed_arkitscenes" "$ARKIT_OUT/processed_arkitscenes_highres"; do
    [ -d "$d" ] || { echo "[fatal] missing dataset root: $d"; exit 1; }
done

echo "=== arm $ARM: $ARM_NAME  (${ARM_FLAGS[*]}) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"

python finetune_depth.py \
    --exp-group "$EXP_GROUP" \
    --pretrained "$PRETRAINED" \
    --save-dir "$CKPT_DIR" \
    "${ARM_FLAGS[@]}" \
    --depth-cond.heads DEPTH \
    --train.train-heads DEPTH \
    --loss.depth-log-space \
    --loss.depth-alpha 0.02 \
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
    --val-dataset.root "$DATA/processed_scannet" \
    --val-dataset.dataset SCANNET \
    --val-dataset.stride-range 1 1 \
    --val-dataset.epoch-size 1000 \
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
    --print-freq 10
