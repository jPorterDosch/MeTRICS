#!/bin/bash
#SBATCH --job-name=all_ds_spot
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_spot_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_spot_%j.out

# =============================================================================
# The TOKEN half of the injection ladder, re-run under the EMPIRICAL SPOT mask
# instead of uniform random patch masking.
#
#   arm  injection  trainable                      inject-sweep counterpart
#   ---  ---------  -----------------------------  ------------------------
#    0   TOKEN      + LoRA on decoder attention    arm 2 (66c8672b41ac8421)
#    1   TOKEN      heads + conditioner (no LoRA)  arm 3 (54bbf74c1353d559)
#
# Contrast: 0 vs 1 = does TOKEN still need decoder plasticity once the prompt
# is the real sensor pattern rather than a uniform random one. The HEAD arms
# are dropped: the inject sweep settled that question at 0.95 random sparsity
# (TOKEN without LoRA beat HEAD with LoRA, 0.0329 vs 0.0563 val absrel), and
# re-running them here would cost ~36 GPU-hours to re-answer it.
#
# ONLY the mask changes from train_all_datasets_inject_sweep.sh -- data,
# stride (1,1), epoch sizes, 15 epochs, lr schedule, loss, heads are verbatim.
#
# WHAT THE MASK CHANGES. The inject sweep trained at sim_mask_ratio 0.95 with
# RANDOM patch masking: 5% of patches visible, uniform over the frame. The SPOT
# map is both DENSER and STRUCTURED -- mean validity 0.4205, so a ~0.58 mask
# ratio, with per-pixel validity drawn from the real sensor's hole pattern
# (assets/spot/valid_freq_640x480.npz, 2558 frames over seqs 0 and 1). Expect
# lower absrel across the board from the denser prompt; the comparison that
# matters is arm 0 vs arm 1, not these numbers against the inject sweep's.
#
# NO --depth-cond.sim-mask-ratio FLAG. Under sim_mode=pixel_freq the density is
# the map's, derived in DepthCondCfg.validate(); passing a ratio is a hard
# error. The map's content sha256 is part of the experiment hash, so rebuilding
# the .npz gives these arms new hashes rather than silently redefining them.
#
# The arms run one after another in this job. An arm whose run dir already
# exists is refused by finetune_depth and the loop moves on to the next arm;
# any other failure also moves on.
#
#   sbatch experiments/all_datasets_finetune/train_all_datasets_spot_mask_sweep.sh
#
# Cost: the inject sweep's TOKEN arms took 23h10m (LoRA) and 20h49m (no LoRA)
# on an H100. Masking cost is identical at any density, so budget ~44h total,
# inside the 3-day wall clock.
# =============================================================================

set -euo pipefail

ARM_NAMES=(spot_token_lora spot_token_headonly)
ARM_FLAGS=(
    "--depth-cond.injection TOKEN --lora.enabled"
    "--depth-cond.injection TOKEN --lora.no-enabled"
)

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
DATA=/lustre/isaac24/proj/UTK0516/metrics_data/processed
ARKIT_OUT="${ARKIT_OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/processed_jd}"
# same exp_group as train_all_datasets.sh, so these sit beside the random-mask
# cells and can be filtered apart by depth_cond.sim_mode in the manifest
EXP_GROUP=metric_all_datasets
CKPT_DIR="${CKPT_DIR:-/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd}"
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
# absolute: the loop runs from $REPO/src, so a relative path would resolve
# against src/ instead of the repo root
SPOT_FREQ_MAP="${SPOT_FREQ_MAP:-$REPO/assets/spot/valid_freq_640x480.npz}"
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
# the mask is committed to the repo; a missing file means a bad checkout or a
# hand-edited SPOT_FREQ_MAP, and finetune_depth would refuse anyway
[ -f "$SPOT_FREQ_MAP" ] || {
    echo "[fatal] missing SPOT frequency map: $SPOT_FREQ_MAP"
    echo "        rebuild it with: python src/build_spot_freq_map.py --data-root <spot_data>"
    exit 1
}
for d in "$DATA/processed_scannetpp" "$DATA/processed_tartanair" "$DATA/processed_scannet" \
         "$ARKIT_OUT/processed_arkitscenes" "$ARKIT_OUT/processed_arkitscenes_highres"; do
    [ -d "$d" ] || { echo "[fatal] missing dataset root: $d"; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"

for i in "${!ARM_NAMES[@]}"; do
    echo "=== arm $i: ${ARM_NAMES[$i]}  (${ARM_FLAGS[$i]}) ==="
    # ARM_FLAGS[$i] is deliberately unquoted: it word-splits into separate args
    # shellcheck disable=SC2086
    python finetune_depth.py \
        --exp-group "$EXP_GROUP" \
        --pretrained "$PRETRAINED" \
        --save-dir "$CKPT_DIR" \
        ${ARM_FLAGS[$i]} \
        --depth-cond.heads DEPTH \
        --train.train-heads DEPTH \
        --depth-cond.sim-mode PIXEL_FREQ \
        --depth-cond.sim-freq-map-path "$SPOT_FREQ_MAP" \
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
                           "$ARKIT_OUT/processed_arkitscenes" \
                           "$ARKIT_OUT/processed_arkitscenes_highres" \
                           "$DATA/processed_hammer" \
        --val-dataset.dataset SCANNET ARKITSCENES_LOWRES ARKITSCENES_HIGHRES HAMMER \
        --val-dataset.stride-range 1 1 1 1 1 1 1 1 \
        --val-dataset.epoch-size 1000 1000 1000 1000 \
        --val-dataset.highres-root None "$ARKIT_OUT/processed_arkitscenes_highres" None None \
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
        || echo "=== arm $i did not run to completion (exists or failed); continuing ==="
done

echo "=== sweep loop finished ==="
