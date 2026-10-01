#!/bin/bash
#SBATCH --job-name=all_ds_mask
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_mask_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_mask_%j.out

# =============================================================================
# Masking-strategy sweep on the winning injection cell (TOKEN + LoRA, from the
# inject sweep on the rebuilt video data: best on 3 of 4 benchmark datasets).
# ONLY the training-time sparse-depth simulation changes:
#
#   arm  sim mode    density              what the model sees per clip
#   ---  ----------  -------------------  ---------------------------------------
#    0   RANDOM      5% of 14px patches   a new random patch set every frame
#    1   TUBE_MASK   5% of 14px patches   one patch set, fixed for the whole clip
#    2   PIXEL_FREQ  ~42% (map density)   per-pixel draws from the real SPOT
#                                         sensor's validity map
#
# Arm 0 is the inject sweep's token_lora cell with an identical config, so it
# hashes to the same run (e9263a9af33e8195) and finetune_depth refuses it:
# submit --array=1-2 and compare against that run. It is listed so the sweep
# reads as the full comparison.
#
# CONFOUND in arm 2: PIXEL_FREQ's density is the map's (~0.58 mask ratio), not
# 5% -- load_freq_map refuses any other ratio -- so 0 vs 1 is the clean
# pattern contrast (moving vs fixed patches at equal density), and 2 changes
# pattern AND density at once. The end-of-training benchmark scores all three
# identically (TUBE_MASK at 1/5/40%, plus real SPOT sensor input), which is
# where the density difference should show.
#
# Everything else is train_all_datasets_inject_sweep.sh verbatim -- data,
# stride (1,1), epoch sizes, 15 epochs, lr schedule, loss, heads -- plus the
# end-of-training benchmark (BENCH=0 skips it).
#
#   sbatch --array=1-2 experiments/all_datasets_finetune/train_all_datasets_spot_mask_sweep.sh
#
# Cost: ~23 h per arm on an H100 (the inject sweep's TOKEN+LoRA cell) plus
# ~6 h of benchmark; one arm per array task, inside the 3-day limit.
# =============================================================================

set -euo pipefail

ARM_NAMES=(random_token_lora tube_token_lora spot_token_lora)
REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
DATA=/lustre/isaac24/proj/UTK0516/metrics_data/processed
# The datasets rebuilt as video are NOT under $DATA, which still holds the
# PRE-REBUILD ScanNet++ (~143 thinned iPhone frames per scene plus DSLR stills,
# against ~637 contiguous iPhone frames here). Both load, so reading ScanNet++
# from $DATA silently trains on the old selection.
REBUILT="${REBUILT:-/lustre/isaac24/proj/UTK0516/metrics_data/processed_jd}"
ARKIT_OUT="${ARKIT_OUT:-$REBUILT}"
# same exp_group as train_all_datasets.sh, so these sit beside the random-mask
# cells and can be filtered apart by depth_cond.sim_mode in the manifest
EXP_GROUP=metric_all_datasets
CKPT_DIR="${CKPT_DIR:-/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd}"
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
# absolute: the loop runs from $REPO/src, so a relative path would resolve
# against src/ instead of the repo root
SPOT_FREQ_MAP="${SPOT_FREQ_MAP:-$REPO/assets/spot/valid_freq_640x480.npz}"
mkdir -p "$CKPT_DIR" "$REPO/logs"
ARM_FLAGS=(
    "--depth-cond.sim-mode RANDOM"
    "--depth-cond.sim-mode TUBE_MASK"
    "--depth-cond.sim-mode PIXEL_FREQ --depth-cond.sim-freq-map-path $SPOT_FREQ_MAP"
)
BENCH=${BENCH:-1}
BENCH_ROOT=${BENCH_ROOT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}
BENCH_ARGS=()
[ "$BENCH" = 1 ] && BENCH_ARGS=(--bench.enabled --bench.root "$BENCH_ROOT")

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
for d in "$REBUILT/processed_scannetpp" "$DATA/processed_tartanair" "$DATA/processed_scannet" \
         "$ARKIT_OUT/processed_arkitscenes" "$ARKIT_OUT/processed_arkitscenes_highres"; do
    [ -d "$d" ] || { echo "[fatal] missing dataset root: $d"; exit 1; }
done

if [ "$BENCH" = 1 ]; then
    for d in sintel scannet kitti bonn nyuv2; do
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
        --depth-cond.injection TOKEN \
        --lora.enabled \
        ${ARM_FLAGS[$i]} \
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
        "${BENCH_ARGS[@]}" \
        || echo "=== arm $i did not run to completion (exists or failed); continuing ==="
done

echo "=== sweep loop finished ==="
