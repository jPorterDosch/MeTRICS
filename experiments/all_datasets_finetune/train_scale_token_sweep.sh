#!/bin/bash
#SBATCH --job-name=all_ds_scaletok
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_scaletok_%A_%a.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/all_ds_scaletok_%A_%a.out
# =============================================================================
# Scale-token ablation on the five-dataset mixture, after the depth-dropout
# drift run showed every depth-conditioned model losing its metric scale
# within ~30 frames of a dropout: median |log s| at frame 191 of 0.35 (tube
# run 50e636145fa9afbb) and 0.39 (conv-stem 960e8e2d2bc2b5cc), worse than the
# RGB-only LoRA arm (c46b17af477343ce, 0.07), while PromptDA holding its last
# depth frame reaches 0.19. Source: eval_jd/scale_drift/
# drop20_all_models_w20_v2/summary.json (100 ScanNet TAE clips, 5% tube
# draw, dropout at frame 20, KV window 20, A40; PromptDA job 6509066).
#
#   arm  scale token              readout     role
#   ---  -----------------------  ----------  --------------------------------
#    0   none                     -           control: the windowed training
#                                             and depth dropout alone
#    1   recurrent                aux         token carried across frames,
#                                             supervised by a train-only head
#    2   recurrent                factorised  depth = exp(readout) x depth
#                                             head: the token IS the scale
#    3   no carry                 aux         a token with no explicit carry:
#                                             it reaches past frames only
#                                             through attention to the cached
#                                             K/V (multi-hop, as every token
#                                             in every arm does)
#
# Every arm: the conv-stem encoder (PREFILL_CONV, the encoder sweep's best,
# 960e8e2d2bc2b5cc, no edge loss), token injection, LoRA, tube-mask sparsity,
# and the changes to training that the token needs to mean anything:
#   --train.window 8          the clip runs one frame at a time through a KV
#                             cache cut to the last 8 frames, the window these
#                             arms are also DEPLOYED with (scale_drift.py
#                             --window 8, the long-clip table), so training
#                             and deployment see the same cache. Close to the
#                             backbone's own 10-frame pretraining context.
#   --train-dataset.num-views 16
#                             clips long enough to evict: frame t sees frames
#                             t-8..t, so frames 9-15 run without frame 0 (the
#                             parallel forward on 10-frame clips never evicts)
#   --train.depth-dropout-p   half the clips lose their sparse depth from a
#                             random frame in [1, 15] on, so holding a scale
#                             without measurements is something training
#                             asks. A frame with NO fed frame left in its
#                             cache needs t >= dropout frame + 8: only clips
#                             dropped by frame 7, at their last frames.
#   --epochs 10               budgeted by frame exposures, not steps: 16-frame
#                             clips at 10 epochs (45,000 steps) show 1.07x the
#                             frames of the encoder sweep's 10-frame clips at
#                             15 (67,500), which also keeps an arm inside the
#                             3-day wall clock (cost below)
# Readout target: the log of the mean valid GT depth over frames 0..t,
# causal like the stream (depth_cond/scale_token.py), L1, weight 0.1.
#
# Contrasts: 3 vs 0 = an extra (supervised) token, which also moves the
# pretrained output from step 0 (it starts as a copy of register 0); 1 vs 3 =
# the explicit carry; 2 vs 1 = factorised vs auxiliary readout.
#
# The end-of-run benchmark (bench_eval) streams with an UNBOUNDED cache, as
# for every other row of the main table: these arms train with 8 cached
# frames and are scored there with all of them. Arm 0 carries the same
# mismatch, so the contrasts above hold; a contrast against the encoder sweep
# mixes in the window change. Their deployment numbers come from window 8:
# scale_drift.py --window 8 (rerun the earlier models at 8 too, so every curve
# shares the deployment setting) and the long-clip table. The non-learned bar is
# not a training arm: scale_drift.py's +ema correction on arm 0 (and on the
# encoder sweep's conv-stem run) is what the token has to beat.
#
#   sbatch --array=0-3 experiments/all_datasets_finetune/train_scale_token_sweep.sh
#   REPO=<worktree> sbatch --array=0-3 ...   # before the branch is merged
#
# Extra arguments go to finetune_depth.py after the ones above (the last
# occurrence of a flag wins), e.g. a smoke run of two arms, a few steps each:
#
#   BENCH=0 REPO=<worktree> sbatch --array=1,2 -p ai-tenn-debug -q ai-tenn-debug \
#       --time=01:00:00 experiments/all_datasets_finetune/train_scale_token_sweep.sh \
#       --exp-group scale_token_smoke --epochs 1 \
#       --train-dataset.epoch-size 8 8 8 8 8 --val-dataset.epoch-size 4 4 4 4
#
# Cost (estimate, not measured): the conv-stem arm of the encoder sweep took
# 41 h 57 m on an H100 (job 6463259: 67,500 steps, ~1.9 s/step, with
# validation and the end-of-run benchmark). A 16-frame windowed step is
# estimated at ~1.8x, so 45,000 steps at ~3.4 s ~ 43 h + validation and
# benchmark, ~50 h per arm. The smoke run above measures the step time (and,
# with --print-freq 1, the logged max mem) before this is queued. BENCH=0
# skips the end-of-run benchmark.
# =============================================================================

set -euo pipefail

ARM_NAMES=(window_ctrl token_rec_aux token_rec_fact token_cache_aux)
COMMON="--depth-cond.encoder PREFILL_CONV --train.window 8 --train-dataset.num-views 16 --train.depth-dropout-p 0.5"
ARM_FLAGS=(
    "$COMMON --scale-token.no-enabled"
    "$COMMON --scale-token.enabled --scale-token.recurrent --scale-token.readout AUX"
    "$COMMON --scale-token.enabled --scale-token.recurrent --scale-token.readout FACTORISED"
    "$COMMON --scale-token.enabled --scale-token.no-recurrent --scale-token.readout AUX"
)

REPO=${REPO:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS}  # a worktree before merge
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
# its own group: four arms, one ladder
EXP_GROUP=metric_scale_token_sweep
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
        --epochs 10 \
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
        "$@" \
        || echo "=== arm $i did not run to completion (exists or failed); continuing ==="
done

echo "=== sweep loop finished ==="
