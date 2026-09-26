#!/bin/bash
#SBATCH --job-name=bench_smoke
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn-debug
#SBATCH --qos=ai-tenn-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_smoke_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_smoke_%j.out

# GPU smoke test of the end-of-training benchmark (src/bench_eval.py) on the
# PRETRAINED backbone with zero-init conditioning -- the baseline arm -- via
# the --epochs 0 pure-eval path. No training, no val loaders (--val-freq 0),
# so the only thing that runs is the benchmark: one sequence per dataset at
# one density by default, every mode, both TAEs, clouds + GLBs.
#
#   sbatch experiments/all_datasets_finetune/bench_smoke.sh
#   DATASETS="scannet" DENSITIES="0.01 0.05 0.4" sbatch experiments/all_datasets_finetune/bench_smoke.sh
#   MAX_SEQ=0 sbatch ...          # the full benchmark (~1 h), still the baseline arm
#
# Needs the benchmark tree (prepare_vda_benchmark.sbatch). The training
# mixture is still constructed (finetune_depth builds it before anything
# else), which is a few minutes of dataset walking -- expected.
#
# Output: <CKPT_ROOT>/bench_smoke/<run_id>/bench_results.json + bench_clouds/,
# and final_bench/* in wandb (offline unless .secrets/wandb-personal.env).

set -euo pipefail

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
CKPT_ROOT=/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
BENCH_ROOT=${BENCH_ROOT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}
DATASETS=${DATASETS:-sintel scannet kitti bonn nyuv2}
DENSITIES=${DENSITIES:-0.05}
MAX_SEQ=${MAX_SEQ:-1}

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [ -f "$REPO/.secrets/wandb-personal.env" ]; then
    # shellcheck disable=SC1091
    source "$REPO/.secrets/wandb-personal.env"
else
    echo "[warn] no .secrets/wandb-personal.env -- running wandb offline"
    export WANDB_MODE=offline
fi

[ -f "$PRETRAINED" ] || { echo "[fatal] pretrained weights not found: $PRETRAINED"; exit 1; }
for d in $DATASETS; do
    ls "$BENCH_ROOT/$d"/*.json >/dev/null 2>&1 || { echo "[fatal] no manifest under $BENCH_ROOT/$d"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"
# shellcheck disable=SC2086
python finetune_depth.py \
    --exp-group bench_smoke \
    --save-dir "$CKPT_ROOT" \
    --pretrained "$PRETRAINED" \
    --epochs 0 \
    --val-freq 0 \
    --bench.enabled \
    --bench.root "$BENCH_ROOT" \
    --bench.datasets $DATASETS \
    --bench.densities $DENSITIES \
    --bench.cloud-density "${DENSITIES%% *}" \
    --bench.max-sequences "$MAX_SEQ"

echo "done: $(ls -d "$CKPT_ROOT"/bench_smoke/*/ | tail -n 1)bench_results.json"
