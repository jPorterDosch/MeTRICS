#!/bin/bash
#SBATCH --job-name=probe_window
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn-debug
#SBATCH --qos=ai-tenn-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/probe_window_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/probe_window_%j.out
# =============================================================================
# Peak memory and step time of the windowed training forward over clip length
# x KV window, with and without the recurrent scale token (src/
# probe_window_memory.py). On an H100, the GPU the sweep trains on.
#
#   REPO=<worktree> sbatch experiments/all_datasets_finetune/probe_window_memory.sh
#   GRID="16:8 20:10" sbatch ...          # other points (S:W, S:none = parallel)
#
# Writes $OUT (default logs/probe_window_<job>.json next to the log).
# =============================================================================
set -euo pipefail

REPO=${REPO:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS}
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
GRID=${GRID:-10:none 10:4 16:none 16:8 24:12 32:20}
OUT=${OUT:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/probe_window_${SLURM_JOB_ID:-local}.json}

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

[ -f "$PRETRAINED" ] || { echo "[fatal] missing $PRETRAINED"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[probe] REPO=$REPO HEAD=$(git -C "$REPO" rev-parse --short HEAD) grid=$GRID"
cd "$REPO/src"
# shellcheck disable=SC2086
python probe_window_memory.py --pretrained "$PRETRAINED" --grid $GRID --out "$OUT"
