#!/bin/bash
#SBATCH --job-name=scannet_extract
#SBATCH --array=0-31              # 32 shards; scenes assigned round-robin
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/scannet_extract_%A_%a.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Batch-extract raw ScanNet .sens -> per-frame color/depth/pose/intrinsic.
# Stage 1 of 3 (then preprocess_scannet.py, then generate_set_scannet.py).
# Resumable: already-extracted scenes are skipped, so re-submitting continues.

set -euo pipefail
mkdir -p logs

PY=/users/jdosch/miniconda3/envs/StreamVGGT/bin/python
REPO=/oscar/home/jdosch/MeTRIC
RAW=/gpfs/data/jtompki1/cli277/metric/scannet

cd "$REPO"

$PY datasets_preprocess/extract_scannet_sens.py \
    --raw-root "$RAW" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard "$SLURM_ARRAY_TASK_ID"
