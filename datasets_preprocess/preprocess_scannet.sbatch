#!/bin/bash
#SBATCH --job-name=scannet_preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scannet_preprocess_%j.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Stage 2 + 3 of the ScanNet pipeline (after extract_scannet.sbatch):
# preprocess extracted frames -> processed color/depth/cam tree, then
# generate_set -> per-scene new_scene_metadata.npz (the loader input).
# CPU only. Re-running after a failure redoes the preprocess from scratch
# (it has no per-scene skip logic) but simply overwrites, so it is safe.

set -euo pipefail
mkdir -p logs

PY=/users/jdosch/miniconda3/envs/StreamVGGT/bin/python
REPO=/oscar/home/jdosch/MeTRIC
RAW=/gpfs/data/jtompki1/cli277/metric/scannet
OUT=/gpfs/data/jtompki1/cli277/metric/processed_scannet

# keep tqdm from flooding the log (per-frame bars across 16 workers);
# honored by tqdm >= 4.66, harmless otherwise
export TQDM_MININTERVAL=60

cd "$REPO"

$PY datasets_preprocess/preprocess_scannet.py \
    --scannet_dir "$RAW" \
    --output_dir "$OUT"

# Max interval 150 is the default, hardcoded here for readability
$PY datasets_preprocess/generate_set_scannet.py \
    --root "$OUT" \
    --splits scans_test scans_train \
    --max_interval 150 \
    --num_workers "$SLURM_CPUS_PER_TASK"
