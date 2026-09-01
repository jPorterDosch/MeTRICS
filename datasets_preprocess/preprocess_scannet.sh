#!/bin/bash
#SBATCH --job-name=scannet_preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scannet_preprocess_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# Stage 3 + 4 of the ScanNet pipeline (after extract_scannet.sh): preprocess
# extracted frames -> processed color/depth/cam tree, then generate_set ->
# per-scene new_scene_metadata.npz (the loader input). CPU only.
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_preprocess/preprocess_scannet.sh   # batch (needs ./logs)
#   bash   datasets_preprocess/preprocess_scannet.sh   # login node
#
# Re-running after a failure redoes the preprocess from scratch (it has no
# per-scene skip logic) but simply overwrites, so it is safe.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../datasets_download/env.sh"

OUT="$METRICS_PROCESSED_ROOT/processed_scannet"

# keep tqdm from flooding the log (per-frame bars across 16 workers);
# honored by tqdm >= 4.66, harmless otherwise
export TQDM_MININTERVAL=60

cd "$METRICS_REPO"

"$METRICS_PY" datasets_preprocess/preprocess_scannet.py \
    --scannet_dir "$SCANNET_DIR" \
    --output_dir "$OUT"

# Max interval 150 is the default, hardcoded here for readability.
# Outside Slurm SLURM_CPUS_PER_TASK is unset. Fall back to 8, NOT nproc: the
# header offers a login-node run, where nproc reports the whole shared node.
"$METRICS_PY" datasets_preprocess/generate_set_scannet.py \
    --root "$OUT" \
    --splits scans_test scans_train \
    --max_interval 150 \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}"
