#!/bin/bash
#SBATCH --job-name=arkit_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/arkit_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long

# ARKitScenes (https://github.com/apple/ARKitScenes)
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_arkit_scenes.sh   # batch (needs ./logs)
#   bash   datasets_download/download_arkit_scenes.sh   # login node
#
# Resumable: already-downloaded files/extracted dirs are skipped, so re-running
# after a timeout or failure continues where it left off. ~5 TB, so expect more
# than one submission.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# NB: the downloader appends <dataset>/<split> to this (download_arkit_scenes.py:239),
# so it must be a dataset-specific dir -- pointing it at a shared project root would
# create a bare "raw/" there.
DOWNLOAD_DIR="$ARKIT_DIR"
mkdir -p "$DOWNLOAD_DIR"

# Network-bound, so oversubscribe the CPU allocation under Slurm; stay modest
# when run by hand on a shared login node.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    NUM_WORKERS="${NUM_WORKERS:-16}"
else
    NUM_WORKERS="${NUM_WORKERS:-8}"
fi

# Reproduce the DUSt3R/StreamVGGT ARKitScenes recipe: two variants from ONE download of
# the full raw split. Both variants share the 640x480 vga_wide RGB + vga_wide_intrinsics
# + lowres_wide.traj and differ only in the depth source:
#   * lowres  -> preprocess_arkitscenes.py         (vga_wide + lowres_depth  LiDAR)      => processed_arkitscenes/
#   * highres -> preprocess_arkitscenes_highres.py (vga_wide + highres_depth laser GT)   => processed_arkitscenes_highres/
# highres_depth only exists for the ~2257 "upsampling" scenes; the downloader auto-skips
# it on the rest (download_arkit_scenes.py:66-70). The highres preprocess then "owns"
# those scenes and the lowres loader excludes them at load time (arkitscenes.py:85-90),
# so the two variants partition the scenes without overlap.
"$METRICS_PY" "$SCRIPT_DIR/download_arkit_scenes.py" raw \
    --download_dir "$DOWNLOAD_DIR" \
    --video_id_csv "$SCRIPT_DIR/raw/raw_train_val_splits.csv" \
    --raw_dataset_assets highres_depth lowres_depth vga_wide vga_wide_intrinsics lowres_wide.traj \
    --num_workers "$NUM_WORKERS"
