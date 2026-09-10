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

# Resolving this script's own directory has to survive both invocation paths.
# Under sbatch, Slurm COPIES the script to /var/spool/.../slurm_script and runs
# it from there, so BASH_SOURCE points at the spool dir and every path built
# from it is wrong -- that is why sourcing env.sh failed with "No such file or
# directory". `scontrol show job` still knows the path the job was submitted
# with, so ask Slurm inside a job and fall back to BASH_SOURCE outside one.
# (sbatch --test-only does NOT run the body, so it cannot catch a bug here.)
#
# Query SLURM_ARRAY_JOB_ID, not SLURM_JOB_ID, and keep only the first
# Command=. In a job ARRAY exactly one task inherits the array's master job id
# -- observed on 6177682_31, whose JobIDRaw was the array id 6177682 itself --
# and `scontrol show job` on that id describes the ARRAY rather than the one
# task, so the sed did not yield a single usable path. That task then exited 1
# before doing any work, which is invisible until an afterok dependent sits at
# DependencyNeverSatisfied. Every other task in the array was unaffected,
# which is what made it look like a data problem.
#
# SLURM_SUBMIT_DIR is the belt to that braces: Slurm sets it to the directory
# the job was submitted from, and every documented invocation here submits
# from the repo root.
_rel="datasets_download/download_arkit_scenes.sh"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _self="$(scontrol show job "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" 2>/dev/null \
             | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)"
    if [ -n "$_self" ] && [ -f "$_self" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$_self")" && pwd)"
    elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/$_rel" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$SLURM_SUBMIT_DIR/$_rel")" && pwd)"
    elif [ -n "${METRICS_REPO:-}" ] && [ -f "$METRICS_REPO/$_rel" ]; then
        SCRIPT_DIR="$(cd "$(dirname "$METRICS_REPO/$_rel")" && pwd)"
    else
        echo "could not resolve this script's path: scontrol gave" \
             "'${_self:-<empty>}', SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}'." \
             "Set METRICS_REPO to the repo root, submit from it, or run this" \
             "with bash instead of sbatch" >&2
        exit 1
    fi
    unset _self
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
unset _rel
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
