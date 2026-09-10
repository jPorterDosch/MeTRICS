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
_rel="datasets_preprocess/preprocess_scannet.sh"
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
