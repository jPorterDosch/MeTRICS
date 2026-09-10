#!/bin/bash
#SBATCH --job-name=scannet_cleanup
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scannet_cleanup_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# Delete the ScanNet stage-2 intermediates once stage 3 has consumed them.
#
# extract_scannet.sh writes a frames.zip INTO each raw scene dir, next to the
# .sens. preprocess_scannet.sh is the only thing that ever reads them: the
# loader reads $METRICS_PROCESSED_ROOT/processed_scannet, and verify_scannet.py
# reads the .sens. So after stage 3 they are ~651 GiB of dead weight, and on a
# 10 TiB per-uid quota that is the difference between finishing TartanAir with
# 0.69 TiB to spare and finishing with 0.05 TiB.
#
# They are the right thing to drop rather than the .sens: regenerating them is
# ~1.5 h per shard of pure CPU, while re-fetching the .sens is a multi-hour
# download gated on a signed ScanNet ToU.
#
# Meant to be chained, so the ordering is enforced by Slurm rather than by
# remembering:
#   ejid=$(sbatch --parsable datasets_download/extract_scannet.sh)
#   pjid=$(sbatch --parsable --dependency=afterok:$ejid \
#              datasets_preprocess/preprocess_scannet.sh)
#   cjid=$(sbatch --parsable --dependency=afterok:$pjid \
#              datasets_preprocess/cleanup_scannet_frames.sh)
#   sbatch --dependency=afterok:$cjid datasets_preprocess/preprocess_tartanair.sh
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_preprocess/cleanup_scannet_frames.sh
#   bash   datasets_preprocess/cleanup_scannet_frames.sh
#
# DRY_RUN=1 reports what it would delete and touches nothing.

set -eu

# Resolving this script's own directory has to survive both invocation paths.
# Under sbatch, Slurm COPIES the script to /var/spool/.../slurm_script and runs
# it from there, so BASH_SOURCE points at the spool dir and every path built
# from it is wrong. `scontrol show job` still knows the path the job was
# submitted with, so ask Slurm inside a job and fall back outside one.
#
# Query SLURM_ARRAY_JOB_ID and keep only the first Command=: in a job ARRAY
# one task inherits the array's master job id, and `scontrol show job` on that
# id describes the array rather than the task. This script is not an array,
# but the block is shared across every wrapper here and must not diverge.
_rel="datasets_preprocess/cleanup_scannet_frames.sh"
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

# scans_train is a SYMLINK to scans, so listing both would visit every scene
# twice. Name the two real split dirs and let find refuse to follow symlinks
# (its default), rather than globbing scans*.
RAW_SPLITS=("$SCANNET_DIR/scans" "$SCANNET_DIR/scans_test")
for d in "${RAW_SPLITS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "raw split dir missing: $d" >&2
        exit 1
    fi
done

if [ ! -d "$OUT" ]; then
    echo "refusing to delete: no processed tree at $OUT" >&2
    exit 1
fi

# --- gate -------------------------------------------------------------------
# A dependency of afterok on preprocess_scannet.sh already means it exited 0,
# which now means zero scene failures. This is the second lock on a 651 GiB
# irreversible delete: count what stage 3 actually produced and refuse if it
# looks short. new_scene_metadata.npz is the real end of the pipeline -- it is
# what src/dust3r/datasets/scannet.py loads -- so it, not frames.zip, is the
# completeness signal.
n_raw=$(find "${RAW_SPLITS[@]}" -maxdepth 2 -name frames.zip | wc -l)
n_out=$(find "$OUT" -maxdepth 3 -name new_scene_metadata.npz | wc -l)

echo "raw scenes holding a stage-2 frames.zip : $n_raw"
echo "processed scenes with new_scene_metadata: $n_out"

if [ "$n_raw" -eq 0 ]; then
    echo "nothing to delete; already cleaned up."
    exit 0
fi

# preprocess_scannet.py legitimately skips scenes whose intrinsics are
# non-finite, so an exact match is not the right test -- but a large shortfall
# means stage 3 did not really finish and the intermediates are still needed.
min_out=$(( n_raw * 90 / 100 ))
if [ "$n_out" -lt "$min_out" ]; then
    echo "refusing to delete: only $n_out processed scenes against $n_raw" \
         "extracted (expected at least $min_out). Stage 3 looks incomplete --" \
         "re-run preprocess_scannet.sh, which resumes." >&2
    exit 1
fi

echo "freeing:"
du -sh --apparent-size "${RAW_SPLITS[@]}" 2>/dev/null || true

if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN set: would delete $n_raw frames.zip files; nothing removed."
    exit 0
fi

# -name frames.zip is deliberately narrow: the only other things in these dirs
# are the .sens files, and nothing here should ever match them.
find "${RAW_SPLITS[@]}" -maxdepth 2 -name frames.zip -delete

left=$(find "${RAW_SPLITS[@]}" -maxdepth 2 -name frames.zip | wc -l)
echo "deleted $((n_raw - left)) stage-2 archives; $left remain"
lfs quota -h -u "$USER" "$SCANNET_DIR" 2>/dev/null | tail -2 || true
