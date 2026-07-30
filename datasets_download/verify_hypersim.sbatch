#!/bin/bash
#SBATCH --job-name=verify_hypersim
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/home/jdosch/MeTRIC/datasets_download/logs/verify_hypersim_%j.out
#SBATCH --error=/oscar/home/jdosch/MeTRIC/datasets_download/logs/verify_hypersim_%j.out

# =============================================================================
# Integrity sweep of the Hypersim RAW tree (266 GB of .hdf5).
#
# NOTE WHAT THIS DOES AND DOES NOT CHECK. verify_hypersim.py globs **/*.hdf5,
# so it verifies the RAW download only -- the PROCESSED tree is .png/.npy/.npz
# and is not touched by this. It is still the right first step for the open
# question, which is the count gap:
#     raw        458 scenes / 74,619 frames
#     processed  457 scenes / 73,627 frames
# Two skipped cams (ai_011_010/cam_00, ai_015_007/cam_00) explain part of it.
# If the remainder is corrupt raw .hdf5 that preprocessing silently skipped,
# this sweep finds it. A clean report means the gap is NOT raw corruption and
# the processed tree needs its own audit before the raw tree can be retired.
#
# CPU only (h5py reads), no GPU. It is I/O bound over 266 GB on /gpfs, hence
# the 12h wall -- it does not need it if the filesystem is quiet, but a
# timeout mid-sweep tells you nothing.
#
# REPORT ONLY -- deliberately NO --delete. The tree is in cli277's directory,
# not ours; a corrupt-file list is something to hand them, not to act on.
# Exit code 0 iff clean.
#
# Submit:  sbatch datasets_download/verify_hypersim.sbatch
# =============================================================================

set -euo pipefail

REPO=/oscar/home/jdosch/MeTRIC
RAW=/gpfs/data/jtompki1/cli277/metric/hypersim

export PATH=/users/jdosch/miniconda3/envs/StreamVGGT/bin:$PATH

mkdir -p "$REPO/datasets_download/logs"

echo "raw tree: $RAW"
echo "started:  $(date)"
echo "=============================================================="

# --min-age 0: nothing is in flight (no download job has run since 2026-07-24),
# so the possibly-still-writing guard would only hide real corruption.
python "$REPO/datasets_download/verify_hypersim.py" "$RAW" --min-age 0
status=$?

echo "=============================================================="
echo "finished: $(date)   exit=$status"
exit $status
