#!/bin/bash
#SBATCH --job-name=scannet_download
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/scannet_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long

# ScanNet v2 (http://www.scan-net.org)
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_scannet.sh   # batch (needs ./logs to exist)
#   bash   datasets_download/download_scannet.sh   # login node
#
# Stage 0 of 4 (then extract_scannet.sh, then preprocess_scannet.sh).
# Downloads only the .sens files, which is all the MeTRICS pipeline consumes:
#   1. datasets_preprocess/extract_scannet_sens.py  .sens -> color/depth/pose/intrinsic
#   2. datasets_preprocess/preprocess_scannet.py    -> processed color/depth/cam tree
#   3. datasets_preprocess/generate_set_scannet.py  -> new_scene_metadata.npz
# Dropping the meshes/labels/2d-zips cuts this from ~1.2 TB to the .sens subset.
#
# Network-bound: the downloader uses a 6-thread pool (download_scannet.py:78),
# hence 6 cpus. Requires outbound HTTP to kaldir.vc.cit.tum.de.
#
# Resumable: --skip_existing skips scenes already on disk, so re-run after the
# 48h walltime; expect more than one submission. Note it skips on *filename*,
# not size, so a .sens truncated by a killed job is NOT re-fetched -- delete a
# suspect scene dir before re-running. verify_scannet.py at the end catches it.
#
# Running this constitutes agreeing to the ScanNet Terms of Use:
# http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf
# The whole-release path prompts THREE times -- the TOS, a "press any key to
# continue" size warning, and a ".sens are the same as v1, press n to exclude"
# question -- so feed an unbounded stream of empty lines rather than a fixed
# count: one newline dies with EOFError at the second prompt, before a single
# byte is fetched. Empty is the right answer to all three (accept, continue,
# and do NOT exclude .sens). This is the unattended TOS acknowledgement.

# -e so a failure here stops the script instead of falling through to the
# steps after it. NOT -o pipefail: `yes` exits 141 on SIGPIPE once the reader
# is done, and pipefail would turn that into a spurious failure after a
# successful run.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

# TUM mails download-scannet.py out only after a signed ToU is approved, so it
# is deliberately NOT in this repo (see .gitignore). Fail here with an
# actionable message rather than letting python report "No such file".
if [ ! -f "$SCRIPT_DIR/download_scannet.py" ]; then
    echo "download_scannet.sh: $SCRIPT_DIR/download_scannet.py is missing." >&2
    echo "  It is not redistributable: mail a signed ScanNet Terms of Use" >&2
    echo "  (http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf) from an" >&2
    echo "  institutional address to scannet@googlegroups.com, then save the" >&2
    echo "  download-scannet.py they send you to that path." >&2
    exit 1
fi

mkdir -p "$SCANNET_DIR"

yes '' | "$METRICS_PY" "$SCRIPT_DIR/download_scannet.py" \
    -o "$SCANNET_DIR" \
    --type .sens \
    --skip_existing

# The downloader writes the train split to scans/ and the test split to
# scans_test/, but every downstream stage expects scans_train/ + scans_test/
# (preprocess_scannet.py:67 hardcodes that pair). Bridge with a symlink rather
# than moving ~1 TB of data.
if [ -d "$SCANNET_DIR/scans" ] && [ ! -e "$SCANNET_DIR/scans_train" ]; then
    ln -s scans "$SCANNET_DIR/scans_train"
    echo "linked $SCANNET_DIR/scans_train -> scans"
fi

# download_scannet.py prints per-scan failures but returns normally, so without
# this the job would report SUCCESS with scenes missing and extraction would
# then silently skip them. Exits non-zero on any gap.
"$METRICS_PY" "$SCRIPT_DIR/verify_scannet.py" "$SCANNET_DIR"
