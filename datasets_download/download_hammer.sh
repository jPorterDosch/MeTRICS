#!/bin/bash
#SBATCH --job-name=hammer_download
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hammer_download_%j.out
#SBATCH --account=isaac-utk0516
#SBATCH --partition=campus
#SBATCH --qos=campus

# HAMMER (https://github.com/Junggy/HAMMER-dataset)
#
# Runs either way -- the #SBATCH block above is a comment to bash and a
# directive to Slurm:
#   sbatch datasets_download/download_hammer.sh   # batch (needs ./logs to exist)
#   bash   datasets_download/download_hammer.sh   # login node
#
# Downloads only the polarization (RGB) camera subset needed for CUT3R
# (~24 GB of the 170 GB official zip) via HTTP range requests.
# Single-stream, so this is network-bound rather than CPU-bound.
#
# Resumable: members already on disk at the expected size are skipped and every
# file is CRC32-checked, so re-running after a timeout continues safely.
#
# Requires outbound HTTPS to campar.in.tum.de. If the compute nodes are
# firewalled, run it on a login node with bash.

# -e so a failure here stops the script instead of falling through to the
# steps after it. NOT -o pipefail: a `yes`/`printf` feeding a prompt exits 141
# on SIGPIPE once the reader is done, and pipefail would turn that into a
# spurious failure after a successful run.
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$HAMMER_DIR"
"$METRICS_PY" "$SCRIPT_DIR/download_hammer.py" --out "$HAMMER_DIR"
