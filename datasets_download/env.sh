# Shared storage layout for the MeTRICS dataset download scripts.
#
# Every download_*.sh sources this file, so the storage location lives in
# exactly one place. Anything here can be overridden from the environment:
#
#   METRICS_DATA_ROOT=/some/other/place bash download_hammer.sh
#   SCANNET_DIR=/tmp/sn bash download_scannet.sh
#
# Layout:
#   $METRICS_DATA_ROOT/
#   ├── arkit_scenes/ hammer/ scannet/ scannetpp/ tartanair/   raw downloads
#   └── processed/                                             preprocess outputs

METRICS_DATA_ROOT="${METRICS_DATA_ROOT:-/lustre/isaac24/proj/UTK0516/metrics_data}"

# raw downloads, one dir per dataset
ARKIT_DIR="${ARKIT_DIR:-$METRICS_DATA_ROOT/arkit_scenes}"
HAMMER_DIR="${HAMMER_DIR:-$METRICS_DATA_ROOT/hammer}"
SCANNET_DIR="${SCANNET_DIR:-$METRICS_DATA_ROOT/scannet}"
SCANNETPP_DIR="${SCANNETPP_DIR:-$METRICS_DATA_ROOT/scannetpp}"
TARTANAIR_DIR="${TARTANAIR_DIR:-$METRICS_DATA_ROOT/tartanair}"

# where the datasets_preprocess/* scripts write their output
METRICS_PROCESSED_ROOT="${METRICS_PROCESSED_ROOT:-$METRICS_DATA_ROOT/processed}"

# repo root, derived from this file's location
METRICS_REPO="${METRICS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Interpreter every script in this directory runs Python through, as
# "$METRICS_PY". Point it at a REAL interpreter, not the pyenv shim: a shim
# resolves through pyenv's version files at run time and lands on the base
# 3.11.13, not the project virtualenv, so a job would silently use the wrong
# packages. Override for a different env, e.g.
#   METRICS_PY=~/miniconda3/envs/metrics/bin/python sbatch ...
METRICS_PY="${METRICS_PY:-$HOME/.pyenv/versions/3.11.13/envs/metrics/bin/python}"

# If that interpreter is missing, fall back to PATH rather than dying: the
# scripts call "$METRICS_PY" directly, so leaving it pointing at a nonexistent
# file would fail with a bare "No such file or directory" some lines later.
# Resolving it here means a call site never has to care which case it got.
if [ ! -x "$METRICS_PY" ]; then
    _metrics_fallback="$(command -v python3 || command -v python || true)"
    if [ -z "$_metrics_fallback" ]; then
        echo "env.sh: error: METRICS_PY=$METRICS_PY is not executable and no" \
             "python/python3 is on PATH. Set METRICS_PY to a real interpreter." >&2
        return 1 2>/dev/null || exit 1
    fi
    echo "env.sh: warning: METRICS_PY=$METRICS_PY is not executable;" \
         "falling back to $_metrics_fallback" >&2
    METRICS_PY="$_metrics_fallback"
    unset _metrics_fallback
fi
export METRICS_PY

# Also put it first on PATH, so a bare `python` in any child process (a
# setup.py, a nested tool) agrees with "$METRICS_PY". Idempotent.
export PATH="$(dirname "$METRICS_PY"):$PATH"

# NOTE: there are deliberately no Slurm account/partition variables here.
# #SBATCH directives are parsed by Slurm before the job's shell ever runs, so a
# variable set in this file could not affect them -- it would look configurable
# and silently do nothing. The account/partition/qos live in the #SBATCH block
# at the top of each download_*.sh; override per submission with
#   sbatch --account=... --partition=... --qos=... <script>
