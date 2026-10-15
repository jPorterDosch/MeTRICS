#!/bin/bash
#SBATCH --job-name=dump_baseline
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/dump_baseline_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/dump_baseline_%j.out

# Raw benchmark predictions of an arm that runs outside the metrics venv
# (dav: pinned torch/diffusers), for `benchmark --predictions` to score in the
# metrics venv (src/bench_baselines.py dump-predictions). Resumable: rerun the
# same command and finished sequences are skipped. The gate applies as for
# benchmark.sh: the arm must be verified (or close, with --allow-close). Writes
#   /lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/baselines/pred_<ARM>/
#
#   ARM=dav sbatch -J dump_dav experiments/baselines/dump_predictions.sh
#   ARM=dav DATASETS="nyuv2" sbatch ...          # subset (a later run fills in the rest)
# then, in the metrics venv:
#   ARM=dav sbatch experiments/baselines/benchmark.sh \
#       --predictions /lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/baselines/pred_dav
#
# Budget (estimate): dav runs 20 seeded passes per clip; the five benchmark
# datasets plus ScanNet's 100 TAE clips are ~1 GPU-day on an H100.

set -euo pipefail

ARM="${ARM:?set ARM (dav)}"
DATASETS=${DATASETS:-sintel scannet kitti bonn_all nyuv2}
TAE_DATASETS=${TAE_DATASETS:-sintel scannet kitti bonn_all}
MAX_SEQ=${MAX_SEQ:-0}
REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
ENVS=/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd

if [ "$ARM" = dav ]; then
    DAV_ENV=${DAV_ENV:-$ENVS/dav}
    [ -x "$DAV_ENV/bin/python" ] || { echo "[fatal] no venv at $DAV_ENV: run experiments/baselines/setup_dav_env.sh" >&2; exit 1; }
    export PATH="$DAV_ENV/bin:$PATH"
else
    export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
    export PYTHONPATH="$ENVS/baselines_pydeps${PYTHONPATH:+:$PYTHONPATH}"
fi
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "arm: $ARM datasets=$DATASETS tae=$TAE_DATASETS max_seq=$MAX_SEQ python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
# shellcheck disable=SC2086
python src/bench_baselines.py dump-predictions --arm "$ARM" --datasets $DATASETS \
    --tae-datasets $TAE_DATASETS --max-sequences "$MAX_SEQ" "$@"
