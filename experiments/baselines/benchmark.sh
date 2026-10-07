#!/bin/bash
#SBATCH --job-name=bench_baseline
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_baseline_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_baseline_%j.out

# A VERIFIED baseline arm under OUR benchmark protocols (src/bench_baselines.py
# benchmark): published / sparse_aligned / metric per density and both TAEs,
# with the identical sparse-depth draws our model gets (same --seed and patch
# size as bench_checkpoint.py). Refused for an arm the reproduction record
# does not mark `verified`. Writes
#   /lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/baselines/bench_<ARM>/bench_results.json
# in bench_eval's layout, so it sits next to a run's bench_results.json.
#
#   ARM=vda sbatch -J bench_vda experiments/baselines/benchmark.sh
#   ARM=vda DATASETS="bonn_all scannet" sbatch ...            # subset
#   ARM=vda MAX_SEQ=2 DENSITIES=0.05 sbatch -p ai-tenn-debug -q ai-tenn-debug --time=01:00:00 ...   # smoke
#
# Budget (estimate): VDA-L fp32 at the five datasets, three densities, is one
# inference per sequence (the arm never sees the sparse depth), so ~1.5 h on an
# H100 plus the TAE pass (100 ScanNet scenes x 192 frames).

set -euo pipefail

ARM="${ARM:?set ARM to a verified arm (python src/bench_baselines.py status)}"
DATASETS=${DATASETS:-sintel scannet kitti bonn_all nyuv2}
TAE_DATASETS=${TAE_DATASETS:-sintel scannet kitti bonn_all}
DENSITIES=${DENSITIES:-0.01 0.05 0.4}
MAX_SEQ=${MAX_SEQ:-0}
REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
ENVS=/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export PYTHONPATH="$ENVS/baselines_pydeps${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "arm: $ARM datasets=$DATASETS tae=$TAE_DATASETS densities=$DENSITIES max_seq=$MAX_SEQ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
# shellcheck disable=SC2086
python src/bench_baselines.py benchmark --arm "$ARM" --datasets $DATASETS --tae-datasets $TAE_DATASETS \
    --densities $DENSITIES --max-sequences "$MAX_SEQ" "$@"
