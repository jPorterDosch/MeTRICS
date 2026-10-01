#!/bin/bash
#SBATCH --job-name=bench_ckpt
#SBATCH --account=isaac-utk0256
#SBATCH --partition=ai-tenn
#SBATCH --qos=ai-tenn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_ckpt_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/bench_ckpt_%j.out

# Video-depth benchmark (src/bench_checkpoint.py) on an already trained run:
# the newest checkpoint-<CKPT>.pth under CKPT_ROOT by default, like
# visualize_latest.sh. Writes <run>/bench_<CKPT>/bench_results.json + clouds.
#
#   sbatch experiments/all_datasets_finetune/bench_checkpoint.sh
#   WEIGHTS=/path/to/run_dir sbatch ...                 # a specific run
#   CKPT=last sbatch ...                                # checkpoint-last instead of best
#   DATASETS="scannet bonn" DENSITIES="0.05" sbatch ... # subset
#   MAX_SEQ=2 sbatch ...                                # smoke test
#   BASE=1 sbatch ...                                   # pretrained backbone, same config
#   IMAGE_SIZE=1036 DATASETS=kitti sbatch ...           # 2x input (long side); default 518
#   OUT_DIR=/path sbatch ...                            # write results elsewhere
#   sbatch ... --bench.spot-starts 998                  # extra args go to bench_checkpoint.py
# SPOT (spot_data/seq_0, seq_1) is scored on every run; a quick SPOT-only
# look: DATASETS=nyuv2 MAX_SEQ=4 DENSITIES=0.05 OUT_DIR=<run>/bench_spot
# (MAX_SEQ also caps the SPOT windows, so 4 = both captures x both starts)
#
# Measured: the full benchmark at three densities is ~5.5 h on one H100
# (ScanNet alone ~4.5 h: 100 scenes x 3 densities + the TAE split; each other
# dataset 10-20 min) plus SPOT, so it runs on ai-tenn, not the 3 h debug QOS.
# A subset (DATASETS=..., DENSITIES=0.05, MAX_SEQ=...) fits ai-tenn-debug:
#   sbatch -p ai-tenn-debug -q ai-tenn-debug --time=03:00:00 ...

set -euo pipefail

REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
CKPT_ROOT=/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd
PRETRAINED=${PRETRAINED:-/lustre/isaac24/proj/UTK0516/ckpt/checkpoints.pth}
BENCH_ROOT=${BENCH_ROOT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/bench}
CKPT="${CKPT:-best}"
DATASETS=${DATASETS:-sintel scannet kitti bonn nyuv2}
DENSITIES=${DENSITIES:-0.01 0.05 0.4}
MAX_SEQ=${MAX_SEQ:-0}
BASE=${BASE:-0}
IMAGE_SIZE=${IMAGE_SIZE:-518}
# clouds at 5% (the headline density) when it is swept, else the first density
case " $DENSITIES " in *" 0.05 "*) CLOUD_DENSITY=${CLOUD_DENSITY:-0.05} ;; *) CLOUD_DENSITY=${CLOUD_DENSITY:-${DENSITIES%% *}} ;; esac

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [ -z "${WEIGHTS:-}" ]; then
    newest="$(find "$CKPT_ROOT" -name "checkpoint-${CKPT}.pth" -printf '%T@ %p\n' \
              2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
    [ -n "$newest" ] || { echo "no checkpoint-${CKPT}.pth under $CKPT_ROOT" >&2; exit 1; }
    WEIGHTS="$(dirname "$newest")"
fi
for d in $DATASETS; do
    ls "$BENCH_ROOT/${d%_500}"/*.json >/dev/null 2>&1 || { echo "[fatal] no manifest under $BENCH_ROOT/${d%_500}"; exit 1; }
done
BASE_ARGS=()
[ "$BASE" = 1 ] && BASE_ARGS=(--base --pretrained "$PRETRAINED")
# a non-default input size writes to its own dir, so it never overwrites the 518 run
OUT_ARGS=()
if [ "$IMAGE_SIZE" != 518 ]; then
    OUT_ARGS=(--out-dir "$WEIGHTS/bench_${CKPT}_s${IMAGE_SIZE}")
    [ "$BASE" = 1 ] && OUT_ARGS=(--out-dir "$WEIGHTS/bench_base_s${IMAGE_SIZE}")
fi
[ -n "${OUT_DIR:-}" ] && OUT_ARGS=(--out-dir "$OUT_DIR")
echo "image size: $IMAGE_SIZE"
echo "weights: $WEIGHTS (checkpoint-${CKPT}.pth) base=$BASE datasets=$DATASETS densities=$DENSITIES clouds@$CLOUD_DENSITY max_seq=$MAX_SEQ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO/src"
# shellcheck disable=SC2086
python bench_checkpoint.py \
    --weights "$WEIGHTS" --checkpoint "$CKPT" "${BASE_ARGS[@]}" \
    --bench.root "$BENCH_ROOT" \
    --bench.datasets $DATASETS \
    --bench.densities $DENSITIES \
    --bench.cloud-density "$CLOUD_DENSITY" \
    --bench.max-sequences "$MAX_SEQ" \
    --bench.image-size "$IMAGE_SIZE" \
    "${OUT_ARGS[@]}" \
    "$@"
