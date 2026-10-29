#!/bin/bash
#SBATCH --job-name=slit_view
#SBATCH --account=isaac-utk0256
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/slit_view_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/slit_view_%j.out
# =============================================================================
# Slit-window view (src/slit_view.py), VDA's temporal-profile figure: one
# pixel column of every frame stacked over time, per model, aligned as the
# benchmark's `published` protocol scores it, with the benchmark's 5% draw.
# Default: the ScanNet TAE clips (192 frames), columns at 1/4, 1/2, 3/4 width,
#
#   Ours        the tube-mask run (checkpoint-last)
#   StreamVGGT  the pretrained backbone in that architecture (RGB only)
#   VDA         Video Depth Anything (verified arm)
#   DAV         Depth Any Video, from its dumped predictions (own venv)
#
#   sbatch experiments/all_datasets_finetune/slit_view.sh
#   SEQS="scene0707_00 scene0710_00" sbatch ...       # named clips
#   MAX_SEQ=1 sbatch -p ai-tenn-debug -q ai-tenn-debug --gres=gpu:h100:1 --time=01:00:00 ...   # smoke
#   EXTRA="--protocol metric" MODELS="Ours=ckpt:<run>" sbatch ...   # raw metric depth
#
# Then render on any node (seconds, no GPU):
#   python src/slit_view.py render --out <OUT>
# The job renders once at the end too.
# =============================================================================
set -euo pipefail

# the checkout sbatch was run from (a worktree runs its own code), not a
# hard-coded path; SLURM copies the script, so its own location is no guide
REPO=${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)}
CK=/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd/metric_all_datasets
RUN=${RUN:-$CK/50e636145fa9afbb}
DAV_DUMP=${DAV_DUMP:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/baselines/pred_dav}
DATASET=${DATASET:-scannet}
TAE=${TAE:-1}
SEQS=${SEQS:-}
MAX_SEQ=${MAX_SEQ:-4}
XS=${XS:-0.25 0.5 0.75}
EXTRA=${EXTRA:-}
TAG=$DATASET
if [ "$TAE" = 1 ]; then TAG=${TAG}_tae; fi
if [ -n "$SEQS" ]; then TAG=${TAG}_named; else TAG=${TAG}_n${MAX_SEQ}; fi
OUT=${OUT:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/slit_view/$(basename "$RUN")_${TAG}}

export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
# the arms' extra dependencies (easydict, ...), as experiments/baselines/benchmark.sh
export PYTHONPATH="/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd/baselines_pydeps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

[ -f "$RUN/checkpoint-last.pth" ] || { echo "[fatal] no checkpoint-last.pth in $RUN"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MODEL_ARGS=()
if [ -n "${MODELS:-}" ]; then
    read -r -a _models <<< "$MODELS"
    for m in "${_models[@]}"; do MODEL_ARGS+=(--model "$m"); done
else
    MODEL_ARGS=(--model "Ours=ckpt:$RUN" --model "StreamVGGT=base:$RUN"
                --model "VDA=arm:vda" --model "DAV=dump:$DAV_DUMP")
fi
SEQ_ARGS=(--max-sequences "$MAX_SEQ")
# shellcheck disable=SC2206
if [ -n "$SEQS" ]; then SEQ_ARGS=(--sequences $SEQS); fi
TAE_ARGS=()
if [ "$TAE" = 1 ]; then TAE_ARGS=(--tae); fi

cd "$REPO/src"
# shellcheck disable=SC2086
python slit_view.py extract --out "$OUT" --dataset "$DATASET" "${TAE_ARGS[@]}" \
    "${SEQ_ARGS[@]}" --x $XS "${MODEL_ARGS[@]}" $EXTRA
python slit_view.py render --out "$OUT"
echo "[slit] done -> $OUT"
