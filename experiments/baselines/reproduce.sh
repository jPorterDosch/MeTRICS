#!/bin/bash
#SBATCH --job-name=repro_baseline
#SBATCH --account=isaac-utk0256
#SBATCH --partition=campus-gpu-large
#SBATCH --qos=campus-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/repro_baseline_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/repro_baseline_%j.out

# Reproduce one baseline arm's own paper through our loaders and scoring
# (src/bench_baselines.py reproduce). A complete run writes the measured
# values, the commit and this job's id into
# src/eval/baselines/reproduction.json and recomputes the arm's status; until
# that says `verified`, the arm cannot be built for any other protocol.
#
#   ARM=vda        sbatch -J repro_vda        experiments/baselines/reproduce.sh
#   ARM=vda_metric sbatch -J repro_vda_metric experiments/baselines/reproduce.sh
#   ARM=ovda       sbatch -J repro_ovda --mem=300G experiments/baselines/reproduce.sh
#   ARM=promptda   sbatch -J repro_promptda   experiments/baselines/reproduce.sh
#   ARM=dav        sbatch -J repro_dav        experiments/baselines/reproduce.sh
#   ARM=vda MAX_SEQ=2 sbatch -p ai-tenn-debug -q ai-tenn-debug --gres=gpu:h100:1 --time=01:00:00 ...   # smoke test, record untouched
#
# Defaults are what verified vda and ovda (2026-10-08): one 32 GB V100S;
# 16 GB cards run out of memory on VDA-L. ovda needs --mem=300G: the
# authors' scoring holds a whole sequence in memory, and the 10,000-frame
# Bonn sequence peaked at 215 GB. Split a long arm with --datasets (any
# subset; jobs at the same commit combine into one verification).
#
# Before the first run of an arm, on a login node (compute nodes are offline):
#   python src/bench_baselines.py fetch --arm <arm>          # weights -> Lustre
# promptda also needs the ARKitScenes upsampling Validation fold (14.9 GB):
#   python datasets_download/download_arkit_scenes.py upsampling --split Validation \
#       --video_id_csv datasets_download/raw/upsampling_train_val_splits.csv --num_workers 8
# dav runs from its own venv (experiments/baselines/setup_dav_env.sh).
#
# Measured on one V100S (2026-10-08): vda scannet 3.3 h, scannet_500 5.8 h,
# kitti kitti_500 bonn_all bonn_all_500 sintel_bgr 2.3 h; ovda ovda_kitti
# 3.5 h, ovda_sintel ovda_bonn 1.3-1.5 h. The other arms: under 2 h each on
# an H100 (estimate).

set -euo pipefail

ARM="${ARM:?set ARM to one of: vda vda_metric ovda promptda dav}"
MAX_SEQ=${MAX_SEQ:-0}
REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
ENVS=/lustre/isaac24/proj/UTK0516/metrics_data/envs_jd

if [ "$ARM" = dav ]; then
    # Depth Any Video pins torch/diffusers/transformers versions the metrics
    # venv does not have; see setup_dav_env.sh
    DAV_ENV=${DAV_ENV:-$ENVS/dav}
    [ -x "$DAV_ENV/bin/python" ] || { echo "[fatal] no venv at $DAV_ENV: run experiments/baselines/setup_dav_env.sh" >&2; exit 1; }
    export PATH="$DAV_ENV/bin:$PATH"
else
    export PATH=/nfs/home/jdosch1/.pyenv/versions/3.11.13/envs/metrics/bin:$PATH
    # easydict (VDA / oVDA import it) lives in an overlay directory, not in
    # the metrics venv
    export PYTHONPATH="$ENVS/baselines_pydeps${PYTHONPATH:+:$PYTHONPATH}"
fi
export LD_LIBRARY_PATH=/nfs/home/jdosch1/.local/lib:${LD_LIBRARY_PATH:-}  # libbz2 shim
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "arm: $ARM max_seq=$MAX_SEQ python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
python src/bench_baselines.py reproduce --arm "$ARM" --max-sequences "$MAX_SEQ" "$@"
