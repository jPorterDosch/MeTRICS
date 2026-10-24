#!/bin/bash
#SBATCH --job-name=eval_download
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/eval_download_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/eval_download_%j.out

# Raw sources for the video-depth benchmark (Video Depth Anything protocol:
# Sintel 23x50, KITTI 13x110, Bonn 5x110, NYUv2 654 stills), fetched by the
# per-dataset scripts in this directory, in sequence, into
#   $EVAL_RAW = /lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw/<name>
#
#   sbatch datasets_download/download_eval_benchmarks.sbatch
#   DATASETS="bonn nyu" sbatch datasets_download/download_eval_benchmarks.sbatch
#
# long/isaac-utk0516, as download_arkit_scenes.sh: those CPU nodes have
# outbound network and a 48 h cap. The first attempt ran on ai-tenn-debug and
# timed out at 3 h with ~700 KB/s to the KITTI S3 bucket -- Sintel finished,
# KITTI did not. ~55 GB total (KITTI 14 GB annotated + 13 raw drives, Bonn
# 5 x 0.27 GB, Sintel ~7 GB, NYU 3 GB); every wget is -c and the per-dataset
# scripts skip what is already unpacked, so a rerun resumes rather than
# restarts.
#
# ScanNet is not fetched: VDA's 100 scenes are the first 100 of scans_test
# sorted, i.e. metrics_data/scannet/scans_test on Lustre already (.sens only).

set -eu

# under sbatch the script runs from a Slurm spool copy, so the sibling scripts
# are located by fixed repo path rather than BASH_SOURCE
REPO=/nfs/home/jdosch1/brown-visual-computing/MeTRICS
export EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
mkdir -p "$EVAL_RAW"

for name in ${DATASETS:-sintel kitti bonn nyu}; do
    echo "== $name -> $EVAL_RAW/$name  ($(date))"
    bash "$REPO/datasets_download/download_${name}.sh"
done

echo "== done ($(date))"
du -sh "$EVAL_RAW"/*
