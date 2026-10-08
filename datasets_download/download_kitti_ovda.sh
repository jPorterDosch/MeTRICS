#!/bin/bash
#SBATCH --job-name=kitti_ovda_download
#SBATCH --account=isaac-utk0516
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/kitti_ovda_download_%j.out
#SBATCH --error=/nfs/home/jdosch1/brown-visual-computing/MeTRICS/logs/kitti_ovda_download_%j.out

# KITTI for reproducing Online Video Depth Anything's tables: the raw colour
# frames (image_02 and image_03 only) of the 138 drives that have
# data_depth_annotated/train GT, the set the authors evaluate
# (third_party/ovda/evaluation/sequences.csv). Each ~1.3 GB raw sync zip is
# streamed: downloaded, its two colour folders extracted, deleted. Kept: about
# 60-70 GB of PNGs (estimate); peak extra disk one zip.
#
# Nothing existing is overwritten: unzip -n never replaces a file, so the 13
# validation drives already on disk are left as they are. Resumable: a drive
# whose two colour folders exist is skipped.
#
# Then lays out $EVAL_RAW/kitti_ovda as the authors' kitti_pairs() expects
# (kitti_depth/data_depth_annotated/train, kitti_raw/<date>/<drive>/...),
# as two symlinks onto $EVAL_RAW/kitti -- no data is copied.
#
#   sbatch datasets_download/download_kitti_ovda.sh

set -euo pipefail
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
REPO="${REPO:-/nfs/home/jdosch1/brown-visual-computing/MeTRICS}"
CSV="$REPO/third_party/ovda/evaluation/sequences.csv"
KITTI="$EVAL_RAW/kitti"
ZIPS="$EVAL_RAW/kitti_zips_tmp"
mkdir -p "$ZIPS"

drives=$(awk -F, '$1=="kitti"{sub(/-image_0[23]$/,"",$3); print $3}' "$CSV" | sort -u)
echo "$(echo "$drives" | wc -l) drives"
for drive_sync in $drives; do
    drive="${drive_sync%_sync}"
    date="${drive:0:10}"
    if [ -d "$KITTI/$date/$drive_sync/image_02/data" ] && [ -d "$KITTI/$date/$drive_sync/image_03/data" ]; then
        continue
    fi
    zip="$ZIPS/${drive_sync}.zip"
    wget -q -c -O "$zip" "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${drive}/${drive_sync}.zip"
    unzip -q -n "$zip" "*/image_02/data/*" "*/image_03/data/*" -d "$KITTI"
    rm -f "$zip"
    echo "$(date '+%F %T') $drive_sync"
done
rmdir "$ZIPS" 2>/dev/null || true

mkdir -p "$EVAL_RAW/kitti_ovda/kitti_depth"
[ -e "$EVAL_RAW/kitti_ovda/kitti_depth/data_depth_annotated" ] || ln -s "$KITTI" "$EVAL_RAW/kitti_ovda/kitti_depth/data_depth_annotated"
[ -e "$EVAL_RAW/kitti_ovda/kitti_raw" ] || ln -s "$KITTI" "$EVAL_RAW/kitti_ovda/kitti_raw"
du -sh "$KITTI"
