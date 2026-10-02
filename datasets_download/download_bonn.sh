# bonn -- the five sequences of DepthCrafter's benchmark/csv/meta_bonn.csv (the
# 5 x 110 protocol VDA reports against). NOTE person_tracking, not the
# person_tracking2 MonST3R uses (src/eval/video_depth/metadata.py),
# as per-sequence zips (~270 MB each) rather than the 16 GB full dataset zip;
# the Bonn server serves ~1 MB/s, so the full zip is hours for 21 unused
# sequences. Unpacks into rgbd_bonn_dataset/rgbd_bonn_<seq>/, the layout the
# full zip would give, so prepare_vda_benchmark.py reads either.
set -eu
EVAL_RAW="${EVAL_RAW:-/lustre/isaac24/proj/UTK0516/metrics_data/eval_jd/raw}"
mkdir -p "$EVAL_RAW/bonn/rgbd_bonn_dataset"
cd "$EVAL_RAW/bonn/rgbd_bonn_dataset"
for seq in balloon2 crowd2 crowd3 person_tracking synchronous; do
    # skipped once unpacked (the zip is deleted afterwards)
    [ -d "rgbd_bonn_$seq" ] && continue
    wget -c "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_${seq}.zip"
    unzip -q "rgbd_bonn_${seq}.zip"
    rm "rgbd_bonn_${seq}.zip"
    [ -d "rgbd_bonn_$seq" ] || { echo "rgbd_bonn_${seq}.zip did not unpack to rgbd_bonn_$seq" >&2; exit 1; }
done
cd ..
