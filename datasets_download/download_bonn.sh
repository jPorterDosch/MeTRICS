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

# BONN_ALL=1: the other 21 sequences too (~14.5 GB), into bonn_full/ next to
# symlinks of the five above -- the 26-sequence set Video Depth Anything's
# Bonn numbers are over (prepare_vda_benchmark.py bonn_all).
if [ "${BONN_ALL:-0}" = 1 ]; then
    mkdir -p "$EVAL_RAW/bonn_full"
    cd "$EVAL_RAW/bonn_full"
    for seq in balloon2 crowd2 crowd3 person_tracking synchronous; do
        [ -e "rgbd_bonn_$seq" ] || ln -s "../bonn/rgbd_bonn_dataset/rgbd_bonn_$seq" "rgbd_bonn_$seq"
    done
    for seq in balloon balloon_tracking balloon_tracking2 crowd kidnapping_box \
        kidnapping_box2 moving_nonobstructing_box moving_nonobstructing_box2 \
        moving_obstructing_box moving_obstructing_box2 person_tracking2 \
        placing_nonobstructing_box placing_nonobstructing_box2 \
        placing_nonobstructing_box3 placing_obstructing_box \
        removing_nonobstructing_box removing_nonobstructing_box2 \
        removing_obstructing_box synchronous2 static static_close_far; do
        [ -d "rgbd_bonn_$seq" ] && continue
        wget -c "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_${seq}.zip"
        unzip -q "rgbd_bonn_${seq}.zip"
        rm "rgbd_bonn_${seq}.zip"
    done
    cd ..
fi
