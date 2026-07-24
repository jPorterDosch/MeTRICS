#!/bin/bash
#SBATCH --job-name=hypersim_download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hypersim_download_%j.out
# SBATCH --account=jdosch
#SBATCH --partition=batch

# Selective Hypersim download (~230GB, vs 1.57TB full dataset): only the
# render passes preprocess_hypersim.py consumes. The vendored downloader
# extracts matching members from Apple's per-scene zips via HTTP range
# requests -- nothing else is transferred. Members are STORED in the zips,
# so download bytes == on-disk bytes. Colour dominates (~214GB); depth
# compresses ~15x better (~14GB).
#
# SUBMIT WITH sbatch, NOT `bash`: running it in a login/interactive shell
# dies with the session (that is how the first attempt was lost at 160/457
# scenes). Resumable either way -- existing files are skipped by name, so
# re-submitting continues where it stopped.
#
# The downloader is single-threaded and re-reads each zip's central
# directory once per pass, so expect the full run to take many hours.
#
# NOTE: the downloader ANDs all --contains words together, so each pass
# pattern needs its own invocation.

set -euo pipefail
mkdir -p logs

PY=/users/jdosch/miniconda3/envs/StreamVGGT/bin/python
REPO=/oscar/home/jdosch/MeTRIC
OUT=/gpfs/data/jtompki1/cli277/metric/hypersim

mkdir -p "$OUT"
cd "$REPO/datasets_download"

# global camera-parameters CSV (preprocess reads it from the dataset root;
# it ships in the ml-hypersim repo, not in the scene zips)
[ -f "$OUT/metadata_camera_parameters.csv" ] || curl -sS --fail \
    "https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv" \
    -o "$OUT/metadata_camera_parameters.csv"

# The downloader retries individual range requests internally and exits
# nonzero if anything still could not be fetched. Loop each pass until it
# comes back clean (already-downloaded files are skipped by name, so a repeat
# sweep is cheap) rather than making a human requeue the job.
MAX_SWEEPS=5

for pattern in .color.hdf5 .depth_meters.hdf5 .render_entity_id.hdf5 _detail; do
    for sweep in $(seq 1 $MAX_SWEEPS); do
        echo "=== pass: $pattern (sweep $sweep/$MAX_SWEEPS) ==="
        if $PY download_hypersim_subset.py -d "$OUT" -c "$pattern" --silent; then
            echo "=== pass $pattern clean ==="
            break
        fi
        if [ "$sweep" -eq "$MAX_SWEEPS" ]; then
            echo "WARNING: $pattern still incomplete after $MAX_SWEEPS sweeps" >&2
        else
            echo "--- retrying $pattern after failures ---"
            sleep 60
        fi
    done
done
echo "done: $(date)"
echo "next: python verify_hypersim.py $OUT --delete"
