# oVDA evaluation protocol

Two reference scripts explain the published evaluation:

- [`protocol.py`](protocol.py): fit inverse depth, convert to metres, and compute AbsRel and delta1. Requires NumPy only.
- [`data_sorting.py`](data_sorting.py): pair and order RGB/GT frames. Uses the Python standard library only.

[`sequences.csv`](sequences.csv) records the exact sequence names, historical order, frame counts, and first/last paired frames. [`bonn_pairs.json`](bonn_pairs.json) records every historical Bonn RGB/depth correspondence. These are reference data; neither script reads them.

## Alignment and scoring

1. Use GT in **metres**, with dataset-valid pixels `0 < GT < limit`: Sintel **10000**, KITTI **255.9**, Bonn **100**.
2. For first-frame alignment, fit `a * prediction + b` to `1 / GT` using every dataset-valid pixel of the first **paired** frame. **GT beyond 80 m stays in the fit, without clipping.** Global alignment fits all paired frames instead. Each sequence has its own fit.
3. The original convention is `scale = 1/a`, `shift = -b/a`. Apply `(prediction - shift) / scale`, replace **exactly zero** inverse depths with `1e-4`, take the reciprocal, then clip predicted metric depth to `[0, 80]`. Negative inverse depth becomes zero metric depth and remains an error.
4. Score only dataset-valid pixels with **original GT < 80 m**. GT at or beyond 80 m is excluded, rather than clipped and retained.
5. **Exclude frame 0** of every sequence from scoring. Add error sums, delta1 outlier counts, and valid-pixel counts over the whole dataset. Divide once; do not average frame or sequence means. Delta1 counts ratios **<= 1.25** as correct.

Predictions and GT must have matching `(frames, height, width)` shapes. Historical arrays were float32; original prediction resizing used bilinear interpolation with `align_corners=True`. Use RGB inputs and `infer_video_depth(input_size=518)`. Model inference and depth-file decoding are outside these small reference functions. KITTI PNG GT is divided by 256; Bonn PNG GT by 5000; Sintel `.dpt` is already in metres.

```python
from evaluation.protocol import evaluate_sequence, summarize

# inverse_prediction and gt_metres are arrays for one ordered sequence.
totals = evaluate_sequence(inverse_prediction, gt_metres, "sintel", alignment="first")
print(summarize([totals]))

# For the dataset, append one totals dictionary per sequence, then summarize.
# alignment="all" gives the global-fit protocol.
```

## Data pairing

| Dataset | Selection | Paired sequences | Paired frames | Mean length |
| --- | --- | ---: | ---: | ---: |
| Sintel | `training/final`, all scenes | 23 | 1064 | 46.2609 |
| KITTI | 138 `data_depth_annotated/train` drives; both cameras | 276 | 85898 | 311.2246 |
| Bonn | All 26 sequences | 26 | 29064 | 1117.8462 |

Sintel pairs matching frame IDs with `training/depth`. KITTI pairs identical filenames in annotation and raw RGB directories, with **image_03 before image_02** as independent sequences. Feed only paired RGB frames to the model; no Eigen/Garg crop is used.

Bonn uses `rgb.txt` and `depth.txt`, **timestamp matching**, zero offset, and a strict difference **< 0.02 seconds**. Sort candidates by `(difference, RGB timestamp, depth timestamp)`, greedily accept pairs whose timestamps are unused, then sort matches by RGB path. Repeated timestamps retain the last entry; discard pairs with missing files. This explains why extra RGB frames are not evaluated. Historical scene order followed directory iteration; the CSV freezes that order. Use sequence names when comparing different filesystem copies.

## Reference results

These values were recovered from the original evaluation records:

| Dataset | First-frame AbsRel / delta1 | Global AbsRel / delta1 |
| --- | --- | --- |
| Sintel | 0.379854371 / 0.547662502 | 0.294014490 / 0.603567900 |
| KITTI | 0.139869800 / 0.809422488 | 0.112348743 / 0.876156608 |
| Bonn | 0.118279751 / 0.870804365 | 0.109251045 / 0.887003540 |

Fresh released-oVDA inference on all Sintel sequences gave **0.379632528 / 0.547859875** for first-frame alignment, rounding to Table 1's **0.380 / 0.548**. Global alignment gave **0.294031296 / 0.603484525**; delta1 is just below the 0.6035 rounding boundary. Fresh historical inference reproduced both paper rows at three decimals. Full KITTI/Bonn inference was not rerun; their archived results and pairing were checked. Small floating-point differences are expected; these functions use float64 error accumulation instead of historical float32 sums.

For Table 8's 500-frame variant, the old notebook fitted frames **0–499** but scored **1–500**, where available. That variant is a historical slicing detail; the two modes above cover first-frame and full-sequence global alignment.
