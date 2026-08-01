# Evaluator parity review

This is the Cycle 1, Phase 2 parity review for PR #26 (`review/codebase-sweep`). The protected and reachable parity baseline is the evaluator as it stood on `main` at `489d28f`, because that is the code with which numbers in this repository could actually have been produced.

This review did **not** compare the vendored files with upstream bytes. The sandbox has no network, all attempted StreamVGGT, CUT3R, MonST3R, DUSt3R, and DepthAnyVideo clones failed with `Could not resolve host: github.com`, and no upstream checkout is available. Whether the six `main` blobs match true upstream is still open. The settling operation is to byte-compare those six blobs from `main` with the matching StreamVGGT revision and its CUT3R/DUSt3R ancestors once those sources are reachable.

Numerical parity outranks independent correctness here: a correction that changes a baseline-comparable number must either be reverted or explicitly opt-in with behavior from `main` as the default.

## Verdicts

| Rank | Commit and hunk | Verdict | Parity reason and disposition |
| --- | --- | --- | --- |
| 1 | `db38142`: custom-mask pixels excluded from affine fitting in the three depth evaluators | **REVERT — APPLIED** | Nontrivial custom masks change affine metrics. Restored `main` fitting and scoring arithmetic. The defect may re-land separately as an explicit default-off keyword selecting mask-aware fitting; not re-landed in this round. |
| 2 | `9b39ce6`: video `scale&shift` routes changed from Adam L1 to exact least squares | **REVERT — APPLIED** | This changes both the fitted objective and reported depth metrics. Restored the `main` Adam-L1 route. Exact L2 may re-land separately as an explicit alignment choice that defaults to Adam L1; not re-landed in this round. |
| 3 | `ebe25d7`: multi-rank global reduction and blended-loss reporting | **REVERT** | Unequal multi-rank shards change every validation metric. Restore `main`. A corrected reducer may re-land separately behind a default-off config flag; not re-landed in this round. |
| 4 | `6005530`: z-buffered temporal reprojection | **REVERT — APPLIED** | Colliding reprojections change TAE, a DepthAnyVideo-attributed metric. Restored last-write behavior from `main`. Z-buffering may re-land separately behind a default-off keyword; not re-landed in this round. |
| 5 | `f941f65`: clip-weighted loss averages | **REVERT** | Uneven final batches change loss averages. Restore `main`. Clip weighting may re-land separately behind a default-off config flag; not re-landed in this round. |
| 5 | `2160b1c`: global loss medians | **REVERT** | Differing multi-rank shards change loss medians. Restore `main`. Global medians may re-land separately behind a default-off config flag; not re-landed in this round. |
| 6 | `c3da4b7`: invalid-frame TAE reset | **REVERT** | Valid/invalid/valid gaps change which temporal pair is scored. Restore cross-gap scoring from `main`. Reset-on-gap may re-land separately behind a default-off config flag; not re-landed in this round. |
| 7 | `5ba721c`: tuple finiteness filtering | **REVERT — APPLIED** | Non-finites change which point/color tuples enter reconstruction metrics. Restored `main`'s component mask, including its fabricated-coordinate behavior. Tuple-wise filtering may re-land separately behind a default-off CLI option; not re-landed in this round. |
| 8 | `6fd172d`: rank-log collection | **REVERT — APPLIED** | More than eight ranks or a missing log followed by a present log changes the collected metric population. Baseline configurations cannot be proven to exclude both cases. Restored `main`'s eight-rank cap and stop-at-gap behavior; dynamic collection may re-land separately behind a default-off CLI/config option. |
| 9 | `294039c`: contradictory alignment modes rejected | **GATE — APPLIED** | `reject_contradictory_modes=False` fits each existing evaluator signature without a helper or restructure. The default preserves `main` precedence; explicit strict mode rejects contradictory flags. |
| 10 | `08e3951`: per-sample point-cloud normalization | **KEEP — RECORDED** | `src/eval/mv_recon/launch.py:110` constructs `Regr3D_t_ScaleShiftInv(..., norm_mode=False, gt_scale=True)`, so its `compute_loss` skips both normalization calls and the changed function is unreachable from the shipped launcher. External callers that enable normalization can still observe different results. |
| 11 | `cf9a799`: `delta < 1.` boundary | **REVERT — APPLIED** | Bit-exact pixels change the nonstandard `delta < 1.` field. Restored strict comparison from `main`. The inclusive boundary may re-land separately behind a default-off metric option; not re-landed in this round. |

Decisions marked REVERT or GATE but not represented by a later commit in this branch remain decided but not yet applied.

## Retained or gated divergences

The behavior columns below compare `main` with the PR; “upstream behavior” means the reachable parity baseline on `main`, not a claim about true upstream source bytes.

| File and current line | Upstream behavior (`main` parity baseline) | Our behavior | Metrics affected | Datasets/configs affected | Is any existing reported number stale? |
| --- | --- | --- | --- | --- | --- |
| `src/eval/monodepth/tools.py:152`; `src/eval/video_depth/tools.py:152`; `src/eval/temporal_consistency/metrics.py:173` | Contradictory alignment modes are resolved by existing precedence and produce a score. | **GATED:** default preserves precedence; `reject_contradictory_modes=True` raises. | All depth metrics returned by the selected alignment path. | Only calls enabling more than one alignment mode; no shipped dataset is proven to exclude this misconfiguration. | Only results made with the opt-in strict mode would be unavailable rather than numerically stale; default results remain comparable. |
| `src/eval/mv_recon/criterion.py:178` | Normalization aggregates the batch when normalization is enabled. | Per-sample valid counts produce per-sample normalization factors. | Point-cloud reconstruction metrics for normalized calls. | External callers with normalization enabled. The shipped launcher fixes `norm_mode=False`, so shipped baseline configurations cannot reach it. | No shipped-launcher reported number is stale; external normalized results must record evaluator version. |

## Undecidable divergence

`af60361` recomputes the criterion per clip at `src/finetune_depth.py:1222` and `:1386`. Its numerical effect is **UNDECIDABLE** without the prohibited real-data/GPU entrypoint. The settling run is: on the same checkpoint and the same real validation batch with `B=2`, run the configured criterion once on the batch and once through `_criterion_per_clip`; compare every `loss` and detail value and the final `_reduce_metrics` result. That run was not performed in this review.

## Transient, self-cancelled history

- `196d8e2` removed `--landscape-crop --crop-anchor top` from the SPOT run; `3419f1f` restored it within this PR. There is no net divergence to decide.
- `2e89c94` made the zero-support check unconditionally raise where `main` returned an all-zero record; `091aeb5` restored the `main` behavior within this PR. There is no net divergence to decide.

## Existing local results that require care

No metric dump, comparison table, or notebook is tracked: `git ls-files eval_results viz logs checkpoints | wc -l` returned `0`. The following untracked user artifacts predate relevant evaluator patches and must not be reused as parity-comparable without re-scoring:

- `eval_results/video_depth/{sintel,bonn,kitti}_streamvggt/result_scale.json` from 2026-07-04. These use `align=scale`, so `9b39ce6` does not affect them; `cf9a799` affects them only if a post-scale pixel is bit-exact to ground truth.
- Six `checkpoints/*/.../wandb-summary.json` summaries from 2026-07-21 through 2026-07-31 04:18 carrying `absrel_affine`, `rmse_affine`, `tae`, `tae_sq`, loss averages, and loss medians. All predate the 2026-07-31 14:39–18:12 evaluator patches.
- Pre-z-buffer `tcons` reports under `viz/eval_5978e1b00c3663be/`, `viz/hammer_stride_20/`, `viz/hammer_stride_1/`, `viz/hammer_stride_1_spot1/`, and `viz/eval_aaa6afa69197e511/`, through 2026-07-31 13:37.

The post-patch families `viz/eval_60ede373febf7c9b/` and `viz/spot_dyn_timing/` are not part of that stale pre-z-buffer list. No artifact listed here was modified, moved, deleted, or staged by this review.

## Deferred Area C findings under the parity lens

- **R2-3, no-overlap TAE:** this is a parity risk. Changing `(nan, nan)` or caller filtering would change the TAE denominator for clips with no reprojection overlap while depth-metric denominators remain unchanged. It is deferred; retain `main` behavior unless a correction is default-off.
- **R2-5, independently sorted prediction/GT pairing:** this is a parity risk. Adding identity/count failure or changing pairing can remove, reject, or remap evaluated sequences and therefore change every affected metric. It is deferred; retain `main` behavior until provenance establishes whether historical inputs ever mismatched.
