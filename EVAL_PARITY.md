# Evaluator parity review

This is the Cycle 1, Phase 2 parity review for PR #26 (`review/codebase-sweep`). The operative parity baseline is the repository's import state, recovered offline from the object database: repository root `bff5684` (`Initial commit`, 2025-07-14); the imported evaluator files described below; and earliest `experiments/eval_all.sh` at `27d7a33` (the file was added twice).

`src/eval/monodepth/tools.py`, `src/eval/video_depth/tools.py`, `src/eval/video_depth/eval_depth.py`, `src/eval/mv_recon/criterion.py`, `src/eval/mv_recon/launch.py`, and `src/eval/pose_evaluation/test_co3d.py` were added by `49656da` (`Add files via upload`). `src/eval/temporal_consistency/metrics.py` was added by `de3ae67` (`Added validation loop and evaluation metric suite`). A repetition hazard: `git log --diff-filter=A --follow` misattributes the latter file to `49656da:src/eval/video_depth/tools.py` through `C057` copy detection; `49656da:src/eval/temporal_consistency/metrics.py` does not exist.

This review still cannot establish that those import blobs are verbatim upstream bytes: that requires network access unavailable in the sandbox. The import state is therefore the operative baseline, not proven upstream. `src/eval/mv_recon/criterion.py` differs from its import blob by one insertion and one deletion in the retained per-sample normalization change described below; most other `src/eval/` files are byte-identical, which bounds rather than eliminates this provenance risk.

Numerical parity outranks independent correctness here: a correction that changes a baseline-comparable number must either be reverted or explicitly opt-in with import-state behavior as the default. In every reverted or gated region below, a later comparison established that `main` and the import state agree; references to both are retained only where that historical comparison is relevant, not as competing baselines.

## Verdicts

| Rank | Commit and hunk | Verdict | Parity reason and disposition |
| --- | --- | --- | --- |
| 1 | `db38142`: custom-mask pixels excluded from affine fitting in the three depth evaluators | **REVERT — APPLIED** | Nontrivial custom masks change affine metrics. Restored import-state fitting and scoring arithmetic. The defect may re-land separately as an explicit default-off keyword selecting mask-aware fitting; not re-landed in this round. |
| 2 | `9b39ce6`: video `scale&shift` routes changed from Adam L1 to exact least squares | **REVERT — APPLIED** | This changes both the fitted objective and reported depth metrics. Restored the import-state Adam-L1 route. Exact L2 may re-land separately as an explicit alignment choice that defaults to Adam L1; not re-landed in this round. |
| 3 | `ebe25d7`: multi-rank global reduction and blended-loss reporting | **KEEP — REACHABILITY PROVED** | `_reduce_metrics` is called only by `finetune_depth.py`'s `val_loop` and `streaming_eval`; its output goes to `_log_val_stats` (W&B) and `val_loop`'s return, whose `absrel_metric_avg`/`loss_avg` selects our training checkpoint at lines 610–635. No `src/eval/` launcher imports or reaches `finetune_depth.py`. The seven pre-patch `checkpoints/*/.../wandb-summary.json` files carrying these averages are stale, but they are our-run monitoring/selection artifacts, not baseline scores. |
| 4 | `6005530`: z-buffered temporal reprojection | **REVERT — APPLIED** | Colliding reprojections change TAE, a DepthAnyVideo-attributed metric. Restored import-state last-write behavior. Z-buffering may re-land separately behind a default-off keyword; not re-landed in this round. |
| 5 | `f941f65`: clip-weighted loss averages | **KEEP — REACHABILITY PROVED** | `_accumulate_batch_loss` is called only inside `finetune_depth.py::val_loop`; its sums flow through `_reduce_metrics` to `_log_val_stats` and the `loss_avg` fallback for our checkpoint selection. No `src/eval/` launcher calls this training validation loop. The seven pre-patch `checkpoints/*/.../wandb-summary.json` files carrying loss averages are stale, but no baseline was scored through this path. |
| 5 | `2160b1c`: global loss medians | **KEEP — REACHABILITY PROVED** | The gather is inside `finetune_depth.py::_log_val_stats`; `val_loop` and `streaming_eval` are its only callers, and the resulting `*_med` keys are sent to W&B but are not read by checkpoint selection. No `src/eval/` launcher reaches it. The seven pre-patch `checkpoints/*/.../wandb-summary.json` files carrying loss medians are stale; they contain our training-monitoring numbers only. |
| 6 | `c3da4b7`: invalid-frame TAE reset | **KEEP — REACHABILITY PROVED** | `_streaming_depth_metrics` is called only by `finetune_depth.py::streaming_eval`; its TAE accumulators flow through `_reduce_metrics` to `_log_val_stats` for our W&B monitoring. Neither checkpoint selection nor any `src/eval/` launcher reaches this helper. The seven pre-patch `checkpoints/*/.../wandb-summary.json` files carrying `tae`/`tae_sq` are stale, but they are not baseline scores. |
| 7 | `5ba721c`: tuple finiteness filtering | **REVERT — APPLIED** | Non-finites change which point/color tuples enter reconstruction metrics. Restored the import-state component mask, including its fabricated-coordinate behavior. Tuple-wise filtering may re-land separately behind a default-off CLI option; not re-landed in this round. |
| 8 | `6fd172d`: rank-log collection | **REVERT — APPLIED** | More than eight ranks or a missing log followed by a present log changes the collected metric population. Baseline configurations cannot be proven to exclude both cases. Restored the import-state eight-rank cap and stop-at-gap behavior; dynamic collection may re-land separately behind a default-off CLI/config option. |
| 9 | `294039c`: contradictory alignment modes rejected | **GATE — APPLIED** | `reject_contradictory_modes=False` fits each existing evaluator signature without a helper or restructure. The default preserves import-state precedence; explicit strict mode rejects contradictory flags. |
| 10 | `08e3951`: per-sample point-cloud normalization | **KEEP — RECORDED** | `src/eval/mv_recon/launch.py:110` constructs `Regr3D_t_ScaleShiftInv(..., norm_mode=False, gt_scale=True)`, so its `compute_loss` skips both normalization calls and the changed function is unreachable from the shipped launcher. External callers that enable normalization can still observe different results. |
| 11 | `cf9a799`: `delta < 1.` boundary | **REVERT — APPLIED** | Bit-exact pixels change the nonstandard `delta < 1.` field. Restored the import-state strict comparison. The inclusive boundary may re-land separately behind a default-off metric option; not re-landed in this round. |

Every REVERT and GATE verdict in this table is represented by a later commit in this branch and is applied in the current tree.

## Cycle 1 actions checked against import state

All seven Cycle 1 revert or gate actions restore import behavior; none restores only `main`. `main` had not drifted in any reverted or gated region.

| Action | Evidence at import |
| --- | --- |
| `9e8b27a` revert `db38142` custom-mask affine fit | Both imported `tools.py` blobs (`49656da`, lines 178-291) and imported temporal metrics (`de3ae67`, lines 231-323) fit on the ordinary valid-pixel population and apply `custom_mask` only after alignment, for scoring. |
| `79ae439` revert `9b39ce6` affine route | Imported `eval_depth.py` selects `align_with_lad2=True` at all three `scale&shift` call sites (`49656da`, lines 124-130, 232-238, 332-338). |
| `15e8c4d` revert `6005530` z-buffering | Imported `point2depth` uses zero initialization plus advanced-index last-write assignment (`de3ae67`, lines 55-57). |
| `e6b93ef` revert `5ba721c` tuple finiteness | Imported `mv_recon/launch.py:327-331` filters component-wise, computes but does not use `mask_gt`, and indexes GT with the predicted component mask. |
| `a2c1e5a` revert `6fd172d` rank-log collection | Imported `mv_recon/launch.py:454-458` loops over `range(8)` and breaks at the first missing log. |
| `9a4798d` revert `cf9a799` delta-one boundary | All three imported evaluators compute strict `max_ratio < 1.0`. |
| `366e005` gate `294039c` | Imported evaluators resolve multiple true modes by ordered `if`/`elif` precedence and never reject; the default `reject_contradictory_modes=False` preserves exactly that. |

## Known defects restored for parity

The six applied reverts intentionally put these defects from the import state back into the evaluator. They remain live unless a future default-off correction is selected explicitly:

- Custom masks are applied only after affine fitting, so excluded pixels can still influence the fitted alignment.
- Video `scale&shift` uses the historical Adam L1 route rather than the exact affine solver.
- Temporal reprojection uses NumPy last-write collision handling rather than a nearest-depth z-buffer, so source order can change the projected depth.
- Reconstruction finiteness filtering applies a component-wise mask and can fabricate coordinate tuples instead of preserving point/ground-truth/color rows together.
- Reconstruction log collection examines at most ranks 0–7 and stops at the first missing rank log, so later present logs can be omitted.
- The reported `delta < 1.` metric uses a strict comparison, so even an exactly correct depth pixel does not satisfy it.

## Retained or gated divergences

The behavior columns below compare import state with the PR; “upstream behavior” means the operative import baseline, not a claim about true upstream source bytes.

| File and current line | Upstream behavior (import parity baseline) | Our behavior | Metrics affected | Datasets/configs affected | Is any existing reported number stale? |
| --- | --- | --- | --- | --- | --- |
| `src/eval/monodepth/tools.py:160`; `src/eval/video_depth/tools.py:160`; `src/eval/temporal_consistency/metrics.py:176` | Contradictory alignment modes are resolved by existing precedence and produce a score. | **GATED:** default preserves precedence; `reject_contradictory_modes=True` raises. | All depth metrics returned by the selected alignment path. | Only calls enabling more than one alignment mode; no shipped dataset is proven to exclude this misconfiguration. | Only results made with the opt-in strict mode would be unavailable rather than numerically stale; default results remain comparable. |
| `src/eval/mv_recon/criterion.py:178` | Normalization aggregates the batch when normalization is enabled. | Per-sample valid counts produce per-sample normalization factors. | Point-cloud reconstruction metrics for normalized calls. | External callers with normalization enabled. The shipped launcher fixes `norm_mode=False`, so shipped baseline configurations cannot reach it. | No shipped-launcher reported number is stale; external normalized results must record evaluator version. |
| `src/eval/temporal_consistency/metrics.py` (affine solver), introduced by pre-PR `6a1f5ee` | `absolute_value_scaling2`: a 1000-step Adam L1 fit. | `closed_form_scale_and_shift`: closed-form L2 normal equations. This PR did not introduce or remediate the divergence. | Affine AbsRel and RMSE; these are different estimators, not a bug fix. Probe (`scale&shift`, pred `[1,2,10]`, gt `[2,4,5]`): import AbsRel `0.9485160708`, RMSE `8.1417541504`; current AbsRel `0.2208903879`, RMSE `0.7167277336`. | Every validation-loop affine metric. | Any temporal-consistency affine number produced by the validation loop is not import-parity-comparable. |
| `src/eval/temporal_consistency/metrics.py` (pixel grid), introduced by pre-PR `6a1f5ee` | Half-pixel centers: `linspace(0.5, h-0.5, h)`. | Integer `arange` via `_pixel_grid`. This PR did not introduce or remediate the divergence. | Reprojection and therefore TAE. | Every non-identity temporal reprojection. A non-degenerate probe with realistic non-identity `K`, yaw plus translation at 8x10, 31x47, and 64x96 found lower TAE for the integer grid at every transformed resolution (64x96: integer `0.000299339` versus half-pixel `0.000554264`); both were exactly `0` under identity. The change is defensible and the code comment's rationale holds: this is a parity divergence, not a defect. | Any temporal-consistency TAE number produced by the validation loop is not import-parity-comparable. |

These two metric-arithmetic divergences are live, predate this PR, lie outside every Cycle 1 region, and were hidden by comparison only to `main`. They do not alter any Cycle 1 verdict.

The similarly labelled affine evaluators solve different objectives. Adam L1 is reached only through `src/eval/video_depth/eval_depth.py` (lines 132-169, 245-283, 347-380) and writes standalone `result_scale&shift.json`; closed-form L2 is reached only through `src/finetune_depth.py:1023-1025,1088-1090` and is logged as `absrel_affine`/`rmse_affine`. No tracked table, figure, summary script, or W&B panel consumes both, but a human could compare the similar labels by hand. Those numbers are not comparable.

## Undecidable divergence

`af60361` recomputes the criterion per clip at `src/finetune_depth.py:1222` and `:1395`. Its numerical effect is **UNDECIDABLE** without the prohibited real-data/GPU entrypoint. The settling run is: on the same checkpoint and the same real validation batch with `B=2`, run the configured criterion once on the batch and once through `_criterion_per_clip`; compare every `loss` and detail value and the final `_reduce_metrics` result. That run was not performed in this review.

The following pre-PR divergences are also **UNDECIDABLE** without unavailable real data. Neither run was performed; neither changes a Cycle 1 verdict.

| File and change | Import behavior | Current behavior | Metrics affected | Settling run |
| --- | --- | --- | --- | --- |
| `src/eval/pose_evaluation/test_co3d.py` (`c6437bd`) | JSON/NPZ sequence population and manual c2w inversion. | JGZ/PyTorch3D records with fast-eval sampling, and extrinsics built by `convert_pt3d_RT_to_opencv`. | CO3D Racc, Tacc, and AUC, through both population and GT-extrinsic changes. | Use the same checkpoint, seed, categories, and frame IDs through both complete pipelines on real annotations; compare generated GT extrinsic matrices first, then Racc/Tacc/AUC. |
| `experiments/eval_all.sh` (`74e3fd5`) | The hammer stage was not pinned with `--dataset hammer`. | `--dataset hammer` changes which clip is exported on a mixed HAMMER+ScanNet validation run. | The exported clip and paired CSV. | Use the same mixed-run checkpoint, config, and seed with import and current scripts; log the first sampled clip and compare the paired CSV. |

## Transient, self-cancelled history

- `196d8e2` removed `--landscape-crop --crop-anchor top` from the SPOT run; `3419f1f` restored it within this PR. There is no net divergence to decide.
- `2e89c94` made the zero-support check unconditionally raise where `main` returned an all-zero record; `091aeb5` restored the `main` behavior within this PR. There is no net divergence to decide.

## Existing local results that require care

No metric dump, comparison table, or notebook is tracked: `git ls-files eval_results viz logs checkpoints | wc -l` returned `0`. The following untracked user artifacts predate relevant evaluator patches and must not be reused as parity-comparable without re-scoring:

- `eval_results/video_depth/{sintel,bonn,kitti}_streamvggt/result_scale.json` from 2026-07-04. These use `align=scale`, so `9b39ce6` does not affect them; `cf9a799` affects them only if a post-scale pixel is bit-exact to ground truth.
- Seven `checkpoints/*/.../wandb-summary.json` summaries from 2026-07-21 through 2026-07-31 04:18 carrying `absrel_affine`, `rmse_affine`, `tae`, `tae_sq`, loss averages, and loss medians. All predate the 2026-07-31 14:39–18:12 evaluator patches.
- Pre-z-buffer `tcons` reports under `viz/eval_5978e1b00c3663be/`, `viz/hammer_stride_20/`, `viz/hammer_stride_1/`, `viz/hammer_stride_1_spot1/`, and `viz/eval_aaa6afa69197e511/`, through 2026-07-31 13:37.

The post-patch families `viz/eval_60ede373febf7c9b/` and `viz/spot_dyn_timing/` are not part of that stale pre-z-buffer list. No artifact listed here was modified, moved, deleted, or staged by this review.

## Deferred Area C findings under the parity lens

- **R2-3, no-overlap TAE:** this is a parity risk. Changing `(nan, nan)` or caller filtering would change the TAE denominator for clips with no reprojection overlap while depth-metric denominators remain unchanged. It is deferred; retain `main` behavior unless a correction is default-off.
- **R2-5, independently sorted prediction/GT pairing:** this is a parity risk. Adding identity/count failure or changing pairing can remove, reject, or remap evaluated sequences and therefore change every affected metric. It is deferred; retain `main` behavior until provenance establishes whether historical inputs ever mismatched.
