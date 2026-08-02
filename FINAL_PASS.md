# PR #26 final pass — Phase 10 decisions

This document states the current Phase 10 position and preserves earlier states
as explicit history. Report durability is intentional: `CODE_REVIEW_FINDINGS.md`
and `IMPLEMENTATION_NOTES.md` stay local and untracked by user decision;
`FINAL_PASS.md` and `EVAL_PARITY.md` are the tracked deliverables. Item 9b was
deferred in rounds 1 and 2, then partly completed by `89f5a10` and `7140e32`:
the affine and rank-log regions are test-guarded, while component-mask tuple
fabrication remains byte-comparison-only. DUST-1 is the sole open item.

## Manager questions

### 1. Are the parity reverts intact?

Yes at the behavior-bearing regions, with two disclosed qualifications. The import byte probes matched the restored custom-mask fit (`9e8b27a`), Adam-L1 route (`79ae439`), last-write temporal reprojection (`15e8c4d`), eight-rank/stop-at-gap collection (`a2c1e5a`), and strict delta boundary (`9a4798d`). The component-mask region (`e6b93ef`) differs only because two trailing spaces present in `49656da` were removed; its executable statements match. The `f47ae0f` region deliberately does not match: `426809c` restores the fail-fast `ValueError` at the user's explicit PHASE 7 direction to prevent misleading findings.

Cycle 1 commit `1545cba` deleted four tests that pinned the affine route (two
tests), malformed point tuples, and rank cap/gap. At the Cycle 3 checkpoint the
affine and rank-log regions were byte-comparison-only; that stale state is
retracted. The affine route is test-guarded as of `7140e32`, and rank-log cap
and first-gap truncation are test-guarded as of `89f5a10`. Component-mask tuple
fabrication remains byte-comparison-only, the one parity region still without a
behavior test. Current tests also exercise restored import behavior for
custom-mask fitting, last-write reprojection, and strict delta.

Commands and literal results:

```text
byte-sequence probe:
9e8b27a mono custom-mask: import_count=1 head_count=1 byte_match=True
9e8b27a video custom-mask: import_count=1 head_count=1 byte_match=True
9e8b27a temporal custom-mask: import_count=1 head_count=1 byte_match=True
79ae439 route: import_count=6 head_count=3 byte_match=True
15e8c4d last-write: import_count=1 head_count=1 byte_match=True
e6b93ef component-mask: import_count=0 head_count=1 byte_match=False
a2c1e5a rank-cap: import_count=1 head_count=1 byte_match=True
9a4798d mono delta: import_count=1 head_count=1 byte_match=True
9a4798d video delta: import_count=1 head_count=1 byte_match=True
9a4798d temporal delta: import_count=1 head_count=1 byte_match=True
f47ae0f/426809c fail-fast divergence: import_count=0 head_count=1 authorized_divergence=True
```

The temporal file did not exist at `49656da`; its operative import blob is `de3ae67`, as already documented in tracked `EVAL_PARITY.md`. The correspondence file was compared to its copied import origin, `49656da:src/eval/mv_recon/dataset_utils/corr.py`.

### 2. Will all intended report deliverables survive merge?

Yes. The two local readability reports are intentionally untracked and are not
deliverables; `EVAL_PARITY.md` is the tracked parity report. The dangling tracked
reference to a local report has been removed while retaining the factual
zero-initialization statement.

```text
git ls-files --error-unmatch <local-review-report>
error: pathspec '<local-review-report>' did not match any file(s) known to git
exit 1

git ls-files --error-unmatch <local-implementation-report>
error: pathspec '<local-implementation-report>' did not match any file(s) known to git
exit 1

git ls-files --error-unmatch EVAL_PARITY.md
EVAL_PARITY.md
exit 0

git grep -n '<local report names>' HEAD
exit 1
```

### 3. Did this PR violate the vendored constraint?

DUST-1 remains open awaiting the user. The branch contains 13 changed lines in
read-only `src/dust3r/inference.py`; commit evidence shows these were the user's
own in-flight edits carried by `196d8e2`, not an agent editing vendored code.
The user must choose either keep-and-waive, or change both visualizers to stop
routing through the vendored entry point and then restore the vendored file.

Read-only call-graph investigation answers:

1. Every training and evaluator caller reaches `loss_of_one_batch`; inference
   callers reach the edited inference branch. Only the two visualizers pass
   `frame_times_ms` (`src/visualize_depth.py:936` and
   `src/visualize_spot.py:447`); all other tracked and untracked callers omit it.
2. First-party `StreamVGGT.inference` and the depth-conditioned override perform
   the timing, while the visualizers request and report it. The dust3r function
   is not a second implementation, but it is the current forwarding adapter
   between those first-party pieces and is therefore load-bearing for both
   visualizer entrypoints.
3. Removing the 13 lines without changing callers would break both visualizers
   at their unconditional `frame_times_ms=` keyword, even when timing is off;
   training and evaluator calls omit that keyword and would retain their old
   route. This conclusion is from the static call graph; no entrypoint ran.

Recommendation: relocate the optional timing forwarding to a first-party batch
inference wrapper and have the two visualizers use it, then restore the vendored
file. Until that replacement exists, removing the lines alone would break two
reachable first-party entrypoints, so keep the user decision open for the next
round rather than editing vendored code here.

```text
git diff --numstat main...HEAD -- src/dust3r/inference.py
11  2  src/dust3r/inference.py

git diff --raw main...HEAD -- src/dust3r/inference.py
:100644 100644 f5e585b b02a55f M src/dust3r/inference.py

git merge-base --is-ancestor 196d8e2 main
exit 1

git show -s --format='%h %s' 196d8e2
196d8e2 chore: carry the user's in-flight working-tree edits
```

`git blame -L 73,108 HEAD -- src/dust3r/inference.py` assigns every changed line (79, 81–83, 98–103, and 105) to `196d8e23`.

## Frozen contracts

They are unchanged. The metric-key CPU emitter passed and printed the per-dataset and `all/` key set. `val_images_smoke.py` separately emitted `val/samples`. Checkpoint selection still reads the flat `absrel_metric_avg` name with `loss_avg` fallback at `src/finetune_depth.py:633`. AST comparison against `main` proved every field and exclusion entry unchanged.

```text
PYTHONPATH=src .../python tests/val_metric_sections.py
_log_val_stats: ['val/all/absrel_metric_avg', 'val/all/loss_avg', 'val/all/loss_med', 'val/hammer/absrel_metric_avg', 'val/hammer/loss_avg', 'val/scannet/absrel_metric_avg', 'val/scannet/loss_avg']
PASS
exit 0

FinetuneDepthCfg: main_present=True head_present=True ast_byte_equal=True
_NON_IDENTITY_FIELDS: main_present=True head_present=True ast_byte_equal=True
```

## Full CPU gate

The planned pytest line cannot run in this environment:

```text
/users/jdosch/miniconda3/envs/StreamVGGT/bin/python -m pytest --version
/users/jdosch/miniconda3/envs/StreamVGGT/bin/python: No module named pytest
exit 1
```

I therefore ran the required direct equivalents with `PYTHONPATH=src` and `/users/jdosch/miniconda3/envs/StreamVGGT/bin/python`; the eight-script loop allowed 240 seconds per script.

| Command | Literal terminal result | Exit |
| --- | --- | ---: |
| `tests/val_images_smoke.py` | `PASS` | 0 |
| `tests/val_metric_sections.py` | `PASS` | 0 |
| `tests/val_loop_multi_loader.py` | `PASS` | 0 |
| `tests/test_loss_area_a.py` | no failure output | 0 |
| `tests/test_datasets_area_b.py` | no failure output | 0 |
| `tests/test_eval_area_c.py` | `Ran 11 tests ... OK` | 0 |
| `tests/test_train_area_d.py` | `Ran 15 tests ... OK` | 0 |
| `tests/test_depth_cond_area_e.py` | completed | 0 |
| `tests/test_review_diff_audit.py` | completed | 0 |

Ruff was run over all 42 Python files named by `git diff --name-only main...HEAD -- '*.py'`. For provenance, the same commands were run against the 36 paths that exist at `main`, materialized with `git archive main` under repository-local `.round9_main_lint/`; the other six paths are new tests and had no HEAD finding. The scratch-path prefix is the only path difference in the main commands.

```text
mapfile -t round9_py_files < <(git diff --name-only main...HEAD -- '*.py')
/users/jdosch/miniconda3/envs/StreamVGGT/bin/python -m ruff check "${round9_py_files[@]}"
Found 25 errors.
HEAD ruff check exit 1

/users/jdosch/miniconda3/envs/StreamVGGT/bin/python -m ruff format --check "${round9_py_files[@]}"
Would reformat: src/dust3r/inference.py
Would reformat: src/eval/mv_recon/criterion.py
Would reformat: src/eval/mv_recon/launch.py
Would reformat: src/eval/pose_evaluation/test_co3d.py
Would reformat: src/streamvggt/depth_cond/model.py
Would reformat: src/streamvggt/models/streamvggt.py
Would reformat: src/visualize_depth.py
7 files would be reformatted, 35 files already formatted
HEAD ruff format --check exit 1

/users/jdosch/miniconda3/envs/StreamVGGT/bin/python -m ruff check "${round9_main_args[@]}"
Found 25 errors.
main ruff check exit 1

/users/jdosch/miniconda3/envs/StreamVGGT/bin/python -m ruff format --check "${round9_main_args[@]}"
Would reformat: .round9_main_lint/src/dust3r/inference.py
Would reformat: .round9_main_lint/src/eval/mv_recon/criterion.py
Would reformat: .round9_main_lint/src/eval/mv_recon/launch.py
Would reformat: .round9_main_lint/src/eval/pose_evaluation/test_co3d.py
Would reformat: .round9_main_lint/src/streamvggt/models/streamvggt.py
5 files would be reformatted, 31 files already formatted
main ruff format --check exit 1
```

All 25 check violations reproduce on `main`; the PR introduces no new ruff rule violation. Their per-finding provenance is:

| Rule | HEAD file:line | Provenance |
| --- | --- | --- |
| F401 | `src/dust3r/inference.py:1` | pre-existing |
| F401 | `src/dust3r/inference.py:3` (`to_cpu`) | pre-existing |
| F401 | `src/dust3r/inference.py:3` (`collate_with_cat`) | pre-existing |
| F401 | `src/dust3r/inference.py:6` | pre-existing |
| F401 | `src/dust3r/inference.py:9` | pre-existing |
| F401 | `src/eval/mv_recon/criterion.py:5` | pre-existing |
| F401 | `src/eval/mv_recon/criterion.py:6` | pre-existing |
| F821 | `src/eval/mv_recon/criterion.py:132` | pre-existing |
| F841 | `src/eval/mv_recon/criterion.py:136` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:5` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:11` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:15` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:17` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:18` | pre-existing |
| F401 | `src/eval/mv_recon/launch.py:97` | pre-existing |
| F841 | `src/eval/mv_recon/launch.py:127` | pre-existing |
| F841 | `src/eval/mv_recon/launch.py:128` | pre-existing |
| E741 | `src/eval/mv_recon/launch.py:254` | pre-existing |
| F841 | `src/eval/mv_recon/launch.py:302` | pre-existing |
| F541 | `src/eval/mv_recon/launch.py:432` | pre-existing |
| E402 | `src/eval/mv_recon/launch.py:464` | pre-existing |
| F811 | `src/eval/mv_recon/launch.py:464` | pre-existing |
| E402 | `src/eval/mv_recon/launch.py:465` | pre-existing |
| F401 | `src/eval/pose_evaluation/test_co3d.py:10` | pre-existing |
| F401 | `src/eval/pose_evaluation/test_co3d.py:15` | pre-existing |

Format provenance is:

| Rule | File | Provenance |
| --- | --- | --- |
| format | `src/dust3r/inference.py` | pre-existing |
| format | `src/eval/mv_recon/criterion.py` | pre-existing |
| format | `src/eval/mv_recon/launch.py` | pre-existing |
| format | `src/eval/pose_evaluation/test_co3d.py` | pre-existing |
| format | `src/streamvggt/depth_cond/model.py` | introduced-by-this-PR |
| format | `src/streamvggt/models/streamvggt.py` | pre-existing |
| format | `src/visualize_depth.py` | introduced-by-this-PR |

Historical correction: at the Cycle 3 checkpoint no lint or formatting
violation had been fixed, and the two PR-owned format failures were outstanding.
That statement ceased to be current when `075ed03` fixed those two format
failures. Item 8 remains declined by the user: all 25 pre-existing ruff errors
and five pre-existing format failures are deliberately untouched because the
user did not want to risk parity against existing code.

## Uncovered

- Independent red reproduction covered 3 commits (`0a6a819`, `c03c57b`,
  `c018c86`) out of roughly 40 commits with test claims: about **3/40, or
  7.5%**. Phase 10 subsequently added five more literal red reproductions, but
  no new independent sampling denominator was established, so the independently
  reproduced coverage figure remains 7.5%. This is the central limitation.
- Nothing requiring a GPU, real data, real checkpoints, multi-rank execution, or a live wandb service was run.
- No training, evaluation, export, `experiments/eval_all.sh`, `srun`, `sbatch`, `tests/val_images_wandb_check.py`, or actual CUDA timing/cache arithmetic was run.
- The import blobs were not verified as verbatim upstream; that requires network access.
- Real evaluator numerical parity, DDP loader sharding, checkpoint failure injection, stale-result re-scoring, and every deferred experiment below remain unperformed.
- Ruff provenance found 25 rule violations and five format failures to be
  pre-existing. Two additional PR-owned format failures in
  `src/streamvggt/depth_cond/model.py` and `src/visualize_depth.py` were present
  at the Cycle 3 checkpoint and were fixed by `075ed03`.

## Current item status at HEAD

This ledger supersedes stale Cycle 3 conclusions while retaining their history:

1. **Resolved — durability.** The two local reports remain untracked by user
   decision, so this is not a defect. `FINAL_PASS.md` and `EVAL_PARITY.md` are
   the tracked deliverables.
2. **Resolved by `6b83815`.** The dangling tracked reference to the local
   implementation report was removed while its factual zero-init statement was
   preserved.
3. **Resolved by revert — DUST-1.** The earlier open finding is retained below
   as history. `bfad0af` restored `src/dust3r/inference.py` byte-for-byte from
   `main`; `6889915` moved the visualizers onto the existing first-party model
   inference path. The cost is a small first-party wrapper around that path;
   the per-frame timing columns remain available and no vendored logic was
   copied.
4. **Resolved by `2a6a2da`.** `reject_contradictory_modes` is documented as a
   library opt-in with no shipped caller; its false default preserves parity.
5. **Fixed by `8717715` — TU-1.** Global rank precedence is `RANK`,
   `SLURM_PROCID`, then `LOCAL_RANK`. It was rejected as pre-existing in Cycle 3
   and was later fixed by explicit user decision.
6. **Fixed by `118e683` — VS-1.** Empty sparse-depth windows report zero valid
   pixels without fabricating a statistic. It was rejected as pre-existing in
   Cycle 3 and was later fixed by explicit user decision.
7. **Fixed by `075ed03`.** The two PR-owned format failures were outstanding at
   the Cycle 3 checkpoint and were subsequently formatted; no other file was
   formatted for this item.
8. **Declined by the user.** The 25 pre-existing ruff errors and five
   pre-existing format failures remain deliberately untouched because changing
   existing code posed parity risk.
9. **Resolved.** The parity reverts remain intact at their behavior-bearing
   regions, subject to the disclosed whitespace-only component-mask difference
   and the authorized fail-fast divergence.
9b. **Partly completed after deferral.** Rounds 1 and 2 recorded this as not
   attempted; `89f5a10` and `7140e32` then added public-boundary rank-log and
   affine parity tests. Component-mask tuple fabrication remains
   byte-comparison-only.
10. **Declined by the user.** Teacher-confidence weighting was not changed.
11. **Declined by the user.** Inert `lambda_track` was not changed.
12. **Declined by the user.** Unused `CameraLoss.delta` was not changed.
13. **Accepted pre-existing, live defect.** Failed video-evaluation sequences
    can be omitted from the denominator at `49656da`, `489d28f`, and HEAD; the
    user chose not to address it here.
14. **Accepted pre-existing, live defect.** Prediction and GT files can be
    paired by independent sort position without identity/count assertions at
    `49656da`, `489d28f`, and HEAD; the user chose not to address it here.
15. **Fixed by `85163d2`.** All three prepared-loader kinds fail fast at zero
    batches; a one-batch loader does not trigger the guard.
16. **Accepted by the user.** The 6000-second peer timeout after a rank-0
    checkpoint-commit failure stands unchanged.

No GPU test ran during this documentation correction, no source or test file
was changed, and nothing was pushed.

## Phase 10, round 3 item 9b coverage

- The affine `scale&shift` route is now guarded at the public `main(args)`
  boundary: the emitted result records selection of the import-state Adam-L1
  route. The test is red at pre-revert `9b39ce6` and green at HEAD.
- The rank-log cap and first-gap truncation are now guarded at the public
  `main(args)` boundary: created rank logs demonstrate the eight-rank cap and
  exclusion of logs after the first gap. The test is red at pre-revert
  `6fd172d` and green at HEAD.
- Component-mask tuple fabrication remains byte-comparison-only. It was not
  reached before this round's hard timeout; its only public boundary is the
  substantially heavier nonempty reconstruction path through
  `eval.mv_recon.launch.main`.

The earlier statement that item 9b “was not attempted” remains true only as
round 1/2 history; it is not the current status after `89f5a10` and `7140e32`.

## Phase 10, round 2 decisions

- Item 10 (teacher-confidence weighting, A R1-6) was **declined by the user**.
  No code change was made; it is not an open review item.
- Item 11 (inert `lambda_track`, A R2-5) was **declined by the user**. No code
  change was made; it is not an open review item.
- Item 12 (unused `CameraLoss.delta`, A R2-8) was **declined by the user**. No
  code change was made; it is not an open review item.
- Item 13 (failed video-evaluation sequences silently omitted from the dataset
  denominator) is **accepted pre-existing behavior**. The defect is present at
  import state `49656da`, `main` (`489d28f`), and HEAD. It remains live, but
  the user chose not to address it here.
- Item 14 (GT and prediction files paired by independent sort position without
  identity or count assertions) is **accepted pre-existing behavior**. The
  defect is present at import state `49656da`, `main` (`489d28f`), and HEAD. It
  remains live, but the user chose not to address it here.
- Item 16 (rank-0 checkpoint-commit failure can leave peers waiting for the
  6000-second barrier timeout, SA R2-3) is **accepted by user decision**. No
  code change was made; it is not an open review item.

## Restated verdict

**FINAL VERDICT: DUST-1 is resolved by the user's explicit revert decision and
first-party rerouting; all other Phase 10 items have a current disposition
above.**

## Phase 11, round 1 DUST-1 resolution

- `src/dust3r/inference.py` was restored to the `main` blob without formatting
  or adjacent cleanup.
- Both visualizers now call the existing first-party streaming model inference
  path, whose base and depth-conditioned implementations already own per-frame
  timing. Measuring only around the complete call would have lost the
  per-frame columns; recreating the vendored wrapper would have duplicated its
  query-point and inference plumbing.
- The timing columns are preserved. The behavior cost is that these depth-only
  visualizers pass no tracking query points; they do not consume tracking
  outputs.
