# PR #26 final pass — Phase 10 decisions

This round records the user's final-pass decisions without restating the merge
verdict. Report durability is intended: the two local readability reports remain
untracked and are not deliverables. Items 2, 4, 5, 6, and 7 were applied; item
8 is declined. Item 9b was not attempted because the remaining hard-timeout
budget could not safely accommodate three sanctioned source reverts and
restorations. Item 3 remains open for a user decision after the read-only
investigation recorded below.

## Manager questions

### 1. Are the parity reverts intact?

Yes at the behavior-bearing regions, with two disclosed qualifications. The import byte probes matched the restored custom-mask fit (`9e8b27a`), Adam-L1 route (`79ae439`), last-write temporal reprojection (`15e8c4d`), eight-rank/stop-at-gap collection (`a2c1e5a`), and strict delta boundary (`9a4798d`). The component-mask region (`e6b93ef`) differs only because two trailing spaces present in `49656da` were removed; its executable statements match. The `f47ae0f` region deliberately does not match: `426809c` restores the fail-fast `ValueError` at the user's explicit PHASE 7 direction to prevent misleading findings.

Cycle 1 commit `1545cba` deleted four tests that pinned the affine route (two tests), malformed point tuples, and rank cap/gap. That decision was accepted, but it leaves the byte comparison as the only remaining guard on those regions. Current tests still exercise restored import behavior for custom-mask fitting, last-write reprojection, and strict delta. Search found no test that asserts the reverted-away mask-aware fit, exact-L2 route, z-buffer, tuple-safe filtering, dynamic rank scan, or inclusive delta behavior.

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

The branch contains 13 changed lines in read-only `src/dust3r/inference.py`, so the literal tree constraint is violated. Commit evidence shows these were the user's own in-flight edits carried by `196d8e2`, not an agent editing vendored code. The change is an ordinary modification, forwards optional timing only when `frame_times_ms` is supplied, and is numerically inert for existing callers that omit it. I do not treat DUST-1 as independently merge-blocking once disclosed, but the user must explicitly keep/waive or relocate it.

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

No lint or formatting violation was fixed. The 25 rule violations and five pre-existing format failures are debt for the user to waive or address separately; the two introduced format failures remain part of this PR's NOT MERGE-READY basis.

## Uncovered

- Independent red reproduction covered 3 commits (`0a6a819`, `c03c57b`, `c018c86`) out of roughly 40 commits with test claims: about **3/40, or 7.5%**. Every other UPHELD red-before/green-after verdict still rests on a claim made by the same agent that wrote both the test and fix. This is the central limitation of the pass.
- Nothing requiring a GPU, real data, real checkpoints, multi-rank execution, or a live wandb service was run.
- No training, evaluation, export, `experiments/eval_all.sh`, `srun`, `sbatch`, `tests/val_images_wandb_check.py`, or actual CUDA timing/cache arithmetic was run.
- The import blobs were not verified as verbatim upstream; that requires network access.
- Real evaluator numerical parity, DDP loader sharding, checkpoint failure injection, stale-result re-scoring, and every deferred experiment below remain unperformed.
- Ruff provenance classification was performed after the rest of this gate: all 25 rule violations and five format failures are pre-existing, while the format failures in `src/streamvggt/depth_cond/model.py` and `src/visualize_depth.py` were introduced by this PR.

## User decisions owed

These are merger/product decisions, not omitted agent work:

1. **Area A R1-6:** choose teacher-confidence semantics for DISTILL after the specified controlled GPU experiment.
2. **Area A R2-5:** decide whether FINETUNE `lambda_track` should gain a real track term or reject/deprecate the no-op while preserving frozen identity compatibility.
3. **Area A R2-8:** establish whether `CameraLoss.delta` defines a robust-loss curve or should be deprecated as a compatibility API.
4. **Area C R2-3:** choose fail-fast completeness versus explicitly labelled partial scores for failed video sequences.
5. **Area C R2-5:** approve a canonical prediction/GT identity mapping after inspecting real NYUv2, Sintel, Bonn, and KITTI layouts.
6. **Self-audit R2-1:** decide the distributed policy for a rank whose prepared loader is empty after the specified two-rank reproduction.
7. **Self-audit R2-3:** decide the cross-rank checkpoint failure protocol after `fsync`/`os.replace` fault injection.
8. **DUST-1:** keep the 13 vendored lines and explicitly waive the read-only constraint, or restore the vendored file and move the timing shim into first-party code.
9. **`reject_contradictory_modes`:** reachability fact—only `tests/test_eval_area_c.py:183` passes `True`; no shipped path does. Cycle 2's separate justification is that this is a deliberate library-level opt-in for external callers whose false default preserves parity. Decide whether to keep that external API, wire a shipped selector, or remove it.
10. **TU-1 (`src/train_utils.py:27,41`):** decide rank-variable precedence. `LOCAL_RANK` outranks `SLURM_PROCID`, so multi-node process 8 can believe it is global rank zero. This is a live pre-existing defect left unfixed, not a disproved defect.
11. **VS-1 (`src/visualize_spot.py:431`):** decide behavior when `dvals` is empty; current min/max/median crashes. This is a live pre-existing defect left unfixed, not a disproved defect.
12. Resolved: the local readability reports intentionally remain untracked.
13. Resolved: remove the dangling tracked reference while preserving its factual statement.
14. Resolved: item 8 is declined; do not change the 25 pre-existing ruff rule violations or five pre-existing format failures.

No GPU test ran, no source/test fix was made, and nothing was pushed.

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
