# Depth Anything 3 -- vendored model code

Source: https://github.com/ByteDance-Seed/Depth-Anything-3 at commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` (2026-07-27), fetched 2026-10-10:
the `src/depth_anything_3/` package verbatim (without `__pycache__`), and
`LICENSE` from the repository root. License: Apache-2.0. The released
DA3METRIC-LARGE weights (Apache-2.0, `depth-anything/DA3METRIC-LARGE`) are not
vendored; `src/bench_baselines.py fetch --arm da3_metric` downloads them.

What is used, and from where:

| file | used by | for |
|---|---|---|
| `src/depth_anything_3/api.py` | `src/eval/baselines/arms.py::DA3MetricArm` | `DepthAnything3.from_pretrained` and `inference` (`process_res=504`, `lower_bound_resize`) |
| `src/depth_anything_3/{cfg,registry,specs}.py`, `model/`, `configs/`, `utils/` | imported by `api.py` | the network, input/output processing, sky post-process |
| `src/depth_anything_3/utils/alignment.py::apply_metric_scaling` | reference only | the arm repeats its rule, metric depth = depth * mean(fx, fy) / 300 with the intrinsics at the processed resolution |

Nothing in this directory is edited. `api.py` imports
`depth_anything_3.utils.export` at module level, and that package needs
`moviepy`, `plyfile` and others the `metrics` venv lacks; the export path is
never used for inference, so the arm registers an empty module under that name
before importing (`arms.py::load_da3`). `model/da3.py` also needs `addict`
and `omegaconf`. `app/`, `bench/`, `services/`, `cli.py` are vendored with the
package but unused.
