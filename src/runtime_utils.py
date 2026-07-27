"""Process/runtime environment helpers shared by entrypoints and tests.

Named `runtime_utils` rather than `utils` for the same reason as
`train_utils`: a top-level `utils` module shadows the vendored
`src/croco/utils` namespace package once `src/` is on the path, breaking
croco's own `import utils.misc`.
"""

from __future__ import annotations

import os


def n_cpus() -> int:
    """CPUs this process may actually use.

    NOT os.cpu_count(): that reports the machine's cores and ignores the
    cpuset a login node or batch job puts us in. The gap is not academic --
    on this cluster's login nodes cpu_count() reports 48 while the affinity
    mask grants 1, and libraries that size thread pools from the former spawn
    48 threads to fight over one core (onnxruntime additionally floods stderr
    with pthread_setaffinity_np failures as it tries to pin them)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def set_torch_threads(verbose: bool = True) -> int:
    """Size torch's thread pool from the affinity mask and report it.

    torch, like onnxruntime, defaults to the machine's core count; on a
    restricted or busy node that is pure contention. Returns the count so
    callers can pass it to other pools."""
    import torch

    n = n_cpus()
    torch.set_num_threads(n)
    if verbose:
        print(f"CPUs available to this process: {n}")
    return n
