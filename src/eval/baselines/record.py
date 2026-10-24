"""The reproduction record of the baseline arms, and the gate it implies.

reproduction.json lists, per arm, the numbers its own paper publishes
(transcribed with table and row), what our pipeline measured for them, and a
status. Two rules live here so they are enforced, not remembered:

  * An arm whose status is not `verified` may run only the (dataset,
    protocol) pairs of its own targets -- the paper's settings. Anything else
    (sparse_aligned, a density sweep, SPOT, another dataset) raises GateError.
  * `verified` is computed, never typed: every gating target measured and
    within max(relative * |published|, last_digit_units * the last printed
    digit). A gating target outside it makes the arm `failed`.

No torch here: the gate is importable (and tested) without a model.
"""

from __future__ import annotations

import json
from pathlib import Path

RECORD_PATH = Path(__file__).with_name("reproduction.json")

VERIFIED = "verified"
PENDING = "pending-gpu"
FAILED = "failed"


class GateError(RuntimeError):
    """An unverified baseline arm was asked for a run its paper does not report."""


def load(path: Path = RECORD_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def save(record: dict, path: Path = RECORD_PATH) -> None:
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def arm_entry(record: dict, arm: str) -> dict:
    if arm not in record["arms"]:
        raise KeyError(f"unknown baseline arm {arm!r}; known: {sorted(record['arms'])}")
    return record["arms"][arm]


def tolerance(published: str, tol: dict) -> float:
    """Allowed |measured - published|, in the printed unit. The last-digit
    unit comes from the string as printed ("0.570" -> 0.001, "5.1" -> 0.1)."""
    decimals = len(published.split(".")[1]) if "." in published else 0
    return max(
        tol["relative"] * abs(float(published)),
        tol["last_digit_units"] * 10.0**-decimals,
    )


def within(target: dict, measured: float, tol: dict) -> bool:
    """`measured` is in our unit; target['scale'] converts it to the printed one."""
    allowed = tolerance(target["published"], tol)
    # 1e-9: "exactly on the boundary" must not depend on float rounding
    return (
        abs(measured * target["scale"] - float(target["published"])) <= allowed + 1e-9
    )


def runnable_targets(entry: dict) -> list[dict]:
    return [t for t in entry["targets"] if t["dataset"] is not None]


def paper_uses(entry: dict) -> set[tuple[str, str]]:
    """The (dataset, protocol) pairs the arm's own paper reports."""
    return {(t["dataset"], t["protocol"]) for t in runnable_targets(entry)}


def _identifiable(commit) -> bool:
    """A code string two runs can be matched on: present, known, and not the
    legacy '+dirty' marker that named no code state."""
    return (
        isinstance(commit, str)
        and commit not in ("", "unknown")
        and not commit.endswith("+dirty")
    )


def _passes(t: dict, tol: dict):
    """True / False from the stored measurement, recomputed against the
    published number and tolerance (never the stored flag, which a hand edit
    of `published`, `scale` or the tolerance would leave stale); None if
    unmeasured."""
    if t.get("measured") is None:
        return None
    return within(t, t["measured"], tol)


def computed_status(entry: dict, tol: dict) -> str:
    """The status the stored measurements support: `verified` needs every
    gating target measured, within tolerance, and by ONE identifiable code
    state; any gating miss is `failed`. Pass/fail is recomputed from
    `measured` against the record's tolerance `tol`, never read from the
    stored `within_tolerance` flags."""
    gating = [t for t in entry["targets"] if t["gate"]]
    commits = {t.get("measured_commit") for t in gating}
    same_code = len(commits) == 1 and _identifiable(next(iter(commits), None))
    passes = [_passes(t, tol) for t in gating]
    if any(p is False for p in passes):
        return FAILED
    if gating and all(passes) and same_code:
        return VERIFIED
    return PENDING


def recorded_close(entry: dict) -> bool:
    """The user's decision, written into the record as `"close": true`, that
    an arm which does not reach parity is close enough to compare, with the
    gap reported (bench_baselines benchmark --allow-close)."""
    return bool(entry.get("close"))


def close_allowed(entry: dict, tol: dict) -> bool:
    """--allow-close opens the gate only for an arm the record marks close
    AND whose every gating target is measured, by ONE identifiable code
    state, with a computed `failed`: the gap it is compared with is a
    measured one that a code state can be tied to, as `verified` requires."""
    gating = [t for t in entry["targets"] if t["gate"]]
    commits = {t.get("measured_commit") for t in gating}
    one_code = len(commits) == 1 and _identifiable(next(iter(commits), None))
    return (
        recorded_close(entry)
        and one_code
        and all(t.get("measured") is not None for t in gating)
        and computed_status(entry, tol) == FAILED
    )


def require_allowed(
    record: dict,
    arm: str,
    uses: list[tuple[str, str]] | set[tuple[str, str]],
    allow_close: bool = False,
) -> None:
    """Raise GateError unless every (dataset, protocol) in `uses` is open to
    `arm`: all of them once it is verified, only its paper's before. With
    `allow_close`, all of them for an arm the record marks close whose
    measured status is `failed` (close_allowed), and nothing else."""
    entry = arm_entry(record, arm)
    if allow_close:
        if close_allowed(entry, record["tolerance"]):
            return
        raise GateError(
            f"baseline arm {arm!r} is "
            f"{computed_status(entry, record['tolerance'])!r}: --allow-close needs "
            'an arm the record marks "close": true with every gating target '
            "measured and the status computed failed"
        )
    # recomputed, never read from the stored string or flags: a hand-edited or
    # stale `verified` / `within_tolerance` does not open the gate
    status = computed_status(entry, record["tolerance"])
    if status == VERIFIED:
        return
    blocked = sorted(set(uses) - paper_uses(entry))
    if blocked:
        raise GateError(
            f"baseline arm {arm!r} is {status!r}, not verified: it may run "
            f"only its paper's settings {sorted(paper_uses(entry))}, not {blocked}. "
            "Reproduce it first (src/bench_baselines.py reproduce)."
        )


def apply_measurements(
    record: dict, arm: str, measured: dict[str, float], run: dict
) -> str:
    """Store `measured` (target id -> value in our unit) and `run` (commit,
    job id, date) on the arm, recompute its status and return it. A target
    absent from `measured` keeps its previous value; the status is `verified`
    only when every gating target has one, all are within tolerance, AND all
    were measured by the same code (`run["commit"]`, which bench_baselines
    hashes together with any uncommitted code change): a partial rerun after
    a protocol change cannot verify an arm on stale numbers."""
    entry = arm_entry(record, arm)
    known = {t["id"] for t in entry["targets"]}
    unknown = sorted(set(measured) - known)
    if unknown:
        raise KeyError(f"{arm}: measurements for unknown targets {unknown}")
    tol = record["tolerance"]
    for t in entry["targets"]:
        if t["id"] in measured:
            t["measured"] = float(measured[t["id"]])
            t["within_tolerance"] = within(t, t["measured"], tol)
            t["measured_commit"] = run.get("commit")
    entry["status"] = computed_status(entry, tol)
    entry["runs"].append(run)
    return entry["status"]
