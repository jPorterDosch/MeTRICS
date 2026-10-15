"""Which code and which numeric precision produced a result, written into
every results file so results scored differently are never tabled together
(src/compare_table.py checks both)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CODE_PATHS = ("src", "datasets_preprocess", "experiments", "third_party")
RECORD = "src/eval/baselines/reproduction.json"


def code_fingerprint() -> str:
    """HEAD, plus a hash of the uncommitted CODE changes -- tracked diffs and
    untracked files under src/, datasets_preprocess/, experiments/ and
    third_party/, the reproduction record excluded since every run rewrites
    it: two runs with the same string ran the same code. Call it when a run
    STARTS: the code imported then is the code that runs."""
    git = ["git", "-C", str(REPO)]
    try:
        head = subprocess.check_output(
            [*git, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            [*git, "diff", "HEAD", "--", *CODE_PATHS, f":!{RECORD}"], text=True
        )
        untracked = [
            p
            for p in subprocess.check_output(
                [
                    *git,
                    "ls-files",
                    "-z",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *CODE_PATHS,
                ],
                text=True,
            ).split("\0")
            if p
        ]
        h = hashlib.sha1(diff.encode())
        for rel in sorted(untracked):
            h.update(rel.encode())
            h.update((REPO / rel).read_bytes())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "unknown"
    if not diff.strip() and not untracked:
        return head
    return f"{head}+{h.hexdigest()[:8]}"


def device_info() -> dict:
    """The GPU, and the autocast dtype dust3r.inference.loss_of_one_batch
    picks on it for OUR model (bf16 on compute capability >= 8, else fp16):
    the same checkpoint scores differently on a V100 than on an A40/H100."""
    if not torch.cuda.is_available():
        return {"gpu": "cpu", "autocast_dtype": None}
    major = torch.cuda.get_device_capability()[0]
    return {
        "gpu": torch.cuda.get_device_name(),
        "autocast_dtype": "bfloat16" if major >= 8 else "float16",
    }


def identifiable(commit) -> bool:
    """A fingerprint two runs can be matched on: present and not "unknown"
    (git failed) -- two "unknown" strings say nothing about the code."""
    return isinstance(commit, str) and commit not in ("", "unknown")


def manifest_hashes(root, specs) -> dict[str, str]:
    """sha256 (first 16 hex) of each benchmark manifest a run reads (main and
    TAE), keyed by its path relative to the bench root: two results scored on
    re-exported trees at the same path then differ here. `specs`: the
    datasets the run SCORES (a TAE dataset outside them is never read)."""
    out = {}
    for spec in specs:
        for rel in (spec.json, spec.tae_json):
            if rel and rel not in out:
                path = Path(root) / rel
                if path.is_file():
                    out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out
