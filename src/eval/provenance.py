"""Which code and which numeric precision produced a result, written into
every results file so results scored differently are never tabled together
(src/compare_table.py checks both)."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
_REPO_STR = str(REPO) + os.sep
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


# ---------------------------------------------------------------------------
# The code that ran, file by file.
#
# A results file's `commit` names a repository state; `code_files` names the
# code this run actually executed: a content hash of every repository source
# file whose module body ran in the process. Two rows are comparable when every
# file BOTH ran is byte-identical, so a commit that only touches training
# scripts, launchers or another model's code does not separate rows scored
# before it from rows scored after. compare_table checks this when every row
# carries it.
#
# Recorded through the interpreter's "exec" audit event, which fires when a
# module's code object runs -- on import, including the vendored baselines
# that eval.baselines.arms._vendored later drops from sys.modules, and modules
# executed by path (spec_from_file_location). The file is hashed at that
# moment, so the record is what ran; reading a file (git, linecache) does not
# fire it. Modules imported before this module was (an entry point should
# import it first) and compiled extensions come from sys.modules instead,
# hashed when the record is taken. Files run inside display_only() -- the
# cloud and GIF renderers, which only draw what was already scored -- and
# this module itself, which only records, are left out.
# ---------------------------------------------------------------------------

MODIFIED = "modified-during-run"
_SELF = os.path.realpath(__file__)
_EXEC: dict[str, str] = {}  # realpath -> sha256[:16] when its code ran
_DISPLAY: set[str] = set()
_display_depth = 0


def _hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _audit(event: str, args: tuple) -> None:
    if event != "exec" or not args:
        return
    filename = getattr(args[0], "co_filename", None)
    if not isinstance(filename, str) or not filename.endswith(".py"):
        return
    path = os.path.realpath(filename)
    if not path.startswith(_REPO_STR) or path in _EXEC:
        return
    try:
        _EXEC[path] = _hash(path)
    except OSError:
        return
    if _display_depth:
        _DISPLAY.add(path)


sys.addaudithook(_audit)


@contextlib.contextmanager
def display_only():
    """Code first run inside this block draws or exports what was already
    scored (cloud renders, GIFs, viewers) and is left out of loaded_code(): a
    change to a renderer must not separate two rows. Scoring code must never
    run in here."""
    global _display_depth
    _display_depth += 1
    try:
        yield
    finally:
        _display_depth -= 1


def loaded_code() -> dict[str, str]:
    """{repo-relative path: sha256[:16]} of every repository source file this
    process ran (see the block comment above). A file whose content on disk no
    longer matches what ran maps to MODIFIED."""
    out: dict[str, str] = {}
    for path, ran in list(_EXEC.items()):
        if path == _SELF or path in _DISPLAY:
            continue
        try:
            now = _hash(path)
        except OSError:
            now = None
        out[Path(path).relative_to(REPO).as_posix()] = ran if now == ran else MODIFIED
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not isinstance(f, str):
            continue
        path = os.path.realpath(f)
        if path.endswith(".pyc"):
            try:
                path = os.path.realpath(importlib.util.source_from_cache(path))
            except ValueError:
                continue
        if (
            not path.startswith(_REPO_STR)
            or path in _EXEC
            or path == _SELF
            or not os.path.isfile(path)
        ):
            continue
        out[Path(path).relative_to(REPO).as_posix()] = _hash(path)
    return dict(sorted(out.items()))


def code_differences(
    named: dict[str, dict[str, str] | None],
) -> list[str]:
    """Problems between runs' code_files: every file two or more runs ran must
    be identical in all of them, and none may be MODIFIED. A file only one run
    ran (its own model or wrapper) is not compared."""
    problems = []
    paths = sorted({p for files in named.values() if files for p in files})
    for path in paths:
        seen = {
            label: files[path]
            for label, files in named.items()
            if files and path in files
        }
        if MODIFIED in seen.values():
            problems.append(
                f"{path} changed during {sorted(k for k, v in seen.items() if v == MODIFIED)}"
            )
        elif len(seen) > 1 and len(set(seen.values())) > 1:
            problems.append(f"{path} differs: {seen}")
    return problems


def to_primitive(obj):
    """Recursively strip a config to builtin types (enum -> value, dataclass /
    mapping / sequence -> dict / list) so a snapshot of it pickles or
    serializes without needing any project module to reconstruct it. Lives
    here, not in train_utils, so a benchmark run need not import training
    helpers (train_utils re-exports it)."""
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_primitive(getattr(obj, f.name)) for f in dataclasses.fields(obj)
        }
    if isinstance(obj, dict):
        return {k: to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]
    return obj
