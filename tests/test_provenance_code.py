"""eval.provenance.loaded_code / code_differences / display_only: the per-file
record of the code a benchmark run executed."""

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from eval import provenance as P


def _probe(tmp: Path, name: str, body: str = "X = 1\n") -> str:
    """Write a module under the repo, import it, return its repo path."""
    (tmp / f"{name}.py").write_text(body)
    sys.path.insert(0, str(tmp))
    try:
        importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp))
    return (tmp / f"{name}.py").resolve().relative_to(P.REPO).as_posix()


def test_loaded_code_lists_imported_repo_files() -> None:
    import eval.protocols  # noqa: F401

    files = P.loaded_code()
    assert "src/eval/protocols.py" in files
    assert len(files["src/eval/protocols.py"]) == 16
    assert "src/eval/provenance.py" not in files  # records, never scores
    assert not any(f.startswith(("/", "..")) for f in files)


def test_module_dropped_from_sys_modules_is_still_recorded() -> None:
    # what eval.baselines.arms._vendored does to the vendored baselines
    with tempfile.TemporaryDirectory(dir=P.REPO / "src") as tmp:
        rel = _probe(Path(tmp), "fp_probe_dropped")
        del sys.modules["fp_probe_dropped"]
        files = P.loaded_code()
    # the file is gone now, so it reads as changed -- but it was recorded
    assert rel in files


def test_file_edited_after_it_ran_is_marked() -> None:
    with tempfile.TemporaryDirectory(dir=P.REPO / "src") as tmp:
        rel = _probe(Path(tmp), "fp_probe_late")
        ran = P.loaded_code()[rel]
        assert ran != P.MODIFIED
        (Path(tmp) / "fp_probe_late.py").write_text("X = 2\n")
        assert P.loaded_code()[rel] == P.MODIFIED
        sys.modules.pop("fp_probe_late", None)


def test_reading_a_file_does_not_record_it() -> None:
    with tempfile.TemporaryDirectory(dir=P.REPO / "src") as tmp:
        path = Path(tmp) / "fp_probe_read.py"
        path.write_text("X = 1\n")
        path.read_bytes()
        rel = path.resolve().relative_to(P.REPO).as_posix()
        assert rel not in P.loaded_code()


def test_display_only_code_is_left_out() -> None:
    with tempfile.TemporaryDirectory(dir=P.REPO / "src") as tmp:
        with P.display_only():
            rel = _probe(Path(tmp), "fp_probe_render")
        assert rel not in P.loaded_code()
        sys.modules.pop("fp_probe_render", None)


def test_code_differences_compares_only_shared_files() -> None:
    a = {"x.py": "1", "only_a.py": "2"}
    b = {"x.py": "1", "only_b.py": "3"}
    assert P.code_differences({"a": a, "b": b}) == []
    assert P.code_differences({"a": a, "b": {**b, "x.py": "9"}})
    assert P.code_differences({"a": {**a, "x.py": P.MODIFIED}, "b": b})


def _fresh(code: str) -> list[str]:
    """sorted(loaded_code()) of a fresh interpreter after `code` (other tests
    in this process import training code)."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\nimport eval.provenance as P\n"
            + code
            + "\nprint(json.dumps(sorted(P.loaded_code())))",
        ],
        cwd=P.REPO / "src",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return json.loads(out.splitlines()[-1])


def test_benchmark_entry_point_runs_no_training_code() -> None:
    files = _fresh("import bench_checkpoint")
    assert "src/model_io.py" in files and "src/bench_eval.py" in files
    for training in (
        "src/finetune_depth.py",
        "src/visualize_depth.py",
        "src/train_utils.py",
    ):
        assert training not in files, training


def test_scoring_import_leaves_out_our_model() -> None:
    # a baseline row (bench_baselines -> bench_eval) must not carry our
    # model's code: config and sparse simulation only
    files = _fresh("import bench_eval")
    assert "src/streamvggt/depth_cond/sparse.py" in files
    assert "src/streamvggt/depth_cond/model.py" not in files
    assert "src/streamvggt/depth_cond/conditioner.py" not in files
