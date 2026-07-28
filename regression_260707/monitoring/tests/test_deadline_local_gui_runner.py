import hashlib
from pathlib import Path

import pytest

from regression_260707.monitoring import (
    deadline_design,
    deadline_local_gui_runner as runner,
)


SOLVER_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/"
    "mft_deadline_timk3_solver_260723/worktree"
)
LIBRARY_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/"
    "mft_deadline_timk3_library_260723/worktree"
)


def test_gui_wrapper_hash_is_pinned_by_publication_adapter():
    digest = hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()

    assert digest == deadline_design.DEADLINE_LOCAL_GUI_RUNNER_SHA256
    assert (
        deadline_design.EXPECTED_LOCAL_GUI_SOLVER["gui_runner_sha256"]
        == digest
    )


def test_exact_solver_and_library_sources_are_clean_and_authenticated():
    try:
        exact_runner = runner._verify_sources(SOLVER_ROOT, LIBRARY_ROOT)
    except runner.DeadlineLocalGuiRunnerError as exc:
        pytest.skip(f"retired historical deadline sources are unavailable: {exc}")

    assert exact_runner == (
        SOLVER_ROOT / "run_simulation_260706.py"
    ).resolve()
    assert runner._sha256(exact_runner) == runner.EXACT_SOLVER_RUNNER_SHA256
    assert runner._sha256(
        SOLVER_ROOT / "module" / "thermal_260706.py"
    ) == runner.EXACT_THERMAL_MODULE_SHA256


def test_source_verification_fails_closed_when_sources_are_missing(tmp_path):
    with pytest.raises(
        runner.DeadlineLocalGuiRunnerError,
        match="source is unavailable",
    ):
        runner._verify_sources(
            (tmp_path / "missing-solver").resolve(),
            (tmp_path / "missing-library").resolve(),
        )


def test_wrapper_requires_absolute_result_path_and_hold(tmp_path):
    result = (tmp_path / "result.json").resolve()

    forwarded, parsed = runner._extract_result_argument([
        "--fixed",
        "--hold",
        "--result-json",
        str(result),
    ])

    assert forwarded == ["--fixed", "--hold"]
    assert parsed == result
    with pytest.raises(
        runner.DeadlineLocalGuiRunnerError, match="requires --hold"
    ):
        runner._extract_result_argument([
            "--fixed", "--result-json", str(result)
        ])
