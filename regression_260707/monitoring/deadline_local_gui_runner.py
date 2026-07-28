"""Retained local-GUI harness for the immutable deadline TIM-k3 solver.

The physics source is never copied or edited.  This small executable imports
the exact, clean solver checkout after verifying its Git identity and adds
only the interactive GUI lifecycle/result-artifact behavior needed by 8010.
The solver therefore continues to report the reviewed physics revision rather
than a synthetic revision created merely to add a GUI wrapper.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


EXACT_SOLVER_REVISION = "8a8d90f68e8728669282f586f24304c7cc807029"
EXACT_SOLVER_BRANCH = "deadline-timk3-k3-20260723"
EXACT_SOLVER_RUNNER_SHA256 = (
    "c3e2b8a2dce2dbf6f87e93723ffabd864702c579db6d2fe9b2f78494b227a14c"
)
EXACT_THERMAL_MODULE_SHA256 = (
    "58b217e35ad1a6d4c1ffa6aafd88462872ee5c0da600c287c37b9a08d2bd3224"
)
EXACT_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
GUI_RESULT_SCHEMA = "mft-local-aedt-gui-result-v1"
SOLVER_ROOT_ENV = "MFT_DEADLINE_EXACT_SOLVER_ROOT"
LIBRARY_ROOT_ENV = "MFT_PYAEDT_LIBRARY_ROOT"


class DeadlineLocalGuiRunnerError(RuntimeError):
    """The immutable source identity or retained-GUI contract failed."""


class _HoldRunFailure(Exception):
    """Prevent the exact runner's normal retry loop from leaking a held GUI."""


class _RunState:
    def __init__(self) -> None:
        self.sim: Any = None
        self.result: Any = None
        self.outcome: bool | None = None
        self.error: BaseException | None = None
        self.aedt_process: dict[str, Any] | None = None
        self.detach_confirmed: bool | None = None
        self.detach_error: str | None = None
        self.inspection_reports: list[dict[str, Any]] = []
        self.reports_attempted = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return completed.stdout.strip()


def _verify_sources(solver_root: Path, library_root: Path) -> Path:
    if not solver_root.is_absolute() or not library_root.is_absolute():
        raise DeadlineLocalGuiRunnerError(
            "deadline solver and library roots must be absolute"
        )
    try:
        solver_root = solver_root.resolve(strict=True)
        library_root = library_root.resolve(strict=True)
        runner = (solver_root / "run_simulation_260706.py").resolve(strict=True)
        thermal_module = (
            solver_root / "module" / "thermal_260706.py"
        ).resolve(strict=True)
    except OSError as exc:
        raise DeadlineLocalGuiRunnerError(
            "deadline solver or library source is unavailable"
        ) from exc
    if (
        not runner.is_file()
        or not thermal_module.is_file()
        or _git(solver_root, "rev-parse", "HEAD").lower()
        != EXACT_SOLVER_REVISION
        or _git(solver_root, "branch", "--show-current")
        != EXACT_SOLVER_BRANCH
        or _git(
            solver_root, "status", "--porcelain", "--untracked-files=no"
        )
        or _sha256(runner) != EXACT_SOLVER_RUNNER_SHA256
        or _sha256(thermal_module) != EXACT_THERMAL_MODULE_SHA256
    ):
        raise DeadlineLocalGuiRunnerError(
            "deadline TIM-k3 solver revision/branch/hash/cleanliness mismatch"
        )
    if (
        _git(library_root, "rev-parse", "HEAD").lower()
        != EXACT_LIBRARY_REVISION
        or _git(
            library_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "src",
        )
    ):
        raise DeadlineLocalGuiRunnerError(
            "deadline PyAEDT library revision/cleanliness mismatch"
        )
    return runner


def _extract_result_argument(arguments: list[str]) -> tuple[list[str], Path]:
    forwarded: list[str] = []
    result_path: Path | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--result-json":
            if index + 1 >= len(arguments):
                raise DeadlineLocalGuiRunnerError(
                    "--result-json requires a path"
                )
            result_path = Path(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--result-json="):
            result_path = Path(argument.split("=", 1)[1])
            index += 1
            continue
        forwarded.append(argument)
        index += 1
    if result_path is None:
        raise DeadlineLocalGuiRunnerError("--result-json is mandatory")
    if "--hold" not in forwarded:
        raise DeadlineLocalGuiRunnerError("--result-json requires --hold")
    if not result_path.is_absolute():
        raise DeadlineLocalGuiRunnerError("--result-json must be absolute")
    return forwarded, result_path.resolve()


def _frame_row(frame: Any) -> dict[str, Any] | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    value = json.loads(frame.iloc[0].to_json())
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _aedt_identity(desktop: Any) -> dict[str, Any] | None:
    if desktop is None:
        return None
    candidate_pids: list[int] = []
    for raw in (
        getattr(desktop, "pid", None),
        getattr(desktop, "aedt_process_id", None),
    ):
        try:
            parsed = int(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed > 0:
            candidate_pids.append(parsed)
    native = getattr(desktop, "odesktop", None)
    getter = getattr(native, "GetProcessID", None)
    if callable(getter):
        try:
            parsed = int(getter())
            if parsed > 0:
                candidate_pids.append(parsed)
        except Exception:
            pass
    candidate_pids = sorted(set(candidate_pids))
    if len(candidate_pids) != 1:
        return None
    port = getattr(desktop, "port", None) or getattr(
        desktop, "grpc_port", None
    )
    try:
        port = int(port) if port is not None else None
    except (TypeError, ValueError, OverflowError):
        port = None
    try:
        import psutil

        process = psutil.Process(candidate_pids[0])
        return {
            "pid": candidate_pids[0],
            "create_time": float(process.create_time()),
            "name": str(process.name() or ""),
            "executable": str(process.exe() or ""),
            "grpc_port": port if port and 0 < port <= 65_535 else None,
            "alive": bool(process.is_running()),
            "error": None,
        }
    except Exception as exc:
        return {
            "pid": candidate_pids[0],
            "create_time": 0.0,
            "name": "",
            "executable": "",
            "grpc_port": port if port and 0 < port <= 65_535 else None,
            "alive": False,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _inspection_reports(sim: Any) -> list[dict[str, Any]]:
    specifications = []
    matrix = getattr(sim, "design_matrix", None)
    if matrix is not None:
        specifications.append({
            "design": matrix,
            "design_name": "maxwell_matrix",
            "plot_name": "MFT_FEA_Matrix_Results",
            "expressions": [
                "Matrix.L(Tx_winding,Tx_winding)",
                "Matrix.L(Rx_winding,Rx_winding)",
                "Matrix.L(Tx_winding,Rx_winding)",
                "abs(Matrix.CplCoef(Tx_winding,Rx_winding))",
            ],
            "category": "AC Magnetic",
            "context": "Matrix",
        })
    records: list[dict[str, Any]] = []
    for specification in specifications:
        public = {
            key: specification[key]
            for key in ("design_name", "plot_name", "expressions")
        }
        try:
            post = specification["design"].post
            if callable(post) and not hasattr(post, "create_report"):
                post = post()
            report = post.create_report(
                expressions=specification["expressions"],
                setup_sweep_name="Setup1 : LastAdaptive",
                variations={"Freq": ["All"]},
                primary_sweep_variable="Freq",
                report_category=specification["category"],
                plot_type="Data Table",
                context=specification["context"],
                plot_name=specification["plot_name"],
            )
            if report is None or report is False:
                raise RuntimeError("create_report returned no report")
            records.append({**public, "created": True, "error": None})
        except Exception as exc:
            records.append({
                **public,
                "created": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": (str(exc).strip() or repr(exc))[:2000],
                },
            })
    return records


def _install_hold_harness(solver: Any, state: _RunState) -> None:
    original_init = solver.Simulation.__init__
    original_save_results = solver.Simulation.save_results_to_csv
    original_save_project = solver.Simulation.save_project
    original_desktop = solver.pyDesktop
    original_run = solver.run_one_loop

    def capture_init(sim: Any, *args: Any, **kwargs: Any) -> None:
        original_init(sim, *args, **kwargs)
        state.sim = sim

    def capture_results(sim: Any, results: Any, *args: Any, **kwargs: Any) -> Any:
        state.result = results
        return original_save_results(sim, results, *args, **kwargs)

    def save_with_reports(sim: Any, *args: Any, **kwargs: Any) -> Any:
        if state.result is not None and not state.reports_attempted:
            state.reports_attempted = True
            state.inspection_reports = _inspection_reports(sim)
        return original_save_project(sim, *args, **kwargs)

    class HoldDesktop(original_desktop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["close_on_exit"] = False
            super().__init__(*args, **kwargs)

        def release_desktop(
            self, close_projects: bool = True, close_on_exit: bool = True
        ) -> Any:
            sim = state.sim
            if sim is None:
                return super().release_desktop(
                    close_projects=close_projects,
                    close_on_exit=close_on_exit,
                )
            if state.result is None:
                try:
                    original_save_project(sim, strict=True)
                except Exception:
                    pass
            state.aedt_process = _aedt_identity(self)
            try:
                released = super().release_desktop(
                    close_projects=False,
                    close_on_exit=False,
                )
            except Exception as exc:
                state.detach_confirmed = False
                state.detach_error = (
                    f"{type(exc).__name__}:{str(exc).strip() or repr(exc)}"
                )
                raise
            state.detach_confirmed = released is True
            state.detach_error = (
                None
                if state.detach_confirmed
                else f"release_desktop_returned:{released!r}"
            )
            return released

    def hold_run(*args: Any, **kwargs: Any) -> Any:
        try:
            state.outcome = original_run(*args, **kwargs)
            return state.outcome
        except BaseException as exc:
            state.error = exc
            raise _HoldRunFailure(str(exc)) from exc

    solver.Simulation.__init__ = capture_init
    solver.Simulation.save_results_to_csv = capture_results
    solver.Simulation.save_project = save_with_reports
    solver.pyDesktop = HoldDesktop
    solver.run_one_loop = hold_run
    solver._finalize_run_cleanup = lambda *args, **kwargs: None


def _publish(
    result_path: Path,
    solver: Any,
    state: _RunState,
    *,
    model_only: bool,
) -> None:
    sim = state.sim
    detach_ok = state.detach_confirmed is True
    if state.error is not None:
        status = "failed_held" if detach_ok else "failed"
    elif model_only and detach_ok:
        status = "model_ready"
    elif state.outcome is True and state.result is not None and detach_ok:
        status = "completed"
    else:
        status = "failed_held" if detach_ok else "failed"
    error = state.error
    if error is None and status not in {"completed", "model_ready"}:
        error = DeadlineLocalGuiRunnerError(
            state.detach_error or "solver returned an invalid result"
        )
    result = _frame_row(state.result)
    if result is not None and sim is not None:
        result.update(getattr(sim, "last_save_meta", {}) or {})
    project_path = getattr(sim, "project_path", None) if sim is not None else None
    payload = {
        "schema": GUI_RESULT_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_id": os.environ.get("MFT_GUI_LAUNCH_ID") or None,
        "candidate_id": os.environ.get("MFT_GUI_CANDIDATE_ID") or None,
        "held_open": bool(detach_ok),
        "detach_confirmed": state.detach_confirmed,
        "detach_error": state.detach_error,
        "solver_revision": solver.GIT_HASH,
        "solver_dirty": int(solver.GIT_DIRTY),
        "solver_branch": EXACT_SOLVER_BRANCH,
        "solver_source_runner_sha256": EXACT_SOLVER_RUNNER_SHA256,
        "solver_thermal_module_sha256": EXACT_THERMAL_MODULE_SHA256,
        "library_revision": solver.PYAEDT_LIBRARY_GIT_HASH,
        "library_dirty": int(solver.PYAEDT_LIBRARY_GIT_DIRTY),
        "project_name": (
            str(getattr(sim, "PROJECT_NAME", "") or "") or None
            if sim is not None else None
        ),
        "project_path": (
            str(Path(project_path).resolve()) if project_path else None
        ),
        "aedt_process": state.aedt_process,
        "parameters": (
            _frame_row(getattr(sim, "input_df", None))
            if sim is not None else None
        ),
        "result": result,
        "inspection_reports": state.inspection_reports,
        "error": (
            {
                "type": type(error).__name__,
                "message": (str(error).strip() or repr(error))[:8000],
            }
            if error is not None else None
        ),
    }
    _atomic_json(result_path, payload)


def main() -> int:
    forwarded, result_path = _extract_result_argument(sys.argv[1:])
    solver_root_text = os.environ.get(SOLVER_ROOT_ENV, "").strip()
    library_root_text = os.environ.get(LIBRARY_ROOT_ENV, "").strip()
    if not solver_root_text or not library_root_text:
        raise DeadlineLocalGuiRunnerError(
            f"{SOLVER_ROOT_ENV} and {LIBRARY_ROOT_ENV} are mandatory"
        )
    solver_root = Path(solver_root_text)
    library_root = Path(library_root_text)
    exact_runner = _verify_sources(solver_root, library_root)
    os.environ["MFT_AEDT_BACKEND"] = "standalone"
    os.environ[LIBRARY_ROOT_ENV] = str(library_root.resolve())
    sys.path.insert(0, str(solver_root.resolve()))
    specification = importlib.util.spec_from_file_location(
        "_mft_deadline_exact_timk3_solver", exact_runner
    )
    if specification is None or specification.loader is None:
        raise DeadlineLocalGuiRunnerError("cannot import exact solver source")
    solver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(solver)
    if (
        solver.GIT_HASH != EXACT_SOLVER_REVISION
        or int(solver.GIT_DIRTY) != 0
        or solver.PYAEDT_LIBRARY_GIT_HASH != EXACT_LIBRARY_REVISION
        or int(solver.PYAEDT_LIBRARY_GIT_DIRTY) != 0
    ):
        raise DeadlineLocalGuiRunnerError(
            "imported exact solver/library provenance drifted"
        )

    state = _RunState()
    _install_hold_harness(solver, state)
    model_only = "--model-only" in forwarded
    original_argv = sys.argv
    sys.argv = [str(exact_runner), *forwarded]
    try:
        try:
            solver.main()
        except _HoldRunFailure as exc:
            if state.error is None:
                state.error = exc
        except BaseException as exc:
            state.error = exc
    finally:
        sys.argv = original_argv
    _publish(result_path, solver, state, model_only=model_only)
    return 0 if state.error is None and (
        (model_only and state.detach_confirmed is True)
        or (
            state.outcome is True
            and state.result is not None
            and state.detach_confirmed is True
        )
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
