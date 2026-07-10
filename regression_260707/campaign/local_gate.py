"""Run and attest the exact-profile local three-consecutive simulation gate."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
sys.path.insert(0, str(REGRESSION_ROOT))
sys.path.insert(0, str(REGRESSION_ROOT / "verify"))
sys.path.insert(0, str(REPO_ROOT))

import al_driver
import pinned_pilot
import scheduler_client


_LOCAL_PROFILE_CONTRACT = dict(scheduler_client.STANDARD_PROFILE_CONTRACT)
_LOCAL_PROFILE_CONTRACT["keep_project"] = 1
EXACT_ARGS = [
    "--headless",
    "--thermal",
    "--count", "3",
    "--require-consecutive",
]
for _key, _value in _LOCAL_PROFILE_CONTRACT.items():
    EXACT_ARGS.extend(("--set", f"{_key}={_value}"))


def _capture_process_tree(root_pid, captured):
    try:
        import psutil
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
        for process in processes:
            try:
                captured[process.pid] = process.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass


def _terminate_captured_processes(captured, wait_seconds=10):
    """Terminate only the PID/create-time identities captured from this gate."""
    import psutil
    live = []
    for pid, create_time in reversed(list(captured.items())):
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - create_time) > 0.01:
                continue
            process.terminate()
            live.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, survivors = psutil.wait_procs(live, timeout=wait_seconds)
    for process in survivors:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if survivors:
        _, survivors = psutil.wait_procs(survivors, timeout=wait_seconds)
    remaining = []
    for pid, create_time in captured.items():
        try:
            process = psutil.Process(pid)
            if abs(process.create_time() - create_time) <= 0.01 and process.is_running():
                remaining.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if remaining:
        raise RuntimeError(f"local3 process-tree cleanup left live PIDs: {remaining}")


def _manifest_path(solver_revision, library_revision):
    tag = pinned_pilot.local_gate_tag(solver_revision, library_revision)
    return pinned_pilot.campaign_manifest_dir() / f"{tag}.json"


def _validate_result(result, index, solver_revision, library_revision):
    profile = pinned_pilot.local_gate_profile_contract()
    if not scheduler_client.is_valid_result(
            result, expected_revision=solver_revision,
            expected_library_revision=library_revision,
            expected_profile=profile):
        raise RuntimeError(f"local3 result {index} failed the strict result contract")
    if result.get("matrix_extraction_backend") != "export_rl_matrix":
        raise RuntimeError(f"local3 result {index} did not use export_rl_matrix")
    for label in ("matrix", "loss"):
        if int(float(result.get(f"{label}_solve_attempts", -1))) != 1:
            raise RuntimeError(f"local3 result {index} did not use one {label} solve")


def _validate_results(results, solver_revision, library_revision):
    if len(results) != pinned_pilot.LOCAL_GATE_COUNT:
        raise RuntimeError(
            f"local3 emitted {len(results)} RESULT_JSON rows, expected exactly three")
    for index, result in enumerate(results):
        _validate_result(result, index, solver_revision, library_revision)
    projects = [str(result["project_name"]) for result in results]
    if len(set(projects)) != len(projects):
        raise RuntimeError("local3 results contain duplicate project identities")


def _parse_result_line(line, result_index, solver_revision, library_revision):
    try:
        payload = json.loads(line[len("RESULT_JSON "):])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"local3 emitted malformed RESULT_JSON after {result_index} valid rows"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"local3 emitted non-object RESULT_JSON after {result_index} valid rows"
        )
    _validate_result(payload, result_index, solver_revision, library_revision)
    return payload


def run_gate(python_executable=sys.executable, force=False):
    solver_revision = al_driver._current_solver_revision()
    library_revision = al_driver._current_library_revision()
    manifest_path = _manifest_path(solver_revision, library_revision)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file() and not force:
        try:
            evidence = pinned_pilot.validate_local_gate(
                solver_revision, library_revision, manifest_dir=manifest_path.parent)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(json.dumps({**evidence, "reused": True}, ensure_ascii=False))
            return manifest
        except RuntimeError:
            pass
    timestamp = datetime.now().astimezone().strftime("%y%m%d_%H%M%S")
    log_path = REGRESSION_ROOT / "logs" / f"local3_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_executable), "-u", str(REPO_ROOT / "run_simulation_260706.py"),
        *EXACT_ARGS,
    ]
    results = []
    captured_processes = {}
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        popen_options = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            **popen_options,
        )
        _capture_process_tree(process.pid, captured_processes)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                _capture_process_tree(process.pid, captured_processes)
                print(line, end="", flush=True)
                log.write(line)
                if not line.startswith("RESULT_JSON "):
                    continue
                results.append(_parse_result_line(
                    line, len(results), solver_revision, library_revision))
            return_code = process.wait()
        finally:
            if process.poll() is None:
                _capture_process_tree(process.pid, captured_processes)
            _terminate_captured_processes(captured_processes)
    if return_code != 0:
        raise RuntimeError(
            f"local3 process failed with exit code {return_code}; inspect {log_path}")
    _validate_results(results, solver_revision, library_revision)
    tag = pinned_pilot.local_gate_tag(solver_revision, library_revision)
    manifest = {
        "tag": tag,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "sample_count": pinned_pilot.LOCAL_GATE_COUNT,
        "passed": True,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "log": str(log_path),
        "profile_contract": pinned_pilot.local_gate_profile_contract(),
        "results": results,
    }
    pinned_pilot._atomic_manifest(manifest, manifest_path)
    pinned_pilot.validate_local_gate(
        solver_revision, library_revision, manifest_dir=manifest_path.parent)
    print(json.dumps({
        "manifest": str(manifest_path),
        "log": str(log_path),
        "projects": [result["project_name"] for result in results],
    }, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_gate(args.python, force=args.force)


if __name__ == "__main__":
    main()
