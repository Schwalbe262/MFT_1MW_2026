"""Build the sealed SHA-b171 recovery4 terminal gate from read-only evidence.

This is deliberately a one-shot evidence collector.  It has no scheduler
mutation path, requires all four exact tasks to be completed before reading
any result stdout, and refuses to replace an existing gate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for item in (HERE, REGRESSION_ROOT, VERIFY_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import _submit_production300_b171c7c as production
import rapid_campaign
import scheduler_client


SOLVER = production.SOLVER
LIBRARY = production.LIBRARY
RECOVERY_IDS = production.RECOVERY_IDS
GATE_PATH = production.GATE_PATH
THERMAL_SOURCE_PATH = "module/thermal_260706.py"
THERMAL_SOLVE_FUNCTION = "_solve_exact_thermal_setup"
CONVERGED_REASONS = frozenset(("converged", "residual_threshold"))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_bundle():
    """Use the production submitter's exact sealed-artifact validators."""
    plan, submission = production._load_recovery()
    plan_tasks = plan.get("tasks")
    submitted = submission.get("tasks")
    if not isinstance(plan_tasks, list) or not isinstance(submitted, list) \
            or len(plan_tasks) != 4 or len(submitted) != 4:
        raise RuntimeError("sealed recovery4 plan/submission is incomplete")
    if tuple(row.get("task_id") for row in submitted) != RECOVERY_IDS:
        raise RuntimeError("sealed recovery4 task IDs drifted")
    return plan, submission


def _expected_metadata(row, expected):
    resources = production.EXPECTED_RESOURCES
    task_id = int(expected["task_id"])
    checks = {
        "id": row.get("id", row.get("task_id")) == task_id,
        "name": row.get("name") == expected.get("name"),
        "dedupe": row.get("dedupe_key") == expected.get("dedupe_key"),
        "project": row.get("project") == resources["project"],
        "cpus": row.get("cpus") == resources["cpus"],
        "memory_mb": row.get("memory_mb") == resources["memory_mb"],
        "gpus": row.get("gpus") == resources["gpus"],
        "timeout_seconds": row.get("timeout_seconds")
            == resources["timeout_seconds"],
        "required_capability": row.get("required_capability")
            == resources["required_capability"],
        "env_profile": row.get("env_profile") == resources["env_profile"],
        "scheduling_profile": row.get("scheduling_profile")
            == resources["scheduling_profile"],
        "remote_cwd": row.get("remote_cwd") == resources["remote_cwd"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"recovery4 task {task_id} scheduler metadata drifted: {checks}"
        )
    status = str(row.get("status") or "").strip().lower()
    return {
        "id": task_id,
        "name": row["name"],
        "dedupe_key": row["dedupe_key"],
        "project": row["project"],
        "status": status,
        "cpus": row["cpus"],
        "memory_mb": row["memory_mb"],
        "gpus": row["gpus"],
        "timeout_seconds": row["timeout_seconds"],
        "required_capability": row["required_capability"],
        "env_profile": row["env_profile"],
        "scheduling_profile": row["scheduling_profile"],
        "remote_cwd": row["remote_cwd"],
        # Priority is scheduling-only evidence.  The exact pilot was
        # intentionally raised after submission, so it is not an identity gate.
        "observed_priority": row.get("priority"),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
    }


def _collect_completed_metadata(submission):
    evidence = []
    for expected in submission["tasks"]:
        task_id = int(expected["task_id"])
        evidence.append(_expected_metadata(
            production._task_detail(task_id), expected,
        ))
    statuses = {row["id"]: row["status"] for row in evidence}
    if any(status != "completed" for status in statuses.values()):
        raise RuntimeError(
            "recovery4 terminal gate requires all exact tasks completed: "
            + repr(statuses)
        )
    return evidence


def _git_thermal_source():
    spec = f"{SOLVER}:{THERMAL_SOURCE_PATH}"
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", spec],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"cannot read exact solver thermal source {spec}: {detail}"
        ) from exc
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeError("exact solver thermal source is not UTF-8") from exc


def _normalized_call_name(call):
    target = call.func
    if isinstance(target, ast.Attribute):
        name = target.attr
    elif isinstance(target, ast.Name):
        name = target.id
    else:
        return ""
    return str(name).replace("_", "").casefold()


def _static_dispatch_proof(source=None):
    """Prove the exact SHA's thermal function has one setup-scoped analyze call."""
    source = _git_thermal_source() if source is None else source
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        raise RuntimeError("exact solver thermal source cannot be parsed") from exc

    setup_constant = None
    target = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for lhs in node.targets:
                if isinstance(lhs, ast.Name) and lhs.id == "_THERMAL_SETUP_NAME" \
                        and isinstance(node.value, ast.Constant):
                    setup_constant = node.value.value
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == THERMAL_SOLVE_FUNCTION:
            if target is not None:
                raise RuntimeError("exact thermal solve function is duplicated")
            target = node
    if setup_constant != "ThermalSetup" or target is None:
        raise RuntimeError("exact ThermalSetup function/constant proof failed")

    args = list(target.args.args)
    defaults = [None] * (len(args) - len(target.args.defaults)) \
        + list(target.args.defaults)
    setup_default = next(
        (default for arg, default in zip(args, defaults)
         if arg.arg == "setup_name"), None,
    )
    if not isinstance(setup_default, ast.Name) \
            or setup_default.id != "_THERMAL_SETUP_NAME":
        raise RuntimeError("thermal solve setup_name default drifted")

    all_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = [call for call in all_calls
                 if _normalized_call_name(call) == "analyzeall"]
    target_analyze = [
        call for call in ast.walk(target)
        if isinstance(call, ast.Call) and _normalized_call_name(call) == "analyze"
    ]
    if forbidden or len(target_analyze) != 1:
        raise RuntimeError(
            "exact solver dispatch is not one analyze(setup=...) with zero AnalyzeAll calls"
        )
    call = target_analyze[0]
    keywords = {item.arg: item.value for item in call.keywords if item.arg}
    setup_arg = keywords.get("setup")
    blocking_arg = keywords.get("blocking")
    if not isinstance(setup_arg, ast.Name) or setup_arg.id != "setup_name" \
            or not isinstance(blocking_arg, ast.Constant) \
            or blocking_arg.value is not True:
        raise RuntimeError("exact solver analyze keyword dispatch drifted")

    function_source = ast.get_source_segment(source, target)
    if not function_source:
        raise RuntimeError("cannot isolate exact thermal solve function")
    return {
        "solver_revision": SOLVER,
        "source_path": THERMAL_SOURCE_PATH,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "function": THERMAL_SOLVE_FUNCTION,
        "function_sha256": hashlib.sha256(
            function_source.encode("utf-8")
        ).hexdigest(),
        "entrypoint": "ThermalSetup",
        "native_analyze_call_count": 1,
        "analyze_all_call_count": 0,
        "proof_kind": "exact-git-blob-python-ast",
    }


def _exact_positive_int(value, label):
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not an integer")
    try:
        numeric = int(value)
        if float(value) != float(numeric) or numeric < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} is not a positive integer") from exc
    return numeric


def _strict_result(fetched, planned, task_id, expected_profile):
    if fetched.state != scheduler_client.RESULT_VALID \
            or not isinstance(fetched.result, dict):
        raise RuntimeError(
            f"recovery4 task {task_id} result is not scheduler-client strict valid: "
            f"{fetched.state}"
        )
    result = fetched.result
    if not scheduler_client.is_valid_result(
            result, expected_revision=SOLVER,
            expected_library_revision=LIBRARY,
            expected_profile=expected_profile):
        raise RuntimeError(f"recovery4 task {task_id} failed strict result recheck")
    if not scheduler_client.result_matches_params(
            result, planned.get("effective_params")):
        raise RuntimeError(f"recovery4 task {task_id} effective params drifted")
    saturated = rapid_campaign.thermal_saturation_columns(result)
    if saturated:
        raise RuntimeError(
            f"recovery4 task {task_id} has thermal saturation: {saturated}"
        )
    return result


def _analyze_all_evidence(result, static_proof, task_id):
    result_keys = (
        "thermal_analyze_all_call_count",
        "analyze_all_call_count",
    )
    present = {key: result[key] for key in result_keys if key in result}
    for key, value in present.items():
        if isinstance(value, bool):
            raise RuntimeError(f"recovery4 task {task_id} {key} is not integer zero")
        try:
            if int(value) != 0 or float(value) != 0.0:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"recovery4 task {task_id} {key} is not integer zero"
            ) from exc
    if static_proof.get("analyze_all_call_count") != 0:
        raise RuntimeError("exact solver AnalyzeAll static proof is absent")
    return {
        "source": "result+exact-solver-static-proof" if present
            else "exact-solver-static-proof",
        "result_fields": present,
        "static_source_sha256": static_proof["source_sha256"],
        "static_function_sha256": static_proof["function_sha256"],
    }


def _failed_source_dispatch(result, static_proof, task_id):
    raw = result.get("thermal_dispatch_forensic_json")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            f"recovery4 task {task_id} thermal dispatch forensic JSON is absent"
        )
    try:
        forensic = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"recovery4 task {task_id} thermal dispatch forensic JSON is malformed"
        ) from exc
    if not isinstance(forensic, dict) \
            or forensic.get("schema") != "thermal-dispatch-forensic-v1":
        raise RuntimeError(f"recovery4 task {task_id} forensic schema drifted")
    attempts = forensic.get("attempts")
    final = forensic.get("final_convergence")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2 \
            or not isinstance(final, dict):
        raise RuntimeError(f"recovery4 task {task_id} forensic attempts drifted")

    for index, attempt in enumerate(attempts, start=1):
        identity = attempt.get("identity") if isinstance(attempt, dict) else None
        if not isinstance(attempt, dict) or attempt.get("attempt") != index \
                or not isinstance(identity, dict) \
                or identity.get("design") != "icepak_thermal" \
                or identity.get("design_type") != "Icepak" \
                or identity.get("setups") != ["ThermalSetup"] \
                or identity.get("wrapper_setups") != ["ThermalSetup"]:
            raise RuntimeError(
                f"recovery4 task {task_id} attempt {index} setup identity drifted"
            )
        if index < len(attempts) and (
                attempt.get("dispatch_status") not in ("false", "exception")
                or attempt.get("monitor_reason") != "monitor_missing"
                or attempt.get("native_running") is not False):
            raise RuntimeError(
                f"recovery4 task {task_id} retry was not evidence-gated"
            )

    attempt_count = _exact_positive_int(
        result.get("thermal_solve_attempts"),
        f"recovery4 task {task_id} thermal_solve_attempts",
    )
    if attempt_count != len(attempts):
        raise RuntimeError(f"recovery4 task {task_id} attempt count drifted")
    last = attempts[-1]
    monitor_file = str(result.get("thermal_monitor_file") or "").strip()
    reason = str(final.get("reason") or "").strip()
    checks = {
        "dispatch_success": last.get("dispatch_status") == "success",
        "result_dispatch_success": result.get("thermal_dispatch_status") == "success",
        "analyze_call_ok": result.get("thermal_analyze_call_ok") == 1,
        "solution_available": result.get("thermal_solution_data_available") == 1,
        "fresh_monitor_nonempty": bool(monitor_file),
        "attempt_monitor": last.get("monitor_file") == monitor_file,
        "final_monitor": final.get("monitor_file") == monitor_file,
        "converged": final.get("converged") == 1,
        "convergence_available": final.get("available") == 1,
        "convergence_reason": reason in CONVERGED_REASONS,
        "attempt_reason": last.get("monitor_reason") == reason,
        "result_reason": result.get("thermal_convergence_reason") == reason,
        "retry_max_one": len(attempts) - 1 <= 1,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"recovery4 task {task_id} final thermal dispatch drifted: {checks}"
        )
    analyze_all = _analyze_all_evidence(result, static_proof, task_id)
    return {
        "entrypoint": "ThermalSetup",
        "analyze_all_call_count": 0,
        "fresh_monitor": True,
        "startup_retry_count": len(attempts) - 1,
        "forensic_schema": forensic["schema"],
        "attempt_count": len(attempts),
        "monitor_file": monitor_file,
        "convergence_reason": reason,
        "analyze_all_evidence": analyze_all,
    }


def _atomic_create_json(path, payload):
    """Atomically publish a complete JSON file without replacement semantics."""
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"terminal gate already exists: {path}")
    if not path.parent.is_dir():
        raise RuntimeError(f"terminal gate parent directory is absent: {path.parent}")
    data = (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n").encode("utf-8")
    fd, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard-link publication is atomic and fails if the final name
            # appeared after our initial check.  It never replaces evidence.
            os.link(staged, path)
        except FileExistsError as exc:
            raise RuntimeError(f"terminal gate already exists: {path}") from exc
        except OSError as link_exc:
            # Mapped Windows/SMB filesystems commonly reject hard links.  On
            # Windows, rename is still an atomic no-replace operation; unlike
            # POSIX rename it fails when the destination already exists.
            if os.name != "nt":
                raise RuntimeError(
                    f"terminal gate filesystem lacks atomic create: {path.parent}"
                ) from link_exc
            try:
                os.rename(staged, path)
            except OSError as rename_exc:
                if path.exists():
                    raise RuntimeError(
                        f"terminal gate already exists: {path}"
                    ) from rename_exc
                raise RuntimeError(
                    f"terminal gate atomic publication failed: {path}"
                ) from rename_exc
        if path.read_bytes() != data:
            raise RuntimeError("terminal gate atomic readback mismatch")
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.remove(staged)
        except FileNotFoundError:
            pass


def build_gate(output_path=None):
    output = Path(GATE_PATH if output_path is None else output_path)
    if output.exists():
        raise RuntimeError(f"terminal gate already exists: {output}")

    plan, submission = _load_bundle()
    metadata = _collect_completed_metadata(submission)
    static_proof = _static_dispatch_proof()
    expected_profile = plan.get("profile", {}).get("param_overrides")
    if not isinstance(expected_profile, dict):
        raise RuntimeError("sealed recovery4 expected profile is absent")

    rows = []
    for index, (planned, expected, live) in enumerate(zip(
            plan["tasks"], submission["tasks"], metadata)):
        task_id = int(expected["task_id"])
        fetched = scheduler_client.fetch_result(
            task_id, attempts=1, retry_delay=0,
            expected_revision=SOLVER,
            expected_library_revision=LIBRARY,
            expected_profile=expected_profile,
        )
        result = _strict_result(
            fetched, planned, task_id, expected_profile,
        )
        row = {
            "ordinal": expected["ordinal"],
            "task_id": task_id,
            "source_task_id": expected["source_task_id"],
            "name": expected["name"],
            "dedupe_key": expected["dedupe_key"],
            "status": live["status"],
            "result_state": scheduler_client.RESULT_VALID,
            "strict_valid": True,
            "result_sha256": production._sha(result),
            "scheduler_metadata": live,
            "scheduler_metadata_sha256": production._sha(live),
            "saturation_columns": [],
            "effective_params_match": True,
        }
        if index < 3:
            row["thermal_dispatch"] = _failed_source_dispatch(
                result, static_proof, task_id,
            )
        else:
            row["known_good_nonregression"] = True
        rows.append(row)

    gate = {
        "schema": production.GATE_SCHEMA,
        "created_at": _now(),
        "gate_decision": "pass",
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "recovery_plan_sha256": production.RECOVERY_PLAN_SHA256,
        "recovery_submission_sha256": production.RECOVERY_SUBMISSION_SHA256,
        "task_count": 4,
        "strict_valid_count": 4,
        "all_strict_valid": True,
        "partial_pass_allowed": False,
        "scheduler_query_count": {
            "task_metadata_get": 4,
            "result_stdout_get": 4,
        },
        "scheduler_mutation_count": 0,
        "solver_static_dispatch_proof": static_proof,
        "tasks": rows,
    }
    gate["gate_sha256"] = production._sha(gate)
    production._validate_gate(gate, gate["gate_sha256"], submission)
    _atomic_create_json(output, gate)
    readback = production._read_json(output, "terminal recovery gate readback")
    production._validate_gate(
        readback, readback.get("gate_sha256"), submission,
    )
    if readback != gate:
        raise RuntimeError("terminal recovery gate semantic readback mismatch")
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    gate = build_gate()
    print(json.dumps({
        "gate": str(GATE_PATH.resolve()),
        "gate_sha256": gate["gate_sha256"],
        "task_ids": [row["task_id"] for row in gate["tasks"]],
        "scheduler_mutation_count": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
