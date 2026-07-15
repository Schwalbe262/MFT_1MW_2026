"""Reviewed scheduler-client execution for pipeline verification manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


ADAPTER_NAME = "mft_scheduler_v1"
TERMINAL = {"completed", "failed", "cancelled"}
BASE_CONFIG_FIELDS = {
    "adapter", "execute", "library_root", "poll_seconds", "timeout_seconds"
}
EXPERIMENTAL_CONFIG_FIELDS = {
    "solver_root",
    "priority",
    "required_hard_cap",
    "max_project_active_tasks",
    "aedt_backend",
    "submission_env",
    "accounts",
    "q7_guard_task_ids",
    "q7_guard_timeout_seconds",
    "experimental_quality_status_sha256",
}
EXPERIMENTAL_SOLVER_REVISION = "267860a86dc8c8017c4b713f6674c0614cc365ce"
EXPERIMENTAL_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
Q7_GUARD_TASK_IDS = (41692, 41693, 41694)
Q7_GUARD_ACCOUNTS = ("r1jae262", "dhj02", "r1jae262")
Q7_NAME_PREFIX = "mft-1to3-q7-full-267860a-"
Q7_SUBMISSION_ENV = {
    "MFT_AEDT_BACKEND": "pooled",
    "MFT_AEDT_ISOLATION_POLICY": "family",
    "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
    "MFT_AEDT_POOL_WORKSPACE": (
        "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
    ),
    "MFT_AEDT_SCHEDULER_URL": "http://172.16.10.37:18790",
    "MFT_AEDT_SESSION_PROFILE": (
        '{"aedt_version":"2025.2","desktop_dso":{"config_name":'
        '"pyaedt_config","designs":{"Icepak":{"cores":4,"gpus":0,'
        '"tasks":1,"use_auto_settings":false},"Maxwell 2D":{"cores":4,'
        '"gpus":0,"tasks":1,"use_auto_settings":true},"Maxwell 3D":'
        '{"cores":4,"gpus":0,"tasks":1,"use_auto_settings":true}}},'
        '"filesystem":"gpfs-shared-v1","profile_version":2,'
        '"pyaedt_version":"0.22.0","python_environment":"pyaedt2026v1"}'
    ),
    "MFT_AEDT_SESSION_VERSION": "2025.2",
    "MFT_AEDT_SHARED_CANARY": "1",
    "MFT_AEDT_WORKSPACE_PATH": (
        "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
    ),
    "MFT_SLURM_SCHEDULER_ROOT": "$HOME/slurm_scheduler/aedt_pool_pkg",
    "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE": (
        "$HOME/slurm_scheduler/aedt_pool_client"
    ),
}


def _full_sha(value) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _experimental_execution_contract(request, config):
    """Return sealed non-production execution options, or ``None``.

    These options deliberately cannot be supplied to the production adapter.
    The only accepted experimental request is a three-candidate fine/full FEA
    feedback wave tied to the failed provisional quality evidence.
    """

    requested = request.get("experimental_active_learning") is True
    configured = bool(set(config).intersection(EXPERIMENTAL_CONFIG_FIELDS))
    if not requested and not configured:
        return None
    if not requested or not configured:
        raise RuntimeError("experimental verification mode is not explicit")
    candidates = request.get("candidates")
    blockers = request.get("quality_blockers")
    expected_quality_sha = str(
        config.get("experimental_quality_status_sha256") or ""
    ).lower()
    actual_quality_sha = str(request.get("quality_status_sha256") or "").lower()
    if (
        request.get("stage") != "fine"
        or request.get("expected_count") != 3
        or not isinstance(candidates, list)
        or len(candidates) != 3
        or len({item.get("candidate_id") for item in candidates}) != 3
        or request.get("quality_gate_passed") is not False
        or request.get("production_eligible") is not False
        or request.get("selection_policy") != "experimental_pareto_span_3_v1"
        or not isinstance(blockers, dict)
        or not blockers
        or not _full_sha(actual_quality_sha)
        or actual_quality_sha != expected_quality_sha
    ):
        raise RuntimeError("experimental verification evidence contract is invalid")
    if (
        str(request.get("solver_revision") or "").lower()
        != EXPERIMENTAL_SOLVER_REVISION
        or str(request.get("library_revision") or "").lower()
        != EXPERIMENTAL_LIBRARY_REVISION
    ):
        raise RuntimeError("experimental verification revision pins changed")

    priority = config.get("priority")
    accounts = config.get("accounts")
    guard_ids = config.get("q7_guard_task_ids")
    submission_env = config.get("submission_env")
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or priority >= 10
        or priority > 0
        or config.get("required_hard_cap") != 3
        or config.get("max_project_active_tasks") != 600
        or config.get("aedt_backend") != "pooled"
        or accounts != list(Q7_GUARD_ACCOUNTS)
        or guard_ids != list(Q7_GUARD_TASK_IDS)
        or submission_env != Q7_SUBMISSION_ENV
    ):
        raise RuntimeError("experimental scheduler safety contract is invalid")
    solver_root = os.path.abspath(os.fspath(config.get("solver_root") or ""))
    if not solver_root or not os.path.isdir(solver_root):
        raise RuntimeError("experimental verification needs a solver_root")
    return {
        "priority": priority,
        "accounts": list(accounts),
        "guard_ids": list(guard_ids),
        "submission_env": dict(submission_env),
        "required_hard_cap": 3,
        "max_project_active_tasks": 600,
        "aedt_backend": "pooled",
        "solver_root": solver_root,
        "guard_timeout_seconds": max(
            60, int(config.get("q7_guard_timeout_seconds", 48 * 3600))
        ),
    }


def _wait_for_q7_guard(scheduler, state, state_path, options, poll_seconds):
    """Wait until the exact q7 trio has completed successfully.

    Merely seeing terminal task states is insufficient: identity, priority,
    pooled backend, account placement, and exit code must all remain sealed.
    The durable state is updated on every observation so operators can see why
    no experimental scheduler mutation has happened yet.
    """

    started = time.monotonic()
    guard = state.setdefault(
        "pre_submission_guard",
        {
            "schema_version": 1,
            "task_ids": options["guard_ids"],
            "outcome": "waiting",
            "observations": {},
        },
    )
    if guard.get("task_ids") != options["guard_ids"]:
        raise RuntimeError("q7 guard task inventory changed across retry")
    while True:
        all_passed = True
        observations = {}
        for task_id, expected_account in zip(
            options["guard_ids"], Q7_GUARD_ACCOUNTS
        ):
            try:
                response = scheduler.requests.get(
                    f"{scheduler.SCHEDULER}/api/tasks/{task_id}", timeout=30
                )
                response.raise_for_status()
                task = response.json()
            except Exception as exc:
                observations[str(task_id)] = {
                    "status": "read_error",
                    "error": f"{type(exc).__name__}:{exc}",
                }
                all_passed = False
                continue
            status = str(task.get("status") or task.get("state") or "").lower()
            identity_ok = (
                task.get("id", task.get("task_id")) == task_id
                and str(task.get("name") or "").startswith(Q7_NAME_PREFIX)
                and task.get("priority") == 10
                and task.get("project") == scheduler.MFT_PROJECT
                and task.get("aedt_backend") == "pooled"
                and task.get("account_name") == expected_account
            )
            observations[str(task_id)] = {
                "status": status,
                "exit_code": task.get("exit_code"),
                "identity_ok": identity_ok,
                "finished_at": task.get("finished_at"),
            }
            if not identity_ok:
                guard.update(outcome="failed", observations=observations)
                _atomic_json(state, state_path)
                raise RuntimeError(f"q7 guard task identity changed: {task_id}")
            if status in {"failed", "cancelled"}:
                guard.update(outcome="failed", observations=observations)
                _atomic_json(state, state_path)
                raise RuntimeError(f"q7 guard task did not succeed: {task_id}:{status}")
            if status == "completed":
                if task.get("exit_code") != 0:
                    guard.update(outcome="failed", observations=observations)
                    _atomic_json(state, state_path)
                    raise RuntimeError(
                        f"q7 guard task has nonzero/unknown exit code: {task_id}"
                    )
            elif status in {"queued", "attaching", "running"}:
                all_passed = False
            else:
                guard.update(outcome="failed", observations=observations)
                _atomic_json(state, state_path)
                raise RuntimeError(f"q7 guard task state is invalid: {task_id}:{status}")
        guard.update(
            outcome="passed" if all_passed else "waiting",
            observations=observations,
            observed_at_epoch=time.time(),
        )
        _atomic_json(state, state_path)
        if all_passed:
            return
        if time.monotonic() - started >= options["guard_timeout_seconds"]:
            raise RuntimeError("q7 guard timed out before successful completion")
        time.sleep(poll_seconds)


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=1, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _load_state(path: Path, request_sha256: str, candidates: list[dict]):
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("request_sha256") != request_sha256:
            raise RuntimeError("scheduler verification request changed across retry")
        return state
    state = {
        "schema_version": 1,
        "request_sha256": request_sha256,
        "records": {
            item["candidate_id"]: {
                "rank": rank,
                "attempt": 0,
                "active_id": None,
                "outcome": "new",
            }
            for rank, item in enumerate(candidates)
        },
    }
    _atomic_json(state, path)
    return state


def _candidate_params(candidate):
    from module.input_parameter_260706 import KEYS

    raw = candidate.get("parameters")
    if not isinstance(raw, dict):
        raise RuntimeError("verification candidate parameters are unavailable")
    params = {key: raw[key] for key in KEYS if key in raw and raw[key] is not None}
    if set(params) != set(KEYS):
        raise RuntimeError(
            "verification candidate does not contain the sealed input schema"
        )
    return params


def run_scheduler_verification(request_path, result_path, config):
    """Submit/reconcile one exact manifest through the hardened scheduler client."""
    if not isinstance(config, dict) or set(config) - (
        BASE_CONFIG_FIELDS | EXPERIMENTAL_CONFIG_FIELDS
    ):
        raise RuntimeError("unknown scheduler verification configuration field")
    if config.get("adapter") != ADAPTER_NAME or config.get("execute") is not True:
        raise RuntimeError("reviewed scheduler verification is not explicitly enabled")
    library_root = os.path.abspath(os.fspath(config.get("library_root") or ""))
    if not library_root or not os.path.isdir(library_root):
        raise RuntimeError("reviewed scheduler verification needs a library_root")

    request_path = Path(request_path).resolve()
    result_path = Path(result_path).resolve()
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    experimental_options = _experimental_execution_contract(request, config)
    stage = request.get("stage")
    candidates = request.get("candidates")
    solver_revision = str(request.get("solver_revision") or "").lower()
    library_revision = str(request.get("library_revision") or "").lower()
    if stage not in ("standard", "fine") or not isinstance(candidates, list):
        raise RuntimeError("verification request schema is invalid")
    for label, revision in (
        ("solver", solver_revision), ("library", library_revision)
    ):
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise RuntimeError(f"verification request has no pinned {label} revision")

    repo_root = Path(__file__).resolve().parents[2]
    deployment_solver_root = (
        Path(experimental_options["solver_root"])
        if experimental_options else repo_root
    )
    from campaign.deployment_gate import validate_deployment

    validate_deployment(
        deployment_solver_root, solver_revision, library_root, library_revision
    )
    from verify import scheduler_client as scheduler

    profile_path = repo_root / "regression_260707" / "verify" / "profiles" / f"{stage}.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    state_path = result_path.parent / "scheduler_state.json"
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    state = _load_state(state_path, request_sha256, candidates)
    records = state["records"]
    by_id = {item["candidate_id"]: item for item in candidates}
    if set(records) != set(by_id):
        raise RuntimeError("scheduler task inventory does not match selection")

    poll_seconds = max(5, int(config.get("poll_seconds", 180)))
    if experimental_options and any(
        record.get("outcome") in ("new", "submitting", "retry_submitting")
        for record in records.values()
    ):
        _wait_for_q7_guard(
            scheduler, state, state_path, experimental_options, poll_seconds
        )

    def submit_record(candidate_id, record, *, retry):
        params = _candidate_params(by_id[candidate_id])
        submitted = scheduler.effective_verification_params(params, profile)
        rank = int(record["rank"])
        short = request["input_generation_id"][:10]
        base_name = f"mft-pipe-{stage}-{short}-{rank:02d}"
        base_workdir = f"mft_pipe_{stage}_{short}_{rank:02d}"
        name = f"{base_name}-retry" if retry else base_name
        workdir = f"{base_workdir}_retry" if retry else base_workdir
        submitting = "retry_submitting" if retry else "submitting"
        expected_attempt = 1 if retry else 0
        if record.get("outcome") != submitting:
            record.update(
                active_id=None,
                attempt=expected_attempt,
                outcome=submitting,
                submitted_params=submitted,
                name=name,
                workdir=workdir,
            )
            # Seal the deterministic scheduler identity before the call.  If
            # the process dies after POST, a queue retry invokes the hardened
            # scheduler client's exact name/dedupe reconciliation instead of
            # inventing a different task identity.
            _atomic_json(state, state_path)
        elif (
            record.get("name") != name
            or record.get("workdir") != workdir
            or int(record.get("attempt", -1)) != expected_attempt
            or record.get("submitted_params") != submitted
        ):
            raise RuntimeError("sealed scheduler submission identity changed")
        experimental_kwargs = {}
        if experimental_options:
            experimental_kwargs = {
                "priority": experimental_options["priority"],
                "account_name": experimental_options["accounts"][rank],
                "aedt_backend": experimental_options["aedt_backend"],
                "submission_env": experimental_options["submission_env"],
                "required_hard_cap": experimental_options["required_hard_cap"],
                "max_project_active_tasks": experimental_options[
                    "max_project_active_tasks"
                ],
            }
        task_id = scheduler.submit_verification(
            name,
            workdir,
            params,
            profile,
            mem_mb=(
                max(
                    int(profile.get("mem_mb", 32768)),
                    131072 if stage == "fine" else 65536,
                )
                if retry else int(profile.get("mem_mb", 32768))
            ),
            cpus=int(profile.get("cpus", 4)),
            solver_revision=solver_revision,
            library_revision=library_revision,
            **experimental_kwargs,
        )
        if task_id is None:
            record["last_submission_error"] = "task_identity_unknown"
            _atomic_json(state, state_path)
            raise RuntimeError(
                f"scheduler submission identity is unknown: {candidate_id}; "
                "the sealed identity will be reconciled on retry"
            )
        record.update(active_id=task_id, outcome="pending")
        record.pop("last_submission_error", None)
        _atomic_json(state, state_path)
        time.sleep(0.05)

    for candidate_id, record in records.items():
        outcome = record.get("outcome")
        if outcome in ("new", "submitting"):
            submit_record(candidate_id, record, retry=False)
        elif outcome == "retry_submitting":
            submit_record(candidate_id, record, retry=True)

    timeout_seconds = max(60, int(config.get("timeout_seconds", 6 * 3600)))
    while True:
        pending = {
            candidate_id: record
            for candidate_id, record in records.items()
            if record.get("outcome") == "pending"
        }
        if not pending:
            break
        task_ids = [int(record["active_id"]) for record in pending.values()]
        statuses = scheduler.wait_all(
            task_ids, poll_s=poll_seconds, timeout_s=timeout_seconds
        )
        retried = False
        unresolved = False
        for candidate_id, record in pending.items():
            task_id = int(record["active_id"])
            status = statuses.get(task_id)
            record["last_status"] = status
            if status not in TERMINAL:
                unresolved = True
                continue
            fetched = scheduler.fetch_result(
                task_id,
                expected_revision=solver_revision,
                expected_library_revision=library_revision,
                expected_profile=profile.get("param_overrides"),
            )
            matches = (
                fetched.state == scheduler.RESULT_VALID
                and scheduler.result_matches_params(
                    fetched.result,
                    record["submitted_params"],
                    required_keys=record["submitted_params"].keys(),
                )
            )
            if matches:
                from optimization.geometry_metrics import bounding_box_lit
                from verify.finalize import physical_spec_reasons

                result = fetched.result
                record["result"] = result
                record["actual_volume_L"] = float(bounding_box_lit(result)[0])
                losses = (
                    result.get("P_winding_total"),
                    result.get("P_core_total"),
                    result.get("P_core_plate_total"),
                    result.get("P_wcp_total"),
                )
                if all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in losses
                ):
                    record["actual_total_loss_W"] = sum(float(value) for value in losses)
                record["valid"] = not physical_spec_reasons(result)
                record["outcome"] = "terminal"
                _atomic_json(state, state_path)
                continue
            record["last_result_state"] = fetched.state
            if int(record.get("attempt", 0)) >= 1:
                record.update(outcome="terminal", valid=False)
                _atomic_json(state, state_path)
                continue
            submit_record(candidate_id, record, retry=True)
            retried = True
        if unresolved:
            _atomic_json(state, state_path)
            raise RuntimeError("verification tasks did not reach a terminal state")
        if not retried and not any(
            record.get("outcome") == "pending" for record in records.values()
        ):
            break

    output = []
    for candidate in candidates:
        record = records[candidate["candidate_id"]]
        if record.get("outcome") != "terminal":
            raise RuntimeError("verification task inventory is not terminal")
        output.append(
            {
                "candidate_id": candidate["candidate_id"],
                "completed": True,
                "valid": bool(record.get("valid")),
                "task_id": record.get("active_id"),
                "attempt": int(record.get("attempt", 0)),
                "actual_volume_L": record.get("actual_volume_L"),
                "actual_total_loss_W": record.get("actual_total_loss_W"),
                "result": record.get("result"),
            }
        )
    document = {"schema_version": 1, "stage": stage, "results": output}
    _atomic_json(document, result_path)
    return document
