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
    "guard_cohort",
    "guard_cohort_sha256",
    "guard_timeout_seconds",
    "experimental_quality_status_sha256",
}
EXPERIMENTAL_SUBMISSION_ACCOUNTS = ("r1jae262", "dhj02", "r1jae262")
EXPERIMENTAL_SUBMISSION_ENV = {
    "MFT_AEDT_BACKEND": "pooled",
    "MFT_AEDT_ISOLATION_POLICY": "family",
    "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
    "MFT_AEDT_RELEASE_WAIT_SECONDS": "7200",
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
EXACT_COHORT_SCHEMA_VERSION = "mft-aedt-exact-cohort-v1"
EXACT_COHORT_PROJECT = "MFT_1MW_2026v1"
EXACT_COHORT_FIELDS = {
    "schema_version",
    "cohort_id",
    "session_id",
    "session_generation",
    "solve_batch_generation",
    "solver_revision",
    "library_revision",
    "members",
}
EXACT_COHORT_MEMBER_FIELDS = {
    "task_id",
    "lease_id",
    "session_id",
    "session_generation",
    "solve_generation",
    "slot_index",
    "name",
    "dedupe_key",
    "account_name",
    "project",
    "priority",
    "aedt_backend",
    "solver_revision",
    "library_revision",
    "lease_request_key",
    "lease_project_name",
}
EXACT_COHORT_RESULT_FLAGS = (
    "result_valid_em",
    "result_valid_thermal",
    "thermal_solved",
    "thermal_extraction_complete",
    "thermal_convergence_available",
    "thermal_converged",
    "core_native_material_readback_attested",
    "core_loss_native_attested",
    "B_mean_faraday_attested",
    "native_core_report_coverage_attested",
)


def _full_sha(value) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _git_sha(value) -> bool:
    text = str(value or "").lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _exact_cohort_contract(cohort, supplied_sha256, solver_revision, library_revision):
    """Authenticate one not-yet-known exact 1-AEDT/3-project cohort.

    The cohort is deliberately configuration data rather than source constants:
    q21b task and lease identities do not exist until that canary is submitted.
    A canonical digest prevents a later retry from silently swapping any member.
    """

    if not isinstance(cohort, dict) or set(cohort) != EXACT_COHORT_FIELDS:
        raise RuntimeError("experimental exact cohort schema is invalid")
    digest = str(supplied_sha256 or "").lower()
    if not _full_sha(digest) or digest != _canonical_sha256(cohort):
        raise RuntimeError("experimental exact cohort seal is invalid")
    session_id = cohort.get("session_id")
    session_generation = cohort.get("session_generation")
    solve_generation = cohort.get("solve_batch_generation")
    members = cohort.get("members")
    if (
        cohort.get("schema_version") != EXACT_COHORT_SCHEMA_VERSION
        or not str(cohort.get("cohort_id") or "").strip()
        or not _positive_int(session_id)
        or not _positive_int(session_generation)
        or not _positive_int(solve_generation)
        or str(cohort.get("solver_revision") or "").lower() != solver_revision
        or str(cohort.get("library_revision") or "").lower() != library_revision
        or not isinstance(members, list)
        or len(members) != 3
    ):
        raise RuntimeError("experimental exact cohort contract is invalid")
    task_ids = set()
    lease_ids = set()
    slots = set()
    for member in members:
        if not isinstance(member, dict) or set(member) != EXACT_COHORT_MEMBER_FIELDS:
            raise RuntimeError("experimental exact cohort member schema is invalid")
        task_id = member.get("task_id")
        lease_id = member.get("lease_id")
        slot_index = member.get("slot_index")
        if (
            not _positive_int(task_id)
            or not _positive_int(lease_id)
            or isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or slot_index not in {0, 1, 2}
            or member.get("session_id") != session_id
            or member.get("session_generation") != session_generation
            or member.get("solve_generation") != solve_generation
            or member.get("project") != EXACT_COHORT_PROJECT
            or member.get("priority") != 10
            or member.get("aedt_backend") != "pooled"
            or str(member.get("solver_revision") or "").lower() != solver_revision
            or str(member.get("library_revision") or "").lower() != library_revision
            or not str(member.get("name") or "").strip()
            or not str(member.get("dedupe_key") or "").strip()
            or not str(member.get("account_name") or "").strip()
            or not str(member.get("lease_request_key") or "").strip()
            or not str(member.get("lease_project_name") or "").strip()
        ):
            raise RuntimeError("experimental exact cohort member contract is invalid")
        task_ids.add(task_id)
        lease_ids.add(lease_id)
        slots.add(slot_index)
    if len(task_ids) != 3 or len(lease_ids) != 3 or slots != {0, 1, 2}:
        raise RuntimeError("experimental exact cohort inventory is not one exact triple")
    return json.loads(json.dumps(cohort, sort_keys=True))


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
    solver_revision = str(request.get("solver_revision") or "").lower()
    library_revision = str(request.get("library_revision") or "").lower()
    training_solver_revision = str(
        request.get("training_solver_revision") or ""
    ).lower()
    training_library_revision = str(
        request.get("training_library_revision") or ""
    ).lower()
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
        or not _git_sha(solver_revision)
        or not _git_sha(library_revision)
        or not _git_sha(training_solver_revision)
        or not _git_sha(training_library_revision)
        or str(request.get("fea_solver_revision") or "").lower()
        != solver_revision
        or str(request.get("fea_library_revision") or "").lower()
        != library_revision
    ):
        raise RuntimeError("experimental verification evidence contract is invalid")

    priority = config.get("priority")
    accounts = config.get("accounts")
    submission_env = config.get("submission_env")
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or priority >= 10
        or priority > 0
        or config.get("required_hard_cap") != 3
        or config.get("max_project_active_tasks") != 600
        or config.get("aedt_backend") != "pooled"
        or accounts != list(EXPERIMENTAL_SUBMISSION_ACCOUNTS)
        or submission_env != EXPERIMENTAL_SUBMISSION_ENV
    ):
        raise RuntimeError("experimental scheduler safety contract is invalid")
    solver_root = os.path.abspath(os.fspath(config.get("solver_root") or ""))
    if not solver_root or not os.path.isdir(solver_root):
        raise RuntimeError("experimental verification needs a solver_root")
    guard_cohort = _exact_cohort_contract(
        config.get("guard_cohort"),
        config.get("guard_cohort_sha256"),
        solver_revision,
        library_revision,
    )
    timeout = config.get("guard_timeout_seconds", 48 * 3600)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 60 <= timeout <= 48 * 3600
    ):
        raise RuntimeError("experimental exact cohort timeout is invalid")
    return {
        "priority": priority,
        "accounts": list(accounts),
        "guard_cohort": guard_cohort,
        "guard_cohort_sha256": str(config["guard_cohort_sha256"]).lower(),
        "submission_env": dict(submission_env),
        "required_hard_cap": 3,
        "max_project_active_tasks": 600,
        "aedt_backend": "pooled",
        "solver_root": solver_root,
        "guard_timeout_seconds": timeout,
    }


class _GuardEvidencePending(RuntimeError):
    """A sealed identity exists but its public terminal evidence is not ready."""


def _guard_result_evidence(scheduler, member, solver_revision, library_revision):
    try:
        response = scheduler.requests.get(
            f"{scheduler.SCHEDULER}/api/tasks/{member['task_id']}/stdout",
            params={"max_bytes": 1_048_576},
            timeout=30,
        )
        response.raise_for_status()
        output = response.text
    except Exception as exc:
        raise _GuardEvidencePending(
            f"stdout read pending:{type(exc).__name__}:{exc}"
        ) from exc
    library_marker = ""
    result = None
    for line in reversed(output.splitlines()):
        if not library_marker and line.startswith("MFT_LIBRARY_GIT_HASH "):
            candidate = line[len("MFT_LIBRARY_GIT_HASH "):].strip().lower()
            if _git_sha(candidate):
                library_marker = candidate
        if result is None and line.startswith("RESULT_JSON "):
            try:
                candidate = json.loads(line[len("RESULT_JSON "):])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict):
                result = candidate
        if result is not None and library_marker:
            break
    if result is None or not library_marker:
        raise _GuardEvidencePending("terminal RESULT_JSON/library marker is unavailable")
    bad_flags = [key for key in EXACT_COHORT_RESULT_FLAGS if result.get(key) != 1]
    if (
        bad_flags
        or result.get("thermal_required_missing_count") != 0
        or result.get("thermal_probe_failure_count") != 0
        or result.get("thermal_dispatch_status") != "success"
        or result.get("matrix_solve_attempts") != 1
        or result.get("loss_solve_attempts") != 1
        or result.get("thermal_solve_attempts") != 1
        or result.get("git_dirty") != 0
        or result.get("pyaedt_library_git_dirty") != 0
        or str(result.get("git_hash") or "").lower() != solver_revision
        or str(result.get("pyaedt_library_git_hash") or "").lower()
        != library_revision
        or library_marker != library_revision
        or str(result.get("project_name") or "")
        != member["lease_project_name"]
        or not str(result.get("saved_at") or "").strip()
    ):
        raise RuntimeError(
            f"exact cohort result evidence changed for task {member['task_id']}: "
            f"bad_flags={bad_flags}"
        )
    return {
        "result_valid_em": 1,
        "result_valid_thermal": 1,
        "thermal_converged": 1,
        "native_attestation_count": sum(
            result.get(key) == 1 for key in EXACT_COHORT_RESULT_FLAGS
        ),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "project_name": result["project_name"],
        "saved_at": result.get("saved_at"),
    }


def _guard_pool_evidence(scheduler, cohort):
    try:
        response = scheduler.requests.get(
            f"{scheduler.SCHEDULER}/api/aedt-pool", timeout=30
        )
        response.raise_for_status()
        summary = response.json()
    except Exception as exc:
        raise _GuardEvidencePending(
            f"AEDT pool evidence pending:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(summary, dict):
        raise _GuardEvidencePending("AEDT pool summary is unavailable")
    sessions = [
        item
        for collection in (summary.get("sessions"), summary.get("session_history"))
        if isinstance(collection, list)
        for item in collection
        if isinstance(item, dict) and item.get("id") == cohort["session_id"]
    ]
    if not sessions:
        raise _GuardEvidencePending("sealed AEDT session is absent from public evidence")
    session = sessions[0]
    if any(item != session for item in sessions[1:]):
        raise RuntimeError("sealed AEDT session has conflicting public evidence")
    try:
        last_fault_evidence = json.loads(
            str(session.get("last_fault_evidence_json") or "{}")
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("sealed AEDT session fault evidence is invalid") from exc
    if (
        session.get("generation") != cohort["session_generation"]
        or session.get("solve_batch_generation")
        != cohort["solve_batch_generation"]
        or session.get("slots_total") != 3
        or str(session.get("failure_message") or "")
        or str(session.get("quarantine_reason") or "")
        or session.get("last_fault_at") not in (None, "")
        or last_fault_evidence != {}
        or not str(session.get("process_id") or "").strip()
    ):
        raise RuntimeError("sealed AEDT session identity/health evidence changed")
    session_state = str(session.get("state") or "").lower()
    currently_reusable = (
        session_state == "ready"
        and session.get("active_lease_count") == 0
        and session.get("free_slot_count") == 3
    )
    idle_since = str(session.get("idle_since") or "")
    drain_requested_at = str(session.get("drain_requested_at") or "")
    closed_at = str(session.get("closed_at") or "")
    clean_idle_close = (
        session_state == "closed"
        and session.get("active_lease_count") == 0
        and session.get("free_slot_count") == 3
        and bool(idle_since and drain_requested_at and closed_at)
        and idle_since <= drain_requested_at <= closed_at
    )
    reusable_evidence = currently_reusable or clean_idle_close
    if session_state in {"failed", "unhealthy"} or (
        session_state == "closed" and not clean_idle_close
    ):
        raise RuntimeError(f"sealed AEDT session is not reusable: {session_state}")
    leases = summary.get("leases")
    if not isinstance(leases, list):
        raise _GuardEvidencePending("AEDT lease inventory is unavailable")
    by_id = {
        item.get("id"): item for item in leases if isinstance(item, dict)
    }
    lease_observations = {}
    leases_released = True
    for member in cohort["members"]:
        lease = by_id.get(member["lease_id"])
        if lease is None:
            raise _GuardEvidencePending(
                f"sealed AEDT lease is absent: {member['lease_id']}"
            )
        identity_ok = (
            lease.get("task_id") == member["task_id"]
            and lease.get("session_id") == cohort["session_id"]
            and lease.get("slot_index") == member["slot_index"]
            and lease.get("request_key") == member["lease_request_key"]
            and lease.get("project_name") == member["lease_project_name"]
            and not str(lease.get("failure_message") or "")
        )
        if not identity_ok:
            raise RuntimeError(
                f"sealed AEDT lease identity changed: {member['lease_id']}"
            )
        lease_state = str(lease.get("state") or "").lower()
        if lease_state in {"failed", "cancelled", "faulted"}:
            raise RuntimeError(
                f"sealed AEDT lease did not release: {member['lease_id']}:{lease_state}"
            )
        released = lease_state == "released"
        leases_released = leases_released and released
        lease_observations[str(member["lease_id"])] = {
            "state": lease_state,
            "released": released,
            "task_id": member["task_id"],
            "session_id": cohort["session_id"],
            "slot_index": member["slot_index"],
        }
    return reusable_evidence and leases_released, {
        "session": {
            "id": cohort["session_id"],
            "state": session_state,
            "generation": session.get("generation"),
            "solve_batch_generation": session.get("solve_batch_generation"),
            "active_lease_count": session.get("active_lease_count"),
            "free_slot_count": session.get("free_slot_count"),
            "reusable_evidence": (
                "currently_ready"
                if currently_reusable
                else "clean_idle_close_after_release"
                if clean_idle_close
                else "waiting"
            ),
        },
        "leases": lease_observations,
    }


def _wait_for_exact_cohort_guard(
    scheduler, state, state_path, options, poll_seconds
):
    """Require terminal, released, native-valid evidence for one sealed triple."""

    started = time.monotonic()
    cohort = options["guard_cohort"]
    task_ids = [member["task_id"] for member in cohort["members"]]
    guard = state.setdefault(
        "pre_submission_guard",
        {
            "schema_version": 2,
            "cohort_sha256": options["guard_cohort_sha256"],
            "task_ids": task_ids,
            "outcome": "waiting",
            "observations": {},
        },
    )
    if (
        guard.get("schema_version") != 2
        or guard.get("cohort_sha256") != options["guard_cohort_sha256"]
        or guard.get("task_ids") != task_ids
    ):
        raise RuntimeError("exact cohort guard inventory changed across retry")
    while True:
        all_passed = True
        observations = {}
        try:
            for member in cohort["members"]:
                task_id = member["task_id"]
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
                    and task.get("name") == member["name"]
                    and task.get("dedupe_key") == member["dedupe_key"]
                    and task.get("priority") == member["priority"]
                    and task.get("project") == member["project"]
                    and task.get("aedt_backend") == member["aedt_backend"]
                    and task.get("account_name") == member["account_name"]
                )
                task_observation = {
                    "status": status,
                    "exit_code": task.get("exit_code"),
                    "identity_ok": identity_ok,
                    "finished_at": task.get("finished_at"),
                }
                observations[str(task_id)] = task_observation
                if not identity_ok:
                    raise RuntimeError(
                        f"exact cohort task identity changed: {task_id}"
                    )
                if status in {"failed", "cancelled"}:
                    raise RuntimeError(
                        f"exact cohort task did not succeed: {task_id}:{status}"
                    )
                if status == "completed":
                    if task.get("exit_code") != 0:
                        raise RuntimeError(
                            f"exact cohort task has nonzero/unknown exit code: {task_id}"
                        )
                    try:
                        task_observation["result_evidence"] = _guard_result_evidence(
                            scheduler,
                            member,
                            cohort["solver_revision"],
                            cohort["library_revision"],
                        )
                    except _GuardEvidencePending as exc:
                        task_observation["result_evidence_pending"] = str(exc)
                        all_passed = False
                elif status in {"queued", "attaching", "running"}:
                    all_passed = False
                else:
                    raise RuntimeError(
                        f"exact cohort task state is invalid: {task_id}:{status}"
                    )
            try:
                pool_passed, pool_observation = _guard_pool_evidence(
                    scheduler, cohort
                )
                observations["aedt_pool"] = pool_observation
                all_passed = all_passed and pool_passed
            except _GuardEvidencePending as exc:
                observations["aedt_pool"] = {
                    "status": "evidence_pending", "reason": str(exc)
                }
                all_passed = False
        except RuntimeError:
            guard.update(
                outcome="failed",
                observations=observations,
                observed_at_epoch=time.time(),
            )
            _atomic_json(state, state_path)
            raise
        guard.update(
            outcome="passed" if all_passed else "waiting",
            observations=observations,
            observed_at_epoch=time.time(),
        )
        _atomic_json(state, state_path)
        if all_passed:
            return
        if time.monotonic() - started >= options["guard_timeout_seconds"]:
            raise RuntimeError(
                "exact cohort guard timed out before reusable terminal evidence"
            )
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
        _wait_for_exact_cohort_guard(
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
