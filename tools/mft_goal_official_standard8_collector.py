"""GET-only watcher/collector for official Standard candidate #8 task 96329.

This adapter is deliberately bound to the immutable candidate-eight plan,
single POST receipt, and final seal.  A queued strict-n114 task is observable,
but any attached task must prove actual n114 placement without node relaxation.
Only Scheduler GET and retained remote-file GET paths are reachable here.
Collected solver output remains a noncanonical observation and never becomes a
scientific PASS or infeasibility claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import mft_goal_official_standard8_postdeadline as official  # noqa: E402
from tools import mft_goal_postdeadline_standard_collector as base  # noqa: E402


TASK_ID = 96_329
TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official8-"
    "s96141-622097dde126-n114"
)
DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-postdeadline-official8-"
    "s96141-622097dde126-n114:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "6ad4b1c5a342de6b"
)
CANDIDATE_PHYSICS_SHA256 = (
    "622097dde126e12540a9aad52cb992d880ddb47a7d881f722858f624fd8878f4"
)
PLAN_FILE_SHA256 = (
    "fe8a0cf379570d27878e3d2073557bd53ffbfaf4445e9625083d1bb5184eac18"
)
PLAN_PAYLOAD_SHA256 = (
    "bb2b885eb308546d400a0caaf49ce439e7d0397ac1c0926e347a27dc7a9a3678"
)
INTENT_FILE_SHA256 = (
    "7a1e309924f032af45ce6b6608cfaccabf71991fa6824daacae1843788027523"
)
INTENT_PAYLOAD_SHA256 = (
    "d24d7a1820fa64234de2eca777b0be0c929debd019fd02bccd1020ff3d893855"
)
ATTEMPT_FILE_SHA256 = (
    "8369dc620773b50a8b2ff5281b760a52d5b043d885fad720972174dbd2cb25c1"
)
ATTEMPT_PAYLOAD_SHA256 = (
    "1847f93e8edf5a5a703ac7530d6d6f30183e35a32d7f3cf5479864531c3364b0"
)
SUBMISSION_FILE_SHA256 = (
    "8d7e59fe51523f22e3692ef87ee529d619b4a18b5e0a5353664bef9a73500ff5"
)
SUBMISSION_PAYLOAD_SHA256 = (
    "dd2310cb02dd41166a10fdb1c5a53beb0c1879f6f8971029c50536f5730b1362"
)
FINAL_FILE_SHA256 = (
    "5319a8a4dceb27082b91fc6badd540221298eeb9f324e2fa30e76e529313d3ec"
)
FINAL_PAYLOAD_SHA256 = (
    "664ac155eb62e5b2448a5ef4ea3f20ff22d525ed0195bb84e3c51f957cefef1e"
)
ATTEMPT_SCHEMA = (
    "mft-goal-official-standard-postdeadline-scheduler-post-attempt-v1"
)
EXPECTED_NODE = "n114"
EXPECTED_ACCOUNT = "jji0930"
SCHEDULER_URL = "http://127.0.0.1:8002"
RUNTIME_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official8_622097dde126_n114_260726_v1"
)
DEFAULT_PLAN = RUNTIME_ROOT / "diagnostic_postdeadline_plan.json"
DEFAULT_SUBMISSION = RUNTIME_ROOT / "submission" / "submission_receipt.json"
DEFAULT_OUTPUT = RUNTIME_ROOT / "authenticated_get_collection_task96329_v1"

JsonGetter = Callable[..., bytes]
PlanLoader = Callable[
    [Path],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]


class Official8CollectionError(base.CollectionError):
    """Candidate-eight source, placement, or retained evidence drifted."""


def _raise(message: str) -> None:
    raise Official8CollectionError(message)


def _validate_safety(value: Mapping[str, Any], label: str) -> None:
    if any(
        value.get(key) is not expected
        for key, expected in official.SAFETY_FLAGS.items()
    ):
        _raise(f"{label} safety classification drifted")


def _read_sealed(path: Path, schema: str, label: str) -> dict[str, Any]:
    value = base._read_json(path, label)  # noqa: SLF001
    try:
        base._validate_seal(value, schema, label)  # noqa: SLF001
    except base.CollectionError as exc:
        raise Official8CollectionError(str(exc)) from exc
    return value


def _verify_record(
    record: Any,
    label: str,
    *,
    expected_path: Path,
) -> Path:
    try:
        path = base._verify_file_record(record, label)  # noqa: SLF001
    except base.CollectionError as exc:
        raise Official8CollectionError(str(exc)) from exc
    if path != expected_path.resolve(strict=True):
        _raise(f"{label} does not bind the exact expected file")
    return path


def _validate_opening_preflight(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        _raise(f"{label} is malformed")
    capacity = value.get("capacity")
    if (
        value.get("schema_version") != official.PREFLIGHT_SCHEMA
        or value.get("queue_state") != "opening"
        or value.get("queue_reason") != official.OPENING_QUEUE_REASON
        or value.get("ready_fit_slots") != 0
        or value.get("pending_fit_slots") != 0
        or value.get("inflight_fit_slots") != 0
        or value.get("active_n114_fea_count") != 0
        or value.get("collision_count") != 0
        or value.get("preferred_node_relaxed") is not False
        or value.get("exact_account_node_opening") is not True
        or value.get("strict_node_required") is not True
        or value.get("fixed_physics_unchanged") is not True
        or not isinstance(capacity, Mapping)
        or capacity.get("queue_state") != "opening"
        or capacity.get("queue_reason") != official.OPENING_QUEUE_REASON
        or capacity.get("fit_slots") != 0
        or capacity.get("ready_fit_slots") != 0
        or capacity.get("pending_fit_slots") != 0
        or capacity.get("inflight_fit_slots") != 0
        or capacity.get("preferred_node_relaxed") is not False
        or capacity.get("allocations") != []
    ):
        _raise(f"{label} strict opening identity drifted")


def _validate_plan(
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    scheduler_url: str,
) -> dict[str, Any]:
    _validate_safety(plan, "official #8 plan")
    payload = plan.get("scheduler_payload")
    resources = plan.get("resources")
    placement = plan.get("placement")
    opening = plan.get("opening_demand_pool_contract")
    retained = plan.get("retained_aedt_bundle")
    single_attempt = plan.get("single_attempt_contract")
    transport = (
        retained.get("transport") if isinstance(retained, Mapping) else None
    )
    parameter_digest = (
        str(retained.get("parameter_digest") or "")
        if isinstance(retained, Mapping)
        else ""
    )
    task_identity = base.sha256_bytes(DEDUPE_KEY.encode("utf-8"))[:16]
    retained_root = f"goal-fea-retained/{task_identity}"
    marker = {
        "schema": base.MARKER_SCHEMA,
        "preserve": True,
        "reason": (
            "Retain diagnostic Standard AEDT and AEDT results until "
            "authenticated diagnostic collection; stage=standard; "
            f"dedupe_key={DEDUPE_KEY}"
        ),
        "owner": base.SCHEDULER_PROJECT,
    }
    if (
        plan.get("schema_version") != official.PLAN_SCHEMA
        or plan.get("payload_sha256") != PLAN_PAYLOAD_SHA256
        or plan.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or plan.get("official_standard_selection_order") != 8
        or plan.get("source_seed") != official.SOURCE_SEED
        or plan.get("source_task_id") != official.SOURCE_TASK_ID
        or plan.get("task_name") != TASK_NAME
        or plan.get("dedupe_key") != DEDUPE_KEY
        or plan.get("standard_only") is not True
        or plan.get("symmetric_model") is not True
        or plan.get("full_model") is not False
        or plan.get("thermal_symmetry") != "eighth"
        or plan.get("fixed_boundary") != official.FIXED_BOUNDARY
        or plan.get("fixed_physics_unchanged") is not True
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or str(plan.get("scheduler_url") or "").rstrip("/")
        != scheduler_url.rstrip("/")
        or plan.get("scheduler_project") != base.SCHEDULER_PROJECT
        or plan.get("solver_revision") != official.SOLVER_REVISION
        or plan.get("library_revision") != official.LIBRARY_REVISION
        or plan.get("execution_profile_canonical_sha256")
        != base.payload_sha256(profile)
        or not isinstance(resources, Mapping)
        or resources.get("cpus") != 8
        or resources.get("memory_mb") != 98_304
        or resources.get("solver_seconds") != 43_200
        or resources.get("kill_grace_seconds") != 300
        or resources.get("retention_seconds") != 1_800
        or resources.get("scheduler_timeout_seconds") != 45_300
        or resources.get("max_workers_per_node") != 1
        or not isinstance(placement, Mapping)
        or placement.get("account_name") != EXPECTED_ACCOUNT
        or placement.get("node_name") != EXPECTED_NODE
        or placement.get("node_name_policy") != "strict"
        or placement.get("allocation_state_at_prepare") != "opening"
        or placement.get("allocation_id_at_prepare") is not None
        or placement.get("slurm_job_id_at_prepare") is not None
        or placement.get("preferred_node_relaxed_allowed") is not False
        or not isinstance(opening, Mapping)
        or opening.get("allowed_pre_submit_queue_state") != "opening"
        or opening.get("required_queue_reason")
        != official.OPENING_QUEUE_REASON
        or opening.get("active_target_fea_count") != 0
        or opening.get("require_reget_inside_mutation_lock") is not True
        or opening.get("relaxed_allocation_allowed") is not False
        or not isinstance(payload, Mapping)
        or plan.get("scheduler_payload_sha256")
        != base.payload_sha256(payload)
        or payload.get("name") != TASK_NAME
        or payload.get("dedupe_key") != DEDUPE_KEY
        or payload.get("project") != base.SCHEDULER_PROJECT
        or payload.get("account_name") != EXPECTED_ACCOUNT
        or payload.get("node_name") != EXPECTED_NODE
        or payload.get("node_name_policy") != "strict"
        or payload.get("cpus") != 8
        or payload.get("memory_mb") != 98_304
        or payload.get("timeout_seconds") != 45_300
        or payload.get("max_workers_per_node") != 1
        or payload.get("aedt_backend") != "standalone"
        or "same_node_as_task_id" in payload
        or not isinstance(retained, Mapping)
        or retained.get("stage") != "standard"
        or retained.get("dedupe_key") != DEDUPE_KEY
        or not DEDUPE_KEY.endswith(f":{parameter_digest}")
        or retained.get("relative_directory") != retained_root
        or retained.get("artifact_path") != f"{retained_root}/symmetric.aedt"
        or retained.get("receipt_path")
        != f"{retained_root}/symmetric.aedt.receipt.json"
        or retained.get("marker_path")
        != f"{retained_root}/.slurm-scheduler-preserve.json"
        or retained.get("results_path")
        != f"{retained_root}/symmetric.aedtresults"
        or retained.get("results_manifest_path")
        != f"{retained_root}/symmetric.aedtresults.manifest.json"
        or retained.get("marker_contract") != marker
        or retained.get("marker_contract_sha256")
        != base.payload_sha256(marker)
        or not isinstance(transport, Mapping)
        or transport.get("schema_version") != base.TRANSPORT_SCHEMA
        or transport.get("encoding") != "base64"
        or transport.get("raw_chunk_bytes") != base.RAW_CHUNK_BYTES
        or transport.get("chunk_directory")
        != f"{retained_root}/symmetric.aedt.chunks"
        or not isinstance(single_attempt, Mapping)
        or single_attempt.get("schema_version")
        != official.SINGLE_ATTEMPT_SCHEMA
        or single_attempt.get("post_call_budget") != 1
        or single_attempt.get("ledger_consumed_before_network") is not True
        or single_attempt.get("output_override_allowed") is not False
    ):
        _raise("official #8 plan fixed identity/retention contract drifted")
    _validate_opening_preflight(plan.get("preflight"), "plan preflight")
    return {
        "parameter_digest": parameter_digest,
        "profile_sha256": base.payload_sha256(profile),
        "retained": copy.deepcopy(dict(retained)),
        "retained_root": retained_root,
        "marker_contract": marker,
    }


def _validate_task_snapshot(
    task: Any,
    *,
    label: str,
    require_bound_placement: bool | None = None,
) -> dict[str, Any]:
    if not isinstance(task, dict):
        _raise(f"{label} is not an object")
    expected = {
        "name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "project": base.SCHEDULER_PROJECT,
        "requested_account_name": EXPECTED_ACCOUNT,
        "requested_node_name": EXPECTED_NODE,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
    }
    task_id = task.get("task_id", task.get("id"))
    if task_id != TASK_ID or any(
        task.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        _raise(f"{label} exact task identity drifted")
    if (
        task.get("node_name") != EXPECTED_NODE
        or task.get("node_name_policy") != "strict"
        or task.get("strict_node_placement") is not True
        or task.get("preferred_node_relaxed") is not False
        or task.get("account_name") != EXPECTED_ACCOUNT
    ):
        _raise(f"{label} strict requested placement drifted")
    attached = bool(
        task.get("attached_at")
        or task.get("allocation_id")
        or task.get("assigned_allocation")
        or task.get("slurm_job_id")
        or str(task.get("state") or "")
        in {"attaching", "running", "succeeded", "failed", "cancelled"}
        or str(task.get("status") or "")
        in {"attaching", "running", "completed", "failed", "cancelled"}
    )
    if require_bound_placement is True:
        attached = True
    elif require_bound_placement is False:
        attached = False
    if attached:
        allocation_id = int(
            task.get("allocation_id")
            or task.get("assigned_allocation")
            or 0
        )
        if (
            task.get("actual_node_name") != EXPECTED_NODE
            or task.get("allocation_node_name") != EXPECTED_NODE
            or allocation_id <= 0
            or not str(task.get("slurm_job_id") or "")
            or task.get("placement_contract_satisfied") is not True
        ):
            _raise(f"{label} attached strict n114 placement drifted")
    else:
        if (
            str(task.get("actual_node_name") or "")
            or str(task.get("allocation_node_name") or "")
            or task.get("allocation_id") is not None
            or task.get("assigned_allocation") is not None
            or str(task.get("slurm_job_id") or "")
            or task.get("placement_contract_satisfied") is not False
        ):
            _raise(f"{label} queued placement unexpectedly bound")
    return copy.deepcopy(task)


def _validate_submission(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    submission_path: Path,
) -> dict[str, Any]:
    submission = _read_sealed(
        submission_path,
        official.SUBMISSION_SCHEMA,
        "official #8 submission receipt",
    )
    if (
        base.sha256_file(submission_path) != SUBMISSION_FILE_SHA256
        or submission.get("payload_sha256") != SUBMISSION_PAYLOAD_SHA256
    ):
        _raise("official #8 submission receipt bytes drifted")
    intent_path = Path(str(submission.get("intent", {}).get("path") or ""))
    attempt_path = Path(
        str(submission.get("attempt_ledger", {}).get("path") or "")
    )
    final_path = submission_path.parent / "final_seal.json"
    _verify_record(
        submission.get("plan"),
        "submission plan",
        expected_path=plan_path,
    )
    _verify_record(
        submission.get("intent"),
        "submission intent",
        expected_path=intent_path,
    )
    _verify_record(
        submission.get("attempt_ledger"),
        "submission attempt ledger",
        expected_path=attempt_path,
    )
    intent = _read_sealed(intent_path, official.INTENT_SCHEMA, "submit intent")
    attempt = _read_sealed(attempt_path, ATTEMPT_SCHEMA, "attempt ledger")
    final = _read_sealed(
        final_path,
        official.FINAL_SEAL_SCHEMA,
        "final submission seal",
    )
    expected_files = (
        (intent_path, INTENT_FILE_SHA256, intent, INTENT_PAYLOAD_SHA256),
        (attempt_path, ATTEMPT_FILE_SHA256, attempt, ATTEMPT_PAYLOAD_SHA256),
        (final_path, FINAL_FILE_SHA256, final, FINAL_PAYLOAD_SHA256),
    )
    for path, file_sha, value, payload_sha in expected_files:
        if (
            base.sha256_file(path) != file_sha
            or value.get("payload_sha256") != payload_sha
        ):
            _raise(f"durable submission artifact bytes drifted: {path.name}")
        _validate_safety(value, path.name)
    _validate_safety(submission, "submission receipt")
    for record, label, expected in (
        (intent.get("plan"), "intent plan", plan_path),
        (attempt.get("plan"), "attempt plan", plan_path),
        (attempt.get("intent"), "attempt intent", intent_path),
        (final.get("plan"), "final plan", plan_path),
        (final.get("intent"), "final intent", intent_path),
        (final.get("attempt_ledger"), "final attempt", attempt_path),
        (final.get("receipt"), "final receipt", submission_path),
    ):
        _verify_record(record, label, expected_path=expected)
    single_attempt = plan["single_attempt_contract"]
    if (
        submission.get("plan_payload_sha256") != PLAN_PAYLOAD_SHA256
        or submission.get("scheduler_payload_sha256")
        != plan["scheduler_payload_sha256"]
        or submission.get("task_id") != TASK_ID
        or submission.get("task_name") != TASK_NAME
        or submission.get("dedupe_key") != DEDUPE_KEY
        or submission.get("account_name") != EXPECTED_ACCOUNT
        or submission.get("node_name") != EXPECTED_NODE
        or submission.get("campaign_mutation_lock_acquired") is not True
        or submission.get("pre_submit_get_repeated_inside_lock") is not True
        or submission.get("scheduler_post_calls") != 1
        or submission.get("scheduler_submission_performed") is not True
        or submission.get("scheduler_post_http_status") != 201
        or submission.get("scheduler_post_error") is not None
        or submission.get("scheduler_repository_modified") is not False
        or submission.get("scheduler_project_mutation_performed") is not False
        or intent.get("plan_payload_sha256") != PLAN_PAYLOAD_SHA256
        or intent.get("scheduler_payload_sha256")
        != plan["scheduler_payload_sha256"]
        or intent.get("task_name") != TASK_NAME
        or intent.get("dedupe_key") != DEDUPE_KEY
        or intent.get("authorization") != official.POST_AUTHORIZATION
        or intent.get("attempt_nonce") != single_attempt["attempt_nonce"]
        or intent.get("scheduler_post_calls_before") != 0
        or intent.get("post_authorized") is not True
        or attempt.get("plan_payload_sha256") != PLAN_PAYLOAD_SHA256
        or attempt.get("intent_payload_sha256") != INTENT_PAYLOAD_SHA256
        or attempt.get("attempt_nonce") != single_attempt["attempt_nonce"]
        or attempt.get("campaign_mutation_lock_acquired") is not True
        or attempt.get("post_call_budget") != 1
        or attempt.get("post_call_consumed_before_network") is not True
        or attempt.get("scheduler_post_calls_before") != 0
        or attempt.get("scheduler_post_calls_authorized") != 1
        or final.get("plan_payload_sha256") != PLAN_PAYLOAD_SHA256
        or final.get("intent_payload_sha256") != INTENT_PAYLOAD_SHA256
        or final.get("attempt_ledger_payload_sha256")
        != ATTEMPT_PAYLOAD_SHA256
        or final.get("receipt_payload_sha256")
        != SUBMISSION_PAYLOAD_SHA256
        or final.get("task_id") != TASK_ID
        or final.get("scheduler_post_calls") != 1
        or final.get("immutable_evidence_complete") is not True
    ):
        _raise("official #8 single-POST/final-seal evidence drifted")
    for value, label in (
        (submission.get("initial_live_preflight"), "initial preflight"),
        (
            submission.get("locked_pre_submit_live_preflight"),
            "locked preflight",
        ),
        (intent.get("initial_live_preflight"), "intent preflight"),
        (
            attempt.get("locked_pre_submit_live_preflight"),
            "attempt preflight",
        ),
    ):
        _validate_opening_preflight(value, label)
    for task, label in (
        (submission.get("scheduler_post_response"), "POST response"),
        (submission.get("task_readback"), "submission readback"),
    ):
        _validate_task_snapshot(
            task,
            label=label,
            require_bound_placement=False,
        )
    if submission.get("task_readback_sha256") != base.payload_sha256(
        submission["task_readback"]
    ):
        _raise("official #8 task readback seal drifted")
    return copy.deepcopy(submission)


def load_contract(
    *,
    plan_path: Path = DEFAULT_PLAN,
    submission_path: Path = DEFAULT_SUBMISSION,
    scheduler_url: str = SCHEDULER_URL,
    plan_loader: PlanLoader = official.load_plan,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    submission_path = submission_path.resolve(strict=True)
    if base.sha256_file(plan_path) != PLAN_FILE_SHA256:
        _raise("official #8 plan file SHA drifted")
    try:
        plan, _params, profile = plan_loader(plan_path)
    except (OSError, ValueError, official.PostdeadlineContractError) as exc:
        raise Official8CollectionError(
            "official #8 plan reauthentication failed"
        ) from exc
    authenticated = _validate_plan(
        plan,
        profile,
        scheduler_url=scheduler_url,
    )
    submission = _validate_submission(
        plan=plan,
        plan_path=plan_path,
        submission_path=submission_path,
    )
    retained = authenticated["retained"]
    return {
        "plan": copy.deepcopy(plan),
        "submission": submission,
        "plan_path": plan_path,
        "submission_path": submission_path,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "submission_file_sha256": SUBMISSION_FILE_SHA256,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "dedupe_key": DEDUPE_KEY,
        "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
        "solver_revision": official.SOLVER_REVISION,
        "library_revision": official.LIBRARY_REVISION,
        "profile_sha256": authenticated["profile_sha256"],
        "parameter_digest": authenticated["parameter_digest"],
        "scheduler_url": scheduler_url.rstrip("/"),
        "node_name": EXPECTED_NODE,
        "retained": {
            "root": authenticated["retained_root"],
            "artifact_path": retained["artifact_path"],
            "receipt_path": retained["receipt_path"],
            "marker_path": retained["marker_path"],
            "chunk_directory": retained["transport"]["chunk_directory"],
            "results_path": retained["results_path"],
            "results_manifest_path": retained["results_manifest_path"],
            "marker_contract": authenticated["marker_contract"],
            "marker_contract_sha256": retained["marker_contract_sha256"],
        },
    }


def get_task(
    contract: Mapping[str, Any],
    *,
    getter: JsonGetter = base.http_get,
) -> dict[str, Any]:
    raw = getter(
        f"{contract['scheduler_url']}/api/tasks/{contract['task_id']}",
        max_bytes=1024 * 1024,
        timeout=30.0,
    )
    try:
        task = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Official8CollectionError(
            "Scheduler task GET is invalid JSON"
        ) from exc
    return _validate_task_snapshot(task, label="live Scheduler GET")


def poll_once(
    *,
    contract: Mapping[str, Any],
    output: Path,
    sequence: int = 0,
    getter: JsonGetter = base.http_get,
) -> tuple[bool, dict[str, Any]]:
    task = get_task(contract, getter=getter)
    poll = base._poll_record(contract, task)  # noqa: SLF001
    poll_path = base._record_poll(output, poll, sequence=sequence)  # noqa: SLF001
    state = str(task.get("state") or "")
    status = str(task.get("status") or "")
    if (status, state) in base.TERMINAL_SUCCESS:
        result = base.collect_success(
            contract=contract,
            task=task,
            output=output,
            getter=getter,
        )
        result["poll_path"] = str(poll_path)
        return True, result
    if (
        state in base.TERMINAL_FAILURE_STATES
        or status in base.TERMINAL_FAILURE_STATES
    ):
        result = base.write_failure_ledger(
            contract=contract,
            task=task,
            output=output,
        )
        result["poll_path"] = str(poll_path)
        return True, result
    if state not in base.ACTIVE_STATES and status not in base.ACTIVE_STATES:
        _raise(
            f"unsupported Scheduler transition: "
            f"status={status!r}, state={state!r}"
        )
    return False, {
        "event": "active",
        "task_id": TASK_ID,
        "state": state,
        "status": status,
        "actual_node_name": task.get("actual_node_name"),
        "allocation_id": task.get("allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "preferred_node_relaxed": task.get("preferred_node_relaxed"),
        "strict_requested_node": EXPECTED_NODE,
        "poll_path": str(poll_path),
        "scheduler_get_only": True,
        "scheduler_mutation_performed": False,
        "scientific_pass_claimed": False,
        "scientific_infeasible_claimed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 10 <= args.interval <= 3_600:
        _raise("watch interval must be between 10 and 3600 seconds")
    contract = load_contract(
        plan_path=args.plan,
        submission_path=args.submission,
    )
    sequence = 0
    while True:
        terminal, event = poll_once(
            contract=contract,
            output=args.output,
            sequence=sequence,
        )
        print(json.dumps(event, sort_keys=True), flush=True)
        if terminal or args.once:
            return 0
        sequence += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Official8CollectionError, base.CollectionError) as exc:
        print(
            json.dumps(
                {
                    "event": "official8_collector_error",
                    "error": str(exc),
                    "scheduler_get_only": True,
                    "scheduler_mutation_performed": False,
                    "scientific_pass_claimed": False,
                    "scientific_infeasible_claimed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from exc
