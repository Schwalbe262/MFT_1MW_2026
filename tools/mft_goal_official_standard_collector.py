"""GET-only watcher/collector for official Standard candidate #6 task 96328.

The submission was produced by :mod:`mft_goal_official_standard_postdeadline`,
whose sealed plan and receipt intentionally differ from the older custom
post-deadline Standard schema.  This adapter reauthenticates that official
plan, every durable submission artifact, the fixed candidate/physics identity,
the exact Scheduler task and placement, and the retained AEDT bundle before it
delegates bounded artifact reconstruction to the existing GET-only collector.

There is deliberately no Scheduler mutation path in this module.  A terminal
solver result is collected as an observation only and never promoted to a
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

from tools import mft_goal_official_standard_postdeadline as official  # noqa: E402
from tools import mft_goal_postdeadline_standard_collector as base  # noqa: E402


TASK_ID = 96_328
TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official6-"
    "s96185-2772aed82a8c-n113"
)
DEDUPE_KEY = (
    "mft-al:mft-goal-diag-standard-postdeadline-official6-"
    "s96185-2772aed82a8c-n113:"
    "a1e4f70cefa1af04673c73a6131bf490c0cc14b5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "b713a9235ad804af"
)
CANDIDATE_PHYSICS_SHA256 = (
    "2772aed82a8c2c5d7e861b7b864e24d11d3ccfb601dfb1b0535b5f1e1ceb9a9e"
)
PLAN_FILE_SHA256 = (
    "24465ef093ef0b4ebf05fd9d0d585a235d66a4a0183eecd7a95f4800b0d29e98"
)
PLAN_PAYLOAD_SHA256 = (
    "49fb840f8e39235cfe139685416e7181099edb7d043bcd0dad0b9afa317c7526"
)
SUBMISSION_FILE_SHA256 = (
    "42942394a873e40181f9074f2625807224239b291bcf2f4cf2ccacde50b11edc"
)
SUBMISSION_PAYLOAD_SHA256 = (
    "847c7604316ccd03a52afbc7c75db1d832f0df333bbd84b40c9e8ba35fec9f99"
)
EXPECTED_NODE = "n113"
EXPECTED_ACCOUNT = "dw16"
EXPECTED_ALLOCATION_ID = 14_620
EXPECTED_SLURM_JOB_ID = "829579"
SCHEDULER_URL = "http://127.0.0.1:8002"
RUNTIME_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official6_2772aed82a8c_n113_260726_v2"
)
DEFAULT_PLAN = RUNTIME_ROOT / "diagnostic_postdeadline_plan.json"
DEFAULT_SUBMISSION = RUNTIME_ROOT / "submission" / "submission_receipt.json"
DEFAULT_OUTPUT = (
    RUNTIME_ROOT / "authenticated_get_collection_task96328_v1"
)

ATTEMPT_SCHEMA = (
    "mft-goal-official-standard-postdeadline-scheduler-post-attempt-v1"
)
SINGLE_ATTEMPT_SCHEMA = (
    "mft-goal-official-standard-postdeadline-single-attempt-contract-v1"
)
LIVE_PREFLIGHT_SCHEMA = "mft-goal-official-standard-live-preflight-v1"

JsonGetter = Callable[..., bytes]
PlanLoader = Callable[
    [Path],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]


class OfficialCollectionError(base.CollectionError):
    """The fixed official #6 evidence or GET-only observation drifted."""


def _raise(message: str) -> None:
    raise OfficialCollectionError(message)


def _validate_safety(value: Mapping[str, Any], label: str) -> None:
    if any(value.get(key) is not expected for key, expected in official.SAFETY_FLAGS.items()):
        _raise(f"{label} safety classification drifted")


def _read_sealed(path: Path, schema: str, label: str) -> dict[str, Any]:
    value = base._read_json(path, label)  # noqa: SLF001
    try:
        base._validate_seal(value, schema, label)  # noqa: SLF001
    except base.CollectionError as exc:
        raise OfficialCollectionError(str(exc)) from exc
    return value


def _verify_record(
    record: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    try:
        path = base._verify_file_record(record, label)  # noqa: SLF001
    except base.CollectionError as exc:
        raise OfficialCollectionError(str(exc)) from exc
    if expected_path is not None and path != expected_path.resolve(strict=True):
        _raise(f"{label} does not bind the exact expected file")
    return path


def _validate_preflight(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        _raise(f"{label} is malformed")
    allocation = value.get("selected_allocation")
    if (
        value.get("schema_version") != LIVE_PREFLIGHT_SCHEMA
        or value.get("collision_count") != 0
        or value.get("active_fea_on_target_count") != 0
        or value.get("exact_account_node_ready") is not True
        or value.get("fixed_physics_unchanged") is not True
        or value.get("selected_allocation_id") != EXPECTED_ALLOCATION_ID
        or str(value.get("selected_slurm_job_id") or "")
        != EXPECTED_SLURM_JOB_ID
        or not isinstance(allocation, Mapping)
        or allocation.get("account_name") != EXPECTED_ACCOUNT
        or allocation.get("node_name") != EXPECTED_NODE
        or int(allocation.get("id") or 0) != EXPECTED_ALLOCATION_ID
        or str(allocation.get("slurm_job_id") or "")
        != EXPECTED_SLURM_JOB_ID
    ):
        _raise(f"{label} target-lane identity drifted")


def _validate_plan(
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_task_name: str,
    expected_dedupe_key: str,
    scheduler_url: str,
) -> dict[str, Any]:
    _validate_safety(plan, "official plan")
    payload = plan.get("scheduler_payload")
    resources = plan.get("resources")
    placement = plan.get("placement")
    retained = plan.get("retained_aedt_bundle")
    single_attempt = plan.get("single_attempt_contract")
    transport = retained.get("transport") if isinstance(retained, Mapping) else None
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, Mapping):
        _raise("official execution profile overrides are malformed")
    # The Scheduler identity was derived from the freshly authenticated
    # in-memory parameter insertion order.  The persisted params file is
    # canonical-key-sorted, so recomputing from that file would create a
    # different digest despite identical physics.  ``official.load_plan``
    # already regenerates and byte-compares the Scheduler payload; bind the
    # retained digest to that reauthenticated payload and dedupe suffix.
    parameter_digest = (
        str(retained.get("parameter_digest") or "")
        if isinstance(retained, Mapping)
        else ""
    )
    task_identity = base.sha256_bytes(expected_dedupe_key.encode("utf-8"))[:16]
    retained_root = f"goal-fea-retained/{task_identity}"
    expected_marker = {
        "schema": base.MARKER_SCHEMA,
        "preserve": True,
        "reason": (
            "Retain diagnostic Standard AEDT and AEDT results until "
            "authenticated diagnostic collection; stage=standard; "
            f"dedupe_key={expected_dedupe_key}"
        ),
        "owner": base.SCHEDULER_PROJECT,
    }
    if (
        plan.get("schema_version") != official.PLAN_SCHEMA
        or plan.get("payload_sha256") != base.payload_sha256(
            {key: copy.deepcopy(value) for key, value in plan.items() if key != "payload_sha256"}
        )
        or plan.get("candidate_physics_sha256") != expected_candidate_sha256
        or plan.get("task_name") != expected_task_name
        or plan.get("dedupe_key") != expected_dedupe_key
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
        or placement.get("same_node_as_task_id") != 0
        or placement.get("dependency_task_id") != 0
        or placement.get("allocation_id_at_prepare") != EXPECTED_ALLOCATION_ID
        or str(placement.get("slurm_job_id_at_prepare") or "")
        != EXPECTED_SLURM_JOB_ID
        or not isinstance(payload, Mapping)
        or plan.get("scheduler_payload_sha256") != base.payload_sha256(payload)
        or payload.get("name") != expected_task_name
        or payload.get("dedupe_key") != expected_dedupe_key
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
        or str(payload.get("command") or "").count(
            "timeout --signal=TERM --kill-after=300s 43200s "
            "python run_simulation_260706.py"
        )
        != 1
        or "sleep $((RANDOM % 300))" in str(payload.get("command") or "")
        or not isinstance(retained, Mapping)
        or plan.get("retained_aedt_bundle_sha256")
        != base.payload_sha256(retained)
        or retained.get("schema_version")
        != "mft-goal-diagnostic-retained-aedt-bundle-v1"
        or retained.get("stage") != "standard"
        or retained.get("dedupe_key") != expected_dedupe_key
        or len(parameter_digest) != 16
        or any(character not in "0123456789abcdef" for character in parameter_digest)
        or not expected_dedupe_key.endswith(f":{parameter_digest}")
        or retained.get("solver_revision") != official.SOLVER_REVISION
        or retained.get("library_revision") != official.LIBRARY_REVISION
        or retained.get("profile_sha256") != base.payload_sha256(profile)
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
        or retained.get("results_manifest_schema_version")
        != base.RESULTS_MANIFEST_SCHEMA
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
        or retained.get("marker_contract") != expected_marker
        or retained.get("marker_contract_sha256")
        != base.payload_sha256(expected_marker)
        or not isinstance(transport, Mapping)
        or transport.get("schema_version") != base.TRANSPORT_SCHEMA
        or transport.get("encoding") != "base64"
        or transport.get("raw_chunk_bytes") != base.RAW_CHUNK_BYTES
        or transport.get("max_encoded_chunk_bytes")
        != base.MAX_ENCODED_CHUNK_BYTES
        or transport.get("chunk_directory")
        != f"{retained_root}/symmetric.aedt.chunks"
        or not isinstance(single_attempt, Mapping)
        or single_attempt.get("schema_version") != SINGLE_ATTEMPT_SCHEMA
        or single_attempt.get("post_call_budget") != 1
        or single_attempt.get("ledger_consumed_before_network") is not True
        or single_attempt.get("output_override_allowed") is not False
    ):
        _raise("official plan fixed identity/retention contract drifted")
    _validate_preflight(plan.get("preflight"), "official plan preflight")
    return {
        "parameter_digest": parameter_digest,
        "profile_sha256": base.payload_sha256(profile),
        "retained": copy.deepcopy(dict(retained)),
        "single_attempt": copy.deepcopy(dict(single_attempt)),
        "marker_contract": expected_marker,
        "retained_root": retained_root,
    }


def _validate_submission_artifacts(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    submission: Mapping[str, Any],
    submission_path: Path,
    expected_task_id: int,
    expected_task_name: str,
    expected_dedupe_key: str,
) -> None:
    _validate_safety(submission, "official submission")
    plan_record_path = _verify_record(
        submission.get("plan"),
        "official submission plan",
        expected_path=plan_path,
    )
    intent_path = _verify_record(
        submission.get("intent"), "official submission intent"
    )
    attempt_path = _verify_record(
        submission.get("attempt_ledger"), "official attempt ledger"
    )
    intent = _read_sealed(intent_path, official.INTENT_SCHEMA, "official intent")
    attempt = _read_sealed(attempt_path, ATTEMPT_SCHEMA, "official attempt ledger")
    final_path = submission_path.parent / "final_seal.json"
    final_seal = _read_sealed(
        final_path, official.FINAL_SEAL_SCHEMA, "official final seal"
    )
    _validate_safety(intent, "official intent")
    _validate_safety(attempt, "official attempt ledger")
    _validate_safety(final_seal, "official final seal")
    _verify_record(intent.get("plan"), "intent plan", expected_path=plan_record_path)
    _verify_record(attempt.get("plan"), "attempt plan", expected_path=plan_record_path)
    _verify_record(attempt.get("intent"), "attempt intent", expected_path=intent_path)
    _verify_record(final_seal.get("plan"), "final-seal plan", expected_path=plan_path)
    _verify_record(final_seal.get("intent"), "final-seal intent", expected_path=intent_path)
    _verify_record(
        final_seal.get("attempt_ledger"),
        "final-seal attempt ledger",
        expected_path=attempt_path,
    )
    _verify_record(
        final_seal.get("receipt"),
        "final-seal receipt",
        expected_path=submission_path,
    )
    single_attempt = plan["single_attempt_contract"]
    expected_attempt = Path(single_attempt["attempt_ledger_path"]).resolve(strict=True)
    expected_submission_dir = Path(
        single_attempt["submission_output_path"]
    ).resolve(strict=True)
    if (
        plan_record_path != plan_path
        or expected_attempt != attempt_path
        or expected_submission_dir != submission_path.parent
        or submission.get("schema_version") != official.SUBMISSION_SCHEMA
        or submission.get("payload_sha256")
        != base.payload_sha256(
            {
                key: copy.deepcopy(value)
                for key, value in submission.items()
                if key != "payload_sha256"
            }
        )
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
        or submission.get("scheduler_payload_sha256")
        != plan["scheduler_payload_sha256"]
        or submission.get("task_id") != expected_task_id
        or submission.get("task_name") != expected_task_name
        or submission.get("dedupe_key") != expected_dedupe_key
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
        or intent.get("plan_payload_sha256") != plan["payload_sha256"]
        or intent.get("scheduler_payload_sha256")
        != plan["scheduler_payload_sha256"]
        or intent.get("task_name") != expected_task_name
        or intent.get("dedupe_key") != expected_dedupe_key
        or intent.get("authorization") != official.POST_AUTHORIZATION
        or intent.get("attempt_ledger_path") != str(expected_attempt)
        or intent.get("attempt_nonce") != single_attempt["attempt_nonce"]
        or intent.get("scheduler_post_calls_before") != 0
        or intent.get("post_authorized") is not True
        or attempt.get("plan_payload_sha256") != plan["payload_sha256"]
        or attempt.get("intent_payload_sha256") != intent["payload_sha256"]
        or attempt.get("attempt_nonce") != single_attempt["attempt_nonce"]
        or Path(str(attempt.get("submission_output_path") or "")).resolve()
        != expected_submission_dir
        or attempt.get("campaign_mutation_lock_acquired") is not True
        or attempt.get("post_call_budget") != 1
        or attempt.get("post_call_consumed_before_network") is not True
        or attempt.get("scheduler_post_calls_before") != 0
        or attempt.get("scheduler_post_calls_authorized") != 1
        or final_seal.get("plan_payload_sha256") != plan["payload_sha256"]
        or final_seal.get("intent_payload_sha256") != intent["payload_sha256"]
        or final_seal.get("attempt_ledger_payload_sha256")
        != attempt["payload_sha256"]
        or final_seal.get("receipt_payload_sha256")
        != submission["payload_sha256"]
        or final_seal.get("task_id") != expected_task_id
        or final_seal.get("scheduler_post_calls") != 1
        or final_seal.get("immutable_evidence_complete") is not True
    ):
        _raise("official durable submission evidence drifted")
    for key, expected in (
        ("intent_payload_sha256", intent["payload_sha256"]),
        ("attempt_ledger_payload_sha256", attempt["payload_sha256"]),
    ):
        if submission.get(key) != expected:
            _raise(f"official submission {key} drifted")
    _validate_preflight(
        submission.get("initial_live_preflight"), "initial live preflight"
    )
    _validate_preflight(
        submission.get("locked_pre_submit_live_preflight"),
        "locked live preflight",
    )
    _validate_preflight(
        intent.get("initial_live_preflight"), "intent initial preflight"
    )
    _validate_preflight(
        attempt.get("locked_pre_submit_live_preflight"),
        "attempt locked preflight",
    )
    for label, task in (
        ("submission POST response", submission.get("scheduler_post_response")),
        ("submission readback", submission.get("task_readback")),
    ):
        _validate_task_snapshot(
            task,
            label=label,
            expected_task_id=expected_task_id,
            expected_task_name=expected_task_name,
            expected_dedupe_key=expected_dedupe_key,
            require_bound_placement=False,
        )
    if submission.get("task_readback_sha256") != base.payload_sha256(
        submission["task_readback"]
    ):
        _raise("official task readback seal drifted")


def _validate_task_snapshot(
    task: Any,
    *,
    label: str,
    expected_task_id: int,
    expected_task_name: str,
    expected_dedupe_key: str,
    require_bound_placement: bool,
) -> dict[str, Any]:
    if not isinstance(task, dict):
        _raise(f"{label} is not an object")
    task_id = task.get("task_id", task.get("id"))
    expected = {
        "name": expected_task_name,
        "dedupe_key": expected_dedupe_key,
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
    if task_id != expected_task_id or any(task.get(key) != value for key, value in expected.items()):
        _raise(f"{label} exact task identity drifted")
    if require_bound_placement:
        if (
            task.get("actual_node_name") != EXPECTED_NODE
            or int(task.get("allocation_id") or task.get("assigned_allocation") or 0)
            != EXPECTED_ALLOCATION_ID
            or str(task.get("slurm_job_id") or "") != EXPECTED_SLURM_JOB_ID
            or task.get("account_name") != EXPECTED_ACCOUNT
        ):
            _raise(f"{label} bound allocation/Slurm identity drifted")
    return copy.deepcopy(task)


def load_contract(
    *,
    plan_path: Path,
    submission_path: Path,
    expected_plan_sha256: str = PLAN_FILE_SHA256,
    expected_receipt_sha256: str = SUBMISSION_FILE_SHA256,
    expected_plan_payload_sha256: str = PLAN_PAYLOAD_SHA256,
    expected_submission_payload_sha256: str = SUBMISSION_PAYLOAD_SHA256,
    expected_candidate_sha256: str = CANDIDATE_PHYSICS_SHA256,
    expected_task_id: int = TASK_ID,
    expected_task_name: str = TASK_NAME,
    expected_dedupe_key: str = DEDUPE_KEY,
    scheduler_url: str = SCHEDULER_URL,
    plan_loader: PlanLoader = official.load_plan,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    submission_path = submission_path.resolve(strict=True)
    if (
        base.sha256_file(plan_path) != expected_plan_sha256
        or base.sha256_file(submission_path) != expected_receipt_sha256
    ):
        _raise("official plan or submission receipt file SHA drifted")
    try:
        plan, params, profile = plan_loader(plan_path)
    except (OSError, ValueError, official.PostdeadlineContractError) as exc:
        raise OfficialCollectionError(
            "official plan reauthentication failed"
        ) from exc
    if (
        plan.get("payload_sha256") != expected_plan_payload_sha256
        or expected_candidate_sha256 != CANDIDATE_PHYSICS_SHA256
        or expected_task_id != TASK_ID
        or expected_task_name != TASK_NAME
        or expected_dedupe_key != DEDUPE_KEY
    ):
        _raise("fixed official #6 source identity drifted")
    authenticated = _validate_plan(
        plan,
        params,
        profile,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_task_name=expected_task_name,
        expected_dedupe_key=expected_dedupe_key,
        scheduler_url=scheduler_url,
    )
    submission = _read_sealed(
        submission_path,
        official.SUBMISSION_SCHEMA,
        "official submission receipt",
    )
    if submission.get("payload_sha256") != expected_submission_payload_sha256:
        _raise("official submission receipt payload seal drifted")
    _validate_submission_artifacts(
        plan=plan,
        plan_path=plan_path,
        submission=submission,
        submission_path=submission_path,
        expected_task_id=expected_task_id,
        expected_task_name=expected_task_name,
        expected_dedupe_key=expected_dedupe_key,
    )
    retained = authenticated["retained"]
    return {
        "plan": copy.deepcopy(plan),
        "submission": copy.deepcopy(submission),
        "plan_path": plan_path,
        "submission_path": submission_path,
        "plan_file_sha256": expected_plan_sha256,
        "submission_file_sha256": expected_receipt_sha256,
        "task_id": expected_task_id,
        "task_name": expected_task_name,
        "dedupe_key": expected_dedupe_key,
        "candidate_physics_sha256": expected_candidate_sha256,
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
        raise OfficialCollectionError("Scheduler task GET is invalid JSON") from exc
    state = str(task.get("state") or "") if isinstance(task, Mapping) else ""
    status = str(task.get("status") or "") if isinstance(task, Mapping) else ""
    require_bound = (
        state in {"running", "succeeded", "completed", "failed", "cancelled"}
        or status in {"running", "succeeded", "completed", "failed", "cancelled"}
    )
    return _validate_task_snapshot(
        task,
        label="live Scheduler GET",
        expected_task_id=contract["task_id"],
        expected_task_name=contract["task_name"],
        expected_dedupe_key=contract["dedupe_key"],
        require_bound_placement=require_bound,
    )


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
    if state in base.TERMINAL_FAILURE_STATES or status in base.TERMINAL_FAILURE_STATES:
        result = base.write_failure_ledger(
            contract=contract,
            task=task,
            output=output,
        )
        result["poll_path"] = str(poll_path)
        return True, result
    if state not in base.ACTIVE_STATES and status not in base.ACTIVE_STATES:
        _raise(
            f"unsupported Scheduler transition: status={status!r}, state={state!r}"
        )
    return False, {
        "event": "active",
        "task_id": contract["task_id"],
        "state": state,
        "status": status,
        "actual_node_name": task.get("actual_node_name"),
        "allocation_id": task.get("allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "poll_path": str(poll_path),
        "scheduler_get_only": True,
        "scheduler_mutation_performed": False,
        "scientific_pass_claimed": False,
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
    except (OfficialCollectionError, base.CollectionError) as exc:
        print(
            json.dumps(
                {
                    "event": "official_collector_error",
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
