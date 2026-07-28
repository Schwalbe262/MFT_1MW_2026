"""GET-only collector for the three official post-deadline hedge lanes.

The one-shot submitter has a new, stricter evidence topology than the older
post-deadline Standard lanes.  This collector authenticates that complete
topology, derives immutable compatibility authorities for the existing
Standard artifact collector, and then performs Scheduler GET requests only.

No function in this module submits, cancels, reprioritizes, or restarts
anything.  A successful terminal task is collected through the reviewed
Standard AEDT/AEDT-results transport validator.  Each lane output therefore
remains directly consumable by
``tools.mft_goal_postdeadline_standard_postsuccess``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import (  # noqa: E402
    mft_goal_official_standard_hedge_prepare as prepare_only,
)
from tools import (  # noqa: E402
    mft_goal_official_standard_hedge_submit as submitter,
)
from tools import (  # noqa: E402
    mft_goal_postdeadline_standard_collector as standard,
)


CollectionError = standard.CollectionError
Getter = Callable[..., bytes]
PrepareAuthenticator = Callable[
    [Path],
    tuple[dict[str, Any], list[dict[str, Any]]],
]
RuntimeAuthenticator = Callable[[], dict[str, Any]]

SUBMISSION_ROOT = submitter.OUTPUT_ROOT
AGGREGATE_SUBMIT_MANIFEST = SUBMISSION_ROOT / submitter.AGGREGATE_SUBMIT_MANIFEST_NAME
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official_hedge3_collection_260726_v1"
)

LINEAGE_SCHEMA = "mft-goal-official-standard-hedge-collection-lineage-v1"
SELECTED_ADAPTER_SCHEMA = "mft-goal-official-standard-hedge-selected-adapter-v1"
STATE_SCHEMA = "mft-goal-official-standard-hedge3-collection-state-v1"
AUTHORITY_DIRECTORY = ".authenticated_authority"
STATE_NAME = "state.json"

LIVE_ALLOCATION_STATES = {
    "pending",
    "warm",
    "active",
    "draining",
    "closing",
}

PARAMETER_COMMAND = re.compile(
    r"printf '%s' '(\{.*?\})' > cand\.json",
)


def _sealed(
    value: Mapping[str, Any],
    schema: str,
    label: str,
) -> dict[str, Any]:
    try:
        result = submitter.validate_seal(dict(value), schema)
    except (OSError, prepare_only.PostdeadlineContractError) as exc:
        raise CollectionError(f"{label} seal drifted") from exc
    if not isinstance(result, dict):
        raise CollectionError(f"{label} must be an object")
    return result


def _read_sealed(path: Path, schema: str, label: str) -> dict[str, Any]:
    try:
        value = submitter.read_json(path.resolve(strict=True))
    except (OSError, prepare_only.PostdeadlineContractError) as exc:
        raise CollectionError(f"{label} is unavailable") from exc
    return _sealed(value, schema, label)


def _safety(value: Mapping[str, Any], label: str) -> None:
    drift = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in prepare_only.SAFETY_FLAGS.items()
        if value.get(key) is not expected
    }
    if drift:
        raise CollectionError(f"{label} safety classification drifted")


def _record_path(
    record: Any,
    label: str,
    *,
    contained_by: Path | None = None,
    expected: Path | None = None,
) -> Path:
    try:
        path = standard._verify_file_record(record, label)  # noqa: SLF001
    except (OSError, standard.CollectionError) as exc:
        raise CollectionError(f"{label} file record drifted") from exc
    if contained_by is not None:
        try:
            path.relative_to(contained_by.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise CollectionError(f"{label} escapes its authenticated root") from exc
    if expected is not None:
        try:
            expected_path = expected.resolve(strict=True)
        except OSError as exc:
            raise CollectionError(f"{label} expected file is absent") from exc
        if path != expected_path:
            raise CollectionError(f"{label} path identity drifted")
    return path


def _same_record(record: Any, path: Path, label: str) -> None:
    _record_path(record, label, expected=path)


def _task_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CollectionError(f"{label} is invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CollectionError(f"{label} is invalid") from exc
    if parsed <= 0:
        raise CollectionError(f"{label} is invalid")
    return parsed


def _validate_runtime_authority(
    value: Any,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise CollectionError(f"{label} runtime authority is absent")
    authority = _sealed(
        value,
        "mft-goal-scheduler-runtime-walltime-authority-v1",
        f"{label} runtime authority",
    )
    if authority != expected:
        raise CollectionError(f"{label} runtime authority drifted")
    try:
        submitter._validate_historical_launcher_record(
            authority.get("launcher")
        )
    except prepare_only.PostdeadlineContractError as exc:
        raise CollectionError(
            f"{label} runtime authority launcher drifted"
        ) from exc
    for key in (
        "live_config",
        "deployed_config_py",
        "deployed_scheduler_py",
        "cutover_receipt",
    ):
        _record_path(
            authority.get(key),
            f"{label} runtime authority {key}",
        )
    if (
        authority.get("allocation_walltime_seconds") != 172_800
        or authority.get("allocation_drain_after_seconds") != 129_600
        or authority.get("fresh_pool_attach_window_seconds") != 127_800
        or authority.get("ordinary_cpu_pending_timeout_seconds") != 1_800
        or authority.get("ordinary_cpu_pending_timeout_exempt") is not False
        or authority.get("minimum_required_residual_seconds") != 50_400
        or authority.get("nominal_fresh_pool_attach_window_exceeds_minimum") is not True
        or authority.get("hard_residual_guarantee") is not False
        or authority.get("hard_residual_seconds_api_proven") is not False
        or authority.get("operational_source_inference") is not True
    ):
        raise CollectionError(f"{label} runtime lifecycle drifted")


def _validate_live_gate(
    value: Any,
    lane: Mapping[str, Any],
    runtime_authority: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectionError(f"{label} is absent")
    gate = _sealed(value, submitter.LIVE_GATE_SCHEMA, label)
    _safety(gate, label)
    spec = lane["spec"]
    plan = lane["plan"]
    expected = {
        "selection_order": spec.selection_order,
        "task_name": spec.task_name,
        "dedupe_key": plan["dedupe_key"],
        "account_name": spec.account_name,
        "node_name": spec.node_name,
        "node_name_policy": "strict",
        "scheduler_get_only": True,
        "scheduler_post_calls": 0,
    }
    if any(gate.get(key) != expected_value for key, expected_value in expected.items()):
        raise CollectionError(f"{label} identity drifted")
    preflight = gate.get("strict_opening_preflight")
    node = gate.get("node_gate")
    walltime = gate.get("allocation_walltime_contract")
    capacity = preflight.get("capacity") if isinstance(preflight, Mapping) else None
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("queue_state") != "opening"
        or preflight.get("ready_fit_slots") != 0
        or preflight.get("pending_fit_slots") != 0
        or preflight.get("inflight_fit_slots") != 0
        or preflight.get("preferred_node_relaxed") is not False
        or preflight.get("active_target_fea_count") != 0
        or preflight.get("collision_count") != 0
        or not isinstance(capacity, Mapping)
        or capacity.get("allocations") != []
        or not isinstance(node, Mapping)
        or node.get("node_name") != spec.node_name
        or node.get("exact_account_node_live_allocation_count") != 0
        or node.get("exact_account_node_live_allocation_must_be_absent") is not True
        or not isinstance(walltime, Mapping)
        or walltime.get("minimum_seconds") != 50_400
        or walltime.get("task_timeout_seconds") != 45_300
        or walltime.get("fresh_strict_opening_requires_new_pool") is not True
        or walltime.get("capacity_allocations_empty") is not True
        or walltime.get("api_residual_seconds_proven") is not False
        or walltime.get("source_contract_derived_not_api_residual") is not True
    ):
        raise CollectionError(f"{label} strict fresh-opening gate drifted")
    _validate_runtime_authority(
        walltime.get("runtime_source_authority"),
        runtime_authority,
        label,
    )
    return gate


def _validate_task_identity(
    task: Mapping[str, Any],
    lane: Mapping[str, Any],
    task_id: int,
    label: str,
    *,
    receipt_readback: bool,
) -> None:
    spec = lane["spec"]
    plan = lane["plan"]
    expected = {
        "name": spec.task_name,
        "dedupe_key": plan["dedupe_key"],
        "project": submitter.PROJECT,
        "requested_account_name": spec.account_name,
        "requested_node_name": spec.node_name,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": submitter.CPUS,
        "memory_mb": submitter.MEMORY_MB,
        "timeout_seconds": submitter.SCHEDULER_SECONDS,
        "max_workers_per_node": submitter.MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
    }
    actual_id = task.get("task_id", task.get("id"))
    if actual_id != task_id or any(
        task.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise CollectionError(f"{label} Scheduler identity drifted")
    if (
        task.get("preferred_node_relaxed") is not False
        or task.get("node_name_policy") not in {None, "", "strict"}
        or (
            task.get("status") in {"attaching", "running", "completed"}
            and task.get("placement_contract_satisfied") is False
        )
    ):
        raise CollectionError(f"{label} strict placement drifted")
    for key in (
        "account_name",
        "requested_account_name",
    ):
        actual = str(task.get(key) or "")
        if actual and actual != spec.account_name:
            raise CollectionError(f"{label} account placement drifted")
    for key in (
        "node_name",
        "requested_node_name",
        "actual_node_name",
        "allocation_node_name",
    ):
        actual = str(task.get(key) or "")
        if actual and actual != spec.node_name:
            raise CollectionError(f"{label} node placement drifted")
    if receipt_readback and task.get("status") not in {
        "queued",
        "attaching",
        "running",
    }:
        raise CollectionError(f"{label} creation readback state drifted")


def _lane_artifact(
    lane_root: Path,
    record: Any,
    filename: str,
    schema: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = _record_path(
        record,
        label,
        contained_by=lane_root,
        expected=lane_root / filename,
    )
    return path, _read_sealed(path, schema, label)


def _authenticate_lane_submission(
    *,
    aggregate_lane: Mapping[str, Any],
    prepared_lane: Mapping[str, Any],
    aggregate_path: Path,
    batch_intent_path: Path,
    runtime_authority: Mapping[str, Any],
) -> dict[str, Any]:
    spec = prepared_lane["spec"]
    plan = prepared_lane["plan"]
    submission_root = aggregate_path.parent
    lane_root = submission_root / spec.lane_name
    if not lane_root.is_dir():
        raise CollectionError(
            f"official{spec.selection_order} submission directory is absent"
        )
    task_id = _task_id(
        aggregate_lane.get("task_id"),
        f"official{spec.selection_order} task ID",
    )
    expected = {
        "selection_order": spec.selection_order,
        "candidate_physics_sha256": spec.candidate_sha256,
        "lane_nonce": prepared_lane["lane_nonce"],
        "task_name": spec.task_name,
        "dedupe_key": plan["dedupe_key"],
        "account_name": spec.account_name,
        "node_name": spec.node_name,
        "node_name_policy": "strict",
    }
    if any(
        aggregate_lane.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise CollectionError(
            f"official{spec.selection_order} aggregate identity drifted"
        )
    _same_record(
        aggregate_lane.get("prepare_plan"),
        prepared_lane["plan_path"],
        f"official{spec.selection_order} prepare plan",
    )
    intent_path, intent = _lane_artifact(
        lane_root,
        aggregate_lane.get("lane_intent"),
        submitter.LANE_INTENT_NAME,
        submitter.LANE_INTENT_SCHEMA,
        f"official{spec.selection_order} lane intent",
    )
    attempt_path, attempt = _lane_artifact(
        lane_root,
        aggregate_lane.get("attempt_ledger"),
        submitter.ATTEMPT_NAME,
        submitter.ATTEMPT_SCHEMA,
        f"official{spec.selection_order} attempt",
    )
    receipt_path, receipt = _lane_artifact(
        lane_root,
        aggregate_lane.get("submission_receipt"),
        submitter.RECEIPT_NAME,
        submitter.RECEIPT_SCHEMA,
        f"official{spec.selection_order} submission receipt",
    )
    final_path, final = _lane_artifact(
        lane_root,
        aggregate_lane.get("lane_final_seal"),
        submitter.LANE_FINAL_NAME,
        submitter.LANE_FINAL_SCHEMA,
        f"official{spec.selection_order} final seal",
    )
    for label, value in (
        ("lane intent", intent),
        ("attempt", attempt),
        ("submission receipt", receipt),
        ("lane final", final),
    ):
        _safety(value, f"official{spec.selection_order} {label}")

    _same_record(
        intent.get("batch_intent"),
        batch_intent_path,
        "lane intent batch authority",
    )
    _same_record(
        intent.get("prepare_plan"),
        prepared_lane["plan_path"],
        "lane intent prepare plan",
    )
    initial_gate = _validate_live_gate(
        intent.get("initial_live_gate"),
        prepared_lane,
        runtime_authority,
        f"official{spec.selection_order} initial live gate",
    )
    if (
        intent.get("selection_order") != spec.selection_order
        or intent.get("candidate_physics_sha256") != spec.candidate_sha256
        or intent.get("lane_nonce") != prepared_lane["lane_nonce"]
        or intent.get("prepare_plan_payload_sha256") != plan["payload_sha256"]
        or intent.get("scheduler_payload_sha256") != plan["scheduler_payload_sha256"]
        or intent.get("task_name") != spec.task_name
        or intent.get("dedupe_key") != plan["dedupe_key"]
        or intent.get("account_name") != spec.account_name
        or intent.get("node_name") != spec.node_name
        or intent.get("node_name_policy") != "strict"
        or intent.get("post_call_budget") != 1
        or intent.get("attempt_ledger_must_precede_network") is not True
        or intent.get("retry_after_ambiguity_allowed") is not False
    ):
        raise CollectionError(f"official{spec.selection_order} lane intent drifted")

    _same_record(
        attempt.get("batch_intent"),
        batch_intent_path,
        "attempt batch authority",
    )
    _same_record(
        attempt.get("lane_intent"),
        intent_path,
        "attempt lane intent",
    )
    _same_record(
        attempt.get("prepare_plan"),
        prepared_lane["plan_path"],
        "attempt prepare plan",
    )
    locked_gate = _validate_live_gate(
        attempt.get("locked_pre_submit_live_gate"),
        prepared_lane,
        runtime_authority,
        f"official{spec.selection_order} locked live gate",
    )
    if (
        attempt.get("selection_order") != spec.selection_order
        or attempt.get("lane_nonce") != prepared_lane["lane_nonce"]
        or attempt.get("prepare_plan_payload_sha256") != plan["payload_sha256"]
        or attempt.get("scheduler_payload_sha256") != plan["scheduler_payload_sha256"]
        or attempt.get("initial_live_gate") != initial_gate
        or attempt.get("campaign_mutation_lock_acquired") is not True
        or attempt.get("post_call_budget") != 1
        or attempt.get("post_call_consumed_before_network") is not True
        or attempt.get("scheduler_post_calls_before") != 0
        or attempt.get("scheduler_post_calls_authorized") != 1
        or attempt.get("ambiguous_retry_allowed") is not False
    ):
        raise CollectionError(f"official{spec.selection_order} attempt ledger drifted")

    _same_record(
        receipt.get("plan"),
        prepared_lane["plan_path"],
        "submission receipt prepare plan",
    )
    _same_record(
        receipt.get("attempt_ledger"),
        attempt_path,
        "submission receipt attempt",
    )
    readback = receipt.get("task_readback")
    if not isinstance(readback, Mapping):
        raise CollectionError("submission receipt readback is absent")
    _validate_task_identity(
        readback,
        prepared_lane,
        task_id,
        f"official{spec.selection_order} submission readback",
        receipt_readback=True,
    )
    if (
        receipt.get("selection_order") != spec.selection_order
        or receipt.get("candidate_physics_sha256") != spec.candidate_sha256
        or receipt.get("lane_nonce") != prepared_lane["lane_nonce"]
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("scheduler_payload_sha256") != plan["scheduler_payload_sha256"]
        or receipt.get("scheduler_post_calls") != 1
        or receipt.get("task_id") != task_id
        or receipt.get("task_name") != spec.task_name
        or receipt.get("dedupe_key") != plan["dedupe_key"]
        or receipt.get("account_name") != spec.account_name
        or receipt.get("node_name") != spec.node_name
        or receipt.get("node_name_policy") != "strict"
        or receipt.get("preferred_node_relaxed") is not False
        or receipt.get("task_readback_sha256") != standard.payload_sha256(readback)
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("scheduler_project_mutation_performed") is not False
    ):
        raise CollectionError(
            f"official{spec.selection_order} submission receipt drifted"
        )

    _same_record(
        final.get("attempt_ledger"),
        attempt_path,
        "lane final attempt",
    )
    _same_record(
        final.get("receipt"),
        receipt_path,
        "lane final receipt",
    )
    if (
        final.get("selection_order") != spec.selection_order
        or final.get("lane_nonce") != prepared_lane["lane_nonce"]
        or final.get("receipt_payload_sha256") != receipt["payload_sha256"]
        or final.get("task_id") != task_id
        or final.get("scheduler_post_calls") != 1
        or final.get("immutable_evidence_complete") is not True
    ):
        raise CollectionError(f"official{spec.selection_order} lane final drifted")
    return {
        "prepared_lane": prepared_lane,
        "task_id": task_id,
        "lane_root": lane_root,
        "lane_intent_path": intent_path,
        "attempt_path": attempt_path,
        "receipt_path": receipt_path,
        "final_path": final_path,
        "intent": intent,
        "attempt": attempt,
        "receipt": receipt,
        "final": final,
        "initial_gate": initial_gate,
        "locked_gate": locked_gate,
    }


def _authenticate_batch_intent(
    aggregate: Mapping[str, Any],
    aggregate_path: Path,
    prepare_path: Path,
    prepare_manifest: Mapping[str, Any],
    prepared_lanes: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = _record_path(
        aggregate.get("batch_intent"),
        "aggregate batch intent",
        contained_by=aggregate_path.parent,
        expected=aggregate_path.parent / submitter.BATCH_INTENT_NAME,
    )
    intent = _read_sealed(path, submitter.BATCH_INTENT_SCHEMA, "batch intent")
    _safety(intent, "batch intent")
    _same_record(
        intent.get("prepare_manifest"),
        prepare_path,
        "batch intent prepare manifest",
    )
    expected_nonces = [
        {
            "selection_order": lane["spec"].selection_order,
            "lane_name": lane["spec"].lane_name,
            "lane_nonce": lane["lane_nonce"],
            "task_name": lane["spec"].task_name,
            "dedupe_key": lane["plan"]["dedupe_key"],
            "account_name": lane["spec"].account_name,
            "node_name": lane["spec"].node_name,
            "post_call_budget": 1,
        }
        for lane in prepared_lanes
    ]
    if (
        intent.get("prepare_manifest_payload_sha256")
        != prepare_manifest["payload_sha256"]
        or intent.get("authorization_contract") != submitter.AUTHORIZATION
        or intent.get("run_command_argv") != submitter.RUN_COMMAND_ARGV
        or intent.get("lane_count") != len(prepared_lanes)
        or intent.get("lane_nonces") != expected_nonces
        or intent.get("total_post_call_budget") != len(prepared_lanes)
        or intent.get("attempt_ledger_consumed_before_each_network_call") is not True
        or intent.get("ambiguous_post_retry_allowed") is not False
        or intent.get("node_relaxation_allowed") is not False
        or intent.get("minimum_allocation_walltime_seconds") != 50_400
        or intent.get("reviewed_allocation_walltime_seconds") != 172_800
        or intent.get("walltime_evidence_classification")
        != "fresh_opening_runtime_source_contract"
        or intent.get("api_residual_seconds_proven") is not False
    ):
        raise CollectionError("batch submission intent drifted")
    return path, intent


def authenticate_submission_package(
    *,
    aggregate_path: Path = AGGREGATE_SUBMIT_MANIFEST,
    require_fixed_path: bool = True,
    prepare_authenticator: PrepareAuthenticator = (
        submitter.authenticate_prepare_package
    ),
    runtime_authenticator: RuntimeAuthenticator = (
        submitter.authenticate_runtime_walltime_contract
    ),
) -> dict[str, Any]:
    """Authenticate the prepare, submit, and exact three-lane evidence graph."""

    resolved_aggregate = aggregate_path.resolve(strict=True)
    if require_fixed_path:
        try:
            fixed = AGGREGATE_SUBMIT_MANIFEST.resolve(strict=True)
        except OSError as exc:
            raise CollectionError("fixed aggregate submit manifest is absent") from exc
        if resolved_aggregate != fixed:
            raise CollectionError("aggregate submit manifest path is not fixed")
    aggregate = _read_sealed(
        resolved_aggregate,
        submitter.AGGREGATE_SUBMIT_MANIFEST_SCHEMA,
        "aggregate submit manifest",
    )
    _safety(aggregate, "aggregate submit manifest")
    prepare_path = _record_path(
        aggregate.get("prepare_manifest"),
        "aggregate prepare manifest",
    )
    try:
        prepare_manifest, prepared_lanes = prepare_authenticator(prepare_path)
    except (OSError, prepare_only.PostdeadlineContractError) as exc:
        raise CollectionError("prepare package reauthentication failed") from exc
    if len(prepared_lanes) != len(prepare_only.CANDIDATES):
        raise CollectionError("prepare package lane count drifted")
    if (
        aggregate.get("prepare_manifest_payload_sha256")
        != prepare_manifest["payload_sha256"]
    ):
        raise CollectionError("aggregate prepare manifest binding drifted")
    batch_intent_path, batch_intent = _authenticate_batch_intent(
        aggregate,
        resolved_aggregate,
        prepare_path,
        prepare_manifest,
        prepared_lanes,
    )
    lanes = aggregate.get("lanes")
    expected_orders = [lane["spec"].selection_order for lane in prepared_lanes]
    expected_names = [lane["spec"].task_name for lane in prepared_lanes]
    expected_nodes = [lane["spec"].node_name for lane in prepared_lanes]
    if (
        not isinstance(lanes, list)
        or len(lanes) != len(prepared_lanes)
        or aggregate.get("lane_count") != len(prepared_lanes)
        or aggregate.get("selection_orders") != expected_orders
        or aggregate.get("task_names") != expected_names
        or aggregate.get("strict_nodes") != expected_nodes
        or aggregate.get("scheduler_post_call_budget") != len(prepared_lanes)
        or aggregate.get("scheduler_post_calls_evidenced") != len(prepared_lanes)
        or aggregate.get("maximum_one_post_per_lane") is not True
        or aggregate.get("node_relaxation_allowed") is not False
        or aggregate.get("get_only_collector_ready") is not True
    ):
        raise CollectionError("aggregate submission topology drifted")
    task_ids = [_task_id(item, "aggregate task ID") for item in aggregate["task_ids"]]
    if len(task_ids) != len(set(task_ids)):
        raise CollectionError("aggregate task IDs are duplicated")
    runtime_authority = runtime_authenticator()
    authenticated_lanes = [
        _authenticate_lane_submission(
            aggregate_lane=aggregate_lane,
            prepared_lane=prepared_lane,
            aggregate_path=resolved_aggregate,
            batch_intent_path=batch_intent_path,
            runtime_authority=runtime_authority,
        )
        for aggregate_lane, prepared_lane in zip(lanes, prepared_lanes, strict=True)
    ]
    if [lane["task_id"] for lane in authenticated_lanes] != task_ids:
        raise CollectionError("aggregate lane/task ordering drifted")
    return {
        "aggregate": aggregate,
        "aggregate_path": resolved_aggregate,
        "prepare_manifest": prepare_manifest,
        "prepare_manifest_path": prepare_path,
        "batch_intent": batch_intent,
        "batch_intent_path": batch_intent_path,
        "runtime_authority": runtime_authority,
        "lanes": authenticated_lanes,
    }


def _plan_relative_record(
    plan_path: Path,
    record: Any,
    label: str,
) -> Path:
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise CollectionError(f"{label} record is malformed")
    candidate = plan_path.parent / record["path"]
    absolute_record = dict(record)
    absolute_record["path"] = str(candidate)
    return _record_path(
        absolute_record,
        label,
        expected=candidate,
    )


def _compatibility_authority(
    package: Mapping[str, Any],
    lane: Mapping[str, Any],
    authority_root: Path,
) -> dict[str, Any]:
    prepared = lane["prepared_lane"]
    spec = prepared["spec"]
    plan = prepared["plan"]
    lane_root = authority_root / spec.lane_name
    lane_root.mkdir(parents=True, exist_ok=True)

    params_path = _plan_relative_record(
        prepared["plan_path"],
        plan.get("fea_params"),
        "prepared FEA params",
    )
    selected_path = _plan_relative_record(
        prepared["plan_path"],
        plan.get("selected_candidate"),
        "prepared selected candidate",
    )
    profile_path = _plan_relative_record(
        prepared["plan_path"],
        plan.get("execution_profile"),
        "prepared execution profile",
    )
    params = standard._read_json(params_path, "prepared FEA params")  # noqa: SLF001
    selected = standard._read_json(  # noqa: SLF001
        selected_path, "prepared selected candidate"
    )
    profile = standard._read_json(  # noqa: SLF001
        profile_path, "prepared execution profile"
    )
    original_row = selected.get("official_row")
    decoded = selected.get("decoded_physical_params")
    if (
        not isinstance(original_row, Mapping)
        or not isinstance(decoded, Mapping)
        or original_row.get("candidate_physics_sha") != spec.candidate_sha256
        or plan.get("raw_fea_params_sha256") != standard.payload_sha256(params)
        or profile.get("fixed_boundary_contract") != standard.FIXED_BOUNDARY
    ):
        raise CollectionError("prepared candidate adapter inputs drifted")

    command = str(plan.get("scheduler_payload", {}).get("command") or "")
    matches = PARAMETER_COMMAND.findall(command)
    if len(matches) != 1:
        raise CollectionError("prepared Scheduler command parameter payload drifted")
    try:
        submitted_params = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise CollectionError(
            "prepared Scheduler command parameter payload is malformed"
        ) from exc
    effective_params = dict(params)
    overrides = profile.get("param_overrides")
    if not isinstance(submitted_params, dict) or not isinstance(overrides, Mapping):
        raise CollectionError("prepared parameter override contract drifted")
    effective_params.update(overrides)
    if submitted_params != effective_params:
        raise CollectionError(
            "prepared Scheduler command differs from reviewed parameters"
        )
    parameter_json = json.dumps(
        submitted_params,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    parameter_digest = standard.sha256_bytes(parameter_json.encode("utf-8"))[:16]
    if parameter_digest != str(plan["dedupe_key"]).rsplit(":", 1)[-1]:
        raise CollectionError(
            "prepared Scheduler command/dedupe parameter identity drifted"
        )
    adapter_params_path = lane_root / "fea_params.json"
    standard._write_atomic(  # noqa: SLF001
        adapter_params_path,
        parameter_json.encode("utf-8") + b"\n",
    )

    lineage_path = lane_root / "hedge_lineage.json"
    lineage = standard.seal(
        {
            "schema_version": LINEAGE_SCHEMA,
            **prepare_only.SAFETY_FLAGS,
            "selection_order": spec.selection_order,
            "candidate_physics_sha256": spec.candidate_sha256,
            "task_id": lane["task_id"],
            "task_name": spec.task_name,
            "dedupe_key": plan["dedupe_key"],
            "account_name": spec.account_name,
            "node_name": spec.node_name,
            "node_name_policy": "strict",
            "aggregate_submit_manifest": standard.file_record(
                package["aggregate_path"]
            ),
            "aggregate_submit_manifest_payload_sha256": package["aggregate"][
                "payload_sha256"
            ],
            "prepare_manifest": standard.file_record(package["prepare_manifest_path"]),
            "prepare_manifest_payload_sha256": package["prepare_manifest"][
                "payload_sha256"
            ],
            "prepare_plan": standard.file_record(prepared["plan_path"]),
            "prepare_plan_payload_sha256": plan["payload_sha256"],
            "batch_intent": standard.file_record(package["batch_intent_path"]),
            "lane_intent": standard.file_record(lane["lane_intent_path"]),
            "attempt_ledger": standard.file_record(lane["attempt_path"]),
            "submission_receipt": standard.file_record(lane["receipt_path"]),
            "lane_final_seal": standard.file_record(lane["final_path"]),
            "scheduler_post_calls_authenticated": 1,
            "collector_scheduler_get_only": True,
            "collector_scheduler_mutation_performed": False,
        }
    )
    standard._write_json(lineage_path, lineage)  # noqa: SLF001

    selected_adapter_path = lane_root / "selected_candidate.json"
    selected_adapter = standard.seal(
        {
            "schema_version": SELECTED_ADAPTER_SCHEMA,
            **standard.CLASSIFICATION,
            "noncanonical": True,
            "automatic_promotion": False,
            "scientific_pass_claimed": False,
            "selected_row": copy.deepcopy(dict(original_row)),
            "row_contract": {
                "candidate_physics_sha256": spec.candidate_sha256,
                "fea_params_sha256": standard.payload_sha256(submitted_params),
                "canonical_physical_params_sha256": plan[
                    "canonical_physical_params_sha256"
                ],
                "canonical_official_row_json_sha256": plan[
                    "canonical_official_row_json_sha256"
                ],
            },
            "decoded_physical_params": copy.deepcopy(dict(decoded)),
            "source_selected_candidate": standard.file_record(selected_path),
            "source_selected_candidate_payload_sha256": selected["payload_sha256"],
            "hedge_lineage": standard.file_record(lineage_path),
        }
    )
    standard._write_json(  # noqa: SLF001
        selected_adapter_path,
        selected_adapter,
    )

    resources = plan.get("resources")
    payload = plan.get("scheduler_payload")
    if (
        not isinstance(resources, Mapping)
        or not isinstance(payload, Mapping)
        or payload.get("node_name_policy") != "strict"
    ):
        raise CollectionError("prepared Scheduler payload drifted")
    source_artifacts = {
        "fea_params.json": standard.file_record(adapter_params_path),
        "selected_candidate.json": standard.file_record(selected_adapter_path),
        "hedge_lineage.json": standard.file_record(lineage_path),
    }
    adapter_plan_path = lane_root / "diagnostic_postdeadline_plan.json"
    adapter_plan = standard.seal(
        {
            "schema_version": standard.PLAN_SCHEMA,
            **standard.CLASSIFICATION,
            "noncanonical": True,
            "automatic_promotion": False,
            "scientific_pass_claimed": False,
            "fixed_physics_unchanged": True,
            "candidate_physics_sha256": spec.candidate_sha256,
            "task_name": spec.task_name,
            "dedupe_key": plan["dedupe_key"],
            "scheduler_url": plan["scheduler_url"],
            "scheduler_project": plan["scheduler_project"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "scheduler_payload": copy.deepcopy(dict(payload)),
            "scheduler_payload_sha256": plan["scheduler_payload_sha256"],
            "resources": {
                "cpus": resources["cpus"],
                "memory_mb": resources["memory_mb"],
                "node_name": spec.node_name,
                "node_name_policy": "strict",
                "same_node_as_task_id": 0,
                "dependency_task_id": 0,
            },
            "timeout_envelope": {
                "solver_seconds": resources["solver_seconds"],
                "kill_grace_seconds": resources["kill_grace_seconds"],
                "retention_seconds": resources["retention_seconds"],
                "scheduler_timeout_seconds": resources["scheduler_timeout_seconds"],
            },
            "fixed_boundary": copy.deepcopy(standard.FIXED_BOUNDARY),
            "preflight": {
                "fixed_boundary": copy.deepcopy(standard.FIXED_BOUNDARY),
                "candidate_boundary_projection": copy.deepcopy(
                    standard.CANDIDATE_BOUNDARY
                ),
                "candidate_physics_sha256": spec.candidate_sha256,
                "candidate_reauthenticated": True,
                "fixed_physics_unchanged": True,
                "revision_reauthenticated": True,
                "strict_account_name": spec.account_name,
                "strict_node_name": spec.node_name,
                "node_relaxation_allowed": False,
                "source_hedge_lineage_payload_sha256": lineage["payload_sha256"],
            },
            "source_artifacts": source_artifacts,
            "execution_profile": standard.file_record(profile_path),
            "execution_profile_canonical_sha256": plan[
                "execution_profile_canonical_sha256"
            ],
            "hedge_lineage": standard.file_record(lineage_path),
        }
    )
    standard._write_json(adapter_plan_path, adapter_plan)  # noqa: SLF001

    adapter_submission_path = lane_root / "submission_receipt.json"
    adapter_submission = standard.seal(
        {
            "schema_version": standard.SUBMISSION_SCHEMA,
            **standard.CLASSIFICATION,
            "noncanonical": True,
            "automatic_promotion": False,
            "scientific_pass_claimed": False,
            "fixed_physics_unchanged": True,
            "candidate_physics_sha256": spec.candidate_sha256,
            "task_id": lane["task_id"],
            "task_name": spec.task_name,
            "dedupe_key": plan["dedupe_key"],
            "account_name": spec.account_name,
            "node_name": spec.node_name,
            "node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "plan": standard.file_record(adapter_plan_path),
            "plan_payload_sha256": adapter_plan["payload_sha256"],
            "scheduler_payload_sha256": plan["scheduler_payload_sha256"],
            "hedge_submission_receipt": standard.file_record(lane["receipt_path"]),
            "hedge_submission_receipt_payload_sha256": lane["receipt"][
                "payload_sha256"
            ],
            "hedge_lineage": standard.file_record(lineage_path),
            "scheduler_post_calls_authenticated": 1,
            "collector_scheduler_get_only": True,
            "collector_scheduler_mutation_performed": False,
        }
    )
    standard._write_json(  # noqa: SLF001
        adapter_submission_path,
        adapter_submission,
    )
    contract = standard.load_contract(
        plan_path=adapter_plan_path,
        submission_path=adapter_submission_path,
        expected_task_id=lane["task_id"],
        expected_task_name=spec.task_name,
        expected_dedupe_key=plan["dedupe_key"],
        expected_candidate_sha256=spec.candidate_sha256,
        expected_receipt_sha256=standard.sha256_file(adapter_submission_path),
        scheduler_url=plan["scheduler_url"],
    )
    contract["hedge_lineage_path"] = lineage_path
    contract["hedge_selection_order"] = spec.selection_order
    contract["strict_account_name"] = spec.account_name
    contract["strict_node_name"] = spec.node_name
    return contract


def build_collection_contracts(
    package: Mapping[str, Any],
    *,
    output_root: Path = OUTPUT_ROOT,
) -> list[dict[str, Any]]:
    authority_root = output_root.resolve() / AUTHORITY_DIRECTORY
    authority_root.mkdir(parents=True, exist_ok=True)
    contracts = [
        _compatibility_authority(package, lane, authority_root)
        for lane in package["lanes"]
    ]
    if len({contract["task_id"] for contract in contracts}) != len(contracts):
        raise CollectionError("collection task identities are duplicated")
    return contracts


def get_task(
    contract: Mapping[str, Any],
    *,
    getter: Getter = standard.http_get,
) -> dict[str, Any]:
    task = standard.get_task(contract, getter=getter)
    prepared_lane = {
        "spec": type(
            "StrictIdentity",
            (),
            {
                "task_name": contract["task_name"],
                "candidate_sha256": contract["candidate_physics_sha256"],
                "account_name": contract["strict_account_name"],
                "node_name": contract["strict_node_name"],
            },
        )(),
        "plan": {"dedupe_key": contract["dedupe_key"]},
    }
    _validate_task_identity(
        task,
        prepared_lane,
        contract["task_id"],
        f"task {contract['task_id']} live GET",
        receipt_readback=False,
    )
    return task


def _failure_path(output: Path) -> Path:
    return output.resolve().parent / f"{output.name}.failure_ledger.json"


def poll_lane_once(
    *,
    contract: Mapping[str, Any],
    output: Path,
    sequence: int,
    getter: Getter = standard.http_get,
) -> tuple[bool, dict[str, Any]]:
    task = get_task(contract, getter=getter)
    poll = standard._poll_record(contract, task)  # noqa: SLF001
    poll_path = standard._record_poll(  # noqa: SLF001
        output,
        poll,
        sequence=sequence,
    )
    state = str(task.get("state") or "")
    status = str(task.get("status") or "")
    if (status, state) in standard.TERMINAL_SUCCESS:
        if task.get("exit_code") != 0:
            raise CollectionError(
                f"task {contract['task_id']} succeeded without exit code 0"
            )
        result = standard.collect_success(
            contract=contract,
            task=task,
            output=output,
            getter=getter,
        )
        result["poll_path"] = str(poll_path)
        return True, result
    if (
        state in standard.TERMINAL_FAILURE_STATES
        or status in standard.TERMINAL_FAILURE_STATES
    ):
        existing_path = _failure_path(output)
        if existing_path.is_file():
            ledger = standard._read_json(  # noqa: SLF001
                existing_path,
                "existing failure ledger",
            )
            standard._validate_seal(  # noqa: SLF001
                ledger,
                standard.FAILURE_SCHEMA,
                "existing failure ledger",
            )
            if ledger.get("task_id") != contract["task_id"]:
                raise CollectionError("existing failure ledger drifted")
            result = {
                "event": "existing_failure_ledger",
                "task_id": contract["task_id"],
                "path": str(existing_path),
                "failure_ledger_payload_sha256": ledger["payload_sha256"],
            }
        else:
            result = standard.write_failure_ledger(
                contract=contract,
                task=task,
                output=output,
            )
        result["poll_path"] = str(poll_path)
        return True, result
    if state not in standard.ACTIVE_STATES and status not in standard.ACTIVE_STATES:
        raise CollectionError(
            f"unsupported Scheduler task transition: status={status!r}, state={state!r}"
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
    }


def lane_output_path(
    output_root: Path,
    contract: Mapping[str, Any],
) -> Path:
    return output_root.resolve() / (
        f"official{contract['hedge_selection_order']}_task{contract['task_id']}"
    )


def _write_state(
    output_root: Path,
    package: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    terminal: Sequence[bool],
) -> Path:
    lanes = []
    for contract, event, is_terminal in zip(
        contracts,
        events,
        terminal,
        strict=True,
    ):
        collection_root = lane_output_path(output_root, contract)
        if event.get("event") in {"collected", "already_collected"}:
            lane_status = "collected"
        elif event.get("event") in {
            "failure_ledger",
            "existing_failure_ledger",
        }:
            lane_status = "terminal_failure"
        else:
            lane_status = "pending"
        lanes.append(
            {
                "selection_order": contract["hedge_selection_order"],
                "task_id": contract["task_id"],
                "task_name": contract["task_name"],
                "candidate_physics_sha256": contract["candidate_physics_sha256"],
                "account_name": contract["strict_account_name"],
                "node_name": contract["strict_node_name"],
                "collection_root": str(collection_root),
                "failure_ledger": str(_failure_path(collection_root)),
                "status": lane_status,
                "terminal": is_terminal,
                "event": copy.deepcopy(dict(event)),
            }
        )
    state = standard.seal(
        {
            "schema_version": STATE_SCHEMA,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            **prepare_only.SAFETY_FLAGS,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
            "aggregate_submit_manifest": standard.file_record(
                package["aggregate_path"]
            ),
            "aggregate_submit_manifest_payload_sha256": package["aggregate"][
                "payload_sha256"
            ],
            "lane_count": len(lanes),
            "terminal_count": sum(terminal),
            "pending_count": sum(not value for value in terminal),
            "collected_count": sum(lane["status"] == "collected" for lane in lanes),
            "failed_count": sum(lane["status"] == "terminal_failure" for lane in lanes),
            "all_terminal": all(terminal),
            "postsuccess_lane_inputs": [
                f"{lane['task_id']}={lane['collection_root']}" for lane in lanes
            ],
            "lanes": lanes,
        }
    )
    path = output_root.resolve() / STATE_NAME
    standard._write_json(path, state, replace=True)  # noqa: SLF001
    return path


def poll_all_once(
    *,
    package: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
    output_root: Path = OUTPUT_ROOT,
    sequence: int = 0,
    getter: Getter = standard.http_get,
) -> tuple[bool, dict[str, Any]]:
    events: list[dict[str, Any]] = []
    terminal: list[bool] = []
    for contract in contracts:
        finished, event = poll_lane_once(
            contract=contract,
            output=lane_output_path(output_root, contract),
            sequence=sequence,
            getter=getter,
        )
        terminal.append(finished)
        events.append(event)
    state_path = _write_state(
        output_root,
        package,
        contracts,
        events,
        terminal,
    )
    state = standard._read_json(state_path, "collection state")  # noqa: SLF001
    standard._validate_seal(  # noqa: SLF001
        state,
        STATE_SCHEMA,
        "collection state",
    )
    return all(terminal), state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-submit-manifest",
        type=Path,
        default=AGGREGATE_SUBMIT_MANIFEST,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 10 <= args.interval <= 3_600:
        raise CollectionError("watch interval must be between 10 and 3600 seconds")
    package = authenticate_submission_package(
        aggregate_path=args.aggregate_submit_manifest,
        require_fixed_path=True,
    )
    contracts = build_collection_contracts(
        package,
        output_root=args.output_root,
    )
    sequence = 0
    while True:
        terminal, state = poll_all_once(
            package=package,
            contracts=contracts,
            output_root=args.output_root,
            sequence=sequence,
        )
        print(json.dumps(state, sort_keys=True), flush=True)
        if terminal or args.once:
            return 0
        sequence += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as exc:
        print(
            json.dumps(
                {
                    "event": "hedge_collector_error",
                    "error": str(exc),
                    "scheduler_get_only": True,
                    "scheduler_post_calls": 0,
                    "scheduler_mutation_performed": False,
                    "scientific_pass_claimed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from exc
