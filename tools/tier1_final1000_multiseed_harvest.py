"""Incrementally flatten mixed single-seed and finite-batch final1000 lanes.

The Scheduler parent remains the only physical task identity.  Batch children
are scientific seed records, never fabricated Scheduler tasks.  A running
batch may be harvested only through its atomic journal and immutable child
receipts; undeclared result files are never read or trusted.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

try:
    import tier1_final1000_multiseed_phase_b_contract as phase_b_contract
    from tier1_corrected_current7_slurm_harvest import (
        MAX_RESULT_BYTES,
        MAX_STATUS_BYTES,
        RemoteReader,
        _fail_closed,
        _harvest_result_artifacts,
        _hex,
        _public_observation,
        _remote_child,
        deduplicate_observations,
        harvest_terminal_task,
        read_stable_remote_json,
    )
    from tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tier1_final1000_multiseed_contract import (
        BATCH_PAYLOAD_SCHEMA,
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_batch_task,
        validate_child_receipt,
        validate_task_status,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_final1000_multiseed_phase_b_contract as phase_b_contract
    from tools.tier1_corrected_current7_slurm_harvest import (
        MAX_RESULT_BYTES,
        MAX_STATUS_BYTES,
        RemoteReader,
        _fail_closed,
        _harvest_result_artifacts,
        _hex,
        _public_observation,
        _remote_child,
        deduplicate_observations,
        harvest_terminal_task,
        read_stable_remote_json,
    )
    from tools.tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tools.tier1_final1000_multiseed_contract import (
        BATCH_PAYLOAD_SCHEMA,
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_batch_task,
        validate_child_receipt,
        validate_task_status,
    )


MAX_BATCH_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TASK_STATUS_BYTES = 512 * 1024
TERMINAL_SCHEDULER_STATES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)
HARVESTABLE_BATCH_STATES = frozenset({"running"}) | TERMINAL_SCHEDULER_STATES


def _batch_parent_task(
    item: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = item.get("parent_task")
    if not isinstance(candidate, dict):
        raise RuntimeError("batch inventory item has no authenticated parent envelope")
    candidate_payload = candidate.get("payload_json")
    if (
        isinstance(candidate_payload, dict)
        and candidate_payload.get("schema_version")
        == phase_b_contract.BATCH_PAYLOAD_SCHEMA
    ):
        task = phase_b_contract.validate_runtime_batch_task(
            candidate,
            submission_intent=item.get("phase_b_submission_intent"),
            authority=item.get("phase_b_authority_lease"),
            reconciliation_plan=item.get("phase_b_reconciliation_plan"),
        )
    else:
        task = validate_batch_task(candidate)
    payload = task["payload_json"]
    task_id = item.get("task_id")
    scheduler_state = str(item.get("status") or "").lower()
    observed_envelope = item.get("scheduler_task")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or scheduler_state not in HARVESTABLE_BATCH_STATES
        or item.get("dedupe_key") != task.get("dedupe_key")
        or item.get("name") != task.get("name")
        or not isinstance(observed_envelope, dict)
        or observed_envelope != task
        or payload.get("bundle_id") != plan.get("bundle_id")
        or payload.get("bundle_id") != manifest.get("bundle_id")
        or payload.get("bundle_manifest_sha256") != plan.get("bundle_manifest_sha256")
        or task.get("remote_cwd") != plan.get("remote_bundle")
    ):
        raise RuntimeError("batch parent/plan/bundle identity mismatch")
    return task


def _read_batch_journal(
    item: Mapping[str, Any],
    *,
    parent_task: Mapping[str, Any],
    remote: RemoteReader,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    account = str(item.get("account_name") or "")
    if not account:
        raise RuntimeError("batch Scheduler task has no account identity")
    task_id = int(item["task_id"])
    remote_root = str(parent_task["remote_cwd"])
    manifest_path = _remote_child(
        remote_root, "runs", f"task-{task_id}", "batch_manifest.json"
    )
    stable_manifest = read_stable_remote_json(
        remote,
        account_name=account,
        path=manifest_path,
        maximum_bytes=MAX_BATCH_MANIFEST_BYTES,
    )
    if stable_manifest.stat.mode & 0o222:
        raise RuntimeError("remote batch manifest is not immutable")
    is_phase_b = (
        parent_task["payload_json"].get("schema_version")
        == phase_b_contract.BATCH_PAYLOAD_SCHEMA
    )
    manifest = (
        phase_b_contract.validate_batch_manifest(stable_manifest.value)
        if is_phase_b
        else validate_batch_manifest(stable_manifest.value)
    )
    expected_manifest = (
        phase_b_contract.batch_manifest_from_payload(parent_task["payload_json"])
        if is_phase_b
        else batch_manifest_from_payload(parent_task["payload_json"])
    )
    if manifest != expected_manifest:
        raise RuntimeError("remote batch manifest differs from parent payload")
    status_path = _remote_child(
        remote_root, "runs", f"task-{task_id}", "task_status.json"
    )
    stable_status = read_stable_remote_json(
        remote,
        account_name=account,
        path=status_path,
        maximum_bytes=MAX_TASK_STATUS_BYTES,
    )
    status = (
        phase_b_contract.validate_task_status(stable_status.value)
        if is_phase_b
        else validate_task_status(stable_status.value)
    )
    if (
        str(status["task_id"]) != str(task_id)
        or status["manifest_sha256"] != manifest["manifest_sha256"]
        or int(status["sealed_child_count"]) > int(manifest["batch_length"])
        or int(status["completed_child_count"]) + int(status["failed_child_count"])
        > int(status["sealed_child_count"])
    ):
        raise RuntimeError("batch task journal/manifest binding mismatch")
    if status["state"] in {"completed", "completed_with_failures"} and int(
        status["sealed_child_count"]
    ) != int(manifest["batch_length"]):
        raise RuntimeError("completed batch journal is not a full receipt prefix")
    current_ordinal = status.get("current_ordinal")
    if current_ordinal is not None:
        children = manifest["ordered_children"]
        if int(current_ordinal) >= len(children) or int(
            children[int(current_ordinal)]["seed"]
        ) != int(status["current_seed"]):
            raise RuntimeError("batch task cursor/manifest binding mismatch")
    return manifest, status, remote_root


def harvest_batch_lane(
    item: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    remote: RemoteReader,
    result_validator: Callable[..., None] = validate_current7_result,
    journal_snapshot: tuple[
        Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str
    ]
    | None = None,
) -> list[dict[str, Any]]:
    """Harvest journal-declared children even while the parent is running."""

    if journal_snapshot is None:
        parent_task = _batch_parent_task(item, plan=plan, manifest=manifest)
        batch_manifest, task_status, remote_root = _read_batch_journal(
            item, parent_task=parent_task, remote=remote
        )
    else:
        parent_task, batch_manifest, task_status, remote_root = journal_snapshot
        expected_parent = _batch_parent_task(item, plan=plan, manifest=manifest)
        if dict(parent_task) != expected_parent:
            raise RuntimeError("preloaded batch journal parent identity mismatch")
    parent_payload = parent_task["payload_json"]
    account = str(item["account_name"])
    task_id = int(item["task_id"])
    scheduler_state = str(item.get("status") or "").lower()
    observations: list[dict[str, Any]] = []
    declared_sealed_count = int(task_status["sealed_child_count"])
    declared_completed = 0
    declared_failed = 0
    # The mutable task-status cursor is the only authority that exposes child
    # receipts to a consumer.  A receipt may already exist in the narrow
    # receipt-before-cursor crash window, but reading it would make a later
    # cursor rollback indistinguishable from an authenticated publication.
    # Consume exactly the durable prefix and let the next poll observe the
    # cursor advance. A terminal watcher may close that narrow crash window,
    # but only through the shared sealed complete-scan attestation. Missing or
    # invalid evidence never widens this prefix.
    exposed_receipt_count = declared_sealed_count
    recovery_attestation: dict[str, Any] | None = None
    raw_recovery = item.get("phase_b_terminal_receipt_recovery_attestation")
    if raw_recovery is not None:
        if (
            batch_manifest.get("schema_version")
            != phase_b_contract.BATCH_MANIFEST_SCHEMA
        ):
            raise RuntimeError("terminal receipt recovery is Phase-B-only")
        if not isinstance(raw_recovery, Mapping):
            raise RuntimeError("terminal receipt recovery attestation is invalid")
        recovery_attestation = (
            phase_b_contract.validate_terminal_receipt_recovery_attestation(
                raw_recovery,
                parent_task=parent_task,
                manifest=batch_manifest,
                task_status=task_status,
                task_id=task_id,
                scheduler_state=scheduler_state,
                expected_watcher_capability_sha256=str(
                    item.get("phase_b_watcher_capability_sha256") or ""
                ),
                expected_watcher_revision_sha256=str(
                    item.get("phase_b_watcher_revision_sha256") or ""
                ),
            )
        )
        exposed_receipt_count = int(
            recovery_attestation["recovered_receipt_count"]
        )
    for ordinal in range(exposed_receipt_count):
        child = parent_payload["children"][ordinal]
        seed = int(child["seed"])
        receipt_path = _remote_child(
            remote_root,
            "runs",
            f"task-{task_id}",
            f"seed-{seed}",
            "seed_status.json",
        )
        stable_receipt = read_stable_remote_json(
            remote,
            account_name=account,
            path=receipt_path,
            maximum_bytes=MAX_STATUS_BYTES,
        )
        if stable_receipt.stat.mode & 0o222:
            raise RuntimeError("remote batch child receipt is not immutable")
        receipt = (
            phase_b_contract.validate_child_receipt(
                stable_receipt.value, manifest=batch_manifest
            )
            if batch_manifest.get("schema_version")
            == phase_b_contract.BATCH_MANIFEST_SCHEMA
            else validate_child_receipt(stable_receipt.value, manifest=batch_manifest)
        )
        if recovery_attestation is not None:
            recovery_record = recovery_attestation["receipt_scan"][ordinal]
            if (
                recovery_record.get("present") is not True
                or recovery_record.get("remote_path") != receipt_path
                or recovery_record.get("remote_receipt_sha256")
                != stable_receipt.sha256
                or recovery_record.get("remote_receipt_size")
                != stable_receipt.size
                or recovery_record.get("remote_mode") != stable_receipt.stat.mode
                or recovery_record.get("receipt") != receipt
            ):
                raise RuntimeError(
                    "terminal recovery attestation differs from remote receipt bytes"
                )
        legacy_status = receipt["legacy_status"]
        if (
            str(receipt["task_id"]) != str(task_id)
            or int(receipt["ordinal"]) != ordinal
            or int(receipt["seed"]) != seed
            or legacy_status.get("seed") != seed
            or legacy_status.get("payload_sha256") != child["payload_sha256"]
        ):
            raise RuntimeError("batch child receipt/legacy status identity mismatch")
        observation: dict[str, Any] = {
            "bundle_id": parent_payload["bundle_id"],
            "seed": seed,
            "island_id": child["task"]["payload_json"]["lane"]["island_id"],
            "task_id": task_id,
            "physical_parent_task_id": task_id,
            "batch_ordinal": ordinal,
            "batch_manifest_sha256": batch_manifest["manifest_sha256"],
            "scheduler_state": scheduler_state,
            "scheduler_exit_code": item.get("exit_code"),
            "account_name": account,
            "terminal_state": receipt["state"],
            "child_wall_time_seconds": float(receipt["wall_time_seconds"]),
            "status_object": {
                "sha256": stable_receipt.sha256,
                "size": stable_receipt.size,
                "remote_path": receipt_path,
                "stable_stat": {
                    "size": stable_receipt.stat.size,
                    "mtime": stable_receipt.stat.mtime,
                    "mode": stable_receipt.stat.mode,
                },
            },
            "_status_bytes": stable_receipt.payload,
        }
        if receipt["state"] != "completed":
            if ordinal < declared_sealed_count:
                declared_failed += 1
            observation["failure"] = receipt.get("failure")
            observations.append(observation)
            continue
        if ordinal < declared_sealed_count:
            declared_completed += 1
        result_path = _remote_child(
            remote_root,
            "runs",
            f"task-{task_id}",
            f"seed-{seed}",
            str(manifest["search_execution"]["result_filename"]),
        )
        stable_result = read_stable_remote_json(
            remote,
            account_name=account,
            path=result_path,
            maximum_bytes=MAX_RESULT_BYTES,
        )
        if stable_result.sha256 != _hex(
            receipt.get("result_sha256"), "batch child result_sha256"
        ):
            raise RuntimeError("batch child result SHA does not match receipt")
        result_validator(
            stable_result.value,
            payload=child["task"]["payload_json"],
            manifest=manifest,
        )
        _fail_closed(stable_result.value, "remote batch child terminal result")
        output_root = _remote_child(
            remote_root, "runs", f"task-{task_id}", f"seed-{seed}"
        )
        artifact_objects, artifact_payloads, candidates = _harvest_result_artifacts(
            result=stable_result.value,
            account_name=account,
            output_root=output_root,
            remote=remote,
        )
        observation.update(
            {
                "result_object": {
                    "sha256": stable_result.sha256,
                    "size": stable_result.size,
                    "remote_path": result_path,
                    "stable_stat": {
                        "size": stable_result.stat.size,
                        "mtime": stable_result.stat.mtime,
                        "mode": stable_result.stat.mode,
                    },
                },
                "completed_generations": int(
                    stable_result.value["completed_generations"]
                ),
                "feasible_pareto_count": int(
                    stable_result.value.get("feasible_pareto_count", 0)
                ),
                "artifact_objects": artifact_objects,
                "_result_bytes": stable_result.payload,
                "_result": stable_result.value,
                "_artifact_payloads": artifact_payloads,
                "_candidates": candidates,
            }
        )
        observations.append(observation)
    if declared_completed != int(
        task_status["completed_child_count"]
    ) or declared_failed != int(task_status["failed_child_count"]):
        raise RuntimeError(
            "batch task counters do not replay the declared receipt prefix"
        )
    return observations


def harvest_mixed_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    remote: RemoteReader,
    result_validator: Callable[..., None] = validate_current7_result,
    cohort_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    journal_snapshots: Mapping[
        int,
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str],
    ]
    | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Flatten v1 terminal tasks and v2 journal progress into seed records."""

    observations: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    physical_task_ids: set[int] = set()
    v1_count = 0
    v2_count = 0
    for item in inventory:
        task_id = int(item["task_id"])
        if task_id in physical_task_ids:
            raise RuntimeError("mixed inventory repeats a physical Scheduler task id")
        physical_task_ids.add(task_id)
        parent = item.get("parent_task")
        is_v2 = (
            isinstance(parent, dict)
            and isinstance(parent.get("payload_json"), dict)
            and parent["payload_json"].get("schema_version")
            in {BATCH_PAYLOAD_SCHEMA, phase_b_contract.BATCH_PAYLOAD_SCHEMA}
        )
        try:
            if is_v2:
                v2_count += 1
                if str(item.get("status") or "").lower() in {
                    "queued",
                    "attaching",
                }:
                    continue
                observations.extend(
                    harvest_batch_lane(
                        item,
                        plan=plan,
                        manifest=manifest,
                        remote=remote,
                        result_validator=result_validator,
                        journal_snapshot=(journal_snapshots or {}).get(task_id),
                    )
                )
            else:
                v1_count += 1
                if (
                    str(item.get("status") or "").lower()
                    not in TERMINAL_SCHEDULER_STATES
                ):
                    continue
                cohort_id = item.get("source_harvest_cohort_id")
                selected_plan = plan
                selected_manifest = manifest
                if cohort_id is not None:
                    context = (cohort_contexts or {}).get(str(cohort_id))
                    if (
                        not isinstance(context, dict)
                        or not isinstance(context.get("plan"), dict)
                        or not isinstance(context.get("manifest"), dict)
                        or context["plan"].get("launch_plan_sha256")
                        != item.get("source_launch_plan_sha256")
                    ):
                        raise RuntimeError(
                            "v1 mixed harvest cohort context is missing or mismatched"
                        )
                    selected_plan = context["plan"]
                    selected_manifest = context["manifest"]
                observation = harvest_terminal_task(
                    item,
                    plan=selected_plan,
                    manifest=selected_manifest,
                    remote=remote,
                    result_validator=result_validator,
                )
                if cohort_id is not None:
                    observation["harvest_cohort_id"] = str(cohort_id)
                observations.append(observation)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            if strict:
                raise
            refusals.append(
                {
                    "task_id": task_id,
                    "protocol_version": "v2" if is_v2 else "v1",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    records = deduplicate_observations(observations)
    provenance_by_identity: dict[tuple[str, int], tuple[int, int, str]] = {}
    for observation in observations:
        parent_task_id = observation.get("physical_parent_task_id")
        if parent_task_id is None:
            continue
        identity = (str(observation["bundle_id"]), int(observation["seed"]))
        provenance = (
            int(parent_task_id),
            int(observation["batch_ordinal"]),
            str(observation["batch_manifest_sha256"]),
        )
        prior = provenance_by_identity.setdefault(identity, provenance)
        if prior != provenance:
            raise RuntimeError(
                "one scientific seed has divergent physical batch provenance"
            )
    for record in records:
        provenance = provenance_by_identity.get(
            (str(record["bundle_id"]), int(record["seed"]))
        )
        if provenance is None:
            continue
        (
            record["physical_parent_task_id"],
            record["batch_ordinal"],
            record["batch_manifest_sha256"],
        ) = provenance
    return {
        "physical_task_count": len(physical_task_ids),
        "v1_physical_task_count": v1_count,
        "v2_physical_task_count": v2_count,
        "authenticated_seed_count": len(records),
        "refused_physical_task_count": len(refusals),
        "refusals": refusals,
        "records": records,
        "physical_task_ids": sorted(physical_task_ids),
        "virtual_scheduler_task_ids_created": False,
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
    }


def public_records(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_public_observation(item) for item in value.get("records") or []]
