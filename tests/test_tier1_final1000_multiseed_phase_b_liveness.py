from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.test_tier1_final1000_multiseed_phase_a import _rendered_plan
from tools import tier1_final1000_multiseed_contract as phase_a
from tools import tier1_final1000_multiseed_harvest as harvest
from tools import tier1_final1000_multiseed_phase_b_contract as phase_b
from tools import tier1_final1000_multiseed_phase_b_liveness as liveness
from tools import tier1_final1000_multiseed_phase_b_successor as successor
from tools import tier1_final1000_stage_profiles as profiles


def _reseal(value: dict[str, Any], seal_field: str) -> dict[str, Any]:
    sealed = copy.deepcopy(value)
    unsigned = {key: item for key, item in sealed.items() if key != seal_field}
    sealed[seal_field] = phase_a.canonical_sha256(unsigned)
    return sealed


@pytest.fixture(scope="module")
def cohort() -> dict[str, Any]:
    launch_plan = _rendered_plan()
    seed_starts = {
        stage.stage_id: stage.seed_start + 10_000 for stage in profiles.STAGES
    }
    placement = successor.build_placement_plan(
        launch_plan,
        next_seed_by_stage=seed_starts,
        allocation_inventory=successor.current_empty_pool_inventory(),
    )
    parents = placement.pop("parent_tasks")
    assert sum(task["payload_json"]["batch_length"] for task in parents) == 500
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in launch_plan["task_waves"]["canaries"]
    }
    control_fence = liveness.build_control_fence(
        owner_identity="fixture-controller-1",
        epoch=1,
        state="active",
        phase_a_handoff_lease_sha256="b" * 64,
        control_store_capability_sha256="1" * 64,
        watcher_capability_sha256="2" * 64,
        submitter_capability_sha256="3" * 64,
        cas_commit_receipt_sha256="4" * 64,
        cas_store_revision=1,
    )
    authority = liveness.build_authority_lease(control_fence=control_fence)
    checkpoint = liveness.build_checkpoint(
        authority=authority,
        next_seed_by_stage={
            stage.stage_id: seed_starts[stage.stage_id]
            + successor.STAGE_LOGICAL_QUOTAS[stage.stage_id]
            for stage in profiles.STAGES
        },
    )
    observations = [
        liveness.seal_parent_observation(
            task_id=80_000 + index,
            scheduler_status="queued",
            parent_task=task,
        )
        for index, task in enumerate(parents)
    ]
    victim_index = next(
        index
        for index, task in enumerate(parents)
        if task["payload_json"]["batch_length"] == 4
    )
    return {
        "parents": parents,
        "templates": templates,
        "authority": authority,
        "control_fence": control_fence,
        "checkpoint": checkpoint,
        "observations": observations,
        "victim_index": victim_index,
        "victim_task_id": 80_000 + victim_index,
    }


def _pagination_receipt(
    cohort: dict[str, Any], observations: list[dict[str, Any]], *, cursor: str
) -> dict[str, Any]:
    task_ids = [int(item["task_id"]) for item in observations]
    return liveness.build_watcher_pagination_receipt(
        authority=cohort["authority"],
        current_control_fence=cohort["control_fence"],
        namespace_fence_sha256="c" * 64,
        page_task_ids=[task_ids],
        page_cursor_sha256s=[cursor * 64],
    )


def _snapshot(
    cohort: dict[str, Any], observations: list[dict[str, Any]], *, cursor: str
) -> dict[str, Any]:
    receipt = _pagination_receipt(cohort, observations, cursor=cursor)
    return liveness.build_bounded_inventory_snapshot(
        observations,
        authority=cohort["authority"],
        watcher_pagination_receipt=receipt,
    )


def _running_status(
    task: dict[str, Any],
    *,
    task_id: int,
    sealed_count: int = 0,
    completed_count: int = 0,
) -> dict[str, Any]:
    payload = task["payload_json"]
    return phase_b.seal_task_status(
        {
            "schema_version": phase_b.TASK_STATUS_SCHEMA,
            "protocol_version": phase_b.PROTOCOL_VERSION,
            "task_id": str(task_id),
            "manifest_sha256": phase_b.batch_manifest_from_payload(payload)[
                "manifest_sha256"
            ],
            "state": "running",
            "stop_requested": False,
            "stop_reason": None,
            "active_children": [],
            "launched_child_count": sealed_count,
            "finished_child_count": sealed_count,
            "sealed_child_count": sealed_count,
            "completed_child_count": completed_count,
            "failed_child_count": sealed_count - completed_count,
            "started_at": "2026-07-22T00:00:00+09:00",
            "updated_at": "2026-07-22T00:00:01+09:00",
            "finished_at": None,
            "execution_mode": "concurrent",
            "concurrent_children": payload["concurrent_children"],
            "subprocess_per_seed": True,
            "model_context_reuse": False,
            "rng_context_reuse": False,
            "cpu_isolation": copy.deepcopy(phase_b.CPU_ISOLATION),
            "resource_telemetry": None,
            "physical_scheduler_task_count": 1,
            "virtual_scheduler_task_ids_created": False,
            "scheduler_mutation_performed": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )


def _completed_receipt(
    task: dict[str, Any], *, task_id: int, ordinal: int
) -> dict[str, Any]:
    manifest = phase_b.batch_manifest_from_payload(task["payload_json"])
    child = task["payload_json"]["children"][ordinal]
    legacy = {
        "schema_version": "fixture-legacy-terminal-v1",
        "state": "completed",
        "seed": child["seed"],
        "payload_sha256": child["payload_sha256"],
        "terminal": True,
    }
    return phase_b.seal_child_receipt(
        {
            "schema_version": phase_b.CHILD_RECEIPT_SCHEMA,
            "protocol_version": phase_b.PROTOCOL_VERSION,
            "task_id": str(task_id),
            "manifest_sha256": manifest["manifest_sha256"],
            "ordinal": ordinal,
            "seed": child["seed"],
            "payload_sha256": child["payload_sha256"],
            "logical_dedupe_key": child["logical_dedupe_key"],
            "state": "completed",
            "terminal": True,
            "lane_fatal": False,
            "exit_code": 0,
            "legacy_status": legacy,
            "legacy_status_sha256": phase_a.canonical_sha256(legacy),
            "result_sha256": "d" * 64,
            "started_at": "2026-07-22T00:00:00+09:00",
            "finished_at": "2026-07-22T00:00:01+09:00",
            "wall_time_seconds": 1.0,
            "child_resource_telemetry": {
                "schema_version": phase_b.CHILD_RESOURCE_TELEMETRY_SCHEMA,
                "available": True,
                "measurement": "resource.getrusage(self+children)-delta",
                "child_cpus": 4,
                "wall_time_seconds": 1.0,
                "process_tree_cpu_seconds": 1.0,
                "cpu_capacity_seconds": 4.0,
                "cpu_utilization_fraction": 0.25,
            },
            "failure": None,
            "cpu_set": list(range(ordinal * 4, ordinal * 4 + 4)),
            "child_cpus": 4,
            "child_memory_mb": phase_b.CHILD_MEMORY_MB,
            "runtime_scratch_relative_path": f"seed-{child['seed']}/runtime",
            "runtime_scratch_cleanup_performed": True,
            "shared_tmp_deleted": False,
            "production_eligible": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )


def _replace_victim(
    cohort: dict[str, Any],
    *,
    include_manifest: bool,
    scheduler_status: str = "failed",
    status: dict[str, Any] | None = None,
    receipts: list[dict[str, Any]] | None = None,
    recovery_attestation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    observations = copy.deepcopy(cohort["observations"])
    task = cohort["parents"][cohort["victim_index"]]
    observations[cohort["victim_index"]] = liveness.seal_parent_observation(
        task_id=cohort["victim_task_id"],
        scheduler_status=scheduler_status,
        parent_task=task,
        batch_manifest=(
            phase_b.batch_manifest_from_payload(task["payload_json"])
            if include_manifest
            else None
        ),
        task_status=status,
        child_receipts=receipts or [],
        authority=(cohort["authority"] if recovery_attestation is not None else None),
        terminal_receipt_recovery_attestation=recovery_attestation,
    )
    return observations


def _reconcile(
    cohort: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    checkpoint: dict[str, Any] | None = None,
    cursor: str = "e",
) -> dict[str, Any]:
    return liveness.reconcile_deficit(
        authority=cohort["authority"],
        current_control_fence=cohort["control_fence"],
        checkpoint=checkpoint or cohort["checkpoint"],
        inventory_snapshot=_snapshot(cohort, observations, cursor=cursor),
        fresh_task_templates=cohort["templates"],
    )


def _intent_child_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        child
        for intent in plan["submission_intents"]
        for child in intent["ordered_children"]
    ]


def test_initial_exact500_is_stable_and_dry(cohort: dict[str, Any]):
    plan = _reconcile(cohort, copy.deepcopy(cohort["observations"]))
    assert plan["logical_active_before_by_stage"] == {
        profiles.STAGES[0].stage_id: 300,
        profiles.STAGES[1].stage_id: 150,
        profiles.STAGES[2].stage_id: 40,
        profiles.STAGES[3].stage_id: 10,
    }
    assert plan["logical_active_after"] == 500
    assert plan["submission_intents"] == []
    assert plan["scheduler_write_performed"] is False
    assert plan["http_request_performed"] is False
    assert plan["cancellation_performed"] is False
    assert plan["preemption_performed"] is False


def test_manifest_and_status_crash_points_recover_only_incomplete_identity(
    cohort: dict[str, Any],
):
    task = cohort["parents"][cohort["victim_index"]]
    manifest_only = _replace_victim(cohort, include_manifest=True)
    after_manifest = _reconcile(cohort, manifest_only, cursor="f")
    status = _running_status(task, task_id=cohort["victim_task_id"])
    after_status = _reconcile(
        cohort,
        _replace_victim(cohort, include_manifest=True, status=status),
        cursor="1",
    )
    original = {
        child["logical_dedupe_key"] for child in task["payload_json"]["children"]
    }
    manifest_rows = _intent_child_rows(after_manifest)
    status_rows = _intent_child_rows(after_status)
    assert {row["logical_dedupe_key"] for row in manifest_rows} == original
    assert all(row["disposition"] == "recover-incomplete" for row in manifest_rows)
    assert manifest_rows == status_rows
    assert [
        intent["physical_parent_dedupe_key"]
        for intent in after_manifest["submission_intents"]
    ] == [
        intent["physical_parent_dedupe_key"]
        for intent in after_status["submission_intents"]
    ]
    assert all(
        intent["physical_parent_dedupe_key"] != task["dedupe_key"]
        for intent in after_manifest["submission_intents"]
    )
    assert after_manifest["logical_active_after"] == 500


def test_terminal_receipt_beyond_stale_cursor_fails_closed_then_cursor_recovers(
    cohort: dict[str, Any],
):
    task = cohort["parents"][cohort["victim_index"]]
    receipt = _completed_receipt(task, task_id=cohort["victim_task_id"], ordinal=0)
    cursor_before = _running_status(
        task, task_id=cohort["victim_task_id"], sealed_count=0
    )
    cursor_after = _running_status(
        task,
        task_id=cohort["victim_task_id"],
        sealed_count=1,
        completed_count=1,
    )
    with pytest.raises(RuntimeError, match="shared recovery attestation"):
        _replace_victim(
            cohort,
            include_manifest=True,
            status=cursor_before,
            receipts=[receipt],
        )
    watcher_receipt = _pagination_receipt(cohort, cohort["observations"], cursor="2")
    manifest = phase_b.batch_manifest_from_payload(task["payload_json"])
    receipt_scan: list[dict[str, Any]] = []
    for ordinal, child in enumerate(task["payload_json"]["children"]):
        present = ordinal == 0
        receipt_scan.append(
            {
                "ordinal": ordinal,
                "seed": child["seed"],
                "present": present,
                "remote_path": (
                    f"{task['remote_cwd'].rstrip('/')}/runs/"
                    f"task-{cohort['victim_task_id']}/seed-{child['seed']}/"
                    "seed_status.json"
                ),
                "receipt": receipt if present else None,
                "remote_receipt_sha256": (
                    phase_a.canonical_sha256(receipt) if present else None
                ),
                "remote_receipt_size": 1 if present else None,
                "remote_mode": 0o444 if present else None,
                "stable_stat_count": 3 if present else 2,
                "stable_byte_read_count": 2 if present else 0,
                "absence_confirmed": not present,
            }
        )
    recovery_attestation = phase_b.build_terminal_receipt_recovery_attestation(
        parent_task=task,
        manifest=manifest,
        task_status=cursor_before,
        task_id=cohort["victim_task_id"],
        scheduler_state="failed",
        receipt_scan=receipt_scan,
        watcher_capability_sha256=cohort["control_fence"]["watcher_capability_sha256"],
        watcher_revision_sha256=watcher_receipt["watcher_pagination_receipt_sha256"],
        scheduler_get_count=1,
    )
    recovered_observations = _replace_victim(
        cohort,
        include_manifest=True,
        status=cursor_before,
        receipts=[receipt],
        recovery_attestation=recovery_attestation,
    )
    recovered = _reconcile(cohort, recovered_observations, cursor="2")
    wrong_revision_attestation = phase_b.build_terminal_receipt_recovery_attestation(
        parent_task=task,
        manifest=manifest,
        task_status=cursor_before,
        task_id=cohort["victim_task_id"],
        scheduler_state="failed",
        receipt_scan=receipt_scan,
        watcher_capability_sha256=cohort["control_fence"]["watcher_capability_sha256"],
        watcher_revision_sha256="f" * 64,
        scheduler_get_count=1,
    )
    wrong_revision_observations = _replace_victim(
        cohort,
        include_manifest=True,
        status=cursor_before,
        receipts=[receipt],
        recovery_attestation=wrong_revision_attestation,
    )
    with pytest.raises(RuntimeError, match="pagination mismatch"):
        _reconcile(cohort, wrong_revision_observations, cursor="2")
    after_observations = _replace_victim(
        cohort,
        include_manifest=True,
        status=cursor_after,
        receipts=[receipt],
    )
    after = _reconcile(cohort, after_observations, cursor="3")
    assert recovered["submission_intents"] == after["submission_intents"]
    completed = task["payload_json"]["children"][0]
    expected_recovery = {
        child["logical_dedupe_key"] for child in task["payload_json"]["children"][1:]
    }
    after_rows = _intent_child_rows(after)
    assert completed["logical_dedupe_key"] not in {
        row["logical_dedupe_key"] for row in after_rows
    }
    assert {
        row["logical_dedupe_key"]
        for row in after_rows
        if row["disposition"] == "recover-incomplete"
    } == expected_recovery
    fresh_rows = [row for row in after_rows if row["disposition"] == "fresh-refill"]
    assert len(fresh_rows) == 1
    stage_id = task["payload_json"]["stage_id"]
    assert fresh_rows[0]["seed"] == cohort["checkpoint"]["next_seed_by_stage"][stage_id]
    assert len({row["seed"] for row in after_rows}) == len(after_rows)
    assert len({row["logical_dedupe_key"] for row in after_rows}) == len(after_rows)
    assert after["logical_active_after"] == 500

    # Persisting the outbox checkpoint before the Scheduler reflects its POST
    # still reserves the exact four children.  No second seed or parent intent
    # is created after the status cursor catches up with the durable receipt.
    replay = _reconcile(
        cohort,
        after_observations,
        checkpoint=after["next_checkpoint"],
        cursor="4",
    )
    assert replay["submission_intents"] == []
    assert replay["pending_intent_count_after"] == len(after["submission_intents"])
    assert replay["logical_active_after"] == 500


def test_deterministic_tail_regrouping_is_at_most_eight(cohort: dict[str, Any]):
    task = cohort["parents"][cohort["victim_index"]]
    receipt = _completed_receipt(task, task_id=cohort["victim_task_id"], ordinal=0)
    plan = _reconcile(
        cohort,
        _replace_victim(
            cohort,
            include_manifest=True,
            status=_running_status(
                task,
                task_id=cohort["victim_task_id"],
                sealed_count=1,
                completed_count=1,
            ),
            receipts=[receipt],
        ),
        cursor="5",
    )
    lengths = [
        intent["parent_task"]["payload_json"]["batch_length"]
        for intent in plan["submission_intents"]
    ]
    assert sorted(lengths) == [1, 3]
    assert max(lengths) <= 8
    assert all(
        child["seed"]
        == intent["parent_task"]["payload_json"]["children"][0]["seed"] + ordinal
        for intent in plan["submission_intents"]
        for ordinal, child in enumerate(
            intent["parent_task"]["payload_json"]["children"]
        )
    )


def test_liveness_attempt_is_harvestable_but_science_envelope_drift_is_refused(
    cohort: dict[str, Any],
):
    plan = _reconcile(
        cohort,
        _replace_victim(cohort, include_manifest=True),
        cursor="7",
    )
    intent = plan["submission_intents"][0]
    task = intent["parent_task"]
    with pytest.raises(RuntimeError, match="lacks its sealed intent/authority/outbox"):
        phase_b.validate_runtime_batch_task(task)
    validated = phase_b.validate_runtime_batch_task(
        task,
        submission_intent=intent,
        authority=cohort["authority"],
        reconciliation_plan=plan,
    )
    assert validated == task
    item = {
        "task_id": 100_007,
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "status": "running",
        "parent_task": copy.deepcopy(task),
        "scheduler_task": copy.deepcopy(task),
        "phase_b_submission_intent": copy.deepcopy(intent),
        "phase_b_authority_lease": copy.deepcopy(cohort["authority"]),
        "phase_b_reconciliation_plan": copy.deepcopy(plan),
    }
    launch_identity = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "bundle_manifest_sha256": task["payload_json"]["bundle_manifest_sha256"],
        "remote_bundle": task["remote_cwd"],
    }
    assert (
        harvest._batch_parent_task(
            item,
            plan=launch_identity,
            manifest={"bundle_id": launch_identity["bundle_id"]},
        )
        == task
    )

    for field, replacement in (
        ("command", ["python", "tampered.py"]),
        ("cpus", int(task["cpus"]) + 1),
        ("memory_mb", int(task["memory_mb"]) + 1),
    ):
        drifted = copy.deepcopy(task)
        drifted[field] = replacement
        with pytest.raises(RuntimeError, match="canonical payload"):
            phase_b.validate_runtime_batch_task(
                drifted,
                submission_intent=intent,
                authority=cohort["authority"],
                reconciliation_plan=plan,
            )

    payload_drift = copy.deepcopy(task)
    payload_drift["payload_json"]["children"][0]["seed"] += 1
    with pytest.raises(RuntimeError):
        phase_b.validate_runtime_batch_task(
            payload_drift,
            submission_intent=intent,
            authority=cohort["authority"],
            reconciliation_plan=plan,
        )

    invalid_attempt = copy.deepcopy(task)
    invalid_attempt["dedupe_key"] = phase_b.PHYSICAL_ATTEMPT_DEDUPE_PREFIX + "0"
    with pytest.raises(RuntimeError, match="attempt dedupe identity"):
        phase_b.validate_runtime_batch_task(
            invalid_attempt,
            submission_intent=intent,
            authority=cohort["authority"],
            reconciliation_plan=plan,
        )

    with pytest.raises(RuntimeError, match="outbox lineage"):
        phase_b.validate_runtime_batch_task(
            task,
            submission_intent=intent,
            authority=cohort["authority"],
        )
    tampered_plan = copy.deepcopy(plan)
    tampered_plan["next_checkpoint"]["pending_submission_intents"] = []
    checkpoint_unsigned = {
        key: value
        for key, value in tampered_plan["next_checkpoint"].items()
        if key != "checkpoint_sha256"
    }
    tampered_plan["next_checkpoint"]["checkpoint_sha256"] = phase_a.canonical_sha256(
        checkpoint_unsigned
    )
    tampered_plan["next_checkpoint_sha256"] = tampered_plan["next_checkpoint"][
        "checkpoint_sha256"
    ]
    tampered_plan["post_precondition_checkpoint_sha256"] = tampered_plan[
        "next_checkpoint"
    ]["checkpoint_sha256"]
    plan_unsigned = {
        key: value for key, value in tampered_plan.items() if key != "plan_sha256"
    }
    tampered_plan["plan_sha256"] = phase_a.canonical_sha256(plan_unsigned)
    with pytest.raises(RuntimeError):
        phase_b.validate_runtime_batch_task(
            task,
            submission_intent=intent,
            authority=cohort["authority"],
            reconciliation_plan=tampered_plan,
        )


def test_duplicate_active_seed_dedupe_is_refused(cohort: dict[str, Any]):
    observations = copy.deepcopy(cohort["observations"])
    victim = cohort["parents"][cohort["victim_index"]]
    child = victim["payload_json"]["children"][0]
    legacy = phase_a._legacy_child_task_from_phase_a(
        child["task"], expected_stage=profiles.BY_ID[victim["payload_json"]["stage_id"]]
    )
    duplicate_parent = phase_b.build_concurrent_batch_task([legacy])
    assert duplicate_parent["dedupe_key"] != victim["dedupe_key"]
    observations.append(
        liveness.seal_parent_observation(
            task_id=99_999,
            scheduler_status="queued",
            parent_task=duplicate_parent,
        )
    )
    with pytest.raises(RuntimeError, match="duplicate active Phase-B logical"):
        _reconcile(cohort, observations, cursor="6")


def test_unsealed_or_ambiguous_artifacts_fail_closed(
    cohort: dict[str, Any],
):
    task = cohort["parents"][cohort["victim_index"]]
    status = _running_status(task, task_id=cohort["victim_task_id"])
    status["status_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="task status seal mismatch"):
        _replace_victim(cohort, include_manifest=True, status=status)

    receipt = _completed_receipt(task, task_id=cohort["victim_task_id"], ordinal=0)
    receipt["receipt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="receipt seal mismatch"):
        _replace_victim(
            cohort,
            include_manifest=True,
            status=_running_status(task, task_id=cohort["victim_task_id"]),
            receipts=[receipt],
        )

    with pytest.raises(RuntimeError, match="completed Scheduler parent"):
        liveness.seal_parent_observation(
            task_id=cohort["victim_task_id"],
            scheduler_status="completed",
            parent_task=task,
        )


def test_cas_fence_is_deterministic_exclusive_monotonic_and_stoppable(
    cohort: dict[str, Any],
):
    fence = cohort["control_fence"]
    authority = cohort["authority"]
    assert liveness.build_authority_lease(control_fence=fence) == authority
    assert (
        liveness.build_authority_lease(control_fence=copy.deepcopy(fence)) == authority
    )

    failed = _replace_victim(cohort, include_manifest=True)
    first = _reconcile(cohort, failed, cursor="8")
    second = _reconcile(cohort, failed, cursor="8")
    assert first["submission_intents"] == second["submission_intents"]
    assert [
        item["physical_parent_dedupe_key"] for item in first["submission_intents"]
    ] == [item["physical_parent_dedupe_key"] for item in second["submission_intents"]]

    alternate_fence = liveness.build_control_fence(
        owner_identity="self-asserted-alternate-controller",
        epoch=1,
        state="active",
        phase_a_handoff_lease_sha256=fence["phase_a_handoff_lease_sha256"],
        control_store_capability_sha256=fence["control_store_capability_sha256"],
        watcher_capability_sha256=fence["watcher_capability_sha256"],
        submitter_capability_sha256=fence["submitter_capability_sha256"],
        cas_commit_receipt_sha256="5" * 64,
        cas_store_revision=1,
    )
    alternate_authority = liveness.build_authority_lease(control_fence=alternate_fence)
    with pytest.raises(RuntimeError, match="not the current durable CAS owner"):
        liveness.require_current_authority(
            alternate_authority, current_control_fence=fence
        )

    receipt_tamper = copy.deepcopy(fence)
    receipt_tamper["cas_commit_receipt"]["expected_store_revision"] += 1
    receipt_tamper["cas_commit_receipt"] = _reseal(
        receipt_tamper["cas_commit_receipt"], "cas_receipt_sha256"
    )
    receipt_tamper["cas_commit_receipt_sha256"] = receipt_tamper["cas_commit_receipt"][
        "cas_receipt_sha256"
    ]
    receipt_tamper = _reseal(receipt_tamper, "control_fence_sha256")
    with pytest.raises(RuntimeError, match="CAS control fence mismatch"):
        liveness.validate_control_fence(receipt_tamper)

    stopped = liveness.build_control_fence(
        owner_identity=fence["current_owner_identity"],
        epoch=2,
        state="stopped",
        phase_a_handoff_lease_sha256=fence["phase_a_handoff_lease_sha256"],
        control_store_capability_sha256=fence["control_store_capability_sha256"],
        watcher_capability_sha256=fence["watcher_capability_sha256"],
        submitter_capability_sha256=fence["submitter_capability_sha256"],
        cas_commit_receipt_sha256="6" * 64,
        cas_store_revision=2,
        previous_fence=fence,
    )
    with pytest.raises(RuntimeError, match="is stopped"):
        liveness.reconcile_deficit(
            authority=authority,
            current_control_fence=stopped,
            checkpoint=cohort["checkpoint"],
            inventory_snapshot=_snapshot(
                cohort, copy.deepcopy(cohort["observations"]), cursor="9"
            ),
            fresh_task_templates=cohort["templates"],
        )
    with pytest.raises(RuntimeError, match="cannot transition"):
        liveness.build_control_fence(
            owner_identity=fence["current_owner_identity"],
            epoch=3,
            state="active",
            phase_a_handoff_lease_sha256=fence["phase_a_handoff_lease_sha256"],
            control_store_capability_sha256=fence["control_store_capability_sha256"],
            watcher_capability_sha256=fence["watcher_capability_sha256"],
            submitter_capability_sha256=fence["submitter_capability_sha256"],
            cas_commit_receipt_sha256="7" * 64,
            cas_store_revision=3,
            previous_fence=stopped,
        )
    with pytest.raises(RuntimeError, match="epoch is not monotonic"):
        liveness.build_control_fence(
            owner_identity=fence["current_owner_identity"],
            epoch=3,
            state="active",
            phase_a_handoff_lease_sha256=fence["phase_a_handoff_lease_sha256"],
            control_store_capability_sha256=fence["control_store_capability_sha256"],
            watcher_capability_sha256=fence["watcher_capability_sha256"],
            submitter_capability_sha256=fence["submitter_capability_sha256"],
            cas_commit_receipt_sha256="8" * 64,
            cas_store_revision=2,
            previous_fence=fence,
        )
    with pytest.raises(RuntimeError, match="store revision is not monotonic"):
        liveness.build_control_fence(
            owner_identity=fence["current_owner_identity"],
            epoch=2,
            state="active",
            phase_a_handoff_lease_sha256=fence["phase_a_handoff_lease_sha256"],
            control_store_capability_sha256=fence["control_store_capability_sha256"],
            watcher_capability_sha256=fence["watcher_capability_sha256"],
            submitter_capability_sha256=fence["submitter_capability_sha256"],
            cas_commit_receipt_sha256="9" * 64,
            cas_store_revision=3,
            previous_fence=fence,
        )


def test_pagination_receipt_binds_exact_pages_tasks_capability_and_zero_post(
    cohort: dict[str, Any],
):
    observations = copy.deepcopy(cohort["observations"])
    task_ids = [int(item["task_id"]) for item in observations]
    split = len(task_ids) // 2
    receipt = liveness.build_watcher_pagination_receipt(
        authority=cohort["authority"],
        current_control_fence=cohort["control_fence"],
        namespace_fence_sha256="a" * 64,
        page_task_ids=[task_ids[:split], task_ids[split:]],
        page_cursor_sha256s=["b" * 64, "c" * 64],
    )
    assert receipt["page_count"] == 2
    assert receipt["exact_task_count"] == len(task_ids)
    assert receipt["scheduler_get_count"] == 2
    assert receipt["scheduler_post_count"] == 0
    snapshot = liveness.build_bounded_inventory_snapshot(
        observations,
        authority=cohort["authority"],
        watcher_pagination_receipt=receipt,
    )
    assert snapshot["observation_count"] == len(task_ids)

    with pytest.raises(RuntimeError, match="receipt/observation inventory mismatch"):
        liveness.build_bounded_inventory_snapshot(
            observations[:-1],
            authority=cohort["authority"],
            watcher_pagination_receipt=receipt,
        )
    with pytest.raises(RuntimeError, match="repeat a physical task id"):
        liveness.build_watcher_pagination_receipt(
            authority=cohort["authority"],
            current_control_fence=cohort["control_fence"],
            namespace_fence_sha256="a" * 64,
            page_task_ids=[[task_ids[0]], [task_ids[0]]],
            page_cursor_sha256s=["b" * 64, "c" * 64],
        )
    post_tamper = copy.deepcopy(receipt)
    post_tamper["scheduler_post_count"] = 1
    post_tamper = _reseal(post_tamper, "watcher_pagination_receipt_sha256")
    with pytest.raises(RuntimeError, match="pagination receipt seal mismatch"):
        liveness.validate_watcher_pagination_receipt(
            post_tamper, authority=cohort["authority"]
        )


def test_plan_validator_rechecks_arithmetic_generation_unique_outbox_and_post_order(
    cohort: dict[str, Any],
):
    plan = _reconcile(
        cohort,
        _replace_victim(cohort, include_manifest=True),
        cursor="d",
    )
    liveness.validate_reconciliation_plan(plan, authority=cohort["authority"])
    assert plan["checkpoint_must_be_durable_before_post"] is True
    assert plan["submit_before_checkpoint_allowed"] is False
    assert plan["post_requires_current_control_fence_cas_recheck"] is True
    assert plan["post_precondition_checkpoint_sha256"] == plan["next_checkpoint_sha256"]
    assert all(
        intent["checkpoint_before_post_required"] is True
        and intent["submit_before_checkpoint_allowed"] is False
        and intent["post_requires_current_control_fence_cas_recheck"] is True
        for intent in plan["submission_intents"]
    )

    stage_id = cohort["parents"][cohort["victim_index"]]["payload_json"]["stage_id"]
    arithmetic_tamper = copy.deepcopy(plan)
    arithmetic_tamper["logical_active_before_by_stage"][stage_id] -= 1
    arithmetic_tamper = _reseal(arithmetic_tamper, "plan_sha256")
    with pytest.raises(RuntimeError, match="plan seal mismatch"):
        liveness.validate_reconciliation_plan(
            arithmetic_tamper, authority=cohort["authority"]
        )

    empty_outbox = liveness.build_checkpoint(
        authority=cohort["authority"],
        next_seed_by_stage=plan["next_checkpoint"]["next_seed_by_stage"],
        generation=plan["plan_generation"],
        pending_submission_intents=[],
        last_inventory_snapshot_sha256=plan["inventory_snapshot_sha256"],
    )
    outbox_tamper = copy.deepcopy(plan)
    outbox_tamper["next_checkpoint"] = empty_outbox
    outbox_tamper["next_checkpoint_sha256"] = empty_outbox["checkpoint_sha256"]
    outbox_tamper["post_precondition_checkpoint_sha256"] = empty_outbox[
        "checkpoint_sha256"
    ]
    outbox_tamper["pending_intent_count_after"] = 0
    outbox_tamper = _reseal(outbox_tamper, "plan_sha256")
    with pytest.raises(RuntimeError, match="plan seal mismatch"):
        liveness.validate_reconciliation_plan(
            outbox_tamper, authority=cohort["authority"]
        )

    generation_two_checkpoint = liveness.build_checkpoint(
        authority=cohort["authority"],
        next_seed_by_stage=plan["next_checkpoint"]["next_seed_by_stage"],
        generation=2,
        pending_submission_intents=plan["submission_intents"],
        last_inventory_snapshot_sha256=plan["inventory_snapshot_sha256"],
    )
    generation_tamper = copy.deepcopy(plan)
    generation_tamper["plan_generation"] = 2
    generation_tamper["next_checkpoint"] = generation_two_checkpoint
    generation_tamper["next_checkpoint_sha256"] = generation_two_checkpoint[
        "checkpoint_sha256"
    ]
    generation_tamper["post_precondition_checkpoint_sha256"] = (
        generation_two_checkpoint["checkpoint_sha256"]
    )
    generation_tamper = _reseal(generation_tamper, "plan_sha256")
    with pytest.raises(RuntimeError, match="plan seal mismatch"):
        liveness.validate_reconciliation_plan(
            generation_tamper, authority=cohort["authority"]
        )

    intent = plan["submission_intents"][0]
    canonical = phase_b.canonical_parent_from_runtime_task(intent["parent_task"])
    duplicate_intent = liveness._build_submission_intent(
        canonical,
        intent["ordered_children"],
        authority=cohort["authority"],
        plan_generation=plan["plan_generation"],
        intent_ordinal=len(plan["submission_intents"]),
    )
    with pytest.raises(RuntimeError, match="repeats a pending logical child"):
        liveness.build_checkpoint(
            authority=cohort["authority"],
            next_seed_by_stage=plan["next_checkpoint"]["next_seed_by_stage"],
            generation=plan["plan_generation"],
            pending_submission_intents=[
                *plan["submission_intents"],
                duplicate_intent,
            ],
            last_inventory_snapshot_sha256=plan["inventory_snapshot_sha256"],
        )
