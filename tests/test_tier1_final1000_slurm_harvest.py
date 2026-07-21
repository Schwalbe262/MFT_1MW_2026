from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse

import pytest

from tests.test_tier1_final1000_stage_launch import _rendered_plan
from tests import test_tier1_final1000_rolling_migration as rolling_fixtures
from tools import tier1_corrected_current7_slurm_harvest as current_harvest
from tools import tier1_corrected_current7_slurm_seed_runner as seed_runner
from tools import tier1_final1000_slurm_controller as controller
from tools import tier1_final1000_slurm_harvest as harvest
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


REPO = Path(__file__).resolve().parents[1]


class Scheduler:
    def __init__(self, tasks, *, batch_omit_ids=()):
        self.tasks = {int(task["id"]): copy.deepcopy(task) for task in tasks}
        self.batch_omit_ids = {int(value) for value in batch_omit_ids}
        self.get_count = 0
        self.batch_get_count = 0
        self.task_get_count = 0

    def list_tasks(self, *, name_prefix, limit):
        assert name_prefix == controller.TASK_NAME_PREFIX
        assert limit == harvest.SCHEDULER_BATCH_LIMIT
        self.get_count += 1
        self.batch_get_count += 1
        return [
            copy.deepcopy(task)
            for task_id, task in sorted(self.tasks.items(), reverse=True)
            if task_id not in self.batch_omit_ids
            and str(task.get("name") or "").startswith(name_prefix)
        ][:limit]

    def get_task(self, task_id):
        self.get_count += 1
        self.task_get_count += 1
        value = self.tasks.get(int(task_id))
        return copy.deepcopy(value) if value is not None else None


class NoRemoteReads:
    def stat(self, _account_name, _path):
        raise AssertionError("running-only projection must not read SFTP")

    def read_bytes(self, _account_name, _path, *, maximum_bytes):
        del maximum_bytes
        raise AssertionError("running-only projection must not read SFTP")


def _state_and_tasks(plan, *, scheduler_state="running", all_entries=False):
    state = controller._initial_state(plan)
    plan_index = controller._task_index(plan)
    templates = controller._task_templates(plan)
    tasks = []
    task_id = 91_000
    seen_stages = set()
    for entry in state["entries"]:
        if not all_entries and entry["stage_id"] in seen_stages:
            continue
        seen_stages.add(entry["stage_id"])
        task_id += 1
        entry["task_id"] = task_id
        entry["state"] = (
            scheduler_state
            if scheduler_state in controller.ACTIVE_STATES | controller.TERMINAL_STATES
            else "running"
        )
        expected = controller._task_for_entry(
            entry, plan_index=plan_index, templates=templates
        )
        tasks.append(
            {
                **{
                    field: copy.deepcopy(expected[field])
                    for field in harvest.OBSERVED_TASK_SEAL_FIELDS
                },
                "id": task_id,
                "status": scheduler_state,
                "account_name": "account-a",
                "slurm_job_id": str(task_id),
                "exit_code": 0 if scheduler_state == "completed" else None,
                "created_at": "2026-07-21T01:00:00+00:00",
                "started_at": "2026-07-21T01:00:01+00:00",
                "finished_at": (
                    "2026-07-21T01:10:00+00:00"
                    if scheduler_state == "completed"
                    else None
                ),
            }
        )
    state = controller._seal_state(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    return controller._validate_state(state, plan), tasks


def _bindings(plan):
    values = {}
    temperature_targets = list(seed_runner.CURRENT_TEMPERATURE_TARGETS)
    for stage in profiles.STAGES:
        hard_spec, constraint_names = launch.optimizer_stage_contract(stage)
        summary = plan["stage_bindings"][stage.stage_id]
        manifest = {
            "contract_sha256": "c" * 64,
            "task_schema_version": "mft-tier1-current7-slurm-seed-task-v1",
            "status_schema_version": current_harvest.STATUS_SCHEMA,
            "constraint_version": (f"final1000-{stage.stage_id}-res15to20k-n1-6-v1"),
            "hard_spec": hard_spec,
            "hard_spec_sha256": launch.canonical_sha256(hard_spec),
            "hard_constraint_contract_sha256": "d" * 64,
            "temperature_contract_sha256": "e" * 64,
            "constraint_names": constraint_names,
            "temperature_targets": temperature_targets,
            "search_execution": {
                "result_schema_version": current_harvest.RESULT_SCHEMA,
                "result_filename": "result.json",
            },
        }
        values[stage.stage_id] = {
            "plan": {
                "bundle_id": summary["bundle_id"],
                "bundle_manifest_sha256": summary["bundle_manifest_sha256"],
                "remote_bundle": summary["remote_bundle"],
            },
            "manifest": manifest,
            "publication": {
                "receipt_sha256": summary["publication_receipt_sha256"],
                "ready": copy.deepcopy(summary["ready"]),
                "ready_sha256": summary["ready_sha256"],
            },
        }
    return values


def _cohorts(plan):
    return {
        "successor": harvest._validate_plan_bindings(
            role="successor",
            plan=plan,
            bindings=_bindings(plan),
            resource_policy_ids=[controller.SUCCESSOR_RESOURCE_POLICY_ID],
        )
    }


def _rolling_cohort(role, plan, policy_ids):
    return harvest._validate_plan_bindings(
        role=role,
        plan=plan,
        bindings=_bindings(plan),
        resource_policy_ids=policy_ids,
    )


def _resource_only_mixed_campaign(tmp_path):
    fixture = rolling_fixtures._fixture(tmp_path / "rolling")
    resource_plan = rolling_fixtures._resource_only_successor(
        fixture["predecessor"], fixture["successor"]
    )
    fixture["successor"] = resource_plan
    fixture["successor_plan_path"].write_text(json.dumps(resource_plan))
    fixture["ready"] = rolling_fixtures._Ready(fixture["predecessor"], resource_plan)
    rolling_fixtures._prepare(
        fixture,
        apply=True,
        transition_mode=rolling_fixtures.migration.RESOURCE_QUOTA_ONLY,
    )
    controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    state = json.loads(fixture["successor_state_path"].read_text())
    cohorts = {
        "predecessor": _rolling_cohort(
            "predecessor",
            fixture["predecessor"],
            [controller.LEGACY_RESOURCE_POLICY_ID],
        ),
        "successor": _rolling_cohort(
            "successor",
            resource_plan,
            [controller.SUCCESSOR_RESOURCE_POLICY_ID],
        ),
    }
    tasks = list(fixture["scheduler"].by_id.values())
    return fixture, resource_plan, state, cohorts, tasks


def test_apply_publishes_four_isolated_condition_indexes_and_never_primary(
    tmp_path, monkeypatch
):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    runtime = tmp_path / "final1000-runtime"
    state_path = runtime / "controller" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    launch_path = tmp_path / "launch.json"
    bindings_path = tmp_path / "bindings.json"
    launch_path.write_text("{}", encoding="utf-8")
    bindings_path.write_text("{}", encoding="utf-8")
    current7_root = tmp_path / "current7-primary"
    protected = current7_root / "canonical" / "current7-index.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"current7-primary-sentinel\n")
    monkeypatch.setattr(
        harvest,
        "_validate_inputs",
        lambda *_args, **_kwargs: (plan, _cohorts(plan), state),
    )

    result = harvest.harvest_once(
        launch_path,
        bindings_path,
        state_path,
        scheduler=Scheduler(tasks),
        remote=NoRemoteReads(),
        runtime=runtime,
        protected_current7_index=protected,
        apply=True,
        observed_at="2026-07-21 01:02:03",
    )

    assert result["scheduler_mutation_count"] == 0
    assert result["remote_write_count"] == 0
    assert result["scheduler_get_count"] == 1
    assert result["scheduler_batch_get_count"] == 1
    assert result["scheduler_task_get_count"] == 0
    assert protected.read_bytes() == b"current7-primary-sentinel\n"
    indexes = result["condition_inventory"]["indexes"]
    assert [item["stage_id"] for item in indexes] == [
        stage.stage_id for stage in profiles.STAGES
    ]
    assert result["condition_inventory"]["ui_environment"]["value"] == (
        os.pathsep.join(item["path"] for item in indexes)
    )
    for item, stage in zip(indexes, profiles.STAGES, strict=True):
        path = Path(item["path"])
        assert (
            path
            == (
                runtime
                / "conditions"
                / stage.stage_id
                / "canonical"
                / "current7-index.json"
            ).resolve()
        )
        index = json.loads(path.read_text(encoding="utf-8"))
        assert index["schema_version"] == current_harvest.CURRENT7_INDEX_SCHEMA
        assert index["final_goal_stage_id"] == stage.stage_id
        assert index["hard_spec"]["resonance_min_Hz"] == 15_000.0
        assert index["hard_spec"]["resonance_max_Hz"] == 20_000.0
        assert index["harvest_observed_at"] == "2026-07-21 01:02:03"
        assert index["condition_display_only"] is True


def test_terminal_scheduler_envelope_uses_detail_get_then_caches(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, scheduler_state="completed")
    runtime = tmp_path / "final1000-runtime"
    scheduler = Scheduler(tasks)
    by_stage, pending, hits = harvest._inventory_tasks(
        plan, state, cohorts=_cohorts(plan), scheduler=scheduler, runtime=runtime
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 5
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 4
    assert hits == 0
    assert len(pending) == 4
    for path, payload in pending:
        current_harvest._atomic_replace(path, payload)

    scheduler.tasks.clear()
    by_stage, pending, hits = harvest._inventory_tasks(
        plan, state, cohorts=_cohorts(plan), scheduler=scheduler, runtime=runtime
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 5
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 4
    assert hits == 4
    assert pending == []


def test_500_running_tasks_use_one_campaign_inventory_get(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, all_entries=True)
    assert len(tasks) == 500
    scheduler = Scheduler(tasks)

    by_stage, pending, hits = harvest._inventory_tasks(
        plan,
        state,
        cohorts=_cohorts(plan),
        scheduler=scheduler,
        runtime=tmp_path / "final1000-runtime",
    )

    assert sum(len(items) for items in by_stage.values()) == 500
    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0
    assert pending == []
    assert hits == 0


def test_resource_only_mixed_ledger_replays_8c_and_4c_task_seals(tmp_path):
    _fixture, plan, state, cohorts, tasks = _resource_only_mixed_campaign(tmp_path)
    scheduler = Scheduler(tasks)

    by_stage, _pending, _hits = harvest._inventory_tasks(
        plan,
        state,
        cohorts=cohorts,
        scheduler=scheduler,
        runtime=tmp_path / "harvest",
    )
    items = [item for values in by_stage.values() for item in values]
    predecessor = [item for item in items if item["binding_role"] == "predecessor"]
    successor = [item for item in items if item["binding_role"] == "successor"]

    assert len(predecessor) == 500
    assert len(successor) == 4
    assert {item["resource_policy_id"] for item in predecessor} == {
        controller.LEGACY_RESOURCE_POLICY_ID
    }
    assert {item["resource_policy_id"] for item in successor} == {
        controller.SUCCESSOR_RESOURCE_POLICY_ID
    }
    assert {item["_expected_task"]["cpus"] for item in predecessor} == {8}
    assert {item["_expected_task"]["max_workers_per_node"] for item in predecessor} == {
        8
    }
    assert {item["_expected_task"]["cpus"] for item in successor} == {4}
    assert {item["_expected_task"]["max_workers_per_node"] for item in successor} == {
        32
    }
    for item in items:
        assert item["bundle_id"] == item["_binding"]["plan"]["bundle_id"]
        assert (
            item["bundle_manifest_sha256"]
            == item["_binding"]["plan"]["bundle_manifest_sha256"]
        )


def test_full_harvest_authenticates_legacy_and_successor_results_in_one_index(
    tmp_path, monkeypatch, request
):
    fixture, plan, state, cohorts, _tasks = _resource_only_mixed_campaign(tmp_path)
    stage = profiles.STAGES[0]
    successor_entry = next(
        entry
        for entry in state["entries"]
        if entry["origin"] == "successor" and entry["stage_id"] == stage.stage_id
    )
    successor_entry["state"] = "completed"
    fixture["scheduler"].by_id[int(successor_entry["task_id"])]["status"] = "completed"
    state = controller._seal_state(state)
    scheduler = Scheduler(fixture["scheduler"].by_id.values())
    # Keep content-addressed cache paths below the legacy Windows MAX_PATH
    # limit used by the CI interpreter.
    runtime = Path(tmp_path.anchor) / (
        "f1k-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:10]
    )
    request.addfinalizer(lambda: shutil.rmtree(runtime, ignore_errors=True))
    state_path = runtime / "controller" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state))
    launch_path = tmp_path / "launch.json"
    bindings_path = tmp_path / "bindings.json"
    launch_path.write_text("{}")
    bindings_path.write_text("{}")
    protected = tmp_path / "primary" / "canonical" / "current7-index.json"
    protected.parent.mkdir(parents=True)
    protected.write_text("primary")
    monkeypatch.setattr(
        harvest,
        "_validate_inputs",
        lambda *_args, **_kwargs: (plan, cohorts, state),
    )
    monkeypatch.setattr(
        harvest,
        "validate_current7_result",
        lambda _result, **_kwargs: None,
    )

    def fake_terminal(item, *, plan, manifest, remote, result_validator):
        del remote
        payload = item["payload"]
        constraints = list(payload["constraint_names"])
        physical = {name: -1.0 for name in constraints}
        result_value = {
            "hard_spec": copy.deepcopy(payload["hard_spec"]),
            "stage_spec_sha256": payload["stage_spec_sha256"],
            "constraint_names": constraints,
            "constraint_version": manifest["constraint_version"],
            "hard_constraint_contract_sha256": manifest[
                "hard_constraint_contract_sha256"
            ],
            "temperature_contract_sha256": manifest["temperature_contract_sha256"],
            "temperature_targets": list(manifest["temperature_targets"]),
            "seed": payload["seed"],
            "fixed_primary_turns": 6,
            "terminal_population_primary_turn_values": [6],
            "terminal_population_fixed_primary_turns_verified": True,
            "terminal_physical_replay_attested": True,
            "constraint_minimum_G": physical,
            "terminal_population_best_constraint_G": physical,
            "optimizer_terminal_best_physical_constraint_G": physical,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
        }
        result_validator(result_value, payload=payload, manifest=manifest)
        status_bytes = json.dumps(
            {"task_id": item["task_id"], "seed": item["seed"]}, sort_keys=True
        ).encode()
        result_bytes = json.dumps(result_value, sort_keys=True).encode()
        return {
            "task_id": item["task_id"],
            "bundle_id": plan["bundle_id"],
            "seed": item["seed"],
            "island_id": item["island_id"],
            "scheduler_state": "completed",
            "terminal_state": "completed",
            "completed_generations": 201,
            "feasible_pareto_count": 0,
            "status_object": {
                "remote_path": f"/status/{item['task_id']}.json",
                "sha256": hashlib.sha256(status_bytes).hexdigest(),
                "size_bytes": len(status_bytes),
            },
            "result_object": {
                "remote_path": f"/result/{item['task_id']}.json",
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
                "size_bytes": len(result_bytes),
            },
            "artifact_objects": {},
            "_status_bytes": status_bytes,
            "_result_bytes": result_bytes,
            "_result": result_value,
            "_artifact_payloads": {},
            "_candidates": [],
        }

    monkeypatch.setattr(harvest, "harvest_terminal_task", fake_terminal)
    result = harvest.harvest_once(
        launch_path,
        bindings_path,
        state_path,
        scheduler=scheduler,
        remote=NoRemoteReads(),
        runtime=runtime,
        protected_current7_index=protected,
        apply=True,
        observed_at="2026-07-22 01:00:00",
    )

    stage_result = next(
        item for item in result["stages"] if item["stage_id"] == stage.stage_id
    )
    assert stage_result["authenticated_seed_count"] == 2
    index_path = Path(
        next(
            item["path"]
            for item in result["condition_inventory"]["indexes"]
            if item["stage_id"] == stage.stage_id
        )
    )
    index = json.loads(index_path.read_text())
    status = json.loads(Path(index["status"]["path"]).read_text())
    assert index["mixed_bundle_projection"] is False
    assert index["mixed_resource_policy_projection"] is True
    assert status["authenticated_terminal_seed_count"] == 2
    assert {item["resource_policy_id"] for item in status["latest_tasks"]} == {
        controller.LEGACY_RESOURCE_POLICY_ID,
        controller.SUCCESSOR_RESOURCE_POLICY_ID,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("cpus", 99), ("aedt_backend", "attached")),
)
def test_scheduler_resource_drift_is_refused_before_projection(
    tmp_path, field, replacement
):
    _fixture, plan, state, cohorts, tasks = _resource_only_mixed_campaign(tmp_path)
    tasks[0][field] = replacement
    scheduler = Scheduler(tasks)

    with pytest.raises(RuntimeError, match="scheduler task identity changed"):
        harvest._inventory_tasks(
            plan,
            state,
            cohorts=cohorts,
            scheduler=scheduler,
            runtime=tmp_path / "harvest",
        )


def test_migration_inputs_require_both_authenticated_plan_binding_pairs(
    tmp_path, monkeypatch
):
    fixture, plan, state, cohorts, _tasks = _resource_only_mixed_campaign(tmp_path)
    successor_plan_path = tmp_path / "successor-plan.json"
    predecessor_plan_path = tmp_path / "predecessor-plan.json"
    successor_bindings_path = tmp_path / "successor-bindings.json"
    predecessor_bindings_path = tmp_path / "predecessor-bindings.json"
    state_path = tmp_path / "state.json"
    for path in (
        successor_plan_path,
        predecessor_plan_path,
        successor_bindings_path,
        predecessor_bindings_path,
        state_path,
    ):
        path.write_text("{}")
    values = {
        successor_plan_path.resolve(): plan,
        predecessor_plan_path.resolve(): fixture["predecessor"],
        state_path.resolve(): state,
    }
    binding_values = {
        successor_bindings_path.resolve(): cohorts["successor"]["bindings"],
        predecessor_bindings_path.resolve(): cohorts["predecessor"]["bindings"],
    }
    monkeypatch.setattr(harvest, "_read_json", lambda path: values[Path(path)])
    monkeypatch.setattr(
        harvest,
        "load_stage_bindings",
        lambda path: binding_values[Path(path).resolve()],
    )

    with pytest.raises(RuntimeError, match="requires predecessor launch"):
        harvest._validate_inputs(
            successor_plan_path,
            successor_bindings_path,
            state_path,
        )
    validated_plan, validated_cohorts, validated_state = harvest._validate_inputs(
        successor_plan_path,
        successor_bindings_path,
        state_path,
        predecessor_launch_plan_path=predecessor_plan_path,
        predecessor_bindings_path=predecessor_bindings_path,
    )
    assert validated_plan["launch_plan_sha256"] == plan["launch_plan_sha256"]
    assert {
        role: value["identity"] for role, value in validated_cohorts.items()
    } == state["rolling_migration"]["harvest_cohorts"]
    assert validated_state["state_sha256"] == state["state_sha256"]

    live_compatible = copy.deepcopy(state)
    live_compatible["rolling_migration"].pop("harvest_cohorts")
    live_compatible = controller._seal_state(live_compatible)
    values[state_path.resolve()] = live_compatible
    _plan, compatible_cohorts, compatible_state = harvest._validate_inputs(
        successor_plan_path,
        successor_bindings_path,
        state_path,
        predecessor_launch_plan_path=predecessor_plan_path,
        predecessor_bindings_path=predecessor_bindings_path,
    )
    assert compatible_state["state_sha256"] == live_compatible["state_sha256"]
    assert {
        role: value["identity"] for role, value in compatible_cohorts.items()
    } == controller._derived_resource_only_harvest_cohorts(
        plan, live_compatible["rolling_migration"]
    )


def test_patched_bundle_mixed_ledger_replays_old_and_new_bindings(tmp_path):
    fixture, resource_plan, _state, _cohorts_value, _tasks = (
        _resource_only_mixed_campaign(tmp_path)
    )
    fixture["scheduler"].pass_successor_canaries()
    controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
        request_stop=True,
    )
    patched_plan = rolling_fixtures._successor_plan()
    patched_plan_path = tmp_path / "patched-plan.json"
    patched_state_path = tmp_path / "patched-state.json"
    patched_plan_path.write_text(json.dumps(patched_plan))
    ready = rolling_fixtures._Ready(fixture["predecessor"], resource_plan, patched_plan)
    rolling_fixtures.migration.prepare_successor_state(
        predecessor_plan_path=fixture["successor_plan_path"],
        predecessor_state_path=fixture["successor_state_path"],
        successor_plan_path=patched_plan_path,
        successor_state_path=patched_state_path,
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=ready,
        successor_ready_probe=ready,
        apply=True,
        transition_mode=rolling_fixtures.migration.PATCHED_BUNDLE,
    )
    patched_state = json.loads(patched_state_path.read_text())
    vacated = next(
        entry
        for entry in patched_state["entries"]
        if entry["state"] in controller.ACTIVE_STATES
    )
    fixture["scheduler"].by_id[int(vacated["task_id"])]["status"] = "completed"
    controller.control_once(
        patched_plan_path,
        state_path=patched_state_path,
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=ready,
    )
    patched_state = json.loads(patched_state_path.read_text())
    predecessor_policies = patched_state["rolling_migration"]["harvest_cohorts"][
        "predecessor"
    ]["resource_policy_ids"]
    cohorts = {
        "predecessor": _rolling_cohort(
            "predecessor", resource_plan, predecessor_policies
        ),
        "successor": _rolling_cohort(
            "successor", patched_plan, [controller.SUCCESSOR_RESOURCE_POLICY_ID]
        ),
    }

    by_stage, _pending, _hits = harvest._inventory_tasks(
        patched_plan,
        patched_state,
        cohorts=cohorts,
        scheduler=Scheduler(fixture["scheduler"].by_id.values()),
        runtime=tmp_path / "patched-harvest",
    )
    items = [item for values in by_stage.values() for item in values]
    old = [item for item in items if item["binding_role"] == "predecessor"]
    new = [item for item in items if item["binding_role"] == "successor"]
    assert old
    assert new
    assert {item["bundle_id"] for item in old} == {
        binding["bundle_id"] for binding in resource_plan["stage_bindings"].values()
    }
    assert {item["bundle_id"] for item in new}.issubset(
        {binding["bundle_id"] for binding in patched_plan["stage_bindings"].values()}
    )
    assert all(
        item["bundle_id"] == item["_binding"]["plan"]["bundle_id"] for item in items
    )


def test_mixed_bundle_projection_uses_each_originating_manifest(tmp_path, monkeypatch):
    fixture, old_plan, state, old_cohorts, _tasks = _resource_only_mixed_campaign(
        tmp_path
    )
    patched_plan = fixture["successor"] = rolling_fixtures._successor_plan()
    predecessor_policy_ids = sorted(
        {
            str(entry["resource_policy_id"])
            for entry in state["entries"]
            if entry["origin"] == "predecessor"
        }
    )
    old_cohort = _rolling_cohort("predecessor", old_plan, predecessor_policy_ids)
    new_cohort = _rolling_cohort(
        "successor", patched_plan, [controller.SUCCESSOR_RESOURCE_POLICY_ID]
    )
    stage = profiles.STAGES[0]
    old_binding = old_cohort["bindings"][stage.stage_id]
    new_binding = new_cohort["bindings"][stage.stage_id]
    calls = []

    def fake_terminal(item, *, plan, manifest, remote, result_validator):
        del remote, result_validator
        calls.append((plan["bundle_id"], manifest["contract_sha256"]))
        payload = b"{}\n"
        return {
            "task_id": item["task_id"],
            "bundle_id": plan["bundle_id"],
            "seed": item["seed"],
            "island_id": item["island_id"],
            "scheduler_state": "completed",
            "terminal_state": "completed",
            "status_object": {
                "remote_path": f"/status/{item['task_id']}.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            },
            "_status_bytes": payload,
        }

    monkeypatch.setattr(harvest, "harvest_terminal_task", fake_terminal)
    inventory = []
    for task_id, binding, role, policy_id, seed in (
        (
            101,
            old_binding,
            "predecessor",
            controller.LEGACY_RESOURCE_POLICY_ID,
            stage.seed_start,
        ),
        (
            102,
            new_binding,
            "successor",
            controller.SUCCESSOR_RESOURCE_POLICY_ID,
            stage.seed_start + 1,
        ),
    ):
        template = old_cohort if role == "predecessor" else new_cohort
        expected = rolling_fixtures.migration._render_from_template_for_policy(
            template["templates"][stage.stage_id],
            stage_id=stage.stage_id,
            seed=seed,
            wave="refill",
            policy=rolling_fixtures.migration.RESOURCE_POLICIES[policy_id],
        )
        inventory.append(
            {
                "task_id": task_id,
                "name": expected["name"],
                "dedupe_key": expected["dedupe_key"],
                "bundle_id": binding["plan"]["bundle_id"],
                "bundle_manifest_sha256": binding["plan"]["bundle_manifest_sha256"],
                "binding_role": role,
                "resource_policy_id": policy_id,
                "seed": seed,
                "island_id": expected["payload_json"]["lane"]["island_id"],
                "wave": "refill",
                "status": "completed",
                "account_name": "account-a",
                "slurm_job_id": str(task_id),
                "exit_code": 0,
                "created_at": "2026-07-22T00:00:00+00:00",
                "started_at": "2026-07-22T00:00:01+00:00",
                "finished_at": "2026-07-22T00:01:00+00:00",
                "payload": expected["payload_json"],
                "payload_sha256": harvest.canonical_sha256(expected["payload_json"]),
                "_expected_task": expected,
                "_binding": binding,
            }
        )

    projection = harvest._project_stage(
        stage_id=stage.stage_id,
        anchor_binding=new_binding,
        source_cohorts=[old_cohort, new_cohort],
        inventory=inventory,
        remote=NoRemoteReads(),
        runtime=tmp_path / "projection",
        observed_at="2026-07-22 00:02:00",
        fallback_event_at="2026-07-22T00:00:00+00:00",
    )

    assert calls == [
        (old_binding["plan"]["bundle_id"], old_binding["manifest"]["contract_sha256"]),
        (new_binding["plan"]["bundle_id"], new_binding["manifest"]["contract_sha256"]),
    ]
    assert projection["index"]["mixed_bundle_projection"] is True
    assert projection["index"]["mixed_resource_policy_projection"] is True
    assert [item["role"] for item in projection["index"]["source_bundle_cohorts"]] == [
        "predecessor",
        "successor",
    ]


def test_task_missing_from_capped_batch_uses_one_fail_closed_detail_get(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    omitted_id = int(tasks[0]["id"])
    scheduler = Scheduler(tasks, batch_omit_ids={omitted_id})

    by_stage, pending, hits = harvest._inventory_tasks(
        plan,
        state,
        cohorts=_cohorts(plan),
        scheduler=scheduler,
        runtime=tmp_path / "final1000-runtime",
    )

    assert sum(len(items) for items in by_stage.values()) == 4
    assert scheduler.get_count == 2
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 1
    assert pending == []
    assert hits == 0


def test_batch_inventory_identity_mismatch_fails_closed(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan)
    tasks[0]["dedupe_key"] = "foreign-dedupe"
    scheduler = Scheduler(tasks)

    with pytest.raises(RuntimeError, match="inventory identity changed"):
        harvest._inventory_tasks(
            plan,
            state,
            cohorts=_cohorts(plan),
            scheduler=scheduler,
            runtime=tmp_path / "final1000-runtime",
        )

    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0


def test_terminal_batch_row_requires_matching_detail_evidence(tmp_path):
    plan = _rendered_plan()
    state, tasks = _state_and_tasks(plan, scheduler_state="completed")

    class ChangedTerminalScheduler(Scheduler):
        def get_task(self, task_id):
            value = super().get_task(task_id)
            assert value is not None
            value["status"] = "failed"
            return value

    scheduler = ChangedTerminalScheduler(tasks)

    with pytest.raises(RuntimeError, match="terminal scheduler evidence changed"):
        harvest._inventory_tasks(
            plan,
            state,
            cohorts=_cohorts(plan),
            scheduler=scheduler,
            runtime=tmp_path / "final1000-runtime",
        )

    assert scheduler.get_count == 2
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 1


def test_read_only_api_uses_filtered_max_limit_inventory_get(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b"[]"

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(harvest.urllib.request, "urlopen", fake_urlopen)
    scheduler = harvest.Final1000ReadOnlySchedulerApi(
        "http://scheduler:8002", timeout=7
    )

    assert (
        scheduler.list_tasks(
            name_prefix=controller.TASK_NAME_PREFIX,
            limit=harvest.SCHEDULER_BATCH_LIMIT,
        )
        == []
    )

    assert len(requests) == 1
    request, timeout = requests[0]
    parsed = urllib.parse.urlparse(request.full_url)
    assert request.get_method() == "GET"
    assert parsed.path == "/api/tasks"
    assert urllib.parse.parse_qs(parsed.query) == {
        "name_prefix": [controller.TASK_NAME_PREFIX],
        "limit": [str(harvest.SCHEDULER_BATCH_LIMIT)],
    }
    assert timeout == 7
    assert scheduler.get_count == 1
    assert scheduler.batch_get_count == 1
    assert scheduler.task_get_count == 0


def test_runtime_overlap_with_current7_primary_fails_before_writes(tmp_path):
    protected = tmp_path / "current7" / "canonical" / "current7-index.json"
    runtime = tmp_path / "current7" / "conditions" / "final1000"
    with pytest.raises(RuntimeError, match="overlaps"):
        harvest._assert_runtime_isolated(runtime, protected)


def test_cli_exposes_watch_apply_but_no_scheduler_mutation_surface():
    process = subprocess.run(
        [sys.executable, "tools/tier1_final1000_slurm_harvest.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    help_text = process.stdout.lower()
    assert "--watch" in help_text
    assert "--apply" in help_text
    assert "--scheduler-url" in help_text
    assert "--predecessor-launch-plan" in help_text
    assert "--predecessor-bindings" in help_text
    assert "--submit" not in help_text
    assert "--cancel" not in help_text
    assert "--close" not in help_text
