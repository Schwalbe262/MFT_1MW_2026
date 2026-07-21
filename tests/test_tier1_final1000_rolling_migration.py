from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import urllib.error

import pytest

from tools import tier1_final1000_rolling_migration as migration
from tools import tier1_corrected_current7_slurm_controller as current7_controller
from tools import tier1_final1000_slurm_controller as controller
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


def _base_task(stage: profiles.FinalGoalStage, seed: int) -> dict:
    payload = {
        "schema_version": "mft-tier1-current7-slurm-seed-task-v1",
        "bundle_id": f"successor-{stage.stage_id}",
        "bundle_manifest_sha256": "a" * 64,
        "lane": {
            "island_id": "n1-6-base",
            "variant": "n1-6-base-current7",
            "seed": seed,
            "fixed_primary_turns": 6,
            "wave": "refill",
        },
        "seed": seed,
        "population": 320,
        "max_generations": profiles.FIXED_GENERATIONS,
        "inference_threads": 8,
        "optimizer_processes": 1,
        "maximum_peak_rss_bytes": 42 * 1024**3,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    payload_sha = launch.canonical_sha256(payload)
    return {
        "name": "base",
        "remote_cwd": f"/gpfs/successor/{stage.stage_id}",
        "command": (
            "set -euo pipefail\nexec python -u artifacts/code/tools/"
            "tier1_corrected_current7_slurm_seed_runner.py "
            f"--payload-sha256 {payload_sha}"
        ),
        "payload_json": payload,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": 8,
        "memory_mb": 65_536,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": 0,
        "timeout_seconds": 86_400,
        "dedupe_key": "base:" + "b" * 64,
        "max_workers_per_node": 4,
    }


def _successor_plan() -> dict:
    canaries: list[dict] = []
    ramp: list[dict] = []
    for lane in [*launch.logical_lanes()[0], *launch.logical_lanes()[1]]:
        stage = profiles.BY_ID[lane["stage_id"]]
        task = launch.build_stage_task(
            _base_task(stage, int(lane["seed"])),
            stage=stage,
            wave=str(lane["wave"]),
        )
        (canaries if lane["wave"] == "canary" else ramp).append(task)
    bindings = {}
    for stage in profiles.STAGES:
        ready = {
            "schema_version": "mft-tier1-current7-slurm-ready-v1",
            "bundle_id": f"successor-{stage.stage_id}",
            "stage_spec_sha256": profiles.stage_profile(stage)["stage_spec_sha256"],
        }
        bindings[stage.stage_id] = {
            "bundle_id": f"successor-{stage.stage_id}",
            "bundle_manifest_sha256": "a" * 64,
            "remote_bundle": f"/gpfs/successor/{stage.stage_id}",
            "publication_receipt_sha256": "b" * 64,
            "ready": ready,
            "ready_sha256": launch.canonical_sha256(ready),
            "base_island_id": "n1-6-base",
            "stage_spec_sha256": profiles.stage_profile(stage)["stage_spec_sha256"],
        }
    unsigned = {
        "schema_version": launch.LAUNCH_SCHEMA,
        "created_at": "2026-07-21T00:00:00+09:00",
        "stage_inventory": profiles.stage_inventory(),
        "stage_bindings": bindings,
        "resources": {
            "cpus_per_task": 4,
            "memory_mb_per_task": 28 * 1024,
            "max_workers_per_node": 32,
            "priority": 1,
            "scheduling_profile": "standard",
            "gpus": 0,
        },
        "fast_ramp": {"canary_count": 4, "ramp_count": 496},
        "open_ended_refill": {
            "logical_active_target": 500,
            "stage_active_quotas": launch.SUCCESSOR_ACTIVE_QUOTAS,
        },
        "task_waves": {"canaries": canaries, "ramp": ramp},
        "surrogate_only": True,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return {**unsigned, "launch_plan_sha256": launch.canonical_sha256(unsigned)}


def _predecessor_plan(successor: dict) -> dict:
    plan = copy.deepcopy(successor)
    successor_templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in successor["task_waves"]["canaries"]
    }
    for stage in profiles.STAGES:
        binding = plan["stage_bindings"][stage.stage_id]
        binding["bundle_id"] = f"predecessor-{stage.stage_id}"
        binding["bundle_manifest_sha256"] = "c" * 64
        binding["remote_bundle"] = f"/gpfs/predecessor/{stage.stage_id}"
        binding["publication_receipt_sha256"] = "d" * 64
        binding["ready"]["bundle_id"] = binding["bundle_id"]
        binding["ready_sha256"] = launch.canonical_sha256(binding["ready"])
    plan["task_waves"] = {"canaries": [], "ramp": []}
    for stage in profiles.STAGES:
        for offset in range(stage.active_quota):
            wave = "canary" if offset == 0 else "ramp"
            template = copy.deepcopy(successor_templates[stage.stage_id])
            payload = template["payload_json"]
            stage_id = stage.stage_id
            binding = plan["stage_bindings"][stage_id]
            payload["bundle_id"] = binding["bundle_id"]
            payload["bundle_manifest_sha256"] = binding["bundle_manifest_sha256"]
            template["remote_cwd"] = binding["remote_bundle"]
            task = migration._render_from_template(
                template,
                stage_id=stage_id,
                seed=stage.seed_start + offset,
                wave=wave,
            )
            plan["task_waves"]["canaries" if wave == "canary" else "ramp"].append(task)
    plan["resources"] = {
        key: migration.PREDECESSOR_POLICY[key]
        for key in (
            "cpus_per_task",
            "memory_mb_per_task",
            "max_workers_per_node",
            "priority",
            "scheduling_profile",
            "gpus",
        )
    }
    plan["open_ended_refill"]["stage_active_quotas"] = {
        stage.stage_id: stage.active_quota for stage in profiles.STAGES
    }
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return migration.validate_predecessor_plan(plan)


def _resource_only_successor(predecessor: dict, successor: dict) -> dict:
    plan = copy.deepcopy(successor)
    plan["stage_bindings"] = copy.deepcopy(predecessor["stage_bindings"])
    for wave_name in ("canaries", "ramp"):
        rendered = []
        for source in successor["task_waves"][wave_name]:
            template = copy.deepcopy(source)
            payload = template["payload_json"]
            stage_id = payload["final_goal_stage_id"]
            binding = plan["stage_bindings"][stage_id]
            payload["bundle_id"] = binding["bundle_id"]
            payload["bundle_manifest_sha256"] = binding["bundle_manifest_sha256"]
            template["remote_cwd"] = binding["remote_bundle"]
            rendered.append(
                migration._render_from_template_for_policy(
                    template,
                    stage_id=stage_id,
                    seed=int(payload["seed"]),
                    wave=str(payload["lane"]["wave"]),
                    policy=migration.SUCCESSOR_POLICY,
                )
            )
        plan["task_waves"][wave_name] = rendered
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return migration.validate_successor_plan(plan)


def _historical_resource_quota_successor(
    predecessor: dict, successor: dict
) -> dict:
    """Render the exact 0033c48 phase-1 160/140/120/80 plan identity."""

    plan = _resource_only_successor(predecessor, successor)
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in plan["task_waves"]["canaries"]
    }
    canaries: list[dict] = []
    ramp: list[dict] = []
    for stage in profiles.STAGES:
        quota = controller.HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS[
            stage.stage_id
        ]
        for offset in range(quota):
            wave = "canary" if offset == 0 else "ramp"
            task = migration._render_from_template_for_policy(
                templates[stage.stage_id],
                stage_id=stage.stage_id,
                seed=stage.seed_start + offset,
                wave=wave,
                policy=migration.SUCCESSOR_POLICY,
            )
            (canaries if wave == "canary" else ramp).append(task)
    plan["task_waves"] = {"canaries": canaries, "ramp": ramp}
    plan["open_ended_refill"]["stage_active_quotas"] = copy.deepcopy(
        controller.HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS
    )
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return migration.validate_historical_resource_quota_successor_plan(plan)


def _stopped_historical_resource_quota_state(plan: dict) -> dict:
    """Build a sealed stopped phase-1 state without running old release code."""

    state = controller._initial_state(plan)
    next_task_id = 90_000
    for entry in state["entries"]:
        next_task_id += 1
        entry["task_id"] = next_task_id
        entry["state"] = "running"
        entry["origin"] = "successor"
        entry["resource_policy_id"] = controller.SUCCESSOR_RESOURCE_POLICY_ID
    state["stop_requested"] = True
    state["ramp_released"] = True
    state["canary_passed_stage_ids"] = sorted(profiles.BY_ID)
    state["scheduler_submit_count"] = 500
    legacy_plan_identity = copy.deepcopy(plan)
    legacy_plan_identity["launch_plan_sha256"] = "e" * 64
    canary_ids = {
        stage.stage_id: sorted(
            int(entry["task_id"])
            for entry in state["entries"]
            if entry["stage_id"] == stage.stage_id and entry["wave"] == "canary"
        )
        for stage in profiles.STAGES
    }
    state["rolling_migration"] = {
        "schema_version": migration.MIGRATION_SCHEMA,
        "transition_mode": migration.RESOURCE_QUOTA_ONLY,
        "predecessor_controller_kind": "legacy_8c",
        "predecessor_launch_plan_sha256": "e" * 64,
        "predecessor_state_sha256": "f" * 64,
        "predecessor_state_revision": 1,
        "successor_launch_plan_sha256": plan["launch_plan_sha256"],
        "scheduler_inventory_sha256": "a" * 64,
        "predecessor_entry_count": 0,
        "imported_active_count_by_stage": {
            stage.stage_id: 0 for stage in profiles.STAGES
        },
        "next_seed_by_stage": copy.deepcopy(state["next_seed_by_stage"]),
        "successor_resource_policy": copy.deepcopy(
            controller.ROLLING_SUCCESSOR_RESOURCE_POLICY
        ),
        "successor_active_quotas": copy.deepcopy(
            controller.HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS
        ),
        "harvest_cohorts": {
            "predecessor": migration.harvest_cohort_identity(
                role="predecessor",
                plan=legacy_plan_identity,
                resource_policy_ids=[controller.LEGACY_RESOURCE_POLICY_ID],
            ),
            "successor": migration.harvest_cohort_identity(
                role="successor",
                plan=plan,
                resource_policy_ids=[controller.SUCCESSOR_RESOURCE_POLICY_ID],
            ),
        },
        "refill_policy": controller.REFILL_POLICY,
        "successor_canary_task_ids_by_stage": canary_ids,
        "successor_canary_status_by_stage": {
            stage.stage_id: "remote_preflight_passed" for stage in profiles.STAGES
        },
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "cancellation_performed": False,
        "preemption_performed": False,
        "controller_stop_performed_by_this_tool": False,
    }
    sealed = controller._seal_state(state)
    return controller._validate_historical_resource_quota_successor_state(
        sealed, plan
    )


def _historical_transition_fixture(tmp_path: Path) -> dict:
    fixture = _fixture(tmp_path)
    patched_plan = fixture["successor"]
    historical_plan = _historical_resource_quota_successor(
        fixture["predecessor"], patched_plan
    )
    historical_state = _stopped_historical_resource_quota_state(historical_plan)
    predecessor_plan_path = tmp_path / "historical-resource-plan.json"
    predecessor_state_path = tmp_path / "historical-resource-state.json"
    successor_plan_path = tmp_path / "patched-plan.json"
    successor_state_path = tmp_path / "patched-state.json"
    predecessor_plan_path.write_text(json.dumps(historical_plan))
    predecessor_state_path.write_text(json.dumps(historical_state))
    successor_plan_path.write_text(json.dumps(patched_plan))
    return {
        "predecessor": historical_plan,
        "state": historical_state,
        "successor": patched_plan,
        "predecessor_plan_path": predecessor_plan_path,
        "predecessor_state_path": predecessor_state_path,
        "successor_plan_path": successor_plan_path,
        "successor_state_path": successor_state_path,
        "scheduler": _Scheduler(historical_plan, historical_state),
        "ready": _Ready(historical_plan, patched_plan),
    }


def _predecessor_state(plan: dict) -> dict:
    state = controller._initial_state(plan)
    terminal_stage_ids: set[str] = set()
    next_id = 70_000
    for entry in state["entries"]:
        next_id += 1
        entry["task_id"] = next_id
        if entry["stage_id"] not in terminal_stage_ids:
            terminal_stage_ids.add(entry["stage_id"])
            entry["state"] = "completed"
        else:
            entry["state"] = "running"
    state["stop_requested"] = True
    state["ramp_released"] = True
    state["canary_passed_stage_ids"] = sorted(profiles.BY_ID)
    state["scheduler_submit_count"] = 500
    return controller._seal_state(state)


class _Ready:
    def __init__(self, *plans: dict):
        self.read_count = 0
        self.values = {
            binding["remote_bundle"]: copy.deepcopy(binding["ready"])
            for plan in plans
            for binding in plan["stage_bindings"].values()
        }

    def read_ready(self, remote_bundle: str):
        self.read_count += 1
        return copy.deepcopy(self.values.get(remote_bundle))


class _Scheduler:
    def __init__(self, plan: dict, state: dict):
        tasks = {task["dedupe_key"]: task for task in migration._plan_tasks(plan)}
        self.by_id: dict[int, dict] = {}
        self.by_dedupe: dict[str, dict] = {}
        self.seed_status: dict[int, dict] = {}
        self.next_id = 80_000
        self.post_count = 0
        self.mutations: list[str] = []
        self.submitted_stage_ids: list[str] = []
        for entry in state["entries"]:
            task = copy.deepcopy(tasks[entry["dedupe_key"]])
            task.update(
                {
                    "id": int(entry["task_id"]),
                    "task_id": int(entry["task_id"]),
                    "status": entry["state"],
                }
            )
            self.by_id[task["id"]] = task
            self.by_dedupe[task["dedupe_key"]] = task

    def list_namespace_tasks(self):
        return [copy.deepcopy(task) for task in self.by_id.values()]

    def find_task_by_dedupe(self, dedupe_key: str):
        value = self.by_dedupe.get(dedupe_key)
        return copy.deepcopy(value) if value else None

    def submit_task(self, payload: dict):
        assert payload["cpus"] == 4
        assert payload["memory_mb"] == 28 * 1024
        assert payload["max_workers_per_node"] == 32
        assert payload["payload_json"]["inference_threads"] == 8
        self.next_id += 1
        task = copy.deepcopy(payload)
        task.update({"id": self.next_id, "task_id": self.next_id, "status": "queued"})
        self.by_id[self.next_id] = task
        self.by_dedupe[task["dedupe_key"]] = task
        self.post_count += 1
        self.mutations.append("POST /api/tasks")
        self.submitted_stage_ids.append(
            str(payload["payload_json"]["final_goal_stage_id"])
        )
        return copy.deepcopy(task)

    def get_task(self, task_id: int):
        value = self.by_id.get(task_id)
        return copy.deepcopy(value) if value else None

    def read_seed_status(self, task_id: int):
        return copy.deepcopy(self.seed_status.get(task_id))

    def pass_successor_canaries(self):
        for task_id, task in self.by_id.items():
            payload = task["payload_json"]
            if payload["lane"]["wave"] == "canary" and task["cpus"] == 4:
                self.seed_status[task_id] = {
                    "seed": payload["seed"],
                    "bundle_id": payload["bundle_id"],
                    "ramp_gate_passed": True,
                    "aedt_used": False,
                    "fea_submission_performed": False,
                }


def _fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    successor = _successor_plan()
    predecessor = _predecessor_plan(successor)
    state = _predecessor_state(predecessor)
    predecessor_plan_path = tmp_path / "predecessor-plan.json"
    predecessor_state_path = tmp_path / "predecessor-state.json"
    successor_plan_path = tmp_path / "successor-plan.json"
    successor_state_path = tmp_path / "successor-state.json"
    predecessor_plan_path.write_text(json.dumps(predecessor), encoding="utf-8")
    predecessor_state_path.write_text(json.dumps(state), encoding="utf-8")
    successor_plan_path.write_text(json.dumps(successor), encoding="utf-8")
    scheduler = _Scheduler(predecessor, state)
    ready = _Ready(predecessor, successor)
    return {
        "predecessor": predecessor,
        "successor": successor,
        "state": state,
        "predecessor_plan_path": predecessor_plan_path,
        "predecessor_state_path": predecessor_state_path,
        "successor_plan_path": successor_plan_path,
        "successor_state_path": successor_state_path,
        "scheduler": scheduler,
        "ready": ready,
    }


def _prepare(fixture: dict, *, apply: bool, **kwargs):
    return migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan_path"],
        predecessor_state_path=fixture["predecessor_state_path"],
        successor_plan_path=fixture["successor_plan_path"],
        successor_state_path=fixture["successor_state_path"],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        apply=apply,
        **kwargs,
    )


def _prepare_historical(fixture: dict, *, apply: bool = False):
    return migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan_path"],
        predecessor_state_path=fixture["predecessor_state_path"],
        successor_plan_path=fixture["successor_plan_path"],
        successor_state_path=fixture["successor_state_path"],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        apply=apply,
        transition_mode=migration.PATCHED_BUNDLE,
    )


def test_successor_policy_preserves_science_threads_and_unlocks_allocation_cap():
    plan = migration.validate_successor_plan(_successor_plan())
    assert plan["resources"] == {
        "cpus_per_task": 4,
        "memory_mb_per_task": 28 * 1024,
        "max_workers_per_node": 32,
        "priority": 1,
        "scheduling_profile": "standard",
        "gpus": 0,
    }
    assert all(
        task["cpus"] == 4
        and task["max_workers_per_node"] == 32
        and task["payload_json"]["inference_threads"] == 8
        for task in migration._plan_tasks(plan)
    )


def test_seed_status_http_429_retries_are_bounded(monkeypatch):
    attempts = []
    sleeps = []

    def busy_urlopen(_request, *, timeout):
        attempts.append(timeout)
        raise urllib.error.HTTPError(
            "http://scheduler/remote-file",
            429,
            "remote reads busy limit 4",
            None,
            io.BytesIO(b"remote reads busy limit 4"),
        )

    monkeypatch.setattr(controller.urllib.request, "urlopen", busy_urlopen)
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    client = controller.SchedulerApiClient("http://scheduler:8002", timeout=7)

    with pytest.raises(controller.SeedStatusReadBusy, match="after 3 attempts"):
        client.read_seed_status(12345)

    assert attempts == [7, 7, 7]
    assert sleeps == [0.25, 0.5]


def test_busy_seed_status_holds_canary_pending_without_false_pass(tmp_path):
    fixture = _fixture(tmp_path)

    class BusyScheduler(_Scheduler):
        def read_seed_status(self, task_id: int):
            raise controller.SeedStatusReadBusy(f"busy task {task_id}")

    scheduler = BusyScheduler(fixture["predecessor"], fixture["state"])
    result = controller.control_once(
        fixture["successor_plan_path"],
        state_path=tmp_path / "busy-controller-state.json",
        apply=True,
        scheduler=scheduler,
        ready_probe=fixture["ready"],
    )

    assert result["ramp_released"] is False
    assert result["canary_passed_stage_ids"] == []
    assert result["active_count"] == 500
    held = result["actions"][-1]
    assert held["action"] == "ramp_held"
    assert held["passed_stage_count"] == 0
    assert held["reasons"] == [
        f"{stage.stage_id}:remote_preflight_status_read_busy"
        for stage in profiles.STAGES
    ]


def test_prepare_dry_run_imports_every_entry_and_seed_without_writes(tmp_path):
    fixture = _fixture(tmp_path)
    value = _prepare(fixture, apply=False)
    imported = value["successor_state"]
    assert value["scheduler_post_count"] == 0
    assert value["predecessor_entry_count"] == 500
    assert value["imported_active_count"] == 496
    assert not fixture["successor_state_path"].exists()
    assert all(entry["origin"] == "predecessor" for entry in imported["entries"])
    assert imported["next_seed_by_stage"] == fixture["state"]["next_seed_by_stage"]
    assert (
        len({(entry["stage_id"], entry["seed"]) for entry in imported["entries"]})
        == 500
    )
    assert imported["rolling_migration"]["scheduler_mutation_endpoints"] == [
        "POST /api/tasks"
    ]


def test_resource_quota_only_handoff_reuses_exact_current_bundle_bindings(tmp_path):
    fixture = _fixture(tmp_path)
    resource_plan = _resource_only_successor(
        fixture["predecessor"], fixture["successor"]
    )
    fixture["successor"] = resource_plan
    fixture["successor_plan_path"].write_text(json.dumps(resource_plan))
    fixture["ready"] = _Ready(fixture["predecessor"], resource_plan)
    value = _prepare(
        fixture,
        apply=False,
        transition_mode=migration.RESOURCE_QUOTA_ONLY,
    )
    assert value["transition_mode"] == migration.RESOURCE_QUOTA_ONLY
    assert value["scheduler_post_count"] == 0
    assert value["successor_state"]["rolling_migration"]["transition_mode"] == (
        migration.RESOURCE_QUOTA_ONLY
    )
    cohorts = value["successor_state"]["rolling_migration"]["harvest_cohorts"]
    assert set(cohorts) == {"predecessor", "successor"}
    assert cohorts["predecessor"]["resource_policy_ids"] == [
        controller.LEGACY_RESOURCE_POLICY_ID
    ]
    assert cohorts["successor"]["resource_policy_ids"] == [
        controller.SUCCESSOR_RESOURCE_POLICY_ID
    ]
    assert len(cohorts["predecessor"]["stage_bindings"]) == 4
    assert len(cohorts["successor"]["stage_bindings"]) == 4
    assert len(json.dumps(cohorts, sort_keys=True)) < 10_000
    for stage in profiles.STAGES:
        assert (
            resource_plan["stage_bindings"][stage.stage_id]
            == fixture["predecessor"]["stage_bindings"][stage.stage_id]
        )


def test_stopped_resource_successor_can_dual_bind_patched_bundles(tmp_path):
    fixture = _fixture(tmp_path)
    patched_plan = fixture["successor"]
    resource_plan = _historical_resource_quota_successor(
        fixture["predecessor"], patched_plan
    )
    resource_state = _stopped_historical_resource_quota_state(resource_plan)
    resource_plan_path = tmp_path / "historical-resource-plan.json"
    resource_state_path = tmp_path / "historical-resource-state.json"
    resource_plan_path.write_text(json.dumps(resource_plan))
    resource_state_path.write_text(json.dumps(resource_state))
    scheduler = _Scheduler(resource_plan, resource_state)
    ready = _Ready(resource_plan, patched_plan)

    patched_path = tmp_path / "patched-plan.json"
    patched_state_path = tmp_path / "patched-state.json"
    patched_path.write_text(json.dumps(patched_plan))
    value = migration.prepare_successor_state(
        predecessor_plan_path=resource_plan_path,
        predecessor_state_path=resource_state_path,
        successor_plan_path=patched_path,
        successor_state_path=patched_state_path,
        scheduler=scheduler,
        predecessor_ready_probe=ready,
        successor_ready_probe=ready,
        apply=False,
        transition_mode=migration.PATCHED_BUNDLE,
    )
    assert value["transition_mode"] == migration.PATCHED_BUNDLE
    assert value["scheduler_post_count"] == 0
    assert value["predecessor_entry_count"] == 500
    assert value["imported_active_count"] == 500
    assert value["imported_active_count_by_stage"] == (
        controller.HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS
    )
    assert value["next_seed_by_stage"] == resource_state["next_seed_by_stage"]
    assert value["scheduler_post_count"] == 0
    assert scheduler.post_count == 0
    assert scheduler.mutations == []
    assert value["cancellation_performed"] is False
    assert value["preemption_performed"] is False
    assert (
        value["successor_state"]["rolling_migration"]["predecessor_controller_kind"]
        == "resource_quota_successor"
    )
    assert all(
        entry["origin"] == "predecessor"
        for entry in value["successor_state"]["entries"]
    )
    cohorts = value["successor_state"]["rolling_migration"]["harvest_cohorts"]
    assert cohorts["predecessor"]["resource_policy_ids"] == [
        controller.SUCCESSOR_RESOURCE_POLICY_ID
    ]
    assert cohorts["successor"]["resource_policy_ids"] == [
        controller.SUCCESSOR_RESOURCE_POLICY_ID
    ]
    for stage in profiles.STAGES:
        assert (
            cohorts["predecessor"]["stage_bindings"][stage.stage_id]["bundle_id"]
            != cohorts["successor"]["stage_bindings"][stage.stage_id]["bundle_id"]
        )


def test_historical_transition_rejects_wrong_predecessor_quota(tmp_path):
    fixture = _historical_transition_fixture(tmp_path)
    plan = copy.deepcopy(fixture["predecessor"])
    quotas = plan["open_ended_refill"]["stage_active_quotas"]
    quotas["entry-1200-t125"] += 1
    quotas["final-1000-t100"] -= 1
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    fixture["predecessor_plan_path"].write_text(json.dumps(plan))

    with pytest.raises(RuntimeError, match="launch-plan seal mismatch"):
        _prepare_historical(fixture)


def test_historical_transition_rejects_wrong_successor_quota(tmp_path):
    fixture = _historical_transition_fixture(tmp_path)
    plan = copy.deepcopy(fixture["successor"])
    quotas = plan["open_ended_refill"]["stage_active_quotas"]
    quotas["entry-1200-t125"] -= 1
    quotas["final-1000-t100"] += 1
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    fixture["successor_plan_path"].write_text(json.dumps(plan))

    with pytest.raises(RuntimeError, match="launch-plan top-level seal mismatch"):
        _prepare_historical(fixture)


def test_historical_transition_rejects_state_plan_seal_mismatch(tmp_path):
    fixture = _historical_transition_fixture(tmp_path)
    state = copy.deepcopy(fixture["state"])
    state["launch_plan_sha256"] = "b" * 64
    state = controller._seal_state(state)
    fixture["predecessor_state_path"].write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="state identity/SHA mismatch"):
        _prepare_historical(fixture)


def test_historical_transition_rejects_active_total_above_500(tmp_path):
    fixture = _historical_transition_fixture(tmp_path)
    state = copy.deepcopy(fixture["state"])
    stage = profiles.BY_ID["entry-1200-t125"]
    seed = int(state["next_seed_by_stage"][stage.stage_id])
    template = next(
        task
        for task in fixture["predecessor"]["task_waves"]["canaries"]
        if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
    )
    task = migration._render_from_template_for_policy(
        template,
        stage_id=stage.stage_id,
        seed=seed,
        wave="refill",
        policy=migration.SUCCESSOR_POLICY,
    )
    task_id = max(int(entry["task_id"]) for entry in state["entries"]) + 1
    entry = controller._entry(task, origin="successor")
    entry.update({"task_id": task_id, "state": "running"})
    state["entries"].append(entry)
    state["next_seed_by_stage"][stage.stage_id] = seed + 1
    state["rolling_migration"]["next_seed_by_stage"] = copy.deepcopy(
        state["next_seed_by_stage"]
    )
    state = controller._seal_state(state)
    fixture["predecessor_state_path"].write_text(json.dumps(state))
    row = copy.deepcopy(task)
    row.update({"id": task_id, "task_id": task_id, "status": "running"})
    fixture["scheduler"].by_id[task_id] = row
    fixture["scheduler"].by_dedupe[row["dedupe_key"]] = row

    with pytest.raises(RuntimeError, match="exceeds total active quota"):
        _prepare_historical(fixture)


def test_historical_transition_rejects_cancellation_endpoint(tmp_path):
    fixture = _historical_transition_fixture(tmp_path)
    state = copy.deepcopy(fixture["state"])
    state["scheduler_mutation_endpoints"] = [
        "POST /api/tasks",
        "POST /api/tasks/{id}/cancel",
    ]
    state = controller._seal_state(state)
    fixture["predecessor_state_path"].write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="state identity/SHA mismatch"):
        _prepare_historical(fixture)


def test_controller_rejects_entry_that_crosses_harvest_cohort_boundary(tmp_path):
    fixture = _fixture(tmp_path)
    value = _prepare(fixture, apply=False)
    state = value["successor_state"]
    entry = next(item for item in state["entries"] if item["origin"] == "predecessor")
    stage_id = entry["stage_id"]
    entry["bundle_id"] = state["rolling_migration"]["harvest_cohorts"]["successor"][
        "stage_bindings"
    ][stage_id]["bundle_id"]
    state = controller._seal_state(state)

    with pytest.raises(RuntimeError, match="ledger cohort drifted"):
        controller._validate_state(state, fixture["successor"])


def test_controller_rejects_tampered_harvest_cohort_seal(tmp_path):
    fixture = _fixture(tmp_path)
    value = _prepare(fixture, apply=False)
    state = value["successor_state"]
    state["rolling_migration"]["harvest_cohorts"]["predecessor"][
        "resource_policy_ids"
    ] = [controller.SUCCESSOR_RESOURCE_POLICY_ID]
    state = controller._seal_state(state)

    with pytest.raises(RuntimeError, match="harvest cohort seal mismatch"):
        controller._validate_state(state, fixture["successor"])


def test_live_phase1_state_without_catalog_derives_same_bundle_cohorts(tmp_path):
    fixture = _fixture(tmp_path)
    resource_plan = _resource_only_successor(
        fixture["predecessor"], fixture["successor"]
    )
    fixture["successor"] = resource_plan
    fixture["successor_plan_path"].write_text(json.dumps(resource_plan))
    fixture["ready"] = _Ready(fixture["predecessor"], resource_plan)
    value = _prepare(
        fixture,
        apply=False,
        transition_mode=migration.RESOURCE_QUOTA_ONLY,
    )
    live_compatible = copy.deepcopy(value["successor_state"])
    live_compatible["rolling_migration"].pop("harvest_cohorts")
    live_compatible = controller._seal_state(live_compatible)

    validated = controller._validate_state(live_compatible, resource_plan)
    assert validated["entries"] == live_compatible["entries"]
    assert validated["next_seed_by_stage"] == live_compatible["next_seed_by_stage"]
    assert "harvest_cohorts" not in validated["rolling_migration"]
    derived = controller._derived_resource_only_harvest_cohorts(
        resource_plan, validated["rolling_migration"]
    )
    assert (
        derived["predecessor"]["stage_bindings"]
        == derived["successor"]["stage_bindings"]
    )
    assert derived["predecessor"]["resource_policy_ids"] == [
        controller.LEGACY_RESOURCE_POLICY_ID
    ]
    assert derived["successor"]["resource_policy_ids"] == [
        controller.SUCCESSOR_RESOURCE_POLICY_ID
    ]

    invalid = copy.deepcopy(live_compatible)
    invalid["rolling_migration"]["transition_mode"] = migration.PATCHED_BUNDLE
    invalid = controller._seal_state(invalid)
    with pytest.raises(RuntimeError, match="derivation is not applicable"):
        controller._validate_state(invalid, resource_plan)


def test_controller_keeps_canary_pending_on_transient_remote_status_429(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path)
    _prepare(fixture, apply=True)
    controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )

    def busy(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "http://scheduler/remote-file",
            429,
            "busy",
            {},
            io.BytesIO(b'{"detail":"remote read busy"}'),
        )

    monkeypatch.setattr(current7_controller.urllib.request, "urlopen", busy)
    api = controller.SchedulerApiClient("http://scheduler")
    fixture["scheduler"].read_seed_status = api.read_seed_status
    value = controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )

    assert value["ramp_released"] is False
    assert value["active_count"] == 500
    assert value["canary_passed_stage_ids"] == []
    assert set(value["successor_canary_status_by_stage"].values()) == {
        "remote_preflight_pending"
    }
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 4


def test_prepare_fails_closed_if_predecessor_state_drifts(tmp_path):
    fixture = _fixture(tmp_path)

    def drift():
        state = json.loads(fixture["predecessor_state_path"].read_text())
        state["revision"] += 1
        state = controller._seal_state(state)
        fixture["predecessor_state_path"].write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="drifted during migration"):
        _prepare(fixture, apply=True, before_final_state_read=drift)
    assert not fixture["successor_state_path"].exists()
    assert fixture["scheduler"].mutations == []


def test_prepare_state_write_is_crash_atomic(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)

    def crash(_source, _target):
        raise OSError("injected replace crash")

    monkeypatch.setattr(controller.os, "replace", crash)
    with pytest.raises(OSError, match="injected replace crash"):
        _prepare(fixture, apply=True)
    assert not fixture["successor_state_path"].exists()
    assert not list(tmp_path.glob(".successor-state.json.*.tmp"))
    assert fixture["scheduler"].mutations == []


def test_rolling_controller_fills_only_natural_gaps_and_keeps_exact_500(tmp_path):
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture, apply=True)
    old_next = copy.deepcopy(prepared["next_seed_by_stage"])

    first = controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert first["rolling_migration"] is True
    assert first["active_count"] == 500
    assert first["active_count_by_stage"] == {
        stage.stage_id: stage.active_quota for stage in profiles.STAGES
    }
    assert first["scheduler_post_count"] == 4
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 4
    successor_canaries = [
        task
        for task in fixture["scheduler"].by_id.values()
        if task["payload_json"]["bundle_id"].startswith("successor-")
    ]
    assert {
        (task["payload_json"]["final_goal_stage_id"], task["payload_json"]["seed"])
        for task in successor_canaries
    } == set(old_next.items())

    fixture["scheduler"].pass_successor_canaries()
    released = controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert released["ramp_released"] is True
    assert released["active_count"] == 500
    assert set(released["canary_passed_stage_ids"]) == set(profiles.BY_ID)

    # Drain only the predecessor excess (close +38, final +162).  The
    # successor must transfer those 200 naturally vacated slots into entry
    # +136 and bridge +64 without cancelling anything.
    remaining = {"close-1075-t107p5": 38, "final-1000-t100": 162}
    for task in fixture["scheduler"].by_id.values():
        stage_id = task["payload_json"]["final_goal_stage_id"]
        if (
            task["payload_json"]["bundle_id"].startswith("predecessor-")
            and task["status"] == "running"
            and remaining.get(stage_id, 0) > 0
        ):
            task["status"] = "completed"
            fixture["scheduler"].by_dedupe[task["dedupe_key"]]["status"] = "completed"
            remaining[stage_id] -= 1
    assert remaining == {"close-1075-t107p5": 0, "final-1000-t100": 0}
    refilled = controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert refilled["active_count"] == 500
    assert refilled["active_count_by_stage"] == launch.SUCCESSOR_ACTIVE_QUOTAS
    assert refilled["scheduler_post_count"] == 204
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 204
    transfer_order = fixture["scheduler"].submitted_stage_ids[4:]
    assert transfer_order.count("entry-1200-t125") == 136
    assert transfer_order.count("bridge-1150-t115") == 64
    assert "close-1075-t107p5" not in transfer_order
    assert "final-1000-t100" not in transfer_order
    # Weighted scheduling must interleave while both deficits remain; it may
    # not append one complete stage block followed by the other.
    assert len(set(transfer_order[:12])) == 2
    assert any(
        transfer_order[index] != transfer_order[index + 1] for index in range(20)
    )
    assert refilled["cancellation_performed"] is False
    assert not hasattr(controller.SchedulerApiClient, "cancel_task")
    assert not hasattr(controller.SchedulerApiClient, "preempt_task")


def test_prepare_rejects_unstopped_controller_and_scheduler_identity_tamper(tmp_path):
    fixture = _fixture(tmp_path)
    state = json.loads(fixture["predecessor_state_path"].read_text())
    state["stop_requested"] = False
    state = controller._seal_state(state)
    fixture["predecessor_state_path"].write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="stopped predecessor"):
        _prepare(fixture, apply=False)

    fixture = _fixture(tmp_path / "scheduler-tamper")
    first = next(iter(fixture["scheduler"].by_id.values()))
    first["cpus"] = 1
    with pytest.raises(RuntimeError, match="seal differs from ledger"):
        _prepare(fixture, apply=False)
