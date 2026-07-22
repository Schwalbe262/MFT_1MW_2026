from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import urllib.error

import pytest

from tools import tier1_final1000_rolling_migration as migration
from tools import tier1_corrected_current7_slurm_controller as current7_controller
from tools import tier1_corrected_current7_slurm_bundle as single_contract
from tools import tier1_final1000_multiseed_contract as multiseed_contract
from tools import tier1_final1000_multiseed_controller as multiseed_controller
from tools import tier1_final1000_multiseed_driver as multiseed_driver
from tools import tier1_final1000_multiseed_monitor as multiseed_monitor
from tools import tier1_final1000_multiseed_status as multiseed_status
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
        self.recent_inventory_read_count = 0
        self.complete_inventory_read_count = 0
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
        self.recent_inventory_read_count += 1
        return [copy.deepcopy(task) for task in self.by_id.values()]

    def list_complete_namespace_tasks(self):
        self.complete_inventory_read_count += 1
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


def _plan_for_generations(plan: dict, generations: int) -> dict:
    value = copy.deepcopy(plan)
    value["stage_inventory"] = migration._stage_inventory_for_generations(
        generations
    )
    policy, _quotas = migration._plan_policy_and_quotas(value)
    for wave_name in ("canaries", "ramp"):
        rendered = []
        for source in value["task_waves"][wave_name]:
            template = copy.deepcopy(source)
            payload = template["payload_json"]
            stage_id = payload["final_goal_stage_id"]
            payload["max_generations"] = generations
            payload["final_goal_stage_profile_sha256"] = (
                migration._stage_profile_for_generations(stage_id, generations)[
                    "sha256"
                ]
            )
            rendered.append(
                migration._render_from_template_for_policy(
                    template,
                    stage_id=stage_id,
                    seed=int(payload["seed"]),
                    wave=str(payload["lane"]["wave"]),
                    policy=policy,
                    fixed_generations=generations,
                )
            )
        value["task_waves"][wave_name] = rendered
    unsigned = {
        key: item for key, item in value.items() if key != "launch_plan_sha256"
    }
    value["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return migration.validate_chained_predecessor_plan(value)


def _rebundle_plan(plan: dict, prefix: str) -> dict:
    value = copy.deepcopy(plan)
    generations = migration._plan_fixed_generations(value)
    policy, _quotas = migration._plan_policy_and_quotas(value)
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in value["task_waves"]["canaries"]
    }
    for stage in profiles.STAGES:
        binding = value["stage_bindings"][stage.stage_id]
        binding["bundle_id"] = f"{prefix}-{stage.stage_id}"
        binding["bundle_manifest_sha256"] = launch.canonical_sha256(
            {"prefix": prefix, "stage_id": stage.stage_id, "kind": "manifest"}
        )
        binding["remote_bundle"] = f"/gpfs/{prefix}/{stage.stage_id}"
        binding["publication_receipt_sha256"] = launch.canonical_sha256(
            {"prefix": prefix, "stage_id": stage.stage_id, "kind": "receipt"}
        )
        binding["ready"]["bundle_id"] = binding["bundle_id"]
        binding["ready_sha256"] = launch.canonical_sha256(binding["ready"])
    for wave_name in ("canaries", "ramp"):
        rendered = []
        for source in value["task_waves"][wave_name]:
            payload = source["payload_json"]
            stage_id = payload["final_goal_stage_id"]
            binding = value["stage_bindings"][stage_id]
            template = copy.deepcopy(templates[stage_id])
            template["payload_json"]["bundle_id"] = binding["bundle_id"]
            template["payload_json"]["bundle_manifest_sha256"] = binding[
                "bundle_manifest_sha256"
            ]
            template["remote_cwd"] = binding["remote_bundle"]
            rendered.append(
                migration._render_from_template_for_policy(
                    template,
                    stage_id=stage_id,
                    seed=int(payload["seed"]),
                    wave=str(payload["lane"]["wave"]),
                    policy=policy,
                    fixed_generations=generations,
                )
            )
        value["task_waves"][wave_name] = rendered
    unsigned = {
        key: item for key, item in value.items() if key != "launch_plan_sha256"
    }
    value["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return value


def _chained_fixture(tmp_path: Path, *, stopped: bool) -> dict:
    base_successor = _successor_plan()
    legacy = _predecessor_plan(base_successor)
    historical = _historical_resource_quota_successor(legacy, base_successor)
    historical = _plan_for_generations(historical, 200)
    current = _plan_for_generations(_rebundle_plan(base_successor, "current"), 200)
    gen300 = migration.validate_successor_plan(
        _rebundle_plan(base_successor, "gen300")
    )

    entries = []
    expected_tasks: dict[int, dict] = {}
    task_id = 100_000
    for task in migration._plan_tasks(historical):
        task_id += 1
        payload = task["payload_json"]
        entries.append(
            {
                "stage_id": payload["final_goal_stage_id"],
                "bundle_id": payload["bundle_id"],
                "seed": int(payload["seed"]),
                "wave": payload["lane"]["wave"],
                "dedupe_key": task["dedupe_key"],
                "task_id": task_id,
                "state": "completed",
                "origin": "predecessor",
                "resource_policy_id": controller.SUCCESSOR_RESOURCE_POLICY_ID,
            }
        )
        expected_tasks[task_id] = task

    current_templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in current["task_waves"]["canaries"]
    }
    next_seed_by_stage = {}
    canary_ids = {}
    for stage in profiles.STAGES:
        stage_ids = []
        quota = launch.SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
        start = stage.seed_start + 10_000
        for offset in range(quota):
            task_id += 1
            wave = "canary" if offset == 0 else "refill"
            task = migration._render_from_template_for_policy(
                current_templates[stage.stage_id],
                stage_id=stage.stage_id,
                seed=start + offset,
                wave=wave,
                policy=migration.SUCCESSOR_POLICY,
                fixed_generations=200,
            )
            payload = task["payload_json"]
            entries.append(
                {
                    "stage_id": stage.stage_id,
                    "bundle_id": payload["bundle_id"],
                    "seed": int(payload["seed"]),
                    "wave": wave,
                    "dedupe_key": task["dedupe_key"],
                    "task_id": task_id,
                    "state": "running",
                    "origin": "successor",
                    "resource_policy_id": controller.SUCCESSOR_RESOURCE_POLICY_ID,
                }
            )
            expected_tasks[task_id] = task
            if wave == "canary":
                stage_ids.append(task_id)
        canary_ids[stage.stage_id] = stage_ids
        next_seed_by_stage[stage.stage_id] = start + quota

    migration_value = {
        "schema_version": migration.MIGRATION_SCHEMA,
        "transition_mode": migration.PATCHED_BUNDLE,
        "predecessor_controller_kind": "resource_quota_successor",
        "predecessor_launch_plan_sha256": historical["launch_plan_sha256"],
        "predecessor_state_sha256": "f" * 64,
        "predecessor_state_revision": 1,
        "successor_launch_plan_sha256": current["launch_plan_sha256"],
        "scheduler_inventory_sha256": "e" * 64,
        "predecessor_entry_count": 500,
        "imported_active_count_by_stage": {
            stage.stage_id: 0 for stage in profiles.STAGES
        },
        "next_seed_by_stage": copy.deepcopy(next_seed_by_stage),
        "successor_resource_policy": copy.deepcopy(migration.SUCCESSOR_POLICY),
        "successor_active_quotas": copy.deepcopy(launch.SUCCESSOR_ACTIVE_QUOTAS),
        "harvest_cohorts": {
            "predecessor": migration.harvest_cohort_identity(
                role="predecessor",
                plan=historical,
                resource_policy_ids=[controller.SUCCESSOR_RESOURCE_POLICY_ID],
            ),
            "successor": migration.harvest_cohort_identity(
                role="successor",
                plan=current,
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
    state = controller._seal_state(
        {
            "schema_version": controller.STATE_SCHEMA,
            "launch_plan_sha256": current["launch_plan_sha256"],
            "revision": 9,
            "parent_state_sha256": "d" * 64,
            "stop_requested": stopped,
            "ramp_released": True,
            "canary_passed_stage_ids": sorted(profiles.BY_ID),
            "entries": entries,
            "next_seed_by_stage": next_seed_by_stage,
            "scheduler_name_prefix": controller.TASK_NAME_PREFIX,
            "scheduler_dedupe_prefix": controller.DEDUPE_PREFIX,
            "scheduler_mutation_endpoints": ["POST /api/tasks"],
            "scheduler_submit_count": len(entries),
            "refill_policy": controller.REFILL_POLICY,
            "refill_stage_cursor": 0,
            "refill_deficit_credit_by_stage": {
                stage.stage_id: 0 for stage in profiles.STAGES
            },
            "rolling_migration": migration_value,
        }
    )

    class ChainedScheduler:
        def __init__(self):
            self.by_id = {}
            self.by_dedupe = {}
            self.seed_status = {}
            self.post_count = 0
            self.mutations = []
            self.next_id = 200_000
            for entry in entries:
                task = copy.deepcopy(expected_tasks[int(entry["task_id"])])
                task.update(
                    {
                        "id": int(entry["task_id"]),
                        "task_id": int(entry["task_id"]),
                        "status": entry["state"],
                    }
                )
                self.by_id[task["id"]] = task
                self.by_dedupe[task["dedupe_key"]] = task
            self.by_id[72164] = {
                "id": 72164,
                "task_id": 72164,
                "name": "mft-t1fg-c-refill-2457499999",
                "dedupe_key": "mft-tier1-final1000:" + "9" * 64,
                "status": "running",
            }

        def list_namespace_tasks(self):
            return copy.deepcopy(list(self.by_id.values()))

        def find_task_by_dedupe(self, dedupe_key: str):
            return copy.deepcopy(self.by_dedupe.get(dedupe_key))

        def submit_task(self, payload: dict):
            self.next_id += 1
            task = copy.deepcopy(payload)
            task.update(
                {"id": self.next_id, "task_id": self.next_id, "status": "queued"}
            )
            self.by_id[self.next_id] = task
            self.by_dedupe[task["dedupe_key"]] = task
            self.post_count += 1
            self.mutations.append("POST /api/tasks")
            return copy.deepcopy(task)

        def get_task(self, task_id: int):
            return copy.deepcopy(self.by_id.get(task_id))

        def read_seed_status(self, task_id: int):
            return copy.deepcopy(self.seed_status.get(task_id))

    paths = {}
    for name, value in (
        ("ancestor_plan", historical),
        ("predecessor_plan", current),
        ("predecessor_state", state),
        ("successor_plan", gen300),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    paths["successor_state"] = tmp_path / "successor_state.json"
    return {
        "ancestor": historical,
        "predecessor": current,
        "state": state,
        "successor": gen300,
        "scheduler": ChainedScheduler(),
        "ready": _Ready(historical, current, gen300),
        **paths,
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


def test_multiseed_upgrade_observes_exact_rolling_predecessor_envelope(tmp_path):
    fixture = _fixture(tmp_path)
    # Replace one active planned-wave task with a genuine predecessor refill
    # that is absent from both launch-plan task waves.  This is the identity
    # that cannot be reconstructed from the successor template after handoff.
    predecessor_state = copy.deepcopy(fixture["state"])
    source_entry = next(
        item for item in predecessor_state["entries"] if item["state"] == "running"
    )
    stage_id = source_entry["stage_id"]
    refill_seed = int(predecessor_state["next_seed_by_stage"][stage_id])
    template = next(
        task
        for task in fixture["predecessor"]["task_waves"]["canaries"]
        if task["payload_json"]["final_goal_stage_id"] == stage_id
    )
    refill_task = migration._render_from_template(
        template,
        stage_id=stage_id,
        seed=refill_seed,
        wave="refill",
    )
    old_dedupe = source_entry["dedupe_key"]
    source_entry.update(
        bundle_id=refill_task["payload_json"]["bundle_id"],
        seed=refill_seed,
        wave="refill",
        dedupe_key=refill_task["dedupe_key"],
    )
    predecessor_state["next_seed_by_stage"][stage_id] = refill_seed + 1
    predecessor_state = controller._seal_state(predecessor_state)
    fixture["state"] = predecessor_state
    fixture["predecessor_state_path"].write_text(
        json.dumps(predecessor_state), encoding="utf-8"
    )
    task_id = int(source_entry["task_id"])
    scheduler_task = copy.deepcopy(refill_task)
    scheduler_task.update({"id": task_id, "task_id": task_id, "status": "running"})
    fixture["scheduler"].by_dedupe.pop(old_dedupe)
    fixture["scheduler"].by_dedupe[refill_task["dedupe_key"]] = scheduler_task
    fixture["scheduler"].by_id[task_id] = scheduler_task

    prepared = _prepare(fixture, apply=False)
    rolling_state = prepared["successor_state"]

    with pytest.raises(RuntimeError, match="predecessor plan is required"):
        multiseed_controller.upgrade_v1_state(
            rolling_state,
            fixture["successor"],
            batch_length=4,
        )

    state = multiseed_controller.upgrade_v1_state(
        rolling_state,
        fixture["successor"],
        predecessor_plan=fixture["predecessor"],
        batch_length=4,
    )
    entry = next(
        item
        for item in state["entries"]
        if item["origin"] == "predecessor"
        and item["wave"] == "refill"
        and item["seeds"] == [refill_seed]
    )
    task = copy.deepcopy(entry["task_envelope"])
    assert task["payload_json"]["bundle_id"] == entry["bundle_id"]
    assert task["payload_json"]["bundle_id"].startswith("predecessor-")
    assert entry["source_launch_plan_sha256"] == fixture["predecessor"][
        "launch_plan_sha256"
    ]
    observed = multiseed_controller.observe_scheduler_task(
        state,
        fixture["successor"],
        parent_dedupe_key=entry["parent_dedupe_key"],
        scheduler_task={
            "id": entry["task_id"],
            "task_id": entry["task_id"],
            "name": task["name"],
            "dedupe_key": task["dedupe_key"],
            "status": "running",
            "state": "running",
            "task_json": task,
        },
    )
    observed_entry = next(
        item
        for item in observed["entries"]
        if item["parent_dedupe_key"] == entry["parent_dedupe_key"]
    )
    assert observed_entry["state"] == "running"


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


def test_migration_cli_alone_uses_complete_inventory_reader(
    monkeypatch, tmp_path, capsys
):
    constructed: list[tuple[str, object]] = []

    class CompleteReader:
        def __init__(self, scheduler_url):
            constructed.append((scheduler_url, self))

    class TransportContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    def prepare(**kwargs):
        assert kwargs["scheduler"] is constructed[0][1]
        return {
            "apply": False,
            "scheduler_post_count": 0,
            "successor_state": {},
        }

    monkeypatch.setattr(
        migration, "CompleteInventorySchedulerApiClient", CompleteReader
    )
    monkeypatch.setattr(
        migration,
        "scheduler_publication_transport",
        lambda **_kwargs: TransportContext(),
    )
    monkeypatch.setattr(migration, "prepare_successor_state", prepare)
    migration.main(
        [
            "--predecessor-plan",
            str(tmp_path / "predecessor-plan.json"),
            "--predecessor-state",
            str(tmp_path / "predecessor-state.json"),
            "--successor-plan",
            str(tmp_path / "successor-plan.json"),
            "--successor-state",
            str(tmp_path / "successor-state.json"),
            "--scheduler-url",
            "http://scheduler:8002",
        ]
    )

    assert constructed == [("http://scheduler:8002", constructed[0][1])]
    assert json.loads(capsys.readouterr().out) == {
        "apply": False,
        "scheduler_post_count": 0,
    }


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
    assert fixture["scheduler"].complete_inventory_read_count == 1
    assert fixture["scheduler"].recent_inventory_read_count == 0
    imported = value["successor_state"]
    assert value["scheduler_post_count"] == 0
    assert value["predecessor_entry_count"] == 500
    assert value["imported_active_count"] == 496
    assert not fixture["successor_state_path"].exists()
    assert all(entry["origin"] == "predecessor" for entry in imported["entries"])
    planned_seeds = {
        stage.stage_id: {
            int(task["payload_json"]["seed"])
            for wave in ("canaries", "ramp")
            for task in fixture["successor"]["task_waves"][wave]
            if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
        }
        for stage in profiles.STAGES
    }
    assert imported["next_seed_by_stage"] == {
        stage.stage_id: max(
            int(fixture["state"]["next_seed_by_stage"][stage.stage_id]),
            max(planned_seeds[stage.stage_id]) + 1,
        )
        for stage in profiles.STAGES
    }
    assert all(
        imported["next_seed_by_stage"][stage.stage_id]
        not in planned_seeds[stage.stage_id]
        for stage in profiles.STAGES
    )
    assert (
        len({(entry["stage_id"], entry["seed"]) for entry in imported["entries"]})
        == 500
    )
    assert imported["rolling_migration"]["scheduler_mutation_endpoints"] == [
        "POST /api/tasks"
    ]


def test_chained_live_shadow_imports_all_cohorts_without_external_task(tmp_path):
    fixture = _chained_fixture(tmp_path, stopped=False)
    value = migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan"],
        predecessor_state_path=fixture["predecessor_state"],
        successor_plan_path=fixture["successor_plan"],
        successor_state_path=fixture["successor_state"],
        ancestor_plan_paths=[fixture["ancestor_plan"]],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        allow_running_predecessor_shadow=True,
    )

    shadow = value["successor_state"]
    assert value["schema_version"] == migration.CHAINED_MIGRATION_SCHEMA
    assert value["shadow_only"] is True
    assert value["cutover_ready"] is False
    assert value["scheduler_post_count"] == 0
    assert value["predecessor_entry_count"] == 1_000
    assert value["imported_active_count"] == 500
    assert value["imported_active_count_by_stage"] == launch.SUCCESSOR_ACTIVE_QUOTAS
    assert value["namespace_extra_task_ids"] == [72164]
    assert not fixture["successor_state"].exists()
    assert 72164 not in {int(entry["task_id"]) for entry in shadow["entries"]}
    assert all(entry["origin"] == "predecessor" for entry in shadow["entries"])
    assert len({entry["harvest_cohort_id"] for entry in shadow["entries"]}) == 2
    assert len(shadow["rolling_migration"]["harvest_cohorts"]) == 3
    assert shadow["rolling_migration"]["successor_canary_status_by_stage"] == {
        stage.stage_id: "waiting_for_natural_terminal_gap"
        for stage in profiles.STAGES
    }
    assert shadow["rolling_migration"]["successor_canary_task_ids_by_stage"] == {
        stage.stage_id: [] for stage in profiles.STAGES
    }
    with pytest.raises(RuntimeError, match="not cutover-ready"):
        controller._validate_state(shadow, fixture["successor"])


def test_chained_live_shadow_can_never_apply(tmp_path):
    fixture = _chained_fixture(tmp_path, stopped=False)
    with pytest.raises(RuntimeError, match="can never be applied"):
        migration.prepare_successor_state(
            predecessor_plan_path=fixture["predecessor_plan"],
            predecessor_state_path=fixture["predecessor_state"],
            successor_plan_path=fixture["successor_plan"],
            successor_state_path=fixture["successor_state"],
            ancestor_plan_paths=[fixture["ancestor_plan"]],
            scheduler=fixture["scheduler"],
            predecessor_ready_probe=fixture["ready"],
            successor_ready_probe=fixture["ready"],
            allow_running_predecessor_shadow=True,
            apply=True,
        )
    assert fixture["scheduler"].post_count == 0
    assert not fixture["successor_state"].exists()


def test_chained_predecessor_requires_every_ancestor_plan(tmp_path):
    fixture = _chained_fixture(tmp_path, stopped=True)
    with pytest.raises(RuntimeError, match="cohort plan is missing"):
        migration.prepare_successor_state(
            predecessor_plan_path=fixture["predecessor_plan"],
            predecessor_state_path=fixture["predecessor_state"],
            successor_plan_path=fixture["successor_plan"],
            successor_state_path=fixture["successor_state"],
            scheduler=fixture["scheduler"],
            predecessor_ready_probe=fixture["ready"],
            successor_ready_probe=fixture["ready"],
        )
    assert fixture["scheduler"].post_count == 0


def test_chained_cutover_waits_for_four_natural_gaps(tmp_path):
    fixture = _chained_fixture(tmp_path, stopped=True)
    prepared = migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan"],
        predecessor_state_path=fixture["predecessor_state"],
        successor_plan_path=fixture["successor_plan"],
        successor_state_path=fixture["successor_state"],
        ancestor_plan_paths=[fixture["ancestor_plan"]],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        apply=True,
    )
    assert prepared["cutover_ready"] is True
    assert prepared["scheduler_post_count"] == 0
    validated = controller._validate_state(
        json.loads(fixture["successor_state"].read_text()), fixture["successor"]
    )
    assert sum(
        entry["state"] in controller.ACTIVE_STATES
        for entry in validated["entries"]
    ) == 500

    for stage in profiles.STAGES:
        entry = next(
            item
            for item in validated["entries"]
            if item["stage_id"] == stage.stage_id
            and item["state"] in controller.ACTIVE_STATES
        )
        fixture["scheduler"].by_id[int(entry["task_id"])]["status"] = "completed"

    result = controller.control_once(
        fixture["successor_plan"],
        state_path=fixture["successor_state"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert result["active_count"] == 500
    assert result["ramp_released"] is False
    assert fixture["scheduler"].post_count == 4
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 4
    assert result["successor_canary_status_by_stage"] == {
        stage.stage_id: "remote_preflight_pending" for stage in profiles.STAGES
    }
    assert all(
        len(result["successor_canary_task_ids_by_stage"][stage.stage_id]) == 1
        for stage in profiles.STAGES
    )


def test_chained_cutover_holds_extra_gaps_until_exactly_four_canaries_pass(
    tmp_path,
):
    fixture = _chained_fixture(tmp_path, stopped=True)
    migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan"],
        predecessor_state_path=fixture["predecessor_state"],
        successor_plan_path=fixture["successor_plan"],
        successor_state_path=fixture["successor_state"],
        ancestor_plan_paths=[fixture["ancestor_plan"]],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        apply=True,
    )
    state = controller._validate_state(
        json.loads(fixture["successor_state"].read_text()), fixture["successor"]
    )
    for stage in profiles.STAGES:
        victims = [
            entry
            for entry in state["entries"]
            if entry["stage_id"] == stage.stage_id
            and entry["state"] in controller.ACTIVE_STATES
        ][:3]
        assert len(victims) == 3
        for entry in victims:
            fixture["scheduler"].by_id[int(entry["task_id"])]["status"] = "completed"

    held = controller.control_once(
        fixture["successor_plan"],
        state_path=fixture["successor_state"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert held["active_count"] == 492
    assert held["ramp_released"] is False
    assert fixture["scheduler"].post_count == 4
    assert sum(
        len(task_ids)
        for task_ids in held["successor_canary_task_ids_by_stage"].values()
    ) == 4
    hold_action = next(
        action for action in held["actions"] if action["action"] == "rolling_canary_held"
    )
    assert hold_action["active_target_temporarily_relaxed"] is True
    assert hold_action["active_shortfall_held_until_all_canaries_passed"] == 8
    assert hold_action["successor_canary_gap_policy"] == (
        controller.CHAINED_CANARY_GAP_POLICY
    )

    held_again = controller.control_once(
        fixture["successor_plan"],
        state_path=fixture["successor_state"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert held_again["active_count"] == 492
    assert fixture["scheduler"].post_count == 4
    assert sum(
        len(task_ids)
        for task_ids in held_again["successor_canary_task_ids_by_stage"].values()
    ) == 4

    state = json.loads(fixture["successor_state"].read_text())
    canary_entries = [
        entry
        for entry in state["entries"]
        if entry["wave"] == "canary"
        and str(entry.get("origin") or "successor") == "successor"
    ]
    for entry in canary_entries:
        fixture["scheduler"].seed_status[int(entry["task_id"])] = {
            "seed": entry["seed"],
            "bundle_id": entry["bundle_id"],
            "ramp_gate_passed": True,
            "aedt_used": False,
            "fea_submission_performed": False,
        }
    released = controller.control_once(
        fixture["successor_plan"],
        state_path=fixture["successor_state"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert released["ramp_released"] is True
    assert released["active_count"] == 500
    assert fixture["scheduler"].post_count == 12
    final_state = controller._validate_state(
        json.loads(fixture["successor_state"].read_text()), fixture["successor"]
    )
    final_canaries = [
        entry
        for entry in final_state["entries"]
        if entry["wave"] == "canary"
        and str(entry.get("origin") or "successor") == "successor"
    ]
    assert len(final_canaries) == 4
    assert {entry["stage_id"] for entry in final_canaries} == set(profiles.BY_ID)
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 12

    tampered = copy.deepcopy(final_state)
    controller._append_refill(
        tampered,
        templates=controller._task_templates(fixture["successor"]),
        stage_id=profiles.STAGES[0].stage_id,
        wave="canary",
    )
    controller._refresh_migration_observation(tampered)
    tampered = controller._advance_state(tampered)
    with pytest.raises(RuntimeError, match="canary cardinality drifted"):
        controller._validate_state(tampered, fixture["successor"])


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
    planned_seeds = {
        stage.stage_id: {
            int(task["payload_json"]["seed"])
            for wave in ("canaries", "ramp")
            for task in patched_plan["task_waves"][wave]
            if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
        }
        for stage in profiles.STAGES
    }
    assert value["next_seed_by_stage"] == {
        stage.stage_id: max(
            int(resource_state["next_seed_by_stage"][stage.stage_id]),
            max(planned_seeds[stage.stage_id]) + 1,
        )
        for stage in profiles.STAGES
    }
    assert all(
        value["next_seed_by_stage"][stage.stage_id]
        not in planned_seeds[stage.stage_id]
        for stage in profiles.STAGES
    )
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

    # Drain only the predecessor excess (close +88, final +202).  The
    # successor must transfer those 290 naturally vacated slots into entry
    # +236 and bridge +54 without cancelling anything.
    remaining = {"close-1075-t107p5": 88, "final-1000-t100": 202}
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
    assert refilled["scheduler_post_count"] == 294
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 294
    transfer_order = fixture["scheduler"].submitted_stage_ids[4:]
    assert transfer_order.count("entry-1200-t125") == 236
    assert transfer_order.count("bridge-1150-t115") == 54
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


def _phase_a_safe_successor_plan() -> dict:
    value = _successor_plan()
    for stage in profiles.STAGES:
        value["stage_bindings"][stage.stage_id]["remote_bundle"] = (
            f"/gpfs/successor-{stage.stage_id}"
        )
    for wave_name in ("canaries", "ramp"):
        for task in value["task_waves"][wave_name]:
            stage_id = task["payload_json"]["final_goal_stage_id"]
            task["remote_cwd"] = value["stage_bindings"][stage_id]["remote_bundle"]
            task["command"] = multiseed_contract._single_seed_command(
                launch.canonical_sha256(task["payload_json"])
            )
    unsigned = {
        key: item for key, item in value.items() if key != "launch_plan_sha256"
    }
    value["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    return migration.validate_successor_plan(value)


def _external_monitoring_capability(tmp_path: Path) -> dict:
    snapshot = multiseed_status.build_compact_snapshot(
        stage_id=profiles.STAGES[0].stage_id,
        physical_lanes=[],
        seed_records=[],
    )
    snapshot_root = tmp_path / "monitor-snapshot"
    multiseed_status.publish_compact_snapshot(snapshot_root, *snapshot)
    backend = tmp_path / "backend.py"
    backend.write_text("CAPABILITY = 'compact-v2'\n", encoding="utf-8")
    evidence = tmp_path / "monitor-tests.json"
    evidence.write_text('{"passed":true,"tests":1}\n', encoding="utf-8")
    return multiseed_monitor.build_backend_capability_receipt(
        index_path=snapshot_root / "index.json",
        code_files={
            "tools/tier1_final1000_multiseed_monitor.py": Path(
                multiseed_monitor.__file__
            ),
            "tools/tier1_final1000_multiseed_status.py": Path(
                multiseed_status.__file__
            ),
        },
        code_revision="a" * 40,
        backend_files={"regression_260707/monitor.py": backend},
        backend_revision="b" * 40,
        test_evidence_files={"tests/monitor.json": evidence},
        test_revision="c" * 40,
    )


class _ExternalGateReader:
    def __init__(self, scheduler: _Scheduler, *, mode: str = "pass"):
        self.scheduler = scheduler
        self.mode = mode

    def lane_evidence(self, task_id: int, seeds):
        if self.mode == "missing":
            return None
        if self.mode == "429":
            raise urllib.error.HTTPError(
                "http://scheduler/remote-file",
                429,
                "busy",
                {},
                io.BytesIO(b"busy"),
            )
        task = self.scheduler.by_id[int(task_id)]["task_json"]
        manifest = multiseed_contract.batch_manifest_from_payload(
            task["payload_json"]
        )
        receipts = []
        for ordinal, child in enumerate(manifest["ordered_children"]):
            legacy = {
                "schema_version": single_contract.STATUS_SCHEMA,
                "state": "completed",
                "terminal": True,
                "seed": child["seed"],
            }
            receipts.append(
                multiseed_contract.seal_child_receipt(
                    {
                        "schema_version": multiseed_contract.CHILD_RECEIPT_SCHEMA,
                        "protocol_version": multiseed_contract.PROTOCOL_VERSION,
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
                        "legacy_status_sha256": launch.canonical_sha256(legacy),
                        "result_sha256": "d" * 64,
                        "started_at": "2026-07-22T00:00:00+00:00",
                        "finished_at": "2026-07-22T00:01:00+00:00",
                        "wall_time_seconds": 60.0,
                        "failure": None,
                        "production_eligible": False,
                        "fea_submission_performed": False,
                        "aedt_used": False,
                    }
                )
            )
        failed = self.mode == "failed"
        status = multiseed_contract.seal_task_status(
            {
                "schema_version": multiseed_contract.TASK_STATUS_SCHEMA,
                "protocol_version": multiseed_contract.PROTOCOL_VERSION,
                "task_id": str(task_id),
                "manifest_sha256": manifest["manifest_sha256"],
                "state": "failed" if failed else "completed",
                "stop_requested": False,
                "stop_reason": None,
                "current_ordinal": None,
                "current_seed": None,
                "sealed_child_count": len(seeds),
                "completed_child_count": 0 if failed else len(seeds),
                "failed_child_count": len(seeds) if failed else 0,
                "started_at": "2026-07-22T00:00:00+00:00",
                "updated_at": "2026-07-22T00:01:00+00:00",
                "finished_at": "2026-07-22T00:01:00+00:00",
                "subprocess_per_seed": True,
                "model_context_reuse": False,
                "scheduler_mutation_performed": False,
                "fea_submission_performed": False,
                "aedt_used": False,
            }
        )
        return {
            "manifest": manifest,
            "task_status": status,
            "child_receipts": receipts,
        }


def _external_canary_fixture(tmp_path: Path) -> dict:
    successor = _phase_a_safe_successor_plan()
    predecessor = _predecessor_plan(successor)
    state = _predecessor_state(predecessor)
    for entry in state["entries"]:
        entry["state"] = "running"
    terminal_budget = {
        "entry-1200-t125": 6,
        "bridge-1150-t115": 4,
        "close-1075-t107p5": 2,
        "final-1000-t100": 0,
    }
    for entry in state["entries"]:
        stage_id = entry["stage_id"]
        if terminal_budget[stage_id] > 0:
            entry["state"] = "completed"
            terminal_budget[stage_id] -= 1
    assert not any(terminal_budget.values())
    state = controller._seal_state(state)
    scheduler = _Scheduler(predecessor, state)
    capability = _external_monitoring_capability(tmp_path)
    driver_state = multiseed_driver.initial_driver_state(
        state, predecessor, successor, capability
    )
    gates = {"batch1": {}, "batch4": {}}
    next_task_id = 810_000
    for stage in profiles.STAGES:
        next_task_id += 1
        task = multiseed_controller.build_reserved_gate_task(
            successor, stage_id=stage.stage_id, phase="batch1"
        )
        row = {
            **copy.deepcopy(task),
            "id": next_task_id,
            "task_id": next_task_id,
            "status": "completed",
            "state": "succeeded",
            "task_json": copy.deepcopy(task),
        }
        scheduler.by_id[next_task_id] = row
        scheduler.by_dedupe[task["dedupe_key"]] = row
        gates["batch1"][task["dedupe_key"]] = {
            "parent_dedupe_key": task["dedupe_key"],
            "stage_id": stage.stage_id,
            "batch_length": 1,
            "task_id": next_task_id,
            "state": "passed",
            "task": task,
            "task_sha256": launch.canonical_sha256(task),
        }
    driver_state = multiseed_driver._advance_driver(
        driver_state,
        successor,
        capability,
        gate_lanes=gates,
        phase="batch4",
    )
    paths = {}
    for name, value in (
        ("predecessor_plan", predecessor),
        ("predecessor_state", state),
        ("successor_plan", successor),
        ("driver_state", driver_state),
        ("monitoring_capability", capability),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[f"{name}_path"] = path
    return {
        "predecessor": predecessor,
        "state": state,
        "successor": successor,
        "driver_state": driver_state,
        "capability": capability,
        "scheduler": scheduler,
        "ready": _Ready(predecessor, successor),
        "successor_state_path": tmp_path / "successor-state.json",
        **paths,
    }


def _prepare_external(fixture: dict, *, reader=None):
    return migration.prepare_successor_state(
        predecessor_plan_path=fixture["predecessor_plan_path"],
        predecessor_state_path=fixture["predecessor_state_path"],
        successor_plan_path=fixture["successor_plan_path"],
        successor_state_path=fixture["successor_state_path"],
        scheduler=fixture["scheduler"],
        predecessor_ready_probe=fixture["ready"],
        successor_ready_probe=fixture["ready"],
        apply=False,
        transition_mode=migration.PATCHED_BUNDLE,
        external_canary_driver_state_path=fixture["driver_state_path"],
        external_canary_monitoring_capability_path=fixture[
            "monitoring_capability_path"
        ],
        external_canary_gate_reader=(
            reader or _ExternalGateReader(fixture["scheduler"])
        ),
    )


def test_external_batch1_attestation_releases_first_cycle_refill_post0_shadow(
    tmp_path,
):
    fixture = _external_canary_fixture(tmp_path)
    result = _prepare_external(fixture)
    state = result["successor_state"]
    migration_value = state["rolling_migration"]
    receipt = migration_value["external_canary_attestation"]

    assert result["imported_active_count"] == 488
    assert result["initial_active_shortfall"] == 12
    assert result["scheduler_post_count"] == 0
    assert fixture["scheduler"].post_count == 0
    assert fixture["scheduler"].mutations == []
    assert state["ramp_released"] is True
    assert set(state["canary_passed_stage_ids"]) == set(profiles.BY_ID)
    assert migration_value["successor_canary_task_ids_by_stage"] == {
        stage.stage_id: [] for stage in profiles.STAGES
    }
    assert migration_value["successor_canary_status_by_stage"] == {
        stage.stage_id: "remote_preflight_passed" for stage in profiles.STAGES
    }
    assert receipt["source_driver_state_sha256"] == fixture["driver_state"][
        "state_sha256"
    ]
    assert receipt["successor_launch_plan_sha256"] == fixture["successor"][
        "launch_plan_sha256"
    ]
    assert sorted(
        item["task_id"] for item in receipt["stage_attestations"].values()
    ) == [810_001, 810_002, 810_003, 810_004]
    assert all(
        item["seeds"] == [profiles.BY_ID[stage_id].seed_window_end_exclusive - 5]
        for stage_id, item in receipt["stage_attestations"].items()
    )
    assert not any(entry["state"] == "planned" for entry in state["entries"])
    controller._validate_state(state, fixture["successor"])

    # Two more natural terminals may appear after the migration snapshot but
    # before the first successor cycle.  Because the external receipt already
    # released the ramp, that same first cycle must reconcile all 14 gaps,
    # submit safe-bundle refills, and return to exact500.
    newly_terminal = 0
    for task in fixture["scheduler"].by_id.values():
        if (
            task.get("status") == "running"
            and task.get("payload_json", {}).get("bundle_id", "").startswith(
                "predecessor-"
            )
            and newly_terminal < 2
        ):
            task["status"] = "completed"
            task["state"] = "succeeded"
            fixture["scheduler"].by_dedupe[task["dedupe_key"]]["status"] = (
                "completed"
            )
            fixture["scheduler"].by_dedupe[task["dedupe_key"]]["state"] = (
                "succeeded"
            )
            newly_terminal += 1
    assert newly_terminal == 2
    task_ids_before_first_cycle = set(fixture["scheduler"].by_id)
    fixture["successor_state_path"].write_text(json.dumps(state), encoding="utf-8")
    first = controller.control_once(
        fixture["successor_plan_path"],
        state_path=fixture["successor_state_path"],
        apply=True,
        scheduler=fixture["scheduler"],
        ready_probe=fixture["ready"],
    )
    assert first["active_count"] == 500
    assert sum(first["active_count_by_stage"].values()) == 500
    assert first["scheduler_post_count"] == 14
    refill_action = next(
        action for action in first["actions"] if action["action"] == "refill"
    )
    assert refill_action["reserved"] == 14
    assert refill_action["submitted"] == 14
    assert refill_action["reconciled"] == 0
    assert first["cancellation_performed"] is False
    assert state["rolling_migration"]["preemption_performed"] is False
    assert fixture["scheduler"].mutations == ["POST /api/tasks"] * 14
    submitted = [
        task
        for task_id, task in fixture["scheduler"].by_id.items()
        if task_id not in task_ids_before_first_cycle
    ]
    assert len(submitted) == 14
    assert all(
        task["payload_json"]["bundle_id"]
        == fixture["successor"]["stage_bindings"][
            task["payload_json"]["final_goal_stage_id"]
        ]["bundle_id"]
        for task in submitted
    )


@pytest.mark.parametrize("mode", ["missing", "429", "failed"])
def test_external_batch1_remote_evidence_failures_fail_closed(tmp_path, mode):
    fixture = _external_canary_fixture(tmp_path / mode)
    with pytest.raises(RuntimeError, match="external successor canary"):
        _prepare_external(
            fixture,
            reader=_ExternalGateReader(fixture["scheduler"], mode=mode),
        )
    assert fixture["scheduler"].post_count == 0
    assert fixture["scheduler"].mutations == []


def test_external_batch1_driver_and_scheduler_tamper_fail_closed(tmp_path):
    fixture = _external_canary_fixture(tmp_path / "driver")
    damaged = copy.deepcopy(fixture["driver_state"])
    damaged["phase"] = "refill"
    fixture["driver_state_path"].write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(RuntimeError, match="driver state seal mismatch"):
        _prepare_external(fixture)

    fixture = _external_canary_fixture(tmp_path / "scheduler")
    task = fixture["scheduler"].by_id[810_001]
    task["task_json"] = copy.deepcopy(task["task_json"])
    task["task_json"]["cpus"] = 1
    with pytest.raises(RuntimeError, match="Scheduler task identity drifted"):
        _prepare_external(fixture)
    assert fixture["scheduler"].post_count == 0


def test_external_attestation_semantic_linkage_and_seed_are_fail_closed(tmp_path):
    fixture = _external_canary_fixture(tmp_path)
    state = _prepare_external(fixture)["successor_state"]

    missing = copy.deepcopy(state)
    missing["rolling_migration"].pop("external_canary_attestation")
    missing["rolling_migration"] = migration._seal_nested(
        missing["rolling_migration"]
    )
    missing = controller._seal_state(missing)
    with pytest.raises(RuntimeError, match="rolling migration ledger seal mismatch"):
        controller._validate_state(missing, fixture["successor"])

    wrong_ramp = copy.deepcopy(state)
    wrong_ramp["ramp_released"] = False
    wrong_ramp = controller._seal_state(wrong_ramp)
    with pytest.raises(RuntimeError, match="rolling migration ledger seal mismatch"):
        controller._validate_state(wrong_ramp, fixture["successor"])

    wrong_seed = copy.deepcopy(state)
    receipt = wrong_seed["rolling_migration"]["external_canary_attestation"]
    first_stage = profiles.STAGES[0]
    receipt["stage_attestations"][first_stage.stage_id]["seeds"] = [
        first_stage.seed_window_end_exclusive - 4
    ]
    receipt_unsigned = {key: item for key, item in receipt.items() if key != "sha256"}
    receipt["sha256"] = migration.canonical_sha256(receipt_unsigned)
    wrong_seed["rolling_migration"] = migration._seal_nested(
        wrong_seed["rolling_migration"]
    )
    wrong_seed = controller._seal_state(wrong_seed)
    with pytest.raises(RuntimeError, match="stage attestation drifted"):
        controller._validate_state(wrong_seed, fixture["successor"])

def test_external_canary_prepare_fails_on_final_predecessor_drift(tmp_path):
    fixture = _external_canary_fixture(tmp_path)

    def drift():
        current = json.loads(fixture["predecessor_state_path"].read_text())
        current["revision"] += 1
        current = controller._seal_state(current)
        fixture["predecessor_state_path"].write_text(
            json.dumps(current), encoding="utf-8"
        )

    with pytest.raises(RuntimeError, match="drifted during migration"):
        migration.prepare_successor_state(
            predecessor_plan_path=fixture["predecessor_plan_path"],
            predecessor_state_path=fixture["predecessor_state_path"],
            successor_plan_path=fixture["successor_plan_path"],
            successor_state_path=fixture["successor_state_path"],
            scheduler=fixture["scheduler"],
            predecessor_ready_probe=fixture["ready"],
            successor_ready_probe=fixture["ready"],
            apply=False,
            transition_mode=migration.PATCHED_BUNDLE,
            before_final_state_read=drift,
            external_canary_driver_state_path=fixture["driver_state_path"],
            external_canary_monitoring_capability_path=fixture[
                "monitoring_capability_path"
            ],
            external_canary_gate_reader=_ExternalGateReader(fixture["scheduler"]),
        )
