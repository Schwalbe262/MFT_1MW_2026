from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_slurm_controller as controller
from tools import tier1_final1000_stage_profiles as profiles


REPO = Path(__file__).resolve().parents[1]


def _base_task(stage: profiles.FinalGoalStage, seed: int | None = None) -> dict:
    seed = stage.seed_start if seed is None else int(seed)
    payload = {
        "schema_version": "mft-tier1-current7-slurm-seed-task-v1",
        "bundle_id": f"current7-{stage.stage_id}",
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
        "max_generations": 200,
        "inference_threads": 8,
        "optimizer_processes": 1,
        "maximum_peak_rss_bytes": 42 * 1024**3,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    old_sha = launch.canonical_sha256(payload)
    return {
        "name": "base",
        "remote_cwd": f"/gpfs/final1000/{stage.stage_id}",
        "command": (
            "set -euo pipefail\n"
            "exec python -u artifacts/code/tools/"
            "tier1_corrected_current7_slurm_seed_runner.py "
            f"--payload-sha256 {old_sha}"
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
        "dedupe_key": "mft-tier1-current7:" + "b" * 64,
        "max_workers_per_node": 4,
    }


def _rendered_plan() -> dict:
    canaries = []
    ramp = []
    for lane in [*launch.logical_lanes()[0], *launch.logical_lanes()[1]]:
        stage = profiles.BY_ID[lane["stage_id"]]
        task = launch.build_stage_task(
            _base_task(stage, lane["seed"]),
            stage=stage,
            wave=lane["wave"],
        )
        (canaries if lane["wave"] == "canary" else ramp).append(task)
    stage_bindings = {}
    for stage in profiles.STAGES:
        ready = {
            "schema_version": "mft-tier1-current7-slurm-ready-v1",
            "bundle_id": f"current7-{stage.stage_id}",
            "stage_spec_sha256": profiles.stage_profile(stage)[
                "stage_spec_sha256"
            ],
        }
        stage_bindings[stage.stage_id] = {
            "bundle_id": f"current7-{stage.stage_id}",
            "bundle_manifest_sha256": "a" * 64,
            "remote_bundle": f"/gpfs/final1000/{stage.stage_id}",
            "publication_receipt_sha256": "b" * 64,
            "ready": ready,
            "ready_sha256": launch.canonical_sha256(ready),
            "base_island_id": "n1-6-base",
            "stage_spec_sha256": profiles.stage_profile(stage)[
                "stage_spec_sha256"
            ],
        }
    unsigned = {
        "schema_version": launch.LAUNCH_SCHEMA,
        "created_at": "2026-07-21T00:00:00+09:00",
        "stage_inventory": profiles.stage_inventory(),
        "stage_bindings": stage_bindings,
        "resources": {
            "cpus_per_task": 4,
            "memory_mb_per_task": 28 * 1024,
            "max_workers_per_node": 32,
            "priority": 0,
            "scheduling_profile": "standard",
            "gpus": 0,
        },
        "fast_ramp": {
            "canary_count": 4,
            "ramp_count": 496,
        },
        "open_ended_refill": {
            "logical_active_target": 500,
            "stage_active_quotas": {
                stage.stage_id: stage.active_quota for stage in profiles.STAGES
            },
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
    return {
        **unsigned,
        "launch_plan_sha256": launch.canonical_sha256(unsigned),
    }


def test_profiles_form_exact_parallel_staircase_and_final_goal():
    assert len(profiles.STAGES) == 4
    assert profiles.TOTAL_ACTIVE_QUOTA == 500
    assert [stage.active_quota for stage in profiles.STAGES] == [64, 96, 128, 212]
    assert [stage.size_W_max_mm for stage in profiles.STAGES] == [
        1200.0,
        1150.0,
        1075.0,
        1000.0,
    ]
    assert [stage.robust_temperature_limit_C for stage in profiles.STAGES] == [
        125.0,
        115.0,
        107.5,
        100.0,
    ]
    for stage in profiles.STAGES:
        spec, names = launch.optimizer_stage_contract(stage)
        assert spec == profiles.hard_spec(stage)
        assert spec["resonance_min_Hz"] == 15_000.0
        assert spec["resonance_max_Hz"] == 20_000.0
        assert spec["size_H_max_mm"] == 750.0
        assert spec["n_core_group_max"] == 4.0
        assert spec["primary_conductor_thickness_mm"] == 5.0
        assert profiles.RESONANCE_CONSTRAINT_NAMES == (
            "half_magnetizing_resonance_minimum",
            "half_magnetizing_resonance_maximum",
        )
        assert all(name in names for name in profiles.RESONANCE_CONSTRAINT_NAMES)
        profile = profiles.stage_profile(stage)
        assert profile["parallel_launch_required"] is True
        assert profile["predecessor_completion_required"] is False
        assert profile["resonance_band"]["upper_exclusive"] is True
        assert profile["fixed_primary_turns"] == 6
        assert profile["surrogate_only"] is True
        assert profile["aedt_used"] is False
        warm = profile["warm_pool_input_contract"]
        assert warm["output_mix"] == {
            "shape": [64, 25],
            "terminal_replay_N1_6": 48,
            "independently_repaired_geometry_N1_6": 16,
            "interleave": "three_terminal_replay_then_one_repaired_geometry",
            "decoded_params_sha256_duplicate_cap": 1,
        }
        assert warm["dynamic_G_replay"][
            "old_dynamic_G_must_not_be_reused"
        ] is True
        assert warm["repaired_geometry_input_contract"][
            "unrepaired_or_N1_other_than_6_rejected"
        ] is True

    final = profiles.hard_spec(profiles.STAGES[-1])
    assert final["size_W_max_mm"] == final["size_L_max_mm"] == 1000.0
    assert final["size_H_max_mm"] == 750.0
    assert final["T_limit_C"] == 100.0


def test_logical_wave_is_4_canaries_plus_496_and_preserves_stage_quotas():
    canaries, ramp = launch.logical_lanes()
    assert len(canaries) == 4
    assert len(ramp) == 496
    assert {lane["stage_id"] for lane in canaries} == set(profiles.BY_ID)
    assert all(lane["wave"] == "canary" for lane in canaries)
    assert all(lane["wave"] == "ramp" for lane in ramp)
    all_lanes = [*canaries, *ramp]
    assert len({lane["seed"] for lane in all_lanes}) == 500
    assert {
        stage.stage_id: sum(
            lane["stage_id"] == stage.stage_id for lane in all_lanes
        )
        for stage in profiles.STAGES
    } == {stage.stage_id: stage.active_quota for stage in profiles.STAGES}


def test_task_renderer_pins_stage_payload_and_requested_scheduler_resources():
    stage = profiles.STAGES[-1]
    task = launch.build_stage_task(_base_task(stage), stage=stage, wave="canary")
    payload = task["payload_json"]
    assert set(task) == launch.REQUIRED_SCHEDULER_FIELDS
    assert payload["hard_spec"] == profiles.hard_spec(stage)
    assert payload["stage_spec_sha256"] == profiles.canonical_sha256(
        profiles.hard_spec(stage)
    )
    assert payload["final_goal_stage_id"] == stage.stage_id
    assert payload["lane"]["fixed_primary_turns"] == 6
    assert payload["maximum_peak_rss_bytes"] == 22 * 1024**3
    assert task["cpus"] == 4
    assert task["memory_mb"] == 28 * 1024
    assert task["max_workers_per_node"] == 32
    assert task["priority"] == 0
    assert task["scheduling_profile"] == "standard"
    assert task["gpus"] == 0
    assert task["aedt_backend"] == "standalone"
    assert "ansysedt" not in task["command"].lower()
    assert "pyaedt.desktop" not in task["command"].lower()
    assert (
        f"--payload-sha256 {launch.canonical_sha256(payload)}"
        in task["command"]
    )
    assert task["dedupe_key"].startswith("mft-tier1-final1000:")


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("task", "cpus", 8),
        ("task", "memory_mb", 65_536),
        ("task", "priority", 10),
        ("task", "max_workers_per_node", 4),
        ("payload", "aedt_used", True),
        ("payload", "fea_submission_performed", True),
        ("payload", "stage_spec_sha256", "0" * 64),
    ],
)
def test_task_validator_fails_closed_on_resource_or_science_drift(
    location, field, value
):
    stage = profiles.STAGES[0]
    task = launch.build_stage_task(_base_task(stage), stage=stage, wave="canary")
    mutated = copy.deepcopy(task)
    target = mutated if location == "task" else mutated["payload_json"]
    target[field] = value
    with pytest.raises(RuntimeError, match="task seal mismatch"):
        launch.validate_task(mutated)


def test_full_launch_plan_is_sealed_open_ended_500_and_detects_tamper():
    plan = _rendered_plan()
    validated = launch.validate_launch_plan(plan)
    assert validated["open_ended_refill"]["logical_active_target"] == 500
    assert len(validated["task_waves"]["canaries"]) == 4
    assert len(validated["task_waves"]["ramp"]) == 496
    assert validated["submission_performed"] is False
    assert validated["aedt_used"] is False

    mutated = copy.deepcopy(plan)
    mutated["task_waves"]["ramp"][0]["cpus"] = 8
    unsigned = {
        key: value
        for key, value in mutated.items()
        if key != "launch_plan_sha256"
    }
    mutated["launch_plan_sha256"] = launch.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="task seal mismatch"):
        launch.validate_launch_plan(mutated)


def test_result_preflight_requires_exact_dynamic_stage_and_physical_replay():
    stage = profiles.STAGES[-1]
    task = launch.build_stage_task(_base_task(stage), stage=stage, wave="canary")
    spec, names = launch.optimizer_stage_contract(stage)
    physical = {name: -1.0 for name in names}
    result = {
        "hard_spec": spec,
        "stage_spec_sha256": launch.canonical_sha256(spec),
        "constraint_names": names,
        "seed": stage.seed_start,
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
    receipt = launch.validate_stage_result(result, task)
    assert receipt["stage_id"] == stage.stage_id
    assert receipt["terminal_physical_replay_attested"] is True
    assert receipt["scheduler_write_performed"] is False

    mutated = copy.deepcopy(result)
    mutated["hard_spec"]["resonance_max_Hz"] = 20_001.0
    with pytest.raises(RuntimeError, match="result stage/replay seal mismatch"):
        launch.validate_stage_result(mutated, task)


def test_cli_is_render_validate_only_and_has_no_apply_or_submit_surface():
    process = subprocess.run(
        [sys.executable, "tools/tier1_final1000_slurm_launch.py", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    help_text = process.stdout.lower()
    assert "render" in help_text
    assert "validate" in help_text
    assert "--apply" not in help_text
    assert "{profiles,render,validate,validate-result}" in help_text


def test_binding_schema_rejects_missing_stage_before_any_remote_action(tmp_path):
    path = tmp_path / "bindings.json"
    path.write_text(
        json.dumps({"schema_version": launch.BINDINGS_SCHEMA, "stages": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="binding inventory mismatch"):
        launch.load_stage_bindings(path)


class _ReadyProbe:
    def __init__(self, plan: dict):
        self.read_count = 0
        self.by_remote = {
            binding["remote_bundle"]: binding["ready"]
            for binding in plan["stage_bindings"].values()
        }

    def read_ready(self, remote_bundle: str):
        self.read_count += 1
        return copy.deepcopy(self.by_remote.get(remote_bundle))


class _Scheduler:
    def __init__(self):
        self.post_count = 0
        self.next_id = 10_000
        self.by_dedupe = {}
        self.by_id = {}
        self.seed_status = {}

    def find_task_by_dedupe(self, dedupe_key: str):
        value = self.by_dedupe.get(dedupe_key)
        return copy.deepcopy(value) if value is not None else None

    def submit_task(self, payload: dict):
        assert payload["name"].startswith(controller.TASK_NAME_PREFIX)
        assert payload["dedupe_key"].startswith(controller.DEDUPE_PREFIX)
        assert "requested_allocation_id" not in payload
        self.next_id += 1
        value = {
            "id": self.next_id,
            "task_id": self.next_id,
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "status": "queued",
            "payload_json": copy.deepcopy(payload["payload_json"]),
        }
        self.post_count += 1
        self.by_dedupe[value["dedupe_key"]] = value
        self.by_id[value["id"]] = value
        return copy.deepcopy(value)

    def get_task(self, task_id: int):
        value = self.by_id.get(task_id)
        return copy.deepcopy(value) if value is not None else None

    def read_seed_status(self, task_id: int):
        value = self.seed_status.get(task_id)
        return copy.deepcopy(value) if value is not None else None

    def pass_all_canaries(self):
        for task_id, task in self.by_id.items():
            payload = task["payload_json"]
            if payload["lane"]["wave"] != "canary":
                continue
            self.seed_status[task_id] = {
                "seed": payload["seed"],
                "bundle_id": payload["bundle_id"],
                "ramp_gate_passed": True,
                "aedt_used": False,
                "fea_submission_performed": False,
            }


def _write_plan(tmp_path: Path) -> tuple[Path, dict]:
    plan = _rendered_plan()
    path = tmp_path / "launch.json"
    path.write_text(
        json.dumps(plan, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    return path, plan


def test_controller_dry_run_is_write_zero_and_namespace_isolated(tmp_path):
    plan_path, _plan = _write_plan(tmp_path)
    state_path = tmp_path / "state.json"
    scheduler = _Scheduler()
    value = controller.control_once(
        plan_path,
        state_path=state_path,
        apply=False,
        scheduler=scheduler,
    )
    assert value["apply"] is False
    assert value["state_writes"] == 0
    assert value["scheduler_post_count"] == 0
    assert value["scheduler_name_prefix"] == "mft-t1fg-"
    assert value["scheduler_dedupe_prefix"] == "mft-tier1-final1000:"
    assert value["actions"] == [
        {"action": "would_submit_canaries", "count": 4}
    ]
    assert not state_path.exists()
    assert not hasattr(controller.SchedulerApiClient, "cancel_task")
    assert not hasattr(controller.SchedulerApiClient, "close_task")


def test_controller_releases_496_then_refills_each_terminal_gap(tmp_path):
    plan_path, plan = _write_plan(tmp_path)
    state_path = tmp_path / "state.json"
    scheduler = _Scheduler()
    ready = _ReadyProbe(plan)

    first = controller.control_once(
        plan_path,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=ready,
    )
    assert first["ramp_released"] is False
    assert first["scheduler_post_count"] == 4
    assert first["active_count"] == 500
    assert len(scheduler.by_id) == 4
    assert ready.read_count == 4

    scheduler.pass_all_canaries()
    second = controller.control_once(
        plan_path,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=ready,
    )
    assert second["ramp_released"] is True
    assert second["scheduler_post_count"] == 500
    assert second["active_count"] == 500
    assert second["active_count_by_stage"] == {
        stage.stage_id: stage.active_quota for stage in profiles.STAGES
    }
    assert len(scheduler.by_id) == 500
    assert all(
        task["name"].startswith("mft-t1fg-")
        and task["dedupe_key"].startswith("mft-tier1-final1000:")
        for task in scheduler.by_id.values()
    )

    completed_id = next(
        task_id
        for task_id, task in scheduler.by_id.items()
        if task["payload_json"]["lane"]["wave"] == "ramp"
    )
    scheduler.by_id[completed_id]["status"] = "completed"
    scheduler.by_dedupe[
        scheduler.by_id[completed_id]["dedupe_key"]
    ]["status"] = "completed"
    third = controller.control_once(
        plan_path,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=ready,
    )
    assert third["scheduler_post_count"] == 501
    assert third["active_count"] == 500
    assert third["entry_count"] == 501
    assert third["terminal_count"] == 1
    assert third["actions"][-1] == {
        "action": "refill",
        "reserved": 1,
        "submitted": 1,
        "reconciled": 0,
    }


def test_controller_stop_seals_no_refill_and_never_cancels(tmp_path):
    plan_path, plan = _write_plan(tmp_path)
    state_path = tmp_path / "state.json"
    scheduler = _Scheduler()
    ready = _ReadyProbe(plan)
    controller.control_once(
        plan_path,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=ready,
    )
    posts = scheduler.post_count
    stopped = controller.control_once(
        plan_path,
        state_path=state_path,
        apply=True,
        scheduler=scheduler,
        ready_probe=ready,
        request_stop=True,
    )
    assert stopped["stop_requested"] is True
    assert stopped["scheduler_post_count"] == posts
    assert stopped["cancellation_performed"] is False
    assert stopped["actions"][-1]["running_tasks_cancelled"] is False
