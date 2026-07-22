from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
from typing import Any

import pytest

from tools import tier1_corrected_current7_slurm_bundle as single_contract
from tools import tier1_corrected_current7_slurm_harvest as single_harvest
from tools import tier1_corrected_current7_slurm_seed_runner as single_runner
from tools import tier1_corrected_generation_preflight as generation_preflight
from tools import tier1_final1000_multiseed_contract as contract
from tools import tier1_final1000_multiseed_controller as controller
from tools import tier1_final1000_multiseed_harvest as harvest
from tools import tier1_final1000_multiseed_lane_runner as runner
from tools import tier1_final1000_multiseed_status as compact
from tools import tier1_final1000_slurm_controller as single_controller
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_stage_profiles as profiles


def _base_task(stage: profiles.FinalGoalStage, seed: int) -> dict[str, Any]:
    payload = {
        "schema_version": single_contract.TASK_SCHEMA,
        "bundle_id": f"current7-{stage.stage_id}",
        "bundle_manifest_sha256": "a" * 64,
        "lane": {
            "island_id": "n1-6-base",
            "variant": "n1-6-base-current7",
            "seed": int(seed),
            "fixed_primary_turns": 6,
            "wave": "refill",
        },
        "seed": int(seed),
        "population": 320,
        "max_generations": 300,
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
        "remote_cwd": f"/gpfs/final1000/current7-{stage.stage_id}",
        "command": contract._single_seed_command(payload_sha),
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


def _child_tasks(length: int = 4, *, stage_index: int = 0) -> list[dict[str, Any]]:
    stage = profiles.STAGES[stage_index]
    return [
        launch.build_stage_task(
            _base_task(stage, stage.seed_start + offset),
            stage=stage,
            wave="refill",
        )
        for offset in range(length)
    ]


def _rendered_plan() -> dict[str, Any]:
    canaries: list[dict[str, Any]] = []
    ramp: list[dict[str, Any]] = []
    for lane in [*launch.logical_lanes()[0], *launch.logical_lanes()[1]]:
        stage = profiles.BY_ID[lane["stage_id"]]
        task = launch.build_stage_task(
            _base_task(stage, lane["seed"]),
            stage=stage,
            wave=lane["wave"],
        )
        (canaries if lane["wave"] == "canary" else ramp).append(task)
    bindings = {}
    for stage in profiles.STAGES:
        ready = {
            "schema_version": "mft-tier1-current7-slurm-ready-v1",
            "bundle_id": f"current7-{stage.stage_id}",
            "stage_spec_sha256": profiles.stage_profile(stage)["stage_spec_sha256"],
        }
        bindings[stage.stage_id] = {
            "bundle_id": f"current7-{stage.stage_id}",
            "bundle_manifest_sha256": "a" * 64,
            "remote_bundle": f"/gpfs/final1000/current7-{stage.stage_id}",
            "publication_receipt_sha256": "b" * 64,
            "ready": ready,
            "ready_sha256": launch.canonical_sha256(ready),
            "base_island_id": "n1-6-base",
            "stage_spec_sha256": profiles.stage_profile(stage)["stage_spec_sha256"],
        }
    unsigned = {
        "schema_version": launch.LAUNCH_SCHEMA,
        "created_at": "2026-07-22T00:00:00+09:00",
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


def _prepare_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: int = 99001,
    length: int = 4,
    internal_deadline_seconds: int = 82_800,
    cleanup_reserve_seconds: int = 1_800,
    minimum_child_start_budget_seconds: int = 7_200,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    task = contract.build_batch_task(
        _child_tasks(length),
        internal_deadline_seconds=internal_deadline_seconds,
        cleanup_reserve_seconds=cleanup_reserve_seconds,
        minimum_child_start_budget_seconds=minimum_child_start_budget_seconds,
    )
    bundle = tmp_path / "bundle"
    (bundle / "artifacts" / "code" / "tools").mkdir(parents=True)
    payload_root = tmp_path / "scheduler" / "runs"
    payload_path = payload_root / "request-1" / "payload.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(contract.json_bytes(task["payload_json"]))
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", str(task_id))
    return bundle, payload_root, payload_path, task


class FakeChildExecutor:
    def __init__(self, modes: list[str]):
        self.modes = list(modes)
        self.calls: list[int] = []
        self.payload_basenames: list[str] = []
        self.prior_receipt_counts: list[int] = []
        self.cursor_snapshots: list[dict[str, Any]] = []
        self._active = False

    def __call__(
        self,
        *,
        command,
        cwd,
        environment,
        stop_latch,
        heartbeat_seconds,
        termination_grace_seconds,
        deadline_monotonic,
        monotonic,
        on_heartbeat,
    ) -> int:
        assert not self._active
        self._active = True
        try:
            payload_path = Path(command[command.index("--payload") + 1])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            seed = int(payload["seed"])
            self.calls.append(seed)
            self.payload_basenames.append(payload_path.name)
            self.prior_receipt_counts.append(
                len(list(payload_path.parent.parent.glob("seed-*/seed_status.json")))
            )
            mode = self.modes[len(self.calls) - 1]
            on_heartbeat(70_000 + len(self.calls))
            self.cursor_snapshots.append(
                json.loads(
                    (payload_path.parent.parent / "task_status.json").read_text()
                )
            )
            if mode == "raise":
                raise OSError("synthetic child launch failure")
            task_id = str(environment["SLURM_SCHED_TASK_ID"])
            status_path = payload_path.parent.parent / "seed_status.json"
            status: dict[str, Any] = {
                "schema_version": single_contract.STATUS_SCHEMA,
                "state": "failed",
                "phase": "terminal",
                "terminal": True,
                "task_id": task_id,
                "seed": seed,
                "payload_sha256": contract.canonical_sha256(payload),
                "loaded_model_count": 8,
                "exit_code": 17,
                "failure": "synthetic seed-local optimizer failure",
            }
            exit_code = 17
            if mode == "completed":
                result_path = payload_path.parent / "result.json"
                result_path.write_bytes(contract.json_bytes({"seed": seed}))
                status.update(
                    state="completed",
                    exit_code=0,
                    result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                )
                status.pop("failure")
                exit_code = 0
            elif mode == "fatal":
                status.update(
                    phase="remote_model_load_failed",
                    exit_code=72,
                    failure="authenticated bundle/model preflight failed",
                )
                status.pop("loaded_model_count")
                exit_code = 72
            elif mode == "stop":
                stop_latch.request("test_stop")
                exit_code = -15
            status_path.write_bytes(contract.json_bytes(status))
            return exit_code
        finally:
            self._active = False


def _run_lane(
    bundle: Path,
    payload_root: Path,
    payload_path: Path,
    task: dict[str, Any],
    executor: FakeChildExecutor,
    *,
    monotonic=None,
) -> int:
    kwargs = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    return runner.run(
        bundle,
        payload_path,
        payload_root,
        contract.canonical_sha256(task["payload_json"]),
        heartbeat_seconds=0.01,
        termination_grace_seconds=0.01,
        child_executor=executor,
        **kwargs,
    )


def test_batch4_contract_preserves_exact_child_identity_and_detects_tamper():
    children = _child_tasks()
    task = contract.build_batch_task(children)
    payload = task["payload_json"]
    manifest = contract.batch_manifest_from_payload(payload)

    assert payload["batch_length"] == 4
    assert [child["seed"] for child in payload["children"]] == list(
        range(payload["children"][0]["seed"], payload["children"][0]["seed"] + 4)
    )
    assert payload["resource_policy"]["remote_cwd"] == task["remote_cwd"]
    assert payload["subprocess_per_seed"] is True
    assert payload["model_context_reuse"] is False
    assert manifest["seed_reuse_allowed"] is False
    assert task["dedupe_key"].startswith(contract.DEDUPE_PREFIX)
    for child in payload["children"]:
        child_task = child["task"]
        assert child_task["cpus"] == 4
        assert child_task["payload_json"]["scheduler_cpus"] == 4
        assert child_task["payload_json"]["inference_threads"] == 4
        assert child_task["payload_json"]["optimizer_processes"] == 1
    assert (
        contract.build_batch_task(_child_tasks(1))["payload_json"]["batch_length"] == 1
    )
    with pytest.raises(RuntimeError, match="operational batch length"):
        contract.build_batch_task(_child_tasks(8))
    assert contract.PHASE_A_REMOTE_CODE_FILES == (
        "tools/tier1_corrected_current7_slurm_seed_runner.py",
        "tools/tier1_final1000_multiseed_contract.py",
        "tools/tier1_final1000_multiseed_lane_runner.py",
    )

    tampered = copy.deepcopy(task)
    tampered["remote_cwd"] += "-other"
    with pytest.raises(RuntimeError, match="seal mismatch"):
        contract.validate_batch_task(tampered)

    extra_manifest = copy.deepcopy(manifest)
    extra_manifest["unexpected"] = False
    unsigned = {
        key: value for key, value in extra_manifest.items() if key != "manifest_sha256"
    }
    extra_manifest["manifest_sha256"] = contract.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="manifest seal mismatch"):
        contract.validate_batch_manifest(extra_manifest)

    hard_spec_tamper = copy.deepcopy(payload)
    hard_spec_tamper["children"][0]["task"]["payload_json"]["hard_spec"] = {
        "tampered": True
    }
    with pytest.raises(RuntimeError, match="surrogate-only task seal mismatch"):
        contract.validate_batch_payload(hard_spec_tamper)

    stage_profile_tamper = copy.deepcopy(payload)
    stage_profile_tamper["stage_profile_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="parent payload identity mismatch"):
        contract.validate_batch_payload(stage_profile_tamper)

    capability_tamper = copy.deepcopy(task)
    capability_tamper["required_capability"] = "conda:wrong"
    capability_tamper["payload_json"]["resource_policy"]["required_capability"] = (
        "conda:wrong"
    )
    for child in capability_tamper["payload_json"]["children"]:
        child["task"]["required_capability"] = "conda:wrong"
    with pytest.raises(RuntimeError, match="surrogate-only task seal mismatch"):
        contract.validate_batch_task(capability_tamper)


def test_phase_a_child_thread_environment_matches_scheduler_cpu_request():
    child = contract.build_batch_task(_child_tasks(1))["payload_json"]["children"][
        0
    ]["task"]
    environment = single_runner.optimizer_environment(
        {"OMP_NUM_THREADS": "99", "UNRELATED": "preserved"},
        child["payload_json"],
    )
    assert environment["UNRELATED"] == "preserved"
    assert {
        environment[name]
        for name in single_runner.THREAD_LIMIT_ENVIRONMENT_VARIABLES
    } == {"4"}
    assert generation_preflight.validate_search_thread_contract(
        profile_inference_threads=8,
        inference_threads=4,
        scheduler_cpus=4,
    ) == 4
    mismatched = copy.deepcopy(child["payload_json"])
    mismatched["inference_threads"] = 8
    with pytest.raises(RuntimeError, match="CPU/thread"):
        single_runner.optimizer_environment({}, mismatched)
    with pytest.raises(RuntimeError, match="CPU/thread allocation"):
        generation_preflight.validate_search_thread_contract(
            profile_inference_threads=8,
            inference_threads=8,
            scheduler_cpus=4,
        )


def test_runner_seals_each_child_before_next_and_seed_local_failure_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare_lane(tmp_path, monkeypatch)
    executor = FakeChildExecutor(
        ["local_failure", "completed", "completed", "completed"]
    )

    assert _run_lane(bundle, payload_root, payload_path, task, executor) == 0
    seeds = [child["seed"] for child in task["payload_json"]["children"]]
    assert executor.calls == seeds
    assert executor.payload_basenames == ["payload.json"] * 4
    assert executor.prior_receipt_counts == [0, 1, 2, 3]
    assert [item["current_ordinal"] for item in executor.cursor_snapshots] == [
        0,
        1,
        2,
        3,
    ]
    assert [item["current_seed"] for item in executor.cursor_snapshots] == seeds
    run_root = bundle / "runs" / "task-99001"
    receipts = [
        contract.validate_child_receipt(
            json.loads((run_root / f"seed-{seed}" / "seed_status.json").read_text()),
            manifest=json.loads((run_root / "batch_manifest.json").read_text()),
        )
        for seed in seeds
    ]
    assert [receipt["state"] for receipt in receipts] == [
        "failed",
        "completed",
        "completed",
        "completed",
    ]
    status = contract.validate_task_status(
        json.loads((run_root / "task_status.json").read_text())
    )
    assert status["state"] == "completed_with_failures"
    assert status["sealed_child_count"] == 4
    assert status["completed_child_count"] == 3
    assert status["failed_child_count"] == 1

    with pytest.raises(RuntimeError, match="new parent identity"):
        _run_lane(bundle, payload_root, payload_path, task, executor)
    with pytest.raises(RuntimeError, match="immutable multi-seed evidence collision"):
        runner._immutable_json(
            run_root / f"seed-{seeds[0]}" / "seed_status.json", {"tampered": True}
        )


def test_lane_fatal_aborts_batch_and_stop_never_starts_next_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fatal_root = tmp_path / "fatal"
    bundle, payload_root, payload_path, task = _prepare_lane(
        fatal_root, monkeypatch, task_id=99002
    )
    fatal = FakeChildExecutor(["fatal"])
    assert (
        _run_lane(bundle, payload_root, payload_path, task, fatal)
        == runner.LANE_FATAL_EXIT_CODE
    )
    assert len(fatal.calls) == 1
    fatal_status = json.loads(
        (bundle / "runs" / "task-99002" / "task_status.json").read_text()
    )
    assert fatal_status["state"] == "failed"
    assert fatal_status["sealed_child_count"] == 1

    stop_root = tmp_path / "stop"
    bundle, payload_root, payload_path, task = _prepare_lane(
        stop_root, monkeypatch, task_id=99003
    )
    stopped = FakeChildExecutor(["stop"])
    assert (
        _run_lane(bundle, payload_root, payload_path, task, stopped)
        == runner.STOPPED_EXIT_CODE
    )
    assert len(stopped.calls) == 1
    run_root = bundle / "runs" / "task-99003"
    stop_status = json.loads((run_root / "task_status.json").read_text())
    receipt = json.loads(
        (run_root / f"seed-{stopped.calls[0]}" / "seed_status.json").read_text()
    )
    assert stop_status["state"] == "stopped"
    assert stop_status["stop_requested"] is True
    assert receipt["state"] == "stopped"

    executor_root = tmp_path / "executor-error"
    bundle, payload_root, payload_path, task = _prepare_lane(
        executor_root, monkeypatch, task_id=99005
    )
    broken = FakeChildExecutor(["raise"])
    assert (
        _run_lane(bundle, payload_root, payload_path, task, broken)
        == runner.LANE_FATAL_EXIT_CODE
    )
    assert len(broken.calls) == 1
    run_root = bundle / "runs" / "task-99005"
    failed_receipt = json.loads(
        (run_root / f"seed-{broken.calls[0]}" / "seed_status.json").read_text()
    )
    assert failed_receipt["state"] == "refused"
    assert failed_receipt["exit_code"] == runner.CHILD_EXECUTOR_FAILURE_EXIT_CODE
    assert "synthetic child launch failure" in failed_receipt["failure"]


def test_internal_deadline_refuses_next_seed_and_seals_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle, payload_root, payload_path, task = _prepare_lane(
        tmp_path,
        monkeypatch,
        task_id=99004,
        internal_deadline_seconds=100,
        cleanup_reserve_seconds=10,
        minimum_child_start_budget_seconds=50,
    )
    executor = FakeChildExecutor(["completed"])
    values = iter([0.0, 0.0, 60.0])

    assert (
        _run_lane(
            bundle,
            payload_root,
            payload_path,
            task,
            executor,
            monotonic=lambda: next(values),
        )
        == runner.STOPPED_EXIT_CODE
    )
    assert len(executor.calls) == 1
    status = json.loads(
        (bundle / "runs" / "task-99004" / "task_status.json").read_text()
    )
    assert status["state"] == "deadline"
    assert status["stop_reason"] == "internal_deadline_before_next_seed"
    assert status["sealed_child_count"] == 1


@pytest.mark.parametrize(
    "unsafe_task_id",
    ["../escape", "1/../../escape", "0", "-1", "01", "9" * 21],
)
def test_runner_rejects_unsafe_scheduler_task_id_before_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_task_id: str,
):
    bundle, payload_root, payload_path, task = _prepare_lane(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", unsafe_task_id)
    with pytest.raises(RuntimeError, match="positive decimal integer"):
        _run_lane(
            bundle,
            payload_root,
            payload_path,
            task,
            FakeChildExecutor(["completed"]),
        )
    assert not (bundle.parent / "escape").exists()


def test_default_executor_kills_stubborn_child_at_internal_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class StubbornProcess:
        pid = 77123

        def __init__(self):
            self.returncode = None
            self.terminate_count = 0
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def terminate(self):
            self.terminate_count += 1

        def wait(self, timeout=None):
            if not self.killed:
                raise runner.subprocess.TimeoutExpired("fixture", timeout)
            self.returncode = -9
            return -9

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = StubbornProcess()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    latch = runner.StopLatch()
    heartbeats: list[int] = []
    exit_code = runner._default_child_executor(
        command=["fixture"],
        cwd=tmp_path,
        environment={},
        stop_latch=latch,
        heartbeat_seconds=0.01,
        termination_grace_seconds=0.01,
        deadline_monotonic=1.0,
        monotonic=lambda: 1.0,
        on_heartbeat=heartbeats.append,
    )
    assert exit_code == -9
    assert latch.reason == "internal_deadline_elapsed"
    assert process.terminate_count >= 1
    assert process.killed is True
    assert heartbeats == [process.pid]


def test_posix_executor_terminates_the_entire_dedicated_child_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class GroupProcess:
        pid = 88123

        def __init__(self):
            self.returncode = None
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def wait(self, timeout=None):
            if not self.killed:
                raise runner.subprocess.TimeoutExpired("fixture", timeout)
            self.returncode = -9
            return -9

    process = GroupProcess()
    popen_options: dict[str, Any] = {}
    signals = []

    def fake_popen(*args, **kwargs):
        popen_options.update(kwargs)
        return process

    def fake_killpg(pgid, signum):
        assert pgid == process.pid
        signals.append(signum)
        if signum == runner.FORCE_KILL_SIGNAL:
            process.killed = True
            process.returncode = -9

    monkeypatch.setattr(runner.os, "name", "posix")
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    latch = runner.StopLatch()
    exit_code = runner._default_child_executor(
        command=["fixture"],
        cwd=tmp_path,
        environment={},
        stop_latch=latch,
        heartbeat_seconds=0.01,
        termination_grace_seconds=0.01,
        deadline_monotonic=1.0,
        monotonic=lambda: 1.0,
        on_heartbeat=lambda _pid: None,
    )
    assert exit_code == -9
    assert popen_options["start_new_session"] is True
    assert signal.SIGTERM in signals
    assert runner.FORCE_KILL_SIGNAL in signals


def test_controller_v2_reserves_four_seeds_per_physical_gap_and_maps_timeout():
    plan = launch.validate_launch_plan(_rendered_plan())
    v1_state = single_controller._initial_state(plan)
    vacated_stage = str(v1_state["entries"][0]["stage_id"])
    changed = copy.deepcopy(v1_state)
    changed["entries"][0]["state"] = "completed"
    changed["entries"][0]["task_id"] = 98_000
    v1_state = single_controller._advance_state(changed)

    state = controller.upgrade_v1_state(v1_state, plan, batch_length=4)
    assert controller.active_physical_count(state) == 499
    old_next = state["next_seed_by_stage"][vacated_stage]
    state, task = controller.reserve_lane(state, plan, stage_id=vacated_stage)
    assert controller.active_physical_count(state) == 500
    assert {
        stage.stage_id: controller.active_physical_count(state, stage.stage_id)
        for stage in profiles.STAGES
    } == launch.SUCCESSOR_ACTIVE_QUOTAS
    assert state["next_seed_by_stage"][vacated_stage] == old_next + 4
    assert task["payload_json"]["batch_length"] == 4
    assert controller.logical_seed_counts(state)["logical_future_seed_backlog"] == 4

    dedupe = task["dedupe_key"]

    def observed(status: str) -> dict[str, Any]:
        return {
            "id": 99100,
            "name": task["name"],
            "dedupe_key": dedupe,
            "status": status,
            "task_json": copy.deepcopy(task),
        }

    with pytest.raises(RuntimeError, match="dedupe identity"):
        controller.observe_scheduler_task(
            state,
            plan,
            parent_dedupe_key=dedupe,
            scheduler_task={
                "id": 99100,
                "name": task["name"],
                "status": "running",
                "task_json": copy.deepcopy(task),
            },
        )
    state = controller.observe_scheduler_task(
        state,
        plan,
        parent_dedupe_key=dedupe,
        scheduler_task=observed("queued"),
    )
    assert controller.logical_seed_counts(state)["logical_future_seed_backlog"] == 4
    state = controller.observe_scheduler_task(
        state,
        plan,
        parent_dedupe_key=dedupe,
        scheduler_task=observed("running"),
    )
    counts = controller.logical_seed_counts(state)
    assert counts["logical_current_seeds"] == 1
    assert counts["logical_future_seed_backlog"] == 3
    state = controller.observe_scheduler_task(
        state,
        plan,
        parent_dedupe_key=dedupe,
        scheduler_task=observed("timed_out"),
    )
    assert controller.active_physical_count(state) == 499
    assert single_controller._scheduler_state("timed_out") == "timeout"
    with pytest.raises(RuntimeError, match="terminal lane state cannot change"):
        controller.observe_scheduler_task(
            state,
            plan,
            parent_dedupe_key=dedupe,
            scheduler_task=observed("running"),
        )

    malformed_v1 = copy.deepcopy(state)
    v1_entry = next(
        entry
        for entry in malformed_v1["entries"]
        if entry["protocol_version"] == controller.SINGLE_SEED_PROTOCOL
    )
    v1_entry["seeds"] = [v1_entry["seeds"][0], v1_entry["seeds"][0] + 1]
    v1_entry["batch_length"] = 2
    malformed_v1 = controller._seal(malformed_v1)
    with pytest.raises(RuntimeError, match="physical lane entry drifted"):
        controller.validate_state(malformed_v1, plan)

    with pytest.raises(RuntimeError, match="operational batch length"):
        controller.reserve_lane(state, plan, stage_id=vacated_stage, batch_length=8)

    stopped = controller.request_stop(state, plan)
    with pytest.raises(RuntimeError, match="stop latch"):
        controller.reserve_lane(stopped, plan, stage_id=vacated_stage)


class MemoryRemote:
    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)
        self.modes = {path: 0o444 for path in files}

    def stat(self, account_name: str, path: str) -> single_harvest.RemoteFileStat:
        try:
            payload = self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        return single_harvest.RemoteFileStat(
            size=len(payload), mtime=1, mode=self.modes[path]
        )

    def read_bytes(self, account_name: str, path: str, *, maximum_bytes: int) -> bytes:
        try:
            payload = self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        if len(payload) > maximum_bytes:
            raise RuntimeError("fixture exceeded read bound")
        return payload


def _stopped_remote_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], MemoryRemote, dict[str, Any], dict[str, Any]]:
    bundle, payload_root, payload_path, task = _prepare_lane(
        tmp_path, monkeypatch, task_id=99200
    )
    executor = FakeChildExecutor(["stop"])
    assert (
        _run_lane(bundle, payload_root, payload_path, task, executor)
        == runner.STOPPED_EXIT_CODE
    )
    seed = executor.calls[0]
    run_root = bundle / "runs" / "task-99200"
    remote_root = task["remote_cwd"]
    files = {
        f"{remote_root}/runs/task-99200/batch_manifest.json": (
            run_root / "batch_manifest.json"
        ).read_bytes(),
        f"{remote_root}/runs/task-99200/task_status.json": (
            run_root / "task_status.json"
        ).read_bytes(),
        f"{remote_root}/runs/task-99200/seed-{seed}/seed_status.json": (
            run_root / f"seed-{seed}" / "seed_status.json"
        ).read_bytes(),
    }
    item = {
        "task_id": 99200,
        "name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "account_name": "local-fixture",
        "status": "running",
        "exit_code": None,
        "parent_task": task,
        "scheduler_task": copy.deepcopy(task),
    }
    plan = {
        "bundle_id": task["payload_json"]["bundle_id"],
        "bundle_manifest_sha256": task["payload_json"]["bundle_manifest_sha256"],
        "remote_bundle": task["remote_cwd"],
    }
    manifest = {"bundle_id": task["payload_json"]["bundle_id"]}
    return item, MemoryRemote(files), plan, manifest


def test_running_batch_harvests_only_journal_declared_receipts_and_mixes_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    item, remote, plan, manifest = _stopped_remote_lane(tmp_path, monkeypatch)
    observations = harvest.harvest_batch_lane(
        item, plan=plan, manifest=manifest, remote=remote
    )
    assert len(observations) == 1
    assert observations[0]["physical_parent_task_id"] == 99200
    assert observations[0]["batch_ordinal"] == 0
    assert observations[0]["terminal_state"] == "stopped"

    missing_dedupe = copy.deepcopy(item)
    del missing_dedupe["dedupe_key"]
    with pytest.raises(RuntimeError, match="parent/plan/bundle identity mismatch"):
        harvest.harvest_batch_lane(
            missing_dedupe, plan=plan, manifest=manifest, remote=remote
        )

    # Crash window: the immutable receipt is visible before the mutable cursor
    # advances. The receipt remains harvestable and the seed is never rerun.
    task_status_path = next(
        path for path in remote.files if path.endswith("task_status.json")
    )
    original_task_status = remote.files[task_status_path]
    stale = json.loads(original_task_status)
    first_child = item["parent_task"]["payload_json"]["children"][0]
    stale.update(
        state="running",
        stop_requested=False,
        stop_reason=None,
        current_ordinal=0,
        current_seed=first_child["seed"],
        sealed_child_count=0,
        completed_child_count=0,
        failed_child_count=0,
        finished_at=None,
    )
    remote.files[task_status_path] = contract.json_bytes(
        contract.seal_task_status(stale)
    )
    crash_window = harvest.harvest_batch_lane(
        item, plan=plan, manifest=manifest, remote=remote
    )
    assert len(crash_window) == 1
    remote.files[task_status_path] = original_task_status

    manifest_path = next(
        path for path in remote.files if path.endswith("batch_manifest.json")
    )
    remote.modes[manifest_path] = 0o644
    with pytest.raises(RuntimeError, match="manifest is not immutable"):
        harvest.harvest_batch_lane(item, plan=plan, manifest=manifest, remote=remote)
    remote.modes[manifest_path] = 0o444

    def fake_v1(item, *, plan, manifest, remote, result_validator):
        return {
            "bundle_id": "legacy-bundle",
            "seed": 1,
            "island_id": "legacy-island",
            "task_id": int(item["task_id"]),
            "scheduler_state": "completed",
            "terminal_state": "failed",
            "status_object": {"sha256": "c" * 64},
            "_status_bytes": b"legacy",
        }

    monkeypatch.setattr(harvest, "harvest_terminal_task", fake_v1)
    mixed = harvest.harvest_mixed_inventory(
        [item, {"task_id": 99201, "status": "completed"}],
        plan=plan,
        manifest=manifest,
        remote=remote,
    )
    assert mixed["physical_task_count"] == 2
    assert mixed["authenticated_seed_count"] == 2
    assert mixed["v1_physical_task_count"] == 1
    assert mixed["v2_physical_task_count"] == 1
    assert mixed["virtual_scheduler_task_ids_created"] is False

    receipt_path = next(
        path for path in remote.files if path.endswith("seed_status.json")
    )
    del remote.files[receipt_path]
    refused = harvest.harvest_mixed_inventory(
        [item], plan=plan, manifest=manifest, remote=remote
    )
    assert refused["authenticated_seed_count"] == 0
    assert refused["refused_physical_task_count"] == 1


def test_compact_status_shards_history_and_atomically_advances_index(tmp_path: Path):
    lanes = [
        {
            "task_id": 100_000 + index,
            "stage_id": profiles.STAGES[0].stage_id,
            "protocol_version": contract.PROTOCOL_VERSION,
            "state": "running",
            "account_name": "fixture",
            "batch_length": 4,
            "current_seed": 200_000 + index,
            "sealed_child_count": 3,
        }
        for index in range(500)
    ]
    records = [
        {
            "bundle_id": "bundle-a",
            "seed": index,
            "task_ids": [100_000 + index % 500],
            "details": "x" * 4096,
        }
        for index in range(8192)
    ]
    status, manifest, shards, index = compact.build_compact_snapshot(
        stage_id=profiles.STAGES[0].stage_id,
        physical_lanes=lanes,
        seed_records=records,
        shard_size=128,
    )
    hot = contract.json_bytes(status)
    assert len(hot) < compact.MAX_HOT_WIRE_BYTES
    assert b'"details"' not in hot
    assert (
        sum(len(contract.json_bytes(record)) for record in records)
        > compact.MAX_HOT_WIRE_BYTES
    )
    assert manifest["record_count"] == 8192
    assert manifest["shard_count"] == 64
    by_path = {
        reference["path"]: shard
        for reference, shard in zip(manifest["shards"], shards, strict=True)
    }
    page = compact.adapt_status_for_frontend(
        status,
        manifest,
        lambda reference: by_path[reference["path"]],
        offset=127,
        limit=4,
    )
    assert [record["seed"] for record in page["terminal_results"]] == [
        127,
        128,
        129,
        130,
    ]
    assert page["scheduler_task_count"] == 500
    assert page["virtual_scheduler_task_ids_created"] is False
    default_page = compact.adapt_status_for_frontend(
        status,
        manifest,
        lambda reference: by_path[reference["path"]],
    )
    assert default_page["terminal_results_returned_count"] == 256
    assert default_page["terminal_results_has_more"] is True
    with pytest.raises(RuntimeError, match="record bound"):
        compact.adapt_status_for_frontend(
            status,
            manifest,
            lambda reference: by_path[reference["path"]],
            limit=compact.MAX_FRONTEND_PAGE_SIZE + 1,
        )

    assert (
        compact.publish_compact_snapshot(tmp_path, status, manifest, shards, index) > 0
    )
    old_manifest_path = tmp_path / Path(
        *Path(index["seed_result_shards"]["path"]).parts
    )
    assert old_manifest_path.is_file()
    second = compact.build_compact_snapshot(
        stage_id=profiles.STAGES[0].stage_id,
        physical_lanes=lanes,
        seed_records=[*records, {"bundle_id": "bundle-a", "seed": 8192}],
        shard_size=128,
    )
    assert compact.publish_compact_snapshot(tmp_path, *second) > 0
    published_index = json.loads((tmp_path / "index.json").read_text())
    assert published_index == second[3]
    assert old_manifest_path.is_file()

    bad_index = copy.deepcopy(second[3])
    bad_index["status"]["path"] = "../escaped.json"
    unsigned = {key: value for key, value in bad_index.items() if key != "index_sha256"}
    bad_index["index_sha256"] = contract.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="unsafe|escaped"):
        compact.publish_compact_snapshot(tmp_path, *second[:3], bad_index)
    assert not (tmp_path.parent / "escaped.json").exists()

    with pytest.raises(RuntimeError, match="shard exceeds"):
        compact.build_compact_snapshot(
            stage_id=profiles.STAGES[0].stage_id,
            physical_lanes=[],
            seed_records=[
                {
                    "bundle_id": "oversized",
                    "seed": 1,
                    "details": "x" * compact.MAX_HOT_WIRE_BYTES,
                }
            ],
            shard_size=1,
        )


def test_compact_status_gates_full_frontend_wrapper_and_shrinks_large_page():
    with pytest.raises(RuntimeError, match="cannot fit.*frontend wire cap"):
        compact.build_compact_snapshot(
            stage_id=profiles.STAGES[0].stage_id,
            physical_lanes=[],
            seed_records=[
                {
                    "bundle_id": "large-but-shard-servable",
                    "seed": 1,
                    "details": "x" * (compact.MAX_HOT_WIRE_BYTES - 200_000),
                }
            ],
            frontend_static={"wrapper_padding": "y" * 256_000},
            shard_size=1,
        )

    records = [
        {
            "bundle_id": "paged",
            "seed": seed,
            "details": "z" * 140_000,
        }
        for seed in range(compact.DEFAULT_FRONTEND_PAGE_SIZE)
    ]
    status, manifest, shards, _index = compact.build_compact_snapshot(
        stage_id=profiles.STAGES[0].stage_id,
        physical_lanes=[],
        seed_records=records,
        shard_size=64,
    )
    by_path = {
        reference["path"]: shard
        for reference, shard in zip(manifest["shards"], shards, strict=True)
    }
    page = compact.adapt_status_for_frontend(
        status,
        manifest,
        lambda reference: by_path[reference["path"]],
    )
    assert 0 < page["terminal_results_returned_count"] < len(records)
    assert page["terminal_results_has_more"] is True
    assert len(contract.json_bytes(page)) < compact.MAX_HOT_WIRE_BYTES
