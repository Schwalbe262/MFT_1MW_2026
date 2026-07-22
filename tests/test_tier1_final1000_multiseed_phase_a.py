from __future__ import annotations

import copy
from contextlib import contextmanager
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
from tools import tier1_final1000_multiseed_consumer as consumer
from tools import tier1_final1000_multiseed_controller as controller
from tools import tier1_final1000_multiseed_driver as driver
from tools import tier1_final1000_multiseed_harvest as harvest
from tools import tier1_final1000_multiseed_lane_runner as runner
from tools import tier1_final1000_multiseed_monitor as monitor
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
    child = contract.build_batch_task(_child_tasks(1))["payload_json"]["children"][0][
        "task"
    ]
    environment = single_runner.optimizer_environment(
        {"OMP_NUM_THREADS": "99", "UNRELATED": "preserved"},
        child["payload_json"],
    )
    assert environment["UNRELATED"] == "preserved"
    assert {
        environment[name] for name in single_runner.THREAD_LIMIT_ENVIRONMENT_VARIABLES
    } == {"4"}
    assert (
        generation_preflight.validate_search_thread_contract(
            profile_inference_threads=8,
            inference_threads=4,
            scheduler_cpus=4,
        )
        == 4
    )
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


def _monitoring_capability(tmp_path: Path) -> dict[str, Any]:
    snapshot = compact.build_compact_snapshot(
        stage_id=profiles.STAGES[0].stage_id,
        physical_lanes=[],
        seed_records=[],
    )
    snapshot_root = tmp_path / "monitor-snapshot"
    compact.publish_compact_snapshot(snapshot_root, *snapshot)
    backend = tmp_path / "backend.py"
    backend.write_text("CAPABILITY = 'compact-v2'\n", encoding="utf-8")
    evidence = tmp_path / "monitor-tests.json"
    evidence.write_text('{"passed":true,"tests":1}\n', encoding="utf-8")
    return monitor.build_backend_capability_receipt(
        index_path=snapshot_root / "index.json",
        code_files={
            "tools/tier1_final1000_multiseed_monitor.py": Path(monitor.__file__),
            "tools/tier1_final1000_multiseed_status.py": Path(compact.__file__),
        },
        code_revision="a" * 40,
        backend_files={"regression_260707/monitor.py": backend},
        backend_revision="b" * 40,
        test_evidence_files={"tests/monitor.json": evidence},
        test_revision="c" * 40,
    )


@contextmanager
def _consumer_capability(
    tmp_path: Path,
    plan: dict[str, Any],
    controller_state: dict[str, Any],
    *,
    publish_mode: str = "canonical",
):
    root = tmp_path / f"consumer-{publish_mode}"
    runtime = root / "runtime"
    publication = runtime if publish_mode == "canonical" else root / "shadow"
    indexes = []
    for stage in profiles.STAGES:
        pointer_root = publication / "conditions" / stage.stage_id / "canonical"
        snapshot = compact.build_compact_snapshot(
            stage_id=stage.stage_id,
            physical_lanes=[],
            seed_records=[],
            frontend_static={},
            updated_at="2026-07-22T05:00:00+00:00",
        )
        compact.publish_compact_snapshot(
            pointer_root, *snapshot, pointer_name=consumer.POINTER_NAME
        )
        indexes.append(
            {
                "stage_id": stage.stage_id,
                "path": str((pointer_root / consumer.POINTER_NAME).resolve()),
            }
        )
    inputs = consumer.ConsumerInputs(
        plan=plan,
        state=controller_state,
        cohorts_by_plan_sha={},
    )
    inventory = consumer._condition_inventory(
        output_root=publication,
        inputs=inputs,
        indexes=indexes,
        observed_at="2026-07-22T05:00:00+00:00",
    )
    inventory_path = publication / "canonical" / "condition-indexes.json"
    consumer._atomic_json(inventory_path, inventory)

    handoff_path = None
    if publish_mode == "canonical":
        old_stop = root / "STOP_OLD_V1_HARVESTER"
        old_stop.write_text("stopped\n", encoding="utf-8")
        old_indexes = {}
        for stage in profiles.STAGES:
            path = root / "sealed-v1" / f"{stage.stage_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                contract.json_bytes(
                    {
                        "schema_version": consumer.CURRENT7_INDEX_SCHEMA,
                        "final_goal_stage_id": stage.stage_id,
                        "identity": f"sealed-{stage.stage_id}",
                    }
                )
            )
            old_indexes[stage.stage_id] = path
        handoff = consumer.build_v1_handoff_receipt(
            old_pid=2_000_000_000,
            stop_file=old_stop,
            condition_indexes=old_indexes,
        )
        handoff_path = root / "v1-handoff.json"
        consumer._atomic_json(handoff_path, handoff)

    result = {
        "publish_mode": publish_mode,
        "observed_at": "2026-07-22T05:00:00+00:00",
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "controller_state_sha256": controller_state["state_sha256"],
        "condition_inventory": inventory,
    }
    receipt_path = publication / "canonical" / "consumer-capability.json"
    with consumer.WriterLease(
        publication / "canonical" / "consumer-writer.lock"
    ) as lease:
        receipt = consumer.build_capability_receipt(
            result=result,
            inventory_path=inventory_path,
            poll_seconds=15,
            freshness_deadline_seconds=120,
            runtime_root=runtime,
            writer_lease=lease,
            handoff_receipt_path=handoff_path,
        )
        consumer._atomic_json(receipt_path, receipt)
        yield receipt_path, receipt


def _running_v1_with_scheduler(
    plan: dict[str, Any], scheduler: "FakePhaseAScheduler"
) -> dict[str, Any]:
    v1 = single_controller._initial_state(plan)
    entries = copy.deepcopy(v1["entries"])
    index = single_controller._task_index(plan)
    templates = single_controller._task_templates(plan)
    for offset, entry in enumerate(entries):
        task = single_controller._task_for_entry(
            entry, plan_index=index, templates=templates
        )
        entry["state"] = "running"
        entry["task_id"] = 880_000 + offset
        scheduler.rows[entry["dedupe_key"]] = {
            "id": entry["task_id"],
            "name": task["name"],
            "dedupe_key": task["dedupe_key"],
            "status": "running",
            "task_json": task,
        }
    return single_controller._advance_state({**v1, "entries": entries})


class FakePhaseAScheduler:
    def __init__(self):
        self.rows: dict[str, dict[str, Any]] = {}
        self.post_count = 0
        self.next_id = 990_000
        self.detail_reads: list[int] = []

    def latest_10000(self):
        return [copy.deepcopy(row) for row in self.rows.values()]

    def task_detail(self, task_id: int):
        self.detail_reads.append(int(task_id))
        for row in self.rows.values():
            if row["id"] == int(task_id):
                return copy.deepcopy(row)
        return None

    def post_task(self, task):
        self.post_count += 1
        self.next_id += 1
        row = {
            "id": self.next_id,
            "name": task["name"],
            "dedupe_key": task["dedupe_key"],
            "status": "queued",
            "task_json": copy.deepcopy(task),
        }
        self.rows[task["dedupe_key"]] = row
        return copy.deepcopy(row)


class PassingGateReader:
    def __init__(self, scheduler: FakePhaseAScheduler):
        self.scheduler = scheduler

    def lane_evidence(self, task_id: int, seeds):
        row = next(row for row in self.scheduler.rows.values() if row["id"] == task_id)
        task = row["task_json"]
        manifest = contract.batch_manifest_from_payload(task["payload_json"])
        receipts = []
        for ordinal, child in enumerate(manifest["ordered_children"]):
            legacy = {
                "schema_version": single_contract.STATUS_SCHEMA,
                "state": "completed",
                "terminal": True,
                "seed": child["seed"],
            }
            receipts.append(
                contract.seal_child_receipt(
                    {
                        "schema_version": contract.CHILD_RECEIPT_SCHEMA,
                        "protocol_version": contract.PROTOCOL_VERSION,
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
                        "legacy_status_sha256": contract.canonical_sha256(legacy),
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
        status = contract.seal_task_status(
            {
                "schema_version": contract.TASK_STATUS_SCHEMA,
                "protocol_version": contract.PROTOCOL_VERSION,
                "task_id": str(task_id),
                "manifest_sha256": manifest["manifest_sha256"],
                "state": "completed",
                "stop_requested": False,
                "stop_reason": None,
                "current_ordinal": None,
                "current_seed": None,
                "sealed_child_count": len(seeds),
                "completed_child_count": len(seeds),
                "failed_child_count": 0,
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


def _reach_prepared_cutover(tmp_path: Path):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    reader = PassingGateReader(scheduler)
    initial = driver.initial_driver_state(predecessor, plan, plan, capability)
    batch1, _ = driver.cycle(initial, plan, capability, scheduler, reader, apply=True)
    for dedupe in batch1["gate_lanes"]["batch1"]:
        scheduler.rows[dedupe]["status"] = "completed"
    batch4, _ = driver.cycle(batch1, plan, capability, scheduler, reader, apply=True)
    for dedupe in batch4["gate_lanes"]["batch4"]:
        scheduler.rows[dedupe]["status"] = "completed"
    awaiting, _ = driver.cycle(batch4, plan, capability, scheduler, reader, apply=True)
    final_entries = copy.deepcopy(predecessor["entries"])
    final_entries[0]["state"] = "completed"
    scheduler.rows[final_entries[0]["dedupe_key"]]["status"] = "completed"
    final_predecessor = single_controller._advance_state(
        {**predecessor, "entries": final_entries, "stop_requested": True}
    )
    stop = driver.write_stop_file(tmp_path / "PREPARED-STOP.json", final_predecessor)
    prepared, intents = driver.cycle(
        awaiting,
        plan,
        capability,
        scheduler,
        reader,
        apply=True,
        stop=stop,
        final_predecessor_state=final_predecessor,
        source_plan=plan,
    )
    assert intents == []
    assert prepared["phase"] == "cutover_prepared"
    assert scheduler.post_count == 8
    return plan, capability, scheduler, reader, awaiting, prepared


def test_production_driver_is_post0_by_default_and_gates_refill_until_two_waves(
    tmp_path: Path,
):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    state = driver.initial_driver_state(predecessor, plan, plan, capability)
    reader = PassingGateReader(scheduler)

    preview, intents = driver.cycle(
        state, plan, capability, scheduler, reader, apply=False
    )
    assert scheduler.post_count == 0
    assert len(intents) == 4
    assert {task["payload_json"]["batch_length"] for task in intents} == {1}
    assert preview["phase"] == "batch1"
    assert preview["refill_released"] is False
    assert preview["controller_state"] is None
    assert state["source_start"]["state_sha256"] == predecessor["state_sha256"]

    applied, intents = driver.cycle(
        state, plan, capability, scheduler, reader, apply=True
    )
    assert len(intents) == 4
    assert scheduler.post_count == 4
    assert applied["phase"] == "batch1"
    assert applied["refill_released"] is False
    assert applied["controller_state"] is None
    assert (
        sum(row["status"] in {"queued", "running"} for row in scheduler.rows.values())
        == 504
    )

    unchanged, intents = driver.cycle(
        applied, plan, capability, scheduler, reader, apply=True
    )
    assert intents == []
    assert scheduler.post_count == 4
    assert unchanged["source_start"] == applied["source_start"]

    for dedupe in applied["gate_lanes"]["batch1"]:
        scheduler.rows[dedupe]["status"] = "completed"
    batch4, intents = driver.cycle(
        unchanged, plan, capability, scheduler, reader, apply=True
    )
    assert batch4["phase"] == "batch4"
    assert batch4["refill_released"] is False
    assert len(intents) == 4
    assert {task["payload_json"]["batch_length"] for task in intents} == {4}
    assert scheduler.post_count == 8
    assert (
        sum(row["status"] in {"queued", "running"} for row in scheduler.rows.values())
        == 504
    )

    batch4_dedupes = set(batch4["gate_lanes"]["batch4"])
    for dedupe in batch4_dedupes:
        scheduler.rows[dedupe]["status"] = "completed"
    awaiting, intents = driver.cycle(
        batch4, plan, capability, scheduler, reader, apply=True
    )
    assert awaiting["phase"] == "awaiting_predecessor_stop"
    assert awaiting["refill_released"] is False
    assert awaiting["controller_state"] is None
    assert intents == []
    assert scheduler.post_count == 8

    stale_predecessor = single_controller._seal_state(
        {**predecessor, "stop_requested": True}
    )
    stale_stop = driver.write_stop_file(tmp_path / "STALE-STOP.json", stale_predecessor)
    with pytest.raises(RuntimeError, match="stale initial predecessor"):
        driver.cycle(
            awaiting,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            stop=stale_stop,
            final_predecessor_state=stale_predecessor,
            source_plan=plan,
        )

    final_predecessor = copy.deepcopy(predecessor)
    final_entries = copy.deepcopy(final_predecessor["entries"])
    final_entries[0]["state"] = "completed"
    scheduler.rows[final_entries[0]["dedupe_key"]]["status"] = "completed"
    final_predecessor = single_controller._advance_state(
        {
            **final_predecessor,
            "entries": final_entries,
            "stop_requested": True,
        }
    )
    stop = driver.write_stop_file(
        tmp_path / "STOP.json",
        final_predecessor,
    )
    prepared, intents = driver.cycle(
        awaiting,
        plan,
        capability,
        scheduler,
        reader,
        apply=True,
        stop=stop,
        final_predecessor_state=final_predecessor,
        source_plan=plan,
    )
    assert prepared["phase"] == "cutover_prepared"
    assert prepared["refill_released"] is False
    assert prepared["consumer_cutover_capability"] is None
    assert intents == []
    assert scheduler.post_count == 8
    with _consumer_capability(tmp_path, plan, prepared["controller_state"]) as (
        consumer_receipt,
        _receipt,
    ):
        refill, intents = driver.cycle(
            prepared,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=consumer_receipt,
        )
    assert refill["phase"] == "refill"
    assert refill["refill_released"] is True
    assert refill["cutover_source"]["state_sha256"] == final_predecessor["state_sha256"]
    assert len(intents) == 1
    assert scheduler.post_count == 9
    assert controller.active_physical_count(refill["controller_state"]) == 500


def test_production_driver_fails_closed_on_gate_terminal_and_capability_tamper(
    tmp_path: Path,
):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    state = driver.initial_driver_state(predecessor, plan, plan, capability)
    reader = PassingGateReader(scheduler)
    applied, _ = driver.cycle(state, plan, capability, scheduler, reader, apply=True)
    first = scheduler.rows[next(iter(applied["gate_lanes"]["batch1"]))]
    first["status"] = "failed"
    failed, intents = driver.cycle(
        applied, plan, capability, scheduler, reader, apply=True
    )
    assert failed["phase"] == "failed"
    assert intents == []
    assert scheduler.post_count == 4

    tampered = {**capability, "running_child_journal_visible": False}
    tampered_unsigned = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = contract.canonical_sha256(tampered_unsigned)
    with pytest.raises(RuntimeError, match="capability"):
        driver.initial_driver_state(predecessor, plan, plan, tampered)

    store = driver.AtomicStateStore(tmp_path / "driver-state.json")
    store.write(None, state, plan, capability)
    store.write(state, applied, plan, capability)
    assert store.load(plan, capability) == applied
    history = store.history / f"{state['state_sha256']}.json"
    damaged = json.loads(history.read_text())
    damaged["phase"] = "failed"
    history.write_text(json.dumps(damaged))
    with pytest.raises(RuntimeError, match="state seal|ancestor"):
        store.load(plan, capability)


def test_v2_gate_keeps_refill_zero_while_old_controller_and_harvester_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    state = driver.initial_driver_state(predecessor, plan, plan, capability)
    monkeypatch.setattr(consumer, "_pid_is_running", lambda _pid: True)

    preview, intents = driver.cycle(
        state,
        plan,
        capability,
        scheduler,
        PassingGateReader(scheduler),
        apply=False,
    )
    assert preview["schema_version"] == driver.DRIVER_STATE_SCHEMA
    assert preview["refill_released"] is False
    assert preview["controller_state"] is None
    assert {task["payload_json"]["batch_length"] for task in intents} == {1}
    assert scheduler.post_count == 0


def test_consumer_gate_refuses_invalid_receipts_and_catches_up_before_refill(
    tmp_path: Path,
):
    plan, capability, scheduler, reader, _awaiting, prepared = _reach_prepared_cutover(
        tmp_path
    )
    posts_before = scheduler.post_count

    with _consumer_capability(
        tmp_path, plan, prepared["controller_state"], publish_mode="shadow"
    ) as (shadow_path, _shadow):
        with pytest.raises(RuntimeError, match="capability/freshness"):
            driver.cycle(
                prepared,
                plan,
                capability,
                scheduler,
                reader,
                apply=True,
                consumer_capability_path=shadow_path,
            )
    assert scheduler.post_count == posts_before

    with _consumer_capability(tmp_path, plan, prepared["controller_state"]) as (
        receipt_path,
        receipt,
    ):
        tampered = copy.deepcopy(receipt)
        tampered["run_id"] = "tampered"
        consumer._atomic_json(receipt_path, tampered)
        with pytest.raises(RuntimeError, match="capability/freshness"):
            driver.cycle(
                prepared,
                plan,
                capability,
                scheduler,
                reader,
                apply=True,
                consumer_capability_path=receipt_path,
            )
        assert scheduler.post_count == posts_before

        stale = copy.deepcopy(receipt)
        stale["published_at"] = "2000-01-01T00:00:00+00:00"
        stale["receipt_sha256"] = contract.canonical_sha256(
            {key: item for key, item in stale.items() if key != "receipt_sha256"}
        )
        consumer._atomic_json(receipt_path, stale)
        with pytest.raises(RuntimeError, match="capability/freshness"):
            driver.cycle(
                prepared,
                plan,
                capability,
                scheduler,
                reader,
                apply=True,
                consumer_capability_path=receipt_path,
            )
        assert scheduler.post_count == posts_before

        consumer._atomic_json(receipt_path, receipt)
        state_path = tmp_path / "dry-run-driver-state.json"
        driver._atomic_write(state_path, prepared)
        state_before = state_path.read_bytes()
        preview, intents = driver.cycle(
            prepared,
            plan,
            capability,
            scheduler,
            reader,
            apply=False,
            consumer_capability_path=receipt_path,
        )
        assert preview["phase"] == "refill"
        assert len(intents) == 1
        assert scheduler.post_count == posts_before
        assert state_path.read_bytes() == state_before

        refill, intents = driver.cycle(
            prepared,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=receipt_path,
        )
        assert len(intents) == 1
        assert scheduler.post_count == posts_before + 1

        # The still-fresh C(n-1) capability is a normal poll race.  It pauses
        # rather than terminating watch or allowing an unaudited refill POST.
        catching_up, intents = driver.cycle(
            refill,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=receipt_path,
        )
        assert intents == []
        assert catching_up["controller_state"] == refill["controller_state"]
        assert scheduler.post_count == posts_before + 1

    with pytest.raises(RuntimeError, match="writer lease"):
        driver.cycle(
            catching_up,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=receipt_path,
        )
    assert scheduler.post_count == posts_before + 1

    with _consumer_capability(tmp_path, plan, catching_up["controller_state"]) as (
        caught_up_path,
        caught_up_receipt,
    ):
        stale = copy.deepcopy(caught_up_receipt)
        stale["published_at"] = "2000-01-01T00:00:00+00:00"
        stale["receipt_sha256"] = contract.canonical_sha256(
            {key: item for key, item in stale.items() if key != "receipt_sha256"}
        )
        consumer._atomic_json(caught_up_path, stale)
        with pytest.raises(RuntimeError, match="capability/freshness"):
            driver.cycle(
                catching_up,
                plan,
                capability,
                scheduler,
                reader,
                apply=True,
                consumer_capability_path=caught_up_path,
            )
        assert scheduler.post_count == posts_before + 1

        consumer._atomic_json(caught_up_path, caught_up_receipt)
        victim = next(
            entry
            for entry in catching_up["controller_state"]["entries"]
            if entry["parent_dedupe_key"] in scheduler.rows
            and entry["state"] in controller.ACTIVE_STATES
        )
        scheduler.rows[victim["parent_dedupe_key"]]["status"] = "completed"
        resumed, intents = driver.cycle(
            catching_up,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=caught_up_path,
        )
        assert resumed["phase"] == "refill"
        assert len(intents) == 1
        assert scheduler.post_count == posts_before + 2


def test_prepared_controller_artifact_crash_is_restart_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan, capability, scheduler, reader, awaiting, prepared = _reach_prepared_cutover(
        tmp_path
    )
    awaiting_root = driver._seal(
        {
            **{key: item for key, item in awaiting.items() if key != "state_sha256"},
            "revision": 0,
            "parent_state_sha256": None,
        }
    )
    prepared_root = driver._seal(
        {
            **{key: item for key, item in prepared.items() if key != "state_sha256"},
            "revision": 1,
            "parent_state_sha256": awaiting_root["state_sha256"],
        }
    )
    driver.validate_driver_state(awaiting_root, plan, capability)
    driver.validate_driver_state(prepared_root, plan, capability)
    store = driver.AtomicStateStore(tmp_path / "artifact-crash-state.json")
    store.write(None, awaiting_root, plan, capability)
    original_write = driver._atomic_write

    def crash_before_state(path: Path, value):
        if path == store.path:
            raise OSError("synthetic crash before driver-state replace")
        return original_write(path, value)

    monkeypatch.setattr(driver, "_atomic_write", crash_before_state)
    with pytest.raises(OSError, match="synthetic crash"):
        store.write(awaiting_root, prepared_root, plan, capability)
    assert json.loads(store.path.read_text()) == awaiting_root
    assert (
        json.loads(store.controller_path.read_text())
        == prepared_root["controller_state"]
    )

    monkeypatch.setattr(driver, "_atomic_write", original_write)
    store.write(awaiting_root, prepared_root, plan, capability)
    assert store.load(plan, capability) == prepared_root

    with _consumer_capability(tmp_path, plan, prepared_root["controller_state"]) as (
        receipt_path,
        _receipt,
    ):
        refill, _ = driver.cycle(
            prepared_root,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=receipt_path,
        )
    store.write(prepared_root, refill, plan, capability)
    future_controller = controller.request_stop(refill["controller_state"], plan)
    future = driver._advance_driver(
        refill, plan, capability, controller_state=future_controller
    )
    monkeypatch.setattr(driver, "_atomic_write", crash_before_state)
    with pytest.raises(OSError, match="synthetic crash"):
        store.write(refill, future, plan, capability)
    assert store.load(plan, capability) == refill
    assert json.loads(store.controller_path.read_text()) == future_controller


def test_watch_restart_keeps_500_and_never_duplicates_a_submission(tmp_path: Path):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    initial = driver.initial_driver_state(predecessor, plan, plan, capability)
    store = driver.AtomicStateStore(tmp_path / "watch-state.json")
    store.write(None, initial, plan, capability)
    reader = PassingGateReader(scheduler)

    def finish_batch1(_seconds: float) -> None:
        saved = store.load(plan, capability)
        for dedupe in saved["gate_lanes"]["batch1"]:
            scheduler.rows[dedupe]["status"] = "completed"

    batch4, summaries = driver.run_cycles(
        initial,
        plan,
        plan,
        (),
        scheduler,
        reader,
        capability_loader=lambda: capability,
        consumer_capability_path_loader=lambda: None,
        predecessor_loader=lambda: predecessor,
        predecessor_stop_loader=lambda: None,
        supervisor_stop_loader=lambda: None,
        store=store,
        apply=True,
        watch=True,
        poll_seconds=0.1,
        max_cycles=2,
        sleeper=finish_batch1,
    )
    assert [item["physical_active_lanes"] for item in summaries] == [504, 504]
    assert scheduler.post_count == 8
    assert store.load(plan, capability) == batch4

    for dedupe in batch4["gate_lanes"]["batch4"]:
        scheduler.rows[dedupe]["status"] = "completed"
    final_entries = copy.deepcopy(predecessor["entries"])
    final_entries[0]["state"] = "completed"
    scheduler.rows[final_entries[0]["dedupe_key"]]["status"] = "completed"
    final_predecessor = single_controller._advance_state(
        {**predecessor, "entries": final_entries, "stop_requested": True}
    )
    stop = driver.write_stop_file(tmp_path / "PREDECESSOR-STOP.json", final_predecessor)
    restarted = store.load(plan, capability)
    prepared, summaries = driver.run_cycles(
        restarted,
        plan,
        plan,
        (),
        scheduler,
        reader,
        capability_loader=lambda: capability,
        consumer_capability_path_loader=lambda: None,
        predecessor_loader=lambda: final_predecessor,
        predecessor_stop_loader=lambda: stop,
        supervisor_stop_loader=lambda: None,
        store=store,
        apply=True,
        watch=True,
        poll_seconds=0.1,
        max_cycles=1,
        sleeper=lambda _seconds: None,
    )
    assert prepared["phase"] == "cutover_prepared"
    assert summaries[0]["planned_task_count"] == 0
    assert scheduler.post_count == 8
    assert json.loads(store.controller_path.read_text()) == prepared["controller_state"]

    with _consumer_capability(tmp_path, plan, prepared["controller_state"]) as (
        consumer_receipt,
        _receipt,
    ):
        refill, summaries = driver.run_cycles(
            store.load(plan, capability),
            plan,
            plan,
            (),
            scheduler,
            reader,
            capability_loader=lambda: capability,
            consumer_capability_path_loader=lambda: consumer_receipt,
            predecessor_loader=lambda: final_predecessor,
            predecessor_stop_loader=lambda: stop,
            supervisor_stop_loader=lambda: None,
            store=store,
            apply=True,
            watch=True,
            poll_seconds=0.1,
            max_cycles=1,
            sleeper=lambda _seconds: None,
        )
    assert refill["phase"] == "refill"
    assert summaries[0]["physical_active_lanes"] == 500
    assert scheduler.post_count == 9

    restarted = store.load(plan, capability)
    with _consumer_capability(tmp_path, plan, restarted["controller_state"]) as (
        consumer_receipt,
        _receipt,
    ):
        stable, summaries = driver.run_cycles(
            restarted,
            plan,
            plan,
            (),
            scheduler,
            reader,
            capability_loader=lambda: capability,
            consumer_capability_path_loader=lambda: consumer_receipt,
            predecessor_loader=lambda: final_predecessor,
            predecessor_stop_loader=lambda: stop,
            supervisor_stop_loader=lambda: None,
            store=store,
            apply=True,
            watch=True,
            poll_seconds=0.1,
            max_cycles=1,
            sleeper=lambda _seconds: None,
        )
    assert summaries[0]["planned_task_count"] == 0
    assert scheduler.post_count == 9
    assert driver.operational_counts(stable)["physical_active_lanes"] == 500


def test_post_before_state_write_crash_recovers_gate_and_refill_without_duplicate(
    tmp_path: Path,
):
    plan = launch.validate_launch_plan(_rendered_plan())
    capability = _monitoring_capability(tmp_path)
    scheduler = FakePhaseAScheduler()
    predecessor = _running_v1_with_scheduler(plan, scheduler)
    initial = driver.initial_driver_state(predecessor, plan, plan, capability)
    store = driver.AtomicStateStore(tmp_path / "crash-window-state.json")
    store.write(None, initial, plan, capability)
    reader = PassingGateReader(scheduler)

    # Simulate death after all four POSTs return but before run_cycles writes the
    # returned state.  Reconciliation from the old state must find the exact
    # deterministic tasks in latest10k and recover their Scheduler ids.
    crashed_gate, first_gate_intents = driver.cycle(
        initial, plan, capability, scheduler, reader, apply=True
    )
    assert len(first_gate_intents) == 4
    assert scheduler.post_count == 4
    assert store.load(plan, capability) == initial
    gate_task_ids = {
        task["dedupe_key"]: scheduler.rows[task["dedupe_key"]]["id"]
        for task in first_gate_intents
    }
    recovered_gate, recovered_gate_intents = driver.cycle(
        store.load(plan, capability),
        plan,
        capability,
        scheduler,
        reader,
        apply=True,
    )
    assert scheduler.post_count == 4
    assert recovered_gate_intents == first_gate_intents
    assert {
        dedupe: lane["task_id"]
        for dedupe, lane in recovered_gate["gate_lanes"]["batch1"].items()
    } == gate_task_ids
    assert crashed_gate["gate_lanes"] == recovered_gate["gate_lanes"]
    store.write(initial, recovered_gate, plan, capability)

    for dedupe in recovered_gate["gate_lanes"]["batch1"]:
        scheduler.rows[dedupe]["status"] = "completed"
    batch4, _ = driver.cycle(
        recovered_gate, plan, capability, scheduler, reader, apply=True
    )
    store.write(recovered_gate, batch4, plan, capability)
    for dedupe in batch4["gate_lanes"]["batch4"]:
        scheduler.rows[dedupe]["status"] = "completed"
    awaiting, _ = driver.cycle(batch4, plan, capability, scheduler, reader, apply=True)
    store.write(batch4, awaiting, plan, capability)
    assert scheduler.post_count == 8

    final_entries = copy.deepcopy(predecessor["entries"])
    final_entries[0]["state"] = "completed"
    scheduler.rows[final_entries[0]["dedupe_key"]]["status"] = "completed"
    final_predecessor = single_controller._advance_state(
        {**predecessor, "entries": final_entries, "stop_requested": True}
    )
    stop = driver.write_stop_file(
        tmp_path / "CRASH-WINDOW-PREDECESSOR-STOP.json", final_predecessor
    )

    # The first cutover phase persists an exact controller artifact without a
    # refill POST, breaking the controller/canonical-consumer dependency cycle.
    prepared, prepare_intents = driver.cycle(
        awaiting,
        plan,
        capability,
        scheduler,
        reader,
        apply=True,
        stop=stop,
        final_predecessor_state=final_predecessor,
        source_plan=plan,
    )
    assert prepared["phase"] == "cutover_prepared"
    assert prepare_intents == []
    assert scheduler.post_count == 8
    store.write(awaiting, prepared, plan, capability)
    assert json.loads(store.controller_path.read_text()) == prepared["controller_state"]

    # Repeat the POST-before-state-write crash window for the first natural
    # refill. latest10k dedupe recovers the exact posted row on restart.
    with _consumer_capability(tmp_path, plan, prepared["controller_state"]) as (
        consumer_receipt,
        _receipt,
    ):
        crashed_refill, first_refill_intents = driver.cycle(
            prepared,
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=consumer_receipt,
        )
        assert len(first_refill_intents) == 1
        assert scheduler.post_count == 9
        assert store.load(plan, capability) == prepared
        refill_task = first_refill_intents[0]
        refill_task_id = scheduler.rows[refill_task["dedupe_key"]]["id"]
        recovered_refill, recovered_refill_intents = driver.cycle(
            store.load(plan, capability),
            plan,
            capability,
            scheduler,
            reader,
            apply=True,
            consumer_capability_path=consumer_receipt,
        )
    assert scheduler.post_count == 9
    assert recovered_refill_intents == first_refill_intents
    recovered_entry = next(
        entry
        for entry in recovered_refill["controller_state"]["entries"]
        if entry["parent_dedupe_key"] == refill_task["dedupe_key"]
    )
    assert recovered_entry["task_id"] == refill_task_id
    assert crashed_refill["controller_state"] == recovered_refill["controller_state"]
    all_seeds = [
        seed
        for entry in recovered_refill["controller_state"]["entries"]
        for seed in entry["seeds"]
    ]
    assert len(all_seeds) == len(set(all_seeds))
    assert driver.operational_counts(recovered_refill)["physical_active_lanes"] == 500


def test_scheduler_client_retries_transient_get_but_never_blindly_retries_post(
    monkeypatch: pytest.MonkeyPatch,
):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"[]"

    calls: list[str] = []
    sleeps: list[float] = []

    def flaky(request, timeout):
        calls.append(request.get_method())
        if len(calls) < 3:
            raise driver.urllib.error.URLError("transient")
        return Response()

    monkeypatch.setattr(driver.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(driver.time, "sleep", sleeps.append)
    api = driver.SchedulerApi(
        "http://scheduler", get_attempts=3, get_backoff_seconds=0.25
    )
    assert api.latest_10000() == []
    assert calls == ["GET", "GET", "GET"]
    assert sleeps == [0.25, 0.5]

    calls.clear()

    def failed_post(request, timeout):
        calls.append(request.get_method())
        raise driver.urllib.error.URLError("ambiguous post")

    monkeypatch.setattr(driver.urllib.request, "urlopen", failed_post)
    api = driver.SchedulerApi("http://scheduler", apply=True, get_attempts=5)
    with pytest.raises(driver.urllib.error.URLError):
        api.post_task(contract.build_batch_task(_child_tasks(1)))
    assert calls == ["POST"]


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

    # Crash window: an immutable receipt that lands before the mutable cursor
    # advances is deliberately not public. Running-parent visibility is exactly
    # the journal-declared sealed prefix; terminal recovery needs a separate
    # attestation rather than an opportunistic receipt scan.
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
    assert crash_window == []
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

    cohort_plan = {**plan, "launch_plan_sha256": "e" * 64}
    cohort_item = {
        "task_id": 99202,
        "status": "completed",
        "source_harvest_cohort_id": "legacy-200g",
        "source_launch_plan_sha256": "e" * 64,
    }
    cohort_mixed = harvest.harvest_mixed_inventory(
        [cohort_item],
        plan=plan,
        manifest=manifest,
        remote=remote,
        cohort_contexts={
            "legacy-200g": {"plan": cohort_plan, "manifest": {"cohort": 200}}
        },
    )
    assert cohort_mixed["records"][0]["harvest_cohort_id"] == "legacy-200g"
    refused_cohort = harvest.harvest_mixed_inventory(
        [cohort_item], plan=plan, manifest=manifest, remote=remote
    )
    assert refused_cohort["authenticated_seed_count"] == 0
    assert refused_cohort["refused_physical_task_count"] == 1

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
