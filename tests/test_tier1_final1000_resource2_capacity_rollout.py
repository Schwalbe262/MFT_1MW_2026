from __future__ import annotations

import copy

import pytest

from tests.test_tier1_final1000_rolling_migration import _successor_plan
from tests.test_tier1_final1000_slurm_harvest import _bindings
from tools import tier1_final1000_resource2_canary as canary
from tools import tier1_final1000_resource2_capacity_rollout as rollout
from tools import tier1_final1000_rolling_migration as migration
from tools import tier1_final1000_slurm_controller as controller
from tools import tier1_final1000_slurm_launch as launch
from tools import tier1_final1000_slurm_harvest as harvest
from tools import tier1_final1000_stage_profiles as profiles


PLAN_SHA = "90af982bf32b8fb005b8557db77f1822f5cc8a2c3dc6e377c29da41e4cdfe730"
PACKAGE_SHA = "12f8ca284d4cf3d12f8bc9e6ac528473e9872e8e81300849dfc1ae03a6e9a7b8"
SUBMISSION_SHA = "da0bcbde3c626ce2b9f003746b8f0bb5b6e4f2130edc54ac6bc8123545bdd742"


def _config(*, plan_sha: str = PLAN_SHA) -> dict:
    unsigned = {
        "schema_version": rollout.CONFIG_SCHEMA,
        "predecessor_launch_plan_sha256": plan_sha,
        "predecessor_controller_revision": "4aaf99a02f54d134dfe202b2867a73bab1460577",
        "resource2_canary_task_id": 86501,
        "resource2_canary_package_sha256": PACKAGE_SHA,
        "resource2_canary_submission_receipt_sha256": SUBMISSION_SHA,
        "minimum_active_target": 500,
        "maximum_active_target": 1000,
        "stage_weight_basis": copy.deepcopy(rollout.STAGE_WEIGHT_BASIS),
        "resource2_promoted_stage_ids": [rollout.ENTRY_STAGE_ID],
        "cross_stage_resource2_expansion_allowed": False,
        "promotion_basis": {
            "entry-1200-t125": "terminal-gate-task-86501",
            "bridge-1150-t115": "not-promoted-requires-own-2cpu-terminal-gate",
            "close-1075-t107p5": "not-promoted-requires-own-2cpu-terminal-gate",
            "final-1000-t100": "not-promoted-requires-own-2cpu-terminal-gate",
        },
        "resource2_policy": copy.deepcopy(migration.RESOURCE2_POLICY),
        "fallback_resource4_policy": copy.deepcopy(migration.SUCCESSOR_POLICY),
        "capacity_policy": {
            "allocation_states_counted": ["active"],
            "resource_pool": "cpu",
            "exclusive_node": False,
            "scheduling_profile": "standard",
            "slot_formula": "exact-bin-pack(2cpu-promoted-stages,4cpu-unpromoted-stages,28672MiB,32-workers)",
            "draining_counted": False,
            "pending_counted": False,
            "warm_counted": False,
            "recompute_each_cycle": True,
            "target_formula": "max(500,min(exact_mixed_fit,maximum_active_target))",
        },
        "failure_policy": {
            "resource2_terminal_failure_threshold": 1,
            "fallback_mode": rollout.REFILL_MODE_RESOURCE4,
            "fallback_active_target": 500,
            "cancel_existing_tasks": False,
            "retry_resource2_automatically": False,
        },
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return rollout.validate_config(
        {**unsigned, "config_sha256": rollout.canonical_sha256(unsigned)}
    )


def _gate(config: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    terminal_unsigned = {
        "task_id": 86501,
        "package_sha256": PACKAGE_SHA,
        "scheduler_status": "completed",
        "gate_results": {
            name: True for name in rollout.REQUIRED_RESOURCE2_TERMINAL_GATES
        },
        "promotion_eligible": True,
    }
    terminal = {
        **terminal_unsigned,
        "terminal_sha256": rollout.canonical_sha256(terminal_unsigned),
    }

    def validate_terminal(value: dict) -> dict:
        unsigned = {
            key: item for key, item in value.items() if key != "terminal_sha256"
        }
        if value.get("terminal_sha256") != rollout.canonical_sha256(unsigned):
            raise RuntimeError("resource2 terminal evidence seal mismatch")
        return copy.deepcopy(dict(value))

    monkeypatch.setattr(
        rollout,
        "validate_terminal_evidence",
        validate_terminal,
    )
    unsigned = {
        "schema_version": canary.REMOTE_TERMINAL_SCHEMA,
        "task_id": 86501,
        "package_sha256": PACKAGE_SHA,
        "submission_receipt_sha256": SUBMISSION_SHA,
        "scheduler_status": "completed",
        "terminal_evidence": terminal,
        "terminal_sha256": terminal["terminal_sha256"],
        "promotion_eligible": True,
        "scheduler_access": "GET-only",
        "remote_access": "read-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "publication_count": 0,
        "remote_write_count": 0,
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {
        **unsigned,
        "remote_terminal_sha256": rollout.canonical_sha256(unsigned),
    }


def _reseal_gate(gate: dict, *, reseal_terminal: bool = True) -> None:
    terminal = gate["terminal_evidence"]
    if reseal_terminal:
        terminal_unsigned = {
            key: value for key, value in terminal.items() if key != "terminal_sha256"
        }
        terminal["terminal_sha256"] = rollout.canonical_sha256(terminal_unsigned)
    gate["terminal_sha256"] = terminal["terminal_sha256"]
    unsigned = {
        key: value for key, value in gate.items() if key != "remote_terminal_sha256"
    }
    gate["remote_terminal_sha256"] = rollout.canonical_sha256(unsigned)


def _allocation_rows(*, include_41st: bool = True) -> list[dict]:
    shapes = [
        (13741, 64, 700751),
        (13742, 64, 666799),
        (13744, 44, 691200),
        (13745, 44, 691200),
        (13747, 44, 691200),
        (13749, 64, 754907),
        (13750, 32, 631613),
        (13754, 32, 520087),
        (13756, 32, 472983),
        (13759, 28, 729988),
        (13762, 28, 693862),
        (13765, 64, 180491),
        (13768, 28, 673137),
        (13772, 28, 663248),
        (13774, 20, 938463),
        (13776, 36, 947945),
        (13777, 28, 658914),
        (13780, 36, 928621),
        (13781, 28, 508836),
        (13782, 8, 750969),
        (13783, 34, 568387),
        (13784, 28, 369915),
        (13785, 20, 910571),
        (13899, 64, 407117),
        (13957, 48, 964242),
        (13963, 40, 691200),
        (14004, 20, 887454),
        (14009, 64, 764630),
        (14010, 64, 764630),
        (14011, 64, 489536),
        (14021, 64, 587519),
        (14022, 64, 480466),
        (14029, 64, 668608),
        (14032, 64, 552382),
        (14033, 64, 585135),
        (14042, 64, 500578),
        (14043, 64, 596882),
        (14045, 64, 574605),
        (14046, 64, 460772),
        (14060, 64, 629892),
    ]
    if include_41st:
        shapes.append((14099, 48, 698814))
    return [
        {
            "id": allocation_id,
            "slurm_job_id": str(700000 + allocation_id),
            "account_name": "test",
            "partition": "cpu2",
            "node_name": f"n{offset:03d}",
            "state": "active",
            "resource_pool": "cpu",
            "exclusive_node": 0,
            "total_cpus": cpus,
            "total_memory_mb": memory,
        }
        for offset, (allocation_id, cpus, memory) in enumerate(shapes)
    ]


def _previous_plan() -> dict:
    plan = copy.deepcopy(_successor_plan())
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in plan["task_waves"]["canaries"]
    }
    canaries = []
    ramp = []
    for stage in profiles.STAGES:
        quota = launch.PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
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
        launch.PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS
    )
    unsigned = {
        key: value for key, value in plan.items() if key != "launch_plan_sha256"
    }
    plan["launch_plan_sha256"] = rollout.canonical_sha256(unsigned)
    return migration.validate_previous_successor_plan(plan)


def _predecessor_state(plan: dict) -> dict:
    state = controller._initial_state(plan)
    cohort_id = "successor-123456789abc"
    for task_id, entry in enumerate(state["entries"], start=100_000):
        entry.update(
            {
                "task_id": task_id,
                "state": "running",
                "origin": "successor",
                "resource_policy_id": migration.SUCCESSOR_RESOURCE_POLICY_ID,
                "harvest_cohort_id": cohort_id,
            }
        )
    state.update(
        {
            "revision": 17,
            "stop_requested": True,
            "ramp_released": True,
            "canary_passed_stage_ids": sorted(profiles.BY_ID),
            "scheduler_submit_count": 500,
        }
    )
    migration_value = {
        "schema_version": migration.CHAINED_MIGRATION_SCHEMA,
        "predecessor_harvest_cohort_ids": [],
        "successor_harvest_cohort_id": cohort_id,
        "harvest_cohorts": {
            cohort_id: migration.chained_harvest_cohort_identity(
                cohort_id=cohort_id,
                plan=plan,
                resource_policy_ids=[migration.SUCCESSOR_RESOURCE_POLICY_ID],
            )
        },
    }
    state["rolling_migration"] = rollout._seal_nested(migration_value)
    return controller._seal_state(state)


class _Scheduler:
    def __init__(self):
        self.post_count = 0
        self.next_id = 200_000
        self.by_id: dict[int, dict] = {}
        self.by_dedupe: dict[str, dict] = {}

    def list_namespace_tasks(self):
        return copy.deepcopy(list(self.by_id.values()))

    def list_allocations(self):
        return _allocation_rows()

    def get_task(self, task_id: int):
        return copy.deepcopy(self.by_id.get(task_id))

    def find_task_by_dedupe(self, dedupe_key: str):
        return copy.deepcopy(self.by_dedupe.get(dedupe_key))

    def submit_task(self, payload: dict):
        self.next_id += 1
        task = copy.deepcopy(payload)
        task.update({"id": self.next_id, "task_id": self.next_id, "status": "queued"})
        self.by_id[self.next_id] = task
        self.by_dedupe[task["dedupe_key"]] = task
        self.post_count += 1
        return copy.deepcopy(task)


def _prepared(monkeypatch: pytest.MonkeyPatch):
    plan = _previous_plan()
    config = _config(plan_sha=plan["launch_plan_sha256"])
    gate = _gate(config, monkeypatch)
    predecessor = _predecessor_state(plan)
    monkeypatch.setattr(
        rollout,
        "_validate_previous_successor_state",
        lambda value, _plan: copy.deepcopy(dict(value)),
    )
    state = rollout.prepare_state(
        predecessor,
        plan=plan,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        require_predecessor_stopped=True,
    )
    return plan, config, gate, predecessor, state


def test_capacity_is_exact_mixed_fit_and_ignores_draining():
    config = _config()
    forty = rollout.capacity_snapshot(
        _allocation_rows(include_41st=False), config=config
    )
    assert forty["resource4_exact_fit_slots"] == 447
    assert forty["resource2_exact_fit_slots"] == 695
    assert forty["mixed_exact_fit_slots"] == 559
    assert forty["bounded_active_target"] == 559
    assert forty["stage_active_targets"] == {
        "entry-1200-t125": 223,
        "bridge-1150-t115": 179,
        "close-1075-t107p5": 101,
        "final-1000-t100": 56,
    }

    forty_one = rollout.capacity_snapshot(_allocation_rows(), config=config)
    assert forty_one["resource4_exact_fit_slots"] == 459
    assert forty_one["resource2_exact_fit_slots"] == 719
    assert forty_one["mixed_exact_fit_slots"] == 574
    assert forty_one["bounded_active_target"] == 574
    assert forty_one["stage_active_targets"] == {
        "entry-1200-t125": 230,
        "bridge-1150-t115": 184,
        "close-1075-t107p5": 103,
        "final-1000-t100": 57,
    }
    draining = _allocation_rows()
    draining[-1]["state"] = "draining"
    assert (
        rollout.capacity_snapshot(draining, config=config)["mixed_exact_fit_slots"]
        == 559
    )


def test_entry_gate_cannot_promote_the_other_three_stages():
    config = _config()
    assert config["resource2_promoted_stage_ids"] == [rollout.ENTRY_STAGE_ID]
    for stage_id in set(profiles.BY_ID) - {rollout.ENTRY_STAGE_ID}:
        assert (
            rollout.capacity_snapshot(_allocation_rows(), config=config)[
                "stage_refill_policy_ids"
            ][stage_id]
            == migration.SUCCESSOR_RESOURCE_POLICY_ID
        )
    widened = copy.deepcopy(config)
    widened["resource2_promoted_stage_ids"].append("bridge-1150-t115")
    unsigned = {key: value for key, value in widened.items() if key != "config_sha256"}
    widened["config_sha256"] = rollout.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="config seal mismatch"):
        rollout.validate_config(widened)


def test_rollout_preserves_prefix_and_refills_mixed_policies(monkeypatch):
    plan, config, gate, predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    writes = []
    updated, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda value: writes.append(copy.deepcopy(dict(value))),
    )
    prefix_count = updated["resource2_rollout"]["predecessor_entry_count"]
    assert updated["entries"][:prefix_count] == predecessor["entries"]
    assert (
        rollout.canonical_sha256(
            rollout._entry_identities(updated["entries"][:prefix_count])
        )
        == updated["resource2_rollout"]["predecessor_entry_identities_sha256"]
    )
    new = updated["entries"][prefix_count:]
    assert len(new) == 74
    assert scheduler.post_count == 74
    assert result["logical_active_target"] == 574
    assert (
        sum(
            item["resource_policy_id"] == migration.RESOURCE2_RESOURCE_POLICY_ID
            for item in new
        )
        == 30
    )
    assert (
        sum(
            item["resource_policy_id"] == migration.SUCCESSOR_RESOURCE_POLICY_ID
            for item in new
        )
        == 44
    )
    assert {
        item["stage_id"]
        for item in new
        if item["resource_policy_id"] == migration.RESOURCE2_RESOURCE_POLICY_ID
    } == {rollout.ENTRY_STAGE_ID}
    for task in scheduler.by_id.values():
        stage_id = task["payload_json"]["final_goal_stage_id"]
        if stage_id == rollout.ENTRY_STAGE_ID:
            assert task["cpus"] == 2
            assert task["payload_json"]["inference_threads"] == 2
            assert task["payload_json"]["scheduler_cpus"] == 2
            for name in migration.RESOURCE2_THREAD_ENVIRONMENT:
                assert f"export {name}=2" in task["command"]
        else:
            assert task["cpus"] == 4
            assert task["payload_json"]["inference_threads"] == 8
    cohort_id = updated["rolling_migration"]["successor_harvest_cohort_id"]
    cohort = harvest._validate_plan_bindings(
        role="successor",
        cohort_id=cohort_id,
        plan=plan,
        bindings=_bindings(plan),
        resource_policy_ids=[
            migration.SUCCESSOR_RESOURCE_POLICY_ID,
            migration.RESOURCE2_RESOURCE_POLICY_ID,
        ],
    )
    resource2_entry = next(
        item
        for item in new
        if item["resource_policy_id"] == migration.RESOURCE2_RESOURCE_POLICY_ID
    )
    reproduced, _binding, observed_cohort, observed_policy = (
        harvest._entry_task_context(
            resource2_entry,
            plan=plan,
            plan_index=controller._task_index(plan),
            templates=controller._task_templates(plan),
            cohorts={cohort_id: cohort},
            rolling_migration=True,
        )
    )
    assert reproduced["cpus"] == 2
    assert observed_cohort == cohort_id
    assert observed_policy == migration.RESOURCE2_RESOURCE_POLICY_ID
    assert writes


def test_first_resource2_failure_switches_only_future_refills_to_4cpu(
    monkeypatch,
):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    state, _ = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    failed = next(task for task in scheduler.by_id.values() if task["cpus"] == 2)
    scheduler.by_id[failed["id"]]["status"] = "failed"
    scheduler.by_dedupe[failed["dedupe_key"]]["status"] = "failed"
    state, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert result["refill_mode"] == rollout.REFILL_MODE_RESOURCE4
    assert result["actions"][-1]["cancelled_task_ids"] == []
    posts_before = scheduler.post_count

    active_prefix = [
        entry
        for entry in state["entries"]
        if entry["state"] in controller.ACTIVE_STATES
        and entry["resource_policy_id"] == migration.SUCCESSOR_RESOURCE_POLICY_ID
    ]
    plan_index = controller._task_index(plan)
    for entry in active_prefix[:80]:
        task = copy.deepcopy(plan_index[entry["dedupe_key"]])
        task.update(
            {
                "id": entry["task_id"],
                "task_id": entry["task_id"],
                "status": "completed",
            }
        )
        scheduler.by_id[int(entry["task_id"])] = task
        scheduler.by_dedupe[entry["dedupe_key"]] = task
    state, _ = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    new_tasks = list(scheduler.by_id.values())[posts_before:]
    assert new_tasks
    assert {task["cpus"] for task in new_tasks} == {4}
    assert state["resource2_rollout"]["refill_mode"] == rollout.REFILL_MODE_RESOURCE4


def test_terminal_gate_is_mandatory_before_any_scheduler_post(monkeypatch):
    plan = _previous_plan()
    config = _config(plan_sha=plan["launch_plan_sha256"])
    gate = _gate(config, monkeypatch)
    gate["scheduler_status"] = "running"
    unsigned = {
        key: value for key, value in gate.items() if key != "remote_terminal_sha256"
    }
    gate["remote_terminal_sha256"] = rollout.canonical_sha256(unsigned)
    monkeypatch.setattr(
        rollout,
        "_validate_previous_successor_state",
        lambda value, _plan: copy.deepcopy(dict(value)),
    )
    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        rollout.prepare_state(
            _predecessor_state(plan),
            plan=plan,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            require_predecessor_stopped=True,
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("task_id", 90_001),
        ("package_sha256", "8" * 64),
        ("scheduler_status", "failed"),
    ],
)
def test_terminal_gate_cross_binds_inner_evidence(
    monkeypatch, field: str, changed: object
):
    config = _config()
    gate = _gate(config, monkeypatch)
    gate["terminal_evidence"][field] = changed
    _reseal_gate(gate)
    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        rollout.validate_promotion_gate(gate, config=config)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("task_id", 90_001),
        ("package_sha256", "8" * 64),
        ("submission_receipt_sha256", "9" * 64),
        ("scheduler_status", "failed"),
        ("terminal_sha256", "a" * 64),
    ],
)
def test_terminal_gate_binds_outer_identity_to_config_and_terminal(
    monkeypatch, field: str, changed: object
):
    config = _config()
    gate = _gate(config, monkeypatch)
    gate[field] = changed
    unsigned = {
        key: value for key, value in gate.items() if key != "remote_terminal_sha256"
    }
    gate["remote_terminal_sha256"] = rollout.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        rollout.validate_promotion_gate(gate, config=config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "false", "truthy-int"])
def test_terminal_gate_requires_exact_identity_typed_true_gate_set(
    monkeypatch, mutation: str
):
    config = _config()
    gate = _gate(config, monkeypatch)
    gates = gate["terminal_evidence"]["gate_results"]
    name = "exact_two_cpu_affinity"
    if mutation == "missing":
        gates.pop(name)
    elif mutation == "extra":
        gates["unsealed_extra_gate"] = True
    elif mutation == "false":
        gates[name] = False
    else:
        gates[name] = 1
    _reseal_gate(gate)
    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        rollout.validate_promotion_gate(gate, config=config)


def test_terminal_gate_rejects_inner_and_outer_sha_drift(monkeypatch):
    config = _config()
    gate = _gate(config, monkeypatch)
    rollout.validate_promotion_gate(gate, config=config)

    inner_drift = copy.deepcopy(gate)
    inner_drift["terminal_evidence"]["task_id"] = 90_001
    _reseal_gate(inner_drift, reseal_terminal=False)
    with pytest.raises(RuntimeError, match="terminal evidence seal mismatch"):
        rollout.validate_promotion_gate(inner_drift, config=config)

    outer_drift = copy.deepcopy(gate)
    outer_drift["remote_terminal_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="promotion gate did not pass"):
        rollout.validate_promotion_gate(outer_drift, config=config)


def test_immediate_resource2_submit_failure_is_persisted_as_fallback(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)

    class ImmediateFailureScheduler(_Scheduler):
        failed_once = False

        def submit_task(self, payload: dict):
            task = super().submit_task(payload)
            if payload["cpus"] == 2 and not self.failed_once:
                self.failed_once = True
                self.by_id[task["id"]]["status"] = "failed"
                self.by_dedupe[task["dedupe_key"]]["status"] = "failed"
                task["status"] = "failed"
            return task

    scheduler = ImmediateFailureScheduler()
    state, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert result["refill_mode"] == rollout.REFILL_MODE_RESOURCE4
    assert result["logical_active_target"] == 500
    assert state["resource2_rollout"]["resource2_terminal_failure_task_ids"]
    assert scheduler.post_count == 1
    assert len(scheduler.by_id) == 1
    rollout.validate_state(
        state,
        plan=plan,
        config=config,
        terminal_gate=gate,
    )


def test_failure_after_n_halts_every_later_post(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)

    class ThirdResource2FailureScheduler(_Scheduler):
        resource2_count = 0
        failed_at_post: int | None = None

        def submit_task(self, payload: dict):
            task = super().submit_task(payload)
            if payload["cpus"] == 2:
                self.resource2_count += 1
                if self.resource2_count == 3:
                    self.failed_at_post = self.post_count
                    self.by_id[task["id"]]["status"] = "failed"
                    self.by_dedupe[task["dedupe_key"]]["status"] = "failed"
                    task["status"] = "failed"
            return task

    scheduler = ThirdResource2FailureScheduler()
    state, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.failed_at_post is not None
    assert scheduler.post_count == scheduler.failed_at_post
    assert scheduler.resource2_count == 3
    assert result["refill_mode"] == rollout.REFILL_MODE_RESOURCE4
    assert result["actions"][-1]["detected_failure_task_ids"]

    posts = scheduler.post_count
    state, _ = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == posts
    assert state["resource2_rollout"]["refill_mode"] == rollout.REFILL_MODE_RESOURCE4


def test_queued_and_running_responses_are_not_terminal_failures(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)

    class ActiveResponseScheduler(_Scheduler):
        def submit_task(self, payload: dict):
            task = super().submit_task(payload)
            status = "running" if self.post_count % 2 else "queued"
            self.by_id[task["id"]]["status"] = status
            self.by_dedupe[task["dedupe_key"]]["status"] = status
            task["status"] = status
            return task

    scheduler = ActiveResponseScheduler()
    state, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == 74
    assert result["refill_mode"] == rollout.REFILL_MODE_RESOURCE2
    assert result["active_count"] == 574
    assert not state["resource2_rollout"]["resource2_terminal_failure_task_ids"]


def test_crash_after_post_before_persist_reconciles_dedupe_once(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    durable = []
    persist_calls = 0

    def crash_before_second_persist(value):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise RuntimeError("simulated crash after POST before state persist")
        durable.append(copy.deepcopy(dict(value)))

    with pytest.raises(RuntimeError, match="simulated crash"):
        rollout.control_once(
            plan=plan,
            state=state,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            scheduler=scheduler,
            apply=True,
            persist=crash_before_second_persist,
        )
    assert scheduler.post_count == 1
    assert len(durable) == 1
    assert durable[0]["entries"][-1]["state"] == "planned"
    assert durable[0]["entries"][-1]["task_id"] is None

    state, result = rollout.control_once(
        plan=plan,
        state=durable[0],
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == 74
    assert len(scheduler.by_dedupe) == 74
    assert result["active_count"] == 574
    assert not any(entry["state"] == "planned" for entry in state["entries"])


def test_detected_failure_releases_unposted_durable_reservation(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    state, _ = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    completed = next(task for task in scheduler.by_id.values() if task["cpus"] == 4)
    scheduler.by_id[completed["id"]]["status"] = "completed"
    scheduler.by_dedupe[completed["dedupe_key"]]["status"] = "completed"
    durable = None
    persist_calls = 0

    def crash_after_reservation(value):
        nonlocal durable, persist_calls
        persist_calls += 1
        if persist_calls == 2:
            durable = copy.deepcopy(dict(value))
            raise RuntimeError("simulated crash after durable reservation")

    with pytest.raises(RuntimeError, match="durable reservation"):
        rollout.control_once(
            plan=plan,
            state=state,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            scheduler=scheduler,
            apply=True,
            persist=crash_after_reservation,
        )
    assert durable is not None
    planned = durable["entries"][-1]
    assert planned["state"] == "planned"
    assert scheduler.find_task_by_dedupe(planned["dedupe_key"]) is None
    planned_seed = planned["seed"]
    planned_stage = planned["stage_id"]

    failed = next(task for task in scheduler.by_id.values() if task["cpus"] == 2)
    scheduler.by_id[failed["id"]]["status"] = "failed"
    scheduler.by_dedupe[failed["dedupe_key"]]["status"] = "failed"
    posts = scheduler.post_count
    state, result = rollout.control_once(
        plan=plan,
        state=durable,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == posts
    assert result["actions"][-1]["pending_released_without_post"] == 1
    assert state["next_seed_by_stage"][planned_stage] == planned_seed
    assert all(entry["dedupe_key"] != planned["dedupe_key"] for entry in state["entries"])
    assert state["resource2_rollout"]["refill_mode"] == rollout.REFILL_MODE_RESOURCE4


def test_manual_fallback_releases_pending_resource2_without_post(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    durable = None

    def crash_after_first_reservation(value):
        nonlocal durable
        durable = copy.deepcopy(dict(value))
        raise RuntimeError("simulated stop after reservation")

    with pytest.raises(RuntimeError, match="after reservation"):
        rollout.control_once(
            plan=plan,
            state=state,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            scheduler=scheduler,
            apply=True,
            persist=crash_after_first_reservation,
        )
    assert durable is not None
    planned = durable["entries"][-1]
    assert planned["state"] == "planned"
    assert planned["resource_policy_id"] == migration.RESOURCE2_RESOURCE_POLICY_ID
    state = rollout.activate_manual_fallback(
        durable,
        plan=plan,
        config=config,
        terminal_gate=gate,
        expected_state_sha256=durable["state_sha256"],
    )
    state, result = rollout.control_once(
        plan=plan,
        state=state,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == 0
    assert result["refill_mode"] == rollout.REFILL_MODE_RESOURCE4
    assert state["next_seed_by_stage"][planned["stage_id"]] == planned["seed"]
    assert all(entry["dedupe_key"] != planned["dedupe_key"] for entry in state["entries"])


def test_stale_second_controller_reconciles_all_dedupes_without_duplicate_posts(
    monkeypatch,
):
    plan, config, gate, _predecessor, initial = _prepared(monkeypatch)
    scheduler = _Scheduler()
    first, _ = rollout.control_once(
        plan=plan,
        state=initial,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == 74
    second, result = rollout.control_once(
        plan=plan,
        state=initial,
        config=config,
        terminal_gate=gate,
        allocations=_allocation_rows(),
        scheduler=scheduler,
        apply=True,
        persist=lambda _value: None,
    )
    assert scheduler.post_count == 74
    assert result["active_count"] == 574
    assert rollout._entry_identities(first["entries"]) == rollout._entry_identities(
        second["entries"]
    )


def test_recovered_dedupe_with_changed_task_identity_fails_closed(monkeypatch):
    plan, config, gate, _predecessor, state = _prepared(monkeypatch)
    scheduler = _Scheduler()
    durable = None

    def crash_after_first_reservation(value):
        nonlocal durable
        durable = copy.deepcopy(dict(value))
        raise RuntimeError("simulated stop after reservation")

    with pytest.raises(RuntimeError, match="after reservation"):
        rollout.control_once(
            plan=plan,
            state=state,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            scheduler=scheduler,
            apply=True,
            persist=crash_after_first_reservation,
        )
    assert durable is not None
    entry = durable["entries"][-1]
    task = rollout._task_for_rollout_entry(entry, templates=controller._task_templates(plan))
    task.update(
        {
            "id": 300_001,
            "task_id": 300_001,
            "status": "queued",
            "cpus": 4,
        }
    )
    scheduler.by_id[300_001] = task
    scheduler.by_dedupe[entry["dedupe_key"]] = task
    with pytest.raises(RuntimeError, match="changed capacity refill task identity"):
        rollout.control_once(
            plan=plan,
            state=durable,
            config=config,
            terminal_gate=gate,
            allocations=_allocation_rows(),
            scheduler=scheduler,
            apply=True,
            persist=lambda _value: None,
        )
    assert scheduler.post_count == 0
