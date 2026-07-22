"""Pure controller-v2 ledger operations for finite final1000 seed lanes.

The module contains no HTTP client and no apply/watch entry point.  It is the
reviewable state transition layer used before a separately authorized rollout:
one physical Scheduler entry can own an exact ordered finite child block, while
active quotas always count physical lanes (500 total, 200/160/90/50).
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_final1000_multiseed_contract import (
        MAX_BATCH_LENGTH,
        PHASE_A_OPERATIONAL_BATCH_LENGTHS,
        PROTOCOL_VERSION,
        batch_manifest_from_payload,
        build_batch_task,
        validate_batch_task,
    )
    from tier1_final1000_slurm_controller import (
        SUCCESSOR_RESOURCE_POLICY_ID,
        _refill_task,
        _task_for_entry,
        _task_index,
        _task_templates,
        _validate_state as validate_v1_state,
    )
    from tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        SUCCESSOR_ACTIVE_QUOTAS,
        validate_launch_plan,
    )
    from tier1_final1000_rolling_migration import (
        RESOURCE_POLICIES,
        _render_from_template_for_policy,
        _validate_task_for_policy,
        validate_historical_resource_quota_successor_plan,
        validate_predecessor_plan,
    )
    from tier1_final1000_stage_profiles import BY_ID, STAGES, TOTAL_ACTIVE_QUOTA
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_final1000_multiseed_contract import (
        MAX_BATCH_LENGTH,
        PHASE_A_OPERATIONAL_BATCH_LENGTHS,
        PROTOCOL_VERSION,
        batch_manifest_from_payload,
        build_batch_task,
        validate_batch_task,
    )
    from tools.tier1_final1000_slurm_controller import (
        SUCCESSOR_RESOURCE_POLICY_ID,
        _refill_task,
        _task_for_entry,
        _task_index,
        _task_templates,
        _validate_state as validate_v1_state,
    )
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        SUCCESSOR_ACTIVE_QUOTAS,
        validate_launch_plan,
    )
    from tools.tier1_final1000_rolling_migration import (
        RESOURCE_POLICIES,
        _render_from_template_for_policy,
        _validate_task_for_policy,
        validate_historical_resource_quota_successor_plan,
        validate_predecessor_plan,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, STAGES, TOTAL_ACTIVE_QUOTA


CONTROLLER_STATE_SCHEMA = "mft-tier1-final1000-multiseed-controller-state-v2"
SINGLE_SEED_PROTOCOL = "final1000-single-seed-v1"
REFILL_POLICY = "physical-lane-stage-deficit-v2"
ACTIVE_STATES = frozenset({"planned", "submitted", "queued", "running"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})


def scheduler_state(value: Any) -> str | None:
    """Map every Scheduler terminal, including both timeout spellings."""

    return {
        "queued": "queued",
        "attaching": "queued",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "timeout": "timeout",
        "timed_out": "timeout",
    }.get(str(value or "").lower())


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        key: copy.deepcopy(item)
        for key, item in unsigned.items()
        if key != "state_sha256"
    }
    return {**value, "state_sha256": canonical_sha256(value)}


def _advance(state: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in state.items() if key != "state_sha256"
    }
    unsigned.update(copy.deepcopy(changes))
    unsigned["parent_state_sha256"] = state["state_sha256"]
    unsigned["revision"] = int(state["revision"]) + 1
    return _seal(unsigned)


def _v1_entry(
    value: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    origin: str,
    source_launch_plan_sha256: str,
    source_resource_policy_id: str,
) -> dict[str, Any]:
    seed = int(value["seed"])
    task_envelope = copy.deepcopy(dict(task))
    return {
        "protocol_version": SINGLE_SEED_PROTOCOL,
        "stage_id": str(value["stage_id"]),
        "bundle_id": str(value["bundle_id"]),
        "wave": str(value["wave"]),
        "seeds": [seed],
        "batch_length": 1,
        "parent_dedupe_key": str(value["dedupe_key"]),
        "batch_manifest_sha256": None,
        "task_id": value.get("task_id"),
        "state": str(value["state"]),
        "origin": origin,
        "source_launch_plan_sha256": source_launch_plan_sha256,
        "source_resource_policy_id": source_resource_policy_id,
        "task_envelope": task_envelope,
        "task_envelope_sha256": canonical_sha256(task_envelope),
    }


def _cohort_task_for_entry(
    entry: Mapping[str, Any], plan: Mapping[str, Any], *, policy_id: str
) -> dict[str, Any]:
    policy = RESOURCE_POLICIES.get(policy_id)
    if policy is None:
        raise RuntimeError("controller-v2 predecessor resource policy is unknown")
    candidates = [
        task
        for task in [
            *plan["task_waves"]["canaries"],
            *plan["task_waves"]["ramp"],
        ]
        if task["dedupe_key"] == entry["dedupe_key"]
    ]
    if len(candidates) > 1:
        raise RuntimeError("controller-v2 predecessor task identity is ambiguous")
    if candidates:
        task = _validate_task_for_policy(
            candidates[0], policy=policy, expected_wave=str(entry["wave"])
        )
    else:
        templates = {
            task["payload_json"]["final_goal_stage_id"]: task
            for task in plan["task_waves"]["canaries"]
        }
        if set(templates) != set(BY_ID):
            raise RuntimeError("controller-v2 predecessor templates are incomplete")
        task = _render_from_template_for_policy(
            templates[str(entry["stage_id"])],
            stage_id=str(entry["stage_id"]),
            seed=int(entry["seed"]),
            wave=str(entry["wave"]),
            policy=policy,
        )
    if (
        task["dedupe_key"] != entry["dedupe_key"]
        or task["payload_json"]["bundle_id"] != entry["bundle_id"]
    ):
        raise RuntimeError("controller-v2 predecessor task/cohort identity mismatch")
    return copy.deepcopy(task)


def _stored_v1_task(entry: Mapping[str, Any]) -> dict[str, Any]:
    task = entry.get("task_envelope")
    policy_id = str(entry.get("source_resource_policy_id") or "")
    source_plan_sha = str(entry.get("source_launch_plan_sha256") or "")
    if (
        not isinstance(task, dict)
        or set(task) != REQUIRED_SCHEDULER_FIELDS
        or canonical_sha256(task) != entry.get("task_envelope_sha256")
        or len(source_plan_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_plan_sha)
        or policy_id not in RESOURCE_POLICIES
    ):
        raise RuntimeError("controller-v2 stored v1 task binding is invalid")
    task = _validate_task_for_policy(
        task,
        policy=RESOURCE_POLICIES[policy_id],
        expected_wave=str(entry["wave"]),
    )
    payload = task["payload_json"]
    if (
        task["dedupe_key"] != entry["parent_dedupe_key"]
        or payload["final_goal_stage_id"] != entry["stage_id"]
        or payload["bundle_id"] != entry["bundle_id"]
        or payload["seed"] != entry["seeds"][0]
    ):
        raise RuntimeError("controller-v2 stored v1 task identity drifted")
    return copy.deepcopy(task)


def upgrade_v1_state(
    v1_state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    predecessor_plan: Mapping[str, Any] | None = None,
    batch_length: int = 4,
) -> dict[str, Any]:
    """Create a mixed v1/v2 ledger without changing or cancelling a v1 lane."""

    plan = validate_launch_plan(plan)
    v1_state = validate_v1_state(v1_state, plan)
    if int(batch_length) not in PHASE_A_OPERATIONAL_BATCH_LENGTHS:
        raise RuntimeError("controller-v2 operational batch length must be 1 or 4")
    migration = v1_state.get("rolling_migration")
    validated_predecessor_plan: dict[str, Any] | None = None
    if isinstance(migration, dict) and any(
        entry.get("origin") == "predecessor" for entry in v1_state["entries"]
    ):
        if predecessor_plan is None:
            raise RuntimeError(
                "controller-v2 rolling predecessor plan is required for upgrade"
            )
        if migration.get("predecessor_controller_kind") == "legacy_8c":
            validated_predecessor_plan = validate_predecessor_plan(predecessor_plan)
        elif migration.get("predecessor_controller_kind") == "resource_quota_successor":
            validated_predecessor_plan = (
                validate_historical_resource_quota_successor_plan(predecessor_plan)
            )
        else:  # pragma: no cover - v1 validation guards this
            raise RuntimeError("controller-v2 predecessor controller kind is unknown")
        if validated_predecessor_plan["launch_plan_sha256"] != migration.get(
            "predecessor_launch_plan_sha256"
        ):
            raise RuntimeError("controller-v2 predecessor launch identity mismatch")
    entries = []
    successor_index = _task_index(plan)
    successor_templates = _task_templates(plan)
    for entry in v1_state["entries"]:
        origin = str(entry.get("origin") or "successor")
        policy_id = str(
            entry.get("resource_policy_id") or SUCCESSOR_RESOURCE_POLICY_ID
        )
        if origin == "predecessor" and isinstance(migration, dict):
            if validated_predecessor_plan is None:  # pragma: no cover - guarded above
                raise RuntimeError("controller-v2 predecessor plan was not validated")
            task = _cohort_task_for_entry(
                entry, validated_predecessor_plan, policy_id=policy_id
            )
            source_plan_sha = validated_predecessor_plan["launch_plan_sha256"]
        else:
            task = _task_for_entry(
                entry,
                plan_index=successor_index,
                templates=successor_templates,
            )
            source_plan_sha = plan["launch_plan_sha256"]
        entries.append(
            _v1_entry(
                entry,
                task,
                origin=origin,
                source_launch_plan_sha256=source_plan_sha,
                source_resource_policy_id=policy_id,
            )
        )
    unsigned = {
        "schema_version": CONTROLLER_STATE_SCHEMA,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "revision": 0,
        "parent_state_sha256": None,
        "stop_requested": bool(v1_state.get("stop_requested")),
        "batch_length": int(batch_length),
        "physical_active_target": TOTAL_ACTIVE_QUOTA,
        "stage_physical_active_quotas": copy.deepcopy(SUCCESSOR_ACTIVE_QUOTAS),
        "entries": entries,
        "next_seed_by_stage": {
            stage_id: int(seed)
            for stage_id, seed in v1_state["next_seed_by_stage"].items()
        },
        "refill_policy": REFILL_POLICY,
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return validate_state(_seal(unsigned), plan)


def validate_state(value: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    plan = validate_launch_plan(plan)
    unsigned = {key: item for key, item in value.items() if key != "state_sha256"}
    entries = value.get("entries")
    next_seeds = value.get("next_seed_by_stage")
    if (
        value.get("schema_version") != CONTROLLER_STATE_SCHEMA
        or value.get("launch_plan_sha256") != plan["launch_plan_sha256"]
        or value.get("state_sha256") != canonical_sha256(unsigned)
        or not isinstance(entries, list)
        or not isinstance(next_seeds, dict)
        or set(next_seeds) != set(BY_ID)
        or value.get("physical_active_target") != TOTAL_ACTIVE_QUOTA
        or value.get("stage_physical_active_quotas") != SUCCESSOR_ACTIVE_QUOTAS
        or value.get("refill_policy") != REFILL_POLICY
        or value.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or value.get("cancellation_performed") is not False
        or value.get("preemption_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or int(value.get("batch_length", 0)) not in PHASE_A_OPERATIONAL_BATCH_LENGTHS
    ):
        raise RuntimeError("controller-v2 state identity/SHA mismatch")
    task_ids: set[int] = set()
    dedupes: set[str] = set()
    identities: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("controller-v2 entry must be an object")
        stage_id = str(entry.get("stage_id") or "")
        protocol = str(entry.get("protocol_version") or "")
        seeds = entry.get("seeds")
        state = str(entry.get("state") or "")
        task_id = entry.get("task_id")
        stage = BY_ID.get(stage_id)
        batch_length = entry.get("batch_length")
        if (
            set(entry)
            != {
                "protocol_version",
                "stage_id",
                "bundle_id",
                "wave",
                "seeds",
                "batch_length",
                "parent_dedupe_key",
                "batch_manifest_sha256",
                "task_id",
                "state",
                "origin",
                "source_launch_plan_sha256",
                "source_resource_policy_id",
                "task_envelope",
                "task_envelope_sha256",
            }
            or stage is None
            or protocol not in {SINGLE_SEED_PROTOCOL, PROTOCOL_VERSION}
            or not str(entry.get("bundle_id") or "")
            or entry.get("wave") not in {"canary", "ramp", "refill"}
            or not isinstance(seeds, list)
            or not seeds
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
            )
            or isinstance(batch_length, bool)
            or not isinstance(batch_length, int)
            or len(seeds) != batch_length
            or len(seeds) > MAX_BATCH_LENGTH
            or seeds != list(range(seeds[0], seeds[0] + len(seeds)))
            or any(
                not stage.seed_start <= int(seed) < stage.seed_window_end_exclusive
                for seed in seeds
            )
            or (
                protocol == SINGLE_SEED_PROTOCOL
                and (batch_length != 1 or len(seeds) != 1)
            )
            or (
                protocol == PROTOCOL_VERSION
                and batch_length not in PHASE_A_OPERATIONAL_BATCH_LENGTHS
            )
            or state not in ACTIVE_STATES | TERMINAL_STATES
            or not str(entry.get("parent_dedupe_key") or "").startswith(
                "mft-tier1-final1000:"
            )
            or (protocol == PROTOCOL_VERSION)
            != isinstance(entry.get("batch_manifest_sha256"), str)
            or (
                protocol == PROTOCOL_VERSION
                and (
                    len(str(entry.get("batch_manifest_sha256") or "")) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in str(entry["batch_manifest_sha256"])
                    )
                    or not str(entry["parent_dedupe_key"]).startswith(
                        "mft-tier1-final1000:lane:"
                    )
                    or entry.get("origin") != "successor"
                )
            )
            or (
                protocol == SINGLE_SEED_PROTOCOL
                and entry.get("origin") not in {"predecessor", "successor"}
            )
            or (
                protocol == SINGLE_SEED_PROTOCOL
                and (
                    not isinstance(entry.get("task_envelope"), dict)
                    or not isinstance(entry.get("task_envelope_sha256"), str)
                    or not isinstance(entry.get("source_launch_plan_sha256"), str)
                    or not isinstance(entry.get("source_resource_policy_id"), str)
                )
            )
            or (
                protocol == PROTOCOL_VERSION
                and any(
                    entry.get(field) is not None
                    for field in (
                        "source_launch_plan_sha256",
                        "source_resource_policy_id",
                        "task_envelope",
                        "task_envelope_sha256",
                    )
                )
            )
            or (state == "planned") != (task_id is None)
        ):
            raise RuntimeError("controller-v2 physical lane entry drifted")
        if protocol == SINGLE_SEED_PROTOCOL:
            _stored_v1_task(entry)
        dedupe = str(entry["parent_dedupe_key"])
        if dedupe in dedupes:
            raise RuntimeError("controller-v2 duplicate parent dedupe")
        dedupes.add(dedupe)
        if task_id is not None:
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
            ):
                raise RuntimeError("controller-v2 task id is invalid")
            if task_id in task_ids:
                raise RuntimeError("one Scheduler task id represents multiple lanes")
            task_ids.add(task_id)
        for seed in seeds:
            identity = (str(entry["bundle_id"]), int(seed))
            if identity in identities:
                raise RuntimeError("controller-v2 duplicate child scientific identity")
            identities.add(identity)
    for stage in STAGES:
        stage_entries = [
            entry for entry in entries if entry["stage_id"] == stage.stage_id
        ]
        maximum = max(
            (seed for entry in stage_entries for seed in entry["seeds"]),
            default=stage.seed_start - 1,
        )
        if int(next_seeds[stage.stage_id]) <= maximum:
            raise RuntimeError("controller-v2 next seed was not atomically advanced")
        if not (
            stage.seed_start
            <= int(next_seeds[stage.stage_id])
            <= stage.seed_window_end_exclusive
        ):
            raise RuntimeError("controller-v2 next seed escaped its stage window")
    return copy.deepcopy(dict(value))


def active_physical_count(state: Mapping[str, Any], stage_id: str | None = None) -> int:
    return sum(
        entry["state"] in ACTIVE_STATES
        and (stage_id is None or entry["stage_id"] == stage_id)
        for entry in state["entries"]
    )


def logical_seed_counts(state: Mapping[str, Any]) -> dict[str, int]:
    running = sum(1 for entry in state["entries"] if entry["state"] == "running")
    backlog = sum(
        max(0, len(entry["seeds"]) - (1 if entry["state"] == "running" else 0))
        for entry in state["entries"]
        if entry["state"] in ACTIVE_STATES
        and entry["protocol_version"] == PROTOCOL_VERSION
    )
    return {
        "physical_active_lanes": active_physical_count(state),
        "logical_current_seeds": running,
        "logical_future_seed_backlog": backlog,
    }


def _batch_task_for(
    plan: Mapping[str, Any],
    *,
    stage_id: str,
    seeds: Sequence[int],
    wave: str,
) -> dict[str, Any]:
    templates = _task_templates(plan)
    children = [
        _refill_task(
            templates[stage_id],
            stage_id=stage_id,
            seed=int(seed),
            wave=wave,
        )
        for seed in seeds
    ]
    return build_batch_task(children)


def reserve_lane(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    stage_id: str,
    wave: str = "refill",
    batch_length: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Atomically reserve one exact child block and return its parent task."""

    state = validate_state(state, plan)
    if state["stop_requested"]:
        raise RuntimeError("controller-v2 stop latch prohibits a new lane")
    if stage_id not in BY_ID:
        raise RuntimeError("controller-v2 stage is unknown")
    length = int(state["batch_length"] if batch_length is None else batch_length)
    if length not in PHASE_A_OPERATIONAL_BATCH_LENGTHS:
        raise RuntimeError("controller-v2 operational batch length must be 1 or 4")
    if active_physical_count(state) >= TOTAL_ACTIVE_QUOTA:
        raise RuntimeError("controller-v2 has no physical lane gap")
    if active_physical_count(state, stage_id) >= SUCCESSOR_ACTIVE_QUOTAS[stage_id]:
        raise RuntimeError("controller-v2 stage has no physical lane gap")
    start = int(state["next_seed_by_stage"][stage_id])
    seeds = list(range(start, start + length))
    stage = BY_ID[stage_id]
    if seeds[-1] >= stage.seed_window_end_exclusive:
        raise RuntimeError(
            "controller-v2 exact child block escapes the stage seed window"
        )
    task = _batch_task_for(plan, stage_id=stage_id, seeds=seeds, wave=wave)
    payload = task["payload_json"]
    manifest = batch_manifest_from_payload(payload)
    entry = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": stage_id,
        "bundle_id": payload["bundle_id"],
        "wave": wave,
        "seeds": seeds,
        "batch_length": length,
        "parent_dedupe_key": task["dedupe_key"],
        "batch_manifest_sha256": manifest["manifest_sha256"],
        "task_id": None,
        "state": "planned",
        "origin": "successor",
        "source_launch_plan_sha256": None,
        "source_resource_policy_id": None,
        "task_envelope": None,
        "task_envelope_sha256": None,
    }
    unsigned = {
        key: copy.deepcopy(item) for key, item in state.items() if key != "state_sha256"
    }
    unsigned["entries"].append(entry)
    unsigned["next_seed_by_stage"][stage_id] = start + length
    next_state = _advance(
        state,
        entries=unsigned["entries"],
        next_seed_by_stage=unsigned["next_seed_by_stage"],
    )
    validate_state(next_state, plan)
    return next_state, task


def reproduce_lane_task(
    entry: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if entry.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("only v2 lane tasks can be reproduced by this adapter")
    task = _batch_task_for(
        validate_launch_plan(plan),
        stage_id=str(entry["stage_id"]),
        seeds=[int(seed) for seed in entry["seeds"]],
        wave=str(entry["wave"]),
    )
    validate_batch_task(task)
    manifest = batch_manifest_from_payload(task["payload_json"])
    if (
        task["dedupe_key"] != entry["parent_dedupe_key"]
        or manifest["manifest_sha256"] != entry["batch_manifest_sha256"]
    ):
        raise RuntimeError("controller-v2 cannot reproduce its parent lane identity")
    return task


def _expected_task_for_entry(
    entry: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    if entry.get("protocol_version") == PROTOCOL_VERSION:
        return reproduce_lane_task(entry, plan)
    if entry.get("protocol_version") != SINGLE_SEED_PROTOCOL:
        raise RuntimeError("controller-v2 entry protocol cannot be reproduced")
    # A rolling predecessor can have a different immutable bundle and resource
    # policy from the successor plan.  Upgrade therefore stores the exact task
    # envelope authenticated against its source cohort; observation never tries
    # to recreate it from the successor template.
    return _stored_v1_task(entry)


def observe_scheduler_task(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    parent_dedupe_key: str,
    scheduler_task: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one read-only observation, including timeout as terminal."""

    state = validate_state(state, plan)
    entries = copy.deepcopy(state["entries"])
    matches = [
        entry for entry in entries if entry["parent_dedupe_key"] == parent_dedupe_key
    ]
    if len(matches) != 1:
        raise RuntimeError("controller-v2 scheduler observation has no unique lane")
    entry = matches[0]
    if scheduler_task.get("dedupe_key") != parent_dedupe_key:
        raise RuntimeError("Scheduler changed controller-v2 parent dedupe identity")
    mapped = scheduler_state(scheduler_task.get("status"))
    if mapped is None:
        raise RuntimeError("controller-v2 observed an unknown Scheduler state")
    task_id = scheduler_task.get("id") or scheduler_task.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise RuntimeError("controller-v2 Scheduler observation lacks a task id")
    expected_task = _expected_task_for_entry(entry, plan)
    observed_envelope = scheduler_task.get("task_json")
    if (
        not isinstance(observed_envelope, dict)
        or set(observed_envelope) != REQUIRED_SCHEDULER_FIELDS
        or dict(observed_envelope) != expected_task
        or scheduler_task.get("name") != expected_task["name"]
    ):
        raise RuntimeError("controller-v2 Scheduler envelope identity changed")
    if entry.get("task_id") not in (None, int(task_id)):
        raise RuntimeError("controller-v2 Scheduler task identity changed")
    if entry["state"] in TERMINAL_STATES and entry["state"] != mapped:
        raise RuntimeError("controller-v2 terminal lane state cannot change")
    entry["task_id"] = int(task_id)
    entry["state"] = mapped
    return validate_state(_advance(state, entries=entries), plan)


def request_stop(state: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    state = validate_state(state, plan)
    if state["stop_requested"]:
        return state
    return validate_state(_advance(state, stop_requested=True), plan)


def deficit_stage_order(state: Mapping[str, Any]) -> list[str]:
    """Return one stage per physical gap; child backlog never counts as active."""

    current = {
        stage.stage_id: active_physical_count(state, stage.stage_id) for stage in STAGES
    }
    order: list[str] = []
    while sum(current.values()) < TOTAL_ACTIVE_QUOTA:
        eligible = [
            stage.stage_id
            for stage in STAGES
            if current[stage.stage_id] < SUCCESSOR_ACTIVE_QUOTAS[stage.stage_id]
        ]
        if not eligible:
            raise RuntimeError(
                "controller-v2 physical target/quota identity is inconsistent"
            )
        # Fill the largest normalized physical deficit first, with stable stage order.
        stage_id = max(
            eligible,
            key=lambda item: (
                (SUCCESSOR_ACTIVE_QUOTAS[item] - current[item])
                / SUCCESSOR_ACTIVE_QUOTAS[item],
                -list(BY_ID).index(item),
            ),
        )
        order.append(stage_id)
        current[stage_id] += 1
    return order
