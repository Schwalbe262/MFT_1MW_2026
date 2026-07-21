"""Prepare a fail-closed, cancellation-free final1000 successor handoff.

The predecessor controller must first be stopped through its normal sealed
stop flag.  This module never stops a controller and never cancels, preempts,
or edits a scheduler task.  Preparation performs GET/read-only checks and can
atomically create a successor controller state.  The only scheduler mutation
available after handoff remains ``POST /api/tasks`` in the regular controller.

Every predecessor ledger entry (terminal and active) and every next-seed
cursor is imported.  The migration-aware controller consequently submits a
successor task only after an imported active task naturally becomes terminal.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_controller import TransportReadyProbe
    from tier1_corrected_current7_slurm_publish import scheduler_publication_transport
    from tier1_final1000_slurm_controller import (
        ACTIVE_STATES,
        DEDUPE_PREFIX,
        HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
        LEGACY_RESOURCE_POLICY_ID,
        REFILL_POLICY,
        STATE_SCHEMA,
        TASK_NAME_PREFIX,
        TERMINAL_STATES,
        SchedulerApiClient,
        SUCCESSOR_RESOURCE_POLICY_ID,
        _scheduler_state,
        _validate_historical_resource_quota_successor_state,
        _write_state,
    )
    from tier1_final1000_slurm_launch import (
        DEFAULT_MEMORY_MB,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        LAUNCH_SCHEMA,
        REQUIRED_SCHEDULER_FIELDS,
        SUCCESSOR_ACTIVE_QUOTAS,
        optimizer_stage_contract,
        validate_launch_plan,
    )
    from tier1_final1000_stage_profiles import (
        BY_ID,
        FIXED_GENERATIONS,
        FIXED_PRIMARY_TURNS,
        INFERENCE_THREADS,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        stage_inventory,
        stage_profile,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_controller import TransportReadyProbe
    from tools.tier1_corrected_current7_slurm_publish import (
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_slurm_controller import (
        ACTIVE_STATES,
        DEDUPE_PREFIX,
        HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
        LEGACY_RESOURCE_POLICY_ID,
        REFILL_POLICY,
        STATE_SCHEMA,
        TASK_NAME_PREFIX,
        TERMINAL_STATES,
        SchedulerApiClient,
        SUCCESSOR_RESOURCE_POLICY_ID,
        _scheduler_state,
        _validate_historical_resource_quota_successor_state,
        _write_state,
    )
    from tools.tier1_final1000_slurm_launch import (
        DEFAULT_MEMORY_MB,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        LAUNCH_SCHEMA,
        REQUIRED_SCHEDULER_FIELDS,
        SUCCESSOR_ACTIVE_QUOTAS,
        optimizer_stage_contract,
        validate_launch_plan,
    )
    from tools.tier1_final1000_stage_profiles import (
        BY_ID,
        FIXED_GENERATIONS,
        FIXED_PRIMARY_TURNS,
        INFERENCE_THREADS,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        stage_inventory,
        stage_profile,
    )


MIGRATION_SCHEMA = "mft-tier1-final1000-rolling-migration-v1"
HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v1"
RESOURCE_QUOTA_ONLY = "resource_quota_only"
PATCHED_BUNDLE = "patched_bundle"
PREDECESSOR_CPUS = 8
PREDECESSOR_MAX_WORKERS = 8
SUCCESSOR_CPUS = 4
SUCCESSOR_MAX_WORKERS = 32

PREDECESSOR_POLICY = {
    "cpus_per_task": PREDECESSOR_CPUS,
    "memory_mb_per_task": DEFAULT_MEMORY_MB,
    "max_workers_per_node": PREDECESSOR_MAX_WORKERS,
    "priority": DEFAULT_PRIORITY,
    "scheduling_profile": "standard",
    "gpus": 0,
    "inference_threads": INFERENCE_THREADS,
}
SUCCESSOR_POLICY = {
    "cpus_per_task": SUCCESSOR_CPUS,
    "memory_mb_per_task": DEFAULT_MEMORY_MB,
    "max_workers_per_node": SUCCESSOR_MAX_WORKERS,
    "priority": DEFAULT_PRIORITY,
    "scheduling_profile": "standard",
    "gpus": 0,
    "inference_threads": INFERENCE_THREADS,
}

RESOURCE_POLICIES = {
    LEGACY_RESOURCE_POLICY_ID: PREDECESSOR_POLICY,
    SUCCESSOR_RESOURCE_POLICY_ID: SUCCESSOR_POLICY,
}

PAYLOAD_SHA_PATTERN = re.compile(
    r"(--payload-sha256\s+)([0-9a-f]{64})(?=\s*$)", re.MULTILINE
)


class ReadyProbe(Protocol):
    read_count: int

    def read_ready(self, remote_bundle: str) -> Mapping[str, Any] | None: ...


class InventoryClient(Protocol):
    def list_namespace_tasks(self) -> Sequence[Mapping[str, Any]]: ...


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _seal_nested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "sha256"
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "state_sha256"
    }
    migration = unsigned.get("rolling_migration")
    if isinstance(migration, dict):
        unsigned["rolling_migration"] = _seal_nested(migration)
    return {**unsigned, "state_sha256": canonical_sha256(unsigned)}


def _plan_tasks(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    waves = plan.get("task_waves") or {}
    return [
        *[dict(task) for task in waves.get("canaries") or []],
        *[dict(task) for task in waves.get("ramp") or []],
    ]


def harvest_cohort_identity(
    *,
    role: str,
    plan: Mapping[str, Any],
    resource_policy_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the bounded identity needed to replay one ledger cohort.

    The controller ledger keeps only a bundle id and resource-policy id per
    task.  Full launch plans and manifests remain immutable external inputs;
    this two-role seal proves which of those inputs a harvester may use without
    copying an ever-growing task history into migration metadata.
    """

    if role not in {"predecessor", "successor"}:
        raise ValueError("unknown final1000 harvest cohort role")
    policies = sorted({str(value) for value in resource_policy_ids})
    if not policies or any(value not in RESOURCE_POLICIES for value in policies):
        raise RuntimeError("harvest cohort resource policy identity is invalid")
    bindings = plan.get("stage_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(BY_ID):
        raise RuntimeError("harvest cohort stage binding set is incomplete")
    stage_bindings: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        binding = bindings.get(stage.stage_id)
        if not isinstance(binding, Mapping):
            raise RuntimeError("harvest cohort stage binding is invalid")
        stage_bindings[stage.stage_id] = {
            key: copy.deepcopy(binding.get(key))
            for key in (
                "bundle_id",
                "bundle_manifest_sha256",
                "remote_bundle",
                "publication_receipt_sha256",
                "ready_sha256",
                "stage_spec_sha256",
            )
        }
    unsigned = {
        "schema_version": HARVEST_COHORT_SCHEMA,
        "role": role,
        "launch_plan_sha256": plan.get("launch_plan_sha256"),
        "resource_policy_ids": policies,
        "stage_bindings": stage_bindings,
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _task_resources(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cpus": int(policy["cpus_per_task"]),
        "memory_mb": int(policy["memory_mb_per_task"]),
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": int(policy["priority"]),
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "max_workers_per_node": int(policy["max_workers_per_node"]),
    }


def _validate_task_for_policy(
    task: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    expected_wave: str | None = None,
) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError("final1000 migration task fields drifted")
    payload = task.get("payload_json") or {}
    stage = BY_ID.get(str(payload.get("final_goal_stage_id") or ""))
    if stage is None:
        raise RuntimeError("final1000 migration task stage is unknown")
    spec, constraints = optimizer_stage_contract(stage)
    lane = payload.get("lane") or {}
    seed = int(payload.get("seed", -1))
    wave = str(lane.get("wave") or "")
    resources = _task_resources(policy)
    dedupe = canonical_sha256(
        {
            "goal": "final1000",
            "stage_profile_sha256": stage_profile(stage)["sha256"],
            "bundle_id": payload.get("bundle_id"),
            "payload": payload,
            "resources": resources,
        }
    )
    if (
        (expected_wave is not None and wave != expected_wave)
        or wave not in {"canary", "ramp", "refill"}
        or lane.get("seed") != seed
        or lane.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or not stage.seed_start <= seed < stage.seed_window_end_exclusive
        or payload.get("hard_spec") != spec
        or payload.get("hard_spec_sha256") != canonical_sha256(spec)
        or payload.get("stage_spec_sha256") != canonical_sha256(spec)
        or payload.get("constraint_names") != constraints
        or payload.get("final_goal_stage_profile_sha256")
        != stage_profile(stage)["sha256"]
        or payload.get("max_generations") != FIXED_GENERATIONS
        or payload.get("inference_threads") != INFERENCE_THREADS
        or payload.get("optimizer_processes") != 1
        or payload.get("maximum_peak_rss_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or any(
            payload.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
        or any(task.get(key) != value for key, value in resources.items())
        or task.get("required_capability") != "conda:pyaedt2026v1"
        or task.get("env_profile") != "pyaedt2026v1"
        or not str(task.get("remote_cwd") or "").startswith("/")
        or task.get("name") != f"{stage.task_name_stem}-{wave}-{seed}"
        or task.get("dedupe_key") != f"{DEDUPE_PREFIX}{dedupe}"
        or f"--payload-sha256 {canonical_sha256(payload)}"
        not in str(task.get("command") or "")
        or any(
            token in str(task.get("command") or "").lower()
            for token in ("ansysedt", "pyaedt.desktop")
        )
    ):
        raise RuntimeError("final1000 migration task policy/science seal mismatch")
    return dict(task)


def _validate_plan_for_policy(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    expected_active_quotas: Mapping[str, int],
) -> dict[str, Any]:
    expected_quotas = dict(expected_active_quotas)
    legacy_quotas = {stage.stage_id: stage.active_quota for stage in STAGES}
    policy_identity = dict(policy)
    if not (
        (policy_identity == PREDECESSOR_POLICY and expected_quotas == legacy_quotas)
        or (
            policy_identity == SUCCESSOR_POLICY
            and expected_quotas
            in (
                HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
                SUCCESSOR_ACTIVE_QUOTAS,
            )
        )
    ):
        raise RuntimeError("unsealed final1000 migration quota policy")
    unsigned = {key: item for key, item in value.items() if key != "launch_plan_sha256"}
    waves = value.get("task_waves") or {}
    canaries = waves.get("canaries") if isinstance(waves, dict) else None
    ramp = waves.get("ramp") if isinstance(waves, dict) else None
    bindings = value.get("stage_bindings")
    expected_resources = {
        key: policy[key]
        for key in (
            "cpus_per_task",
            "memory_mb_per_task",
            "max_workers_per_node",
            "priority",
            "scheduling_profile",
            "gpus",
        )
    }
    if (
        value.get("schema_version") != LAUNCH_SCHEMA
        or value.get("launch_plan_sha256") != canonical_sha256(unsigned)
        or value.get("stage_inventory") != stage_inventory()
        or value.get("resources") != expected_resources
        or (value.get("open_ended_refill") or {}).get("stage_active_quotas")
        != expected_quotas
        or not isinstance(canaries, list)
        or not isinstance(ramp, list)
        or len(canaries) != len(STAGES)
        or len(ramp) != TOTAL_ACTIVE_QUOTA - len(STAGES)
        or not isinstance(bindings, dict)
        or set(bindings) != set(BY_ID)
        or value.get("surrogate_only") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "submission_performed",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("final1000 migration launch-plan seal mismatch")
    for stage in STAGES:
        binding = bindings[stage.stage_id]
        ready = binding.get("ready") if isinstance(binding, dict) else None
        if (
            not isinstance(ready, dict)
            or binding.get("ready_sha256") != canonical_sha256(ready)
            or binding.get("stage_spec_sha256")
            != stage_profile(stage)["stage_spec_sha256"]
            or not str(binding.get("remote_bundle") or "").startswith("/")
            or not str(binding.get("bundle_id") or "")
        ):
            raise RuntimeError("final1000 migration bundle/READY seal mismatch")
    tasks = [*canaries, *ramp]
    for task in canaries:
        _validate_task_for_policy(task, policy=policy, expected_wave="canary")
    for task in ramp:
        _validate_task_for_policy(task, policy=policy, expected_wave="ramp")
    dedupes = [task["dedupe_key"] for task in tasks]
    identities = [
        (
            task["payload_json"]["final_goal_stage_id"],
            int(task["payload_json"]["seed"]),
        )
        for task in tasks
    ]
    if len(set(dedupes)) != TOTAL_ACTIVE_QUOTA or len(set(identities)) != len(tasks):
        raise RuntimeError("final1000 migration plan contains duplicate work")
    stage_counts = {
        stage.stage_id: sum(identity[0] == stage.stage_id for identity in identities)
        for stage in STAGES
    }
    if stage_counts != expected_quotas:
        raise RuntimeError("final1000 migration plan stage quotas drifted")
    for task in tasks:
        payload = task["payload_json"]
        binding = bindings[payload["final_goal_stage_id"]]
        if payload.get("bundle_id") != binding.get("bundle_id") or task.get(
            "remote_cwd"
        ) != binding.get("remote_bundle"):
            raise RuntimeError("final1000 migration task/bundle identity mismatch")
    return copy.deepcopy(dict(value))


def validate_predecessor_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_plan_for_policy(
        value,
        policy=PREDECESSOR_POLICY,
        expected_active_quotas={
            stage.stage_id: stage.active_quota for stage in STAGES
        },
    )


def validate_historical_resource_quota_successor_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact 160/140/120/80 phase-1 predecessor plan."""

    return _validate_plan_for_policy(
        value,
        policy=SUCCESSOR_POLICY,
        expected_active_quotas=HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
    )


def validate_successor_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_launch_plan(value)
    return _validate_plan_for_policy(
        validated,
        policy=SUCCESSOR_POLICY,
        expected_active_quotas=SUCCESSOR_ACTIVE_QUOTAS,
    )


def _render_from_template_for_policy(
    template: Mapping[str, Any],
    *,
    stage_id: str,
    seed: int,
    wave: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    stage = BY_ID[stage_id]
    task = copy.deepcopy(dict(template))
    payload = task["payload_json"]
    lane = dict(payload["lane"])
    payload["seed"] = int(seed)
    lane["seed"] = int(seed)
    lane["wave"] = wave
    payload["lane"] = lane
    task["payload_json"] = payload
    task["name"] = f"{stage.task_name_stem}-{wave}-{int(seed)}"
    payload_sha = canonical_sha256(payload)
    command, count = PAYLOAD_SHA_PATTERN.subn(
        lambda match: match.group(1) + payload_sha,
        str(task.get("command") or ""),
    )
    if count != 1:
        raise RuntimeError("predecessor task command payload seal is ambiguous")
    task["command"] = command
    resources = _task_resources(policy)
    task.update(resources)
    digest = canonical_sha256(
        {
            "goal": "final1000",
            "stage_profile_sha256": stage_profile(stage)["sha256"],
            "bundle_id": payload.get("bundle_id"),
            "payload": payload,
            "resources": resources,
        }
    )
    task["dedupe_key"] = f"{DEDUPE_PREFIX}{digest}"
    return _validate_task_for_policy(task, policy=policy)


def _render_from_template(
    template: Mapping[str, Any], *, stage_id: str, seed: int, wave: str
) -> dict[str, Any]:
    """Backward-compatible predecessor renderer used by evidence tests."""

    return _render_from_template_for_policy(
        template,
        stage_id=stage_id,
        seed=seed,
        wave=wave,
        policy=PREDECESSOR_POLICY,
    )


def _validate_predecessor_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in state.items() if key != "state_sha256"}
    entries = state.get("entries")
    next_seeds = state.get("next_seed_by_stage")
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("launch_plan_sha256") != plan.get("launch_plan_sha256")
        or state.get("state_sha256") != canonical_sha256(unsigned)
        or state.get("scheduler_name_prefix") != TASK_NAME_PREFIX
        or state.get("scheduler_dedupe_prefix") != DEDUPE_PREFIX
        or state.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or state.get("stop_requested") is not True
        or state.get("ramp_released") is not True
        or isinstance(state.get("revision"), bool)
        or not isinstance(state.get("revision"), int)
        or int(state["revision"]) < 0
        or isinstance(state.get("scheduler_submit_count"), bool)
        or not isinstance(state.get("scheduler_submit_count"), int)
        or int(state["scheduler_submit_count"]) < 0
        or set(state.get("canary_passed_stage_ids") or []) != set(BY_ID)
        or not isinstance(entries, list)
        or not isinstance(next_seeds, dict)
        or set(next_seeds) != set(BY_ID)
    ):
        raise RuntimeError("stopped predecessor state identity/SHA mismatch")
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in (plan.get("task_waves") or {}).get("canaries") or []
    }
    if set(templates) != set(BY_ID):
        raise RuntimeError("predecessor plan has no unique stage templates")
    task_ids: set[int] = set()
    dedupes: set[str] = set()
    seed_ids: set[tuple[str, int]] = set()
    expected_by_id: dict[int, dict[str, Any]] = {}
    max_seed = {stage.stage_id: stage.seed_start - 1 for stage in STAGES}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("predecessor ledger entry is not an object")
        stage_id = str(entry.get("stage_id") or "")
        stage = BY_ID.get(stage_id)
        task_id = entry.get("task_id")
        seed = int(entry.get("seed", -1))
        wave = str(entry.get("wave") or "")
        if (
            stage is None
            or entry.get("state") not in ACTIVE_STATES | TERMINAL_STATES
            or entry.get("state") == "planned"
            or wave not in {"canary", "ramp", "refill"}
            or isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not stage.seed_start <= seed < stage.seed_window_end_exclusive
        ):
            raise RuntimeError("predecessor ledger entry drifted")
        expected = _render_from_template(
            templates[stage_id], stage_id=stage_id, seed=seed, wave=wave
        )
        if (
            entry.get("bundle_id") != expected["payload_json"]["bundle_id"]
            or entry.get("dedupe_key") != expected["dedupe_key"]
        ):
            raise RuntimeError("predecessor ledger cannot reproduce sealed task")
        seed_id = (stage_id, seed)
        if task_id in task_ids or entry["dedupe_key"] in dedupes or seed_id in seed_ids:
            raise RuntimeError("predecessor ledger contains duplicate identity")
        task_ids.add(task_id)
        dedupes.add(entry["dedupe_key"])
        seed_ids.add(seed_id)
        expected_by_id[task_id] = expected
        max_seed[stage_id] = max(max_seed[stage_id], seed)
    for stage in STAGES:
        candidate = int(next_seeds[stage.stage_id])
        if (
            candidate <= max_seed[stage.stage_id]
            or not stage.seed_start <= candidate < stage.seed_window_end_exclusive
        ):
            raise RuntimeError("predecessor next-seed cursor can reuse prior work")
    value = copy.deepcopy(dict(state))
    value["_expected_tasks_by_id"] = expected_by_id
    value["_resource_policy_id_by_id"] = {
        task_id: LEGACY_RESOURCE_POLICY_ID for task_id in expected_by_id
    }
    return value


def _validate_resource_successor_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate phase-1 state before dual-binding patched bundles."""

    value = _validate_historical_resource_quota_successor_state(state, plan)
    migration = value.get("rolling_migration") or {}
    if (
        value.get("stop_requested") is not True
        or value.get("ramp_released") is not True
        or set(value.get("canary_passed_stage_ids") or []) != set(BY_ID)
        or migration.get("transition_mode") != RESOURCE_QUOTA_ONLY
    ):
        raise RuntimeError("resource/quota successor must be stopped and released")
    templates = {
        task["payload_json"]["final_goal_stage_id"]: task
        for task in (plan.get("task_waves") or {}).get("canaries") or []
    }
    if set(templates) != set(BY_ID):
        raise RuntimeError("resource/quota plan has no unique stage templates")
    expected_by_id: dict[int, dict[str, Any]] = {}
    policy_id_by_id: dict[int, str] = {}
    for entry in value["entries"]:
        policy_id = str(entry.get("resource_policy_id") or "")
        policy = {
            LEGACY_RESOURCE_POLICY_ID: PREDECESSOR_POLICY,
            SUCCESSOR_RESOURCE_POLICY_ID: SUCCESSOR_POLICY,
        }.get(policy_id)
        if policy is None:
            raise RuntimeError("resource/quota ledger policy identity drifted")
        stage_id = str(entry["stage_id"])
        expected = _render_from_template_for_policy(
            templates[stage_id],
            stage_id=stage_id,
            seed=int(entry["seed"]),
            wave=str(entry["wave"]),
            policy=policy,
        )
        if (
            entry.get("bundle_id") != expected["payload_json"]["bundle_id"]
            or entry.get("dedupe_key") != expected["dedupe_key"]
        ):
            raise RuntimeError("resource/quota ledger cannot reproduce sealed task")
        task_id = int(entry["task_id"])
        expected_by_id[task_id] = expected
        policy_id_by_id[task_id] = policy_id
    value["_expected_tasks_by_id"] = expected_by_id
    value["_resource_policy_id_by_id"] = policy_id_by_id
    return value


def _verify_ready(plan: Mapping[str, Any], probe: ReadyProbe, *, role: str) -> None:
    for stage in STAGES:
        binding = plan["stage_bindings"][stage.stage_id]
        expected = binding["ready"]
        if probe.read_ready(str(binding["remote_bundle"])) != expected:
            raise RuntimeError(f"{role} {stage.stage_id} live READY identity mismatch")


def _scheduler_inventory(
    scheduler: InventoryClient,
    *,
    predecessor_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows = [dict(row) for row in scheduler.list_namespace_tasks()]
    by_id: dict[int, dict[str, Any]] = {}
    by_dedupe: dict[str, int] = {}
    for row in rows:
        task_id = row.get("task_id") or row.get("id")
        dedupe = str(row.get("dedupe_key") or "")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not str(row.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not dedupe.startswith(DEDUPE_PREFIX)
        ):
            raise RuntimeError("scheduler final1000 inventory identity drifted")
        if task_id in by_id or (dedupe in by_dedupe and by_dedupe[dedupe] != task_id):
            raise RuntimeError("scheduler final1000 inventory duplicated identity")
        by_id[task_id] = row
        by_dedupe[dedupe] = task_id
    expected_by_id = predecessor_state["_expected_tasks_by_id"]
    observed_states: dict[int, str] = {}
    for entry in predecessor_state["entries"]:
        task_id = int(entry["task_id"])
        expected = expected_by_id[task_id]
        row = by_id.get(task_id)
        observed = _scheduler_state(row.get("status") if row else None)
        if row is None or observed is None:
            raise RuntimeError(f"scheduler task {task_id} is unavailable or invalid")
        checked_fields = (
            "name",
            "dedupe_key",
            "remote_cwd",
            "required_capability",
            "env_profile",
            "cpus",
            "memory_mb",
            "scheduling_profile",
            "aedt_backend",
            "gpus",
            "priority",
            "timeout_seconds",
            "max_workers_per_node",
        )
        if any(row.get(field) != expected.get(field) for field in checked_fields):
            raise RuntimeError(f"scheduler task {task_id} seal differs from ledger")
        prior = str(entry["state"])
        if prior in TERMINAL_STATES and observed in ACTIVE_STATES:
            raise RuntimeError(f"scheduler task {task_id} regressed from terminal")
        observed_states[task_id] = observed
    return rows, observed_states


def prepare_successor_state(
    *,
    predecessor_plan_path: Path,
    predecessor_state_path: Path,
    successor_plan_path: Path,
    successor_state_path: Path,
    scheduler: InventoryClient,
    predecessor_ready_probe: ReadyProbe,
    successor_ready_probe: ReadyProbe,
    apply: bool = False,
    transition_mode: str = PATCHED_BUNDLE,
    before_final_state_read: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Validate and optionally atomically write a migration controller state."""

    predecessor_plan_raw = _read_json(predecessor_plan_path.resolve(strict=True))
    predecessor_resources = predecessor_plan_raw.get("resources")
    predecessor_controller_kind: str
    if predecessor_resources == {
        key: PREDECESSOR_POLICY[key]
        for key in (
            "cpus_per_task",
            "memory_mb_per_task",
            "max_workers_per_node",
            "priority",
            "scheduling_profile",
            "gpus",
        )
    }:
        predecessor_plan = validate_predecessor_plan(predecessor_plan_raw)
        predecessor_controller_kind = "legacy_8c"
    elif predecessor_resources == {
        key: SUCCESSOR_POLICY[key]
        for key in (
            "cpus_per_task",
            "memory_mb_per_task",
            "max_workers_per_node",
            "priority",
            "scheduling_profile",
            "gpus",
        )
    }:
        predecessor_plan = validate_historical_resource_quota_successor_plan(
            predecessor_plan_raw
        )
        predecessor_controller_kind = "resource_quota_successor"
    else:
        raise RuntimeError("predecessor launch resource policy is unsupported")
    successor_plan = validate_successor_plan(
        _read_json(successor_plan_path.resolve(strict=True))
    )
    if transition_mode not in {RESOURCE_QUOTA_ONLY, PATCHED_BUNDLE}:
        raise ValueError("unknown final1000 transition mode")
    if (
        predecessor_controller_kind == "resource_quota_successor"
        and transition_mode != PATCHED_BUNDLE
    ):
        raise RuntimeError(
            "resource/quota successor may transition only to patched bundles"
        )
    for stage in STAGES:
        old = predecessor_plan["stage_bindings"][stage.stage_id]
        new = successor_plan["stage_bindings"][stage.stage_id]
        same_binding = (
            old["bundle_id"] == new["bundle_id"]
            and old["remote_bundle"] == new["remote_bundle"]
            and old["bundle_manifest_sha256"] == new["bundle_manifest_sha256"]
        )
        different_binding = (
            old["bundle_id"] != new["bundle_id"]
            and old["remote_bundle"] != new["remote_bundle"]
            and old["bundle_manifest_sha256"] != new["bundle_manifest_sha256"]
        )
        if (
            old["stage_spec_sha256"] != new["stage_spec_sha256"]
            or (transition_mode == RESOURCE_QUOTA_ONLY and not same_binding)
            or (transition_mode == PATCHED_BUNDLE and not different_binding)
        ):
            raise RuntimeError("successor bundle transition mode/identity mismatch")
    predecessor_state_raw = _read_json(predecessor_state_path.resolve(strict=True))
    predecessor_state = (
        _validate_predecessor_state(predecessor_state_raw, predecessor_plan)
        if predecessor_controller_kind == "legacy_8c"
        else _validate_resource_successor_state(predecessor_state_raw, predecessor_plan)
    )
    _verify_ready(predecessor_plan, predecessor_ready_probe, role="predecessor")
    _verify_ready(successor_plan, successor_ready_probe, role="successor")
    inventory, observed_states = _scheduler_inventory(
        scheduler, predecessor_state=predecessor_state
    )
    if before_final_state_read is not None:
        before_final_state_read()
    final_predecessor_state = _read_json(predecessor_state_path.resolve(strict=True))
    if final_predecessor_state.get("state_sha256") != predecessor_state_raw.get(
        "state_sha256"
    ) or canonical_sha256(final_predecessor_state) != canonical_sha256(
        predecessor_state_raw
    ):
        raise RuntimeError("predecessor state drifted during migration preflight")

    entries = []
    for old_entry in predecessor_state["entries"]:
        entry = copy.deepcopy(dict(old_entry))
        entry["state"] = observed_states[int(entry["task_id"])]
        entry["origin"] = "predecessor"
        entry["resource_policy_id"] = predecessor_state["_resource_policy_id_by_id"][
            int(entry["task_id"])
        ]
        entries.append(entry)
    active_by_stage = {
        stage.stage_id: sum(
            entry["stage_id"] == stage.stage_id and entry["state"] in ACTIVE_STATES
            for entry in entries
        )
        for stage in STAGES
    }
    if sum(active_by_stage.values()) > TOTAL_ACTIVE_QUOTA:
        raise RuntimeError("predecessor scheduler inventory exceeds total active quota")
    normalized_inventory = sorted(
        (
            {
                "task_id": int(row.get("task_id") or row.get("id")),
                "name": row.get("name"),
                "dedupe_key": row.get("dedupe_key"),
                "status": row.get("status"),
            }
            for row in inventory
        ),
        key=lambda row: row["task_id"],
    )
    predecessor_policy_ids = sorted(
        {
            str(value)
            for value in predecessor_state["_resource_policy_id_by_id"].values()
        }
    )
    migration = _seal_nested(
        {
            "schema_version": MIGRATION_SCHEMA,
            "transition_mode": transition_mode,
            "predecessor_controller_kind": predecessor_controller_kind,
            "predecessor_launch_plan_sha256": predecessor_plan["launch_plan_sha256"],
            "predecessor_state_sha256": predecessor_state_raw["state_sha256"],
            "predecessor_state_revision": int(predecessor_state_raw["revision"]),
            "successor_launch_plan_sha256": successor_plan["launch_plan_sha256"],
            "scheduler_inventory_sha256": canonical_sha256(normalized_inventory),
            "predecessor_entry_count": len(entries),
            "imported_active_count_by_stage": active_by_stage,
            "next_seed_by_stage": copy.deepcopy(
                predecessor_state["next_seed_by_stage"]
            ),
            "successor_resource_policy": copy.deepcopy(SUCCESSOR_POLICY),
            "successor_active_quotas": copy.deepcopy(SUCCESSOR_ACTIVE_QUOTAS),
            "harvest_cohorts": {
                "predecessor": harvest_cohort_identity(
                    role="predecessor",
                    plan=predecessor_plan,
                    resource_policy_ids=predecessor_policy_ids,
                ),
                "successor": harvest_cohort_identity(
                    role="successor",
                    plan=successor_plan,
                    resource_policy_ids=[SUCCESSOR_RESOURCE_POLICY_ID],
                ),
            },
            "refill_policy": REFILL_POLICY,
            "successor_canary_task_ids_by_stage": {
                stage.stage_id: [] for stage in STAGES
            },
            "successor_canary_status_by_stage": {
                stage.stage_id: "waiting_for_natural_terminal_gap" for stage in STAGES
            },
            "scheduler_mutation_endpoints": ["POST /api/tasks"],
            "cancellation_performed": False,
            "preemption_performed": False,
            "controller_stop_performed_by_this_tool": False,
        }
    )
    unsigned_state = {
        "schema_version": STATE_SCHEMA,
        "launch_plan_sha256": successor_plan["launch_plan_sha256"],
        "revision": 0,
        "parent_state_sha256": predecessor_state_raw["state_sha256"],
        "stop_requested": False,
        "ramp_released": False,
        "canary_passed_stage_ids": [],
        "entries": entries,
        "next_seed_by_stage": copy.deepcopy(predecessor_state["next_seed_by_stage"]),
        "scheduler_name_prefix": TASK_NAME_PREFIX,
        "scheduler_dedupe_prefix": DEDUPE_PREFIX,
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "scheduler_submit_count": int(
            predecessor_state.get("scheduler_submit_count") or 0
        ),
        "refill_policy": REFILL_POLICY,
        "refill_stage_cursor": 0,
        "refill_deficit_credit_by_stage": {stage.stage_id: 0 for stage in STAGES},
        "rolling_migration": migration,
    }
    successor_state = _seal_state(unsigned_state)
    if successor_state_path.exists():
        existing = _read_json(successor_state_path)
        if existing != successor_state:
            raise RuntimeError(
                "successor state target already exists with other identity"
            )
    elif apply:
        _write_state(successor_state_path, successor_state)
    return {
        "schema_version": MIGRATION_SCHEMA,
        "apply": bool(apply),
        "transition_mode": transition_mode,
        "predecessor_entry_count": len(entries),
        "imported_active_count": sum(active_by_stage.values()),
        "imported_active_count_by_stage": active_by_stage,
        "logical_active_target": TOTAL_ACTIVE_QUOTA,
        "next_seed_by_stage": copy.deepcopy(predecessor_state["next_seed_by_stage"]),
        "successor_state_sha256": successor_state["state_sha256"],
        "successor_state_written": bool(apply and successor_state_path.is_file()),
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "scheduler_post_count": 0,
        "cancellation_performed": False,
        "preemption_performed": False,
        "successor_state": successor_state,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--successor-plan", type=Path, required=True)
    parser.add_argument("--successor-state", type=Path, required=True)
    parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--resource-quota-only",
        action="store_true",
        help="reuse the exact predecessor bundles while changing only resources/quotas",
    )
    parser.add_argument(
        "--accounts",
        type=Path,
        default=Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml"),
    )
    parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=Path(r"C:\Users\peets\NEC\slurm_scheduler"),
    )
    parser.add_argument("--publication-account", default="harry261")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    scheduler = SchedulerApiClient(args.scheduler_url)
    with scheduler_publication_transport(
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        account_name=args.publication_account,
    ) as transport:
        probe = TransportReadyProbe(transport)
        value = prepare_successor_state(
            predecessor_plan_path=args.predecessor_plan,
            predecessor_state_path=args.predecessor_state,
            successor_plan_path=args.successor_plan,
            successor_state_path=args.successor_state,
            scheduler=scheduler,
            predecessor_ready_probe=probe,
            successor_ready_probe=probe,
            apply=args.apply,
            transition_mode=(
                RESOURCE_QUOTA_ONLY if args.resource_quota_only else PATCHED_BUNDLE
            ),
        )
    compact = {key: item for key, item in value.items() if key != "successor_state"}
    print(json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
