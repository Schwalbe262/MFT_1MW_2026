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
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_controller import TransportReadyProbe
    from tier1_corrected_current7_slurm_publish import scheduler_publication_transport
    from tier1_final1000_slurm_controller import (
        ACTIVE_STATES,
        CHAINED_CANARY_GAP_POLICY,
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
        PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS,
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
        CHAINED_CANARY_GAP_POLICY,
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
        PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS,
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
CHAINED_MIGRATION_SCHEMA = "mft-tier1-final1000-rolling-migration-v2"
HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v1"
CHAINED_HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v2"
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
SUPPORTED_PREDECESSOR_GENERATIONS = frozenset({200, FIXED_GENERATIONS})


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


def _stage_profile_for_generations(stage_id: str, generations: int) -> dict[str, Any]:
    """Recreate one explicitly supported historical science profile.

    Generation 200 is the only historical science identity accepted by this
    release.  Everything else in the profile remains byte-for-byte identical
    to the current sealed profile.  This avoids trusting a predecessor plan's
    self-described science while permitting the intentional 200 -> 300 handoff.
    """

    if generations not in SUPPORTED_PREDECESSOR_GENERATIONS:
        raise RuntimeError("unsupported final1000 predecessor generation policy")
    stage = BY_ID.get(stage_id)
    if stage is None:
        raise RuntimeError("unknown final1000 stage in historical profile")
    profile = copy.deepcopy(stage_profile(stage))
    profile["fixed_generations"] = int(generations)
    profile["sha256"] = canonical_sha256(
        {key: item for key, item in profile.items() if key != "sha256"}
    )
    return profile


def _stage_inventory_for_generations(generations: int) -> dict[str, Any]:
    inventory = copy.deepcopy(stage_inventory())
    profiles = [
        _stage_profile_for_generations(stage.stage_id, generations)
        for stage in STAGES
    ]
    inventory["profiles"] = profiles
    inventory["profile_sha256_by_stage"] = {
        profile["stage_id"]: profile["sha256"] for profile in profiles
    }
    inventory["sha256"] = canonical_sha256(
        {key: item for key, item in inventory.items() if key != "sha256"}
    )
    return inventory


def _plan_fixed_generations(value: Mapping[str, Any]) -> int:
    inventory = value.get("stage_inventory")
    profiles = inventory.get("profiles") if isinstance(inventory, Mapping) else None
    if not isinstance(profiles, list) or len(profiles) != len(STAGES):
        raise RuntimeError("final1000 plan has no complete science profile inventory")
    generations = {
        int(profile.get("fixed_generations", -1))
        for profile in profiles
        if isinstance(profile, Mapping)
    }
    if len(generations) != 1:
        raise RuntimeError("final1000 plan mixes generation policies")
    generation = generations.pop()
    if inventory != _stage_inventory_for_generations(generation):
        raise RuntimeError("final1000 plan science inventory is not a sealed identity")
    return generation


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


def chained_harvest_cohort_identity(
    *,
    cohort_id: str,
    plan: Mapping[str, Any],
    resource_policy_ids: Sequence[str],
) -> dict[str, Any]:
    """Seal one member of an append-only multi-generation cohort catalog."""

    if not re.fullmatch(r"(?:predecessor|successor)-[0-9a-f]{12}", cohort_id):
        raise RuntimeError("chained harvest cohort id is invalid")
    policies = sorted({str(value) for value in resource_policy_ids})
    if not policies or any(value not in RESOURCE_POLICIES for value in policies):
        raise RuntimeError("chained harvest cohort resource policy is invalid")
    bindings = plan.get("stage_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(BY_ID):
        raise RuntimeError("chained harvest cohort bindings are incomplete")
    stage_bindings = {
        stage.stage_id: {
            key: copy.deepcopy(bindings[stage.stage_id].get(key))
            for key in (
                "bundle_id",
                "bundle_manifest_sha256",
                "remote_bundle",
                "publication_receipt_sha256",
                "ready_sha256",
                "stage_spec_sha256",
            )
        }
        for stage in STAGES
    }
    unsigned = {
        "schema_version": CHAINED_HARVEST_COHORT_SCHEMA,
        "cohort_id": cohort_id,
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
    fixed_generations: int = FIXED_GENERATIONS,
) -> dict[str, Any]:
    if set(task) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError("final1000 migration task fields drifted")
    payload = task.get("payload_json") or {}
    stage = BY_ID.get(str(payload.get("final_goal_stage_id") or ""))
    if stage is None:
        raise RuntimeError("final1000 migration task stage is unknown")
    spec, constraints = optimizer_stage_contract(stage)
    expected_profile = _stage_profile_for_generations(
        stage.stage_id, fixed_generations
    )
    lane = payload.get("lane") or {}
    seed = int(payload.get("seed", -1))
    wave = str(lane.get("wave") or "")
    resources = _task_resources(policy)
    dedupe = canonical_sha256(
        {
            "goal": "final1000",
            "stage_profile_sha256": expected_profile["sha256"],
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
        != expected_profile["sha256"]
        or payload.get("max_generations") != fixed_generations
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
    expected_generations: int = FIXED_GENERATIONS,
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
                PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS,
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
        or value.get("stage_inventory")
        != _stage_inventory_for_generations(expected_generations)
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
        _validate_task_for_policy(
            task,
            policy=policy,
            expected_wave="canary",
            fixed_generations=expected_generations,
        )
    for task in ramp:
        _validate_task_for_policy(
            task,
            policy=policy,
            expected_wave="ramp",
            fixed_generations=expected_generations,
        )
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


def validate_previous_successor_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the running 200/160/90/50 patched predecessor plan."""

    return _validate_plan_for_policy(
        value,
        policy=SUCCESSOR_POLICY,
        expected_active_quotas=PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS,
    )


def validate_successor_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_launch_plan(value)
    return _validate_plan_for_policy(
        validated,
        policy=SUCCESSOR_POLICY,
        expected_active_quotas=SUCCESSOR_ACTIVE_QUOTAS,
    )


def _plan_policy_and_quotas(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    resources = value.get("resources")
    resource_keys = (
        "cpus_per_task",
        "memory_mb_per_task",
        "max_workers_per_node",
        "priority",
        "scheduling_profile",
        "gpus",
    )
    if resources == {key: PREDECESSOR_POLICY[key] for key in resource_keys}:
        policy = PREDECESSOR_POLICY
    elif resources == {key: SUCCESSOR_POLICY[key] for key in resource_keys}:
        policy = SUCCESSOR_POLICY
    else:
        raise RuntimeError("predecessor launch resource policy is unsupported")
    refill = value.get("open_ended_refill")
    quotas = refill.get("stage_active_quotas") if isinstance(refill, Mapping) else None
    if not isinstance(quotas, Mapping):
        raise RuntimeError("predecessor launch quota policy is unavailable")
    return dict(policy), {str(key): int(item) for key, item in quotas.items()}


def validate_chained_predecessor_plan(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a bounded 200/300-generation predecessor cohort plan."""

    generations = _plan_fixed_generations(value)
    policy, quotas = _plan_policy_and_quotas(value)
    legacy_quotas = {stage.stage_id: stage.active_quota for stage in STAGES}
    allowed = (
        quotas == legacy_quotas
        if policy == PREDECESSOR_POLICY
        else quotas
        in (
            HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
            PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS,
            SUCCESSOR_ACTIVE_QUOTAS,
        )
    )
    if not allowed:
        raise RuntimeError("unsealed chained predecessor quota policy")
    return _validate_plan_for_policy(
        value,
        policy=policy,
        expected_active_quotas=quotas,
        expected_generations=generations,
    )


def _render_from_template_for_policy(
    template: Mapping[str, Any],
    *,
    stage_id: str,
    seed: int,
    wave: str,
    policy: Mapping[str, Any],
    fixed_generations: int = FIXED_GENERATIONS,
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
            "stage_profile_sha256": _stage_profile_for_generations(
                stage_id, fixed_generations
            )["sha256"],
            "bundle_id": payload.get("bundle_id"),
            "payload": payload,
            "resources": resources,
        }
    )
    task["dedupe_key"] = f"{DEDUPE_PREFIX}{digest}"
    return _validate_task_for_policy(
        task, policy=policy, fixed_generations=fixed_generations
    )


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


def _cohort_id(role: str, launch_plan_sha256: str) -> str:
    if role not in {"predecessor", "successor"} or not re.fullmatch(
        r"[0-9a-f]{64}", launch_plan_sha256
    ):
        raise RuntimeError("cannot derive chained harvest cohort identity")
    return f"{role}-{launch_plan_sha256[:12]}"


def _validate_chained_predecessor_state(
    state: Mapping[str, Any],
    primary_plan: Mapping[str, Any],
    *,
    plan_by_sha256: Mapping[str, Mapping[str, Any]],
    allow_running_shadow: bool,
) -> dict[str, Any]:
    """Authenticate every cohort and ledger entry in an existing rollout.

    Unlike the original one-hop validator this path accepts an already patched
    200/160/90/50 successor and an append-only v2 cohort catalog.  Every cohort
    must have its full immutable launch plan supplied by the caller; a catalog
    hash alone is never treated as enough evidence to reproduce a task.
    """

    unsigned = {key: item for key, item in state.items() if key != "state_sha256"}
    entries = state.get("entries")
    next_seeds = state.get("next_seed_by_stage")
    migration = state.get("rolling_migration")
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("launch_plan_sha256") != primary_plan.get("launch_plan_sha256")
        or state.get("state_sha256") != canonical_sha256(unsigned)
        or state.get("scheduler_name_prefix") != TASK_NAME_PREFIX
        or state.get("scheduler_dedupe_prefix") != DEDUPE_PREFIX
        or state.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or (
            state.get("stop_requested") is not True
            and state.get("stop_requested") is not False
        )
        or (state.get("stop_requested") is not True and not allow_running_shadow)
        or state.get("ramp_released") is not True
        or set(state.get("canary_passed_stage_ids") or []) != set(BY_ID)
        or isinstance(state.get("revision"), bool)
        or not isinstance(state.get("revision"), int)
        or int(state["revision"]) < 0
        or isinstance(state.get("scheduler_submit_count"), bool)
        or not isinstance(state.get("scheduler_submit_count"), int)
        or int(state["scheduler_submit_count"]) < 0
        or not isinstance(entries, list)
        or not isinstance(next_seeds, dict)
        or set(next_seeds) != set(BY_ID)
        or state.get("refill_policy") != REFILL_POLICY
        or isinstance(state.get("refill_stage_cursor"), bool)
        or not isinstance(state.get("refill_stage_cursor"), int)
        or not 0 <= int(state["refill_stage_cursor"]) < len(STAGES)
        or set(state.get("refill_deficit_credit_by_stage") or {}) != set(BY_ID)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (state.get("refill_deficit_credit_by_stage") or {}).values()
        )
        or not isinstance(migration, dict)
    ):
        raise RuntimeError("chained predecessor state identity/SHA mismatch")
    migration_unsigned = {
        key: item for key, item in migration.items() if key != "sha256"
    }
    primary_quotas = dict(
        (primary_plan.get("open_ended_refill") or {}).get(
            "stage_active_quotas"
        )
        or {}
    )
    if (
        migration.get("schema_version")
        not in {MIGRATION_SCHEMA, CHAINED_MIGRATION_SCHEMA}
        or migration.get("sha256") != canonical_sha256(migration_unsigned)
        or migration.get("transition_mode") != PATCHED_BUNDLE
        or migration.get("successor_launch_plan_sha256")
        != primary_plan.get("launch_plan_sha256")
        or migration.get("successor_resource_policy") != SUCCESSOR_POLICY
        or migration.get("successor_active_quotas") != primary_quotas
        or migration.get("refill_policy") != REFILL_POLICY
        or migration.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or migration.get("cancellation_performed") is not False
        or migration.get("preemption_performed") is not False
        or migration.get("controller_stop_performed_by_this_tool") is not False
    ):
        raise RuntimeError("chained predecessor rolling migration seal mismatch")

    stored_cohorts = migration.get("harvest_cohorts")
    if not isinstance(stored_cohorts, dict) or not stored_cohorts:
        raise RuntimeError("chained predecessor has no harvest cohort evidence")
    entry_cohort_id: dict[int, str] = {}
    catalog: dict[str, dict[str, Any]] = {}
    if migration["schema_version"] == MIGRATION_SCHEMA:
        if (
            migration.get("predecessor_controller_kind")
            != "resource_quota_successor"
            or set(stored_cohorts) != {"predecessor", "successor"}
        ):
            raise RuntimeError("legacy rolling cohort roles drifted")
        role_to_cohort_id: dict[str, str] = {}
        for role in ("predecessor", "successor"):
            stored = stored_cohorts[role]
            if not isinstance(stored, Mapping):
                raise RuntimeError("legacy rolling cohort evidence is invalid")
            plan_sha = str(stored.get("launch_plan_sha256") or "")
            plan = plan_by_sha256.get(plan_sha)
            policies = stored.get("resource_policy_ids")
            if plan is None or not isinstance(policies, list):
                raise RuntimeError("a sealed predecessor cohort plan is missing")
            expected = harvest_cohort_identity(
                role=role, plan=plan, resource_policy_ids=policies
            )
            if stored != expected:
                raise RuntimeError("legacy rolling harvest cohort seal mismatch")
            cohort_id = _cohort_id(role, plan_sha)
            role_to_cohort_id[role] = cohort_id
            catalog[cohort_id] = chained_harvest_cohort_identity(
                cohort_id=cohort_id,
                plan=plan,
                resource_policy_ids=policies,
            )
        if stored_cohorts["successor"].get("launch_plan_sha256") != primary_plan.get(
            "launch_plan_sha256"
        ):
            raise RuntimeError("legacy rolling successor plan identity drifted")
        for index, entry in enumerate(entries):
            origin = str(entry.get("origin") or "successor")
            if origin not in role_to_cohort_id:
                raise RuntimeError("legacy rolling ledger origin is unknown")
            entry_cohort_id[index] = role_to_cohort_id[origin]
    else:
        predecessor_ids = migration.get("predecessor_harvest_cohort_ids")
        successor_id = migration.get("successor_harvest_cohort_id")
        if (
            migration.get("predecessor_controller_kind")
            != "chained_patched_successor"
            or migration.get("predecessor_stop_observed") is not True
            or migration.get("shadow_only") is not False
            or not isinstance(predecessor_ids, list)
            or predecessor_ids != sorted(set(predecessor_ids))
            or not isinstance(successor_id, str)
            or set(stored_cohorts) != set(predecessor_ids) | {successor_id}
        ):
            raise RuntimeError("chained harvest cohort catalog membership drifted")
        for cohort_id, stored in stored_cohorts.items():
            if not isinstance(stored, Mapping):
                raise RuntimeError("chained harvest cohort evidence is invalid")
            plan_sha = str(stored.get("launch_plan_sha256") or "")
            plan = plan_by_sha256.get(plan_sha)
            policies = stored.get("resource_policy_ids")
            if plan is None or not isinstance(policies, list):
                raise RuntimeError("a chained predecessor cohort plan is missing")
            expected = chained_harvest_cohort_identity(
                cohort_id=cohort_id,
                plan=plan,
                resource_policy_ids=policies,
            )
            if stored != expected:
                raise RuntimeError("chained harvest cohort seal mismatch")
            catalog[cohort_id] = copy.deepcopy(dict(stored))
        for index, entry in enumerate(entries):
            cohort_id = str(entry.get("harvest_cohort_id") or "")
            if cohort_id not in catalog:
                raise RuntimeError("chained rolling ledger cohort is unknown")
            entry_cohort_id[index] = cohort_id

    referenced_plan_shas = {
        str(cohort["launch_plan_sha256"]) for cohort in catalog.values()
    }
    if set(plan_by_sha256) != referenced_plan_shas:
        raise RuntimeError("predecessor cohort plan inventory is incomplete or excessive")
    successor_candidates = [
        cohort_id
        for cohort_id, cohort in catalog.items()
        if cohort["launch_plan_sha256"] == primary_plan["launch_plan_sha256"]
    ]
    if len(successor_candidates) != 1:
        raise RuntimeError("primary predecessor plan has ambiguous cohort identity")

    templates_by_plan: dict[str, dict[str, Mapping[str, Any]]] = {}
    generations_by_plan: dict[str, int] = {}
    for plan_sha, plan in plan_by_sha256.items():
        generations_by_plan[plan_sha] = _plan_fixed_generations(plan)
        templates = {
            task["payload_json"]["final_goal_stage_id"]: task
            for task in (plan.get("task_waves") or {}).get("canaries") or []
        }
        if set(templates) != set(BY_ID):
            raise RuntimeError("predecessor cohort plan lacks stage templates")
        templates_by_plan[plan_sha] = templates

    task_ids: set[int] = set()
    dedupes: set[str] = set()
    seed_ids: set[tuple[str, int]] = set()
    expected_by_id: dict[int, dict[str, Any]] = {}
    policy_by_id: dict[int, str] = {}
    cohort_by_id: dict[int, str] = {}
    max_seed = {stage.stage_id: stage.seed_start - 1 for stage in STAGES}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError("chained predecessor ledger entry is not an object")
        stage_id = str(entry.get("stage_id") or "")
        stage = BY_ID.get(stage_id)
        task_id = entry.get("task_id")
        seed = int(entry.get("seed", -1))
        wave = str(entry.get("wave") or "")
        policy_id = str(entry.get("resource_policy_id") or "")
        cohort_id = entry_cohort_id[index]
        cohort = catalog[cohort_id]
        plan_sha = str(cohort["launch_plan_sha256"])
        policy = RESOURCE_POLICIES.get(policy_id)
        if (
            stage is None
            or entry.get("state") not in ACTIVE_STATES | TERMINAL_STATES
            or entry.get("state") == "planned"
            or wave not in {"canary", "ramp", "refill"}
            or isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not stage.seed_start <= seed < stage.seed_window_end_exclusive
            or policy is None
            or policy_id not in cohort["resource_policy_ids"]
            or entry.get("bundle_id")
            != cohort["stage_bindings"][stage_id]["bundle_id"]
        ):
            raise RuntimeError("chained predecessor ledger entry drifted")
        expected = _render_from_template_for_policy(
            templates_by_plan[plan_sha][stage_id],
            stage_id=stage_id,
            seed=seed,
            wave=wave,
            policy=policy,
            fixed_generations=generations_by_plan[plan_sha],
        )
        if (
            entry.get("bundle_id") != expected["payload_json"]["bundle_id"]
            or entry.get("dedupe_key") != expected["dedupe_key"]
        ):
            raise RuntimeError("chained predecessor task cannot be reproduced")
        seed_id = (stage_id, seed)
        if task_id in task_ids or entry["dedupe_key"] in dedupes or seed_id in seed_ids:
            raise RuntimeError("chained predecessor ledger contains duplicate identity")
        task_ids.add(task_id)
        dedupes.add(entry["dedupe_key"])
        seed_ids.add(seed_id)
        expected_by_id[task_id] = expected
        policy_by_id[task_id] = policy_id
        cohort_by_id[task_id] = cohort_id
        max_seed[stage_id] = max(max_seed[stage_id], seed)
    for stage in STAGES:
        candidate = int(next_seeds[stage.stage_id])
        if (
            candidate <= max_seed[stage.stage_id]
            or not stage.seed_start <= candidate < stage.seed_window_end_exclusive
        ):
            raise RuntimeError("chained predecessor next-seed cursor can reuse work")

    expected_canary_ids = {
        stage.stage_id: sorted(
            int(entry["task_id"])
            for entry in entries
            if entry["stage_id"] == stage.stage_id
            and entry["wave"] == "canary"
            and str(entry.get("origin") or "successor") == "successor"
        )
        for stage in STAGES
    }
    if (
        migration.get("successor_canary_task_ids_by_stage") != expected_canary_ids
        or (
            migration.get("schema_version") == CHAINED_MIGRATION_SCHEMA
            and (
                migration.get("successor_canary_gap_policy")
                != CHAINED_CANARY_GAP_POLICY
                or any(
                    len(task_ids) > 1
                    for task_ids in expected_canary_ids.values()
                )
            )
        )
        or migration.get("successor_canary_status_by_stage")
        != {stage.stage_id: "remote_preflight_passed" for stage in STAGES}
        or migration.get("next_seed_by_stage")
        != {stage.stage_id: int(next_seeds[stage.stage_id]) for stage in STAGES}
        or migration.get("predecessor_entry_count")
        != sum(
            str(entry.get("origin") or "successor") == "predecessor"
            for entry in entries
        )
    ):
        raise RuntimeError("chained predecessor canary/seed seal mismatch")

    value = copy.deepcopy(dict(state))
    value["_expected_tasks_by_id"] = expected_by_id
    value["_resource_policy_id_by_id"] = policy_by_id
    value["_harvest_cohort_id_by_id"] = cohort_by_id
    value["_harvest_cohort_catalog"] = catalog
    value["_predecessor_stopped"] = state.get("stop_requested") is True
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
    complete_reader = getattr(scheduler, "list_complete_namespace_tasks", None)
    rows = [
        dict(row)
        for row in (
            complete_reader()
            if callable(complete_reader)
            else scheduler.list_namespace_tasks()
        )
    ]
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
    ancestor_plan_paths: Sequence[Path] = (),
    allow_running_predecessor_shadow: bool = False,
) -> dict[str, Any]:
    """Validate and optionally atomically write a migration controller state."""

    if apply and allow_running_predecessor_shadow:
        raise RuntimeError("a live predecessor shadow can never be applied")
    predecessor_plan_raw = _read_json(predecessor_plan_path.resolve(strict=True))
    predecessor_resources = predecessor_plan_raw.get("resources")
    predecessor_quotas = (
        (predecessor_plan_raw.get("open_ended_refill") or {}).get(
            "stage_active_quotas"
        )
    )
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
    } and predecessor_quotas == HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS:
        predecessor_plan = validate_historical_resource_quota_successor_plan(
            predecessor_plan_raw
        )
        predecessor_controller_kind = "resource_quota_successor"
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
    } and predecessor_quotas == PREVIOUS_SUCCESSOR_ACTIVE_QUOTAS:
        predecessor_plan = validate_previous_successor_plan(
            predecessor_plan_raw
        )
        predecessor_controller_kind = "chained_patched_successor"
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
    } and predecessor_quotas == SUCCESSOR_ACTIVE_QUOTAS:
        predecessor_plan = validate_chained_predecessor_plan(predecessor_plan_raw)
        predecessor_controller_kind = "chained_patched_successor"
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
        raise RuntimeError("final1000 migration launch-plan seal mismatch")
    else:
        raise RuntimeError("predecessor launch resource policy is unsupported")
    successor_plan = validate_successor_plan(
        _read_json(successor_plan_path.resolve(strict=True))
    )
    successor_plan_next_seed_by_stage = {
        stage.stage_id: max(
            int(task["payload_json"]["seed"])
            for wave in ("canaries", "ramp")
            for task in successor_plan["task_waves"][wave]
            if task["payload_json"]["final_goal_stage_id"] == stage.stage_id
        )
        + 1
        for stage in STAGES
    }
    predecessor_generations = _plan_fixed_generations(predecessor_plan)
    if predecessor_generations not in SUPPORTED_PREDECESSOR_GENERATIONS:
        raise RuntimeError("unsupported predecessor science transition")
    if transition_mode not in {RESOURCE_QUOTA_ONLY, PATCHED_BUNDLE}:
        raise ValueError("unknown final1000 transition mode")
    if (
        predecessor_controller_kind
        in {
            "resource_quota_successor",
            "chained_patched_successor",
        }
        and transition_mode != PATCHED_BUNDLE
    ):
        raise RuntimeError(
            "an existing successor may transition only to patched bundles"
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
    cohort_plan_by_sha: dict[str, dict[str, Any]] = {
        predecessor_plan["launch_plan_sha256"]: predecessor_plan
    }
    for ancestor_path in ancestor_plan_paths:
        ancestor = validate_chained_predecessor_plan(
            _read_json(ancestor_path.resolve(strict=True))
        )
        ancestor_sha = str(ancestor["launch_plan_sha256"])
        if ancestor_sha in cohort_plan_by_sha:
            raise RuntimeError("predecessor cohort plan was supplied more than once")
        cohort_plan_by_sha[ancestor_sha] = ancestor
    if predecessor_controller_kind == "legacy_8c":
        if ancestor_plan_paths:
            raise RuntimeError("legacy predecessor cannot accept ancestor plans")
        predecessor_state = _validate_predecessor_state(
            predecessor_state_raw, predecessor_plan
        )
    elif predecessor_controller_kind == "resource_quota_successor":
        if ancestor_plan_paths:
            raise RuntimeError("historical predecessor cannot accept ancestor plans")
        predecessor_state = _validate_resource_successor_state(
            predecessor_state_raw, predecessor_plan
        )
    else:
        predecessor_state = _validate_chained_predecessor_state(
            predecessor_state_raw,
            predecessor_plan,
            plan_by_sha256=cohort_plan_by_sha,
            allow_running_shadow=allow_running_predecessor_shadow,
        )
    successor_next_seed_by_stage = {
        stage.stage_id: max(
            int(predecessor_state["next_seed_by_stage"][stage.stage_id]),
            int(successor_plan_next_seed_by_stage[stage.stage_id]),
        )
        for stage in STAGES
    }
    for stage in STAGES:
        candidate = successor_next_seed_by_stage[stage.stage_id]
        if not stage.seed_start <= candidate < stage.seed_window_end_exclusive:
            raise RuntimeError("successor topology seed cursor escaped stage window")
    for index, cohort_plan in enumerate(cohort_plan_by_sha.values()):
        _verify_ready(
            cohort_plan,
            predecessor_ready_probe,
            role=f"predecessor-cohort-{index}",
        )
    _verify_ready(successor_plan, successor_ready_probe, role="successor")
    if (
        predecessor_controller_kind == "chained_patched_successor"
        and allow_running_predecessor_shadow
    ):
        # Remote READY checks can take longer than the active controller poll.
        # Refresh the live snapshot only after those checks, immediately before
        # the single scheduler inventory GET, to minimize benign shadow drift.
        refreshed = _read_json(predecessor_state_path.resolve(strict=True))
        if refreshed.get("state_sha256") != predecessor_state_raw.get(
            "state_sha256"
        ):
            predecessor_state_raw = refreshed
            predecessor_state = _validate_chained_predecessor_state(
                predecessor_state_raw,
                predecessor_plan,
                plan_by_sha256=cohort_plan_by_sha,
                allow_running_shadow=True,
            )
            successor_next_seed_by_stage = {
                stage.stage_id: max(
                    int(predecessor_state["next_seed_by_stage"][stage.stage_id]),
                    int(successor_plan_next_seed_by_stage[stage.stage_id]),
                )
                for stage in STAGES
            }
            if any(
                not stage.seed_start
                <= successor_next_seed_by_stage[stage.stage_id]
                < stage.seed_window_end_exclusive
                for stage in STAGES
            ):
                raise RuntimeError(
                    "refreshed successor seed cursor escaped stage window"
                )
    inventory, observed_states = _scheduler_inventory(
        scheduler, predecessor_state=predecessor_state
    )
    inventory_snapshot_receipt = copy.deepcopy(
        getattr(scheduler, "inventory_snapshot_receipt", None)
    )
    if inventory_snapshot_receipt is not None:
        if not isinstance(inventory_snapshot_receipt, dict):
            raise RuntimeError("scheduler complete inventory receipt mismatch")
        snapshot_unsigned = {
            key: item
            for key, item in inventory_snapshot_receipt.items()
            if key != "sha256"
        }
        if (
            inventory_snapshot_receipt.get("schema_version")
            != "mft-tier1-final1000-inventory-snapshot-v1"
            or inventory_snapshot_receipt.get("sha256")
            != canonical_sha256(snapshot_unsigned)
            or inventory_snapshot_receipt.get("filtered_total") != len(inventory)
        ):
            raise RuntimeError("scheduler complete inventory receipt mismatch")
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
        if predecessor_controller_kind == "chained_patched_successor":
            entry["harvest_cohort_id"] = predecessor_state[
                "_harvest_cohort_id_by_id"
            ][int(entry["task_id"])]
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
    ledger_task_ids = {int(entry["task_id"]) for entry in entries}
    namespace_extra_task_ids = sorted(
        int(row.get("task_id") or row.get("id"))
        for row in inventory
        if int(row.get("task_id") or row.get("id")) not in ledger_task_ids
    )
    predecessor_policy_ids = sorted(
        {
            str(value)
            for value in predecessor_state["_resource_policy_id_by_id"].values()
        }
    )
    migration_payload: dict[str, Any] = {
        "schema_version": MIGRATION_SCHEMA,
        "transition_mode": transition_mode,
        "predecessor_controller_kind": predecessor_controller_kind,
        "predecessor_launch_plan_sha256": predecessor_plan["launch_plan_sha256"],
        "predecessor_state_sha256": predecessor_state_raw["state_sha256"],
        "predecessor_state_revision": int(predecessor_state_raw["revision"]),
        "successor_launch_plan_sha256": successor_plan["launch_plan_sha256"],
        "scheduler_inventory_sha256": canonical_sha256(normalized_inventory),
        "scheduler_complete_inventory_snapshot": inventory_snapshot_receipt,
        "predecessor_entry_count": len(entries),
        "imported_active_count_by_stage": active_by_stage,
        "predecessor_next_seed_by_stage": copy.deepcopy(
            predecessor_state["next_seed_by_stage"]
        ),
        "successor_plan_next_seed_by_stage": copy.deepcopy(
            successor_plan_next_seed_by_stage
        ),
        "next_seed_by_stage": copy.deepcopy(successor_next_seed_by_stage),
        "successor_resource_policy": copy.deepcopy(SUCCESSOR_POLICY),
        "successor_active_quotas": copy.deepcopy(SUCCESSOR_ACTIVE_QUOTAS),
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
    if predecessor_controller_kind == "chained_patched_successor":
        catalog = copy.deepcopy(predecessor_state["_harvest_cohort_catalog"])
        successor_cohort_id = _cohort_id(
            "successor", str(successor_plan["launch_plan_sha256"])
        )
        if successor_cohort_id in catalog:
            raise RuntimeError("successor cohort reuses a predecessor plan identity")
        successor_cohort = chained_harvest_cohort_identity(
            cohort_id=successor_cohort_id,
            plan=successor_plan,
            resource_policy_ids=[SUCCESSOR_RESOURCE_POLICY_ID],
        )
        for prior in catalog.values():
            for stage in STAGES:
                old_binding = prior["stage_bindings"][stage.stage_id]
                new_binding = successor_cohort["stage_bindings"][stage.stage_id]
                if any(
                    old_binding[key] == new_binding[key]
                    for key in (
                        "bundle_id",
                        "bundle_manifest_sha256",
                        "remote_bundle",
                    )
                ):
                    raise RuntimeError("successor reuses an earlier bundle identity")
        predecessor_cohort_ids = sorted(catalog)
        catalog[successor_cohort_id] = successor_cohort
        migration_payload.update(
            {
                "schema_version": CHAINED_MIGRATION_SCHEMA,
                "predecessor_harvest_cohort_ids": predecessor_cohort_ids,
                "successor_harvest_cohort_id": successor_cohort_id,
                "harvest_cohorts": catalog,
                "successor_canary_gap_policy": copy.deepcopy(
                    CHAINED_CANARY_GAP_POLICY
                ),
                "predecessor_stop_observed": bool(
                    predecessor_state["_predecessor_stopped"]
                ),
                "shadow_only": not bool(predecessor_state["_predecessor_stopped"]),
            }
        )
    else:
        migration_payload["harvest_cohorts"] = {
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
        }
    migration = _seal_nested(migration_payload)
    unsigned_state = {
        "schema_version": STATE_SCHEMA,
        "launch_plan_sha256": successor_plan["launch_plan_sha256"],
        "revision": 0,
        "parent_state_sha256": predecessor_state_raw["state_sha256"],
        "stop_requested": False,
        "ramp_released": False,
        "canary_passed_stage_ids": [],
        "entries": entries,
        "next_seed_by_stage": copy.deepcopy(successor_next_seed_by_stage),
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
        "schema_version": migration["schema_version"],
        "apply": bool(apply),
        "transition_mode": transition_mode,
        "predecessor_controller_kind": predecessor_controller_kind,
        "predecessor_launch_plan_sha256": predecessor_plan["launch_plan_sha256"],
        "predecessor_state_sha256": predecessor_state_raw["state_sha256"],
        "successor_launch_plan_sha256": successor_plan["launch_plan_sha256"],
        "scheduler_inventory_sha256": migration["scheduler_inventory_sha256"],
        "scheduler_complete_inventory_snapshot": copy.deepcopy(
            migration.get("scheduler_complete_inventory_snapshot")
        ),
        "predecessor_stopped": bool(
            predecessor_state.get("_predecessor_stopped", True)
        ),
        "shadow_only": bool(migration.get("shadow_only", False)),
        "cutover_ready": not bool(migration.get("shadow_only", False)),
        "predecessor_entry_count": len(entries),
        "imported_active_count": sum(active_by_stage.values()),
        "imported_active_count_by_stage": active_by_stage,
        "logical_active_target": TOTAL_ACTIVE_QUOTA,
        "predecessor_next_seed_by_stage": copy.deepcopy(
            predecessor_state["next_seed_by_stage"]
        ),
        "successor_plan_next_seed_by_stage": copy.deepcopy(
            successor_plan_next_seed_by_stage
        ),
        "next_seed_by_stage": copy.deepcopy(successor_next_seed_by_stage),
        "successor_state_sha256": successor_state["state_sha256"],
        "successor_state_shadow_generated": True,
        "successor_state_written": bool(apply and successor_state_path.is_file()),
        "successor_canary_task_ids_by_stage": copy.deepcopy(
            migration["successor_canary_task_ids_by_stage"]
        ),
        "successor_canary_status_by_stage": copy.deepcopy(
            migration["successor_canary_status_by_stage"]
        ),
        "successor_canary_gap_policy": copy.deepcopy(
            migration.get("successor_canary_gap_policy")
        ),
        "harvest_cohort_ids": sorted(migration["harvest_cohorts"]),
        "namespace_extra_task_ids": namespace_extra_task_ids,
        "namespace_extra_task_count": len(namespace_extra_task_ids),
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
        "--ancestor-plan",
        type=Path,
        action="append",
        default=[],
        help=(
            "immutable launch plan for each older cohort named by an already "
            "rolling predecessor state; repeat once per ancestor"
        ),
    )
    parser.add_argument(
        "--allow-running-predecessor-shadow",
        action="store_true",
        help=(
            "read-only planning evidence while the predecessor still runs; "
            "incompatible with --apply and rejected by the successor controller"
        ),
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        help="atomically write a sealed local receipt for a non-applying dry-run",
    )
    parser.add_argument(
        "--shadow-state-evidence-output",
        type=Path,
        help=(
            "atomically persist the sealed, non-runnable shadow state for a "
            "read-only chained harvester preflight"
        ),
    )
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
            ancestor_plan_paths=args.ancestor_plan,
            allow_running_predecessor_shadow=args.allow_running_predecessor_shadow,
            transition_mode=(
                RESOURCE_QUOTA_ONLY if args.resource_quota_only else PATCHED_BUNDLE
            ),
        )
    compact = {key: item for key, item in value.items() if key != "successor_state"}
    if args.shadow_state_evidence_output is not None:
        if args.apply or not args.allow_running_predecessor_shadow:
            raise RuntimeError(
                "shadow state evidence requires a non-applying running-predecessor shadow"
            )
        shadow_path = args.shadow_state_evidence_output.resolve()
        protected_shadow_targets = {
            path.resolve()
            for path in (
                args.predecessor_plan,
                args.predecessor_state,
                args.successor_plan,
                args.successor_state,
                *args.ancestor_plan,
            )
        }
        if shadow_path in protected_shadow_targets:
            raise RuntimeError("shadow state evidence cannot overwrite an input/state file")
        if (
            args.evidence_output is not None
            and shadow_path == args.evidence_output.resolve()
        ):
            raise RuntimeError("shadow state and receipt evidence paths must differ")
        shadow_state = value["successor_state"]
        if shadow_path.exists() and _read_json(shadow_path) != shadow_state:
            raise RuntimeError("shadow state evidence target has another identity")
        if not shadow_path.exists():
            _write_state(shadow_path, shadow_state)
        compact["shadow_state_evidence_path"] = str(shadow_path)
        compact["shadow_state_evidence_sha256"] = shadow_state["state_sha256"]
    if args.evidence_output is not None:
        if args.apply:
            raise RuntimeError("dry-run evidence output is incompatible with --apply")
        protected = {
            path.resolve()
            for path in (
                args.predecessor_plan,
                args.predecessor_state,
                args.successor_plan,
                args.successor_state,
                *args.ancestor_plan,
            )
        }
        if args.shadow_state_evidence_output is not None:
            protected.add(args.shadow_state_evidence_output.resolve())
        evidence_path = args.evidence_output.resolve()
        if evidence_path in protected:
            raise RuntimeError("dry-run evidence cannot overwrite an input/state file")
        unsigned_evidence = {
            "schema_version": "mft-tier1-final1000-migration-dry-run-evidence-v1",
            "scheduler_url": args.scheduler_url,
            "predecessor_plan": str(args.predecessor_plan.resolve()),
            "predecessor_state": str(args.predecessor_state.resolve()),
            "ancestor_plans": sorted(str(path.resolve()) for path in args.ancestor_plan),
            "successor_plan": str(args.successor_plan.resolve()),
            "successor_state_target": str(args.successor_state.resolve()),
            "result": compact,
        }
        evidence = {
            **unsigned_evidence,
            "evidence_sha256": canonical_sha256(unsigned_evidence),
        }
        if evidence_path.exists() and _read_json(evidence_path) != evidence:
            raise RuntimeError("dry-run evidence target already has another identity")
        if not evidence_path.exists():
            _write_state(evidence_path, evidence)
    print(json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
