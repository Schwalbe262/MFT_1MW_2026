"""Namespace-isolated open-ended controller for final1000 surrogate seeds.

Only task names beginning ``mft-t1fg-`` and dedupe keys beginning
``mft-tier1-final1000:`` are visible to this controller.  Its sole scheduler
mutation is ``POST /api/tasks``.  It has no cancel, close, preempt, AEDT, FEA,
or scheduler-administration method.

Dry-run is the default and performs neither scheduler POSTs nor state writes.
In apply mode, one canary per stage is submitted first; after all four remote
model-load/RSS/repair gates pass, the remaining wave is submitted and each
terminal gap is refilled to maintain exactly 500 logical active tasks until an
explicit stop flag is sealed in the controller state.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import math
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error
import urllib.parse
import urllib.request

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_controller import (
        SchedulerApiClient as Current7SchedulerApiClient,
        TransportReadyProbe,
    )
    from tier1_corrected_current7_slurm_publish import (
        scheduler_publication_transport,
    )
    from tier1_final1000_slurm_launch import (
        SUCCESSOR_ACTIVE_QUOTAS,
        build_stage_task,
        validate_launch_plan,
        validate_task,
    )
    from tier1_final1000_stage_profiles import (
        BY_ID,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        stage_profile,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_controller import (
        SchedulerApiClient as Current7SchedulerApiClient,
        TransportReadyProbe,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_slurm_launch import (
        SUCCESSOR_ACTIVE_QUOTAS,
        build_stage_task,
        validate_launch_plan,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import (
        BY_ID,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
        stage_profile,
    )


STATE_SCHEMA = "mft-tier1-final1000-slurm-controller-state-v1"
RESULT_SCHEMA = "mft-tier1-final1000-slurm-controller-result-v1"
ROLLING_MIGRATION_SCHEMA = "mft-tier1-final1000-rolling-migration-v1"
HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v1"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
TASK_NAME_PREFIX = "mft-t1fg-"
DEDUPE_PREFIX = "mft-tier1-final1000:"
ROLLING_SUCCESSOR_RESOURCE_POLICY = {
    "cpus_per_task": 4,
    "memory_mb_per_task": 28 * 1024,
    "max_workers_per_node": 32,
    "priority": 1,
    "scheduling_profile": "standard",
    "gpus": 0,
    "inference_threads": 8,
}
LEGACY_RESOURCE_POLICY_ID = "legacy-8c-28672m-mw8-t8"
SUCCESSOR_RESOURCE_POLICY_ID = "successor-4c-28672m-mw32-t8"
HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS = {
    "entry-1200-t125": 160,
    "bridge-1150-t115": 140,
    "close-1075-t107p5": 120,
    "final-1000-t100": 80,
}
if (
    set(HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS) != set(BY_ID)
    or sum(HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS.values())
    != TOTAL_ACTIVE_QUOTA
):  # pragma: no cover - immutable release invariant
    raise RuntimeError("historical resource/quota successor policy is invalid")

ACTIVE_STATES = frozenset({"planned", "submitted", "queued", "running"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
REFILL_POLICY = "smooth-weighted-deficit-round-robin-v1"
SEED_STATUS_BUSY_MAX_ATTEMPTS = 3
SEED_STATUS_BUSY_RETRY_SECONDS = 0.25


class SeedStatusReadBusy(RuntimeError):
    """Scheduler remote-file readers are at their bounded concurrency limit."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class SchedulerClient(Protocol):
    post_count: int

    def find_task_by_dedupe(self, dedupe_key: str) -> Mapping[str, Any] | None: ...

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_task(self, task_id: int) -> Mapping[str, Any] | None: ...

    def read_seed_status(self, task_id: int) -> Mapping[str, Any] | None: ...


class ReadyProbe(Protocol):
    read_count: int

    def read_ready(self, remote_bundle: str) -> Mapping[str, Any] | None: ...


class SchedulerApiClient(Current7SchedulerApiClient):
    """Scheduler client restricted to the final-goal task namespace."""

    def _load_inventory(self) -> None:
        query = urllib.parse.urlencode(
            {"name_prefix": TASK_NAME_PREFIX, "limit": 10_000}
        )
        value = self._request(f"/api/tasks?{query}")
        if not isinstance(value, list):
            raise RuntimeError("final1000 scheduler inventory is not a list")
        inventory: dict[str, Mapping[str, Any]] = {}
        for task in value:
            if not isinstance(task, dict):
                continue
            name = str(task.get("name") or "")
            dedupe = str(task.get("dedupe_key") or "")
            if not name.startswith(TASK_NAME_PREFIX):
                continue
            if not dedupe.startswith(DEDUPE_PREFIX):
                raise RuntimeError("final1000 namespace task has foreign dedupe key")
            if dedupe in inventory and int(inventory[dedupe]["id"]) != int(task["id"]):
                raise RuntimeError(
                    f"scheduler contains duplicate final1000 dedupe: {dedupe}"
                )
            inventory[dedupe] = task
        self._inventory = inventory

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            not str(payload.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not str(payload.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
            or "requested_allocation_id" in payload
        ):
            raise RuntimeError("foreign or allocation-pinned final1000 submission")
        return super().submit_task(payload)

    def list_namespace_tasks(self) -> list[Mapping[str, Any]]:
        """Return the read-only final1000 inventory used by migration prep."""

        self._load_inventory()
        return [dict(task) for task in (self._inventory or {}).values()]

    def read_seed_status(self, task_id: int) -> Mapping[str, Any] | None:
        """Read a canary seal with bounded retry for scheduler HTTP 429.

        The scheduler permits only a small number of simultaneous remote-file
        reads. Exhausting that transient limit is not evidence that a canary
        failed or passed, so surface a typed busy condition for the controller
        to hold pending without terminating its watch loop.
        """

        path = f"runs/task-{int(task_id)}/seed_status.json"
        query = urllib.parse.urlencode({"path": path, "base": "remote_cwd"})
        request = urllib.request.Request(
            self.base_url + f"/api/tasks/{int(task_id)}/remote-file?{query}",
            method="GET",
        )
        for attempt in range(SEED_STATUS_BUSY_MAX_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in {404, 409}:
                    return None
                if exc.code == 429:
                    if attempt + 1 < SEED_STATUS_BUSY_MAX_ATTEMPTS:
                        time.sleep(SEED_STATUS_BUSY_RETRY_SECONDS * (attempt + 1))
                        continue
                    raise SeedStatusReadBusy(
                        "scheduler remote status busy after "
                        f"{SEED_STATUS_BUSY_MAX_ATTEMPTS} attempts: task {task_id}"
                    ) from exc
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"scheduler remote status read failed: {detail}"
                ) from exc
            if not raw.strip():
                return None
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else None
        raise AssertionError("bounded seed-status retry loop fell through")


def _seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "state_sha256"
    }
    migration = unsigned.get("rolling_migration")
    if isinstance(migration, dict):
        migration_unsigned = {
            key: copy.deepcopy(item)
            for key, item in migration.items()
            if key != "sha256"
        }
        unsigned["rolling_migration"] = {
            **migration_unsigned,
            "sha256": canonical_sha256(migration_unsigned),
        }
    return {**unsigned, "state_sha256": canonical_sha256(unsigned)}


def _advance_state(state: Mapping[str, Any]) -> dict[str, Any]:
    previous = str(state["state_sha256"])
    value = copy.deepcopy(dict(state))
    value["revision"] = int(value["revision"]) + 1
    value["parent_state_sha256"] = previous
    value.pop("state_sha256", None)
    return _seal_state(value)


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_json_bytes(state))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _all_tasks(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    waves = plan["task_waves"]
    return [
        *[dict(task) for task in waves["canaries"]],
        *[dict(task) for task in waves["ramp"]],
    ]


def _derived_resource_only_harvest_cohorts(
    plan: Mapping[str, Any], migration: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Derive the bounded catalog for the already-running phase-1 state.

    The first resource-only successor release predates the explicit catalog.
    Its old and new tasks intentionally share identical bundle bindings, so
    those bindings can be reconstructed without ambiguity from the sealed
    successor plan.  This compatibility path is prohibited for patched-bundle
    transitions where two distinct binding sets are required.
    """

    if (
        migration.get("transition_mode") != "resource_quota_only"
        or migration.get("predecessor_controller_kind") != "legacy_8c"
        or migration.get("harvest_cohorts") is not None
    ):
        raise RuntimeError("legacy harvest cohort derivation is not applicable")
    summaries = plan.get("stage_bindings")
    if not isinstance(summaries, dict) or set(summaries) != set(BY_ID):
        raise RuntimeError("legacy harvest cohort plan bindings are incomplete")

    def cohort(role: str, launch_plan_sha256: str, policy_id: str) -> dict[str, Any]:
        stage_bindings = {
            stage.stage_id: {
                key: copy.deepcopy(summaries[stage.stage_id].get(key))
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
            "schema_version": HARVEST_COHORT_SCHEMA,
            "role": role,
            "launch_plan_sha256": launch_plan_sha256,
            "resource_policy_ids": [policy_id],
            "stage_bindings": stage_bindings,
        }
        return {**unsigned, "sha256": canonical_sha256(unsigned)}

    predecessor_sha = str(migration.get("predecessor_launch_plan_sha256") or "")
    successor_sha = str(plan.get("launch_plan_sha256") or "")
    if len(predecessor_sha) != 64 or len(successor_sha) != 64:
        raise RuntimeError("legacy harvest cohort launch identity is invalid")
    return {
        "predecessor": cohort(
            "predecessor", predecessor_sha, LEGACY_RESOURCE_POLICY_ID
        ),
        "successor": cohort("successor", successor_sha, SUCCESSOR_RESOURCE_POLICY_ID),
    }


def _entry(task: Mapping[str, Any], *, origin: str | None = None) -> dict[str, Any]:
    payload = task["payload_json"]
    lane = payload["lane"]
    entry = {
        "stage_id": payload["final_goal_stage_id"],
        "bundle_id": payload["bundle_id"],
        "seed": int(payload["seed"]),
        "wave": lane["wave"],
        "dedupe_key": task["dedupe_key"],
        "task_id": None,
        "state": "planned",
    }
    if origin is not None:
        entry["origin"] = origin
        if origin == "successor":
            entry["resource_policy_id"] = SUCCESSOR_RESOURCE_POLICY_ID
    return entry


def _initial_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    tasks = _all_tasks(plan)
    entries = [_entry(task) for task in tasks]
    next_seed_by_stage = {
        stage.stage_id: max(
            int(entry["seed"])
            for entry in entries
            if entry["stage_id"] == stage.stage_id
        )
        + 1
        for stage in STAGES
    }
    unsigned = {
        "schema_version": STATE_SCHEMA,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "revision": 0,
        "parent_state_sha256": None,
        "stop_requested": False,
        "ramp_released": False,
        "canary_passed_stage_ids": [],
        "entries": entries,
        "next_seed_by_stage": next_seed_by_stage,
        "scheduler_name_prefix": TASK_NAME_PREFIX,
        "scheduler_dedupe_prefix": DEDUPE_PREFIX,
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "scheduler_submit_count": 0,
        "refill_policy": REFILL_POLICY,
        "refill_stage_cursor": 0,
        "refill_deficit_credit_by_stage": {stage.stage_id: 0 for stage in STAGES},
    }
    return _seal_state(unsigned)


def _validate_state_for_sealed_active_quotas(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    expected_active_quotas: Mapping[str, int],
) -> dict[str, Any]:
    expected_quotas = dict(expected_active_quotas)
    if expected_quotas not in (
        SUCCESSOR_ACTIVE_QUOTAS,
        HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
    ):
        raise RuntimeError("unsealed final1000 active quota policy")
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
        or not isinstance(entries, list)
        or not isinstance(next_seeds, dict)
        or set(next_seeds) != set(BY_ID)
        or set(state.get("canary_passed_stage_ids") or []) - set(BY_ID)
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
    ):
        raise RuntimeError("final1000 controller state identity/SHA mismatch")
    migration = state.get("rolling_migration")
    validated_harvest_cohorts: dict[str, dict[str, Any]] | None = None
    if migration is not None:
        migration_unsigned = (
            {key: item for key, item in migration.items() if key != "sha256"}
            if isinstance(migration, dict)
            else {}
        )
        expected_status_keys = set(BY_ID)
        if (
            not isinstance(migration, dict)
            or migration.get("schema_version") != ROLLING_MIGRATION_SCHEMA
            or migration.get("transition_mode")
            not in {"resource_quota_only", "patched_bundle"}
            or migration.get("predecessor_controller_kind")
            not in {"legacy_8c", "resource_quota_successor"}
            or migration.get("sha256") != canonical_sha256(migration_unsigned)
            or migration.get("successor_launch_plan_sha256")
            != plan.get("launch_plan_sha256")
            or migration.get("successor_resource_policy")
            != ROLLING_SUCCESSOR_RESOURCE_POLICY
            or migration.get("successor_active_quotas") != expected_quotas
            or migration.get("refill_policy") != REFILL_POLICY
            or migration.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
            or migration.get("cancellation_performed") is not False
            or migration.get("preemption_performed") is not False
            or set(migration.get("successor_canary_task_ids_by_stage") or {})
            != expected_status_keys
            or set(migration.get("successor_canary_status_by_stage") or {})
            != expected_status_keys
        ):
            raise RuntimeError("final1000 rolling migration seal mismatch")
        cohorts = migration.get("harvest_cohorts")
        if cohorts is None:
            cohorts = _derived_resource_only_harvest_cohorts(plan, migration)
        if not isinstance(cohorts, dict) or set(cohorts) != {
            "predecessor",
            "successor",
        }:
            raise RuntimeError("final1000 harvest cohort inventory mismatch")
        for role in ("predecessor", "successor"):
            cohort = cohorts.get(role)
            cohort_unsigned = (
                {key: item for key, item in cohort.items() if key != "sha256"}
                if isinstance(cohort, dict)
                else {}
            )
            bindings = (
                cohort.get("stage_bindings") if isinstance(cohort, dict) else None
            )
            policies = (
                cohort.get("resource_policy_ids") if isinstance(cohort, dict) else None
            )
            expected_plan_sha = (
                migration.get("predecessor_launch_plan_sha256")
                if role == "predecessor"
                else plan.get("launch_plan_sha256")
            )
            if (
                not isinstance(cohort, dict)
                or cohort.get("schema_version") != HARVEST_COHORT_SCHEMA
                or cohort.get("role") != role
                or cohort.get("launch_plan_sha256") != expected_plan_sha
                or cohort.get("sha256") != canonical_sha256(cohort_unsigned)
                or not isinstance(policies, list)
                or not policies
                or policies != sorted(set(policies))
                or set(policies)
                - {LEGACY_RESOURCE_POLICY_ID, SUCCESSOR_RESOURCE_POLICY_ID}
                or not isinstance(bindings, dict)
                or set(bindings) != set(BY_ID)
            ):
                raise RuntimeError("final1000 harvest cohort seal mismatch")
            for stage in STAGES:
                binding = bindings.get(stage.stage_id)
                if (
                    not isinstance(binding, dict)
                    or set(binding)
                    != {
                        "bundle_id",
                        "bundle_manifest_sha256",
                        "remote_bundle",
                        "publication_receipt_sha256",
                        "ready_sha256",
                        "stage_spec_sha256",
                    }
                    or not str(binding.get("bundle_id") or "")
                    or len(str(binding.get("bundle_manifest_sha256") or "")) != 64
                    or not str(binding.get("remote_bundle") or "").startswith("/")
                    or len(str(binding.get("publication_receipt_sha256") or "")) != 64
                    or len(str(binding.get("ready_sha256") or "")) != 64
                    or binding.get("stage_spec_sha256")
                    != stage_profile(stage)["stage_spec_sha256"]
                ):
                    raise RuntimeError("final1000 harvest stage binding seal mismatch")
        predecessor_cohort = cohorts["predecessor"]
        successor_cohort = cohorts["successor"]
        if successor_cohort["resource_policy_ids"] != [SUCCESSOR_RESOURCE_POLICY_ID]:
            raise RuntimeError("final1000 successor harvest policy drifted")
        for stage in STAGES:
            old = predecessor_cohort["stage_bindings"][stage.stage_id]
            new = successor_cohort["stage_bindings"][stage.stage_id]
            same_binding = old == new
            different_bundle = all(
                old[key] != new[key]
                for key in ("bundle_id", "bundle_manifest_sha256", "remote_bundle")
            )
            if (
                migration.get("transition_mode") == "resource_quota_only"
                and not same_binding
            ) or (
                migration.get("transition_mode") == "patched_bundle"
                and not different_bundle
            ):
                raise RuntimeError("final1000 harvest bundle transition drifted")
        validated_harvest_cohorts = copy.deepcopy(cohorts)
    dedupe: set[str] = set()
    task_ids: set[int] = set()
    seed_identities: set[tuple[str, int]] = set()
    max_seed = {stage.stage_id: stage.seed_start - 1 for stage in STAGES}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("final1000 controller entry is not an object")
        stage = BY_ID.get(str(entry.get("stage_id") or ""))
        task_id = entry.get("task_id")
        stage_id = str(entry.get("stage_id") or "")
        seed = int(entry.get("seed", -1))
        identity = (stage_id, seed)
        origin = str(entry.get("origin") or "successor")
        resource_policy_id = entry.get("resource_policy_id")
        if (
            stage is None
            or entry.get("state") not in ACTIVE_STATES | TERMINAL_STATES
            or entry.get("wave") not in {"canary", "ramp", "refill"}
            or not str(entry.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
            or not stage.seed_start
            <= int(entry.get("seed", -1))
            < stage.seed_window_end_exclusive
            or (entry.get("state") == "planned" and task_id is not None)
            or (
                entry.get("state") != "planned"
                and (
                    isinstance(task_id, bool)
                    or not isinstance(task_id, int)
                    or task_id <= 0
                )
            )
            or (migration is not None and origin not in {"predecessor", "successor"})
            or (
                migration is not None
                and resource_policy_id
                not in {LEGACY_RESOURCE_POLICY_ID, SUCCESSOR_RESOURCE_POLICY_ID}
            )
            or (migration is None and origin != "successor")
        ):
            raise RuntimeError("final1000 controller ledger entry drifted")
        if isinstance(migration, dict):
            if validated_harvest_cohorts is None:  # pragma: no cover - guarded above
                raise RuntimeError("final1000 harvest cohorts were not validated")
            cohort = validated_harvest_cohorts[origin]
            cohort_binding = cohort["stage_bindings"][stage_id]
            if (
                resource_policy_id not in cohort["resource_policy_ids"]
                or entry.get("bundle_id") != cohort_binding["bundle_id"]
                or (
                    origin == "successor"
                    and resource_policy_id != SUCCESSOR_RESOURCE_POLICY_ID
                )
            ):
                raise RuntimeError("final1000 controller ledger cohort drifted")
        if entry["dedupe_key"] in dedupe or identity in seed_identities:
            raise RuntimeError("final1000 controller ledger contains duplicate work")
        dedupe.add(entry["dedupe_key"])
        seed_identities.add(identity)
        if task_id is not None:
            if task_id in task_ids:
                raise RuntimeError("one scheduler task maps to multiple entries")
            task_ids.add(task_id)
        max_seed[stage_id] = max(max_seed[stage_id], seed)
    for stage in STAGES:
        candidate = int(next_seeds[stage.stage_id])
        if (
            not stage.seed_start <= candidate < stage.seed_window_end_exclusive
            or candidate <= max_seed[stage.stage_id]
        ):
            raise RuntimeError("final1000 next seed escaped its sealed window")
    if isinstance(migration, dict):
        expected_canary_ids = {
            stage.stage_id: sorted(
                int(entry["task_id"])
                for entry in entries
                if entry["stage_id"] == stage.stage_id
                and entry["wave"] == "canary"
                and str(entry.get("origin") or "successor") == "successor"
                and entry.get("task_id") is not None
            )
            for stage in STAGES
        }
        passed = set(state.get("canary_passed_stage_ids") or [])
        expected_canary_status = {
            stage.stage_id: (
                "remote_preflight_passed"
                if stage.stage_id in passed
                else (
                    "remote_preflight_pending"
                    if expected_canary_ids[stage.stage_id]
                    else "waiting_for_natural_terminal_gap"
                )
            )
            for stage in STAGES
        }
        if (
            migration.get("successor_canary_task_ids_by_stage") != expected_canary_ids
            or migration.get("successor_canary_status_by_stage")
            != expected_canary_status
            or migration.get("predecessor_entry_count")
            != sum(
                str(entry.get("origin") or "successor") == "predecessor"
                for entry in entries
            )
            or migration.get("next_seed_by_stage")
            != {stage.stage_id: int(next_seeds[stage.stage_id]) for stage in STAGES}
        ):
            raise RuntimeError("final1000 rolling migration ledger seal mismatch")
    return copy.deepcopy(dict(state))


def _validate_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a state owned by this release's 200/160/90/50 controller."""

    return _validate_state_for_sealed_active_quotas(
        state,
        plan,
        expected_active_quotas=SUCCESSOR_ACTIVE_QUOTAS,
    )


def _validate_historical_resource_quota_successor_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the one sealed 160/140/120/80 phase-1 predecessor."""

    return _validate_state_for_sealed_active_quotas(
        state,
        plan,
        expected_active_quotas=HISTORICAL_RESOURCE_QUOTA_SUCCESSOR_ACTIVE_QUOTAS,
    )


def _verify_live_ready(
    plan: Mapping[str, Any], *, apply: bool, ready_probe: ReadyProbe | None
) -> None:
    bindings = plan.get("stage_bindings") or {}
    if set(bindings) != set(BY_ID):
        raise RuntimeError("final1000 launch plan lacks four bundle bindings")
    if not apply:
        return
    if ready_probe is None:
        raise RuntimeError("--apply requires a live READY probe")
    for stage in STAGES:
        binding = bindings[stage.stage_id]
        expected = binding.get("ready")
        if (
            not isinstance(expected, dict)
            or binding.get("ready_sha256") != canonical_sha256(expected)
            or ready_probe.read_ready(str(binding.get("remote_bundle") or ""))
            != expected
        ):
            raise RuntimeError(f"{stage.stage_id} live READY identity mismatch")


def _task_templates(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    templates = {}
    for task in plan["task_waves"]["canaries"]:
        validated = validate_task(task, expected_wave="canary")
        stage_id = validated["payload_json"]["final_goal_stage_id"]
        if stage_id in templates:
            raise RuntimeError("final1000 stage has duplicate canary templates")
        templates[stage_id] = validated
    if set(templates) != set(BY_ID):
        raise RuntimeError("final1000 canary template set is incomplete")
    return templates


def _task_index(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["dedupe_key"]: task for task in _all_tasks(plan)}


def _refill_task(
    template: Mapping[str, Any], *, stage_id: str, seed: int, wave: str = "refill"
) -> dict[str, Any]:
    stage = BY_ID[stage_id]
    base = copy.deepcopy(dict(template))
    payload = base["payload_json"]
    payload["seed"] = int(seed)
    payload["lane"]["seed"] = int(seed)
    payload["lane"]["wave"] = wave
    return build_stage_task(base, stage=stage, wave=wave)


def _task_for_entry(
    entry: Mapping[str, Any],
    *,
    plan_index: Mapping[str, Mapping[str, Any]],
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    existing = plan_index.get(str(entry["dedupe_key"]))
    task = (
        dict(existing)
        if existing is not None
        else _refill_task(
            templates[str(entry["stage_id"])],
            stage_id=str(entry["stage_id"]),
            seed=int(entry["seed"]),
            wave=str(entry["wave"]),
        )
    )
    validate_task(task, expected_stage=BY_ID[str(entry["stage_id"])])
    if task["dedupe_key"] != entry["dedupe_key"]:
        raise RuntimeError("controller entry cannot reproduce its sealed task")
    return task


def _append_refill(
    state: dict[str, Any],
    *,
    templates: Mapping[str, Mapping[str, Any]],
    stage_id: str,
    wave: str = "refill",
) -> dict[str, Any]:
    stage = BY_ID[stage_id]
    seed = int(state["next_seed_by_stage"][stage_id])
    if not stage.seed_start <= seed < stage.seed_window_end_exclusive:
        raise RuntimeError(f"{stage_id} seed window exhausted")
    task = _refill_task(templates[stage_id], stage_id=stage_id, seed=seed, wave=wave)
    entry = _entry(
        task,
        origin="successor" if state.get("rolling_migration") is not None else None,
    )
    state["entries"].append(entry)
    state["next_seed_by_stage"][stage_id] = seed + 1
    return entry


def _scheduler_state(value: Any) -> str | None:
    return {
        "queued": "queued",
        "attaching": "queued",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(str(value or "").lower())


def _task_id(value: Mapping[str, Any]) -> int:
    candidate = value.get("task_id") or value.get("id")
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
        raise RuntimeError("scheduler response has no positive task id")
    return candidate


def _submit_planned(
    state: dict[str, Any],
    *,
    entries: Sequence[dict[str, Any]],
    plan_index: Mapping[str, Mapping[str, Any]],
    templates: Mapping[str, Mapping[str, Any]],
    scheduler: SchedulerClient,
) -> tuple[int, int]:
    submitted = 0
    reconciled = 0
    for entry in entries:
        if entry["state"] != "planned":
            continue
        if str(entry.get("origin") or "successor") != "successor":
            raise RuntimeError("predecessor ledger entry cannot be submitted")
        task = _task_for_entry(entry, plan_index=plan_index, templates=templates)
        existing = scheduler.find_task_by_dedupe(entry["dedupe_key"])
        if existing is None:
            existing = scheduler.submit_task(task)
            submitted += 1
        else:
            reconciled += 1
        if (
            str(existing.get("dedupe_key") or entry["dedupe_key"])
            != entry["dedupe_key"]
        ):
            raise RuntimeError("scheduler response changed final1000 dedupe identity")
        entry["task_id"] = _task_id(existing)
        entry["state"] = _scheduler_state(existing.get("status")) or "submitted"
    state["scheduler_submit_count"] = int(state["scheduler_submit_count"]) + submitted
    return submitted, reconciled


def _reconcile(state: dict[str, Any], scheduler: SchedulerClient) -> int:
    changed = 0
    for entry in state["entries"]:
        if entry["task_id"] is None or entry["state"] in TERMINAL_STATES:
            continue
        observed = scheduler.get_task(int(entry["task_id"]))
        if observed is None:
            continue
        if (
            str(observed.get("dedupe_key") or entry["dedupe_key"])
            != entry["dedupe_key"]
        ):
            raise RuntimeError("scheduler task changed final1000 dedupe identity")
        mapped = _scheduler_state(observed.get("status"))
        if mapped is not None and mapped != entry["state"]:
            entry["state"] = mapped
            changed += 1
    return changed


def _observe_canary_gates(
    state: dict[str, Any], scheduler: SchedulerClient
) -> tuple[set[str], list[str]]:
    passed = set(state["canary_passed_stage_ids"])
    reasons = []
    for stage in STAGES:
        if stage.stage_id in passed:
            continue
        candidates = [
            entry
            for entry in state["entries"]
            if entry["stage_id"] == stage.stage_id
            and entry["wave"] == "canary"
            and str(entry.get("origin") or "successor") == "successor"
            and entry["task_id"] is not None
        ]
        stage_passed = False
        status_read_busy = False
        for entry in candidates:
            try:
                status = scheduler.read_seed_status(int(entry["task_id"]))
            except SeedStatusReadBusy:
                status_read_busy = True
                continue
            if status is None:
                continue
            if (
                status.get("seed") == entry["seed"]
                and status.get("bundle_id") == entry["bundle_id"]
                and status.get("ramp_gate_passed") is True
                and status.get("aedt_used") is False
                and status.get("fea_submission_performed") is False
            ):
                stage_passed = True
                passed.add(stage.stage_id)
                break
        if not stage_passed:
            reason = (
                "remote_preflight_status_read_busy"
                if status_read_busy
                else "remote_preflight_pending"
            )
            reasons.append(f"{stage.stage_id}:{reason}")
    state["canary_passed_stage_ids"] = sorted(passed)
    return passed, reasons


def _refresh_migration_observation(state: dict[str, Any]) -> None:
    migration = state.get("rolling_migration")
    if not isinstance(migration, dict):
        return
    passed = set(state.get("canary_passed_stage_ids") or [])
    ids: dict[str, list[int]] = {}
    statuses: dict[str, str] = {}
    for stage in STAGES:
        candidates = [
            entry
            for entry in state["entries"]
            if entry["stage_id"] == stage.stage_id
            and entry["wave"] == "canary"
            and str(entry.get("origin") or "successor") == "successor"
            and entry.get("task_id") is not None
        ]
        ids[stage.stage_id] = sorted(int(entry["task_id"]) for entry in candidates)
        if stage.stage_id in passed:
            statuses[stage.stage_id] = "remote_preflight_passed"
        elif candidates:
            statuses[stage.stage_id] = "remote_preflight_pending"
        else:
            statuses[stage.stage_id] = "waiting_for_natural_terminal_gap"
    migration["successor_canary_task_ids_by_stage"] = ids
    migration["successor_canary_status_by_stage"] = statuses
    migration["next_seed_by_stage"] = {
        stage.stage_id: int(state["next_seed_by_stage"][stage.stage_id])
        for stage in STAGES
    }


def _active_count(state: Mapping[str, Any], stage_id: str | None = None) -> int:
    return sum(
        entry["state"] in ACTIVE_STATES
        and (stage_id is None or entry["stage_id"] == stage_id)
        for entry in state["entries"]
    )


def _weighted_deficit_refill_order(
    state: dict[str, Any],
    *,
    slots: int,
    reserve_missing_migration_canaries: bool,
) -> list[str]:
    """Choose a deterministic interleaved stage sequence for refill slots.

    Smooth weighted round-robin operates only on stages below the successor
    target.  During migration, one natural gap is first reserved for each
    stage that has no live successor canary, even when that predecessor stage
    is temporarily above its successor quota.  No task is cancelled to force
    convergence; excess drains naturally and its slots move to deficits.
    """

    if slots < 0:
        raise ValueError("refill slots cannot be negative")
    stage_ids = [stage.stage_id for stage in STAGES]
    current = {stage_id: _active_count(state, stage_id) for stage_id in stage_ids}
    credits = {
        stage_id: float(state["refill_deficit_credit_by_stage"][stage_id])
        for stage_id in stage_ids
    }
    cursor = int(state["refill_stage_cursor"])
    selected: list[str] = []

    if reserve_missing_migration_canaries:
        passed = set(state.get("canary_passed_stage_ids") or [])
        missing = {
            stage_id
            for stage_id in stage_ids
            if stage_id not in passed
            and not any(
                entry["stage_id"] == stage_id
                and entry["wave"] == "canary"
                and str(entry.get("origin") or "successor") == "successor"
                and entry["state"] in ACTIVE_STATES
                for entry in state["entries"]
            )
        }
        while slots > 0 and missing:
            offsets = range(len(stage_ids))
            stage_id = next(
                stage_ids[(cursor + offset) % len(stage_ids)]
                for offset in offsets
                if stage_ids[(cursor + offset) % len(stage_ids)] in missing
            )
            selected.append(stage_id)
            current[stage_id] += 1
            slots -= 1
            missing.remove(stage_id)
            cursor = (stage_ids.index(stage_id) + 1) % len(stage_ids)

    while slots > 0:
        eligible = [
            stage_id
            for stage_id in stage_ids
            if current[stage_id] < SUCCESSOR_ACTIVE_QUOTAS[stage_id]
        ]
        if not eligible:
            raise RuntimeError("no successor stage deficit exists for refill slot")
        for stage_id in eligible:
            credits[stage_id] += SUCCESSOR_ACTIVE_QUOTAS[stage_id]
        total_weight = sum(SUCCESSOR_ACTIVE_QUOTAS[stage_id] for stage_id in eligible)
        cyclic_rank = {
            stage_ids[(cursor + offset) % len(stage_ids)]: offset
            for offset in range(len(stage_ids))
        }
        stage_id = min(
            eligible,
            key=lambda candidate: (-credits[candidate], cyclic_rank[candidate]),
        )
        credits[stage_id] -= total_weight
        selected.append(stage_id)
        current[stage_id] += 1
        slots -= 1
        cursor = (stage_ids.index(stage_id) + 1) % len(stage_ids)

    state["refill_stage_cursor"] = cursor
    state["refill_deficit_credit_by_stage"] = {
        stage_id: credits[stage_id] for stage_id in stage_ids
    }
    return selected


def control_once(
    plan_path: Path,
    *,
    state_path: Path,
    apply: bool = False,
    scheduler: SchedulerClient | None = None,
    ready_probe: ReadyProbe | None = None,
    request_stop: bool = False,
    stop_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    plan = validate_launch_plan(_read_json(plan_path.resolve(strict=True)))
    _verify_live_ready(plan, apply=apply, ready_probe=ready_probe)
    if apply and scheduler is None:
        raise RuntimeError("--apply requires an explicit scheduler client")
    if state_path.is_file():
        state = _validate_state(_read_json(state_path), plan)
        created = False
    else:
        state = _initial_state(plan)
        created = True
    plan_index = _task_index(plan)
    templates = _task_templates(plan)
    actions: list[dict[str, Any]] = []
    writes = 0

    def persist() -> None:
        nonlocal state, writes
        _refresh_migration_observation(state)
        state = _advance_state(state)
        if apply:
            _write_state(state_path, state)
            writes += 1

    if apply and created:
        _write_state(state_path, state)
        writes += 1

    stop_requested = bool(
        state["stop_requested"]
        or request_stop
        or (stop_probe is not None and stop_probe())
    )
    if stop_requested:
        if not state["stop_requested"]:
            state["stop_requested"] = True
            persist()
        actions.append(
            {
                "action": "stop",
                "reason": "explicit_stop_requested",
                "running_tasks_cancelled": False,
            }
        )
        return _result(state, actions, apply, writes, scheduler)

    rolling_migration = state.get("rolling_migration") is not None
    canary_entries = [
        entry
        for entry in state["entries"]
        if entry["wave"] == "canary"
        and (
            not rolling_migration
            or str(entry.get("origin") or "successor") == "successor"
        )
    ]
    if not apply:
        actions.append(
            {
                "action": "would_submit_canaries",
                "count": sum(entry["state"] == "planned" for entry in canary_entries),
            }
        )
        return _result(state, actions, False, 0, None)

    assert scheduler is not None  # guarded above
    reconciled_changes = _reconcile(state, scheduler)
    submitted, reconciled = _submit_planned(
        state,
        entries=canary_entries,
        plan_index=plan_index,
        templates=templates,
        scheduler=scheduler,
    )
    if reconciled_changes or submitted or reconciled:
        persist()
    actions.append(
        {
            "action": "canary_submission",
            "submitted": submitted,
            "reconciled": reconciled,
        }
    )

    if not state["ramp_released"]:
        prior_passed = set(state["canary_passed_stage_ids"])
        passed, reasons = _observe_canary_gates(state, scheduler)
        replacement_entries = []
        replacement_stage_order: list[str] = []
        if rolling_migration:
            # Preserve every predecessor task and transfer only naturally
            # vacated slots.  The old 64/96/128/212 distribution is allowed
            # to be temporarily over/under the successor 200/160/90/50
            # targets; weighted deficit refill converges without cancellation.
            replacement_stage_order = _weighted_deficit_refill_order(
                state,
                slots=TOTAL_ACTIVE_QUOTA - _active_count(state),
                reserve_missing_migration_canaries=True,
            )
            for stage_id in replacement_stage_order:
                wave = "refill" if stage_id in passed else "canary"
                replacement_entries.append(
                    _append_refill(
                        state,
                        templates=templates,
                        stage_id=stage_id,
                        wave=wave,
                    )
                )
        else:
            for stage in STAGES:
                if stage.stage_id in passed:
                    continue
                stage_canaries = [
                    entry
                    for entry in state["entries"]
                    if entry["stage_id"] == stage.stage_id and entry["wave"] == "canary"
                ]
                if stage_canaries and all(
                    entry["state"] in TERMINAL_STATES for entry in stage_canaries
                ):
                    replacement_entries.append(
                        _append_refill(
                            state,
                            templates=templates,
                            stage_id=stage.stage_id,
                            wave="canary",
                        )
                    )
        replacement_submitted = 0
        replacement_reconciled = 0
        if replacement_entries:
            replacement_submitted, replacement_reconciled = _submit_planned(
                state,
                entries=replacement_entries,
                plan_index=plan_index,
                templates=templates,
                scheduler=scheduler,
            )
            persist()
        if len(passed) != len(STAGES):
            # Persist newly observed successful gates even without replacement.
            if passed != prior_passed and not replacement_entries:
                persist()
            if rolling_migration and _active_count(state) != TOTAL_ACTIVE_QUOTA:
                raise RuntimeError(
                    "rolling migration failed to preserve exact active target"
                )
            actions.append(
                {
                    "action": (
                        "rolling_canary_held" if rolling_migration else "ramp_held"
                    ),
                    "passed_stage_count": len(passed),
                    "reasons": reasons,
                    "replacement_submitted": replacement_submitted,
                    "replacement_reconciled": replacement_reconciled,
                    "replacement_stage_order": replacement_stage_order,
                }
            )
            return _result(state, actions, True, writes, scheduler)
        state["ramp_released"] = True
        persist()
        actions.append(
            {
                "action": (
                    "rolling_successor_released"
                    if rolling_migration
                    else "ramp_released"
                ),
                "count": 0 if rolling_migration else 496,
            }
        )

    planned_entries = [
        entry for entry in state["entries"] if entry["state"] == "planned"
    ]
    submitted, reconciled = _submit_planned(
        state,
        entries=planned_entries,
        plan_index=plan_index,
        templates=templates,
        scheduler=scheduler,
    )
    if submitted or reconciled:
        persist()
    actions.append(
        {
            "action": "ramp_or_reserved_submission",
            "submitted": submitted,
            "reconciled": reconciled,
        }
    )

    changed = _reconcile(state, scheduler)
    refill_entries = []
    refill_stage_order = _weighted_deficit_refill_order(
        state,
        slots=TOTAL_ACTIVE_QUOTA - _active_count(state),
        reserve_missing_migration_canaries=False,
    )
    for stage_id in refill_stage_order:
        refill_entries.append(
            _append_refill(state, templates=templates, stage_id=stage_id)
        )
    refill_submitted = 0
    refill_reconciled = 0
    if refill_entries:
        refill_submitted, refill_reconciled = _submit_planned(
            state,
            entries=refill_entries,
            plan_index=plan_index,
            templates=templates,
            scheduler=scheduler,
        )
    if changed or refill_entries or refill_submitted or refill_reconciled:
        persist()
    actions.append(
        {
            "action": "refill",
            "reserved": len(refill_entries),
            "submitted": refill_submitted,
            "reconciled": refill_reconciled,
            "stage_order": refill_stage_order,
        }
    )
    if _active_count(state) != TOTAL_ACTIVE_QUOTA:
        raise RuntimeError("final1000 controller failed to restore active target")
    return _result(state, actions, True, writes, scheduler)


def _result(
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    apply: bool,
    state_writes: int,
    scheduler: SchedulerClient | None,
) -> dict[str, Any]:
    stage_active = {
        stage.stage_id: _active_count(state, stage.stage_id) for stage in STAGES
    }
    migration = state.get("rolling_migration")
    return {
        "schema_version": RESULT_SCHEMA,
        "apply": bool(apply),
        "state_revision": int(state["revision"]),
        "state_sha256": state["state_sha256"],
        "state_writes": int(state_writes),
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "scheduler_name_prefix": TASK_NAME_PREFIX,
        "scheduler_dedupe_prefix": DEDUPE_PREFIX,
        "stop_requested": bool(state["stop_requested"]),
        "ramp_released": bool(state["ramp_released"]),
        "canary_passed_stage_ids": list(state["canary_passed_stage_ids"]),
        "logical_active_target": TOTAL_ACTIVE_QUOTA,
        "active_count": sum(stage_active.values()),
        "active_count_by_stage": stage_active,
        "terminal_count": sum(
            entry["state"] in TERMINAL_STATES for entry in state["entries"]
        ),
        "entry_count": len(state["entries"]),
        "actions": [dict(action) for action in actions],
        "rolling_migration": isinstance(migration, dict),
        "successor_canary_task_ids_by_stage": (
            copy.deepcopy(migration.get("successor_canary_task_ids_by_stage"))
            if isinstance(migration, dict)
            else None
        ),
        "successor_canary_status_by_stage": (
            copy.deepcopy(migration.get("successor_canary_status_by_stage"))
            if isinstance(migration, dict)
            else None
        ),
        "cancellation_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
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
    if args.watch and not args.apply:
        raise RuntimeError("--watch requires --apply")
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    scheduler = SchedulerApiClient(args.scheduler_url) if args.apply else None

    def stop_probe() -> bool:
        return bool(args.stop_file is not None and args.stop_file.is_file())

    if args.apply:
        with scheduler_publication_transport(
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            account_name=args.publication_account,
        ) as transport:
            ready_probe = TransportReadyProbe(transport)
            while True:
                value = control_once(
                    args.plan,
                    state_path=args.state,
                    apply=True,
                    scheduler=scheduler,
                    ready_probe=ready_probe,
                    request_stop=args.request_stop,
                    stop_probe=stop_probe,
                )
                print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
                if not args.watch or value["stop_requested"]:
                    break
                time.sleep(args.poll_seconds)
    else:
        value = control_once(
            args.plan,
            state_path=args.state,
            apply=False,
            scheduler=None,
            ready_probe=None,
            request_stop=args.request_stop,
            stop_probe=stop_probe,
        )
        print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
