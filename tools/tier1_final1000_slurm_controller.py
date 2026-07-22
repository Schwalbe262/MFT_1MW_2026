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
CHAINED_ROLLING_MIGRATION_SCHEMA = "mft-tier1-final1000-rolling-migration-v2"
HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v1"
CHAINED_HARVEST_COHORT_SCHEMA = "mft-tier1-final1000-harvest-cohort-v2"
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
CHAINED_CANARY_GAP_POLICY = {
    "required_stage_ids": sorted(BY_ID),
    "maximum_canary_tasks_per_stage": 1,
    "replacement_canaries_allowed": False,
    "additional_natural_gaps_before_all_passed": "held_unfilled",
    "active_target_before_all_passed": "temporarily_at_or_below_500",
    "post_pass_refill": "weighted_deficit_to_exact_500",
}
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
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})
REFILL_POLICY = "smooth-weighted-deficit-round-robin-v1"
SEED_STATUS_BUSY_MAX_ATTEMPTS = 3
SEED_STATUS_BUSY_RETRY_SECONDS = 0.25
INVENTORY_PAGE_SIZE = 10_000
INVENTORY_BUSY_MAX_ATTEMPTS = 3
INVENTORY_BUSY_RETRY_SECONDS = 0.25
INVENTORY_SNAPSHOT_SCHEMA = "mft-tier1-final1000-inventory-snapshot-v1"
EXTERNAL_CANARY_ATTESTATION_SCHEMA = (
    "mft-tier1-final1000-external-canary-attestation-v1"
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_external_canary_attestation(
    value: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    stages = value.get("stage_attestations")
    if (
        set(value)
        != {
            "schema_version",
            "source_driver_state_schema_version",
            "source_driver_state_sha256",
            "source_driver_state_revision",
            "source_driver_launch_plan_sha256",
            "source_driver_phase",
            "source_monitoring_capability_sha256",
            "successor_launch_plan_sha256",
            "gate_phase",
            "stage_attestations",
            "scheduler_mutation_endpoints",
            "scheduler_post_count_delta",
            "cancellation_performed",
            "preemption_performed",
            "sha256",
        }
        or value.get("schema_version") != EXTERNAL_CANARY_ATTESTATION_SCHEMA
        or value.get("sha256") != canonical_sha256(unsigned)
        or value.get("source_driver_state_schema_version")
        != "mft-tier1-final1000-multiseed-production-driver-v2"
        or not _is_sha256(value.get("source_driver_state_sha256"))
        or isinstance(value.get("source_driver_state_revision"), bool)
        or not isinstance(value.get("source_driver_state_revision"), int)
        or int(value["source_driver_state_revision"]) < 0
        or value.get("source_driver_launch_plan_sha256")
        != plan.get("launch_plan_sha256")
        or value.get("successor_launch_plan_sha256")
        != plan.get("launch_plan_sha256")
        or value.get("source_driver_phase")
        not in {"batch4", "awaiting_predecessor_stop", "cutover_prepared", "refill"}
        or not _is_sha256(value.get("source_monitoring_capability_sha256"))
        or value.get("gate_phase") != "batch1"
        or not isinstance(stages, dict)
        or set(stages) != set(BY_ID)
        or value.get("scheduler_mutation_endpoints") != []
        or value.get("scheduler_post_count_delta") != 0
        or value.get("cancellation_performed") is not False
        or value.get("preemption_performed") is not False
    ):
        raise RuntimeError("external successor canary attestation seal mismatch")
    task_ids: set[int] = set()
    for stage in STAGES:
        item = stages.get(stage.stage_id)
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "task_id",
                "parent_dedupe_key",
                "task_sha256",
                "bundle_id",
                "seeds",
                "scheduler_status",
                "batch_manifest_sha256",
                "task_status_sha256",
                "child_receipt_sha256s",
                "remote_evidence_sha256",
            }
            or isinstance(item.get("task_id"), bool)
            or not isinstance(item.get("task_id"), int)
            or int(item["task_id"]) <= 0
            or int(item["task_id"]) in task_ids
            or not str(item.get("parent_dedupe_key") or "").startswith(
                "mft-tier1-final1000:lane:"
            )
            or not _is_sha256(item.get("task_sha256"))
            or item.get("bundle_id")
            != plan["stage_bindings"][stage.stage_id]["bundle_id"]
            or item.get("seeds") != [stage.seed_window_end_exclusive - 5]
            or item.get("scheduler_status") != "completed"
            or not _is_sha256(item.get("batch_manifest_sha256"))
            or not _is_sha256(item.get("task_status_sha256"))
            or not isinstance(item.get("child_receipt_sha256s"), list)
            or len(item["child_receipt_sha256s"]) != 1
            or not all(_is_sha256(digest) for digest in item["child_receipt_sha256s"])
            or not _is_sha256(item.get("remote_evidence_sha256"))
        ):
            raise RuntimeError("external successor canary stage attestation drifted")
        task_ids.add(int(item["task_id"]))


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

    def list_namespace_tasks(self) -> Sequence[Mapping[str, Any]]: ...

    def read_seed_status(self, task_id: int) -> Mapping[str, Any] | None: ...


class ReadyProbe(Protocol):
    read_count: int

    def read_ready(self, remote_bundle: str) -> Mapping[str, Any] | None: ...


class SchedulerApiClient(Current7SchedulerApiClient):
    """Bounded final-goal client for the long-running watch controller."""

    def _load_inventory(self) -> None:
        query = urllib.parse.urlencode(
            {
                "name_prefix": TASK_NAME_PREFIX,
                "sort_by": "id",
                "sort_order": "desc",
                "limit": 10_000,
            }
        )
        value = self._request(f"/api/tasks?{query}")
        if not isinstance(value, list):
            raise RuntimeError("final1000 scheduler inventory is not a list")
        inventory: dict[str, Mapping[str, Any]] = {}
        task_ids: dict[int, str] = {}
        for task in value:
            if not isinstance(task, dict):
                continue
            name = str(task.get("name") or "")
            dedupe = str(task.get("dedupe_key") or "")
            if not name.startswith(TASK_NAME_PREFIX):
                continue
            if not dedupe.startswith(DEDUPE_PREFIX):
                raise RuntimeError("final1000 namespace task has foreign dedupe key")
            task_id = _task_id(task)
            if task_id in task_ids and task_ids[task_id] != dedupe:
                raise RuntimeError(
                    "scheduler contains one final1000 task id with multiple dedupes"
                )
            task_ids[task_id] = dedupe
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
        """Return the bounded namespace window used by watch reconciliation."""

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


class CompleteInventorySchedulerApiClient(SchedulerApiClient):
    """One-shot, full namespace reader restricted to migration preparation."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        super().__init__(base_url, timeout=timeout)
        self.inventory_get_count = 0
        self.inventory_snapshot_receipt: dict[str, Any] | None = None

    def _request_inventory_page(self, path: str) -> Mapping[str, Any]:
        """Read one inventory page with bounded retry for HTTP 429 only."""

        request = urllib.request.Request(self.base_url + path, method="GET")
        for attempt in range(INVENTORY_BUSY_MAX_ATTEMPTS):
            self.inventory_get_count += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    if attempt + 1 < INVENTORY_BUSY_MAX_ATTEMPTS:
                        time.sleep(INVENTORY_BUSY_RETRY_SECONDS * (attempt + 1))
                        continue
                    raise RuntimeError(
                        "scheduler inventory remained rate-limited after "
                        f"{INVENTORY_BUSY_MAX_ATTEMPTS} attempts"
                    ) from exc
                raise RuntimeError(
                    f"scheduler inventory GET {path} failed with HTTP "
                    f"{exc.code}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"scheduler inventory GET {path} failed: {exc.reason}"
                ) from exc
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("scheduler inventory page is not valid JSON") from exc
            if not isinstance(value, dict):
                raise RuntimeError("scheduler paged inventory is not an object")
            return value
        raise AssertionError("bounded inventory retry loop fell through")

    @staticmethod
    def _inventory_query(*, page: int, page_size: int, before_id: int) -> str:
        query = urllib.parse.urlencode(
            {
                "name_prefix": TASK_NAME_PREFIX,
                "sort_by": "id",
                "sort_order": "desc",
                "paged": "true",
                "page": page,
                "page_size": page_size,
                "before_id": before_id,
            }
        )
        return f"/api/tasks?{query}"

    @staticmethod
    def _validate_inventory_page(
        value: Mapping[str, Any],
        *,
        page: int,
        page_size: int,
        before_id: int,
        expected_total: int | None,
        expected_page_count: int | None,
        expected_server_revision: Any,
        check_server_revision: bool,
    ) -> tuple[list[Mapping[str, Any]], int, int, Any]:
        def strict_int(name: str) -> int:
            candidate = value.get(name)
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, int)
                or candidate < 0
            ):
                raise RuntimeError(
                    f"scheduler inventory {name} is not a non-negative integer"
                )
            return candidate

        filtered_total = strict_int("filtered_total")
        observed_page = strict_int("page")
        observed_page_size = strict_int("page_size")
        page_count = strict_int("page_count")
        calculated_page_count = max(1, math.ceil(filtered_total / page_size))
        if (
            observed_page != page
            or observed_page_size != page_size
            or page_count != calculated_page_count
            or value.get("has_previous") is not (page > 1)
            or value.get("has_next") is not (page < page_count)
            or value.get("sort_by") != "id"
            or value.get("sort_order") != "desc"
        ):
            raise RuntimeError("scheduler inventory page metadata mismatch")
        filters = value.get("filters")
        if filters != {
            "project": "",
            "name_prefix": TASK_NAME_PREFIX,
            "name_contains": "",
            "status": None,
            "before_id": before_id,
        }:
            raise RuntimeError("scheduler inventory page filters changed")
        if (
            expected_total is not None
            and (
                filtered_total != expected_total
                or (
                    expected_page_count is not None
                    and page_count != expected_page_count
                )
            )
        ):
            raise RuntimeError("scheduler inventory snapshot metadata changed")
        server_revision = value.get("snapshot_revision")
        if check_server_revision and server_revision != expected_server_revision:
            raise RuntimeError("scheduler inventory snapshot revision changed")
        items = value.get("items")
        if not isinstance(items, list):
            raise RuntimeError("scheduler inventory page items are not a list")
        expected_items = min(
            page_size,
            max(0, filtered_total - ((page - 1) * page_size)),
        )
        if len(items) != expected_items:
            raise RuntimeError("scheduler inventory page cardinality mismatch")
        if not all(isinstance(item, dict) for item in items):
            raise RuntimeError("scheduler inventory page contains a non-object row")
        return items, filtered_total, page_count, server_revision

    def _load_inventory(self) -> None:
        # Establish a high-watermark first.  Authoritative page reads are then
        # pinned below it so concurrent inserts cannot shift OFFSET pages.
        anchor = self._request_inventory_page(
            self._inventory_query(page=1, page_size=1, before_id=0)
        )
        anchor_items, anchor_total, _anchor_page_count, _anchor_revision = (
            self._validate_inventory_page(
                anchor,
                page=1,
                page_size=1,
                before_id=0,
                expected_total=None,
                expected_page_count=None,
                expected_server_revision=None,
                check_server_revision=False,
            )
        )
        if anchor_items:
            high_watermark_task_id = _task_id(anchor_items[0])
        else:
            high_watermark_task_id = 0
        before_id = high_watermark_task_id + 1

        rows: list[Mapping[str, Any]] = []
        seen_task_ids: set[int] = set()
        previous_task_id: int | None = None
        expected_page_count: int | None = None
        expected_server_revision: Any = None
        page = 1
        while expected_page_count is None or page <= expected_page_count:
            value = self._request_inventory_page(
                self._inventory_query(
                    page=page,
                    page_size=INVENTORY_PAGE_SIZE,
                    before_id=before_id,
                )
            )
            page_items, _filtered_total, page_count, server_revision = (
                self._validate_inventory_page(
                    value,
                    page=page,
                    page_size=INVENTORY_PAGE_SIZE,
                    before_id=before_id,
                    expected_total=anchor_total,
                    expected_page_count=expected_page_count,
                    expected_server_revision=expected_server_revision,
                    check_server_revision=page > 1,
                )
            )
            if expected_page_count is None:
                expected_page_count = page_count
                expected_server_revision = server_revision
            for task in page_items:
                task_id = _task_id(task)
                if task_id >= before_id:
                    raise RuntimeError(
                        "scheduler inventory escaped its fixed high-watermark"
                    )
                if task_id in seen_task_ids:
                    raise RuntimeError("scheduler inventory repeats a task id")
                if previous_task_id is not None and task_id >= previous_task_id:
                    raise RuntimeError(
                        "scheduler inventory task ids are not strictly descending"
                    )
                seen_task_ids.add(task_id)
                previous_task_id = task_id
                rows.append(task)
            page += 1

        if len(rows) != anchor_total:
            raise RuntimeError("scheduler inventory snapshot is missing task ids")
        if rows and _task_id(rows[0]) != high_watermark_task_id:
            raise RuntimeError("scheduler inventory high-watermark identity changed")

        inventory: dict[str, Mapping[str, Any]] = {}
        task_ids: dict[int, str] = {}
        for task in rows:
            name = str(task.get("name") or "")
            dedupe = str(task.get("dedupe_key") or "")
            if not name.startswith(TASK_NAME_PREFIX):
                raise RuntimeError("final1000 inventory escaped its name namespace")
            if not dedupe.startswith(DEDUPE_PREFIX):
                raise RuntimeError("final1000 namespace task has foreign dedupe key")
            task_id = _task_id(task)
            if task_id in task_ids and task_ids[task_id] != dedupe:
                raise RuntimeError(
                    "scheduler contains one final1000 task id with multiple dedupes"
                )
            task_ids[task_id] = dedupe
            if dedupe in inventory and int(inventory[dedupe]["id"]) != int(task["id"]):
                raise RuntimeError(
                    f"scheduler contains duplicate final1000 dedupe: {dedupe}"
                )
            inventory[dedupe] = task
        snapshot_unsigned = {
            "schema_version": INVENTORY_SNAPSHOT_SCHEMA,
            "high_watermark_task_id": high_watermark_task_id,
            "before_id": before_id,
            "filtered_total": anchor_total,
            "page_size": INVENTORY_PAGE_SIZE,
            "page_count": expected_page_count,
            "task_ids_sha256": canonical_sha256([_task_id(task) for task in rows]),
            "server_snapshot_revision": expected_server_revision,
        }
        self.inventory_snapshot_receipt = {
            **snapshot_unsigned,
            "sha256": canonical_sha256(snapshot_unsigned),
        }
        self._inventory = inventory

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
    allow_chained_shadow: bool = False,
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
            or migration.get("schema_version")
            not in {ROLLING_MIGRATION_SCHEMA, CHAINED_ROLLING_MIGRATION_SCHEMA}
            or migration.get("transition_mode")
            not in {"resource_quota_only", "patched_bundle"}
            or migration.get("predecessor_controller_kind")
            not in {
                "legacy_8c",
                "resource_quota_successor",
                "chained_patched_successor",
            }
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
        external_attestation = migration.get("external_canary_attestation")
        if external_attestation is not None:
            if not isinstance(external_attestation, Mapping):
                raise RuntimeError(
                    "external successor canary attestation is not an object"
                )
            _validate_external_canary_attestation(external_attestation, plan)
        if migration.get("schema_version") == CHAINED_ROLLING_MIGRATION_SCHEMA:
            cutover_ready = (
                migration.get("predecessor_stop_observed") is True
                and migration.get("shadow_only") is False
            )
            authenticated_shadow = (
                allow_chained_shadow
                and migration.get("predecessor_stop_observed") is False
                and migration.get("shadow_only") is True
            )
            if (
                migration.get("predecessor_controller_kind")
                != "chained_patched_successor"
                or migration.get("transition_mode") != "patched_bundle"
                or not (cutover_ready or authenticated_shadow)
            ):
                raise RuntimeError("final1000 chained migration is not cutover-ready")
        cohorts = migration.get("harvest_cohorts")
        chained_catalog = (
            migration.get("schema_version") == CHAINED_ROLLING_MIGRATION_SCHEMA
        )
        if cohorts is None:
            cohorts = _derived_resource_only_harvest_cohorts(plan, migration)
        if not isinstance(cohorts, dict):
            raise RuntimeError("final1000 harvest cohort inventory mismatch")
        if chained_catalog:
            predecessor_cohort_ids = migration.get(
                "predecessor_harvest_cohort_ids"
            )
            successor_cohort_id = migration.get("successor_harvest_cohort_id")
            if (
                not isinstance(predecessor_cohort_ids, list)
                or predecessor_cohort_ids != sorted(set(predecessor_cohort_ids))
                or not predecessor_cohort_ids
                or not isinstance(successor_cohort_id, str)
                or set(cohorts)
                != set(predecessor_cohort_ids) | {successor_cohort_id}
            ):
                raise RuntimeError("final1000 chained cohort membership mismatch")
            cohort_ids = sorted(cohorts)
        else:
            if set(cohorts) != {"predecessor", "successor"}:
                raise RuntimeError("final1000 harvest cohort inventory mismatch")
            predecessor_cohort_ids = ["predecessor"]
            successor_cohort_id = "successor"
            cohort_ids = ["predecessor", "successor"]
        for cohort_id in cohort_ids:
            cohort = cohorts.get(cohort_id)
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
            expected_schema = (
                CHAINED_HARVEST_COHORT_SCHEMA
                if chained_catalog
                else HARVEST_COHORT_SCHEMA
            )
            if (
                not isinstance(cohort, dict)
                or cohort.get("schema_version") != expected_schema
                or (
                    chained_catalog
                    and cohort.get("cohort_id") != cohort_id
                )
                or (
                    not chained_catalog
                    and cohort.get("role") != cohort_id
                )
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
        successor_cohort = cohorts[successor_cohort_id]
        if successor_cohort["resource_policy_ids"] != [SUCCESSOR_RESOURCE_POLICY_ID]:
            raise RuntimeError("final1000 successor harvest policy drifted")
        if successor_cohort.get("launch_plan_sha256") != plan.get(
            "launch_plan_sha256"
        ):
            raise RuntimeError("final1000 successor cohort plan identity drifted")
        if not any(
            cohorts[cohort_id].get("launch_plan_sha256")
            == migration.get("predecessor_launch_plan_sha256")
            for cohort_id in predecessor_cohort_ids
        ):
            raise RuntimeError("final1000 primary predecessor cohort is missing")
        for predecessor_cohort_id in predecessor_cohort_ids:
            predecessor_cohort = cohorts[predecessor_cohort_id]
            for stage in STAGES:
                old = predecessor_cohort["stage_bindings"][stage.stage_id]
                new = successor_cohort["stage_bindings"][stage.stage_id]
                same_binding = old == new
                different_bundle = all(
                    old[key] != new[key]
                    for key in (
                        "bundle_id",
                        "bundle_manifest_sha256",
                        "remote_bundle",
                    )
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
        harvest_cohort_id = str(entry.get("harvest_cohort_id") or "")
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
            if chained_catalog:
                expected_origin = (
                    "successor"
                    if harvest_cohort_id == successor_cohort_id
                    else "predecessor"
                )
                if (
                    harvest_cohort_id not in validated_harvest_cohorts
                    or origin != expected_origin
                ):
                    raise RuntimeError("final1000 chained ledger cohort drifted")
                cohort = validated_harvest_cohorts[harvest_cohort_id]
            else:
                if harvest_cohort_id:
                    raise RuntimeError("legacy rolling ledger gained a cohort id")
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
        successor_canary_entry_count_by_stage = {
            stage.stage_id: sum(
                entry["stage_id"] == stage.stage_id
                and entry["wave"] == "canary"
                and str(entry.get("origin") or "successor") == "successor"
                for entry in entries
            )
            for stage in STAGES
        }
        if chained_catalog and (
            migration.get("successor_canary_gap_policy")
            != CHAINED_CANARY_GAP_POLICY
            or any(
                count > 1
                for count in successor_canary_entry_count_by_stage.values()
            )
        ):
            raise RuntimeError("final1000 chained canary cardinality drifted")
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
        external_attestation_present = (
            migration.get("external_canary_attestation") is not None
        )
        external_prepassed_shape = (
            state.get("ramp_released") is True
            and passed == set(BY_ID)
            and all(not task_ids for task_ids in expected_canary_ids.values())
            and expected_canary_status
            == {stage.stage_id: "remote_preflight_passed" for stage in STAGES}
        )
        if (
            migration.get("successor_canary_task_ids_by_stage") != expected_canary_ids
            or migration.get("successor_canary_status_by_stage")
            != expected_canary_status
            or external_attestation_present != external_prepassed_shape
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


def _validate_chained_shadow_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a non-runnable v2 state for read-only preflight only."""

    value = _validate_state_for_sealed_active_quotas(
        state,
        plan,
        expected_active_quotas=SUCCESSOR_ACTIVE_QUOTAS,
        allow_chained_shadow=True,
    )
    migration = value.get("rolling_migration")
    if (
        not isinstance(migration, dict)
        or migration.get("schema_version") != CHAINED_ROLLING_MIGRATION_SCHEMA
        or migration.get("predecessor_stop_observed") is not False
        or migration.get("shadow_only") is not True
    ):
        raise RuntimeError("final1000 state is not an authenticated chained shadow")
    return value


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
    migration = state.get("rolling_migration")
    if (
        isinstance(migration, dict)
        and migration.get("schema_version") == CHAINED_ROLLING_MIGRATION_SCHEMA
    ):
        cohort_id = migration.get("successor_harvest_cohort_id")
        if not isinstance(cohort_id, str) or not cohort_id:
            raise RuntimeError("chained successor cohort id is unavailable")
        entry["harvest_cohort_id"] = cohort_id
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
        "timeout": "timeout",
        "timed_out": "timeout",
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
    active_entries = [
        entry
        for entry in state["entries"]
        if entry["task_id"] is not None and entry["state"] not in TERMINAL_STATES
    ]
    bulk_by_id: dict[int, Mapping[str, Any]] | None = None
    list_namespace_tasks = getattr(scheduler, "list_namespace_tasks", None)
    if callable(list_namespace_tasks):
        inventory = list_namespace_tasks()
        if not isinstance(inventory, Sequence):
            raise RuntimeError("scheduler bulk inventory is not a sequence")
        bulk_by_id = {}
        for observed in inventory:
            if not isinstance(observed, Mapping):
                raise RuntimeError("scheduler bulk inventory row is not an object")
            task_id = _task_id(observed)
            dedupe = str(observed.get("dedupe_key") or "")
            if (
                not str(observed.get("name") or "").startswith(TASK_NAME_PREFIX)
                or not dedupe.startswith(DEDUPE_PREFIX)
            ):
                raise RuntimeError("scheduler bulk inventory escaped final1000 namespace")
            prior = bulk_by_id.get(task_id)
            if prior is not None and prior.get("dedupe_key") != dedupe:
                raise RuntimeError(
                    "scheduler bulk inventory repeats a task id with changed identity"
                )
            bulk_by_id[task_id] = observed

    changed = 0
    for entry in active_entries:
        expected_task_id = int(entry["task_id"])
        observed = (
            bulk_by_id.get(expected_task_id) if bulk_by_id is not None else None
        )
        # A task may disappear between the sealed bulk snapshot and this
        # reconciliation pass, so only missing active IDs use the detail read.
        if observed is None:
            observed = scheduler.get_task(expected_task_id)
        if observed is None:
            continue
        observed_ids = {
            int(value)
            for value in (observed.get("id"), observed.get("task_id"))
            if isinstance(value, int) and not isinstance(value, bool)
        }
        expected_name = (
            f"{BY_ID[str(entry['stage_id'])].task_name_stem}-"
            f"{entry['wave']}-{int(entry['seed'])}"
        )
        if (
            not observed_ids
            or observed_ids != {expected_task_id}
            or str(observed.get("dedupe_key") or "") != entry["dedupe_key"]
            or str(observed.get("name") or "") != expected_name
        ):
            raise RuntimeError("scheduler task changed final1000 task identity")
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
    stage that has never received its one successor canary, even when that
    predecessor stage is temporarily above its successor quota.  No task is
    cancelled to force convergence; excess drains naturally and its slots
    move to deficits.
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
            missing_canary_stage_count = sum(
                stage.stage_id not in passed
                and not any(
                    entry["stage_id"] == stage.stage_id
                    and entry["wave"] == "canary"
                    and str(entry.get("origin") or "successor") == "successor"
                    for entry in state["entries"]
                )
                for stage in STAGES
            )
            # Before all four remote preflights pass, reserve at most one
            # natural gap per still-unrepresented stage.  Any extra natural
            # gaps intentionally remain empty.  Filling those gaps as
            # ``canary`` would create an unbounded canary batch; filling them
            # as ``refill`` would bypass the remote preflight gate.
            replacement_stage_order = _weighted_deficit_refill_order(
                state,
                slots=min(
                    TOTAL_ACTIVE_QUOTA - _active_count(state),
                    missing_canary_stage_count,
                ),
                reserve_missing_migration_canaries=True,
            )
            for stage_id in replacement_stage_order:
                replacement_entries.append(
                    _append_refill(
                        state,
                        templates=templates,
                        stage_id=stage_id,
                        wave="canary",
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
            active_shortfall = max(0, TOTAL_ACTIVE_QUOTA - _active_count(state))
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
                    "successor_canary_gap_policy": (
                        copy.deepcopy(CHAINED_CANARY_GAP_POLICY)
                        if isinstance(state.get("rolling_migration"), dict)
                        and state["rolling_migration"].get("schema_version")
                        == CHAINED_ROLLING_MIGRATION_SCHEMA
                        else None
                    ),
                    "active_target_temporarily_relaxed": bool(active_shortfall),
                    "active_shortfall_held_until_all_canaries_passed": (
                        active_shortfall
                    ),
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
