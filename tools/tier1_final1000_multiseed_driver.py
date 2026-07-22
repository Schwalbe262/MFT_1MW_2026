"""Restartable production driver for the pure Phase-A controller-v3.

Dry-run is the default and performs no POST and no state write.  ``--apply``
is deliberately required for the only Scheduler mutation implemented here,
``POST /api/tasks``.  Existing v1 lanes are observed in place and count toward
the physical 500-lane target; they are never replaced until they terminate
naturally.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.parse
import urllib.error
import urllib.request

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import read_json
    from tier1_final1000_multiseed_contract import (
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_child_receipt,
        validate_task_status,
    )
    from tier1_final1000_multiseed_consumer import (
        CAPABILITIES as CONSUMER_CAPABILITIES,
        CAPABILITY_RECEIPT_SCHEMA as CONSUMER_CAPABILITY_RECEIPT_SCHEMA,
        require_consumer_capability,
    )
    from tier1_final1000_multiseed_monitor import (
        CAPABILITY_RECEIPT_SCHEMA,
        default_adapter_code_files,
        require_backend_capability,
        validate_backend_capability_receipt,
    )
    from tier1_final1000_multiseed_controller import (
        ACTIVE_STATES,
        active_physical_count,
        build_reserved_gate_task,
        deficit_stage_order,
        observe_scheduler_tasks,
        reserve_lanes,
        upgrade_v1_state,
        validate_state as validate_controller_state,
    )
    from tier1_final1000_slurm_controller import (
        _validate_chained_shadow_state,
        _validate_state as validate_v1_state,
    )
    from tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_launch_plan,
    )
    from tier1_final1000_stage_profiles import STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import read_json
    from tools.tier1_final1000_multiseed_contract import (
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_child_receipt,
        validate_task_status,
    )
    from tools.tier1_final1000_multiseed_consumer import (
        CAPABILITIES as CONSUMER_CAPABILITIES,
        CAPABILITY_RECEIPT_SCHEMA as CONSUMER_CAPABILITY_RECEIPT_SCHEMA,
        require_consumer_capability,
    )
    from tools.tier1_final1000_multiseed_monitor import (
        CAPABILITY_RECEIPT_SCHEMA,
        default_adapter_code_files,
        require_backend_capability,
        validate_backend_capability_receipt,
    )
    from tools.tier1_final1000_multiseed_controller import (
        ACTIVE_STATES,
        active_physical_count,
        build_reserved_gate_task,
        deficit_stage_order,
        observe_scheduler_tasks,
        reserve_lanes,
        upgrade_v1_state,
        validate_state as validate_controller_state,
    )
    from tools.tier1_final1000_slurm_controller import (
        _validate_chained_shadow_state,
        _validate_state as validate_v1_state,
    )
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_launch_plan,
    )
    from tools.tier1_final1000_stage_profiles import STAGES


DRIVER_STATE_SCHEMA = "mft-tier1-final1000-multiseed-production-driver-v2"
MONITORING_CAPABILITY_SCHEMA = CAPABILITY_RECEIPT_SCHEMA
STOP_SCHEMA = "mft-tier1-final1000-predecessor-stop-observation-v1"
SUPERVISOR_STOP_SCHEMA = "mft-tier1-final1000-multiseed-supervisor-stop-v1"
GATE_LANES_PER_STAGE = 1
NAMESPACE_PREFIX = "mft-t1fg-"
DEDUPE_PREFIX = "mft-tier1-final1000:"


class Scheduler(Protocol):
    post_count: int

    def latest_10000(self) -> Sequence[Mapping[str, Any]]: ...

    def task_detail(self, task_id: int) -> Mapping[str, Any] | None: ...

    def post_task(self, task: Mapping[str, Any]) -> Mapping[str, Any]: ...


class GateReader(Protocol):
    def lane_evidence(
        self, task_id: int, seeds: Sequence[int]
    ) -> Mapping[str, Any] | None: ...


def _seal(value: Mapping[str, Any], field: str = "state_sha256") -> dict[str, Any]:
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    return {**unsigned, field: canonical_sha256(unsigned)}


def validate_monitoring_capability(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        receipt = validate_backend_capability_receipt(value)
    except RuntimeError as exc:
        raise RuntimeError(
            "monitoring/status backend capability is not deployable"
        ) from exc
    contract = receipt["capability_contract"]
    if (
        contract.get("scheduler_task_count_semantics") != "physical_lane_count"
        or contract.get("logical_seed_count_field") != "logical_seed_count"
        or contract.get("running_parent_visibility_rule")
        != "batch_ordinal < sealed_child_count"
        or contract.get("scheduler_mutation_performed") is not False
        or contract.get("remote_write_performed") is not False
    ):
        raise RuntimeError("monitoring/status backend capability is not deployable")
    return receipt


def validate_stop(
    value: Mapping[str, Any], final_predecessor_state: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema_version") != STOP_SCHEMA
        or value.get("predecessor_stopped") is not True
        or value.get("final_state_sha256")
        != final_predecessor_state.get("state_sha256")
        or value.get("final_state_revision") != final_predecessor_state.get("revision")
        or value.get("final_entry_count")
        != len(final_predecessor_state.get("entries") or [])
        or final_predecessor_state.get("stop_requested") is not True
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("predecessor stop observation is not sealed")
    return copy.deepcopy(dict(value))


def validate_supervisor_stop(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema_version") != SUPERVISOR_STOP_SCHEMA
        or value.get("stop_requested") is not True
        or not str(value.get("reason") or "")
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("multi-seed supervisor stop latch is not sealed")
    return copy.deepcopy(dict(value))


def _validate_predecessor(
    state: Mapping[str, Any], source_plan: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return validate_v1_state(state, source_plan)
    except RuntimeError:
        return _validate_chained_shadow_state(state, source_plan)


def _consumer_cutover_binding(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact, already live-validated consumer receipt to cutover state."""

    value = copy.deepcopy(dict(receipt))
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    handoff = value.get("v1_handoff")
    lease = value.get("writer_lease")
    indexes = value.get("condition_indexes")
    if (
        value.get("schema_version") != CONSUMER_CAPABILITY_RECEIPT_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("capabilities") != CONSUMER_CAPABILITIES
        or value.get("publish_mode") != "canonical"
        or value.get("runtime_root") != value.get("publication_root")
        or value.get("launch_plan_sha256") != plan.get("launch_plan_sha256")
        or value.get("controller_state_sha256") != controller.get("state_sha256")
        or not isinstance(indexes, list)
        or len(indexes) != len(STAGES)
        or {str(item.get("stage_id") or "") for item in indexes}
        != {stage.stage_id for stage in STAGES}
        or not isinstance(handoff, dict)
        or handoff.get("old_writer_exited") is not True
        or handoff.get("sealed_v1_condition_index_count") != len(STAGES)
        or not str(handoff.get("receipt_sha256") or "")
        or not isinstance(lease, dict)
        or lease.get("run_id") != value.get("run_id")
        or lease.get("lease_held_at_publication") is not True
    ):
        raise RuntimeError(
            "canonical consumer capability does not bind the exact cutover state"
        )
    return {
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "receipt": value,
    }


def _consumer_capability_caught_up(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> bool:
    if receipt.get("launch_plan_sha256") != plan.get("launch_plan_sha256"):
        raise RuntimeError("consumer capability launch-plan identity drifted")
    return receipt.get("controller_state_sha256") == controller.get("state_sha256")


def _validate_consumer_cutover_binding(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    prepared_controller_state_sha256: str,
) -> dict[str, Any]:
    if set(value) != {"receipt_path", "receipt"} or not str(
        value.get("receipt_path") or ""
    ):
        raise RuntimeError("consumer cutover capability binding is invalid")
    receipt = value.get("receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("consumer cutover capability binding is invalid")
    return _consumer_cutover_binding(
        Path(str(value["receipt_path"])),
        receipt,
        plan,
        {"state_sha256": prepared_controller_state_sha256},
    )


def initial_driver_state(
    predecessor_state: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    monitoring_capability: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor = _validate_predecessor(predecessor_state, source_plan)
    capability = validate_monitoring_capability(monitoring_capability)
    unsigned = {
        "schema_version": DRIVER_STATE_SCHEMA,
        "revision": 0,
        "parent_state_sha256": None,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "source_start": {
            "launch_plan_sha256": source_plan["launch_plan_sha256"],
            "state_sha256": predecessor["state_sha256"],
            "revision": int(predecessor["revision"]),
            "entry_count": len(predecessor["entries"]),
        },
        "controller_state": None,
        "controller_state_sha256": None,
        "cutover_source": None,
        "consumer_cutover_capability": None,
        "phase": "batch1",
        "gate_lanes": {"batch1": {}, "batch4": {}},
        "monitoring_capability_sha256": capability["receipt_sha256"],
        "refill_released": False,
        "failure": None,
        "predecessor_stop_observed": False,
        "baseline_physical_target": 500,
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
    }
    return validate_driver_state(_seal(unsigned), plan, capability)


def validate_driver_state(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    monitoring_capability: Mapping[str, Any],
) -> dict[str, Any]:
    plan = validate_launch_plan(plan)
    capability = validate_monitoring_capability(monitoring_capability)
    unsigned = {key: item for key, item in value.items() if key != "state_sha256"}
    gates = value.get("gate_lanes")
    controller = value.get("controller_state")
    source_start = value.get("source_start")
    if (
        value.get("schema_version") != DRIVER_STATE_SCHEMA
        or value.get("launch_plan_sha256") != plan["launch_plan_sha256"]
        or value.get("state_sha256") != canonical_sha256(unsigned)
        or value.get("controller_state_sha256")
        != (controller.get("state_sha256") if isinstance(controller, dict) else None)
        or value.get("monitoring_capability_sha256") != capability["receipt_sha256"]
        or value.get("phase")
        not in {
            "batch1",
            "batch4",
            "awaiting_predecessor_stop",
            "cutover_prepared",
            "refill",
            "failed",
        }
        or not isinstance(gates, dict)
        or set(gates) != {"batch1", "batch4"}
        or any(not isinstance(gates[key], dict) for key in gates)
        or value.get("scheduler_mutation_endpoints") != ["POST /api/tasks"]
        or not isinstance(value.get("refill_released"), bool)
        or not isinstance(value.get("predecessor_stop_observed"), bool)
        or value.get("baseline_physical_target") != 500
        or not isinstance(source_start, dict)
        or set(source_start)
        != {"launch_plan_sha256", "state_sha256", "revision", "entry_count"}
        or not isinstance(source_start.get("revision"), int)
        or not isinstance(source_start.get("entry_count"), int)
    ):
        raise RuntimeError("multi-seed driver state seal mismatch")
    for phase, lanes in gates.items():
        expected_length = 1 if phase == "batch1" else 4
        if len(lanes) > len(STAGES):
            raise RuntimeError("multi-seed rollout gate cardinality drifted")
        stage_ids: set[str] = set()
        for dedupe, lane in lanes.items():
            expected_task = (
                build_reserved_gate_task(
                    plan, stage_id=str(lane.get("stage_id") or ""), phase=phase
                )
                if isinstance(lane, dict)
                else {}
            )
            if (
                not isinstance(lane, dict)
                or lane.get("parent_dedupe_key") != dedupe
                or lane.get("stage_id") not in {stage.stage_id for stage in STAGES}
                or lane.get("batch_length") != expected_length
                or lane.get("state") not in {"planned", "active", "passed", "failed"}
                or not isinstance(lane.get("task_id"), (int, type(None)))
                or lane.get("task") != expected_task
                or lane.get("task_sha256") != canonical_sha256(expected_task)
                or lane.get("stage_id") in stage_ids
            ):
                raise RuntimeError("multi-seed rollout gate ledger drifted")
            stage_ids.add(str(lane["stage_id"]))
    cutover = value.get("cutover_source")
    consumer_cutover = value.get("consumer_cutover_capability")
    if value["phase"] in {"cutover_prepared", "refill"}:
        if (
            value["predecessor_stop_observed"] is not True
            or not isinstance(controller, dict)
            or not isinstance(cutover, dict)
            or set(cutover)
            != {
                "state_sha256",
                "revision",
                "entry_count",
                "prepared_controller_state_sha256",
            }
            or cutover.get("state_sha256")
            != controller.get("source_controller_state", {}).get("state_sha256")
            or not isinstance(cutover.get("revision"), int)
            or not isinstance(cutover.get("entry_count"), int)
        ):
            raise RuntimeError("multi-seed cutover lacks its exact prepared controller")
        validate_controller_state(controller, plan)
        prepared_sha = str(cutover["prepared_controller_state_sha256"])
        if value["phase"] == "cutover_prepared":
            if (
                value["refill_released"] is not False
                or controller.get("state_sha256") != prepared_sha
                or consumer_cutover is not None
            ):
                raise RuntimeError("prepared cutover released refill prematurely")
        else:
            if value["refill_released"] is not True or not isinstance(
                consumer_cutover, dict
            ):
                raise RuntimeError(
                    "multi-seed refill was released without consumer capability"
                )
            _validate_consumer_cutover_binding(consumer_cutover, plan, prepared_sha)
    elif (
        controller is not None
        or value["refill_released"] is not False
        or consumer_cutover is not None
    ):
        raise RuntimeError("multi-seed driver copied predecessor before cutover")
    return copy.deepcopy(dict(value))


def _advance_driver(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    capability: Mapping[str, Any],
    **changes: Any,
) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in state.items() if key != "state_sha256"
    }
    unsigned.update(copy.deepcopy(changes))
    unsigned["revision"] = int(state["revision"]) + 1
    unsigned["parent_state_sha256"] = state["state_sha256"]
    controller = unsigned["controller_state"]
    unsigned["controller_state_sha256"] = (
        controller["state_sha256"] if isinstance(controller, dict) else None
    )
    return validate_driver_state(_seal(unsigned), plan, capability)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class AtomicStateStore:
    path: Path
    controller_export: Path | None = None

    @property
    def history(self) -> Path:
        return self.path.parent / f"{self.path.name}.history"

    @property
    def controller_path(self) -> Path:
        return self.controller_export or self.path.with_name(
            f"{self.path.stem}.prepared-controller.json"
        )

    def _validate_current_controller_export(
        self, current: Mapping[str, Any], plan: Mapping[str, Any]
    ) -> None:
        if current["phase"] not in {"cutover_prepared", "refill"}:
            return
        if not self.controller_path.is_file():
            return
        # The driver state is authoritative. Artifact-first persistence can
        # legitimately leave a sealed next-state export after a crash before
        # the state replace. Such an export cannot release refill because its
        # consumer receipt will not match the authoritative controller SHA;
        # the next apply cycle rewrites it from ``current``.
        validate_controller_state(read_json(self.controller_path), plan)

    def load(
        self, plan: Mapping[str, Any], capability: Mapping[str, Any]
    ) -> dict[str, Any]:
        current = validate_driver_state(read_json(self.path), plan, capability)
        cursor = current
        visited: set[str] = set()
        while cursor.get("parent_state_sha256") is not None:
            parent_sha = str(cursor["parent_state_sha256"])
            if parent_sha in visited:
                raise RuntimeError("driver state ancestor chain contains a cycle")
            visited.add(parent_sha)
            parent_path = self.history / f"{parent_sha}.json"
            parent = validate_driver_state(read_json(parent_path), plan, capability)
            if parent["state_sha256"] != parent_sha or int(
                parent["revision"]
            ) + 1 != int(cursor["revision"]):
                raise RuntimeError("driver state ancestor chain is discontinuous")
            cursor = parent
        self._validate_current_controller_export(current, plan)
        return current

    def write(
        self,
        prior: Mapping[str, Any] | None,
        state: Mapping[str, Any],
        plan: Mapping[str, Any],
        capability: Mapping[str, Any],
    ) -> None:
        validate_driver_state(state, plan, capability)
        if prior is not None:
            validate_driver_state(prior, plan, capability)
            if state.get("parent_state_sha256") != prior.get("state_sha256"):
                raise RuntimeError("atomic state write lost its parent identity")
            self.history.mkdir(parents=True, exist_ok=True)
            history_path = self.history / f"{prior['state_sha256']}.json"
            if history_path.exists() and read_json(history_path) != prior:
                raise RuntimeError("content-addressed state history collision")
            if not history_path.exists():
                _atomic_write(history_path, prior)
        controller = state.get("controller_state")
        if state["phase"] in {"cutover_prepared", "refill"}:
            if not isinstance(controller, dict):
                raise RuntimeError("prepared controller export has no controller state")
            # Artifact first is intentional: a crash before the driver state
            # replace leaves the old authority intact; restart deterministically
            # overwrites this exact sealed candidate before advancing again.
            _atomic_write(self.controller_path, controller)
        _atomic_write(self.path, state)


def write_stop_file(
    path: Path, final_predecessor_state: Mapping[str, Any]
) -> dict[str, Any]:
    if final_predecessor_state.get("stop_requested") is not True:
        raise RuntimeError("predecessor state has not latched its stop")
    unsigned = {
        "schema_version": STOP_SCHEMA,
        "predecessor_stopped": True,
        "final_state_sha256": final_predecessor_state.get("state_sha256"),
        "final_state_revision": final_predecessor_state.get("revision"),
        "final_entry_count": len(final_predecessor_state.get("entries") or []),
    }
    value = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    validate_stop(value, final_predecessor_state)
    _atomic_write(path, value)
    return value


def write_supervisor_stop(path: Path, reason: str) -> dict[str, Any]:
    unsigned = {
        "schema_version": SUPERVISOR_STOP_SCHEMA,
        "stop_requested": True,
        "reason": str(reason),
    }
    value = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    validate_supervisor_stop(value)
    _atomic_write(path, value)
    return value


class SchedulerApi:
    """One bulk GET, missing-detail fallback, and an apply-only task POST."""

    def __init__(
        self,
        base_url: str,
        *,
        apply: bool = False,
        timeout: float = 30.0,
        get_attempts: int = 5,
        get_backoff_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.apply = bool(apply)
        self.timeout = float(timeout)
        self.get_attempts = int(get_attempts)
        self.get_backoff_seconds = float(get_backoff_seconds)
        if self.get_attempts < 1 or self.get_backoff_seconds < 0:
            raise ValueError("Scheduler GET retry policy is invalid")
        self.post_count = 0

    def _request(self, path: str, *, body: Mapping[str, Any] | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        attempts = self.get_attempts if body is None else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                    raise
            except urllib.error.URLError:
                if attempt + 1 >= attempts:
                    raise
            time.sleep(min(30.0, self.get_backoff_seconds * (2**attempt)))
        raise AssertionError("bounded Scheduler GET retry fell through")

    def latest_10000(self) -> Sequence[Mapping[str, Any]]:
        query = urllib.parse.urlencode(
            {
                "name_prefix": NAMESPACE_PREFIX,
                "sort_by": "id",
                "sort_order": "desc",
                "limit": 10000,
            }
        )
        value = self._request(f"/api/tasks?{query}")
        if not isinstance(value, list):
            raise RuntimeError("Scheduler latest-10000 inventory is not a list")
        return value

    def task_detail(self, task_id: int) -> Mapping[str, Any] | None:
        value = self._request(f"/api/tasks/{int(task_id)}")
        return value if isinstance(value, dict) else None

    def post_task(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.apply:
            raise RuntimeError("Scheduler POST requires explicit --apply")
        if set(task) != REQUIRED_SCHEDULER_FIELDS:
            raise RuntimeError("Scheduler task envelope is not exact")
        value = self._request("/api/tasks", body=task)
        self.post_count += 1
        if not isinstance(value, dict):
            raise RuntimeError("Scheduler POST response is not an object")
        return value


class SchedulerGateReader:
    """Read only the immutable/status files required to authenticate a gate."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        get_attempts: int = 5,
        get_backoff_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.get_attempts = int(get_attempts)
        self.get_backoff_seconds = float(get_backoff_seconds)

    def _json_file(self, task_id: int, relative: str) -> Mapping[str, Any] | None:
        query = urllib.parse.urlencode({"path": relative, "base": "remote_cwd"})
        request = urllib.request.Request(
            self.base_url + f"/api/tasks/{int(task_id)}/remote-file?{query}",
            method="GET",
        )
        raw: bytes | None = None
        for attempt in range(self.get_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code in {404, 409}:
                    return None
                if (
                    exc.code not in {429, 500, 502, 503, 504}
                    or attempt + 1 >= self.get_attempts
                ):
                    raise
            except urllib.error.URLError:
                if attempt + 1 >= self.get_attempts:
                    raise
            time.sleep(min(30.0, self.get_backoff_seconds * (2**attempt)))
        if raw is None:
            raise AssertionError("bounded Scheduler remote-file retry fell through")
        if not raw.strip():
            return None
        value = json.loads(raw.decode())
        if not isinstance(value, dict):
            raise RuntimeError("Scheduler remote gate object is not JSON object")
        return value

    def lane_evidence(
        self, task_id: int, seeds: Sequence[int]
    ) -> Mapping[str, Any] | None:
        root = f"runs/task-{int(task_id)}"
        status = self._json_file(task_id, f"{root}/task_status.json")
        manifest = self._json_file(task_id, f"{root}/batch_manifest.json")
        if status is None or manifest is None:
            return None
        receipts = []
        for seed in seeds:
            receipt = self._json_file(
                task_id, f"{root}/seed-{int(seed)}/seed_status.json"
            )
            if receipt is None:
                return None
            receipts.append(receipt)
        return {
            "task_status": status,
            "manifest": manifest,
            "child_receipts": receipts,
        }


def _exact_task(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        row.get("name") == expected["name"]
        and row.get("dedupe_key") == expected["dedupe_key"]
        and row.get("task_json") == expected
    )


def scheduler_inventory(
    scheduler: Scheduler, state: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    rows = list(scheduler.latest_10000())
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("name") or "").startswith(
            NAMESPACE_PREFIX
        ):
            continue
        if not str(row.get("dedupe_key") or "").startswith(DEDUPE_PREFIX):
            raise RuntimeError("Scheduler namespace contains a foreign dedupe")
        task_id = row.get("id") or row.get("task_id")
        if isinstance(task_id, int):
            by_id[task_id] = row
    for entry in state["controller_state"]["entries"]:
        task_id = entry.get("task_id")
        if (
            task_id is not None
            and entry["state"] in ACTIVE_STATES
            and task_id not in by_id
        ):
            detail = scheduler.task_detail(int(task_id))
            if detail is None:
                raise RuntimeError(f"active Scheduler task {task_id} is missing")
            by_id[int(task_id)] = detail
    return {str(row["dedupe_key"]): row for row in by_id.values()}


def gate_inventory(
    scheduler: Scheduler, gates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    rows = list(scheduler.latest_10000())
    by_dedupe = {
        str(row.get("dedupe_key") or ""): row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("name") or "").startswith(NAMESPACE_PREFIX)
        and str(row.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
    }
    for lane in gates.values():
        task_id = lane.get("task_id")
        dedupe = str(lane["parent_dedupe_key"])
        if (
            task_id is not None
            and lane["state"] == "active"
            and dedupe not in by_dedupe
        ):
            detail = scheduler.task_detail(int(task_id))
            if detail is None:
                raise RuntimeError(f"active additive gate task {task_id} is missing")
            by_dedupe[dedupe] = detail
    return by_dedupe


def _lane_passed(
    lane: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
) -> bool:
    if evidence is None:
        return False
    expected = lane["task"]
    manifest = validate_batch_manifest(evidence["manifest"])
    expected_manifest = batch_manifest_from_payload(expected["payload_json"])
    status = validate_task_status(evidence["task_status"])
    receipts = evidence.get("child_receipts")
    if (
        manifest != expected_manifest
        or status["manifest_sha256"] != manifest["manifest_sha256"]
        or status["state"] != "completed"
        or not isinstance(receipts, list)
        or len(receipts) != lane["batch_length"]
    ):
        raise RuntimeError("terminal gate lane did not authenticate as all-pass")
    for receipt in receipts:
        if validate_child_receipt(receipt, manifest=manifest)["state"] != "completed":
            raise RuntimeError("terminal gate child is not completed")
    return True


def _phase_complete(gates: Mapping[str, Any], phase: str) -> bool:
    lanes = gates[phase].values()
    return all(
        sum(
            lane["stage_id"] == stage.stage_id and lane["state"] == "passed"
            for lane in lanes
        )
        == GATE_LANES_PER_STAGE
        for stage in STAGES
    )


def cycle(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    capability: Mapping[str, Any],
    scheduler: Scheduler,
    gate_reader: GateReader,
    *,
    apply: bool = False,
    stop: Mapping[str, Any] | None = None,
    final_predecessor_state: Mapping[str, Any] | None = None,
    source_plan: Mapping[str, Any] | None = None,
    ancestor_plans: Sequence[Mapping[str, Any]] = (),
    consumer_capability_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one bounded reconcile.  Dry-run returns intents and performs POST0."""

    current = validate_driver_state(state, plan, capability)
    post_count_start = scheduler.post_count
    gates = copy.deepcopy(current["gate_lanes"])
    phase = str(current["phase"])
    controller = current["controller_state"]
    consumer_cutover = current["consumer_cutover_capability"]
    cutover_source = current["cutover_source"]
    stop_observed = current["predecessor_stop_observed"]
    refill_released = current["refill_released"]
    refill_capability_ready = False
    inventory: dict[str, Mapping[str, Any]] = {}

    if phase in {"batch1", "batch4"}:
        inventory = gate_inventory(scheduler, gates[phase])
        for lane in gates[phase].values():
            row = inventory.get(lane["parent_dedupe_key"])
            if row is None:
                if lane["state"] != "planned":
                    raise RuntimeError(
                        "active additive gate disappeared from Scheduler"
                    )
                continue
            if not _exact_task(row, lane["task"]):
                raise RuntimeError("Scheduler changed additive gate identity")
            task_id = row.get("id") or row.get("task_id")
            if not isinstance(task_id, int) or task_id <= 0:
                raise RuntimeError("additive gate lacks a Scheduler task id")
            lane["task_id"] = task_id
            observed = {
                "queued": "active",
                "attaching": "active",
                "running": "active",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "failed",
                "timeout": "failed",
                "timed_out": "failed",
            }.get(str(row.get("status") or "").lower())
            if observed is None:
                raise RuntimeError("additive gate has an unknown Scheduler state")
            if observed == "failed":
                lane["state"] = "failed"
                return _advance_driver(
                    current,
                    plan,
                    capability,
                    gate_lanes=gates,
                    phase="failed",
                    failure=f"terminal_gate_failure:{lane['parent_dedupe_key']}",
                ), []
            if observed == "completed":
                evidence = gate_reader.lane_evidence(
                    task_id,
                    [
                        child["seed"]
                        for child in lane["task"]["payload_json"]["children"]
                    ],
                )
                try:
                    passed = _lane_passed(lane, evidence)
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    lane["state"] = "failed"
                    return _advance_driver(
                        current,
                        plan,
                        capability,
                        gate_lanes=gates,
                        phase="failed",
                        failure=f"terminal_gate_evidence:{type(exc).__name__}:{exc}",
                    ), []
                lane["state"] = "passed" if passed else "active"
            else:
                lane["state"] = "active"
        if _phase_complete(gates, phase):
            phase = "batch4" if phase == "batch1" else "awaiting_predecessor_stop"

    intents: list[dict[str, Any]] = []
    if phase in {"batch1", "batch4"}:
        length = 1 if phase == "batch1" else 4
        for stage in STAGES:
            if not any(
                lane["stage_id"] == stage.stage_id for lane in gates[phase].values()
            ):
                task = build_reserved_gate_task(
                    plan, stage_id=stage.stage_id, phase=phase
                )
                intents.append(task)
                gates[phase][task["dedupe_key"]] = {
                    "parent_dedupe_key": task["dedupe_key"],
                    "stage_id": stage.stage_id,
                    "batch_length": length,
                    "task_id": None,
                    "state": "planned",
                    "task": task,
                    "task_sha256": canonical_sha256(task),
                }

    if phase == "awaiting_predecessor_stop" and stop is not None:
        if final_predecessor_state is None or source_plan is None:
            raise RuntimeError("cutover requires the final predecessor state and plan")
        validated_final = validate_v1_state(final_predecessor_state, source_plan)
        validate_stop(stop, validated_final)
        start = current["source_start"]
        if (
            source_plan.get("launch_plan_sha256") != start["launch_plan_sha256"]
            or validated_final["state_sha256"] == start["state_sha256"]
            or int(validated_final["revision"]) <= int(start["revision"])
        ):
            raise RuntimeError("stale initial predecessor snapshot cannot cut over")
        candidate_controller = upgrade_v1_state(
            validated_final,
            plan,
            source_plan=source_plan,
            ancestor_plans=ancestor_plans,
            batch_length=4,
            clear_stop_for_successor=True,
        )
        controller = candidate_controller
        cutover_source = {
            "state_sha256": validated_final["state_sha256"],
            "revision": validated_final["revision"],
            "entry_count": len(validated_final["entries"]),
            "prepared_controller_state_sha256": candidate_controller["state_sha256"],
        }
        stop_observed = True
        phase = "cutover_prepared"

    if current["phase"] == "cutover_prepared" and controller is not None:
        if consumer_capability_path is not None and consumer_capability_path.is_file():
            receipt = require_consumer_capability(
                consumer_capability_path,
                required_publish_mode="canonical",
            )
            if _consumer_capability_caught_up(receipt, plan, controller):
                consumer_cutover = _consumer_cutover_binding(
                    consumer_capability_path,
                    receipt,
                    plan,
                    controller,
                )
                refill_released = True
                refill_capability_ready = True
                phase = "refill"

    if phase == "refill" and controller is not None:
        if not refill_capability_ready:
            if (
                consumer_capability_path is not None
                and consumer_capability_path.is_file()
            ):
                receipt = require_consumer_capability(
                    consumer_capability_path,
                    required_publish_mode="canonical",
                )
                if _consumer_capability_caught_up(receipt, plan, controller):
                    _consumer_cutover_binding(
                        consumer_capability_path,
                        receipt,
                        plan,
                        controller,
                    )
                    refill_capability_ready = True
        if refill_capability_ready:
            working = {**current, "controller_state": controller}
            inventory = scheduler_inventory(scheduler, working)
            observations = [
                (
                    entry["parent_dedupe_key"],
                    inventory[entry["parent_dedupe_key"]],
                )
                for entry in controller["entries"]
                if entry["parent_dedupe_key"] in inventory
            ]
            controller = observe_scheduler_tasks(
                controller, plan, observations=observations
            )
        requests = (
            [(stage_id, "refill", 4) for stage_id in deficit_stage_order(controller)]
            if refill_capability_ready
            else []
        )
        if requests:
            controller, intents = reserve_lanes(controller, plan, requests=requests)

    if not apply:
        if scheduler.post_count != post_count_start:
            raise RuntimeError("dry-run observed a Scheduler POST")
        changes: dict[str, Any] = {"gate_lanes": gates, "phase": phase}
        if phase in {"cutover_prepared", "refill"}:
            changes.update(
                {
                    "controller_state": controller,
                    "cutover_source": cutover_source,
                    "predecessor_stop_observed": stop_observed,
                    "refill_released": refill_released,
                    "consumer_cutover_capability": consumer_cutover,
                }
            )
        preview = _advance_driver(current, plan, capability, **changes)
        return preview, intents

    for task in intents:
        existing = inventory.get(task["dedupe_key"])
        row = existing or scheduler.post_task(task)
        task_id = row.get("id") or row.get("task_id")
        detail = (
            scheduler.task_detail(int(task_id)) if isinstance(task_id, int) else None
        )
        sealed = detail or row
        if not _exact_task(sealed, task):
            raise RuntimeError("Scheduler submission response changed the exact task")
        if phase in {"batch1", "batch4"}:
            gates[phase][task["dedupe_key"]]["task_id"] = int(task_id)
            gates[phase][task["dedupe_key"]]["state"] = "active"
        elif controller is not None:
            controller = observe_scheduler_tasks(
                controller,
                plan,
                observations=[(task["dedupe_key"], sealed)],
            )
    result = _advance_driver(
        current,
        plan,
        capability,
        controller_state=controller,
        gate_lanes=gates,
        phase=phase,
        cutover_source=cutover_source,
        predecessor_stop_observed=stop_observed,
        refill_released=refill_released,
        consumer_cutover_capability=consumer_cutover,
    )
    return result, intents


def run_cycles(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    ancestor_plans: Sequence[Mapping[str, Any]],
    scheduler: Scheduler,
    gate_reader: GateReader,
    *,
    capability_loader: Callable[[], Mapping[str, Any]],
    consumer_capability_path_loader: Callable[[], Path | None],
    predecessor_loader: Callable[[], Mapping[str, Any]],
    predecessor_stop_loader: Callable[[], Mapping[str, Any] | None],
    supervisor_stop_loader: Callable[[], Mapping[str, Any] | None],
    store: AtomicStateStore | None,
    apply: bool = False,
    watch: bool = False,
    poll_seconds: float = 30.0,
    max_cycles: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one cycle or an explicitly authorized, restartable watch loop."""

    if watch and not apply:
        raise RuntimeError("continuous watch requires explicit --apply")
    if apply and store is None:
        raise RuntimeError("--apply requires an atomic state store")
    if not 0.1 <= float(poll_seconds) <= 3600:
        raise ValueError("poll interval must be from 0.1 through 3600 seconds")
    if max_cycles is not None and max_cycles < 1:
        raise ValueError("max_cycles must be positive when supplied")
    current = copy.deepcopy(dict(state))
    summaries: list[dict[str, Any]] = []
    cycle_index = 0
    while True:
        supervisor_stop = supervisor_stop_loader()
        if supervisor_stop is not None:
            validate_supervisor_stop(supervisor_stop)
            break
        capability = capability_loader()
        validate_driver_state(current, plan, capability)
        predecessor_stop = predecessor_stop_loader()
        consumer_capability_path = consumer_capability_path_loader()
        final_predecessor = (
            predecessor_loader() if predecessor_stop is not None else None
        )
        next_state, intents = cycle(
            current,
            plan,
            capability,
            scheduler,
            gate_reader,
            apply=apply,
            stop=predecessor_stop,
            final_predecessor_state=final_predecessor,
            source_plan=source_plan,
            ancestor_plans=ancestor_plans,
            consumer_capability_path=consumer_capability_path,
        )
        if apply:
            assert store is not None
            store.write(current, next_state, plan, capability)
        summaries.append(
            {
                "cycle": cycle_index,
                "phase": next_state["phase"],
                "planned_task_count": len(intents),
                "state_sha256": next_state["state_sha256"],
                **operational_counts(next_state),
            }
        )
        current = next_state
        cycle_index += 1
        if current["phase"] == "failed":
            raise RuntimeError(f"multi-seed watch failed closed: {current['failure']}")
        if not watch or (max_cycles is not None and cycle_index >= max_cycles):
            break
        sleeper(float(poll_seconds))
    return current, summaries


def operational_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Keep physical Scheduler lanes distinct from logical seed work."""

    controller = state.get("controller_state")
    active_gates = [
        lane
        for phase in ("batch1", "batch4")
        for lane in state["gate_lanes"][phase].values()
        if lane["state"] in {"planned", "active"}
    ]
    if isinstance(controller, dict):
        physical = active_physical_count(controller)
        logical = sum(
            len(entry["seeds"])
            for entry in controller["entries"]
            if entry["state"] in ACTIVE_STATES
        )
    else:
        physical = int(state["baseline_physical_target"])
        logical = int(state["baseline_physical_target"])
    return {
        "physical_active_lanes": physical + len(active_gates),
        "logical_active_seeds": logical
        + sum(int(lane["batch_length"]) for lane in active_gates),
        "additive_gate_lanes": len(active_gates),
    }


def _named_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise RuntimeError(f"{label} must be NAME=PATH")
        name, path = raw.split("=", 1)
        if not name or name in result:
            raise RuntimeError(f"{label} name is empty or duplicated")
        result[name] = Path(path)
    if not result:
        raise RuntimeError(f"at least one {label} is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--prepared-controller-state", type=Path)
    parser.add_argument("--monitoring-capability", type=Path, required=True)
    parser.add_argument("--consumer-capability", type=Path, required=True)
    parser.add_argument("--condition-index", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--backend-file", action="append", default=[])
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--test-evidence", action="append", default=[])
    parser.add_argument("--test-revision", required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--ancestor-plan", action="append", type=Path, default=[])
    parser.add_argument("--scheduler-url", required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--watch-stop-file", type=Path)
    parser.add_argument("--watch-stop-reason")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--seal-predecessor-stop", action="store_true")
    parser.add_argument("--seal-watch-stop", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = read_json(args.plan)
    source_plan = read_json(args.source_plan)
    predecessor_state = read_json(args.predecessor_state)
    backend_files = _named_paths(args.backend_file, "backend file")
    test_evidence_files = _named_paths(args.test_evidence, "test evidence")

    def load_capability() -> dict[str, Any]:
        return require_backend_capability(
            args.monitoring_capability,
            index_path=args.condition_index,
            code_files=default_adapter_code_files(args.code_root),
            code_revision=args.code_revision,
            backend_files=backend_files,
            backend_revision=args.backend_revision,
            test_evidence_files=test_evidence_files,
            test_revision=args.test_revision,
        )

    capability = load_capability()
    if args.seal_watch_stop:
        if not args.apply:
            raise RuntimeError("writing a watch stop latch requires --apply")
        if args.watch_stop_file is None or not args.watch_stop_reason:
            raise RuntimeError(
                "--seal-watch-stop requires --watch-stop-file and reason"
            )
        receipt = write_supervisor_stop(args.watch_stop_file, args.watch_stop_reason)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.seal_predecessor_stop:
        _validate_predecessor(predecessor_state, source_plan)
        if not args.apply:
            raise RuntimeError("writing a predecessor stop receipt requires --apply")
        if args.stop_file is None:
            raise RuntimeError("--seal-predecessor-stop requires --stop-file")
        receipt = write_stop_file(args.stop_file, predecessor_state)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    store = AtomicStateStore(
        args.state,
        controller_export=args.prepared_controller_state,
    )
    initialized = False
    if args.state.exists():
        state = store.load(plan, capability)
    else:
        if not args.initialize:
            raise RuntimeError("missing driver state requires --initialize")
        state = initial_driver_state(predecessor_state, source_plan, plan, capability)
        initialized = True
        if args.apply:
            store.write(None, state, plan, capability)
    ancestors = [read_json(path) for path in args.ancestor_plan]
    scheduler = SchedulerApi(args.scheduler_url, apply=args.apply)
    reader = SchedulerGateReader(args.scheduler_url)
    next_state, summaries = run_cycles(
        state,
        plan,
        source_plan,
        ancestors,
        scheduler,
        reader,
        capability_loader=load_capability,
        consumer_capability_path_loader=lambda: (
            args.consumer_capability if args.consumer_capability.is_file() else None
        ),
        predecessor_loader=lambda: read_json(args.predecessor_state),
        predecessor_stop_loader=lambda: (
            read_json(args.stop_file)
            if args.stop_file and args.stop_file.exists()
            else None
        ),
        supervisor_stop_loader=lambda: (
            read_json(args.watch_stop_file)
            if args.watch_stop_file and args.watch_stop_file.exists()
            else None
        ),
        store=store,
        apply=args.apply,
        watch=args.watch,
        poll_seconds=args.poll_seconds,
        max_cycles=args.max_cycles,
    )
    counts = operational_counts(next_state)
    print(
        json.dumps(
            {
                "apply": bool(args.apply),
                "scheduler_post_count": scheduler.post_count,
                "state_write_count": len(summaries) * int(bool(args.apply))
                + int(initialized and args.apply),
                "phase": next_state["phase"],
                "prepared_controller_state": (
                    str(store.controller_path.resolve())
                    if next_state["phase"] in {"cutover_prepared", "refill"}
                    else None
                ),
                **counts,
                "cycle_count": len(summaries),
                "planned_task_count": sum(
                    item["planned_task_count"] for item in summaries
                ),
                "cycles": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
