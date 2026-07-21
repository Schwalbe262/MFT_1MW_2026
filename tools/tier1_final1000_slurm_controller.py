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
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.parse

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
        build_stage_task,
        validate_launch_plan,
        validate_task,
    )
    from tier1_final1000_stage_profiles import (
        BY_ID,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
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
        build_stage_task,
        validate_launch_plan,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import (
        BY_ID,
        STAGES,
        TOTAL_ACTIVE_QUOTA,
    )


STATE_SCHEMA = "mft-tier1-final1000-slurm-controller-state-v1"
RESULT_SCHEMA = "mft-tier1-final1000-slurm-controller-result-v1"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
TASK_NAME_PREFIX = "mft-t1fg-"
DEDUPE_PREFIX = "mft-tier1-final1000:"

ACTIVE_STATES = frozenset({"planned", "submitted", "queued", "running"})
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


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
            if dedupe in inventory and int(inventory[dedupe]["id"]) != int(
                task["id"]
            ):
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


def _seal_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "state_sha256"
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


def _entry(task: Mapping[str, Any]) -> dict[str, Any]:
    payload = task["payload_json"]
    lane = payload["lane"]
    return {
        "stage_id": payload["final_goal_stage_id"],
        "bundle_id": payload["bundle_id"],
        "seed": int(payload["seed"]),
        "wave": lane["wave"],
        "dedupe_key": task["dedupe_key"],
        "task_id": None,
        "state": "planned",
    }


def _initial_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    tasks = _all_tasks(plan)
    entries = [_entry(task) for task in tasks]
    unsigned = {
        "schema_version": STATE_SCHEMA,
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "revision": 0,
        "parent_state_sha256": None,
        "stop_requested": False,
        "ramp_released": False,
        "canary_passed_stage_ids": [],
        "entries": entries,
        "next_seed_by_stage": {
            stage.stage_id: stage.seed_start + stage.active_quota
            for stage in STAGES
        },
        "scheduler_name_prefix": TASK_NAME_PREFIX,
        "scheduler_dedupe_prefix": DEDUPE_PREFIX,
        "scheduler_mutation_endpoints": ["POST /api/tasks"],
        "scheduler_submit_count": 0,
    }
    return _seal_state(unsigned)


def _validate_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in state.items() if key != "state_sha256"
    }
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
    ):
        raise RuntimeError("final1000 controller state identity/SHA mismatch")
    dedupe: set[str] = set()
    task_ids: set[int] = set()
    seed_identities: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("final1000 controller entry is not an object")
        stage = BY_ID.get(str(entry.get("stage_id") or ""))
        task_id = entry.get("task_id")
        identity = (str(entry.get("bundle_id") or ""), int(entry.get("seed", -1)))
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
                and (isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0)
            )
        ):
            raise RuntimeError("final1000 controller ledger entry drifted")
        if entry["dedupe_key"] in dedupe or identity in seed_identities:
            raise RuntimeError("final1000 controller ledger contains duplicate work")
        dedupe.add(entry["dedupe_key"])
        seed_identities.add(identity)
        if task_id is not None:
            if task_id in task_ids:
                raise RuntimeError("one scheduler task maps to multiple entries")
            task_ids.add(task_id)
    for stage in STAGES:
        candidate = int(next_seeds[stage.stage_id])
        if not stage.seed_start <= candidate < stage.seed_window_end_exclusive:
            raise RuntimeError("final1000 next seed escaped its sealed window")
    return copy.deepcopy(dict(state))


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
    task = _refill_task(
        templates[stage_id], stage_id=stage_id, seed=seed, wave=wave
    )
    entry = _entry(task)
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
        task = _task_for_entry(
            entry, plan_index=plan_index, templates=templates
        )
        existing = scheduler.find_task_by_dedupe(entry["dedupe_key"])
        if existing is None:
            existing = scheduler.submit_task(task)
            submitted += 1
        else:
            reconciled += 1
        if str(existing.get("dedupe_key") or entry["dedupe_key"]) != entry[
            "dedupe_key"
        ]:
            raise RuntimeError("scheduler response changed final1000 dedupe identity")
        entry["task_id"] = _task_id(existing)
        entry["state"] = _scheduler_state(existing.get("status")) or "submitted"
    state["scheduler_submit_count"] = int(state["scheduler_submit_count"]) + submitted
    return submitted, reconciled


def _reconcile(
    state: dict[str, Any], scheduler: SchedulerClient
) -> int:
    changed = 0
    for entry in state["entries"]:
        if entry["task_id"] is None or entry["state"] in TERMINAL_STATES:
            continue
        observed = scheduler.get_task(int(entry["task_id"]))
        if observed is None:
            continue
        if str(observed.get("dedupe_key") or entry["dedupe_key"]) != entry[
            "dedupe_key"
        ]:
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
            and entry["task_id"] is not None
        ]
        stage_passed = False
        for entry in candidates:
            status = scheduler.read_seed_status(int(entry["task_id"]))
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
            reasons.append(f"{stage.stage_id}:remote_preflight_pending")
    state["canary_passed_stage_ids"] = sorted(passed)
    return passed, reasons


def _active_count(state: Mapping[str, Any], stage_id: str | None = None) -> int:
    return sum(
        entry["state"] in ACTIVE_STATES
        and (stage_id is None or entry["stage_id"] == stage_id)
        for entry in state["entries"]
    )


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

    canary_entries = [
        entry for entry in state["entries"] if entry["wave"] == "canary"
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
        for stage in STAGES:
            if stage.stage_id in passed:
                continue
            stage_canaries = [
                entry
                for entry in state["entries"]
                if entry["stage_id"] == stage.stage_id
                and entry["wave"] == "canary"
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
            actions.append(
                {
                    "action": "ramp_held",
                    "passed_stage_count": len(passed),
                    "reasons": reasons,
                    "replacement_submitted": replacement_submitted,
                    "replacement_reconciled": replacement_reconciled,
                }
            )
            return _result(state, actions, True, writes, scheduler)
        state["ramp_released"] = True
        persist()
        actions.append({"action": "ramp_released", "count": 496})

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
    for stage in STAGES:
        gap = stage.active_quota - _active_count(state, stage.stage_id)
        if gap < 0:
            raise RuntimeError(f"{stage.stage_id} exceeds its active quota")
        for _ in range(gap):
            refill_entries.append(
                _append_refill(
                    state, templates=templates, stage_id=stage.stage_id
                )
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
