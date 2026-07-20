"""Fail-closed fast-ramp and open-ended refill controller for current7 seeds.

Only ``POST /api/tasks`` is used for scheduler mutation.  The controller is a
dry-run unless ``apply=True``; dry-run neither writes controller state nor calls
the scheduler's POST endpoint.  A hash-chained state file plus scheduler dedupe
reconciliation makes every submission restartable.
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
import urllib.error
import urllib.parse
import urllib.request

try:
    from tier1_corrected_current7_receipt import (
        canonical_sha256,
        validate_adapter_receipt,
    )
    from tier1_corrected_current7_slurm_bundle import (
        ACTIVE_LEDGER_STATES,
        TERMINAL_LEDGER_STATES,
        assess_fast_ramp,
        build_task_payload,
        build_task_waves,
        initial_seed_ledger,
        plan_refill_wave,
        read_json,
        seal_seed_ledger,
        validate_seed_ledger,
    )
    from tier1_corrected_current7_slurm_publish import (
        PublicationTransport,
        _ready_matches,
        load_bundle_plan,
        scheduler_publication_transport,
        validate_publication_receipt,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import (
        canonical_sha256,
        validate_adapter_receipt,
    )
    from tools.tier1_corrected_current7_slurm_bundle import (
        ACTIVE_LEDGER_STATES,
        TERMINAL_LEDGER_STATES,
        assess_fast_ramp,
        build_task_payload,
        build_task_waves,
        initial_seed_ledger,
        plan_refill_wave,
        read_json,
        seal_seed_ledger,
        validate_seed_ledger,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        PublicationTransport,
        _ready_matches,
        load_bundle_plan,
        scheduler_publication_transport,
        validate_publication_receipt,
    )


CONTROLLER_STATE_SCHEMA = "mft-tier1-current7-slurm-controller-state-v1"
CONTROLLER_RESULT_SCHEMA = "mft-tier1-current7-slurm-controller-result-v1"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class ReadyProbe(Protocol):
    read_count: int

    def read_ready(self, remote_bundle: str) -> Mapping[str, Any] | None: ...


class SchedulerClient(Protocol):
    post_count: int

    def find_task_by_dedupe(self, dedupe_key: str) -> Mapping[str, Any] | None: ...

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def get_task(self, task_id: int) -> Mapping[str, Any] | None: ...

    def read_seed_status(self, task_id: int) -> Mapping[str, Any] | None: ...


class TransportReadyProbe:
    def __init__(self, transport: PublicationTransport):
        self.transport = transport
        self.read_count = 0

    def read_ready(self, remote_bundle: str) -> Mapping[str, Any] | None:
        self.read_count += 1
        path = remote_bundle.rstrip("/") + "/READY.json"
        if not self.transport.exists(path):
            return None
        payload = json.loads(self.transport.read_bytes(path).decode("utf-8"))
        return payload if isinstance(payload, dict) else None


class SchedulerApiClient:
    """Small HTTP client whose sole mutating method is POST /api/tasks."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.post_count = 0
        self._inventory: dict[str, Mapping[str, Any]] | None = None

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"scheduler API {method} {path} failed: {detail}"
            ) from exc
        value = json.loads(raw.decode("utf-8"))
        return value

    def _load_inventory(self) -> None:
        query = urllib.parse.urlencode({"name_prefix": "mft-t1c7-", "limit": 10000})
        value = self._request(f"/api/tasks?{query}")
        if not isinstance(value, list):
            raise RuntimeError("scheduler task inventory is not a list")
        inventory: dict[str, Mapping[str, Any]] = {}
        for task in value:
            if not isinstance(task, dict):
                continue
            dedupe = str(task.get("dedupe_key") or "")
            if not dedupe:
                continue
            if dedupe in inventory and int(inventory[dedupe]["id"]) != int(task["id"]):
                raise RuntimeError(
                    f"scheduler contains duplicate dedupe identity: {dedupe}"
                )
            inventory[dedupe] = task
        self._inventory = inventory

    def find_task_by_dedupe(self, dedupe_key: str) -> Mapping[str, Any] | None:
        if self._inventory is None:
            self._load_inventory()
        return self._inventory.get(dedupe_key)  # type: ignore[union-attr]

    def submit_task(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if "requested_allocation_id" in payload:
            raise RuntimeError(
                "requested_allocation_id is prohibited for current7 seeds"
            )
        response = self._request("/api/tasks", method="POST", payload=payload)
        self.post_count += 1
        if not isinstance(response, dict):
            raise RuntimeError("scheduler submit response is not an object")
        dedupe = str(payload["dedupe_key"])
        if self._inventory is None:
            self._inventory = {}
        self._inventory[dedupe] = response
        return response

    def get_task(self, task_id: int) -> Mapping[str, Any] | None:
        try:
            value = self._request(f"/api/tasks/{int(task_id)}")
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise
        return value if isinstance(value, dict) else None

    def read_seed_status(self, task_id: int) -> Mapping[str, Any] | None:
        path = f"runs/task-{int(task_id)}/seed_status.json"
        query = urllib.parse.urlencode({"path": path, "base": "remote_cwd"})
        request = urllib.request.Request(
            self.base_url + f"/api/tasks/{int(task_id)}/remote-file?{query}",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 409}:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"scheduler remote status read failed: {detail}"
            ) from exc
        if not raw.strip():
            return None
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None


def _seal_controller_state(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != "state_sha256"
    }
    return {**unsigned, "state_sha256": canonical_sha256(unsigned)}


def _initial_state(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    priority: int | None,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": CONTROLLER_STATE_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "publication_receipt_sha256": publication["receipt_sha256"],
        "revision": 0,
        "parent_state_sha256": None,
        "stop_requested": False,
        "ramp_released": False,
        "ramp_gate": None,
        "priority_override": priority,
        "ledger": initial_seed_ledger(plan, manifest, priority=priority),
        "submissions": {},
        "scheduler_mutation_endpoint": "/api/tasks",
        "scheduler_submit_count": 0,
    }
    return _seal_controller_state(unsigned)


def _validate_state(
    state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {key: item for key, item in state.items() if key != "state_sha256"}
    if (
        state.get("schema_version") != CONTROLLER_STATE_SCHEMA
        or state.get("bundle_id") != plan.get("bundle_id")
        or state.get("bundle_manifest_sha256") != plan.get("bundle_manifest_sha256")
        or state.get("publication_receipt_sha256") != publication.get("receipt_sha256")
        or state.get("scheduler_mutation_endpoint") != "/api/tasks"
        or state.get("state_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("controller state identity/SHA mismatch")
    ledger = validate_seed_ledger(manifest, state.get("ledger") or {})
    submissions = state.get("submissions")
    if not isinstance(submissions, dict):
        raise RuntimeError("controller submissions map is invalid")
    ledger_dedupe = {entry["dedupe_key"]: entry for entry in ledger["entries"]}
    task_ids = []
    for dedupe, record in submissions.items():
        if dedupe not in ledger_dedupe or not isinstance(record, dict):
            raise RuntimeError("controller submission is absent from seed ledger")
        if record.get("dedupe_key") != dedupe:
            raise RuntimeError("controller submission dedupe identity mismatch")
        task_ids.append(int(record["task_id"]))
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("controller state maps one scheduler task more than once")
    return copy.deepcopy(dict(state))


def _advance_state(state: dict[str, Any]) -> dict[str, Any]:
    previous = str(state["state_sha256"])
    state = copy.deepcopy(state)
    state["revision"] = int(state["revision"]) + 1
    state["parent_state_sha256"] = previous
    state.pop("state_sha256", None)
    return _seal_controller_state(state)


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(_json_bytes(state))
    os.replace(temporary, path)


def _receipt_launch_gate(
    *, manifest: Mapping[str, Any], source_map: Mapping[str, Path]
) -> None:
    relative = str(manifest["adapter_receipt"]["path"])
    source = source_map.get(relative)
    if source is None:
        raise RuntimeError("adapter receipt is absent from local bundle sources")
    identity = validate_adapter_receipt(read_json(source)).to_dict()
    expected = manifest["adapter_receipt"]["identity"]
    repair_contracts = identity.get(
        "optimizer_repair_contract_sha256_by_fixed_primary_turns"
    )
    mandatory = (
        identity == expected
        and identity.get("launch_eligible") is True
        and identity.get("offspring_physics_repair") is True
        and identity.get("fixed_primary_turns_supported") == [5, 6]
        and identity.get("initial_repair_attested") is True
        and identity.get("warm_repair_attested") is True
        and identity.get("every_offspring_decode_repair_attested") is True
        and identity.get("terminal_physical_replay_attested") is True
        and isinstance(repair_contracts, dict)
        and set(repair_contracts) == {"5", "6"}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in repair_contracts.values()
        )
    )
    if not mandatory:
        raise RuntimeError(
            "current7 controller blocked: receipt lacks repair=true N1=5|6 launch gate"
        )


def _verify_ready_gate(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    publication: Mapping[str, Any],
    ready_probe: ReadyProbe | None,
    apply: bool,
) -> None:
    validate_publication_receipt(publication, plan=plan, manifest=manifest)
    if ready_probe is None:
        if apply:
            raise RuntimeError("--apply requires a live remote READY probe")
        return
    live = ready_probe.read_ready(str(plan["remote_bundle"]))
    if (
        live is None
        or dict(live) != publication.get("ready")
        or not _ready_matches(live, plan, manifest)
    ):
        raise RuntimeError("live remote READY does not match publication receipt")


def _ledger_with_states(
    manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    updates: Mapping[str, str],
    *,
    stop_requested: bool | None = None,
) -> dict[str, Any]:
    ledger = validate_seed_ledger(manifest, ledger)
    entries = []
    changed = False
    for original in ledger["entries"]:
        entry = dict(original)
        desired = updates.get(entry["dedupe_key"])
        if desired is not None and desired != entry["state"]:
            current = str(entry["state"])
            if current in TERMINAL_LEDGER_STATES and desired != current:
                raise RuntimeError("terminal seed ledger state cannot change")
            if desired not in ACTIVE_LEDGER_STATES | TERMINAL_LEDGER_STATES:
                raise RuntimeError("unknown seed ledger transition")
            entry["state"] = desired
            changed = True
        entries.append(entry)
    desired_stop = (
        bool(ledger["stop_requested"])
        if stop_requested is None
        else bool(stop_requested)
    )
    if desired_stop != bool(ledger["stop_requested"]):
        changed = True
    if not changed:
        return dict(ledger)
    return seal_seed_ledger(
        manifest,
        entries,
        revision=int(ledger["revision"]) + 1,
        parent_ledger_sha256=ledger["ledger_sha256"],
        stop_requested=desired_stop,
    )


def _scheduler_ledger_state(status: str) -> str | None:
    return {
        "queued": "queued",
        "attaching": "queued",
        "running": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(status)


def _validate_task_envelope(
    task: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    required = set(manifest["scheduler_api_contract"]["required_fields"])
    if set(task) != required or "requested_allocation_id" in task:
        raise RuntimeError("current7 scheduler envelope field contract mismatch")
    if (
        task.get("aedt_backend") != "standalone"
        or task.get("gpus") != 0
        or task.get("cpus") != 8
        or task.get("memory_mb") != 65_536
        or task.get("max_workers_per_node") != 4
        or task.get("scheduling_profile") != "standard"
    ):
        raise RuntimeError("current7 scheduler envelope resource contract mismatch")


def control_once(
    plan_path: Path,
    publication_receipt: Mapping[str, Any],
    *,
    state_path: Path,
    apply: bool = False,
    scheduler: SchedulerClient | None = None,
    ready_probe: ReadyProbe | None = None,
    priority: int | None = None,
    request_stop: bool = False,
    stop_probe: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one idempotent reconciliation tick."""

    plan, manifest, source_map = load_bundle_plan(plan_path)
    _receipt_launch_gate(manifest=manifest, source_map=source_map)
    publication = validate_publication_receipt(
        publication_receipt, plan=plan, manifest=manifest
    )
    _verify_ready_gate(
        plan=plan,
        manifest=manifest,
        publication=publication,
        ready_probe=ready_probe,
        apply=apply,
    )
    if apply and scheduler is None:
        raise RuntimeError("--apply requires an explicit scheduler API client")

    state_created = not state_path.is_file()
    if not state_created:
        state = _validate_state(
            read_json(state_path),
            plan=plan,
            manifest=manifest,
            publication=publication,
        )
        if state.get("priority_override") != priority:
            raise RuntimeError("controller priority override changed across restart")
    else:
        state = _initial_state(plan, manifest, publication, priority=priority)
        if apply:
            _write_state(state_path, state)

    actions: list[dict[str, Any]] = []
    state_writes = 1 if apply and state_created else 0

    def persist() -> None:
        nonlocal state, state_writes
        state = _advance_state(state)
        if apply:
            _write_state(state_path, state)
            state_writes += 1

    def stop_is_requested() -> bool:
        return (
            bool(state["stop_requested"])
            or bool(request_stop)
            or bool(stop_probe is not None and stop_probe())
        )

    def honor_stop() -> bool:
        if not stop_is_requested():
            return False
        changed = not state["stop_requested"] or not state["ledger"]["stop_requested"]
        state["stop_requested"] = True
        state["ledger"] = _ledger_with_states(
            manifest, state["ledger"], {}, stop_requested=True
        )
        if changed:
            persist()
        actions.append({"action": "stop", "reason": "explicit_stop_requested"})
        return True

    if honor_stop():
        return _controller_result(
            state=state,
            actions=actions,
            apply=apply,
            state_writes=state_writes,
            scheduler=scheduler,
        )

    waves = build_task_waves(plan, manifest, priority=priority)
    tasks_by_dedupe = {
        task["dedupe_key"]: task for task in [*waves["canaries"], *waves["ramp"]]
    }

    def reserve_submission(
        task: Mapping[str, Any],
        scheduler_task: Mapping[str, Any],
        *,
        submitted_by_this_controller: bool = False,
    ) -> None:
        nonlocal state
        dedupe = str(task["dedupe_key"])
        task_id = int(scheduler_task.get("task_id") or scheduler_task.get("id") or 0)
        if task_id <= 0:
            raise RuntimeError("scheduler response has no positive task id")
        response_dedupe = str(scheduler_task.get("dedupe_key") or dedupe)
        if response_dedupe != dedupe:
            raise RuntimeError("scheduler response dedupe identity mismatch")
        state["submissions"][dedupe] = {
            "dedupe_key": dedupe,
            "task_id": task_id,
            "name": task["name"],
            "seed": int(task["payload_json"]["seed"]),
            "island_id": task["payload_json"]["lane"]["island_id"],
            "wave": task["payload_json"]["lane"]["wave"],
        }
        state["ledger"] = _ledger_with_states(
            manifest, state["ledger"], {dedupe: "submitted"}
        )
        if submitted_by_this_controller:
            state["scheduler_submit_count"] = int(state["scheduler_submit_count"]) + 1
        persist()

    def ensure_submitted(task: Mapping[str, Any]) -> bool:
        nonlocal state
        _validate_task_envelope(task, manifest)
        dedupe = str(task["dedupe_key"])
        if dedupe in state["submissions"]:
            return True
        if honor_stop():
            return False
        existing = (
            scheduler.find_task_by_dedupe(dedupe) if scheduler is not None else None
        )
        if existing is not None:
            reserve_submission(task, existing)
            actions.append(
                {
                    "action": "reconciled",
                    "dedupe_key": dedupe,
                    "task_id": int(existing.get("task_id") or existing.get("id")),
                }
            )
            return True
        if not apply:
            actions.append(
                {
                    "action": "would_submit",
                    "dedupe_key": dedupe,
                    "seed": int(task["payload_json"]["seed"]),
                    "wave": task["payload_json"]["lane"]["wave"],
                }
            )
            return False
        if scheduler is None:  # pragma: no cover - guarded above
            raise RuntimeError("scheduler unavailable")
        response = scheduler.submit_task(task)
        reserve_submission(task, response, submitted_by_this_controller=True)
        actions.append(
            {
                "action": "submitted",
                "dedupe_key": dedupe,
                "task_id": int(response.get("task_id") or response.get("id")),
                "seed": int(task["payload_json"]["seed"]),
                "wave": task["payload_json"]["lane"]["wave"],
            }
        )
        return True

    for task in waves["canaries"]:
        ensure_submitted(task)
    if not apply and len(state["submissions"]) < 4:
        return _controller_result(
            state=state,
            actions=actions,
            apply=False,
            state_writes=state_writes,
            scheduler=scheduler,
        )
    if honor_stop():
        return _controller_result(
            state=state,
            actions=actions,
            apply=apply,
            state_writes=state_writes,
            scheduler=scheduler,
        )

    canary_statuses = []
    for task in waves["canaries"]:
        submission = state["submissions"].get(task["dedupe_key"])
        status = (
            scheduler.read_seed_status(int(submission["task_id"]))
            if scheduler is not None and submission is not None
            else None
        )
        if status is not None:
            canary_statuses.append(status)
    assessment = assess_fast_ramp(manifest, canary_statuses)
    state["ramp_gate"] = assessment
    if not state["ramp_released"] and assessment["eligible"]:
        state["ramp_released"] = True
        persist()
        actions.append({"action": "ramp_released", "count": 32})
    elif not state["ramp_released"]:
        actions.append({"action": "ramp_held", "reasons": assessment["reasons"]})
        return _controller_result(
            state=state,
            actions=actions,
            apply=apply,
            state_writes=state_writes,
            scheduler=scheduler,
        )

    # Ramp and any refill entries reserved before an interrupted controller
    # invocation are submitted immediately once the gate has been sealed.
    for entry in list(state["ledger"]["entries"]):
        if entry["state"] != "planned" or entry["wave"] == "canary":
            continue
        task = tasks_by_dedupe.get(entry["dedupe_key"])
        if task is None:
            task = build_task_payload(
                plan, manifest, seed=int(entry["seed"]), priority=priority
            )
            tasks_by_dedupe[task["dedupe_key"]] = task
        ensure_submitted(task)
    if honor_stop():
        return _controller_result(
            state=state,
            actions=actions,
            apply=apply,
            state_writes=state_writes,
            scheduler=scheduler,
        )

    updates: dict[str, str] = {}
    if scheduler is not None:
        for dedupe, record in state["submissions"].items():
            task = scheduler.get_task(int(record["task_id"]))
            if task is None:
                continue
            response_dedupe = str(task.get("dedupe_key") or dedupe)
            if response_dedupe != dedupe:
                raise RuntimeError("scheduler task changed dedupe identity")
            mapped = _scheduler_ledger_state(str(task.get("status") or ""))
            if mapped is not None:
                updates[dedupe] = mapped
    next_ledger = _ledger_with_states(manifest, state["ledger"], updates)
    if next_ledger["ledger_sha256"] != state["ledger"]["ledger_sha256"]:
        state["ledger"] = next_ledger
        persist()

    refill = plan_refill_wave(plan, manifest, state["ledger"], priority=priority)
    if refill["eligible"]:
        state["ledger"] = refill["next_ledger"]
        persist()
        actions.append(
            {"action": "refill_reserved", "count": int(refill["refill_count"])}
        )
        for task in refill["tasks"]:
            tasks_by_dedupe[task["dedupe_key"]] = task
            ensure_submitted(task)
    else:
        actions.append({"action": "refill_idle", "reason": refill["reason"]})
    return _controller_result(
        state=state,
        actions=actions,
        apply=apply,
        state_writes=state_writes,
        scheduler=scheduler,
    )


def _controller_result(
    *,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    apply: bool,
    state_writes: int,
    scheduler: SchedulerClient | None,
) -> dict[str, Any]:
    active = sum(
        entry["state"] in ACTIVE_LEDGER_STATES for entry in state["ledger"]["entries"]
    )
    terminal = sum(
        entry["state"] in TERMINAL_LEDGER_STATES for entry in state["ledger"]["entries"]
    )
    return {
        "schema_version": CONTROLLER_RESULT_SCHEMA,
        "bundle_id": state["bundle_id"],
        "apply": bool(apply),
        "state_revision": int(state["revision"]),
        "state_sha256": state["state_sha256"],
        "state_writes": int(state_writes),
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "stop_requested": bool(state["stop_requested"]),
        "ramp_released": bool(state["ramp_released"]),
        "submission_count": len(state["submissions"]),
        "ledger_seed_count": len(state["ledger"]["entries"]),
        "ledger_active_count": active,
        "ledger_terminal_count": terminal,
        "ledger_sha256": state["ledger"]["ledger_sha256"],
        "actions": [dict(action) for action in actions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--priority", type=int)
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
    args = parser.parse_args()
    if args.watch and not args.apply:
        raise RuntimeError("--watch requires --apply")
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    publication = read_json(args.publication_receipt)
    scheduler = SchedulerApiClient(args.scheduler_url) if args.apply else None

    def stop_probe() -> bool:
        return bool(args.stop_file is not None and args.stop_file.is_file())

    if args.apply:
        context = scheduler_publication_transport(
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            account_name=args.publication_account,
        )
        with context as transport:
            ready_probe = TransportReadyProbe(transport)
            while True:
                result = control_once(
                    args.plan,
                    publication,
                    state_path=args.state,
                    apply=True,
                    scheduler=scheduler,
                    ready_probe=ready_probe,
                    priority=args.priority,
                    request_stop=args.request_stop,
                    stop_probe=stop_probe,
                )
                print(
                    json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True
                )
                if not args.watch or result["stop_requested"]:
                    break
                time.sleep(args.poll_seconds)
    else:
        result = control_once(
            args.plan,
            publication,
            state_path=args.state,
            apply=False,
            scheduler=None,
            ready_probe=None,
            priority=args.priority,
            request_stop=args.request_stop,
            stop_probe=stop_probe,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
