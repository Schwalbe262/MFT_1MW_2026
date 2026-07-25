"""Restart-safe GET-only terminal watcher for sealed MFT diagnostic slots.

The watcher never submits, retries, cancels, or otherwise mutates Scheduler
state.  A successful terminal execution is consumed by the existing strict
diagnostic collector and authenticators.  Failed executions are classified in
an immutable ledger and are never represented as physical truth.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_strict_al_ingest as strict_al  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


WATCH_PLAN_SCHEMA = "mft-goal-terminal-success-watch-plan-v1"
SUCCESS_RECEIPT_SCHEMA = "mft-goal-terminal-success-receipt-v1"
FAILURE_LEDGER_SCHEMA = "mft-goal-terminal-failure-ledger-v1"
NDS_RECEIPT_SCHEMA = "mft-goal-terminal-nds-receipt-v1"
STATE_SCHEMA = "mft-goal-terminal-watcher-state-v1"
PID_SCHEMA = "mft-goal-terminal-watcher-pid-v1"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_PROJECT = "MFT_1MW_2026v1"
DEFAULT_POLL_SECONDS = 45
MIN_POLL_SECONDS = 30
MAX_POLL_SECONDS = 60
ACTIVE_STATUSES = frozenset(
    {"queued", "pending", "attaching", "starting", "running"}
)
FAILED_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "timed_out", "timeout"}
)


class WatcherContractError(RuntimeError):
    """Raised when immutable watcher authority or evidence drifts."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"at_utc": _now(), "event": event, **values},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WatcherContractError(f"JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise WatcherContractError(f"JSON artifact is not an object: {path}")
    return value


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise WatcherContractError("watcher payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != canonical_sha256(
        unsigned
    ):
        raise WatcherContractError(f"{schema} payload seal mismatch")
    return dict(value)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    try:
        return production._write_immutable_json(path.resolve(), value)
    except Exception as exc:
        raise WatcherContractError(
            f"immutable watcher output could not be written: {path}"
        ) from exc


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(production._json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


class SingleInstanceLock:
    """Process-lifetime advisory lock; the OS releases it after a crash."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.stream: Any | None = None

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise WatcherContractError(
                f"another watcher holds the lock: {self.path}"
            ) from exc
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def _retry_record(plan: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    keys = [
        name
        for name in (
            "retry_of_operational_pressure",
            "retry_of_dependency_failure",
            "retry_of_mesh_quality_canary",
            "retry_of_timeout12h",
            "retry_of_timeout",
        )
        if plan.get(name) is not None
    ]
    if len(keys) != 1 or not isinstance(plan.get(keys[0]), Mapping):
        raise WatcherContractError(
            "watched plan must have exactly one recognized retry authority"
        )
    return keys[0], plan[keys[0]]


def _logical_authority_id(plan: Mapping[str, Any]) -> int:
    _kind, record = _retry_record(plan)
    value = record.get("logical_authority_task_id")
    if value is None:
        value = record.get("retry_of_task_id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WatcherContractError("logical authority task ID is invalid")
    return value


def _slot_name(logical_id: int, execution_id: int) -> str:
    return f"l{logical_id}-t{execution_id}"


def _slot_paths(root: Path, slot: Mapping[str, Any]) -> dict[str, Path]:
    name = _slot_name(
        int(slot["logical_authority_task_id"]),
        int(slot["execution_task_id"]),
    )
    return {
        "directory": root / "slots" / name,
        "collection": root / "slots" / name / "collection.json",
        "receipt": root / "slots" / name / "success_receipt.json",
        "failure": root / "failures" / f"{name}.json",
    }


def _load_slot_source(
    expected_logical_id: int,
    submission_path: Path,
    *,
    scheduler_url: str,
    project: str,
) -> dict[str, Any]:
    raw_submission = production._read_json(submission_path.resolve(strict=True))
    plan_record = raw_submission.get("plan")
    if not isinstance(plan_record, Mapping):
        raise WatcherContractError("submission plan record is absent")
    plan_path = Path(str(plan_record.get("path") or ""))
    if production._file_record(plan_path) != plan_record:
        raise WatcherContractError("submission plan file record drifted")
    plan, _params, selected = diagnostic._load_collectible_plan(plan_path)
    submission = diagnostic._load_collectible_submission(
        submission_path, plan=plan
    )
    logical_id = _logical_authority_id(plan)
    if logical_id != expected_logical_id:
        raise WatcherContractError(
            f"logical slot mismatch: expected {expected_logical_id}, got {logical_id}"
        )
    if (
        submission.get("scheduler_url", "").rstrip("/")
        != scheduler_url.rstrip("/")
        or submission.get("project", project) != project
    ):
        raise WatcherContractError("submission Scheduler authority drifted")
    fixed = selected["row_contract"]["fixed_identity_attestation"]
    expected = fixed.get("expected")
    if (
        fixed.get("attested") is not True
        or not isinstance(expected, Mapping)
        or expected.get("fan_velocity") != 1.5
        or expected.get("fan_config") != "dual"
        or expected.get("thermal_pad_conductivity_W_mK") != 0.2
        or expected.get("core_plate_pad_t") != 2.0
        or expected.get("wcp_pad_t") != 2.0
    ):
        raise WatcherContractError("fixed fan/TIM/pad identity drifted")
    retry_kind, _record = _retry_record(plan)
    return {
        "logical_authority_task_id": logical_id,
        "execution_task_id": submission["task_id"],
        "task_name": submission["task_name"],
        "dedupe_key": submission["dedupe_key"],
        "retry_authority": retry_kind,
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "plan": production._file_record(plan_path),
        "plan_payload_sha256": plan["payload_sha256"],
        "submission": production._file_record(submission_path),
        "submission_payload_sha256": submission["payload_sha256"],
        "fixed_identity_attestation_sha256": fixed["sha256"],
    }


def initialize_watch_plan(
    *,
    output_root: Path,
    slot_specs: Sequence[tuple[int, Path]],
    scheduler_url: str,
    project: str,
    poll_seconds: int,
    code_revision: str,
) -> Path:
    if not MIN_POLL_SECONDS <= poll_seconds <= MAX_POLL_SECONDS:
        raise WatcherContractError("poll interval must be between 30 and 60 seconds")
    if not slot_specs:
        raise WatcherContractError("at least one exact logical slot is required")
    slots = [
        _load_slot_source(
            logical_id,
            submission,
            scheduler_url=scheduler_url,
            project=project,
        )
        for logical_id, submission in slot_specs
    ]
    logical_ids = [slot["logical_authority_task_id"] for slot in slots]
    execution_ids = [slot["execution_task_id"] for slot in slots]
    if len(set(logical_ids)) != len(slots) or len(set(execution_ids)) != len(slots):
        raise WatcherContractError("watch plan contains duplicate logical/execution slots")
    revision = str(code_revision).strip().lower()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise WatcherContractError("watcher code revision is not an exact git SHA")
    root = output_root.resolve()
    value = _sealed(
        {
            "schema_version": WATCH_PLAN_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "scheduler_url": scheduler_url.rstrip("/"),
            "scheduler_project": project,
            "scheduler_methods_allowed": ["GET"],
            "scheduler_submission_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
            "poll_seconds": poll_seconds,
            "output_root": str(root),
            "code_revision": revision,
            "exact_logical_slot_count": len(slots),
            "slots": sorted(
                slots, key=lambda item: item["logical_authority_task_id"]
            ),
            "fixed_boundary_policy": {
                "fan_velocity_m_s": 1.5,
                "fan_config": "dual",
                "thermal_pad_conductivity_W_mK": 0.2,
                "core_plate_pad_thickness_mm": 2.0,
                "wcp_pad_thickness_mm": 2.0,
                "mutable": False,
            },
        }
    )
    path = root / "watch_plan.json"
    if path.exists():
        if _read_json(path) != value:
            raise WatcherContractError("existing immutable watch plan differs")
        return path
    return _write_immutable(path, value)


def _load_watch_plan(path: Path) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), WATCH_PLAN_SCHEMA)
    root = Path(str(value.get("output_root") or "")).resolve()
    if root / "watch_plan.json" != path.resolve():
        raise WatcherContractError("watch plan output root/path drifted")
    poll = value.get("poll_seconds")
    slots = value.get("slots")
    if (
        not isinstance(poll, int)
        or not MIN_POLL_SECONDS <= poll <= MAX_POLL_SECONDS
        or not isinstance(slots, list)
        or len(slots) != value.get("exact_logical_slot_count")
    ):
        raise WatcherContractError("watch plan interval/slot count drifted")
    refreshed = []
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise WatcherContractError("watch plan slot is malformed")
        expected = _load_slot_source(
            int(slot["logical_authority_task_id"]),
            Path(slot["submission"]["path"]),
            scheduler_url=value["scheduler_url"],
            project=value["scheduler_project"],
        )
        if expected != slot:
            raise WatcherContractError("watched source artifact drifted")
        refreshed.append(expected)
    if len({item["logical_authority_task_id"] for item in refreshed}) != len(
        refreshed
    ):
        raise WatcherContractError("watch plan logical slots are not unique")
    return value


def _plan_with_authorized_extensions(
    base: Mapping[str, Any],
    *,
    watch_plan_path: Path,
    extension_authority_path: Path | None,
) -> dict[str, Any]:
    """Reauthenticate and merge locally authorized refill receipts."""
    merged = copy.deepcopy(dict(base))
    merged["authorized_extension_count"] = 0
    if extension_authority_path is None:
        return merged
    refill = importlib.import_module("tools.mft_goal_safe_refill")
    authority = refill.authenticate_extension_authority(
        extension_authority_path,
        watch_plan_path=watch_plan_path,
    )
    directory = Path(authority["extension_directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    extensions = []
    for receipt_path in sorted(directory.glob("*.json")):
        extensions.append(
            refill.authenticate_watcher_extension_receipt(
                receipt_path, authority=authority
            )
        )
    all_slots = [*merged["slots"], *extensions]
    logical = [slot["logical_authority_task_id"] for slot in all_slots]
    execution = [slot["execution_task_id"] for slot in all_slots]
    if len(set(logical)) != len(logical) or len(set(execution)) != len(
        execution
    ):
        raise WatcherContractError(
            "authorized watcher extensions duplicate an exact slot"
        )
    merged["slots"] = sorted(
        all_slots, key=lambda item: item["logical_authority_task_id"]
    )
    merged["exact_logical_slot_count"] = len(all_slots)
    merged["authorized_extension_count"] = len(extensions)
    merged["extension_authority"] = production._file_record(
        extension_authority_path
    )
    return merged


def _validate_snapshot(
    snapshot: Mapping[str, Any],
    slot: Mapping[str, Any],
    *,
    project: str,
) -> None:
    task_id = snapshot.get("task_id", snapshot.get("id"))
    if (
        task_id != slot["execution_task_id"]
        or snapshot.get("name") != slot["task_name"]
        or snapshot.get("project") != project
        or snapshot.get("dedupe_key") != slot["dedupe_key"]
    ):
        raise WatcherContractError("Scheduler GET snapshot identity drifted")


def _terminal_kind(snapshot: Mapping[str, Any]) -> str:
    status = str(snapshot.get("status") or "").strip().lower()
    state = str(snapshot.get("state") or "").strip().lower()
    exit_code = snapshot.get("exit_code")
    if status == "completed":
        if state == "succeeded" and exit_code == 0:
            return "success"
        return "failure"
    if status in FAILED_STATUSES or state in FAILED_STATUSES:
        return "failure"
    if snapshot.get("finished_at") and exit_code is not None:
        return "failure"
    if status in ACTIVE_STATUSES or state in ACTIVE_STATUSES:
        return "active"
    return "unknown"


def _authenticate_slot_collection(
    collection_path: Path,
    slot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    view = diagnostic.authenticate_collection(collection_path)
    strict = strict_al.authenticate_collection(collection_path)
    collection = view["collection"]
    plan = view["plan"]
    if (
        collection.get("task_id") != slot["execution_task_id"]
        or collection.get("plan") != slot["plan"]
        or collection.get("submission") != slot["submission"]
        or _logical_authority_id(plan) != slot["logical_authority_task_id"]
        or strict.collection_file_sha256
        != production._sha256_file(collection_path)
    ):
        raise WatcherContractError("authenticated collection escaped exact slot")
    truth, status = promotion._actual_standard_observation(
        view, collection_path=collection_path
    )
    if status["actual_truth_feasible"] is True:
        if promotion._actual_standard_truth(
            view, collection_path=collection_path
        ) != truth:
            raise WatcherContractError("strict feasible truth recomputation drifted")
    actual_fixed = truth.get("actual_fixed_identity_attestation")
    selected_fixed = truth.get("selected_fixed_identity_attestation")
    if (
        not isinstance(actual_fixed, Mapping)
        or not isinstance(selected_fixed, Mapping)
        or actual_fixed.get("sha256")
        != slot["fixed_identity_attestation_sha256"]
        or selected_fixed.get("sha256")
        != slot["fixed_identity_attestation_sha256"]
    ):
        raise WatcherContractError("authenticated fan/TIM/pad identity drifted")
    strict_summary = {
        "adapter_kind": strict.adapter_kind,
        "collection_file_sha256": strict.collection_file_sha256,
        "solver_revision": strict.solver_revision,
        "library_revision": strict.library_revision,
        "source_task_payload_sha256": strict.source_task_payload_sha256,
        "source_seed": strict.source_seed,
        "source_fixed_primary_turns": strict.source_fixed_primary_turns,
    }
    return truth, status, strict_summary


def _success_receipt(
    *,
    slot: Mapping[str, Any],
    collection_path: Path,
) -> dict[str, Any]:
    truth, status, strict_summary = _authenticate_slot_collection(
        collection_path, slot
    )
    return _sealed(
        {
            "schema_version": SUCCESS_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "logical_authority_task_id": slot["logical_authority_task_id"],
            "execution_task_id": slot["execution_task_id"],
            "collection": production._file_record(collection_path),
            "strict_authentication": strict_summary,
            "hard_constraint_status": status,
            "actual_truth": truth,
            "eligible_for_global_truth_nds": status["actual_truth_feasible"],
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "fixed_boundary_policy_preserved": True,
            "authenticated_at_utc": _now(),
        }
    )


def _load_success_receipt(
    path: Path,
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), SUCCESS_RECEIPT_SCHEMA)
    collection = Path(str((value.get("collection") or {}).get("path") or ""))
    recomputed = _success_receipt(slot=slot, collection_path=collection)
    for key in (
        "logical_authority_task_id",
        "execution_task_id",
        "collection",
        "strict_authentication",
        "hard_constraint_status",
        "actual_truth",
        "eligible_for_global_truth_nds",
        "scheduler_get_only",
        "scheduler_mutation_performed",
        "fixed_boundary_policy_preserved",
    ):
        if value.get(key) != recomputed.get(key):
            raise WatcherContractError("success receipt reauthentication drifted")
    return value


def _write_failure(
    *,
    path: Path,
    slot: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Path:
    value = _sealed(
        {
            "schema_version": FAILURE_LEDGER_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "logical_authority_task_id": slot["logical_authority_task_id"],
            "execution_task_id": slot["execution_task_id"],
            "classification": "terminal_execution_not_successful",
            "scheduler_snapshot": copy.deepcopy(dict(snapshot)),
            "scheduler_snapshot_sha256": canonical_sha256(snapshot),
            "collection_performed": False,
            "physical_truth_claimed": False,
            "global_truth_nds_included": False,
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "classified_at_utc": _now(),
        }
    )
    return _write_immutable(path, value)


def _nds_key(receipts: Sequence[Mapping[str, Any]]) -> str:
    authority = [
        {
            "logical_authority_task_id": item["logical_authority_task_id"],
            "collection_payload_sha256": item["collection"]["sha256"],
        }
        for item in sorted(
            receipts, key=lambda item: item["logical_authority_task_id"]
        )
    ]
    return canonical_sha256(authority)[:16]


def _promote_nds(
    *,
    root: Path,
    receipts: Sequence[Mapping[str, Any]],
    promoter: Callable[..., Path],
) -> Path | None:
    eligible = [
        receipt
        for receipt in receipts
        if receipt["eligible_for_global_truth_nds"] is True
    ]
    if not eligible:
        return None
    key = _nds_key(eligible)
    directory = root / "truth_snapshots" / f"n{len(eligible):02d}-{key}"
    manifest = directory / "truth_pareto_manifest.json"
    receipt_path = root / "truth_snapshots" / f"n{len(eligible):02d}-{key}.receipt.json"
    collections = [
        Path(item["collection"]["path"])
        for item in sorted(
            eligible, key=lambda item: item["logical_authority_task_id"]
        )
    ]
    if not manifest.exists():
        promoter(standard_collection_paths=collections, output=directory)
    manifest_value = production._validate_seal(
        production._read_json(manifest), promotion.TRUTH_MANIFEST_SCHEMA
    )
    if manifest_value.get("input_collection_count") != len(collections):
        raise WatcherContractError("global truth NDS input count drifted")
    if not receipt_path.exists():
        _write_immutable(
            receipt_path,
            _sealed(
                {
                    "schema_version": NDS_RECEIPT_SCHEMA,
                    "campaign_id": "mft-goal-20260726",
                    "authority_key": key,
                    "logical_authority_task_ids": [
                        item["logical_authority_task_id"]
                        for item in sorted(
                            eligible,
                            key=lambda item: item[
                                "logical_authority_task_id"
                            ],
                        )
                    ],
                    "input_collections": [
                        production._file_record(path) for path in collections
                    ],
                    "truth_manifest": production._file_record(manifest),
                    "global_nondominated_sort_performed": True,
                    "scheduler_mutation_performed": False,
                    "created_at_utc": _now(),
                }
            ),
        )
    else:
        _validate_seal(_read_json(receipt_path), NDS_RECEIPT_SCHEMA)
    return manifest


def process_cycle(
    watch_plan: Mapping[str, Any],
    *,
    task_reader: Callable[..., dict[str, Any]] = diagnostic._scheduler_task_snapshot,
    collector: Callable[..., Path] = diagnostic.collect_standard,
    promoter: Callable[..., Path] = promotion.create_truth_promotion,
) -> dict[str, Any]:
    root = Path(watch_plan["output_root"])
    receipts: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for slot in watch_plan["slots"]:
        paths = _slot_paths(root, slot)
        state = {
            "logical_authority_task_id": slot["logical_authority_task_id"],
            "execution_task_id": slot["execution_task_id"],
        }
        try:
            if paths["failure"].exists():
                failure = _validate_seal(
                    _read_json(paths["failure"]), FAILURE_LEDGER_SCHEMA
                )
                state.update(
                    {"state": "terminal_failure", "ledger": str(paths["failure"])}
                )
                if failure.get("physical_truth_claimed") is not False:
                    raise WatcherContractError("failure ledger claims physical truth")
            elif paths["receipt"].exists():
                receipt = _load_success_receipt(paths["receipt"], slot)
                receipts.append(receipt)
                state.update(
                    {
                        "state": (
                            "authenticated_feasible"
                            if receipt["eligible_for_global_truth_nds"]
                            else "authenticated_infeasible"
                        ),
                        "receipt": str(paths["receipt"]),
                    }
                )
            elif paths["collection"].exists():
                receipt = _success_receipt(
                    slot=slot, collection_path=paths["collection"]
                )
                _write_immutable(paths["receipt"], receipt)
                receipts.append(receipt)
                state.update(
                    {
                        "state": (
                            "authenticated_feasible"
                            if receipt["eligible_for_global_truth_nds"]
                            else "authenticated_infeasible"
                        ),
                        "receipt": str(paths["receipt"]),
                        "recovered_after_restart": True,
                    }
                )
            else:
                snapshot = task_reader(
                    scheduler_url=watch_plan["scheduler_url"],
                    task_id=slot["execution_task_id"],
                )
                _validate_snapshot(
                    snapshot, slot, project=watch_plan["scheduler_project"]
                )
                kind = _terminal_kind(snapshot)
                if kind == "success":
                    collector(
                        plan_path=Path(slot["plan"]["path"]),
                        submission_path=Path(slot["submission"]["path"]),
                        output=paths["collection"],
                        scheduler_url=watch_plan["scheduler_url"],
                    )
                    receipt = _success_receipt(
                        slot=slot, collection_path=paths["collection"]
                    )
                    _write_immutable(paths["receipt"], receipt)
                    receipts.append(receipt)
                    state.update(
                        {
                            "state": (
                                "authenticated_feasible"
                                if receipt["eligible_for_global_truth_nds"]
                                else "authenticated_infeasible"
                            ),
                            "receipt": str(paths["receipt"]),
                        }
                    )
                elif kind == "failure":
                    _write_failure(
                        path=paths["failure"], slot=slot, snapshot=snapshot
                    )
                    state.update(
                        {
                            "state": "terminal_failure",
                            "ledger": str(paths["failure"]),
                        }
                    )
                else:
                    state.update(
                        {
                            "state": kind,
                            "scheduler_status": snapshot.get("status"),
                            "scheduler_state": snapshot.get("state"),
                        }
                    )
        except Exception as exc:
            state.update(
                {
                    "state": "transient_or_contract_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _log(
                "slot_error",
                logical_authority_task_id=slot["logical_authority_task_id"],
                execution_task_id=slot["execution_task_id"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
        states.append(state)
    nds_manifest = None
    try:
        nds_manifest = _promote_nds(
            root=root, receipts=receipts, promoter=promoter
        )
    except Exception as exc:
        _log("nds_error", error_type=type(exc).__name__, error=str(exc))
    state_value = _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "watch_plan_payload_sha256": watch_plan["payload_sha256"],
            "observed_at_utc": _now(),
            "slots": states,
            "counts": {
                name: sum(item["state"] == name for item in states)
                for name in sorted({item["state"] for item in states})
            },
            "latest_truth_manifest": (
                str(nds_manifest) if nds_manifest is not None else None
            ),
            "scheduler_methods_used": ["GET"],
            "scheduler_submission_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
        }
    )
    _write_atomic(root / "state.json", state_value)
    _log("cycle_complete", counts=state_value["counts"])
    return state_value


def run_watcher(
    path: Path,
    *,
    once: bool = False,
    extension_authority_path: Path | None = None,
) -> None:
    plan = _load_watch_plan(path)
    root = Path(plan["output_root"])
    with SingleInstanceLock(root / "watcher.lock"):
        _write_atomic(
            root / "watcher.pid.json",
            _sealed(
                {
                    "schema_version": PID_SCHEMA,
                    "pid": os.getpid(),
                    "watch_plan": production._file_record(path),
                    "extension_authority": (
                        production._file_record(extension_authority_path)
                        if extension_authority_path is not None
                        else None
                    ),
                    "started_at_utc": _now(),
                    "scheduler_methods_allowed": ["GET"],
                    "scheduler_mutation_performed": False,
                }
            ),
        )
        _log(
            "watcher_started",
            pid=os.getpid(),
            poll_seconds=plan["poll_seconds"],
            slot_count=len(plan["slots"]),
        )
        while True:
            effective = _plan_with_authorized_extensions(
                plan,
                watch_plan_path=path,
                extension_authority_path=extension_authority_path,
            )
            process_cycle(effective)
            if once:
                return
            time.sleep(plan["poll_seconds"])


def _parse_slot(value: str) -> tuple[int, Path]:
    logical, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("slot must be LOGICAL_ID=SUBMISSION_PATH")
    try:
        logical_id = int(logical)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("slot logical ID must be an integer") from exc
    if logical_id <= 0 or not path.strip():
        raise argparse.ArgumentTypeError("slot logical ID/path is invalid")
    return logical_id, Path(path)


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={REPOSITORY_ROOT}", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GET-only restart-safe MFT diagnostic terminal watcher"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--slot", type=_parse_slot, action="append", required=True)
    init.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    init.add_argument("--project", default=DEFAULT_PROJECT)
    init.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    init.add_argument("--code-revision", default="")
    run = commands.add_parser("run")
    run.add_argument("--watch-plan", type=Path, required=True)
    run.add_argument("--extension-authority", type=Path)
    once = commands.add_parser("once")
    once.add_argument("--watch-plan", type=Path, required=True)
    once.add_argument("--extension-authority", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        path = initialize_watch_plan(
            output_root=args.output_root,
            slot_specs=args.slot,
            scheduler_url=args.scheduler_url,
            project=args.project,
            poll_seconds=args.poll_seconds,
            code_revision=args.code_revision or _git_revision(),
        )
        print(path)
        return 0
    run_watcher(
        args.watch_plan,
        once=args.command == "once",
        extension_authority_path=args.extension_authority,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
