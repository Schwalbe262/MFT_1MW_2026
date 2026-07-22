"""Production read-only consumer for mixed final1000 v1/v2 Scheduler lanes.

The consumer performs GET-only Scheduler inventory reads and read-only SFTP
stable reads.  Its only mutations are local, content-addressed result/cache
objects plus atomic condition-index pointers beneath ``--output-root``.

Shadow publication is the default.  Writing the live four condition pointers
requires an authenticated v1-writer handoff and an exclusive process lease;
this prevents the old v1 harvester and this consumer from racing the same
``current7-index.json`` files during a canary.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence
import urllib.error
import uuid

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_harvest import (
        CACHE_RECORD_SCHEMA,
        COHORT_STATUS_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
        AccountSftpReader,
        RemoteReader,
        _json_bytes,
        _sha_bytes,
        _strict_json,
        aggregate_candidates,
        authenticated_constraint_identity,
        build_current7_snapshot,
        cache_seed_records,
    )
    from tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        PROTOCOL_VERSION,
        json_bytes,
        validate_batch_task,
    )
    from tier1_final1000_multiseed_controller import (
        SINGLE_SEED_PROTOCOL,
        _expected_task_for_entry,
        validate_state,
    )
    from tier1_final1000_multiseed_harvest import (
        _batch_parent_task,
        _read_batch_journal,
        harvest_mixed_inventory,
    )
    from tier1_final1000_multiseed_monitor import (
        _read_reference,
        load_compact_condition_index,
    )
    from tier1_final1000_multiseed_status import (
        build_compact_snapshot,
        publish_compact_snapshot,
        validate_shard,
    )
    from tier1_final1000_rolling_migration import (
        validate_chained_predecessor_plan,
        validate_historical_resource_quota_successor_plan,
        validate_predecessor_plan,
        validate_successor_plan,
    )
    from tier1_final1000_slurm_harvest import (
        DEFAULT_ACCOUNTS,
        DEFAULT_CURRENT7_INDEX,
        DEFAULT_RUNTIME,
        DEFAULT_SCHEDULER_SOURCE,
        DEFAULT_SCHEDULER_URL,
        DEDUPE_PREFIX,
        OBSERVED_TASK_SEAL_FIELDS,
        SCHEDULER_BATCH_LIMIT,
        TASK_NAME_PREFIX,
        Final1000ReadOnlySchedulerApi,
        SchedulerReader,
        _assert_runtime_isolated,
        _load_scheduler_cache,
        _result_validator,
        _scheduler_cache_path,
        _scheduler_task_id,
        _sealed_scheduler_cache,
    )
    from tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        load_stage_bindings,
        validate_stage_result,
    )
    from tier1_final1000_stage_profiles import BY_ID, STAGES, stage_profile
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_harvest import (
        CACHE_RECORD_SCHEMA,
        COHORT_STATUS_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
        AccountSftpReader,
        RemoteReader,
        _json_bytes,
        _sha_bytes,
        _strict_json,
        aggregate_candidates,
        authenticated_constraint_identity,
        build_current7_snapshot,
        cache_seed_records,
    )
    from tools.tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tools.tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        PROTOCOL_VERSION,
        json_bytes,
        validate_batch_task,
    )
    from tools.tier1_final1000_multiseed_controller import (
        SINGLE_SEED_PROTOCOL,
        _expected_task_for_entry,
        validate_state,
    )
    from tools.tier1_final1000_multiseed_harvest import (
        _batch_parent_task,
        _read_batch_journal,
        harvest_mixed_inventory,
    )
    from tools.tier1_final1000_multiseed_monitor import (
        _read_reference,
        load_compact_condition_index,
    )
    from tools.tier1_final1000_multiseed_status import (
        build_compact_snapshot,
        publish_compact_snapshot,
        validate_shard,
    )
    from tools.tier1_final1000_rolling_migration import (
        validate_chained_predecessor_plan,
        validate_historical_resource_quota_successor_plan,
        validate_predecessor_plan,
        validate_successor_plan,
    )
    from tools.tier1_final1000_slurm_harvest import (
        DEFAULT_ACCOUNTS,
        DEFAULT_CURRENT7_INDEX,
        DEFAULT_RUNTIME,
        DEFAULT_SCHEDULER_SOURCE,
        DEFAULT_SCHEDULER_URL,
        DEDUPE_PREFIX,
        OBSERVED_TASK_SEAL_FIELDS,
        SCHEDULER_BATCH_LIMIT,
        TASK_NAME_PREFIX,
        Final1000ReadOnlySchedulerApi,
        SchedulerReader,
        _assert_runtime_isolated,
        _load_scheduler_cache,
        _result_validator,
        _scheduler_cache_path,
        _scheduler_task_id,
        _sealed_scheduler_cache,
    )
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        load_stage_bindings,
        validate_stage_result,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, STAGES, stage_profile


CONSUMER_SCHEMA = "mft-tier1-final1000-multiseed-consumer-result-v1"
CONDITION_INVENTORY_SCHEMA = (
    "mft-tier1-final1000-multiseed-condition-index-inventory-v2"
)
CAPABILITY_RECEIPT_SCHEMA = "mft-tier1-final1000-multiseed-consumer-capability-v2"
V1_HANDOFF_SCHEMA = "mft-tier1-final1000-v1-harvester-handoff-v1"
STOP_HANDOFF_SCHEMA = "mft-tier1-final1000-multiseed-consumer-stop-v1"
WRITER_LEASE_SCHEMA = "mft-tier1-final1000-multiseed-writer-lease-v1"
MAX_POLL_SECONDS = 60.0
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_FRESHNESS_SECONDS = 120
DEFAULT_SHADOW_NAME = "multiseed-shadow"
POINTER_NAME = "current7-index.json"
VISIBLE_STATES = frozenset(
    {
        "queued",
        "attaching",
        "running",
        "completed",
        "failed",
        "cancelled",
        "timeout",
        "timed_out",
    }
)
TERMINAL_STATES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)
CAPABILITIES = {
    "mixed_v1_v2_scheduler_inventory": True,
    "running_parent_stable_journal_read": True,
    "exact_result_validation": True,
    "bundle_seed_deduplication": True,
    "compact_v2_atomic_publication": True,
    "sealed_child_count_prefix_only": True,
    "live_v1_cohort_identity_preserved": True,
    "scheduler_access": "GET-only",
    "remote_access": "read-only",
    "scheduler_mutation_count": 0,
    "remote_write_count": 0,
    "virtual_scheduler_task_ids_created": False,
    "concurrent_writer_refusal": True,
    "stop_restart_handoff": True,
    "aedt_used": False,
    "fea_submission_performed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> int:
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == payload:
        return 0
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 1


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _record_matches(path: Path, record: Mapping[str, Any]) -> bool:
    try:
        actual = _file_record(path)
    except OSError:
        return False
    return all(actual[key] == record.get(key) for key in actual)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows ``os.kill(pid, 0)`` terminates the target process instead
        # of providing the POSIX existence probe.  Query a process handle and
        # never send a signal to either the old writer or this consumer.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # access denied still proves existence
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _lease_is_actively_locked(path: Path) -> bool:
    """Return true only when another open file description holds the lease."""

    try:
        stream = path.resolve(strict=True).open("r+b")
    except OSError:
        return False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised on Linux CI
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, BlockingIOError):
            return True
        return False
    finally:
        stream.close()


def _lease_sidecar_path(lock_path: Path) -> Path:
    return lock_path.with_name(f"{lock_path.name}.json")


def build_v1_handoff_receipt(
    *,
    old_pid: int,
    stop_file: Path,
    condition_indexes: Mapping[str, Path],
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Seal proof that the old writer exited and its four pointers are final."""

    if _pid_is_running(int(old_pid)):
        raise RuntimeError("old v1 harvester is still running")
    stop_record = _file_record(stop_file)
    if set(condition_indexes) != set(BY_ID):
        raise RuntimeError("v1 handoff must seal all four condition indexes")
    indexes = []
    for stage in STAGES:
        path = condition_indexes[stage.stage_id]
        first = _file_record(path)
        value = _read_json(path)
        second = _file_record(path)
        if first != second or value.get("schema_version") != CURRENT7_INDEX_SCHEMA:
            raise RuntimeError("v1 condition pointer was unstable or not v1")
        indexes.append({"stage_id": stage.stage_id, **first, "index": value})
    unsigned = {
        "schema_version": V1_HANDOFF_SCHEMA,
        "observed_at": observed_at or _utc_now(),
        "old_writer_pid": int(old_pid),
        "old_writer_exited": True,
        "stop_file": stop_record,
        "condition_indexes": indexes,
        "condition_index_count": len(indexes),
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def validate_v1_handoff_receipt(
    value: Mapping[str, Any], *, require_live_pointer_match: bool
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    indexes = value.get("condition_indexes")
    stop_record = value.get("stop_file")
    if (
        value.get("schema_version") != V1_HANDOFF_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("old_writer_exited") is not True
        or _pid_is_running(int(value.get("old_writer_pid") or 0))
        or not isinstance(stop_record, dict)
        or not _record_matches(Path(str(stop_record.get("path") or "")), stop_record)
        or not isinstance(indexes, list)
        or len(indexes) != len(BY_ID)
        or {str(item.get("stage_id") or "") for item in indexes} != set(BY_ID)
        or value.get("condition_index_count") != len(BY_ID)
        or value.get("scheduler_mutation_count") != 0
        or value.get("remote_write_count") != 0
    ):
        raise RuntimeError("v1 harvester handoff receipt is invalid")
    for item in indexes:
        embedded = item.get("index")
        if (
            not isinstance(embedded, dict)
            or embedded.get("schema_version") != CURRENT7_INDEX_SCHEMA
            or int(item.get("size") or -1) != len(json_bytes(embedded))
            or str(item.get("sha256") or "")
            != hashlib.sha256(json_bytes(embedded)).hexdigest()
        ):
            raise RuntimeError("v1 handoff index identity is invalid")
        if require_live_pointer_match and not _record_matches(
            Path(str(item.get("path") or "")), item
        ):
            raise RuntimeError("v1 final condition pointer advanced after handoff")
    return copy.deepcopy(dict(value))


class WriterLease:
    """Cross-process nonblocking lock; the sidecar is evidence, not authority."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.sidecar_path = _lease_sidecar_path(self.path)
        self.stream: Any | None = None
        self.run_id = uuid.uuid4().hex
        self.holder_pid = os.getpid()
        self.holder_host = socket.gethostname()

    def __enter__(self) -> "WriterLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                if self.stream.read(1) == b"":
                    self.stream.write(b"\0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised on Linux CI
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                "another multi-seed consumer owns the writer lease"
            ) from exc
        sidecar_unsigned = {
            "schema_version": WRITER_LEASE_SCHEMA,
            "run_id": self.run_id,
            "pid": self.holder_pid,
            "host": self.holder_host,
            "acquired_at": _utc_now(),
        }
        sidecar = {
            **sidecar_unsigned,
            "lease_sha256": canonical_sha256(sidecar_unsigned),
        }
        try:
            _atomic_json(self.sidecar_path, sidecar)
        except BaseException:
            self.__exit__()
            raise
        return self

    def active_attestation(self) -> dict[str, Any]:
        if self.stream is None:
            raise RuntimeError("writer lease is not active")
        sidecar = _read_json(self.sidecar_path)
        unsigned = {key: item for key, item in sidecar.items() if key != "lease_sha256"}
        if (
            sidecar.get("schema_version") != WRITER_LEASE_SCHEMA
            or sidecar.get("lease_sha256") != canonical_sha256(unsigned)
            or sidecar.get("run_id") != self.run_id
            or sidecar.get("pid") != self.holder_pid
            or sidecar.get("host") != self.holder_host
        ):
            raise RuntimeError("active writer lease sidecar identity drifted")
        return {
            "run_id": self.run_id,
            "holder_pid": self.holder_pid,
            "holder_host": self.holder_host,
            "lock_path": str(self.path),
            "lease_file": _file_record(self.sidecar_path),
            "sidecar_sha256": sidecar["lease_sha256"],
            "lease_held_at_publication": True,
        }

    def __exit__(self, *_args: Any) -> None:
        if self.stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised on Linux CI
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def _validate_writer_lease_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    lease_file = value.get("lease_file")
    if not isinstance(lease_file, dict):
        raise RuntimeError("consumer writer lease attestation is invalid")
    path = Path(str(lease_file.get("path") or ""))
    lock_path = Path(str(value.get("lock_path") or ""))
    if not _record_matches(path, lease_file):
        raise RuntimeError("consumer writer lease sidecar advanced")
    sidecar = _read_json(path)
    unsigned = {key: item for key, item in sidecar.items() if key != "lease_sha256"}
    pid = int(value.get("holder_pid") or 0)
    host = str(value.get("holder_host") or "")
    if (
        sidecar.get("schema_version") != WRITER_LEASE_SCHEMA
        or sidecar.get("lease_sha256") != canonical_sha256(unsigned)
        or value.get("sidecar_sha256") != sidecar.get("lease_sha256")
        or value.get("run_id") != sidecar.get("run_id")
        or value.get("lease_held_at_publication") is not True
        or pid != sidecar.get("pid")
        or host != sidecar.get("host")
        or host != socket.gethostname()
        or not _pid_is_running(pid)
        or path.resolve() != _lease_sidecar_path(lock_path.resolve())
        or not _lease_is_actively_locked(lock_path)
    ):
        raise RuntimeError("consumer writer lease attestation is not active")
    return copy.deepcopy(dict(value))


@dataclass(frozen=True)
class ConsumerInputs:
    plan: dict[str, Any]
    state: dict[str, Any]
    cohorts_by_plan_sha: dict[str, dict[str, Any]]


def _validate_any_source_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for validator in (
        validate_successor_plan,
        validate_chained_predecessor_plan,
        validate_historical_resource_quota_successor_plan,
        validate_predecessor_plan,
    ):
        try:
            return validator(value)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{validator.__name__}:{exc}")
    raise RuntimeError("source launch plan is unauthenticated: " + "; ".join(errors))


def _validate_binding_pair(
    plan: Mapping[str, Any], bindings_path: Path
) -> dict[str, Mapping[str, Any]]:
    bindings = load_stage_bindings(bindings_path)
    if set(bindings) != set(BY_ID):
        raise RuntimeError("source binding set is incomplete")
    for stage in STAGES:
        summary = plan["stage_bindings"][stage.stage_id]
        binding = bindings[stage.stage_id]
        bundle_plan = binding["plan"]
        publication = binding["publication"]
        if (
            summary.get("bundle_id") != bundle_plan.get("bundle_id")
            or summary.get("bundle_manifest_sha256")
            != bundle_plan.get("bundle_manifest_sha256")
            or summary.get("remote_bundle") != bundle_plan.get("remote_bundle")
            or summary.get("publication_receipt_sha256")
            != publication.get("receipt_sha256")
            or summary.get("ready") != publication.get("ready")
            or summary.get("ready_sha256") != publication.get("ready_sha256")
            or summary.get("stage_spec_sha256")
            != stage_profile(stage)["stage_spec_sha256"]
        ):
            raise RuntimeError("source launch/bundle/READY binding mismatch")
    return bindings


def load_inputs(
    launch_plan_path: Path,
    bindings_path: Path,
    controller_state_path: Path,
    *,
    source_launch_plan_paths: Sequence[Path] = (),
    source_bindings_paths: Sequence[Path] = (),
) -> ConsumerInputs:
    if len(source_launch_plan_paths) != len(source_bindings_paths):
        raise RuntimeError("source launch-plan/bindings counts differ")
    plan = validate_successor_plan(_read_json(launch_plan_path.resolve(strict=True)))
    state = validate_state(_read_json(controller_state_path.resolve(strict=True)), plan)
    supplied = [(plan, bindings_path)]
    supplied.extend(
        (
            _validate_any_source_plan(_read_json(plan_path.resolve(strict=True))),
            source_bindings_path,
        )
        for plan_path, source_bindings_path in zip(
            source_launch_plan_paths, source_bindings_paths, strict=True
        )
    )
    cohorts: dict[str, dict[str, Any]] = {}
    for source_plan, source_bindings in supplied:
        plan_sha = str(source_plan["launch_plan_sha256"])
        candidate = {
            "plan": source_plan,
            "bindings": _validate_binding_pair(source_plan, source_bindings),
        }
        prior = cohorts.setdefault(plan_sha, candidate)
        if prior != candidate:
            raise RuntimeError("one source plan SHA has divergent bindings")
    required = {
        str(entry["source_launch_plan_sha256"])
        for entry in state["entries"]
        if entry["protocol_version"] == SINGLE_SEED_PROTOCOL
    }
    required.add(str(plan["launch_plan_sha256"]))
    if set(cohorts) != required:
        raise RuntimeError("source cohort inputs are incomplete or excessive")
    return ConsumerInputs(plan=plan, state=state, cohorts_by_plan_sha=cohorts)


def _expected_entry_context(
    entry: Mapping[str, Any], inputs: ConsumerInputs
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    expected = _expected_task_for_entry(entry, inputs.plan)
    if set(expected) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError("expected Scheduler envelope fields drifted")
    source_sha = (
        str(inputs.plan["launch_plan_sha256"])
        if entry["protocol_version"] == PROTOCOL_VERSION
        else str(entry["source_launch_plan_sha256"])
    )
    cohort = inputs.cohorts_by_plan_sha.get(source_sha)
    if cohort is None:
        raise RuntimeError("entry source cohort is unavailable")
    binding = cohort["bindings"][str(entry["stage_id"])]
    payload = expected["payload_json"]
    if (
        payload.get("bundle_id") != binding["plan"].get("bundle_id")
        or payload.get("bundle_manifest_sha256")
        != binding["plan"].get("bundle_manifest_sha256")
        or expected.get("remote_cwd") != binding["plan"].get("remote_bundle")
    ):
        raise RuntimeError("entry task does not match its exact source binding")
    return expected, binding


def _validate_observed(
    observed: Mapping[str, Any], *, task_id: int, expected: Mapping[str, Any]
) -> str:
    state = str(observed.get("status") or "").lower()
    embedded = observed.get("task_json")
    if (
        _scheduler_task_id(observed) != task_id
        or state not in VISIBLE_STATES
        or observed.get("name") != expected.get("name")
        or observed.get("dedupe_key") != expected.get("dedupe_key")
        or any(
            observed.get(field) != expected.get(field)
            for field in OBSERVED_TASK_SEAL_FIELDS
        )
        or (embedded is not None and embedded != expected)
    ):
        raise RuntimeError(f"mixed Scheduler task identity changed: {task_id}")
    # The Scheduler has emitted both spellings over its lifetime, while the
    # inherited immutable terminal-cache validator accepts ``timeout``.  Seal
    # one canonical spelling so the first observation and every restart
    # project the same physical-lane state.
    return "timeout" if state == "timed_out" else state


def _index_scheduler_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    by_dedupe: dict[str, int] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("mixed Scheduler inventory row is not an object")
        row = copy.deepcopy(dict(raw))
        task_id = _scheduler_task_id(row)
        dedupe = str(row.get("dedupe_key") or "")
        if (
            not str(row.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not dedupe.startswith(DEDUPE_PREFIX)
            or str(row.get("status") or "").lower() not in VISIBLE_STATES
        ):
            raise RuntimeError("mixed Scheduler inventory row identity drifted")
        if task_id in by_id or (dedupe in by_dedupe and by_dedupe[dedupe] != task_id):
            raise RuntimeError("mixed Scheduler inventory identity is duplicated")
        by_id[task_id] = row
        by_dedupe[dedupe] = task_id
    return by_id


def _scheduler_inventory(
    inputs: ConsumerInputs,
    *,
    scheduler: SchedulerReader,
    runtime: Path,
    apply: bool,
) -> tuple[list[dict[str, Any]], list[tuple[Path, bytes]], int]:
    prepared = []
    cache_hits = 0
    seen_task_ids: set[int] = set()
    for entry in inputs.state["entries"]:
        task_id = entry.get("task_id")
        if task_id is None:
            continue
        task_id = int(task_id)
        if task_id in seen_task_ids:
            raise RuntimeError("controller-v2 repeats a physical Scheduler task id")
        seen_task_ids.add(task_id)
        expected, binding = _expected_entry_context(entry, inputs)
        cache_path = _scheduler_cache_path(
            runtime / "conditions" / str(entry["stage_id"]), task_id
        )
        cached = None
        # Once a terminal Scheduler envelope has been sealed locally, it is
        # sufficient evidence even if controller-v2 has not yet persisted its
        # matching terminal state or the Scheduler has already aged the row
        # out of the live list response.
        if cache_path.is_file() or str(entry.get("state") or "") in TERMINAL_STATES:
            cached = _load_scheduler_cache(
                cache_path,
                task_id=task_id,
                dedupe_key=str(entry["parent_dedupe_key"]),
            )
            cache_hits += cached is not None
        prepared.append(
            {
                "entry": entry,
                "task_id": task_id,
                "expected": expected,
                "binding": binding,
                "cache_path": cache_path,
                "observed": cached,
            }
        )
    by_id = _index_scheduler_rows(
        scheduler.list_tasks(name_prefix=TASK_NAME_PREFIX, limit=SCHEDULER_BATCH_LIMIT)
    )
    pending_cache: list[tuple[Path, bytes]] = []
    inventory = []
    terminal_confirmation: dict[int, dict[str, Any]] | None = None
    for item in prepared:
        observed = item["observed"]
        task_id = int(item["task_id"])
        observed_from_batch = False
        if observed is None:
            observed = by_id.get(task_id)
            observed_from_batch = observed is not None
            if observed is None:
                observed = scheduler.get_task(task_id)
            if observed is None:
                raise RuntimeError(f"mixed Scheduler task disappeared: {task_id}")
        state = _validate_observed(observed, task_id=task_id, expected=item["expected"])
        if state in TERMINAL_STATES and not item["cache_path"].is_file():
            if observed_from_batch:
                if terminal_confirmation is None:
                    terminal_confirmation = _index_scheduler_rows(
                        scheduler.list_tasks(
                            name_prefix=TASK_NAME_PREFIX,
                            limit=SCHEDULER_BATCH_LIMIT,
                        )
                    )
                confirmation = terminal_confirmation.get(task_id)
                if confirmation is None:
                    confirmation = scheduler.get_task(task_id)
                if confirmation is None:
                    raise RuntimeError(
                        f"terminal mixed Scheduler task disappeared: {task_id}"
                    )
                confirmed_state = _validate_observed(
                    confirmation, task_id=task_id, expected=item["expected"]
                )
                if confirmed_state != state:
                    raise RuntimeError(
                        f"terminal mixed Scheduler state changed: {task_id}"
                    )
                observed = confirmation
            cache_observed = copy.deepcopy(dict(observed))
            cache_observed["status"] = state
            pending_cache.append(
                (
                    item["cache_path"],
                    _json_bytes(_sealed_scheduler_cache(cache_observed)),
                )
            )
        entry = item["entry"]
        expected = item["expected"]
        payload = expected["payload_json"]
        common = {
            "task_id": task_id,
            "stage_id": str(entry["stage_id"]),
            "protocol_version": str(entry["protocol_version"]),
            "name": expected["name"],
            "dedupe_key": expected["dedupe_key"],
            "account_name": str(observed.get("account_name") or ""),
            "status": state,
            "state": state,
            "exit_code": observed.get("exit_code"),
            "created_at": observed.get("created_at"),
            "started_at": observed.get("started_at"),
            "finished_at": observed.get("finished_at"),
            "_expected_task": expected,
            "_binding": item["binding"],
            "_entry": entry,
        }
        if entry["protocol_version"] == PROTOCOL_VERSION:
            common.update(
                {
                    "parent_task": expected,
                    "scheduler_task": copy.deepcopy(expected),
                    "batch_length": int(entry["batch_length"]),
                }
            )
        else:
            lane = payload["lane"]
            common.update(
                {
                    "bundle_id": payload["bundle_id"],
                    "seed": int(payload["seed"]),
                    "island_id": lane["island_id"],
                    "wave": lane["wave"],
                    "payload": copy.deepcopy(payload),
                    "payload_sha256": canonical_sha256(payload),
                }
            )
        inventory.append(common)
    if not apply:
        # Explicitly document the dry-run boundary: callers may inspect these
        # proposed cache writes, but this function never performs them.
        pass
    return inventory, pending_cache, cache_hits


def _v2_result_validator(parent_task: Mapping[str, Any]):
    parent = validate_batch_task(parent_task)
    by_seed = {
        int(child["seed"]): child["task"]
        for child in parent["payload_json"]["children"]
    }

    def validate(
        result: Mapping[str, Any],
        *,
        payload: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        validate_current7_result(result, payload=payload, manifest=manifest)
        child = by_seed.get(int(payload["seed"]))
        if child is None or child.get("payload_json") != payload:
            raise RuntimeError("v2 terminal result child identity is unknown")

        def exact_task(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            if dict(candidate) != child:
                raise RuntimeError("v2 terminal child envelope drifted")
            return child

        validate_stage_result(result, child, task_validator=exact_task)

    return validate


def _load_legacy_snapshot(
    index: Mapping[str, Any], *, index_path: Path, stage_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(str(index.get("path_containment_root") or "")).resolve(strict=True)
    if not index_path.resolve().is_relative_to(root):
        raise RuntimeError("legacy condition index escaped its containment root")
    status_ref = index.get("status")
    if (
        not isinstance(status_ref, dict)
        or status_ref.get("schema_version") != COHORT_STATUS_SCHEMA
    ):
        raise RuntimeError("legacy condition status reference is invalid")
    status_path = Path(str(status_ref.get("path") or "")).resolve(strict=True)
    if not status_path.is_relative_to(root):
        raise RuntimeError("legacy condition status escaped its containment root")
    payload = status_path.read_bytes()
    status = _strict_json(payload, str(status_path))
    if (
        _sha_bytes(payload) != status_ref.get("sha256")
        or status.get("schema_version") != COHORT_STATUS_SCHEMA
        or index.get("final_goal_stage_id", stage_id) != stage_id
    ):
        raise RuntimeError("legacy condition status/index seal mismatch")
    records = status.get("terminal_results")
    if not isinstance(records, list) or any(
        not isinstance(item, dict) for item in records
    ):
        raise RuntimeError("legacy terminal-result inventory is invalid")
    for record in records:
        unsigned = {key: item for key, item in record.items() if key != "record_sha256"}
        if record.get("record_sha256") != canonical_sha256(unsigned):
            raise RuntimeError("legacy seed record seal mismatch")
    static = copy.deepcopy(status)
    for key in (
        "schema_version",
        "updated_at",
        "scheduler_task_count",
        "state_counts",
        "latest_tasks",
        "authenticated_terminal_seed_count",
        "terminal_results",
    ):
        static.pop(key, None)
    return copy.deepcopy(records), static


def _load_existing_snapshot(
    index_path: Path, *, stage_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not index_path.is_file():
        return [], {}, []
    first = index_path.read_bytes()
    index = _strict_json(first, str(index_path))
    if index.get("schema_version") == CURRENT7_INDEX_SCHEMA:
        records, static = _load_legacy_snapshot(
            index, index_path=index_path, stage_id=stage_id
        )
        return records, static, []
    if index.get("schema_version") != COMPACT_INDEX_SCHEMA:
        raise RuntimeError("condition pointer schema is unsupported")
    snapshot = load_compact_condition_index(index_path)
    records: list[dict[str, Any]] = []
    for reference in snapshot["manifest"]["shards"]:
        _path, _payload, value = _read_reference(
            snapshot["root"], reference, label="seed-result shard"
        )
        shard = validate_shard(value, reference)
        records.extend(copy.deepcopy(shard["records"]))
    if index_path.read_bytes() != first:
        raise RuntimeError("condition pointer advanced during parent import")
    return (
        records,
        copy.deepcopy(snapshot["status"].get("frontend_static") or {}),
        copy.deepcopy(snapshot["status"].get("latest_tasks") or []),
    )


def _validate_public_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in record.items() if key != "record_sha256"}
    if (
        record.get("schema_version") != CACHE_RECORD_SCHEMA
        or not str(record.get("bundle_id") or "")
        or isinstance(record.get("seed"), bool)
        or not isinstance(record.get("seed"), int)
        or record.get("authenticated") is not True
        or record.get("record_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("canonical seed record identity is invalid")
    return record


def _merge_records(
    historical: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in [*historical, *current]:
        record = _validate_public_record(raw)
        identity = (str(record["bundle_id"]), int(record["seed"]))
        prior = merged.get(identity)
        if prior is None:
            merged[identity] = record
            continue
        prior_result = (prior.get("result_object") or {}).get("sha256")
        next_result = (record.get("result_object") or {}).get("sha256")
        if prior_result and next_result and prior_result != next_result:
            raise RuntimeError("canonical bundle+seed result identity diverged")
        if prior_result and not next_result:
            continue
        if next_result and not prior_result:
            merged[identity] = record
            continue
        # Idempotent replay normally has equal records.  Permit only the
        # monotonic task/status union produced by cache_seed_records.
        if prior != record:
            prior_ids = {int(item) for item in prior.get("task_ids") or []}
            next_ids = {int(item) for item in record.get("task_ids") or []}
            if not prior_ids.issubset(next_ids):
                raise RuntimeError("canonical seed replay lost a real task id")
            merged[identity] = record
    return [merged[key] for key in sorted(merged)]


def _aggregate_records(
    records: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current_by_identity = {
        (str(item["bundle_id"]), int(item["seed"])): item for item in current
    }
    hydrated = []
    for raw in records:
        record = copy.deepcopy(dict(raw))
        source = current_by_identity.get(
            (str(record["bundle_id"]), int(record["seed"]))
        )
        if source is not None and isinstance(source.get("_candidates"), list):
            record["_candidates"] = copy.deepcopy(source["_candidates"])
        elif record.get("result_object"):
            candidates = []
            artifacts = record.get("artifact_objects") or {}
            for name in ("pareto_candidates", "least_violation_candidates"):
                artifact = artifacts.get(name)
                if not isinstance(artifact, dict):
                    raise RuntimeError("canonical candidate artifact is missing")
                path = Path(str(artifact.get("local_cache_path") or ""))
                payload = path.read_bytes()
                if _sha_bytes(payload) != artifact.get("sha256") or len(payload) != int(
                    artifact.get("size_bytes") or -1
                ):
                    raise RuntimeError("canonical candidate cache object drifted")
                value = _strict_json(payload, str(path))
                rows = value.get("candidates")
                if not isinstance(rows, list) or any(
                    not isinstance(item, dict) for item in rows
                ):
                    raise RuntimeError("canonical candidate cache schema drifted")
                candidates.extend(copy.deepcopy(rows))
            record["_candidates"] = candidates
        else:
            record["_candidates"] = []
        hydrated.append(record)
    return aggregate_candidates(hydrated)


def _frontend_static(
    historical: Mapping[str, Any], *, binding: Mapping[str, Any], stage_runtime: Path
) -> dict[str, Any]:
    if historical:
        return copy.deepcopy(dict(historical))
    status, _compatibility, _index = build_current7_snapshot(
        plan=binding["plan"],
        manifest=binding["manifest"],
        inventory=[],
        records=[],
        refusals=[],
        runtime=stage_runtime,
    )
    static = copy.deepcopy(status)
    for key in (
        "schema_version",
        "updated_at",
        "scheduler_task_count",
        "state_counts",
        "latest_tasks",
        "authenticated_terminal_seed_count",
        "terminal_results",
    ):
        static.pop(key, None)
    return static


def _source_cohort_projection(
    inputs: ConsumerInputs, *, stage_id: str
) -> list[dict[str, Any]]:
    policy_ids_by_plan: dict[str, set[str]] = {
        plan_sha: set() for plan_sha in inputs.cohorts_by_plan_sha
    }
    for entry in inputs.state["entries"]:
        plan_sha = (
            str(inputs.plan["launch_plan_sha256"])
            if entry["protocol_version"] == PROTOCOL_VERSION
            else str(entry["source_launch_plan_sha256"])
        )
        policy_id = entry.get("source_resource_policy_id")
        if policy_id:
            policy_ids_by_plan[plan_sha].add(str(policy_id))
    projection = []
    for plan_sha, cohort in sorted(inputs.cohorts_by_plan_sha.items()):
        binding = cohort["bindings"][stage_id]
        projection.append(
            {
                "launch_plan_sha256": plan_sha,
                "bundle_id": binding["plan"]["bundle_id"],
                "bundle_manifest_sha256": binding["plan"]["bundle_manifest_sha256"],
                "manifest_contract_sha256": binding["manifest"]["contract_sha256"],
                "source_resource_policy_ids": sorted(policy_ids_by_plan[plan_sha]),
            }
        )
    return projection


def _stage_parent_path(
    *,
    stage_id: str,
    runtime: Path,
    output_root: Path,
    publish_mode: str,
    handoff: Mapping[str, Any] | None,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    output_pointer = output_root / "conditions" / stage_id / "canonical" / POINTER_NAME
    if output_pointer.is_file():
        return output_pointer, None
    live = runtime / "conditions" / stage_id / "canonical" / POINTER_NAME
    if publish_mode == "canonical" and handoff is not None:
        item = next(
            entry
            for entry in handoff["condition_indexes"]
            if entry["stage_id"] == stage_id
        )
        return Path(str(item["path"])), item["index"]
    return (live if live.is_file() else None), None


def _load_parent(
    path: Path | None,
    embedded: Mapping[str, Any] | None,
    *,
    stage_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return [], {}, []
    if embedded is None:
        return _load_existing_snapshot(path, stage_id=stage_id)
    return (*_load_legacy_snapshot(embedded, index_path=path, stage_id=stage_id), [])


def _hot_lanes(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    active = [item for item in lanes if str(item["state"]) not in TERMINAL_STATES]
    terminal = [item for item in lanes if str(item["state"]) in TERMINAL_STATES]
    active.sort(key=lambda item: int(item["task_id"]), reverse=True)
    terminal.sort(key=lambda item: int(item["task_id"]), reverse=True)
    if len(active) > 500:
        raise RuntimeError("more than 500 physical lanes are active")
    return [*active, *terminal[: 500 - len(active)]]


def _condition_inventory(
    *,
    output_root: Path,
    inputs: ConsumerInputs,
    indexes: Sequence[Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    unsigned = {
        "schema_version": CONDITION_INVENTORY_SCHEMA,
        "launch_plan_sha256": inputs.plan["launch_plan_sha256"],
        "controller_state_sha256": inputs.state["state_sha256"],
        "updated_at": observed_at,
        "indexes": copy.deepcopy(list(indexes)),
        "ui_environment": {
            "name": "MFT_TIER1_CURRENT7_CONDITION_INDEXES",
            "value": os.pathsep.join(str(item["path"]) for item in indexes),
            "adapter": "tools.tier1_final1000_multiseed_monitor.adapt_condition_index",
            "primary_authority_unchanged": True,
        },
        "path_containment_root": str(output_root.resolve()),
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
        "virtual_scheduler_task_ids_created": False,
        "aedt_used": False,
        "fea_submission_performed": False,
    }
    return {**unsigned, "inventory_sha256": canonical_sha256(unsigned)}


def build_capability_receipt(
    *,
    result: Mapping[str, Any],
    inventory_path: Path,
    poll_seconds: float,
    freshness_deadline_seconds: int,
    runtime_root: Path,
    writer_lease: WriterLease,
    handoff_receipt_path: Path | None = None,
) -> dict[str, Any]:
    if not 0 < float(poll_seconds) <= MAX_POLL_SECONDS:
        raise RuntimeError("consumer poll interval exceeds 60 seconds")
    publish_mode = str(result.get("publish_mode") or "")
    if publish_mode not in {"shadow", "canonical"}:
        raise RuntimeError("consumer capability publish mode is invalid")
    runtime_root = runtime_root.resolve()
    publication_root = Path(
        str(result["condition_inventory"]["path_containment_root"])
    ).resolve()
    if (
        (publish_mode == "canonical" and publication_root != runtime_root)
        or (publish_mode == "shadow" and publication_root == runtime_root)
        or not _path_is_within(inventory_path, publication_root)
    ):
        raise RuntimeError("consumer capability publication containment is invalid")
    inventory_record = _file_record(inventory_path)
    indexes = []
    for item in result["condition_inventory"]["indexes"]:
        index_path = Path(str(item["path"]))
        if not _path_is_within(index_path, publication_root):
            raise RuntimeError("consumer condition index escaped publication root")
        load_compact_condition_index(index_path)
        indexes.append({"stage_id": item["stage_id"], **_file_record(index_path)})
    handoff_binding = None
    if publish_mode == "canonical":
        if handoff_receipt_path is None:
            raise RuntimeError("canonical capability requires the v1 handoff file")
        handoff_path = handoff_receipt_path.resolve(strict=True)
        handoff = validate_v1_handoff_receipt(
            _read_json(handoff_path), require_live_pointer_match=False
        )
        handoff_binding = {
            "receipt_file": _file_record(handoff_path),
            "receipt_sha256": handoff["receipt_sha256"],
            "old_writer_pid": handoff["old_writer_pid"],
            "old_writer_exited": handoff["old_writer_exited"],
            "sealed_v1_condition_index_count": handoff["condition_index_count"],
        }
    elif handoff_receipt_path is not None:
        raise RuntimeError("shadow capability must not claim a canonical handoff")
    lease_attestation = writer_lease.active_attestation()
    unsigned = {
        "schema_version": CAPABILITY_RECEIPT_SCHEMA,
        "observed_at": result["observed_at"],
        "published_at": _utc_now(),
        "freshness_deadline_seconds": int(freshness_deadline_seconds),
        "poll_seconds": float(poll_seconds),
        "run_id": writer_lease.run_id,
        "publish_mode": publish_mode,
        "runtime_root": str(runtime_root),
        "publication_root": str(publication_root),
        "launch_plan_sha256": result["launch_plan_sha256"],
        "controller_state_sha256": result["controller_state_sha256"],
        "condition_inventory": inventory_record,
        "condition_indexes": indexes,
        "v1_handoff": handoff_binding,
        "writer_lease": lease_attestation,
        "capabilities": copy.deepcopy(CAPABILITIES),
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def require_consumer_capability(
    receipt_path: Path,
    *,
    now: datetime | None = None,
    maximum_age_seconds: int | None = None,
    required_publish_mode: str = "canonical",
) -> dict[str, Any]:
    value = _read_json(receipt_path.resolve(strict=True))
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    deadline = int(value.get("freshness_deadline_seconds") or 0)
    try:
        published = datetime.fromisoformat(str(value.get("published_at") or ""))
    except ValueError as exc:
        raise RuntimeError("consumer capability timestamp is invalid") from exc
    if published.tzinfo is None:
        raise RuntimeError("consumer capability timestamp lacks a timezone")
    current = now or datetime.now(timezone.utc)
    age = (
        current.astimezone(timezone.utc) - published.astimezone(timezone.utc)
    ).total_seconds()
    allowed = (
        deadline
        if maximum_age_seconds is None
        else min(deadline, int(maximum_age_seconds))
    )
    inventory = value.get("condition_inventory")
    indexes = value.get("condition_indexes")
    publish_mode = str(value.get("publish_mode") or "")
    runtime_root = Path(str(value.get("runtime_root") or "")).resolve()
    publication_root = Path(str(value.get("publication_root") or "")).resolve()
    if (
        value.get("schema_version") != CAPABILITY_RECEIPT_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("capabilities") != CAPABILITIES
        or required_publish_mode not in {"shadow", "canonical"}
        or publish_mode != required_publish_mode
        or (publish_mode == "canonical" and publication_root != runtime_root)
        or (publish_mode == "shadow" and publication_root == runtime_root)
        or not 0 < float(value.get("poll_seconds") or 0) <= MAX_POLL_SECONDS
        or deadline <= 0
        or allowed <= 0
        or age < -5
        or age > allowed
        or not isinstance(inventory, dict)
        or not _record_matches(Path(str(inventory.get("path") or "")), inventory)
        or not _path_is_within(Path(str(inventory.get("path") or "")), publication_root)
        or not isinstance(indexes, list)
        or len(indexes) != len(BY_ID)
        or {str(item.get("stage_id") or "") for item in indexes} != set(BY_ID)
    ):
        raise RuntimeError("consumer capability/freshness receipt is invalid")
    inventory_value = _read_json(Path(str(inventory["path"])))
    inventory_indexes = inventory_value.get("indexes")
    if not isinstance(inventory_indexes, list) or any(
        not isinstance(item, dict) for item in inventory_indexes
    ):
        raise RuntimeError("consumer capability inventory indexes are invalid")
    inventory_unsigned = {
        key: item for key, item in inventory_value.items() if key != "inventory_sha256"
    }
    if (
        inventory_value.get("schema_version") != CONDITION_INVENTORY_SCHEMA
        or inventory_value.get("inventory_sha256")
        != canonical_sha256(inventory_unsigned)
        or inventory_value.get("launch_plan_sha256") != value.get("launch_plan_sha256")
        or inventory_value.get("controller_state_sha256")
        != value.get("controller_state_sha256")
        or Path(str(inventory_value.get("path_containment_root") or "")).resolve()
        != publication_root
        or {
            (
                str(item.get("stage_id") or ""),
                str(Path(str(item.get("path") or "")).resolve()),
            )
            for item in inventory_indexes
        }
        != {
            (
                str(item.get("stage_id") or ""),
                str(Path(str(item.get("path") or "")).resolve()),
            )
            for item in indexes
        }
    ):
        raise RuntimeError("consumer capability inventory identity is invalid")
    for item in indexes:
        path = Path(str(item.get("path") or ""))
        if not _path_is_within(path, publication_root) or not _record_matches(
            path, item
        ):
            raise RuntimeError("consumer condition index advanced past its receipt")
        load_compact_condition_index(path)
    handoff_binding = value.get("v1_handoff")
    if publish_mode == "canonical":
        if not isinstance(handoff_binding, dict):
            raise RuntimeError("canonical consumer capability lacks v1 handoff")
        receipt_file = handoff_binding.get("receipt_file")
        if not isinstance(receipt_file, dict) or not _record_matches(
            Path(str(receipt_file.get("path") or "")), receipt_file
        ):
            raise RuntimeError("canonical consumer v1 handoff file advanced")
        handoff = validate_v1_handoff_receipt(
            _read_json(Path(str(receipt_file["path"]))),
            require_live_pointer_match=False,
        )
        if (
            handoff_binding.get("receipt_sha256") != handoff["receipt_sha256"]
            or handoff_binding.get("old_writer_pid") != handoff["old_writer_pid"]
            or handoff_binding.get("old_writer_exited") is not True
            or handoff_binding.get("sealed_v1_condition_index_count") != len(BY_ID)
        ):
            raise RuntimeError("canonical consumer v1 handoff identity is invalid")
    elif handoff_binding is not None:
        raise RuntimeError("shadow consumer capability claims a v1 handoff")
    lease = value.get("writer_lease")
    if not isinstance(lease, dict) or lease.get("run_id") != value.get("run_id"):
        raise RuntimeError("consumer capability writer lease identity is invalid")
    _validate_writer_lease_attestation(lease)
    return value


def consume_once(
    inputs: ConsumerInputs,
    *,
    scheduler: SchedulerReader,
    remote: RemoteReader,
    runtime: Path,
    output_root: Path,
    protected_current7_index: Path = DEFAULT_CURRENT7_INDEX,
    apply: bool = False,
    publish_mode: str = "shadow",
    handoff_receipt: Mapping[str, Any] | None = None,
    writer_lease: WriterLease | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    runtime = runtime.resolve()
    output_root = output_root.resolve()
    _assert_runtime_isolated(runtime, protected_current7_index)
    if publish_mode not in {"shadow", "canonical"}:
        raise RuntimeError("publish mode must be shadow or canonical")
    if publish_mode == "canonical" and output_root != runtime:
        raise RuntimeError("canonical publication root must equal the campaign runtime")
    if publish_mode == "shadow" and output_root == runtime:
        raise RuntimeError("shadow publication must not share the live runtime root")
    if publish_mode == "canonical" and handoff_receipt is None:
        raise RuntimeError("canonical publication requires a v1 handoff receipt")
    if publish_mode == "canonical":
        all_compact = all(
            (
                runtime / "conditions" / stage.stage_id / "canonical" / POINTER_NAME
            ).is_file()
            and _read_json(
                runtime / "conditions" / stage.stage_id / "canonical" / POINTER_NAME
            ).get("schema_version")
            == COMPACT_INDEX_SCHEMA
            for stage in STAGES
        )
        validate_v1_handoff_receipt(
            handoff_receipt or {}, require_live_pointer_match=not all_compact
        )
        if apply and (writer_lease is None or writer_lease.stream is None):
            raise RuntimeError("canonical publication requires the active writer lease")
    inventory, pending_scheduler_cache, cache_hits = _scheduler_inventory(
        inputs,
        scheduler=scheduler,
        runtime=output_root,
        apply=apply,
    )
    heartbeat = observed_at or _utc_now()
    stages = []
    total_writes = 0
    index_inventory = []
    projections = []
    for stage in STAGES:
        stage_id = stage.stage_id
        stage_runtime = output_root / "conditions" / stage_id
        pointer_root = stage_runtime / "canonical"
        parent_path, embedded = _stage_parent_path(
            stage_id=stage_id,
            runtime=runtime,
            output_root=output_root,
            publish_mode=publish_mode,
            handoff=handoff_receipt,
        )
        historical, prior_static, _prior_lanes = _load_parent(
            parent_path, embedded, stage_id=stage_id
        )
        historical_identities = {
            (str(item["bundle_id"]), int(item["seed"])) for item in historical
        }
        lanes = []
        new_records = []
        refusals = []
        for item in [row for row in inventory if row["stage_id"] == stage_id]:
            protocol = str(item["protocol_version"])
            lane = {
                key: copy.deepcopy(item.get(key))
                for key in (
                    "task_id",
                    "stage_id",
                    "protocol_version",
                    "name",
                    "dedupe_key",
                    "state",
                    "account_name",
                    "created_at",
                    "started_at",
                    "finished_at",
                )
            }
            lane["batch_length"] = int(item.get("batch_length") or 1)
            lane["sealed_child_count"] = 1 if item["state"] in TERMINAL_STATES else 0
            if protocol == PROTOCOL_VERSION and item["state"] not in {
                "queued",
                "attaching",
            }:
                parent = _batch_parent_task(
                    item,
                    plan=item["_binding"]["plan"],
                    manifest=item["_binding"]["manifest"],
                )
                batch_manifest, task_status, remote_root = _read_batch_journal(
                    item, parent_task=parent, remote=remote
                )
                lane.update(
                    {
                        "batch_length": int(batch_manifest["batch_length"]),
                        "sealed_child_count": int(task_status["sealed_child_count"]),
                        "current_seed": task_status.get("current_seed"),
                    }
                )
                mixed = harvest_mixed_inventory(
                    [item],
                    plan=item["_binding"]["plan"],
                    manifest=item["_binding"]["manifest"],
                    remote=remote,
                    result_validator=_v2_result_validator(item["_expected_task"]),
                    journal_snapshots={
                        int(item["task_id"]): (
                            parent,
                            batch_manifest,
                            task_status,
                            remote_root,
                        )
                    },
                    strict=True,
                )
                if mixed.get("refusals"):
                    raise RuntimeError(
                        "v2 result/journal authentication refusal: "
                        + json.dumps(mixed["refusals"], sort_keys=True)
                    )
                new_records.extend(mixed["records"])
            elif protocol == SINGLE_SEED_PROTOCOL:
                payload = item["_expected_task"]["payload_json"]
                identity = (str(payload["bundle_id"]), int(payload["seed"]))
                if (
                    item["state"] in TERMINAL_STATES
                    and identity not in historical_identities
                ):
                    mixed = harvest_mixed_inventory(
                        [item],
                        plan=item["_binding"]["plan"],
                        manifest=item["_binding"]["manifest"],
                        remote=remote,
                        result_validator=_result_validator(
                            item["_expected_task"],
                            str(item["_entry"]["source_resource_policy_id"]),
                        ),
                        strict=True,
                    )
                    if mixed.get("refusals"):
                        raise RuntimeError(
                            "v1 terminal result authentication refusal: "
                            + json.dumps(mixed["refusals"], sort_keys=True)
                        )
                    new_records.extend(mixed["records"])
            lanes.append(lane)
        cached_records, cache_writes = cache_seed_records(
            stage_runtime, new_records, apply=False
        )
        if cache_writes != 0:
            raise RuntimeError("dry-run cache projection unexpectedly wrote files")
        records = _merge_records(historical, cached_records)
        successor_binding = inputs.cohorts_by_plan_sha[
            str(inputs.plan["launch_plan_sha256"])
        ]["bindings"][stage_id]
        static = _frontend_static(
            prior_static,
            binding=successor_binding,
            stage_runtime=stage_runtime,
        )
        source_cohorts = _source_cohort_projection(inputs, stage_id=stage_id)
        aggregate = _aggregate_records(records, new_records)
        aggregate.update(
            authenticated_constraint_identity(records, successor_binding["manifest"])
        )
        static.update(
            {
                "harvest_observed_at": heartbeat,
                "final_goal_stage_id": stage_id,
                "final_goal_stage_profile_sha256": stage_profile(stage)["sha256"],
                "refused_terminal_count": len(refusals),
                "refusals": refusals,
                "aggregate": aggregate,
                "consumer_source_bundle_cohorts": source_cohorts,
                "consumer_source_bundle_cohorts_sha256": canonical_sha256(
                    source_cohorts
                ),
                "consumer_capabilities": copy.deepcopy(CAPABILITIES),
            }
        )
        snapshot = build_compact_snapshot(
            stage_id=stage_id,
            physical_lanes=_hot_lanes(lanes),
            seed_records=records,
            frontend_static=static,
            updated_at=heartbeat,
        )
        status, _manifest, _shards, index = snapshot
        pointer = pointer_root / POINTER_NAME
        writes = 0
        index_inventory.append(
            {
                "stage_id": stage_id,
                "path": str(pointer.resolve()),
                "schema_version": COMPACT_INDEX_SCHEMA,
                "projected_file_sha256": hashlib.sha256(json_bytes(index)).hexdigest(),
                "index_sha256": index["index_sha256"],
                "physical_lane_count": status["physical_lane_count"],
                "logical_seed_count": status["logical_seed_count"],
                "authenticated_terminal_seed_count": status[
                    "authenticated_terminal_seed_count"
                ],
            }
        )
        total_writes += writes
        projections.append(
            {
                "stage_runtime": stage_runtime,
                "pointer_root": pointer_root,
                "new_records": new_records,
                "cached_records": cached_records,
                "snapshot": snapshot,
            }
        )
        stages.append(
            {
                "stage_id": stage_id,
                "physical_lane_count": status["physical_lane_count"],
                "logical_seed_count": status["logical_seed_count"],
                "authenticated_terminal_seed_count": len(records),
                "new_authenticated_seed_count": len(cached_records),
                "refused_physical_task_count": len(refusals),
                "publication_write_count": writes,
            }
        )
    condition_inventory = _condition_inventory(
        output_root=output_root,
        inputs=inputs,
        indexes=index_inventory,
        observed_at=heartbeat,
    )
    inventory_path = output_root / "canonical" / "condition-indexes.json"
    if apply:
        for path, payload in pending_scheduler_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and path.read_bytes() == payload:
                continue
            total_writes += _atomic_json(path, _strict_json(payload, str(path)))
        for stage_result, projection in zip(stages, projections, strict=True):
            applied_records, writes = cache_seed_records(
                projection["stage_runtime"],
                projection["new_records"],
                apply=True,
            )
            if applied_records != projection["cached_records"]:
                raise RuntimeError("seed cache projection changed before apply")
            writes += publish_compact_snapshot(
                projection["pointer_root"],
                *projection["snapshot"],
                pointer_name=POINTER_NAME,
            )
            stage_result["publication_write_count"] = writes
            total_writes += writes
        total_writes += _atomic_json(inventory_path, condition_inventory)
    return {
        "schema_version": CONSUMER_SCHEMA,
        "apply": bool(apply),
        "publish_mode": publish_mode,
        "observed_at": heartbeat,
        "launch_plan_sha256": inputs.plan["launch_plan_sha256"],
        "controller_state_sha256": inputs.state["state_sha256"],
        "physical_scheduler_task_count": len(inventory),
        "scheduler_terminal_cache_hit_count": cache_hits,
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
        "local_write_count": total_writes,
        "virtual_scheduler_task_ids_created": False,
        "aedt_used": False,
        "fea_submission_performed": False,
        "condition_inventory": condition_inventory,
        "condition_inventory_path": str(inventory_path.resolve()),
        "stages": stages,
    }


def build_stop_handoff(
    *, run_id: str, capability_receipt_path: Path | None
) -> dict[str, Any]:
    capability = (
        _file_record(capability_receipt_path)
        if capability_receipt_path is not None and capability_receipt_path.is_file()
        else None
    )
    unsigned = {
        "schema_version": STOP_HANDOFF_SCHEMA,
        "run_id": run_id,
        "stopped_at": _utc_now(),
        "clean_shutdown": True,
        "last_capability_receipt": capability,
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def validate_stop_handoff(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    capability = value.get("last_capability_receipt")
    if (
        value.get("schema_version") != STOP_HANDOFF_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("clean_shutdown") is not True
        or not str(value.get("run_id") or "")
        or value.get("scheduler_mutation_count") != 0
        or value.get("remote_write_count") != 0
        or (
            capability is not None
            and (
                not isinstance(capability, dict)
                or not _record_matches(
                    Path(str(capability.get("path") or "")), capability
                )
            )
        )
    ):
        raise RuntimeError("multi-seed consumer stop handoff is invalid")
    return copy.deepcopy(dict(value))


def _named_stage_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        stage_id, separator, path = value.partition("=")
        if not separator or stage_id not in BY_ID or stage_id in result or not path:
            raise RuntimeError("condition index must be STAGE_ID=PATH")
        result[stage_id] = Path(path)
    return result


def _transient_transport_error(error: BaseException) -> bool:
    """Only retry failures that explicitly prove a transport outage."""

    if isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    return isinstance(error, OSError) and getattr(error, "errno", None) in {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--launch-plan", type=Path, required=True)
    run.add_argument("--bindings", type=Path, required=True)
    run.add_argument("--controller-state", type=Path, required=True)
    run.add_argument("--source-launch-plan", type=Path, action="append", default=[])
    run.add_argument("--source-bindings", type=Path, action="append", default=[])
    run.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    run.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    run.add_argument("--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE)
    run.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    run.add_argument("--output-root", type=Path)
    run.add_argument(
        "--protected-current7-index", type=Path, default=DEFAULT_CURRENT7_INDEX
    )
    run.add_argument(
        "--publish-mode", choices=("shadow", "canonical"), default="shadow"
    )
    run.add_argument("--handoff-receipt", type=Path)
    run.add_argument("--capability-receipt", type=Path)
    run.add_argument("--apply", action="store_true")
    run.add_argument("--watch", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    run.add_argument("--freshness-seconds", type=int, default=DEFAULT_FRESHNESS_SECONDS)
    run.add_argument("--stop-file", type=Path)
    run.add_argument("--previous-stop-handoff", type=Path)
    seal = commands.add_parser("seal-v1-handoff")
    seal.add_argument("--old-pid", type=int, required=True)
    seal.add_argument("--stop-file", type=Path, required=True)
    seal.add_argument("--condition-index", action="append", default=[], required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--apply", action="store_true")
    check = commands.add_parser("check-capability")
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument("--maximum-age-seconds", type=int)
    check.add_argument(
        "--require-publish-mode",
        choices=("canonical", "shadow"),
        default="canonical",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command
    if command == "seal-v1-handoff":
        receipt = build_v1_handoff_receipt(
            old_pid=args.old_pid,
            stop_file=args.stop_file,
            condition_indexes=_named_stage_paths(args.condition_index),
        )
        if args.apply:
            _atomic_json(args.output, receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    if command == "check-capability":
        receipt = require_consumer_capability(
            args.receipt,
            maximum_age_seconds=args.maximum_age_seconds,
            required_publish_mode=args.require_publish_mode,
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    if args.watch and not args.apply:
        raise RuntimeError("--watch requires --apply")
    if not 0 < float(args.poll_seconds) <= MAX_POLL_SECONDS:
        raise RuntimeError("--poll-seconds must be in (0, 60]")
    if int(args.freshness_seconds) < int(args.poll_seconds):
        raise RuntimeError("freshness deadline must cover one poll interval")
    runtime = args.runtime.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (runtime / DEFAULT_SHADOW_NAME).resolve()
    )
    handoff = None
    if args.handoff_receipt is not None:
        handoff = validate_v1_handoff_receipt(
            _read_json(args.handoff_receipt.resolve(strict=True)),
            require_live_pointer_match=args.publish_mode == "canonical"
            and not all(
                (
                    runtime / "conditions" / stage.stage_id / "canonical" / POINTER_NAME
                ).is_file()
                and _read_json(
                    runtime / "conditions" / stage.stage_id / "canonical" / POINTER_NAME
                ).get("schema_version")
                == COMPACT_INDEX_SCHEMA
                for stage in STAGES
            ),
        )

    def cycle_inputs() -> ConsumerInputs:
        return load_inputs(
            args.launch_plan,
            args.bindings,
            args.controller_state,
            source_launch_plan_paths=args.source_launch_plan,
            source_bindings_paths=args.source_bindings,
        )

    scheduler = Final1000ReadOnlySchedulerApi(args.scheduler_url)
    remote = AccountSftpReader(args.accounts, args.scheduler_source)
    lease_path = output_root / "canonical" / ".multiseed-consumer.lock"
    capability_path = args.capability_receipt or (
        output_root / "canonical" / "multiseed-consumer-capability.json"
    )
    stop_handoff_path = output_root / "canonical" / "multiseed-consumer-stop.json"
    previous_stop = args.previous_stop_handoff
    if previous_stop is None and stop_handoff_path.is_file():
        previous_stop = stop_handoff_path
    if previous_stop is not None:
        validate_stop_handoff(_read_json(previous_stop.resolve(strict=True)))
    try:
        if not args.apply:
            result = consume_once(
                cycle_inputs(),
                scheduler=scheduler,
                remote=remote,
                runtime=runtime,
                output_root=output_root,
                protected_current7_index=args.protected_current7_index,
                apply=False,
                publish_mode=args.publish_mode,
                handoff_receipt=handoff,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        with WriterLease(lease_path) as lease:
            transient_failures = 0
            while True:
                if args.stop_file is not None and args.stop_file.is_file():
                    break
                try:
                    result = consume_once(
                        cycle_inputs(),
                        scheduler=scheduler,
                        remote=remote,
                        runtime=runtime,
                        output_root=output_root,
                        protected_current7_index=args.protected_current7_index,
                        apply=args.apply,
                        publish_mode=args.publish_mode,
                        handoff_receipt=handoff,
                        writer_lease=lease,
                    )
                    if args.apply:
                        receipt = build_capability_receipt(
                            result=result,
                            inventory_path=Path(result["condition_inventory_path"]),
                            poll_seconds=args.poll_seconds,
                            freshness_deadline_seconds=args.freshness_seconds,
                            runtime_root=runtime,
                            writer_lease=lease,
                            handoff_receipt_path=(
                                args.handoff_receipt
                                if args.publish_mode == "canonical"
                                else None
                            ),
                        )
                        _atomic_json(capability_path, receipt)
                        result["capability_receipt_sha256"] = receipt["receipt_sha256"]
                        result["local_write_count"] += 1
                    transient_failures = 0
                    print(json.dumps(result, sort_keys=True), flush=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    if not args.watch or not _transient_transport_error(exc):
                        raise
                    transient_failures += 1
                    if transient_failures > 5:
                        raise RuntimeError(
                            "consumer transport retry budget exhausted"
                        ) from exc
                    print(
                        json.dumps(
                            {
                                "schema_version": CONSUMER_SCHEMA,
                                "apply": True,
                                "error": f"{type(exc).__name__}:{exc}",
                                "scheduler_mutation_count": 0,
                                "remote_write_count": 0,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    time.sleep(min(args.poll_seconds, 2 ** (transient_failures - 1)))
                    continue
                if not args.watch:
                    break
                time.sleep(args.poll_seconds)
            if args.apply:
                _atomic_json(
                    stop_handoff_path,
                    build_stop_handoff(
                        run_id=lease.run_id,
                        capability_receipt_path=capability_path,
                    ),
                )
    finally:
        remote.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
