"""Authenticate and publish additive current7 Slurm terminal snapshots.

This reader has no scheduler mutation method.  It inventories tasks with GET,
reads terminal artifacts through SFTP, and requires two identical reads across
three stable stat observations before accepting a file.  Local writes are also
disabled unless ``--apply`` is explicit.  Publication is additive:
``canonical/index.json`` (the deployed all11 source) is never touched; current7
uses ``canonical/current7-index.json`` until a compatible monitor is deployed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error
import urllib.request

try:
    from tier1_corrected_current7_slurm_bundle import (
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        build_task_payload,
        canonical_sha256,
    )
    from tier1_corrected_current7_slurm_controller import CONTROLLER_STATE_SCHEMA
    from tier1_corrected_current7_slurm_publish import load_bundle_plan, read_json
    from tier1_corrected_current7_slurm_seed_runner import validate_result
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_slurm_bundle import (
        RESULT_SCHEMA,
        STATUS_SCHEMA,
        build_task_payload,
        canonical_sha256,
    )
    from tools.tier1_corrected_current7_slurm_controller import (
        CONTROLLER_STATE_SCHEMA,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        load_bundle_plan,
        read_json,
    )
    from tools.tier1_corrected_current7_slurm_seed_runner import validate_result


HARVEST_RESULT_SCHEMA = "mft-tier1-current7-slurm-harvest-result-v1"
CACHE_RECORD_SCHEMA = "mft-tier1-current7-content-addressed-seed-record-v1"
COHORT_STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
CURRENT7_INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
AGGREGATE_SCHEMA = "mft-tier1-current7-slurm-aggregate-v1"
COMPATIBILITY_SCHEMA = "mft-tier1-current7-monitor-compatibility-v1"
SNAPSHOT_IDENTITY_SCHEMA = "mft-tier1-current7-snapshot-identity-v1"
LEGACY_MONITOR_GENERATION = "84ff69f-cap-parity-overlay-20260719"

DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_RUNTIME = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling"
)
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"Y:\git\slurm_scheduler")

TERMINAL_SCHEDULER_STATES = frozenset({"completed", "failed", "cancelled"})
FAIL_CLOSED_FLAGS = (
    "production_eligible",
    "fea_submission_approved",
    "fea_submission_performed",
    "aedt_used",
    "automatic_promotion_allowed",
)
MAX_STATUS_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
PARETO_CANDIDATES_SCHEMA = "mft-tier1-current7-pareto-candidates-v1"
LEAST_VIOLATION_CANDIDATES_SCHEMA = (
    "mft-tier1-current7-least-violation-candidates-v1"
)
INFEASIBILITY_REPORT_SCHEMA = "mft-tier1-current7-infeasibility-report-v1"
REQUIRED_SEARCH_ARTIFACTS = frozenset(
    {
        "pareto_X",
        "pareto_F",
        "pareto_G_physical",
        "pareto_front",
        "pareto_candidates",
        "terminal_X",
        "terminal_F",
        "terminal_G_optimizer",
        "terminal_G_physical",
        "least_violation_X",
        "least_violation_F",
        "least_violation_G_physical",
        "least_violation_front",
        "least_violation_candidates",
        "infeasibility_report",
    }
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_json(value: bytes, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")

    try:
        parsed = json.loads(
            value.decode("utf-8"), parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON object: {label}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"JSON object required: {label}")
    return parsed


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} is not a SHA-256 digest")
    return normalized


def _fail_closed(value: Mapping[str, Any], label: str) -> None:
    if any(value.get(field) is not False for field in FAIL_CLOSED_FLAGS):
        raise RuntimeError(f"{label} is not fail-closed")


def _remote_child(root: str, *parts: str) -> str:
    base = PurePosixPath(str(root))
    if not base.is_absolute() or any(part in ("", ".", "..") for part in base.parts):
        raise RuntimeError("remote bundle root is unsafe")
    result = base
    for raw in parts:
        relative = PurePosixPath(str(raw))
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise RuntimeError("remote artifact suffix is unsafe")
        result /= relative
    if base not in result.parents:
        raise RuntimeError("remote artifact escaped the immutable bundle")
    return result.as_posix()


@dataclass(frozen=True)
class RemoteFileStat:
    size: int
    mtime: int
    mode: int = 0


@dataclass(frozen=True)
class StableRemoteFile:
    path: str
    payload: bytes
    sha256: str
    size: int
    stat: RemoteFileStat


@dataclass(frozen=True)
class StableRemoteJson(StableRemoteFile):
    value: dict[str, Any]


class SchedulerReader(Protocol):
    def get_task(self, task_id: int) -> Mapping[str, Any] | None: ...


class RemoteReader(Protocol):
    def stat(self, account_name: str, path: str) -> RemoteFileStat: ...

    def read_bytes(
        self, account_name: str, path: str, *, maximum_bytes: int
    ) -> bytes: ...


class ReadOnlySchedulerApi:
    """GET-only scheduler client; it intentionally exposes no mutation API."""

    def __init__(self, base_url: str = DEFAULT_SCHEDULER_URL, timeout: float = 20):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.get_count = 0

    def get_task(self, task_id: int) -> Mapping[str, Any] | None:
        request = urllib.request.Request(
            self.base_url + f"/api/tasks/{int(task_id)}", method="GET"
        )
        self.get_count += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = _strict_json(response.read(), f"scheduler task {task_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"scheduler GET task {task_id} failed: {detail}"
            ) from exc
        return value


class AccountSftpReader:
    """Read-only account-aware SFTP transport backed by scheduler config."""

    def __init__(self, accounts_path: Path, scheduler_source: Path):
        source = scheduler_source.resolve(strict=True)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from slurm_scheduler.config import load_accounts
        from slurm_scheduler.slurm import SSHSession

        self._session_type = SSHSession
        self._accounts = {
            account.name: account for account in load_accounts(accounts_path)
        }
        self._sessions: dict[str, Any] = {}
        self.read_count = 0
        self.stat_count = 0

    def _session(self, account_name: str):
        if account_name not in self._accounts:
            raise RuntimeError(f"unknown scheduler account: {account_name}")
        session = self._sessions.get(account_name)
        if session is None:
            session = self._session_type(self._accounts[account_name])
            session.ensure_connected()
            self._sessions[account_name] = session
        return session

    def stat(self, account_name: str, path: str) -> RemoteFileStat:
        session = self._session(account_name)
        sftp = session.client.open_sftp()
        self.stat_count += 1
        try:
            value = sftp.stat(path)
            return RemoteFileStat(
                size=int(value.st_size),
                mtime=int(value.st_mtime),
                mode=int(value.st_mode),
            )
        finally:
            sftp.close()

    def read_bytes(
        self, account_name: str, path: str, *, maximum_bytes: int
    ) -> bytes:
        session = self._session(account_name)
        sftp = session.client.open_sftp()
        self.read_count += 1
        try:
            with sftp.file(path, "rb") as stream:
                value = stream.read(int(maximum_bytes) + 1)
        finally:
            sftp.close()
        if len(value) > int(maximum_bytes):
            raise RuntimeError(f"remote artifact exceeds byte limit: {path}")
        return bytes(value)

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()


def read_stable_remote_file(
    reader: RemoteReader,
    *,
    account_name: str,
    path: str,
    maximum_bytes: int,
) -> StableRemoteFile:
    """Require three equal stats and two byte-identical reads."""

    first_stat = reader.stat(account_name, path)
    if first_stat.size <= 0 or first_stat.size > maximum_bytes:
        raise RuntimeError(f"remote artifact size is invalid: {path}")
    first = reader.read_bytes(
        account_name, path, maximum_bytes=maximum_bytes
    )
    middle_stat = reader.stat(account_name, path)
    second = reader.read_bytes(
        account_name, path, maximum_bytes=maximum_bytes
    )
    last_stat = reader.stat(account_name, path)
    first_sha = _sha_bytes(first)
    second_sha = _sha_bytes(second)
    if (
        first_stat != middle_stat
        or middle_stat != last_stat
        or len(first) != first_stat.size
        or len(second) != first_stat.size
        or first_sha != second_sha
    ):
        raise RuntimeError(f"remote artifact changed during stable read: {path}")
    return StableRemoteFile(
        path=path,
        payload=first,
        sha256=first_sha,
        size=len(first),
        stat=first_stat,
    )


def read_stable_remote_json(
    reader: RemoteReader,
    *,
    account_name: str,
    path: str,
    maximum_bytes: int,
) -> StableRemoteJson:
    stable = read_stable_remote_file(
        reader,
        account_name=account_name,
        path=path,
        maximum_bytes=maximum_bytes,
    )
    return StableRemoteJson(
        path=stable.path,
        payload=stable.payload,
        sha256=stable.sha256,
        size=stable.size,
        stat=stable.stat,
        value=_strict_json(stable.payload, path),
    )


def _validate_controller_state(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    if (
        state.get("schema_version") != CONTROLLER_STATE_SCHEMA
        or state.get("bundle_id") != plan.get("bundle_id")
        or state.get("bundle_manifest_sha256")
        != plan.get("bundle_manifest_sha256")
        or state.get("state_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("controller state identity/SHA mismatch")
    submissions = state.get("submissions")
    if not isinstance(submissions, dict):
        raise RuntimeError("controller submissions map is invalid")
    ids: list[int] = []
    for dedupe, record in submissions.items():
        if not isinstance(record, dict) or record.get("dedupe_key") != dedupe:
            raise RuntimeError("controller submission identity mismatch")
        task_id = int(record.get("task_id") or 0)
        if task_id <= 0:
            raise RuntimeError("controller submission task id is invalid")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise RuntimeError("controller submission task id is duplicated")
    return copy.deepcopy(dict(state))


def inventory_tasks(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    scheduler: SchedulerReader,
    *,
    payload_builder: Callable[..., Mapping[str, Any]] = build_task_payload,
) -> list[dict[str, Any]]:
    state = _validate_controller_state(controller_state, plan)
    priority = state.get("priority_override")
    inventory: list[dict[str, Any]] = []
    for dedupe, submission in sorted(
        state["submissions"].items(), key=lambda item: int(item[1]["task_id"])
    ):
        task_id = int(submission["task_id"])
        task = scheduler.get_task(task_id)
        if task is None:
            inventory.append(
                {
                    **submission,
                    "status": "missing",
                    "account_name": "",
                    "inventory_error": "scheduler_task_missing",
                }
            )
            continue
        expected = payload_builder(
            plan, manifest, seed=int(submission["seed"]), priority=priority
        )
        observed_id = int(task.get("id") or task.get("task_id") or 0)
        if (
            observed_id != task_id
            or str(task.get("name") or "") != str(expected["name"])
            or str(task.get("dedupe_key") or "") != dedupe
            or str(expected["dedupe_key"]) != dedupe
        ):
            raise RuntimeError(f"scheduler task identity changed: {task_id}")
        inventory.append(
            {
                "task_id": task_id,
                "name": expected["name"],
                "dedupe_key": dedupe,
                "bundle_id": plan["bundle_id"],
                "seed": int(submission["seed"]),
                "island_id": submission["island_id"],
                "wave": submission["wave"],
                "status": str(task.get("status") or "unknown").lower(),
                "account_name": str(task.get("account_name") or ""),
                "slurm_job_id": str(task.get("slurm_job_id") or ""),
                "exit_code": task.get("exit_code"),
                "created_at": task.get("created_at"),
                "started_at": task.get("started_at"),
                "finished_at": task.get("finished_at"),
                "payload": copy.deepcopy(dict(expected["payload_json"])),
                "payload_sha256": canonical_sha256(expected["payload_json"]),
            }
        )
    return inventory


def _validate_terminal_status(
    status: Mapping[str, Any],
    *,
    item: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    scheduler_state = item["status"]
    state = str(status.get("state") or "")
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("terminal") is not True
        or str(status.get("task_id")) != str(item["task_id"])
        or int(status.get("seed", -1)) != int(item["seed"])
        or status.get("bundle_id") != plan["bundle_id"]
        or status.get("bundle_manifest_sha256")
        != plan["bundle_manifest_sha256"]
        or status.get("payload_sha256") != item["payload_sha256"]
        or status.get("island_id") != item["island_id"]
        or state not in {"completed", "failed", "cancelled"}
        or (scheduler_state == "completed" and state != "completed")
        or (scheduler_state != "completed" and state == "completed")
    ):
        raise RuntimeError("remote terminal seed status identity mismatch")
    _fail_closed(status, "remote terminal status")


def _artifact_json_flags(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("production_eligible") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError(f"{label} is not fail-closed")


def _harvest_result_artifacts(
    *,
    result: Mapping[str, Any],
    account_name: str,
    output_root: str,
    remote: RemoteReader,
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]]]:
    inventory = result.get("artifact_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != REQUIRED_SEARCH_ARTIFACTS
        or result.get("artifact_inventory_sha256") != canonical_sha256(inventory)
    ):
        raise RuntimeError("terminal result artifact inventory seal mismatch")
    objects: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    json_values: dict[str, dict[str, Any]] = {}
    for name, raw_record in sorted(inventory.items()):
        if not isinstance(raw_record, dict):
            raise RuntimeError("terminal artifact record is invalid")
        relative = str(raw_record.get("path") or "")
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.name != relative
        ):
            raise RuntimeError("terminal artifact path is not a safe basename")
        expected_size = int(raw_record.get("size_bytes") or 0)
        expected_sha = _hex(raw_record.get("sha256"), f"artifact {name} SHA")
        remote_path = _remote_child(output_root, relative)
        stable = read_stable_remote_file(
            remote,
            account_name=account_name,
            path=remote_path,
            maximum_bytes=MAX_ARTIFACT_BYTES,
        )
        if stable.size != expected_size or stable.sha256 != expected_sha:
            raise RuntimeError(f"terminal artifact bytes differ from inventory: {name}")
        objects[name] = {
            **copy.deepcopy(raw_record),
            "remote_path": remote_path,
            "stable_stat": {
                "size": stable.stat.size,
                "mtime": stable.stat.mtime,
                "mode": stable.stat.mode,
            },
        }
        payloads[name] = stable.payload
        if name in {
            "pareto_candidates",
            "least_violation_candidates",
            "infeasibility_report",
        }:
            json_values[name] = _strict_json(stable.payload, remote_path)

    pareto = json_values["pareto_candidates"]
    least = json_values["least_violation_candidates"]
    report = json_values["infeasibility_report"]
    pareto_candidates = pareto.get("candidates")
    least_candidates = least.get("candidates")
    if (
        pareto.get("schema_version") != PARETO_CANDIDATES_SCHEMA
        or pareto.get("authoritative_constraints")
        != "terminal_unscaled_physical_replay"
        or not isinstance(pareto_candidates, list)
        or pareto.get("candidate_count") != len(pareto_candidates)
        or any(not isinstance(item, dict) for item in pareto_candidates)
        or any(item.get("physical_feasible") is not True for item in pareto_candidates)
        or least.get("schema_version") != LEAST_VIOLATION_CANDIDATES_SCHEMA
        or least.get("ranking")
        != "minimum_sum_positive_optimizer_normalized_G"
        or not isinstance(least_candidates, list)
        or least.get("candidate_count") != len(least_candidates)
        or len(least_candidates) != 1
        or any(not isinstance(item, dict) for item in least_candidates)
        or report.get("schema_version") != INFEASIBILITY_REPORT_SCHEMA
        or report.get("authoritative_constraints")
        != "terminal_unscaled_physical_replay"
        or int(result.get("feasible_pareto_count", -1)) != len(pareto_candidates)
        or int(result.get("least_violation_count", -1)) != len(least_candidates)
        or int(result.get("terminal_population_count", -1))
        != int(report.get("population_size", -2))
        or int(result.get("physical_feasible_count", -1))
        != int(report.get("physical_feasible_count", -2))
    ):
        raise RuntimeError("terminal candidate/infeasibility artifact mismatch")
    _artifact_json_flags(pareto, "pareto candidates")
    _artifact_json_flags(least, "least-violation candidates")
    _artifact_json_flags(report, "infeasibility report")
    return objects, payloads, [*pareto_candidates, *least_candidates]


def harvest_terminal_task(
    item: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    remote: RemoteReader,
    result_validator: Callable[..., None] = validate_result,
) -> dict[str, Any]:
    if item.get("status") not in TERMINAL_SCHEDULER_STATES:
        raise RuntimeError("SFTP harvest is prohibited for a nonterminal task")
    account = str(item.get("account_name") or "")
    if not account:
        raise RuntimeError("terminal scheduler task has no account identity")
    task_id = int(item["task_id"])
    seed = int(item["seed"])
    status_path = _remote_child(
        str(plan["remote_bundle"]), "runs", f"task-{task_id}", "seed_status.json"
    )
    stable_status = read_stable_remote_json(
        remote,
        account_name=account,
        path=status_path,
        maximum_bytes=MAX_STATUS_BYTES,
    )
    _validate_terminal_status(stable_status.value, item=item, plan=plan)
    observation: dict[str, Any] = {
        "bundle_id": plan["bundle_id"],
        "seed": seed,
        "island_id": item["island_id"],
        "task_id": task_id,
        "scheduler_state": item["status"],
        "scheduler_exit_code": item.get("exit_code"),
        "account_name": account,
        "terminal_state": stable_status.value["state"],
        "status_object": {
            "sha256": stable_status.sha256,
            "size": stable_status.size,
            "remote_path": status_path,
            "stable_stat": {
                "size": stable_status.stat.size,
                "mtime": stable_status.stat.mtime,
                "mode": stable_status.stat.mode,
            },
        },
        "_status_bytes": stable_status.payload,
    }
    if stable_status.value["state"] != "completed":
        observation["failure"] = stable_status.value.get("failure")
        return observation

    result_path = _remote_child(
        str(plan["remote_bundle"]),
        "runs",
        f"task-{task_id}",
        f"seed-{seed}",
        str(manifest["search_execution"]["result_filename"]),
    )
    stable_result = read_stable_remote_json(
        remote,
        account_name=account,
        path=result_path,
        maximum_bytes=MAX_RESULT_BYTES,
    )
    if stable_result.sha256 != _hex(
        stable_status.value.get("result_sha256"), "remote status result_sha256"
    ):
        raise RuntimeError("remote result SHA does not match terminal seed status")
    if stable_result.value.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("remote result schema mismatch")
    result_validator(
        stable_result.value,
        payload=item["payload"],
        manifest=manifest,
    )
    _fail_closed(stable_result.value, "remote terminal result")
    output_root = _remote_child(
        str(plan["remote_bundle"]), "runs", f"task-{task_id}", f"seed-{seed}"
    )
    artifact_objects, artifact_payloads, candidates = _harvest_result_artifacts(
        result=stable_result.value,
        account_name=account,
        output_root=output_root,
        remote=remote,
    )
    observation.update(
        {
            "result_object": {
                "sha256": stable_result.sha256,
                "size": stable_result.size,
                "remote_path": result_path,
                "stable_stat": {
                    "size": stable_result.stat.size,
                    "mtime": stable_result.stat.mtime,
                    "mode": stable_result.stat.mode,
                },
            },
            "completed_generations": int(
                stable_result.value["completed_generations"]
            ),
            "feasible_pareto_count": int(
                stable_result.value.get("feasible_pareto_count", 0)
            ),
            "artifact_objects": artifact_objects,
            "_result_bytes": stable_result.payload,
            "_result": stable_result.value,
            "_artifact_payloads": artifact_payloads,
            "_candidates": candidates,
        }
    )
    return observation


def _public_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def deduplicate_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in observations:
        key = (str(item["bundle_id"]), int(item["seed"]))
        groups.setdefault(key, []).append(item)
    records: list[dict[str, Any]] = []
    for (bundle_id, seed), group in sorted(groups.items()):
        completed = [item for item in group if item.get("result_object")]
        result_hashes = {
            str(item["result_object"]["sha256"]) for item in completed
        }
        if len(result_hashes) > 1:
            raise RuntimeError(
                f"divergent terminal results for bundle+seed {bundle_id}:{seed}"
            )
        chosen = min(completed or group, key=lambda item: int(item["task_id"]))
        status_objects = sorted(
            (
                {
                    **copy.deepcopy(dict(item["status_object"])),
                    "task_id": int(item["task_id"]),
                }
                for item in group
            ),
            key=lambda item: item["task_id"],
        )
        record: dict[str, Any] = {
            "schema_version": CACHE_RECORD_SCHEMA,
            "bundle_id": bundle_id,
            "seed": seed,
            "island_id": chosen["island_id"],
            "task_ids": sorted({int(item["task_id"]) for item in group}),
            "scheduler_states": dict(
                sorted(Counter(str(item["scheduler_state"]) for item in group).items())
            ),
            "terminal_state": chosen["terminal_state"],
            "status_objects": status_objects,
            "authenticated": True,
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
            "_status_payloads": {
                str(item["status_object"]["sha256"]): item["_status_bytes"]
                for item in group
            },
        }
        if completed:
            result_value = chosen["_result"]
            record.update(
                {
                    "terminal_state": "completed",
                    "result_object": copy.deepcopy(dict(chosen["result_object"])),
                    "completed_generations": chosen["completed_generations"],
                    "feasible_pareto_count": chosen["feasible_pareto_count"],
                    "constraint_version": result_value["constraint_version"],
                    "hard_spec_sha256": result_value["stage_spec_sha256"],
                    "hard_constraint_contract_sha256": result_value[
                        "hard_constraint_contract_sha256"
                    ],
                    "temperature_contract_sha256": result_value[
                        "temperature_contract_sha256"
                    ],
                    "artifact_objects": copy.deepcopy(
                        dict(chosen["artifact_objects"])
                    ),
                    "_result_bytes": chosen["_result_bytes"],
                    "_result": copy.deepcopy(dict(result_value)),
                    "_artifact_payloads": copy.deepcopy(
                        dict(chosen["_artifact_payloads"])
                    ),
                    "_candidates": copy.deepcopy(list(chosen["_candidates"])),
                }
            )
        else:
            record["failure"] = chosen.get("failure")
        records.append(record)
    return records


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"candidate {label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"candidate {label} is not numeric") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"candidate {label} is not finite")
    return result


def _constraint_values(candidate: Mapping[str, Any]) -> list[float] | None:
    raw = None
    for name in ("physical_constraint_G", "constraint_G", "constraints", "G"):
        if name in candidate:
            raw = candidate[name]
            break
    if raw is None:
        return None
    values = raw.values() if isinstance(raw, dict) else raw
    if not isinstance(values, (list, tuple, dict_values_type())):
        raise RuntimeError("candidate constraint vector is invalid")
    return [_finite_number(value, "constraint") for value in values]


def dict_values_type():
    return type({}.values())


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    decoded = candidate.get("decoded_params")
    if isinstance(decoded, dict) and decoded:
        return canonical_sha256(decoded)
    return canonical_sha256(candidate)


def _candidate_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        result = record.get("_result")
        if not isinstance(result, dict):
            continue
        raw_candidates = record.get("_candidates")
        if raw_candidates is None:
            raw_candidates = result.get("candidates")
        if raw_candidates is None:
            raw_candidates = result.get("pareto_candidates") or []
        if not isinstance(raw_candidates, list):
            raise RuntimeError("terminal result candidates inventory is invalid")
        for position, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                raise RuntimeError("terminal candidate must be an object")
            candidate = copy.deepcopy(raw)
            volume = _finite_number(candidate.get("volume_L"), "volume_L")
            loss = _finite_number(candidate.get("total_loss_W"), "total_loss_W")
            if volume < 0 or loss < 0:
                raise RuntimeError("candidate objective is negative")
            constraints = _constraint_values(candidate)
            explicit = candidate.get("physical_feasible", candidate.get("feasible"))
            computed = None if constraints is None else all(x <= 0 for x in constraints)
            if explicit is not None and not isinstance(explicit, bool):
                raise RuntimeError("candidate feasible flag is invalid")
            if explicit is not None and computed is not None and explicit != computed:
                raise RuntimeError("candidate feasible flag contradicts constraints")
            feasible = bool(explicit if explicit is not None else computed)
            raw_violation = candidate.get("total_positive_violation")
            if raw_violation is None and constraints is not None:
                violation = sum(max(0.0, value) for value in constraints)
            elif raw_violation is not None:
                violation = _finite_number(
                    raw_violation, "total_positive_violation"
                )
            elif feasible:
                violation = 0.0
            else:
                raise RuntimeError("infeasible candidate has no violation evidence")
            if violation < 0 or (feasible and violation > 1e-12):
                raise RuntimeError("candidate violation evidence is inconsistent")
            identity = _candidate_identity(candidate)
            row = {
                **candidate,
                "candidate_identity_sha256": identity,
                "source_bundle_id": record["bundle_id"],
                "source_seed": int(record["seed"]),
                "source_candidate_position": position,
                "volume_L": volume,
                "total_loss_W": loss,
                "feasible": feasible,
                "total_positive_violation": violation,
            }
            prior = unique.get(identity)
            if prior is not None and (
                prior["volume_L"],
                prior["total_loss_W"],
                prior["feasible"],
                prior["total_positive_violation"],
            ) != (
                volume,
                loss,
                feasible,
                violation,
            ):
                raise RuntimeError("candidate identity has divergent objective evidence")
            unique.setdefault(identity, row)
    return list(unique.values())


def aggregate_candidates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = _candidate_rows(records)
    feasible = [row for row in rows if row["feasible"]]
    pareto = []
    for row in feasible:
        if any(
            other["volume_L"] <= row["volume_L"]
            and other["total_loss_W"] <= row["total_loss_W"]
            and (
                other["volume_L"] < row["volume_L"]
                or other["total_loss_W"] < row["total_loss_W"]
            )
            for other in feasible
            if other is not row
        ):
            continue
        pareto.append(row)
    pareto.sort(key=lambda item: (item["volume_L"], item["total_loss_W"]))
    least = min(
        rows,
        key=lambda item: (
            item["total_positive_violation"],
            item["volume_L"],
            item["total_loss_W"],
            item["candidate_identity_sha256"],
        ),
        default=None,
    )
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "authenticated_seed_count": len(records),
        "candidate_count": len(rows),
        "feasible_candidate_count": len(feasible),
        "pareto_count": len(pareto),
        "pareto_candidates": pareto[:256],
        "pareto_preview_truncated": len(pareto) > 256,
        "least_violation_candidate": least,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
    }


def authenticated_constraint_identity(
    records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "constraint_version": manifest.get("constraint_version"),
        "hard_spec": copy.deepcopy(manifest.get("hard_spec")),
        "hard_spec_sha256": manifest.get("hard_spec_sha256"),
        "hard_constraint_contract_sha256": manifest.get(
            "hard_constraint_contract_sha256"
        ),
        "temperature_contract_sha256": manifest.get(
            "temperature_contract_sha256"
        ),
        "constraint_names": copy.deepcopy(manifest.get("constraint_names")),
        "temperature_targets": copy.deepcopy(manifest.get("temperature_targets")),
    }
    if (
        not isinstance(expected["constraint_version"], str)
        or not expected["constraint_version"]
        or not isinstance(expected["hard_spec"], dict)
        or not expected["hard_spec"]
        or expected["hard_spec_sha256"]
        != canonical_sha256(expected["hard_spec"])
        or not isinstance(expected["constraint_names"], list)
        or not expected["constraint_names"]
        or any(
            not isinstance(name, str) or not name
            for name in expected["constraint_names"]
        )
        or len(expected["constraint_names"])
        != len(set(expected["constraint_names"]))
        or not isinstance(expected["temperature_targets"], list)
        or not expected["temperature_targets"]
        or any(
            not isinstance(target, str) or not target
            for target in expected["temperature_targets"]
        )
        or len(expected["temperature_targets"])
        != len(set(expected["temperature_targets"]))
    ):
        raise RuntimeError("bundle hard-constraint identity is invalid")
    expected["hard_spec_sha256"] = _hex(
        expected["hard_spec_sha256"], "bundle hard spec SHA"
    )
    expected["hard_constraint_contract_sha256"] = _hex(
        expected["hard_constraint_contract_sha256"],
        "bundle hard constraint contract SHA",
    )
    expected["temperature_contract_sha256"] = _hex(
        expected["temperature_contract_sha256"],
        "bundle temperature contract SHA",
    )
    identities = []
    for record in records:
        result = record.get("_result")
        if not isinstance(result, dict):
            continue
        hard_spec = result.get("hard_spec")
        hard_spec_sha = result.get("stage_spec_sha256")
        constraint_names = result.get("constraint_names")
        temperatures = result.get("temperature_targets")
        constraint_version = result.get("constraint_version")
        if (
            not isinstance(hard_spec, dict)
            or not hard_spec
            or hard_spec_sha != canonical_sha256(hard_spec)
            or not isinstance(constraint_names, list)
            or not constraint_names
            or any(not isinstance(name, str) or not name for name in constraint_names)
            or len(constraint_names) != len(set(constraint_names))
            or temperatures != expected["temperature_targets"]
            or not isinstance(constraint_version, str)
            or not constraint_version
        ):
            raise RuntimeError("terminal result hard-constraint identity is invalid")
        identities.append(
            {
                "constraint_version": constraint_version,
                "hard_spec": copy.deepcopy(hard_spec),
                "hard_spec_sha256": _hex(hard_spec_sha, "hard spec SHA"),
                "hard_constraint_contract_sha256": _hex(
                    result.get("hard_constraint_contract_sha256"),
                    "hard constraint contract SHA",
                ),
                "temperature_contract_sha256": _hex(
                    result.get("temperature_contract_sha256"),
                    "temperature contract SHA",
                ),
                "constraint_names": list(constraint_names),
                "temperature_targets": list(temperatures),
            }
        )
    if any(item != expected for item in identities):
        raise RuntimeError("terminal results mix hard-constraint identities")
    return expected


def legacy_8010_compatibility(manifest: Mapping[str, Any]) -> dict[str, Any]:
    temperatures = list(manifest.get("temperature_targets") or [])
    return {
        "schema_version": COMPATIBILITY_SCHEMA,
        "monitor_generation": LEGACY_MONITOR_GENERATION,
        "compatible": False,
        "activation_allowed": False,
        "legacy_canonical_index_write_allowed": False,
        "reasons": [
            "index_schema_current7_not_mft-tier1-slurm-rolling-index-v1",
            "status_schema_current7_not_mft-tier1-slurm-rolling-status-v1",
            f"task_schema_{manifest.get('task_schema_version')}_not_v3",
            "result_schema_current7_not_mft-tier1-corrected-search-seed-v1",
            f"temperature_target_count_{len(temperatures)}_not_all11",
        ],
        "required_action": "deploy_new_immutable_8010_current7_reader",
    }


def _immutable_write(path: Path, payload: bytes) -> int:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable cache/cohort collision: {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale atomic write temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return 1


def _atomic_replace(path: Path, payload: bytes) -> int:
    if path.is_file() and path.read_bytes() == payload:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale atomic write temporary exists: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return 1


def cache_seed_records(
    runtime: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    apply: bool,
) -> tuple[list[dict[str, Any]], int]:
    runtime = runtime.resolve()
    cache = runtime / "current7-cache"
    writes = 0
    public_records: list[dict[str, Any]] = []
    for source in records:
        record = _public_observation(source)
        status_payloads = source.get("_status_payloads") or {}
        for status_object in record["status_objects"]:
            digest = str(status_object["sha256"])
            status_object["local_cache_path"] = str(
                cache / "objects" / "sha256" / digest[:2] / f"{digest}.json"
            )
        for digest, payload in status_payloads.items():
            if _sha_bytes(payload) != digest:
                raise RuntimeError("status cache object SHA mismatch")
            path = cache / "objects" / "sha256" / digest[:2] / f"{digest}.json"
            if apply:
                writes += _immutable_write(path, payload)
        result_payload = source.get("_result_bytes")
        if result_payload is not None:
            digest = str(record["result_object"]["sha256"])
            if _sha_bytes(result_payload) != digest:
                raise RuntimeError("result cache object SHA mismatch")
            path = cache / "objects" / "sha256" / digest[:2] / f"{digest}.json"
            record["result_object"]["local_cache_path"] = str(path)
            if apply:
                writes += _immutable_write(path, result_payload)
        artifact_payloads = source.get("_artifact_payloads") or {}
        for name, payload in sorted(artifact_payloads.items()):
            artifact = record["artifact_objects"][name]
            digest = str(artifact["sha256"])
            if _sha_bytes(payload) != digest:
                raise RuntimeError("search artifact cache object SHA mismatch")
            path = cache / "objects" / "sha256" / digest[:2] / f"{digest}.artifact"
            artifact["local_cache_path"] = str(path)
            if apply:
                writes += _immutable_write(path, payload)
        record["cache_objects_root"] = str(cache / "objects" / "sha256")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        record["record_sha256"] = canonical_sha256(unsigned)
        ref = cache / "refs" / str(record["bundle_id"]) / f"seed-{record['seed']}.json"
        if ref.is_file():
            prior = _strict_json(ref.read_bytes(), str(ref))
            prior_result = (prior.get("result_object") or {}).get("sha256")
            next_result = (record.get("result_object") or {}).get("sha256")
            if prior_result and next_result and prior_result != next_result:
                raise RuntimeError("cached bundle+seed result identity diverged")
            if prior.get("record_sha256") != canonical_sha256(
                {key: value for key, value in prior.items() if key != "record_sha256"}
            ):
                raise RuntimeError("cached bundle+seed record SHA mismatch")
            merged_ids = sorted(
                {int(value) for value in [*(prior.get("task_ids") or []), *record["task_ids"]]}
            )
            if prior_result and not next_result:
                record = prior
            else:
                record["task_ids"] = merged_ids
                unsigned = {
                    key: value for key, value in record.items() if key != "record_sha256"
                }
                record["record_sha256"] = canonical_sha256(unsigned)
        if apply:
            writes += _atomic_replace(ref, _json_bytes(record))
        public_records.append(record)
    public_records.sort(key=lambda item: (str(item["bundle_id"]), int(item["seed"])))
    return public_records, writes


def _latest_inventory_time(inventory: Sequence[Mapping[str, Any]]) -> str | None:
    values = [
        str(value)
        for item in inventory
        for value in (
            item.get("finished_at"),
            item.get("started_at"),
            item.get("created_at"),
        )
        if value
    ]
    return max(values, default=None)


def build_current7_snapshot(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    refusals: Sequence[Mapping[str, Any]],
    runtime: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = runtime.resolve()
    aggregate = aggregate_candidates(records)
    constraint_identity = authenticated_constraint_identity(records, manifest)
    aggregate.update(copy.deepcopy(constraint_identity))
    compatibility = legacy_8010_compatibility(manifest)
    public_records = [_public_observation(record) for record in records]
    compact_inventory = [
        {key: copy.deepcopy(value) for key, value in item.items() if key != "payload"}
        for item in inventory
    ]
    snapshot_identity = {
        "schema_version": SNAPSHOT_IDENTITY_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "task_schema_version": manifest["task_schema_version"],
        "result_schema_version": manifest["search_execution"][
            "result_schema_version"
        ],
        "inventory_sha256": canonical_sha256(compact_inventory),
        "terminal_record_sha256": sorted(
            str(record["record_sha256"]) for record in public_records
        ),
        "refusals_sha256": canonical_sha256(list(refusals)),
        "aggregate_sha256": canonical_sha256(aggregate),
    }
    snapshot_sha = canonical_sha256(snapshot_identity)
    cohort_id = f"current7-{str(plan['bundle_id'])[-12:]}-{snapshot_sha[:16]}"
    cohort_dir = runtime / "cohorts" / cohort_id
    status_path = cohort_dir / "status.json"
    compatibility_path = cohort_dir / "compatibility.json"
    status = {
        "schema_version": COHORT_STATUS_SCHEMA,
        "cohort_id": cohort_id,
        "snapshot_identity": snapshot_identity,
        "snapshot_sha256": snapshot_sha,
        "updated_at": _latest_inventory_time(inventory),
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "bundle_contract_sha256": manifest["contract_sha256"],
        "task_schema_version": manifest["task_schema_version"],
        "seed_status_schema_version": manifest["status_schema_version"],
        "result_schema_version": manifest["search_execution"][
            "result_schema_version"
        ],
        **copy.deepcopy(constraint_identity),
        "scheduler_task_count": len(inventory),
        "state_counts": dict(
            sorted(Counter(str(item["status"]) for item in inventory).items())
        ),
        "latest_tasks": compact_inventory,
        "authenticated_terminal_seed_count": len(public_records),
        "refused_terminal_count": len(refusals),
        "refusals": list(refusals),
        "terminal_results": public_records,
        "aggregate": aggregate,
        "legacy_8010_compatibility": compatibility,
        "healthy": not refusals,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    status_bytes = _json_bytes(status)
    compatibility_bytes = _json_bytes(compatibility)
    index = {
        "schema_version": CURRENT7_INDEX_SCHEMA,
        "active_cohort_id": cohort_id,
        "snapshot_sha256": snapshot_sha,
        "updated_at": status["updated_at"],
        "path_containment_root": str(runtime),
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        **copy.deepcopy(constraint_identity),
        "status": {
            "path": str(status_path),
            "schema_version": COHORT_STATUS_SCHEMA,
            "sha256": _sha_bytes(status_bytes),
        },
        "compatibility": {
            "path": str(compatibility_path),
            "schema_version": COMPATIBILITY_SCHEMA,
            "sha256": _sha_bytes(compatibility_bytes),
            "legacy_84ff69f_compatible": False,
        },
        "legacy_canonical_index_path": str(runtime / "canonical" / "index.json"),
        "legacy_canonical_index_touched": False,
        "activation_allowed": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return status, compatibility, index


def publish_current7_snapshot(
    runtime: Path,
    status: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    runtime = runtime.resolve()
    cohort_id = str(status["cohort_id"])
    final = runtime / "cohorts" / cohort_id
    current7_index = runtime / "canonical" / "current7-index.json"
    legacy_index = runtime / "canonical" / "index.json"
    legacy_before = legacy_index.read_bytes() if legacy_index.is_file() else None
    status_ref = index.get("status") or {}
    compatibility_ref = index.get("compatibility") or {}
    constraint_identity_fields = (
        "constraint_version",
        "hard_spec",
        "hard_spec_sha256",
        "hard_constraint_contract_sha256",
        "temperature_contract_sha256",
        "constraint_names",
        "temperature_targets",
    )
    if (
        Path(str(index.get("path_containment_root") or "")).resolve() != runtime
        or Path(str(status_ref.get("path") or "")).resolve()
        != (final / "status.json").resolve()
        or status_ref.get("schema_version") != COHORT_STATUS_SCHEMA
        or status_ref.get("sha256") != _sha_bytes(_json_bytes(status))
        or Path(str(compatibility_ref.get("path") or "")).resolve()
        != (final / "compatibility.json").resolve()
        or compatibility_ref.get("schema_version") != COMPATIBILITY_SCHEMA
        or compatibility_ref.get("sha256")
        != _sha_bytes(_json_bytes(compatibility))
        or index.get("active_cohort_id") != cohort_id
        or index.get("snapshot_sha256") != status.get("snapshot_sha256")
        or any(index.get(field) != status.get(field) for field in constraint_identity_fields)
        or index.get("legacy_canonical_index_touched") is not False
        or index.get("activation_allowed") is not False
    ):
        raise RuntimeError("current7 secondary index containment/SHA seal mismatch")
    if not apply:
        return {
            "apply": False,
            "write_count": 0,
            "cohort_path": str(final),
            "current7_index_path": str(current7_index),
            "legacy_index_touched": False,
        }
    status_bytes = _json_bytes(status)
    compatibility_bytes = _json_bytes(compatibility)
    writes = 0
    if final.exists():
        if not final.is_dir():
            raise RuntimeError("current7 cohort path is not a directory")
        expected = {
            "status.json": status_bytes,
            "compatibility.json": compatibility_bytes,
        }
        for name, payload in expected.items():
            path = final / name
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError("immutable current7 cohort collision")
    else:
        incoming = runtime / "cohorts" / f".incoming-{cohort_id}-{os.getpid()}"
        if incoming.exists():
            raise RuntimeError("stale current7 cohort incoming directory exists")
        incoming.mkdir(parents=True)
        (incoming / "status.json").write_bytes(status_bytes)
        (incoming / "compatibility.json").write_bytes(compatibility_bytes)
        os.replace(incoming, final)
        writes += 2
    writes += _atomic_replace(current7_index, _json_bytes(index))
    legacy_after = legacy_index.read_bytes() if legacy_index.is_file() else None
    if legacy_before != legacy_after:
        raise RuntimeError("legacy canonical index changed during current7 publish")
    return {
        "apply": True,
        "write_count": writes,
        "cohort_path": str(final),
        "current7_index_path": str(current7_index),
        "legacy_index_touched": False,
    }


def harvest_snapshot(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    scheduler: SchedulerReader,
    remote: RemoteReader,
    runtime: Path,
    apply: bool = False,
    payload_builder: Callable[..., Mapping[str, Any]] = build_task_payload,
    result_validator: Callable[..., None] = validate_result,
) -> dict[str, Any]:
    inventory = inventory_tasks(
        plan,
        manifest,
        controller_state,
        scheduler,
        payload_builder=payload_builder,
    )
    observations = []
    refusals = []
    for item in inventory:
        if item["status"] not in TERMINAL_SCHEDULER_STATES:
            continue
        try:
            observations.append(
                harvest_terminal_task(
                    item,
                    plan=plan,
                    manifest=manifest,
                    remote=remote,
                    result_validator=result_validator,
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            refusals.append(
                {
                    "task_id": int(item["task_id"]),
                    "bundle_id": plan["bundle_id"],
                    "seed": int(item["seed"]),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    deduplicated = deduplicate_observations(observations)
    # Validate the complete projected cache/status/aggregate before allowing the
    # first local write.  Content-addressed objects may remain after a later OS
    # interruption, but malformed candidate evidence can never create them.
    cached, _projected_writes = cache_seed_records(
        runtime, deduplicated, apply=False
    )
    # Preserve parsed result payloads for this process's aggregate while only
    # publishing/cache-sealing the public record fields.
    by_key = {
        (str(item["bundle_id"]), int(item["seed"])): item
        for item in deduplicated
    }
    aggregate_records = []
    for item in cached:
        source = by_key.get((str(item["bundle_id"]), int(item["seed"])))
        aggregate_records.append(
            {
                **item,
                "_result": (source or {}).get("_result"),
                "_candidates": (source or {}).get("_candidates"),
            }
        )
    status, compatibility, index = build_current7_snapshot(
        plan=plan,
        manifest=manifest,
        inventory=inventory,
        records=aggregate_records,
        refusals=refusals,
        runtime=runtime,
    )
    cache_writes = 0
    if apply:
        applied_cached, cache_writes = cache_seed_records(
            runtime, deduplicated, apply=True
        )
        if applied_cached != cached:
            raise RuntimeError("current7 cache projection changed before apply")
    publication = publish_current7_snapshot(
        runtime, status, compatibility, index, apply=apply
    )
    return {
        "schema_version": HARVEST_RESULT_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "apply": bool(apply),
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
        "local_write_count": cache_writes + int(publication["write_count"]),
        "inventory_count": len(inventory),
        "nonterminal_skipped_count": sum(
            item["status"] not in TERMINAL_SCHEDULER_STATES for item in inventory
        ),
        "terminal_observation_count": len(observations),
        "authenticated_seed_count": len(cached),
        "refused_terminal_count": len(refusals),
        "refusals": refusals,
        "cohort_id": status["cohort_id"],
        "aggregate": status["aggregate"],
        "compatibility": compatibility,
        "publication": publication,
    }


def run_from_paths(
    plan_path: Path,
    controller_state_path: Path,
    *,
    scheduler: SchedulerReader,
    remote: RemoteReader,
    runtime: Path = DEFAULT_RUNTIME,
    apply: bool = False,
) -> dict[str, Any]:
    plan, manifest, _source_map = load_bundle_plan(plan_path)
    state = read_json(controller_state_path.resolve(strict=True))
    return harvest_snapshot(
        plan=plan,
        manifest=manifest,
        controller_state=state,
        scheduler=scheduler,
        remote=remote,
        runtime=runtime,
        apply=apply,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--controller-state", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    scheduler = ReadOnlySchedulerApi(args.scheduler_url)
    remote = AccountSftpReader(args.accounts, args.scheduler_source)
    try:
        result = run_from_paths(
            args.plan,
            args.controller_state,
            scheduler=scheduler,
            remote=remote,
            runtime=args.runtime,
            apply=args.apply,
        )
    finally:
        remote.close()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
