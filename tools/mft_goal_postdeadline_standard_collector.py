"""GET-only watcher and collector for custom post-deadline Standard tasks.

The custom post-deadline plans intentionally use schemas that the canonical
diagnostic collector does not accept.  This adapter preserves the reviewed
remote bundle contract while refusing all Scheduler mutations and all
scientific PASS/infeasible claims.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error, parse, request


PLAN_SCHEMA = "mft-goal-standard-postdeadline-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-standard-postdeadline-submission-v1"
POLL_SCHEMA = "mft-goal-postdeadline-standard-watch-poll-v1"
FAILURE_SCHEMA = "mft-goal-postdeadline-standard-failure-ledger-v1"
COLLECTION_SCHEMA = "mft-goal-postdeadline-standard-collection-v1"
COLLECTION_SEAL_SCHEMA = "mft-goal-postdeadline-standard-collection-seal-v1"
REMOTE_RECEIPT_SCHEMA = "mft-goal-diagnostic-remote-artifact-bundle-receipt-v1"
RESULTS_MANIFEST_SCHEMA = "mft-goal-diagnostic-aedtresults-manifest-v1"
MARKER_SCHEMA = "slurm-scheduler-prune-protection-v1"
TRANSPORT_SCHEMA = "mft-goal-fea-base64-chunks-v1"
SCHEDULER_PROJECT = "MFT_1MW_2026v1"
TERMINAL_SUCCESS = {("completed", "succeeded")}
TERMINAL_FAILURE_STATES = {"failed", "cancelled"}
ACTIVE_STATES = {"queued", "attaching", "running"}
RAW_CHUNK_BYTES = 768_000
MAX_ENCODED_CHUNK_BYTES = 1_024_000
MAX_METADATA_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_STDOUT_BYTES = 1024 * 1024
MAX_AEDT_BYTES = 64 * 1024 * 1024 * 1024
MAX_RESULTS_BYTES = 512 * 1024 * 1024 * 1024
MAX_RESULTS_FILES = 250_000
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
FIXED_BOUNDARY = {
    "core_plate_on": 1,
    "core_plate_pad_t_mm": 2.0,
    "fan_config": "dual",
    "fan_velocity_m_s": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "wcp_on": 1,
    "wcp_pad_t_mm": 2.0,
}
CANDIDATE_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity": 1.5,
    "k_ins": 0.2,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}
RESULT_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity": 1.5,
    "k_ins": 0.2,
    "core_plate_on": 1,
    "core_plate_pad_t": 2.0,
    "wcp_on": 1,
    "wcp_pad_t": 2.0,
}
CLASSIFICATION = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "production_eligible": False,
    "original_deadline_missed": True,
}
REMOTE_RECEIPT_FIELDS = {
    "schema_version",
    "stage",
    "dedupe_key",
    "parameter_digest",
    "solver_revision",
    "library_revision",
    "profile_sha256",
    "artifact_path",
    "marker_path",
    "retention_required",
    "prune_protection_required",
    "scheduler_cleanup_exclusion_required",
    "artifact_sha256",
    "artifact_size_bytes",
    "marker_sha256",
    "marker_contract_sha256",
    "transport_schema_version",
    "transport_encoding",
    "transport_chunk_directory",
    "transport_raw_chunk_bytes",
    "transport_max_encoded_chunk_bytes",
    "transport_chunk_count",
    "source_project_filename",
    "source_project_name",
    "results_path",
    "results_manifest_path",
    "results_manifest_schema_version",
    "results_manifest_sha256",
    "results_tree_sha256",
    "results_file_count",
    "results_size_bytes",
    "source_results_directory_name",
}


class CollectionError(RuntimeError):
    """A source identity, Scheduler GET, or retained artifact drifted."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise CollectionError("object is already sealed")
    result["payload_sha256"] = payload_sha256(result)
    return result


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    label = (
        resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"{label} must be a JSON object")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    body = copy.deepcopy(dict(value))
    expected = body.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(expected, str)
        or not HEX64.fullmatch(expected)
        or payload_sha256(body) != expected
    ):
        raise CollectionError(f"{label} seal drifted")


def _write_atomic(path: Path, data: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() == data:
            return
        raise CollectionError(f"immutable output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    _write_atomic(path, canonical_bytes(value) + b"\n", replace=replace)


def _safe_relative(value: Any, label: str) -> str:
    text = str(value or "")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in text
        or pure.as_posix() != text
    ):
        raise CollectionError(f"{label} path is unsafe")
    return text


def _finite_equal(actual: Any, expected: float | int) -> bool:
    try:
        return math.isfinite(float(actual)) and float(actual) == float(expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _verify_file_record(record: Any, label: str) -> Path:
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or not HEX64.fullmatch(str(record["sha256"]))
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
    ):
        raise CollectionError(f"{label} file record is malformed")
    path = Path(record["path"]).resolve(strict=True)
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise CollectionError(f"{label} file bytes drifted")
    return path


def _find_source(plan: Mapping[str, Any], filename: str, label: str) -> Path:
    records = plan.get("source_artifacts")
    record = records.get(filename) if isinstance(records, Mapping) else None
    return _verify_file_record(record, label)


def _classification_valid(value: Mapping[str, Any]) -> bool:
    return all(value.get(key) is expected for key, expected in CLASSIFICATION.items())


def load_contract(
    *,
    plan_path: Path,
    submission_path: Path,
    expected_task_id: int,
    expected_task_name: str,
    expected_dedupe_key: str,
    expected_candidate_sha256: str,
    expected_receipt_sha256: str,
    scheduler_url: str,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    submission_path = submission_path.resolve(strict=True)
    if (
        isinstance(expected_task_id, bool)
        or not isinstance(expected_task_id, int)
        or expected_task_id <= 0
        or not expected_task_name
        or not expected_dedupe_key
        or not HEX64.fullmatch(expected_candidate_sha256)
        or not HEX64.fullmatch(expected_receipt_sha256)
        or sha256_file(submission_path) != expected_receipt_sha256
    ):
        raise CollectionError("expected watcher identity is malformed or drifted")
    plan = _read_json(plan_path, "post-deadline plan")
    submission = _read_json(submission_path, "post-deadline submission")
    _validate_seal(plan, PLAN_SCHEMA, "post-deadline plan")
    _validate_seal(submission, SUBMISSION_SCHEMA, "post-deadline submission")
    if not _classification_valid(plan) or not _classification_valid(submission):
        raise CollectionError("post-deadline classification drifted")
    if (
        plan.get("fixed_physics_unchanged") is not True
        or submission.get("fixed_physics_unchanged") is not True
        or plan.get("candidate_physics_sha256") != expected_candidate_sha256
        or submission.get("candidate_physics_sha256") != expected_candidate_sha256
        or plan.get("task_name") != expected_task_name
        or submission.get("task_name") != expected_task_name
        or plan.get("dedupe_key") != expected_dedupe_key
        or submission.get("dedupe_key") != expected_dedupe_key
        or submission.get("task_id") != expected_task_id
        or str(plan.get("scheduler_url") or "").rstrip("/") != scheduler_url.rstrip("/")
    ):
        raise CollectionError("post-deadline plan/submission identity drifted")
    plan_record = submission.get("plan")
    if (
        not isinstance(plan_record, Mapping)
        or _verify_file_record(plan_record, "submission plan") != plan_path
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
    ):
        raise CollectionError("submission does not bind the exact plan")
    scheduler_payload = plan.get("scheduler_payload")
    resources = plan.get("resources")
    envelope = plan.get("timeout_envelope")
    if (
        not isinstance(scheduler_payload, Mapping)
        or payload_sha256(scheduler_payload) != plan.get("scheduler_payload_sha256")
        or submission.get("scheduler_payload_sha256")
        != plan.get("scheduler_payload_sha256")
        or not isinstance(resources, Mapping)
        or resources.get("cpus") != 8
        or resources.get("memory_mb") != 98_304
        or resources.get("same_node_as_task_id") != 0
        or resources.get("dependency_task_id") != 0
        or not isinstance(envelope, Mapping)
        or envelope.get("solver_seconds") != 43_200
        or envelope.get("kill_grace_seconds") != 300
        or envelope.get("retention_seconds") != 1_800
        or envelope.get("scheduler_timeout_seconds") != 45_300
        or scheduler_payload.get("name") != expected_task_name
        or scheduler_payload.get("dedupe_key") != expected_dedupe_key
        or scheduler_payload.get("project") != SCHEDULER_PROJECT
        or scheduler_payload.get("cpus") != 8
        or scheduler_payload.get("memory_mb") != 98_304
        or scheduler_payload.get("timeout_seconds") != 45_300
        or scheduler_payload.get("node_name_policy") != "strict"
        or scheduler_payload.get("same_node_as_task_id", 0) != 0
        or "same_node_as_task_id" in scheduler_payload
        or scheduler_payload.get("aedt_backend") != "standalone"
    ):
        raise CollectionError("sealed Scheduler payload/resources drifted")
    command = str(scheduler_payload.get("command") or "")
    if (
        command.count(
            "timeout --signal=TERM --kill-after=300s 43200s "
            "python run_simulation_260706.py"
        )
        != 1
        or "sleep $((RANDOM % 300))" in command
    ):
        raise CollectionError("sealed solver/retention command drifted")
    preflight = plan.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("fixed_boundary") != FIXED_BOUNDARY
        or preflight.get("candidate_boundary_projection") != CANDIDATE_BOUNDARY
        or preflight.get("candidate_physics_sha256") != expected_candidate_sha256
        or preflight.get("candidate_reauthenticated") is not True
        or preflight.get("fixed_physics_unchanged") is not True
        or preflight.get("revision_reauthenticated") is not True
    ):
        raise CollectionError("candidate/fixed-boundary preflight drifted")
    source_records = plan.get("source_artifacts")
    if not isinstance(source_records, Mapping) or not source_records:
        raise CollectionError("source artifact records are absent")
    for name, record in source_records.items():
        _verify_file_record(record, f"source artifact {name}")
    params_path = _find_source(plan, "fea_params.json", "source FEA params")
    selected_path = _find_source(plan, "selected_candidate.json", "selected candidate")
    profile_path = _verify_file_record(
        plan.get("execution_profile"), "execution profile"
    )
    params = _read_json(params_path, "source FEA params")
    selected = _read_json(selected_path, "selected candidate")
    profile = _read_json(profile_path, "execution profile")
    _validate_seal(
        selected,
        str(selected.get("schema_version") or ""),
        "selected candidate",
    )
    if (
        selected.get("selected_row", {}).get("candidate_physics_sha")
        != expected_candidate_sha256
        or profile.get("fixed_boundary_contract") != FIXED_BOUNDARY
        or profile.get("param_overrides", {}).get("thermal_symmetry") != "eighth"
        or profile.get("param_overrides", {}).get("full_model") != 0
        or profile.get("timeout_seconds") != 43_200
        or payload_sha256(profile) != plan.get("execution_profile_canonical_sha256")
    ):
        raise CollectionError("selected candidate/execution profile drifted")
    merged = dict(params)
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, Mapping):
        raise CollectionError("execution profile overrides are malformed")
    merged.update(overrides)
    parameter_json = json.dumps(
        merged,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    parameter_digest = sha256_bytes(parameter_json.encode("utf-8"))[:16]
    solver_revision = str(plan.get("solver_revision") or "")
    library_revision = str(plan.get("library_revision") or "")
    if not HEX40.fullmatch(solver_revision) or not HEX40.fullmatch(library_revision):
        raise CollectionError("execution revisions are malformed")
    recomputed_dedupe = (
        f"mft-al:{expected_task_name}:{solver_revision}:"
        f"{library_revision}:{parameter_digest}"
    )
    if recomputed_dedupe != expected_dedupe_key:
        raise CollectionError("candidate parameter/dedupe identity drifted")
    task_identity = sha256_bytes(expected_dedupe_key.encode("utf-8"))[:16]
    retained_root = f"goal-fea-retained/{task_identity}"
    marker_contract = {
        "schema": MARKER_SCHEMA,
        "preserve": True,
        "reason": (
            "Retain diagnostic Standard AEDT and AEDT results until "
            "authenticated diagnostic collection; stage=standard; "
            f"dedupe_key={expected_dedupe_key}"
        ),
        "owner": SCHEDULER_PROJECT,
    }
    return {
        "plan": plan,
        "submission": submission,
        "plan_path": plan_path,
        "submission_path": submission_path,
        "plan_file_sha256": sha256_file(plan_path),
        "submission_file_sha256": expected_receipt_sha256,
        "task_id": expected_task_id,
        "task_name": expected_task_name,
        "dedupe_key": expected_dedupe_key,
        "candidate_physics_sha256": expected_candidate_sha256,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "profile_sha256": payload_sha256(profile),
        "parameter_digest": parameter_digest,
        "scheduler_url": scheduler_url.rstrip("/"),
        "node_name": str(resources.get("node_name") or ""),
        "retained": {
            "root": retained_root,
            "artifact_path": f"{retained_root}/symmetric.aedt",
            "receipt_path": (f"{retained_root}/symmetric.aedt.receipt.json"),
            "marker_path": f"{retained_root}/.slurm-scheduler-preserve.json",
            "chunk_directory": (f"{retained_root}/symmetric.aedt.chunks"),
            "results_path": f"{retained_root}/symmetric.aedtresults",
            "results_manifest_path": (
                f"{retained_root}/symmetric.aedtresults.manifest.json"
            ),
            "marker_contract": marker_contract,
            "marker_contract_sha256": payload_sha256(marker_contract),
        },
    }


def http_get(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 120.0,
    opener: Callable[..., Any] = request.urlopen,
) -> bytes:
    if not url.startswith(("http://", "https://")) or max_bytes <= 0:
        raise CollectionError("unsafe GET request")
    req = request.Request(url, headers={"Accept": "*/*"}, method="GET")
    try:
        with opener(req, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
    except (OSError, error.HTTPError, error.URLError) as exc:
        raise CollectionError(f"Scheduler GET failed: {url}") from exc
    if len(raw) > max_bytes:
        raise CollectionError(f"Scheduler GET exceeded byte bound: {url}")
    return raw


def get_task(
    contract: Mapping[str, Any],
    *,
    getter: Callable[..., bytes] = http_get,
) -> dict[str, Any]:
    raw = getter(
        f"{contract['scheduler_url']}/api/tasks/{contract['task_id']}",
        max_bytes=1024 * 1024,
        timeout=30.0,
    )
    try:
        task = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError("Scheduler task GET is invalid JSON") from exc
    if not isinstance(task, dict):
        raise CollectionError("Scheduler task GET is not an object")
    expected = {
        "task_id": contract["task_id"],
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": SCHEDULER_PROJECT,
        "requested_node_name": contract["node_name"],
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
    }
    for key, value in expected.items():
        actual = (
            task.get("task_id", task.get("id")) if key == "task_id" else task.get(key)
        )
        if actual != value:
            raise CollectionError(f"Scheduler task identity drifted: {key}")
    return task


def _remote_url(
    contract: Mapping[str, Any],
    relative_path: str,
    max_bytes: int,
) -> str:
    safe = _safe_relative(relative_path, "remote file")
    query = parse.urlencode(
        {"path": safe, "base": "remote_cwd", "max_bytes": max_bytes}
    )
    return (
        f"{contract['scheduler_url']}/api/tasks/{contract['task_id']}"
        f"/remote-file?{query}"
    )


def remote_get(
    contract: Mapping[str, Any],
    relative_path: str,
    max_bytes: int,
    *,
    getter: Callable[..., bytes] = http_get,
) -> bytes:
    return getter(
        _remote_url(contract, relative_path, max_bytes),
        max_bytes=max_bytes,
        timeout=120.0,
    )


def get_result(
    contract: Mapping[str, Any],
    *,
    getter: Callable[..., bytes] = http_get,
) -> tuple[dict[str, Any], bytes]:
    query = parse.urlencode({"max_bytes": MAX_STDOUT_BYTES})
    stdout = getter(
        f"{contract['scheduler_url']}/api/tasks/{contract['task_id']}/stdout?{query}",
        max_bytes=MAX_STDOUT_BYTES,
        timeout=60.0,
    )
    try:
        text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise CollectionError("Scheduler stdout is not UTF-8") from exc
    library = None
    result = None
    for line in reversed(text.splitlines()):
        if library is None and line.startswith("MFT_LIBRARY_GIT_HASH "):
            candidate = line.removeprefix("MFT_LIBRARY_GIT_HASH ").strip()
            if HEX40.fullmatch(candidate):
                library = candidate
        if result is None and line.startswith("RESULT_JSON "):
            try:
                candidate_result = json.loads(line.removeprefix("RESULT_JSON "))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate_result, dict):
                result = candidate_result
        if result is not None and library is not None:
            break
    if result is None or library != contract["library_revision"]:
        raise CollectionError("valid RESULT_JSON/library marker is absent")
    if (
        result.get("git_hash") != contract["solver_revision"]
        or result.get("pyaedt_library_git_hash") != contract["library_revision"]
        or not _finite_equal(result.get("full_model"), 0)
        or str(result.get("thermal_symmetry") or "").lower() != "eighth"
    ):
        raise CollectionError("RESULT_JSON execution identity drifted")
    for key, expected in RESULT_BOUNDARY.items():
        actual = result.get(key)
        if isinstance(expected, str):
            passed = actual == expected
        else:
            passed = _finite_equal(actual, expected)
        if not passed:
            raise CollectionError(f"RESULT_JSON fixed physics drifted: {key}")
    return result, stdout


def _validate_remote_receipt(
    value: Any, contract: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    retained = contract["retained"]
    if not isinstance(value, dict) or set(value) != REMOTE_RECEIPT_FIELDS:
        raise CollectionError("remote bundle receipt fields drifted")
    integer_fields = {
        "artifact_size_bytes": (1, MAX_AEDT_BYTES),
        "transport_chunk_count": (1, math.ceil(MAX_AEDT_BYTES / RAW_CHUNK_BYTES)),
        "results_file_count": (1, MAX_RESULTS_FILES),
        "results_size_bytes": (0, MAX_RESULTS_BYTES),
    }
    for key, (minimum, maximum) in integer_fields.items():
        raw = value.get(key)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int)
            or not minimum <= raw <= maximum
        ):
            raise CollectionError(f"remote receipt integer drifted: {key}")
    sha_fields = {
        "artifact_sha256",
        "marker_sha256",
        "marker_contract_sha256",
        "results_manifest_sha256",
        "results_tree_sha256",
    }
    if any(
        not isinstance(value.get(key), str) or not HEX64.fullmatch(str(value[key]))
        for key in sha_fields
    ):
        raise CollectionError("remote receipt SHA fields drifted")
    expected = {
        "schema_version": REMOTE_RECEIPT_SCHEMA,
        "stage": "standard",
        "dedupe_key": contract["dedupe_key"],
        "parameter_digest": contract["parameter_digest"],
        "solver_revision": contract["solver_revision"],
        "library_revision": contract["library_revision"],
        "profile_sha256": contract["profile_sha256"],
        "artifact_path": retained["artifact_path"],
        "marker_path": retained["marker_path"],
        "marker_contract_sha256": retained["marker_contract_sha256"],
        "transport_schema_version": TRANSPORT_SCHEMA,
        "transport_encoding": "base64",
        "transport_chunk_directory": retained["chunk_directory"],
        "transport_raw_chunk_bytes": RAW_CHUNK_BYTES,
        "transport_max_encoded_chunk_bytes": MAX_ENCODED_CHUNK_BYTES,
        "results_path": retained["results_path"],
        "results_manifest_path": retained["results_manifest_path"],
        "results_manifest_schema_version": RESULTS_MANIFEST_SCHEMA,
        "retention_required": True,
        "prune_protection_required": True,
        "scheduler_cleanup_exclusion_required": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CollectionError(f"remote receipt identity drifted: {key}")
    if (
        value["transport_chunk_count"]
        != math.ceil(value["artifact_size_bytes"] / RAW_CHUNK_BYTES)
        or value.get("source_project_filename")
        != f"{value.get('source_project_name')}.aedt"
        or value.get("source_results_directory_name")
        != f"{value.get('source_project_name')}.aedtresults"
        or result.get("project_name") != value.get("source_project_name")
    ):
        raise CollectionError("remote receipt project/chunk identity drifted")
    return copy.deepcopy(value)


def _validate_marker(
    value: Any, raw: bytes, receipt: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionError("remote marker is not an object")
    marker = copy.deepcopy(value)
    created_at = marker.pop("created_at", None)
    try:
        parsed = datetime.fromisoformat(str(created_at or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise CollectionError("remote marker timestamp is invalid") from exc
    if (
        parsed.tzinfo is None
        or marker != contract["retained"]["marker_contract"]
        or sha256_bytes(raw) != receipt["marker_sha256"]
    ):
        raise CollectionError("remote marker identity drifted")
    return copy.deepcopy(value)


def _validate_manifest(
    value: Any,
    raw: bytes,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "source_project_name",
        "source_results_directory_name",
        "retained_results_directory_name",
        "file_count",
        "size_bytes",
        "tree_sha256",
        "files",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version") != RESULTS_MANIFEST_SCHEMA
        or value.get("source_project_name") != result.get("project_name")
        or value.get("source_results_directory_name")
        != f"{result.get('project_name')}.aedtresults"
        or value.get("retained_results_directory_name") != "symmetric.aedtresults"
        or not isinstance(value.get("files"), list)
        or sha256_bytes(raw) != receipt["results_manifest_sha256"]
    ):
        raise CollectionError("remote results manifest identity drifted")
    files = value["files"]
    paths: list[str] = []
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size_bytes"}:
            raise CollectionError("results manifest record is malformed")
        relative = _safe_relative(item.get("path"), "results manifest file")
        size = item.get("size_bytes")
        if (
            not isinstance(item.get("sha256"), str)
            or not HEX64.fullmatch(item["sha256"])
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise CollectionError("results manifest record identity drifted")
        paths.append(relative)
        total += size
    tree = payload_sha256(files)
    if (
        not 1 <= len(files) <= MAX_RESULTS_FILES
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or value.get("file_count") != len(files)
        or receipt["results_file_count"] != len(files)
        or value.get("size_bytes") != total
        or receipt["results_size_bytes"] != total
        or value.get("tree_sha256") != tree
        or receipt["results_tree_sha256"] != tree
    ):
        raise CollectionError("results manifest content inventory drifted")
    return copy.deepcopy(value)


def fetch_remote_metadata(
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    getter: Callable[..., bytes] = http_get,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes, bytes]:
    retained = contract["retained"]
    receipt_raw = remote_get(
        contract,
        retained["receipt_path"],
        MAX_METADATA_BYTES,
        getter=getter,
    )
    try:
        receipt_value = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError("remote receipt is invalid JSON") from exc
    receipt = _validate_remote_receipt(receipt_value, contract, result)
    marker_raw = remote_get(
        contract,
        retained["marker_path"],
        MAX_METADATA_BYTES,
        getter=getter,
    )
    try:
        marker_value = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError("remote marker is invalid JSON") from exc
    marker = _validate_marker(marker_value, marker_raw, receipt, contract)
    manifest_raw = remote_get(
        contract,
        retained["results_manifest_path"],
        MAX_MANIFEST_BYTES,
        getter=getter,
    )
    try:
        manifest_value = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionError("remote results manifest is invalid JSON") from exc
    manifest = _validate_manifest(manifest_value, manifest_raw, receipt, result)
    return (
        receipt,
        marker,
        manifest,
        receipt_raw,
        marker_raw,
        manifest_raw,
    )


def _fetch_aedt_and_chunks(
    *,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    staging: Path,
    getter: Callable[..., bytes] = http_get,
) -> tuple[Path, list[dict[str, Any]]]:
    chunks = staging / "symmetric.aedt.chunks"
    chunks.mkdir()
    artifact = staging / "symmetric.aedt"
    temporary = staging / ".symmetric.aedt.tmp"
    digest = hashlib.sha256()
    written = 0
    inventory = []
    try:
        with temporary.open("xb") as artifact_handle:
            for index in range(receipt["transport_chunk_count"]):
                relative = f"{receipt['transport_chunk_directory']}/{index:08d}.b64"
                encoded = remote_get(
                    contract,
                    relative,
                    MAX_ENCODED_CHUNK_BYTES,
                    getter=getter,
                )
                expected_raw = min(
                    RAW_CHUNK_BYTES,
                    receipt["artifact_size_bytes"] - written,
                )
                expected_encoded = 4 * math.ceil(expected_raw / 3)
                if len(encoded) != expected_encoded:
                    raise CollectionError(f"AEDT chunk size drifted: {index}")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise CollectionError(
                        f"AEDT chunk base64 drifted: {index}"
                    ) from exc
                if len(raw) != expected_raw:
                    raise CollectionError(f"AEDT raw chunk drifted: {index}")
                chunk_path = chunks / f"{index:08d}.b64"
                chunk_path.write_bytes(encoded)
                artifact_handle.write(raw)
                digest.update(raw)
                written += len(raw)
                inventory.append(file_record(chunk_path, relative_to=staging))
            artifact_handle.flush()
            os.fsync(artifact_handle.fileno())
        if (
            written != receipt["artifact_size_bytes"]
            or digest.hexdigest() != receipt["artifact_sha256"]
        ):
            raise CollectionError("reconstructed AEDT identity drifted")
        os.replace(temporary, artifact)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return artifact, inventory


def collect_success(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    output: Path,
    getter: Callable[..., bytes] = http_get,
) -> dict[str, Any]:
    destination = output.resolve()
    if destination.exists():
        existing = _read_json(
            destination / "collection_seal.json", "existing collection seal"
        )
        _validate_seal(existing, COLLECTION_SEAL_SCHEMA, "collection seal")
        if (
            existing.get("task_id") != contract["task_id"]
            or existing.get("source_submission_file_sha256")
            != contract["submission_file_sha256"]
        ):
            raise CollectionError("existing collection identity drifted")
        return {
            "event": "already_collected",
            "task_id": contract["task_id"],
            "output": str(destination),
            "collection_seal_payload_sha256": existing["payload_sha256"],
        }
    result, stdout = get_result(contract, getter=getter)
    (
        remote_receipt,
        marker,
        manifest,
        remote_receipt_raw,
        marker_raw,
        manifest_raw,
    ) = fetch_remote_metadata(contract, result, getter=getter)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        artifact, chunk_inventory = _fetch_aedt_and_chunks(
            contract=contract,
            receipt=remote_receipt,
            staging=staging,
            getter=getter,
        )
        (staging / "result.json").write_bytes(canonical_bytes(result) + b"\n")
        (staging / "scheduler_stdout.log").write_bytes(stdout)
        (staging / "remote_bundle_receipt.json").write_bytes(remote_receipt_raw)
        (staging / "prune_protection_marker.json").write_bytes(marker_raw)
        (staging / "symmetric.aedtresults.manifest.json").write_bytes(manifest_raw)
        (staging / "scheduler_terminal_task.json").write_bytes(
            canonical_bytes(task) + b"\n"
        )
        shutil.copy2(contract["plan_path"], staging / "source_plan.json")
        shutil.copy2(
            contract["submission_path"],
            staging / "source_submission_receipt.json",
        )
        artifact_record = file_record(artifact, relative_to=staging)
        manifest_record = file_record(
            staging / "symmetric.aedtresults.manifest.json",
            relative_to=staging,
        )
        result_record = file_record(staging / "result.json", relative_to=staging)
        source_records = {
            name: file_record(staging / name, relative_to=staging)
            for name in (
                "source_plan.json",
                "source_submission_receipt.json",
                "scheduler_terminal_task.json",
                "scheduler_stdout.log",
                "remote_bundle_receipt.json",
                "prune_protection_marker.json",
            )
        }
        receipt = seal(
            {
                "schema_version": COLLECTION_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **CLASSIFICATION,
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "task_id": contract["task_id"],
                "task_name": contract["task_name"],
                "dedupe_key": contract["dedupe_key"],
                "candidate_physics_sha256": contract["candidate_physics_sha256"],
                "fixed_physics_unchanged": True,
                "source_plan_file_sha256": contract["plan_file_sha256"],
                "source_plan_payload_sha256": contract["plan"]["payload_sha256"],
                "source_submission_file_sha256": contract["submission_file_sha256"],
                "source_submission_payload_sha256": contract["submission"][
                    "payload_sha256"
                ],
                "scheduler_terminal_task_sha256": payload_sha256(task),
                "result_sha256": payload_sha256(result),
                "retained_symmetric_aedt": artifact_record,
                "retained_symmetric_aedt_remote_sha256": remote_receipt[
                    "artifact_sha256"
                ],
                "retained_aedt_chunks": chunk_inventory,
                "retained_aedt_chunk_inventory_sha256": payload_sha256(chunk_inventory),
                "aedtresults_manifest": manifest_record,
                "aedtresults_manifest_payload_sha256": payload_sha256(manifest),
                "aedtresults_remote_tree_sha256": remote_receipt["results_tree_sha256"],
                "aedtresults_remote_file_count": remote_receipt["results_file_count"],
                "aedtresults_remote_size_bytes": remote_receipt["results_size_bytes"],
                "result_json": result_record,
                "remote_bundle_receipt_payload_sha256": payload_sha256(remote_receipt),
                "prune_protection_marker_payload_sha256": payload_sha256(marker),
                "source_files": source_records,
            }
        )
        receipt_path = staging / "collection_receipt.json"
        receipt_path.write_bytes(canonical_bytes(receipt) + b"\n")
        seal_value = seal(
            {
                "schema_version": COLLECTION_SEAL_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **CLASSIFICATION,
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "task_id": contract["task_id"],
                "task_name": contract["task_name"],
                "dedupe_key": contract["dedupe_key"],
                "candidate_physics_sha256": contract["candidate_physics_sha256"],
                "source_submission_file_sha256": contract["submission_file_sha256"],
                "collection_receipt": file_record(receipt_path, relative_to=staging),
                "collection_receipt_payload_sha256": receipt["payload_sha256"],
                "artifact_sha256": remote_receipt["artifact_sha256"],
                "aedtresults_manifest_sha256": remote_receipt[
                    "results_manifest_sha256"
                ],
                "aedtresults_tree_sha256": remote_receipt["results_tree_sha256"],
                "atomic_directory_collection": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
            }
        )
        (staging / "collection_seal.json").write_bytes(
            canonical_bytes(seal_value) + b"\n"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "event": "collected",
        "task_id": contract["task_id"],
        "output": str(destination),
        "artifact_sha256": remote_receipt["artifact_sha256"],
        "aedtresults_tree_sha256": remote_receipt["results_tree_sha256"],
        "collection_receipt_payload_sha256": receipt["payload_sha256"],
        "collection_seal_payload_sha256": seal_value["payload_sha256"],
    }


def write_failure_ledger(
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    path = output.resolve().parent / f"{output.name}.failure_ledger.json"
    ledger = seal(
        {
            "schema_version": FAILURE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **CLASSIFICATION,
            "scientific_pass_claimed": False,
            "scientific_infeasible_claimed": False,
            "scheduler_failure_only": True,
            "collection_performed": False,
            "scheduler_get_only_watch": True,
            "scheduler_mutation_performed": False,
            "task_id": contract["task_id"],
            "task_name": contract["task_name"],
            "dedupe_key": contract["dedupe_key"],
            "candidate_physics_sha256": contract["candidate_physics_sha256"],
            "fixed_physics_unchanged": True,
            "source_plan_file_sha256": contract["plan_file_sha256"],
            "source_plan_payload_sha256": contract["plan"]["payload_sha256"],
            "source_submission_file_sha256": contract["submission_file_sha256"],
            "source_submission_payload_sha256": contract["submission"][
                "payload_sha256"
            ],
            "scheduler_terminal_task": copy.deepcopy(dict(task)),
            "scheduler_terminal_task_sha256": payload_sha256(task),
            "failure_class": "scheduler_terminal_failure_or_timeout",
        }
    )
    _write_json(path, ledger)
    return {
        "event": "failure_ledger",
        "task_id": contract["task_id"],
        "path": str(path),
        "failure_ledger_payload_sha256": ledger["payload_sha256"],
    }


def _poll_record(
    contract: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    return seal(
        {
            "schema_version": POLL_SCHEMA,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "task_id": contract["task_id"],
            "task_name": contract["task_name"],
            "dedupe_key": contract["dedupe_key"],
            "candidate_physics_sha256": contract["candidate_physics_sha256"],
            "source_submission_file_sha256": contract["submission_file_sha256"],
            "state": task.get("state"),
            "status": task.get("status"),
            "actual_node_name": task.get("actual_node_name"),
            "allocation_id": task.get("allocation_id"),
            "slurm_job_id": str(task.get("slurm_job_id") or ""),
            "task_snapshot_sha256": payload_sha256(task),
        }
    )


def _record_poll(
    output: Path,
    poll: Mapping[str, Any],
    *,
    sequence: int,
) -> Path:
    watch_root = output.resolve().parent / f".{output.name}.watch"
    polls = watch_root / "polls"
    polls.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    immutable = polls / f"{stamp}-{sequence:06d}.json"
    _write_json(immutable, poll)
    _write_json(watch_root / "latest_poll.json", poll, replace=True)
    return immutable


def poll_once(
    *,
    contract: Mapping[str, Any],
    output: Path,
    sequence: int = 0,
    getter: Callable[..., bytes] = http_get,
) -> tuple[bool, dict[str, Any]]:
    task = get_task(contract, getter=getter)
    poll = _poll_record(contract, task)
    poll_path = _record_poll(output, poll, sequence=sequence)
    state = str(task.get("state") or "")
    status = str(task.get("status") or "")
    if (status, state) in TERMINAL_SUCCESS:
        result = collect_success(
            contract=contract,
            task=task,
            output=output,
            getter=getter,
        )
        result["poll_path"] = str(poll_path)
        return True, result
    if state in TERMINAL_FAILURE_STATES or status in TERMINAL_FAILURE_STATES:
        result = write_failure_ledger(contract=contract, task=task, output=output)
        result["poll_path"] = str(poll_path)
        return True, result
    if state not in ACTIVE_STATES and status not in ACTIVE_STATES:
        raise CollectionError(
            f"unsupported Scheduler task transition: status={status!r}, state={state!r}"
        )
    return False, {
        "event": "active",
        "task_id": contract["task_id"],
        "state": state,
        "status": status,
        "actual_node_name": task.get("actual_node_name"),
        "allocation_id": task.get("allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "poll_path": str(poll_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-task-id", required=True, type=int)
    parser.add_argument("--expected-task-name", required=True)
    parser.add_argument("--expected-dedupe-key", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 10 <= args.interval <= 3600:
        raise CollectionError("watch interval must be between 10 and 3600 seconds")
    contract = load_contract(
        plan_path=args.plan,
        submission_path=args.submission,
        expected_task_id=args.expected_task_id,
        expected_task_name=args.expected_task_name,
        expected_dedupe_key=args.expected_dedupe_key,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_receipt_sha256=args.expected_receipt_sha256,
        scheduler_url=args.scheduler_url,
    )
    sequence = 0
    while True:
        terminal, event = poll_once(
            contract=contract,
            output=args.output,
            sequence=sequence,
        )
        print(json.dumps(event, sort_keys=True), flush=True)
        if terminal or args.once:
            return 0
        sequence += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as exc:
        print(
            json.dumps(
                {
                    "event": "collector_error",
                    "error": str(exc),
                    "scheduler_mutation_performed": False,
                    "scientific_pass_claimed": False,
                    "scientific_infeasible_claimed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from exc
