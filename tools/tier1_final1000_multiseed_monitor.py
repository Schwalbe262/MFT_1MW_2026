"""Read-only 8010 adapter and sealed backend gate for multi-seed indexes.

The adapter consumes the compact v2 condition index produced by
``tier1_final1000_multiseed_status`` and hydrates the existing Current7 status
wire on demand.  It has no Scheduler, remote transport, AEDT, or FEA client.

The capability receipt is deliberately tied to exact adapter code, backend
code, condition-index snapshot, and test-evidence bytes.  A production driver
can call :func:`require_backend_capability` immediately before enabling a new
index; a missing receipt or any drift is a hard refusal.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_harvest import COHORT_STATUS_SCHEMA
    from tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        COMPACT_STATUS_SCHEMA,
        SHARD_MANIFEST_SCHEMA,
        json_bytes,
        now,
    )
    from tier1_final1000_multiseed_status import (
        DEFAULT_FRONTEND_PAGE_SIZE,
        MAX_HOT_WIRE_BYTES,
        adapt_status_for_frontend,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_harvest import COHORT_STATUS_SCHEMA
    from tools.tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        COMPACT_STATUS_SCHEMA,
        SHARD_MANIFEST_SCHEMA,
        json_bytes,
        now,
    )
    from tools.tier1_final1000_multiseed_status import (
        DEFAULT_FRONTEND_PAGE_SIZE,
        MAX_HOT_WIRE_BYTES,
        adapt_status_for_frontend,
    )


CAPABILITY_RECEIPT_SCHEMA = "mft-tier1-final1000-multiseed-backend-capability-v1"
CAPABILITY_CONTRACT_SCHEMA = (
    "mft-tier1-final1000-multiseed-monitor-capability-contract-v1"
)
FILE_SET_IDENTITY_SCHEMA = "mft-tier1-final1000-exact-file-set-v1"
INDEX_IDENTITY_SCHEMA = "mft-tier1-final1000-compact-index-identity-v1"
TEST_SEALS_SCHEMA = "mft-tier1-final1000-multiseed-monitor-test-seals-v1"
MAX_CAPABILITY_RECEIPT_BYTES = 4 * 1024 * 1024
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")

DEFAULT_ADAPTER_CODE_FILES = (
    "tools/tier1_final1000_multiseed_monitor.py",
    "tools/tier1_final1000_multiseed_status.py",
    "tools/tier1_final1000_multiseed_contract.py",
    "tools/tier1_corrected_current7_receipt.py",
)

CAPABILITY_CONTRACT = {
    "schema_version": CAPABILITY_CONTRACT_SCHEMA,
    "input_index_schema_version": COMPACT_INDEX_SCHEMA,
    "input_status_schema_version": COMPACT_STATUS_SCHEMA,
    "input_shard_manifest_schema_version": SHARD_MANIFEST_SCHEMA,
    "normalized_status_schema_version": COHORT_STATUS_SCHEMA,
    "scheduler_task_count_semantics": "physical_lane_count",
    "logical_seed_count_field": "logical_seed_count",
    "running_parent_visibility_rule": "batch_ordinal < sealed_child_count",
    "maximum_frontend_wire_bytes_exclusive": MAX_HOT_WIRE_BYTES,
    "virtual_scheduler_task_ids_created": False,
    "scheduler_mutation_performed": False,
    "remote_write_performed": False,
    "aedt_used": False,
    "fea_submission_performed": False,
}
CAPABILITY_CONTRACT_SHA256 = canonical_sha256(CAPABILITY_CONTRACT)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _read_bounded(path: Path, maximum_bytes: int, label: str) -> bytes:
    size = path.stat().st_size
    if size < 0 or size >= maximum_bytes:
        raise RuntimeError(f"{label} exceeds its sealed byte bound")
    payload = path.read_bytes()
    if len(payload) != size:
        raise RuntimeError(f"{label} changed during its bounded read")
    return payload


def _safe_relative(value: str) -> str:
    relative = PurePosixPath(str(value))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in str(value)
        or "\x00" in str(value)
        or str(relative) != str(value)
    ):
        raise RuntimeError("compact condition-index reference is unsafe")
    return str(relative)


def _safe_child(root: Path, relative: str) -> Path:
    relative = _safe_relative(relative)
    candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise RuntimeError("compact condition-index reference escaped its root")
    return candidate


def _validate_reference(
    reference: Mapping[str, Any], *, schema_version: str, label: str
) -> dict[str, Any]:
    required = {"path", "schema_version", "sha256", "size"}
    if not required.issubset(reference):
        raise RuntimeError(f"compact {label} reference fields are incomplete")
    size = reference.get("size")
    if (
        reference.get("schema_version") != schema_version
        or not isinstance(reference.get("path"), str)
        or _safe_relative(str(reference["path"])) != reference["path"]
        or not HEX_SHA256.fullmatch(str(reference.get("sha256") or ""))
        or isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size < MAX_HOT_WIRE_BYTES
    ):
        raise RuntimeError(f"compact {label} reference identity is invalid")
    return copy.deepcopy(dict(reference))


def _validate_index(index: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in index.items() if key != "index_sha256"}
    status_ref = index.get("status")
    manifest_ref = index.get("seed_result_shards")
    expected_adapter = {
        "normalized_status_schema_version": COHORT_STATUS_SCHEMA,
        "normalized_index_schema_version": (
            "mft-tier1-current7-slurm-rolling-index-v1"
        ),
        "terminal_results_field": "terminal_results",
        "on_demand": True,
    }
    expected_counts = {
        "scheduler_task_count_field": "physical_lane_count",
        "logical_seed_count_field": "logical_seed_count",
        "sealed_prefix_rule": "batch_ordinal < sealed_child_count",
    }
    if (
        index.get("schema_version") != COMPACT_INDEX_SCHEMA
        or index.get("index_sha256") != canonical_sha256(unsigned)
        or not isinstance(index.get("stage_id"), str)
        or not str(index["stage_id"]).strip()
        or not isinstance(status_ref, dict)
        or not isinstance(manifest_ref, dict)
        or index.get("frontend_adapter") != expected_adapter
        or index.get("count_contract") != expected_counts
        or index.get("virtual_scheduler_task_ids_created") is not False
    ):
        raise RuntimeError("compact condition index identity is invalid")
    _validate_reference(
        status_ref, schema_version=COMPACT_STATUS_SCHEMA, label="status"
    )
    _validate_reference(
        manifest_ref,
        schema_version=SHARD_MANIFEST_SCHEMA,
        label="shard manifest",
    )
    return copy.deepcopy(dict(index))


def _read_reference(
    root: Path, reference: Mapping[str, Any], *, label: str
) -> tuple[Path, bytes, dict[str, Any]]:
    path = _safe_child(root, str(reference["path"]))
    payload = _read_bounded(path, MAX_HOT_WIRE_BYTES, label)
    if len(payload) != reference["size"] or _sha256(payload) != reference["sha256"]:
        raise RuntimeError(f"compact {label} file/reference seal mismatch")
    return path, payload, _json_object(payload, label)


def _validate_status_counts(status: Mapping[str, Any]) -> None:
    lanes = status.get("latest_tasks")
    state_counts = status.get("state_counts")
    if not isinstance(lanes, list) or not isinstance(state_counts, dict):
        raise RuntimeError("compact status physical lane inventory is invalid")
    task_ids: set[int] = set()
    logical = 0
    sealed = 0
    observed_states: Counter[str] = Counter()
    for lane in lanes:
        if not isinstance(lane, dict):
            raise RuntimeError("compact status physical lane record is invalid")
        task_id = lane.get("task_id")
        batch_length = lane.get("batch_length")
        sealed_count = lane.get("sealed_child_count")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or task_id in task_ids
            or isinstance(batch_length, bool)
            or not isinstance(batch_length, int)
            or batch_length <= 0
            or isinstance(sealed_count, bool)
            or not isinstance(sealed_count, int)
            or not 0 <= sealed_count <= batch_length
        ):
            raise RuntimeError("compact status physical/logical lane identity drifted")
        task_ids.add(task_id)
        logical += batch_length
        sealed += sealed_count
        observed_states[str(lane.get("state") or "unknown")] += 1
    hidden = status.get("hidden_unsealed_seed_record_count")
    if (
        status.get("physical_lane_count") != len(lanes)
        or status.get("logical_seed_count") != logical
        or status.get("logical_sealed_seed_count") != sealed
        or status.get("logical_unsealed_seed_count") != logical - sealed
        or dict(sorted(observed_states.items())) != state_counts
        or isinstance(hidden, bool)
        or not isinstance(hidden, int)
        or hidden < 0
    ):
        raise RuntimeError("compact status physical/logical counters drifted")


def _validate_snapshot_contract(
    index: Mapping[str, Any],
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    references = manifest.get("shards")
    record_count = manifest.get("record_count")
    shard_size = manifest.get("shard_size")
    if (
        status.get("stage_id") != index.get("stage_id")
        or manifest.get("stage_id") != index.get("stage_id")
        or status.get("terminal_results_sharded") is not True
        or status.get("terminal_results_manifest_sha256")
        != manifest.get("manifest_sha256")
        or status.get("virtual_scheduler_task_ids_created") is not False
        or manifest.get("virtual_scheduler_task_ids_created") is not False
        or any(
            status.get(field) is not False
            for field in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "aedt_used",
                "automatic_promotion_allowed",
            )
        )
        or isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count < 0
        or status.get("authenticated_terminal_seed_count") != record_count
        or index["seed_result_shards"].get("record_count") != record_count
        or isinstance(shard_size, bool)
        or not isinstance(shard_size, int)
        or not 1 <= shard_size <= 4096
        or not isinstance(references, list)
        or manifest.get("shard_count") != len(references)
    ):
        raise RuntimeError("compact index/status/manifest contract drifted")
    cursor = 0
    paths: set[str] = set()
    for ordinal, reference in enumerate(references):
        count = reference.get("record_count") if isinstance(reference, dict) else None
        size = reference.get("size") if isinstance(reference, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if (
            not isinstance(reference, dict)
            or reference.get("ordinal") != ordinal
            or reference.get("record_offset") != cursor
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= shard_size
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size < MAX_HOT_WIRE_BYTES
            or not isinstance(path, str)
            or _safe_relative(path) != path
            or path in paths
            or not HEX_SHA256.fullmatch(str(reference.get("sha256") or ""))
        ):
            raise RuntimeError("compact shard manifest range/reference drifted")
        paths.add(path)
        cursor += count
    if cursor != record_count:
        raise RuntimeError("compact shard manifest does not cover its record count")


def load_compact_condition_index(index_path: Path) -> dict[str, Any]:
    """Load one coherent local compact snapshot without hydrating shards."""

    path = index_path.resolve(strict=True)
    root = path.parent.resolve(strict=True)
    first_index_payload = _read_bounded(
        path, MAX_HOT_WIRE_BYTES, "compact condition index"
    )
    index = _validate_index(
        _json_object(first_index_payload, "compact condition index")
    )
    status_path, status_payload, status = _read_reference(
        root, index["status"], label="status"
    )
    manifest_path, manifest_payload, manifest = _read_reference(
        root, index["seed_result_shards"], label="shard manifest"
    )
    # limit=0 performs the complete status/manifest seal validation without
    # requesting a shard.  Page hydration performs shard validation below.
    adapt_status_for_frontend(status, manifest, lambda _reference: {}, limit=0)
    _validate_status_counts(status)
    _validate_snapshot_contract(index, status, manifest)
    if _read_bounded(path, MAX_HOT_WIRE_BYTES, "compact condition index") != (
        first_index_payload
    ):
        raise RuntimeError("compact condition index advanced during snapshot load")
    return {
        "root": root,
        "index_path": path,
        "index_bytes": first_index_payload,
        "index": index,
        "status_path": status_path,
        "status_bytes": status_payload,
        "status": status,
        "manifest_path": manifest_path,
        "manifest_bytes": manifest_payload,
        "manifest": manifest,
    }


def adapt_condition_index(
    index_path: Path, *, offset: int = 0, limit: int | None = None
) -> dict[str, Any]:
    """Hydrate a compact index into the existing 8010 Current7 status wire."""

    snapshot = load_compact_condition_index(index_path)

    def load_shard(reference: Mapping[str, Any]) -> dict[str, Any]:
        _path, _payload, value = _read_reference(
            snapshot["root"], reference, label="seed-result shard"
        )
        return value

    normalized = adapt_status_for_frontend(
        snapshot["status"],
        snapshot["manifest"],
        load_shard,
        offset=offset,
        limit=limit,
    )
    wire_size = len(json_bytes(normalized))
    if wire_size >= MAX_HOT_WIRE_BYTES:
        raise RuntimeError("normalized 8010 status exceeds the 32-MiB wire cap")
    if (
        _read_bounded(
            snapshot["index_path"], MAX_HOT_WIRE_BYTES, "compact condition index"
        )
        != snapshot["index_bytes"]
    ):
        raise RuntimeError("compact condition index advanced during page hydration")
    return normalized


def _file_record(logical_name: str, path: Path) -> dict[str, Any]:
    logical = _safe_relative(logical_name)
    resolved = path.resolve(strict=True)
    payload = _read_bounded(resolved, MAX_HOT_WIRE_BYTES, logical)
    return {
        "logical_name": logical,
        "resolved_path": str(resolved),
        "sha256": _sha256(payload),
        "size": len(payload),
    }


def _file_set_identity(
    named_paths: Mapping[str, Path], *, kind: str, revision: str
) -> dict[str, Any]:
    if not named_paths or not HEX_REVISION.fullmatch(str(revision)):
        raise RuntimeError(f"exact {kind} identity is incomplete")
    files = [_file_record(name, path) for name, path in sorted(named_paths.items())]
    names = [record["logical_name"] for record in files]
    resolved_paths = [record["resolved_path"] for record in files]
    if len(names) != len(set(names)) or len(resolved_paths) != len(set(resolved_paths)):
        raise RuntimeError(f"exact {kind} identity repeats a logical file")
    unsigned = {
        "schema_version": FILE_SET_IDENTITY_SCHEMA,
        "kind": kind,
        "revision": str(revision),
        "files": files,
    }
    return {**unsigned, "identity_sha256": canonical_sha256(unsigned)}


def _validate_file_set_identity(
    value: Mapping[str, Any], *, kind: str
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    files = value.get("files")
    if (
        set(value)
        != {
            "schema_version",
            "kind",
            "revision",
            "files",
            "identity_sha256",
        }
        or value.get("schema_version") != FILE_SET_IDENTITY_SCHEMA
        or value.get("kind") != kind
        or not HEX_REVISION.fullmatch(str(value.get("revision") or ""))
        or not isinstance(files, list)
        or not files
        or value.get("identity_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"sealed {kind} file-set identity is invalid")
    names: set[str] = set()
    resolved_paths: set[str] = set()
    for record in files:
        size = record.get("size") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or set(record) != {"logical_name", "resolved_path", "sha256", "size"}
            or not isinstance(record.get("logical_name"), str)
            or _safe_relative(record["logical_name"]) != record["logical_name"]
            or record["logical_name"] in names
            or not isinstance(record.get("resolved_path"), str)
            or not Path(record["resolved_path"]).is_absolute()
            or record["resolved_path"] in resolved_paths
            or not HEX_SHA256.fullmatch(str(record.get("sha256") or ""))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size < MAX_HOT_WIRE_BYTES
        ):
            raise RuntimeError(f"sealed {kind} file record is invalid")
        names.add(record["logical_name"])
        resolved_paths.add(record["resolved_path"])
    if files != sorted(files, key=lambda item: item["logical_name"]):
        raise RuntimeError(f"sealed {kind} file records are not canonical")
    return copy.deepcopy(dict(value))


def _adapter_probe(index_path: Path) -> dict[str, Any]:
    page = adapt_condition_index(index_path, offset=0, limit=DEFAULT_FRONTEND_PAGE_SIZE)
    payload = json_bytes(page)
    unsigned = {
        "normalized_status_schema_version": page["schema_version"],
        "normalized_status_sha256": _sha256(payload),
        "normalized_wire_size": len(payload),
        "physical_lane_count": int(page["physical_lane_count"]),
        "logical_seed_count": int(page["logical_seed_count"]),
        "logical_sealed_seed_count": int(page["logical_sealed_seed_count"]),
        "logical_unsealed_seed_count": int(page["logical_unsealed_seed_count"]),
        "returned_terminal_seed_count": int(page["terminal_results_returned_count"]),
        "hidden_unsealed_seed_record_count": int(
            page["hidden_unsealed_seed_record_count"]
        ),
        "wire_below_32_mib": len(payload) < MAX_HOT_WIRE_BYTES,
    }
    return {**unsigned, "probe_sha256": canonical_sha256(unsigned)}


def _index_identity(index_path: Path) -> dict[str, Any]:
    snapshot = load_compact_condition_index(index_path)
    probe = _adapter_probe(index_path)
    unsigned = {
        "schema_version": INDEX_IDENTITY_SCHEMA,
        "index_path": str(snapshot["index_path"]),
        "index_file_sha256": _sha256(snapshot["index_bytes"]),
        "index_file_size": len(snapshot["index_bytes"]),
        "index_schema_version": snapshot["index"]["schema_version"],
        "index_sha256": snapshot["index"]["index_sha256"],
        "stage_id": snapshot["index"]["stage_id"],
        "status_path": str(snapshot["status_path"]),
        "status_file_sha256": _sha256(snapshot["status_bytes"]),
        "status_file_size": len(snapshot["status_bytes"]),
        "status_sha256": snapshot["status"]["status_sha256"],
        "manifest_path": str(snapshot["manifest_path"]),
        "manifest_file_sha256": _sha256(snapshot["manifest_bytes"]),
        "manifest_file_size": len(snapshot["manifest_bytes"]),
        "manifest_sha256": snapshot["manifest"]["manifest_sha256"],
        "adapter_probe": probe,
    }
    return {**unsigned, "identity_sha256": canonical_sha256(unsigned)}


def _validate_index_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "identity_sha256"}
    required = {
        "schema_version",
        "index_path",
        "index_file_sha256",
        "index_file_size",
        "index_schema_version",
        "index_sha256",
        "stage_id",
        "status_path",
        "status_file_sha256",
        "status_file_size",
        "status_sha256",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_file_size",
        "manifest_sha256",
        "adapter_probe",
        "identity_sha256",
    }
    probe = value.get("adapter_probe")
    probe_unsigned = (
        {key: item for key, item in probe.items() if key != "probe_sha256"}
        if isinstance(probe, dict)
        else {}
    )
    size_fields = (
        "index_file_size",
        "status_file_size",
        "manifest_file_size",
    )
    count_fields = (
        "physical_lane_count",
        "logical_seed_count",
        "logical_sealed_seed_count",
        "logical_unsealed_seed_count",
        "returned_terminal_seed_count",
        "hidden_unsealed_seed_record_count",
        "normalized_wire_size",
    )
    if (
        set(value) != required
        or value.get("schema_version") != INDEX_IDENTITY_SCHEMA
        or value.get("index_schema_version") != COMPACT_INDEX_SCHEMA
        or value.get("identity_sha256") != canonical_sha256(unsigned)
        or not isinstance(probe, dict)
        or probe.get("probe_sha256") != canonical_sha256(probe_unsigned)
        or probe.get("normalized_status_schema_version") != COHORT_STATUS_SCHEMA
        or probe.get("wire_below_32_mib") is not True
        or any(
            isinstance(value.get(field), bool)
            or not isinstance(value.get(field), int)
            or value[field] <= 0
            or value[field] >= MAX_HOT_WIRE_BYTES
            for field in size_fields
        )
        or any(
            isinstance(probe.get(field), bool)
            or not isinstance(probe.get(field), int)
            or probe[field] < 0
            for field in count_fields
        )
        or probe.get("logical_sealed_seed_count", 0)
        + probe.get("logical_unsealed_seed_count", 0)
        != probe.get("logical_seed_count")
        or probe.get("normalized_wire_size", MAX_HOT_WIRE_BYTES) >= MAX_HOT_WIRE_BYTES
        or any(
            not isinstance(value.get(field), str)
            or not Path(value[field]).is_absolute()
            for field in ("index_path", "status_path", "manifest_path")
        )
        or any(
            not HEX_SHA256.fullmatch(str(value.get(field) or ""))
            for field in (
                "index_file_sha256",
                "index_sha256",
                "status_file_sha256",
                "status_sha256",
                "manifest_file_sha256",
                "manifest_sha256",
            )
        )
    ):
        raise RuntimeError("sealed compact condition-index identity is invalid")
    return copy.deepcopy(dict(value))


def _test_seals(
    evidence_files: Mapping[str, Path], *, revision: str, probe: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {
        "schema_version": TEST_SEALS_SCHEMA,
        "evidence_identity": _file_set_identity(
            evidence_files, kind="test_evidence", revision=revision
        ),
        "capability_contract_sha256": CAPABILITY_CONTRACT_SHA256,
        "adapter_probe_sha256": probe["probe_sha256"],
    }
    return {**unsigned, "seals_sha256": canonical_sha256(unsigned)}


def _validate_test_seals(value: Mapping[str, Any], probe: Mapping[str, Any]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "seals_sha256"}
    if (
        set(value)
        != {
            "schema_version",
            "evidence_identity",
            "capability_contract_sha256",
            "adapter_probe_sha256",
            "seals_sha256",
        }
        or value.get("schema_version") != TEST_SEALS_SCHEMA
        or value.get("capability_contract_sha256") != CAPABILITY_CONTRACT_SHA256
        or value.get("adapter_probe_sha256") != probe.get("probe_sha256")
        or value.get("seals_sha256") != canonical_sha256(unsigned)
        or not isinstance(value.get("evidence_identity"), dict)
    ):
        raise RuntimeError("sealed backend test evidence is invalid")
    _validate_file_set_identity(value["evidence_identity"], kind="test_evidence")


def build_backend_capability_receipt(
    *,
    index_path: Path,
    code_files: Mapping[str, Path],
    code_revision: str,
    backend_files: Mapping[str, Path],
    backend_revision: str,
    test_evidence_files: Mapping[str, Path],
    test_revision: str,
) -> dict[str, Any]:
    """Seal exact local identities after a deterministic adapter probe."""

    index_identity = _index_identity(index_path)
    unsigned = {
        "schema_version": CAPABILITY_RECEIPT_SCHEMA,
        "created_at": now(),
        "capability_contract": copy.deepcopy(CAPABILITY_CONTRACT),
        "code_identity": _file_set_identity(
            code_files, kind="adapter_code", revision=code_revision
        ),
        "backend_identity": _file_set_identity(
            backend_files, kind="backend_code", revision=backend_revision
        ),
        "condition_index_identity": index_identity,
        "test_seals": _test_seals(
            test_evidence_files,
            revision=test_revision,
            probe=index_identity["adapter_probe"],
        ),
        "scheduler_mutation_performed": False,
        "remote_write_performed": False,
        "aedt_used": False,
        "fea_submission_performed": False,
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def validate_backend_capability_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a receipt's internal seals without trusting local files."""

    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "created_at",
        "capability_contract",
        "code_identity",
        "backend_identity",
        "condition_index_identity",
        "test_seals",
        "scheduler_mutation_performed",
        "remote_write_performed",
        "aedt_used",
        "fea_submission_performed",
        "receipt_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != CAPABILITY_RECEIPT_SCHEMA
        or not isinstance(value.get("created_at"), str)
        or not str(value["created_at"]).strip()
        or value.get("capability_contract") != CAPABILITY_CONTRACT
        or canonical_sha256(value["capability_contract"]) != CAPABILITY_CONTRACT_SHA256
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_mutation_performed",
                "remote_write_performed",
                "aedt_used",
                "fea_submission_performed",
            )
        )
        or not isinstance(value.get("code_identity"), dict)
        or not isinstance(value.get("backend_identity"), dict)
        or not isinstance(value.get("condition_index_identity"), dict)
        or not isinstance(value.get("test_seals"), dict)
    ):
        raise RuntimeError("backend capability receipt identity is invalid")
    _validate_file_set_identity(value["code_identity"], kind="adapter_code")
    _validate_file_set_identity(value["backend_identity"], kind="backend_code")
    index_identity = _validate_index_identity(value["condition_index_identity"])
    _validate_test_seals(value["test_seals"], index_identity["adapter_probe"])
    return copy.deepcopy(dict(value))


def require_backend_capability(
    receipt_path: Path,
    *,
    index_path: Path,
    code_files: Mapping[str, Path],
    code_revision: str,
    backend_files: Mapping[str, Path],
    backend_revision: str,
    test_evidence_files: Mapping[str, Path],
    test_revision: str,
) -> dict[str, Any]:
    """Refuse missing, stale, relocated, or otherwise drifted capability."""

    if not receipt_path.is_file():
        raise RuntimeError("sealed multi-seed backend capability receipt is required")
    receipt = validate_backend_capability_receipt(
        _json_object(
            _read_bounded(
                receipt_path.resolve(strict=True),
                MAX_CAPABILITY_RECEIPT_BYTES,
                "backend capability receipt",
            ),
            "backend capability receipt",
        )
    )
    expected = {
        "code_identity": _file_set_identity(
            code_files, kind="adapter_code", revision=code_revision
        ),
        "backend_identity": _file_set_identity(
            backend_files, kind="backend_code", revision=backend_revision
        ),
        "condition_index_identity": _index_identity(index_path),
    }
    expected["test_seals"] = _test_seals(
        test_evidence_files,
        revision=test_revision,
        probe=expected["condition_index_identity"]["adapter_probe"],
    )
    for field, identity in expected.items():
        if receipt[field] != identity:
            raise RuntimeError(f"backend capability receipt {field} drifted")
    return receipt


def default_adapter_code_files(code_root: Path) -> dict[str, Path]:
    root = code_root.resolve(strict=True)
    return {
        relative: root / Path(*PurePosixPath(relative).parts)
        for relative in DEFAULT_ADAPTER_CODE_FILES
    }


def _named_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise RuntimeError(f"{label} must use LOGICAL_NAME=PATH")
        logical = _safe_relative(name)
        if logical in result:
            raise RuntimeError(f"{label} repeats {logical}")
        result[logical] = Path(raw_path)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(json_bytes(value))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument(
        "--backend-file", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument(
        "--test-evidence", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--test-revision", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    adapt = commands.add_parser("adapt", help="print one normalized 8010 page")
    adapt.add_argument("--index", type=Path, required=True)
    adapt.add_argument("--offset", type=int, default=0)
    adapt.add_argument("--limit", type=int)
    seal = commands.add_parser("seal-capability")
    _identity_arguments(seal)
    seal.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-capability")
    _identity_arguments(validate)
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def _identities(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "index_path": args.index,
        "code_files": default_adapter_code_files(args.code_root),
        "code_revision": args.code_revision,
        "backend_files": _named_paths(args.backend_file, "backend file"),
        "backend_revision": args.backend_revision,
        "test_evidence_files": _named_paths(args.test_evidence, "test evidence"),
        "test_revision": args.test_revision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "adapt":
        print(
            json.dumps(
                adapt_condition_index(args.index, offset=args.offset, limit=args.limit),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    identities = _identities(args)
    if args.command == "seal-capability":
        receipt = build_backend_capability_receipt(**identities)
        _atomic_json(args.output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    receipt = require_backend_capability(args.receipt, **identities)
    print(
        json.dumps(
            {
                "valid": True,
                "receipt_sha256": receipt["receipt_sha256"],
                "condition_index_identity_sha256": receipt["condition_index_identity"][
                    "identity_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
