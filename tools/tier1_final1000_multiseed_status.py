"""Bounded status/index projection and legacy frontend adapter for lane v2.

Full seed history is split into immutable content-addressed shards.  The hot
status contains at most the physical lane inventory and small counters.  A
backend request can hydrate the same ``terminal_results`` field consumed by
the existing frontend without inventing Scheduler task IDs.
"""

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_harvest import (
        COHORT_STATUS_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
    )
    from tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        COMPACT_STATUS_SCHEMA,
        MAX_PHYSICAL_LANES,
        MAX_BATCH_LENGTH,
        PROTOCOL_VERSION,
        SHARD_MANIFEST_SCHEMA,
        json_bytes,
        now,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_harvest import (
        COHORT_STATUS_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
    )
    from tools.tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        COMPACT_STATUS_SCHEMA,
        MAX_PHYSICAL_LANES,
        MAX_BATCH_LENGTH,
        PROTOCOL_VERSION,
        SHARD_MANIFEST_SCHEMA,
        json_bytes,
        now,
    )


SEED_RESULT_SHARD_SCHEMA = "mft-tier1-final1000-seed-result-shard-v2"
DEFAULT_SHARD_SIZE = 256
DEFAULT_FRONTEND_PAGE_SIZE = 256
MAX_FRONTEND_PAGE_SIZE = 4096
MAX_HOT_WIRE_BYTES = 32 * 1024 * 1024
SINGLE_SEED_PROTOCOL = "final1000-single-seed-v1"
TERMINAL_LANE_STATES = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
        "canceled",
        "timeout",
        "timed_out",
        "stopped",
        "deadline",
    }
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _public(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if not str(key).startswith("_")
    }


def _physical_lane(value: Mapping[str, Any]) -> dict[str, Any]:
    task_id = value.get("task_id", value.get("id"))
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise RuntimeError("compact status physical lane lacks a Scheduler task id")
    protocol = value.get("protocol_version") or SINGLE_SEED_PROTOCOL
    if protocol not in {SINGLE_SEED_PROTOCOL, PROTOCOL_VERSION}:
        raise RuntimeError("compact status physical lane protocol is unknown")
    state = str(value.get("state") or "unknown").lower()
    if protocol == PROTOCOL_VERSION:
        batch_length = value.get("batch_length")
        sealed_child_count = value.get("sealed_child_count")
        if (
            isinstance(batch_length, bool)
            or not isinstance(batch_length, int)
            or not 1 <= batch_length <= MAX_BATCH_LENGTH
            or isinstance(sealed_child_count, bool)
            or not isinstance(sealed_child_count, int)
            or not 0 <= sealed_child_count <= batch_length
        ):
            raise RuntimeError("compact status batch lane cursor is invalid")
    else:
        batch_length = 1
        sealed_child_count = 1 if state in TERMINAL_LANE_STATES else 0
    # Payloads are authenticated elsewhere and intentionally excluded from the
    # hot wire.  This is a bounded physical-lane summary, not scientific history.
    return {
        key: copy.deepcopy(value.get(key))
        for key in (
            "stage_id",
            "protocol_version",
            "state",
            "account_name",
            "batch_length",
            "current_seed",
            "sealed_child_count",
            "started_at",
            "updated_at",
        )
    } | {
        "task_id": int(task_id),
        "protocol_version": protocol,
        "state": state,
        "batch_length": int(batch_length),
        "sealed_child_count": int(sealed_child_count),
    }


def _visible_seed_records(
    lanes: Sequence[Mapping[str, Any]],
    seed_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return only child records proven inside each visible parent's seal.

    Historical records whose physical parent is no longer in the bounded hot
    lane inventory remain visible.  When a batch parent is present, however,
    its ordinal must be authenticated and strictly below ``sealed_child_count``.
    This keeps a receipt observed in the cursor/receipt crash window out of the
    8010 projection until the parent journal seals that exact prefix.
    """

    by_task_id = {int(lane["task_id"]): lane for lane in lanes}
    visible: list[dict[str, Any]] = []
    hidden = 0
    for raw in seed_records:
        record = _public(raw)
        parent_task_id = record.get("physical_parent_task_id")
        if parent_task_id is None:
            visible.append(record)
            continue
        if (
            isinstance(parent_task_id, bool)
            or not isinstance(parent_task_id, int)
            or parent_task_id <= 0
        ):
            raise RuntimeError("compact status child record parent id is invalid")
        lane = by_task_id.get(parent_task_id)
        if lane is None:
            # The compact hot inventory is bounded to current physical lanes;
            # immutable history from an older parent is already sealed by the
            # harvester and must not disappear merely because that lane drained.
            visible.append(record)
            continue
        if lane.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(
                "compact status child record points at a non-batch physical lane"
            )
        ordinal = record.get("batch_ordinal")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or not 0 <= ordinal < int(lane["batch_length"])
        ):
            raise RuntimeError("compact status child record ordinal is invalid")
        if ordinal >= int(lane["sealed_child_count"]):
            hidden += 1
            continue
        visible.append(record)
    return visible, hidden


def _normalized_frontend_status(
    status: Mapping[str, Any],
    *,
    record_count: int,
    offset: int,
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    static = copy.deepcopy(status.get("frontend_static") or {})
    returned = len(selected)
    return {
        **static,
        "schema_version": COHORT_STATUS_SCHEMA,
        "updated_at": status["updated_at"],
        "scheduler_task_count": status["physical_lane_count"],
        "physical_lane_count": status["physical_lane_count"],
        "logical_seed_count": status["logical_seed_count"],
        "logical_sealed_seed_count": status["logical_sealed_seed_count"],
        "logical_unsealed_seed_count": status["logical_unsealed_seed_count"],
        "hidden_unsealed_seed_record_count": status[
            "hidden_unsealed_seed_record_count"
        ],
        "state_counts": copy.deepcopy(status["state_counts"]),
        "latest_tasks": copy.deepcopy(status["latest_tasks"]),
        "authenticated_terminal_seed_count": status[
            "authenticated_terminal_seed_count"
        ],
        "terminal_results": copy.deepcopy(list(selected)),
        "terminal_results_offset": int(offset),
        "terminal_results_returned_count": returned,
        "terminal_results_total_count": int(record_count),
        "terminal_results_has_more": int(offset) + returned < int(record_count),
        "virtual_scheduler_task_ids_created": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }


def build_compact_snapshot(
    *,
    stage_id: str,
    physical_lanes: Sequence[Mapping[str, Any]],
    seed_records: Sequence[Mapping[str, Any]],
    frontend_static: Mapping[str, Any] | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    relative_root: str = "seed-results",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Return compact status, shard manifest, shard objects, and compact index."""

    if not 1 <= int(shard_size) <= 4096:
        raise RuntimeError("seed-result shard size is outside the sealed bound")
    if len(physical_lanes) > MAX_PHYSICAL_LANES:
        raise RuntimeError("compact status exceeds 500 physical lanes")
    lanes = [_physical_lane(item) for item in physical_lanes]
    task_ids = [int(item["task_id"]) for item in lanes]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("compact status repeats a physical Scheduler task id")
    visible_records, hidden_unsealed_count = _visible_seed_records(lanes, seed_records)
    records = sorted(
        visible_records,
        key=lambda item: (str(item.get("bundle_id") or ""), int(item.get("seed", -1))),
    )
    identities = [
        (str(item.get("bundle_id") or ""), int(item.get("seed", -1)))
        for item in records
    ]
    if any(seed < 0 or not bundle for bundle, seed in identities) or len(
        identities
    ) != len(set(identities)):
        raise RuntimeError(
            "compact status seed-result identity is missing or duplicated"
        )
    shard_objects: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for ordinal, offset in enumerate(range(0, len(records), int(shard_size))):
        chunk = records[offset : offset + int(shard_size)]
        unsigned = {
            "schema_version": SEED_RESULT_SHARD_SCHEMA,
            "stage_id": stage_id,
            "ordinal": ordinal,
            "record_offset": offset,
            "record_count": len(chunk),
            "records": chunk,
        }
        shard = {**unsigned, "shard_sha256": canonical_sha256(unsigned)}
        payload = json_bytes(shard)
        if len(payload) >= MAX_HOT_WIRE_BYTES:
            raise RuntimeError("seed-result shard exceeds the deployed 32-MiB wire cap")
        path = f"{relative_root}/shard-{ordinal:06d}-{_sha_bytes(payload)[:16]}.json"
        references.append(
            {
                "ordinal": ordinal,
                "path": path,
                "sha256": _sha_bytes(payload),
                "size": len(payload),
                "record_offset": offset,
                "record_count": len(chunk),
                "first_seed": int(chunk[0]["seed"]) if chunk else None,
                "last_seed": int(chunk[-1]["seed"]) if chunk else None,
            }
        )
        shard_objects.append(shard)
    manifest_unsigned = {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "stage_id": stage_id,
        "record_count": len(records),
        "shard_size": int(shard_size),
        "shard_count": len(references),
        "shards": references,
        "virtual_scheduler_task_ids_created": False,
    }
    shard_manifest = {
        **manifest_unsigned,
        "manifest_sha256": canonical_sha256(manifest_unsigned),
    }
    static = copy.deepcopy(dict(frontend_static or {}))
    for forbidden in (
        "latest_tasks",
        "terminal_results",
        "authenticated_terminal_seed_count",
    ):
        static.pop(forbidden, None)
    timestamp = now()
    status_unsigned = {
        "schema_version": COMPACT_STATUS_SCHEMA,
        "stage_id": stage_id,
        "updated_at": timestamp,
        "physical_lane_count": len(lanes),
        "logical_seed_count": sum(int(item["batch_length"]) for item in lanes),
        "logical_sealed_seed_count": sum(
            int(item["sealed_child_count"]) for item in lanes
        ),
        "logical_unsealed_seed_count": sum(
            int(item["batch_length"]) - int(item["sealed_child_count"])
            for item in lanes
        ),
        "hidden_unsealed_seed_record_count": hidden_unsealed_count,
        "state_counts": dict(
            sorted(
                Counter(str(item.get("state") or "unknown") for item in lanes).items()
            )
        ),
        "latest_tasks": lanes,
        "authenticated_terminal_seed_count": len(records),
        "terminal_results_sharded": True,
        "terminal_results_manifest_sha256": shard_manifest["manifest_sha256"],
        "frontend_static": static,
        "virtual_scheduler_task_ids_created": False,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    status = {**status_unsigned, "status_sha256": canonical_sha256(status_unsigned)}
    status_payload = json_bytes(status)
    if records:
        # A shard smaller than 32 MiB is not sufficient: one record must also
        # fit after the legacy frontend wrapper and physical-lane hot status are
        # serialized.  Rank by nested JSON contribution, then perform an exact
        # whole-response gate on the worst record.
        largest = max(
            records,
            key=lambda record: len(json_bytes({"terminal_results": [record]})),
        )
        one_record_wire = json_bytes(
            _normalized_frontend_status(
                status,
                record_count=len(records),
                offset=max(0, len(records) - 1),
                selected=[largest],
            )
        )
        if len(one_record_wire) >= MAX_HOT_WIRE_BYTES:
            raise RuntimeError(
                "one seed-result record cannot fit the deployed frontend wire cap"
            )
    manifest_payload = json_bytes(shard_manifest)
    status_path = f"snapshots/status-{_sha_bytes(status_payload)[:16]}.json"
    manifest_path = f"seed-results/manifest-{_sha_bytes(manifest_payload)[:16]}.json"
    index_unsigned = {
        "schema_version": COMPACT_INDEX_SCHEMA,
        "stage_id": stage_id,
        "updated_at": timestamp,
        "status": {
            "path": status_path,
            "schema_version": COMPACT_STATUS_SCHEMA,
            "sha256": _sha_bytes(status_payload),
            "size": len(status_payload),
        },
        "seed_result_shards": {
            "path": manifest_path,
            "schema_version": SHARD_MANIFEST_SCHEMA,
            "sha256": _sha_bytes(manifest_payload),
            "size": len(manifest_payload),
            "record_count": len(records),
        },
        "frontend_adapter": {
            "normalized_status_schema_version": COHORT_STATUS_SCHEMA,
            "normalized_index_schema_version": CURRENT7_INDEX_SCHEMA,
            "terminal_results_field": "terminal_results",
            "on_demand": True,
        },
        "count_contract": {
            "scheduler_task_count_field": "physical_lane_count",
            "logical_seed_count_field": "logical_seed_count",
            "sealed_prefix_rule": "batch_ordinal < sealed_child_count",
        },
        "virtual_scheduler_task_ids_created": False,
    }
    index = {**index_unsigned, "index_sha256": canonical_sha256(index_unsigned)}
    if (
        len(status_payload) >= MAX_HOT_WIRE_BYTES
        or len(json_bytes(index)) >= MAX_HOT_WIRE_BYTES
    ):
        raise RuntimeError(
            "compact hot status/index exceeds the deployed 32-MiB wire cap"
        )
    return status, shard_manifest, shard_objects, index


def validate_shard(
    value: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "shard_sha256"}
    payload = json_bytes(value)
    records = value.get("records")
    if (
        value.get("schema_version") != SEED_RESULT_SHARD_SCHEMA
        or value.get("shard_sha256") != canonical_sha256(unsigned)
        or not isinstance(records, list)
        or len(records) != value.get("record_count")
        or value.get("ordinal") != reference.get("ordinal")
        or value.get("record_offset") != reference.get("record_offset")
        or len(records) != reference.get("record_count")
        or _sha_bytes(payload) != reference.get("sha256")
        or len(payload) != reference.get("size")
        or len(payload) >= MAX_HOT_WIRE_BYTES
    ):
        raise RuntimeError("seed-result shard/reference seal mismatch")
    return copy.deepcopy(dict(value))


def adapt_status_for_frontend(
    status: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    shard_loader: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Hydrate the current frontend wire without changing its field names."""

    status_unsigned = {
        key: item for key, item in status.items() if key != "status_sha256"
    }
    manifest_unsigned = {
        key: item for key, item in shard_manifest.items() if key != "manifest_sha256"
    }
    if (
        status.get("schema_version") != COMPACT_STATUS_SCHEMA
        or status.get("status_sha256") != canonical_sha256(status_unsigned)
        or shard_manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA
        or shard_manifest.get("manifest_sha256") != canonical_sha256(manifest_unsigned)
        or status.get("terminal_results_manifest_sha256")
        != shard_manifest.get("manifest_sha256")
        or int(offset) < 0
        or (limit is not None and int(limit) < 0)
    ):
        raise RuntimeError("compact status/shard manifest adapter identity mismatch")
    effective_limit = DEFAULT_FRONTEND_PAGE_SIZE if limit is None else int(limit)
    if effective_limit > MAX_FRONTEND_PAGE_SIZE:
        raise RuntimeError("frontend seed-result page exceeds the sealed record bound")
    end = int(offset) + effective_limit
    selected: list[dict[str, Any]] = []
    for reference in shard_manifest["shards"]:
        shard_start = int(reference["record_offset"])
        shard_end = shard_start + int(reference["record_count"])
        if shard_end <= int(offset) or shard_start >= end:
            continue
        shard = validate_shard(shard_loader(reference), reference)
        local_start = max(0, int(offset) - shard_start)
        local_end = min(len(shard["records"]), end - shard_start)
        selected.extend(copy.deepcopy(shard["records"][local_start:local_end]))
    total = int(shard_manifest["record_count"])
    expected_count = min(effective_limit, max(0, total - int(offset)))
    if len(selected) != expected_count:
        raise RuntimeError("seed-result shard manifest has a page gap")

    def normalized_prefix(count: int) -> dict[str, Any]:
        return _normalized_frontend_status(
            status,
            record_count=total,
            offset=int(offset),
            selected=selected[:count],
        )

    normalized = normalized_prefix(len(selected))
    if len(json_bytes(normalized)) >= MAX_HOT_WIRE_BYTES and selected:
        # Keep the caller's requested offset stable, but return the largest
        # contiguous prefix that is actually serializable under the deployed
        # wire cap.  This keeps default pagination usable for large records.
        low = 0
        high = len(selected)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json_bytes(normalized_prefix(middle))) < MAX_HOT_WIRE_BYTES:
                low = middle
            else:
                high = middle - 1
        if low == 0:
            raise RuntimeError(
                "one seed-result record exceeds the deployed frontend wire cap"
            )
        normalized = normalized_prefix(low)
    if len(json_bytes(normalized)) >= MAX_HOT_WIRE_BYTES:
        raise RuntimeError(
            "normalized frontend page exceeds the deployed 32-MiB wire cap"
        )
    return normalized


def _immutable_write(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable compact snapshot collision: {path}")
        return 0
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return 1


def _safe_child(root: Path, relative: str) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or any(part in ("", ".", "..") for part in value.parts):
        raise RuntimeError("compact snapshot path is unsafe")
    path = (root / Path(*value.parts)).resolve()
    if path == root or not path.is_relative_to(root):
        raise RuntimeError("compact snapshot path escaped its local root")
    return path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_compact_snapshot(
    root: Path,
    status: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    shard_objects: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
) -> int:
    """Publish only beneath a caller-supplied local root."""

    root = root.resolve()
    writes = 0
    references = shard_manifest.get("shards") or []
    status_unsigned = {
        key: item for key, item in status.items() if key != "status_sha256"
    }
    manifest_unsigned = {
        key: item for key, item in shard_manifest.items() if key != "manifest_sha256"
    }
    index_unsigned = {key: item for key, item in index.items() if key != "index_sha256"}
    if (
        status.get("schema_version") != COMPACT_STATUS_SCHEMA
        or status.get("status_sha256") != canonical_sha256(status_unsigned)
        or shard_manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA
        or shard_manifest.get("manifest_sha256") != canonical_sha256(manifest_unsigned)
        or index.get("schema_version") != COMPACT_INDEX_SCHEMA
        or index.get("index_sha256") != canonical_sha256(index_unsigned)
        or len(references) != len(shard_objects)
    ):
        raise RuntimeError("compact shard publication inventory mismatch")
    for reference, shard in zip(references, shard_objects, strict=True):
        validate_shard(shard, reference)
        writes += _immutable_write(
            _safe_child(root, str(reference["path"])), json_bytes(shard)
        )
    manifest_ref = index.get("seed_result_shards") or {}
    status_ref = index.get("status") or {}
    manifest_payload = json_bytes(shard_manifest)
    status_payload = json_bytes(status)
    if (
        manifest_ref.get("sha256") != _sha_bytes(manifest_payload)
        or manifest_ref.get("size") != len(manifest_payload)
        or status_ref.get("sha256") != _sha_bytes(status_payload)
        or status_ref.get("size") != len(status_payload)
    ):
        raise RuntimeError("compact index/status publication binding mismatch")
    writes += _immutable_write(
        _safe_child(root, str(manifest_ref.get("path") or "")), manifest_payload
    )
    writes += _immutable_write(
        _safe_child(root, str(status_ref.get("path") or "")), status_payload
    )
    _atomic_write(_safe_child(root, "index.json"), json_bytes(index))
    return writes + 1
