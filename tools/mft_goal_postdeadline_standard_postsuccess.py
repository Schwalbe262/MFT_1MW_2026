"""Fail-closed post-success bridge for the two post-deadline Standard lanes.

The custom Standard collector owns Scheduler transport and uses GET requests
only.  This module starts at its immutable ``collection_receipt.json``.  It
reauthenticates the collection lineage, exact submitted candidate, retained
result, fixed operating/cooling identity, and strict-training validity before
emitting any downstream input.

Measured observations and the 512-seed surrogate aggregate are deliberately
kept as separate authorities.  The module emits an exact two-objective
non-dominated sort over the available measured observations and a sealed
rerank-input bridge, but it never labels that diagnostic front as a production
Pareto front and never submits, cancels, or mutates a Scheduler task.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    TEMPERATURE_FAMILY_LIMITS_C,
    TEMPERATURE_TARGET_FAMILIES,
    attest_fixed_identity,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_20260726_launch as launch  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_postdeadline_standard_collector as collector  # noqa: E402


COLLECTION_SCHEMA = collector.COLLECTION_SCHEMA
AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-postdeadline-standard-authenticated-collection-v1"
)
OBSERVATION_SCHEMA = (
    "mft-goal-postdeadline-standard-measured-observation-v1"
)
RERANK_INPUT_SCHEMA = "mft-goal-postdeadline-standard-rerank-input-v1"
SNAPSHOT_SCHEMA = "mft-goal-postdeadline-standard-postsuccess-snapshot-v1"
STATE_SCHEMA = "mft-goal-postdeadline-standard-postsuccess-state-v1"
PID_SCHEMA = "mft-goal-postdeadline-standard-postsuccess-pid-v1"
OFFICIAL5_DIRECT_COLLECTION_SCHEMA = (
    "mft-goal-official5-direct-terminal-collection-v1"
)
OFFICIAL5_DIRECT_COLLECTION_SEAL_SCHEMA = (
    "mft-goal-official5-direct-terminal-collection-seal-v1"
)
OFFICIAL5_DIRECT_AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-official5-direct-authenticated-collection-v1"
)
OFFICIAL5_DIRECT_TASK_ID = 96_338
OFFICIAL5_SOURCE_TASK_ID = 96_332
OFFICIAL5_SOURCE_ALLOCATION_ID = 14_650
OFFICIAL5_SOURCE_SLURM_JOB_ID = "840582"
OFFICIAL5_CANDIDATE_SHA256 = (
    "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
)
CAMPAIGN_ID = "mft-goal-20260726"
DEFAULT_AGGREGATE_MANIFEST = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\aggregate_rolling512_d4e4d60\aggregate_manifest.json"
)
DEFAULT_BASE_DATASET = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs"
    r"\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet"
)
DEFAULT_BASE_SHA256 = (
    "0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3"
)
DEFAULT_BASE_ROWS = 6_151
DEFAULT_INTERVAL_SECONDS = 30
MIN_INTERVAL_SECONDS = 10
MAX_INTERVAL_SECONDS = 300

SAFETY_FLAGS = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "production_eligible": False,
    "original_deadline_missed": True,
}

COLLECTION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        *SAFETY_FLAGS,
        "scientific_pass_claimed",
        "scientific_infeasible_claimed",
        "result_observation_only",
        "scheduler_get_only_collection",
        "scheduler_mutation_performed",
        "task_id",
        "task_name",
        "dedupe_key",
        "candidate_physics_sha256",
        "fixed_physics_unchanged",
        "source_plan_file_sha256",
        "source_plan_payload_sha256",
        "source_submission_file_sha256",
        "source_submission_payload_sha256",
        "scheduler_terminal_task_sha256",
        "result_sha256",
        "retained_symmetric_aedt",
        "retained_symmetric_aedt_remote_sha256",
        "retained_aedt_chunks",
        "retained_aedt_chunk_inventory_sha256",
        "aedtresults_manifest",
        "aedtresults_manifest_payload_sha256",
        "aedtresults_remote_tree_sha256",
        "aedtresults_remote_file_count",
        "aedtresults_remote_size_bytes",
        "result_json",
        "remote_bundle_receipt_payload_sha256",
        "prune_protection_marker_payload_sha256",
        "source_files",
        "payload_sha256",
    }
)

COLLECTION_SEAL_FIELDS = frozenset(
    {
        "schema_version",
        "created_at_utc",
        *SAFETY_FLAGS,
        "scientific_pass_claimed",
        "scientific_infeasible_claimed",
        "result_observation_only",
        "task_id",
        "task_name",
        "dedupe_key",
        "candidate_physics_sha256",
        "source_submission_file_sha256",
        "collection_receipt",
        "collection_receipt_payload_sha256",
        "artifact_sha256",
        "aedtresults_manifest_sha256",
        "aedtresults_tree_sha256",
        "atomic_directory_collection",
        "scheduler_get_only_collection",
        "scheduler_mutation_performed",
        "payload_sha256",
    }
)

SOURCE_FILE_NAMES = frozenset(
    {
        "source_plan.json",
        "source_submission_receipt.json",
        "scheduler_terminal_task.json",
        "scheduler_stdout.log",
        "remote_bundle_receipt.json",
        "prune_protection_marker.json",
    }
)

OBSERVATION_COLUMNS = (
    "task_id",
    "candidate_physics_sha256",
    "actual_volume_L",
    "actual_total_loss_W",
    "actual_width_mm",
    "actual_length_mm",
    "actual_height_mm",
    "actual_resonance_Hz",
    "actual_winding_max_C",
    "actual_core_max_C",
    "measured_hard_constraints_passed",
    "strict_al_row_eligible",
    "audit_non_dominated_rank",
    "hard_feasible_non_dominated_rank",
    "source_seed",
    "source_fixed_primary_turns",
    "solver_revision",
    "library_revision",
    "physics_data_revision",
)


class PostSuccessContractError(RuntimeError):
    """A custom collection, measured truth, or downstream authority drifted."""


@dataclass(frozen=True)
class Lane:
    """One expected Standard collector output."""

    task_id: int
    collection_root: Path
    selection_effective: bool = True


_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PostSuccessContractError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise PostSuccessContractError(f"{label} is not finite")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PostSuccessContractError(f"{label} is invalid")
    return value


def _sha256_file(path: Path) -> str:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise PostSuccessContractError(f"artifact is not a regular file: {path}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _future_file_record(source: Path, destination: Path) -> dict[str, Any]:
    """Record staged bytes under the path they will have after atomic publish."""

    staged = source.resolve(strict=True)
    return {
        "path": str(destination.resolve()),
        "sha256": _sha256_file(staged),
        "size_bytes": staged.stat().st_size,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostSuccessContractError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise PostSuccessContractError(f"{label} is not an object")
    return value


def _validate_seal(
    value: Mapping[str, Any],
    schema: str,
    label: str,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    unsigned = dict(output)
    observed = unsigned.pop("payload_sha256", None)
    if (
        output.get("schema_version") != schema
        or not isinstance(observed, str)
        or observed != collector.payload_sha256(unsigned)
    ):
        raise PostSuccessContractError(f"{label} payload seal drifted")
    return output


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    if "payload_sha256" in output:
        raise PostSuccessContractError("payload is already sealed")
    output["payload_sha256"] = collector.payload_sha256(output)
    return output


def _resolve_record(
    record: Any,
    *,
    root: Path,
    label: str,
    verify_hash: bool = True,
) -> Path:
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or len(str(record["sha256"])) != 64
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or int(record["size_bytes"]) < 0
    ):
        raise PostSuccessContractError(f"{label} file record is malformed")
    raw = Path(str(record["path"]))
    target = raw.resolve(strict=True) if raw.is_absolute() else (root / raw).resolve(
        strict=True
    )
    try:
        target.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise PostSuccessContractError(f"{label} escapes its collection") from exc
    if not target.is_file() or target.is_symlink():
        raise PostSuccessContractError(f"{label} is not a regular file")
    if target.stat().st_size != int(record["size_bytes"]):
        raise PostSuccessContractError(f"{label} size drifted")
    if verify_hash and _sha256_file(target) != str(record["sha256"]):
        raise PostSuccessContractError(f"{label} bytes drifted")
    return target


def _classification_valid(value: Mapping[str, Any]) -> bool:
    return all(value.get(key) is expected for key, expected in SAFETY_FLAGS.items())


def _load_collection_envelope(
    collection_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Authenticate the collector directory and exact submitted result."""

    receipt_path = collection_path.resolve(strict=True)
    root = receipt_path.parent
    if receipt_path != root / "collection_receipt.json":
        raise PostSuccessContractError(
            "post-deadline adapter requires collection_receipt.json"
        )
    seal_path = root / "collection_seal.json"
    receipt = _validate_seal(
        _read_json(receipt_path, "collection receipt"),
        COLLECTION_SCHEMA,
        "collection receipt",
    )
    seal = _validate_seal(
        _read_json(seal_path, "collection seal"),
        collector.COLLECTION_SEAL_SCHEMA,
        "collection seal",
    )
    if set(receipt) != COLLECTION_RECEIPT_FIELDS:
        raise PostSuccessContractError("collection receipt fields drifted")
    if set(seal) != COLLECTION_SEAL_FIELDS:
        raise PostSuccessContractError("collection seal fields drifted")
    if not _classification_valid(receipt) or not _classification_valid(seal):
        raise PostSuccessContractError("post-deadline classification drifted")
    receipt_record_path = _resolve_record(
        seal.get("collection_receipt"),
        root=root,
        label="collection receipt",
    )
    if (
        receipt_record_path != receipt_path
        or seal.get("collection_receipt_payload_sha256")
        != receipt["payload_sha256"]
        or any(
            receipt.get(key) != seal.get(key)
            for key in (
                "task_id",
                "task_name",
                "dedupe_key",
                "candidate_physics_sha256",
                "source_submission_file_sha256",
            )
        )
        or any(
            receipt.get(key) is not expected
            for key, expected in {
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "fixed_physics_unchanged": True,
            }.items()
        )
        or any(
            seal.get(key) is not expected
            for key, expected in {
                "scientific_pass_claimed": False,
                "scientific_infeasible_claimed": False,
                "result_observation_only": True,
                "atomic_directory_collection": True,
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
            }.items()
        )
    ):
        raise PostSuccessContractError("collection receipt/seal binding drifted")

    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != SOURCE_FILE_NAMES:
        raise PostSuccessContractError("collection source file inventory drifted")
    resolved_sources = {
        name: _resolve_record(
            record,
            root=root,
            label=f"collection source {name}",
        )
        for name, record in source_files.items()
    }
    source_submission_path = resolved_sources[
        "source_submission_receipt.json"
    ]
    source_submission = _read_json(
        source_submission_path, "source submission receipt"
    )
    source_plan_record = source_submission.get("plan")
    if not isinstance(source_plan_record, Mapping):
        raise PostSuccessContractError("source submission plan record is absent")
    source_plan_path = Path(str(source_plan_record.get("path") or "")).resolve(
        strict=True
    )
    source_plan = _read_json(source_plan_path, "source post-deadline plan")
    task_id = _positive_int(receipt.get("task_id"), "collection task ID")
    try:
        contract = collector.load_contract(
            plan_path=source_plan_path,
            submission_path=source_submission_path,
            expected_task_id=task_id,
            expected_task_name=str(receipt.get("task_name") or ""),
            expected_dedupe_key=str(receipt.get("dedupe_key") or ""),
            expected_candidate_sha256=str(
                receipt.get("candidate_physics_sha256") or ""
            ),
            expected_receipt_sha256=_sha256_file(source_submission_path),
            scheduler_url=str(source_plan.get("scheduler_url") or ""),
        )
    except (OSError, collector.CollectionError) as exc:
        raise PostSuccessContractError(
            "source plan/submission contract authentication failed"
        ) from exc
    if (
        receipt.get("source_plan_file_sha256") != contract["plan_file_sha256"]
        or receipt.get("source_plan_payload_sha256")
        != contract["plan"]["payload_sha256"]
        or receipt.get("source_submission_file_sha256")
        != contract["submission_file_sha256"]
        or receipt.get("source_submission_payload_sha256")
        != contract["submission"]["payload_sha256"]
        or _sha256_file(resolved_sources["source_plan.json"])
        != contract["plan_file_sha256"]
    ):
        raise PostSuccessContractError("source collection lineage drifted")

    result_path = _resolve_record(
        receipt.get("result_json"),
        root=root,
        label="collected result JSON",
    )
    result = _read_json(result_path, "collected result")
    if receipt.get("result_sha256") != collector.payload_sha256(result):
        raise PostSuccessContractError("collected result payload drifted")

    params_path = collector._find_source(  # noqa: SLF001
        contract["plan"], "fea_params.json", "source FEA params"
    )
    selected_path = collector._find_source(  # noqa: SLF001
        contract["plan"], "selected_candidate.json", "selected candidate"
    )
    profile_path = collector._verify_file_record(  # noqa: SLF001
        contract["plan"].get("execution_profile"), "execution profile"
    )
    params = _read_json(params_path, "source FEA params")
    selected = _read_json(selected_path, "selected candidate")
    profile = _read_json(profile_path, "execution profile")
    effective = dict(params)
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, Mapping):
        raise PostSuccessContractError("execution profile overrides are malformed")
    effective.update(overrides)
    if not scheduler_client.result_matches_params(
        result,
        effective,
        required_keys=set(ALL_INPUT_KEYS),
    ):
        raise PostSuccessContractError(
            "collected result differs from the exact submitted candidate"
        )
    row_contract = selected.get("row_contract")
    if (
        not isinstance(row_contract, Mapping)
        or row_contract.get("fea_params_sha256")
        != collector.payload_sha256(params)
        or selected.get("selected_row", {}).get("candidate_physics_sha")
        != receipt.get("candidate_physics_sha256")
    ):
        raise PostSuccessContractError("selected candidate parameter identity drifted")

    terminal_path = resolved_sources["scheduler_terminal_task.json"]
    terminal = _read_json(terminal_path, "Scheduler terminal task")
    terminal_task_id = terminal.get("task_id", terminal.get("id"))
    if (
        terminal_task_id != task_id
        or terminal.get("name") != receipt.get("task_name")
        or terminal.get("dedupe_key") != receipt.get("dedupe_key")
        or terminal.get("project") != collector.SCHEDULER_PROJECT
        or str(terminal.get("status") or "").lower() != "completed"
        or str(terminal.get("state") or "").lower() != "succeeded"
        or terminal.get("exit_code") != 0
        or receipt.get("scheduler_terminal_task_sha256")
        != collector.payload_sha256(terminal)
    ):
        raise PostSuccessContractError(
            "Scheduler terminal-success evidence drifted"
        )

    remote_receipt_path = resolved_sources["remote_bundle_receipt.json"]
    marker_path = resolved_sources["prune_protection_marker.json"]
    manifest_path = _resolve_record(
        receipt.get("aedtresults_manifest"),
        root=root,
        label="AEDT results manifest",
    )
    remote_receipt = _read_json(remote_receipt_path, "remote bundle receipt")
    marker = _read_json(marker_path, "prune protection marker")
    manifest = _read_json(manifest_path, "AEDT results manifest")
    try:
        authenticated_remote = collector._validate_remote_receipt(  # noqa: SLF001
            remote_receipt, contract, result
        )
        collector._validate_marker(  # noqa: SLF001
            marker,
            marker_path.read_bytes(),
            authenticated_remote,
            contract,
        )
        authenticated_manifest = collector._validate_manifest(  # noqa: SLF001
            manifest,
            manifest_path.read_bytes(),
            authenticated_remote,
            result,
        )
    except collector.CollectionError as exc:
        raise PostSuccessContractError(
            "retained AEDT bundle metadata authentication failed"
        ) from exc
    if (
        receipt.get("remote_bundle_receipt_payload_sha256")
        != collector.payload_sha256(authenticated_remote)
        or receipt.get("prune_protection_marker_payload_sha256")
        != collector.payload_sha256(marker)
        or receipt.get("aedtresults_manifest_payload_sha256")
        != collector.payload_sha256(authenticated_manifest)
        or seal.get("artifact_sha256")
        != authenticated_remote["artifact_sha256"]
        or seal.get("aedtresults_manifest_sha256")
        != authenticated_remote["results_manifest_sha256"]
        or seal.get("aedtresults_tree_sha256")
        != authenticated_remote["results_tree_sha256"]
    ):
        raise PostSuccessContractError("retained bundle receipt binding drifted")

    artifact_path = _resolve_record(
        receipt.get("retained_symmetric_aedt"),
        root=root,
        label="retained symmetric AEDT",
    )
    if (
        _sha256_file(artifact_path) != authenticated_remote["artifact_sha256"]
        or receipt.get("retained_symmetric_aedt_remote_sha256")
        != authenticated_remote["artifact_sha256"]
    ):
        raise PostSuccessContractError("retained symmetric AEDT bytes drifted")
    chunks = receipt.get("retained_aedt_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise PostSuccessContractError("retained AEDT chunk inventory is absent")
    for index, record in enumerate(chunks):
        _resolve_record(
            record,
            root=root,
            label=f"retained AEDT chunk {index}",
        )
    if (
        receipt.get("retained_aedt_chunk_inventory_sha256")
        != collector.payload_sha256(chunks)
        or len(chunks) != authenticated_remote["transport_chunk_count"]
    ):
        raise PostSuccessContractError("retained AEDT chunk inventory drifted")

    return (
        receipt,
        seal,
        contract,
        params,
        selected,
        result,
        {
            "receipt_path": receipt_path,
            "seal_path": seal_path,
            "result_path": result_path,
            "artifact_path": artifact_path,
        },
    )


DirectAuthenticator = Callable[[Path], Mapping[str, Any]]


def _authenticate_official5_direct_collection(
    collection_path: Path,
    *,
    authenticator: DirectAuthenticator | None = None,
) -> dict[str, Any]:
    """Validate the named official5 direct collector's lossless view."""

    receipt_path = collection_path.resolve(strict=True)
    if receipt_path.is_dir():
        receipt_path = (
            receipt_path / "collection_receipt.json"
        ).resolve(strict=True)
    if receipt_path.name != "collection_receipt.json":
        raise PostSuccessContractError(
            "official5 direct adapter requires collection_receipt.json"
        )
    receipt = _validate_seal(
        _read_json(receipt_path, "official5 direct collection receipt"),
        OFFICIAL5_DIRECT_COLLECTION_SCHEMA,
        "official5 direct collection receipt",
    )
    if authenticator is None:
        try:
            from tools import (  # noqa: PLC0415
                mft_goal_official5_direct_terminal_collector as direct,
            )
        except ImportError as exc:
            raise PostSuccessContractError(
                "official5 direct terminal collector adapter is unavailable"
            ) from exc
        authenticator = direct.authenticate_collection
    try:
        raw_view = authenticator(receipt_path)
    except Exception as exc:
        raise PostSuccessContractError(
            "official5 direct collection authentication failed"
        ) from exc
    if not isinstance(raw_view, Mapping):
        raise PostSuccessContractError(
            "official5 direct authenticated view is malformed"
        )
    view = copy.deepcopy(dict(raw_view))
    required_view_fields = {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    }
    if (
        set(view) != required_view_fields
        or view.get("schema_version")
        != OFFICIAL5_DIRECT_AUTHENTICATED_COLLECTION_SCHEMA
        or not all(
            isinstance(view.get(name), Mapping)
            for name in (
                "collection",
                "plan",
                "params",
                "selected",
                "submission",
            )
        )
    ):
        raise PostSuccessContractError(
            "official5 direct authenticated view contract drifted"
        )
    collection = _validate_seal(
        view["collection"],
        OFFICIAL5_DIRECT_AUTHENTICATED_COLLECTION_SCHEMA,
        "official5 direct authenticated collection",
    )
    view["collection"] = collection
    plan = view["plan"]
    params = view["params"]
    selected = view["selected"]
    submission = view["submission"]
    placement = plan.get("placement")
    task_identity = selected.get("task_identity")
    row_contract = selected.get("row_contract")
    result = collection.get("result")
    fixed_params = {
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "thermal_symmetry": "eighth",
        "full_model": 0,
    }
    if (
        not _classification_valid(collection)
        or any(
            collection.get(key) is not expected
            for key, expected in {
                "scheduler_get_only_collection": True,
                "scheduler_mutation_performed": False,
                "scientific_pass_claimed": False,
                "production_claimed": False,
                "strict_al_adapter_authorized": True,
                "direct_analyze_authentication_passed": True,
                "solver_core_authentication_passed": True,
                "thermal_truth_authentication_passed": True,
            }.items()
        )
        or collection.get("task_id") != OFFICIAL5_DIRECT_TASK_ID
        or collection.get("candidate_physics_sha256")
        != OFFICIAL5_CANDIDATE_SHA256
        or collection.get("same_node_as_task_id")
        != OFFICIAL5_SOURCE_TASK_ID
        or collection.get("same_node_as_allocation_id")
        != OFFICIAL5_SOURCE_ALLOCATION_ID
        or str(collection.get("slurm_job_id") or "")
        != OFFICIAL5_SOURCE_SLURM_JOB_ID
        or not isinstance(
            collection.get("scientific_gate_evidence"), Mapping
        )
        or not isinstance(result, Mapping)
        or collection.get("result_sha256")
        != collector.payload_sha256(result)
        or plan.get("candidate_physics_sha256")
        != OFFICIAL5_CANDIDATE_SHA256
        or plan.get("standard_only") is not True
        or plan.get("symmetric_model") is not True
        or plan.get("full_model") is not False
        or plan.get("thermal_symmetry") != "eighth"
        or plan.get("fixed_physics_unchanged") is not True
        or not isinstance(placement, Mapping)
        or placement.get("same_node_as_task_id")
        != OFFICIAL5_SOURCE_TASK_ID
        or placement.get("same_node_as_allocation_id")
        != OFFICIAL5_SOURCE_ALLOCATION_ID
        or str(placement.get("same_node_as_slurm_job_id") or "")
        != OFFICIAL5_SOURCE_SLURM_JOB_ID
        or submission.get("task_id") != OFFICIAL5_DIRECT_TASK_ID
        or submission.get("task_name") != collection.get("task_name")
        or submission.get("dedupe_key") != collection.get("dedupe_key")
        or submission.get("candidate_physics_sha256")
        != OFFICIAL5_CANDIDATE_SHA256
        or submission.get("same_node_as_task_id")
        != OFFICIAL5_SOURCE_TASK_ID
        or any(params.get(key) != value for key, value in fixed_params.items())
        or not isinstance(task_identity, Mapping)
        or isinstance(task_identity.get("seed"), bool)
        or not isinstance(task_identity.get("seed"), int)
        or int(task_identity["seed"]) <= 0
        or task_identity.get("fixed_primary_turns")
        != params.get("N1_main")
        or not isinstance(row_contract, Mapping)
        or row_contract.get("fea_params_sha256")
        != collector.payload_sha256(params)
    ):
        raise PostSuccessContractError(
            "official5 direct physics/placement adapter drifted"
        )
    record_names = (
        "source_collection_receipt",
        "source_collection_seal",
        "result_json",
        "retained_symmetric_aedt",
    )
    resolved_records: dict[str, Path] = {}
    for name in record_names:
        record = collection.get(name)
        if not isinstance(record, Mapping):
            raise PostSuccessContractError(
                f"official5 direct {name} record is absent"
            )
        try:
            target = Path(str(record.get("path") or "")).resolve(
                strict=True
            )
        except OSError as exc:
            raise PostSuccessContractError(
                f"official5 direct {name} file is unavailable"
            ) from exc
        if _file_record(target) != record:
            raise PostSuccessContractError(
                f"official5 direct {name} bytes drifted"
            )
        resolved_records[name] = target
    if resolved_records["source_collection_receipt"] != receipt_path:
        raise PostSuccessContractError(
            "official5 direct receipt path binding drifted"
        )
    seal = _validate_seal(
        _read_json(
            resolved_records["source_collection_seal"],
            "official5 direct collection seal",
        ),
        OFFICIAL5_DIRECT_COLLECTION_SEAL_SCHEMA,
        "official5 direct collection seal",
    )
    collected_result = _read_json(
        resolved_records["result_json"],
        "official5 direct collected result",
    )
    if (
        collection.get("source_collection_receipt_payload_sha256")
        != receipt["payload_sha256"]
        or collection.get("source_collection_seal_payload_sha256")
        != seal["payload_sha256"]
        or collected_result != result
    ):
        raise PostSuccessContractError(
            "official5 direct collection source binding drifted"
        )
    return view


def authenticate_collection(collection_path: Path) -> dict[str, Any]:
    """Return the stable custom-schema adapter consumed by strict AL."""

    source = collection_path.resolve(strict=True)
    receipt_path = (
        (source / "collection_receipt.json").resolve(strict=True)
        if source.is_dir()
        else source
    )
    raw = _read_json(receipt_path, "collection receipt")
    if (
        raw.get("schema_version")
        == OFFICIAL5_DIRECT_COLLECTION_SCHEMA
    ):
        return _authenticate_official5_direct_collection(receipt_path)
    (
        receipt,
        seal,
        contract,
        params,
        selected,
        result,
        paths,
    ) = _load_collection_envelope(collection_path)
    try:
        reasons = production._goal_result_reasons(result, selected)  # noqa: SLF001
        fixed = attest_fixed_identity(
            result,
            require_thermal_pad_metadata=True,
        )
    except Exception as exc:
        raise PostSuccessContractError(
            "actual result hard-contract authentication failed"
        ) from exc
    adapter_collection = _sealed(
        {
            "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
            **SAFETY_FLAGS,
            "stage": "standard",
            "scheduler_status": "completed",
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "task_id": receipt["task_id"],
            "task_name": receipt["task_name"],
            "dedupe_key": receipt["dedupe_key"],
            "candidate_physics_sha256": receipt[
                "candidate_physics_sha256"
            ],
            "source_collection_receipt": _file_record(paths["receipt_path"]),
            "source_collection_receipt_payload_sha256": receipt[
                "payload_sha256"
            ],
            "source_collection_seal": _file_record(paths["seal_path"]),
            "source_collection_seal_payload_sha256": seal["payload_sha256"],
            "retained_symmetric_aedt": _file_record(paths["artifact_path"]),
            "result": copy.deepcopy(result),
            "result_sha256": collector.payload_sha256(result),
            "result_json": _file_record(paths["result_path"]),
            "goal_physical_spec_reasons": list(reasons),
            "goal_physical_spec_passed": not reasons,
            "fixed_identity_attestation": fixed,
            "strict_al_adapter_authorized": True,
        }
    )
    return {
        "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
        "collection": adapter_collection,
        "plan": copy.deepcopy(contract["plan"]),
        "params": copy.deepcopy(params),
        "selected": copy.deepcopy(selected),
        "submission": copy.deepcopy(contract["submission"]),
    }


def _measured_classification(view: Mapping[str, Any]) -> dict[str, Any]:
    collection = view["collection"]
    result = collection["result"]
    try:
        volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
        width, length, height = (
            _finite(value, "actual exterior dimension") for value in dimensions
        )
        resonance = _finite(
            result.get("f_res_min_tx_rx_only_Hz"),
            "actual minimum self resonance",
        )
        active, temperatures, temperature_gate = (
            production._temperature_gate_evidence(result)  # noqa: SLF001
        )
        fixed = attest_fixed_identity(
            result,
            require_thermal_pad_metadata=True,
        )
    except Exception as exc:
        raise PostSuccessContractError(
            "measured hard-constraint evidence is incomplete"
        ) from exc
    family_values: dict[str, list[float]] = {"winding": [], "core": []}
    for target in active:
        family = TEMPERATURE_TARGET_FAMILIES[target]
        family_values[family].append(
            _finite(temperatures[target]["actual_C"], f"actual {target}")
        )
    if not family_values["winding"] or not family_values["core"]:
        raise PostSuccessContractError(
            "measured temperature family evidence is incomplete"
        )
    winding_max = max(family_values["winding"])
    core_max = max(family_values["core"])
    limits = {
        "width_mm": float(GOAL_SIZE_LIMITS_MM["W"]),
        "length_mm": float(GOAL_SIZE_LIMITS_MM["L"]),
        "height_mm": float(GOAL_SIZE_LIMITS_MM["H"]),
        "resonance_Hz": float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
        "winding_max_C": float(TEMPERATURE_FAMILY_LIMITS_C["winding"]),
        "core_max_C": float(TEMPERATURE_FAMILY_LIMITS_C["core"]),
    }
    actuals = {
        "width_mm": width,
        "length_mm": length,
        "height_mm": height,
        "resonance_Hz": resonance,
        "winding_max_C": winding_max,
        "core_max_C": core_max,
    }
    upper_bound = {
        "width_mm",
        "length_mm",
        "height_mm",
        "winding_max_C",
        "core_max_C",
    }
    evidence = {}
    for name, actual in actuals.items():
        if name in upper_bound:
            margin = limits[name] - actual
            passed = actual <= limits[name]
            relation = "<="
        else:
            margin = actual - limits[name]
            passed = actual >= limits[name]
            relation = ">="
        evidence[name] = {
            "actual": actual,
            "limit": limits[name],
            "relation": relation,
            "margin": margin,
            "passed": passed,
        }
    if temperature_gate is not (
        evidence["winding_max_C"]["passed"]
        and evidence["core_max_C"]["passed"]
        and all(item["passed"] for item in temperatures.values())
    ):
        raise PostSuccessContractError(
            "temperature family and target gates disagree"
        )
    losses = {
        name: _finite(result.get(name), f"actual {name}")
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    if any(value < 0 for value in losses.values()):
        raise PostSuccessContractError("actual loss component is negative")
    return {
        "task_id": collection["task_id"],
        "candidate_physics_sha256": collection[
            "candidate_physics_sha256"
        ],
        "source_seed": int(view["selected"]["task_identity"]["seed"]),
        "source_fixed_primary_turns": int(
            view["selected"]["task_identity"]["fixed_primary_turns"]
        ),
        "solver_revision": view["plan"]["solver_revision"],
        "library_revision": view["plan"]["library_revision"],
        "physics_data_revision": str(
            result.get("physics_data_revision") or ""
        ),
        "actual_volume_L": float(volume_l),
        "actual_total_loss_W": sum(losses.values()),
        "actual_loss_components_W": losses,
        "actual_dimensions_mm": {"W": width, "L": length, "H": height},
        "actual_resonance_Hz": resonance,
        "actual_winding_max_C": winding_max,
        "actual_core_max_C": core_max,
        "active_temperature_targets": list(active),
        "actual_temperature_targets": copy.deepcopy(temperatures),
        "fixed_identity_attestation": fixed,
        "hard_constraint_evidence": evidence,
        "measured_hard_constraints_passed": all(
            item["passed"] for item in evidence.values()
        ),
        "campaign_full_physical_spec_reasons": list(
            collection["goal_physical_spec_reasons"]
        ),
        "campaign_full_physical_spec_passed": collection[
            "goal_physical_spec_passed"
        ],
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    payload = collector.canonical_bytes(value) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise PostSuccessContractError(
                f"immutable output bytes differ: {target}"
            )
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return target


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(collector.canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return target


def _observation_path(output_root: Path, task_id: int) -> Path:
    return output_root / "observations" / f"task{task_id}.json"


def _load_observation(path: Path) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path, "measured observation"),
        OBSERVATION_SCHEMA,
        "measured observation",
    )
    for name in (
        "source_collection_receipt",
        "source_collection_seal",
        "source_result_json",
    ):
        record = value.get(name)
        if not isinstance(record, Mapping):
            raise PostSuccessContractError(
                f"measured observation {name} is absent"
            )
        target = Path(str(record.get("path") or ""))
        if _file_record(target) != record:
            raise PostSuccessContractError(
                f"measured observation {name} bytes drifted"
            )
    if (
        value.get("scientific_pass_claimed") is not False
        or value.get("production_claimed") is not False
        or not _classification_valid(value)
    ):
        raise PostSuccessContractError(
            "measured observation safety flags drifted"
        )
    return value


def _create_or_load_observation(
    *,
    collection_path: Path,
    output_root: Path,
    expected_task_id: int,
) -> dict[str, Any]:
    path = _observation_path(output_root, expected_task_id)
    if path.exists():
        value = _load_observation(path)
        if value.get("task_id") != expected_task_id:
            raise PostSuccessContractError(
                "existing observation task identity drifted"
            )
        return value
    view = authenticate_collection(collection_path)
    measured = _measured_classification(view)
    if measured["task_id"] != expected_task_id:
        raise PostSuccessContractError(
            "collection task differs from expected lane"
        )
    collection = view["collection"]
    observation = _sealed(
        {
            "schema_version": OBSERVATION_SCHEMA,
            "created_at_utc": _now(),
            **SAFETY_FLAGS,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "measured_constraint_classification_only": True,
            "source_collection_receipt": _file_record(collection_path),
            "source_collection_receipt_payload_sha256": collection[
                "source_collection_receipt_payload_sha256"
            ],
            "source_collection_seal": copy.deepcopy(
                collection["source_collection_seal"]
            ),
            "source_result_json": copy.deepcopy(collection["result_json"]),
            "source_authenticated_adapter_payload_sha256": collection[
                "payload_sha256"
            ],
            "retained_symmetric_aedt": copy.deepcopy(
                collection["retained_symmetric_aedt"]
            ),
            **measured,
        }
    )
    _write_immutable_json(path, observation)
    return observation


def _aggregate_artifact_path(
    manifest_path: Path,
    record: Mapping[str, Any],
    label: str,
) -> Path:
    raw = Path(str(record.get("path") or ""))
    target = (manifest_path.parent / raw).resolve(strict=True)
    try:
        target.relative_to(manifest_path.parent.resolve(strict=True))
    except ValueError as exc:
        raise PostSuccessContractError(
            f"aggregate {label} artifact escapes its root"
        ) from exc
    return target


def authenticate_aggregate_reference(path: Path) -> dict[str, Any]:
    """Authenticate aggregate manifest and published tables, without re-NDS."""

    manifest_path = path.resolve(strict=True)
    try:
        aggregate = launch._validate_seal(  # noqa: SLF001
            production._read_json(manifest_path),
            schema=launch.GLOBAL_PARETO_SCHEMA,
        )
    except Exception as exc:
        raise PostSuccessContractError(
            "512-seed aggregate manifest seal drifted"
        ) from exc
    artifacts = aggregate.get("artifacts")
    if (
        aggregate.get("campaign_id") != CAMPAIGN_ID
        or aggregate.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or aggregate.get("hard_spec") != GOAL_STAGE_SPEC
        or aggregate.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or aggregate.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or aggregate.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or aggregate.get("seed_local_pareto_merge_used") is not False
        or aggregate.get("production_eligible") is not False
        or aggregate.get("automatic_promotion_allowed") is not False
        or int(aggregate.get("seed_count", 0)) < 512
        or not isinstance(artifacts, Mapping)
    ):
        raise PostSuccessContractError(
            "512-seed aggregate goal authority drifted"
        )
    artifact_records = {}
    for name in (
        "global_terminal_candidates",
        "global_pareto_front",
        "global_objective_front",
        "standard_candidates",
    ):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise PostSuccessContractError(
                f"aggregate artifact is absent: {name}"
            )
        artifact_path = _aggregate_artifact_path(
            manifest_path, record, name
        )
        if (
            _sha256_file(artifact_path) != record.get("sha256")
            or len(pd.read_csv(artifact_path))
            != int(record.get("row_count", -1))
        ):
            raise PostSuccessContractError(
                f"aggregate artifact bytes/rows drifted: {name}"
            )
        artifact_records[name] = _file_record(artifact_path)
        artifact_records[name]["row_count"] = int(record["row_count"])
    if (
        int(aggregate.get("physical_feasible_count", -1))
        < artifact_records["global_pareto_front"]["row_count"]
        or int(aggregate.get("global_pareto_count", -1))
        != artifact_records["global_pareto_front"]["row_count"]
        or int(aggregate.get("global_objective_front_count", -1))
        != artifact_records["global_objective_front"]["row_count"]
    ):
        raise PostSuccessContractError("aggregate published counts drifted")
    return {
        "manifest": _file_record(manifest_path),
        "manifest_payload_sha256": aggregate["payload_sha256"],
        "seed_count": int(aggregate["seed_count"]),
        "input_terminal_row_count": int(
            aggregate["input_terminal_row_count"]
        ),
        "deduplicated_physical_geometry_count": int(
            aggregate["deduplicated_physical_geometry_count"]
        ),
        "physical_feasible_count": int(
            aggregate["physical_feasible_count"]
        ),
        "global_pareto_count": int(aggregate["global_pareto_count"]),
        "global_objective_front_count": int(
            aggregate["global_objective_front_count"]
        ),
        "sorting_authority": aggregate["sorting_authority"],
        "artifacts": artifact_records,
        "manifest_and_published_artifact_bytes_authenticated": True,
        "all_512_seed_inputs_reauthenticated_by_this_cycle": False,
        "global_nds_recomputed_by_this_cycle": False,
    }


def _nondominated_ranks(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Return exact strict-Pareto ranks for measured volume/loss."""

    remaining = set(range(len(rows)))
    output: dict[str, int] = {}
    rank = 0
    while remaining:
        front = []
        for index in sorted(remaining):
            candidate = rows[index]
            objective = (
                _finite(candidate["actual_volume_L"], "actual volume"),
                _finite(candidate["actual_total_loss_W"], "actual loss"),
            )
            dominated = False
            for other_index in remaining:
                if other_index == index:
                    continue
                other = rows[other_index]
                other_objective = (
                    _finite(other["actual_volume_L"], "actual volume"),
                    _finite(other["actual_total_loss_W"], "actual loss"),
                )
                if (
                    other_objective[0] <= objective[0]
                    and other_objective[1] <= objective[1]
                    and (
                        other_objective[0] < objective[0]
                        or other_objective[1] < objective[1]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                front.append(index)
        if not front:
            raise PostSuccessContractError(
                "measured non-dominated sorting stalled"
            )
        for index in front:
            candidate = str(rows[index]["candidate_physics_sha256"])
            if candidate in output:
                raise PostSuccessContractError(
                    "measured candidate identity is duplicated"
                )
            output[candidate] = rank
        remaining.difference_update(front)
        rank += 1
    return output


def _strict_readiness(
    collection_paths: Sequence[Path],
    *,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
) -> tuple[dict[str, Any], dict[int, bool]]:
    from tools import mft_goal_strict_al_ingest as strict_al

    eligibility: dict[int, bool] = {}
    try:
        prepared = strict_al.prepare_ingest(
            base_dataset=base_dataset,
            expected_base_sha256=expected_base_sha256,
            expected_base_rows=expected_base_rows,
            collection_paths=collection_paths,
        )
    except strict_al.StrictALIngestError as exc:
        profile = strict_al._profile_content()  # noqa: SLF001
        try:
            _base, base_record, revision, _counts = strict_al._base_audit(  # noqa: SLF001
                base_dataset,
                expected_sha256=expected_base_sha256,
                expected_rows=expected_base_rows,
                profile=profile,
            )
        except strict_al.StrictALIngestError as base_exc:
            return (
                {
                    "status": "ineligible_fail_closed",
                    "reason": str(base_exc),
                    "strict_al_adapter_installed": True,
                    "base_dataset_authenticated": False,
                    "validated_row_count": 0,
                    "retraining_admission": {
                        "allowed": False,
                        "reasons": ["base_dataset_authentication_failed"],
                    },
                    "dataset_write_performed": False,
                    "surrogate_retraining_performed": False,
                },
                eligibility,
            )
        row_errors = {}
        facts = []
        for path in collection_paths:
            try:
                truth = strict_al.authenticate_collection(path)
                _row, fact = strict_al._validate_truth_row(  # noqa: SLF001
                    truth,
                    profile=profile,
                    physics_data_revision=revision,
                )
                task_id = int(fact["task_id"])
                eligibility[task_id] = True
                facts.append(fact)
            except strict_al.StrictALIngestError as row_exc:
                try:
                    raw = _read_json(path, "strict AL source collection")
                    task_id = int(raw.get("task_id", -1))
                except Exception:
                    task_id = -1
                eligibility[task_id] = False
                row_errors[str(task_id)] = str(row_exc)
        admission = strict_al._admission(  # noqa: SLF001
            facts,
            minimum_useful_rows=strict_al.DEFAULT_MINIMUM_USEFUL_ROWS,
            minimum_source_tasks=strict_al.DEFAULT_MINIMUM_SOURCE_TASKS,
        )
        reasons = list(admission["reasons"])
        if row_errors:
            reasons.append("one_or_more_rows_failed_strict_validation")
        admission["allowed"] = False
        admission["reasons"] = reasons
        return (
            {
                "status": "ineligible_fail_closed",
                "reason": str(exc),
                "strict_al_adapter_installed": True,
                "base_dataset_authenticated": True,
                "base_dataset": base_record,
                "validated_row_count": len(facts),
                "row_errors": row_errors,
                "retraining_admission": admission,
                "dataset_write_performed": False,
                "surrogate_retraining_performed": False,
            },
            eligibility,
        )
    for fact in prepared.collection_records:
        eligibility[int(fact["task_id"])] = True
    summary = strict_al._summary(prepared)  # noqa: SLF001
    summary.update(
        {
            "status": (
                "eligible"
                if prepared.retraining_admission["allowed"]
                else "rows_valid_admission_closed"
            ),
            "strict_al_adapter_installed": True,
            "base_dataset_authenticated": True,
            "validated_row_count": len(prepared.collection_records),
            "dataset_write_performed": False,
            "surrogate_retraining_performed": False,
        }
    )
    return summary, eligibility


def _snapshot_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    strict_eligibility: Mapping[int, bool],
) -> list[dict[str, Any]]:
    all_ranks = _nondominated_ranks(observations)
    feasible = [
        row
        for row in observations
        if row["measured_hard_constraints_passed"] is True
    ]
    feasible_ranks = _nondominated_ranks(feasible)
    rows = []
    for observation in observations:
        candidate = str(observation["candidate_physics_sha256"])
        rows.append(
            {
                "task_id": int(observation["task_id"]),
                "candidate_physics_sha256": candidate,
                "actual_volume_L": float(observation["actual_volume_L"]),
                "actual_total_loss_W": float(
                    observation["actual_total_loss_W"]
                ),
                "actual_width_mm": float(
                    observation["actual_dimensions_mm"]["W"]
                ),
                "actual_length_mm": float(
                    observation["actual_dimensions_mm"]["L"]
                ),
                "actual_height_mm": float(
                    observation["actual_dimensions_mm"]["H"]
                ),
                "actual_resonance_Hz": float(
                    observation["actual_resonance_Hz"]
                ),
                "actual_winding_max_C": float(
                    observation["actual_winding_max_C"]
                ),
                "actual_core_max_C": float(
                    observation["actual_core_max_C"]
                ),
                "measured_hard_constraints_passed": bool(
                    observation["measured_hard_constraints_passed"]
                ),
                "strict_al_row_eligible": bool(
                    strict_eligibility.get(int(observation["task_id"]), False)
                ),
                "audit_non_dominated_rank": all_ranks[candidate],
                "hard_feasible_non_dominated_rank": feasible_ranks.get(
                    candidate, -1
                ),
                "source_seed": int(observation["source_seed"]),
                "source_fixed_primary_turns": int(
                    observation["source_fixed_primary_turns"]
                ),
                "solver_revision": observation["solver_revision"],
                "library_revision": observation["library_revision"],
                "physics_data_revision": observation[
                    "physics_data_revision"
                ],
            }
        )
    return sorted(rows, key=lambda row: (row["audit_non_dominated_rank"], row["task_id"]))


def _load_snapshot(path: Path, *, expected_snapshot_id: str) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path, "post-success snapshot"),
        SNAPSHOT_SCHEMA,
        "post-success snapshot",
    )
    if (
        value.get("snapshot_id") != expected_snapshot_id
        or not _classification_valid(value)
        or value.get("scientific_pass_claimed") is not False
        or value.get("production_claimed") is not False
        or value.get("production_pareto_emitted") is not False
        or value.get("measured_actual_nds_performed") is not True
        or value.get("scheduler_mutation_performed") is not False
        or value.get("orchestrator_scheduler_methods_used") != []
    ):
        raise PostSuccessContractError(
            "existing post-success snapshot safety contract drifted"
        )
    for name, extras in (
        (
            "measured_actual_observations_csv",
            {"row_count", "columns"},
        ),
        (
            "diagnostic_hard_feasible_actual_front_csv",
            {"row_count", "columns"},
        ),
        ("global_rerank_input", {"payload_sha256"}),
    ):
        record = value.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {"path", "sha256", "size_bytes", *extras}
        ):
            raise PostSuccessContractError(
                f"existing snapshot {name} record drifted"
            )
        file_record = {
            key: record[key] for key in ("path", "sha256", "size_bytes")
        }
        if _file_record(Path(str(record["path"]))) != file_record:
            raise PostSuccessContractError(
                f"existing snapshot {name} bytes drifted"
            )
    rerank_record = value["global_rerank_input"]
    rerank = _validate_seal(
        _read_json(Path(rerank_record["path"]), "global rerank input"),
        RERANK_INPUT_SCHEMA,
        "global rerank input",
    )
    if (
        rerank.get("payload_sha256") != rerank_record["payload_sha256"]
        or rerank.get("scientific_pass_claimed") is not False
        or rerank.get("production_claimed") is not False
        or rerank.get("production_pareto_emitted") is not False
        or rerank.get("scheduler_mutation_performed") is not False
        or rerank.get("surrogate_and_measured_rows_directly_unioned")
        is not False
    ):
        raise PostSuccessContractError(
            "existing global rerank input safety contract drifted"
        )
    return value


def _build_snapshot(
    *,
    observations: Sequence[Mapping[str, Any]],
    collection_paths: Sequence[Path],
    output_root: Path,
    expected_lane_count: int,
    aggregate_manifest: Path,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
) -> dict[str, Any]:
    aggregate = authenticate_aggregate_reference(aggregate_manifest)
    identity = {
        "observation_payload_sha256": sorted(
            str(item["payload_sha256"]) for item in observations
        ),
        "aggregate_manifest_payload_sha256": aggregate[
            "manifest_payload_sha256"
        ],
    }
    snapshot_id = collector.payload_sha256(identity)[:16]
    destination = output_root / "snapshots" / snapshot_id
    manifest_path = destination / "snapshot_manifest.json"
    if manifest_path.exists():
        return _load_snapshot(
            manifest_path,
            expected_snapshot_id=snapshot_id,
        )
    strict_summary, eligibility = _strict_readiness(
        collection_paths,
        base_dataset=base_dataset,
        expected_base_sha256=expected_base_sha256,
        expected_base_rows=expected_base_rows,
    )
    rows = _snapshot_rows(
        observations,
        strict_eligibility=eligibility,
    )
    staging = Path(
        tempfile.mkdtemp(
            dir=(output_root / "snapshots").resolve(),
            prefix=f".{snapshot_id}.",
            suffix=".tmp",
        )
    )
    try:
        observations_csv = staging / "measured_actual_observations.csv"
        pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).to_csv(
            observations_csv,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        feasible_rows = [
            row
            for row in rows
            if row["measured_hard_constraints_passed"]
            and row["hard_feasible_non_dominated_rank"] == 0
        ]
        feasible_csv = staging / "diagnostic_hard_feasible_actual_front.csv"
        pd.DataFrame(feasible_rows, columns=OBSERVATION_COLUMNS).to_csv(
            feasible_csv,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        all_expected = len(observations) == expected_lane_count
        rerank = _sealed(
            {
                "schema_version": RERANK_INPUT_SCHEMA,
                "created_at_utc": _now(),
                **SAFETY_FLAGS,
                "scientific_pass_claimed": False,
                "production_claimed": False,
                "production_pareto_emitted": False,
                "source_512_seed_surrogate_aggregate": aggregate,
                "measured_standard_observations": [
                    {
                        "observation": _file_record(
                            _observation_path(
                                output_root, int(item["task_id"])
                            )
                        ),
                        "observation_payload_sha256": item[
                            "payload_sha256"
                        ],
                        "candidate_physics_sha256": item[
                            "candidate_physics_sha256"
                        ],
                        "task_id": item["task_id"],
                        "measured_hard_constraints_passed": item[
                            "measured_hard_constraints_passed"
                        ],
                        "actual_volume_L": item["actual_volume_L"],
                        "actual_total_loss_W": item[
                            "actual_total_loss_W"
                        ],
                    }
                    for item in observations
                ],
                "strict_al_readiness": strict_summary,
                "all_expected_postdeadline_lanes_collected": all_expected,
                "measured_actual_nds_performed": True,
                "measured_actual_nds_scope": (
                    "all_available_authenticated_postdeadline_standard_"
                    "observations"
                ),
                "surrogate_and_measured_rows_directly_unioned": False,
                "direct_mixed_authority_sort_allowed": False,
                "handoff_to_next_surrogate_generation_allowed": bool(
                    strict_summary["retraining_admission"]["allowed"]
                ),
                "new_512_seed_nsga2_required_after_retraining": True,
                "scheduler_methods_used": [],
                "scheduler_mutation_performed": False,
                "dataset_write_performed": False,
                "surrogate_retraining_performed": False,
            }
        )
        rerank_path = staging / "global_rerank_input.json"
        rerank_path.write_bytes(collector.canonical_bytes(rerank) + b"\n")
        revisions = sorted({row["solver_revision"] for row in rows})
        physics_revisions = sorted(
            {row["physics_data_revision"] for row in rows}
        )
        manifest = _sealed(
            {
                "schema_version": SNAPSHOT_SCHEMA,
                "created_at_utc": _now(),
                **SAFETY_FLAGS,
                "scientific_pass_claimed": False,
                "production_claimed": False,
                "production_pareto_emitted": False,
                "snapshot_id": snapshot_id,
                "expected_lane_count": expected_lane_count,
                "authenticated_observation_count": len(observations),
                "all_expected_postdeadline_lanes_collected": all_expected,
                "measured_hard_feasible_count": sum(
                    row["measured_hard_constraints_passed"] for row in rows
                ),
                "diagnostic_actual_rank0_count": len(feasible_rows),
                "measured_actual_nds_performed": True,
                "measured_actual_nds_complete_for_expected_lanes": all_expected,
                "solver_revisions": revisions,
                "physics_data_revisions": physics_revisions,
                "mixed_solver_revisions": len(revisions) > 1,
                "mixed_physics_data_revisions": len(physics_revisions) > 1,
                "ranked_rows": rows,
                "strict_al_readiness": strict_summary,
                "source_aggregate_reference": aggregate,
                "measured_actual_observations_csv": {
                    **_future_file_record(
                        observations_csv,
                        destination / observations_csv.name,
                    ),
                    "row_count": len(rows),
                    "columns": list(OBSERVATION_COLUMNS),
                },
                "diagnostic_hard_feasible_actual_front_csv": {
                    **_future_file_record(
                        feasible_csv,
                        destination / feasible_csv.name,
                    ),
                    "row_count": len(feasible_rows),
                    "columns": list(OBSERVATION_COLUMNS),
                },
                "global_rerank_input": {
                    **_future_file_record(
                        rerank_path,
                        destination / rerank_path.name,
                    ),
                    "payload_sha256": rerank["payload_sha256"],
                },
                "sorting_authority": (
                    "exact_two_objective_nondominated_sort_over_all_available_"
                    "authenticated_postdeadline_standard_actual_observations"
                ),
                "source_collectors_scheduler_methods_used": ["GET"],
                "orchestrator_scheduler_methods_used": [],
                "scheduler_mutation_performed": False,
            }
        )
        (staging / "snapshot_manifest.json").write_bytes(
            collector.canonical_bytes(manifest) + b"\n"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _lane_state(lane: Lane) -> dict[str, Any]:
    root = lane.collection_root.resolve()
    receipt = root / "collection_receipt.json"
    seal = root / "collection_seal.json"
    failure = root.parent / f"{root.name}.failure_ledger.json"
    if receipt.exists() or seal.exists():
        if not receipt.is_file() or not seal.is_file():
            raise PostSuccessContractError(
                f"task {lane.task_id} collection is only partially published"
            )
        raw = _read_json(seal, "collection seal")
        if raw.get("task_id") != lane.task_id:
            raise PostSuccessContractError(
                f"task {lane.task_id} collection identity differs"
            )
        return {
            "task_id": lane.task_id,
            "selection_effective": lane.selection_effective,
            "status": "collection_ready",
            "collection_receipt": str(receipt.resolve(strict=True)),
            "collection_seal": str(seal.resolve(strict=True)),
        }
    if failure.exists():
        value = _read_json(failure, "collection failure ledger")
        if value.get("task_id") != lane.task_id:
            raise PostSuccessContractError(
                f"task {lane.task_id} failure ledger identity differs"
            )
        return {
            "task_id": lane.task_id,
            "selection_effective": lane.selection_effective,
            "status": "terminal_failure",
            "failure_ledger": _file_record(failure),
        }
    return {
        "task_id": lane.task_id,
        "selection_effective": lane.selection_effective,
        "status": "pending",
        "collection_root": str(root),
    }


def process_cycle(
    *,
    lanes: Sequence[Lane],
    output_root: Path,
    aggregate_manifest: Path = DEFAULT_AGGREGATE_MANIFEST,
    base_dataset: Path = DEFAULT_BASE_DATASET,
    expected_base_sha256: str = DEFAULT_BASE_SHA256,
    expected_base_rows: int = DEFAULT_BASE_ROWS,
) -> dict[str, Any]:
    if not lanes:
        raise PostSuccessContractError("at least one Standard lane is required")
    task_ids = [lane.task_id for lane in lanes]
    roots = [str(lane.collection_root.resolve()) for lane in lanes]
    if (
        len(set(task_ids)) != len(task_ids)
        or len(set(roots)) != len(roots)
        or any(task_id <= 0 for task_id in task_ids)
        or not any(lane.selection_effective for lane in lanes)
        or any(
            not isinstance(lane.selection_effective, bool)
            for lane in lanes
        )
    ):
        raise PostSuccessContractError(
            "post-success lane identities are duplicated or invalid"
        )
    root = output_root.resolve()
    (root / "observations").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    lane_states = [_lane_state(lane) for lane in lanes]
    observations = []
    collection_paths = []
    lifecycle_observation_count = 0
    for lane, state in zip(lanes, lane_states, strict=True):
        if state["status"] != "collection_ready":
            continue
        collection_path = Path(state["collection_receipt"])
        observation = _create_or_load_observation(
            collection_path=collection_path,
            output_root=root,
            expected_task_id=lane.task_id,
        )
        state["observation"] = _file_record(
            _observation_path(root, lane.task_id)
        )
        state["measured_hard_constraints_passed"] = observation[
            "measured_hard_constraints_passed"
        ]
        lifecycle_observation_count += 1
        if lane.selection_effective:
            observations.append(observation)
            collection_paths.append(collection_path)
        else:
            state["selection_exclusion_reason"] = (
                "superseded_lifecycle_lane"
            )
    snapshot = None
    effective_lane_count = sum(
        lane.selection_effective for lane in lanes
    )
    if observations:
        snapshot = _build_snapshot(
            observations=observations,
            collection_paths=collection_paths,
            output_root=root,
            expected_lane_count=effective_lane_count,
            aggregate_manifest=aggregate_manifest,
            base_dataset=base_dataset,
            expected_base_sha256=expected_base_sha256,
            expected_base_rows=expected_base_rows,
        )
    effective_states = [
        item
        for item in lane_states
        if item["selection_effective"] is True
    ]
    pending_count = sum(
        item["status"] == "pending" for item in effective_states
    )
    failure_count = sum(
        item["status"] == "terminal_failure" for item in effective_states
    )
    lifecycle_pending_count = sum(
        item["status"] == "pending" for item in lane_states
    )
    lifecycle_failure_count = sum(
        item["status"] == "terminal_failure" for item in lane_states
    )
    if not observations and pending_count:
        status = "pending_standard_collections"
    elif pending_count:
        status = "partial_success_pending_remaining_collections"
    elif failure_count:
        status = "terminal_with_one_or_more_failed_standard_lanes"
    else:
        status = "all_standard_collections_processed"
    state = _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "observed_at_utc": _now(),
            **SAFETY_FLAGS,
            "status": status,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "production_pareto_emitted": False,
            "expected_lane_count": effective_lane_count,
            "lifecycle_lane_count": len(lanes),
            "effective_task_ids": [
                lane.task_id for lane in lanes if lane.selection_effective
            ],
            "lifecycle_task_ids": task_ids,
            "selection_superseded_task_ids": [
                lane.task_id
                for lane in lanes
                if not lane.selection_effective
            ],
            "collection_count": len(observations),
            "pending_count": pending_count,
            "terminal_failure_count": failure_count,
            "lifecycle_collection_count": lifecycle_observation_count,
            "lifecycle_pending_count": lifecycle_pending_count,
            "lifecycle_terminal_failure_count": lifecycle_failure_count,
            "lanes": lane_states,
            "latest_snapshot": (
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "snapshot_manifest": _file_record(
                        root
                        / "snapshots"
                        / snapshot["snapshot_id"]
                        / "snapshot_manifest.json"
                    ),
                    "authenticated_observation_count": snapshot[
                        "authenticated_observation_count"
                    ],
                    "measured_hard_feasible_count": snapshot[
                        "measured_hard_feasible_count"
                    ],
                    "strict_al_status": snapshot[
                        "strict_al_readiness"
                    ]["status"],
                }
                if snapshot is not None
                else None
            ),
            "pending_means_no_scientific_classification": not observations,
            "source_collectors_scheduler_methods_used": ["GET"],
            "orchestrator_scheduler_methods_used": [],
            "scheduler_mutation_performed": False,
            "dataset_write_performed": False,
            "surrogate_retraining_performed": False,
        }
    )
    _write_atomic_json(root / "state.json", state)
    return state


def run_watch(
    *,
    lanes: Sequence[Lane],
    output_root: Path,
    aggregate_manifest: Path = DEFAULT_AGGREGATE_MANIFEST,
    base_dataset: Path = DEFAULT_BASE_DATASET,
    expected_base_sha256: str = DEFAULT_BASE_SHA256,
    expected_base_rows: int = DEFAULT_BASE_ROWS,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS
    ):
        raise PostSuccessContractError("watch interval is invalid")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    effective_lanes = [lane for lane in lanes if lane.selection_effective]
    if not effective_lanes:
        raise PostSuccessContractError(
            "at least one selection-effective lane is required"
        )
    pid = _sealed(
        {
            "schema_version": PID_SCHEMA,
            "created_at_utc": _now(),
            "pid": os.getpid(),
            "interval_seconds": interval_seconds,
            "task_ids": [lane.task_id for lane in lanes],
            "effective_task_ids": [
                lane.task_id for lane in effective_lanes
            ],
            "selection_superseded_task_ids": [
                lane.task_id
                for lane in lanes
                if not lane.selection_effective
            ],
            "scheduler_methods_used": [],
            "scheduler_mutation_performed": False,
        }
    )
    _write_atomic_json(root / "watcher.pid.json", pid)
    while True:
        try:
            state = process_cycle(
                lanes=lanes,
                output_root=root,
                aggregate_manifest=aggregate_manifest,
                base_dataset=base_dataset,
                expected_base_sha256=expected_base_sha256,
                expected_base_rows=expected_base_rows,
            )
        except Exception as exc:
            state = _sealed(
                {
                    "schema_version": STATE_SCHEMA,
                    "observed_at_utc": _now(),
                    **SAFETY_FLAGS,
                    "status": "blocked_fail_closed",
                    "scientific_pass_claimed": False,
                    "production_claimed": False,
                    "production_pareto_emitted": False,
                    "expected_lane_count": len(effective_lanes),
                    "lifecycle_lane_count": len(lanes),
                    "effective_task_ids": [
                        lane.task_id for lane in effective_lanes
                    ],
                    "lifecycle_task_ids": [
                        lane.task_id for lane in lanes
                    ],
                    "selection_superseded_task_ids": [
                        lane.task_id
                        for lane in lanes
                        if not lane.selection_effective
                    ],
                    "collection_count": 0,
                    "pending_count": len(effective_lanes),
                    "terminal_failure_count": 0,
                    "lifecycle_collection_count": 0,
                    "lifecycle_pending_count": len(lanes),
                    "lifecycle_terminal_failure_count": 0,
                    "lanes": [
                        {
                            "task_id": lane.task_id,
                            "selection_effective": (
                                lane.selection_effective
                            ),
                            "status": "authentication_error",
                            "collection_root": str(
                                lane.collection_root.resolve()
                            ),
                        }
                        for lane in lanes
                    ],
                    "latest_snapshot": None,
                    "pending_means_no_scientific_classification": True,
                    "fail_closed_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "source_collectors_scheduler_methods_used": ["GET"],
                    "orchestrator_scheduler_methods_used": [],
                    "scheduler_mutation_performed": False,
                    "dataset_write_performed": False,
                    "surrogate_retraining_performed": False,
                }
            )
            _write_atomic_json(root / "state.json", state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "collection_count": state["collection_count"],
                    "pending_count": state["pending_count"],
                    "terminal_failure_count": state[
                        "terminal_failure_count"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state["pending_count"] == 0:
            return state
        time.sleep(interval_seconds)


def _parse_lane(value: str) -> Lane:
    raw_task, separator, raw_path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "lane must be TASK_ID=COLLECTION_ROOT"
        )
    try:
        task_id = int(raw_task)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lane task ID is invalid") from exc
    if task_id <= 0 or not raw_path.strip():
        raise argparse.ArgumentTypeError("lane identity is invalid")
    return Lane(task_id=task_id, collection_root=Path(raw_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed post-success classification, strict-AL readiness, "
            "and measured actual NDS for custom Standard collections."
        )
    )
    parser.add_argument(
        "--lane",
        action="append",
        required=True,
        type=_parse_lane,
        help="repeat selection-effective TASK_ID=COLLECTION_ROOT",
    )
    parser.add_argument(
        "--lifecycle-lane",
        action="append",
        default=[],
        type=_parse_lane,
        help=(
            "repeat superseded TASK_ID=COLLECTION_ROOT; displayed and "
            "authenticated but excluded from selection/NDS"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--aggregate-manifest",
        type=Path,
        default=DEFAULT_AGGREGATE_MANIFEST,
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=DEFAULT_BASE_DATASET,
    )
    parser.add_argument(
        "--expected-base-sha256",
        default=DEFAULT_BASE_SHA256,
    )
    parser.add_argument(
        "--expected-base-rows",
        type=int,
        default=DEFAULT_BASE_ROWS,
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lifecycle_lanes = [
        Lane(
            task_id=lane.task_id,
            collection_root=lane.collection_root,
            selection_effective=False,
        )
        for lane in args.lifecycle_lane
    ]
    kwargs = {
        "lanes": [*args.lane, *lifecycle_lanes],
        "output_root": args.output_root,
        "aggregate_manifest": args.aggregate_manifest,
        "base_dataset": args.base_dataset,
        "expected_base_sha256": args.expected_base_sha256,
        "expected_base_rows": args.expected_base_rows,
    }
    state = (
        run_watch(**kwargs, interval_seconds=args.interval)
        if args.watch
        else process_cycle(**kwargs)
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "state": str(args.output_root.resolve() / "state.json"),
                "collection_count": state["collection_count"],
                "pending_count": state["pending_count"],
                "scientific_pass_claimed": False,
                "production_claimed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
