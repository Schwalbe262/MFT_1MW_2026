"""Fail-closed production release canary for the MFT monitor.

This module is intended to run from the *candidate* immutable virtual
environment and source checkout before any 8010 process is replaced.  It
starts the candidate ASGI app on an ephemeral loopback socket, performs only
read-only requests, reads the authenticated strict cohort with Arrow itself,
and emits content-addressed evidence.  A deployment manifest is emitted only
after every gate passes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterator
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
EXPECTED_ARROW_MAJOR = 24
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TIER1_TEMPERATURE_TARGETS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
EXPECTED_TIER1_TEMPERATURE_LIMIT_C = 120.0
EXPECTED_TIER1_RESONANCE_MIN_HZ = 10_000.0
EXPECTED_TIER1_NEAR_PREVIEW_LIMIT = 64
EXPECTED_TIER1_CANDIDATE_PREVIEW_LIMIT = 128
CURRENT7_CANDIDATE_PREVIEW_LIMIT = 256
CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS = 300.0
CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS = 60.0
CURRENT7_INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
CURRENT7_STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
CURRENT7_AGGREGATE_SCHEMA = "mft-tier1-current7-slurm-aggregate-v1"
CURRENT7_COMPATIBILITY_SCHEMA = "mft-tier1-current7-monitor-compatibility-v1"
CURRENT7_CONDITION_INDEX_LIMIT = 16
CURRENT7_CONDITION_STATUS_MAX_BYTES = 64 * 1024 * 1024
BLOCKER_HPO_V2_STATUS_SCHEMA = "mft-blocker-hpo-live-status-v2"
CURRENT7_TEMPERATURE_TARGETS = EXPECTED_TIER1_TEMPERATURE_TARGETS[4:]
CURRENT7_HARD_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "T_limit_C": 110.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": 15_000.0,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_200.0,
    "size_L_max_mm": 1_200.0,
    "size_H_max_mm": 750.0,
}
CURRENT7_HARD_SPEC_SHA256 = hashlib.sha256(
    json.dumps(
        CURRENT7_HARD_SPEC,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
CURRENT7_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{name}" for name in CURRENT7_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
    "minimum_physical_insulation",
    "core_group_manufacturability_limit",
    "half_magnetizing_resonance_minimum",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
CURRENT7_AUTHORITY_FALSE_FLAGS = (
    "production_eligible",
    "fea_submission_approved",
    "fea_submission_performed",
    "aedt_used",
    "automatic_promotion_allowed",
)
CURRENT7_TASK_STATES = frozenset({
    "queued", "attaching", "running", "completed", "failed", "cancelled",
    "timeout",
})
CURRENT7_CONSTRAINT_IDENTITY_FIELDS = (
    "constraint_version",
    "hard_spec",
    "hard_spec_sha256",
    "hard_constraint_contract_sha256",
    "temperature_contract_sha256",
    "constraint_names",
    "temperature_targets",
)
HPO_V2_AUTHORITY_FALSE_FLAGS = (
    "production_eligible",
    "production_model_eligible",
    "fea_submission_approved",
    "promotion_approved",
)
EXPECTED_DUAL_STANDARD_FEA_TASKS = {
    "9375b77f0797c3c6dcd55c87594e6b35c59b6629aa295a807af4d69afed97d1b": {
        "candidate_digest": (
            "d65e4cbbdb1925765c293494c2d4b7efcf490db3e00a2f3b07e45d867cef3df6"
        ),
        "task_id": 53841,
        "task_name": "mft-nsgafea-x-d65e4cbbdb192576",
    },
    "be52ebce726446fc5c36f75339b03405be2a7cd009719cf9690019dd375d4498": {
        "candidate_digest": (
            "bb63894ee450a2d1cf062b2d31ef9b3f1a83982b49b1b650b5ee1669ac711bd0"
        ),
        "task_id": 53840,
        "task_name": "mft-nsgafea-x-bb63894ee450a2d1",
    },
}
EXPECTED_DUAL_STANDARD_FEA_STATUSES = frozenset(
    {"queued", "attaching", "running", "completed", "failed", "cancelled", "timeout"}
)
DUAL_SEALED_FALSE_FLAGS = (
    "canonical_candidate",
    "terminal_result",
    "pareto",
    "production",
    "production_eligible",
    "fea",
    "fea_submission_approved",
    "fea_submission_performed",
)
DUAL_SEALED_CANDIDATE_FALSE_FLAGS = (
    *DUAL_SEALED_FALSE_FLAGS,
    "production_submission_enabled",
    "fea_submission_enabled",
)
DUAL_VALIDATION_ENVIRONMENT = {
    "sealed_successor_handoff": "MFT_T120_SEALED_SUCCESSOR_HANDOFF",
    "sealed_successor_handoff_sha256": "MFT_T120_SEALED_SUCCESSOR_HANDOFF_SHA256",
    "dual_successor_plan": "MFT_T120_DUAL_SUCCESSOR_PLAN",
    "dual_successor_plan_sha256": "MFT_T120_DUAL_SUCCESSOR_PLAN_SHA256",
    "dual_successor_ui": "MFT_T120_DUAL_SUCCESSOR_UI",
    "dual_successor_ui_sha256": "MFT_T120_DUAL_SUCCESSOR_UI_SHA256",
    "dual_successor_receipt": "MFT_T120_DUAL_SUCCESSOR_RECEIPT",
    "dual_successor_receipt_sha256": "MFT_T120_DUAL_SUCCESSOR_RECEIPT_SHA256",
    "dual_fea_runtime_root": "MFT_T120_DUAL_FEA_RUNTIME_ROOT",
    "dual_fea_allowed_root": "MFT_T120_DUAL_FEA_ALLOWED_ROOT",
    "dual_fea_manifest_sha256": "MFT_T120_DUAL_FEA_MANIFEST_SHA256",
    "dual_fea_plan_sha256": "MFT_T120_DUAL_FEA_PLAN_SHA256",
    "dual_fea_source_sha256": "MFT_T120_DUAL_FEA_SOURCE_SHA256",
}
NSGA_STABILITY_GATE_TIMEOUT_SECONDS = 120.0
NSGA_STABILITY_GATE_POLL_SECONDS = 2.0
NSGA_STABILITY_GATE_REQUIRED_READS = 2
PRIMARY_PARQUET_COLUMNS = (
    # Provenance and terminal validity.
    "task_id",
    "saved_at",
    "git_hash",
    "git_dirty",
    "pyaedt_library_git_hash",
    "pyaedt_library_git_dirty",
    "physics_data_revision",
    "result_valid_em",
    "result_valid_thermal",
    # Geometry and hard-spec inputs used by training/NSGA.
    "N1_main",
    "N2_main",
    "n_core_group",
    "l1",
    "l2",
    "h1",
    "w1",
    "wcp_t",
    # Matrix, loss, flux, thermal and capacitance outputs.
    "Llt",
    "Llr",
    "Lmt",
    "Lmr",
    "P_core_total",
    "P_core_plate_total",
    "P_wcp_total",
    "P_winding_total",
    "B_mean_core",
    "B_max_core",
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "T_max_core",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "f_res_tx_self_Hz",
    "f_res_rx_self_Hz",
    "f_res_interwinding_Hz",
)


class CanaryFailure(RuntimeError):
    """A release precondition was not proven."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, max_bytes: int = 256 * 1024 * 1024) -> str:
    stat = path.stat()
    if not path.is_file() or path.is_symlink():
        raise CanaryFailure(f"artifact is not a regular non-symlink file: {path}")
    if stat.st_size <= 0 or stat.st_size > max_bytes:
        raise CanaryFailure(
            f"artifact size is outside the canary limit: {path} ({stat.st_size})"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanaryFailure(f"{name} is not a non-negative integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanaryFailure(f"{name} is not a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CanaryFailure(f"{name} is not a finite number")
    return number


def _exact_sha256(value: Any, name: str) -> str:
    digest = str(value or "").strip().lower()
    if not FULL_SHA256.fullmatch(digest):
        raise CanaryFailure(f"{name} is not an exact SHA-256")
    return digest


def _current7_timestamp(value: Any, name: str) -> tuple[str, datetime]:
    text = str(value or "").strip()
    if not text:
        raise CanaryFailure(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanaryFailure(f"{name} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return text, parsed.astimezone(timezone.utc)


def _current7_constraint_identity(document: dict[str, Any]) -> dict[str, Any]:
    return {name: document.get(name) for name in CURRENT7_CONSTRAINT_IDENTITY_FIELDS}


def _current7_constraint_identity_sha256(
    document: dict[str, Any],
) -> str:
    return _sha256_bytes(_canonical_json(_current7_constraint_identity(document)))


def _current7_temperature_contract(
    temperature_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "mft-tier1-current7-temperature-reference-v1",
        "sha256": _exact_sha256(
            temperature_contract_sha256,
            "current7 temperature contract",
        ),
        "robust_upper_bound_C": CURRENT7_HARD_SPEC["T_limit_C"],
        "target_count": len(CURRENT7_TEMPERATURE_TARGETS),
        "targets": list(CURRENT7_TEMPERATURE_TARGETS),
    }


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as exc:
        raise CanaryFailure(f"cannot inspect configured path: {path}") from exc
    return bool(path.is_symlink() or attributes & 0x400)


def _regular_input_identity(
    raw_path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> dict[str, Any]:
    if not raw_path.is_absolute():
        raise CanaryFailure(f"{label} path must be absolute")
    if _is_reparse_point(raw_path):
        raise CanaryFailure(f"{label} cannot be a symlink or reparse point")
    try:
        path = raw_path.resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure(f"{label} is unavailable: {raw_path}") from exc
    if path != raw_path or not path.is_file():
        raise CanaryFailure(f"{label} does not resolve to its declared regular file")
    expected = _exact_sha256(expected_sha256, f"{label} SHA-256")
    actual = _sha256_file(path, max_bytes)
    if actual != expected:
        raise CanaryFailure(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return {
        "path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "regular_non_reparse_file": True,
    }


def _read_runtime_json_identity(
    raw_path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one mutable runtime JSON atomically enough for identity evidence."""

    if not raw_path.is_absolute():
        raise CanaryFailure(f"{label} path must be absolute")
    if _is_reparse_point(raw_path):
        raise CanaryFailure(f"{label} cannot be a symlink or reparse point")
    try:
        path = raw_path.resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure(f"{label} is unavailable: {raw_path}") from exc
    if path != raw_path or not path.is_file():
        raise CanaryFailure(f"{label} does not resolve to its declared regular file")
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise CanaryFailure(f"{label} cannot be read: {path}") from exc
    if not raw or len(raw) > max_bytes:
        raise CanaryFailure(f"{label} size is outside the canary limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CanaryFailure(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CanaryFailure(f"{label} is not a JSON object")
    return value, {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
        "regular_non_reparse_file": True,
    }


def _authenticate_current7_index(raw_path: Path) -> dict[str, Any]:
    """Authenticate the secondary current7 index and both referenced files."""

    last_error = "current7 index was not read"
    for _ in range(8):
        try:
            index, index_identity = _read_runtime_json_identity(
                raw_path,
                label="current7 secondary index",
                max_bytes=256 * 1024,
            )
            if index.get("schema_version") != CURRENT7_INDEX_SCHEMA:
                raise CanaryFailure("current7 secondary index schema mismatch")
            root_text = str(index.get("path_containment_root") or "")
            if not root_text:
                raise CanaryFailure("current7 containment root is missing")
            root = Path(root_text)
            if not root.is_absolute():
                raise CanaryFailure("current7 containment root is not absolute")
            root = root.resolve(strict=True)
            index_path = Path(index_identity["path"])
            if not root.is_dir() or not index_path.is_relative_to(root):
                raise CanaryFailure("current7 index escapes its containment root")

            references: dict[str, dict[str, Any]] = {}
            for name, schema, maximum in (
                ("status", CURRENT7_STATUS_SCHEMA, 32 * 1024 * 1024),
                (
                    "compatibility",
                    CURRENT7_COMPATIBILITY_SCHEMA,
                    512 * 1024,
                ),
            ):
                declared = index.get(name)
                if not isinstance(declared, dict):
                    raise CanaryFailure(f"current7 {name} identity is missing")
                if declared.get("schema_version") != schema:
                    raise CanaryFailure(f"current7 {name} schema mismatch")
                declared_path = Path(str(declared.get("path") or ""))
                if not declared_path.is_absolute():
                    raise CanaryFailure(f"current7 {name} path is not absolute")
                value, identity = _read_runtime_json_identity(
                    declared_path,
                    label=f"current7 {name}",
                    max_bytes=maximum,
                )
                resolved_path = Path(identity["path"])
                if not resolved_path.is_relative_to(root):
                    raise CanaryFailure(f"current7 {name} escapes its containment root")
                expected_sha = _exact_sha256(declared.get("sha256"), f"current7 {name}")
                if identity["sha256"] != expected_sha:
                    raise CanaryFailure(f"current7 {name} SHA-256 mismatch")
                if value.get("schema_version") != schema:
                    raise CanaryFailure(f"current7 {name} payload schema mismatch")
                references[name] = {
                    "identity": identity,
                    "value": value,
                }

            _, final_identity = _read_runtime_json_identity(
                raw_path,
                label="current7 secondary index",
                max_bytes=256 * 1024,
            )
            if final_identity["sha256"] != index_identity["sha256"]:
                raise CanaryFailure("current7 index changed during authentication")
            for flag in CURRENT7_AUTHORITY_FALSE_FLAGS:
                if index.get(flag) is not False:
                    raise CanaryFailure(
                        f"current7 index asserted forbidden authority: {flag}"
                    )
            active_cohort_id = str(index.get("active_cohort_id") or "").strip()
            bundle_id = str(index.get("bundle_id") or "").strip()
            if not active_cohort_id or not bundle_id:
                raise CanaryFailure("current7 cohort or bundle identity is missing")
            snapshot_sha256 = _exact_sha256(
                index.get("snapshot_sha256"), "current7 snapshot"
            )
            bundle_manifest_sha256 = _exact_sha256(
                index.get("bundle_manifest_sha256"),
                "current7 bundle manifest",
            )
            constraint_version = str(index.get("constraint_version") or "").strip()
            if (
                not constraint_version
                or index.get("hard_spec") != CURRENT7_HARD_SPEC
                or index.get("hard_spec_sha256") != CURRENT7_HARD_SPEC_SHA256
                or index.get("constraint_names") != list(CURRENT7_CONSTRAINT_NAMES)
                or index.get("temperature_targets")
                != list(CURRENT7_TEMPERATURE_TARGETS)
            ):
                raise CanaryFailure("current7 index constraint identity diverged")
            for name in (
                "hard_constraint_contract_sha256",
                "temperature_contract_sha256",
            ):
                _exact_sha256(index.get(name), f"current7 index {name}")

            status = references["status"]["value"]
            aggregate = status.get("aggregate")
            if (
                not isinstance(aggregate, dict)
                or aggregate.get("schema_version") != CURRENT7_AGGREGATE_SCHEMA
            ):
                raise CanaryFailure("current7 aggregate schema mismatch")
            status_event_at, _ = _current7_timestamp(
                index.get("status_event_at"),
                "current7 index status_event_at",
            )
            status_updated_at, _ = _current7_timestamp(
                status.get("updated_at"),
                "current7 status updated_at",
            )
            harvest_observed_at, _ = _current7_timestamp(
                index.get("harvest_observed_at"),
                "current7 index harvest_observed_at",
            )
            if status_event_at != status_updated_at:
                raise CanaryFailure("current7 status event identity diverged")
            expected_constraint_identity = _current7_constraint_identity(index)
            for label, document in (
                ("status", status),
                ("aggregate", aggregate),
            ):
                if _current7_constraint_identity(document) != (
                    expected_constraint_identity
                ):
                    raise CanaryFailure(
                        f"current7 {label} constraint identity diverged"
                    )
            for flag in CURRENT7_AUTHORITY_FALSE_FLAGS:
                if status.get(flag) is not False:
                    raise CanaryFailure(
                        f"current7 status asserted forbidden authority: {flag}"
                    )
            for flag in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
            ):
                if aggregate.get(flag) is not False:
                    raise CanaryFailure(
                        f"current7 aggregate asserted forbidden authority: {flag}"
                    )
            if (
                status.get("healthy") is not True
                or _nonnegative_integer(
                    status.get("refused_terminal_count"),
                    "current7 refused_terminal_count",
                )
                != 0
                or status.get("cohort_id") != active_cohort_id
                or status.get("bundle_id") != bundle_id
                or status.get("bundle_manifest_sha256") != bundle_manifest_sha256
                or status.get("snapshot_sha256") != snapshot_sha256
            ):
                raise CanaryFailure("current7 healthy snapshot identity diverged")
            constraint_identity_sha256 = _current7_constraint_identity_sha256(index)
            return {
                **index_identity,
                "schema_version": CURRENT7_INDEX_SCHEMA,
                "active_cohort_id": active_cohort_id,
                "bundle_id": bundle_id,
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "snapshot_sha256": snapshot_sha256,
                "index_file_sha256": index_identity["sha256"],
                "snapshot_file_sha256": references["status"]["identity"]["sha256"],
                "harvest_observed_at": harvest_observed_at,
                "status_event_at": status_event_at,
                "constraint_version": constraint_version,
                "constraint_identity_sha256": constraint_identity_sha256,
                "constraint_identity": expected_constraint_identity,
                "status": references["status"]["identity"],
                "compatibility": references["compatibility"]["identity"],
                "healthy": True,
                "nested_sha256_verified": True,
            }
        except (CanaryFailure, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.02)
    raise CanaryFailure(
        "current7 index authentication did not stabilize: " + last_error
    )


def _authenticate_current7_condition_index(raw_path: Path) -> dict[str, Any]:
    """Authenticate one read-only condition index without fixing its hard spec.

    Condition searches intentionally carry different staged hard specifications.
    The monitor reader performs the semantic mixed-bundle validation; this gate
    independently seals the pointer, containment, nested hashes and the 64 MiB
    status-read boundary used by that reader.
    """

    last_error = "current7 condition index was not read"
    for _ in range(8):
        try:
            index, index_identity = _read_runtime_json_identity(
                raw_path,
                label="current7 condition index",
                max_bytes=256 * 1024,
            )
            if index.get("schema_version") != CURRENT7_INDEX_SCHEMA:
                raise CanaryFailure("current7 condition index schema mismatch")
            root_text = str(index.get("path_containment_root") or "")
            if not root_text:
                raise CanaryFailure("current7 condition containment root is missing")
            root = Path(root_text)
            if not root.is_absolute():
                raise CanaryFailure("current7 condition containment root is not absolute")
            root = root.resolve(strict=True)
            index_path = Path(index_identity["path"])
            if not root.is_dir() or not index_path.is_relative_to(root):
                raise CanaryFailure("current7 condition index escapes its containment root")

            references: dict[str, dict[str, Any]] = {}
            for name, schema, maximum in (
                (
                    "status",
                    CURRENT7_STATUS_SCHEMA,
                    CURRENT7_CONDITION_STATUS_MAX_BYTES,
                ),
                (
                    "compatibility",
                    CURRENT7_COMPATIBILITY_SCHEMA,
                    512 * 1024,
                ),
            ):
                declared = index.get(name)
                if not isinstance(declared, dict):
                    raise CanaryFailure(
                        f"current7 condition {name} identity is missing"
                    )
                if declared.get("schema_version") != schema:
                    raise CanaryFailure(
                        f"current7 condition {name} schema mismatch"
                    )
                declared_path = Path(str(declared.get("path") or ""))
                if not declared_path.is_absolute():
                    raise CanaryFailure(
                        f"current7 condition {name} path is not absolute"
                    )
                value, identity = _read_runtime_json_identity(
                    declared_path,
                    label=f"current7 condition {name}",
                    max_bytes=maximum,
                )
                resolved_path = Path(identity["path"])
                if not resolved_path.is_relative_to(root):
                    raise CanaryFailure(
                        f"current7 condition {name} escapes its containment root"
                    )
                expected_sha = _exact_sha256(
                    declared.get("sha256"), f"current7 condition {name}"
                )
                if identity["sha256"] != expected_sha:
                    raise CanaryFailure(
                        f"current7 condition {name} SHA-256 mismatch"
                    )
                if value.get("schema_version") != schema:
                    raise CanaryFailure(
                        f"current7 condition {name} payload schema mismatch"
                    )
                references[name] = {"identity": identity, "value": value}

            _, final_identity = _read_runtime_json_identity(
                raw_path,
                label="current7 condition index",
                max_bytes=256 * 1024,
            )
            if final_identity["sha256"] != index_identity["sha256"]:
                raise CanaryFailure(
                    "current7 condition index changed during authentication"
                )
            for flag in CURRENT7_AUTHORITY_FALSE_FLAGS:
                if index.get(flag) is not False:
                    raise CanaryFailure(
                        f"current7 condition index asserted forbidden authority: {flag}"
                    )
            status = references["status"]["value"]
            for flag in CURRENT7_AUTHORITY_FALSE_FLAGS:
                if status.get(flag) is not False:
                    raise CanaryFailure(
                        f"current7 condition status asserted forbidden authority: {flag}"
                    )
            aggregate = status.get("aggregate")
            if (
                not isinstance(aggregate, dict)
                or aggregate.get("schema_version") != CURRENT7_AGGREGATE_SCHEMA
            ):
                raise CanaryFailure("current7 condition aggregate schema mismatch")
            constraint_version = str(index.get("constraint_version") or "").strip()
            if not constraint_version:
                raise CanaryFailure("current7 condition constraint version is missing")
            return {
                **index_identity,
                "schema_version": CURRENT7_INDEX_SCHEMA,
                "status": references["status"]["identity"],
                "compatibility": references["compatibility"]["identity"],
                "status_max_bytes": CURRENT7_CONDITION_STATUS_MAX_BYTES,
                "constraint_version": constraint_version,
                "hard_spec_sha256": _exact_sha256(
                    index.get("hard_spec_sha256"),
                    "current7 condition hard spec",
                ),
                "hard_constraint_contract_sha256": _exact_sha256(
                    index.get("hard_constraint_contract_sha256"),
                    "current7 condition hard constraint contract",
                ),
                "temperature_contract_sha256": _exact_sha256(
                    index.get("temperature_contract_sha256"),
                    "current7 condition temperature contract",
                ),
                "bundle_manifest_sha256": _exact_sha256(
                    index.get("bundle_manifest_sha256"),
                    "current7 condition bundle manifest",
                ),
                "snapshot_sha256": _exact_sha256(
                    index.get("snapshot_sha256"),
                    "current7 condition snapshot",
                ),
                "nested_sha256_verified": True,
                "authority_flags_verified_false": True,
            }
        except (CanaryFailure, OSError) as exc:
            last_error = str(exc)
            time.sleep(0.02)
    raise CanaryFailure(
        "current7 condition index authentication did not stabilize: " + last_error
    )


def _authenticate_current7_condition_indexes(
    raw_paths: list[Path],
) -> list[dict[str, Any]]:
    if not raw_paths:
        return []
    if len(raw_paths) > CURRENT7_CONDITION_INDEX_LIMIT:
        raise CanaryFailure(
            "current7 condition index count exceeds "
            f"{CURRENT7_CONDITION_INDEX_LIMIT}"
        )
    identities = [_authenticate_current7_condition_index(path) for path in raw_paths]
    resolved = [identity["path"].casefold() for identity in identities]
    if len(set(resolved)) != len(resolved):
        raise CanaryFailure("current7 condition index paths contain duplicates")
    return identities


def _refresh_current7_index(
    configured_index: dict[str, Any],
) -> dict[str, Any]:
    """Refresh mutable snapshot SHAs while pinning stable search identity."""

    latest = _authenticate_current7_index(Path(configured_index["path"]))
    for name in (
        "path",
        "bundle_id",
        "bundle_manifest_sha256",
        "constraint_version",
        "constraint_identity_sha256",
    ):
        if latest.get(name) != configured_index.get(name):
            raise CanaryFailure(f"current7 stable identity changed: {name}")
    return latest


def _authenticate_blocker_hpo_v2_status(raw_path: Path) -> dict[str, Any]:
    """Authenticate one HPO v2 status snapshot without freezing its heartbeat."""

    value, identity = _read_runtime_json_identity(
        raw_path,
        label="blocker HPO v2 status",
        max_bytes=8 * 1024 * 1024,
    )
    if value.get("schema_version") != BLOCKER_HPO_V2_STATUS_SCHEMA:
        raise CanaryFailure("blocker HPO v2 status schema mismatch")
    phase = str(value.get("phase") or "")
    if phase not in {
        "starting",
        "preflight_complete",
        "optimizing",
        "succeeded",
        "failed",
    }:
        raise CanaryFailure("blocker HPO v2 phase is invalid")
    for flag in HPO_V2_AUTHORITY_FALSE_FLAGS:
        if value.get(flag) is not False:
            raise CanaryFailure(f"blocker HPO v2 asserted forbidden authority: {flag}")
    stage = _nonnegative_integer(
        value.get("selected_cumulative_trials_per_job"),
        "blocker HPO v2 selected_cumulative_trials_per_job",
    )
    jobs = value.get("jobs")
    trials = value.get("trials")
    if stage <= 0 or not isinstance(jobs, dict) or not jobs:
        raise CanaryFailure("blocker HPO v2 job contract is invalid")
    if not isinstance(trials, dict):
        raise CanaryFailure("blocker HPO v2 trial aggregate is missing")
    aggregate = {name: 0 for name in ("complete", "running", "failed")}
    for name, job in jobs.items():
        if not isinstance(name, str) or not isinstance(job, dict):
            raise CanaryFailure("blocker HPO v2 job row is invalid")
        if (
            _nonnegative_integer(job.get("total"), f"blocker HPO v2 job {name}.total")
            != stage
        ):
            raise CanaryFailure("blocker HPO v2 per-job budget drifted")
        observed = 0
        for state in aggregate:
            count = _nonnegative_integer(
                job.get(state), f"blocker HPO v2 job {name}.{state}"
            )
            aggregate[state] += count
            observed += count
        if observed > stage:
            raise CanaryFailure("blocker HPO v2 per-job counts overflow")
    expected_total = len(jobs) * stage
    if _nonnegative_integer(
        trials.get("total"), "blocker HPO v2 trials.total"
    ) != expected_total or any(
        _nonnegative_integer(trials.get(state), f"blocker HPO v2 trials.{state}")
        != aggregate[state]
        for state in aggregate
    ):
        raise CanaryFailure("blocker HPO v2 aggregate trial counts disagree")
    try:
        heartbeat = datetime.fromisoformat(
            str(value.get("heartbeat_at") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise CanaryFailure("blocker HPO v2 heartbeat is invalid") from exc
    if heartbeat.tzinfo is None:
        raise CanaryFailure("blocker HPO v2 heartbeat is not timezone-aware")
    return {
        **identity,
        "schema_version": BLOCKER_HPO_V2_STATUS_SCHEMA,
        "phase": phase,
        "config_sha256": _exact_sha256(
            value.get("config_sha256"), "blocker HPO v2 config"
        ),
        "dataset_sha256": _exact_sha256(
            value.get("dataset_sha256"), "blocker HPO v2 dataset"
        ),
        "job_count": len(jobs),
        "job_identity_sha256": _sha256_bytes(_canonical_json(sorted(jobs))),
        "stage_trials_per_job": stage,
        "trial_total": expected_total,
        "trial_complete": aggregate["complete"],
        "trial_running": aggregate["running"],
        "trial_failed": aggregate["failed"],
        "authority_flags_verified_false": True,
    }


def _refresh_blocker_hpo_v2_status(
    configured_status: dict[str, Any],
) -> dict[str, Any]:
    latest = _authenticate_blocker_hpo_v2_status(Path(configured_status["path"]))
    for name in (
        "path",
        "config_sha256",
        "dataset_sha256",
        "job_identity_sha256",
    ):
        if latest.get(name) != configured_status.get(name):
            raise CanaryFailure(f"blocker HPO v2 stable identity changed: {name}")
    return latest


def _contained_runtime_file(
    root: Path,
    relative: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> dict[str, Any]:
    current = root
    for part in relative.parts:
        current /= part
        if _is_reparse_point(current):
            raise CanaryFailure(f"{label} uses a symlink or reparse point: {current}")
    identity = _regular_input_identity(
        root / relative,
        expected_sha256,
        label=label,
        max_bytes=max_bytes,
    )
    resolved = Path(identity["path"])
    if not resolved.is_relative_to(root):
        raise CanaryFailure(f"{label} escapes the validation runtime root")
    return identity


def _dual_release_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate every ambient input used by the dual validation overlay."""

    if not args.dual_fea_runtime_root.is_absolute():
        raise CanaryFailure("dual Standard-FEA runtime root must be absolute")
    if not args.dual_fea_allowed_root.is_absolute():
        raise CanaryFailure("dual Standard-FEA allowed root must be absolute")
    if _is_reparse_point(args.dual_fea_runtime_root):
        raise CanaryFailure("dual Standard-FEA runtime root is a reparse point")
    if _is_reparse_point(args.dual_fea_allowed_root):
        raise CanaryFailure("dual Standard-FEA allowed root is a reparse point")
    try:
        runtime_root = args.dual_fea_runtime_root.resolve(strict=True)
        allowed_root = args.dual_fea_allowed_root.resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure(
            "dual Standard-FEA runtime/allowed root is unavailable"
        ) from exc
    if not runtime_root.is_dir() or not allowed_root.is_dir():
        raise CanaryFailure("dual Standard-FEA runtime/allowed root is not a directory")
    if runtime_root == allowed_root or not runtime_root.is_relative_to(allowed_root):
        raise CanaryFailure("dual Standard-FEA runtime escapes its allowed root")

    handoff = _regular_input_identity(
        args.sealed_successor_handoff,
        args.sealed_successor_handoff_sha256,
        label="sealed successor handoff",
        max_bytes=128 * 1024,
    )
    dual_plan = _regular_input_identity(
        args.dual_successor_plan,
        args.dual_successor_plan_sha256,
        label="dual successor ValidateOnly plan",
        max_bytes=512 * 1024,
    )
    dual_ui = _regular_input_identity(
        args.dual_successor_ui,
        args.dual_successor_ui_sha256,
        label="dual successor read-only UI handoff",
        max_bytes=128 * 1024,
    )
    dual_receipt = _regular_input_identity(
        args.dual_successor_receipt,
        args.dual_successor_receipt_sha256,
        label="dual successor validation receipt",
        max_bytes=128 * 1024,
    )
    execution_plan = _regular_input_identity(
        args.dual_fea_plan,
        args.dual_fea_plan_sha256,
        label="dual Standard-FEA execution plan",
        max_bytes=2 * 1024 * 1024,
    )
    manifest = _contained_runtime_file(
        runtime_root,
        Path("manifest.json"),
        args.dual_fea_manifest_sha256,
        label="dual Standard-FEA runtime manifest",
        max_bytes=128 * 1024,
    )
    source = _contained_runtime_file(
        runtime_root,
        Path("sealed-plan-source") / "experimental_source.json",
        args.dual_fea_source_sha256,
        label="dual Standard-FEA sealed plan source",
        max_bytes=32 * 1024,
    )
    return {
        "sealed_successor_handoff": handoff,
        "dual_successor_plan": dual_plan,
        "dual_successor_ui": dual_ui,
        "dual_successor_receipt": dual_receipt,
        "standard_fea_validation": {
            "runtime_root": str(runtime_root),
            "allowed_root": str(allowed_root),
            "runtime_manifest": manifest,
            "execution_plan": execution_plan,
            "sealed_plan_source": source,
            "expected_tasks": [
                {
                    "candidate_sha256": candidate_sha,
                    **identity,
                }
                for candidate_sha, identity in sorted(
                    EXPECTED_DUAL_STANDARD_FEA_TASKS.items()
                )
            ],
        },
    }


def _apply_dual_release_environment(
    configuration: dict[str, Any],
) -> dict[str, str]:
    validation = configuration["standard_fea_validation"]
    values = {
        "sealed_successor_handoff": configuration["sealed_successor_handoff"]["path"],
        "sealed_successor_handoff_sha256": configuration["sealed_successor_handoff"][
            "sha256"
        ],
        "dual_successor_plan": configuration["dual_successor_plan"]["path"],
        "dual_successor_plan_sha256": configuration["dual_successor_plan"]["sha256"],
        "dual_successor_ui": configuration["dual_successor_ui"]["path"],
        "dual_successor_ui_sha256": configuration["dual_successor_ui"]["sha256"],
        "dual_successor_receipt": configuration["dual_successor_receipt"]["path"],
        "dual_successor_receipt_sha256": configuration["dual_successor_receipt"][
            "sha256"
        ],
        "dual_fea_runtime_root": validation["runtime_root"],
        "dual_fea_allowed_root": validation["allowed_root"],
        "dual_fea_manifest_sha256": validation["runtime_manifest"]["sha256"],
        "dual_fea_plan_sha256": validation["execution_plan"]["sha256"],
        "dual_fea_source_sha256": validation["sealed_plan_source"]["sha256"],
    }
    applied: dict[str, str] = {}
    for name, environment_name in DUAL_VALIDATION_ENVIRONMENT.items():
        os.environ[environment_name] = values[name]
        applied[environment_name] = values[name]
    return applied


def _require_utf8_display_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanaryFailure(f"{name} is missing")
    text = value.strip()
    try:
        if text.encode("utf-8").decode("utf-8") != text:
            raise CanaryFailure(f"{name} does not round-trip as UTF-8")
    except UnicodeError as exc:
        raise CanaryFailure(f"{name} is not valid UTF-8 text") from exc
    suspicious_fragments = ("\ufffd", "Ã", "Â", "â", "ì", "ë", "í")
    suspicious = any(fragment in text for fragment in suspicious_fragments)
    suspicious = suspicious or any(
        "\u3400" <= character <= "\u9fff" or "\uf900" <= character <= "\ufaff"
        for character in text
    )
    if suspicious:
        raise CanaryFailure(f"{name} contains probable mojibake")
    return text


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _pyvenv_configuration(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CanaryFailure(f"cannot read candidate pyvenv.cfg: {path}") from exc
    values: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip().lower()] = value.strip()
    if not values.get("home") or not values.get("executable"):
        raise CanaryFailure("candidate pyvenv.cfg lacks home/executable identity")
    return values


def _runtime_identity(expected_venv_root: Path) -> dict[str, Any]:
    raw_executable = Path(sys.executable)
    raw_base_executable = Path(str(getattr(sys, "_base_executable", "") or ""))
    if (
        raw_executable.is_symlink()
        or not raw_executable.is_file()
        or raw_base_executable.is_symlink()
        or not raw_base_executable.is_file()
    ):
        raise CanaryFailure("candidate runtime executable pair is not regular")
    executable = raw_executable.resolve(strict=True)
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    base_executable = raw_base_executable.resolve(strict=True)
    expected = expected_venv_root.resolve(strict=True)
    if prefix != expected:
        raise CanaryFailure(
            f"candidate venv mismatch: running {prefix}, expected {expected}"
        )
    if prefix == base_prefix:
        raise CanaryFailure("release canary must run inside a dedicated venv")
    try:
        executable.relative_to(prefix)
    except ValueError as exc:
        raise CanaryFailure("python executable is outside the candidate venv") from exc
    raw_pyvenv_path = prefix / "pyvenv.cfg"
    if raw_pyvenv_path.is_symlink() or not raw_pyvenv_path.is_file():
        raise CanaryFailure("candidate pyvenv.cfg is not a regular file")
    pyvenv_path = raw_pyvenv_path.resolve(strict=True)
    pyvenv = _pyvenv_configuration(pyvenv_path)
    if Path(pyvenv["home"]).resolve(strict=True) != base_prefix:
        raise CanaryFailure("candidate pyvenv.cfg home differs from base_prefix")
    if Path(pyvenv["executable"]).resolve(strict=True) != base_executable:
        raise CanaryFailure(
            "candidate pyvenv.cfg executable differs from _base_executable"
        )
    for runtime_executable in (executable, base_executable):
        if runtime_executable.is_symlink() or not runtime_executable.is_file():
            raise CanaryFailure(
                f"candidate runtime executable is not regular: {runtime_executable}"
            )

    import psutil

    observed_process_executable = Path(psutil.Process().exe()).resolve(strict=True)
    if observed_process_executable not in {executable, base_executable}:
        raise CanaryFailure(
            "candidate OS process image is outside its sealed pyvenv pair"
        )
    runtime_process_identity = {
        "pyvenv": {
            "path": str(pyvenv_path),
            "sha256": _sha256_file(pyvenv_path, 256 * 1024),
            "home": str(base_prefix),
            "executable": str(base_executable),
        },
        "venv_launcher": {
            "path": str(executable),
            "sha256": _sha256_file(executable, 64 * 1024 * 1024),
        },
        "base_interpreter": {
            "path": str(base_executable),
            "sha256": _sha256_file(base_executable, 64 * 1024 * 1024),
        },
        "observed_process_executable": str(observed_process_executable),
    }

    import pyarrow

    try:
        arrow_major = int(str(pyarrow.__version__).split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise CanaryFailure(
            f"unparseable pyarrow version: {pyarrow.__version__!r}"
        ) from exc
    if arrow_major != EXPECTED_ARROW_MAJOR:
        raise CanaryFailure(
            "production parquet compatibility requires pyarrow 24.x; "
            f"candidate has {pyarrow.__version__}"
        )
    return {
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "venv_root": str(prefix),
        "base_prefix": str(base_prefix),
        "runtime_process_identity": runtime_process_identity,
        "pyarrow_version": str(pyarrow.__version__),
        "packages": {
            name: _package_version(name)
            for name in ("fastapi", "pandas", "pyarrow", "psutil", "uvicorn")
        },
    }


def _git(source_root: Path, *arguments: str, timeout: float = 20.0) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_root.as_posix()}",
            "-C",
            str(source_root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CanaryFailure(f"git {' '.join(arguments)} failed: {detail[:500]}")
    return completed.stdout.strip()


def _source_identity(source_root: Path, expected_revision: str) -> dict[str, Any]:
    revision = _git(source_root, "rev-parse", "HEAD").lower()
    expected = expected_revision.strip().lower()
    if not FULL_SHA.fullmatch(expected):
        raise CanaryFailure("--expected-revision must be an exact 40-hex commit")
    if revision != expected:
        raise CanaryFailure(
            f"candidate source revision mismatch: {revision} != {expected}"
        )
    dirty_lines = [
        line
        for line in _git(
            source_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line.strip()
    ]
    if dirty_lines:
        raise CanaryFailure(
            "candidate source checkout is dirty: " + "; ".join(dirty_lines[:20])
        )

    monitoring_root = source_root / "regression_260707" / "monitoring"
    code_files = sorted(
        path
        for path in monitoring_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in {".py", ".html", ".js", ".css", ".txt"}
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    )
    if not code_files:
        raise CanaryFailure("candidate monitoring source files are unavailable")
    code_manifest = {
        path.relative_to(source_root).as_posix(): _sha256_file(path, 8 * 1024 * 1024)
        for path in code_files
    }
    return {
        "source_root": str(source_root),
        "revision": revision,
        "dirty": False,
        "monitoring_file_count": len(code_manifest),
        "monitoring_tree_sha256": _sha256_bytes(_canonical_json(code_manifest)),
        "monitoring_files": code_manifest,
    }


@contextmanager
def _ephemeral_server(app: Any) -> Iterator[str]:
    import uvicorn

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = int(listener.getsockname()[1])
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="mft-monitor-release-canary",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 20.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        raise CanaryFailure("candidate ASGI server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        try:
            listener.close()
        except OSError:
            pass
        if thread.is_alive():
            raise CanaryFailure("candidate ASGI server did not stop cleanly")


def _get_json(
    base_url: str,
    endpoint: str,
    timeout: float,
    *,
    require_utf8_charset: bool = True,
) -> dict[str, Any]:
    request = Request(
        base_url + endpoint,
        method="GET",
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset()
            raw = response.read(16 * 1024 * 1024 + 1)
    except Exception as exc:
        raise CanaryFailure(
            f"GET {endpoint} failed: {type(exc).__name__}: {exc}"
        ) from exc
    if status != 200:
        raise CanaryFailure(f"GET {endpoint} returned HTTP {status}")
    if content_type != "application/json":
        raise CanaryFailure(
            f"GET {endpoint} returned unexpected content type {content_type}"
        )
    if require_utf8_charset and str(charset or "").lower() != "utf-8":
        raise CanaryFailure(
            f"GET {endpoint} JSON charset is not explicit UTF-8: {charset!r}"
        )
    if len(raw) > 16 * 1024 * 1024:
        raise CanaryFailure(f"GET {endpoint} response exceeds 16 MiB")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CanaryFailure(f"GET {endpoint} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CanaryFailure(f"GET {endpoint} did not return a JSON object")
    if (
        json.loads(
            json.dumps(value, ensure_ascii=False).encode("utf-8").decode("utf-8")
        )
        != value
    ):
        raise CanaryFailure(f"GET {endpoint} failed JSON UTF-8 round-trip")
    if value.get("available") is False and value.get("error"):
        raise CanaryFailure(f"GET {endpoint} failed closed: {value['error']}")
    return value


def _get_text(
    base_url: str, endpoint: str, expected_content_type: str, timeout: float
) -> str:
    request = Request(
        base_url + endpoint,
        method="GET",
        headers={"Accept": expected_content_type, "Cache-Control": "no-cache"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            content_type = response.headers.get_content_type()
            cache_control = response.headers.get("Cache-Control", "")
            raw = response.read(4 * 1024 * 1024 + 1)
    except Exception as exc:
        raise CanaryFailure(
            f"GET {endpoint} failed: {type(exc).__name__}: {exc}"
        ) from exc
    allowed_content_types = {expected_content_type}
    if expected_content_type == "text/javascript":
        allowed_content_types.add("application/javascript")
    if status != 200 or content_type not in allowed_content_types:
        raise CanaryFailure(f"GET {endpoint} returned HTTP {status} / {content_type}")
    if "no-store" not in cache_control.lower():
        raise CanaryFailure(f"GET {endpoint} is missing Cache-Control: no-store")
    if len(raw) > 4 * 1024 * 1024:
        raise CanaryFailure(f"GET {endpoint} response exceeds 4 MiB")
    try:
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise CanaryFailure(f"GET {endpoint} is not valid UTF-8") from exc


def _verify_data_api(
    payload: dict[str, Any], pipeline_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise CanaryFailure("/api/data schema version mismatch")
    active_cohort = payload.get("active_cohort")
    if not isinstance(active_cohort, dict):
        raise CanaryFailure("/api/data active cohort is missing")
    active_label = _require_utf8_display_text(
        active_cohort.get("label"), "/api/data active_cohort.label"
    )
    required_authority_flags = (
        "available",
        "counts_available",
        "authority_verified",
        "manifest_matches_audit",
    )
    cohort = payload.get("latest_eligible_cohort")
    cohort_source = "latest_eligible_cohort"
    if not (
        isinstance(cohort, dict)
        and all(cohort.get(key) is True for key in required_authority_flags)
    ):
        controller_stale = bool(
            isinstance(cohort, dict)
            and cohort.get("available") is False
            and cohort.get("counts_available") is False
            and cohort.get("authority_verified") is False
            and cohort.get("scan_truncated") is not True
            and cohort.get("error")
            == "controller role is not alive; controller status is stale"
            and cohort.get("counts_error")
            == "controller role is not alive; controller status is stale"
        )
        if not controller_stale:
            raise CanaryFailure(
                "/api/data latest eligible cohort failed for a non-stale reason"
            )
        completed = payload.get("latest_completed_training")
        sealed = completed.get("snapshot") if isinstance(completed, dict) else None
        if not (
            isinstance(sealed, dict)
            and all(
                sealed.get(key) is True for key in required_authority_flags
            )
            and sealed.get("kind") == "latest_completed_training_snapshot"
            and sealed.get("is_controller_generation") is True
            and sealed.get("generation") == sealed.get("controller_generation")
        ):
            raise CanaryFailure(
                "/api/data has no fully authenticated live or sealed cohort"
            )
        cohort = sealed
        cohort_source = "latest_completed_training.snapshot"
    raw_rows = _nonnegative_integer(cohort.get("raw_rows"), "cohort.raw_rows")
    strict_em = _nonnegative_integer(
        cohort.get("strict_em_rows"), "cohort.strict_em_rows"
    )
    strict_full = _nonnegative_integer(
        cohort.get("strict_full_rows"), "cohort.strict_full_rows"
    )
    if not 0 < strict_full <= strict_em <= raw_rows:
        raise CanaryFailure("/api/data cohort row tiers are incoherent or empty")
    generation_id = str(cohort.get("generation_id") or "").lower()
    artifact_sha = str(cohort.get("artifact_sha256") or "").lower()
    if not FULL_SHA256.fullmatch(generation_id):
        raise CanaryFailure("/api/data cohort generation id is malformed")
    if not FULL_SHA256.fullmatch(artifact_sha):
        raise CanaryFailure("/api/data cohort artifact hash is malformed")
    if cohort.get("counts_source") != ("authenticated_train.parquet_quality_contract"):
        raise CanaryFailure("/api/data cohort was not recomputed by the contract")

    dataset_root = (pipeline_root / "artifacts" / "dataset").resolve(strict=True)
    parquet_path = (dataset_root / generation_id / "train.parquet").resolve(strict=True)
    try:
        parquet_path.relative_to(dataset_root)
    except ValueError as exc:
        raise CanaryFailure(
            "cohort parquet escapes the pipeline artifact root"
        ) from exc
    actual_sha = _sha256_file(parquet_path)
    if actual_sha != artifact_sha:
        raise CanaryFailure("cohort parquet hash changed after API authentication")

    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(parquet_path)
    schema_names = set(parquet_file.schema_arrow.names)
    missing = [name for name in PRIMARY_PARQUET_COLUMNS if name not in schema_names]
    if missing:
        raise CanaryFailure(
            "strict cohort is missing primary columns: " + ", ".join(missing)
        )
    table = parquet_file.read(columns=list(PRIMARY_PARQUET_COLUMNS))
    table.validate(full=True)
    if table.num_rows != raw_rows:
        raise CanaryFailure(
            f"strict cohort primary read row mismatch: {table.num_rows} != {raw_rows}"
        )
    if table.num_columns != len(PRIMARY_PARQUET_COLUMNS):
        raise CanaryFailure("strict cohort primary read column count mismatch")
    return (
        {
            "available": True,
            "generation": cohort.get("generation"),
            "generation_id": generation_id,
            "raw_rows": raw_rows,
            "strict_em_rows": strict_em,
            "strict_full_rows": strict_full,
            "solver_revision": cohort.get("solver_revision"),
            "library_revision": cohort.get("library_revision"),
            "artifact_sha256": artifact_sha,
            "quality_contract_sha256": cohort.get("quality_contract_sha256"),
            "profile_sha256": cohort.get("profile_sha256"),
            "active_cohort_label": active_label,
            "authority_source": cohort_source,
            "utf8_roundtrip_verified": True,
        },
        {
            "path": str(parquet_path),
            "sha256": actual_sha,
            "row_count": table.num_rows,
            "column_count": table.num_columns,
            "columns": list(PRIMARY_PARQUET_COLUMNS),
            "schema_column_count": len(schema_names),
            "validated_full": True,
        },
    )


def _require_false_flags(
    payload: dict[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> None:
    drifted = [name for name in fields if payload.get(name) is not False]
    if drifted:
        raise CanaryFailure(
            f"{label} attempted authority through: {', '.join(drifted)}"
        )


def _verify_projected_actual_t120_gate(
    raw: Any,
    *,
    label: str,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CanaryFailure(f"{label} actual T120 gate is malformed")
    if (
        raw.get("contract") != "mft-tier1-actual-Llt-and-conditional-all11-T120-v1"
        or _finite_number(raw.get("Llt_target_uH"), f"{label}.Llt_target_uH") != 27.5
        or _finite_number(raw.get("Llt_tolerance_uH"), f"{label}.Llt_tolerance_uH")
        != 0.55
        or _finite_number(
            raw.get("temperature_limit_C"), f"{label}.temperature_limit_C"
        )
        != EXPECTED_TIER1_TEMPERATURE_LIMIT_C
        or not isinstance(raw.get("pass"), bool)
        or not isinstance(raw.get("reasons"), list)
        or any(not isinstance(item, str) for item in raw["reasons"])
    ):
        raise CanaryFailure(f"{label} actual T120 gate contract drifted")
    contract = raw.get("temperature_contract")
    if not isinstance(contract, dict) or (
        contract.get("targets") != list(EXPECTED_TIER1_TEMPERATURE_TARGETS)
        or contract.get("target_count") != len(EXPECTED_TIER1_TEMPERATURE_TARGETS)
        or contract.get("robust_upper_bound_C") != EXPECTED_TIER1_TEMPERATURE_LIMIT_C
    ):
        raise CanaryFailure(f"{label} actual all-11 temperature contract drifted")
    actual = raw.get("actual")
    if not isinstance(actual, dict):
        raise CanaryFailure(f"{label} actual result payload is missing")
    llt_full = _finite_number(actual.get("Llt_full_uH"), f"{label}.Llt_full_uH")
    side_turns = actual.get("N2_side")
    if (
        isinstance(side_turns, bool)
        or not isinstance(side_turns, int)
        or side_turns < 0
    ):
        raise CanaryFailure(f"{label} actual N2_side is invalid")
    temperatures = actual.get("temperatures")
    if not isinstance(temperatures, dict) or set(temperatures) != set(
        EXPECTED_TIER1_TEMPERATURE_TARGETS
    ):
        raise CanaryFailure(f"{label} actual all-11 inventory drifted")

    reasons: list[str] = []
    if abs(llt_full - 27.5) > 0.55 + 1e-9:
        reasons.append("Llt_actual_band")
    maximum_temperature: float | None = None
    side_targets = {"T_max_Rx_side", "Tprobe_Rx_side_leeward_max"}
    for name in EXPECTED_TIER1_TEMPERATURE_TARGETS:
        item = temperatures[name]
        if not isinstance(item, dict) or not isinstance(item.get("applicable"), bool):
            raise CanaryFailure(f"{label} actual temperature {name} is malformed")
        expected_applicable = not (name in side_targets and side_turns == 0)
        if item["applicable"] is not expected_applicable:
            raise CanaryFailure(
                f"{label} actual temperature applicability drifted: {name}"
            )
        value = item.get("value_C")
        if expected_applicable:
            value = _finite_number(value, f"{label}.{name}.value_C")
            expected_pass: bool | None = (
                value <= EXPECTED_TIER1_TEMPERATURE_LIMIT_C + 1e-9
            )
            maximum_temperature = (
                value
                if maximum_temperature is None
                else max(maximum_temperature, value)
            )
            if not expected_pass:
                reasons.append(f"temperature_actual_limit:{name}")
        else:
            if value is not None:
                raise CanaryFailure(
                    f"{label} disabled temperature contains a value: {name}"
                )
            expected_pass = None
        if item.get("pass") is not expected_pass:
            raise CanaryFailure(f"{label} actual temperature PASS drifted: {name}")
    recomputed_pass = not reasons
    if raw["pass"] is not recomputed_pass or raw["reasons"] != reasons:
        raise CanaryFailure(
            f"{label} declared actual gate disagrees with canonical T120 recomputation"
        )
    return {
        "contract": raw["contract"],
        "pass": recomputed_pass,
        "reasons": reasons,
        "Llt_full_uH": llt_full,
        "N2_side": side_turns,
        "maximum_applicable_temperature_C": maximum_temperature,
        "temperature_target_count": len(temperatures),
    }


def _verify_dual_standard_fea(
    payload: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    collection = payload.get("sealed_successors")
    if not isinstance(collection, dict):
        raise CanaryFailure("dual sealed successor collection is missing")
    if (
        collection.get("schema_version") != "mft-monitor-dual-sealed-successors-v1"
        or collection.get("status") != "validate_only"
        or collection.get("source_kind") != "sealed_successor_evidence"
    ):
        raise CanaryFailure("dual sealed successor collection identity drifted")
    if not all(
        collection.get(name) is True
        for name in (
            "configured",
            "available",
            "integrity_verified",
            "display_only",
            "read_only",
        )
    ):
        raise CanaryFailure("dual sealed successor collection failed closed")
    if collection.get("candidate_count") != 2:
        raise CanaryFailure("dual sealed successor candidate count drifted")
    _require_false_flags(
        collection,
        (*DUAL_SEALED_FALSE_FLAGS, "scheduler_mutation_performed"),
        "dual sealed successor collection",
    )
    collection_integrity = collection.get("integrity")
    if not isinstance(collection_integrity, dict):
        raise CanaryFailure("dual sealed successor integrity is missing")
    for api_name, config_name in (
        ("validateonly_plan_sha256", "dual_successor_plan"),
        ("read_only_ui_sha256", "dual_successor_ui"),
        ("validation_receipt_sha256", "dual_successor_receipt"),
    ):
        if collection_integrity.get(api_name) != expected[config_name]["sha256"]:
            raise CanaryFailure(f"dual sealed successor {api_name} drifted")

    candidates = collection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise CanaryFailure("dual sealed successor candidate inventory is malformed")
    candidates_by_sha: dict[str, dict[str, Any]] = {}
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise CanaryFailure(
                f"dual sealed successor candidate {position} is malformed"
            )
        if (
            candidate.get("schema_version") != "mft-monitor-sealed-successor-v1"
            or candidate.get("status") != "validate_only"
            or candidate.get("lifecycle_state") != "sealed_successor"
            or candidate.get("source_kind") != "sealed_successor_evidence"
            or candidate.get("fea_status") != "not_submitted"
        ):
            raise CanaryFailure(
                f"dual sealed successor candidate {position} identity drifted"
            )
        if not all(
            candidate.get(name) is True
            for name in (
                "configured",
                "available",
                "integrity_verified",
                "display_only",
                "read_only",
            )
        ):
            raise CanaryFailure(
                f"dual sealed successor candidate {position} failed closed"
            )
        _require_false_flags(
            candidate,
            DUAL_SEALED_CANDIDATE_FALSE_FLAGS,
            f"dual sealed successor candidate {position}",
        )
        if candidate.get("scheduler_task_id") is not None:
            raise CanaryFailure(
                "sealed successor scheduler_task_id must remain empty; "
                "Standard-FEA identity belongs only to its overlay"
            )
        design = candidate.get("candidate")
        candidate_sha = str(
            design.get("candidate_sha256") if isinstance(design, dict) else ""
        ).lower()
        if candidate_sha not in EXPECTED_DUAL_STANDARD_FEA_TASKS:
            raise CanaryFailure(
                f"dual sealed successor candidate {position} identity drifted"
            )
        if candidate_sha in candidates_by_sha:
            raise CanaryFailure("dual sealed successor candidate identity repeated")
        candidates_by_sha[candidate_sha] = candidate
    if set(candidates_by_sha) != set(EXPECTED_DUAL_STANDARD_FEA_TASKS):
        raise CanaryFailure("dual sealed successor exact/interior inventory drifted")

    validation = collection.get("standard_fea_validation")
    if not isinstance(validation, dict) or not all(
        validation.get(name) is True
        for name in ("configured", "available", "integrity_verified", "read_only")
    ):
        raise CanaryFailure("dual Standard-FEA overlay failed closed")
    if validation.get("schema_version") != "mft-monitor-dual-standard-fea-runtime-v1":
        raise CanaryFailure("dual Standard-FEA overlay schema drifted")
    if validation.get("candidate_count") != 2:
        raise CanaryFailure("dual Standard-FEA candidate count drifted")
    expected_validation = expected["standard_fea_validation"]
    try:
        api_runtime_root = Path(str(validation.get("runtime_root"))).resolve(
            strict=True
        )
    except OSError as exc:
        raise CanaryFailure(
            "dual Standard-FEA API runtime root is unavailable"
        ) from exc
    if api_runtime_root != Path(expected_validation["runtime_root"]):
        raise CanaryFailure("dual Standard-FEA API runtime root drifted")
    validation_integrity = validation.get("integrity")
    if not isinstance(validation_integrity, dict):
        raise CanaryFailure("dual Standard-FEA integrity projection is missing")
    expected_integrity = {
        "manifest_sha256": expected_validation["runtime_manifest"]["sha256"],
        "sealed_plan_source_sha256": expected_validation["sealed_plan_source"][
            "sha256"
        ],
        "execution_plan_sha256": expected_validation["execution_plan"]["sha256"],
        "allowed_root": expected_validation["allowed_root"],
    }
    if any(
        validation_integrity.get(name) != value
        for name, value in expected_integrity.items()
    ):
        raise CanaryFailure("dual Standard-FEA runtime/plan/source identity drifted")
    for name in ("state_sha256", "status_sha256"):
        _exact_sha256(validation_integrity.get(name), f"dual Standard-FEA {name}")
    overlay_candidates = validation.get("candidates")
    if not isinstance(overlay_candidates, dict) or set(overlay_candidates) != set(
        EXPECTED_DUAL_STANDARD_FEA_TASKS
    ):
        raise CanaryFailure("dual Standard-FEA overlay candidate mapping drifted")

    task_evidence: list[dict[str, Any]] = []
    runtime_root = Path(expected_validation["runtime_root"])
    for candidate_sha, identity in sorted(EXPECTED_DUAL_STANDARD_FEA_TASKS.items()):
        candidate = candidates_by_sha[candidate_sha]
        overlay = candidate.get("standard_fea_validation")
        if (
            not isinstance(overlay, dict)
            or overlay != overlay_candidates[candidate_sha]
        ):
            raise CanaryFailure(
                f"dual Standard-FEA card/top overlay disagreed: {candidate_sha}"
            )
        if overlay.get("schema_version") != "mft-monitor-standard-fea-validation-v1":
            raise CanaryFailure(
                f"dual Standard-FEA candidate schema drifted: {candidate_sha}"
            )
        if not all(
            overlay.get(name) is True
            for name in (
                "configured",
                "available",
                "integrity_verified",
                "read_only",
            )
        ):
            raise CanaryFailure(
                f"dual Standard-FEA candidate overlay failed closed: {candidate_sha}"
            )
        if overlay.get("candidate_digest") != identity["candidate_digest"]:
            raise CanaryFailure(
                f"dual Standard-FEA candidate digest drifted: {candidate_sha}"
            )
        task = overlay.get("task")
        if not isinstance(task, dict) or (
            task.get("id") != identity["task_id"]
            or task.get("name") != identity["task_name"]
            or task.get("status") != overlay.get("task_status")
            or task.get("status") not in EXPECTED_DUAL_STANDARD_FEA_STATUSES
        ):
            raise CanaryFailure(
                f"dual Standard-FEA task identity/status drifted: {candidate_sha}"
            )
        _require_false_flags(
            overlay,
            ("pareto_eligible", "production_eligible"),
            f"dual Standard-FEA candidate overlay {candidate_sha}",
        )
        result = overlay.get("result_identity")
        if not isinstance(result, dict):
            raise CanaryFailure(
                f"dual Standard-FEA result identity is missing: {candidate_sha}"
            )
        result_available = result.get("available") is True
        if result_available:
            if (
                result.get("task_id") != identity["task_id"]
                or result.get("candidate_digest") != identity["candidate_digest"]
                or result.get("candidate_identity_matches") is not True
                or not isinstance(result.get("contract_valid"), bool)
            ):
                raise CanaryFailure(
                    f"dual Standard-FEA result identity drifted: {candidate_sha}"
                )
            result_sha = _exact_sha256(
                result.get("sha256"),
                f"dual Standard-FEA task {identity['task_id']} result SHA-256",
            )
            expected_result_path = (
                runtime_root / "results" / f"task-{identity['task_id']}.json"
            )
            if Path(str(result.get("path"))) != expected_result_path:
                raise CanaryFailure(
                    f"dual Standard-FEA result path drifted: {candidate_sha}"
                )
            _contained_runtime_file(
                runtime_root,
                Path("results") / f"task-{identity['task_id']}.json",
                result_sha,
                label=f"dual Standard-FEA task {identity['task_id']} result",
                max_bytes=16 * 1024 * 1024,
            )
        elif any(
            (
                result.get("path") is not None,
                result.get("sha256") is not None,
                result.get("contract_valid") is not False,
                result.get("candidate_identity_matches") is not False,
            )
        ):
            raise CanaryFailure(
                f"dual Standard-FEA unavailable result leaks identity: {candidate_sha}"
            )

        collected = overlay.get("collection_state") == "collector_succeeded"
        if overlay.get("collected") is not collected:
            raise CanaryFailure(
                f"dual Standard-FEA collector lifecycle drifted: {candidate_sha}"
            )
        expected_valid = bool(
            task["status"] == "completed"
            and collected
            and result_available
            and result.get("contract_valid") is True
            and result.get("candidate_identity_matches") is True
        )
        if overlay.get("valid") is not expected_valid:
            raise CanaryFailure(
                f"dual Standard-FEA completed/collector validity drifted: {candidate_sha}"
            )
        if task["status"] in {"queued", "attaching", "running"} and (
            result_available or collected or overlay.get("valid") is True
        ):
            raise CanaryFailure(
                f"dual Standard-FEA active task exposes a terminal result: {candidate_sha}"
            )
        actual = _verify_projected_actual_t120_gate(
            overlay.get("actual_gates"), label=f"task {identity['task_id']}"
        )
        if (actual is None) is not (not result_available):
            raise CanaryFailure(
                f"dual Standard-FEA result/actual-gate lifecycle drifted: {candidate_sha}"
            )
        expected_actual_pass = actual["pass"] if actual is not None else None
        if overlay.get("actual_hard_gates_pass") is not expected_actual_pass:
            raise CanaryFailure(
                f"dual Standard-FEA actual gate projection drifted: {candidate_sha}"
            )
        if overlay.get("full_model_validation_candidate_eligible") is True and not (
            expected_valid and expected_actual_pass is True
        ):
            raise CanaryFailure(
                f"dual Standard-FEA full-model eligibility escaped its gate: {candidate_sha}"
            )
        overlay_integrity = overlay.get("integrity")
        if not isinstance(overlay_integrity, dict) or (
            overlay_integrity.get("manifest_sha256")
            != expected_integrity["manifest_sha256"]
            or overlay_integrity.get("execution_plan_sha256")
            != expected_integrity["execution_plan_sha256"]
        ):
            raise CanaryFailure(
                f"dual Standard-FEA candidate integrity drifted: {candidate_sha}"
            )
        task_evidence.append(
            {
                "candidate_sha256": candidate_sha,
                "candidate_digest": identity["candidate_digest"],
                "task_id": identity["task_id"],
                "task_name": identity["task_name"],
                "task_status": task["status"],
                "collection_state": overlay.get("collection_state"),
                "result_available": result_available,
                "result_sha256": result.get("sha256") if result_available else None,
                "valid": expected_valid,
                "actual_t120": actual,
                "full_model_validation_candidate_eligible": overlay.get(
                    "full_model_validation_candidate_eligible"
                )
                is True,
                "warning": overlay.get("warning"),
            }
        )
    return {
        "available": True,
        "integrity_verified": True,
        "read_only": True,
        "sealed_authority_flags_verified_false": True,
        "runtime_root": expected_validation["runtime_root"],
        "allowed_root": expected_validation["allowed_root"],
        "runtime_manifest_sha256": expected_integrity["manifest_sha256"],
        "execution_plan_sha256": expected_integrity["execution_plan_sha256"],
        "sealed_plan_source_sha256": expected_integrity["sealed_plan_source_sha256"],
        "dual_successor_plan_sha256": expected["dual_successor_plan"]["sha256"],
        "dual_successor_ui_sha256": expected["dual_successor_ui"]["sha256"],
        "dual_successor_receipt_sha256": expected["dual_successor_receipt"]["sha256"],
        "candidate_count": 2,
        "task_ids": sorted(item["task_id"] for item in task_evidence),
        "tasks": task_evidence,
        "runtime_state": validation.get("runtime_state"),
        "updated_at": validation.get("updated_at"),
        "warning": validation.get("warning"),
    }


def _verify_nsga_api(
    payload: dict[str, Any],
    expected_dual_validation: dict[str, Any] | None = None,
    *,
    expected_current7_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if expected_current7_index is not None:
        return _verify_current7_authoritative_nsga_api(
            payload,
            expected_current7_index,
            expected_dual_validation=expected_dual_validation,
        )
    if payload.get("schema_version") != 1 or payload.get("available") is not True:
        raise CanaryFailure("/api/nsga2 authoritative payload is unavailable")
    note = _require_utf8_display_text(payload.get("note"), "/api/nsga2 note")
    search = payload.get("tier1_feedback_search")
    if not isinstance(search, dict):
        raise CanaryFailure("/api/nsga2 rolling search status is missing")
    if (
        search.get("available") is not True
        or search.get("integrity_verified") is not True
    ):
        raise CanaryFailure("/api/nsga2 rolling search integrity is not verified")
    counts = {
        name: _nonnegative_integer(search.get(name), f"nsga.{name}")
        for name in (
            "running_count",
            "queued_count",
            "attaching_count",
            "active_plus_queued",
            "completed_count",
            "terminal_results_verified",
            "feasible_pareto_count",
            "near_feasible_count",
        )
    }
    if counts["active_plus_queued"] != (
        counts["running_count"] + counts["queued_count"] + counts["attaching_count"]
    ):
        raise CanaryFailure("/api/nsga2 active/queued counters disagree")
    if counts["completed_count"] != counts["terminal_results_verified"]:
        raise CanaryFailure("/api/nsga2 terminal verification counters disagree")
    candidate_preview_count = _nonnegative_integer(
        search.get("candidate_preview_count"),
        "nsga.candidate_preview_count",
    )
    candidate_preview_limit = _nonnegative_integer(
        search.get("candidate_preview_limit"),
        "nsga.candidate_preview_limit",
    )
    candidate_preview_truncated = search.get("candidate_preview_truncated")
    if (
        candidate_preview_limit != EXPECTED_TIER1_CANDIDATE_PREVIEW_LIMIT
        or candidate_preview_count
        != min(counts["feasible_pareto_count"], candidate_preview_limit)
        or candidate_preview_truncated
        is not (counts["feasible_pareto_count"] > candidate_preview_count)
    ):
        raise CanaryFailure("/api/nsga2 bounded candidate-preview contract disagrees")
    preview_count = _nonnegative_integer(
        search.get("near_feasible_preview_count"),
        "nsga.near_feasible_preview_count",
    )
    preview_limit = _nonnegative_integer(
        search.get("near_feasible_preview_limit"),
        "nsga.near_feasible_preview_limit",
    )
    preview_truncated = search.get("near_feasible_preview_truncated")
    expected_preview_count = min(counts["near_feasible_count"], preview_limit)
    if (
        preview_limit != EXPECTED_TIER1_NEAR_PREVIEW_LIMIT
        or preview_count != expected_preview_count
        or preview_truncated is not (counts["near_feasible_count"] > preview_count)
    ):
        raise CanaryFailure("/api/nsga2 bounded near-preview contract disagrees")
    if search.get("coherent_snapshot_verified") is not True:
        raise CanaryFailure("/api/nsga2 rolling snapshot is not coherent")
    coherent_attempts = _nonnegative_integer(
        search.get("coherent_snapshot_attempts"),
        "nsga.coherent_snapshot_attempts",
    )
    if coherent_attempts < 1 or coherent_attempts > 8:
        raise CanaryFailure("/api/nsga2 coherent snapshot attempts are invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise CanaryFailure("/api/nsga2 candidate list is malformed")
    candidate_count = _nonnegative_integer(
        payload.get("candidate_count"), "nsga.candidate_count"
    )
    if candidate_count != len(candidates):
        raise CanaryFailure("/api/nsga2 candidate count does not match its rows")
    constraint_version = search.get("constraint_version")
    hard_spec = search.get("constraints")
    temperature_contract = search.get("temperature_constraint_contract")
    if not isinstance(hard_spec, dict):
        raise CanaryFailure("/api/nsga2 hard-spec contract is missing")
    if (
        hard_spec.get("T_limit_C") != EXPECTED_TIER1_TEMPERATURE_LIMIT_C
        or hard_spec.get("resonance_min_Hz") != EXPECTED_TIER1_RESONANCE_MIN_HZ
    ):
        raise CanaryFailure("/api/nsga2 T120/res10k hard spec diverged")
    if not isinstance(temperature_contract, dict):
        raise CanaryFailure("/api/nsga2 temperature constraint contract is missing")
    temperature_targets = temperature_contract.get("targets")
    if (
        temperature_contract.get("robust_upper_bound_C")
        != EXPECTED_TIER1_TEMPERATURE_LIMIT_C
        or _nonnegative_integer(
            temperature_contract.get("target_count"),
            "nsga.temperature_constraint_contract.target_count",
        )
        != len(EXPECTED_TIER1_TEMPERATURE_TARGETS)
        or temperature_targets != list(EXPECTED_TIER1_TEMPERATURE_TARGETS)
    ):
        raise CanaryFailure("/api/nsga2 11-target thermal contract diverged")
    source_model_sha = str(search.get("model_manifest_sha256") or "").lower()
    deployment_model_sha = str(
        search.get("deployment_model_manifest_sha256") or ""
    ).lower()
    if not isinstance(constraint_version, str) or not constraint_version.strip():
        raise CanaryFailure("/api/nsga2 constraint identity is missing")
    if not FULL_SHA256.fullmatch(source_model_sha):
        raise CanaryFailure("/api/nsga2 source model hash is malformed")
    if not FULL_SHA256.fullmatch(deployment_model_sha):
        raise CanaryFailure("/api/nsga2 deployment model hash is malformed")
    dual_validation = (
        _verify_dual_standard_fea(payload, expected_dual_validation)
        if expected_dual_validation is not None
        else None
    )
    return {
        "available": True,
        "integrity_verified": True,
        "constraint_version": constraint_version,
        "model_manifest_sha256": source_model_sha,
        "deployment_model_manifest_sha256": deployment_model_sha,
        "candidate_count": candidate_count,
        "constraints": hard_spec,
        "temperature_constraint_contract": temperature_contract,
        "near_feasible_preview_count": preview_count,
        "near_feasible_preview_limit": preview_limit,
        "near_feasible_preview_truncated": preview_truncated,
        "candidate_preview_count": candidate_preview_count,
        "candidate_preview_limit": candidate_preview_limit,
        "candidate_preview_truncated": candidate_preview_truncated,
        "coherent_snapshot_verified": True,
        "coherent_snapshot_attempts": coherent_attempts,
        "note": note,
        "utf8_roundtrip_verified": True,
        "dual_standard_fea_validation": dual_validation,
        **counts,
        "updated_at": search.get("updated_at"),
    }


def _verify_current7_api(
    payload: dict[str, Any],
    configured_index: dict[str, Any],
) -> dict[str, Any]:
    """Verify the additive corrected-current7 projection when configured."""

    search = payload.get("tier1_current7_search")
    if not isinstance(search, dict):
        raise CanaryFailure("/api/nsga2 current7 search projection is missing")
    if not all(
        search.get(name) is True
        for name in ("available", "integrity_verified", "healthy")
    ):
        raise CanaryFailure("/api/nsga2 current7 search failed closed")
    try:
        source = Path(str(search.get("source") or "")).resolve(strict=True)
        expected_source = Path(configured_index["path"]).resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure("/api/nsga2 current7 source is unavailable") from exc
    if source != expected_source:
        raise CanaryFailure("/api/nsga2 current7 source path diverged")
    if search.get("pointer_verified") is not True:
        raise CanaryFailure("/api/nsga2 current7 pointer is not verified")
    if search.get("gui_launch_eligible") is not False:
        raise CanaryFailure("/api/nsga2 current7 unexpectedly enabled GUI launch")
    if search.get("warnings") not in ([], None):
        raise CanaryFailure("/api/nsga2 current7 projection contains warnings")

    count_names = (
        "search_count",
        "running_count",
        "queued_count",
        "attaching_count",
        "completed_count",
        "failed_count",
        "cancelled_count",
        "timeout_count",
        "active_plus_queued",
        "authenticated_terminal_seed_count",
        "refused_terminal_count",
        "feasible_pareto_count",
        "near_feasible_count",
        "candidate_preview_count",
        "candidate_preview_limit",
    )
    counts = {
        name: _nonnegative_integer(search.get(name), f"current7.{name}")
        for name in count_names
    }
    if counts["active_plus_queued"] != sum(
        counts[name] for name in ("running_count", "queued_count", "attaching_count")
    ):
        raise CanaryFailure("/api/nsga2 current7 active counters disagree")
    state_counts = search.get("state_counts")
    if not isinstance(state_counts, dict):
        raise CanaryFailure("/api/nsga2 current7 state counts are missing")
    normalized_states = {
        str(name): _nonnegative_integer(value, f"current7.state_counts.{name}")
        for name, value in state_counts.items()
    }
    if not set(normalized_states).issubset(CURRENT7_TASK_STATES):
        raise CanaryFailure("/api/nsga2 current7 state names diverged")
    if sum(normalized_states.values()) != counts["search_count"]:
        raise CanaryFailure("/api/nsga2 current7 search count disagrees")
    for name in (
        "running", "queued", "attaching", "completed", "failed",
        "cancelled", "timeout",
    ):
        if normalized_states.get(name, 0) != counts[f"{name}_count"]:
            raise CanaryFailure(f"/api/nsga2 current7 {name} count disagrees")
    if counts["refused_terminal_count"] != 0:
        raise CanaryFailure("/api/nsga2 current7 has refused terminal results")
    if counts["completed_count"] != counts["authenticated_terminal_seed_count"]:
        raise CanaryFailure("/api/nsga2 current7 terminal counters disagree")
    if (
        counts["candidate_preview_limit"] != CURRENT7_CANDIDATE_PREVIEW_LIMIT
        or counts["candidate_preview_count"]
        != min(
            counts["feasible_pareto_count"],
            counts["candidate_preview_limit"],
        )
        or search.get("candidate_preview_truncated")
        is not (counts["feasible_pareto_count"] > counts["candidate_preview_count"])
    ):
        raise CanaryFailure("/api/nsga2 current7 candidate preview diverged")
    active_task_count = sum(
        counts[name] for name in ("running_count", "queued_count", "attaching_count")
    )
    freshness_required = search.get("freshness_required")
    freshness_ok = search.get("freshness_ok")
    freshness_age_seconds = _finite_number(
        search.get("freshness_age_seconds"),
        "current7.freshness_age_seconds",
    )
    harvest_observed_at = str(search.get("harvest_observed_at") or "")
    status_event_at = str(search.get("status_event_at") or "")
    if (
        freshness_required is not (active_task_count > 0)
        or freshness_ok is not True
        or harvest_observed_at != configured_index["harvest_observed_at"]
        or status_event_at != configured_index["status_event_at"]
        or search.get("updated_at") != status_event_at
        or (
            active_task_count > 0
            and not (
                -CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS
                <= freshness_age_seconds
                <= CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS
            )
        )
    ):
        raise CanaryFailure("/api/nsga2 current7 freshness contract diverged")
    snapshot_sha = _exact_sha256(
        search.get("snapshot_sha256"), "/api/nsga2 current7 snapshot"
    )
    bundle_manifest_sha = _exact_sha256(
        search.get("bundle_manifest_sha256"),
        "/api/nsga2 current7 bundle manifest",
    )
    index_file_sha = _exact_sha256(
        search.get("index_file_sha256"),
        "/api/nsga2 current7 index file",
    )
    snapshot_file_sha = _exact_sha256(
        search.get("snapshot_file_sha256"),
        "/api/nsga2 current7 snapshot file",
    )
    constraint_identity_sha = _exact_sha256(
        search.get("constraint_identity_sha256"),
        "/api/nsga2 current7 constraint identity",
    )
    temperature_contract_sha = _exact_sha256(
        search.get("temperature_contract_sha256"),
        "/api/nsga2 current7 temperature contract",
    )
    if (
        snapshot_sha != configured_index["snapshot_sha256"]
        or bundle_manifest_sha != configured_index["bundle_manifest_sha256"]
        or index_file_sha != configured_index["index_file_sha256"]
        or snapshot_file_sha != configured_index["snapshot_file_sha256"]
        or constraint_identity_sha != configured_index["constraint_identity_sha256"]
        or temperature_contract_sha
        != configured_index["constraint_identity"]["temperature_contract_sha256"]
        or search.get("cohort_id") != configured_index["active_cohort_id"]
        or search.get("bundle_id") != configured_index["bundle_id"]
    ):
        raise CanaryFailure("/api/nsga2 current7 configured identity diverged")
    attempts = _nonnegative_integer(
        search.get("snapshot_attempts"), "current7.snapshot_attempts"
    )
    if not 1 <= attempts <= 8:
        raise CanaryFailure("/api/nsga2 current7 snapshot attempts are invalid")

    constraints = search.get("constraints")
    constraint_names = search.get("constraint_names")
    temperatures = search.get("temperature_targets")
    if (
        constraints != CURRENT7_HARD_SPEC
        or search.get("hard_spec_sha256") != CURRENT7_HARD_SPEC_SHA256
        or constraint_names != list(CURRENT7_CONSTRAINT_NAMES)
        or temperatures != list(CURRENT7_TEMPERATURE_TARGETS)
        or search.get("constraint_version") != configured_index["constraint_version"]
        or search.get("constraint_contract_verified") is not True
        or search.get("authority_eligible") is not True
    ):
        raise CanaryFailure("/api/nsga2 current7 hard-spec identity diverged")
    return {
        "available": True,
        "integrity_verified": True,
        "healthy": True,
        "authority_eligible": True,
        "constraint_contract_verified": True,
        "source": str(source),
        "configured_index_sha256": configured_index["sha256"],
        "index_file_sha256": index_file_sha,
        "snapshot_file_sha256": snapshot_file_sha,
        "snapshot_sha256": snapshot_sha,
        "cohort_id": search.get("cohort_id"),
        "bundle_id": search.get("bundle_id"),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "constraint_identity_sha256": constraint_identity_sha,
        "temperature_contract_sha256": temperature_contract_sha,
        "constraint_version": search.get("constraint_version"),
        "hard_spec_sha256": search.get("hard_spec_sha256"),
        "constraints": constraints,
        "constraint_names": constraint_names,
        "temperature_targets": temperatures,
        "updated_at": search.get("updated_at"),
        "harvest_observed_at": harvest_observed_at,
        "status_event_at": status_event_at,
        "state_counts": normalized_states,
        "snapshot_attempts": attempts,
        "freshness_required": freshness_required,
        "freshness_ok": freshness_ok,
        "freshness_age_seconds": freshness_age_seconds,
        "candidate_preview_truncated": search.get("candidate_preview_truncated"),
        **counts,
    }


def _verify_current7_condition_searches(
    payload: dict[str, Any],
    configured_indexes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    searches = payload.get("tier1_current7_condition_searches")
    if not isinstance(searches, list):
        raise CanaryFailure("/api/nsga2 current7 condition searches are missing")
    if len(searches) != len(configured_indexes):
        raise CanaryFailure("/api/nsga2 current7 condition search count diverged")
    verified: list[dict[str, Any]] = []
    for position, (search, configured) in enumerate(
        zip(searches, configured_indexes, strict=True)
    ):
        label = f"/api/nsga2 current7 condition {position}"
        if not isinstance(search, dict):
            raise CanaryFailure(f"{label} projection is malformed")
        try:
            source = Path(str(search.get("source") or "")).resolve(strict=True)
            expected_source = Path(configured["path"]).resolve(strict=True)
        except OSError as exc:
            raise CanaryFailure(f"{label} source is unavailable") from exc
        if source != expected_source:
            raise CanaryFailure(f"{label} source path diverged")
        if not all(
            search.get(name) is True
            for name in (
                "configured",
                "available",
                "integrity_verified",
                "constraint_contract_verified",
                "display_only",
                "read_only",
                "pointer_verified",
            )
        ):
            raise CanaryFailure(f"{label} integrity contract failed closed")
        if search.get("authority_eligible") is not False:
            raise CanaryFailure(f"{label} unexpectedly asserted authority")
        if search.get("gui_launch_eligible") is not False:
            raise CanaryFailure(f"{label} unexpectedly enabled GUI launch")
        if search.get("lanes") not in ([], None):
            raise CanaryFailure(f"{label} unexpectedly exposed execution lanes")
        if search.get("index_file_sha256") != configured["sha256"]:
            raise CanaryFailure(f"{label} index generation diverged")
        if search.get("snapshot_file_sha256") != configured["status"]["sha256"]:
            raise CanaryFailure(f"{label} status generation diverged")
        for name in (
            "constraint_version",
            "hard_spec_sha256",
            "hard_constraint_contract_sha256",
            "temperature_contract_sha256",
            "bundle_manifest_sha256",
            "snapshot_sha256",
        ):
            if search.get(name) != configured[name]:
                raise CanaryFailure(f"{label} {name} diverged")

        state_counts = search.get("state_counts")
        if not isinstance(state_counts, dict):
            raise CanaryFailure(f"{label} state counts are missing")
        normalized_states = {
            name: _nonnegative_integer(value, f"{label}.state_counts.{name}")
            for name, value in state_counts.items()
            if name in CURRENT7_TASK_STATES
        }
        search_count = _nonnegative_integer(search.get("search_count"), label)
        if search_count != sum(normalized_states.values()):
            raise CanaryFailure(f"{label} task counts disagree")
        running_count = _nonnegative_integer(search.get("running_count"), label)
        queued_count = _nonnegative_integer(search.get("queued_count"), label)
        attaching_count = _nonnegative_integer(search.get("attaching_count"), label)
        active_plus_queued = _nonnegative_integer(
            search.get("active_plus_queued"), label
        )
        if active_plus_queued != running_count + queued_count + attaching_count:
            raise CanaryFailure(f"{label} active counters disagree")
        if (
            running_count != normalized_states.get("running", 0)
            or queued_count != normalized_states.get("queued", 0)
            or attaching_count != normalized_states.get("attaching", 0)
        ):
            raise CanaryFailure(f"{label} projected states disagree")
        warnings = search.get("warnings")
        if not isinstance(warnings, list) or not all(
            isinstance(item, str) and item for item in warnings
        ):
            raise CanaryFailure(f"{label} warnings are malformed")
        verified.append({
            "position": position,
            "path": configured["path"],
            "index_file_sha256": configured["sha256"],
            "index_size_bytes": configured["size_bytes"],
            "snapshot_file_sha256": configured["status"]["sha256"],
            "snapshot_size_bytes": configured["status"]["size_bytes"],
            "status_max_bytes": configured["status_max_bytes"],
            "constraint_version": configured["constraint_version"],
            "hard_spec_sha256": configured["hard_spec_sha256"],
            "bundle_manifest_sha256": configured["bundle_manifest_sha256"],
            "snapshot_sha256": configured["snapshot_sha256"],
            "search_count": search_count,
            "running_count": running_count,
            "queued_count": queued_count,
            "attaching_count": attaching_count,
            "completed_count": _nonnegative_integer(
                search.get("completed_count"), label
            ),
            "failed_count": _nonnegative_integer(search.get("failed_count"), label),
            "refused_terminal_count": _nonnegative_integer(
                search.get("refused_terminal_count"), label
            ),
            "feasible_pareto_count": _nonnegative_integer(
                search.get("feasible_pareto_count"), label
            ),
            "near_feasible_count": _nonnegative_integer(
                search.get("near_feasible_count"), label
            ),
            "healthy": search.get("healthy") is True,
            "freshness_required": search.get("freshness_required") is True,
            "freshness_ok": search.get("freshness_ok") is True,
            "warnings": warnings,
            "read_only": True,
            "authority_eligible": False,
        })
    return verified


def _verify_current7_condition_archive_nsga_api(
    payload: dict[str, Any],
    configured_indexes: list[dict[str, Any]],
    *,
    expected_dual_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise CanaryFailure("/api/nsga2 schema diverged in condition-archive mode")
    source_kind = payload.get("source_kind")
    deadline_mode = source_kind == "deadline_design_with_condition_archive"
    if not (
        payload.get("available") is True
        and source_kind in {
            "current7_condition_archive",
            "deadline_design_with_condition_archive",
        }
    ):
        raise CanaryFailure(
            "/api/nsga2 condition-archive boundary is unavailable or mislabeled"
        )
    authority = payload.get("search_authority")
    if not isinstance(authority, dict) or not (
        authority.get("configured") is False
        and authority.get("kind") == "condition_archive"
        and authority.get("available") is False
        and authority.get("integrity_verified") is False
        and authority.get("source") is None
        and authority.get("legacy_role") == "archive"
    ):
        raise CanaryFailure(
            "/api/nsga2 condition-archive authority boundary diverged"
        )
    legacy = payload.get("tier1_feedback_search")
    if not isinstance(legacy, dict) or not (
        legacy.get("authority_role") == "archive"
        and legacy.get("archived") is True
    ):
        raise CanaryFailure(
            "/api/nsga2 condition-archive legacy source is not archived"
        )
    conditions = _verify_current7_condition_searches(payload, configured_indexes)
    if not conditions:
        raise CanaryFailure("/api/nsga2 condition-archive mode has no conditions")
    near_preview = payload.get("near_feasible_preview")
    if not isinstance(near_preview, list):
        raise CanaryFailure(
            "/api/nsga2 condition-archive near preview is malformed"
        )
    near_preview_count = _nonnegative_integer(
        payload.get("near_feasible_preview_count"),
        "/api/nsga2 near preview count",
    )
    if near_preview_count != len(near_preview) or near_preview_count > 1:
        raise CanaryFailure(
            "/api/nsga2 condition-archive near preview count diverged"
        )
    dual_validation = (
        _verify_dual_standard_fea(payload, expected_dual_validation)
        if expected_dual_validation is not None
        else None
    )
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict) or not constraints:
        raise CanaryFailure(
            "/api/nsga2 condition-archive selected constraints are missing"
        )
    if deadline_mode:
        selected_condition = conditions[0]
    else:
        selected_hard_spec_sha256 = hashlib.sha256(
            _canonical_json(constraints)
        ).hexdigest()
        selected_constraint_version = str(
            payload.get("constraint_version") or ""
        ).strip()
        selected_conditions = [
            item for item in conditions
            if item["constraint_version"] == selected_constraint_version
            and item["hard_spec_sha256"] == selected_hard_spec_sha256
        ]
        if len(selected_conditions) != 1:
            raise CanaryFailure(
                "/api/nsga2 condition-archive selected condition identity diverged"
            )
        selected_condition = selected_conditions[0]
    return {
        "mode": (
            "deadline_design_with_condition_archive"
            if deadline_mode else "current7_condition_archive"
        ),
        "available": True,
        "integrity_verified": True,
        "read_only": True,
        "authority_configured": False,
        "condition_count": len(conditions),
        "conditions": conditions,
        "constraint_version": selected_condition["constraint_version"],
        "model_manifest_sha256": selected_condition[
            "bundle_manifest_sha256"
        ],
        "deployment_model_manifest_sha256": selected_condition[
            "bundle_manifest_sha256"
        ],
        "candidate_count": sum(
            item["feasible_pareto_count"] for item in conditions
        ),
        "search_count": sum(item["search_count"] for item in conditions),
        "running_count": sum(item["running_count"] for item in conditions),
        "queued_count": sum(item["queued_count"] for item in conditions),
        "attaching_count": sum(item["attaching_count"] for item in conditions),
        "active_plus_queued": sum(
            item["running_count"] + item["queued_count"] + item["attaching_count"]
            for item in conditions
        ),
        "completed_count": sum(item["completed_count"] for item in conditions),
        "failed_count": sum(item["failed_count"] for item in conditions),
        "feasible_pareto_count": sum(
            item["feasible_pareto_count"] for item in conditions
        ),
        "near_feasible_count": sum(
            item["near_feasible_count"] for item in conditions
        ),
        "near_feasible_preview_count": near_preview_count,
        "dual_standard_fea_validation": dual_validation,
    }


def _verify_current7_condition_archive_progress(
    payload: dict[str, Any], nsga: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise CanaryFailure("/api/nsga2/progress schema diverged")
    if payload.get("source_endpoint") != "/api/nsga2":
        raise CanaryFailure("/api/nsga2/progress source endpoint diverged")
    count_contract = {
        "search_count": nsga["search_count"],
        "running_count": nsga["running_count"],
        "queued_count": nsga["queued_count"],
        "attaching_count": nsga["attaching_count"],
        "active_plus_queued": nsga["active_plus_queued"],
        "completed_count": nsga["completed_count"],
        "failed_count": nsga["failed_count"],
        "feasible_pareto_count": nsga["feasible_pareto_count"],
        "candidate_count": nsga["candidate_count"],
        "near_feasible_count": nsga["near_feasible_count"],
        "near_feasible_preview_count": nsga[
            "near_feasible_preview_count"
        ],
    }
    if not (
        payload.get("available") is True
        and payload.get("source_kind") in {
            "current7_condition_archive",
            "deadline_design_with_condition_archive",
        }
        and payload.get("authority_kind") == "none"
        and payload.get("integrity_verified") is True
        and payload.get("condition_count") == nsga["condition_count"]
        and payload.get("source_count") == nsga["condition_count"]
        and payload.get("configured_source_count") == nsga["condition_count"]
        and payload.get("rejected_source_count") == 0
        and payload.get("all_configured_sources_verified") is True
        and payload.get("counters_consistent") is True
        and payload.get("candidate_preview_consistent") is True
        and all(
            payload.get(name) == value
            for name, value in count_contract.items()
        )
    ):
        raise CanaryFailure("/api/nsga2/progress counters are inconsistent")
    return {
        "mode": payload.get("source_kind"),
        "available": payload.get("available") is True,
        "integrity_verified": nsga["integrity_verified"],
        "source_kind": payload.get("source_kind"),
        "status": payload.get("status"),
        "condition_count": nsga["condition_count"],
    }


def _verify_current7_authoritative_nsga_api(
    payload: dict[str, Any],
    configured_index: dict[str, Any],
    *,
    expected_dual_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify current7 as primary without treating the legacy archive as live."""

    if payload.get("schema_version") != 1 or payload.get("available") is not True:
        raise CanaryFailure("/api/nsga2 authoritative payload is unavailable")
    note = _require_utf8_display_text(payload.get("note"), "/api/nsga2 note")
    authority = payload.get("search_authority")
    if not isinstance(authority, dict) or not (
        authority.get("configured") is True
        and authority.get("kind") == "current7"
        and authority.get("available") is True
        and authority.get("integrity_verified") is True
        and authority.get("legacy_role") == "archive"
    ):
        raise CanaryFailure("/api/nsga2 current7 search authority diverged")
    legacy = payload.get("tier1_feedback_search")
    if not isinstance(legacy, dict) or not (
        legacy.get("authority_role") == "archive" and legacy.get("archived") is True
    ):
        raise CanaryFailure("/api/nsga2 legacy search is not marked archive")

    current7 = _verify_current7_api(payload, configured_index)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise CanaryFailure("/api/nsga2 candidate list is malformed")
    candidate_count = _nonnegative_integer(
        payload.get("candidate_count"), "nsga.candidate_count"
    )
    if candidate_count != len(candidates):
        raise CanaryFailure("/api/nsga2 candidate count does not match its rows")
    dual_validation = (
        _verify_dual_standard_fea(payload, expected_dual_validation)
        if expected_dual_validation is not None
        else None
    )
    bundle_sha = current7["bundle_manifest_sha256"]
    return {
        "available": True,
        "integrity_verified": True,
        "authority_kind": "current7",
        "authority_identity_sha256": bundle_sha,
        "constraint_version": current7["constraint_version"],
        "model_manifest_sha256": bundle_sha,
        "deployment_model_manifest_sha256": bundle_sha,
        "candidate_count": candidate_count,
        "constraints": current7["constraints"],
        "constraint_names": current7["constraint_names"],
        "temperature_targets": current7["temperature_targets"],
        "temperature_constraint_contract": _current7_temperature_contract(
            current7["temperature_contract_sha256"]
        ),
        "temperature_contract_sha256": current7[
            "temperature_contract_sha256"
        ],
        "search_count": current7["search_count"],
        "running_count": current7["running_count"],
        "queued_count": current7["queued_count"],
        "attaching_count": current7["attaching_count"],
        "active_plus_queued": current7["active_plus_queued"],
        "completed_count": current7["completed_count"],
        "failed_count": current7["failed_count"],
        "cancelled_count": current7["cancelled_count"],
        "timeout_count": current7["timeout_count"],
        "terminal_results_verified": current7["authenticated_terminal_seed_count"],
        "refused_terminal_count": current7["refused_terminal_count"],
        "feasible_pareto_count": current7["feasible_pareto_count"],
        "candidate_preview_count": current7["candidate_preview_count"],
        "candidate_preview_limit": current7["candidate_preview_limit"],
        "candidate_preview_truncated": current7["candidate_preview_truncated"],
        "near_feasible_count": current7["near_feasible_count"],
        "freshness_required": current7["freshness_required"],
        "freshness_ok": current7["freshness_ok"],
        "freshness_age_seconds": current7["freshness_age_seconds"],
        "harvest_observed_at": current7["harvest_observed_at"],
        "status_event_at": current7["status_event_at"],
        "state_counts": current7["state_counts"],
        "snapshot_attempts": current7["snapshot_attempts"],
        "bundle_manifest_sha256": bundle_sha,
        "index_file_sha256": current7["index_file_sha256"],
        "snapshot_file_sha256": current7["snapshot_file_sha256"],
        "snapshot_sha256": current7["snapshot_sha256"],
        "constraint_identity_sha256": current7["constraint_identity_sha256"],
        "authority_eligible": current7["authority_eligible"],
        "constraint_contract_verified": current7["constraint_contract_verified"],
        "cohort_id": current7["cohort_id"],
        "bundle_id": current7["bundle_id"],
        "note": note,
        "utf8_roundtrip_verified": True,
        "dual_standard_fea_validation": dual_validation,
        "legacy_authority_role": "archive",
        "updated_at": current7["updated_at"],
    }


def _verify_blocker_hpo_v2_api(
    dashboard: dict[str, Any],
    configured_status: dict[str, Any],
) -> dict[str, Any]:
    """Bind the dashboard HPO projection to the configured status artifact."""

    pipeline = dashboard.get("continuous_pipeline")
    pipeline = pipeline if isinstance(pipeline, dict) else {}
    hpo = pipeline.get("blocker_hpo_v2")
    if not isinstance(hpo, dict) or hpo.get("available") is not True:
        raise CanaryFailure("/api/dashboard blocker HPO v2 projection is missing")
    if hpo.get("state") in {None, "unavailable", "invalid"}:
        raise CanaryFailure("/api/dashboard blocker HPO v2 failed closed")
    try:
        source = Path(str(hpo.get("source") or "")).resolve(strict=True)
        expected_source = Path(configured_status["path"]).resolve(strict=True)
    except OSError as exc:
        raise CanaryFailure(
            "/api/dashboard blocker HPO v2 source is unavailable"
        ) from exc
    if source != expected_source:
        raise CanaryFailure("/api/dashboard blocker HPO v2 source path diverged")
    if hpo.get("config_sha256") != configured_status["config_sha256"]:
        raise CanaryFailure("/api/dashboard blocker HPO v2 config hash diverged")
    if hpo.get("dataset_sha256") != configured_status["dataset_sha256"]:
        raise CanaryFailure("/api/dashboard blocker HPO v2 dataset hash diverged")
    job_count = _nonnegative_integer(hpo.get("job_count"), "hpo_v2.job_count")
    trial_total = _nonnegative_integer(hpo.get("trial_total"), "hpo_v2.trial_total")
    stage = _nonnegative_integer(
        hpo.get("stage_trials_per_job"), "hpo_v2.stage_trials_per_job"
    )
    if job_count <= 0 or stage <= 0 or trial_total != job_count * stage:
        raise CanaryFailure("/api/dashboard blocker HPO v2 budget disagrees")
    if (
        job_count != configured_status["job_count"]
        or stage != configured_status["stage_trials_per_job"]
        or trial_total != configured_status["trial_total"]
    ):
        raise CanaryFailure("/api/dashboard blocker HPO v2 sealed budget diverged")
    trial_counts = {
        name: _nonnegative_integer(hpo.get(name), f"hpo_v2.{name}")
        for name in ("trial_complete", "trial_running", "trial_failed")
    }
    observed = sum(trial_counts.values())
    if observed > trial_total:
        raise CanaryFailure("/api/dashboard blocker HPO v2 trial counts overflow")
    if hpo.get("state") != configured_status["phase"] or any(
        trial_counts[name] != configured_status[name] for name in trial_counts
    ):
        raise CanaryFailure("/api/dashboard blocker HPO v2 status snapshot diverged")
    return {
        "available": True,
        "source": str(source),
        "status_sha256": configured_status["sha256"],
        "state": hpo["state"],
        "validated_running": hpo.get("validated_running") is True,
        "config_sha256": hpo["config_sha256"],
        "dataset_sha256": hpo["dataset_sha256"],
        "job_count": job_count,
        "stage_trials_per_job": stage,
        "trial_total": trial_total,
        **trial_counts,
        "trial_observed": observed,
    }


def _wait_for_blocker_hpo_v2_api(
    base_url: str,
    configured_status: dict[str, Any],
    *,
    timeout_seconds: float = 2.0,
    poll_seconds: float = 0.02,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read one dashboard/status generation despite legitimate live updates."""

    if timeout_seconds < 0 or poll_seconds < 0:
        raise ValueError("invalid blocker HPO v2 stability gate bounds")
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    last_error = "no coherent blocker HPO v2 generation was observed"
    retryable_messages = (
        "/api/dashboard blocker HPO v2 sealed budget diverged",
        "/api/dashboard blocker HPO v2 status snapshot diverged",
    )
    while True:
        attempts += 1
        dashboard = _get_json(base_url, "/api/dashboard", 240)
        latest_status = _refresh_blocker_hpo_v2_status(configured_status)
        try:
            verified = _verify_blocker_hpo_v2_api(dashboard, latest_status)
        except CanaryFailure as exc:
            last_error = str(exc)
            if not any(message in last_error for message in retryable_messages):
                raise
        else:
            verified["stability_attempts"] = attempts
            return dashboard, verified, latest_status
        if time.monotonic() >= deadline:
            raise CanaryFailure(
                "blocker HPO v2 stability gate timed out without a coherent "
                f"dashboard/status generation: {last_error}"
            )
        time.sleep(poll_seconds)


def _verify_current7_nsga_progress(
    payload: dict[str, Any], nsga: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise CanaryFailure("/api/nsga2/progress schema version mismatch")
    if not all(
        payload.get(name) is True
        for name in ("available", "integrity_verified", "counters_consistent")
    ):
        raise CanaryFailure("/api/nsga2/progress current7 failed closed")
    if payload.get("source_endpoint") != "/api/nsga2":
        raise CanaryFailure("/api/nsga2/progress declares a separate truth source")
    if (
        payload.get("authority_kind") != "current7"
        or payload.get("authority_identity_sha256") != nsga["authority_identity_sha256"]
    ):
        raise CanaryFailure("NSGA progress current7 authority identity diverged")
    progress_freshness_age = _finite_number(
        payload.get("freshness_age_seconds"),
        "nsga_progress.freshness_age_seconds",
    )
    if (
        payload.get("freshness_required") is not nsga["freshness_required"]
        or payload.get("freshness_ok") is not True
        or payload.get("harvest_observed_at") != nsga["harvest_observed_at"]
        or payload.get("status_event_at") != nsga["status_event_at"]
        or (
            nsga["freshness_required"]
            and not (
                -CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS
                <= progress_freshness_age
                <= CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS
            )
        )
    ):
        raise CanaryFailure("NSGA progress current7 freshness contract diverged")

    count_names = (
        "search_count",
        "running_count",
        "queued_count",
        "attaching_count",
        "active_plus_queued",
        "completed_count",
        "failed_count",
        "cancelled_count",
        "timeout_count",
        "terminal_results_verified",
        "refused_terminal_count",
        "feasible_pareto_count",
        "near_feasible_count",
    )
    counts = {
        name: _nonnegative_integer(payload.get(name), f"nsga_progress.{name}")
        for name in count_names
    }
    if any(counts[name] != nsga[name] for name in count_names):
        raise CanaryFailure("NSGA progress current7 counters diverged")
    if (
        counts["active_plus_queued"]
        != counts["running_count"] + counts["queued_count"] + counts["attaching_count"]
        or counts["completed_count"] != counts["terminal_results_verified"]
        or counts["refused_terminal_count"] != 0
        or counts["search_count"]
        != counts["running_count"]
        + counts["queued_count"]
        + counts["attaching_count"]
        + counts["completed_count"]
        + counts["failed_count"]
        + counts["cancelled_count"]
        + counts["timeout_count"]
    ):
        raise CanaryFailure("NSGA progress current7 counter contract disagreed")

    exact_fields = (
        "constraint_version",
        "constraints",
        "temperature_targets",
        "temperature_constraint_contract",
        "constraint_identity_sha256",
        "model_manifest_sha256",
        "deployment_model_manifest_sha256",
    )
    if any(payload.get(name) != nsga[name] for name in exact_fields):
        raise CanaryFailure("NSGA progress current7 contract identity diverged")
    identity_projection = {
        "authority_identity_sha256": nsga["bundle_manifest_sha256"],
        "authority_index_file_sha256": nsga["index_file_sha256"],
        "authority_snapshot_sha256": nsga["snapshot_file_sha256"],
        "authority_snapshot_identity_sha256": nsga["snapshot_sha256"],
        "constraint_identity_sha256": nsga["constraint_identity_sha256"],
    }
    if any(payload.get(name) != value for name, value in identity_projection.items()):
        raise CanaryFailure("NSGA progress current7 snapshot identity diverged")
    state_counts = payload.get("state_counts")
    if state_counts != nsga["state_counts"]:
        raise CanaryFailure("NSGA progress current7 state counts diverged")
    candidate_preview_count = _nonnegative_integer(
        payload.get("candidate_preview_count"),
        "nsga_progress.candidate_preview_count",
    )
    candidate_preview_limit = _nonnegative_integer(
        payload.get("candidate_preview_limit"),
        "nsga_progress.candidate_preview_limit",
    )
    near_preview_count = _nonnegative_integer(
        payload.get("near_feasible_preview_count"),
        "nsga_progress.near_feasible_preview_count",
    )
    near_preview_limit = _nonnegative_integer(
        payload.get("near_feasible_preview_limit"),
        "nsga_progress.near_feasible_preview_limit",
    )
    if (
        payload.get("candidate_preview_consistent") is not True
        or candidate_preview_count != nsga["candidate_preview_count"]
        or candidate_preview_limit != nsga["candidate_preview_limit"]
        or payload.get("candidate_preview_truncated")
        is not nsga["candidate_preview_truncated"]
        or near_preview_limit != 1
        or near_preview_count != min(counts["near_feasible_count"], 1)
        or payload.get("near_feasible_preview_truncated")
        is not (counts["near_feasible_count"] > near_preview_count)
    ):
        raise CanaryFailure("NSGA progress current7 preview contract diverged")
    coherent_attempts = _nonnegative_integer(
        payload.get("coherent_snapshot_attempts"),
        "nsga_progress.coherent_snapshot_attempts",
    )
    if (
        payload.get("coherent_snapshot") is not True
        or payload.get("coherent_snapshot_verified") is not True
        or coherent_attempts != nsga["snapshot_attempts"]
    ):
        raise CanaryFailure("NSGA progress current7 snapshot is not coherent")
    if (
        payload.get("model_manifest_sha256") != nsga["bundle_manifest_sha256"]
        or payload.get("deployment_model_manifest_sha256")
        != nsga["bundle_manifest_sha256"]
        or payload.get("temperature_constraint_contract")
        != _current7_temperature_contract(nsga["temperature_contract_sha256"])
        or payload.get("temperature_targets") != list(CURRENT7_TEMPERATURE_TARGETS)
    ):
        raise CanaryFailure("NSGA progress current7 authority projection diverged")
    verified = {
        key: payload.get(key)
        for key in (
            "available",
            "integrity_verified",
            "counters_consistent",
            "status",
            "source_endpoint",
            "authority_kind",
            "authority_identity_sha256",
            "authority_index_file_sha256",
            "authority_snapshot_sha256",
            "authority_snapshot_identity_sha256",
            "freshness_required",
            "freshness_ok",
            "freshness_age_seconds",
            "harvest_observed_at",
            "status_event_at",
            *count_names,
            *exact_fields,
            "state_counts",
            "candidate_preview_consistent",
            "candidate_preview_count",
            "candidate_preview_limit",
            "candidate_preview_truncated",
            "near_feasible_preview_count",
            "near_feasible_preview_limit",
            "near_feasible_preview_truncated",
            "coherent_snapshot",
            "coherent_snapshot_verified",
            "coherent_snapshot_attempts",
            "updated_at",
        )
    }
    verified.update(
        {
            "bundle_manifest_sha256": payload["authority_identity_sha256"],
            "index_file_sha256": payload["authority_index_file_sha256"],
            "snapshot_file_sha256": payload["authority_snapshot_sha256"],
            "snapshot_sha256": payload["authority_snapshot_identity_sha256"],
            "constraint_names": list(nsga["constraint_names"]),
        }
    )
    return verified


def _verify_nsga_progress(
    payload: dict[str, Any], nsga: dict[str, Any]
) -> dict[str, Any]:
    if nsga.get("authority_kind") == "current7":
        return _verify_current7_nsga_progress(payload, nsga)
    if payload.get("schema_version") != 1:
        raise CanaryFailure("/api/nsga2/progress schema version mismatch")
    if not all(
        payload.get(key) is True
        for key in (
            "available",
            "integrity_verified",
            "counters_consistent",
            "candidate_preview_consistent",
        )
    ):
        raise CanaryFailure("/api/nsga2/progress failed closed")
    if payload.get("source_endpoint") != "/api/nsga2":
        raise CanaryFailure("/api/nsga2/progress declares a separate truth source")
    progress_counts = {}
    for name in (
        "running_count",
        "queued_count",
        "attaching_count",
        "active_plus_queued",
        "completed_count",
        "terminal_results_verified",
        "feasible_pareto_count",
        "near_feasible_count",
    ):
        progress_counts[name] = _nonnegative_integer(
            payload.get(name), f"nsga_progress.{name}"
        )
    if (
        progress_counts["active_plus_queued"]
        != (
            progress_counts["running_count"]
            + progress_counts["queued_count"]
            + progress_counts["attaching_count"]
        )
        or progress_counts["completed_count"]
        != progress_counts["terminal_results_verified"]
    ):
        raise CanaryFailure("NSGA progress counters disagree")
    candidate_preview_count = _nonnegative_integer(
        payload.get("candidate_preview_count"),
        "nsga_progress.candidate_preview_count",
    )
    candidate_preview_limit = _nonnegative_integer(
        payload.get("candidate_preview_limit"),
        "nsga_progress.candidate_preview_limit",
    )
    if (
        candidate_preview_limit != EXPECTED_TIER1_CANDIDATE_PREVIEW_LIMIT
        or candidate_preview_count
        != min(
            progress_counts["feasible_pareto_count"],
            candidate_preview_limit,
        )
        or payload.get("candidate_preview_truncated")
        is not (progress_counts["feasible_pareto_count"] > candidate_preview_count)
    ):
        raise CanaryFailure(
            "NSGA progress bounded candidate-preview contract disagreed"
        )
    preview_count = _nonnegative_integer(
        payload.get("near_feasible_preview_count"),
        "nsga_progress.near_feasible_preview_count",
    )
    preview_limit = _nonnegative_integer(
        payload.get("near_feasible_preview_limit"),
        "nsga_progress.near_feasible_preview_limit",
    )
    if (
        preview_limit != EXPECTED_TIER1_NEAR_PREVIEW_LIMIT
        or preview_count != min(progress_counts["near_feasible_count"], preview_limit)
        or payload.get("near_feasible_preview_truncated")
        is not (progress_counts["near_feasible_count"] > preview_count)
    ):
        raise CanaryFailure("NSGA progress bounded near-preview contract disagreed")
    if payload.get("coherent_snapshot_verified") is not True:
        raise CanaryFailure("NSGA progress rolling snapshot is not coherent")
    coherent_attempts = _nonnegative_integer(
        payload.get("coherent_snapshot_attempts"),
        "nsga_progress.coherent_snapshot_attempts",
    )
    if coherent_attempts < 1 or coherent_attempts > 8:
        raise CanaryFailure("NSGA progress coherent snapshot attempts are invalid")
    if payload.get("constraint_version") != nsga["constraint_version"]:
        raise CanaryFailure("NSGA progress constraint identity diverged")
    if payload.get("model_manifest_sha256") != nsga["model_manifest_sha256"]:
        raise CanaryFailure("NSGA progress source model identity diverged")
    if (
        payload.get("deployment_model_manifest_sha256")
        != nsga["deployment_model_manifest_sha256"]
    ):
        raise CanaryFailure("NSGA progress deployment model identity diverged")
    if payload.get("constraints") != nsga["constraints"]:
        raise CanaryFailure("NSGA progress T120/res10k hard spec diverged")
    if (
        payload.get("temperature_constraint_contract")
        != nsga["temperature_constraint_contract"]
    ):
        raise CanaryFailure("NSGA progress 11-target thermal contract diverged")
    return {
        key: payload.get(key)
        for key in (
            "available",
            "integrity_verified",
            "counters_consistent",
            "candidate_preview_consistent",
            "status",
            "active_plus_queued",
            "running_count",
            "queued_count",
            "attaching_count",
            "terminal_results_verified",
            "completed_count",
            "feasible_pareto_count",
            "near_feasible_count",
            "candidate_preview_count",
            "candidate_preview_limit",
            "candidate_preview_truncated",
            "near_feasible_preview_count",
            "near_feasible_preview_limit",
            "near_feasible_preview_truncated",
            "coherent_snapshot_verified",
            "coherent_snapshot_attempts",
            "constraints",
            "temperature_constraint_contract",
            "constraint_version",
            "model_manifest_sha256",
            "deployment_model_manifest_sha256",
            "updated_at",
        )
    }


def _nsga_generation_identity(
    nsga: dict[str, Any], progress: dict[str, Any]
) -> dict[str, Any] | None:
    if nsga.get("authority_kind") == "current7":
        fields = (
            "updated_at",
            "authority_kind",
            "authority_identity_sha256",
            "search_count",
            "running_count",
            "queued_count",
            "attaching_count",
            "active_plus_queued",
            "completed_count",
            "failed_count",
            "cancelled_count",
            "timeout_count",
            "terminal_results_verified",
            "refused_terminal_count",
            "feasible_pareto_count",
            "candidate_preview_count",
            "candidate_preview_limit",
            "candidate_preview_truncated",
            "near_feasible_count",
            "freshness_required",
            "freshness_ok",
            "harvest_observed_at",
            "status_event_at",
            "constraint_version",
            "bundle_manifest_sha256",
            "index_file_sha256",
            "snapshot_file_sha256",
            "snapshot_sha256",
            "constraint_identity_sha256",
            "model_manifest_sha256",
            "deployment_model_manifest_sha256",
        )
        nsga_identity = {name: nsga.get(name) for name in fields}
        progress_identity = {name: progress.get(name) for name in fields}
        if nsga_identity != progress_identity:
            return None
        nsga_identity["constraints_sha256"] = _sha256_bytes(
            _canonical_json(nsga["constraints"])
        )
        nsga_identity["constraint_names_sha256"] = _sha256_bytes(
            _canonical_json(nsga["constraint_names"])
        )
        nsga_identity["temperature_targets_sha256"] = _sha256_bytes(
            _canonical_json(nsga["temperature_targets"])
        )
        return nsga_identity
    fields = (
        "updated_at",
        "running_count",
        "queued_count",
        "attaching_count",
        "active_plus_queued",
        "completed_count",
        "terminal_results_verified",
        "feasible_pareto_count",
        "candidate_preview_count",
        "candidate_preview_limit",
        "candidate_preview_truncated",
        "near_feasible_count",
        "near_feasible_preview_count",
        "near_feasible_preview_limit",
        "near_feasible_preview_truncated",
        "constraint_version",
        "model_manifest_sha256",
        "deployment_model_manifest_sha256",
    )
    nsga_identity = {name: nsga.get(name) for name in fields}
    progress_identity = {name: progress.get(name) for name in fields}
    if nsga_identity != progress_identity:
        return None
    nsga_identity["constraints_sha256"] = _sha256_bytes(
        _canonical_json(nsga["constraints"])
    )
    nsga_identity["temperature_constraint_contract_sha256"] = _sha256_bytes(
        _canonical_json(nsga["temperature_constraint_contract"])
    )
    return nsga_identity


def _wait_for_stable_nsga_generation(
    base_url: str,
    *,
    expected_dual_validation: dict[str, Any] | None = None,
    expected_current7_index: dict[str, Any] | None = None,
    expected_current7_condition_indexes: list[dict[str, Any]] | None = None,
    timeout_seconds: float = NSGA_STABILITY_GATE_TIMEOUT_SECONDS,
    poll_seconds: float = NSGA_STABILITY_GATE_POLL_SECONDS,
    required_reads: int = NSGA_STABILITY_GATE_REQUIRED_READS,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Require two consecutive fresh reads of one exact NSGA generation."""
    if timeout_seconds < 0 or poll_seconds < 0 or required_reads < 2:
        raise ValueError("invalid NSGA stability gate bounds")
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    consecutive_reads = 0
    stable_identity: dict[str, Any] | None = None
    last_error = "no authoritative NSGA generation was observed"
    retryable_messages = (
        "/api/nsga2 authoritative payload is unavailable",
        "/api/nsga2 rolling search integrity is not verified",
        "/api/nsga2/progress failed closed",
        "current7 condition index changed during authentication",
        "current7 condition index authentication did not stabilize",
        "current7 condition 0 index generation diverged",
        "current7 condition 1 index generation diverged",
        "current7 condition 2 index generation diverged",
        "current7 condition 3 index generation diverged",
        "current7 condition 4 index generation diverged",
    )
    condition_mode = bool(
        expected_current7_condition_indexes
        and expected_current7_index is None
    )
    while True:
        attempts += 1
        try:
            nsga_payload = _get_json(base_url, "/api/nsga2", 120)
            progress_payload = _get_json(base_url, "/api/nsga2/progress", 120)
            current7_for_attempt = (
                _refresh_current7_index(expected_current7_index)
                if expected_current7_index is not None
                else None
            )
            conditions_for_attempt = (
                _authenticate_current7_condition_indexes([
                    Path(item["path"])
                    for item in expected_current7_condition_indexes or []
                ])
                if expected_current7_condition_indexes
                else []
            )
            if condition_mode:
                nsga = _verify_current7_condition_archive_nsga_api(
                    nsga_payload,
                    conditions_for_attempt,
                    expected_dual_validation=expected_dual_validation,
                )
                progress = _verify_current7_condition_archive_progress(
                    progress_payload, nsga
                )
            else:
                nsga = _verify_nsga_api(
                    nsga_payload,
                    expected_dual_validation=expected_dual_validation,
                    expected_current7_index=current7_for_attempt,
                )
                progress = _verify_nsga_progress(progress_payload, nsga)
                if conditions_for_attempt:
                    nsga["current7_conditions"] = (
                        _verify_current7_condition_searches(
                            nsga_payload, conditions_for_attempt
                        )
                    )
        except CanaryFailure as exc:
            last_error = str(exc)
            current7_snapshot_race = bool(
                expected_current7_index is not None
                and "/api/nsga2 current7 configured identity diverged" in last_error
            )
            condition_snapshot_race = bool(
                expected_current7_condition_indexes
                and "current7 condition" in last_error
                and any(
                    item in last_error
                    for item in (
                        "generation diverged",
                        "changed during authentication",
                        "did not stabilize",
                    )
                )
            )
            if (
                not current7_snapshot_race
                and not condition_snapshot_race
                and not any(item in last_error for item in retryable_messages)
            ):
                raise
            consecutive_reads = 0
            stable_identity = None
        else:
            generation_identity = (
                {
                    "mode": "current7_condition_archive",
                    "conditions_sha256": _sha256_bytes(_canonical_json([
                        {
                            "path": item["path"],
                            "index_file_sha256": item["index_file_sha256"],
                            "snapshot_file_sha256": item[
                                "snapshot_file_sha256"
                            ],
                            "search_count": item["search_count"],
                            "running_count": item["running_count"],
                            "queued_count": item["queued_count"],
                            "attaching_count": item["attaching_count"],
                        }
                        for item in nsga["conditions"]
                    ])),
                }
                if condition_mode
                else _nsga_generation_identity(nsga, progress)
            )
            if generation_identity is None:
                last_error = (
                    "NSGA API/progress did not expose the same authoritative generation"
                )
                consecutive_reads = 0
                stable_identity = None
            elif generation_identity == stable_identity:
                consecutive_reads += 1
            else:
                stable_identity = generation_identity
                consecutive_reads = 1
            if consecutive_reads >= required_reads:
                return (
                    nsga_payload,
                    progress_payload,
                    nsga,
                    progress,
                    {
                        "verified": True,
                        "attempts": attempts,
                        "consecutive_reads": consecutive_reads,
                        "required_reads": required_reads,
                        "poll_seconds": poll_seconds,
                        "timeout_seconds": timeout_seconds,
                        "generation_identity": stable_identity,
                        "current7_index": current7_for_attempt,
                        "current7_condition_indexes": conditions_for_attempt,
                        "mode": (
                            "current7_condition_archive"
                            if condition_mode else "authoritative"
                        ),
                    },
                )
        if time.monotonic() >= deadline:
            raise CanaryFailure(
                "NSGA stability gate timed out without two consecutive fresh "
                f"reads: {last_error}"
            )
        time.sleep(poll_seconds)


def _verify_local_gui_launches(payload: dict[str, Any]) -> dict[str, Any]:
    from .deadline_design import (
        DEADLINE_LOCAL_GUI_RUNNER_SHA256,
        EXPECTED_LOCAL_GUI_SOLVER,
    )
    from .local_aedt_gui import (
        DEFAULT_RUNNER_SHA256,
        DEFAULT_SOLVER_REVISION,
        GUI_RESULT_SCHEMA,
    )

    if payload.get("schema_version") != 1 or payload.get("available") is not True:
        raise CanaryFailure("/api/local-aedt-gui/launches is unavailable")
    if payload.get("backend") != "standalone":
        raise CanaryFailure("local GUI launch backend is not standalone")
    if payload.get("result_schema") != GUI_RESULT_SCHEMA:
        raise CanaryFailure("local GUI result schema mismatch")
    routed = payload.get("routed") is True
    if routed:
        identities = payload.get("solver_identities")
        if not isinstance(identities, list) or len(identities) != 2:
            raise CanaryFailure(
                "routed local GUI solver inventory is malformed"
            )
        by_revision = {
            value.get("revision"): value
            for value in identities
            if isinstance(value, dict)
            and value.get("verified") is True
        }
        identity = by_revision.get(DEFAULT_SOLVER_REVISION)
        deadline_identity = by_revision.get(
            EXPECTED_LOCAL_GUI_SOLVER["solver_revision"]
        )
        if not isinstance(identity, dict) or not isinstance(
            deadline_identity, dict
        ):
            raise CanaryFailure(
                "routed local GUI immutable identities are unverified"
            )
        if (
            str(identity.get("runner_sha256") or "").lower()
            != DEFAULT_RUNNER_SHA256
        ):
            raise CanaryFailure("local GUI baseline runner hash mismatch")
        if (
            deadline_identity.get("branch")
            != EXPECTED_LOCAL_GUI_SOLVER["solver_branch"]
            or str(
                deadline_identity.get("runner_sha256") or ""
            ).lower() != DEADLINE_LOCAL_GUI_RUNNER_SHA256
            or deadline_identity.get("source_runner_sha256")
            != EXPECTED_LOCAL_GUI_SOLVER[
                "solver_source_runner_sha256"
            ]
            or deadline_identity.get("thermal_module_sha256")
            != EXPECTED_LOCAL_GUI_SOLVER[
                "solver_thermal_module_sha256"
            ]
            or deadline_identity.get("library_revision")
            != EXPECTED_LOCAL_GUI_SOLVER["library_revision"]
            or deadline_identity.get("library_dirty") is not False
        ):
            raise CanaryFailure(
                "deadline local GUI solver/library identity mismatch"
            )
    else:
        identity = payload.get("solver_identity")
        deadline_identity = None
        if not isinstance(identity, dict) or identity.get("verified") is not True:
            raise CanaryFailure("local GUI immutable solver identity is unverified")
        if identity.get("revision") != DEFAULT_SOLVER_REVISION:
            raise CanaryFailure("local GUI solver revision mismatch")
        if (
            str(identity.get("runner_sha256") or "").lower()
            != DEFAULT_RUNNER_SHA256
        ):
            raise CanaryFailure("local GUI solver runner hash mismatch")
    launches = payload.get("launches")
    if not isinstance(launches, list):
        raise CanaryFailure("local GUI launches list is malformed")
    counts = {
        name: _nonnegative_integer(payload.get(name), f"local_gui.{name}")
        for name in (
            "max_active",
            "active_count",
            "retained_aedt_count",
            "occupied_count",
        )
    }
    if counts["occupied_count"] < max(
        counts["active_count"], counts["retained_aedt_count"]
    ):
        raise CanaryFailure("local GUI occupied capacity counters disagree")
    if counts["occupied_count"] > counts["max_active"]:
        raise CanaryFailure("local GUI occupied capacity exceeds its hard limit")
    return {
        "available": True,
        "backend": "standalone",
        "result_schema": GUI_RESULT_SCHEMA,
        "solver_revision": identity.get("revision"),
        "runner_sha256": str(identity.get("runner_sha256") or "").lower(),
        "solver_root": identity.get("root"),
        "routed": routed,
        "deadline_solver_revision": (
            deadline_identity.get("revision")
            if isinstance(deadline_identity, dict) else None
        ),
        "deadline_runner_sha256": (
            deadline_identity.get("runner_sha256")
            if isinstance(deadline_identity, dict) else None
        ),
        "deadline_thermal_module_sha256": (
            deadline_identity.get("thermal_module_sha256")
            if isinstance(deadline_identity, dict) else None
        ),
        "launch_rows_returned": len(launches),
        **counts,
    }


def _verify_deadline_design_api(
    nsga_payload: dict[str, Any],
    detail_payload: dict[str, Any],
    *,
    publication_path: Path,
    publication_sha256: str,
) -> dict[str, Any]:
    """Verify the optional hash-authenticated deadline generation end to end."""

    from .deadline_design import load_publication

    authenticated = load_publication(
        publication_path, publication_sha256
    )
    generation = authenticated["generation"]
    candidate = authenticated["candidate"]
    generation_id = generation["id"]
    if nsga_payload.get("selected_generation_id") != generation_id:
        raise CanaryFailure(
            "deadline design is not the selected 8010 generation"
        )
    records = nsga_payload.get("pareto_generations")
    records = records if isinstance(records, list) else []
    matches = [
        value for value in records
        if isinstance(value, dict) and value.get("id") == generation_id
    ]
    if (
        len(matches) != 1
        or matches[0].get("source_kind") != "deadline_validated_design"
        or matches[0].get("gui_build_eligible") is not True
        or matches[0].get("gui_solve_eligible")
        is not candidate.get("gui_solve_eligible")
    ):
        raise CanaryFailure(
            "deadline design generation inventory/authority drifted"
        )
    root_candidates = nsga_payload.get("candidates")
    root_summary = nsga_payload.get("summary")
    if not (
        nsga_payload.get("candidate_count") == 1
        and nsga_payload.get("display_candidate_count") == 1
        and nsga_payload.get("valid_candidate_count") == 1
        and isinstance(root_candidates, list)
        and len(root_candidates) == 1
        and isinstance(root_candidates[0], dict)
        and root_candidates[0].get("id") == candidate.get("id")
        and root_candidates[0].get("final_design_approved")
        is candidate.get("final_design_approved")
        and root_candidates[0].get("gui_solve_eligible")
        is candidate.get("gui_solve_eligible")
        and isinstance(root_summary, dict)
        and root_summary.get("candidate_count") == 1
        and root_summary.get("display_candidate_count") == 1
        and root_summary.get("valid_candidate_count") == 1
    ):
        raise CanaryFailure(
            "selected deadline design is not the coherent top-level view"
        )
    detail_candidates = detail_payload.get("candidates")
    if not (
        detail_payload.get("available") is True
        and detail_payload.get("integrity_verified") is True
        and detail_payload.get("selected_generation_id") == generation_id
        and isinstance(detail_candidates, list)
        and len(detail_candidates) == 1
    ):
        raise CanaryFailure("deadline design detail endpoint failed closed")
    projected = detail_candidates[0]
    if not isinstance(projected, dict) or (
        projected.get("id") != candidate.get("id")
        or projected.get("publication_sha256") != publication_sha256
        or projected.get("fan_config") != candidate.get("fan_config")
        or projected.get("fan_velocity_m_s")
        != candidate.get("fan_velocity_m_s")
        or projected.get("plate_temp_C") != candidate.get("plate_temp_C")
        or projected.get("air_temp_C") != candidate.get("air_temp_C")
        or projected.get("gui_solve_eligible")
        is not candidate.get("gui_solve_eligible")
        or projected.get("local_gui_solver_contract")
        != candidate.get("local_gui_solver_contract")
        or projected.get("pred_f_res_min_screen_Hz")
        != candidate.get("pred_f_res_min_screen_Hz")
        or projected.get("resonance_minimum_required_Hz")
        != candidate.get("resonance_minimum_required_Hz")
        or projected.get("resonance_margin_Hz")
        != candidate.get("resonance_margin_Hz")
        or projected.get("solver_revision")
        != candidate.get("solver_revision")
        or projected.get("thermal_pad_material_policy")
        != candidate.get("thermal_pad_material_policy")
        or projected.get(
            "thermal_pad_native_readback_contract_version"
        )
        != candidate.get(
            "thermal_pad_native_readback_contract_version"
        )
        or projected.get("thermal_pad_native_readback_attested")
        != candidate.get("thermal_pad_native_readback_attested")
        or projected.get(
            "thermal_pad_native_thermal_conductivity_W_mK"
        )
        != candidate.get(
            "thermal_pad_native_thermal_conductivity_W_mK"
        )
        or projected.get(
            "thermal_pad_native_electrical_conductivity_S_m"
        )
        != candidate.get(
            "thermal_pad_native_electrical_conductivity_S_m"
        )
        or (
            projected.get("constraints", {}).get("resonance")
            if isinstance(projected.get("constraints"), dict)
            else None
        )
        != candidate.get("constraints", {}).get("resonance")
    ):
        raise CanaryFailure(
            "deadline design detail/cooling/local-GUI projection drifted"
        )
    validation_state = candidate.get("validation_state")
    if validation_state is not None:
        provisional = (
            validation_state
            == "provisional_split_validated_complete_full_pending"
        )
        complete = (
            validation_state
            == "complete_full_hard_spec_pass_standard_corroborated"
        )
        if not (
            (provisional or complete)
            and candidate.get("final_design_approved") is False
            and candidate.get("standard_corroboration_pass") is True
            and candidate.get("complete_full_pending") is provisional
            and candidate.get("full_actual_hard_pass") is complete
            and candidate.get("gui_build_eligible") is True
            and candidate.get("gui_solve_eligible") is complete
            and generation.get("validation_badge")
            == candidate.get("validation_badge")
            and generation.get("final_design_approved") is False
        ):
            raise CanaryFailure(
                "deadline HIT validation state or GUI authority drifted"
            )
    identity = {
        "verified": True,
        "publication_path": str(publication_path.resolve(strict=True)),
        "generation_id": generation_id,
        "candidate_id": candidate["id"],
        "publication_sha256": publication_sha256,
        "cooling_variant": candidate.get("cooling_variant"),
        "fan_config": candidate.get("fan_config"),
        "fan_velocity_m_s": candidate.get("fan_velocity_m_s"),
        "solver_revision": candidate.get("solver_revision"),
        "thermal_pad_material_policy": candidate.get(
            "thermal_pad_material_policy"
        ),
        "thermal_pad_native_readback_contract_version": candidate.get(
            "thermal_pad_native_readback_contract_version"
        ),
        "thermal_pad_native_readback_attested": candidate.get(
            "thermal_pad_native_readback_attested"
        ),
        "thermal_pad_native_thermal_conductivity_W_mK": candidate.get(
            "thermal_pad_native_thermal_conductivity_W_mK"
        ),
        "thermal_pad_native_electrical_conductivity_S_m": candidate.get(
            "thermal_pad_native_electrical_conductivity_S_m"
        ),
        "half_magnetizing_resonance_Hz": candidate.get(
            "pred_f_res_min_screen_Hz"
        ),
        "half_magnetizing_resonance_minimum_Hz": candidate.get(
            "resonance_minimum_required_Hz"
        ),
        "half_magnetizing_resonance_margin_Hz": candidate.get(
            "resonance_margin_Hz"
        ),
        "full_actual_hard_pass": authenticated["publication"].get(
            "full_actual_hard_pass"
        ) is True,
        "gui_build_eligible": candidate.get("gui_build_eligible") is True,
        "gui_solve_eligible": candidate.get("gui_solve_eligible") is True,
    }
    if validation_state is not None:
        identity.update(
            {
                "complete_full_pending": (
                    candidate.get("complete_full_pending") is True
                ),
                "standard_corroboration_pass": (
                    candidate.get("standard_corroboration_pass") is True
                ),
                "validation_state": validation_state,
                "validation_badge": candidate.get("validation_badge"),
                "hard_spec_authority": candidate.get("hard_spec_authority"),
                "final_design_approved": False,
            }
        )
    return identity


def _write_new_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise CanaryFailure(f"refusing to overwrite release evidence: {path}") from exc
    return _sha256_bytes(raw)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--regression-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--rolling-index", type=Path, required=True)
    parser.add_argument("--current7-index", type=Path)
    parser.add_argument(
        "--current7-condition-index",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable read-only staged current7 index; all configured paths "
            "must authenticate and appear in /api/nsga2 in the same order"
        ),
    )
    parser.add_argument("--blocker-hpo-v2-status", type=Path)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-venv-root", type=Path, required=True)
    parser.add_argument("--sealed-successor-handoff", type=Path, required=True)
    parser.add_argument("--sealed-successor-handoff-sha256", required=True)
    parser.add_argument("--dual-successor-plan", type=Path, required=True)
    parser.add_argument("--dual-successor-plan-sha256", required=True)
    parser.add_argument("--dual-successor-ui", type=Path, required=True)
    parser.add_argument("--dual-successor-ui-sha256", required=True)
    parser.add_argument("--dual-successor-receipt", type=Path, required=True)
    parser.add_argument("--dual-successor-receipt-sha256", required=True)
    parser.add_argument("--dual-fea-runtime-root", type=Path, required=True)
    parser.add_argument("--dual-fea-allowed-root", type=Path, required=True)
    parser.add_argument("--dual-fea-manifest-sha256", required=True)
    parser.add_argument("--dual-fea-plan", type=Path, required=True)
    parser.add_argument("--dual-fea-plan-sha256", required=True)
    parser.add_argument("--dual-fea-source-sha256", required=True)
    parser.add_argument("--deadline-design-publication", type=Path)
    parser.add_argument("--deadline-design-publication-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    generated_at = datetime.now(timezone.utc)
    release_id = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "mft-monitor-production-release-canary",
        "release_id": release_id,
        "generated_at": generated_at.isoformat(timespec="microseconds"),
        "passed": False,
        "deploy_authorized": False,
        "error": None,
    }
    evidence_path = args.evidence_dir / f"canary_evidence_{release_id}.json"
    manifest_path = args.evidence_dir / f"deployment_manifest_{release_id}.json"
    exit_code = 1
    try:
        source_root = args.source_root.resolve(strict=True)
        regression_root = args.regression_root.resolve(strict=True)
        pipeline_root = args.pipeline_root.resolve(strict=True)
        rolling_index = args.rolling_index.resolve(strict=True)
        if rolling_index.is_symlink() or not rolling_index.is_file():
            raise CanaryFailure("rolling NSGA index must be a regular non-symlink file")
        current7_index = (
            _authenticate_current7_index(args.current7_index)
            if args.current7_index is not None
            else None
        )
        current7_condition_indexes = _authenticate_current7_condition_indexes(
            list(args.current7_condition_index)
        )
        blocker_hpo_v2_status = (
            _authenticate_blocker_hpo_v2_status(args.blocker_hpo_v2_status)
            if args.blocker_hpo_v2_status is not None
            else None
        )
        dual_configuration = _dual_release_configuration(args)
        dual_environment = _apply_dual_release_environment(dual_configuration)
        evidence["runtime"] = _runtime_identity(args.expected_venv_root)
        evidence["source"] = _source_identity(source_root, args.expected_revision)
        evidence["configuration"] = {
            "regression_root": str(regression_root),
            "pipeline_root": str(pipeline_root),
            "rolling_index": str(rolling_index),
            "rolling_index_sha256": _sha256_file(rolling_index, 4 * 1024 * 1024),
            "current7_index": current7_index,
            "current7_condition_indexes": current7_condition_indexes,
            "blocker_hpo_v2_status": blocker_hpo_v2_status,
            "history_disabled": True,
            "dual_release": dual_configuration,
            "dual_environment": dual_environment,
        }

        deadline_path = args.deadline_design_publication
        deadline_sha = str(
            args.deadline_design_publication_sha256 or ""
        ).strip().lower()
        if bool(deadline_path) != bool(deadline_sha):
            raise CanaryFailure(
                "deadline publication path/hash command-line configuration "
                "is partial"
            )
        if deadline_path is not None:
            from .deadline_design import load_publication

            deadline_path = deadline_path.resolve(strict=True)
            if _sha256_file(deadline_path, 8 * 1024 * 1024) != deadline_sha:
                raise CanaryFailure(
                    "deadline publication command-line SHA-256 mismatch"
                )
            authenticated_deadline = load_publication(
                deadline_path, deadline_sha
            )
            os.environ["MFT_DEADLINE_DESIGN_PUBLICATION"] = str(
                deadline_path
            )
            os.environ[
                "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256"
            ] = deadline_sha
            evidence["configuration"]["deadline_design_input"] = {
                "path": str(deadline_path),
                "sha256": deadline_sha,
                "generation_id": authenticated_deadline["generation"]["id"],
                "candidate_id": authenticated_deadline["candidate"]["id"],
                "validation_state": authenticated_deadline["candidate"].get(
                    "validation_state"
                ),
            }
        else:
            evidence["configuration"]["deadline_design_input"] = None

        os.environ["MFT_MONITOR_ROOT"] = str(regression_root)
        os.environ["MFT_PIPELINE_ROOT"] = str(pipeline_root)
        os.environ["MFT_TIER1_ROLLING_INDEX"] = str(rolling_index)
        if current7_index is None:
            os.environ.pop("MFT_TIER1_CURRENT7_INDEX", None)
        else:
            os.environ["MFT_TIER1_CURRENT7_INDEX"] = current7_index["path"]
        if current7_condition_indexes:
            os.environ["MFT_TIER1_CURRENT7_CONDITION_INDEXES"] = ";".join(
                item["path"] for item in current7_condition_indexes
            )
        else:
            os.environ.pop("MFT_TIER1_CURRENT7_CONDITION_INDEXES", None)
        if blocker_hpo_v2_status is None:
            os.environ.pop("MFT_BLOCKER_HPO_V2_STATUS", None)
        else:
            os.environ["MFT_BLOCKER_HPO_V2_STATUS"] = blocker_hpo_v2_status["path"]
        os.environ["MFT_MONITOR_DISABLE_HISTORY"] = "1"

        from .app import create_app

        canary_gui_runtime = (
            args.evidence_dir.resolve()
            / "candidate_runtime"
            / release_id
            / "local-aedt-gui"
        )
        os.environ["MFT_LOCAL_AEDT_GUI_RUNTIME"] = str(
            canary_gui_runtime / "baseline"
        )
        os.environ["MFT_DEADLINE_LOCAL_AEDT_GUI_RUNTIME"] = str(
            canary_gui_runtime / "deadline-timk3"
        )
        app = create_app(regression_root=regression_root)
        with _ephemeral_server(app) as base_url:
            dashboard_html = _get_text(base_url, "/", "text/html", 20)
            browser_js = _get_text(base_url, "/static/app.js", "text/javascript", 20)
            if "/static/app.js" not in dashboard_html:
                raise CanaryFailure("dashboard HTML does not load the candidate app.js")
            for endpoint in (
                "/api/dashboard",
                "/api/nsga2/progress",
                "/api/local-aedt-gui/launches",
            ):
                if endpoint not in browser_js:
                    raise CanaryFailure(
                        f"browser bundle does not poll required endpoint {endpoint}"
                    )
            health = _get_json(base_url, "/healthz", 20)
            if health.get("status") != "ok":
                raise CanaryFailure("/healthz is not ok")
            data_payload = _get_json(base_url, "/api/data", 240)
            (
                nsga_payload,
                progress_payload,
                nsga,
                progress,
                nsga_stability_gate,
            ) = _wait_for_stable_nsga_generation(
                base_url,
                expected_dual_validation=dual_configuration,
                expected_current7_index=current7_index,
                expected_current7_condition_indexes=current7_condition_indexes,
            )
            if blocker_hpo_v2_status is not None:
                (
                    dashboard_payload,
                    blocker_hpo_v2_api,
                    blocker_hpo_v2_latest,
                ) = _wait_for_blocker_hpo_v2_api(
                    base_url,
                    blocker_hpo_v2_status,
                )
                blocker_hpo_v2_api["payload_sha256"] = _sha256_bytes(
                    _canonical_json(dashboard_payload)
                )
                evidence["configuration"]["blocker_hpo_v2_status"] = (
                    blocker_hpo_v2_latest
                )
            else:
                blocker_hpo_v2_api = None
            launches_payload = _get_json(base_url, "/api/local-aedt-gui/launches", 60)
            deadline_publication = os.environ.get(
                "MFT_DEADLINE_DESIGN_PUBLICATION", ""
            ).strip()
            deadline_publication_sha256 = os.environ.get(
                "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256", ""
            ).strip().lower()
            if bool(deadline_publication) != bool(
                deadline_publication_sha256
            ):
                raise CanaryFailure(
                    "deadline publication path/hash configuration is partial"
                )
            if deadline_publication:
                generation_id = str(
                    nsga_payload.get("selected_generation_id") or ""
                )
                if not generation_id or len(generation_id) > 200:
                    raise CanaryFailure(
                        "deadline selected generation id is invalid"
                    )
                deadline_detail_payload = _get_json(
                    base_url,
                    f"/api/nsga2/generations/{generation_id}",
                    120,
                )
                deadline_design_api = _verify_deadline_design_api(
                    nsga_payload,
                    deadline_detail_payload,
                    publication_path=Path(deadline_publication),
                    publication_sha256=deadline_publication_sha256,
                )
            else:
                deadline_design_api = None

        cohort, parquet_read = _verify_data_api(data_payload, pipeline_root)
        local_gui = _verify_local_gui_launches(launches_payload)
        current7_api = (
            _verify_current7_api(
                nsga_payload,
                nsga_stability_gate.get("current7_index") or current7_index,
            )
            if current7_index is not None
            else None
        )
        current7_condition_api = (
            _verify_current7_condition_searches(
                nsga_payload,
                nsga_stability_gate.get("current7_condition_indexes")
                or current7_condition_indexes,
            )
            if current7_condition_indexes
            else []
        )
        evidence["strict_cohort"] = cohort
        evidence["parquet_read_canary"] = parquet_read
        evidence["api"] = {
            "browser_bundle": {
                "verified": True,
                "dashboard_html_sha256": _sha256_bytes(dashboard_html.encode("utf-8")),
                "app_js_sha256": _sha256_bytes(browser_js.encode("utf-8")),
                "polled_endpoints": [
                    "/api/dashboard",
                    "/api/nsga2/progress",
                    "/api/local-aedt-gui/launches",
                ],
            },
            "healthz": {"status": "ok"},
            "data": {
                "verified": True,
                "payload_sha256": _sha256_bytes(_canonical_json(data_payload)),
            },
            "nsga2": {
                **nsga,
                "payload_sha256": _sha256_bytes(_canonical_json(nsga_payload)),
            },
            "nsga2_progress": {
                **progress,
                "payload_sha256": _sha256_bytes(_canonical_json(progress_payload)),
            },
            "nsga_stability_gate": nsga_stability_gate,
            "tier1_current7": current7_api,
            "tier1_current7_conditions": current7_condition_api,
            "blocker_hpo_v2": blocker_hpo_v2_api,
            "local_aedt_gui_launches": {
                **local_gui,
                "payload_sha256": _sha256_bytes(_canonical_json(launches_payload)),
            },
            "deadline_design": deadline_design_api,
        }
        evidence["passed"] = True
        evidence["deploy_authorized"] = True
        exit_code = 0
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"

    evidence_sha = _write_new_json(evidence_path, evidence)
    output: dict[str, Any] = {
        "passed": evidence["passed"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha,
        "deployment_manifest_path": None,
        "error": evidence.get("error"),
    }
    if evidence["passed"]:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "mft-monitor-production-deployment-manifest",
            "release_id": release_id,
            "generated_at": evidence["generated_at"],
            "deploy_authorized": True,
            "candidate_revision": evidence["source"]["revision"],
            "candidate_source_root": evidence["source"]["source_root"],
            "monitoring_tree_sha256": evidence["source"]["monitoring_tree_sha256"],
            "regression_root": evidence["configuration"]["regression_root"],
            "pipeline_root": evidence["configuration"]["pipeline_root"],
            "rolling_index": evidence["configuration"]["rolling_index"],
            "rolling_index_sha256": evidence["configuration"]["rolling_index_sha256"],
            "current7_index": evidence["configuration"]["current7_index"],
            "current7_condition_indexes": evidence["configuration"][
                "current7_condition_indexes"
            ],
            "blocker_hpo_v2_status": evidence["configuration"]["blocker_hpo_v2_status"],
            "sealed_successor_handoff": evidence["configuration"]["dual_release"][
                "sealed_successor_handoff"
            ],
            "dual_successor_plan": evidence["configuration"]["dual_release"][
                "dual_successor_plan"
            ],
            "dual_successor_ui": evidence["configuration"]["dual_release"][
                "dual_successor_ui"
            ],
            "dual_successor_receipt": evidence["configuration"]["dual_release"][
                "dual_successor_receipt"
            ],
            "dual_standard_fea_validation": evidence["configuration"]["dual_release"][
                "standard_fea_validation"
            ],
            "python_executable": evidence["runtime"]["python_executable"],
            "venv_root": evidence["runtime"]["venv_root"],
            "runtime_process_identity": evidence["runtime"]["runtime_process_identity"],
            "python_version": evidence["runtime"]["python_version"],
            "pyarrow_version": evidence["runtime"]["pyarrow_version"],
            "strict_cohort_generation": evidence["strict_cohort"]["generation"],
            "strict_cohort_artifact_sha256": evidence["strict_cohort"][
                "artifact_sha256"
            ],
            "nsga_constraint_version": evidence["api"]["nsga2"]["constraint_version"],
            "nsga_model_manifest_sha256": evidence["api"]["nsga2"][
                "model_manifest_sha256"
            ],
            "local_gui_solver_revision": evidence["api"]["local_aedt_gui_launches"][
                "solver_revision"
            ],
            "local_gui_runner_sha256": evidence["api"]["local_aedt_gui_launches"][
                "runner_sha256"
            ],
            "deadline_design": evidence["api"]["deadline_design"],
            "canary_evidence_path": str(evidence_path),
            "canary_evidence_sha256": evidence_sha,
        }
        manifest_sha = _write_new_json(manifest_path, manifest)
        sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
        try:
            with sha_path.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(f"{manifest_sha}  {manifest_path.name}\n")
        except FileExistsError as exc:
            raise CanaryFailure(
                f"refusing to overwrite release hash sidecar: {sha_path}"
            ) from exc
        output["deployment_manifest_path"] = str(manifest_path)
        output["deployment_manifest_sha256"] = manifest_sha
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
