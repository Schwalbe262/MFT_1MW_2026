"""Fail-closed Pareto-to-FEA handoff for the 2026-07-26 MFT goal.

The tool is deliberately stateful and one-way:

``plan -> submit-standard -> collect -> gate -> submit-full -> collect -> package``

Only the two submit commands mutate Scheduler state. All other commands are
local or GET-only. The separate Scheduler repository is never edited.
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
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    FIXED_COOLING_IDENTITY,
    FIXED_OPERATING_IDENTITY,
    GOAL_CONTRACT_SCHEMA,
    GOAL_N1_MAX_TURNS,
    GOAL_N1_MIN_TURNS,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    GOAL_TEMPERATURE_TARGETS,
    TEMPERATURE_TARGET_LIMITS_C,
    attest_fixed_identity,
    canonical_sha256,
    dynamic_core_group_violation,
    validate_cw1_mm,
    validate_goal_stage_spec,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import finalize, scheduler_client  # noqa: E402
from tools import mft_goal_20260726_launch as launch  # noqa: E402
from tools import tier1_corrected_generation_adapter as adapter  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


PLAN_SCHEMA = "mft-goal-fea-handoff-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-fea-submission-v1"
COLLECTION_SCHEMA = "mft-goal-fea-collection-v1"
STANDARD_GATE_SCHEMA = "mft-goal-fea-standard-gate-v1"
PACKAGE_SCHEMA = "mft-goal-fea-package-v1"
REMOTE_RECEIPT_SCHEMA = scheduler_client.RETAINED_AEDT_RECEIPT_SCHEMA
PROFILE_SCHEMA = "mft-goal-fea-profile-v1"
PROFILE_PATHS = {
    "standard": (
        REPOSITORY_ROOT
        / "regression_260707"
        / "verify"
        / "profiles"
        / "goal_standard.json"
    ),
    "full": (
        REPOSITORY_ROOT
        / "regression_260707"
        / "verify"
        / "profiles"
        / "goal_full.json"
    ),
}
STAGE_RESOURCES = {
    "standard": {"cpus": 8, "timeout_seconds": 4 * 3600},
    "full": {"cpus": 16, "timeout_seconds": 12 * 3600},
}
STAGE_MODE = {
    "standard": {
        "full_model": 0,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "matrix_skin_mesh": 0,
        "keep_project": 1,
    },
    "full": {
        "full_model": 1,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "n_explicit_turns": 2,
        "matrix_skin_mesh": 1,
        "keep_project": 1,
    },
}
PROFILE_CANONICAL_SHA256 = {
    "standard": "767f78f2b269bb1196bbfa5fba4f178cc13ee442fa77e004d8674fed0a04e049",
    "full": "975dcfff5ae5ffeea973cf0d7f4892e7809779e4f986c5a7b2a16deb8b682769",
}
PROFILE_CLI_FLAGS = {
    "standard": "--thermal --headless",
    "full": "--thermal --headless --full",
}
PROFILE_REVIEWED_PATH = {
    stage: f"run_simulation_260706.py --fixed {flags}"
    for stage, flags in PROFILE_CLI_FLAGS.items()
}
FIXED_PROFILE_FIELDS = {
    **FIXED_OPERATING_IDENTITY,
    **{
        key: value
        for key, value in FIXED_COOLING_IDENTITY.items()
        if key != "thermal_pad_conductivity_W_mK"
    },
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}
STANDARD_CORE_CONTRACT = "mft-standalone-core-optin-v1"
FULL_CORE_CONTRACT = "mft-standalone-core-16-optin-v1"
FULL_LICENSE_CONTRACT = "mft-aedt-hpc-license-snapshot-v1"
MAX_REMOTE_METADATA_BYTES = 128 * 1024
MAX_REMOTE_TEXT_CHUNK_BYTES = 1024 * 1024
MAX_AEDT_BYTES = scheduler_client.RETAINED_AEDT_MAX_BYTES
MIN_AGGREGATE_SEED_COUNT = launch.ROLLING_SEED_COUNT
EXPECTED_RELOCATION_CONTRACT = {
    "schema_version": launch.RELOCATION_SCHEMA,
    "mode": "source_paths_or_explicit_worker_role_path_relocation",
    "roles": list(launch.RUNTIME_SOURCE_ROLES),
    "code_manifest_path_required_for_relocation": True,
    "relocation_files_emitted": False,
    "stager_must_generate_task_bound_relocation": True,
    "source_absolute_paths_are_not_worker_authority": True,
    "relocated_paths_must_reauthenticate_all_bound_SHA256": True,
    "task_payload_sha256_must_match": True,
    "worker_override_file_supported": True,
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
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
}


class HandoffContractError(RuntimeError):
    """An input, stage transition, result, or retained artifact drifted."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return adapter.sha256_file(path)


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise HandoffContractError(f"artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise HandoffContractError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HandoffContractError(f"{schema} payload is not an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise HandoffContractError(f"{schema} payload seal mismatch")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(f"JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffContractError(f"JSON artifact is not an object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise HandoffContractError(f"immutable output already exists: {target}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def _require_sha(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not HEX64.fullmatch(text):
        raise HandoffContractError(f"{label} must be exact SHA-256 hex")
    return text


def _require_revision(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not HEX40.fullmatch(text):
        raise HandoffContractError(f"{label} must be exact 40-hex Git revision")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise HandoffContractError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HandoffContractError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise HandoffContractError(f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer():
        raise HandoffContractError(f"{label} must be an integer")
    return int(number)


def _csv_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise HandoffContractError(f"{label} must be a canonical CSV boolean")


def _contained_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HandoffContractError(f"{label} path is missing")
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise HandoffContractError(f"{label} escapes its manifest directory") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise HandoffContractError(f"{label} is not a regular file")
    return candidate


def _profile_content(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage not in PROFILE_PATHS:
        raise HandoffContractError(f"unknown FEA stage: {stage}")
    path = PROFILE_PATHS[stage].resolve(strict=True)
    profile = _read_json(path)
    _validate_profile(stage, profile)
    return profile, _file_record(path)


def _validate_profile(stage: str, profile: Mapping[str, Any]) -> None:
    resources = STAGE_RESOURCES[stage]
    overrides = profile.get("param_overrides")
    boundary = profile.get("fixed_boundary_contract")
    if (
        set(profile)
        != {
            "schema_version",
            "stage",
            "comment",
            "reviewed_solver_path",
            "cli_flags",
            "param_overrides",
            "fixed_boundary_contract",
            "artifact_retention",
            "mem_mb",
            "cpus",
            "timeout_seconds",
        }
        or canonical_sha256(profile) != PROFILE_CANONICAL_SHA256[stage]
        or profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("stage") != stage
        or profile.get("cli_flags") != PROFILE_CLI_FLAGS[stage]
        or profile.get("reviewed_solver_path") != PROFILE_REVIEWED_PATH[stage]
        or not isinstance(overrides, dict)
        or not isinstance(boundary, dict)
        or profile.get("cpus") != resources["cpus"]
        or profile.get("timeout_seconds") != resources["timeout_seconds"]
        or not isinstance(profile.get("mem_mb"), int)
        or int(profile["mem_mb"]) <= 0
        or any(overrides.get(key) != value for key, value in STAGE_MODE[stage].items())
        or any(overrides.get(key) != value for key, value in FIXED_PROFILE_FIELDS.items())
        or boundary
        != {
            "thermal_pad_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "core_plate_on": 1,
            "wcp_on": 1,
        }
        or profile.get("artifact_retention", {}).get("stage") != stage
        or profile.get("artifact_retention", {}).get("retention_required")
        is not True
        or profile.get("artifact_retention", {}).get(
            "prune_protection_required"
        )
        is not True
        or profile.get("artifact_retention", {}).get("marker_filename")
        != scheduler_client.SCHEDULER_PRESERVE_MARKER
    ):
        raise HandoffContractError(f"{stage} FEA profile contract drifted")
    if any(
        key in profile or key in overrides
        for key in (
            "T_limit_C",
            "n_core_group_max",
            "primary_conductor_thickness_mm",
            "resonance_max_Hz",
        )
    ):
        raise HandoffContractError(f"{stage} profile contains legacy goal fields")


def _parse_json_cell(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise HandoffContractError(f"{label} is not a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HandoffContractError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise HandoffContractError(f"{label} is not a JSON object")
    return parsed


def _goal_row_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    geometry_sha = _require_sha(
        row.get("physical_geometry_sha256"), "physical_geometry_sha256"
    )
    if (
        _require_sha(row.get("candidate_physics_sha"), "candidate_physics_sha")
        != geometry_sha
        or not _csv_bool(row.get("decoder_valid"), "decoder_valid")
        or not _csv_bool(
            row.get("surrogate_physical_valid"), "surrogate_physical_valid"
        )
        or not _csv_bool(row.get("physical_feasible"), "physical_feasible")
    ):
        raise HandoffContractError("selected row is not canonical physical feasible")
    decoded = _parse_json_cell(
        row.get("decoded_physical_params_json"),
        "decoded_physical_params_json",
    )
    if canonical_sha256(decoded) != _require_sha(
        row.get("canonical_physical_params_sha256"),
        "canonical_physical_params_sha256",
    ):
        raise HandoffContractError("decoded parameter SHA-256 mismatch")
    geometry = {
        name: decoded.get(name)
        for name in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
    }
    if any(value is None for value in geometry.values()):
        raise HandoffContractError("decoded geometry identity is incomplete")
    if canonical_sha256(geometry) != geometry_sha:
        raise HandoffContractError("decoded geometry SHA-256 mismatch")

    physical_g = _parse_json_cell(row.get("physical_G_json"), "physical_G_json")
    normalized_g = _parse_json_cell(
        row.get("normalized_G_json"), "normalized_G_json"
    )
    names = tuple(preflight.GOAL_CONSTRAINT_NAMES)
    if set(physical_g) != set(names) or set(normalized_g) != set(names):
        raise HandoffContractError("goal hard-constraint vector is incomplete")
    physical_values = {name: _finite(physical_g[name], name) for name in names}
    normalized_values = {
        name: _finite(normalized_g[name], f"normalized:{name}") for name in names
    }
    for name in names:
        if not math.isclose(
            physical_values[name],
            _finite(row.get(f"physical_G:{name}"), f"physical_G:{name}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            normalized_values[name],
            _finite(row.get(f"normalized_G:{name}"), f"normalized_G:{name}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise HandoffContractError(f"flattened constraint evidence drifted: {name}")
    violations = {name: value for name, value in physical_values.items() if value > 0}
    if violations:
        raise HandoffContractError(f"selected row violates goal hard spec: {violations}")

    cw1 = validate_cw1_mm(decoded.get("cw1"))
    n1 = _integer(decoded.get("N1_main"), "N1_main") + _integer(
        decoded.get("N1_side"), "N1_side"
    )
    if not GOAL_N1_MIN_TURNS <= n1 <= GOAL_N1_MAX_TURNS:
        raise HandoffContractError("selected row primary turns are outside 5..8")
    if dynamic_core_group_violation(decoded) > 0:
        raise HandoffContractError("selected row dynamic core-group contract failed")
    if (
        _finite(decoded.get("core_plate_pad_t"), "core_plate_pad_t") != 2.0
        or _finite(decoded.get("wcp_pad_t"), "wcp_pad_t") != 2.0
    ):
        raise HandoffContractError("selected row cooling pad thickness drifted")
    identity_values = dict(decoded)
    observed_tim = identity_values.get("thermal_pad_conductivity_W_mK")
    if observed_tim is not None and _finite(observed_tim, "TIM conductivity") != 0.2:
        raise HandoffContractError("selected row TIM conductivity drifted")
    identity_values["thermal_pad_conductivity_W_mK"] = 0.2
    fixed_identity = attest_fixed_identity(identity_values)
    volume_l, dimensions = geometry_metrics.bounding_box_lit(decoded)
    width, length, height = (
        _finite(value, "exterior dimension") for value in dimensions
    )
    for axis, value in zip(("W", "L", "H"), (width, length, height)):
        if value > GOAL_SIZE_LIMITS_MM[axis]:
            raise HandoffContractError(f"decoded exterior {axis} exceeds goal")

    missing = [key for key in ALL_INPUT_KEYS if decoded.get(key) is None]
    if missing:
        raise HandoffContractError(f"decoded FEA parameter schema is incomplete: {missing}")
    params = {key: decoded[key] for key in ALL_INPUT_KEYS}
    if set(params) != set(ALL_INPUT_KEYS):
        raise HandoffContractError("exact FEA parameter projection failed")
    return {
        "physical_geometry_sha256": geometry_sha,
        "canonical_physical_params_sha256": canonical_sha256(decoded),
        "decoded_params": decoded,
        "fea_params": params,
        "fea_params_sha256": canonical_sha256(params),
        "fixed_identity_attestation": fixed_identity,
        "primary_turns": n1,
        "cw1_mm": cw1,
        "n_core_group": _integer(decoded["n_core_group"], "n_core_group"),
        "exterior_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "volume_L": float(volume_l),
        "physical_G": physical_values,
        "normalized_G": normalized_values,
        "goal_hard_spec_passed": True,
        "required_physics": {
            "Llt_robust_band": physical_values["Llt_robust_band"] <= 0,
            "analytical_flux_density_limit": (
                physical_values["analytical_flux_density_limit"] <= 0
            ),
            "minimum_physical_insulation": (
                physical_values["minimum_physical_insulation"] <= 0
            ),
            "half_magnetizing_resonance_minimum_15kHz": (
                physical_values["half_magnetizing_resonance_minimum"] <= 0
            ),
            "all_split_robust_temperature_targets": all(
                physical_values[f"temperature_robust_limit:{target}"] <= 0
                for target in GOAL_TEMPERATURE_TARGETS
            ),
        },
    }


def _rows_equivalent(selected: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    exact = (
        "terminal_population_index",
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
        "source_seed",
        "source_task_id",
        "source_bundle_id",
        "source_island_id",
        "dataset_sha256",
        "evaluation_model_sha256",
        "evaluation_model_artifacts_sha256",
        "constraint_spec_sha256",
        "cooling_contract_sha256",
        "operating_point_sha256",
        "evaluation_model_generation_sha256",
        "evaluation_spec_sha256",
        "evaluation_temperature_contract_sha256",
        "evaluation_hard_constraint_contract_sha256",
        "decoded_physical_params_json",
        "physical_G_json",
        "normalized_G_json",
    )
    for name in exact:
        if str(selected.get(name)) != str(source.get(name)):
            raise HandoffContractError(f"selected/source terminal identity drifted: {name}")
    numeric = (
        "objective_volume_L",
        "objective_total_loss_W",
        *tuple(f"physical_G:{name}" for name in preflight.GOAL_CONSTRAINT_NAMES),
        *tuple(f"normalized_G:{name}" for name in preflight.GOAL_CONSTRAINT_NAMES),
    )
    for name in numeric:
        if not math.isclose(
            _finite(selected.get(name), f"selected:{name}"),
            _finite(source.get(name), f"source:{name}"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise HandoffContractError(f"selected/source terminal value drifted: {name}")
    for name in (
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
        "physical_constraint_feasible",
        "physical_feasible",
    ):
        if _csv_bool(selected.get(name), f"selected:{name}") != _csv_bool(
            source.get(name), f"source:{name}"
        ):
            raise HandoffContractError(f"selected/source boolean drifted: {name}")


def _authenticate_bundle(
    bundle_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, Any],
]:
    bundle_path = bundle_manifest_path.resolve(strict=True)
    try:
        bundle, ledger = launch.load_bundle_task_ledger(bundle_path)
    except RuntimeError as exc:
        raise HandoffContractError(
            "goal bundle/code-manifest/task ledger authentication failed"
        ) from exc
    task_count = _integer(bundle.get("task_count"), "bundle task_count")
    payloads = bundle.get("task_payload_sha256")
    relative_paths = bundle.get("task_relative_paths")
    code_record = bundle.get("code_manifest")
    source_paths = bundle.get("source_bound_paths")
    if (
        bundle.get("campaign_id") != "mft-goal-20260726"
        or bundle.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or bundle.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or bundle.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or bundle.get("search_only_proposal") is not False
        or bundle.get("production_eligible") is not False
        or bundle.get("automatic_promotion_allowed") is not False
        or bundle.get("scheduler_project_modified") is not False
        or bundle.get("legacy_current7_bundle_or_release_identity_reused")
        is not False
        or task_count < MIN_AGGREGATE_SEED_COUNT
        or not isinstance(payloads, list)
        or not isinstance(relative_paths, list)
        or len(payloads) != task_count
        or len(relative_paths) != task_count
        or len(set(map(str, payloads))) != task_count
        or len(set(map(str, relative_paths))) != task_count
        or len(ledger) != task_count
        or bundle.get("relocation_contract")
        != EXPECTED_RELOCATION_CONTRACT
        or not isinstance(code_record, dict)
        or set(code_record)
        != {
            "path",
            "schema_version",
            "payload_sha256",
            "code_inventory_sha256",
            "code_revision",
        }
        or code_record.get("schema_version")
        != launch.CODE_MANIFEST_SCHEMA
        or not isinstance(source_paths, dict)
        or set(source_paths)
        != {*launch.RUNTIME_SOURCE_ROLES, "expected_code_revision"}
        or source_paths.get("expected_code_revision")
        != code_record.get("code_revision")
    ):
        raise HandoffContractError("goal bundle authority drifted")
    code_manifest_path = _contained_file(
        bundle_path.parent,
        code_record["path"],
        "bundle code manifest",
    )
    try:
        code_manifest = launch._validate_seal(
            _read_json(code_manifest_path),
            schema=launch.CODE_MANIFEST_SCHEMA,
        )
    except RuntimeError as exc:
        raise HandoffContractError("bundle code manifest seal failed") from exc
    if (
        code_manifest.get("payload_sha256")
        != code_record.get("payload_sha256")
        or code_manifest.get("code_inventory_sha256")
        != code_record.get("code_inventory_sha256")
        or code_manifest.get("code_revision")
        != code_record.get("code_revision")
        or not isinstance(code_manifest.get("code_inventory"), dict)
        or code_manifest.get("files") != code_manifest.get("code_inventory")
        or code_manifest.get("code_inventory_sha256")
        != canonical_sha256(code_manifest.get("code_inventory"))
        or code_manifest.get("staged_path_rule")
        != "bundle_root/<code_inventory_key>"
        or code_manifest.get("source_checkout_mutated") is not False
        or code_manifest.get("remote_git_checkout_required") is not False
        or code_manifest.get("scheduler_project_code_included") is not False
    ):
        raise HandoffContractError("bundle code manifest identity drifted")
    try:
        code_authentication = preflight.authenticate_goal_code_inventory(
            code_root=bundle_path.parent / "artifacts" / "code",
            code_manifest_path=code_manifest_path,
            expected_manifest_payload_sha256=code_manifest[
                "payload_sha256"
            ],
            expected_code_inventory_sha256=code_manifest[
                "code_inventory_sha256"
            ],
            expected_code_revision=code_manifest["code_revision"],
        )
    except (OSError, RuntimeError) as exc:
        raise HandoffContractError(
            "bundle staged runtime code inventory authentication failed"
        ) from exc
    tasks: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    seeds: set[int] = set()
    strata: dict[int, int] = {turns: 0 for turns in range(5, 9)}
    for expected_sha, relative in zip(payloads, relative_paths):
        digest = _require_sha(expected_sha, "bundle task payload SHA")
        task_path = _contained_file(bundle_path.parent, relative, "bundle task")
        task = launch.validate_task_payload(_read_json(task_path))
        turns = _integer(task.get("fixed_primary_turns"), "bundle task turns")
        seed = _integer(task.get("seed"), "bundle task seed")
        if (
            task.get("payload_sha256") != digest
            or ledger.get(digest) != task
            or turns not in strata
            or seed in seeds
        ):
            raise HandoffContractError("bundle task identity drifted")
        seeds.add(seed)
        strata[turns] += 1
        tasks[digest] = task
        paths[digest] = task_path
    minimum_per_stratum = MIN_AGGREGATE_SEED_COUNT // len(strata)
    if any(count < minimum_per_stratum for count in strata.values()):
        raise HandoffContractError("bundle does not cover the four N1 strata")
    return bundle, tasks, paths, code_authentication


def _authenticate_aggregate_authority(
    *,
    aggregate: Mapping[str, Any],
    manifest_path: Path,
    bundle_manifest_path: Path,
) -> dict[str, Any]:
    from tools.mft_goal_global_pareto import rank_candidates

    (
        bundle,
        bundle_tasks,
        bundle_task_paths,
        bundle_code_authentication,
    ) = _authenticate_bundle(bundle_manifest_path)
    authenticated_bundle = aggregate.get("authenticated_bundle")
    expected_bundle_authority = {
        "path": str(bundle_manifest_path.resolve(strict=True)),
        "sha256": _sha256_file(bundle_manifest_path.resolve(strict=True)),
        "payload_sha256": bundle["payload_sha256"],
        "task_count": len(bundle_tasks),
        "task_ledger_sha256": canonical_sha256(sorted(bundle_tasks)),
        "all_four_N1_strata_covered": True,
    }
    if authenticated_bundle != expected_bundle_authority:
        raise HandoffContractError(
            "aggregate original bundle/task ledger authority drifted"
        )
    minimum = _integer(
        aggregate.get("minimum_seed_count"), "aggregate minimum_seed_count"
    )
    seed_count = _integer(aggregate.get("seed_count"), "aggregate seed_count")
    inputs = aggregate.get("inputs")
    if (
        minimum < MIN_AGGREGATE_SEED_COUNT
        or seed_count < minimum
        or seed_count != len(bundle_tasks)
        or not isinstance(inputs, list)
        or len(inputs) != seed_count
    ):
        raise HandoffContractError("aggregate seed inventory is below goal minimum")
    results = []
    frames = []
    result_paths: dict[str, Path] = {}
    observed_task_payloads: set[str] = set()
    observed_seeds: set[int] = set()
    for record in inputs:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise HandoffContractError("aggregate input record is malformed")
        result_path = Path(str(record["path"])).resolve(strict=True)
        if (
            not result_path.is_file()
            or result_path.is_symlink()
            or _sha256_file(result_path)
            != _require_sha(record["sha256"], "aggregate input SHA")
        ):
            raise HandoffContractError("aggregate input result bytes drifted")
        try:
            envelope = launch._validate_seal(
                _read_json(result_path),
                schema=launch.SEARCH_RESULT_SCHEMA,
            )
            task_sha = _require_sha(
                envelope.get("task_payload_sha256"),
                "result task payload SHA",
            )
            task = bundle_tasks.get(task_sha)
            if task is None:
                raise HandoffContractError(
                    "aggregate result is outside its original bundle task ledger"
                )
            result, frame = launch._validated_seed_table(
                result_path,
                expected_task=task,
            )
        except RuntimeError as exc:
            raise HandoffContractError(
                "aggregate seed terminal evidence failed authentication"
            ) from exc
        if (
            task_sha in observed_task_payloads
            or _integer(result.get("seed"), "result seed")
            != _integer(task.get("seed"), "task seed")
            or _integer(
                result.get("fixed_primary_turns"), "result primary turns"
            )
            != _integer(task.get("fixed_primary_turns"), "task primary turns")
            or _integer(
                result.get("evaluated_generations"),
                "result evaluated generations",
            )
            != launch.GENERATIONS
            or _integer(
                result.get("completed_generations"),
                "result completed generations",
            )
            != launch.EXPECTED_ALGORITHM_N_GEN_COUNTER
            or result.get("fea_submission_performed") is not False
        ):
            raise HandoffContractError("aggregate result is outside its bundle task")
        expected_island = f"n1-{_integer(task['fixed_primary_turns'], 'task turns')}"
        if set(frame["source_island_id"].astype(str)) != {expected_island}:
            raise HandoffContractError("aggregate seed N1 stratum evidence drifted")
        for raw_decoded in frame["decoded_physical_params_json"]:
            decoded = _parse_json_cell(raw_decoded, "terminal decoded params")
            decoded_turns = _integer(decoded.get("N1_main"), "decoded N1_main")
            decoded_turns += _integer(decoded.get("N1_side"), "decoded N1_side")
            if decoded_turns != _integer(
                task["fixed_primary_turns"], "task primary turns"
            ):
                raise HandoffContractError(
                    "aggregate terminal row escaped its N1 stratum"
                )
        observed_task_payloads.add(task_sha)
        observed_seeds.add(_integer(result["seed"], "result seed"))
        result_paths[_sha256_file(result_path)] = result_path
        results.append(result)
        frames.append(frame)
    if observed_task_payloads != set(bundle_tasks):
        raise HandoffContractError("aggregate omits one or more bundle seed results")
    if sorted(observed_seeds) != sorted(
        _integer(value, "aggregate seed") for value in aggregate.get("seeds") or []
    ):
        raise HandoffContractError("aggregate seed list drifted")
    common_fields = (
        "dataset_sha256",
        "evaluation_model_sha256",
        "stage_spec_sha256",
        "temperature_contract_sha256",
        "hard_constraint_contract_sha256",
        "cooling_contract_sha256",
        "operating_point_sha256",
    )
    common = {name: results[0][name] for name in common_fields}
    if (
        aggregate.get("common_identity") != common
        or any(
            any(result.get(name) != value for name, value in common.items())
            for result in results
        )
    ):
        raise HandoffContractError("aggregate mixes seed physics identities")
    merged = pd.concat(frames, ignore_index=True)
    physical_columns = [
        column for column in merged if column.startswith("physical_G:")
    ]
    normalized_columns = [
        column for column in merged if column.startswith("normalized_G:")
    ]
    if (
        len(merged) != launch.POPULATION * seed_count
        or not physical_columns
        or not normalized_columns
    ):
        raise HandoffContractError("aggregate terminal population is incomplete")
    compare_columns = [
        "objective_volume_L",
        "objective_total_loss_W",
        "surrogate_physical_valid",
        "physical_constraint_feasible",
        "physical_feasible",
        *physical_columns,
        *normalized_columns,
    ]
    for _geometry, group in merged.groupby(
        "physical_geometry_sha256", sort=False
    ):
        values = group[compare_columns].to_numpy(dtype=float)
        if len(values) > 1 and not np.allclose(
            values,
            values[[0]],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            raise HandoffContractError(
                "duplicate geometry has inconsistent seed evaluation"
            )
    deduplicated = merged.drop_duplicates(
        "physical_geometry_sha256", keep="first"
    ).copy()
    ranked = rank_candidates(
        deduplicated,
        objective_columns=("objective_volume_L", "objective_total_loss_W"),
        physical_constraint_columns=physical_columns,
        normalized_constraint_columns=normalized_columns,
    )
    ranked["global_physical_feasible"] = ranked["hard_feasible"]
    ranked["global_non_dominated_rank"] = ranked["feasible_rank"]
    pareto = ranked.loc[ranked["global_non_dominated_rank"].eq(0)].copy()
    expected_pareto = set(
        pareto["physical_geometry_sha256"].astype(str).str.lower()
    )
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HandoffContractError("aggregate artifacts are absent")
    loaded_artifacts: dict[str, tuple[Path, pd.DataFrame]] = {}
    for name in (
        "global_terminal_candidates",
        "global_pareto_front",
        "standard_candidates",
    ):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise HandoffContractError(f"aggregate artifact is absent: {name}")
        artifact_path = _contained_file(
            manifest_path.parent, record.get("path"), f"aggregate {name}"
        )
        if _sha256_file(artifact_path) != record.get("sha256"):
            raise HandoffContractError(f"aggregate artifact bytes drifted: {name}")
        frame = pd.read_csv(artifact_path)
        if len(frame) != _integer(record.get("row_count"), f"{name} row_count"):
            raise HandoffContractError(f"aggregate artifact row count drifted: {name}")
        loaded_artifacts[name] = (artifact_path, frame)
    global_terminal = loaded_artifacts["global_terminal_candidates"][1]
    global_pareto = loaded_artifacts["global_pareto_front"][1]
    if (
        set(global_terminal["physical_geometry_sha256"].astype(str).str.lower())
        != set(ranked["physical_geometry_sha256"].astype(str).str.lower())
        or set(global_pareto["physical_geometry_sha256"].astype(str).str.lower())
        != expected_pareto
        or _integer(
            aggregate.get("input_terminal_row_count"),
            "aggregate input terminal rows",
        )
        != len(merged)
        or _integer(
            aggregate.get("deduplicated_physical_geometry_count"),
            "aggregate deduplicated count",
        )
        != len(ranked)
        or _integer(
            aggregate.get("physical_feasible_count"),
            "aggregate feasible count",
        )
        != int(ranked["hard_feasible"].sum())
        or _integer(
            aggregate.get("global_pareto_count"), "aggregate Pareto count"
        )
        != len(pareto)
    ):
        raise HandoffContractError("aggregate global NDS recomputation drifted")
    return {
        "bundle_manifest": _file_record(bundle_manifest_path),
        "bundle_payload_sha256": bundle["payload_sha256"],
        "bundle_task_count": len(bundle_tasks),
        "bundle_task_ledger_sha256": expected_bundle_authority[
            "task_ledger_sha256"
        ],
        "bundle_code_manifest": {
            **copy.deepcopy(bundle["code_manifest"]),
            "file": _file_record(
                _contained_file(
                    bundle_manifest_path.resolve(strict=True).parent,
                    bundle["code_manifest"]["path"],
                    "bundle code manifest",
                )
            ),
            "staged_code_authentication": bundle_code_authentication,
        },
        "bundle_relocation_contract": copy.deepcopy(
            EXPECTED_RELOCATION_CONTRACT
        ),
        "bundle_task_paths": {
            digest: str(path) for digest, path in bundle_task_paths.items()
        },
        "result_paths_by_sha256": {
            digest: str(path) for digest, path in result_paths.items()
        },
        "global_pareto_path": str(
            loaded_artifacts["global_pareto_front"][0]
        ),
        "global_pareto_geometry_sha256": sorted(expected_pareto),
        "global_nds_recomputed": True,
        "all_bundle_seed_results_reauthenticated": True,
        "minimum_seed_count": minimum,
        "seed_count": seed_count,
    }


def authenticate_selection(
    *,
    aggregate_manifest_path: Path | None,
    bundle_manifest_path: Path | None,
    standard_candidates_path: Path | None,
    standard_candidates_sha256: str | None,
    candidate_physics_sha256: str,
    source_result_path: Path,
    task_payload_path: Path,
) -> dict[str, Any]:
    candidate_sha = _require_sha(
        candidate_physics_sha256, "candidate_physics_sha256"
    )
    aggregate = None
    aggregate_authority = None
    if aggregate_manifest_path is not None:
        if bundle_manifest_path is None:
            raise HandoffContractError(
                "aggregate submission authority requires bundle manifest"
            )
        manifest_path = aggregate_manifest_path.resolve(strict=True)
        aggregate = launch._validate_seal(
            _read_json(manifest_path), schema=launch.GLOBAL_PARETO_SCHEMA
        )
        if (
            aggregate.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
            or aggregate.get("hard_spec") != GOAL_STAGE_SPEC
            or aggregate.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
            or aggregate.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
            or aggregate.get("temperature_contract_sha256")
            != GOAL_TEMPERATURE_CONTRACT_SHA256
            or aggregate.get("seed_local_pareto_merge_used") is not False
            or aggregate.get("search_only_proposal") is not False
            or aggregate.get("production_eligible") is not False
            or aggregate.get("automatic_promotion_allowed") is not False
            or "all_authenticated_terminal_rows"
            not in str(aggregate.get("sorting_authority") or "")
        ):
            raise HandoffContractError("aggregate manifest goal authority drifted")
        aggregate_authority = _authenticate_aggregate_authority(
            aggregate=aggregate,
            manifest_path=manifest_path,
            bundle_manifest_path=bundle_manifest_path,
        )
        artifact = (aggregate.get("artifacts") or {}).get("global_pareto_front")
        if not isinstance(artifact, dict):
            raise HandoffContractError("aggregate global Pareto artifact is missing")
        table_path = _contained_file(
            manifest_path.parent,
            artifact.get("path"),
            "aggregate global Pareto",
        )
        if (
            _sha256_file(table_path) != artifact.get("sha256")
            or table_path.stat().st_size <= 0
        ):
            raise HandoffContractError("aggregate global Pareto bytes drifted")
        selection_source = {
            "kind": "aggregate_global_pareto",
            "aggregate_manifest": _file_record(manifest_path),
            "aggregate_authority": aggregate_authority,
            "candidate_table": _file_record(table_path),
            "submission_eligible": True,
        }
    else:
        if bundle_manifest_path is not None:
            raise HandoffContractError(
                "standalone candidate inspection rejects bundle authority"
            )
        if standard_candidates_path is None:
            raise HandoffContractError("candidate table source is missing")
        table_path = standard_candidates_path.resolve(strict=True)
        expected = _require_sha(
            standard_candidates_sha256, "standard_candidates_sha256"
        )
        if _sha256_file(table_path) != expected:
            raise HandoffContractError("standard_candidates bytes SHA-256 mismatch")
        selection_source = {
            "kind": "authenticated_standard_candidates_bytes",
            "candidate_table": _file_record(table_path),
            "operator_supplied_expected_sha256": expected,
            "submission_eligible": False,
            "non_production_reason": (
                "standalone candidate bytes do not prove multi-seed global NDS"
            ),
        }

    table = pd.read_csv(table_path)
    if "physical_geometry_sha256" not in table:
        raise HandoffContractError("candidate table lacks physical geometry identity")
    selected_rows = table.loc[
        table["physical_geometry_sha256"].astype(str).str.lower().eq(candidate_sha)
    ]
    if len(selected_rows) != 1:
        raise HandoffContractError("candidate physical geometry is missing or ambiguous")
    selected = selected_rows.iloc[0].to_dict()
    if aggregate is not None:
        if (
            _integer(selected.get("global_non_dominated_rank"), "global rank")
            != 0
            or not _csv_bool(
                selected.get("global_physical_feasible"),
                "global_physical_feasible",
            )
        ):
            raise HandoffContractError("selected aggregate row is not global Pareto rank 0")
    else:
        rank = selected.get(
            "feasible_rank", selected.get("global_non_dominated_rank")
        )
        if _integer(rank, "standard candidate feasible rank") != 0:
            raise HandoffContractError("standard candidate is not feasible rank 0")

    task_path = task_payload_path.resolve(strict=True)
    task = launch.validate_task_payload(_read_json(task_path))
    if aggregate_authority is not None:
        expected_task_path = Path(
            aggregate_authority["bundle_task_paths"].get(
                task["payload_sha256"], ""
            )
        )
        if (
            task["payload_sha256"]
            not in aggregate_authority["bundle_task_paths"]
            or expected_task_path.resolve(strict=True) != task_path
        ):
            raise HandoffContractError("selected task is outside authenticated bundle")

    source_path = source_result_path.resolve(strict=True)
    try:
        source_result, terminal = launch._validated_seed_table(
            source_path,
            expected_task=task,
        )
    except RuntimeError as exc:
        raise HandoffContractError(
            "selected source result failed original task-ledger authentication"
        ) from exc
    source_result_sha = _sha256_file(source_path)
    if aggregate is not None:
        expected_result_path = Path(
            aggregate_authority["result_paths_by_sha256"].get(
                source_result_sha, ""
            )
        )
        if (
            source_result_sha
            not in aggregate_authority["result_paths_by_sha256"]
            or expected_result_path.resolve(strict=True) != source_path
        ):
            raise HandoffContractError("selected source result is outside aggregate inputs")
    if str(selected.get("source_result_sha256")) != source_result_sha:
        raise HandoffContractError("selected row source result SHA-256 drifted")
    terminal_index = _integer(
        selected.get("terminal_population_index"), "terminal index"
    )
    if not 0 <= terminal_index < launch.POPULATION:
        raise HandoffContractError("selected terminal index is outside 0..319")
    source_row = terminal.iloc[terminal_index].to_dict()
    _rows_equivalent(selected, source_row)

    if (
        source_result.get("task_payload_sha256") != task.get("payload_sha256")
        or str(selected.get("source_bundle_id")) != task.get("payload_sha256")
        or _integer(source_result.get("seed"), "source result seed")
        != _integer(task.get("seed"), "task seed")
        or _integer(selected.get("source_seed"), "selected source seed")
        != _integer(task.get("seed"), "task seed")
        or _integer(
            source_result.get("fixed_primary_turns"),
            "source result primary turns",
        )
        != _integer(task.get("fixed_primary_turns"), "task primary turns")
        or str(selected.get("source_island_id"))
        != f"n1-{_integer(task['fixed_primary_turns'], 'task primary turns')}"
        or source_result.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
    ):
        raise HandoffContractError("source result/task/selected row identity mismatch")
    row_contract = _goal_row_contract(selected)
    if row_contract["primary_turns"] != _integer(
        task["fixed_primary_turns"], "task primary turns"
    ):
        raise HandoffContractError("selected physical row/task N1 stratum drifted")
    return {
        "selection_source": selection_source,
        "selected_row": selected,
        "row_contract": row_contract,
        "source_result": _file_record(source_path),
        "source_result_identity": {
            "schema_version": source_result["schema_version"],
            "payload_sha256": source_result["payload_sha256"],
            "task_payload_sha256": source_result["task_payload_sha256"],
            "seed": _integer(source_result["seed"], "source result seed"),
            "fixed_primary_turns": _integer(
                source_result["fixed_primary_turns"],
                "source result primary turns",
            ),
            "dataset_sha256": source_result["dataset_sha256"],
            "evaluation_model_sha256": source_result["evaluation_model_sha256"],
            "stage_spec_sha256": source_result["stage_spec_sha256"],
            "temperature_contract_sha256": source_result[
                "temperature_contract_sha256"
            ],
            "hard_constraint_contract_sha256": source_result[
                "hard_constraint_contract_sha256"
            ],
        },
        "task_payload": _file_record(task_path),
        "task_identity": {
            "payload_sha256": task["payload_sha256"],
            "task_name": task["task_name"],
            "seed": _integer(task["seed"], "task seed"),
            "fixed_primary_turns": _integer(
                task["fixed_primary_turns"], "task primary turns"
            ),
            "source_code_revision": task["source_identity"]["code_revision"],
            "source_code_manifest_payload_sha256": task["source_identity"][
                "code_manifest_payload_sha256"
            ],
            "source_code_inventory_sha256": task["source_identity"][
                "code_inventory_sha256"
            ],
        },
    }


AUTHENTICATED_SELECTION_FIELDS = (
    "selection_source",
    "selected_row",
    "row_contract",
    "source_result",
    "source_result_identity",
    "task_payload",
    "task_identity",
)


def _selected_authentication_payload(
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    if any(field not in selected for field in AUTHENTICATED_SELECTION_FIELDS):
        raise HandoffContractError("selected candidate authentication is incomplete")
    return {
        field: copy.deepcopy(selected[field])
        for field in AUTHENTICATED_SELECTION_FIELDS
    }


def _reauthenticate_submission_selection(
    *,
    plan: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    selection_source = selected.get("selection_source")
    if not isinstance(selection_source, dict):
        raise HandoffContractError("submission selection source is absent")
    authority = selection_source.get("aggregate_authority")
    if (
        selection_source.get("kind") != "aggregate_global_pareto"
        or selection_source.get("submission_eligible") is not True
        or not isinstance(authority, dict)
    ):
        raise HandoffContractError(
            "submission requires authenticated aggregate global Pareto authority"
        )

    def source_path(record: Any, label: str) -> Path:
        if not isinstance(record, dict):
            raise HandoffContractError(f"{label} source record is absent")
        raw = record.get("path")
        if not isinstance(raw, str) or not raw:
            raise HandoffContractError(f"{label} source path is absent")
        resolved = Path(raw).resolve(strict=True)
        if _file_record(resolved) != record:
            raise HandoffContractError(f"{label} source bytes drifted")
        return resolved

    authenticated = authenticate_selection(
        aggregate_manifest_path=source_path(
            selection_source.get("aggregate_manifest"),
            "aggregate manifest",
        ),
        bundle_manifest_path=source_path(
            authority.get("bundle_manifest"),
            "bundle manifest",
        ),
        standard_candidates_path=None,
        standard_candidates_sha256=None,
        candidate_physics_sha256=str(
            plan.get("candidate_physics_sha256") or ""
        ),
        source_result_path=source_path(
            selected.get("source_result"), "selected source result"
        ),
        task_payload_path=source_path(
            selected.get("task_payload"), "selected task payload"
        ),
    )
    selected_payload = _selected_authentication_payload(selected)
    if authenticated != selected_payload:
        raise HandoffContractError(
            "submission selection differs from freshly authenticated search authority"
        )
    authentication_sha = canonical_sha256(authenticated)
    if authentication_sha != plan.get("search_authority_sha256"):
        raise HandoffContractError(
            "fresh search authority identity differs from handoff plan"
        )
    refreshed_authority = authenticated["selection_source"][
        "aggregate_authority"
    ]
    if (
        refreshed_authority.get("all_bundle_seed_results_reauthenticated")
        is not True
        or refreshed_authority.get("global_nds_recomputed") is not True
        or refreshed_authority.get("seed_count", 0)
        < refreshed_authority.get("minimum_seed_count", 1)
    ):
        raise HandoffContractError(
            "fresh aggregate authority did not reauthenticate every seed and global NDS"
        )
    return {
        "schema_version": "mft-goal-fea-submit-search-reauth-v1",
        "search_authority_sha256": authentication_sha,
        "aggregate_manifest_sha256": authenticated["selection_source"][
            "aggregate_manifest"
        ]["sha256"],
        "bundle_manifest_sha256": refreshed_authority["bundle_manifest"][
            "sha256"
        ],
        "seed_count": refreshed_authority["seed_count"],
        "minimum_seed_count": refreshed_authority["minimum_seed_count"],
        "all_bundle_seed_results_reauthenticated": True,
        "global_nds_recomputed": True,
    }


def _effective_params(
    params: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    effective = scheduler_client.effective_verification_params(
        dict(params), dict(profile)
    )
    if set(effective) != set(ALL_INPUT_KEYS):
        raise HandoffContractError("profile changed exact FEA parameter schema")
    identity = dict(effective)
    identity["thermal_pad_conductivity_W_mK"] = 0.2
    attest_fixed_identity(identity)
    return effective


def create_plan(
    *,
    aggregate_manifest_path: Path | None,
    bundle_manifest_path: Path | None,
    standard_candidates_path: Path | None,
    standard_candidates_sha256: str | None,
    candidate_physics_sha256: str,
    source_result_path: Path,
    task_payload_path: Path,
    solver_revision: str,
    library_revision: str,
    output: Path,
) -> Path:
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    solver = _require_revision(solver_revision, "solver_revision")
    library = _require_revision(library_revision, "library_revision")
    if (aggregate_manifest_path is None) == (standard_candidates_path is None):
        raise HandoffContractError(
            "choose exactly one aggregate manifest or standard_candidates"
        )
    authenticated = authenticate_selection(
        aggregate_manifest_path=aggregate_manifest_path,
        bundle_manifest_path=bundle_manifest_path,
        standard_candidates_path=standard_candidates_path,
        standard_candidates_sha256=standard_candidates_sha256,
        candidate_physics_sha256=candidate_physics_sha256,
        source_result_path=source_result_path,
        task_payload_path=task_payload_path,
    )
    # The Scheduler's established dedupe identity serializes insertion order.
    # Normalize once before deriving either the identity or the on-disk bytes so
    # reloading this immutable plan cannot change the retained-artifact path.
    params = {
        key: authenticated["row_contract"]["fea_params"][key]
        for key in sorted(ALL_INPUT_KEYS)
    }
    profiles = {}
    stage_plans = {}
    stem = authenticated["row_contract"]["physical_geometry_sha256"][:12]
    for stage in ("standard", "full"):
        profile, profile_source = _profile_content(stage)
        effective = _effective_params(params, profile)
        name = f"mft-goal-{stage}-{stem}"
        workdir = f"mft_goal_{stage}_{stem}"
        remote = scheduler_client.retained_aedt_identity(
            name, params, profile, solver, library
        )
        if remote is None:
            raise HandoffContractError(f"{stage} retained AEDT identity is missing")
        profiles[stage] = {
            "profile": profile,
            "source": profile_source,
            "sha256": canonical_sha256(profile),
        }
        stage_plans[stage] = {
            "task_name": name,
            "workdir": workdir,
            "profile_sha256": canonical_sha256(profile),
            "effective_params_sha256": canonical_sha256(effective),
            "resources": copy.deepcopy(STAGE_RESOURCES[stage]),
            "retained_aedt": remote,
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "aedt_backend": "standalone",
        }

    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(f"plan output already exists: {destination}")
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_payload = _seal(
            {
                "schema_version": "mft-goal-fea-selected-candidate-v1",
                "selection_source": authenticated["selection_source"],
                "selected_row": authenticated["selected_row"],
                "row_contract": authenticated["row_contract"],
                "source_result": authenticated["source_result"],
                "source_result_identity": authenticated["source_result_identity"],
                "task_payload": authenticated["task_payload"],
                "task_identity": authenticated["task_identity"],
            }
        )
        selected_path = _write_immutable_json(
            staging / "selected_candidate.json", selected_payload
        )
        params_path = _write_immutable_json(staging / "fea_params.json", params)
        profile_records = {}
        for stage in ("standard", "full"):
            path = _write_immutable_json(
                staging / f"{stage}_profile.json",
                profiles[stage]["profile"],
            )
            profile_records[stage] = {
                "path": path.name,
                "sha256": _sha256_file(path),
                "canonical_sha256": profiles[stage]["sha256"],
                "source": profiles[stage]["source"],
            }
        plan = _seal(
            {
                "schema_version": PLAN_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "solver_revision": solver,
                "library_revision": library,
                "candidate_physics_sha256": authenticated["row_contract"][
                    "physical_geometry_sha256"
                ],
                "search_authority_sha256": canonical_sha256(authenticated),
                "fea_params_sha256": canonical_sha256(params),
                "selected_candidate": {
                    "path": selected_path.name,
                    "sha256": _sha256_file(selected_path),
                },
                "fea_params": {
                    "path": params_path.name,
                    "sha256": _sha256_file(params_path),
                },
                "profiles": profile_records,
                "stages": stage_plans,
                "submission_eligible": authenticated["selection_source"][
                    "submission_eligible"
                ],
                "selection_authority_kind": authenticated[
                    "selection_source"
                ]["kind"],
                "standard_actual_body_probe_gate_required_before_full": True,
                "physics_override_allowed": False,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                "retention_required": True,
                "prune_protection_required": True,
            }
        )
        plan_path = _write_immutable_json(staging / "handoff_plan.json", plan)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _load_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    plan = _validate_seal(_read_json(resolved), PLAN_SCHEMA)
    if (
        plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or plan.get("physics_override_allowed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("retention_required") is not True
        or plan.get("prune_protection_required") is not True
    ):
        raise HandoffContractError("handoff plan contract drifted")
    root = resolved.parent

    def artifact(record: Any, label: str) -> Path:
        if not isinstance(record, dict):
            raise HandoffContractError(f"{label} record is missing")
        target = _contained_file(root, record.get("path"), label)
        if _sha256_file(target) != record.get("sha256"):
            raise HandoffContractError(f"{label} SHA-256 mismatch")
        return target

    params_path = artifact(plan.get("fea_params"), "FEA params")
    params = _read_json(params_path)
    if (
        set(params) != set(ALL_INPUT_KEYS)
        or canonical_sha256(params) != plan.get("fea_params_sha256")
    ):
        raise HandoffContractError("plan FEA parameter identity drifted")
    selected_path = artifact(plan.get("selected_candidate"), "selected candidate")
    selected = _validate_seal(
        _read_json(selected_path), "mft-goal-fea-selected-candidate-v1"
    )
    if (
        selected.get("row_contract", {}).get("physical_geometry_sha256")
        != plan.get("candidate_physics_sha256")
        or selected.get("row_contract", {}).get("fea_params_sha256")
        != plan.get("fea_params_sha256")
        or canonical_sha256(_selected_authentication_payload(selected))
        != plan.get("search_authority_sha256")
    ):
        raise HandoffContractError("plan selected candidate identity drifted")
    selection_source = selected.get("selection_source") or {}
    if (
        plan.get("submission_eligible")
        is not selection_source.get("submission_eligible")
        or plan.get("selection_authority_kind") != selection_source.get("kind")
        or (
            plan.get("submission_eligible") is True
            and (
                selection_source.get("kind") != "aggregate_global_pareto"
                or (
                    selection_source.get("aggregate_authority") or {}
                ).get("global_nds_recomputed")
                is not True
                or (
                    selection_source.get("aggregate_authority") or {}
                ).get("all_bundle_seed_results_reauthenticated")
                is not True
            )
        )
    ):
        raise HandoffContractError("plan submission authority drifted")
    for stage in ("standard", "full"):
        record = (plan.get("profiles") or {}).get(stage)
        profile_path = artifact(record, f"{stage} profile")
        profile = _read_json(profile_path)
        _validate_profile(stage, profile)
        if (
            canonical_sha256(profile) != record.get("canonical_sha256")
            or canonical_sha256(profile)
            != plan["stages"][stage]["profile_sha256"]
        ):
            raise HandoffContractError(f"{stage} profile identity drifted")
        effective = _effective_params(params, profile)
        stage_plan = plan["stages"][stage]
        remote = scheduler_client.retained_aedt_identity(
            stage_plan["task_name"],
            params,
            profile,
            plan["solver_revision"],
            plan["library_revision"],
        )
        if (
            canonical_sha256(effective)
            != stage_plan.get("effective_params_sha256")
            or remote != stage_plan.get("retained_aedt")
            or stage_plan.get("resources") != STAGE_RESOURCES[stage]
            or stage_plan.get("scheduler_project") != scheduler_client.MFT_PROJECT
            or stage_plan.get("aedt_backend") != "standalone"
        ):
            raise HandoffContractError(f"{stage} execution identity drifted")
    return plan, params, selected


def _core_auth(
    solver_revision: str,
    cpus: int,
    *,
    license_contract: str = "",
    license_snapshot_sha256: str = "",
) -> str:
    contract = STANDARD_CORE_CONTRACT if cpus == 8 else FULL_CORE_CONTRACT
    payload: dict[str, Any] = {
        "backend": "standalone",
        "contract_version": contract,
        "requested_num_cores": cpus,
        "required_slurm_cpus_per_task": cpus,
        "solver_revision": solver_revision,
    }
    if cpus == 16:
        payload["license_contract"] = license_contract
        payload["license_snapshot_sha256"] = license_snapshot_sha256
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_license_snapshot(
    path: Path, *, now: datetime | None = None
) -> tuple[str, str]:
    raw = path.resolve(strict=True).read_text(encoding="utf-8").strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HandoffContractError("16-core license snapshot is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "server_up", "server", "checked_at", "features"}
        or value.get("schema") != "mft-aedt-license-headroom-snapshot-v1"
        or value.get("server_up") is not True
        or value.get("server") != "1055@172.16.10.81"
    ):
        raise HandoffContractError("16-core license snapshot identity drifted")
    try:
        checked = datetime.fromisoformat(
            str(value.get("checked_at") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise HandoffContractError("16-core license snapshot timestamp is invalid") from exc
    if checked.tzinfo is None:
        raise HandoffContractError("16-core license snapshot lacks timezone")
    current = now or datetime.now(timezone.utc)
    age = (
        current.astimezone(timezone.utc)
        - checked.astimezone(timezone.utc)
    ).total_seconds()
    required = {
        "anshpc": 16,
        "elec_solve_maxwell": 1,
        "electronics_desktop": 1,
        "electronics3d_gui": 1,
    }
    features = value.get("features")
    if (
        not -30 <= age <= 600
        or not isinstance(features, dict)
        or set(features) != set(required)
    ):
        raise HandoffContractError("16-core license snapshot is stale or incomplete")
    for name, minimum in required.items():
        item = features.get(name)
        if (
            not isinstance(item, dict)
            or set(item) != {"total", "used"}
            or isinstance(item.get("total"), bool)
            or not isinstance(item.get("total"), int)
            or isinstance(item.get("used"), bool)
            or not isinstance(item.get("used"), int)
            or item["total"] < 0
            or item["used"] < 0
            or item["used"] > item["total"]
            or item["total"] - item["used"] < minimum
        ):
            raise HandoffContractError(f"16-core license headroom failed: {name}")
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _submission_environment(
    *, stage: str, solver_revision: str, license_snapshot_path: Path | None
) -> tuple[dict[str, str], dict[str, Any]]:
    cpus = STAGE_RESOURCES[stage]["cpus"]
    contract = STANDARD_CORE_CONTRACT if stage == "standard" else FULL_CORE_CONTRACT
    environment = {
        "MFT_STANDALONE_CORE_CONTRACT": contract,
        "MFT_STANDALONE_CORE_COUNT": str(cpus),
    }
    evidence: dict[str, Any] = {
        "contract": contract,
        "requested_num_cores": cpus,
    }
    if stage == "full":
        if license_snapshot_path is None:
            raise HandoffContractError("Full 16-core submission requires license snapshot")
        _snapshot_json, snapshot_sha = _validate_license_snapshot(
            license_snapshot_path
        )
        environment.update(
            {
                scheduler_client.RUNTIME_LICENSE_REFRESH_ENV: "1",
            }
        )
        evidence["runtime_license_refresh_required"] = True
        evidence["admission_license_snapshot_sha256"] = snapshot_sha
        evidence["admission_license_snapshot"] = _file_record(
            license_snapshot_path
        )
        evidence["admission_auth_sha256"] = _core_auth(
            solver_revision,
            cpus,
            license_contract=FULL_LICENSE_CONTRACT,
            license_snapshot_sha256=snapshot_sha,
        )
    else:
        auth = _core_auth(solver_revision, cpus)
        environment["MFT_STANDALONE_CORE_AUTH_SHA256"] = auth
        evidence["auth_sha256"] = auth
    return environment, evidence


def _temperature_gate_evidence(
    result: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], bool]:
    temperatures = {}
    active = []
    n2_side = _finite(result.get("N2_side"), "Standard N2_side")
    for target in GOAL_TEMPERATURE_TARGETS:
        if target in {
            "T_max_Rx_side",
            "Tprobe_Rx_side_leeward_max",
        } and n2_side <= 0:
            continue
        value = _finite(result.get(target), f"Standard {target}")
        limit = float(TEMPERATURE_TARGET_LIMITS_C[target])
        active.append(target)
        temperatures[target] = {
            "actual_C": value,
            "limit_C": limit,
            "passed": value <= limit,
        }
    passed = (
        len(active)
        in {
            len(GOAL_TEMPERATURE_TARGETS),
            len(GOAL_TEMPERATURE_TARGETS) - 2,
        }
        and all(item["passed"] for item in temperatures.values())
    )
    return active, temperatures, passed


def _load_gate(
    path: Path,
    *,
    plan: Mapping[str, Any],
    standard_collection: Mapping[str, Any],
    standard_collection_path: Path,
) -> dict[str, Any]:
    gate = _validate_seal(_read_json(path.resolve(strict=True)), STANDARD_GATE_SCHEMA)
    active, temperatures, body_probe_pass = _temperature_gate_evidence(
        standard_collection["result"]
    )
    if (
        gate.get("plan_payload_sha256") != plan.get("payload_sha256")
        or gate.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or gate.get("standard_collection")
        != _file_record(standard_collection_path)
        or gate.get("standard_collection_payload_sha256")
        != standard_collection.get("payload_sha256")
        or gate.get("standard_result_sha256")
        != standard_collection.get("result_sha256")
        or gate.get("standard_task_id") != standard_collection.get("task_id")
        or gate.get("active_temperature_targets") != active
        or gate.get("actual_body_probe_temperatures") != temperatures
        or gate.get("goal_physical_spec_reasons")
        != standard_collection.get("goal_physical_spec_reasons")
        or gate.get("passed") is not True
        or gate.get("full_submission_allowed") is not True
        or gate.get("actual_body_probe_temperature_gate_passed")
        is not body_probe_pass
        or gate.get("goal_physical_spec_passed")
        is not standard_collection.get("goal_physical_spec_passed")
        or gate.get("surrogate_only_gate_used") is not False
        or gate.get("legacy_scalar_temperature_gate_used") is not False
    ):
        raise HandoffContractError("authenticated Standard PASS gate is absent")
    return gate


def submit_stage(
    *,
    plan_path: Path,
    stage: str,
    output: Path,
    standard_gate_path: Path | None = None,
    standard_collection_path: Path | None = None,
    license_snapshot_path: Path | None = None,
    priority: int = 0,
    scheduler: Any = scheduler_client,
) -> Path:
    plan, params, selected = _load_plan(plan_path)
    if plan.get("submission_eligible") is not True:
        raise HandoffContractError(
            "standalone candidate plan is inspect-only and cannot submit"
        )
    search_reauthentication = _reauthenticate_submission_selection(
        plan=plan,
        selected=selected,
    )
    preceding_standard = None
    if stage == "full":
        if standard_gate_path is None or standard_collection_path is None:
            raise HandoffContractError(
                "Full submission requires Standard gate and collection"
            )
        standard_collection = _load_collection(
            standard_collection_path,
            plan=plan,
            params=params,
            selected=selected,
            expected_stage="standard",
        )
        gate = _load_gate(
            standard_gate_path,
            plan=plan,
            standard_collection=standard_collection,
            standard_collection_path=standard_collection_path,
        )
        if (
            gate.get("standard_collection_payload_sha256")
            != standard_collection.get("payload_sha256")
            or gate.get("standard_result_sha256")
            != standard_collection.get("result_sha256")
        ):
            raise HandoffContractError("Standard gate/collection identity drifted")
        preceding_standard = {
            "gate": _file_record(standard_gate_path),
            "gate_payload_sha256": gate["payload_sha256"],
            "collection": _file_record(standard_collection_path),
            "collection_payload_sha256": standard_collection[
                "payload_sha256"
            ],
            "result_sha256": standard_collection["result_sha256"],
            "task_id": standard_collection["task_id"],
            "gate_passed": True,
        }
    elif standard_gate_path is not None or standard_collection_path is not None:
        raise HandoffContractError("Standard submission rejects Full gate arguments")
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(f"submission receipt already exists: {target}")
    profile_path = plan_path.resolve(strict=True).parent / plan["profiles"][stage]["path"]
    profile = _read_json(profile_path)
    environment, core_evidence = _submission_environment(
        stage=stage,
        solver_revision=plan["solver_revision"],
        license_snapshot_path=license_snapshot_path,
    )
    stage_plan = plan["stages"][stage]
    task_id = scheduler.submit_verification(
        stage_plan["task_name"],
        stage_plan["workdir"],
        params,
        profile,
        mem_mb=int(profile["mem_mb"]),
        cpus=int(profile["cpus"]),
        solver_revision=plan["solver_revision"],
        library_revision=plan["library_revision"],
        priority=priority,
        aedt_backend="standalone",
        submission_env=environment,
    )
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError("Scheduler submission returned no durable task ID")
    receipt = _seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "stage": stage,
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan["candidate_physics_sha256"],
            "search_authority_reauthentication": search_reauthentication,
            "task_id": task_id,
            "task_name": stage_plan["task_name"],
            "workdir": stage_plan["workdir"],
            "dedupe_key": stage_plan["retained_aedt"]["dedupe_key"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "profile_sha256": stage_plan["profile_sha256"],
            "effective_params_sha256": stage_plan["effective_params_sha256"],
            "resources": stage_plan["resources"],
            "aedt_backend": "standalone",
            "core_policy": core_evidence,
            "retained_aedt": stage_plan["retained_aedt"],
            "scheduler_url": scheduler_client.SCHEDULER.rstrip("/"),
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "preceding_standard_authority": preceding_standard,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": True,
            "retention_required": True,
            "prune_protection_required": True,
        }
    )
    return _write_immutable_json(target, receipt)


def _load_submission(
    path: Path, *, plan: Mapping[str, Any], expected_stage: str
) -> dict[str, Any]:
    receipt = _validate_seal(_read_json(path.resolve(strict=True)), SUBMISSION_SCHEMA)
    stage_plan = plan["stages"][expected_stage]
    search_reauth = receipt.get("search_authority_reauthentication")
    if (
        not isinstance(search_reauth, dict)
        or set(search_reauth)
        != {
            "schema_version",
            "search_authority_sha256",
            "aggregate_manifest_sha256",
            "bundle_manifest_sha256",
            "seed_count",
            "minimum_seed_count",
            "all_bundle_seed_results_reauthenticated",
            "global_nds_recomputed",
        }
        or search_reauth.get("schema_version")
        != "mft-goal-fea-submit-search-reauth-v1"
        or search_reauth.get("search_authority_sha256")
        != plan.get("search_authority_sha256")
        or _require_sha(
            search_reauth.get("aggregate_manifest_sha256"),
            "submission aggregate manifest SHA",
        )
        != search_reauth.get("aggregate_manifest_sha256")
        or _require_sha(
            search_reauth.get("bundle_manifest_sha256"),
            "submission bundle manifest SHA",
        )
        != search_reauth.get("bundle_manifest_sha256")
        or _integer(search_reauth.get("minimum_seed_count"), "minimum seed count")
        < launch.ROLLING_SEED_COUNT
        or _integer(search_reauth.get("seed_count"), "seed count")
        < _integer(search_reauth.get("minimum_seed_count"), "minimum seed count")
        or search_reauth.get("all_bundle_seed_results_reauthenticated")
        is not True
        or search_reauth.get("global_nds_recomputed") is not True
    ):
        raise HandoffContractError(
            f"{expected_stage} submission search reauthentication drifted"
        )
    if (
        receipt.get("stage") != expected_stage
        or receipt.get("plan_payload_sha256") != plan.get("payload_sha256")
        or receipt.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or receipt.get("task_name") != stage_plan["task_name"]
        or receipt.get("dedupe_key")
        != stage_plan["retained_aedt"]["dedupe_key"]
        or receipt.get("retained_aedt") != stage_plan["retained_aedt"]
        or receipt.get("resources") != STAGE_RESOURCES[expected_stage]
        or receipt.get("solver_revision") != plan["solver_revision"]
        or receipt.get("library_revision") != plan["library_revision"]
        or receipt.get("profile_sha256") != stage_plan["profile_sha256"]
        or receipt.get("effective_params_sha256")
        != stage_plan["effective_params_sha256"]
        or receipt.get("aedt_backend") != "standalone"
        or receipt.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or receipt.get("scheduler_url")
        != scheduler_client.SCHEDULER.rstrip("/")
        or receipt.get("scheduler_project_mutation_performed") is not False
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("scheduler_submission_performed") is not True
    ):
        raise HandoffContractError(f"{expected_stage} submission identity drifted")
    preceding = receipt.get("preceding_standard_authority")
    core_policy = receipt.get("core_policy")
    if not isinstance(core_policy, dict):
        raise HandoffContractError(
            f"{expected_stage} submission core policy is absent"
        )
    if expected_stage == "standard":
        if (
            preceding is not None
            or core_policy
            != {
                "contract": STANDARD_CORE_CONTRACT,
                "requested_num_cores": 8,
                "auth_sha256": _core_auth(plan["solver_revision"], 8),
            }
        ):
            raise HandoffContractError(
                "Standard submission contains preceding-stage authority"
            )
    elif (
        not isinstance(preceding, dict)
        or set(preceding)
        != {
            "gate",
            "gate_payload_sha256",
            "collection",
            "collection_payload_sha256",
            "result_sha256",
            "task_id",
            "gate_passed",
        }
        or preceding.get("gate_passed") is not True
        or core_policy.get("contract") != FULL_CORE_CONTRACT
        or core_policy.get("requested_num_cores") != 16
        or core_policy.get("runtime_license_refresh_required") is not True
        or _require_sha(
            core_policy.get("admission_license_snapshot_sha256"),
            "admission license snapshot SHA",
        )
        != core_policy.get("admission_license_snapshot_sha256")
        or _require_sha(
            core_policy.get("admission_auth_sha256"),
            "admission core auth SHA",
        )
        != core_policy.get("admission_auth_sha256")
        or core_policy.get("admission_auth_sha256")
        != _core_auth(
            plan["solver_revision"],
            16,
            license_contract=FULL_LICENSE_CONTRACT,
            license_snapshot_sha256=core_policy.get(
                "admission_license_snapshot_sha256"
            ),
        )
        or not isinstance(
            core_policy.get("admission_license_snapshot"), dict
        )
    ):
        raise HandoffContractError("Full submission lacks Standard PASS authority")
    task_id = receipt.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError("submission task ID is invalid")
    return receipt


def _remote_bytes(
    *, scheduler_url: str, task_id: int, relative_path: str, max_bytes: int
) -> bytes:
    pure = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure.is_absolute()
        or ".." in pure.parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or not 1 <= max_bytes <= MAX_REMOTE_METADATA_BYTES
    ):
        raise HandoffContractError("unsafe Scheduler remote metadata request")
    query = urllib.parse.urlencode(
        {"path": relative_path, "base": "remote_cwd", "max_bytes": max_bytes}
    )
    request = urllib.request.Request(
        scheduler_url.rstrip("/")
        + f"/api/tasks/{task_id}/remote-file?{query}",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response:
            payload = response.read(max_bytes)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise HandoffContractError(
            f"Scheduler remote metadata read failed: {relative_path}"
        ) from exc
    if len(payload) == max_bytes:
        raise HandoffContractError("Scheduler remote metadata reached hard bound")
    return payload


def _validate_remote_receipt_payload(
    receipt: Any,
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    expected = submission["retained_aedt"]
    if (
        not isinstance(receipt, dict)
        or set(receipt) != REMOTE_RECEIPT_FIELDS
        or receipt["schema_version"] != REMOTE_RECEIPT_SCHEMA
        or receipt["stage"] != submission["stage"]
        or receipt["dedupe_key"] != expected["dedupe_key"]
        or receipt["parameter_digest"] != expected["parameter_digest"]
        or receipt["solver_revision"] != submission["solver_revision"]
        or receipt["library_revision"] != submission["library_revision"]
        or receipt["profile_sha256"] != submission["profile_sha256"]
        or receipt["artifact_path"] != expected["artifact_path"]
        or receipt["marker_path"] != expected["marker_path"]
        or receipt["transport_schema_version"]
        != expected["transport"]["schema_version"]
        or receipt["transport_encoding"]
        != expected["transport"]["encoding"]
        or receipt["transport_chunk_directory"]
        != expected["transport"]["chunk_directory"]
        or receipt["transport_raw_chunk_bytes"]
        != expected["transport"]["raw_chunk_bytes"]
        or receipt["transport_max_encoded_chunk_bytes"]
        != expected["transport"]["max_encoded_chunk_bytes"]
        or receipt["retention_required"] is not True
        or receipt["prune_protection_required"] is not True
        or receipt["scheduler_cleanup_exclusion_required"] is not True
        or _require_sha(receipt["artifact_sha256"], "remote AEDT SHA")
        != receipt["artifact_sha256"]
        or isinstance(receipt["artifact_size_bytes"], bool)
        or not isinstance(receipt["artifact_size_bytes"], int)
        or not 0 < receipt["artifact_size_bytes"] <= MAX_AEDT_BYTES
        or isinstance(receipt["transport_chunk_count"], bool)
        or not isinstance(receipt["transport_chunk_count"], int)
        or receipt["transport_chunk_count"]
        != math.ceil(
            receipt["artifact_size_bytes"]
            / receipt["transport_raw_chunk_bytes"]
        )
        or receipt["marker_contract_sha256"]
        != expected["marker_contract_sha256"]
        or receipt["source_project_filename"]
        != f"{receipt['source_project_name']}.aedt"
        or str(result.get("project_name") or "")
        != receipt["source_project_name"]
    ):
        raise HandoffContractError("remote retained AEDT receipt identity drifted")
    return receipt


def _validate_marker_payload(
    marker: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(marker, dict):
        raise HandoffContractError("remote prune-protection marker is not an object")
    contract = dict(marker)
    created_at = contract.pop("created_at", None)
    try:
        parsed_created_at = datetime.fromisoformat(
            str(created_at or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise HandoffContractError(
            "remote prune-protection marker timestamp is invalid"
        ) from exc
    if (
        parsed_created_at.tzinfo is None
        or contract != expected["marker_contract"]
        or canonical_sha256(contract) != expected["marker_contract_sha256"]
    ):
        raise HandoffContractError("remote prune-protection marker payload drifted")
    return marker


def _validated_remote_artifact_receipt(
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
    scheduler_url: str,
    remote_reader: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = int(submission["task_id"])
    expected = submission["retained_aedt"]
    raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=expected["receipt_path"],
        max_bytes=MAX_REMOTE_METADATA_BYTES,
    )
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError("remote AEDT receipt is invalid JSON") from exc
    receipt = _validate_remote_receipt_payload(
        receipt, submission=submission, result=result
    )
    marker_raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=expected["marker_path"],
        max_bytes=MAX_REMOTE_METADATA_BYTES,
    )
    if _sha256_bytes(marker_raw) != receipt["marker_sha256"]:
        raise HandoffContractError("remote prune-protection marker SHA drifted")
    try:
        marker = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError("remote prune-protection marker is invalid") from exc
    marker = _validate_marker_payload(marker, expected=expected)
    return receipt, marker


def _goal_result_reasons(
    result: Mapping[str, Any], selected: Mapping[str, Any]
) -> list[str]:
    augmented = copy.deepcopy(dict(result))
    augmented.update(
        {
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        }
    )
    candidate = copy.deepcopy(selected["row_contract"]["decoded_params"])
    candidate.update(
        {
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
        }
    )
    return finalize.physical_spec_reasons(augmented, candidate=candidate)


def _validate_result_core_policy(
    result: Mapping[str, Any], submission: Mapping[str, Any]
) -> None:
    stage = submission["stage"]
    cpus = STAGE_RESOURCES[stage]["cpus"]
    core_policy = submission["core_policy"]
    license_sha = str(
        result.get("solver_core_license_snapshot_sha256") or ""
    )
    expected_auth = (
        _core_auth(
            submission["solver_revision"],
            cpus,
            license_contract=FULL_LICENSE_CONTRACT,
            license_snapshot_sha256=_require_sha(
                license_sha, "runtime license snapshot SHA"
            ),
        )
        if stage == "full"
        else core_policy["auth_sha256"]
    )
    if (
        result.get("solver_core_policy_schema") != "mft-solver-core-policy-v1"
        or result.get("solver_core_contract_version") != core_policy["contract"]
        or _integer(result.get("solver_core_opt_in"), "solver core opt-in") != 1
        or result.get("solver_core_backend") != "standalone"
        or _integer(
            result.get("solver_num_cores_requested"),
            "solver requested cores",
        )
        != cpus
        or _integer(
            result.get("solver_num_cores_effective"),
            "solver effective cores",
        )
        != cpus
        or _integer(
            result.get("solver_num_tasks_effective"),
            "solver effective tasks",
        )
        != 1
        or _integer(
            result.get("solver_core_affinity_count_readback"),
            "solver affinity readback",
        )
        < cpus
        or str(result.get("solver_core_slurm_cpus_per_task_readback") or "")
        != str(cpus)
        or str(result.get("solver_core_scheduler_task_id_readback") or "")
        != str(submission["task_id"])
        or not re.fullmatch(
            r"[1-9][0-9]*",
            str(result.get("solver_core_slurm_job_id_readback") or ""),
        )
        or result.get("solver_core_auth_sha256") != expected_auth
        or _integer(
            result.get("solver_matrix_hpc_num_cores_readback"),
            "matrix HPC core readback",
        )
        != cpus
        or _integer(
            result.get("solver_matrix_hpc_num_engines_readback"),
            "matrix HPC engine readback",
        )
        != 1
        or not HEX64.fullmatch(
            str(result.get("solver_matrix_hpc_acf_sha256") or "")
        )
    ):
        raise HandoffContractError(f"{stage} solver core readback drifted")
    license_contract = str(result.get("solver_core_license_contract") or "")
    if stage == "full":
        try:
            checked_at = datetime.fromisoformat(
                str(
                    result.get("solver_core_license_checked_at_readback")
                    or ""
                ).replace("Z", "+00:00")
            )
            headroom = json.loads(
                str(
                    result.get(
                        "solver_core_license_headroom_readback_json"
                    )
                    or ""
                )
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise HandoffContractError(
                "full solver license evidence is invalid"
            ) from exc
        required = {
            "anshpc": 16,
            "elec_solve_maxwell": 1,
            "electronics_desktop": 1,
            "electronics3d_gui": 1,
        }
        if (
            license_contract != FULL_LICENSE_CONTRACT
            or checked_at.tzinfo is None
            or not -30
            <= _finite(
                result.get(
                    "solver_core_license_snapshot_age_seconds_readback"
                ),
                "runtime license snapshot age",
            )
            <= 600
            or not isinstance(headroom, dict)
            or set(headroom) != set(required)
            or any(
                isinstance(headroom[name], bool)
                or not isinstance(headroom[name], int)
                or headroom[name] < minimum
                for name, minimum in required.items()
            )
        ):
            raise HandoffContractError("full solver license readback drifted")
    elif license_contract or license_sha:
        raise HandoffContractError(
            "standard solver unexpectedly reported Full license evidence"
        )


def collect_stage(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str = scheduler_client.SCHEDULER,
    scheduler: Any = scheduler_client,
    remote_reader: Any = _remote_bytes,
) -> Path:
    plan, params, selected = _load_plan(plan_path)
    if plan.get("submission_eligible") is not True:
        raise HandoffContractError(
            "inspect-only plan cannot collect Scheduler evidence"
        )
    raw_submission = _read_json(submission_path.resolve(strict=True))
    stage = str(raw_submission.get("stage") or "")
    if stage not in {"standard", "full"}:
        raise HandoffContractError("submission stage is invalid")
    submission = _load_submission(
        submission_path, plan=plan, expected_stage=stage
    )
    normalized_scheduler_url = scheduler_url.rstrip("/")
    if normalized_scheduler_url != submission["scheduler_url"]:
        raise HandoffContractError(
            "collection Scheduler origin differs from submission origin"
        )
    status = scheduler.get_status(int(submission["task_id"]))
    if status != "completed":
        raise HandoffContractError(
            f"{stage} task is not completed: status={status!r}"
        )
    profile_path = plan_path.resolve(strict=True).parent / plan["profiles"][stage]["path"]
    profile = _read_json(profile_path)
    fetched = scheduler.fetch_result(
        int(submission["task_id"]),
        expected_revision=plan["solver_revision"],
        expected_library_revision=plan["library_revision"],
        expected_profile=profile["param_overrides"],
    )
    if fetched.state != scheduler.RESULT_VALID or not isinstance(fetched.result, dict):
        raise HandoffContractError(f"{stage} Scheduler result is not valid")
    result = copy.deepcopy(fetched.result)
    effective = _effective_params(params, profile)
    if not scheduler.result_matches_params(
        result, effective, required_keys=set(ALL_INPUT_KEYS)
    ):
        raise HandoffContractError(f"{stage} result does not echo exact FEA params")
    if (
        _require_revision(result.get("git_hash"), "result git_hash")
        != plan["solver_revision"]
        or _require_revision(
            result.get("pyaedt_library_git_hash"), "result library hash"
        )
        != plan["library_revision"]
        or _integer(result.get("full_model"), "result full_model")
        != STAGE_MODE[stage]["full_model"]
    ):
        raise HandoffContractError(f"{stage} result runtime identity drifted")
    _validate_result_core_policy(result, submission)
    remote_receipt, marker = _validated_remote_artifact_receipt(
        submission=submission,
        result=result,
        scheduler_url=scheduler_url,
        remote_reader=remote_reader,
    )
    reasons = _goal_result_reasons(result, selected)
    result_sha = canonical_sha256(result)
    collection = _seal(
        {
            "schema_version": COLLECTION_SCHEMA,
            "stage": stage,
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission": _file_record(submission_path),
            "submission_payload_sha256": submission["payload_sha256"],
            "candidate_physics_sha256": plan["candidate_physics_sha256"],
            "task_id": submission["task_id"],
            "scheduler_url": submission["scheduler_url"],
            "scheduler_status": status,
            "result": result,
            "result_sha256": result_sha,
            "result_identity": {
                "project_name": result["project_name"],
                "solver_revision": plan["solver_revision"],
                "library_revision": plan["library_revision"],
                "effective_params_sha256": submission["effective_params_sha256"],
                "full_model": STAGE_MODE[stage]["full_model"],
                "solver_core_auth_sha256": result[
                    "solver_core_auth_sha256"
                ],
                "solver_num_cores_effective": STAGE_RESOURCES[stage]["cpus"],
                "solver_core_license_snapshot_sha256": str(
                    result.get("solver_core_license_snapshot_sha256") or ""
                ),
            },
            "remote_aedt_receipt": remote_receipt,
            "prune_protection_marker": marker,
            "prune_protection_marker_verified": True,
            "goal_physical_spec_reasons": reasons,
            "goal_physical_spec_passed": not reasons,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "retention_required_until_package": True,
            "production_eligible": False,
        }
    )
    return _write_immutable_json(output.resolve(), collection)


def _load_collection(
    path: Path,
    *,
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    selected: Mapping[str, Any],
    expected_stage: str,
) -> dict[str, Any]:
    value = _validate_seal(_read_json(path.resolve(strict=True)), COLLECTION_SCHEMA)
    result = value.get("result")
    result_sha = canonical_sha256(result) if isinstance(result, dict) else None
    submission_record = value.get("submission")
    if not isinstance(submission_record, dict):
        raise HandoffContractError(
            f"{expected_stage} collection submission record is absent"
        )
    submission_path = Path(str(submission_record.get("path") or ""))
    if _file_record(submission_path) != submission_record:
        raise HandoffContractError(
            f"{expected_stage} collection submission bytes drifted"
        )
    submission = _load_submission(
        submission_path, plan=plan, expected_stage=expected_stage
    )
    if (
        value.get("stage") != expected_stage
        or value.get("plan_payload_sha256") != plan.get("payload_sha256")
        or value.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or value.get("scheduler_status") != "completed"
        or value.get("scheduler_url") != submission.get("scheduler_url")
        or value.get("submission_payload_sha256")
        != submission.get("payload_sha256")
        or value.get("task_id") != submission.get("task_id")
        or value.get("result_sha256") != result_sha
        or value.get("prune_protection_marker_verified") is not True
        or value.get("scheduler_get_only_collection") is not True
        or value.get("scheduler_mutation_performed") is not False
        or value.get("retention_required_until_package") is not True
    ):
        raise HandoffContractError(f"{expected_stage} collection identity drifted")
    # Callers pass the sealed plan object rather than a plan wrapper; recover
    # the immutable profile beside the plan path recorded by the collection.
    plan_record = value.get("plan")
    if not isinstance(plan_record, dict):
        raise HandoffContractError(
            f"{expected_stage} collection plan record is absent"
        )
    plan_source = Path(str(plan_record.get("path") or ""))
    if _file_record(plan_source) != plan_record:
        raise HandoffContractError(f"{expected_stage} collection plan bytes drifted")
    profile_path = plan_source.parent / plan["profiles"][expected_stage]["path"]
    profile = _read_json(profile_path)
    _validate_profile(expected_stage, profile)
    effective = _effective_params(params, profile)
    if (
        not isinstance(result, dict)
        or not scheduler_client.result_matches_params(
            result, effective, required_keys=set(ALL_INPUT_KEYS)
        )
        or _require_revision(result.get("git_hash"), "result git_hash")
        != plan["solver_revision"]
        or _require_revision(
            result.get("pyaedt_library_git_hash"), "result library hash"
        )
        != plan["library_revision"]
        or _integer(result.get("full_model"), "result full_model")
        != STAGE_MODE[expected_stage]["full_model"]
    ):
        raise HandoffContractError(
            f"{expected_stage} collection result identity drifted"
        )
    _validate_result_core_policy(result, submission)
    receipt = _validate_remote_receipt_payload(
        value.get("remote_aedt_receipt"),
        submission=submission,
        result=result,
    )
    marker = _validate_marker_payload(
        value.get("prune_protection_marker"),
        expected=submission["retained_aedt"],
    )
    if _sha256_bytes(_json_bytes(marker)) != receipt["marker_sha256"]:
        raise HandoffContractError(
            f"{expected_stage} collection marker SHA drifted"
        )
    reasons = _goal_result_reasons(result, selected)
    if (
        value.get("goal_physical_spec_reasons") != reasons
        or value.get("goal_physical_spec_passed") != (not reasons)
        or value.get("result_identity")
        != {
            "project_name": result["project_name"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "effective_params_sha256": submission["effective_params_sha256"],
            "full_model": STAGE_MODE[expected_stage]["full_model"],
            "solver_core_auth_sha256": result["solver_core_auth_sha256"],
            "solver_num_cores_effective": STAGE_RESOURCES[expected_stage][
                "cpus"
            ],
            "solver_core_license_snapshot_sha256": str(
                result.get("solver_core_license_snapshot_sha256") or ""
            ),
        }
    ):
        raise HandoffContractError(
            f"{expected_stage} collection physical/result evidence drifted"
        )
    return value


def create_standard_gate(
    *, plan_path: Path, standard_collection_path: Path, output: Path
) -> tuple[Path, bool]:
    plan, params, selected = _load_plan(plan_path)
    if plan.get("submission_eligible") is not True:
        raise HandoffContractError("inspect-only plan cannot create a FEA gate")
    collection = _load_collection(
        standard_collection_path,
        plan=plan,
        params=params,
        selected=selected,
        expected_stage="standard",
    )
    result = collection["result"]
    active, temperatures, body_probe_pass = _temperature_gate_evidence(
        result
    )
    physical_pass = collection.get("goal_physical_spec_passed") is True
    passed = bool(body_probe_pass and physical_pass)
    gate = _seal(
        {
            "schema_version": STANDARD_GATE_SCHEMA,
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "standard_collection": _file_record(standard_collection_path),
            "standard_collection_payload_sha256": collection["payload_sha256"],
            "standard_result_sha256": collection["result_sha256"],
            "standard_task_id": collection["task_id"],
            "candidate_physics_sha256": plan["candidate_physics_sha256"],
            "active_temperature_targets": active,
            "actual_body_probe_temperatures": temperatures,
            "actual_body_probe_temperature_gate_passed": body_probe_pass,
            "goal_physical_spec_reasons": collection[
                "goal_physical_spec_reasons"
            ],
            "goal_physical_spec_passed": physical_pass,
            "full_submission_allowed": passed,
            "passed": passed,
            "surrogate_only_gate_used": False,
            "legacy_scalar_temperature_gate_used": False,
            "production_eligible": False,
        }
    )
    return _write_immutable_json(output.resolve(), gate), passed


def _fetch_remote_to_path(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    transport_chunk_directory: str,
    transport_raw_chunk_bytes: int,
    transport_max_encoded_chunk_bytes: int,
    transport_chunk_count: int,
    expected_size: int,
    expected_sha256: str,
    destination: Path,
) -> None:
    pure = PurePosixPath(relative_path)
    chunk_root = PurePosixPath(transport_chunk_directory)
    if (
        not relative_path
        or pure.is_absolute()
        or ".." in pure.parts
        or not transport_chunk_directory
        or chunk_root.is_absolute()
        or ".." in chunk_root.parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 0 < expected_size <= MAX_AEDT_BYTES
        or _require_sha(expected_sha256, "remote AEDT SHA") != expected_sha256
        or transport_raw_chunk_bytes
        != scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
        or transport_max_encoded_chunk_bytes
        != scheduler_client.RETAINED_AEDT_MAX_ENCODED_CHUNK_BYTES
        or transport_max_encoded_chunk_bytes > MAX_REMOTE_TEXT_CHUNK_BYTES
        or isinstance(transport_chunk_count, bool)
        or not isinstance(transport_chunk_count, int)
        or transport_chunk_count
        != math.ceil(expected_size / transport_raw_chunk_bytes)
        or destination.exists()
    ):
        raise HandoffContractError("unsafe retained AEDT fetch contract")
    digest = hashlib.sha256()
    count = 0
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            for index in range(transport_chunk_count):
                relative_chunk = (
                    f"{transport_chunk_directory}/{index:08d}.b64"
                )
                query = urllib.parse.urlencode(
                    {
                        "path": relative_chunk,
                        "base": "remote_cwd",
                        "max_bytes": MAX_REMOTE_TEXT_CHUNK_BYTES,
                    }
                )
                request = urllib.request.Request(
                    scheduler_url.rstrip("/")
                    + f"/api/tasks/{task_id}/remote-file?{query}",
                    method="GET",
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=120.0
                    ) as response:
                        encoded = response.read(
                            MAX_REMOTE_TEXT_CHUNK_BYTES + 1
                        )
                except (
                    OSError,
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                ) as exc:
                    raise HandoffContractError(
                        f"Scheduler AEDT text chunk read failed: {index}"
                    ) from exc
                raw_size = min(
                    transport_raw_chunk_bytes,
                    expected_size - count,
                )
                encoded_size = 4 * math.ceil(raw_size / 3)
                if (
                    len(encoded) != encoded_size
                    or len(encoded) > transport_max_encoded_chunk_bytes
                ):
                    raise HandoffContractError(
                        f"remote AEDT text chunk size drifted: {index}"
                    )
                try:
                    chunk = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise HandoffContractError(
                        f"remote AEDT text chunk is invalid base64: {index}"
                    ) from exc
                if len(chunk) != raw_size:
                    raise HandoffContractError(
                        f"remote AEDT raw chunk size drifted: {index}"
                    )
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected_size or digest.hexdigest() != expected_sha256:
            raise HandoffContractError("remote AEDT bytes differ from sealed receipt")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def package_results(
    *,
    plan_path: Path,
    standard_collection_path: Path,
    standard_gate_path: Path,
    full_collection_path: Path,
    output: Path,
    scheduler_url: str = scheduler_client.SCHEDULER,
    remote_fetcher: Any = _fetch_remote_to_path,
) -> Path:
    plan, params, selected = _load_plan(plan_path)
    if plan.get("submission_eligible") is not True:
        raise HandoffContractError("inspect-only plan cannot package FEA evidence")
    standard = _load_collection(
        standard_collection_path,
        plan=plan,
        params=params,
        selected=selected,
        expected_stage="standard",
    )
    gate = _load_gate(
        standard_gate_path,
        plan=plan,
        standard_collection=standard,
        standard_collection_path=standard_collection_path,
    )
    full = _load_collection(
        full_collection_path,
        plan=plan,
        params=params,
        selected=selected,
        expected_stage="full",
    )
    if (
        scheduler_url.rstrip("/") != standard.get("scheduler_url")
        or scheduler_url.rstrip("/") != full.get("scheduler_url")
    ):
        raise HandoffContractError(
            "package Scheduler origin differs from collection origin"
        )
    full_submission_path = Path(full["submission"]["path"])
    full_submission = _load_submission(
        full_submission_path, plan=plan, expected_stage="full"
    )
    preceding = full_submission["preceding_standard_authority"]
    if (
        gate.get("standard_collection_payload_sha256")
        != standard["payload_sha256"]
        or gate.get("standard_result_sha256") != standard["result_sha256"]
        or standard.get("goal_physical_spec_passed") is not True
        or full.get("goal_physical_spec_passed") is not True
        or _integer(standard["result"].get("full_model"), "Standard full_model")
        != 0
        or _integer(full["result"].get("full_model"), "Full full_model") != 1
        or preceding.get("gate") != _file_record(standard_gate_path)
        or preceding.get("gate_payload_sha256") != gate["payload_sha256"]
        or preceding.get("collection")
        != _file_record(standard_collection_path)
        or preceding.get("collection_payload_sha256")
        != standard["payload_sha256"]
        or preceding.get("result_sha256") != standard["result_sha256"]
        or preceding.get("task_id") != standard["task_id"]
    ):
        raise HandoffContractError("Standard/Full package release gates failed")
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(f"package output already exists: {destination}")
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        artifact_records = {}
        marker_records = {}
        for stage, collection, filename in (
            ("standard", standard, "symmetric.aedt"),
            ("full", full, "full.aedt"),
        ):
            receipt = collection["remote_aedt_receipt"]
            target = staging / filename
            remote_fetcher(
                scheduler_url=scheduler_url,
                task_id=int(collection["task_id"]),
                relative_path=receipt["artifact_path"],
                transport_chunk_directory=receipt[
                    "transport_chunk_directory"
                ],
                transport_raw_chunk_bytes=receipt[
                    "transport_raw_chunk_bytes"
                ],
                transport_max_encoded_chunk_bytes=receipt[
                    "transport_max_encoded_chunk_bytes"
                ],
                transport_chunk_count=receipt["transport_chunk_count"],
                expected_size=int(receipt["artifact_size_bytes"]),
                expected_sha256=receipt["artifact_sha256"],
                destination=target,
            )
            if (
                not target.is_file()
                or target.suffix.lower() != ".aedt"
                or _sha256_file(target) != receipt["artifact_sha256"]
                or target.stat().st_size != receipt["artifact_size_bytes"]
            ):
                raise HandoffContractError(f"{stage} packaged AEDT verification failed")
            artifact_records[stage] = {
                "path": target.name,
                "sha256": receipt["artifact_sha256"],
                "size_bytes": receipt["artifact_size_bytes"],
                "source_task_id": collection["task_id"],
                "source_remote_path": receipt["artifact_path"],
                "source_transport": {
                    "schema_version": receipt["transport_schema_version"],
                    "encoding": receipt["transport_encoding"],
                    "chunk_directory": receipt[
                        "transport_chunk_directory"
                    ],
                    "raw_chunk_bytes": receipt[
                        "transport_raw_chunk_bytes"
                    ],
                    "max_encoded_chunk_bytes": receipt[
                        "transport_max_encoded_chunk_bytes"
                    ],
                    "chunk_count": receipt["transport_chunk_count"],
                },
                "source_project_name": receipt["source_project_name"],
            }
            marker = collection["prune_protection_marker"]
            marker_sha256 = _sha256_bytes(_json_bytes(marker))
            if (
                marker_sha256 != receipt["marker_sha256"]
                or receipt["marker_contract_sha256"]
                != plan["stages"][stage]["retained_aedt"][
                    "marker_contract_sha256"
                ]
            ):
                raise HandoffContractError(
                    f"{stage} packaged prune-protection evidence drifted"
                )
            marker_records[stage] = {
                "schema": marker["schema"],
                "created_at": marker["created_at"],
                "sha256": marker_sha256,
                "contract_sha256": receipt["marker_contract_sha256"],
                "source_task_id": collection["task_id"],
                "source_remote_path": receipt["marker_path"],
                "payload": marker,
            }
        copied_json = {}
        plan_root = plan_path.resolve(strict=True).parent
        for label, source, value in (
            ("handoff_plan", plan_path, plan),
            ("selected_candidate", None, selected),
            (
                "fea_params",
                plan_root / plan["fea_params"]["path"],
                params,
            ),
            (
                "standard_profile",
                plan_root / plan["profiles"]["standard"]["path"],
                _read_json(
                    plan_root / plan["profiles"]["standard"]["path"]
                ),
            ),
            (
                "full_profile",
                plan_root / plan["profiles"]["full"]["path"],
                _read_json(plan_root / plan["profiles"]["full"]["path"]),
            ),
            (
                "standard_submission",
                Path(standard["submission"]["path"]),
                _read_json(Path(standard["submission"]["path"])),
            ),
            ("standard_collection", standard_collection_path, standard),
            ("standard_gate", standard_gate_path, gate),
            (
                "full_submission",
                full_submission_path,
                full_submission,
            ),
            ("full_collection", full_collection_path, full),
        ):
            target = staging / f"{label}.json"
            _write_immutable_json(target, value)
            copied_json[label] = {
                "path": target.name,
                "sha256": _sha256_file(target),
                "source": None if source is None else _file_record(source),
            }
        manifest = _seal(
            {
                "schema_version": PACKAGE_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "candidate_physics_sha256": plan["candidate_physics_sha256"],
                "solver_revision": plan["solver_revision"],
                "library_revision": plan["library_revision"],
                "search_source_identities": {
                    "selection_source": selected["selection_source"],
                    "source_result": selected["source_result"],
                    "source_result_identity": selected[
                        "source_result_identity"
                    ],
                    "task_payload": selected["task_payload"],
                    "task_identity": selected["task_identity"],
                },
                "aedt_artifacts": artifact_records,
                "prune_protection_markers": marker_records,
                "result_identities": {
                    "standard": {
                        **standard["result_identity"],
                        "result_sha256": standard["result_sha256"],
                        "task_id": standard["task_id"],
                    },
                    "full": {
                        **full["result_identity"],
                        "result_sha256": full["result_sha256"],
                        "task_id": full["task_id"],
                    },
                },
                "standard_to_full_transition": {
                    "standard_task_id": standard["task_id"],
                    "standard_result_sha256": standard["result_sha256"],
                    "standard_collection_payload_sha256": standard[
                        "payload_sha256"
                    ],
                    "standard_gate_payload_sha256": gate["payload_sha256"],
                    "full_task_id": full["task_id"],
                    "full_submission_payload_sha256": full_submission[
                        "payload_sha256"
                    ],
                    "authenticated_standard_pass_preceded_full_submission": True,
                },
                "evidence": copied_json,
                "standard_actual_body_probe_gate_passed": True,
                "standard_goal_physical_spec_passed": True,
                "full_goal_physical_spec_passed": True,
                "retention_required_during_collection": True,
                "prune_protection_marker_verified_for_both": True,
                "source_remote_artifacts_must_not_be_pruned_before_package": True,
                "package_fea_execution_self_contained": True,
                "search_source_bytes_referenced_by_sha256": True,
                "scheduler_get_only_package_fetch": True,
                "scheduler_mutation_performed": False,
                "scheduler_repository_modified": False,
                "production_eligible": False,
            }
        )
        manifest_path = _write_immutable_json(staging / "package_manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / manifest_path.name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticated goal Pareto-to-Standard/Full FEA handoff"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    sources = plan.add_mutually_exclusive_group(required=True)
    sources.add_argument("--aggregate-manifest", type=Path)
    sources.add_argument("--standard-candidates", type=Path)
    plan.add_argument("--standard-candidates-sha256")
    plan.add_argument("--bundle-manifest", type=Path)
    plan.add_argument("--candidate-physics-sha256", required=True)
    plan.add_argument("--source-result", type=Path, required=True)
    plan.add_argument("--task-payload", type=Path, required=True)
    plan.add_argument("--solver-revision", required=True)
    plan.add_argument("--library-revision", required=True)
    plan.add_argument("--output", type=Path, required=True)

    standard = commands.add_parser("submit-standard")
    standard.add_argument("--plan", type=Path, required=True)
    standard.add_argument("--output", type=Path, required=True)
    standard.add_argument("--priority", type=int, default=0)

    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--submission", type=Path, required=True)
    collect.add_argument("--scheduler-url", default=scheduler_client.SCHEDULER)
    collect.add_argument("--output", type=Path, required=True)

    gate = commands.add_parser("gate")
    gate.add_argument("--plan", type=Path, required=True)
    gate.add_argument("--standard-collection", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    full = commands.add_parser("submit-full")
    full.add_argument("--plan", type=Path, required=True)
    full.add_argument("--standard-gate", type=Path, required=True)
    full.add_argument("--standard-collection", type=Path, required=True)
    full.add_argument("--license-snapshot", type=Path, required=True)
    full.add_argument("--priority", type=int, default=0)
    full.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package")
    package.add_argument("--plan", type=Path, required=True)
    package.add_argument("--standard-collection", type=Path, required=True)
    package.add_argument("--standard-gate", type=Path, required=True)
    package.add_argument("--full-collection", type=Path, required=True)
    package.add_argument("--scheduler-url", default=scheduler_client.SCHEDULER)
    package.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    passed = True
    if args.command == "plan":
        result = create_plan(
            aggregate_manifest_path=args.aggregate_manifest,
            bundle_manifest_path=args.bundle_manifest,
            standard_candidates_path=args.standard_candidates,
            standard_candidates_sha256=args.standard_candidates_sha256,
            candidate_physics_sha256=args.candidate_physics_sha256,
            source_result_path=args.source_result,
            task_payload_path=args.task_payload,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            output=args.output,
        )
    elif args.command == "submit-standard":
        result = submit_stage(
            plan_path=args.plan,
            stage="standard",
            output=args.output,
            priority=args.priority,
        )
    elif args.command == "collect":
        result = collect_stage(
            plan_path=args.plan,
            submission_path=args.submission,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
    elif args.command == "gate":
        result, passed = create_standard_gate(
            plan_path=args.plan,
            standard_collection_path=args.standard_collection,
            output=args.output,
        )
    elif args.command == "submit-full":
        result = submit_stage(
            plan_path=args.plan,
            stage="full",
            output=args.output,
            standard_gate_path=args.standard_gate,
            standard_collection_path=args.standard_collection,
            license_snapshot_path=args.license_snapshot,
            priority=args.priority,
        )
    else:
        result = package_results(
            plan_path=args.plan,
            standard_collection_path=args.standard_collection,
            standard_gate_path=args.standard_gate,
            full_collection_path=args.full_collection,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
    print(
        json.dumps(
            {"status": "ok" if passed else "gate_failed", "path": str(result)},
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
