#!/usr/bin/env python3
"""Prepare and validate the isolated rounded final-package authority.

This gate is deliberately separate from ``mft_goal_final_package_gate.py``.
It is bound to rounded candidate 5 and task 96340, and it has no package
publishing command.  ``prepare`` writes an immutable source/inventory plan;
``validate`` reauthenticates every semantic authority and every source byte.

A task96340 terminal-success Standard collection is the only symmetric
scientific authority.  A raw diagnostic AEDT may be carried as a file-only
fallback, but it always forces ``scientific_package_allowed=false``.  A
terminal rounded Full collection completes the cross-check.  A pre-solve Full
checkpoint may supply the deliverable model file, but it always leaves the
cross-check incomplete and cannot authorize a scientific package.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as rounded_prepare,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_full_collector as full_collector,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_full_prepare as full_prepare,
)
from tools import (  # noqa: E402
    mft_goal_postdeadline_standard_collector as standard_collector,
)


CAMPAIGN_ID = "mft-goal-20260726"
STANDARD_TASK_ID = 96_340
CANDIDATE_PHYSICS_SHA256 = (
    "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
)
PREPARE_SCHEMA = "mft-goal-rounded-final-package-prepare-v1"
VALIDATION_SCHEMA = "mft-goal-rounded-final-package-validation-v1"
TARGET_PACKAGE_SCHEMA = "mft-goal-rounded-final-delivery-package-v1"
TARGET_PACKAGE_SEAL_SCHEMA = (
    "mft-goal-rounded-final-delivery-package-seal-v1"
)
PREPARE_NAME = "rounded_final_package_prepare.json"
VALIDATION_NAME = "rounded_final_package_validation.json"
AGGREGATE_SCHEMA = "mft-goal-20260726-global-pareto-v1"
EXPECTED_SEED_COUNT = 512
EXPECTED_TERMINAL_ROW_COUNT = 163_840
EXPECTED_DEDUPLICATED_GEOMETRY_COUNT = 133_563
EXPECTED_OBJECTIVE_FRONT_COUNT = 22
EXPECTED_PRODUCTION_PARETO_COUNT = 0
EXPECTED_STANDARD_CANDIDATE_COUNT = 12
FIXED_BOUNDARY = copy.deepcopy(rounded_prepare.FIXED_BOUNDARY)
ROUNDING_POLICY = copy.deepcopy(rounded_prepare.ROUNDING_POLICY)

PARETO_ARTIFACTS = {
    "global_terminal_candidates": (
        "global_terminal_candidates.csv",
        "search/global_terminal_candidates.csv",
        "all_seed_global_nondominated_sort",
    ),
    "global_pareto_front": (
        "global_pareto_front.csv",
        "search/global_pareto_front.csv",
        "production_feasible_pareto_front",
    ),
    "global_objective_front": (
        "global_objective_front.csv",
        "search/global_objective_front.csv",
        "audit_objective_front",
    ),
    "standard_candidates": (
        "standard_candidates.csv",
        "search/standard_candidates.csv",
        "deterministic_standard_candidate_selection",
    ),
}


class RoundedPackageGateError(RuntimeError):
    """A rounded package authority or immutable byte invariant failed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RoundedPackageGateError("payload is already sealed")
    result["payload_sha256"] = _payload_sha256(result)
    return result


def _validate_seal(
    value: Any,
    schema: str,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RoundedPackageGateError(f"{label} is not a JSON object")
    result = dict(value)
    claimed = result.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(claimed, str)
        or _payload_sha256(result) != claimed
    ):
        raise RoundedPackageGateError(f"{label} seal drifted")
    return copy.deepcopy(dict(value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label, maximum_bytes=64 * 1024 * 1024)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoundedPackageGateError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RoundedPackageGateError(f"{label} is not a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> Path:
    candidate = Path(path)
    try:
        if candidate.is_symlink():
            raise RoundedPackageGateError(f"{label} cannot be a symlink")
        resolved = candidate.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise RoundedPackageGateError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise RoundedPackageGateError(f"{label} is not a non-empty regular file")
    if maximum_bytes is not None and metadata.st_size > maximum_bytes:
        raise RoundedPackageGateError(f"{label} exceeds its byte limit")
    return resolved


def _source_inventory_record(
    source: Path,
    *,
    package_path: str,
    role: str,
    scientific_authority: bool,
) -> dict[str, Any]:
    resolved = _regular_file(source, label=role)
    package = PurePosixPath(package_path)
    if (
        package.is_absolute()
        or ".." in package.parts
        or any(part in {"", "."} for part in package.parts)
    ):
        raise RoundedPackageGateError(f"unsafe package path for {role}")
    return {
        "source_path": str(resolved),
        "package_path": package.as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "role": role,
        "scientific_authority": scientific_authority,
    }


def _record_path(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise RoundedPackageGateError(f"{label} record is absent")
    relative = record.get("path")
    claimed_sha = record.get("sha256")
    claimed_size = record.get("size_bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(claimed_sha, str)
        or isinstance(claimed_size, bool)
        or not isinstance(claimed_size, int)
    ):
        raise RoundedPackageGateError(f"{label} record is malformed")
    posix = PurePosixPath(relative.replace("\\", "/"))
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        raise RoundedPackageGateError(f"{label} record path is unsafe")
    root_resolved = root.resolve(strict=True)
    candidate = _regular_file(
        root_resolved.joinpath(*posix.parts),
        label=label,
    )
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise RoundedPackageGateError(f"{label} escaped its collection") from exc
    if (
        candidate.stat().st_size != claimed_size
        or _sha256_file(candidate) != claimed_sha
    ):
        raise RoundedPackageGateError(f"{label} bytes drifted")
    return candidate


def _plan_record_path(
    plan_path: Path,
    plan: Mapping[str, Any],
    key: str,
    label: str,
) -> Path:
    return _record_path(plan_path.parent, plan.get(key), label)


def _actual_result_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
        width, length, height = (float(item) for item in dimensions)
        resonance = float(result["f_res_min_tx_rx_only_Hz"])
        active, temperatures, temperature_passed = (
            production._temperature_gate_evidence(result)  # noqa: SLF001
        )
    except Exception as exc:
        raise RoundedPackageGateError(
            "scientific result evidence is incomplete"
        ) from exc
    finite_values = (volume_l, width, length, height, resonance)
    if any(not math.isfinite(float(value)) for value in finite_values):
        raise RoundedPackageGateError("scientific result evidence is malformed")
    goal_passed = (
        width <= float(GOAL_SIZE_LIMITS_MM["W"])
        and length <= float(GOAL_SIZE_LIMITS_MM["L"])
        and height <= float(GOAL_SIZE_LIMITS_MM["H"])
        and resonance >= float(GOAL_STAGE_SPEC["resonance_min_Hz"])
        and temperature_passed
    )
    return {
        "actual_volume_L": float(volume_l),
        "actual_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_resonance_Hz": resonance,
        "active_temperature_targets": active,
        "actual_temperatures": temperatures,
        "goal_constraints_passed": goal_passed,
    }


def _fixed_boundary_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "fan_config": result.get("fan_config"),
            "fan_velocity_m_s": float(result["fan_velocity"]),
            "thermal_pad_conductivity_W_mK": float(result["k_ins"]),
            "core_plate_pad_t_mm": float(result["core_plate_pad_t"]),
            "wcp_pad_t_mm": float(result["wcp_pad_t"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RoundedPackageGateError(
            "result fixed thermal boundary is incomplete"
        ) from exc


def _rounding_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "round_corner": int(result["round_corner"]),
            "corner_radius_mm": float(result["corner_radius"]),
            "corner_segments_per_corner": int(result["corner_segments"]),
            "path_kind": ROUNDING_POLICY["path_kind"],
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RoundedPackageGateError(
            "result rounding identity is incomplete"
        ) from exc


def _standard_collection_envelope(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    collection_root = root.resolve(strict=True)
    receipt_path = collection_root / "collection_receipt.json"
    seal_path = collection_root / "collection_seal.json"
    try:
        receipt = standard_collector._validate_seal(  # noqa: SLF001
            standard_collector._read_json(  # noqa: SLF001
                receipt_path, "rounded Standard collection receipt"
            ),
            standard_collector.COLLECTION_SCHEMA,
            "rounded Standard collection receipt",
        )
        seal = standard_collector._validate_seal(  # noqa: SLF001
            standard_collector._read_json(  # noqa: SLF001
                seal_path, "rounded Standard collection seal"
            ),
            standard_collector.COLLECTION_SEAL_SCHEMA,
            "rounded Standard collection seal",
        )
    except Exception as exc:
        raise RoundedPackageGateError(
            "rounded Standard collection envelope is invalid"
        ) from exc
    if (
        _record_path(
            collection_root,
            seal.get("collection_receipt"),
            "rounded Standard collection receipt",
        )
        != receipt_path.resolve(strict=True)
        or seal.get("collection_receipt_payload_sha256")
        != receipt.get("payload_sha256")
    ):
        raise RoundedPackageGateError(
            "rounded Standard collection receipt binding drifted"
        )
    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise RoundedPackageGateError(
            "rounded Standard source-file inventory is absent"
        )
    paths = {
        "receipt": receipt_path.resolve(strict=True),
        "seal": seal_path.resolve(strict=True),
        "model": _record_path(
            collection_root,
            receipt.get("retained_symmetric_aedt"),
            "rounded symmetric AEDT",
        ),
        "result": _record_path(
            collection_root,
            receipt.get("result_json"),
            "rounded Standard result",
        ),
        "results_manifest": _record_path(
            collection_root,
            receipt.get("aedtresults_manifest"),
            "rounded Standard results manifest",
        ),
        "terminal_task": _record_path(
            collection_root,
            source_files.get("scheduler_terminal_task.json"),
            "rounded Standard terminal task",
        ),
        "source_submission": _record_path(
            collection_root,
            source_files.get("source_submission_receipt.json"),
            "rounded Standard source submission",
        ),
    }
    return receipt, seal, paths


def _authenticate_symmetric_scientific(
    *,
    standard_plan: Path,
    standard_final: Path,
    standard_collection: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        authority = full_prepare.authenticate_standard_success(
            standard_plan_path=standard_plan,
            standard_final_path=standard_final,
            standard_collection_root=standard_collection,
        )
    except Exception as exc:
        raise RoundedPackageGateError(
            "task96340 rounded symmetric scientific authority failed"
        ) from exc
    receipt, _seal_value, paths = _standard_collection_envelope(
        standard_collection
    )
    result = _read_json(paths["result"], "rounded Standard result")
    if (
        authority.get("source_task_id") != STANDARD_TASK_ID
        or authority.get("source_candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or authority.get("measured_hard_constraints_passed") is not True
        or authority.get("fixed_thermal_boundary") != FIXED_BOUNDARY
        or authority.get("rounding_policy") != ROUNDING_POLICY
        or _actual_result_evidence(result)["goal_constraints_passed"] is not True
    ):
        raise RoundedPackageGateError(
            "task96340 symmetric scientific pass is absent"
        )
    inventory = [
        _source_inventory_record(
            paths["model"],
            package_path="models/symmetric.aedt",
            role="task96340_rounded_symmetric_model",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["result"],
            package_path="results/symmetric/result.json",
            role="task96340_actual_symmetric_result",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["results_manifest"],
            package_path=(
                "results/symmetric/symmetric.aedtresults.manifest.json"
            ),
            role="task96340_native_results_tree_manifest",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["receipt"],
            package_path="manifests/symmetric_collection_receipt.json",
            role="task96340_collection_receipt",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["seal"],
            package_path="manifests/symmetric_collection_seal.json",
            role="task96340_collection_seal",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["terminal_task"],
            package_path="provenance/symmetric_scheduler_terminal_task.json",
            role="task96340_scheduler_terminal_provenance",
            scientific_authority=True,
        ),
        _source_inventory_record(
            paths["source_submission"],
            package_path="provenance/symmetric_submission_receipt.json",
            role="task96340_submission_provenance",
            scientific_authority=True,
        ),
        _source_inventory_record(
            standard_final,
            package_path="provenance/symmetric_submission_final.json",
            role="task96340_submission_final",
            scientific_authority=True,
        ),
    ]
    return (
        {
            "mode": "task96340_terminal_scientific_collection",
            "source_task_id": STANDARD_TASK_ID,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "solver_revision": authority["source_solver_revision"],
            "library_revision": authority["source_library_revision"],
            "scientific_result_available": True,
            "scientific_pass": True,
            "model_delivery_allowed": True,
            "diagnostic_file_fallback": False,
            "actual_evidence": {
                "actual_volume_L": authority["actual_volume_L"],
                "actual_dimensions_mm": authority["actual_dimensions_mm"],
                "actual_resonance_Hz": authority["actual_resonance_Hz"],
                "hard_constraint_evidence": authority[
                    "hard_constraint_evidence"
                ],
            },
            "collection_payload_sha256": receipt["payload_sha256"],
            "fixed_boundary_verified": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy_verified": copy.deepcopy(ROUNDING_POLICY),
        },
        inventory,
    )


def _authenticate_symmetric_fallback(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = _regular_file(path, label="diagnostic symmetric AEDT fallback")
    if model.suffix.lower() != ".aedt":
        raise RoundedPackageGateError(
            "diagnostic symmetric fallback must be an AEDT file"
        )
    inventory = [
        _source_inventory_record(
            model,
            package_path="models/symmetric.aedt",
            role="diagnostic_symmetric_model_file_fallback",
            scientific_authority=False,
        )
    ]
    return (
        {
            "mode": "diagnostic_model_file_fallback",
            "source_task_id": None,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "solver_revision": None,
            "library_revision": None,
            "scientific_result_available": False,
            "scientific_pass": False,
            "model_delivery_allowed": True,
            "diagnostic_file_fallback": True,
            "actual_evidence": None,
            "collection_payload_sha256": None,
            "fixed_boundary_verified": None,
            "rounding_policy_verified": None,
        },
        inventory,
    )


def _full_collection_envelope(
    root: Path,
    *,
    checkpoint: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    collection_root = root.resolve(strict=True)
    receipt_path = collection_root / "collection_receipt.json"
    seal_path = collection_root / "collection_seal.json"
    receipt_schema = (
        full_collector.CHECKPOINT_COLLECTION_SCHEMA
        if checkpoint
        else full_collector.COLLECTION_SCHEMA
    )
    seal_schema = (
        full_collector.CHECKPOINT_COLLECTION_SEAL_SCHEMA
        if checkpoint
        else full_collector.COLLECTION_SEAL_SCHEMA
    )
    try:
        receipt = full_prepare.validate_seal(
            full_prepare.read_json(receipt_path), receipt_schema
        )
        seal = full_prepare.validate_seal(
            full_prepare.read_json(seal_path), seal_schema
        )
    except Exception as exc:
        raise RoundedPackageGateError(
            "rounded Full collection envelope is invalid"
        ) from exc
    receipt_record = seal.get("collection_receipt")
    if (
        _record_path(
            collection_root,
            receipt_record,
            "rounded Full collection receipt",
        )
        != receipt_path.resolve(strict=True)
    ):
        raise RoundedPackageGateError(
            "rounded Full collection receipt binding drifted"
        )
    paths = {
        "receipt": receipt_path.resolve(strict=True),
        "seal": seal_path.resolve(strict=True),
    }
    if checkpoint:
        paths["model"] = _record_path(
            collection_root,
            receipt.get("retained_checkpoint_aedt"),
            "rounded Full diagnostic checkpoint AEDT",
        )
        for name in (
            "remote_checkpoint_receipt.json",
            "remote_checkpoint_marker.json",
        ):
            candidate = collection_root / name
            if candidate.exists():
                paths[name] = _regular_file(
                    candidate, label=f"rounded Full {name}"
                )
    else:
        files = receipt.get("files")
        if not isinstance(files, Mapping):
            raise RoundedPackageGateError(
                "rounded Full terminal file inventory is absent"
            )
        expected = {
            "model": "full_model.aedt",
            "result": "result.json",
            "results_manifest": "full_model.aedtresults.manifest.json",
            "terminal_task": "scheduler_terminal_task.json",
            "source_plan": "source_plan.json",
            "source_submission": "source_submission_receipt.json",
        }
        for key, name in expected.items():
            paths[key] = _record_path(
                collection_root,
                files.get(name),
                f"rounded Full {name}",
            )
    return receipt, seal, paths


def _authenticate_full_terminal(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt, seal, paths = _full_collection_envelope(
        root, checkpoint=False
    )
    result = _read_json(paths["result"], "rounded Full result")
    terminal = _read_json(
        paths["terminal_task"], "rounded Full terminal task"
    )
    evidence = _actual_result_evidence(result)
    claimed = receipt.get("actual_evidence")
    if not isinstance(claimed, Mapping):
        raise RoundedPackageGateError(
            "rounded Full actual evidence is absent"
        )
    comparable_keys = (
        "actual_volume_L",
        "actual_dimensions_mm",
        "actual_resonance_Hz",
        "active_temperature_targets",
        "actual_temperatures",
        "goal_constraints_passed",
    )
    if (
        receipt.get("source_standard_task_id") != STANDARD_TASK_ID
        or receipt.get("source_candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or receipt.get("rounding_policy") != ROUNDING_POLICY
        or receipt.get("fixed_boundary") != FIXED_BOUNDARY
        or receipt.get("full_model") != 1
        or receipt.get("thermal_symmetry") != "full"
        or receipt.get("synthetic_aedt_or_results_used") is not False
        or _fixed_boundary_from_result(result) != FIXED_BOUNDARY
        or _rounding_from_result(result) != ROUNDING_POLICY
        or any(claimed.get(key) != evidence[key] for key in comparable_keys)
        or terminal.get("task_id") != receipt.get("task_id")
        or terminal.get("status") != "completed"
        or terminal.get("state") != "succeeded"
        or terminal.get("exit_code") != 0
    ):
        raise RoundedPackageGateError(
            "rounded Full terminal scientific lineage drifted"
        )
    full_pass = evidence["goal_constraints_passed"] is True
    inventory_specs = (
        ("model", "models/full_model.aedt", "rounded_full_model"),
        ("result", "results/full/result.json", "actual_full_result"),
        (
            "results_manifest",
            "results/full/full_model.aedtresults.manifest.json",
            "full_native_results_tree_manifest",
        ),
        (
            "receipt",
            "manifests/full_collection_receipt.json",
            "full_collection_receipt",
        ),
        (
            "seal",
            "manifests/full_collection_seal.json",
            "full_collection_seal",
        ),
        (
            "terminal_task",
            "provenance/full_scheduler_terminal_task.json",
            "full_scheduler_terminal_provenance",
        ),
        (
            "source_plan",
            "provenance/full_source_plan.json",
            "full_source_plan",
        ),
        (
            "source_submission",
            "provenance/full_submission_receipt.json",
            "full_submission_provenance",
        ),
    )
    inventory = [
        _source_inventory_record(
            paths[key],
            package_path=package_path,
            role=role,
            scientific_authority=True,
        )
        for key, package_path, role in inventory_specs
    ]
    return (
        {
            "mode": "terminal_full_collection",
            "source_task_id": receipt["task_id"],
            "source_standard_task_id": STANDARD_TASK_ID,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "solver_revision": receipt["solver_revision"],
            "library_revision": receipt["library_revision"],
            "scientific_result_available": True,
            "scientific_pass": full_pass,
            "crosscheck_complete": True,
            "crosscheck_passed": full_pass,
            "model_delivery_allowed": True,
            "diagnostic_only": False,
            "actual_evidence": evidence,
            "collection_payload_sha256": receipt["payload_sha256"],
            "collection_seal_payload_sha256": seal["payload_sha256"],
            "fixed_boundary_verified": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy_verified": copy.deepcopy(ROUNDING_POLICY),
        },
        inventory,
    )


def _authenticate_full_checkpoint(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt, seal, paths = _full_collection_envelope(root, checkpoint=True)
    if (
        receipt.get("source_standard_task_id") != STANDARD_TASK_ID
        or receipt.get("source_candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or receipt.get("rounded_identity") != ROUNDING_POLICY
        or receipt.get("fixed_boundary") != FIXED_BOUNDARY
        or receipt.get("scientific_result_available") is not False
        or receipt.get("scientific_pass") is not False
        or receipt.get("thermal_pass") is not False
        or receipt.get("production_package_eligible") is not False
        or receipt.get("diagnostic_only") is not True
        or seal.get("scientific_pass") is not False
        or seal.get("production_promotion_eligible") is not False
    ):
        raise RoundedPackageGateError(
            "rounded Full checkpoint classification drifted"
        )
    inventory = [
        _source_inventory_record(
            paths["model"],
            package_path="models/full_model.aedt",
            role="rounded_full_geometry_setup_checkpoint_model",
            scientific_authority=False,
        ),
        _source_inventory_record(
            paths["receipt"],
            package_path="manifests/full_checkpoint_collection_receipt.json",
            role="full_checkpoint_collection_receipt",
            scientific_authority=False,
        ),
        _source_inventory_record(
            paths["seal"],
            package_path="manifests/full_checkpoint_collection_seal.json",
            role="full_checkpoint_collection_seal",
            scientific_authority=False,
        ),
    ]
    for name, package_path in (
        (
            "remote_checkpoint_receipt.json",
            "provenance/full_remote_checkpoint_receipt.json",
        ),
        (
            "remote_checkpoint_marker.json",
            "provenance/full_remote_checkpoint_marker.json",
        ),
    ):
        if name in paths:
            inventory.append(
                _source_inventory_record(
                    paths[name],
                    package_path=package_path,
                    role=name.removesuffix(".json"),
                    scientific_authority=False,
                )
            )
    return (
        {
            "mode": "diagnostic_full_geometry_setup_checkpoint",
            "source_task_id": receipt["task_id"],
            "source_standard_task_id": STANDARD_TASK_ID,
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "solver_revision": receipt["solver_revision"],
            "library_revision": receipt["library_revision"],
            "scientific_result_available": False,
            "scientific_pass": False,
            "crosscheck_complete": False,
            "crosscheck_passed": False,
            "model_delivery_allowed": True,
            "diagnostic_only": True,
            "actual_evidence": None,
            "collection_payload_sha256": receipt["payload_sha256"],
            "collection_seal_payload_sha256": seal["payload_sha256"],
            "fixed_boundary_verified": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy_verified": copy.deepcopy(ROUNDING_POLICY),
        },
        inventory,
    )


def _validate_candidate_row(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            matches = [
                row
                for row in csv.DictReader(handle)
                if row.get("candidate_physics_sha")
                == CANDIDATE_PHYSICS_SHA256
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RoundedPackageGateError(
            "standard candidate CSV is unreadable"
        ) from exc
    if len(matches) != 1:
        raise RoundedPackageGateError(
            "candidate 5 is not unique in standard_candidates.csv"
        )
    row = matches[0]
    try:
        identity = {
            "candidate_physics_sha256": row["candidate_physics_sha"],
            "selection_order": int(row["standard_selection_order"]),
            "source_seed": int(row["source_seed"]),
            "source_task_id": int(row["source_task_id"]),
            "selection_roles": row["standard_selection_roles"],
            "selection_basis": row["standard_selection_basis"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RoundedPackageGateError(
            "candidate 5 selection identity is incomplete"
        ) from exc
    if (
        identity["selection_order"] != 5
        or identity["source_seed"] != 2_607_262_233
        or identity["source_task_id"] != 95_913
    ):
        raise RoundedPackageGateError(
            "candidate 5 search provenance drifted"
        )
    return identity


def _authenticate_pareto(
    aggregate_root: Path,
    pareto_html: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = aggregate_root.resolve(strict=True)
    manifest_path = root / "aggregate_manifest.json"
    manifest = _validate_seal(
        _read_json(manifest_path, "aggregate manifest"),
        AGGREGATE_SCHEMA,
        label="aggregate manifest",
    )
    seeds = manifest.get("seeds")
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("seed_count") != EXPECTED_SEED_COUNT
        or manifest.get("minimum_seed_count") != EXPECTED_SEED_COUNT
        or not isinstance(seeds, list)
        or len(seeds) != EXPECTED_SEED_COUNT
        or len(set(seeds)) != EXPECTED_SEED_COUNT
        or manifest.get("input_terminal_row_count")
        != EXPECTED_TERMINAL_ROW_COUNT
        or manifest.get("deduplicated_physical_geometry_count")
        != EXPECTED_DEDUPLICATED_GEOMETRY_COUNT
        or manifest.get("global_objective_front_count")
        != EXPECTED_OBJECTIVE_FRONT_COUNT
        or manifest.get("global_pareto_count")
        != EXPECTED_PRODUCTION_PARETO_COUNT
        or manifest.get("standard_candidate_count")
        != EXPECTED_STANDARD_CANDIDATE_COUNT
        or manifest.get("seed_local_pareto_merge_used") is not False
        or manifest.get("sorting_authority")
        != (
            "all_authenticated_terminal_rows_then_physical_dedupe_then_"
            "decoder_and_physical_G_and_surrogate_physicality_feasible_then_"
            "exact_2d_nlogn_non_dominated_sort"
        )
        or not isinstance(artifacts, Mapping)
    ):
        raise RoundedPackageGateError(
            "aggregate is not the exact 512-seed global NDS authority"
        )
    inventory = [
        _source_inventory_record(
            manifest_path,
            package_path="search/aggregate_manifest.json",
            role="global_nds_aggregate_manifest",
            scientific_authority=True,
        )
    ]
    artifact_paths: dict[str, Path] = {}
    for key, (filename, package_path, role) in PARETO_ARTIFACTS.items():
        record = artifacts.get(key)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != filename
            or not isinstance(record.get("sha256"), str)
        ):
            raise RoundedPackageGateError(
                f"aggregate {key} artifact record drifted"
            )
        source = _regular_file(root / filename, label=f"aggregate {key}")
        if _sha256_file(source) != record["sha256"]:
            raise RoundedPackageGateError(
                f"aggregate {key} artifact bytes drifted"
            )
        artifact_paths[key] = source
        inventory.append(
            _source_inventory_record(
                source,
                package_path=package_path,
                role=role,
                scientific_authority=True,
            )
        )
    html = _regular_file(pareto_html, label="global Pareto HTML")
    if html.suffix.lower() != ".html":
        raise RoundedPackageGateError("global Pareto visualization must be HTML")
    inventory.append(
        _source_inventory_record(
            html,
            package_path="search/global-pareto-audit.html",
            role="global_pareto_interactive_visualization",
            scientific_authority=True,
        )
    )
    candidate = _validate_candidate_row(artifact_paths["standard_candidates"])
    return (
        {
            "aggregate_schema": AGGREGATE_SCHEMA,
            "aggregate_payload_sha256": manifest["payload_sha256"],
            "seed_count": EXPECTED_SEED_COUNT,
            "input_terminal_row_count": EXPECTED_TERMINAL_ROW_COUNT,
            "deduplicated_physical_geometry_count": (
                EXPECTED_DEDUPLICATED_GEOMETRY_COUNT
            ),
            "production_pareto_count": EXPECTED_PRODUCTION_PARETO_COUNT,
            "audit_objective_front_count": EXPECTED_OBJECTIVE_FRONT_COUNT,
            "all_seed_global_nondominated_sort": True,
            "seed_local_pareto_merge_used": False,
            "sorting_authority": manifest["sorting_authority"],
            "candidate_5_search_identity": candidate,
        },
        inventory,
    )


def _authenticate_candidate_inputs(
    standard_plan_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        plan = rounded_prepare.load_plan(standard_plan_path)
    except Exception as exc:
        raise RoundedPackageGateError(
            "rounded candidate-5 Standard plan is invalid"
        ) from exc
    if (
        plan.get("source_candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        or plan.get("rounding_policy") != ROUNDING_POLICY
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
    ):
        raise RoundedPackageGateError(
            "rounded candidate-5 plan identity drifted"
        )
    specs = (
        (
            "rounded_fea_params",
            "design/candidate5_rounded_params.json",
            "candidate_5_rounded_fea_parameters",
        ),
        (
            "source_selected_candidate",
            "design/candidate5_search_selection.json",
            "candidate_5_search_selection",
        ),
        (
            "drawing_dimensions",
            "design/rounded_drawing_dimensions.json",
            "rounded_model_drawing_dimensions",
        ),
        (
            "rounded_execution_profile",
            "design/rounded_execution_profile.json",
            "rounded_execution_profile",
        ),
    )
    inventory = [
        _source_inventory_record(
            standard_plan_path,
            package_path="provenance/rounded_standard_prepare_plan.json",
            role="rounded_standard_prepare_plan",
            scientific_authority=True,
        )
    ]
    records: dict[str, dict[str, Any]] = {}
    for key, package_path, role in specs:
        source = _plan_record_path(
            standard_plan_path, plan, key, role
        )
        inventory.append(
            _source_inventory_record(
                source,
                package_path=package_path,
                role=role,
                scientific_authority=True,
            )
        )
        records[key] = {
            "payload_sha256": (
                _read_json(source, role).get("payload_sha256")
                if key == "drawing_dimensions"
                else _payload_sha256(_read_json(source, role))
            ),
            "source_sha256": _sha256_file(source),
        }
    params_path = _plan_record_path(
        standard_plan_path,
        plan,
        "rounded_fea_params",
        "rounded FEA parameters",
    )
    params = _read_json(params_path, "rounded FEA parameters")
    expected_params = {
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
    }
    if any(params.get(key) != value for key, value in expected_params.items()):
        raise RoundedPackageGateError(
            "candidate-5 rounded parameter contract drifted"
        )
    return (
        {
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "selection_order": 5,
            "rounded_parameter_payload_sha256": plan[
                "rounded_fea_params_sha256"
            ],
            "effective_rounded_params_sha256": plan[
                "effective_rounded_params_sha256"
            ],
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "records": records,
        },
        inventory,
    )


def _authenticate_drawings(
    pptx: Path | None,
    pdf: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    supplied = {"pptx": pptx is not None, "pdf": pdf is not None}
    for path, suffix, magic, package_path, role in (
        (
            pptx,
            ".pptx",
            b"PK",
            "drawings/final_design.pptx",
            "final_rounded_design_drawing_pptx",
        ),
        (
            pdf,
            ".pdf",
            b"%PDF-",
            "drawings/final_design.pdf",
            "final_rounded_design_drawing_pdf",
        ),
    ):
        if path is None:
            continue
        source = _regular_file(path, label=role)
        if source.suffix.lower() != suffix:
            raise RoundedPackageGateError(f"{role} extension drifted")
        with source.open("rb") as handle:
            if handle.read(len(magic)) != magic:
                raise RoundedPackageGateError(f"{role} file signature drifted")
        inventory.append(
            _source_inventory_record(
                source,
                package_path=package_path,
                role=role,
                scientific_authority=False,
            )
        )
    complete = all(supplied.values())
    return (
        {
            "pptx_supplied": supplied["pptx"],
            "pdf_supplied": supplied["pdf"],
            "complete": complete,
            "pending": not complete,
            "rounded_model_required": True,
        },
        inventory,
    )


def _validate_inventory_uniqueness(
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    package_paths = [str(item.get("package_path")) for item in inventory]
    if len(package_paths) != len(set(package_paths)):
        duplicates = sorted(
            path for path in set(package_paths) if package_paths.count(path) > 1
        )
        raise RoundedPackageGateError(
            f"package inventory paths collide: {duplicates}"
        )
    if "models/symmetric.aedt" not in package_paths:
        raise RoundedPackageGateError("symmetric model inventory is absent")
    if "models/full_model.aedt" not in package_paths:
        raise RoundedPackageGateError("Full model inventory is absent")


def _readiness(
    *,
    symmetric: Mapping[str, Any],
    full: Mapping[str, Any],
    drawings: Mapping[str, Any],
) -> dict[str, Any]:
    symmetric_pass = symmetric.get("scientific_pass") is True
    full_pass = full.get("scientific_pass") is True
    crosscheck_complete = full.get("crosscheck_complete") is True
    model_delivery_allowed = (
        symmetric.get("model_delivery_allowed") is True
        and full.get("model_delivery_allowed") is True
    )
    scientific_allowed = (
        symmetric_pass
        and full_pass
        and crosscheck_complete
        and full.get("crosscheck_passed") is True
    )
    drawings_complete = drawings.get("complete") is True
    publish_eligible = scientific_allowed and drawings_complete
    pending_reasons = []
    if not symmetric_pass:
        pending_reasons.append(
            "task96340 symmetric scientific PASS authority is absent"
        )
    if not crosscheck_complete:
        pending_reasons.append(
            "Full cross-check is incomplete; checkpoint is model-only"
        )
    elif not full_pass:
        pending_reasons.append("terminal Full cross-check did not pass")
    if not drawings_complete:
        pending_reasons.append("final rounded drawing PPTX/PDF are pending")
    return {
        "symmetric_scientific_pass": symmetric_pass,
        "full_scientific_pass": full_pass,
        "full_crosscheck_complete": crosscheck_complete,
        "full_crosscheck_passed": full.get("crosscheck_passed") is True,
        "model_file_delivery_allowed": model_delivery_allowed,
        "scientific_package_allowed": scientific_allowed,
        "drawings_complete": drawings_complete,
        "final_package_publish_eligible": publish_eligible,
        "pending_reasons": pending_reasons,
    }


def _build_payload(
    *,
    standard_plan: Path,
    standard_final: Path | None,
    standard_collection: Path | None,
    symmetric_fallback_aedt: Path | None,
    full_terminal_collection: Path | None,
    full_checkpoint_collection: Path | None,
    aggregate_root: Path,
    pareto_html: Path,
    drawing_pptx: Path | None,
    drawing_pdf: Path | None,
) -> dict[str, Any]:
    scientific_symmetric = (
        standard_final is not None and standard_collection is not None
    )
    if scientific_symmetric == (symmetric_fallback_aedt is not None):
        raise RoundedPackageGateError(
            "choose exactly one symmetric source: task96340 collection "
            "or diagnostic AEDT fallback"
        )
    if (standard_final is None) != (standard_collection is None):
        raise RoundedPackageGateError(
            "scientific symmetric source needs final and collection together"
        )
    if (full_terminal_collection is None) == (
        full_checkpoint_collection is None
    ):
        raise RoundedPackageGateError(
            "choose exactly one Full source: terminal collection or checkpoint"
        )

    candidate, candidate_inventory = _authenticate_candidate_inputs(
        standard_plan
    )
    if scientific_symmetric:
        symmetric, symmetric_inventory = (
            _authenticate_symmetric_scientific(
                standard_plan=standard_plan,
                standard_final=standard_final,
                standard_collection=standard_collection,
            )
        )
    else:
        symmetric, symmetric_inventory = _authenticate_symmetric_fallback(
            symmetric_fallback_aedt
        )
    if full_terminal_collection is not None:
        full, full_inventory = _authenticate_full_terminal(
            full_terminal_collection
        )
    else:
        full, full_inventory = _authenticate_full_checkpoint(
            full_checkpoint_collection
        )
    pareto, pareto_inventory = _authenticate_pareto(
        aggregate_root, pareto_html
    )
    drawings, drawing_inventory = _authenticate_drawings(
        drawing_pptx, drawing_pdf
    )
    if any(
        authority.get("candidate_physics_sha256")
        != CANDIDATE_PHYSICS_SHA256
        for authority in (candidate, symmetric, full)
    ):
        raise RoundedPackageGateError(
            "rounded package candidate identity is inconsistent"
        )
    inventory = [
        *candidate_inventory,
        *symmetric_inventory,
        *full_inventory,
        *pareto_inventory,
        *drawing_inventory,
    ]
    _validate_inventory_uniqueness(inventory)
    readiness = _readiness(
        symmetric=symmetric, full=full, drawings=drawings
    )
    inputs = {
        "standard_plan": str(standard_plan.resolve(strict=True)),
        "standard_final": (
            str(standard_final.resolve(strict=True))
            if standard_final is not None
            else None
        ),
        "standard_collection": (
            str(standard_collection.resolve(strict=True))
            if standard_collection is not None
            else None
        ),
        "symmetric_fallback_aedt": (
            str(symmetric_fallback_aedt.resolve(strict=True))
            if symmetric_fallback_aedt is not None
            else None
        ),
        "full_terminal_collection": (
            str(full_terminal_collection.resolve(strict=True))
            if full_terminal_collection is not None
            else None
        ),
        "full_checkpoint_collection": (
            str(full_checkpoint_collection.resolve(strict=True))
            if full_checkpoint_collection is not None
            else None
        ),
        "aggregate_root": str(aggregate_root.resolve(strict=True)),
        "pareto_html": str(pareto_html.resolve(strict=True)),
        "drawing_pptx": (
            str(drawing_pptx.resolve(strict=True))
            if drawing_pptx is not None
            else None
        ),
        "drawing_pdf": (
            str(drawing_pdf.resolve(strict=True))
            if drawing_pdf is not None
            else None
        ),
    }
    return {
        "schema_version": PREPARE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _now(),
        "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
        "package_target": {
            "package_schema": TARGET_PACKAGE_SCHEMA,
            "package_seal_schema": TARGET_PACKAGE_SEAL_SCHEMA,
            "package_manifest_name": "package_manifest.json",
            "package_seal_name": "package_seal.json",
            "atomic_publish_contract": {
                "same_filesystem_staging_required": True,
                "source_bytes_reauthenticated_before_copy": True,
                "manifest_inventory_sha256_required": True,
                "seal_binds_complete_file_inventory": True,
                "single_os_replace_after_complete_validation": True,
                "partial_destination_forbidden": True,
            },
        },
        "input_paths": inputs,
        "candidate_authority": candidate,
        "symmetric_authority": symmetric,
        "full_authority": full,
        "pareto_authority": pareto,
        "drawing_authority": drawings,
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
        "source_inventory": inventory,
        "source_inventory_sha256": _payload_sha256(inventory),
        "readiness": readiness,
        "prepare_only": True,
        "package_directory_created": False,
        "payload_files_copied": False,
        "package_published": False,
        "scheduler_get_calls": 0,
        "scheduler_post_calls": 0,
        "scheduler_mutation_performed": False,
        "legacy_task96324_96326_gate_used": False,
    }


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RoundedPackageGateError(
            f"immutable output already exists: {target}"
        )
    raw = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare(
    *,
    output: Path,
    standard_plan: Path,
    standard_final: Path | None,
    standard_collection: Path | None,
    symmetric_fallback_aedt: Path | None,
    full_terminal_collection: Path | None,
    full_checkpoint_collection: Path | None,
    aggregate_root: Path,
    pareto_html: Path,
    drawing_pptx: Path | None = None,
    drawing_pdf: Path | None = None,
) -> Path:
    payload = _build_payload(
        standard_plan=standard_plan,
        standard_final=standard_final,
        standard_collection=standard_collection,
        symmetric_fallback_aedt=symmetric_fallback_aedt,
        full_terminal_collection=full_terminal_collection,
        full_checkpoint_collection=full_checkpoint_collection,
        aggregate_root=aggregate_root,
        pareto_html=pareto_html,
        drawing_pptx=drawing_pptx,
        drawing_pdf=drawing_pdf,
    )
    return _atomic_write_new(output, _sealed(payload))


def _path_or_none(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RoundedPackageGateError(f"{label} input path drifted")
    return Path(value)


def validate(
    *,
    plan_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    plan_path = _regular_file(
        plan_path, label="rounded package prepare plan"
    )
    plan = _validate_seal(
        _read_json(plan_path, "rounded package prepare plan"),
        PREPARE_SCHEMA,
        label="rounded package prepare plan",
    )
    inputs = plan.get("input_paths")
    if not isinstance(inputs, Mapping):
        raise RoundedPackageGateError("prepare-plan input paths are absent")
    rebuilt = _build_payload(
        standard_plan=Path(str(inputs["standard_plan"])),
        standard_final=_path_or_none(
            inputs.get("standard_final"), "standard_final"
        ),
        standard_collection=_path_or_none(
            inputs.get("standard_collection"), "standard_collection"
        ),
        symmetric_fallback_aedt=_path_or_none(
            inputs.get("symmetric_fallback_aedt"),
            "symmetric_fallback_aedt",
        ),
        full_terminal_collection=_path_or_none(
            inputs.get("full_terminal_collection"),
            "full_terminal_collection",
        ),
        full_checkpoint_collection=_path_or_none(
            inputs.get("full_checkpoint_collection"),
            "full_checkpoint_collection",
        ),
        aggregate_root=Path(str(inputs["aggregate_root"])),
        pareto_html=Path(str(inputs["pareto_html"])),
        drawing_pptx=_path_or_none(
            inputs.get("drawing_pptx"), "drawing_pptx"
        ),
        drawing_pdf=_path_or_none(
            inputs.get("drawing_pdf"), "drawing_pdf"
        ),
    )
    stable_fields = set(rebuilt) - {"created_at_utc"}
    drift = {
        key: {"planned": plan.get(key), "reauthenticated": rebuilt.get(key)}
        for key in stable_fields
        if plan.get(key) != rebuilt.get(key)
    }
    if drift:
        raise RoundedPackageGateError(
            f"rounded package prepare semantics drifted: {sorted(drift)}"
        )
    report = _sealed(
        {
            "schema_version": VALIDATION_SCHEMA,
            "validated_at_utc": _now(),
            "prepare_plan": _source_inventory_record(
                plan_path,
                package_path="validation/rounded_final_package_prepare.json",
                role="rounded_final_package_prepare_plan",
                scientific_authority=False,
            ),
            "prepare_plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_PHYSICS_SHA256,
            "semantic_authorities_reauthenticated": True,
            "source_inventory_rehashed": True,
            "source_inventory_sha256": plan["source_inventory_sha256"],
            "readiness": copy.deepcopy(plan["readiness"]),
            "prepare_only": True,
            "package_directory_created": False,
            "payload_files_copied": False,
            "package_published": False,
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
            "legacy_task96324_96326_gate_used": False,
        }
    )
    if report_path is not None:
        _atomic_write_new(report_path, report)
    return report


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--standard-plan", type=Path, required=True)
    parser.add_argument("--standard-final", type=Path)
    parser.add_argument("--standard-collection", type=Path)
    parser.add_argument("--symmetric-fallback-aedt", type=Path)
    parser.add_argument("--full-terminal-collection", type=Path)
    parser.add_argument("--full-checkpoint-collection", type=Path)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--pareto-html", type=Path, required=True)
    parser.add_argument("--drawing-pptx", type=Path)
    parser.add_argument("--drawing-pdf", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="seal source authority and intended package inventory"
    )
    _add_common_inputs(prepare_parser)
    prepare_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser(
        "validate", help="reauthenticate a sealed prepare plan; never publish"
    )
    validate_parser.add_argument("--plan", type=Path, required=True)
    validate_parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare(
            output=args.output,
            standard_plan=args.standard_plan,
            standard_final=args.standard_final,
            standard_collection=args.standard_collection,
            symmetric_fallback_aedt=args.symmetric_fallback_aedt,
            full_terminal_collection=args.full_terminal_collection,
            full_checkpoint_collection=args.full_checkpoint_collection,
            aggregate_root=args.aggregate_root,
            pareto_html=args.pareto_html,
            drawing_pptx=args.drawing_pptx,
            drawing_pdf=args.drawing_pdf,
        )
        plan = _validate_seal(
            _read_json(output, "rounded package prepare plan"),
            PREPARE_SCHEMA,
            label="rounded package prepare plan",
        )
        event = {
            "event": "rounded_final_package_prepared",
            "plan": str(output),
            "plan_payload_sha256": plan["payload_sha256"],
            "readiness": plan["readiness"],
            "package_published": False,
            "scheduler_post_calls": 0,
        }
    else:
        report = validate(
            plan_path=args.plan,
            report_path=args.report,
        )
        event = {
            "event": "rounded_final_package_prepare_validated",
            "plan": str(args.plan.resolve()),
            "validation_payload_sha256": report["payload_sha256"],
            "readiness": report["readiness"],
            "package_published": False,
            "scheduler_post_calls": 0,
        }
        if args.report is not None:
            event["report"] = str(args.report.resolve())
    print(json.dumps(event, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RoundedPackageGateError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_final_package_gate_error",
                    "error": str(exc),
                    "package_published": False,
                    "scheduler_post_calls": 0,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
