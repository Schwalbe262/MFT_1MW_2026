"""Prepare bounded same-design corrections after rounded task96340.

This module has no Scheduler client, POST, submit, retry, cancel, or Full
continuation capability.  It accepts only the authenticated terminal-success
collection for rounded candidate-5 task96340.  A plan is emitted only when
the measured result narrowly misses a requested hard constraint.

Candidate geometry is replayed through the exact goal fixed-point repair and
``decode_unit_sample_with_cw1`` path used by the NSGA-II campaign.  The older
``mft_goal_local_trust_acquisition`` raw decoder path is measured, recorded,
and explicitly rejected because it does not reproduce candidate #5.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from module import input_parameter_260706 as input_parameter  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    CORE_TEMPERATURE_TARGETS,
    GOAL_RESONANCE_MIN_HZ,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    TEMPERATURE_TARGET_LIMITS_C,
    WINDING_TEMPERATURE_TARGETS,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_collector as rounded_collector,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as rounded,
)
from tools import (  # noqa: E402
    mft_goal_postdeadline_standard_collector as standard_collector,
)
from tools import tier1_corrected_generation_preflight as goal_repair  # noqa: E402


ContractError = rounded.PostdeadlineContractError
SOURCE_TASK_ID = 96_340
SOURCE_CANDIDATE_SHA256 = rounded.SOURCE_CANDIDATE_SHA256
EXPECTED_SOURCE_SOLVER_REVISION = (
    "623a5345ae1b85b974cb248bcb0c8dfaadf1a867"
)

OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rounded_task96340_bounded_correction_prepare_v1"
)
PLAN_NAME = "bounded_correction_prepare_plan.json"
RECEIPT_NAME = "prepare_receipt.json"
SOURCE_NAME = "authenticated_task96340_source.json"
GATE_NAME = "narrow_miss_gate.json"
LEGACY_REJECTION_NAME = "legacy_local_trust_path_rejection.json"

PLAN_SCHEMA = "mft-goal-rounded-bounded-correction-prepare-plan-v1"
RECEIPT_SCHEMA = "mft-goal-rounded-bounded-correction-receipt-v1"
SOURCE_SCHEMA = "mft-goal-rounded-task96340-correction-source-v1"
GATE_SCHEMA = "mft-goal-rounded-task96340-narrow-miss-gate-v1"
LEGACY_REJECTION_SCHEMA = (
    "mft-goal-rounded-task96340-legacy-path-rejection-v1"
)
CANDIDATE_SCHEMA = "mft-goal-rounded-bounded-correction-candidate-v1"

ROUNDING_POLICY = {
    "round_corner": 1,
    "corner_radius_mm": 10.0,
    "corner_segments_per_corner": 4,
}
FIXED_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity_m_s": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
}
FROZEN_TOPOLOGY = {
    "N1_main": 6,
    "N1_side": 0,
    "N2_main": 37,
    "N2_side": 23,
    "n_core_group": 5,
}

NARROW_LIMITS = {
    "resonance_min_Hz": 14_900.0,
    "resonance_hard_Hz": 15_000.0,
    "winding_hard_C": 100.0,
    "winding_narrow_max_C": 100.5,
    "core_hard_C": 120.0,
    "core_narrow_max_C": 120.5,
    "composite_resonance_deficit_max_Hz": 75.0,
    "composite_winding_excess_max_C": 0.25,
    "composite_core_excess_max_C": 0.20,
}
RESIDUAL_LIMITS = {
    "volume_L": 20.0,
    "total_loss_W": 300.0,
    "resonance_Hz": 500.0,
    "winding_C": 3.0,
    "core_C": 3.0,
}
REQUIRED_GUARDS = {
    "resonance_Hz": 15_020.0,
    "winding_max_C": 99.8,
    "core_max_C": 119.8,
    "W_mm": 1_199.0,
    "L_mm": 999.0,
    "H_mm": 749.0,
}

RESONANCE_CONSTRAINT_SCALE_HZ = 150.0
TEMPERATURE_CONSTRAINT_SCALE_C = 10.0
MAX_TRUST_RADIUS = 0.05
MAX_CANDIDATES = 3
LOCAL_SUPPORT_ROWS = 64
LOCAL_RIDGE = 1.0e-4

FROZEN_UNIT_INDICES = frozenset((0, 1, 2, 7, 8, 9))
ACTIVE_UNIT_INDICES = tuple(
    index
    for index in range(len(input_parameter._SOBOL_DIMS))  # noqa: SLF001
    if index not in FROZEN_UNIT_INDICES
)

MUTABLE_GEOMETRY_KEYS = (
    "N1_main",
    "N1_side",
    "N2_main",
    "N2_side",
    "l1",
    "l2",
    "h1",
    "w1",
    "n_core_group",
    "core_plate_t",
    "wcp_t",
    "wcp_len_x",
    "cw1",
    "gap1",
    "cw2",
    "gap2",
    "nwh1",
    "nwh2",
    "cc_w2c_space_x",
    "cc_w2c_space_y",
    "w2c_w1c_space_x",
    "w2c_w1c_space_y",
    "w1c_w2s_space_x",
    "w2s_w1s_space_x",
    "w1s_w2s_space_y",
    "w1s_cs_space_x",
    "cs_w1s_space_y",
)

# Each template is a bounded perturbation in the repaired goal chromosome.
# A: balanced; B: thermal; C: resonance; D: core-temperature.
TEMPLATES = (
    {
        "template": "A",
        "purpose": "balanced_resonance_and_temperature",
        "raw_modifications": {"f1_split": -0.05, "w1": 0.05},
    },
    {
        "template": "B",
        "purpose": "winding_and_core_temperature",
        "raw_modifications": {"w1": 0.05},
    },
    {
        "template": "C",
        "purpose": "maximum_resonance_recovery",
        "raw_modifications": {"f1_split": -0.05, "wh2": 0.02},
    },
    {
        "template": "D",
        "purpose": "core_temperature_recovery",
        "raw_modifications": {"l1": 0.01, "w1": 0.02},
    },
)
TEMPLATE_PRIORITY = {"A": 0, "B": 1, "C": 2, "D": 3}

sealed = rounded.sealed
validate_seal = rounded.validate_seal
payload_sha256 = rounded.payload_sha256
read_json = rounded.read_json
write_immutable_json = rounded.write_immutable_json
sha256_file = rounded.sha256_file


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ContractError(f"{label} must be finite")
    return number


def _parse_json_mapping(value: Any, label: str) -> dict[str, float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise ContractError(f"{label} must encode an object")
    return {
        str(name): _finite(raw, f"{label}.{name}")
        for name, raw in parsed.items()
    }


def _parse_coordinate(value: Any, label: str) -> tuple[float, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is malformed JSON") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != len(input_parameter._SOBOL_DIMS)  # noqa: SLF001
    ):
        raise ContractError(f"{label} has the wrong coordinate width")
    coordinate = tuple(_finite(item, label) for item in parsed)
    if any(item < 0.0 or item > 1.0 for item in coordinate):
        raise ContractError(f"{label} leaves [0,1]")
    return coordinate


def _record(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ContractError("correction artifact escapes output root") from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _absolute_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(f"not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_collection_record(
    root: Path,
    record: Any,
    label: str,
) -> Path:
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise ContractError(f"{label} record is malformed")
    raw = Path(str(record["path"]))
    target = (
        raw.resolve(strict=True)
        if raw.is_absolute()
        else (root / raw).resolve(strict=True)
    )
    try:
        target.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ContractError(f"{label} escapes collection root") from exc
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_size != int(record["size_bytes"])
        or sha256_file(target) != record["sha256"]
    ):
        raise ContractError(f"{label} bytes drifted")
    return target


def _temperature_measurement(
    result: Mapping[str, Any],
) -> tuple[dict[str, float], float, float]:
    active, evidence, _passed = production._temperature_gate_evidence(  # noqa: SLF001
        result
    )
    if set(active) != set(evidence):
        raise ContractError("task96340 temperature evidence drifted")
    values = {
        target: _finite(item.get("actual_C"), f"temperature {target}")
        for target, item in evidence.items()
    }
    winding = max(
        values[target]
        for target in WINDING_TEMPERATURE_TARGETS
        if target in values
    )
    core = max(
        values[target]
        for target in CORE_TEMPERATURE_TARGETS
        if target in values
    )
    return values, winding, core


def measurement_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    volume_l, raw_dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        _finite(value, "actual exterior dimension")
        for value in raw_dimensions
    )
    temperatures, winding, core = _temperature_measurement(result)
    losses = {
        name: _finite(result.get(name), f"actual {name}")
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    return {
        "actual_volume_L": _finite(volume_l, "actual volume"),
        "actual_total_loss_W": sum(losses.values()),
        "actual_loss_components_W": losses,
        "actual_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_resonance_Hz": _finite(
            result.get("f_res_min_tx_rx_only_Hz"),
            "actual resonance",
        ),
        "actual_temperature_targets_C": temperatures,
        "actual_winding_max_C": winding,
        "actual_core_max_C": core,
        "actual_physical_Llt_uH": 2.0
        * _finite(result.get("Llt"), "actual symmetric Llt"),
    }


def authenticate_task96340_collection(
    *,
    standard_plan_path: Path,
    standard_final_path: Path,
    standard_collection_root: Path,
) -> dict[str, Any]:
    """Authenticate exact rounded terminal success without requiring PASS."""

    contract = rounded_collector.load_contract(
        plan_path=standard_plan_path,
        final_path=standard_final_path,
    )
    if (
        contract.get("task_id") != SOURCE_TASK_ID
        or contract.get("task_name") != rounded.TASK_NAME
        or contract.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or contract.get("solver_revision")
        != EXPECTED_SOURCE_SOLVER_REVISION
        or contract.get("node_name") != rounded.NODE_NAME
    ):
        raise ContractError("correction source is not exact task96340")

    root = standard_collection_root.resolve(strict=True)
    receipt_path = (root / "collection_receipt.json").resolve(strict=True)
    seal_path = (root / "collection_seal.json").resolve(strict=True)
    receipt = standard_collector._validate_seal(  # noqa: SLF001
        standard_collector._read_json(  # noqa: SLF001
            receipt_path, "rounded task96340 collection receipt"
        ),
        standard_collector.COLLECTION_SCHEMA,
        "rounded task96340 collection receipt",
    )
    collection_seal = standard_collector._validate_seal(  # noqa: SLF001
        standard_collector._read_json(  # noqa: SLF001
            seal_path, "rounded task96340 collection seal"
        ),
        standard_collector.COLLECTION_SEAL_SCHEMA,
        "rounded task96340 collection seal",
    )
    receipt_record = _resolve_collection_record(
        root,
        collection_seal.get("collection_receipt"),
        "task96340 collection receipt",
    )
    if (
        receipt_record != receipt_path
        or collection_seal.get("collection_receipt_payload_sha256")
        != receipt["payload_sha256"]
        or receipt.get("task_id") != SOURCE_TASK_ID
        or receipt.get("task_name") != contract["task_name"]
        or receipt.get("dedupe_key") != contract["dedupe_key"]
        or receipt.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or receipt.get("source_plan_file_sha256")
        != contract["plan_file_sha256"]
        or receipt.get("source_submission_file_sha256")
        != contract["submission_file_sha256"]
        or receipt.get("scheduler_get_only_collection") is not True
        or receipt.get("scheduler_mutation_performed") is not False
    ):
        raise ContractError("task96340 collection lineage drifted")

    result_path = _resolve_collection_record(
        root, receipt.get("result_json"), "task96340 result"
    )
    result = read_json(result_path)
    if receipt.get("result_sha256") != payload_sha256(result):
        raise ContractError("task96340 result payload drifted")
    artifact_path = _resolve_collection_record(
        root,
        receipt.get("retained_symmetric_aedt"),
        "task96340 symmetric AEDT",
    )
    if artifact_path.name != "symmetric.aedt":
        raise ContractError("task96340 AEDT filename drifted")

    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ContractError("task96340 source inventory is absent")
    terminal_path = _resolve_collection_record(
        root,
        source_files.get("scheduler_terminal_task.json"),
        "task96340 terminal task",
    )
    terminal = read_json(terminal_path)
    expected_terminal = {
        "task_id": SOURCE_TASK_ID,
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": rounded.PROJECT,
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "actual_node_name": rounded.NODE_NAME,
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
    }
    terminal_drift = {
        key: {"expected": expected, "actual": terminal.get(key)}
        for key, expected in expected_terminal.items()
        if terminal.get(key) != expected
    }
    if terminal_drift:
        raise ContractError(
            f"task96340 terminal identity drifted: {terminal_drift}"
        )

    source = rounded.direct.authenticate_source_authority()
    params, profile = rounded.derive_rounded_params_and_profile(source)
    effective = scheduler_client.effective_verification_params(
        params, profile
    )
    if not scheduler_client.result_matches_params(
        result,
        effective,
        required_keys=set(input_parameter.ALL_INPUT_KEYS),
    ):
        raise ContractError("task96340 result parameters drifted")
    if (
        _finite(result.get("full_model"), "full_model") != 0.0
        or str(result.get("thermal_symmetry") or "").casefold()
        != "eighth"
        or _finite(result.get("round_corner"), "round_corner") != 1.0
        or _finite(result.get("corner_radius"), "corner_radius") != 10.0
        or _finite(result.get("corner_segments"), "corner_segments") != 4.0
    ):
        raise ContractError("task96340 rounded model identity drifted")
    fixed = {
        "fan_config": result.get("fan_config"),
        "fan_velocity_m_s": _finite(
            result.get("fan_velocity"), "fan_velocity"
        ),
        "thermal_pad_conductivity_W_mK": _finite(
            result.get("k_ins"), "k_ins"
        ),
        "core_plate_pad_t_mm": _finite(
            result.get("core_plate_pad_t"), "core_plate_pad_t"
        ),
        "wcp_pad_t_mm": _finite(
            result.get("wcp_pad_t"), "wcp_pad_t"
        ),
    }
    if fixed != FIXED_BOUNDARY:
        raise ContractError(
            f"task96340 fixed boundary drifted: {fixed}"
        )
    measurement = measurement_from_result(result)
    authority = sealed(
        {
            "schema_version": SOURCE_SCHEMA,
            "authenticated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_task_id": SOURCE_TASK_ID,
            "source_task_name": contract["task_name"],
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "source_solver_revision": contract["solver_revision"],
            "source_library_revision": contract["library_revision"],
            "source_node_name": rounded.NODE_NAME,
            "source_plan": _absolute_file_record(standard_plan_path),
            "source_submission_final": _absolute_file_record(
                standard_final_path
            ),
            "source_collection_receipt": _absolute_file_record(receipt_path),
            "source_collection_seal": _absolute_file_record(seal_path),
            "source_result": _absolute_file_record(result_path),
            "source_symmetric_aedt": _absolute_file_record(artifact_path),
            "source_terminal_task": _absolute_file_record(terminal_path),
            "measurement": copy.deepcopy(measurement),
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "terminal_authenticated": True,
            "artifact_authenticated": True,
            "solver_valid": True,
            "physics_valid": True,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
        }
    )
    return {
        "authority": authority,
        "result": result,
        "params": params,
        "profile": profile,
        "selected": source["selected"],
    }


def anchor_context(selected: Mapping[str, Any]) -> dict[str, Any]:
    row = selected.get("official_row")
    if not isinstance(row, Mapping):
        row = selected.get("source_row")
    if not isinstance(row, Mapping):
        raise ContractError("candidate #5 official row is absent")
    if (
        str(row.get("candidate_physics_sha") or "")
        != SOURCE_CANDIDATE_SHA256
        or str(row.get("physical_geometry_sha256") or "")
        != SOURCE_CANDIDATE_SHA256
        or int(row.get("standard_selection_order", 0)) != 5
    ):
        raise ContractError("candidate #5 official row identity drifted")
    coordinate = _parse_coordinate(
        row.get("coordinate_unit_json"), "candidate #5 coordinate"
    )
    physical = _parse_json_mapping(
        row.get("physical_G_json"), "candidate #5 physical constraints"
    )
    normalized = _parse_json_mapping(
        row.get("normalized_G_json"),
        "candidate #5 normalized constraints",
    )
    return {
        "row": copy.deepcopy(dict(row)),
        "coordinate": coordinate,
        "physical_constraints": physical,
        "normalized_constraints": normalized,
        "objective_volume_L": _finite(
            row.get("objective_volume_L"), "candidate #5 volume"
        ),
        "objective_total_loss_W": _finite(
            row.get("objective_total_loss_W"), "candidate #5 loss"
        ),
    }


def predicted_anchor_metrics(anchor: Mapping[str, Any]) -> dict[str, Any]:
    physical = anchor["physical_constraints"]
    temperatures: dict[str, float] = {}
    prefix = "temperature_robust_limit:"
    for name, value in physical.items():
        if name.startswith(prefix):
            target = name[len(prefix) :]
            if target in TEMPERATURE_TARGET_LIMITS_C:
                temperatures[target] = (
                    float(TEMPERATURE_TARGET_LIMITS_C[target])
                    + float(value)
                )
    if not temperatures:
        raise ContractError("candidate #5 temperature prediction is absent")
    dimensions = {
        "W": GOAL_SIZE_LIMITS_MM["W"]
        + float(physical["exterior_width_limit"]),
        "L": GOAL_SIZE_LIMITS_MM["L"]
        + float(physical["exterior_length_limit"]),
        "H": GOAL_SIZE_LIMITS_MM["H"]
        + float(physical["exterior_height_limit"]),
    }
    return {
        "volume_L": float(anchor["objective_volume_L"]),
        "total_loss_W": float(anchor["objective_total_loss_W"]),
        "resonance_Hz": GOAL_RESONANCE_MIN_HZ
        - float(physical["half_magnetizing_resonance_minimum"]),
        "winding_max_C": max(
            temperatures[target]
            for target in WINDING_TEMPERATURE_TARGETS
            if target in temperatures
        ),
        "core_max_C": max(
            temperatures[target]
            for target in CORE_TEMPERATURE_TARGETS
            if target in temperatures
        ),
        "dimensions_mm": dimensions,
        "temperature_targets_C": temperatures,
    }


def narrow_miss_gate(
    *,
    measurement: Mapping[str, Any],
    predicted: Mapping[str, Any],
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Admit only a finite, physically correctable task96340 miss."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ContractError("narrow-miss observation time is naive")
    actual_dimensions = measurement.get("actual_dimensions_mm")
    if not isinstance(actual_dimensions, Mapping):
        raise ContractError("actual dimensions are absent")
    dimension_pass = {
        axis: _finite(actual_dimensions.get(axis), f"actual {axis}")
        <= float(GOAL_SIZE_LIMITS_MM[axis])
        for axis in ("W", "L", "H")
    }
    actual_resonance = _finite(
        measurement.get("actual_resonance_Hz"), "actual resonance"
    )
    actual_winding = _finite(
        measurement.get("actual_winding_max_C"),
        "actual winding maximum",
    )
    actual_core = _finite(
        measurement.get("actual_core_max_C"), "actual core maximum"
    )
    failed: list[str] = []
    if actual_resonance < NARROW_LIMITS["resonance_hard_Hz"]:
        failed.append("resonance")
    if actual_winding > NARROW_LIMITS["winding_hard_C"]:
        failed.append("winding_temperature")
    if actual_core > NARROW_LIMITS["core_hard_C"]:
        failed.append("core_temperature")

    reasons: list[str] = []
    if not all(dimension_pass.values()):
        reasons.append("dimension_miss_is_not_fea_error")
    if not failed:
        reasons.append("source_already_passes_requested_hard_constraints")
    if actual_resonance < NARROW_LIMITS["resonance_min_Hz"]:
        reasons.append("resonance_miss_exceeds_100_Hz")
    if actual_winding > NARROW_LIMITS["winding_narrow_max_C"]:
        reasons.append("winding_miss_exceeds_0p5_C")
    if actual_core > NARROW_LIMITS["core_narrow_max_C"]:
        reasons.append("core_miss_exceeds_0p5_C")

    residuals = {
        "volume_L": abs(
            _finite(measurement.get("actual_volume_L"), "actual volume")
            - _finite(predicted.get("volume_L"), "predicted volume")
        ),
        "total_loss_W": abs(
            _finite(
                measurement.get("actual_total_loss_W"), "actual loss"
            )
            - _finite(predicted.get("total_loss_W"), "predicted loss")
        ),
        "resonance_Hz": abs(
            actual_resonance
            - _finite(predicted.get("resonance_Hz"), "predicted resonance")
        ),
        "winding_C": abs(
            actual_winding
            - _finite(
                predicted.get("winding_max_C"),
                "predicted winding maximum",
            )
        ),
        "core_C": abs(
            actual_core
            - _finite(predicted.get("core_max_C"), "predicted core maximum")
        ),
    }
    residual_pass = {
        name: value <= RESIDUAL_LIMITS[name]
        for name, value in residuals.items()
    }
    if not all(residual_pass.values()):
        reasons.append("surrogate_to_fea_residual_is_not_local")

    if len(failed) > 1:
        if (
            max(
                0.0,
                NARROW_LIMITS["resonance_hard_Hz"]
                - actual_resonance,
            )
            > NARROW_LIMITS["composite_resonance_deficit_max_Hz"]
            or max(
                0.0,
                actual_winding - NARROW_LIMITS["winding_hard_C"],
            )
            > NARROW_LIMITS["composite_winding_excess_max_C"]
            or max(
                0.0,
                actual_core - NARROW_LIMITS["core_hard_C"],
            )
            > NARROW_LIMITS["composite_core_excess_max_C"]
        ):
            reasons.append("composite_miss_exceeds_strict_local_limits")
    eligible = not reasons
    return sealed(
        {
            "schema_version": GATE_SCHEMA,
            "observed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "source_task_id": SOURCE_TASK_ID,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "actual": copy.deepcopy(dict(measurement)),
            "surrogate_anchor": copy.deepcopy(dict(predicted)),
            "failed_requested_constraints": failed,
            "dimension_pass": dimension_pass,
            "narrow_limits": copy.deepcopy(NARROW_LIMITS),
            "residuals": residuals,
            "residual_limits": copy.deepcopy(RESIDUAL_LIMITS),
            "residual_pass": residual_pass,
            "prepare_allowed": eligible,
            "submission_allowed": False,
            "reasons": reasons,
            "scheduler_methods_used": [],
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
        }
    )


class _RepairBase:
    """Minimal model-free base required by the reviewed goal repair class."""

    def __init__(
        self,
        models: Mapping[str, Any],
        spec: Mapping[str, Any] | None = None,
        density_gate: Any = None,
        fixed_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.models = models
        self.spec = dict(spec or {})
        self.density_gate = density_gate
        self.constraint_names = goal_repair.GOAL_BASE_CONSTRAINT_NAMES
        self.n_ieq_constr = len(self.constraint_names)
        self.n_var = len(input_parameter._SOBOL_DIMS)  # noqa: SLF001
        self.xl = np.zeros(self.n_var, dtype=float)
        self.xu = np.ones(self.n_var, dtype=float)
        self.fixed_overrides = dict(fixed_overrides or {})
        self.fixed_overrides.update(
            goal_repair.EXPECTED_SIMPLE_BASE_FIXED_STACK_MM
        )


def exact_goal_repair_problem(
    *,
    core_lamination_factor: float,
) -> Any:
    """Create only the reviewed repair/decoder portion of the goal problem."""

    problem_class = goal_repair.create_current7_problem_class(
        base_problem_class=_RepairBase,
        base_constraint_names=goal_repair.BASE_CONSTRAINT_NAMES,
        base_fixed_stack_mm=(
            goal_repair.EXPECTED_SIMPLE_BASE_FIXED_STACK_MM
        ),
        sobol_dims=input_parameter._SOBOL_DIMS,  # noqa: SLF001
        bounding_box_lit=lambda _row: (0.0, (0.0, 0.0, 0.0)),
        input_parameter_module=input_parameter,
        design_analytical_b_field_t=lambda _row: 0.0,
    )
    problem = problem_class(
        {},
        spec=GOAL_STAGE_SPEC,
        fixed_primary_turns=FROZEN_TOPOLOGY["N1_main"],
    )
    # The production repair consumes this material constant from its enclosing
    # campaign configuration.  It is not a GOAL_STAGE_SPEC decision field.
    problem.spec["core_lamination_factor"] = _finite(
        core_lamination_factor, "core lamination factor"
    )
    return problem


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1.0e-9
        )
    except (TypeError, ValueError, OverflowError):
        return left == right


def attest_anchor_replay(
    *,
    problem: Any,
    anchor_coordinate: Sequence[float],
    base_params: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    coordinate = np.asarray(anchor_coordinate, dtype=float)
    repaired = np.asarray(
        problem.repair_unit_coordinates(coordinate), dtype=float
    )
    if not np.array_equal(repaired, coordinate):
        raise ContractError("candidate #5 coordinate is not a repair fixed point")
    frame, shrink, valid = problem.decode_batch(
        repaired.reshape(1, -1)
    )
    if not bool(valid[0]) or float(shrink[0]) != 0.0:
        raise ContractError("candidate #5 exact decoder replay is invalid")
    row = frame.iloc[0]
    drift = {
        key: {"expected": base_params.get(key), "actual": row.get(key)}
        for key in MUTABLE_GEOMETRY_KEYS
        if key in row and not _same_number(base_params.get(key), row.get(key))
    }
    if drift:
        raise ContractError(f"candidate #5 exact replay drifted: {drift}")
    topology = {
        key: int(_finite(row.get(key), key))
        for key in FROZEN_TOPOLOGY
    }
    if topology != FROZEN_TOPOLOGY:
        raise ContractError("candidate #5 topology drifted")
    evidence = sealed(
        {
            "schema_version": (
                "mft-goal-rounded-exact-repair-anchor-attestation-v1"
            ),
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "coordinate_fixed_point": True,
            "decoder_valid": True,
            "space_shrink_needed": 0.0,
            "replayed_topology": topology,
            "geometry_drift": {},
            "repair_callable": (
                "tier1_corrected_generation_preflight."
                "Current7Tier1Problem.repair_unit_coordinates"
            ),
            "decoder_callable": (
                "tier1_corrected_generation_preflight."
                "decode_unit_sample_with_cw1"
            ),
        }
    )
    return row, evidence


def legacy_path_rejection(
    *,
    anchor_coordinate: Sequence[float],
    base_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the obsolete raw decoder cannot generate these plans."""

    decoded = input_parameter.decode_unit_sample(
        input_parameter.unit_to_dims(anchor_coordinate),
        allow_space_shrink=False,
    )
    required_drift_keys = ("cw1", "cw2", "wcp_len_x")
    drift = {
        key: {
            "sealed_candidate": base_params.get(key),
            "legacy_raw_decode": decoded.get(key),
        }
        for key in required_drift_keys
        if not _same_number(base_params.get(key), decoded.get(key))
    }
    if set(drift) != set(required_drift_keys):
        raise ContractError(
            "legacy local-trust path drift signature changed; "
            "manual review is required"
        )
    return sealed(
        {
            "schema_version": LEGACY_REJECTION_SCHEMA,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "legacy_module": "tools/mft_goal_local_trust_acquisition.py",
            "legacy_neighbor_callable": "generate_local_neighbors",
            "legacy_decoder_path": (
                "decode_unit_sample(unit_to_dims(coordinate))"
            ),
            "observed_anchor_drift": drift,
            "legacy_path_reproduces_sealed_anchor": False,
            "legacy_path_allowed": False,
            "exact_goal_repair_required": True,
            "reason": (
                "legacy raw f1_split semantics bypass goal cw1 grid repair"
            ),
            "scheduler_post_calls": 0,
        }
    )


def _sobol_index(name: str) -> int:
    matches = [
        index
        for index, dimension in enumerate(
            input_parameter._SOBOL_DIMS  # noqa: SLF001
        )
        if dimension[0] == name
    ]
    if len(matches) != 1:
        raise ContractError(f"Sobol coordinate {name} is not unique")
    return matches[0]


def generate_exact_templates(
    *,
    problem: Any,
    anchor_coordinate: Sequence[float],
    base_params: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base = np.asarray(anchor_coordinate, dtype=float)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for template in TEMPLATES:
        raw = base.copy()
        for name, delta in template["raw_modifications"].items():
            index = _sobol_index(name)
            raw[index] = np.clip(raw[index] + float(delta), 0.0, 1.0)
        repaired = np.asarray(
            problem.repair_unit_coordinates(raw), dtype=float
        )
        distance = float(np.max(np.abs(repaired - base)))
        if distance <= 0.0 or distance > MAX_TRUST_RADIUS + 1.0e-12:
            raise ContractError(
                f"template {template['template']} escaped trust radius"
            )
        frame, shrink, valid = problem.decode_batch(
            repaired.reshape(1, -1)
        )
        if not bool(valid[0]) or float(shrink[0]) != 0.0:
            raise ContractError(
                f"template {template['template']} exact decode failed"
            )
        row = frame.iloc[0]
        topology = {
            key: int(_finite(row.get(key), key))
            for key in FROZEN_TOPOLOGY
        }
        if topology != FROZEN_TOPOLOGY:
            raise ContractError(
                f"template {template['template']} changed topology"
            )
        params = copy.deepcopy(dict(base_params))
        for key in MUTABLE_GEOMETRY_KEYS:
            if key in row:
                value = row[key]
                params[key] = value.item() if hasattr(value, "item") else value
        params.update(
            {
                "round_corner": 1,
                "corner_radius": 10.0,
                "corner_segments": 4,
                "full_model": 0,
                "thermal_symmetry": "eighth",
                "fan_config": "dual",
                "fan_velocity": 1.5,
                "k_ins": 0.2,
                "core_plate_pad_t": 2.0,
                "wcp_pad_t": 2.0,
            }
        )
        effective = scheduler_client.effective_verification_params(
            params, dict(profile)
        )
        rounded._validate_effective_contract(effective)
        frozen = {
            **{key: int(params[key]) for key in FROZEN_TOPOLOGY},
            "round_corner": int(params["round_corner"]),
            "corner_radius": float(params["corner_radius"]),
            "corner_segments": int(params["corner_segments"]),
            "full_model": int(effective["full_model"]),
            "thermal_symmetry": effective["thermal_symmetry"],
            "fan_config": effective["fan_config"],
            "fan_velocity": float(effective["fan_velocity"]),
            "k_ins": float(effective["k_ins"]),
            "core_plate_pad_t": float(
                effective["core_plate_pad_t"]
            ),
            "wcp_pad_t": float(effective["wcp_pad_t"]),
        }
        expected_frozen = {
            **FROZEN_TOPOLOGY,
            "round_corner": 1,
            "corner_radius": 10.0,
            "corner_segments": 4,
            "full_model": 0,
            "thermal_symmetry": "eighth",
            "fan_config": "dual",
            "fan_velocity": 1.5,
            "k_ins": 0.2,
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
        }
        if frozen != expected_frozen:
            raise ContractError(
                f"template {template['template']} frozen contract drifted"
            )
        geometry = {
            key: params.get(key) for key in MUTABLE_GEOMETRY_KEYS
        }
        geometry_sha = payload_sha256(geometry)
        if geometry_sha in seen:
            raise ContractError("correction templates are not unique")
        seen.add(geometry_sha)
        output.append(
            {
                "template": template["template"],
                "purpose": template["purpose"],
                "raw_modifications": copy.deepcopy(
                    template["raw_modifications"]
                ),
                "unit_coordinate": repaired.tolist(),
                "distance_linf": distance,
                "decoded_params": params,
                "geometry_sha256": geometry_sha,
                "frozen_contract": frozen,
                "exact_goal_repair_used": True,
                "legacy_local_trust_generator_used": False,
            }
        )
    return output


def _canonical_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ContractError(f"{label} is not a canonical boolean")


def _verified_terminal_table(selected: Mapping[str, Any]) -> pd.DataFrame:
    authentication = selected.get("authentication")
    record = (
        authentication.get("source_terminal_candidates")
        if isinstance(authentication, Mapping)
        else None
    )
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise ContractError("source terminal population record is absent")
    path = Path(record["path"]).resolve(strict=True)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ContractError("source terminal population bytes drifted")
    frame = pd.read_csv(path)
    if len(frame) != 320:
        raise ContractError("source terminal population is not 320 rows")
    return frame


def fit_local_model(
    *,
    training: pd.DataFrame,
    anchor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Fit the campaign's anchor-centred ridge without legacy generation."""

    valid = training.copy()
    for name in (
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
    ):
        if name not in valid:
            raise ContractError(f"source training lacks {name}")
        valid = valid[
            valid[name].map(
                lambda value: _canonical_bool(value, f"training {name}")
            )
        ]
    coordinates = np.vstack(
        [
            np.asarray(
                _parse_coordinate(value, "training coordinate"),
                dtype=float,
            )
            for value in valid["coordinate_unit_json"]
        ]
    )
    anchor_coordinate = np.asarray(anchor["coordinate"], dtype=float)
    active_delta = (
        coordinates[:, ACTIVE_UNIT_INDICES]
        - anchor_coordinate[list(ACTIVE_UNIT_INDICES)]
    )
    distances = np.linalg.norm(active_delta, axis=1)
    count = min(LOCAL_SUPPORT_ROWS, len(valid))
    if count < 10:
        raise ContractError("too few local support rows")
    nearest = np.argsort(distances, kind="stable")[:count]
    constraint_names = tuple(
        sorted(anchor["normalized_constraints"])
    )
    output_names = (
        "objective_volume_L",
        "objective_total_loss_W",
        *(f"normalized_G:{name}" for name in constraint_names),
    )
    missing = sorted(set(output_names).difference(valid.columns))
    if missing:
        raise ContractError(
            "source training lacks outputs: " + ",".join(missing)
        )
    outputs = valid[list(output_names)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(outputs).all():
        raise ContractError("source training outputs are non-finite")
    anchor_output = np.asarray(
        [
            float(anchor["objective_volume_L"]),
            float(anchor["objective_total_loss_W"]),
            *(
                float(anchor["normalized_constraints"][name])
                for name in constraint_names
            ),
        ],
        dtype=float,
    )
    x = active_delta[nearest]
    y = outputs[nearest] - anchor_output
    support_radius = max(float(distances[nearest[-1]]), 0.02)
    weights = np.exp(
        -0.5
        * (
            np.linalg.norm(x, axis=1)
            / max(support_radius, np.finfo(float).eps)
        )
        ** 2
    )
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights[:, None] * y)
    try:
        coefficients = np.linalg.solve(
            gram + LOCAL_RIDGE * np.eye(gram.shape[0]), rhs
        )
    except np.linalg.LinAlgError as exc:
        raise ContractError("local correction model is singular") from exc
    return coefficients, anchor_output, output_names, {
        "method": "anchor_centered_weighted_ridge_delta_linear",
        "support_row_count": count,
        "support_radius_l2_unit": support_radius,
        "ridge": LOCAL_RIDGE,
        "active_unit_indices": list(ACTIVE_UNIT_INDICES),
        "legacy_neighbor_generation_used": False,
    }


def _actual_output_vector(
    *,
    anchor_output: np.ndarray,
    output_names: Sequence[str],
    measurement: Mapping[str, Any],
) -> np.ndarray:
    values = anchor_output.copy()
    index = {name: position for position, name in enumerate(output_names)}
    values[index["objective_volume_L"]] = _finite(
        measurement.get("actual_volume_L"), "actual volume"
    )
    values[index["objective_total_loss_W"]] = _finite(
        measurement.get("actual_total_loss_W"), "actual loss"
    )
    dimensions = measurement["actual_dimensions_mm"]
    for axis, constraint in (
        ("W", "exterior_width_limit"),
        ("L", "exterior_length_limit"),
        ("H", "exterior_height_limit"),
    ):
        name = f"normalized_G:{constraint}"
        if name in index:
            values[index[name]] = (
                _finite(dimensions[axis], f"actual dimension {axis}")
                - GOAL_SIZE_LIMITS_MM[axis]
            )
    resonance_name = (
        "normalized_G:half_magnetizing_resonance_minimum"
    )
    values[index[resonance_name]] = (
        GOAL_RESONANCE_MIN_HZ
        - _finite(
            measurement.get("actual_resonance_Hz"), "actual resonance"
        )
    ) / RESONANCE_CONSTRAINT_SCALE_HZ
    temperatures = measurement["actual_temperature_targets_C"]
    for target, actual in temperatures.items():
        name = f"normalized_G:temperature_robust_limit:{target}"
        if name in index:
            values[index[name]] = (
                _finite(actual, f"actual temperature {target}")
                - TEMPERATURE_TARGET_LIMITS_C[target]
            ) / TEMPERATURE_CONSTRAINT_SCALE_C
    llt_name = "normalized_G:Llt_robust_band"
    if llt_name in index:
        physical_llt = _finite(
            measurement.get("actual_physical_Llt_uH"),
            "actual physical Llt",
        )
        values[index[llt_name]] = (
            abs(physical_llt - 27.5) - 0.55
        ) / 0.55
    return values


def _prediction_metrics(
    values: np.ndarray,
    output_names: Sequence[str],
) -> dict[str, Any]:
    index = {name: position for position, name in enumerate(output_names)}
    resonance = GOAL_RESONANCE_MIN_HZ - (
        values[
            index[
                "normalized_G:half_magnetizing_resonance_minimum"
            ]
        ]
        * RESONANCE_CONSTRAINT_SCALE_HZ
    )
    temperatures: dict[str, float] = {}
    for target, limit in TEMPERATURE_TARGET_LIMITS_C.items():
        name = f"normalized_G:temperature_robust_limit:{target}"
        if name in index:
            temperatures[target] = float(
                limit
                + values[index[name]] * TEMPERATURE_CONSTRAINT_SCALE_C
            )
    dimensions = {}
    for axis, constraint in (
        ("W", "exterior_width_limit"),
        ("L", "exterior_length_limit"),
        ("H", "exterior_height_limit"),
    ):
        dimensions[axis] = float(
            GOAL_SIZE_LIMITS_MM[axis]
            + values[index[f"normalized_G:{constraint}"]]
        )
    return {
        "volume_L": float(values[index["objective_volume_L"]]),
        "total_loss_W": float(
            values[index["objective_total_loss_W"]]
        ),
        "resonance_Hz": float(resonance),
        "winding_max_C": max(
            temperatures[target]
            for target in WINDING_TEMPERATURE_TARGETS
            if target in temperatures
        ),
        "core_max_C": max(
            temperatures[target]
            for target in CORE_TEMPERATURE_TARGETS
            if target in temperatures
        ),
        "dimensions_mm": dimensions,
        "temperature_targets_C": temperatures,
    }


def score_templates(
    *,
    templates: Sequence[Mapping[str, Any]],
    anchor_coordinate: Sequence[float],
    measurement: Mapping[str, Any],
    gate: Mapping[str, Any],
    coefficients: np.ndarray,
    anchor_output: np.ndarray,
    output_names: Sequence[str],
) -> list[dict[str, Any]]:
    actual_output = _actual_output_vector(
        anchor_output=anchor_output,
        output_names=output_names,
        measurement=measurement,
    )
    measurement_delta = actual_output - anchor_output
    base = np.asarray(anchor_coordinate, dtype=float)
    failed = set(gate["failed_requested_constraints"])
    ranked: list[dict[str, Any]] = []
    for raw_template in templates:
        template = copy.deepcopy(dict(raw_template))
        coordinate = np.asarray(template["unit_coordinate"], dtype=float)
        active_delta = (
            coordinate[list(ACTIVE_UNIT_INDICES)]
            - base[list(ACTIVE_UNIT_INDICES)]
        )
        surrogate_neighbor = anchor_output + active_delta @ coefficients
        distance = float(np.max(np.abs(coordinate - base)))
        decay = math.exp(
            -0.5 * (distance / MAX_TRUST_RADIUS) ** 2
        )
        corrected = surrogate_neighbor + decay * measurement_delta
        prediction = _prediction_metrics(corrected, output_names)
        guards = {
            "resonance_Hz": prediction["resonance_Hz"]
            >= REQUIRED_GUARDS["resonance_Hz"],
            "winding_max_C": prediction["winding_max_C"]
            <= REQUIRED_GUARDS["winding_max_C"],
            "core_max_C": prediction["core_max_C"]
            <= REQUIRED_GUARDS["core_max_C"],
            "W_mm": prediction["dimensions_mm"]["W"]
            <= REQUIRED_GUARDS["W_mm"],
            "L_mm": prediction["dimensions_mm"]["L"]
            <= REQUIRED_GUARDS["L_mm"],
            "H_mm": prediction["dimensions_mm"]["H"]
            <= REQUIRED_GUARDS["H_mm"],
        }
        if not all(guards.values()):
            continue
        improvements: dict[str, float] = {}
        if "resonance" in failed:
            improvements["resonance_Hz"] = (
                prediction["resonance_Hz"]
                - float(measurement["actual_resonance_Hz"])
            )
        if "winding_temperature" in failed:
            improvements["winding_C"] = (
                float(measurement["actual_winding_max_C"])
                - prediction["winding_max_C"]
            )
        if "core_temperature" in failed:
            improvements["core_C"] = (
                float(measurement["actual_core_max_C"])
                - prediction["core_max_C"]
            )
        if not improvements or any(value <= 0.0 for value in improvements.values()):
            continue
        normalized_improvement = sum(
            value
            / (
                100.0
                if name == "resonance_Hz"
                else 0.5
            )
            for name, value in improvements.items()
        )
        guard_margin = min(
            (prediction["resonance_Hz"] - 15_020.0) / 100.0,
            (99.8 - prediction["winding_max_C"]) / 0.5,
            (119.8 - prediction["core_max_C"]) / 0.5,
            (1_199.0 - prediction["dimensions_mm"]["W"]) / 5.0,
            (999.0 - prediction["dimensions_mm"]["L"]) / 5.0,
            (749.0 - prediction["dimensions_mm"]["H"]) / 5.0,
        )
        loss_penalty = max(
            0.0,
            prediction["total_loss_W"]
            - float(measurement["actual_total_loss_W"]),
        ) / 300.0
        score = (
            10.0 * normalized_improvement
            + guard_margin
            - distance
            - loss_penalty
        )
        template.update(
            {
                "measurement_delta_decay": decay,
                "corrected_prediction": prediction,
                "expected_improvement_from_measured": improvements,
                "required_guard": copy.deepcopy(REQUIRED_GUARDS),
                "guard_pass": guards,
                "correction_score": score,
            }
        )
        ranked.append(template)
    ranked.sort(
        key=lambda item: (
            -float(item["correction_score"]),
            TEMPLATE_PRIORITY[str(item["template"])],
            str(item["geometry_sha256"]),
        )
    )
    selected = ranked[:MAX_CANDIDATES]
    for rank, item in enumerate(selected, start=1):
        item["correction_rank"] = rank
    return selected


def prepare(
    *,
    standard_plan_path: Path,
    standard_final_path: Path,
    standard_collection_root: Path,
    output: Path = OUTPUT_ROOT,
    observed_at: datetime | None = None,
    collection_authenticator: Callable[..., dict[str, Any]] = (
        authenticate_task96340_collection
    ),
) -> Path:
    """Write one immutable local plan and never contact Scheduler."""

    source = collection_authenticator(
        standard_plan_path=standard_plan_path,
        standard_final_path=standard_final_path,
        standard_collection_root=standard_collection_root,
    )
    authority = source.get("authority")
    params = source.get("params")
    profile = source.get("profile")
    selected = source.get("selected")
    result = source.get("result")
    if (
        not isinstance(authority, dict)
        or validate_seal(authority, SOURCE_SCHEMA) is not authority
        or authority.get("source_task_id") != SOURCE_TASK_ID
        or authority.get("source_candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or authority.get("terminal_authenticated") is not True
        or authority.get("artifact_authenticated") is not True
        or authority.get("solver_valid") is not True
        or authority.get("physics_valid") is not True
        or authority.get("fixed_boundary") != FIXED_BOUNDARY
        or authority.get("rounding_policy") != ROUNDING_POLICY
        or not isinstance(params, Mapping)
        or not isinstance(profile, Mapping)
        or not isinstance(selected, Mapping)
        or not isinstance(result, Mapping)
    ):
        raise ContractError("authenticated task96340 source drifted")
    measurement = authority.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ContractError("task96340 measurement is absent")
    anchor = anchor_context(selected)
    predicted = predicted_anchor_metrics(anchor)
    gate = narrow_miss_gate(
        measurement=measurement,
        predicted=predicted,
        observed_at=observed_at,
    )
    if gate.get("prepare_allowed") is not True:
        raise ContractError(
            "task96340 is not a narrow miss: "
            + ",".join(gate.get("reasons") or [])
        )

    problem = exact_goal_repair_problem(
        core_lamination_factor=_finite(
            params.get("core_lamination_factor"),
            "core lamination factor",
        )
    )
    _base_row, replay = attest_anchor_replay(
        problem=problem,
        anchor_coordinate=anchor["coordinate"],
        base_params=params,
    )
    legacy = legacy_path_rejection(
        anchor_coordinate=anchor["coordinate"],
        base_params=params,
    )
    templates = generate_exact_templates(
        problem=problem,
        anchor_coordinate=anchor["coordinate"],
        base_params=params,
        profile=profile,
    )
    training = _verified_terminal_table(selected)
    coefficients, anchor_output, output_names, support = fit_local_model(
        training=training,
        anchor=anchor,
    )
    candidates = score_templates(
        templates=templates,
        anchor_coordinate=anchor["coordinate"],
        measurement=measurement,
        gate=gate,
        coefficients=coefficients,
        anchor_output=anchor_output,
        output_names=output_names,
    )
    if not candidates:
        raise ContractError(
            "no A/B/C/D template passes corrected prediction guards"
        )
    if len(candidates) > MAX_CANDIDATES:
        raise ContractError("bounded correction candidate cap escaped")

    root = output.resolve()
    if root.exists():
        raise ContractError(
            f"immutable bounded-correction output already exists: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    source_path = write_immutable_json(root / SOURCE_NAME, authority)
    gate_path = write_immutable_json(root / GATE_NAME, gate)
    legacy_path = write_immutable_json(
        root / LEGACY_REJECTION_NAME, legacy
    )
    candidate_paths: list[Path] = []
    for candidate in candidates:
        candidate_path = write_immutable_json(
            root
            / (
                f"candidate_{candidate['correction_rank']}_"
                f"{candidate['template'].lower()}.json"
            ),
            sealed(
                {
                    "schema_version": CANDIDATE_SCHEMA,
                    "source_task_id": SOURCE_TASK_ID,
                    "source_candidate_physics_sha256": (
                        SOURCE_CANDIDATE_SHA256
                    ),
                    "prepare_only": True,
                    "stage": "symmetric_standard",
                    "candidate": candidate,
                    "scheduler_methods_used": [],
                    "scheduler_post_calls": 0,
                    "submission_capability_present": False,
                }
            ),
        )
        candidate_paths.append(candidate_path)

    plan = sealed(
        {
            "schema_version": PLAN_SCHEMA,
            "created_at_utc": (
                observed_at or datetime.now(timezone.utc)
            ).astimezone(timezone.utc).isoformat(),
            "source_task_id": SOURCE_TASK_ID,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "source_authority": _record(root, source_path),
            "narrow_miss_gate": _record(root, gate_path),
            "legacy_local_trust_path_rejection": _record(
                root, legacy_path
            ),
            "exact_anchor_replay": replay,
            "local_model_support": support,
            "candidate_count": len(candidate_paths),
            "candidate_cap": MAX_CANDIDATES,
            "ranked_candidates": [
                _record(root, path) for path in candidate_paths
            ],
            "template_set": ["A", "B", "C", "D"],
            "same_selected_design_neighborhood": True,
            "correction_round": 1,
            "maximum_correction_rounds": 1,
            "parallel_symmetric_validation_allowed": True,
            "full_model_allowed": False,
            "surrogate_retraining_allowed": False,
            "turn_changes_allowed": False,
            "core_group_changes_allowed": False,
            "rounding_changes_allowed": False,
            "cooling_changes_allowed": False,
            "fixed_topology": copy.deepcopy(FROZEN_TOPOLOGY),
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "required_post_prediction_guards": copy.deepcopy(
                REQUIRED_GUARDS
            ),
            "prepare_only": True,
            "scheduler_repository_modified": False,
            "scheduler_methods_used": [],
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
            "scheduler_submission_performed": False,
            "submission_capability_present": False,
            "automatic_continuation": False,
        }
    )
    plan_path = write_immutable_json(root / PLAN_NAME, plan)
    receipt = sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "created_at_utc": plan["created_at_utc"],
            "plan": _record(root, plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_count": len(candidate_paths),
            "prepare_only": True,
            "scheduler_methods_used": [],
            "scheduler_post_calls": 0,
            "scheduler_mutation_performed": False,
            "submission_capability_present": False,
        }
    )
    write_immutable_json(root / RECEIPT_NAME, receipt)
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="prepare at most three local candidate plans"
    )
    prepare_parser.add_argument(
        "--standard-plan", type=Path, required=True
    )
    prepare_parser.add_argument(
        "--standard-final", type=Path, required=True
    )
    prepare_parser.add_argument(
        "--standard-collection", type=Path, required=True
    )
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = prepare(
        standard_plan_path=args.standard_plan,
        standard_final_path=args.standard_final,
        standard_collection_root=args.standard_collection,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "event": "rounded_bounded_correction_prepared",
                "plan": str(plan),
                "prepare_only": True,
                "scheduler_methods_used": [],
                "scheduler_post_calls": 0,
                "submission_capability_present": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "event": (
                        "rounded_bounded_correction_prepare_error"
                    ),
                    "error": str(exc),
                    "prepare_only": True,
                    "scheduler_methods_used": [],
                    "scheduler_post_calls": 0,
                    "submission_capability_present": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
