"""Bounded local acquisition after authenticated symmetric/Standard MFT FEA.

This tool is deliberately *not* a Scheduler client.  It can only prepare a
small, sealed symmetric/Standard candidate batch around one of the current
official anchors (#12, #6, then #1/#5 as backups).  It never creates a Full
continuation and it exposes no submission function.

The decision order is:

1. Reauthenticate the current 512-seed aggregate and measured Standard
   observations.
2. If any current measured symmetric result passes dimensions, resonance,
   winding temperature, and core temperature, stop and select it by actual
   minimum constraint margin, then actual loss and volume.
3. Otherwise classify each measured surrogate-to-FEA residual as invalid,
   small/local, or a large model mismatch.
4. Only a small residual may open a bounded trust region.  Fit a local
   delta-linear model from the authenticated terminal population, score a
   deterministic local stencil, and retain at most three neighbors.
5. Stop after two correction rounds or when no predicted constraint
   improvement remains.  Exact two-objective non-dominated sorting is required
   again after the new authenticated measurements arrive.

The fixed cooling boundary is fan=dual at 1.5 m/s, TIM/insulation conductivity
0.2 W/(m*K), and both thermal pads 2 mm.  Those values are never tunable here.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from module.input_parameter_260706 import (
    _SOBOL_DIMS,
    decode_unit_sample,
    unit_to_dims,
)
from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY,
    GOAL_RESONANCE_MIN_HZ,
    GOAL_SIZE_LIMITS_MM,
    TEMPERATURE_FAMILY_LIMITS_C,
    TEMPERATURE_TARGET_FAMILIES,
)
from tools.mft_goal_global_pareto import nondominated_ranks_2d


MANIFEST_SCHEMA = "mft-goal-local-trust-acquisition-manifest-v1"
CANDIDATE_SET_SCHEMA = "mft-goal-local-trust-candidate-set-v1"
AGGREGATE_SCHEMA = "mft-goal-20260726-global-pareto-v1"
OBSERVATION_SCHEMA = "mft-goal-postdeadline-standard-measured-observation-v1"
CAMPAIGN_ID = "mft-goal-20260726"

ANCHOR_PRIORITY = (12, 6, 1, 5)
PRIMARY_ANCHORS = frozenset((12, 6))
BACKUP_ANCHORS = frozenset((1, 5))
MAX_TRUST_RADIUS = 0.05
MAX_NEIGHBOR_POOL = 64
DEFAULT_MAX_BATCH = 3
DEFAULT_MAX_ROUNDS = 2
EXISTING_FULL_REFERENCE_TASK_ID = 96326
RESONANCE_CONSTRAINT_SCALE_HZ = 150.0

# This compact projection is used in plans and tests.  The complete fixed
# identity, including k_ins/core_k_thermal and on/off switches, is emitted
# separately as ``fixed_cooling_contract``.
FIXED_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}
_WINDING_TEMPERATURE_FAMILIES = (
    "primary_winding",
    "secondary_winding",
)

# Exact copy of the geometry identity used by the terminal-population
# exporter.  Keeping the same projection makes generated hashes directly
# comparable with current NSGA-II physical-geometry hashes.
GEOMETRY_IDENTITY_KEYS = (
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
    "core_plate_pad_t",
    "wcp_t",
    "wcp_pad_t",
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
    "w1c_w2s_gap_x_actual",
    "w2s_w1s_space_x",
    "w1s_w2s_space_y",
    "w1s_cs_space_x",
    "cs_w1s_space_y",
)

# Discrete/topology and cold-plate thickness coordinates remain frozen.  The
# active dimensions are geometry-only and use a deterministic coordinate
# stencil, never a global/random proposal.
FROZEN_UNIT_INDICES = frozenset((0, 1, 2, 7, 8, 9))
ACTIVE_UNIT_INDICES = tuple(
    index for index in range(len(_SOBOL_DIMS)) if index not in FROZEN_UNIT_INDICES
)


class LocalTrustContractError(RuntimeError):
    """Raised when local-acquisition evidence fails closed."""


class ResidualClass(Enum):
    SOLVER_OR_PHYSICS_INVALID = "solver_or_physics_invalid"
    SMALL_LOCAL_CORRECTION = "small_local_correction"
    LARGE_MODEL_MISMATCH = "large_model_mismatch"


@dataclass(frozen=True)
class ResidualThresholds:
    """Scales that define the finite local-correction trust decision."""

    volume_L: float = 20.0
    total_loss_W: float = 300.0
    resonance_frequency_kHz: float = 0.5
    winding_max_C: float = 3.0
    core_max_C: float = 3.0
    normalized_constraint: float = 0.25
    maximum_scaled_residual: float = 1.0
    correction_trigger: float = 0.05


@dataclass(frozen=True)
class ResidualDecision:
    classification: ResidualClass
    eligible_for_local_acquisition: bool
    reasons: tuple[str, ...]
    max_scaled_residual: float
    scaled_residuals: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class BatchSelection:
    selected: tuple[dict[str, Any], ...]
    should_stop: bool
    stop_reason: str | None


@dataclass(frozen=True)
class AnchorEvidence:
    selection_order: int
    role: str
    candidate_physics_sha256: str
    row: dict[str, Any]
    coordinate_unit: tuple[float, ...]
    physical_constraints: dict[str, float]
    normalized_constraints: dict[str, float]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalTrustContractError(
            "payload is not canonical finite JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    if "payload_sha256" in output:
        raise LocalTrustContractError("payload is already sealed")
    output["payload_sha256"] = canonical_sha256(output)
    return output


def validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    observed = output.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(observed, str)
        or observed != canonical_sha256(output)
    ):
        raise LocalTrustContractError("payload SHA256 seal is invalid")
    output["payload_sha256"] = observed
    return output


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise LocalTrustContractError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LocalTrustContractError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise LocalTrustContractError(f"{label} must be finite")
    return number


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def _flatten_measurement_numbers(
    value: Mapping[str, Any],
    *,
    side: str,
) -> dict[str, float]:
    container = value.get(side)
    if not isinstance(container, Mapping):
        raise LocalTrustContractError(f"{side} measurement is absent")
    objectives = container.get("objectives")
    metrics = container.get("metrics")
    constraints = container.get("normalized_constraints")
    if not all(
        isinstance(item, Mapping)
        for item in (objectives, metrics, constraints)
    ):
        raise LocalTrustContractError(
            f"{side} objectives/metrics/constraints are absent"
        )
    required_objectives = ("exterior_volume_L", "total_loss_W")
    required_metrics = (
        "resonance_frequency_kHz",
        "winding_max_C",
        "core_max_C",
    )
    output: dict[str, float] = {}
    for name in required_objectives:
        output[f"objectives.{name}"] = _finite(
            objectives.get(name), f"{side} {name}"
        )
    for name in required_metrics:
        output[f"metrics.{name}"] = _finite(
            metrics.get(name), f"{side} {name}"
        )
    if not constraints:
        raise LocalTrustContractError(
            f"{side} normalized constraints are empty"
        )
    for name, raw in constraints.items():
        output[f"normalized_constraints.{name}"] = _finite(
            raw, f"{side} constraint {name}"
        )
    return output


def _fixed_boundary_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return dict(value) == FIXED_BOUNDARY


def classify_authenticated_residual(
    measurement: Mapping[str, Any],
    *,
    expected_candidate_sha256: str | None = None,
    thresholds: ResidualThresholds | None = None,
) -> ResidualDecision:
    """Classify authenticated FEA residual without initiating any work."""

    limits = thresholds or ResidualThresholds()
    invalid_reasons: list[str] = []
    for field, label in (
        ("terminal_authenticated", "terminal evidence is not authenticated"),
        ("artifact_authenticated", "artifact evidence is not authenticated"),
        ("solver_valid", "solver result is invalid"),
        ("physics_valid", "physics result is invalid"),
    ):
        if measurement.get(field) is not True:
            invalid_reasons.append(label)
    candidate = measurement.get("candidate_physics_sha256")
    if not _is_sha256(candidate):
        invalid_reasons.append("candidate SHA256 is invalid")
    if (
        expected_candidate_sha256 is not None
        and candidate != expected_candidate_sha256
    ):
        invalid_reasons.append("candidate identity does not match anchor")
    if not _fixed_boundary_valid(measurement.get("fixed_boundary")):
        invalid_reasons.append("fixed cooling boundary does not match contract")

    predicted: dict[str, float] = {}
    actual: dict[str, float] = {}
    try:
        predicted = _flatten_measurement_numbers(
            measurement, side="predicted"
        )
        actual = _flatten_measurement_numbers(measurement, side="actual")
    except LocalTrustContractError as exc:
        invalid_reasons.append(str(exc))
    if predicted and actual and set(predicted) != set(actual):
        invalid_reasons.append(
            "predicted/actual metric schemas do not match"
        )
    if invalid_reasons:
        return ResidualDecision(
            classification=ResidualClass.SOLVER_OR_PHYSICS_INVALID,
            eligible_for_local_acquisition=False,
            reasons=tuple(invalid_reasons),
            # No numerical residual exists for an invalid solve.  Keep the
            # sealed representation finite and let the classification/reasons
            # carry that state.
            max_scaled_residual=0.0,
        )

    scale_by_name = {
        "objectives.exterior_volume_L": limits.volume_L,
        "objectives.total_loss_W": limits.total_loss_W,
        "metrics.resonance_frequency_kHz": (
            limits.resonance_frequency_kHz
        ),
        "metrics.winding_max_C": limits.winding_max_C,
        "metrics.core_max_C": limits.core_max_C,
    }
    scaled: list[tuple[str, float]] = []
    for name in sorted(predicted):
        scale = scale_by_name.get(name, limits.normalized_constraint)
        if not math.isfinite(scale) or scale <= 0.0:
            raise LocalTrustContractError(
                f"residual scale for {name} must be positive and finite"
            )
        scaled.append((name, abs(actual[name] - predicted[name]) / scale))
    maximum = max((value for _, value in scaled), default=0.0)
    if maximum > limits.maximum_scaled_residual:
        largest = max(scaled, key=lambda item: item[1])
        return ResidualDecision(
            classification=ResidualClass.LARGE_MODEL_MISMATCH,
            eligible_for_local_acquisition=False,
            reasons=(
                f"scaled residual {largest[0]}={largest[1]:.6g} "
                f"exceeds {limits.maximum_scaled_residual:g}",
            ),
            max_scaled_residual=maximum,
            scaled_residuals=tuple(scaled),
        )

    actual_constraints = {
        name: value
        for name, value in actual.items()
        if name.startswith("normalized_constraints.")
    }
    correction_needed = (
        any(value > 0.0 for value in actual_constraints.values())
        or maximum >= limits.correction_trigger
    )
    return ResidualDecision(
        classification=ResidualClass.SMALL_LOCAL_CORRECTION,
        eligible_for_local_acquisition=correction_needed,
        reasons=()
        if correction_needed
        else ("residual is below the local-correction trigger",),
        max_scaled_residual=maximum,
        scaled_residuals=tuple(scaled),
    )


def _topology(decoded: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(decoded["N1_main"]),
        int(decoded["N1_side"]),
        int(decoded["N2_main"]) + int(decoded["N2_side"]),
        int(decoded["n_core_group"]),
    )


def _apply_symmetric_fixed_boundary(
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(decoded))
    output.update(
        {
            "fan_config": "dual",
            "fan_velocity": 1.5,
            "core_plate_on": 1,
            "wcp_on": 1,
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
            "k_ins": 0.2,
            "core_k_thermal": 2.0,
            "full_model": 0,
            "loss_sym_on": 1,
            "thermal_symmetry": "eighth",
        }
    )
    return output


def _geometry_sha(decoded: Mapping[str, Any]) -> str:
    identity = {
        name: decoded.get(name) for name in GEOMETRY_IDENTITY_KEYS
    }
    return canonical_sha256(identity)


def _candidate_coordinate_stencil(
    anchor: np.ndarray,
    radius: float,
) -> Iterable[np.ndarray]:
    for index in ACTIVE_UNIT_INDICES:
        for sign in (-1.0, 1.0):
            candidate = anchor.copy()
            candidate[index] = np.clip(
                candidate[index] + sign * radius, 0.0, 1.0
            )
            yield candidate

    # Paired directions give the local linear acquisition a few trade-off
    # choices after the axis stencil.  They remain inside the same L-infinity
    # trust region.
    for offset, first in enumerate(ACTIVE_UNIT_INDICES):
        second = ACTIVE_UNIT_INDICES[(offset + 1) % len(ACTIVE_UNIT_INDICES)]
        for first_sign, second_sign in (
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        ):
            candidate = anchor.copy()
            candidate[first] = np.clip(
                candidate[first] + first_sign * radius, 0.0, 1.0
            )
            candidate[second] = np.clip(
                candidate[second] + second_sign * radius, 0.0, 1.0
            )
            yield candidate


def generate_local_neighbors(
    anchor_unit: Sequence[float],
    *,
    selection_order: int,
    trust_radius: float,
    max_pool: int,
) -> list[dict[str, Any]]:
    """Generate a deterministic, topology-preserving local geometry stencil."""

    if selection_order not in ANCHOR_PRIORITY:
        raise LocalTrustContractError(
            "local anchors are restricted to #12, #6, #1, and #5"
        )
    radius = _finite(trust_radius, "trust radius")
    if radius <= 0.0 or radius > MAX_TRUST_RADIUS:
        raise LocalTrustContractError(
            f"trust radius must be in (0, {MAX_TRUST_RADIUS:g}]"
        )
    if (
        isinstance(max_pool, bool)
        or not isinstance(max_pool, int)
        or max_pool <= 0
        or max_pool > MAX_NEIGHBOR_POOL
    ):
        raise LocalTrustContractError(
            f"neighbor pool must be between 1 and {MAX_NEIGHBOR_POOL}"
        )
    anchor = np.asarray(anchor_unit, dtype=float)
    if (
        anchor.shape != (len(_SOBOL_DIMS),)
        or not np.isfinite(anchor).all()
        or np.any(anchor < 0.0)
        or np.any(anchor > 1.0)
    ):
        raise LocalTrustContractError(
            f"anchor unit coordinate must be finite [0,1]^{len(_SOBOL_DIMS)}"
        )
    raw_anchor = decode_unit_sample(
        unit_to_dims(anchor), allow_space_shrink=False
    )
    anchor_topology = _topology(raw_anchor)
    anchor_geometry_sha = _geometry_sha(
        _apply_symmetric_fixed_boundary(raw_anchor)
    )
    role = "primary" if selection_order in PRIMARY_ANCHORS else "backup"
    seen_coordinates: set[str] = set()
    seen_geometries = {anchor_geometry_sha}
    output: list[dict[str, Any]] = []
    for candidate in _candidate_coordinate_stencil(anchor, radius):
        distance = float(np.max(np.abs(candidate - anchor)))
        if distance <= 0.0 or distance > radius + 1e-12:
            continue
        coordinate_key = canonical_sha256(candidate.tolist())
        if coordinate_key in seen_coordinates:
            continue
        seen_coordinates.add(coordinate_key)
        raw_decoded = decode_unit_sample(
            unit_to_dims(candidate), allow_space_shrink=False
        )
        if _topology(raw_decoded) != anchor_topology:
            continue
        decoded = _apply_symmetric_fixed_boundary(raw_decoded)
        geometry_sha = _geometry_sha(decoded)
        if geometry_sha in seen_geometries:
            continue
        seen_geometries.add(geometry_sha)
        output.append(
            {
                "anchor_selection_order": selection_order,
                "anchor_role": role,
                "unit_coordinate": candidate.tolist(),
                "distance_linf": distance,
                "geometry_sha256": geometry_sha,
                "canonical_physical_params_sha256": canonical_sha256(
                    decoded
                ),
                "decoded_params": decoded,
                "stage": "symmetric_standard",
                "prepare_only": True,
            }
        )
        if len(output) >= max_pool:
            break
    if not output:
        raise LocalTrustContractError(
            "local stencil produced no distinct topology-preserving geometry"
        )
    return output


def select_finite_batch(
    scored: Sequence[Mapping[str, Any]],
    *,
    current_round: int,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_batch: int = DEFAULT_MAX_BATCH,
    min_expected_constraint_improvement: float = 0.02,
) -> BatchSelection:
    """Select a finite score-ordered batch or return an explicit stop."""

    for value, label in (
        (current_round, "current round"),
        (max_rounds, "maximum rounds"),
        (max_batch, "maximum batch"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise LocalTrustContractError(f"{label} must be an integer")
    if current_round < 0 or max_rounds <= 0:
        raise LocalTrustContractError("correction rounds are invalid")
    if max_batch <= 0 or max_batch > DEFAULT_MAX_BATCH:
        raise LocalTrustContractError(
            f"maximum batch cannot exceed {DEFAULT_MAX_BATCH}"
        )
    minimum = _finite(
        min_expected_constraint_improvement,
        "minimum expected constraint improvement",
    )
    if minimum < 0.0:
        raise LocalTrustContractError(
            "minimum expected constraint improvement cannot be negative"
        )
    if current_round >= max_rounds:
        return BatchSelection(
            selected=(),
            should_stop=True,
            stop_reason="max_correction_rounds_reached",
        )

    eligible: list[dict[str, Any]] = []
    for item in scored:
        candidate = copy.deepcopy(dict(item))
        improvement = _finite(
            candidate.get("expected_constraint_improvement"),
            "expected constraint improvement",
        )
        _finite(candidate.get("pareto_value"), "Pareto value")
        score = _finite(
            candidate.get("acquisition_score"), "acquisition score"
        )
        if improvement >= minimum:
            candidate["_sort_score"] = score
            eligible.append(candidate)
    if not eligible:
        return BatchSelection(
            selected=(),
            should_stop=True,
            stop_reason="no_expected_constraint_improvement",
        )
    eligible.sort(
        key=lambda item: (
            -item["_sort_score"],
            -float(item["expected_constraint_improvement"]),
            str(item.get("geometry_sha256") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    for item in eligible[:max_batch]:
        item.pop("_sort_score", None)
        selected.append(item)
    return BatchSelection(
        selected=tuple(selected),
        should_stop=False,
        stop_reason=None,
    )


def exact_global_nds_ranks(objectives: Any) -> np.ndarray:
    """Exact all-front two-objective NDS hook used after measurements."""

    return nondominated_ranks_2d(np.asarray(objectives, dtype=float))


def build_prepare_only_manifest(
    *,
    anchor_selection_order: int,
    anchor_candidate_sha256: str,
    residual_decision: ResidualDecision,
    batch: BatchSelection,
    trust_radius: float,
    current_round: int,
    max_rounds: int,
    max_batch: int,
    min_expected_constraint_improvement: float,
    anchor_table: Sequence[Mapping[str, Any]] = (),
    measured_selection: Mapping[str, Any] | None = None,
    authentication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable no-submit handoff for a tiny symmetric batch."""

    if anchor_selection_order not in ANCHOR_PRIORITY:
        raise LocalTrustContractError("manifest anchor priority is invalid")
    if not _is_sha256(anchor_candidate_sha256):
        raise LocalTrustContractError("manifest anchor SHA256 is invalid")
    if len(batch.selected) > max_batch or max_batch > DEFAULT_MAX_BATCH:
        raise LocalTrustContractError("manifest candidate batch exceeds cap")
    candidates = [copy.deepcopy(item) for item in batch.selected]
    for candidate in candidates:
        decoded = candidate.get("decoded_params")
        if (
            not isinstance(decoded, Mapping)
            or decoded.get("full_model") != 0
            or decoded.get("thermal_symmetry") != "eighth"
        ):
            raise LocalTrustContractError(
                "manifest candidates must be symmetric-only"
            )
    return seal(
        {
            "schema_version": MANIFEST_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "prepare_only": True,
            "stage": "symmetric_standard",
            "symmetric_model_primary": True,
            "scheduler_post_calls": 0,
            "scheduler_submission_performed": False,
            "submission_capability_present": False,
            "scheduler_methods_used": [],
            "anchor_priority": list(ANCHOR_PRIORITY),
            "anchor_selection_order": anchor_selection_order,
            "anchor_role": (
                "primary"
                if anchor_selection_order in PRIMARY_ANCHORS
                else "backup"
            ),
            "anchor_candidate_sha256": anchor_candidate_sha256,
            "anchor_table": [
                copy.deepcopy(dict(item)) for item in anchor_table
            ],
            "measured_selection": copy.deepcopy(
                dict(measured_selection or {})
            ),
            "measured_selection_stop_policy": {
                "gate": (
                    "dimensions_and_resonance_and_winding_and_core"
                ),
                "if_any_passes": (
                    "stop_and_select_by_actual_minimum_normalized_margin_"
                    "then_loss_then_volume"
                ),
                "if_none_passes": (
                    "allow_tiny_local_symmetric_batch_only_for_small_"
                    "authenticated_residual"
                ),
            },
            "residual_decision": {
                **asdict(residual_decision),
                "classification": residual_decision.classification.value,
            },
            "trust_radius": _finite(trust_radius, "trust radius"),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "batch_stop": {
                "should_stop": batch.should_stop,
                "reason": batch.stop_reason,
            },
            "finite_stop_criteria": {
                "current_round": current_round,
                "max_correction_rounds": max_rounds,
                "max_parallel_fea_batch": max_batch,
                "min_expected_constraint_improvement": _finite(
                    min_expected_constraint_improvement,
                    "minimum expected constraint improvement",
                ),
            },
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "fixed_cooling_contract": copy.deepcopy(
                FIXED_COOLING_IDENTITY
            ),
            "full_model_policy": "final_one_only",
            "automatic_full_per_candidate": False,
            "automatic_candidate_continuation": False,
            "automatic_full_trigger": False,
            "explicit_final_full_cap": 1,
            "existing_full_reference_task_id": (
                EXISTING_FULL_REFERENCE_TASK_ID
            ),
            "exact_global_nds_reevaluation_hook": {
                "callable": (
                    "tools.mft_goal_global_pareto."
                    "nondominated_ranks_2d"
                ),
                "required_after_authenticated_measurements": True,
                "status": "pending",
            },
            "authentication": copy.deepcopy(dict(authentication or {})),
        }
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalTrustContractError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise LocalTrustContractError(f"{label} must be a JSON object")
    return value


def _validate_generic_seal(
    value: Mapping[str, Any], label: str
) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    observed = output.pop("payload_sha256", None)
    if (
        not isinstance(observed, str)
        or observed != canonical_sha256(output)
    ):
        raise LocalTrustContractError(f"{label} payload seal drifted")
    output["payload_sha256"] = observed
    return output


def _file_record(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    if not target.is_file() or target.is_symlink():
        raise LocalTrustContractError(f"not a regular file: {target}")
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _verify_artifact(
    *,
    root: Path,
    record: Any,
    label: str,
) -> Path:
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not _is_sha256(record.get("sha256"))
    ):
        raise LocalTrustContractError(f"{label} artifact record is invalid")
    raw = Path(str(record["path"]))
    target = raw if raw.is_absolute() else root / raw
    target = target.resolve(strict=True)
    if (
        not target.is_file()
        or target.is_symlink()
        or sha256_file(target) != record["sha256"]
    ):
        raise LocalTrustContractError(f"{label} artifact bytes drifted")
    if (
        "size_bytes" in record
        and (
            isinstance(record["size_bytes"], bool)
            or int(record["size_bytes"]) != target.stat().st_size
        )
    ):
        raise LocalTrustContractError(f"{label} artifact size drifted")
    return target


def _canonical_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise LocalTrustContractError(f"{label} is not a canonical boolean")


def _parse_json_mapping(value: Any, label: str) -> dict[str, float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise LocalTrustContractError(f"{label} is malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise LocalTrustContractError(f"{label} must encode an object")
    return {
        str(name): _finite(raw, f"{label}.{name}")
        for name, raw in parsed.items()
    }


def _parse_coordinate(value: Any, label: str) -> tuple[float, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise LocalTrustContractError(f"{label} is malformed JSON") from exc
    if not isinstance(parsed, list) or len(parsed) != len(_SOBOL_DIMS):
        raise LocalTrustContractError(
            f"{label} must contain {len(_SOBOL_DIMS)} coordinates"
        )
    coordinate = tuple(_finite(item, label) for item in parsed)
    if any(item < 0.0 or item > 1.0 for item in coordinate):
        raise LocalTrustContractError(f"{label} leaves [0,1]")
    return coordinate


def _anchor_from_row(row: Mapping[str, Any]) -> AnchorEvidence:
    try:
        order = int(row["standard_selection_order"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalTrustContractError(
            "standard candidate selection order is invalid"
        ) from exc
    if order not in ANCHOR_PRIORITY:
        raise LocalTrustContractError(
            f"standard candidate #{order} is outside local anchor policy"
        )
    candidate_sha = str(row.get("candidate_physics_sha") or "")
    if (
        not _is_sha256(candidate_sha)
        or row.get("physical_geometry_sha256") != candidate_sha
    ):
        raise LocalTrustContractError(
            f"standard candidate #{order} geometry identity drifted"
        )
    for name in (
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
    ):
        if not _canonical_bool(row.get(name), f"candidate #{order}.{name}"):
            raise LocalTrustContractError(
                f"standard candidate #{order} is not decoder/physics valid"
            )
    return AnchorEvidence(
        selection_order=order,
        role="primary" if order in PRIMARY_ANCHORS else "backup",
        candidate_physics_sha256=candidate_sha,
        row=copy.deepcopy(dict(row)),
        coordinate_unit=_parse_coordinate(
            row.get("coordinate_unit_json"),
            f"candidate #{order}.coordinate_unit_json",
        ),
        physical_constraints=_parse_json_mapping(
            row.get("physical_G_json"),
            f"candidate #{order}.physical_G_json",
        ),
        normalized_constraints=_parse_json_mapping(
            row.get("normalized_G_json"),
            f"candidate #{order}.normalized_G_json",
        ),
    )


def load_authenticated_anchor_set(
    aggregate_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[AnchorEvidence, ...],
    Path,
    Path,
]:
    """Authenticate the aggregate and return the four bounded anchor rows."""

    manifest_path = aggregate_manifest_path.resolve(strict=True)
    manifest = validate_seal(
        _read_json(manifest_path, "aggregate manifest"),
        AGGREGATE_SCHEMA,
    )
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("seed_count") != 512
        or manifest.get("standard_candidate_count") != 12
        or manifest.get("input_terminal_row_count") != 163_840
        or manifest.get("search_only_proposal") is not True
        or manifest.get("production_eligible") is not False
        or manifest.get("automatic_promotion_allowed") is not False
        or manifest.get("seed_local_pareto_merge_used") is not False
    ):
        raise LocalTrustContractError(
            "aggregate campaign/count/safety authority drifted"
        )
    hard_cooling = (
        manifest.get("hard_spec", {}).get("fixed_cooling_identity")
    )
    if hard_cooling != FIXED_COOLING_IDENTITY:
        raise LocalTrustContractError(
            "aggregate fixed cooling contract drifted"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LocalTrustContractError("aggregate artifact inventory is absent")
    standard_path = _verify_artifact(
        root=manifest_path.parent,
        record=artifacts.get("standard_candidates"),
        label="standard candidates",
    )
    objective_front_path = _verify_artifact(
        root=manifest_path.parent,
        record=artifacts.get("global_objective_front"),
        label="global objective front",
    )
    standard = pd.read_csv(standard_path, dtype=object)
    if len(standard) != 12:
        raise LocalTrustContractError(
            "official Standard candidate table must contain 12 rows"
        )
    anchors: list[AnchorEvidence] = []
    for order in ANCHOR_PRIORITY:
        rows = standard[
            pd.to_numeric(
                standard["standard_selection_order"], errors="coerce"
            )
            == order
        ]
        if len(rows) != 1:
            raise LocalTrustContractError(
                f"official Standard candidate #{order} is not unique"
            )
        anchors.append(_anchor_from_row(rows.iloc[0].to_dict()))
    return manifest, tuple(anchors), standard_path, objective_front_path


def _source_training_table(
    anchor: AnchorEvidence,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row = anchor.row
    source_result_path = Path(
        str(row.get("source_result_path") or "")
    ).resolve(strict=True)
    expected_result_sha = str(row.get("source_result_sha256") or "")
    if (
        not _is_sha256(expected_result_sha)
        or sha256_file(source_result_path) != expected_result_sha
    ):
        raise LocalTrustContractError(
            "anchor source result file identity drifted"
        )
    result = _validate_generic_seal(
        _read_json(source_result_path, "anchor source result"),
        "anchor source result",
    )
    source_identity = result.get("source_identity")
    if (
        result.get("search_only_proposal") is not True
        or result.get("production_eligible") is not False
        or result.get("automatic_promotion_allowed") is not False
        or not isinstance(source_identity, Mapping)
        or str(source_identity.get("task_id"))
        != str(row.get("source_task_id"))
    ):
        raise LocalTrustContractError(
            "anchor source result scientific identity drifted"
        )
    inventory = result.get("artifact_inventory")
    terminal_record = (
        inventory.get("terminal_physical_candidates")
        if isinstance(inventory, Mapping)
        else None
    )
    terminal_path = _verify_artifact(
        root=source_result_path.parent,
        record=terminal_record,
        label="source terminal candidates",
    )
    terminal = pd.read_csv(terminal_path)
    if len(terminal) != 320:
        raise LocalTrustContractError(
            "source terminal population is not exactly 320 rows"
        )
    candidate_rows = terminal[
        terminal["candidate_physics_sha"].astype(str)
        == anchor.candidate_physics_sha256
    ]
    if len(candidate_rows) != 1:
        raise LocalTrustContractError(
            "anchor is not unique in authenticated source population"
        )
    source_anchor = candidate_rows.iloc[0]
    exact_values = {
        "terminal_population_index": int(
            anchor.row["terminal_population_index"]
        ),
        "candidate_physics_sha": anchor.candidate_physics_sha256,
    }
    if (
        int(source_anchor["terminal_population_index"])
        != exact_values["terminal_population_index"]
        or str(source_anchor["candidate_physics_sha"])
        != exact_values["candidate_physics_sha"]
        or not np.isclose(
            float(source_anchor["objective_volume_L"]),
            float(anchor.row["objective_volume_L"]),
            rtol=0.0,
            atol=1e-10,
        )
        or not np.isclose(
            float(source_anchor["objective_total_loss_W"]),
            float(anchor.row["objective_total_loss_W"]),
            rtol=0.0,
            atol=1e-10,
        )
    ):
        raise LocalTrustContractError(
            "official anchor differs from authenticated source row"
        )
    return terminal, {
        "source_result": _file_record(source_result_path),
        "source_result_payload_sha256": result["payload_sha256"],
        "source_terminal_candidates": _file_record(terminal_path),
        "source_terminal_population_index": exact_values[
            "terminal_population_index"
        ],
    }


def _predicted_family_max(
    anchor: AnchorEvidence,
    family: str,
) -> float:
    values: list[float] = []
    prefix = "temperature_robust_limit:"
    for constraint, raw in anchor.physical_constraints.items():
        if not constraint.startswith(prefix):
            continue
        target = constraint[len(prefix) :]
        target_family = TEMPERATURE_TARGET_FAMILIES.get(target)
        if family == "winding":
            if target_family not in _WINDING_TEMPERATURE_FAMILIES:
                continue
        elif target_family != family:
            continue
        values.append(
            float(TEMPERATURE_FAMILY_LIMITS_C[target_family])
            + float(raw)
        )
    if not values:
        raise LocalTrustContractError(
            f"anchor lacks {family} temperature predictions"
        )
    return max(values)


def _actual_temperature_value(value: Any, label: str) -> float:
    if isinstance(value, Mapping):
        for key in ("actual_C", "actual", "value_C"):
            if key in value:
                return _finite(value[key], label)
    return _finite(value, label)


def _observation_to_measurement(
    observation: Mapping[str, Any],
    anchor: AnchorEvidence,
) -> dict[str, Any]:
    """Translate a reauthenticated postsuccess observation for residual use."""

    predicted_constraints = copy.deepcopy(anchor.normalized_constraints)
    actual_constraints = copy.deepcopy(predicted_constraints)
    resonance_actual_hz = _finite(
        observation.get("actual_resonance_Hz"), "actual resonance"
    )
    resonance_actual_khz = resonance_actual_hz / 1000.0
    resonance_name = "half_magnetizing_resonance_minimum"
    actual_constraints[resonance_name] = (
        GOAL_RESONANCE_MIN_HZ - resonance_actual_hz
    ) / RESONANCE_CONSTRAINT_SCALE_HZ

    actual_targets = observation.get("actual_temperature_targets")
    if not isinstance(actual_targets, Mapping):
        raise LocalTrustContractError(
            "observation temperature target evidence is absent"
        )
    for target, evidence in actual_targets.items():
        family = TEMPERATURE_TARGET_FAMILIES.get(str(target))
        if family is None:
            continue
        value = _actual_temperature_value(
            evidence, f"actual temperature {target}"
        )
        actual_constraints[
            f"temperature_robust_limit:{target}"
        ] = (
            value - float(TEMPERATURE_FAMILY_LIMITS_C[family])
        ) / 10.0

    dimensions = observation.get("actual_dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise LocalTrustContractError(
            "observation actual dimensions are absent"
        )
    for name, axis in (
        ("exterior_width_limit", "W"),
        ("exterior_length_limit", "L"),
        ("exterior_height_limit", "H"),
    ):
        actual_constraints[name] = _finite(
            dimensions.get(axis), f"actual dimension {axis}"
        ) - float(GOAL_SIZE_LIMITS_MM[axis])

    result_record = observation.get("source_result_json")
    if not isinstance(result_record, Mapping):
        raise LocalTrustContractError(
            "observation source result record is absent"
        )
    result_path = Path(str(result_record.get("path") or "")).resolve(
        strict=True
    )
    if _file_record(result_path) != dict(result_record):
        raise LocalTrustContractError(
            "observation source result bytes drifted"
        )
    result = _read_json(result_path, "observed Standard result")
    actual_physical_llt = 2.0 * _finite(
        result.get("Llt"), "actual symmetric Llt"
    )
    actual_constraints["Llt_robust_band"] = (
        abs(actual_physical_llt - 27.5) - 0.55
    ) / 0.55

    resonance_predicted_khz = (
        GOAL_RESONANCE_MIN_HZ
        - float(anchor.physical_constraints[resonance_name])
    ) / 1000.0
    return {
        "terminal_authenticated": True,
        "artifact_authenticated": True,
        "solver_valid": True,
        "physics_valid": True,
        "candidate_physics_sha256": anchor.candidate_physics_sha256,
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "predicted": {
            "objectives": {
                "exterior_volume_L": float(
                    anchor.row["objective_volume_L"]
                ),
                "total_loss_W": float(
                    anchor.row["objective_total_loss_W"]
                ),
            },
            "metrics": {
                "resonance_frequency_kHz": resonance_predicted_khz,
                "winding_max_C": _predicted_family_max(
                    anchor, "winding"
                ),
                "core_max_C": _predicted_family_max(anchor, "core"),
            },
            "normalized_constraints": predicted_constraints,
        },
        "actual": {
            "objectives": {
                "exterior_volume_L": _finite(
                    observation.get("actual_volume_L"),
                    "actual volume",
                ),
                "total_loss_W": _finite(
                    observation.get("actual_total_loss_W"),
                    "actual total loss",
                ),
            },
            "metrics": {
                "resonance_frequency_kHz": resonance_actual_khz,
                "winding_max_C": _finite(
                    observation.get("actual_winding_max_C"),
                    "actual winding maximum",
                ),
                "core_max_C": _finite(
                    observation.get("actual_core_max_C"),
                    "actual core maximum",
                ),
            },
            "normalized_constraints": actual_constraints,
        },
    }


def load_authenticated_observation(
    path: Path,
    anchors: Sequence[AnchorEvidence],
) -> tuple[dict[str, Any], AnchorEvidence, dict[str, Any], ResidualDecision]:
    """Reauthenticate a postsuccess observation and derive its residual."""

    # The postsuccess bridge owns the strong collection/AEDT authentication.
    # Reuse it instead of weakening that lineage in this separate acquisition
    # tool.
    from tools import (  # noqa: PLC0415
        mft_goal_postdeadline_standard_postsuccess as postsuccess,
    )

    observation_path = path.resolve(strict=True)
    observation = postsuccess._load_observation(  # noqa: SLF001
        observation_path
    )
    collection_record = observation.get("source_collection_receipt")
    if not isinstance(collection_record, Mapping):
        raise LocalTrustContractError(
            "observation collection lineage is absent"
        )
    collection_path = Path(
        str(collection_record.get("path") or "")
    ).resolve(strict=True)
    try:
        view = postsuccess.authenticate_collection(collection_path)
        measured = postsuccess._measured_classification(view)  # noqa: SLF001
    except Exception as exc:
        raise LocalTrustContractError(
            "Standard collection reauthentication failed"
        ) from exc
    comparison_fields = (
        "task_id",
        "candidate_physics_sha256",
        "actual_volume_L",
        "actual_total_loss_W",
        "actual_dimensions_mm",
        "actual_resonance_Hz",
        "actual_winding_max_C",
        "actual_core_max_C",
        "hard_constraint_evidence",
        "measured_hard_constraints_passed",
    )
    if any(
        observation.get(name) != measured.get(name)
        for name in comparison_fields
    ):
        raise LocalTrustContractError(
            "observation differs from reauthenticated Standard result"
        )
    candidate_sha = str(
        observation.get("candidate_physics_sha256") or ""
    )
    matching = [
        anchor
        for anchor in anchors
        if anchor.candidate_physics_sha256 == candidate_sha
    ]
    if len(matching) != 1:
        raise LocalTrustContractError(
            "observation is not one of the bounded official anchors"
        )
    anchor = matching[0]
    measurement = _observation_to_measurement(observation, anchor)
    decision = classify_authenticated_residual(
        measurement,
        expected_candidate_sha256=anchor.candidate_physics_sha256,
    )
    return observation, anchor, measurement, decision


def _minimum_normalized_actual_margin(
    observation: Mapping[str, Any],
) -> float:
    evidence = observation.get("hard_constraint_evidence")
    if not isinstance(evidence, Mapping):
        raise LocalTrustContractError(
            "observation hard-constraint evidence is absent"
        )
    scales = {
        "width_mm": float(GOAL_SIZE_LIMITS_MM["W"]),
        "length_mm": float(GOAL_SIZE_LIMITS_MM["L"]),
        "height_mm": float(GOAL_SIZE_LIMITS_MM["H"]),
        "resonance_Hz": 15_000.0,
        # The legacy observation schema exposes one aggregate winding field.
        # Normalize it against the stricter member of the split Tx/Rx
        # contract instead of inventing a removed "winding" family alias.
        "winding_max_C": min(
            float(TEMPERATURE_FAMILY_LIMITS_C[family])
            for family in _WINDING_TEMPERATURE_FAMILIES
        ),
        "core_max_C": float(TEMPERATURE_FAMILY_LIMITS_C["core"]),
    }
    margins: list[float] = []
    for name, scale in scales.items():
        item = evidence.get(name)
        if not isinstance(item, Mapping):
            raise LocalTrustContractError(
                f"hard-constraint evidence {name} is absent"
            )
        margins.append(_finite(item.get("margin"), f"{name} margin") / scale)
    return min(margins)


def build_anchor_table(
    anchors: Sequence[AnchorEvidence],
    measured: Sequence[
        tuple[dict[str, Any], AnchorEvidence, dict[str, Any], ResidualDecision]
    ],
) -> list[dict[str, Any]]:
    by_order = {item[1].selection_order: item for item in measured}
    output: list[dict[str, Any]] = []
    for anchor in sorted(
        anchors, key=lambda value: ANCHOR_PRIORITY.index(value.selection_order)
    ):
        item = by_order.get(anchor.selection_order)
        row: dict[str, Any] = {
            "priority_index": ANCHOR_PRIORITY.index(
                anchor.selection_order
            ),
            "selection_order": anchor.selection_order,
            "role": anchor.role,
            "candidate_physics_sha256": (
                anchor.candidate_physics_sha256
            ),
            "surrogate_volume_L": float(
                anchor.row["objective_volume_L"]
            ),
            "surrogate_total_loss_W": float(
                anchor.row["objective_total_loss_W"]
            ),
            "measurement_status": "unmeasured",
        }
        if item is not None:
            observation, _anchor, _measurement, decision = item
            row.update(
                {
                    "measurement_status": "authenticated_symmetric_standard",
                    "task_id": int(observation["task_id"]),
                    "actual_volume_L": float(
                        observation["actual_volume_L"]
                    ),
                    "actual_total_loss_W": float(
                        observation["actual_total_loss_W"]
                    ),
                    "actual_dimensions_mm": copy.deepcopy(
                        observation["actual_dimensions_mm"]
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
                    "hard_constraints_passed": bool(
                        observation["measured_hard_constraints_passed"]
                    ),
                    "minimum_normalized_actual_margin": (
                        _minimum_normalized_actual_margin(observation)
                    ),
                    "residual_classification": (
                        decision.classification.value
                    ),
                    "eligible_for_local_acquisition": (
                        decision.eligible_for_local_acquisition
                    ),
                }
            )
        output.append(row)
    return output


def select_current_symmetric_result(
    anchor_table: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select an actual passing Standard result, never a surrogate winner."""

    passing = [
        copy.deepcopy(dict(item))
        for item in anchor_table
        if item.get("measurement_status")
        == "authenticated_symmetric_standard"
        and item.get("hard_constraints_passed") is True
    ]
    if not passing:
        return None
    passing.sort(
        key=lambda item: (
            -_finite(
                item.get("minimum_normalized_actual_margin"),
                "minimum normalized actual margin",
            ),
            _finite(item.get("actual_total_loss_W"), "actual total loss"),
            _finite(item.get("actual_volume_L"), "actual volume"),
            int(item["priority_index"]),
        )
    )
    winner = passing[0]
    return {
        "status": "selected_existing_symmetric_result",
        "new_local_neighbors_required": False,
        "selection_order": int(winner["selection_order"]),
        "candidate_physics_sha256": winner[
            "candidate_physics_sha256"
        ],
        "task_id": int(winner["task_id"]),
        "ranking": (
            "actual_minimum_normalized_constraint_margin_desc,"
            "actual_total_loss_W_asc,actual_volume_L_asc,"
            "anchor_priority"
        ),
        "minimum_normalized_actual_margin": float(
            winner["minimum_normalized_actual_margin"]
        ),
        "actual_total_loss_W": float(winner["actual_total_loss_W"]),
        "actual_volume_L": float(winner["actual_volume_L"]),
    }


def _local_linear_model(
    training: pd.DataFrame,
    anchor: AnchorEvidence,
    *,
    nearest_count: int = 64,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    valid = training.copy()
    for name in (
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
    ):
        if name not in valid:
            raise LocalTrustContractError(
                f"source training table lacks {name}"
            )
        valid = valid[
            valid[name].map(
                lambda value: _canonical_bool(value, f"training {name}")
            )
        ]
    coordinates = np.vstack(
        [
            np.asarray(
                _parse_coordinate(value, "training coordinate_unit_json"),
                dtype=float,
            )
            for value in valid["coordinate_unit_json"]
        ]
    )
    anchor_coordinate = np.asarray(anchor.coordinate_unit, dtype=float)
    active_delta = (
        coordinates[:, ACTIVE_UNIT_INDICES]
        - anchor_coordinate[list(ACTIVE_UNIT_INDICES)]
    )
    distances = np.linalg.norm(active_delta, axis=1)
    count = min(nearest_count, len(valid))
    if count < 10:
        raise LocalTrustContractError(
            "too few authenticated local training rows"
        )
    nearest = np.argsort(distances, kind="stable")[:count]
    constraint_names = tuple(sorted(anchor.normalized_constraints))
    output_names = (
        "objective_volume_L",
        "objective_total_loss_W",
        *(f"normalized_G:{name}" for name in constraint_names),
    )
    missing = sorted(set(output_names).difference(valid.columns))
    if missing:
        raise LocalTrustContractError(
            "source training table lacks local outputs: "
            + ",".join(missing)
        )
    outputs = valid[list(output_names)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(outputs).all():
        raise LocalTrustContractError(
            "source training outputs contain non-finite values"
        )
    anchor_output = np.array(
        [
            float(anchor.row["objective_volume_L"]),
            float(anchor.row["objective_total_loss_W"]),
            *(
                float(anchor.normalized_constraints[name])
                for name in constraint_names
            ),
        ],
        dtype=float,
    )
    x = active_delta[nearest]
    y = outputs[nearest] - anchor_output
    support_radius = max(
        float(distances[nearest[-1]]),
        0.02,
    )
    weights = np.exp(
        -0.5
        * (
            np.linalg.norm(x, axis=1)
            / max(support_radius, np.finfo(float).eps)
        )
        ** 2
    )
    gram = x.T @ (weights[:, None] * x)
    ridge = 1e-4 * np.eye(gram.shape[0])
    rhs = x.T @ (weights[:, None] * y)
    try:
        coefficients = np.linalg.solve(gram + ridge, rhs)
    except np.linalg.LinAlgError as exc:
        raise LocalTrustContractError(
            "local delta-linear model is singular"
        ) from exc
    return coefficients, anchor_output, output_names, {
        "method": "anchor_centered_weighted_ridge_delta_linear",
        "support_row_count": count,
        "support_radius_l2_unit": support_radius,
        "ridge": 1e-4,
        "active_unit_indices": list(ACTIVE_UNIT_INDICES),
    }


def _measurement_delta_vector(
    measurement: Mapping[str, Any],
    output_names: Sequence[str],
) -> np.ndarray:
    predicted = measurement["predicted"]
    actual = measurement["actual"]
    values: list[float] = []
    for name in output_names:
        if name == "objective_volume_L":
            values.append(
                float(actual["objectives"]["exterior_volume_L"])
                - float(predicted["objectives"]["exterior_volume_L"])
            )
        elif name == "objective_total_loss_W":
            values.append(
                float(actual["objectives"]["total_loss_W"])
                - float(predicted["objectives"]["total_loss_W"])
            )
        else:
            constraint = name.removeprefix("normalized_G:")
            values.append(
                float(actual["normalized_constraints"][constraint])
                - float(predicted["normalized_constraints"][constraint])
            )
    return np.asarray(values, dtype=float)


def _objective_front_values(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = ("objective_volume_L", "objective_total_loss_W")
    if any(name not in frame for name in columns):
        raise LocalTrustContractError(
            "global objective front lacks objective columns"
        )
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise LocalTrustContractError(
            "global objective front contains non-finite values"
        )
    return values


def score_local_neighbors(
    neighbors: Sequence[Mapping[str, Any]],
    *,
    training: pd.DataFrame,
    anchor: AnchorEvidence,
    measurement: Mapping[str, Any],
    trust_radius: float,
    global_objective_front: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply local surrogate slopes plus a decaying measured delta."""

    coefficients, anchor_output, names, support = _local_linear_model(
        training, anchor
    )
    delta = _measurement_delta_vector(measurement, names)
    anchor_corrected = anchor_output + delta
    constraint_start = 2
    anchor_violation = float(
        np.maximum(anchor_corrected[constraint_start:], 0.0).sum()
    )
    coordinates = np.vstack(
        [np.asarray(item["unit_coordinate"], dtype=float) for item in neighbors]
    )
    active_delta = (
        coordinates[:, ACTIVE_UNIT_INDICES]
        - np.asarray(anchor.coordinate_unit, dtype=float)[
            list(ACTIVE_UNIT_INDICES)
        ]
    )
    base = anchor_output + active_delta @ coefficients
    distances = np.max(
        np.abs(
            coordinates - np.asarray(anchor.coordinate_unit, dtype=float)
        ),
        axis=1,
    )
    decay = np.exp(
        -0.5 * (distances / max(float(trust_radius), 1e-12)) ** 2
    )
    corrected = base + decay[:, None] * delta
    objective_values = corrected[:, :2]
    combined = (
        np.vstack((global_objective_front, objective_values))
        if len(global_objective_front)
        else objective_values
    )
    ranks = exact_global_nds_ranks(combined)[
        len(global_objective_front) :
    ]
    scored: list[dict[str, Any]] = []
    constraint_names = tuple(
        name.removeprefix("normalized_G:") for name in names[2:]
    )
    for index, raw in enumerate(neighbors):
        candidate = copy.deepcopy(dict(raw))
        constraints = {
            name: float(value)
            for name, value in zip(
                constraint_names,
                corrected[index, constraint_start:],
                strict=True,
            )
        }
        violation = float(
            np.maximum(
                corrected[index, constraint_start:], 0.0
            ).sum()
        )
        improvement = anchor_violation - violation
        pareto_value = 1.0 / (1.0 + int(ranks[index]))
        distance_penalty = 0.05 * (
            float(distances[index]) / float(trust_radius)
        )
        acquisition_score = (
            4.0 * improvement + pareto_value - distance_penalty
        )
        candidate.update(
            {
                "corrected_prediction": {
                    "objective_volume_L": float(
                        corrected[index, 0]
                    ),
                    "objective_total_loss_W": float(
                        corrected[index, 1]
                    ),
                    "normalized_constraints": constraints,
                    "normalized_constraint_violation": violation,
                },
                "anchor_corrected_normalized_constraint_violation": (
                    anchor_violation
                ),
                "expected_constraint_improvement": improvement,
                "objective_rank_against_global_audit_front": int(
                    ranks[index]
                ),
                "pareto_value": pareto_value,
                "delta_correction_decay": float(decay[index]),
                "acquisition_score": acquisition_score,
            }
        )
        scored.append(candidate)
    return scored, support


def _invalid_stop_decision(reason: str) -> ResidualDecision:
    return ResidualDecision(
        classification=ResidualClass.SOLVER_OR_PHYSICS_INVALID,
        eligible_for_local_acquisition=False,
        reasons=(reason,),
        max_scaled_residual=0.0,
    )


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    payload = _canonical_bytes(value) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise LocalTrustContractError(
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


def prepare_local_acquisition(
    *,
    aggregate_manifest_path: Path,
    observation_paths: Sequence[Path],
    output_directory: Path,
    current_results_complete: bool,
    current_round: int = 0,
    trust_radius: float = 0.02,
    max_pool: int = MAX_NEIGHBOR_POOL,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_batch: int = DEFAULT_MAX_BATCH,
    min_expected_constraint_improvement: float = 0.02,
    excluded_invalid_task_ids: Sequence[int] = (),
    pending_official_task_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Execute the read-only decision and publish a prepare-only manifest."""

    for values, label in (
        (excluded_invalid_task_ids, "excluded invalid task IDs"),
        (pending_official_task_ids, "pending official task IDs"),
    ):
        if (
            len(set(values)) != len(values)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in values
            )
        ):
            raise LocalTrustContractError(
                f"{label} must be unique positive integers"
            )
    if set(excluded_invalid_task_ids).intersection(
        pending_official_task_ids
    ):
        raise LocalTrustContractError(
            "a task cannot be both invalid and pending"
        )
    (
        aggregate,
        anchors,
        standard_path,
        objective_front_path,
    ) = load_authenticated_anchor_set(aggregate_manifest_path)
    measured = [
        load_authenticated_observation(path, anchors)
        for path in observation_paths
    ]
    orders = [item[1].selection_order for item in measured]
    if len(set(orders)) != len(orders):
        raise LocalTrustContractError(
            "duplicate observations for one official anchor"
        )
    anchor_table = build_anchor_table(anchors, measured)
    selected_existing = select_current_symmetric_result(anchor_table)
    training_auth: dict[str, Any] = {}
    support: dict[str, Any] = {}

    if selected_existing is not None:
        anchor = next(
            value
            for value in anchors
            if value.selection_order
            == selected_existing["selection_order"]
        )
        decision = next(
            item[3]
            for item in measured
            if item[1].selection_order == anchor.selection_order
        )
        batch = BatchSelection(
            selected=(),
            should_stop=True,
            stop_reason="existing_symmetric_result_passes_hard_gates",
        )
        measured_selection = selected_existing
    elif not current_results_complete:
        anchor = anchors[0]
        decision = _invalid_stop_decision(
            "current symmetric result set is not declared complete"
        )
        batch = BatchSelection(
            selected=(),
            should_stop=True,
            stop_reason="awaiting_current_symmetric_results",
        )
        measured_selection = {
            "status": "awaiting_current_symmetric_results",
            "new_local_neighbors_required": False,
            "excluded_operational_failures": [
                {
                    "task_id": int(task_id),
                    "classification": (
                        ResidualClass.SOLVER_OR_PHYSICS_INVALID.value
                    ),
                    "reason": (
                        "no_authenticated_temperature_residual"
                    ),
                    "local_neighbor_generation_allowed": False,
                }
                for task_id in excluded_invalid_task_ids
            ],
            "pending_official_standard_task_ids": [
                int(task_id) for task_id in pending_official_task_ids
            ],
        }
    else:
        eligible = [
            item
            for item in measured
            if item[3].classification
            is ResidualClass.SMALL_LOCAL_CORRECTION
            and item[3].eligible_for_local_acquisition
        ]
        eligible.sort(
            key=lambda item: ANCHOR_PRIORITY.index(
                item[1].selection_order
            )
        )
        if not eligible:
            anchor = anchors[0]
            decision = (
                measured[0][3]
                if measured
                else _invalid_stop_decision(
                    "no authenticated symmetric Standard observation"
                )
            )
            batch = BatchSelection(
                selected=(),
                should_stop=True,
                stop_reason=(
                    "no_small_authenticated_residual_for_local_acquisition"
                ),
            )
            measured_selection = {
                "status": "no_passing_result_and_no_small_residual",
                "new_local_neighbors_required": False,
            }
        else:
            observation, anchor, measurement, decision = eligible[0]
            training, training_auth = _source_training_table(anchor)
            neighbors = generate_local_neighbors(
                anchor.coordinate_unit,
                selection_order=anchor.selection_order,
                trust_radius=trust_radius,
                max_pool=max_pool,
            )
            scored, support = score_local_neighbors(
                neighbors,
                training=training,
                anchor=anchor,
                measurement=measurement,
                trust_radius=trust_radius,
                global_objective_front=_objective_front_values(
                    objective_front_path
                ),
            )
            batch = select_finite_batch(
                scored,
                current_round=current_round,
                max_rounds=max_rounds,
                max_batch=max_batch,
                min_expected_constraint_improvement=(
                    min_expected_constraint_improvement
                ),
            )
            measured_selection = {
                "status": (
                    "tiny_local_symmetric_batch_prepared"
                    if not batch.should_stop
                    else "local_acquisition_stop"
                ),
                "new_local_neighbors_required": not batch.should_stop,
                "anchor_selection_order": anchor.selection_order,
                "anchor_candidate_sha256": (
                    anchor.candidate_physics_sha256
                ),
                "source_task_id": int(observation["task_id"]),
                "source_residual_classification": (
                    decision.classification.value
                ),
                "local_model_support": support,
            }

    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_set = seal(
        {
            "schema_version": CANDIDATE_SET_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "prepare_only": True,
            "stage": "symmetric_standard",
            "candidate_count": len(batch.selected),
            "candidates": [
                copy.deepcopy(item) for item in batch.selected
            ],
            "scheduler_post_calls": 0,
            "automatic_full_trigger": False,
        }
    )
    candidate_path = _write_immutable_json(
        output_root / "local_symmetric_candidates.json",
        candidate_set,
    )
    authentication = {
        "aggregate_manifest": _file_record(
            aggregate_manifest_path.resolve(strict=True)
        ),
        "aggregate_manifest_payload_sha256": aggregate["payload_sha256"],
        "standard_candidates": _file_record(standard_path),
        "global_objective_front": _file_record(objective_front_path),
        "measured_observations": [
            _file_record(path.resolve(strict=True))
            for path in observation_paths
        ],
        "current_results_complete_declared": bool(
            current_results_complete
        ),
        "source_local_training": training_auth,
        "local_model_support": support,
        "candidate_set": _file_record(candidate_path),
        "candidate_set_payload_sha256": candidate_set["payload_sha256"],
        "scheduler_mutation_performed": False,
    }
    manifest = build_prepare_only_manifest(
        anchor_selection_order=anchor.selection_order,
        anchor_candidate_sha256=anchor.candidate_physics_sha256,
        residual_decision=decision,
        batch=batch,
        trust_radius=trust_radius,
        current_round=current_round,
        max_rounds=max_rounds,
        max_batch=max_batch,
        min_expected_constraint_improvement=(
            min_expected_constraint_improvement
        ),
        anchor_table=anchor_table,
        measured_selection=measured_selection,
        authentication=authentication,
    )
    manifest_path = _write_immutable_json(
        output_root / "acquisition_manifest.json", manifest
    )
    return {
        "manifest_path": str(manifest_path),
        "candidate_set_path": str(candidate_path),
        "payload_sha256": manifest["payload_sha256"],
        "candidate_count": len(batch.selected),
        "stop_reason": batch.stop_reason,
        "measured_selection": measured_selection,
        "scheduler_post_calls": 0,
    }


def write_exact_measured_nds(
    *,
    aggregate_manifest_path: Path,
    observation_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Reevaluate exact NDS over all supplied authenticated measurements."""

    _aggregate, anchors, _standard, _front = (
        load_authenticated_anchor_set(aggregate_manifest_path)
    )
    measured = [
        load_authenticated_observation(path, anchors)
        for path in observation_paths
    ]
    if not measured:
        raise LocalTrustContractError(
            "exact measured NDS requires at least one observation"
        )
    objectives = np.array(
        [
            [
                float(item[0]["actual_volume_L"]),
                float(item[0]["actual_total_loss_W"]),
            ]
            for item in measured
        ],
        dtype=float,
    )
    ranks = exact_global_nds_ranks(objectives)
    rows = []
    for item, rank in zip(measured, ranks, strict=True):
        observation, anchor, _measurement, _decision = item
        rows.append(
            {
                "selection_order": anchor.selection_order,
                "candidate_physics_sha256": (
                    anchor.candidate_physics_sha256
                ),
                "task_id": int(observation["task_id"]),
                "actual_volume_L": float(
                    observation["actual_volume_L"]
                ),
                "actual_total_loss_W": float(
                    observation["actual_total_loss_W"]
                ),
                "exact_measured_non_dominated_rank": int(rank),
            }
        )
    result = seal(
        {
            "schema_version": (
                "mft-goal-local-trust-exact-measured-nds-v1"
            ),
            "campaign_id": CAMPAIGN_ID,
            "diagnostic_measured_truth_only": True,
            "production_pareto_claimed": False,
            "objective_columns": [
                "actual_volume_L",
                "actual_total_loss_W",
            ],
            "row_count": len(rows),
            "rows": rows,
            "sorting_callable": (
                "tools.mft_goal_global_pareto.nondominated_ranks_2d"
            ),
            "scheduler_post_calls": 0,
        }
    )
    _write_immutable_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="prepare a sealed tiny symmetric batch; never submit it",
    )
    prepare.add_argument(
        "--aggregate-manifest", type=Path, required=True
    )
    prepare.add_argument(
        "--observation",
        type=Path,
        action="append",
        default=[],
        help=(
            "authenticated postsuccess measured observation; repeat for "
            "current #12/#6/#1/#5 results"
        ),
    )
    prepare.add_argument(
        "--current-results-complete",
        action="store_true",
        help=(
            "assert the supplied observations are the complete current "
            "symmetric result set; required before local generation"
        ),
    )
    prepare.add_argument("--output-directory", type=Path, required=True)
    prepare.add_argument("--current-round", type=int, default=0)
    prepare.add_argument("--trust-radius", type=float, default=0.02)
    prepare.add_argument("--max-pool", type=int, default=MAX_NEIGHBOR_POOL)
    prepare.add_argument(
        "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS
    )
    prepare.add_argument(
        "--max-batch", type=int, default=DEFAULT_MAX_BATCH
    )
    prepare.add_argument(
        "--min-expected-constraint-improvement",
        type=float,
        default=0.02,
    )
    prepare.add_argument(
        "--excluded-invalid-task-id",
        type=int,
        action="append",
        default=[],
        help=(
            "operationally invalid task with no authenticated residual; "
            "repeat as needed"
        ),
    )
    prepare.add_argument(
        "--pending-official-task-id",
        type=int,
        action="append",
        default=[],
        help="currently pending official Standard task; repeat as needed",
    )

    nds = commands.add_parser(
        "exact-measured-nds",
        help="exactly rerank authenticated measured objectives",
    )
    nds.add_argument("--aggregate-manifest", type=Path, required=True)
    nds.add_argument(
        "--observation", type=Path, action="append", required=True
    )
    nds.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_local_acquisition(
            aggregate_manifest_path=args.aggregate_manifest,
            observation_paths=args.observation,
            output_directory=args.output_directory,
            current_results_complete=args.current_results_complete,
            current_round=args.current_round,
            trust_radius=args.trust_radius,
            max_pool=args.max_pool,
            max_rounds=args.max_rounds,
            max_batch=args.max_batch,
            min_expected_constraint_improvement=(
                args.min_expected_constraint_improvement
            ),
            excluded_invalid_task_ids=args.excluded_invalid_task_id,
            pending_official_task_ids=args.pending_official_task_id,
        )
    else:
        result = write_exact_measured_nds(
            aggregate_manifest_path=args.aggregate_manifest,
            observation_paths=args.observation,
            output_path=args.output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalTrustContractError as exc:
        print(
            json.dumps(
                {
                    "event": "local_trust_acquisition_error",
                    "error": str(exc),
                    "scheduler_post_calls": 0,
                    "automatic_full_trigger": False,
                },
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        raise SystemExit(2) from exc
