"""Physics-informed reranking for 6/60 turn-graded capacitance.

This tool deliberately does not consume the legacy two-net ``C_rx_rx_F``
surrogate.  It builds a 60-node turn-voltage energy network from geometry,
fits a regularized log-delta correction to authenticated turn-graded
electrostatic FEA, and uses the result only to order the next FEA batch.

Neither the fitted prediction nor its upper confidence bound is a physical
feasibility authority.  Final resonance claims still require non-rounded,
turn-graded symmetric FEA with the actual additive main/side voltage schedule.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.electrostatic_cap import (  # noqa: E402
    TURN_GRADED_VOLTAGE_SCHEDULE_SCHEMA_VERSION,
    linear_turn_voltage_schedule,
)
from module.input_parameter_260706 import (  # noqa: E402
    create_input_parameter,
    validation_check,
)
from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402


MODEL_SCHEMA = "mft-goal-turn-graded-physics-delta-model-v2"
RANK_SCHEMA = "mft-goal-turn-graded-physics-rerank-v1"
SELECTION_SCHEMA = "mft-goal-turn-graded-physics-fea-selection-v1"
CALIBRATION_SCHEMA = "mft-goal-current24-turn-graded-cap-thermal-join-v1"
APPROVED_DIELECTRIC_STACK_SCHEMA = (
    "mft-goal-approved-secondary-interturn-dielectric-stack-v1"
)
DIELECTRIC_SENSITIVITY_SCHEMA = (
    "mft-goal-turn-graded-dielectric-stack-sensitivity-v1"
)
EPSILON_0_F_M = 8.854_187_812_8e-12
FIXED_L2_H = 0.2
RESONANCE_MIN_HZ = 15_000.0
RIDGE_ALPHA = 2.0
MIN_CALIBRATION_ROWS = 20
DELTA_FEATURE_NAMES = (
    "log_gap2_mm",
    "log_cw2_mm",
    "log_nwh2_mm",
    "side_turn_fraction",
    "ground_energy_share",
    "log_main_side_perimeter_ratio",
    "log1p_adjacent_edge_to_area_gap_aspect",
)
PHYSICS_COMPONENT_NAMES = (
    "adjacent_turn_energy_Ceq_F",
    "inner_ground_energy_Ceq_F",
    "outer_ground_energy_Ceq_F",
    "axial_ground_energy_Ceq_F",
    "main_side_bridge_energy_Ceq_F",
)
FORBIDDEN_AUTHORITY_INPUTS = (
    "C_rx_rx_F",
    "pred_C_rx_rx_F_fixed_lm2mh",
    "C_rx_rx_corrected_turn_graded_transfer_estimate_F",
)


class RerankError(RuntimeError):
    """The physics/delta authority or candidate contract is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RerankError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RerankError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RerankError("value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RerankError(f"{schema} seal mismatch")
    return result


def _finite(row: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RerankError(f"{key} is unavailable") from exc
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise RerankError(f"{key} is invalid")
    return value


def _integer(row: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = _finite(row, key)
    rounded = round(value)
    if abs(value - rounded) > 1e-9 or rounded < minimum:
        raise RerankError(f"{key} is not an integer >= {minimum}")
    return int(rounded)


def _optional_positive(
    row: Mapping[str, Any],
    keys: Sequence[str],
    fallback: float,
) -> float:
    for key in keys:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    if not math.isfinite(fallback) or fallback <= 0.0:
        raise RerankError("positive geometry fallback is invalid")
    return fallback


def _harmonic_clearance(values: Iterable[float]) -> float:
    cleaned = [float(value) for value in values if float(value) > 0.0]
    if not cleaned:
        raise RerankError("ground-clearance set is empty")
    return len(cleaned) / sum(1.0 / value for value in cleaned)


def _turn_names(n_main: int, n_side: int) -> list[str]:
    return [
        *(f"Rx_main_{index}_0" for index in range(n_main)),
        *(f"Rx_side_{index}_0" for index in range(n_side)),
    ]


def _ensure_derived_geometry(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    required = {
        "sl2_main_x",
        "sl2_main_y",
        "sl2_side_x",
        "sl2_side_y",
        "h_gap2",
    }
    if required.issubset(result) and any(
        key in result
        for key in (
            "w1c_w2s_gap_x_actual",
            "w1c_w2s_gap_actual",
            "w1c_w2s_space_x",
        )
    ):
        return result
    try:
        frame = create_input_parameter(result)
        _valid, derived, errors = validation_check(
            frame,
            strict=True,
            return_errors=True,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RerankError("candidate derived geometry is unavailable") from exc
    if errors:
        raise RerankError(
            "candidate geometry validation failed: " + " / ".join(errors)
        )
    return derived.iloc[0].to_dict()


def _section_turn_geometry(
    *,
    count: int,
    inner_x_mm: float,
    inner_y_mm: float,
    cw2_mm: float,
    gap2_mm: float,
) -> list[dict[str, float]]:
    result = []
    pitch = cw2_mm + gap2_mm
    for index in range(count):
        half_x = 0.5 * inner_x_mm + 0.5 * cw2_mm + index * pitch
        half_y = 0.5 * inner_y_mm + 0.5 * cw2_mm + index * pitch
        result.append(
            {
                "half_x_mm": half_x,
                "half_y_mm": half_y,
                "perimeter_mm": 4.0 * (half_x + half_y),
            }
        )
    return result


def physics_energy_network(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a geometry-only 60-node terminal-capacitance energy kernel.

    Every capacitance component is evaluated with vacuum permittivity.  The
    parallel-plate terms contain no fixed fringing coefficient.  Instead, an
    auditable dimensionless exposed-edge/plate-area geometry kernel is passed
    to the authenticated log-delta calibration as a learned feature.  The
    voltage fractions are produced by the same sealed midpoint schedule used
    by the turn-graded solver.
    """

    row = _ensure_derived_geometry(row)
    n1 = _integer(row, "N1_main", minimum=1) + _integer(row, "N1_side")
    n_main = _integer(row, "N2_main", minimum=1)
    n_side = _integer(row, "N2_side", minimum=1)
    if n1 != 6 or n_main + n_side != 60:
        raise RerankError("physics reranker requires exact 6/60 turns")
    cw2 = _finite(row, "cw2", positive=True)
    gap2 = _finite(row, "gap2", positive=True)
    nwh2 = _finite(row, "nwh2", positive=True)
    sl_main_x = _finite(row, "sl2_main_x", positive=True)
    sl_main_y = _finite(row, "sl2_main_y", positive=True)
    sl_side_x = _finite(row, "sl2_side_x", positive=True)
    sl_side_y = _finite(row, "sl2_side_y", positive=True)

    schedule = linear_turn_voltage_schedule(
        _turn_names(n_main, n_side),
        winding="Rx",
        section_order=("main", "side"),
        reverse_sections=(),
        voltage_policy="turn_midpoint",
        section_polarities={"main": 1, "side": 1},
        full_model=False,
    )
    if (
        schedule["schema_version"]
        != TURN_GRADED_VOLTAGE_SCHEDULE_SCHEMA_VERSION
        or schedule["turn_count"] != 60
    ):
        raise RerankError("turn-voltage schedule drifted")
    voltage_by_section: dict[str, list[float]] = {"main": [], "side": []}
    for turn in schedule["turns"]:
        voltage_by_section[str(turn["section"])].append(
            float(turn["voltage_fraction"])
        )

    geometry_by_section = {
        "main": _section_turn_geometry(
            count=n_main,
            inner_x_mm=sl_main_x,
            inner_y_mm=sl_main_y,
            cw2_mm=cw2,
            gap2_mm=gap2,
        ),
        "side": _section_turn_geometry(
            count=n_side,
            inner_x_mm=sl_side_x,
            inner_y_mm=sl_side_y,
            cw2_mm=cw2,
            gap2_mm=gap2,
        ),
    }
    epsilon_per_mm = EPSILON_0_F_M * 1e-3
    section_multiplicity = {"main": 1.0, "side": 2.0}
    adjacent_by_section = {"main": 0.0, "side": 0.0}
    adjacent_plate_area_voltage2_by_section = {
        "main": 0.0,
        "side": 0.0,
    }
    adjacent_exposed_edge_voltage2_by_section = {
        "main": 0.0,
        "side": 0.0,
    }
    axial_by_section = {"main": 0.0, "side": 0.0}
    h_gap2 = _optional_positive(
        row,
        ("h_gap2",),
        max((_finite(row, "h1", positive=True) - nwh2) / 2.0, 0.1),
    )
    for section in ("main", "side"):
        turns = geometry_by_section[section]
        voltages = voltage_by_section[section]
        for index in range(len(turns) - 1):
            average_perimeter_mm = 0.5 * (
                turns[index]["perimeter_mm"]
                + turns[index + 1]["perimeter_mm"]
            )
            area_mm2 = (
                average_perimeter_mm * nwh2
            )
            capacitance = epsilon_per_mm * area_mm2 / gap2
            voltage_step_squared = (
                voltages[index + 1] - voltages[index]
            ) ** 2
            adjacent_by_section[section] += (
                capacitance * voltage_step_squared
            )
            adjacent_plate_area_voltage2_by_section[section] += (
                area_mm2 * voltage_step_squared
            )
            # The interturn facing surface is a closed rectangular loop.  Its
            # only exposed plate boundaries are the top and bottom edges, so
            # the geometric edge length is exactly twice the loop perimeter.
            # No empirical fringing multiplier is introduced here.
            adjacent_exposed_edge_voltage2_by_section[section] += (
                2.0 * average_perimeter_mm * voltage_step_squared
            )
        for turn, voltage in zip(turns, voltages):
            top_bottom_area_mm2 = 2.0 * turn["perimeter_mm"] * cw2
            axial_by_section[section] += (
                epsilon_per_mm
                * top_bottom_area_mm2
                / h_gap2
                * voltage**2
            )
    adjacent = sum(
        section_multiplicity[section] * adjacent_by_section[section]
        for section in ("main", "side")
    )
    adjacent_plate_area_voltage2 = sum(
        section_multiplicity[section]
        * adjacent_plate_area_voltage2_by_section[section]
        for section in ("main", "side")
    )
    adjacent_exposed_edge_voltage2 = sum(
        section_multiplicity[section]
        * adjacent_exposed_edge_voltage2_by_section[section]
        for section in ("main", "side")
    )
    if adjacent_plate_area_voltage2 <= 0.0:
        raise RerankError("adjacent plate-area geometry kernel is non-positive")
    adjacent_edge_to_area_gap_aspect = (
        gap2
        * adjacent_exposed_edge_voltage2
        / adjacent_plate_area_voltage2
    )
    axial = sum(
        section_multiplicity[section] * axial_by_section[section]
        for section in ("main", "side")
    )

    main_inner_clearance = _harmonic_clearance(
        [
            _optional_positive(row, ("cc_w2c_space_x",), 40.0),
            _optional_positive(row, ("cc_w2c_space_y",), 40.0),
        ]
    )
    main_outer_clearance = _harmonic_clearance(
        [
            _optional_positive(row, ("w2c_w1c_space_x",), 40.0),
            _optional_positive(row, ("w2c_w1c_space_y",), 40.0),
        ]
    )
    side_inner_clearance = _optional_positive(
        row,
        ("w2s_w1s_space_x", "w1s_cs_space_x"),
        40.0,
    )
    side_outer_clearance = _optional_positive(
        row,
        ("w1s_cs_space_x", "h_gap2"),
        h_gap2,
    )
    inner_ground_by_section = {"main": 0.0, "side": 0.0}
    outer_ground_by_section = {"main": 0.0, "side": 0.0}
    for section, d_inner, d_outer in (
        ("main", main_inner_clearance, main_outer_clearance),
        ("side", side_inner_clearance, side_outer_clearance),
    ):
        turns = geometry_by_section[section]
        voltages = voltage_by_section[section]
        inner_ground_by_section[section] += (
            epsilon_per_mm
            * turns[0]["perimeter_mm"]
            * nwh2
            / d_inner
            * voltages[0] ** 2
        )
        outer_ground_by_section[section] += (
            epsilon_per_mm
            * turns[-1]["perimeter_mm"]
            * nwh2
            / d_outer
            * voltages[-1] ** 2
        )
    inner_ground = sum(
        section_multiplicity[section] * inner_ground_by_section[section]
        for section in ("main", "side")
    )
    outer_ground = sum(
        section_multiplicity[section] * outer_ground_by_section[section]
        for section in ("main", "side")
    )

    main_outer = geometry_by_section["main"][-1]
    side_inner = geometry_by_section["side"][0]
    section_gap = _optional_positive(
        row,
        (
            "w1c_w2s_gap_x_actual",
            "w1c_w2s_gap_actual",
            "w1c_w2s_space_x",
        ),
        _optional_positive(row, ("l2",), 300.0) * 0.1,
    )
    bridge_area_mm2 = (
        2.0
        * min(main_outer["half_y_mm"], side_inner["half_y_mm"])
        * nwh2
    )
    retained_bridge = (
        epsilon_per_mm
        * bridge_area_mm2
        / section_gap
        * (
            voltage_by_section["main"][-1]
            - voltage_by_section["side"][0]
        )
        ** 2
    )
    bridge = 2.0 * retained_bridge
    components = {
        "adjacent_turn_energy_Ceq_F": adjacent,
        "inner_ground_energy_Ceq_F": inner_ground,
        "outer_ground_energy_Ceq_F": outer_ground,
        "axial_ground_energy_Ceq_F": axial,
        "main_side_bridge_energy_Ceq_F": bridge,
    }
    total = float(sum(components.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise RerankError("physics energy network is non-positive")
    ground_share = (
        inner_ground + outer_ground + axial
    ) / total
    main_perimeter = float(
        np.mean(
            [turn["perimeter_mm"] for turn in geometry_by_section["main"]]
        )
    )
    side_perimeter = float(
        np.mean(
            [turn["perimeter_mm"] for turn in geometry_by_section["side"]]
        )
    )
    delta_features = {
        "log_gap2_mm": math.log(gap2),
        "log_cw2_mm": math.log(cw2),
        "log_nwh2_mm": math.log(nwh2),
        "side_turn_fraction": n_side / (n_main + n_side),
        "ground_energy_share": ground_share,
        "log_main_side_perimeter_ratio": math.log(
            main_perimeter / side_perimeter
        ),
        "log1p_adjacent_edge_to_area_gap_aspect": math.log1p(
            adjacent_edge_to_area_gap_aspect
        ),
    }
    return {
        "schema_version": "mft-goal-turn-graded-voltage-energy-network-v1",
        "turn_voltage_schedule": {
            "schema_version": schedule["schema_version"],
            "voltage_policy": schedule["voltage_policy"],
            "section_order": schedule["section_order"],
            "section_polarities": schedule["section_polarities"],
            "section_turn_voltage_weights": schedule[
                "section_turn_voltage_weights"
            ],
            "turn_count": schedule["turn_count"],
            "signed_effective_turn_count_geometry": schedule[
                "signed_effective_turn_count_geometry"
            ],
            "full_topology_effective_turn_count": schedule[
                "full_topology_effective_turn_count"
            ],
        },
        "components": components,
        "component_breakdown": {
            "section_physical_multiplicity": section_multiplicity,
            "adjacent_turn_retained_by_section_F": adjacent_by_section,
            "adjacent_turn_physical_by_section_F": {
                section: (
                    section_multiplicity[section]
                    * adjacent_by_section[section]
                )
                for section in ("main", "side")
            },
            "adjacent_parallel_plate_area_voltage2_by_section_mm2": (
                adjacent_plate_area_voltage2_by_section
            ),
            "adjacent_exposed_edge_voltage2_by_section_mm": (
                adjacent_exposed_edge_voltage2_by_section
            ),
            "adjacent_parallel_plate_area_voltage2_physical_mm2": (
                adjacent_plate_area_voltage2
            ),
            "adjacent_exposed_edge_voltage2_physical_mm": (
                adjacent_exposed_edge_voltage2
            ),
            "adjacent_edge_to_area_gap_aspect": (
                adjacent_edge_to_area_gap_aspect
            ),
            "axial_ground_retained_by_section_F": axial_by_section,
            "axial_ground_physical_by_section_F": {
                section: (
                    section_multiplicity[section]
                    * axial_by_section[section]
                )
                for section in ("main", "side")
            },
            "inner_ground_retained_by_section_F": inner_ground_by_section,
            "inner_ground_physical_by_section_F": {
                section: (
                    section_multiplicity[section]
                    * inner_ground_by_section[section]
                )
                for section in ("main", "side")
            },
            "outer_ground_retained_by_section_F": outer_ground_by_section,
            "outer_ground_physical_by_section_F": {
                section: (
                    section_multiplicity[section]
                    * outer_ground_by_section[section]
                )
                for section in ("main", "side")
            },
            "main_side_bridge_retained_F": retained_bridge,
            "main_side_bridge_physical_F": bridge,
        },
        "physics_Ceq_F": total,
        "delta_features": delta_features,
        "raw_two_net_capacitance_used": False,
        "single_truth_transfer_ratio_used": False,
        "dielectric_material_basis": "air_only",
        "dielectric_stack_sensitivity_required": True,
        "parallel_plate_fringe_coefficient": None,
        "fringing_role": "calibrated_residual_not_fixed_truth",
        "physical_feasibility_authority": False,
    }


def _ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or target.shape != (features.shape[0],):
        raise RerankError("ridge calibration dimensions are invalid")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ target,
    )
    return coefficients, mean, scale


def _predict_delta(
    features: np.ndarray,
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    normalized = (features - mean) / scale
    return np.column_stack([np.ones(len(normalized)), normalized]) @ coefficients


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _validate_calibration_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = _read_json(manifest_path)
    unsigned = dict(manifest)
    observed = unsigned.pop("payload_sha256", None)
    ranked = manifest.get("ranked_csv") or {}
    ranked_path = Path(str(ranked.get("path") or ""))
    if (
        manifest.get("schema") != CALIBRATION_SCHEMA
        or observed != canonical_sha256(unsigned)
        or manifest.get("row_count") != 24
        or manifest.get("all_turn_graded_pair_contracts_valid") is not True
        or manifest.get("full_model_series_interconnect_attested") is not False
        or manifest.get("symmetric_even_potential_diagnostic") is not True
        or manifest.get("production_promotion_allowed_from_manifest_alone")
        is not False
        or not ranked_path.is_file()
        or ranked.get("sha256") != _sha256_file(ranked_path)
    ):
        raise RerankError("turn-graded calibration manifest contract mismatch")
    return manifest, ranked_path.resolve(strict=True)


def _calibration_rows(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    manifest, ranked_path = _validate_calibration_manifest(manifest_path)
    by_geometry = {
        str(row["physical_geometry_sha256"]): row
        for row in manifest["rows"]
    }
    rows: list[dict[str, Any]] = []
    with ranked_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            geometry = str(raw.get("physical_geometry_sha256") or "")
            authority = by_geometry.get(geometry)
            if authority is None:
                raise RerankError("calibration CSV contains a foreign geometry")
            params_path = Path(str(raw.get("source_params_path") or ""))
            if (
                raw.get("turn_graded_pair_contract_valid", "").lower()
                != "true"
                or raw.get("legacy_two_net_capacitance_used", "").lower()
                != "false"
                or raw.get("full_model_series_interconnect_attested", "").lower()
                != "false"
                or not params_path.is_file()
                or raw.get("source_params_sha256")
                != _sha256_file(params_path)
                or authority.get("rx_result_json_sha256")
                != raw.get("rx_result_json_sha256")
            ):
                raise RerankError("calibration truth provenance is invalid")
            params = _read_json(params_path)
            truth = _finite(raw, "C_rx_rx_turn_graded_F", positive=True)
            network = physics_energy_network(params)
            rows.append(
                {
                    "physical_geometry_sha256": geometry,
                    "source_params_path": str(params_path.resolve(strict=True)),
                    "source_params_sha256": _sha256_file(params_path),
                    "rx_result_json_sha256": raw["rx_result_json_sha256"],
                    "C_rx_rx_turn_graded_F": truth,
                    "network": network,
                }
            )
    if (
        len(rows) < MIN_CALIBRATION_ROWS
        or len(rows) != len(by_geometry)
        or len({row["physical_geometry_sha256"] for row in rows}) != len(rows)
    ):
        raise RerankError("turn-graded calibration coverage is incomplete")
    return rows, manifest, ranked_path


def fit_model(
    *,
    calibration_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows, manifest, ranked_path = _calibration_rows(calibration_manifest_path)
    feature_matrix = np.asarray(
        [
            [
                row["network"]["delta_features"][name]
                for name in DELTA_FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )
    truth = np.asarray(
        [row["C_rx_rx_turn_graded_F"] for row in rows],
        dtype=float,
    )
    physics = np.asarray(
        [row["network"]["physics_Ceq_F"] for row in rows],
        dtype=float,
    )
    log_delta = np.log(truth / physics)
    coefficients, mean, scale = _ridge_fit(feature_matrix, log_delta)
    fitted = physics * np.exp(
        _predict_delta(feature_matrix, coefficients, mean, scale)
    )
    loo_predictions = []
    for index in range(len(rows)):
        keep = np.arange(len(rows)) != index
        beta, fold_mean, fold_scale = _ridge_fit(
            feature_matrix[keep],
            log_delta[keep],
        )
        delta = _predict_delta(
            feature_matrix[index : index + 1],
            beta,
            fold_mean,
            fold_scale,
        )[0]
        loo_predictions.append(physics[index] * math.exp(float(delta)))
    loo_predictions_array = np.asarray(loo_predictions)
    loo_abs_log_error = np.abs(np.log(loo_predictions_array / truth))
    loo_ape = np.abs(loo_predictions_array / truth - 1.0)
    unsigned = {
        "schema_version": MODEL_SCHEMA,
        "created_at": _now(),
        "target": "C_rx_rx_turn_graded_F",
        "method": (
            "60_node_turn_midpoint_voltage_energy_network_plus_regularized_"
            "log_delta_calibration_with_learned_fringe_geometry_feature"
        ),
        "calibration_manifest": {
            "path": str(calibration_manifest_path.resolve(strict=True)),
            "sha256": _sha256_file(
                calibration_manifest_path.resolve(strict=True)
            ),
            "payload_sha256": manifest["payload_sha256"],
        },
        "calibration_ranked_csv": {
            "path": str(ranked_path),
            "sha256": _sha256_file(ranked_path),
        },
        "calibration_row_count": len(rows),
        "calibration_geometry_sha256": [
            row["physical_geometry_sha256"] for row in rows
        ],
        "physics_network": {
            "schema_version": rows[0]["network"]["schema_version"],
            "component_names": list(PHYSICS_COMPONENT_NAMES),
            "vacuum_permittivity_F_m": EPSILON_0_F_M,
            "section_physical_multiplicity": {
                "main": 1,
                "side": 2,
            },
            "turn_voltage_schedule": rows[0]["network"][
                "turn_voltage_schedule"
            ],
            "raw_two_net_capacitance_used": False,
            "single_truth_transfer_ratio_used": False,
            "dielectric_material_basis": "air_only",
            "parallel_plate_fringe_coefficient": None,
            "fringing_role": "calibrated_residual_not_fixed_truth",
            "fringe_geometry_feature": (
                "log1p_adjacent_edge_to_area_gap_aspect"
            ),
        },
        "delta_model": {
            "feature_names": list(DELTA_FEATURE_NAMES),
            "ridge_alpha": RIDGE_ALPHA,
            "coefficients_intercept_then_standardized_features": (
                coefficients.tolist()
            ),
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "feature_minimum": feature_matrix.min(axis=0).tolist(),
            "feature_maximum": feature_matrix.max(axis=0).tolist(),
        },
        "calibration_metrics": {
            "in_sample_mape_pct": float(
                np.mean(np.abs(fitted / truth - 1.0)) * 100.0
            ),
            "loo_mape_pct": float(np.mean(loo_ape) * 100.0),
            "loo_p90_ape_pct": _quantile(loo_ape, 0.9) * 100.0,
            "loo_q90_abs_log_error": _quantile(loo_abs_log_error, 0.9),
            "loo_max_ape_pct": float(np.max(loo_ape) * 100.0),
        },
        "fixed_lm_primary_referred_H": 0.002,
        "fixed_L2_secondary_H": FIXED_L2_H,
        "resonance_formula": "1/(2*pi*sqrt(0.2*C_rx_turn_graded_F))",
        "resonance_minimum_Hz": RESONANCE_MIN_HZ,
        "forbidden_authority_inputs": list(FORBIDDEN_AUTHORITY_INPUTS),
        "single_0p759701_ratio_role": "not_used_by_this_model",
        "raw_two_net_capacitance_role": "not_used_by_this_model",
        "fringing_role": "calibrated_residual_not_fixed_truth",
        "fixed_fringe_coefficient_used": False,
        "dielectric_material_basis": "air_only",
        "approved_dielectric_stack_sensitivity_required": True,
        "air_only_result_can_close_final_gate": False,
        "prediction_role": "next_FEA_batch_reranking_only",
        "physical_feasibility_authority": False,
        "automatic_final_promotion_allowed": False,
        "final_turn_graded_symmetric_FEA_required": True,
        "full_model_series_interconnect_attested": False,
        "production_eligible": False,
    }
    model = _seal(unsigned)
    _write_json(output_path, model)
    return model


def _load_model(path: Path) -> dict[str, Any]:
    model = _validate_seal(_read_json(path), MODEL_SCHEMA)
    if (
        model.get("physical_feasibility_authority") is not False
        or model.get("automatic_final_promotion_allowed") is not False
        or model.get("final_turn_graded_symmetric_FEA_required") is not True
        or model.get("single_0p759701_ratio_role")
        != "not_used_by_this_model"
        or model.get("raw_two_net_capacitance_role")
        != "not_used_by_this_model"
        or model.get("fringing_role")
        != "calibrated_residual_not_fixed_truth"
        or model.get("fixed_fringe_coefficient_used") is not False
        or model.get("dielectric_material_basis") != "air_only"
        or model.get("approved_dielectric_stack_sensitivity_required")
        is not True
        or model.get("air_only_result_can_close_final_gate") is not False
        or model.get("physics_network", {}).get(
            "raw_two_net_capacitance_used"
        )
        is not False
        or model.get("physics_network", {}).get("fringing_role")
        != "calibrated_residual_not_fixed_truth"
        or model.get("physics_network", {}).get(
            "parallel_plate_fringe_coefficient"
        )
        is not None
    ):
        raise RerankError("physics/delta model authority drifted")
    return model


def _candidate_prediction(
    row: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, float]:
    network = physics_energy_network(row)
    features = np.asarray(
        [
            [
                network["delta_features"][name]
                for name in DELTA_FEATURE_NAMES
            ]
        ],
        dtype=float,
    )
    delta = model["delta_model"]
    coefficients = np.asarray(
        delta["coefficients_intercept_then_standardized_features"],
        dtype=float,
    )
    mean = np.asarray(delta["feature_mean"], dtype=float)
    scale = np.asarray(delta["feature_scale"], dtype=float)
    minimum = np.asarray(delta["feature_minimum"], dtype=float)
    maximum = np.asarray(delta["feature_maximum"], dtype=float)
    predicted_delta = float(
        _predict_delta(features, coefficients, mean, scale)[0]
    )
    predicted_c = network["physics_Ceq_F"] * math.exp(predicted_delta)
    normalized_outside = np.maximum(
        np.maximum(minimum - features[0], features[0] - maximum) / scale,
        0.0,
    )
    extrapolation_distance = float(np.linalg.norm(normalized_outside))
    q90 = float(
        model["calibration_metrics"]["loo_q90_abs_log_error"]
    )
    log_half_width = q90 * (1.0 + extrapolation_distance)
    ucb_c = predicted_c * math.exp(log_half_width)
    lcb_f = 1.0 / (2.0 * math.pi * math.sqrt(FIXED_L2_H * ucb_c))
    mean_f = 1.0 / (
        2.0 * math.pi * math.sqrt(FIXED_L2_H * predicted_c)
    )
    return {
        "physics_network_Ceq_F": network["physics_Ceq_F"],
        "physics_delta_Crx_mean_F": predicted_c,
        "physics_delta_Crx_q90_ucb_F": ucb_c,
        "physics_delta_log_half_width": log_half_width,
        "physics_delta_extrapolation_distance": extrapolation_distance,
        "physics_delta_fRx_mean_Hz": mean_f,
        "physics_delta_fRx_q90_lcb_Hz": lcb_f,
    }


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise RerankError(f"{key} must be an explicit non-empty string")
    return result.strip()


def _validated_approved_dielectric_stack(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate a sealed, explicitly project-approved material stack.

    No relative permittivity is supplied by this tool.  Every dielectric
    property must carry its own project authority reference, and the stack
    itself must be approved for secondary interturn-gap sensitivity.
    """

    if value is None:
        raise RerankError(
            "approved dielectric stack is required; sensitivity is "
            "fail-closed"
        )
    stack = _validate_seal(
        value,
        APPROVED_DIELECTRIC_STACK_SCHEMA,
    )
    approval = stack.get("approval")
    if not isinstance(approval, Mapping):
        raise RerankError("dielectric stack approval record is unavailable")
    if (
        approval.get("status") != "approved"
        or approval.get("authority_kind")
        != "project_approved_dielectric_stack"
        or approval.get("scope")
        != "secondary_interturn_gap_sensitivity"
    ):
        raise RerankError(
            "dielectric stack is not approved for the required scope"
        )
    _required_text(approval, "approved_by")
    _required_text(approval, "approved_at")
    _required_text(approval, "authority_reference")
    _required_text(stack, "case_id")
    if stack.get("gap_fill_policy") != "approved_layers_plus_residual_air":
        raise RerankError("dielectric stack gap-fill policy is unsupported")
    layers = stack.get("dielectric_layers")
    if not isinstance(layers, list) or not layers:
        raise RerankError("approved dielectric stack has no material layers")
    validated_layers = []
    for index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, Mapping):
            raise RerankError(f"dielectric layer {index} is invalid")
        material_id = _required_text(raw_layer, "material_id")
        thickness_mm = _finite(
            raw_layer,
            "thickness_mm",
            positive=True,
        )
        relative_permittivity = _finite(
            raw_layer,
            "relative_permittivity",
            positive=True,
        )
        if relative_permittivity < 1.0:
            raise RerankError(
                f"dielectric layer {index} has relative permittivity < 1"
            )
        property_authority = raw_layer.get(
            "relative_permittivity_authority"
        )
        if (
            not isinstance(property_authority, Mapping)
            or property_authority.get("authority_kind")
            != "project_approved_material_property"
        ):
            raise RerankError(
                f"dielectric layer {index} lacks approved material authority"
            )
        _required_text(property_authority, "reference")
        validated_layers.append(
            {
                "material_id": material_id,
                "thickness_mm": thickness_mm,
                "relative_permittivity": relative_permittivity,
                "relative_permittivity_authority": dict(
                    property_authority
                ),
            }
        )
    result = dict(stack)
    result["dielectric_layers"] = validated_layers
    return result


def dielectric_stack_sensitivity(
    row: Mapping[str, Any],
    model: Mapping[str, Any],
    approved_stack: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply a sealed stack to the adjacent-turn portion of the air UCB.

    The method uses no built-in/example permittivity and no fixed empirical
    fringing coefficient.  Its conservative factor is the calibrated q90
    total/parallel-plate ratio, floored at unity so a dielectric increment
    cannot be reduced by the residual model.
    """

    stack = _validated_approved_dielectric_stack(approved_stack)
    network = physics_energy_network(row)
    prediction = _candidate_prediction(row, model)
    gap_mm = _finite(row, "gap2", positive=True)
    dielectric_thickness_mm = sum(
        layer["thickness_mm"] for layer in stack["dielectric_layers"]
    )
    tolerance_mm = max(1e-9, gap_mm * 1e-9)
    if dielectric_thickness_mm > gap_mm + tolerance_mm:
        raise RerankError(
            "approved dielectric layer thickness exceeds candidate gap2"
        )
    air_thickness_mm = max(0.0, gap_mm - dielectric_thickness_mm)
    equivalent_air_thickness_mm = air_thickness_mm + sum(
        layer["thickness_mm"] / layer["relative_permittivity"]
        for layer in stack["dielectric_layers"]
    )
    if equivalent_air_thickness_mm <= 0.0:
        raise RerankError(
            "dielectric stack equivalent air thickness is non-positive"
        )
    kappa = gap_mm / equivalent_air_thickness_mm
    adjacent_air = float(
        network["components"]["adjacent_turn_energy_Ceq_F"]
    )
    air_ucb = prediction["physics_delta_Crx_q90_ucb_F"]
    calibrated_q90_factor = air_ucb / network["physics_Ceq_F"]
    conservative_fringe_factor = max(1.0, calibrated_q90_factor)
    dielectric_increment = (
        (kappa - 1.0)
        * adjacent_air
        * conservative_fringe_factor
    )
    stack_ucb = air_ucb + dielectric_increment
    stack_resonance_lcb_hz = 1.0 / (
        2.0 * math.pi * math.sqrt(FIXED_L2_H * stack_ucb)
    )
    return _seal(
        {
            "schema_version": DIELECTRIC_SENSITIVITY_SCHEMA,
            "created_at": _now(),
            "approved_stack": {
                "case_id": stack["case_id"],
                "payload_sha256": stack["payload_sha256"],
                "approval": stack["approval"],
                "gap_fill_policy": stack["gap_fill_policy"],
            },
            "model_payload_sha256": model["payload_sha256"],
            "candidate_identity": {
                key: row[key]
                for key in (
                    "physical_geometry_sha256",
                    "canonical_params_sha256",
                    "seed",
                    "task_id",
                    "candidate_index",
                )
                if key in row
            },
            "candidate_gap2_mm": gap_mm,
            "dielectric_layers_total_thickness_mm": (
                dielectric_thickness_mm
            ),
            "residual_air_thickness_mm": air_thickness_mm,
            "equivalent_air_thickness_mm": equivalent_air_thickness_mm,
            "kappa": kappa,
            "kappa_formula": (
                "gap/(t_air+sum(t_dielectric_i/epsr_i))"
            ),
            "C_air_q90_ucb_F": air_ucb,
            "adjacent_parallel_plate_air_Ceq_F": adjacent_air,
            "calibrated_q90_total_to_parallel_plate_factor": (
                calibrated_q90_factor
            ),
            "conservative_fringe_factor": conservative_fringe_factor,
            "fringe_factor_policy": (
                "max(1,calibrated_q90_air_total_over_physics_total)"
            ),
            "fringing_role": "calibrated_residual_not_fixed_truth",
            "fixed_fringe_coefficient_used": False,
            "C_stack_q90_ucb_F": stack_ucb,
            "C_stack_q90_ucb_formula": (
                "C_air_UCB+(kappa-1)*adjacent_air*"
                "conservative_fringe_factor"
            ),
            "fRx_stack_q90_lcb_Hz": stack_resonance_lcb_hz,
            "fixed_L2_secondary_H": FIXED_L2_H,
            "example_or_default_permittivity_used": False,
            "sensitivity_role": "screening_bound_only",
            "physical_feasibility_authority": False,
            "turn_graded_dielectric_FEA_still_required": True,
            "production_eligible": False,
        }
    )


def _column(frame: pd.DataFrame, names: Sequence[str], default: float) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _expanded_candidate_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    encoded = row.get("decoded_physical_params_json")
    if isinstance(encoded, str) and encoded.strip():
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise RerankError(
                "decoded_physical_params_json is invalid"
            ) from exc
        if not isinstance(decoded, dict):
            raise RerankError(
                "decoded_physical_params_json must contain an object"
            )
        expanded.update(decoded)
    for key, value in row.items():
        if value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if not missing:
            expanded[key] = value
    return expanded


def _strata(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise RerankError("stratification column contains non-finite values")
    ranks = numeric.rank(method="first", pct=True)
    return pd.cut(
        ranks,
        bins=[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)


def _selection_order(frame: pd.DataFrame, count: int) -> list[int]:
    ordered = frame.sort_values(
        [
            "physics_delta_fRx_q90_lcb_Hz",
            "physics_delta_extrapolation_distance",
            "_violation",
            "_volume",
            "_loss",
            "_geometry",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    )
    selected: list[int] = []
    roles: dict[int, list[str]] = {}

    def add(index: int, role: str) -> None:
        if index not in selected:
            selected.append(index)
        roles.setdefault(index, []).append(role)

    add(int(ordered.index[0]), "best_physics_delta_q90")
    for column in ("gap2_stratum", "cw2_stratum", "height_stratum"):
        for stratum in ("low", "mid", "high"):
            subset = ordered[ordered[column] == stratum]
            if not subset.empty:
                add(int(subset.index[0]), f"{column}:{stratum}")

    coordinates = frame[
        ["gap2", "cw2", "nwh2", "_W", "_L", "_H"]
    ].to_numpy(dtype=float)
    center = np.nanmean(coordinates, axis=0)
    scale = np.nanstd(coordinates, axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (coordinates - center) / scale
    while len(selected) < min(count, len(frame)):
        remaining = [int(index) for index in frame.index if index not in selected]
        if not selected:
            add(remaining[0], "maximin_seed")
            continue
        selected_matrix = normalized[np.asarray(selected, dtype=int)]
        best = sorted(
            remaining,
            key=lambda index: (
                -float(
                    np.linalg.norm(
                        selected_matrix - normalized[index],
                        axis=1,
                    ).min()
                ),
                -float(frame.loc[index, "physics_delta_fRx_q90_lcb_Hz"]),
                str(frame.loc[index, "_geometry"]),
            ),
        )[0]
        add(best, "design_space_maximin")
    frame["_selection_roles"] = [
        ",".join(roles.get(int(index), [])) for index in frame.index
    ]
    return selected[:count]


def rank_candidates(
    *,
    model_path: Path,
    candidates_path: Path,
    output_dir: Path,
    selection_count: int = 12,
) -> dict[str, Any]:
    if not 9 <= selection_count <= 24:
        raise RerankError("selection_count must be within 9..24")
    model = _load_model(model_path)
    candidates_path = candidates_path.resolve(strict=True)
    if candidates_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(candidates_path)
    else:
        frame = pd.read_csv(candidates_path)
    if frame.empty:
        raise RerankError("candidate table is empty")
    predictions = []
    expanded_rows = []
    errors = []
    for index, row in frame.iterrows():
        expanded: dict[str, Any] = {}
        try:
            expanded = _expanded_candidate_mapping(row.to_dict())
            predictions.append(_candidate_prediction(expanded, model))
            errors.append("")
        except RerankError as exc:
            predictions.append(
                {
                    "physics_network_Ceq_F": np.nan,
                    "physics_delta_Crx_mean_F": np.nan,
                    "physics_delta_Crx_q90_ucb_F": np.nan,
                    "physics_delta_log_half_width": np.nan,
                    "physics_delta_extrapolation_distance": np.nan,
                    "physics_delta_fRx_mean_Hz": np.nan,
                    "physics_delta_fRx_q90_lcb_Hz": np.nan,
                }
            )
            errors.append(str(exc))
        expanded_rows.append(expanded)
    ranked = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(predictions)],
        axis=1,
    )
    for key in (
        "N1_main",
        "N1_side",
        "N2_main",
        "N2_side",
        "cw2",
        "gap2",
        "nwh2",
        "h1",
        "l1",
        "l2",
        "sl2_main_x",
        "sl2_main_y",
        "sl2_side_x",
        "sl2_side_y",
        "h_gap2",
        "w1c_w2s_gap_x_actual",
    ):
        if key not in ranked:
            ranked[key] = [
                expanded.get(key, np.nan) for expanded in expanded_rows
            ]
    ranked["physics_delta_error"] = errors
    ranked = ranked[ranked["physics_delta_error"] == ""].copy()
    if len(ranked) < selection_count:
        raise RerankError("too few valid 6/60 candidates for FEA selection")
    ranked.reset_index(drop=True, inplace=True)
    ranked["gap2_stratum"] = _strata(ranked["gap2"])
    ranked["cw2_stratum"] = _strata(ranked["cw2"])
    ranked["height_stratum"] = _strata(ranked["nwh2"])
    ranked["_violation"] = _column(
        ranked,
        ("normalized_constraint_violation",),
        1e9,
    )
    ranked["_volume"] = _column(
        ranked,
        ("objective_volume_L", "volume_L"),
        1e9,
    )
    ranked["_loss"] = _column(
        ranked,
        ("objective_total_loss_W", "total_loss_W"),
        1e9,
    )
    ranked["_W"] = _column(
        ranked,
        ("exterior_W_drawing_x_mm", "W_mm"),
        1200.0,
    )
    ranked["_L"] = _column(
        ranked,
        ("exterior_L_perpendicular_y_mm", "L_mm"),
        900.0,
    )
    ranked["_H"] = _column(
        ranked,
        ("exterior_H_mm", "H_mm"),
        750.0,
    )
    ranked["_geometry"] = (
        ranked["physical_geometry_sha256"].astype(str)
        if "physical_geometry_sha256" in ranked
        else ranked.index.astype(str)
    )
    selected_indices = _selection_order(ranked, selection_count)
    ranked["physics_delta_rerank_order"] = ranked[
        "physics_delta_fRx_q90_lcb_Hz"
    ].rank(method="first", ascending=False).astype(int)
    ranked["selected_for_turn_graded_FEA"] = False
    ranked.loc[selected_indices, "selected_for_turn_graded_FEA"] = True
    selected = ranked.loc[selected_indices].copy()
    selected["fea_selection_order"] = np.arange(1, len(selected) + 1)
    selected["required_FEA"] = (
        "symmetric_nonrounded_Rx_60turn_additive_main_side_midpoint"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_csv = output_dir / "physics_delta_ranked_candidates.csv"
    selection_csv = output_dir / "turn_graded_fea_selection.csv"
    ranked.drop(
        columns=[
            "_violation",
            "_volume",
            "_loss",
            "_W",
            "_L",
            "_H",
            "_geometry",
            "_selection_roles",
        ],
        errors="ignore",
    ).to_csv(ranked_csv, index=False)
    selected.drop(
        columns=[
            "_violation",
            "_volume",
            "_loss",
            "_W",
            "_L",
            "_H",
            "_geometry",
        ],
        errors="ignore",
    ).to_csv(selection_csv, index=False)
    selection_plan = _seal(
        {
            "schema_version": SELECTION_SCHEMA,
            "created_at": _now(),
            "model": {
                "path": str(model_path.resolve(strict=True)),
                "sha256": _sha256_file(model_path.resolve(strict=True)),
                "payload_sha256": model["payload_sha256"],
            },
            "candidate_table": {
                "path": str(candidates_path),
                "sha256": _sha256_file(candidates_path),
            },
            "selection_csv": {
                "path": str(selection_csv.resolve()),
                "sha256": _sha256_file(selection_csv),
                "row_count": len(selected),
            },
            "selected_geometry_sha256": selected["_geometry"].tolist(),
            "stratification_dimensions": ["gap2", "cw2", "nwh2"],
            "selection_method": (
                "best_q90_then_each_marginal_low_mid_high_then_6D_maximin"
            ),
            "minimum_anchor_geometry_count": 9,
            "recommended_initial_geometry_count": 12,
            "adaptive_followup_geometry_count": 6,
            "final_paired_confirmation_geometry_count": 3,
            "dielectric_stack_sensitivity_geometry_count": 3,
            "dielectric_stack_cases_per_geometry": 3,
            "recommended_total_unique_geometry_count": 21,
            "recommended_solver_task_count": 33,
            "task_count_breakdown": {
                "initial_Rx_only": 12,
                "adaptive_Rx_only": 6,
                "final_top3_Rx_plus_Tx": 6,
                "final_top3_Rx_approved_dielectric_min_nominal_max": 9,
            },
            "FEA_contract": {
                "model": "symmetric_eighth",
                "rounded": False,
                "active_winding_initial": "Rx",
                "turn_count": 60,
                "section_order": ["main", "side"],
                "section_polarities": {"main": 1, "side": 1},
                "voltage_policy": "turn_midpoint",
                "legacy_two_net_capacitance_used": False,
                "single_0p759701_ratio_used": False,
                "fringing_role": "calibrated_residual_not_fixed_truth",
                "fixed_fringe_coefficient_used": False,
                "final_top3_require_paired_Tx_Rx": True,
                "air_only_baseline": True,
                "approved_dielectric_stack_sensitivity_required": True,
                "dielectric_cases": [
                    "approved_stack_minimum_permittivity",
                    "approved_stack_nominal_permittivity",
                    "approved_stack_maximum_permittivity",
                ],
                "dielectric_sensitivity_schema": (
                    DIELECTRIC_SENSITIVITY_SCHEMA
                ),
                "approved_stack_schema": (
                    APPROVED_DIELECTRIC_STACK_SCHEMA
                ),
                "example_or_default_permittivity_allowed": False,
                "arbitrary_dielectric_or_TIM_mutation_allowed": False,
                "air_only_result_can_close_final_gate": False,
            },
            "prediction_role": "next_FEA_batch_reranking_only",
            "physical_feasibility_authority": False,
            "automatic_final_promotion_allowed": False,
            "final_turn_graded_symmetric_FEA_required": True,
            "production_eligible": False,
        }
    )
    selection_plan_path = output_dir / "turn_graded_fea_selection_plan.json"
    _write_json(selection_plan_path, selection_plan)
    summary = _seal(
        {
            "schema_version": RANK_SCHEMA,
            "created_at": _now(),
            "model_payload_sha256": model["payload_sha256"],
            "candidate_row_count": len(frame),
            "valid_physics_row_count": len(ranked),
            "selected_geometry_count": len(selected),
            "ranked_csv": {
                "path": str(ranked_csv.resolve()),
                "sha256": _sha256_file(ranked_csv),
            },
            "selection_plan": {
                "path": str(selection_plan_path.resolve()),
                "sha256": _sha256_file(selection_plan_path),
                "payload_sha256": selection_plan["payload_sha256"],
            },
            "raw_two_net_capacitance_used": False,
            "single_0p759701_ratio_used": False,
            "fringing_role": "calibrated_residual_not_fixed_truth",
            "fixed_fringe_coefficient_used": False,
            "dielectric_material_basis": "air_only",
            "approved_dielectric_stack_sensitivity_required": True,
            "prediction_role": "next_FEA_batch_reranking_only",
            "physical_feasibility_authority": False,
            "production_eligible": False,
        }
    )
    _write_json(output_dir / "physics_delta_rerank_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--calibration-manifest", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    rank = subparsers.add_parser("rank")
    rank.add_argument("--model", type=Path, required=True)
    rank.add_argument("--candidates", type=Path, required=True)
    rank.add_argument("--output-dir", type=Path, required=True)
    rank.add_argument("--selection-count", type=int, default=12)
    sensitivity = subparsers.add_parser("sensitivity")
    sensitivity.add_argument("--model", type=Path, required=True)
    sensitivity.add_argument("--candidate-json", type=Path, required=True)
    sensitivity.add_argument(
        "--approved-stack",
        type=Path,
        required=True,
    )
    sensitivity.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fit":
        result = fit_model(
            calibration_manifest_path=args.calibration_manifest,
            output_path=args.output,
        )
    elif args.command == "rank":
        result = rank_candidates(
            model_path=args.model,
            candidates_path=args.candidates,
            output_dir=args.output_dir,
            selection_count=args.selection_count,
        )
    else:
        model = _load_model(args.model)
        candidate = _expanded_candidate_mapping(
            _read_json(args.candidate_json)
        )
        approved_stack = _read_json(args.approved_stack)
        result = dielectric_stack_sensitivity(
            candidate,
            model,
            approved_stack,
        )
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
