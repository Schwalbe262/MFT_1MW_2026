"""Prepare an aggressive, read-only 6/60 primary-resistance neighborhood.

The plan starts from the lowest-primary-temperature lane in the sealed
last-mile batch.  It spends the remaining W on l1, exchanges core depth for a
shorter Y-directed winding path, increases primary foil height while keeping
the foil exactly 5 mm thick, and uses available L for winding-plate spreading.

Selection is deliberately physics-first: every lane must reduce the decoded
``MLT_Tx/(cw1*nwh1)`` resistance proxy by at least six percent and retain a
robust surrogate core upper bound no higher than 120 C.  Surrogate temperature
means and robust upper bounds are reported, but cannot create a PASS.  This
module has no Scheduler payload materialization, apply option, or submit path.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import warnings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from tools import mft_goal_primary_temp_lastmile_hedge as lastmile  # noqa: E402
from tools import mft_goal_primary_temp_next_neighborhood as v2  # noqa: E402
from tools import mft_goal_targeted_symmetric_fea_batch as targeted  # noqa: E402


PLAN_SCHEMA = "mft-goal-primary-temperature-aggressive-plan-v1"
SCREEN_SCHEMA = "mft-goal-primary-temperature-aggressive-screen-v1"
DRY_RUN_SCHEMA = "mft-goal-primary-temperature-aggressive-dry-run-v1"
DEFAULT_SOURCE_PLAN = v2.DEFAULT_SOURCE_PLAN
DEFAULT_SOURCE_SCREEN = v2.DEFAULT_SOURCE_SCREEN
DEFAULT_FINAL528_AUDIT = v2.DEFAULT_FINAL528_AUDIT
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_aggressive_neighborhood_v3"
)
SOURCE_GEOMETRY_SHA256 = v2.SOURCE_GEOMETRY_SHA256
SOURCE_RANK = v2.SOURCE_RANK
COUNT = 12
MINIMUM_COUNT = 8
R_PROXY_MAX_RATIO = 0.94
CORE_ROBUST_MAX_C = 120.0
SIZE_LIMITS_MM = {"W": 1200.0, "L": 1000.0, "H": 750.0}

STRUCTURAL_STENCIL = (
    ("w18_l1_4_l2_1", 4.0, 1.0, 18.0),
    ("w18_l1_4p5_l2_0", 4.5, 0.0, 18.0),
    ("w18p4_l1_4p6_l2_0", 4.6, 0.0, 18.4),
)
# (nwh1 increase, w1 decrease).  Pairs are chosen around the analytic 0.94
# MLT/area boundary; the decoded proxy remains the authoritative gate.
CONDUCTOR_PATH_STENCIL = (
    ("h18_path52", 18.0, 52.0),
    ("h18_path56", 18.0, 56.0),
    ("h22_path44", 22.0, 44.0),
    ("h22_path48", 22.0, 48.0),
    ("h22_path52", 22.0, 52.0),
    ("h24_path40", 24.0, 40.0),
    ("h24_path44", 24.0, 44.0),
    ("h24_path48", 24.0, 48.0),
)
# 10 mm is the validated minimum.  The 11.1 mm source is retained as a
# thermal-path control; 10 mm recovers part of the core cross-section lost
# when w1 is shortened.
CORE_PLATE_STENCIL_MM = (10.0, 11.1)
# Thick WCP lanes intentionally consume some of the L released by shorter w1.
WINDING_PLATE_STENCIL = (
    ("source_plate_contact79", 16.2, 0.79),
    ("plate20_contact79", 20.0, 0.79),
    ("plate22_contact80", 22.0, 0.80),
)


class AggressiveNeighborhoodError(RuntimeError):
    """The sealed authority or aggressive dry-run contract drifted."""


def _fixed_controls(decoded: Mapping[str, Any]) -> None:
    v2._fixed_controls(decoded)  # noqa: SLF001


def _base_metrics(decoded: Mapping[str, Any]) -> dict[str, float]:
    volume, dimensions = geometry_metrics.bounding_box_lit(decoded)
    conductor_area = float(decoded["cw1"]) * float(decoded["nwh1"])
    resistance_proxy = float(decoded["MLT_Tx_mm"]) / conductor_area
    return {
        "primary_conductor_cross_section_mm2": conductor_area,
        "primary_dc_resistance_proxy": resistance_proxy,
        "primary_MLT_mm": float(decoded["MLT_Tx_mm"]),
        "core_effective_area_m2": float(decoded["Ae_effective_m2"]),
        "winding_plate_contact_area_proxy_mm2": (
            2.0
            * float(decoded["wcp_len_x"])
            * float(decoded["nwh1"])
        ),
        "winding_plate_spreading_section_proxy_mm2": (
            float(decoded["wcp_len_x"]) * float(decoded["wcp_t"])
        ),
        "W_mm": float(dimensions[0]),
        "L_mm": float(dimensions[1]),
        "H_mm": float(dimensions[2]),
        "volume_L": float(volume),
    }


def _candidate_pool(
    base: dict[str, Any],
    existing_geometry_sha256: set[str],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, int]]:
    warnings.filterwarnings(
        "ignore",
        message="DataFrame is highly fragmented",
        category=Warning,
    )
    valid, base_frame = validation_check(
        create_input_parameter(base), strict=False
    )
    if not valid:
        raise AggressiveNeighborhoodError("source params are invalid")
    base_decoded = base_frame.iloc[0].to_dict()
    _fixed_controls(base_decoded)
    baseline = _base_metrics(base_decoded)
    counters = {
        "stencil_combinations": 0,
        "decoder_invalid": 0,
        "envelope_invalid": 0,
        "duplicate_existing_or_local": 0,
        "resistance_proxy_above_0p94": 0,
        "accepted_for_surrogate_screen": 0,
    }
    raw: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (
        structural_family,
        l1_delta,
        l2_delta,
        width_growth,
    ) in STRUCTURAL_STENCIL:
        for (
            conductor_path_family,
            nwh1_delta,
            w1_reduction,
        ) in CONDUCTOR_PATH_STENCIL:
            for core_plate_t in CORE_PLATE_STENCIL_MM:
                for (
                    winding_plate_family,
                    wcp_t,
                    contact_fraction,
                ) in WINDING_PLATE_STENCIL:
                    counters["stencil_combinations"] += 1
                    params = dict(base)
                    params.update(
                        {
                            "l1": round(
                                float(base["l1"]) + l1_delta, 1
                            ),
                            "l2": round(
                                float(base["l2"]) + l2_delta, 1
                            ),
                            "h1": round(
                                float(base["h1"]) - 2.0 * l1_delta,
                                1,
                            ),
                            "nwh1": round(
                                float(base["nwh1"]) + nwh1_delta, 1
                            ),
                            "w1": round(
                                float(base["w1"]) - w1_reduction, 1
                            ),
                            "core_plate_t": core_plate_t,
                            "wcp_t": wcp_t,
                        }
                    )
                    try:
                        valid, decoded_frame = validation_check(
                            create_input_parameter(params), strict=False
                        )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        valid = False
                    if not valid:
                        counters["decoder_invalid"] += 1
                        continue
                    reference_length = float(
                        decoded_frame.iloc[0]["wcp_len_ref_x"]
                    )
                    params["wcp_len_x"] = round(
                        contact_fraction * reference_length, 1
                    )
                    try:
                        valid, decoded_frame = validation_check(
                            create_input_parameter(params), strict=False
                        )
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        OverflowError,
                    ):
                        valid = False
                    if not valid:
                        counters["decoder_invalid"] += 1
                        continue
                    decoded = {
                        key: targeted._builtin(value)  # noqa: SLF001
                        for key, value in decoded_frame.iloc[0]
                        .to_dict()
                        .items()
                    }
                    _fixed_controls(decoded)
                    volume, dimensions = geometry_metrics.bounding_box_lit(
                        decoded
                    )
                    if any(
                        float(value) > SIZE_LIMITS_MM[axis] + 1e-9
                        for axis, value in zip(
                            ("W", "L", "H"), dimensions
                        )
                    ):
                        counters["envelope_invalid"] += 1
                        continue
                    geometry_sha = targeted._geometry_sha(  # noqa: SLF001
                        decoded
                    )
                    if (
                        geometry_sha in existing_geometry_sha256
                        or geometry_sha in seen
                    ):
                        counters["duplicate_existing_or_local"] += 1
                        continue
                    seen.add(geometry_sha)
                    conductor_area = (
                        float(decoded["cw1"]) * float(decoded["nwh1"])
                    )
                    resistance_proxy = (
                        float(decoded["MLT_Tx_mm"]) / conductor_area
                    )
                    resistance_ratio = (
                        resistance_proxy
                        / baseline["primary_dc_resistance_proxy"]
                    )
                    if resistance_ratio > R_PROXY_MAX_RATIO + 1e-12:
                        counters["resistance_proxy_above_0p94"] += 1
                        continue
                    projected = {
                        key: targeted._builtin(decoded[key])  # noqa: SLF001
                        for key in sorted(ALL_INPUT_KEYS)
                    }
                    core_area = float(decoded["Ae_effective_m2"])
                    contact_proxy = (
                        2.0
                        * float(decoded["wcp_len_x"])
                        * float(decoded["nwh1"])
                    )
                    spreading_proxy = (
                        float(decoded["wcp_len_x"])
                        * float(decoded["wcp_t"])
                    )
                    raw.append(
                        {
                            "decoded": decoded,
                            "params": projected,
                            "params_sha256": targeted._sha(  # noqa: SLF001
                                projected
                            ),
                            "physical_geometry_sha256": geometry_sha,
                            "dimensions": [
                                float(value) for value in dimensions
                            ],
                            "envelope_margins_mm": {
                                axis: SIZE_LIMITS_MM[axis] - float(value)
                                for axis, value in zip(
                                    ("W", "L", "H"), dimensions
                                )
                            },
                            "volume_L": float(volume),
                            "mutations": {
                                "structural_family": structural_family,
                                "conductor_path_family": (
                                    conductor_path_family
                                ),
                                "winding_plate_family": (
                                    winding_plate_family
                                ),
                                "l1_core_section_delta_mm": l1_delta,
                                "l2_window_delta_mm": l2_delta,
                                "width_growth_mm": width_growth,
                                "h1_window_height_delta_mm": (
                                    -2.0 * l1_delta
                                ),
                                "nwh1_conductor_height_delta_mm": (
                                    nwh1_delta
                                ),
                                "w1_y_path_reduction_mm": w1_reduction,
                                "core_plate_t_mm": core_plate_t,
                                "wcp_t_mm": wcp_t,
                                "wcp_contact_fraction": contact_fraction,
                                "wcp_len_x_mm": projected["wcp_len_x"],
                                "vertical_clearance_each_mm": (
                                    (
                                        float(decoded["h1"])
                                        - float(decoded["nwh1"])
                                    )
                                    / 2.0
                                ),
                            },
                            "mechanism_proxies": {
                                "primary_MLT_mm": float(
                                    decoded["MLT_Tx_mm"]
                                ),
                                "primary_MLT_reduction_percent": (
                                    100.0
                                    * (
                                        1.0
                                        - float(decoded["MLT_Tx_mm"])
                                        / baseline["primary_MLT_mm"]
                                    )
                                ),
                                "primary_conductor_cross_section_mm2": (
                                    conductor_area
                                ),
                                "primary_conductor_area_gain_percent": (
                                    100.0
                                    * (
                                        conductor_area
                                        / baseline[
                                            "primary_conductor_cross_section_mm2"
                                        ]
                                        - 1.0
                                    )
                                ),
                                "primary_dc_resistance_proxy": (
                                    resistance_proxy
                                ),
                                "primary_dc_resistance_proxy_ratio": (
                                    resistance_ratio
                                ),
                                "primary_dc_resistance_proxy_reduction_percent": (
                                    100.0 * (1.0 - resistance_ratio)
                                ),
                                "core_effective_area_m2": core_area,
                                "core_effective_area_ratio": (
                                    core_area
                                    / baseline["core_effective_area_m2"]
                                ),
                                "core_effective_area_change_percent": (
                                    100.0
                                    * (
                                        core_area
                                        / baseline["core_effective_area_m2"]
                                        - 1.0
                                    )
                                ),
                                "winding_plate_contact_area_proxy_mm2": (
                                    contact_proxy
                                ),
                                "winding_plate_contact_area_gain_percent": (
                                    100.0
                                    * (
                                        contact_proxy
                                        / baseline[
                                            "winding_plate_contact_area_proxy_mm2"
                                        ]
                                        - 1.0
                                    )
                                ),
                                "winding_plate_spreading_section_proxy_mm2": (
                                    spreading_proxy
                                ),
                                "winding_plate_spreading_gain_percent": (
                                    100.0
                                    * (
                                        spreading_proxy
                                        / baseline[
                                            "winding_plate_spreading_section_proxy_mm2"
                                        ]
                                        - 1.0
                                    )
                                ),
                            },
                        }
                    )
    counters["accepted_for_surrogate_screen"] = len(raw)
    if len(raw) < MINIMUM_COUNT:
        raise AggressiveNeighborhoodError(
            f"only {len(raw)} geometry-valid R-proxy candidates exist"
        )
    return raw, baseline, counters


def _selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    proxy = row["mechanism_proxies"]
    return (
        float(proxy["primary_dc_resistance_proxy_ratio"]),
        float(row["secondary_winding_robust_max_C"]),
        float(row["core_robust_max_C"]),
        -float(proxy["core_effective_area_ratio"]),
        str(row["physical_geometry_sha256"]),
    )


def _select(screened: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in screened
        if row["fixed_lm2mh_replay"]["fmin_Hz"]
        >= targeted.RESONANCE_MIN_HZ
        and row["corrected_Crx_transfer_estimate_F"] <= 0.55e-9
        and row["mechanism_proxies"][
            "primary_dc_resistance_proxy_ratio"
        ]
        <= R_PROXY_MAX_RATIO + 1e-12
        and row["core_robust_max_C"] <= CORE_ROBUST_MAX_C
    ]
    if len(eligible) < MINIMUM_COUNT:
        return []
    return sorted(eligible, key=_selection_key)[:COUNT]


def _without_large_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in {"decoded", "params"}
    }


def prepare(
    *,
    source_plan_path: Path,
    source_screen_path: Path,
    final528_audit_path: Path,
    output: Path,
) -> Path:
    source_plan, source_root, _profile = targeted._load_plan(  # noqa: SLF001
        source_plan_path
    )
    source_screen = targeted._validate_seal(  # noqa: SLF001
        targeted._read_json(source_screen_path),  # noqa: SLF001
        lastmile.SCHEMA,
    )
    if (
        source_screen.get("plan_payload_sha256")
        != source_plan["payload_sha256"]
        or source_screen.get("candidate_pool_count") != 72
        or source_screen.get("selected_count") != 8
        or source_screen.get("scheduler_mutation_performed") is not False
    ):
        raise AggressiveNeighborhoodError("last-mile screen drifted")
    source_lanes = [
        lane
        for lane in source_plan["lanes"]
        if lane["candidate"]["physical_geometry_sha256"]
        == SOURCE_GEOMETRY_SHA256
    ]
    if (
        len(source_lanes) != 1
        or int(source_lanes[0]["rank"]) != SOURCE_RANK
    ):
        raise AggressiveNeighborhoodError(
            "minimum-primary source lane drifted"
        )
    source_lane = source_lanes[0]
    base_path = (
        source_root / source_lane["params"]["path"]
    ).resolve(strict=True)
    base = targeted._read_json(base_path)  # noqa: SLF001

    authority, final528_hashes = v2._final528_authority(  # noqa: SLF001
        final528_audit_path
    )
    lastmile_screen_hashes = {
        str(row["physical_geometry_sha256"])
        for row in source_screen.get("rows") or []
    }
    if len(lastmile_screen_hashes) != 72:
        raise AggressiveNeighborhoodError(
            "last-mile geometry inventory drifted"
        )
    existing = set(final528_hashes)
    existing.update(lastmile_screen_hashes)
    existing.update(
        lane["candidate"]["physical_geometry_sha256"]
        for lane in source_plan["lanes"]
    )
    raw, base_metrics, counters = _candidate_pool(base, existing)

    import pandas as pd

    frame = pd.DataFrame([item["decoded"] for item in raw])
    means, upper, model_identity = targeted._n1_6_cooler_predictions(  # noqa: SLF001
        frame
    )
    calibration_ratio, calibration = (
        targeted._turn_graded_crx_calibration()  # noqa: SLF001
    )
    baseline_candidate = source_lane["candidate"]
    baseline_primary = float(
        baseline_candidate["screening_primary_winding_max_C"]
    )
    baseline_secondary = float(
        baseline_candidate["screening_secondary_winding_max_C"]
    )
    baseline_core = float(
        baseline_candidate["screening_core_max_C"]
    )

    screened: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        primary_mean = max(
            float(means[name][index])
            for name in targeted.PRIMARY_WINDING_TEMPERATURES
        )
        primary_upper = max(
            float(upper[name][index])
            for name in targeted.PRIMARY_WINDING_TEMPERATURES
        )
        secondary_mean = max(
            float(means[name][index])
            for name in targeted.SECONDARY_WINDING_TEMPERATURES
        )
        secondary_upper = max(
            float(upper[name][index])
            for name in targeted.SECONDARY_WINDING_TEMPERATURES
        )
        core_mean = max(
            float(means[name][index])
            for name in targeted.CORE_TEMPERATURES
        )
        core_upper = max(
            float(upper[name][index])
            for name in targeted.CORE_TEMPERATURES
        )
        corrected_crx = (
            float(means["C_rx_rx_F"][index]) * calibration_ratio
        )
        replay = targeted._fixed_lm_resonance(  # noqa: SLF001
            llt_uH=float(means["Llt_phys"][index]),
            c_tx_F=float(means["C_tx_tx_F"][index]),
            c_rx_F=corrected_crx,
            c_inter_F=float(means["C_tx_rx_F"][index]),
            n1=6,
            n2=60,
        )
        screened.append(
            {
                **item,
                "primary_winding_mean_max_C": primary_mean,
                "primary_winding_robust_max_C": primary_upper,
                "secondary_winding_mean_max_C": secondary_mean,
                "secondary_winding_robust_max_C": secondary_upper,
                "core_mean_max_C": core_mean,
                "core_robust_max_C": core_upper,
                "primary_robust_change_from_source_C": (
                    primary_upper - baseline_primary
                ),
                "secondary_robust_change_from_source_C": (
                    secondary_upper - baseline_secondary
                ),
                "core_robust_change_from_source_C": (
                    core_upper - baseline_core
                ),
                "preserves_source_secondary_robust_margin": (
                    secondary_upper <= baseline_secondary + 1e-9
                ),
                "corrected_Crx_transfer_estimate_F": corrected_crx,
                "fixed_lm2mh_replay": replay,
                "P_Tx_main_group_W": float(
                    means["P_Tx_main_group"][index]
                ),
                "P_winding_total_W": float(
                    means["P_winding_total"][index]
                ),
                "P_core_total_W": float(means["P_core_total"][index]),
                "P_total_screen_W": (
                    float(means["P_winding_total"][index])
                    + float(means["P_core_total"][index])
                ),
                "surrogate_pass_claimed": False,
                "actual_symmetric_FEA_required": True,
            }
        )
    selected = _select(screened)
    core_safe = [
        row
        for row in screened
        if row["core_robust_max_C"] <= CORE_ROBUST_MAX_C
    ]
    secondary_preserving = [
        row
        for row in core_safe
        if row["preserves_source_secondary_robust_margin"]
    ]
    if not selected:
        best_core = min(
            screened, key=lambda row: row["core_robust_max_C"]
        )
        best_proxy = min(
            screened,
            key=lambda row: row["mechanism_proxies"][
                "primary_dc_resistance_proxy_ratio"
            ],
        )
        destination = output.resolve()
        if destination.exists():
            raise AggressiveNeighborhoodError(
                f"output already exists: {destination}"
            )
        destination.mkdir(parents=True)
        threshold_counts = {
            f"robust_core_at_most_{threshold:.0f}C": sum(
                row["core_robust_max_C"] <= threshold
                for row in screened
            )
            for threshold in (120.0, 122.0, 124.0, 126.0, 128.0, 130.0)
        }
        diagnostic = targeted._seal(  # noqa: SLF001
            {
                "schema_version": SCREEN_SCHEMA,
                "created_at_utc": targeted._now(),  # noqa: SLF001
                "classification": (
                    "sealed-aggressive-no-core-safe-candidate-dry-run"
                ),
                "authority": authority,
                "source_lastmile": {
                    "plan": targeted._file_record(  # noqa: SLF001
                        source_plan_path
                    ),
                    "plan_payload_sha256": source_plan[
                        "payload_sha256"
                    ],
                    "screen": targeted._file_record(  # noqa: SLF001
                        source_screen_path
                    ),
                    "screen_payload_sha256": source_screen[
                        "payload_sha256"
                    ],
                    "source_rank": SOURCE_RANK,
                    "source_geometry_sha256": SOURCE_GEOMETRY_SHA256,
                },
                "fixed_contract": {
                    "size_limits_mm": copy.deepcopy(SIZE_LIMITS_MM),
                    "primary_conductor_thickness_mm": 5.0,
                    "primary_interturn_gap_mm": 1.6,
                    "turns": {"N1": 6, "N2": 60},
                    "Lm_primary_referred_H": 0.002,
                    "fan_velocity_m_s": 1.5,
                    "fan_config": "dual",
                    "TIM_conductivity_W_mK": 0.2,
                    "core_plate_pad_t_mm": 2.0,
                    "wcp_pad_t_mm": 2.0,
                },
                "source_baseline": {
                    "dimensions_mm": {
                        "W": base_metrics["W_mm"],
                        "L": base_metrics["L_mm"],
                        "H": base_metrics["H_mm"],
                    },
                    "primary_winding_robust_max_C": baseline_primary,
                    "secondary_winding_robust_max_C": baseline_secondary,
                    "core_robust_max_C": baseline_core,
                    "mechanism_proxies": base_metrics,
                },
                "candidate_generation_counters": counters,
                "surrogate_screen_count": len(screened),
                "selected_candidate_count": 0,
                "candidate_plan_created": False,
                "R_proxy_max_ratio": R_PROXY_MAX_RATIO,
                "minimum_R_proxy_reduction_percent": 6.0,
                "robust_core_max_C": CORE_ROBUST_MAX_C,
                "active_constraint": {
                    "name": "robust_core_temperature_max_C",
                    "limit_C": CORE_ROBUST_MAX_C,
                    "core_safe_count": len(core_safe),
                    "threshold_counts": threshold_counts,
                    "minimum_observed_robust_core_C": best_core[
                        "core_robust_max_C"
                    ],
                    "minimum_excess_C": (
                        best_core["core_robust_max_C"]
                        - CORE_ROBUST_MAX_C
                    ),
                    "lowest_R_proxy_ratio": best_proxy[
                        "mechanism_proxies"
                    ]["primary_dc_resistance_proxy_ratio"],
                    "lowest_R_proxy_reduction_percent": (
                        best_proxy["mechanism_proxies"][
                            "primary_dc_resistance_proxy_reduction_percent"
                        ]
                    ),
                    "lowest_R_candidate_robust_core_C": best_proxy[
                        "core_robust_max_C"
                    ],
                    "lowest_R_candidate_core_excess_C": (
                        best_proxy["core_robust_max_C"]
                        - CORE_ROBUST_MAX_C
                    ),
                    "conclusion": (
                        "No geometry may be promoted into an 8-16 lane "
                        "plan under simultaneous Rproxy<=0.94 and robust "
                        "core<=120C. Relaxing the core gate is prohibited."
                    ),
                },
                "best_core_candidate": _without_large_fields(best_core),
                "lowest_R_proxy_candidate": _without_large_fields(
                    best_proxy
                ),
                "surrogate_model_identity": model_identity,
                "turn_graded_Crx_calibration": calibration,
                "rows": [
                    _without_large_fields(row)
                    for row in sorted(screened, key=_selection_key)
                ],
                "submission_ready": False,
                "apply_supported": False,
                "scheduler_payloads_materialized": False,
                "scheduler_POST_performed": False,
                "scheduler_submission_performed": False,
                "scheduler_repository_modified": False,
                "scheduler_project_configuration_modified": False,
                "surrogate_pass_claimed": False,
                "actual_symmetric_FEA_required_for_pass": True,
                "production_eligible": False,
            }
        )
        report_path = targeted._atomic_json(  # noqa: SLF001
            destination / "no_candidate_report.json", diagnostic
        )
        dry_run = targeted._seal(  # noqa: SLF001
            {
                "schema_version": DRY_RUN_SCHEMA,
                "created_at_utc": targeted._now(),  # noqa: SLF001
                "report": targeted._file_record(report_path),  # noqa: SLF001
                "report_payload_sha256": diagnostic["payload_sha256"],
                "candidate_count": 0,
                "candidate_plan_created": False,
                "active_constraint": (
                    "robust_core_temperature_max_C"
                ),
                "all_geometry_and_envelope_checks_replayed": True,
                "all_screened_R_proxy_ratios_at_most_0p94": True,
                "surrogate_pass_claimed": False,
                "scheduler_payloads_materialized": False,
                "scheduler_POST_performed": False,
                "scheduler_repository_modified": False,
                "apply_supported": False,
                "classification": (
                    "sealed-aggressive-no-candidate-dry-run"
                ),
            }
        )
        targeted._atomic_json(  # noqa: SLF001
            destination / "dry_run.json", dry_run
        )
        return report_path

    destination = output.resolve()
    if destination.exists():
        raise AggressiveNeighborhoodError(
            f"output already exists: {destination}"
        )
    destination.mkdir(parents=True)
    lanes = []
    for rank, item in enumerate(selected, start=1):
        stem = item["physical_geometry_sha256"][:12]
        params_path = targeted._atomic_json(  # noqa: SLF001
            destination / "params" / f"rank-{rank:02d}-{stem}.json",
            item["params"],
        )
        lanes.append(
            {
                "rank": rank,
                "selection_role": (
                    "minimum_physical_primary_resistance_proxy_with_"
                    "robust_core_gate"
                ),
                "candidate": {
                    "physical_geometry_sha256": item[
                        "physical_geometry_sha256"
                    ],
                    "params_sha256": item["params_sha256"],
                    "dimensions_mm": {
                        "W_drawing_x": item["dimensions"][0],
                        "L_perpendicular_y": item["dimensions"][1],
                        "H": item["dimensions"][2],
                    },
                    "envelope_margins_mm": copy.deepcopy(
                        item["envelope_margins_mm"]
                    ),
                    "objective_volume_L": item["volume_L"],
                    "mechanism_proxies": copy.deepcopy(
                        item["mechanism_proxies"]
                    ),
                    "neighborhood_mutations": copy.deepcopy(
                        item["mutations"]
                    ),
                    "surrogate_screen": {
                        key: copy.deepcopy(item[key])
                        for key in (
                            "primary_winding_mean_max_C",
                            "primary_winding_robust_max_C",
                            "secondary_winding_mean_max_C",
                            "secondary_winding_robust_max_C",
                            "core_mean_max_C",
                            "core_robust_max_C",
                            "primary_robust_change_from_source_C",
                            "secondary_robust_change_from_source_C",
                            "core_robust_change_from_source_C",
                            "preserves_source_secondary_robust_margin",
                            "corrected_Crx_transfer_estimate_F",
                            "fixed_lm2mh_replay",
                            "P_Tx_main_group_W",
                            "P_winding_total_W",
                            "P_core_total_W",
                            "P_total_screen_W",
                        )
                    },
                    "surrogate_pass_claimed": False,
                    "actual_symmetric_FEA_required": True,
                    "production_eligible": False,
                },
                "params": targeted._file_record(  # noqa: SLF001
                    params_path, relative_to=destination
                ),
            }
        )

    plan = targeted._seal(  # noqa: SLF001
        {
            "schema_version": PLAN_SCHEMA,
            "created_at_utc": targeted._now(),  # noqa: SLF001
            "classification": (
                "read-only-aggressive-symmetric-fea-acquisition-design"
            ),
            "authority": authority,
            "source_lastmile": {
                "plan": targeted._file_record(source_plan_path),  # noqa: SLF001
                "plan_payload_sha256": source_plan["payload_sha256"],
                "screen": targeted._file_record(source_screen_path),  # noqa: SLF001
                "screen_payload_sha256": source_screen["payload_sha256"],
                "source_rank": SOURCE_RANK,
                "source_geometry_sha256": SOURCE_GEOMETRY_SHA256,
            },
            "hard_spec": {
                "size_limits_mm": copy.deepcopy(SIZE_LIMITS_MM),
                "self_resonance_min_Hz": 15_000.0,
                "primary_winding_temperature_max_C": 100.0,
                "secondary_winding_temperature_max_C": 100.0,
                "core_temperature_max_C": 120.0,
                "primary_conductor_thickness_mm": 5.0,
                "primary_interturn_gap_mm": 1.6,
                "turns": {"N1": 6, "N2": 60},
                "Lm_primary_referred_H": 0.002,
                "fan_velocity_m_s": 1.5,
                "fan_config": "dual",
                "TIM_conductivity_W_mK": 0.2,
                "core_plate_pad_t_mm": 2.0,
                "wcp_pad_t_mm": 2.0,
                "axis_swap_allowed": False,
            },
            "source_baseline": {
                "dimensions_mm": {
                    "W": base_metrics["W_mm"],
                    "L": base_metrics["L_mm"],
                    "H": base_metrics["H_mm"],
                },
                "unused_envelope_mm": {
                    axis: SIZE_LIMITS_MM[axis] - base_metrics[f"{axis}_mm"]
                    for axis in ("W", "L", "H")
                },
                "primary_winding_robust_max_C": baseline_primary,
                "secondary_winding_robust_max_C": baseline_secondary,
                "core_robust_max_C": baseline_core,
                "mechanism_proxies": base_metrics,
            },
            "selection": {
                "candidate_generation_counters": counters,
                "surrogate_screen_count": len(screened),
                "robust_core_safe_count": len(core_safe),
                "source_secondary_margin_preserving_count": len(
                    secondary_preserving
                ),
                "selected_candidate_count": len(selected),
                "R_proxy_max_ratio": R_PROXY_MAX_RATIO,
                "minimum_R_proxy_reduction_percent": 6.0,
                "robust_core_max_C": CORE_ROBUST_MAX_C,
                "selection_order": (
                    "minimum decoded MLT_Tx/(cw1*nwh1) ratio first; "
                    "then secondary robust temperature; then core robust "
                    "temperature; actual symmetric FEA still required"
                ),
                "final528_geometry_hashes_excluded": len(final528_hashes),
                "lastmile_screen_geometry_hashes_excluded": len(
                    lastmile_screen_hashes
                ),
                "all_selected_geometries_are_new": True,
            },
            "active_constraint_analysis": {
                "primary_resistance_proxy": (
                    "Rdc proportional to MLT_Tx/(5mm*nwh1); decoded ratio "
                    "must be <=0.94 before surrogate inference."
                ),
                "width": (
                    "4*l1+2*l2 growth is 18.0 or 18.4 mm versus "
                    "18.596 mm available; W arithmetic margin is retained."
                ),
                "height": (
                    "h1 decreases by 2*dl1, holding H=750 mm. nwh1 +24 "
                    "leaves only about 2.85-3.45 mm clearance per end and "
                    "is therefore an aggressive FEA acquisition, not a "
                    "manufacturing release."
                ),
                "length": (
                    "w1 reductions free L; 20/22 mm winding plates consume "
                    "part of that margin for spreading while all lanes stay "
                    "within L<=1000 mm."
                ),
                "core": (
                    "shorter w1 reduces core depth; l1 growth and the "
                    "validated-minimum 10 mm core plate recover part of Ae. "
                    "Every selected lane is gated by robust core<=120C."
                ),
                "secondary": (
                    "secondary robust temperature is prioritized after the "
                    "R-proxy/core gates. Preservation relative to the source "
                    "is reported explicitly and never assumed."
                ),
            },
            "surrogate_model_identity": model_identity,
            "turn_graded_Crx_calibration": calibration,
            "lanes": lanes,
            "submission_ready": False,
            "apply_supported": False,
            "scheduler_payloads_materialized": False,
            "scheduler_POST_performed": False,
            "scheduler_submission_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_project_configuration_modified": False,
            "surrogate_pass_claimed": False,
            "actual_symmetric_FEA_required_for_pass": True,
            "production_eligible": False,
        }
    )
    plan_path = targeted._atomic_json(  # noqa: SLF001
        destination / "dry_run_plan.json", plan
    )
    dry_run = targeted._seal(  # noqa: SLF001
        {
            "schema_version": DRY_RUN_SCHEMA,
            "created_at_utc": targeted._now(),  # noqa: SLF001
            "plan": targeted._file_record(plan_path),  # noqa: SLF001
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_count": len(lanes),
            "all_fixed_controls_replayed": True,
            "all_geometry_and_envelope_checks_replayed": True,
            "all_selected_R_proxy_ratios_at_most_0p94": True,
            "all_selected_robust_core_at_most_120C": True,
            "all_geometry_hashes_new_against_final528_and_lastmile": True,
            "surrogate_pass_claimed": False,
            "scheduler_payloads_materialized": False,
            "scheduler_POST_performed": False,
            "scheduler_repository_modified": False,
            "apply_supported": False,
            "classification": "sealed-aggressive-dry-run-only",
        }
    )
    dry_path = targeted._atomic_json(  # noqa: SLF001
        destination / "dry_run.json", dry_run
    )
    report = targeted._seal(  # noqa: SLF001
        {
            "schema_version": SCREEN_SCHEMA,
            "created_at_utc": targeted._now(),  # noqa: SLF001
            "plan": targeted._file_record(plan_path),  # noqa: SLF001
            "plan_payload_sha256": plan["payload_sha256"],
            "dry_run": targeted._file_record(dry_path),  # noqa: SLF001
            "candidate_generation_counters": counters,
            "surrogate_screen_count": len(screened),
            "robust_core_safe_count": len(core_safe),
            "source_secondary_margin_preserving_count": len(
                secondary_preserving
            ),
            "selected_count": len(selected),
            "selected_geometry_sha256": [
                row["physical_geometry_sha256"] for row in selected
            ],
            "rows": [
                _without_large_fields(row)
                for row in sorted(screened, key=_selection_key)
            ],
            "scheduler_mutation_performed": False,
            "surrogate_pass_claimed": False,
            "production_eligible": False,
        }
    )
    targeted._atomic_json(  # noqa: SLF001
        destination / "screen_report.json", report
    )
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-plan", type=Path, default=DEFAULT_SOURCE_PLAN
    )
    parser.add_argument(
        "--source-screen", type=Path, default=DEFAULT_SOURCE_SCREEN
    )
    parser.add_argument(
        "--final528-audit", type=Path, default=DEFAULT_FINAL528_AUDIT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    path = prepare(
        source_plan_path=args.source_plan,
        source_screen_path=args.source_screen,
        final528_audit_path=args.final528_audit,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": (
                    "no_core_safe_candidate"
                    if path.name == "no_candidate_report.json"
                    else "ok"
                ),
                "classification": (
                    "sealed-aggressive-no-candidate-dry-run"
                    if path.name == "no_candidate_report.json"
                    else "sealed-aggressive-dry-run-only"
                ),
                "scheduler_POST_performed": False,
                "path": str(path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
