"""Prepare a sealed, non-submittable 6/60 primary-thermal neighborhood.

This tool is intentionally a design-only acquisition planner.  It authenticates
the final-528 global NSGA audit and the completed primary-temperature last-mile
screen, builds geometries that are absent from both inventories, and evaluates
them with the authenticated surrogate generation.  It writes parameter files,
a screen report, and a dry-run plan, but contains no Scheduler submission path.

The fixed 5 mm primary foil thickness, 1.6 mm primary interturn gap, 1.5 m/s
dual-fan boundary, 0.2 W/(m K) TIM, 2 mm pads, 6/60 turns, 2 mH resonance
replay, and W/L/H limits are rechecked for every candidate.  Surrogate results
are acquisition-ranking evidence only and never constitute a thermal PASS.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
from tools import mft_goal_final528_audit as final528  # noqa: E402
from tools import mft_goal_primary_temp_lastmile_hedge as lastmile  # noqa: E402
from tools import mft_goal_targeted_symmetric_fea_batch as targeted  # noqa: E402


PLAN_SCHEMA = "mft-goal-primary-temperature-next-neighborhood-plan-v1"
SCREEN_SCHEMA = "mft-goal-primary-temperature-next-neighborhood-screen-v1"
DRY_RUN_SCHEMA = "mft-goal-primary-temperature-next-neighborhood-dry-run-v1"
DEFAULT_SOURCE_PLAN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_lastmile_v2\batch_plan.json"
)
DEFAULT_SOURCE_SCREEN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_lastmile_v2\screen_report.json"
)
DEFAULT_FINAL528_AUDIT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_old16_plus_splittemp512_global_nds_v3"
    r"\global_pareto_independent_audit.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_next_neighborhood_v2"
)
SOURCE_GEOMETRY_SHA256 = (
    "b31c08a13f3a4812af0b3839292e9363e706423cebdf888ecd228f520e570bd8"
)
SOURCE_RANK = 5
COUNT = 12
SIZE_LIMITS_MM = {"W": 1200.0, "L": 1000.0, "H": 750.0}
MIN_VERTICAL_CLEARANCE_EACH_MM = 5.5

# The width growth is 4*dl1 + 2*dl2 for this topology.  The 16 mm family
# retains 2.596 mm W margin; the 18 mm families spend 18.0 of the available
# 18.596 mm while preserving a positive 0.596 mm arithmetic margin.
STRUCTURAL_STENCIL = (
    ("w16_l1_4_l2_0", 4.0, 0.0, 16.0),
    ("w18_l1_3_l2_3", 3.0, 3.0, 18.0),
    ("w18_l1_4_l2_1", 4.0, 1.0, 18.0),
    ("w18_l1_4p5_l2_0", 4.5, 0.0, 18.0),
)
PRIMARY_HEIGHT_DELTAS_MM = (12.0, 16.0, 18.0)
# The source already shortened w1 by 4 mm versus the last-mile rank-1
# geometry.  Further 12/16/20 mm reductions shorten the primary mean turn
# path.  Extra l1 replaces the resulting core-area loss using otherwise unused
# W.  The 17.2 mm plate hedge tests spreading/contact at the same short path.
PLATE_PATH_STENCIL = (
    ("balanced_short_path", 16.2, 12.0, 0.79),
    ("short_path", 16.2, 16.0, 0.79),
    ("maximum_short_path", 16.2, 20.0, 0.79),
    ("plate_spreader_contact80", 17.2, 20.0, 0.80),
)


class NeighborhoodError(RuntimeError):
    """The authority, fixed controls, or dry-run design contract drifted."""


def _file_sha(path: Path) -> str:
    return targeted._sha_file(path)  # noqa: SLF001


def _record_from_audit(
    audit: Mapping[str, Any], name: str
) -> dict[str, Any]:
    records = audit.get("authoritative_files")
    record = records.get(name) if isinstance(records, dict) else None
    if not isinstance(record, dict):
        raise NeighborhoodError(f"final528 audit record absent: {name}")
    path = Path(str(record.get("path") or "")).resolve(strict=True)
    if (
        not path.is_file()
        or int(record.get("row_count") or -1) < 0
        or len(str(record.get("sha256") or "")) != 64
    ):
        raise NeighborhoodError(f"final528 audit record invalid: {name}")
    return {
        "path": str(path),
        "sha256": str(record["sha256"]),
        "row_count": int(record["row_count"]),
        "size_bytes": path.stat().st_size,
    }


def _final528_authority(
    audit_path: Path,
) -> tuple[dict[str, Any], set[str]]:
    audit = final528._read_sealed(  # noqa: SLF001
        audit_path.resolve(strict=True), final528.AUDIT_SCHEMA
    )
    controls = audit.get("fixed_controls")
    if (
        audit.get("campaign_id") != final528.CAMPAIGN_ID
        or audit.get("classification") != "independent-final528-audit"
        or audit.get("global_all_row_aggregation_verified") is not True
        or audit.get("global_non_dominated_sorting_recomputed") is not True
        or int(audit.get("selected_logical_seed_count", -1)) != 528
        or int(audit.get("raw_terminal_row_count", -1)) != 168_960
        or int(audit.get("geometry_deduplicated_candidate_count", -1))
        != 2_849
        or int(audit.get("production_hard_feasible_pareto_count", -1))
        != 0
        or controls
        != {
            "N1_strata": [5, 6, 7, 8],
            "primary_conductor_thickness_mm": 5.0,
            "primary_interturn_gap_mm": 1.6,
        }
    ):
        raise NeighborhoodError("final528 independent audit drifted")
    candidates = _record_from_audit(audit, "global_terminal_candidates.csv")
    raw = _record_from_audit(audit, "global_terminal_all_seeds_raw.csv")
    candidate_path = Path(candidates["path"])
    if (
        candidates["row_count"] != 2_849
        or raw["row_count"] != 168_960
        or _file_sha(candidate_path) != candidates["sha256"]
    ):
        raise NeighborhoodError("final528 candidate inventory drifted")
    hashes: set[str] = set()
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=1):
            identity = str(row.get("physical_geometry_sha256") or "")
            if len(identity) != 64:
                raise NeighborhoodError(
                    f"final528 row {row_number} has invalid geometry identity"
                )
            hashes.add(identity)
    if len(hashes) != candidates["row_count"]:
        raise NeighborhoodError("final528 geometry identities are not unique")
    return {
        "audit": targeted._file_record(audit_path),  # noqa: SLF001
        "audit_payload_sha256": audit["payload_sha256"],
        "collector_status_payload_sha256": audit[
            "collector_status_payload_sha256"
        ],
        "pareto_manifest_payload_sha256": audit[
            "pareto_manifest_payload_sha256"
        ],
        "selected_logical_seed_count": 528,
        "raw_terminal_row_count": 168_960,
        "geometry_deduplicated_candidate_count": 2_849,
        "production_hard_feasible_pareto_count": 0,
        "global_non_dominated_sorting_recomputed": True,
        "raw_terminal_rows": raw,
        "deduplicated_candidates": candidates,
    }, hashes


def _fixed_controls(decoded: Mapping[str, Any]) -> None:
    expected = {
        "N1": 6,
        "N2": 60,
        "N1_main": 6,
        "N1_side": 0,
        "cw1": 5.0,
        "gap1": 1.6,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
        "core_center_gap_mm": 0.0,
    }
    for name, value in expected.items():
        observed = decoded.get(name)
        if isinstance(value, float):
            matches = math.isclose(
                float(observed), value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == value
        if not matches:
            raise NeighborhoodError(
                f"fixed topology/cooling/TIM escaped: {name}"
            )


def _candidate_pool(
    base: dict[str, Any],
    existing_geometry_sha256: set[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    warnings.filterwarnings(
        "ignore",
        message="DataFrame is highly fragmented",
        category=Warning,
    )
    valid, base_frame = validation_check(
        create_input_parameter(base), strict=False
    )
    if not valid:
        raise NeighborhoodError(
            "source minimum-primary last-mile params are invalid"
        )
    base_decoded = base_frame.iloc[0].to_dict()
    _fixed_controls(base_decoded)
    base_volume, base_dimensions = geometry_metrics.bounding_box_lit(
        base_decoded
    )
    base_metrics = {
        "primary_conductor_cross_section_mm2": (
            float(base_decoded["cw1"]) * float(base_decoded["nwh1"])
        ),
        "primary_dc_resistance_proxy": (
            float(base_decoded["MLT_Tx_mm"])
            / (
                float(base_decoded["cw1"])
                * float(base_decoded["nwh1"])
            )
        ),
        "core_effective_area_m2": float(base_decoded["Ae_effective_m2"]),
        "winding_plate_contact_area_proxy_mm2": (
            2.0
            * float(base_decoded["wcp_len_x"])
            * float(base_decoded["nwh1"])
        ),
        "winding_plate_spreading_section_proxy_mm2": (
            float(base_decoded["wcp_len_x"])
            * float(base_decoded["wcp_t"])
        ),
        "W_mm": float(base_dimensions[0]),
        "L_mm": float(base_dimensions[1]),
        "H_mm": float(base_dimensions[2]),
        "volume_L": float(base_volume),
    }

    raw: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (
        structural_family,
        l1_delta,
        l2_delta,
        width_growth,
    ) in STRUCTURAL_STENCIL:
        for conductor_height_delta in PRIMARY_HEIGHT_DELTAS_MM:
            for (
                plate_family,
                plate_thickness,
                w1_reduction,
                contact_fraction,
            ) in PLATE_PATH_STENCIL:
                params = dict(base)
                params.update(
                    {
                        "l1": round(float(base["l1"]) + l1_delta, 1),
                        "l2": round(float(base["l2"]) + l2_delta, 1),
                        # H = h1 + 2*l1, so this exchange holds H exactly.
                        "h1": round(float(base["h1"]) - 2.0 * l1_delta, 1),
                        "nwh1": round(
                            float(base["nwh1"]) + conductor_height_delta,
                            1,
                        ),
                        "w1": round(
                            float(base["w1"]) - w1_reduction, 1
                        ),
                        "wcp_t": plate_thickness,
                    }
                )
                try:
                    valid, decoded_frame = validation_check(
                        create_input_parameter(params), strict=False
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if not valid:
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
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if not valid:
                    continue
                decoded = {
                    key: targeted._builtin(value)  # noqa: SLF001
                    for key, value in decoded_frame.iloc[0].to_dict().items()
                }
                _fixed_controls(decoded)
                vertical_clearance = (
                    float(decoded["h1"]) - float(decoded["nwh1"])
                ) / 2.0
                if vertical_clearance < MIN_VERTICAL_CLEARANCE_EACH_MM:
                    continue
                volume, dimensions = geometry_metrics.bounding_box_lit(
                    decoded
                )
                if any(
                    float(value) > SIZE_LIMITS_MM[axis] + 1e-9
                    for axis, value in zip(("W", "L", "H"), dimensions)
                ):
                    continue
                geometry_sha = targeted._geometry_sha(decoded)  # noqa: SLF001
                if (
                    geometry_sha in existing_geometry_sha256
                    or geometry_sha in seen
                ):
                    continue
                seen.add(geometry_sha)
                projected = {
                    key: targeted._builtin(decoded[key])  # noqa: SLF001
                    for key in sorted(ALL_INPUT_KEYS)
                }
                conductor_area = (
                    float(decoded["cw1"]) * float(decoded["nwh1"])
                )
                resistance_proxy = (
                    float(decoded["MLT_Tx_mm"]) / conductor_area
                )
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
                        "params_sha256": targeted._sha(projected),  # noqa: SLF001
                        "physical_geometry_sha256": geometry_sha,
                        "dimensions": [float(value) for value in dimensions],
                        "volume_L": float(volume),
                        "mutations": {
                            "structural_family": structural_family,
                            "plate_family": plate_family,
                            "l1_core_section_delta_mm": l1_delta,
                            "l2_window_delta_mm": l2_delta,
                            "width_growth_mm": width_growth,
                            "h1_window_height_delta_mm": -2.0 * l1_delta,
                            "nwh1_conductor_height_delta_mm": (
                                conductor_height_delta
                            ),
                            "w1_core_depth_path_reduction_mm": w1_reduction,
                            "wcp_t_mm": plate_thickness,
                            "wcp_len_x_mm": projected["wcp_len_x"],
                            "wcp_contact_fraction": contact_fraction,
                            "vertical_clearance_each_mm": vertical_clearance,
                        },
                        "mechanism_proxies": {
                            "primary_conductor_cross_section_mm2": (
                                conductor_area
                            ),
                            "primary_conductor_area_gain_percent": (
                                100.0
                                * (
                                    conductor_area
                                    / base_metrics[
                                        "primary_conductor_cross_section_mm2"
                                    ]
                                    - 1.0
                                )
                            ),
                            "primary_dc_resistance_proxy_ratio": (
                                resistance_proxy
                                / base_metrics["primary_dc_resistance_proxy"]
                            ),
                            "core_effective_area_m2": core_area,
                            "core_effective_area_gain_percent": (
                                100.0
                                * (
                                    core_area
                                    / base_metrics["core_effective_area_m2"]
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
                                    / base_metrics[
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
                                    / base_metrics[
                                        "winding_plate_spreading_section_proxy_mm2"
                                    ]
                                    - 1.0
                                )
                            ),
                        },
                    }
                )
    if len(raw) < COUNT:
        raise NeighborhoodError(
            f"only {len(raw)} non-duplicate valid candidates exist"
        )
    return raw, base_metrics


def _thermal_score(row: Mapping[str, Any]) -> tuple[float, ...]:
    primary = float(row["primary_winding_robust_max_C"])
    secondary = float(row["secondary_winding_robust_max_C"])
    core = float(row["core_robust_max_C"])
    violation = math.sqrt(
        max(0.0, primary - 100.0) ** 2
        + max(0.0, secondary - 100.0) ** 2
        + max(0.0, core - 120.0) ** 2
    )
    return (
        violation,
        max(primary, secondary),
        max(0.0, core - 120.0),
        primary,
        float(row["P_total_screen_W"]),
        str(row["physical_geometry_sha256"]),
    )


def _select(screened: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in screened
        if row["fixed_lm2mh_replay"]["fmin_Hz"]
        >= targeted.RESONANCE_MIN_HZ
        and row["corrected_Crx_transfer_estimate_F"] <= 0.55e-9
    ]
    if len(eligible) < COUNT:
        raise NeighborhoodError(
            "fewer than 12 candidates retain resonance and corrected Crx"
        )
    ordered = sorted(eligible, key=_thermal_score)
    selected: list[dict[str, Any]] = []

    def add(role: str, predicate: Any, count: int) -> None:
        for row in ordered:
            if len([x for x in selected if x["selection_role"] == role]) >= count:
                return
            if (
                row["physical_geometry_sha256"]
                not in {
                    item["physical_geometry_sha256"] for item in selected
                }
                and predicate(row)
            ):
                chosen = dict(row)
                chosen["selection_role"] = role
                selected.append(chosen)

    # Balance the Y-path exchange at three levels, then retain a thicker-plate
    # contact hedge.  Added l1 replaces most or all of the core-area reduction
    # caused by shorter w1, so FEA can identify the copper/contact benefit
    # without selecting only one unidentifiable compound mutation.
    add(
        "balanced_width_for_shorter_y_path",
        lambda row: row["mutations"]["plate_family"]
        == "balanced_short_path",
        3,
    )
    add(
        "shorter_y_path",
        lambda row: row["mutations"]["plate_family"] == "short_path",
        3,
    )
    add(
        "maximum_short_y_path",
        lambda row: row["mutations"]["plate_family"]
        == "maximum_short_path",
        3,
    )
    add(
        "plate_spreading_contact_hedge",
        lambda row: row["mutations"]["plate_family"]
        == "plate_spreader_contact80",
        3,
    )
    for row in ordered:
        if len(selected) >= COUNT:
            break
        if row["physical_geometry_sha256"] in {
            item["physical_geometry_sha256"] for item in selected
        }:
            continue
        chosen = dict(row)
        chosen["selection_role"] = "thermal_score_fill"
        selected.append(chosen)
    if len(selected) != COUNT:
        raise NeighborhoodError("diversified 12-candidate selection failed")
    return sorted(selected, key=_thermal_score)


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
        raise NeighborhoodError("last-mile source screen drifted")
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
        raise NeighborhoodError(
            "exact minimum-primary last-mile source is not unique"
        )
    source_lane = source_lanes[0]
    base_path = (
        source_root / source_lane["params"]["path"]
    ).resolve(strict=True)
    base = targeted._read_json(base_path)  # noqa: SLF001

    authority, final528_hashes = _final528_authority(final528_audit_path)
    lastmile_screen_hashes = {
        str(row["physical_geometry_sha256"])
        for row in source_screen.get("rows") or []
    }
    if len(lastmile_screen_hashes) != 72:
        raise NeighborhoodError("last-mile screen geometry inventory drifted")
    existing = set(final528_hashes)
    existing.update(lastmile_screen_hashes)
    existing.update(
        lane["candidate"]["physical_geometry_sha256"]
        for lane in source_plan["lanes"]
    )
    raw, base_metrics = _candidate_pool(base, existing)

    import pandas as pd

    frame = pd.DataFrame([item["decoded"] for item in raw])
    means, upper, model_identity = targeted._n1_6_cooler_predictions(  # noqa: SLF001
        frame
    )
    calibration_ratio, calibration = (
        targeted._turn_graded_crx_calibration()  # noqa: SLF001
    )
    baseline = source_lane["candidate"]
    baseline_primary = float(
        baseline["screening_primary_winding_max_C"]
    )
    baseline_secondary = float(
        baseline["screening_secondary_winding_max_C"]
    )
    baseline_core = float(baseline["screening_core_max_C"])

    screened: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        primary = max(
            float(upper[name][index])
            for name in targeted.PRIMARY_WINDING_TEMPERATURES
        )
        secondary = max(
            float(upper[name][index])
            for name in targeted.SECONDARY_WINDING_TEMPERATURES
        )
        core = max(
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
        total_loss = (
            float(means["P_winding_total"][index])
            + float(means["P_core_total"][index])
        )
        screened.append(
            {
                **item,
                "primary_winding_robust_max_C": primary,
                "secondary_winding_robust_max_C": secondary,
                "core_robust_max_C": core,
                "primary_improvement_C": baseline_primary - primary,
                "corrected_Crx_transfer_estimate_F": corrected_crx,
                "fixed_lm2mh_replay": replay,
                "P_Tx_main_group_W": float(
                    means["P_Tx_main_group"][index]
                ),
                "P_winding_total_W": float(
                    means["P_winding_total"][index]
                ),
                "P_core_total_W": float(means["P_core_total"][index]),
                "P_total_screen_W": total_loss,
                "surrogate_only_thermal_limits_screened": {
                    "primary_100C": primary <= 100.0,
                    "secondary_100C": secondary <= 100.0,
                    "core_120C": core <= 120.0,
                },
                "surrogate_pass_claimed": False,
                "actual_symmetric_FEA_required": True,
            }
        )
    selected = _select(screened)

    destination = output.resolve()
    if destination.exists():
        raise NeighborhoodError(f"output already exists: {destination}")
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
                "selection_role": item["selection_role"],
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
                    "objective_volume_L": item["volume_L"],
                    "surrogate_screen": {
                        "primary_winding_robust_max_C": item[
                            "primary_winding_robust_max_C"
                        ],
                        "secondary_winding_robust_max_C": item[
                            "secondary_winding_robust_max_C"
                        ],
                        "core_robust_max_C": item["core_robust_max_C"],
                        "P_Tx_main_group_W": item["P_Tx_main_group_W"],
                        "P_winding_total_W": item[
                            "P_winding_total_W"
                        ],
                        "P_core_total_W": item["P_core_total_W"],
                        "P_total_screen_W": item["P_total_screen_W"],
                        "corrected_Crx_transfer_estimate_F": item[
                            "corrected_Crx_transfer_estimate_F"
                        ],
                        **item["fixed_lm2mh_replay"],
                    },
                    "neighborhood_mutations": copy.deepcopy(
                        item["mutations"]
                    ),
                    "mechanism_proxies": copy.deepcopy(
                        item["mechanism_proxies"]
                    ),
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
                "read-only-symmetric-fea-acquisition-design-plan"
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
                "unused_W_mm": (
                    SIZE_LIMITS_MM["W"] - base_metrics["W_mm"]
                ),
                "primary_winding_robust_max_C": baseline_primary,
                "secondary_winding_robust_max_C": baseline_secondary,
                "core_robust_max_C": baseline_core,
                "mechanism_proxies": base_metrics,
            },
            "selection": {
                "candidate_pool_count": len(screened),
                "selected_candidate_count": len(selected),
                "surrogate_primary_improver_count": sum(
                    row["primary_improvement_C"] > 0.0
                    for row in screened
                ),
                "final528_geometry_hashes_excluded": len(final528_hashes),
                "lastmile_screen_geometry_hashes_excluded": len(
                    lastmile_screen_hashes
                ),
                "all_selected_geometries_are_new": True,
                "stencil": {
                    "structural": [
                        {
                            "family": name,
                            "l1_delta_mm": dl1,
                            "l2_delta_mm": dl2,
                            "width_growth_mm": width,
                        }
                        for name, dl1, dl2, width in STRUCTURAL_STENCIL
                    ],
                    "primary_conductor_height_delta_mm": list(
                        PRIMARY_HEIGHT_DELTAS_MM
                    ),
                    "plate_path": [
                        {
                            "family": name,
                            "wcp_t_mm": thickness,
                            "w1_reduction_mm": reduction,
                            "wcp_contact_fraction": contact_fraction,
                        }
                        for (
                            name,
                            thickness,
                            reduction,
                            contact_fraction,
                        ) in PLATE_PATH_STENCIL
                    ],
                    "minimum_vertical_clearance_each_mm": (
                        MIN_VERTICAL_CLEARANCE_EACH_MM
                    ),
                },
                "selection_rule": (
                    "fixed-Lm fmin>=15kHz and corrected Crx<=0.55nF; "
                    "thermal-violation score with mechanism quotas; "
                    "surrogate improvement is reported, not required; "
                    "acquisition ranking only"
                ),
            },
            "physical_rationale": {
                "nwh1": (
                    "cw1 remains exactly 5 mm; increasing foil height raises "
                    "A_cu=cw1*nwh1 and lowers the MLT/A_cu DC-resistance "
                    "proxy without changing turn count or interturn gap."
                ),
                "l1": (
                    "increasing l1 raises Ae_effective and lowers flux density "
                    "for the same volt-seconds; it also lengthens the legal "
                    "79/80% winding-plate contact span. h1 is reduced by "
                    "2*dl1 so H remains exactly 750 mm."
                ),
                "l2": (
                    "small l2 increments consume the remaining W as a control "
                    "for allocating width to window/clearance instead of core "
                    "section; they do not receive the same Ae/contact benefit."
                ),
                "wcp_len_x": (
                    "the primary families retain the source's surrogate-best "
                    "79% fraction and the hedge uses the validated 80% limit; "
                    "l1 makes either absolute contact length larger, reducing "
                    "the contact-area contribution to thermal resistance."
                ),
                "wcp_t_and_w1": (
                    "w1 is shortened by 12/16/20 mm to reduce primary MLT and "
                    "copper resistance; otherwise-unused W is assigned to l1 "
                    "to replace the lost core area. The 17.2 mm hedge adds "
                    "in-plane plate spreading and 80% contact at the same "
                    "maximum-short-path geometry."
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
            "all_size_limits_replayed": True,
            "all_geometry_hashes_new_against_final528_and_lastmile": True,
            "surrogate_ranking_only": True,
            "surrogate_pass_claimed": False,
            "scheduler_payloads_materialized": False,
            "scheduler_POST_performed": False,
            "scheduler_repository_modified": False,
            "apply_supported": False,
            "classification": "sealed-dry-run-only",
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
            "candidate_pool_count": len(screened),
            "selected_count": len(selected),
            "surrogate_primary_improver_count": sum(
                row["primary_improvement_C"] > 0.0 for row in screened
            ),
            "selected_geometry_sha256": [
                row["physical_geometry_sha256"] for row in selected
            ],
            "rows": [
                _without_large_fields(row)
                for row in sorted(screened, key=_thermal_score)
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
                "status": "ok",
                "classification": "sealed-dry-run-only",
                "scheduler_POST_performed": False,
                "path": str(path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
