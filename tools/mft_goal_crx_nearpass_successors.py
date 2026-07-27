"""Prepare a held, bounded Crx rescue batch around exact FEA task 97581.

The tool is deliberately fail-closed:

* the sealed 97581 harvest and exact-gap source plan are the local authority;
* capacitance changes are estimated from four same-core exact FEA points;
* the winding-height exponent is recovered from matched authenticated
  turn-graded FEA pairs, not assumed;
* every candidate keeps the fixed 6/60, cooling, plate, symmetry, and
  identical-three-leg-gap contracts;
* four Matrix + turn-graded-Rx-cap screening lanes and one full-physics lane
  are prepared, but this tool never submits them.

The generated plan is compatible with
``mft_goal_core_rescue_fea_feeder.py submit``.  Running that command without
``--apply`` creates a sealed dry-run receipt and performs no Scheduler POST.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    ROOT,
    ROOT / "regression_260707" / "training",
    ROOT / "regression_260707" / "optimization",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


from module.input_parameter_260706 import (
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from module.mft_goal_20260726_contract import canonical_sha256
from regression_260707.optimization.geometry_metrics import bounding_box_lit
from tools import mft_goal_core_rescue_fea_feeder as feeder
from tools import mft_goal_core_rescue_neighborhood as neighborhood
from tools import mft_goal_turn_graded_physics_reranker as physics


RUNTIME = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
)
BASE_PLAN = RUNTIME / "core_rescue_exact_gap_n4_full_v1" / "plan.json"
BASE_PARAMS = (
    RUNTIME
    / "core_rescue_exact_gap_n4_full_v1"
    / "params"
    / "lane-02.json"
)
SEALED_COLLECTION = (
    RUNTIME
    / "core_rescue_fea_harvest_terminal_v1"
    / "collection.json"
)
HEIGHT_CALIBRATION = (
    RUNTIME
    / "turn_graded_cap_6x60_current25_v1"
    / "bulk24_cap_thermal_join_ranked.csv"
)
QUALITY_STATUS = (
    RUNTIME / "g0b_25target_6151" / "quality_status.json"
)
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
BASE_TASK_ID = 97581
LOCAL_EXACT_TASK_IDS = (97580, 97581, 97582, 97583)
BASE_CRX_F = 589.022e-12
FIXED_SECONDARY_L_H = 0.2
MIN_REDUCTION = 0.06
SIZE_LIMITS_MM = (1200.0, 900.0, 750.0)
MIN_WIDTH_HEADROOM_MM = 1.0
HEX40 = re.compile(r"[0-9a-f]{40}")


SPECS = (
    {
        "candidate_id": "g080_h425_c45",
        "gap2_mm": 0.800,
        "cw2_mm": 0.450,
        "equal_winding_height_mm": 425.0,
    },
    {
        "candidate_id": "g082_h435_c45",
        "gap2_mm": 0.820,
        "cw2_mm": 0.450,
        "equal_winding_height_mm": 435.0,
    },
    {
        "candidate_id": "g0835_h442_c43",
        "gap2_mm": 0.835,
        "cw2_mm": 0.430,
        "equal_winding_height_mm": 442.0,
    },
    {
        "candidate_id": "g084_h445_c43",
        "gap2_mm": 0.840,
        "cw2_mm": 0.430,
        "equal_winding_height_mm": 445.0,
    },
)
FULL_CANDIDATE_ID = "g084_h445_c43"


class NearpassError(RuntimeError):
    """The source evidence, candidate contract, or held plan drifted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NearpassError(f"JSON object required: {path}")
    return value


def _validate_seal(
    value: Mapping[str, Any],
    *,
    schema_key: str,
    schema: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get(schema_key) != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise NearpassError(f"sealed source drifted: {schema}")
    return result


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise NearpassError("value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _candidate_sha(params: Mapping[str, Any]) -> str:
    identity = {
        key: _builtin(params[key])
        for key in (
            "N1_main",
            "N1_side",
            "N2_main",
            "N2_side",
            "l1",
            "l2",
            "h1",
            "w1",
            "n_core_group",
            "cw1",
            "gap1",
            "cw2",
            "gap2",
            "nwh1",
            "nwh2",
            "core_plate_t",
            "core_plate_pad_t",
            "wcp_t",
            "wcp_pad_t",
            "fan_velocity",
            "core_center_gap_mm",
            "core_equal_three_leg_air_gap",
        )
    }
    return canonical_sha256(identity)


def _validated_params(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float, float]]:
    ok, frame, errors = validation_check(
        create_input_parameter(dict(raw)),
        strict=True,
        return_errors=True,
    )
    if not ok or errors or len(frame) != 1:
        raise NearpassError(
            "candidate failed strict validation: " + " / ".join(errors)
        )
    row = frame.iloc[0]
    _volume, dimensions = bounding_box_lit(row)
    width, length, height = map(float, dimensions)
    checks = {
        "turns_6_60": int(row["N1"]) == 6 and int(row["N2"]) == 60,
        "equal_winding_heights": math.isclose(
            float(row["nwh1"]),
            float(row["nwh2"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "primary_foil_and_gap": (
            math.isclose(float(row["cw1"]), 5.0)
            and math.isclose(float(row["gap1"]), 1.6)
        ),
        "secondary_foil_ceiling": 0.0 < float(row["cw2"]) <= 1.0,
        "fixed_plates_and_pads": (
            math.isclose(float(row["core_plate_t"]), 20.0)
            and math.isclose(float(row["wcp_t"]), 20.0)
            and math.isclose(float(row["core_plate_pad_t"]), 2.0)
            and math.isclose(float(row["wcp_pad_t"]), 2.0)
        ),
        "fixed_fan_and_tim": (
            math.isclose(float(row["fan_velocity"]), 1.5)
            and math.isclose(float(row["k_ins"]), 0.2)
        ),
        "nonrounded_eighth": (
            int(row["round_corner"]) == 0
            and int(row["full_model"]) == 0
            and str(row["thermal_symmetry"]) == "eighth"
        ),
        "identical_three_leg_gap": (
            float(row["core_center_gap_mm"]) > 0.0
            and int(row["core_equal_three_leg_air_gap"]) == 1
            and int(row["core_air_gap_gapped_leg_count"]) == 3
        ),
        "size_limits": (
            width <= SIZE_LIMITS_MM[0]
            and length <= SIZE_LIMITS_MM[1]
            and height <= SIZE_LIMITS_MM[2]
        ),
        "strict_width_headroom": (
            SIZE_LIMITS_MM[0] - width >= MIN_WIDTH_HEADROOM_MM - 1e-9
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise NearpassError(
            "candidate physical contract failed: " + ", ".join(failed)
        )
    effective = {
        key: _builtin(row[key])
        for key in sorted(ALL_INPUT_KEYS)
    }
    return effective, checks, (width, length, height)


def _local_exact_rows(
    collection: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = BASE_PLAN.parent
    params_by_candidate: dict[str, dict[str, Any]] = {}
    for lane in source_plan.get("lanes") or []:
        candidate_sha = str(lane.get("candidate_sha256") or "")
        params_path = (root / str(lane["params"]["path"])).resolve(strict=True)
        if _file_sha256(params_path) != lane["params"]["sha256"]:
            raise NearpassError("exact source params SHA drifted")
        params_by_candidate[candidate_sha] = _read_json(params_path)
    records = {
        int(row["task_id"]): row
        for row in collection.get("ranked_records") or []
    }
    rows = []
    invariant_keys = (
        "N1_main",
        "N1_side",
        "N2_main",
        "N2_side",
        "l1",
        "l2",
        "h1",
        "w1",
        "n_core_group",
        "nwh1",
        "nwh2",
    )
    invariant: tuple[Any, ...] | None = None
    for task_id in LOCAL_EXACT_TASK_IDS:
        record = records.get(task_id)
        if (
            not isinstance(record, Mapping)
            or record.get("exact_gap_task") is not True
            or float(record.get("C_rx_turn_graded_full_F") or 0.0) <= 0.0
        ):
            raise NearpassError(f"exact FEA task {task_id} is unavailable")
        candidate_sha = str(record["candidate_sha256"])
        params = params_by_candidate.get(candidate_sha)
        if params is None:
            raise NearpassError("exact FEA geometry params are unavailable")
        current = tuple(params[key] for key in invariant_keys)
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise NearpassError("local exact FEA points are not same-core")
        network = physics.physics_energy_network(params)
        rows.append(
            {
                "task_id": task_id,
                "candidate_sha256": candidate_sha,
                "cw2_mm": float(params["cw2"]),
                "gap2_mm": float(params["gap2"]),
                "nwh2_mm": float(params["nwh2"]),
                "Crx_F": float(record["C_rx_turn_graded_full_F"]),
                "physics_network_Ceq_F": float(network["physics_Ceq_F"]),
                "result_json_sha256": record["result_json_sha256"],
            }
        )
    return rows


def _fit_local_models(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    values = list(rows)
    features = np.asarray(
        [
            [1.0, math.log(row["gap2_mm"]), math.log(row["cw2_mm"])]
            for row in values
        ],
        dtype=float,
    )
    truth = np.asarray([row["Crx_F"] for row in values], dtype=float)
    network = np.asarray(
        [row["physics_network_Ceq_F"] for row in values],
        dtype=float,
    )
    raw_beta = np.linalg.lstsq(features, np.log(truth), rcond=None)[0]
    residual_beta = np.linalg.lstsq(
        features,
        np.log(truth / network),
        rcond=None,
    )[0]

    def loo_errors(target: np.ndarray) -> list[float]:
        result = []
        for index in range(len(values)):
            mask = np.arange(len(values)) != index
            beta = np.linalg.lstsq(
                features[mask],
                target[mask],
                rcond=None,
            )[0]
            result.append(
                float(math.exp(features[index] @ beta - target[index]) - 1.0)
            )
        return result

    return {
        "feature_names": ["intercept", "log_gap2_mm", "log_cw2_mm"],
        "raw_log_coefficients": raw_beta.tolist(),
        "network_residual_log_coefficients": residual_beta.tolist(),
        "raw_log_loo_relative_errors": loo_errors(np.log(truth)),
        "network_residual_log_loo_relative_errors": loo_errors(
            np.log(truth / network)
        ),
    }


def _height_scaling_evidence() -> dict[str, Any]:
    calibration_path = HEIGHT_CALIBRATION.resolve(strict=True)
    rows = []
    with calibration_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        for raw in csv.DictReader(stream):
            params_path = Path(str(raw["source_params_path"])).resolve(
                strict=True
            )
            params = _read_json(params_path)
            rows.append(
                {
                    "geometry_sha256": raw["physical_geometry_sha256"],
                    "params_path": str(params_path),
                    "params_sha256": _file_sha256(params_path),
                    "Crx_F": float(raw["C_rx_rx_turn_graded_F"]),
                    "nwh2_mm": float(params["nwh2"]),
                    "match_key": tuple(
                        params[key]
                        for key in (
                            "N1_main",
                            "N1_side",
                            "N2_main",
                            "N2_side",
                            "l1",
                            "l2",
                            "h1",
                            "w1",
                            "cw2",
                            "gap2",
                            "cc_w2c_space_x",
                            "cc_w2c_space_y",
                            "w2c_w1c_space_x",
                            "w2c_w1c_space_y",
                            "w1c_w2s_space_x",
                            "w1s_cs_space_x",
                            "wcp_len_x",
                        )
                    ),
                }
            )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["match_key"], []).append(row)
    pairs = []
    for group in grouped.values():
        unique = {}
        for row in group:
            unique.setdefault(row["nwh2_mm"], row)
        ordered = sorted(unique.values(), key=lambda item: item["nwh2_mm"])
        if len(ordered) != 2:
            continue
        low, high = ordered
        exponent = math.log(high["Crx_F"] / low["Crx_F"]) / math.log(
            high["nwh2_mm"] / low["nwh2_mm"]
        )
        if math.isfinite(exponent) and 0.4 <= exponent <= 1.2:
            pairs.append(
                {
                    "low_geometry_sha256": low["geometry_sha256"],
                    "high_geometry_sha256": high["geometry_sha256"],
                    "low_nwh2_mm": low["nwh2_mm"],
                    "high_nwh2_mm": high["nwh2_mm"],
                    "low_Crx_F": low["Crx_F"],
                    "high_Crx_F": high["Crx_F"],
                    "power_exponent": exponent,
                    "low_params_sha256": low["params_sha256"],
                    "high_params_sha256": high["params_sha256"],
                }
            )
    if len(pairs) < 2:
        raise NearpassError("matched winding-height FEA evidence is insufficient")
    exponents = np.asarray(
        [item["power_exponent"] for item in pairs],
        dtype=float,
    )
    exponent = float(np.median(exponents))
    if not 0.6 <= exponent <= 0.9:
        raise NearpassError("winding-height exponent drifted")
    return {
        "ranked_csv": {
            "path": str(calibration_path),
            "sha256": _file_sha256(calibration_path),
        },
        "matched_pairs": pairs,
        "median_power_exponent": exponent,
    }


def _thermal_predictions(
    frames: pd.DataFrame,
) -> dict[str, list[float]]:
    generation = neighborhood._load_generation_record(
        neighborhood.DEFAULT_REGISTRY,
        neighborhood.DEFAULT_GENERATION,
    )
    targets = (
        "Llt_phys",
        "T_max_Tx",
        "Tprobe_Tx_leeward_max",
        "T_max_Rx_main",
        "T_max_Rx_side",
        "Tprobe_Rx_main_leeward_max",
        "Tprobe_Rx_side_leeward_max",
        "T_max_core",
        "Tprobe_core_center_max",
        "Tprobe_core_center_leg_max",
        "Tprobe_core_side_leg_max",
        "Tprobe_core_top_yoke_max",
        "P_core_total",
        "P_winding_total",
    )
    result: dict[str, list[float]] = {}
    for target in targets:
        mean, half, _spread = neighborhood._predict(
            generation,
            target,
            frames,
        )
        result[f"{target}_mean"] = mean.tolist()
        result[f"{target}_q90_half_width"] = half.tolist()
        result[f"{target}_robust_upper"] = (mean + half).tolist()
        gc.collect()
    return result


def _mode_profile(*, full: bool) -> dict[str, Any]:
    profile = feeder._full_profile()
    profile["cpus"] = feeder.CPUS
    profile["mem_mb"] = feeder.MEMORY_MB
    if full:
        profile["comment"] = (
            "Held Crx-nearpass best candidate: eighth-symmetry Matrix + "
            "turn-graded Rx Cap + Loss + Thermal"
        )
        profile["timeout_seconds"] = 14_400
        return profile
    profile.update(
        {
            "cli_flags": "--headless",
            "comment": (
                "Held Crx-nearpass screen: eighth-symmetry Matrix plus "
                "turn-graded Rx capacitance; no loss or thermal"
            ),
            "reviewed_solver_path": (
                "run_simulation_260706.py --fixed --headless"
            ),
            "stage": "turn_graded_rx_cap_screen",
            "timeout_seconds": 7_200,
        }
    )
    profile["param_overrides"].update(
        {
            "matrix_on": 1,
            "cap_on": 1,
            "loss_on": 0,
            "thermal_on": 0,
            "keep_project": 0,
            "loss_from_copy": 0,
        }
    )
    return profile


def prepare(output: Path, *, solver_revision: str) -> Path:
    if (
        not HEX40.fullmatch(solver_revision)
        or not HEX40.fullmatch(LIBRARY_REVISION)
    ):
        raise NearpassError("full solver/library revisions are required")
    destination = output.resolve()
    if destination.exists():
        raise NearpassError(f"output exists: {destination}")
    destination.mkdir(parents=True)

    collection_path = SEALED_COLLECTION.resolve(strict=True)
    collection = _validate_seal(
        _read_json(collection_path),
        schema_key="schema_version",
        schema="mft-goal-core-rescue-fea-collection-v1",
    )
    source_plan_path = BASE_PLAN.resolve(strict=True)
    source_plan = _validate_seal(
        _read_json(source_plan_path),
        schema_key="schema_version",
        schema=feeder.SCHEMA,
    )
    if (
        collection.get("complete") is not True
        or int(collection.get("collected_authoritative_task_count", 0)) != 21
        or int(collection.get("feasible_count", -1)) != 0
        or source_plan.get("library_revision") != LIBRARY_REVISION
    ):
        raise NearpassError("sealed core-rescue source is incomplete")
    exact_rows = _local_exact_rows(collection, source_plan)
    local_model = _fit_local_models(exact_rows)
    height_evidence = _height_scaling_evidence()
    height_exponent = float(height_evidence["median_power_exponent"])

    base_params_path = BASE_PARAMS.resolve(strict=True)
    base_params = _read_json(base_params_path)
    base_record = next(
        (
            row
            for row in collection["ranked_records"]
            if int(row["task_id"]) == BASE_TASK_ID
        ),
        None,
    )
    if (
        not isinstance(base_record, Mapping)
        or not math.isclose(
            float(base_record["C_rx_turn_graded_full_F"]),
            BASE_CRX_F,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        or not math.isclose(
            float(base_record["gap_mm"]),
            float(base_params["core_center_gap_mm"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise NearpassError("97581 base result/parameter binding drifted")

    raw_beta = np.asarray(
        local_model["raw_log_coefficients"],
        dtype=float,
    )
    residual_beta = np.asarray(
        local_model["network_residual_log_coefficients"],
        dtype=float,
    )
    candidate_rows = []
    physical_frames = []
    candidate_params: dict[str, dict[str, Any]] = {}
    contract_checks: dict[str, dict[str, bool]] = {}
    for spec in SPECS:
        raw = copy.deepcopy(base_params)
        raw.update(
            {
                "cw2": spec["cw2_mm"],
                "gap2": spec["gap2_mm"],
                "nwh1": spec["equal_winding_height_mm"],
                "nwh2": spec["equal_winding_height_mm"],
                "core_equal_three_leg_air_gap": 1,
                "core_center_gap_mm": base_params["core_center_gap_mm"],
                "round_corner": 0,
                "full_model": 0,
                "fan_velocity": 1.5,
                "core_plate_t": 20.0,
                "core_plate_pad_t": 2.0,
                "wcp_t": 20.0,
                "wcp_pad_t": 2.0,
            }
        )
        effective, checks, dimensions = _validated_params(raw)
        candidate_sha = _candidate_sha(effective)
        candidate_params[spec["candidate_id"]] = effective
        contract_checks[spec["candidate_id"]] = checks
        width, length, height = dimensions

        x = np.asarray(
            [
                1.0,
                math.log(float(effective["gap2"])),
                math.log(float(effective["cw2"])),
            ],
            dtype=float,
        )
        raw_at_base_height = math.exp(float(x @ raw_beta))
        base_height_params = copy.deepcopy(effective)
        base_height_params["nwh1"] = base_params["nwh1"]
        base_height_params["nwh2"] = base_params["nwh2"]
        _ok, base_height_frame, _errors = validation_check(
            create_input_parameter(base_height_params),
            strict=True,
            return_errors=True,
        )
        base_height_network = physics.physics_energy_network(
            base_height_frame.iloc[0].to_dict()
        )["physics_Ceq_F"]
        residual_at_base_height = base_height_network * math.exp(
            float(x @ residual_beta)
        )
        height_scale = (
            float(effective["nwh2"]) / float(base_params["nwh2"])
        ) ** height_exponent
        raw_prediction = raw_at_base_height * height_scale
        residual_prediction = residual_at_base_height * height_scale
        conservative_prediction = max(
            raw_prediction,
            residual_prediction,
        )
        reduction = 1.0 - conservative_prediction / BASE_CRX_F
        predicted_frequency = 1.0 / (
            2.0
            * math.pi
            * math.sqrt(FIXED_SECONDARY_L_H * conservative_prediction)
        )
        if reduction < MIN_REDUCTION:
            raise NearpassError(
                f"{spec['candidate_id']} misses conservative 6% Crx reduction"
            )
        candidate_rows.append(
            {
                **spec,
                "candidate_sha256": candidate_sha,
                "N1": int(effective["N1_main"])
                + int(effective["N1_side"]),
                "N2": int(effective["N2_main"])
                + int(effective["N2_side"]),
                "N2_main": int(effective["N2_main"]),
                "N2_side": int(effective["N2_side"]),
                "core_center_gap_mm": float(
                    effective["core_center_gap_mm"]
                ),
                "W_mm": width,
                "L_mm": length,
                "H_mm": height,
                "W_headroom_mm": SIZE_LIMITS_MM[0] - width,
                "L_headroom_mm": SIZE_LIMITS_MM[1] - length,
                "H_headroom_mm": SIZE_LIMITS_MM[2] - height,
                "raw_local_Crx_prediction_F": raw_prediction,
                "network_residual_Crx_prediction_F": residual_prediction,
                "conservative_Crx_prediction_F": conservative_prediction,
                "conservative_Crx_reduction_fraction": reduction,
                "fixed_L2_predicted_fRx_Hz": predicted_frequency,
                "final_FEA_authority": False,
            }
        )
        _prediction_ok, prediction_derived, prediction_errors = (
            validation_check(
                create_input_parameter(effective),
                strict=True,
                return_errors=True,
            )
        )
        if (
            not _prediction_ok
            or prediction_errors
            or len(prediction_derived) != 1
        ):
            raise NearpassError("candidate surrogate frame is invalid")
        physical_frames.append(prediction_derived.iloc[0].to_dict())

    _base_ok, base_derived, base_errors = validation_check(
        create_input_parameter(base_params),
        strict=True,
        return_errors=True,
    )
    if not _base_ok or base_errors or len(base_derived) != 1:
        raise NearpassError("base surrogate frame is invalid")
    prediction_frame = pd.DataFrame(
        [base_derived.iloc[0].to_dict(), *physical_frames]
    )
    thermal = _thermal_predictions(prediction_frame)
    for index, row in enumerate(candidate_rows, start=1):
        row["Llt_surrogate_mean_uH"] = thermal["Llt_phys_mean"][index]
        row["Llt_surrogate_q90_half_width_uH"] = thermal[
            "Llt_phys_q90_half_width"
        ][index]
        primary_targets = (
            "T_max_Tx",
            "Tprobe_Tx_leeward_max",
        )
        secondary_targets = (
            "T_max_Rx_main",
            "T_max_Rx_side",
            "Tprobe_Rx_main_leeward_max",
            "Tprobe_Rx_side_leeward_max",
        )
        core_targets = (
            "T_max_core",
            "Tprobe_core_center_max",
            "Tprobe_core_center_leg_max",
            "Tprobe_core_side_leg_max",
            "Tprobe_core_top_yoke_max",
        )
        row["primary_surrogate_robust_upper_C"] = max(
            thermal[f"{target}_robust_upper"][index]
            for target in primary_targets
        )
        row["secondary_surrogate_robust_upper_C"] = max(
            thermal[f"{target}_robust_upper"][index]
            for target in secondary_targets
        )
        row["core_surrogate_robust_upper_C"] = max(
            thermal[f"{target}_robust_upper"][index]
            for target in core_targets
        )
        row["P_core_surrogate_mean_W"] = thermal[
            "P_core_total_mean"
        ][index]
        row["P_winding_surrogate_mean_W"] = thermal[
            "P_winding_total_mean"
        ][index]
        row["surrogate_temperature_mean_plus_q90_pass"] = (
            row["primary_surrogate_robust_upper_C"] <= 110.0
            and row["secondary_surrogate_robust_upper_C"] <= 130.0
            and row["core_surrogate_robust_upper_C"] <= 130.0
        )
    best = next(
        row for row in candidate_rows if row["candidate_id"] == FULL_CANDIDATE_ID
    )
    if best["surrogate_temperature_mean_plus_q90_pass"] is not True:
        raise NearpassError("selected full candidate misses surrogate thermal gate")

    csv_path = destination / "bounded_candidates.csv"
    pd.DataFrame(candidate_rows).to_csv(
        csv_path,
        index=False,
        lineterminator="\n",
    )
    quality_path = QUALITY_STATUS.resolve(strict=True)
    quality = _read_json(quality_path)
    analysis = _seal(
        {
            "schema_version": "mft-goal-crx-nearpass-analysis-v1",
            "created_at_utc": _now(),
            "purpose": (
                "bounded contract-preserving >=6% corrected turn-graded Crx "
                "rescue around exact FEA task 97581"
            ),
            "base": {
                "task_id": BASE_TASK_ID,
                "candidate_sha256": base_record["candidate_sha256"],
                "params": {
                    "path": str(base_params_path),
                    "sha256": _file_sha256(base_params_path),
                },
                "Crx_F": BASE_CRX_F,
                "fRx_Hz": float(base_record["f_rx_turn_graded_Hz"]),
                "Lm_primary_full_mH": float(
                    base_record["Lm_primary_full_mH"]
                ),
                "Lm_secondary_full_mH": float(
                    base_record["Lm_secondary_full_mH"]
                ),
                "Llt_full_uH": float(base_record["Llt_full_uH"]),
                "W_mm": float(base_record["W_mm"]),
                "L_mm": float(base_record["L_mm"]),
                "H_mm": float(base_record["H_mm"]),
            },
            "sealed_collection": {
                "path": str(collection_path),
                "sha256": _file_sha256(collection_path),
                "payload_sha256": collection["payload_sha256"],
            },
            "exact_source_plan": {
                "path": str(source_plan_path),
                "sha256": _file_sha256(source_plan_path),
                "payload_sha256": source_plan["payload_sha256"],
            },
            "local_exact_FEA_rows": exact_rows,
            "local_capacitance_models": local_model,
            "height_scaling_evidence": height_evidence,
            "prediction_rule": (
                "max(raw local log-power fit, same-core physics-network "
                "log-residual fit) times authenticated matched-pair "
                "winding-height power law"
            ),
            "prediction_is_final_authority": False,
            "candidate_csv": {
                "path": str(csv_path),
                "sha256": _file_sha256(csv_path),
                "row_count": len(candidate_rows),
            },
            "candidate_rows": candidate_rows,
            "contract_checks": contract_checks,
            "surrogate_generation": {
                "registry": str(neighborhood.DEFAULT_REGISTRY),
                "generation": neighborhood.DEFAULT_GENERATION,
                "quality_status": {
                    "path": str(quality_path),
                    "sha256": _file_sha256(quality_path),
                    "passed": quality.get("passed"),
                    "reasons": quality.get("reasons"),
                },
                "final_authority": False,
            },
            "selected_full_candidate_id": FULL_CANDIDATE_ID,
            "required_FEA": {
                "four_matrix_turn_graded_Rx_cap_screens": True,
                "one_best_full_EM_loss_thermal": True,
                "nonrounded_eighth": True,
                "identical_three_leg_gap": True,
            },
            "scheduler_POST_performed": False,
            "final_design_claim_allowed": False,
        }
    )
    analysis_path = feeder._write(destination / "analysis.json", analysis)

    cap_profile = _mode_profile(full=False)
    full_profile = _mode_profile(full=True)
    cap_profile_record = feeder._profile_record(
        cap_profile,
        destination,
        "profiles/cap-screen.json",
    )
    full_profile_record = feeder._profile_record(
        full_profile,
        destination,
        "profiles/full-best.json",
    )
    lanes = []
    for index, row in enumerate(candidate_rows, start=1):
        candidate_id = str(row["candidate_id"])
        params = copy.deepcopy(candidate_params[candidate_id])
        params.update(
            {
                "matrix_on": 1,
                "cap_on": 1,
                "cap_turn_graded_active_winding": "Rx",
                "cap_turn_graded_voltage_policy": "turn_midpoint",
                "cap_turn_graded_section_order": "main,side",
                "loss_on": 0,
                "thermal_on": 0,
                "keep_project": 0,
                "loss_from_copy": 0,
            }
        )
        effective, _checks, _dimensions = _validated_params(params)
        params_path = feeder._write(
            destination / "params" / f"lane-{index:02d}.json",
            effective,
        )
        name = (
            f"mft-crx-nearpass-cap-{index:02d}-"
            f"{candidate_id}-{row['candidate_sha256'][:10]}"
        )
        workdir = name.replace("-", "_")
        identity = feeder.scheduler_client.verification_submission_identity(
            name,
            effective,
            cap_profile,
            solver_revision,
            LIBRARY_REVISION,
        )
        lanes.append(
            {
                "lane_index": index,
                "candidate_index": index,
                "candidate_id": candidate_id,
                "candidate_sha256": row["candidate_sha256"],
                "source_task_id": BASE_TASK_ID,
                "source_analysis_payload_sha256": analysis["payload_sha256"],
                "core_center_gap_mm": row["core_center_gap_mm"],
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "mode": "matrix_turngraded_rx_cap",
                "params": feeder._record(params_path, destination),
                "params_sha256": feeder._sha(effective),
                "profile": cap_profile_record,
                "scheduler": {
                    "project": feeder.scheduler_client.MFT_PROJECT,
                    "name": name,
                    "workdir": workdir,
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": feeder._sha(
                        identity["merged"]
                    ),
                    "cpus": feeder.CPUS,
                    "memory_mb": feeder.MEMORY_MB,
                    "timeout_seconds": int(cap_profile["timeout_seconds"]),
                    "priority": feeder.PRIORITY,
                    "max_workers_per_node": 1,
                    "environment": feeder._core_environment(
                        solver_revision
                    ),
                },
            }
        )

    full_index = len(lanes) + 1
    full_row = best
    full_params = copy.deepcopy(
        candidate_params[str(full_row["candidate_id"])]
    )
    full_params.update(
        {
            "matrix_on": 1,
            "cap_on": 1,
            "cap_turn_graded_active_winding": "Rx",
            "cap_turn_graded_voltage_policy": "turn_midpoint",
            "cap_turn_graded_section_order": "main,side",
            "loss_on": 1,
            "thermal_on": 1,
            "keep_project": 1,
            "loss_from_copy": 0,
        }
    )
    full_effective, _checks, _dimensions = _validated_params(full_params)
    full_params_path = feeder._write(
        destination / "params" / f"lane-{full_index:02d}.json",
        full_effective,
    )
    full_name = (
        f"mft-crx-nearpass-full-{full_row['candidate_id']}-"
        f"{full_row['candidate_sha256'][:10]}"
    )
    full_identity = (
        feeder.scheduler_client.verification_submission_identity(
            full_name,
            full_effective,
            full_profile,
            solver_revision,
            LIBRARY_REVISION,
        )
    )
    lanes.append(
        {
            "lane_index": full_index,
            "candidate_index": next(
                index
                for index, row in enumerate(candidate_rows, start=1)
                if row["candidate_id"] == FULL_CANDIDATE_ID
            ),
            "candidate_id": FULL_CANDIDATE_ID,
            "candidate_sha256": full_row["candidate_sha256"],
            "source_task_id": BASE_TASK_ID,
            "source_analysis_payload_sha256": analysis["payload_sha256"],
            "core_center_gap_mm": full_row["core_center_gap_mm"],
            "core_equal_three_leg_air_gap": 1,
            "expected_gapped_leg_count": 3,
            "mode": "matrix_turngraded_cap_loss_thermal",
            "params": feeder._record(full_params_path, destination),
            "params_sha256": feeder._sha(full_effective),
            "profile": full_profile_record,
            "scheduler": {
                "project": feeder.scheduler_client.MFT_PROJECT,
                "name": full_name,
                "workdir": full_name.replace("-", "_"),
                "dedupe_key": full_identity["dedupe_key"],
                "parameter_digest": full_identity["parameter_digest"],
                "effective_params_sha256": feeder._sha(
                    full_identity["merged"]
                ),
                "cpus": feeder.CPUS,
                "memory_mb": feeder.MEMORY_MB,
                "timeout_seconds": int(full_profile["timeout_seconds"]),
                "priority": feeder.PRIORITY,
                "max_workers_per_node": 1,
                "environment": feeder._core_environment(solver_revision),
            },
        }
    )
    plan = feeder._seal(
        {
            "schema_version": feeder.SCHEMA,
            "created_at_utc": _now(),
            "source": {
                "analysis": feeder._record(analysis_path, destination),
                "analysis_payload_sha256": analysis["payload_sha256"],
                "base_task_id": BASE_TASK_ID,
                "base_plan_payload_sha256": source_plan["payload_sha256"],
                "collection_payload_sha256": collection["payload_sha256"],
            },
            "solver_revision": solver_revision,
            "library_revision": LIBRARY_REVISION,
            "regression_contract": {
                "native_eighth_to_full_inductance_scale": 2.0,
                "target_physical_primary_referred_Lm_mH": 2.0,
                "preserved_exact_three_leg_gap_mm": float(
                    base_params["core_center_gap_mm"]
                ),
                "geometry_change_requires_post_FEA_Lm_check": True,
                "turns": "6/60",
                "equal_winding_heights": True,
                "cw1_mm": 5.0,
                "gap1_mm": 1.6,
                "cw2_max_mm": 1.0,
                "core_plate_t_mm": 20.0,
                "winding_plate_t_mm": 20.0,
                "pad_t_mm": 2.0,
                "fan_velocity_m_s": 1.5,
                "TIM_k_W_mK": 0.2,
                "W_max_mm": 1200.0,
                "L_max_mm": 900.0,
                "H_max_mm": 750.0,
            },
            "selection": {
                "candidate_count": 4,
                "cap_screen_lane_count": 4,
                "full_physics_lane_count": 1,
                "full_candidate_id": FULL_CANDIDATE_ID,
                "minimum_conservative_predicted_Crx_reduction": (
                    MIN_REDUCTION
                ),
                "prediction_is_final_authority": False,
            },
            "lanes": lanes,
            "parallel_execution_requested": True,
            "scheduler_project_source_included": False,
            "scheduler_project_modified": False,
            "scheduler_submission_performed": False,
            "final_promotion_allowed": False,
        }
    )
    plan_path = feeder._write(destination / "plan.json", plan)
    print(plan_path)
    return plan_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-revision", required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    prepare(args.output, solver_revision=args.solver_revision.strip().lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
