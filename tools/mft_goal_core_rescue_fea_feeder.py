"""Select and submit the compact 6/60 core-rescue FEA tranche.

This feeder is intentionally part of the MFT repository.  It only calls the
separate Scheduler service API and never modifies Scheduler source or project
configuration.

The sealed source is a geometry-only, ungapped sweep.  Selection:

* filters the current compact/fixed-cooling contract,
* enumerates every integer 6/60 secondary main/side split at fixed geometry,
* repairs the split with the authenticated robust leakage model,
* ranks with the physics/delta *mean* capacitance (the extrapolated q90 bound
  is recorded but is not a ranking or rejection authority), and
* evaluates the authenticated winding/core temperature surrogate one target
  at a time to avoid retaining the complete generation in memory.

Submission is fail-closed and produces a three-point physical-gap bracket for
each selected geometry.  The middle point runs Matrix + turn-graded Rx Cap +
Loss + Thermal.  The two outer points run Matrix only.  Every point uses a
positive identical gap in the center and both side legs; no ungapped source
row can be submitted or promoted.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "regression_260707" / "training"
for import_root in (ROOT, TRAINING):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from regression_260707.optimization.geometry_metrics import (  # noqa: E402
    bounding_box_lit,
)
from regression_260707.verify import scheduler_client  # noqa: E402
from predictor import EnsemblePredictor  # noqa: E402
from tools import mft_goal_corrected_physics_nsga_lane as corrected  # noqa: E402
from tools import mft_goal_fixed_lm2mh_rescore as rescore  # noqa: E402
from tools import mft_goal_lm2mh_gap_tuner as gap_tuner  # noqa: E402
from tools import mft_goal_targeted_symmetric_fea_batch as targeted  # noqa: E402
from tools import mft_goal_turn_graded_physics_reranker as physics  # noqa: E402


SCHEMA = "mft-goal-core-rescue-direct-fea-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-core-rescue-direct-fea-submission-v1"
HANDOFF_SCHEMA = "mft-goal-core-rescue-four-lane-handoff-v1"
HANDOFF = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\core_rescue_neighborhood_n4n5_20260727T201404"
    r"\four_lane_handoff.json"
)
HANDOFF_FILE_SHA256 = (
    "14cf6e85ed1986677cbf2f81a674925f30e5ded2594b9492b0f2059287c70400"
)
HANDOFF_PAYLOAD_SHA256 = (
    "e34ffcec44cb5e477424f8651453c467a57dd7feff4f72b8308e5985ccb1df2d"
)
SOURCE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\anchor_core_rescue_6x60_v1\geometry_sweep_20260727T105135Z"
    r"\validated_core_rescue_geometries.csv"
)
SOURCE_SHA256 = (
    "69fa23a52e1a08a961762ce0faa6de3e3a21f06d35cf538eb7a1aee3b8308ce4"
)
PHYSICS_MODEL = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\physics_delta_reranker_v2\model.json"
)
PHYSICS_MODEL_SHA256 = (
    "0a025986c8e9f8b0c1a5615045c62e79e879d0d3517eb3084c79e40614f7669f"
)
SOLVER_REVISION = "383e61d4d4d550ef7a87456c0180e2af0811774f"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SCHEDULER_URL = "http://127.0.0.1:8002"
GAPS_MM = (0.20, 0.35, 0.50)
FULL_PHYSICS_GAP_MM = 0.35
SELECTED_GEOMETRIES = 4
CPUS = 8
MEMORY_MB = 65_536
PRIORITY = 100
PROJECT_ACTIVE_TASK_CAP = 500
TEMPERATURE_TARGETS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "T_max_core",
)
TEMPERATURE_LIMITS_C = {
    "T_max_Tx": 110.0,
    "T_max_Rx_main": 130.0,
    "T_max_Rx_side": 130.0,
    "T_max_core": 130.0,
}
HEX40 = re.compile(r"[0-9a-f]{40}")


class FeederError(RuntimeError):
    """The source, selection, FEA contract, or Scheduler readback drifted."""


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


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _builtin(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise FeederError("payload is already sealed")
    result["payload_sha256"] = _sha(result)
    return result


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeederError(f"{schema} must be an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _sha(unsigned):
        raise FeederError(f"{schema} seal mismatch")
    return dict(value)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FeederError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: Any, *, replace: bool = False) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FeederError(f"output exists: {destination}")
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    _builtin(value),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return destination


def _record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(root.resolve())).replace("\\", "/"),
        "sha256": _file_sha(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _model_prediction(target: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    generation = rescore.GENERATION.resolve(strict=True)
    report = _read(generation / "train_report.json")
    if (
        _file_sha(generation / "train_report.json")
        != rescore.EXPECTED_TRAIN_REPORT_SHA256
        or report.get("dataset_sha256") != rescore.EXPECTED_DATASET_SHA256
        or target not in set(report.get("targets") or [])
    ):
        raise FeederError(f"surrogate generation drifted for {target}")
    active = {"generation": str(generation), "report": report}
    model = EnsemblePredictor._load_record(target, active)
    model.configure_inference_threads(threads=8)
    mean, half = model.predict_mu_sigma(frame, conformal=True)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    half = np.asarray(half, dtype=float).reshape(-1)
    if (
        mean.shape != (len(frame),)
        or half.shape != (len(frame),)
        or not np.isfinite(mean).all()
        or not np.isfinite(half).all()
        or np.any(half < 0.0)
    ):
        raise FeederError(f"surrogate output is invalid for {target}")
    del model
    gc.collect()
    return mean, half


def _filtered_source() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source = SOURCE.resolve(strict=True)
    if _file_sha(source) != SOURCE_SHA256:
        raise FeederError("sealed core-rescue source SHA mismatch")
    frame = pd.read_csv(source)
    rows: list[pd.Series] = []
    params: list[dict[str, Any]] = []
    for _, raw in frame.iterrows():
        value = json.loads(str(raw["full_decoded_params_json"]))
        if not isinstance(value, dict):
            raise FeederError("decoded source cell is not an object")
        if (
            int(value["N1"]) != 6
            or int(value["N2"]) != 60
            or int(value["n_core_group"]) not in (4, 5)
            or float(value["gap2"]) < 0.35
            or float(value["cw2"]) > 1.0
            or not math.isclose(float(value["nwh1"]), float(value["nwh2"]))
            or float(value["core_plate_t"]) != 20.0
            or float(value["wcp_t"]) != 20.0
            or float(value["core_plate_pad_t"]) != 2.0
            or float(value["wcp_pad_t"]) != 2.0
            or float(value["cw1"]) != 5.0
            or float(value["gap1"]) != 1.6
            or float(raw["W_mm"]) > 1200.0
            or float(raw["L_mm"]) > 900.0
            or float(raw["H_mm"]) > 750.0
        ):
            continue
        rows.append(raw)
        params.append(value)
    if len(rows) != 17 or any(int(value["n_core_group"]) != 4 for value in params):
        raise FeederError(
            f"expected exact 17 group-4 candidates after contract filter, got {len(rows)}"
        )
    return pd.DataFrame(rows).reset_index(drop=True), params


def _repair_splits(
    source_frame: pd.DataFrame,
    source_params: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    base = pd.DataFrame(source_params)
    expanded, valid = corrected._expanded_fixed_geometry_split_frame(
        base, decoder_valid=np.ones(len(base), dtype=bool)
    )
    mean, half = _model_prediction("Llt_phys", expanded)
    split_count = len(corrected.SPLIT_VALUES)
    selected_frames: list[pd.Series] = []
    audits: list[dict[str, Any]] = []
    for index in range(len(base)):
        start = index * split_count
        stop = start + split_count
        local_valid = np.asarray(valid[start:stop], dtype=bool)
        robust_g = (
            np.abs(mean[start:stop] - 27.5)
            + half[start:stop]
            - 0.55
        )
        robust_g[~local_valid] = np.inf
        offset = int(np.argmin(robust_g))
        if not math.isfinite(float(robust_g[offset])):
            raise FeederError(f"no valid split for source candidate {index}")
        selected_frames.append(expanded.iloc[start + offset].copy())
        audits.append(
            {
                "source_geometry_sweep_sha256": str(
                    source_frame.iloc[index]["geometry_sweep_sha256"]
                ),
                "enumerated_split_count": split_count,
                "valid_split_count": int(local_valid.sum()),
                "selected_N2_main": int(expanded.iloc[start + offset]["N2_main"]),
                "selected_N2_side": int(expanded.iloc[start + offset]["N2_side"]),
                "Llt_mean_uH": float(mean[start + offset]),
                "Llt_q90_half_width_uH": float(half[start + offset]),
                "robust_Llt_G": float(robust_g[offset]),
                "selection": "minimum_robust_Llt_G",
            }
        )
    selected = pd.DataFrame(selected_frames).reset_index(drop=True)
    repaired_params: list[dict[str, Any]] = []
    repaired_rows: list[pd.Series] = []
    kept_audits: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        projected = {
            key: _builtin(row[key]) for key in sorted(ALL_INPUT_KEYS)
        }
        ok, validated = validation_check(
            create_input_parameter(projected), strict=True
        )
        if not ok:
            continue
        decoded = validated.iloc[0].to_dict()
        _volume_l, raw_dimensions = bounding_box_lit(decoded)
        dimensions = {
            "W_drawing_x": float(raw_dimensions[0]),
            "L_perpendicular_y": float(raw_dimensions[1]),
            "H": float(raw_dimensions[2]),
        }
        if (
            float(dimensions["W_drawing_x"]) > 1200.0
            or float(dimensions["L_perpendicular_y"]) > 900.0
            or float(dimensions["H"]) > 750.0
        ):
            continue
        repaired_params.append(
            {
                key: _builtin(decoded[key])
                for key in sorted(ALL_INPUT_KEYS)
            }
        )
        repaired_rows.append(validated.iloc[0])
        audit = audits[index]
        audit["dimensions_mm"] = {
            "W": float(dimensions["W_drawing_x"]),
            "L": float(dimensions["L_perpendicular_y"]),
            "H": float(dimensions["H"]),
        }
        kept_audits.append(audit)
    if len(repaired_params) < SELECTED_GEOMETRIES:
        raise FeederError("too few size-valid split-repaired candidates")
    return pd.DataFrame(repaired_rows).reset_index(drop=True), kept_audits


def _rank_candidates(
    repaired: pd.DataFrame,
    audits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(repaired) != len(audits):
        raise FeederError("repaired candidate/audit shape mismatch")
    model_path = PHYSICS_MODEL.resolve(strict=True)
    if _file_sha(model_path) != PHYSICS_MODEL_SHA256:
        raise FeederError("physics/delta model SHA mismatch")
    model = physics._load_model(model_path)
    predictions = [
        physics._candidate_prediction(repaired.iloc[index].to_dict(), model)
        for index in range(len(repaired))
    ]
    temperatures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for target in TEMPERATURE_TARGETS:
        temperatures[target] = _model_prediction(target, repaired)
    candidates: list[dict[str, Any]] = []
    for index, row in repaired.iterrows():
        params = {
            key: _builtin(row[key]) for key in sorted(ALL_INPUT_KEYS)
        }
        dimensions = audits[index]["dimensions_mm"]
        prediction = predictions[index]
        temp_record = {
            target: {
                "mean_C": float(temperatures[target][0][index]),
                "q90_half_width_C": float(temperatures[target][1][index]),
                "q90_upper_C": float(
                    temperatures[target][0][index]
                    + temperatures[target][1][index]
                ),
                "limit_C": TEMPERATURE_LIMITS_C[target],
            }
            for target in TEMPERATURE_TARGETS
        }
        upper_violations = sum(
            item["q90_upper_C"] > item["limit_C"]
            for item in temp_record.values()
        )
        upper_ratio = max(
            item["q90_upper_C"] / item["limit_C"]
            for item in temp_record.values()
        )
        mean_ratio = max(
            item["mean_C"] / item["limit_C"]
            for item in temp_record.values()
        )
        identity = _sha(
            {
                "params": params,
                "air_gap_stage": "positive_equal_three_leg_gap_required",
            }
        )
        candidates.append(
            {
                "candidate_sha256": identity,
                "source_geometry_sweep_sha256": audits[index][
                    "source_geometry_sweep_sha256"
                ],
                "params": params,
                "params_sha256": _sha(params),
                "dimensions_mm": dimensions,
                "volume_L": (
                    dimensions["W"] * dimensions["L"] * dimensions["H"] / 1e6
                ),
                "B_design_square_material_analytic_T": (
                    1000.0
                    / (
                        4.0
                        * 1000.0
                        * 6.0
                        * float(row["Ae_effective_m2"])
                    )
                ),
                "split_repair": audits[index],
                "physics_capacitance": {
                    **prediction,
                    "ranking_value": "physics_delta_Crx_mean_F",
                    "q90_excluded_from_rank_and_rejection": True,
                    "reason": "out_of_calibration_geometry_extrapolation",
                    "screening_only": True,
                },
                "temperature_surrogate": temp_record,
                "_rank": (
                    int(upper_violations),
                    float(upper_ratio),
                    float(mean_ratio),
                    max(
                        0.0,
                        15_000.0
                        - float(prediction["physics_delta_fRx_mean_Hz"]),
                    ),
                    float(audits[index]["robust_Llt_G"]),
                    float(
                        prediction["physics_delta_Crx_mean_F"]
                    ),
                    dimensions["W"]
                    * dimensions["L"]
                    * dimensions["H"],
                    identity,
                ),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["physics_capacitance"]["physics_delta_fRx_mean_Hz"]
        >= 15_000.0
    ]
    if len(eligible) < SELECTED_GEOMETRIES:
        raise FeederError(
            "fewer than four group-4 candidates pass physics-C mean screening"
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(role: str, item: dict[str, Any]) -> None:
        if (
            item["candidate_sha256"] not in seen
            and len(selected) < SELECTED_GEOMETRIES
        ):
            item = copy.deepcopy(item)
            item["selection_role"] = role
            item.pop("_rank", None)
            selected.append(item)
            seen.add(item["candidate_sha256"])

    eligible.sort(key=lambda item: item["_rank"])
    add("minimum_combined_surrogate_risk", eligible[0])
    add(
        "maximum_physics_C_mean_resonance",
        max(
            eligible,
            key=lambda item: item["physics_capacitance"][
                "physics_delta_fRx_mean_Hz"
            ],
        ),
    )
    add(
        "minimum_primary_temperature_q90_upper",
        min(
            eligible,
            key=lambda item: item["temperature_surrogate"]["T_max_Tx"][
                "q90_upper_C"
            ],
        ),
    )
    add(
        "minimum_core_temperature_q90_upper",
        min(
            eligible,
            key=lambda item: item["temperature_surrogate"]["T_max_core"][
                "q90_upper_C"
            ],
        ),
    )
    for item in eligible:
        add("next_combined_surrogate_risk", item)
    if len(selected) != SELECTED_GEOMETRIES:
        raise FeederError("failed to select four unique candidates")
    return selected


def _full_profile() -> dict[str, Any]:
    profile = targeted._profile()
    profile["comment"] = (
        "Core-rescue eighth-symmetry direct Matrix + turn-graded Rx Cap + "
        "Loss + Thermal at a positive identical three-leg physical gap"
    )
    profile["param_overrides"]["core_equal_three_leg_air_gap"] = 1
    profile["param_overrides"]["cap_on"] = 1
    profile["param_overrides"]["round_corner"] = 0
    profile["param_overrides"]["full_model"] = 0
    profile["param_overrides"]["thermal_symmetry"] = "eighth"
    profile.pop("artifact_retention", None)
    return profile


def _profile_record(profile: Mapping[str, Any], root: Path, name: str) -> dict[str, Any]:
    path = _write(root / name, profile)
    return _record(path, root)


def _core_environment(solver_revision: str) -> dict[str, str]:
    contract = "mft-standalone-core-optin-v1"
    auth = _sha(
        {
            "backend": "standalone",
            "contract_version": contract,
            "requested_num_cores": CPUS,
            "required_slurm_cpus_per_task": CPUS,
            "solver_revision": solver_revision,
        }
    )
    return {
        "MFT_STANDALONE_CORE_CONTRACT": contract,
        "MFT_STANDALONE_CORE_COUNT": str(CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": auth,
    }


def prepare(output: Path) -> Path:
    if not HEX40.fullmatch(SOLVER_REVISION) or not HEX40.fullmatch(
        LIBRARY_REVISION
    ):
        raise FeederError("full solver/library revisions are required")
    destination = output.resolve()
    if destination.exists():
        raise FeederError(f"output exists: {destination}")
    destination.mkdir(parents=True)
    source_frame, source_params = _filtered_source()
    repaired, audits = _repair_splits(source_frame, source_params)
    selected = _rank_candidates(repaired, audits)
    matrix_profile = gap_tuner._profile()
    full_profile = _full_profile()
    matrix_profile_record = _profile_record(
        matrix_profile, destination, "matrix_profile.json"
    )
    full_profile_record = _profile_record(
        full_profile, destination, "full_physics_profile.json"
    )
    lanes: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(selected, start=1):
        base = candidate["params"]
        for gap_index, gap_mm in enumerate(GAPS_MM, start=1):
            full_physics = math.isclose(
                gap_mm, FULL_PHYSICS_GAP_MM, abs_tol=1e-12
            )
            profile = full_profile if full_physics else matrix_profile
            params = dict(base)
            params.update(profile["param_overrides"])
            params.update(
                {
                    "core_center_gap_mm": gap_mm,
                    "core_equal_three_leg_air_gap": 1,
                    "core_plate_t": 20.0,
                    "wcp_t": 20.0,
                    "core_plate_pad_t": 2.0,
                    "wcp_pad_t": 2.0,
                    "fan_velocity": 1.5,
                    "round_corner": 0,
                    "full_model": 0,
                    "thermal_symmetry": "eighth",
                }
            )
            if full_physics:
                params.update(
                    {
                        "cap_on": 1,
                        "cap_turn_graded_active_winding": "Rx",
                        "cap_turn_graded_voltage_policy": "turn_midpoint",
                        "cap_turn_graded_section_order": "main,side",
                        "cap_turn_graded_reverse_sections": "none",
                        "cap_turn_graded_reverse_terminal_polarity": 0,
                        "cap_turn_graded_side_polarity": 1,
                        "cap_turn_graded_side2_polarity": 1,
                    }
                )
            ok, validated = validation_check(
                create_input_parameter(params), strict=True
            )
            if not ok:
                raise FeederError(
                    f"candidate {candidate_index} gap {gap_mm} failed validation"
                )
            readback = validated.iloc[0]
            if (
                int(readback["core_equal_three_leg_air_gap"]) != 1
                or int(readback["core_air_gap_gapped_leg_count"]) != 3
                or float(readback["core_center_gap_mm"]) <= 0.0
                or float(readback["core_plate_t"]) != 20.0
                or float(readback["wcp_t"]) != 20.0
                or float(readback["gap2"]) < 0.35
                or float(readback["cw2"]) > 1.0
                or not math.isclose(
                    float(readback["nwh1"]), float(readback["nwh2"])
                )
            ):
                raise FeederError("effective positive three-leg contract drifted")
            params = {
                key: _builtin(readback[key])
                for key in sorted(ALL_INPUT_KEYS)
            }
            token = f"{int(round(gap_mm * 1_000_000)):08d}"
            short = candidate["candidate_sha256"][:10]
            mode = "full" if full_physics else "matrix"
            name = f"mft-core-rescue-{mode}-c{candidate_index}-g{token}-{short}"
            workdir = f"mft_core_rescue_{mode}_c{candidate_index}_g{token}_{short}"
            params_path = _write(
                destination
                / "params"
                / f"candidate-{candidate_index:02d}-g{token}-{mode}.json",
                params,
            )
            identity = scheduler_client.verification_submission_identity(
                name,
                params,
                profile,
                SOLVER_REVISION,
                LIBRARY_REVISION,
            )
            lanes.append(
                {
                    "lane_index": len(lanes) + 1,
                    "candidate_index": candidate_index,
                    "candidate_sha256": candidate["candidate_sha256"],
                    "core_center_gap_mm": gap_mm,
                    "core_equal_three_leg_air_gap": 1,
                    "expected_gapped_leg_count": 3,
                    "mode": (
                        "matrix_turngraded_cap_loss_thermal"
                        if full_physics
                        else "matrix_only_lm_bracket"
                    ),
                    "params": _record(params_path, destination),
                    "params_sha256": _sha(params),
                    "profile": (
                        full_profile_record
                        if full_physics
                        else matrix_profile_record
                    ),
                    "scheduler": {
                        "project": scheduler_client.MFT_PROJECT,
                        "name": name,
                        "workdir": workdir,
                        "dedupe_key": identity["dedupe_key"],
                        "parameter_digest": identity["parameter_digest"],
                        "effective_params_sha256": _sha(identity["merged"]),
                        "cpus": CPUS,
                        "memory_mb": MEMORY_MB,
                        "timeout_seconds": int(profile["timeout_seconds"]),
                        "priority": PRIORITY,
                        "max_workers_per_node": 1,
                        "environment": _core_environment(SOLVER_REVISION),
                    },
                }
            )
    plan = _seal(
        {
            "schema_version": SCHEMA,
            "created_at_utc": _now(),
            "source": {
                "path": str(SOURCE.resolve(strict=True)),
                "sha256": SOURCE_SHA256,
                "ungapped_geometry_only": True,
                "source_rows_can_be_final": False,
            },
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "selection": {
                "filtered_group4_candidate_count": 17,
                "group5_candidate_count": 0,
                "split_values": list(corrected.SPLIT_VALUES),
                "split_selection": "minimum_robust_Llt_G",
                "physics_capacitance_ranking": "mean_only",
                "physics_capacitance_q90_used_for_rejection": False,
                "temperature_surrogate": "authenticated_q90_recorded",
                "selected_geometry_count": len(selected),
                "candidates": selected,
            },
            "gap_bracket": {
                "values_mm": list(GAPS_MM),
                "full_physics_gap_mm": FULL_PHYSICS_GAP_MM,
                "topology": "equal_center_and_both_side_legs",
                "core_equal_three_leg_air_gap": 1,
                "gapped_leg_count": 3,
                "target_full_physical_primary_referred_Lm_H": 0.002,
                "post_FEA_interpolation_required": True,
                "pre_gap_rows_final_promotion_allowed": False,
            },
            "fixed_contract": {
                "turns": "6/60",
                "core_plate_t_mm": 20.0,
                "wcp_t_mm": 20.0,
                "pad_t_mm": 2.0,
                "fan_velocity_m_s": 1.5,
                "TIM_mutated": False,
                "cw1_mm": 5.0,
                "gap1_mm": 1.6,
                "gap2_min_mm": 0.35,
                "cw2_max_mm": 1.0,
                "equal_winding_heights": True,
                "size_limits_mm": {"W": 1200.0, "L": 900.0, "H": 750.0},
                "symmetric_eighth": True,
                "rounded": False,
            },
            "matrix_profile": matrix_profile_record,
            "full_physics_profile": full_profile_record,
            "lanes": lanes,
            "parallel_execution_requested": True,
            "scheduler_project_source_included": False,
            "scheduler_project_modified": False,
            "scheduler_submission_performed": False,
            "final_promotion_allowed": False,
        }
    )
    return _write(destination / "plan.json", plan)


def prepare_handoff(output: Path, handoff_path: Path = HANDOFF) -> Path:
    """Prepare the four remaining-capacity lanes from the sealed N4/N5 handoff.

    The compact N5 candidate receives the complete 0.20/0.35/0.50 mm Lm
    bracket, with full physics at 0.35 mm.  The strongest N4 candidate uses
    the fourth slot for an independent full-physics 0.35 mm hedge.
    """
    if not HEX40.fullmatch(SOLVER_REVISION) or not HEX40.fullmatch(
        LIBRARY_REVISION
    ):
        raise FeederError("full solver/library revisions are required")
    destination = output.resolve()
    if destination.exists():
        raise FeederError(f"output exists: {destination}")
    handoff_file = handoff_path.resolve(strict=True)
    if _file_sha(handoff_file) != HANDOFF_FILE_SHA256:
        raise FeederError("sealed four-lane handoff file SHA mismatch")
    handoff = _validate_seal(_read(handoff_file), HANDOFF_SCHEMA)
    if handoff["payload_sha256"] != HANDOFF_PAYLOAD_SHA256:
        raise FeederError("sealed four-lane handoff payload SHA mismatch")
    destination.mkdir(parents=True)

    candidates = {
        str(item["candidate_sha256"]): item
        for item in handoff.get("candidates", [])
    }
    compact_sha = (
        "50f849eb3c0054d21bc7af8259fd3daff2223d37d2458a6ba4f0301d58e689b9"
    )
    hedge_sha = (
        "2f6eb674ebcae02d3b914fa2d688dbec2e845e029ef14fea9ce6ddf05235d1ad"
    )
    if set((compact_sha, hedge_sha)) - set(candidates):
        raise FeederError("required compact/hedge candidates are absent")

    matrix_profile = gap_tuner._profile()
    full_profile = _full_profile()
    matrix_profile_record = _profile_record(
        matrix_profile, destination, "matrix_profile.json"
    )
    full_profile_record = _profile_record(
        full_profile, destination, "full_physics_profile.json"
    )
    lane_specs = (
        (compact_sha, 0.20, False, "compact_n5_lm_lower_gap"),
        (compact_sha, 0.35, True, "compact_n5_full_physics"),
        (compact_sha, 0.50, False, "compact_n5_lm_upper_gap"),
        (hedge_sha, 0.35, True, "best_n4_full_physics_hedge"),
    )
    lanes: list[dict[str, Any]] = []
    selected_candidates: dict[str, dict[str, Any]] = {}
    for lane_index, (candidate_sha, gap_mm, full_physics, role) in enumerate(
        lane_specs, start=1
    ):
        candidate = candidates[candidate_sha]
        params_record = candidate.get("params")
        if not isinstance(params_record, dict):
            raise FeederError("candidate params record is absent")
        source_params_path = Path(str(params_record["path"])).resolve(strict=True)
        if _file_sha(source_params_path) != str(params_record["sha256"]):
            raise FeederError(f"candidate params SHA mismatch: {candidate_sha}")
        source_params = _read(source_params_path)
        if (
            int(source_params["N1_main"]) != 6
            or int(source_params["N1_side"]) != 0
            or int(source_params["N2_main"])
            + int(source_params["N2_side"]) != 60
            or int(source_params["n_core_group"]) not in (4, 5)
            or float(source_params["core_plate_t"]) != 20.0
            or float(source_params["wcp_t"]) != 20.0
            or float(source_params["core_plate_pad_t"]) != 2.0
            or float(source_params["wcp_pad_t"]) != 2.0
            or float(source_params["cw1"]) != 5.0
            or float(source_params["gap1"]) != 1.6
            or float(source_params["gap2"]) < 0.35
            or float(source_params["cw2"]) > 1.0
            or not math.isclose(
                float(source_params["nwh1"]), float(source_params["nwh2"])
            )
            or float(source_params["fan_velocity"]) != 1.5
            or int(source_params["round_corner"]) != 0
            or int(source_params["full_model"]) != 0
        ):
            raise FeederError(f"candidate fixed contract drifted: {candidate_sha}")

        missing_input_keys = [
            key for key in ALL_INPUT_KEYS if key not in source_params
        ]
        if missing_input_keys:
            raise FeederError(
                f"candidate input keys are absent: {missing_input_keys}"
            )
        profile = full_profile if full_physics else matrix_profile
        params = {
            key: _builtin(source_params[key]) for key in sorted(ALL_INPUT_KEYS)
        }
        params.update(profile["param_overrides"])
        params.update(
            {
                "core_center_gap_mm": gap_mm,
                "core_equal_three_leg_air_gap": 1,
                "core_plate_t": 20.0,
                "wcp_t": 20.0,
                "core_plate_pad_t": 2.0,
                "wcp_pad_t": 2.0,
                "fan_velocity": 1.5,
                "round_corner": 0,
                "full_model": 0,
                "thermal_symmetry": "eighth",
            }
        )
        if full_physics:
            params.update(
                {
                    "cap_on": 1,
                    "cap_turn_graded_active_winding": "Rx",
                    "cap_turn_graded_voltage_policy": "turn_midpoint",
                    "cap_turn_graded_section_order": "main,side",
                    "cap_turn_graded_reverse_sections": "none",
                    "cap_turn_graded_reverse_terminal_polarity": 0,
                    "cap_turn_graded_side_polarity": 1,
                    "cap_turn_graded_side2_polarity": 1,
                }
            )
        else:
            params["cap_turn_graded_active_winding"] = "off"
        ok, validated = validation_check(
            create_input_parameter(params), strict=True
        )
        if not ok:
            raise FeederError(
                f"handoff lane {lane_index} failed strict validation"
            )
        readback = validated.iloc[0]
        _volume_l, dimensions_raw = bounding_box_lit(readback.to_dict())
        dimensions = {
            "W": float(dimensions_raw[0]),
            "L": float(dimensions_raw[1]),
            "H": float(dimensions_raw[2]),
        }
        if (
            int(readback["core_equal_three_leg_air_gap"]) != 1
            or int(readback["core_air_gap_gapped_leg_count"]) != 3
            or not math.isclose(
                float(readback["core_center_gap_mm"]), gap_mm, abs_tol=1e-12
            )
            or float(readback["core_plate_t"]) != 20.0
            or float(readback["wcp_t"]) != 20.0
            or float(readback["gap2"]) < 0.35
            or float(readback["cw2"]) > 1.0
            or not math.isclose(
                float(readback["nwh1"]), float(readback["nwh2"])
            )
            or dimensions["W"] > 1200.0
            or dimensions["L"] > 900.0
            or dimensions["H"] > 750.0
        ):
            raise FeederError(
                f"handoff lane {lane_index} effective contract drifted"
            )
        effective = {
            key: _builtin(readback[key]) for key in sorted(ALL_INPUT_KEYS)
        }
        token = f"{int(round(gap_mm * 1_000_000)):08d}"
        short = candidate_sha[:10]
        mode = "full" if full_physics else "matrix"
        name = f"mft-core-rescue2-{mode}-{role}-g{token}-{short}"
        workdir = f"mft_core_rescue2_{mode}_{role}_g{token}_{short}"
        params_path = _write(
            destination
            / "params"
            / f"lane-{lane_index:02d}-g{token}-{mode}.json",
            effective,
        )
        identity = scheduler_client.verification_submission_identity(
            name,
            effective,
            profile,
            SOLVER_REVISION,
            LIBRARY_REVISION,
        )
        selected_candidates[candidate_sha] = copy.deepcopy(candidate)
        lanes.append(
            {
                "lane_index": lane_index,
                "selection_role": role,
                "candidate_sha256": candidate_sha,
                "source_params": {
                    "path": str(source_params_path),
                    "sha256": str(params_record["sha256"]),
                },
                "core_center_gap_mm": gap_mm,
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "dimensions_mm": dimensions,
                "mode": (
                    "matrix_turngraded_cap_loss_thermal"
                    if full_physics
                    else "matrix_only_lm_bracket"
                ),
                "params": _record(params_path, destination),
                "params_sha256": _sha(effective),
                "profile": (
                    full_profile_record
                    if full_physics
                    else matrix_profile_record
                ),
                "scheduler": {
                    "project": scheduler_client.MFT_PROJECT,
                    "name": name,
                    "workdir": workdir,
                    "dedupe_key": identity["dedupe_key"],
                    "parameter_digest": identity["parameter_digest"],
                    "effective_params_sha256": _sha(identity["merged"]),
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "timeout_seconds": int(profile["timeout_seconds"]),
                    "priority": PRIORITY,
                    "max_workers_per_node": 1,
                    "environment": _core_environment(SOLVER_REVISION),
                },
            }
        )

    plan = _seal(
        {
            "schema_version": SCHEMA,
            "created_at_utc": _now(),
            "source": {
                "path": str(handoff_file),
                "file_sha256": HANDOFF_FILE_SHA256,
                "payload_sha256": HANDOFF_PAYLOAD_SHA256,
                "schema_version": HANDOFF_SCHEMA,
                "screening_only": True,
                "pre_gap_rows_final_promotion_allowed": False,
            },
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "selection": {
                "strategy": (
                    "compact_n5_three_point_gap_bracket_plus_best_n4_full_hedge"
                ),
                "physics_capacitance_ranking": "mean_only",
                "physics_capacitance_q90_used_for_rejection": False,
                "selected_candidates": list(selected_candidates.values()),
            },
            "gap_bracket": {
                "compact_n5_values_mm": list(GAPS_MM),
                "full_physics_gap_mm": FULL_PHYSICS_GAP_MM,
                "topology": "equal_center_and_both_side_legs",
                "core_equal_three_leg_air_gap": 1,
                "gapped_leg_count": 3,
                "target_full_physical_primary_referred_Lm_H": 0.002,
                "post_FEA_interpolation_required": True,
                "pre_gap_rows_final_promotion_allowed": False,
            },
            "fixed_contract": handoff["fixed_contract"],
            "required_execution": handoff["required_execution"],
            "matrix_profile": matrix_profile_record,
            "full_physics_profile": full_profile_record,
            "lanes": lanes,
            "parallel_execution_requested": True,
            "scheduler_project_source_included": False,
            "scheduler_project_modified": False,
            "scheduler_submission_performed": False,
            "final_promotion_allowed": False,
        }
    )
    return _write(destination / "plan.json", plan)


def submit(plan_path: Path, output: Path, *, apply: bool) -> Path:
    plan = _validate_seal(_read(plan_path), SCHEMA)
    root = plan_path.resolve(strict=True).parent
    if output.exists():
        raise FeederError(f"output exists: {output}")
    if not apply:
        return _write(
            output,
            _seal(
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "created_at_utc": _now(),
                    "plan_payload_sha256": plan["payload_sha256"],
                    "scheduler_POST_performed": False,
                    "submitted_lane_count": 0,
                    "submissions": [],
                }
            ),
        )
    submissions: list[dict[str, Any]] = []
    for lane in plan["lanes"]:
        params = _read(root / lane["params"]["path"])
        profile = _read(root / lane["profile"]["path"])
        scheduler = lane["scheduler"]
        evidence = scheduler_client.submit_verification(
            scheduler["name"],
            scheduler["workdir"],
            params,
            profile,
            mem_mb=scheduler["memory_mb"],
            cpus=scheduler["cpus"],
            solver_revision=plan["solver_revision"],
            library_revision=plan["library_revision"],
            priority=scheduler["priority"],
            max_workers_per_node=scheduler["max_workers_per_node"],
            aedt_backend="standalone",
            submission_env=scheduler["environment"],
            required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
            return_submission_evidence=True,
            max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
            scheduler_url=SCHEDULER_URL,
        )
        if not isinstance(evidence, dict):
            raise FeederError("Scheduler submission evidence is absent")
        readback = (
            evidence.get("api_pre_submission_readback")
            or evidence.get("api_post_submission_response")
            or evidence
        )
        if isinstance(readback, dict) and isinstance(readback.get("task"), dict):
            readback = readback["task"]
        if not isinstance(readback, dict):
            raise FeederError("Scheduler readback is absent")
        task_id = int(
            evidence.get(
                "task_id",
                evidence.get("id", readback.get("id", readback.get("task_id", 0))),
            )
        )
        if (
            task_id <= 0
            or readback.get("name") != scheduler["name"]
            or readback.get("dedupe_key") != scheduler["dedupe_key"]
            or readback.get("project") != scheduler_client.MFT_PROJECT
            or int(readback.get("cpus") or 0) != scheduler["cpus"]
            or int(readback.get("memory_mb") or 0) != scheduler["memory_mb"]
        ):
            raise FeederError("Scheduler POST readback drifted")
        submissions.append(
            {
                "lane_index": lane["lane_index"],
                "candidate_index": lane.get("candidate_index"),
                "candidate_sha256": lane["candidate_sha256"],
                "core_center_gap_mm": lane["core_center_gap_mm"],
                "core_equal_three_leg_air_gap": 1,
                "expected_gapped_leg_count": 3,
                "mode": lane["mode"],
                "task_id": task_id,
                "name": scheduler["name"],
                "dedupe_key": scheduler["dedupe_key"],
                "readback": readback,
                "submission_evidence": evidence,
            }
        )
    value = _seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "created_at_utc": _now(),
            "plan_payload_sha256": plan["payload_sha256"],
            "scheduler_url": SCHEDULER_URL,
            "scheduler_POST_performed": True,
            "requested_lane_count": len(plan["lanes"]),
            "submitted_lane_count": len(submissions),
            "complete": len(submissions) == len(plan["lanes"]),
            "parallel_execution_requested": True,
            "scheduler_project_modified": False,
            "submissions": submissions,
        }
    )
    return _write(output, value)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    handoff_parser = sub.add_parser("prepare-handoff")
    handoff_parser.add_argument("--output", type=Path, required=True)
    handoff_parser.add_argument("--handoff", type=Path, default=HANDOFF)
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--output", type=Path, required=True)
    submit_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path = prepare(args.output)
    elif args.command == "prepare-handoff":
        path = prepare_handoff(args.output, args.handoff)
    else:
        path = submit(args.plan, args.output, apply=args.apply)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
