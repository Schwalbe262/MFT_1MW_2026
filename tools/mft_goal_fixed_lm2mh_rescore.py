"""Re-score the completed fixed-5T terminal population with fixed Lm=2 mH.

This is a screening-only post-processing lane.  It preserves every original
hard constraint except the obsolete geometry-derived self-resonance gate and
replaces only that gate with the explicitly documented fixed-Lm formula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPOSITORY_ROOT / "regression_260707" / "training"
for import_root in (REPOSITORY_ROOT, TRAINING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

INPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_nsga_v6_global_nds"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_w1200_l1000_fixed_lm2mh_rescore"
)
GENERATION = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\g0b_25target_6151\registry\generations"
    r"\20260724T210635-800c21a0"
)
EXPECTED_TASK_IDS = tuple(range(96397, 96413))
EXPECTED_TRAIN_REPORT_SHA256 = (
    "508d71fd789f29fb7535778a75f4f09d3ea52a7d8b8d50bf46df0bbb7ffa89ab"
)
EXPECTED_DATASET_SHA256 = (
    "0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3"
)
EXPECTED_EVALUATION_MODEL_SHA256 = (
    "b9a2714079317e4b31a5372311991bb91e024d64678618cbc32ab1ed18964dfc"
)
SOURCE_HARD_SPEC_SHA256 = (
    "bb05c758dab06a802627681c0c19be8e7062f423dda54089c0a539e61b0f7d5c"
)
TARGETS = ("Llt_phys", "C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F")
OLD_RESONANCE_CONSTRAINT = "half_magnetizing_resonance_minimum"
NEW_RESONANCE_CONSTRAINT = "fixed_Lm2mH_self_resonance_minimum"
LM_PRIMARY_REFERRED_H = 0.002
RESONANCE_MIN_HZ = 15_000.0
RESONANCE_NORMALIZATION_HZ = 150.0
PRIMARY_CONTROL_ABS_TOL_MM = 1e-12
EFFECTIVE_HARD_SPEC = {
    "schema_version": "mft-goal-fixed-primary-5t-axis-w1200-l1000-lm2mh-v1",
    "primary_conductor_thickness_mm": 5.0,
    "primary_interturn_gap_mm": 1.6,
    "primary_controls_are_hard_fixed": True,
    "size_limits_mm": {"W": 1200.0, "L": 1000.0, "H": 750.0},
    "axis_contract": {
        "W": "drawing_x_original_973mm_direction",
        "L": "drawing_y_perpendicular_direction",
        "rotation_or_axis_swap_allowed": False,
    },
    "magnetizing_inductance_H": LM_PRIMARY_REFERRED_H,
    "magnetizing_inductance_basis": "full-physical-primary-referred",
    "magnetizing_inductance_tuning": "explicit-air-gap",
    "self_resonance_min_Hz": RESONANCE_MIN_HZ,
    "winding_temperature_max_C": 100.0,
    "core_temperature_max_C": 120.0,
    "fan_velocity_m_s": 1.5,
    "cooling_and_TIM_mutation_allowed": False,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


EFFECTIVE_HARD_SPEC_SHA256 = _sha(EFFECTIVE_HARD_SPEC)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _atomic_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(staged, index=False)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _resonance_contract() -> dict[str, Any]:
    contract = {
        "schema_version": "mft-goal-fixed-lm2mh-resonance-contract-v1",
        "source_hard_spec_sha256": SOURCE_HARD_SPEC_SHA256,
        "effective_hard_spec": EFFECTIVE_HARD_SPEC,
        "effective_hard_spec_sha256": EFFECTIVE_HARD_SPEC_SHA256,
        "magnetizing_inductance": {
            "symbol": "Lm",
            "value": LM_PRIMARY_REFERRED_H,
            "unit": "H",
            "basis": "full-physical-primary-referred",
            "tuning": "explicit_air_gap",
            "geometry_prediction_used": False,
            "air_gap_geometry_or_FEA_attested": False,
        },
        "surrogate_inputs": {
            "Llt_phys": {
                "prediction": "ensemble_mean",
                "source_unit": "uH",
                "formula_conversion_to_H": "Llt_phys_uH * 1e-6",
                "basis": "full-physical-primary-referred",
            },
            "C_tx_tx_F": {
                "prediction": "ensemble_mean",
                "unit": "F",
                "basis": "full-transformer-restored",
            },
            "C_rx_rx_F": {
                "prediction": "ensemble_mean",
                "unit": "F",
                "basis": "full-transformer-restored",
            },
            "C_tx_rx_F": {
                "prediction": "ensemble_mean",
                "unit": "F",
                "basis": "full-transformer-restored",
                "use": "diagnostic-only",
            },
        },
        "formulas": {
            "Llt_phys_H": "Llt_phys_uH * 1e-6",
            "Ltx_H": "0.002 + Llt_phys_H",
            "ratio": "N2 / N1",
            "Lrx_H": "Ltx_H * ratio**2",
            "fTx_Hz": "1 / (2*pi*sqrt(Ltx_H*C_tx_tx_F))",
            "fRx_Hz": "1 / (2*pi*sqrt(Lrx_H*C_rx_rx_F))",
            "fInter_Hz": (
                "1 / (2*pi*sqrt(Llt_phys_H*C_tx_rx_F)); diagnostic-only"
            ),
            "hard_gate_Hz": "min(fTx_Hz, fRx_Hz) >= 15000",
            "physical_G": "15000 - min(fTx_Hz, fRx_Hz)",
        },
        "replaced_constraint": OLD_RESONANCE_CONSTRAINT,
        "replacement_constraint": NEW_RESONANCE_CONSTRAINT,
        "all_non_resonance_physical_constraints_preserved": True,
        "axis_contract": {
            "W_max_mm": 1200.0,
            "L_max_mm": 1000.0,
            "H_max_mm": 750.0,
            "rotation_or_axis_swap_allowed": False,
            "source_dimension_recovery": {
                "actual_W_mm": (
                    "1000 + source physical_G.exterior_width_limit"
                ),
                "actual_L_mm": (
                    "1200 + source physical_G.exterior_length_limit"
                ),
                "new_width_G_mm": "actual_W_mm - 1200",
                "new_length_G_mm": "actual_L_mm - 1000",
            },
        },
        "fixed_primary_controls": {"cw1_mm": 5.0, "gap1_mm": 1.6},
        "cooling_contract": {
            "fan_velocity_m_s": 1.5,
            "cooling_or_TIM_modified": False,
        },
        "thermal_scope": {
            "existing_thermal_surrogate_reused": True,
            "surrogate_retrained_for_explicit_gap_or_Lm2mH_current": False,
            "result_classification": "screening-only",
            "production_eligible": False,
        },
    }
    contract["contract_sha256"] = _sha(contract)
    return contract


def _authenticate_and_load_models(generation: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from predictor import EnsemblePredictor

    generation = generation.resolve(strict=True)
    report_path = generation / "train_report.json"
    if _sha_file(report_path) != EXPECTED_TRAIN_REPORT_SHA256:
        raise RuntimeError("training report identity mismatch")
    report = _read_json(report_path)
    if (
        report.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or not set(TARGETS).issubset(report.get("targets") or [])
    ):
        raise RuntimeError("training report source contract mismatch")
    artifact_hashes = report.get("artifacts")
    if not isinstance(artifact_hashes, dict):
        raise RuntimeError("training report artifact inventory is missing")
    model_evidence: dict[str, Any] = {}
    active = {"generation": str(generation), "report": report}
    models = {}
    for target in TARGETS:
        records = {}
        for name in ("models.pkl", "meta.json"):
            relative = f"{target}/{name}"
            path = generation / relative
            expected = artifact_hashes.get(relative)
            observed = _sha_file(path.resolve(strict=True))
            if expected != observed:
                raise RuntimeError(f"model artifact identity mismatch: {relative}")
            records[name] = {"path": str(path), "sha256": observed}
        model = EnsemblePredictor._load_record(target, active)
        binding = model.configure_inference_threads(threads=8)
        models[target] = model
        model_evidence[target] = {
            "artifacts": records,
            "features_sha256": _sha(list(model.features)),
            "inference_binding": binding,
        }
    return models, {
        "generation": str(generation),
        "train_report_sha256": EXPECTED_TRAIN_REPORT_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "evaluation_model_sha256": EXPECTED_EVALUATION_MODEL_SHA256,
        "targets": model_evidence,
    }


def _load_terminal_rows(input_root: Path) -> Any:
    import pandas as pd

    status_path = input_root / "collector_status.json"
    status = _read_json(status_path.resolve(strict=True))
    if (
        status.get("terminal_success_seed_count") != 16
        or status.get("global_nds_final") is not True
        or status.get("hard_spec_sha256") != SOURCE_HARD_SPEC_SHA256
    ):
        raise RuntimeError("the authenticated 16-seed collector is not final")
    collections = status.get("collections")
    if not isinstance(collections, list) or len(collections) != 16:
        raise RuntimeError("collector task coverage is incomplete")
    by_task = {int(item["task_id"]): item for item in collections}
    if tuple(sorted(by_task)) != EXPECTED_TASK_IDS:
        raise RuntimeError("collector task identity set mismatch")
    frames = []
    for task_id in EXPECTED_TASK_IDS:
        record = by_task[task_id]
        path = Path(record["csv"]).resolve(strict=True)
        if _sha_file(path) != record["csv_sha256"]:
            raise RuntimeError(f"collected CSV identity mismatch: task={task_id}")
        frame = pd.read_csv(path)
        if len(frame) != 320:
            raise RuntimeError(f"terminal population mismatch: task={task_id}")
        if set(frame["source_task_id"].astype(int)) != {task_id}:
            raise RuntimeError(f"terminal source task mismatch: task={task_id}")
        frame["scheduler_task_id"] = task_id
        frames.append(frame)
    terminal = pd.concat(frames, ignore_index=True)
    if len(terminal) != 5120:
        raise RuntimeError("16x320 terminal population was not preserved")
    return terminal


def _objective_front(frame: Any) -> Any:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(
        ["objective_volume_L", "objective_total_loss_W", "physical_geometry_sha256"],
        kind="mergesort",
    )
    retained = []
    best_loss = math.inf
    for index, row in ordered.iterrows():
        loss = float(row["objective_total_loss_W"])
        if loss < best_loss - 1e-12:
            retained.append(index)
            best_loss = loss
    return ordered.loc[retained].reset_index(drop=True)


def _selection_record(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "physical_geometry_sha256",
        "scheduler_task_id",
        "source_seed",
        "N1_fixed_lm2mh",
        "N2_fixed_lm2mh",
        "objective_volume_L",
        "objective_total_loss_W",
        "exterior_W_mm_fixed_lm2mh",
        "exterior_L_mm_fixed_lm2mh",
        "exterior_H_mm_fixed_lm2mh",
        "cw1_mm_fixed_lm2mh",
        "gap1_mm_fixed_lm2mh",
        "cw2_mm_fixed_lm2mh",
        "gap2_mm_fixed_lm2mh",
        "pred_Llt_phys_uH_fixed_lm2mh",
        "pred_C_tx_tx_F_fixed_lm2mh",
        "pred_C_rx_rx_F_fixed_lm2mh",
        "pred_C_tx_rx_F_fixed_lm2mh",
        "fTx_Hz_fixed_lm2mh",
        "fRx_Hz_fixed_lm2mh",
        "fInter_Hz_fixed_lm2mh_diagnostic",
        "resonance_min_Hz_fixed_lm2mh",
        "winding_robust_max_C_fixed_lm2mh",
        "core_robust_max_C_fixed_lm2mh",
        "normalized_constraint_violation_l2_fixed_lm2mh",
        "screening_feasible_fixed_lm2mh",
    )
    result = {}
    for field in fields:
        value = row[field]
        if hasattr(value, "item"):
            value = value.item()
        result[field] = value
    return result


def run(*, input_root: Path, output: Path, generation: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    contract = _resonance_contract()
    terminal = _load_terminal_rows(input_root.resolve(strict=True))
    decoded_values = []
    for row_index, raw in enumerate(terminal["decoded_physical_params_json"]):
        decoded = json.loads(raw)
        cw1 = float(decoded["cw1"])
        gap1 = float(decoded["gap1"])
        if (
            not math.isclose(cw1, 5.0, rel_tol=0.0, abs_tol=PRIMARY_CONTROL_ABS_TOL_MM)
            or not math.isclose(
                gap1, 1.6, rel_tol=0.0, abs_tol=PRIMARY_CONTROL_ABS_TOL_MM
            )
        ):
            raise RuntimeError(
                "terminal fixed control escaped: "
                f"row={row_index} task={terminal.iloc[row_index]['scheduler_task_id']} "
                f"cw1={cw1!r} gap1={gap1!r}"
            )
        decoded_values.append(decoded)
    decoded_frame = pd.DataFrame(decoded_values)
    models, model_identity = _authenticate_and_load_models(generation)
    prediction_means = {}
    prediction_half_widths = {}
    for target in TARGETS:
        mean, half_width = models[target].predict_mu_sigma(
            decoded_frame, conformal=True
        )
        mean = np.asarray(mean, dtype=float).reshape(-1)
        half_width = np.asarray(half_width, dtype=float).reshape(-1)
        if (
            mean.shape != (5120,)
            or half_width.shape != (5120,)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise RuntimeError(f"surrogate replay is invalid: {target}")
        prediction_means[target] = mean
        prediction_half_widths[target] = half_width

    llt_h = prediction_means["Llt_phys"] * 1e-6
    c_tx = prediction_means["C_tx_tx_F"]
    c_rx = prediction_means["C_rx_rx_F"]
    c_inter = prediction_means["C_tx_rx_F"]
    n1 = decoded_frame["N1"].to_numpy(dtype=float)
    n2 = decoded_frame["N2"].to_numpy(dtype=float)
    ratio = n2 / n1
    ltx_h = LM_PRIMARY_REFERRED_H + llt_h
    lrx_h = ltx_h * np.square(ratio)
    positive = (
        (llt_h > 0.0)
        & (c_tx > 0.0)
        & (c_rx > 0.0)
        & (c_inter > 0.0)
        & (ltx_h > 0.0)
        & (lrx_h > 0.0)
        & (ratio > 0.0)
    )
    f_tx = np.full(len(terminal), np.nan)
    f_rx = np.full(len(terminal), np.nan)
    f_inter = np.full(len(terminal), np.nan)
    f_tx[positive] = 1.0 / (
        2.0 * math.pi * np.sqrt(ltx_h[positive] * c_tx[positive])
    )
    f_rx[positive] = 1.0 / (
        2.0 * math.pi * np.sqrt(lrx_h[positive] * c_rx[positive])
    )
    f_inter[positive] = 1.0 / (
        2.0 * math.pi * np.sqrt(llt_h[positive] * c_inter[positive])
    )
    f_min = np.minimum(f_tx, f_rx)
    new_resonance_g = RESONANCE_MIN_HZ - f_min

    new_feasible = []
    new_violations = []
    physical_g_values = []
    normalized_g_values = []
    violation_counts: dict[str, int] = {}
    derived = {
        "exterior_W_mm_fixed_lm2mh": [],
        "exterior_L_mm_fixed_lm2mh": [],
        "exterior_H_mm_fixed_lm2mh": [],
        "winding_robust_max_C_fixed_lm2mh": [],
        "core_robust_max_C_fixed_lm2mh": [],
    }
    winding_targets = (
        "T_max_Tx",
        "T_max_Rx_main",
        "T_max_Rx_side",
        "Tprobe_Tx_leeward_max",
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
    for index, row in terminal.iterrows():
        source_physical_g = {
            name: float(value)
            for name, value in json.loads(row["physical_G_json"]).items()
            if name != OLD_RESONANCE_CONSTRAINT
        }
        source_normalized_g = {
            name: float(value)
            for name, value in json.loads(row["normalized_G_json"]).items()
            if name != OLD_RESONANCE_CONSTRAINT
        }
        actual_width_mm = (
            1000.0 + source_physical_g["exterior_width_limit"]
        )
        actual_length_mm = (
            1200.0 + source_physical_g["exterior_length_limit"]
        )
        physical_g = dict(source_physical_g)
        normalized_g = dict(source_normalized_g)
        physical_g["exterior_width_limit"] = actual_width_mm - 1200.0
        physical_g["exterior_length_limit"] = actual_length_mm - 1000.0
        normalized_g["exterior_width_limit"] = (
            physical_g["exterior_width_limit"]
        )
        normalized_g["exterior_length_limit"] = (
            physical_g["exterior_length_limit"]
        )
        resonance_g = (
            float(new_resonance_g[index]) if positive[index] else 1.0e12
        )
        physical_g[NEW_RESONANCE_CONSTRAINT] = resonance_g
        normalized_g[NEW_RESONANCE_CONSTRAINT] = (
            resonance_g / RESONANCE_NORMALIZATION_HZ
        )
        for name, value in physical_g.items():
            if value > 1e-9:
                violation_counts[name] = violation_counts.get(name, 0) + 1
        feasible = (
            _truth(row["decoder_valid"])
            and _truth(row["surrogate_physical_valid"])
            and positive[index]
            and all(value <= 1e-9 for value in physical_g.values())
        )
        new_feasible.append(feasible)
        new_violations.append(
            sum(max(value, 0.0) ** 2 for value in normalized_g.values())
        )
        physical_g_values.append(
            json.dumps(
                physical_g,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        normalized_g_values.append(
            json.dumps(
                normalized_g,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        derived["exterior_W_mm_fixed_lm2mh"].append(
            1200.0 + physical_g["exterior_width_limit"]
        )
        derived["exterior_L_mm_fixed_lm2mh"].append(
            1000.0 + physical_g["exterior_length_limit"]
        )
        derived["exterior_H_mm_fixed_lm2mh"].append(
            750.0 + physical_g["exterior_height_limit"]
        )
        derived["winding_robust_max_C_fixed_lm2mh"].append(
            max(
                100.0 + physical_g[f"temperature_robust_limit:{target}"]
                for target in winding_targets
                if f"temperature_robust_limit:{target}" in physical_g
            )
        )
        derived["core_robust_max_C_fixed_lm2mh"].append(
            max(
                120.0 + physical_g[f"temperature_robust_limit:{target}"]
                for target in core_targets
                if f"temperature_robust_limit:{target}" in physical_g
            )
        )

    terminal["fixed_lm2mh_resonance_contract_sha256"] = contract[
        "contract_sha256"
    ]
    terminal["pred_Llt_phys_uH_fixed_lm2mh"] = prediction_means["Llt_phys"]
    terminal["q90_Llt_phys_uH_fixed_lm2mh"] = prediction_half_widths["Llt_phys"]
    for target in ("C_tx_tx_F", "C_rx_rx_F", "C_tx_rx_F"):
        terminal[f"pred_{target}_fixed_lm2mh"] = prediction_means[target]
        terminal[f"q90_{target}_fixed_lm2mh"] = prediction_half_widths[target]
    terminal["Lm_primary_referred_H_fixed_lm2mh"] = LM_PRIMARY_REFERRED_H
    terminal["Llt_phys_H_fixed_lm2mh"] = llt_h
    terminal["turns_ratio_N2_over_N1_fixed_lm2mh"] = ratio
    terminal["Ltx_H_fixed_lm2mh"] = ltx_h
    terminal["Lrx_H_fixed_lm2mh"] = lrx_h
    terminal["fTx_Hz_fixed_lm2mh"] = f_tx
    terminal["fRx_Hz_fixed_lm2mh"] = f_rx
    terminal["fInter_Hz_fixed_lm2mh_diagnostic"] = f_inter
    terminal["resonance_min_Hz_fixed_lm2mh"] = f_min
    terminal["resonance_G_Hz_fixed_lm2mh"] = np.where(
        positive, new_resonance_g, 1.0e12
    )
    terminal["resonance_prediction_physical_fixed_lm2mh"] = positive
    terminal["physical_G_fixed_lm2mh_json"] = physical_g_values
    terminal["normalized_G_fixed_lm2mh_json"] = normalized_g_values
    terminal["normalized_constraint_violation_l2_fixed_lm2mh"] = new_violations
    terminal["screening_feasible_fixed_lm2mh"] = new_feasible
    terminal["screening_only_fixed_lm2mh"] = True
    terminal["production_eligible_fixed_lm2mh"] = False
    for name, values in derived.items():
        terminal[name] = values
    terminal["N1_fixed_lm2mh"] = decoded_frame["N1"].astype(int)
    terminal["N2_fixed_lm2mh"] = decoded_frame["N2"].astype(int)
    terminal["cw1_mm_fixed_lm2mh"] = decoded_frame["cw1"].astype(float)
    terminal["gap1_mm_fixed_lm2mh"] = decoded_frame["gap1"].astype(float)
    terminal["cw2_mm_fixed_lm2mh"] = decoded_frame["cw2"].astype(float)
    terminal["gap2_mm_fixed_lm2mh"] = decoded_frame["gap2"].astype(float)

    ordered = terminal.sort_values(
        [
            "physical_geometry_sha256",
            "normalized_constraint_violation_l2_fixed_lm2mh",
            "objective_volume_L",
            "objective_total_loss_W",
        ],
        kind="mergesort",
    )
    deduplicated = ordered.drop_duplicates(
        "physical_geometry_sha256", keep="first"
    ).reset_index(drop=True)
    feasible = deduplicated[
        deduplicated["screening_feasible_fixed_lm2mh"].map(_truth)
    ].copy()
    front = _objective_front(feasible)
    objective_front = _objective_front(deduplicated)

    output = output.resolve()
    all_path = output / "all_terminal_rescored_5120.csv"
    dedup_path = output / "global_terminal_rescored_deduplicated.csv"
    front_path = output / "global_pareto_fixed_lm2mh.csv"
    objective_path = output / "global_objective_front_fixed_lm2mh.csv"
    contract_path = output / "fixed_lm2mh_resonance_contract.json"
    _atomic_csv(all_path, terminal)
    _atomic_csv(dedup_path, deduplicated)
    _atomic_csv(front_path, front)
    _atomic_csv(objective_path, objective_front)
    _atomic_json(contract_path, contract)

    selection_pool = (
        front
        if not front.empty
        else deduplicated.sort_values(
            [
                "normalized_constraint_violation_l2_fixed_lm2mh",
                "objective_volume_L",
                "objective_total_loss_W",
            ],
            kind="mergesort",
        ).head(20)
    )
    selections = [_selection_record(row) for _, row in selection_pool.iterrows()]
    status = {
        "schema_version": "mft-goal-fixed-lm2mh-rescore-v1",
        "classification": "screening-only",
        "production_eligible": False,
        "thermal_surrogate_retrained_for_explicit_gap_or_Lm2mH_current": False,
        "thermal_screening_caveat": (
            "The retained thermal surrogate was not retrained for an explicit "
            "air-gap-tuned 2.000 mH magnetizing branch/current; temperatures "
            "are screening values, not final FEA validation."
        ),
        "resonance_contract": contract,
        "resonance_contract_sha256": contract["contract_sha256"],
        "model_identity": model_identity,
        "source": {
            "collector_status": str(input_root / "collector_status.json"),
            "source_hard_spec_sha256": SOURCE_HARD_SPEC_SHA256,
            "effective_hard_spec_sha256": EFFECTIVE_HARD_SPEC_SHA256,
            "scheduler_task_ids": list(EXPECTED_TASK_IDS),
        },
        "counts": {
            "raw_terminal": int(len(terminal)),
            "unique_geometry": int(len(deduplicated)),
            "resonance_prediction_physical": int(positive.sum()),
            "screening_hard_feasible": int(len(feasible)),
            "global_pareto": int(len(front)),
            "global_objective_front": int(len(objective_front)),
        },
        "constraint_positive_row_counts_raw_5120": dict(
            sorted(violation_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "selection_basis": (
            "screening_hard_feasible_global_pareto"
            if not front.empty
            else "least_normalized_violation_fallback_not_production"
        ),
        "selections": selections,
        "artifacts": {},
    }
    for path in (all_path, dedup_path, front_path, objective_path, contract_path):
        status["artifacts"][path.name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha_file(path),
        }
    status["payload_sha256"] = _sha(status)
    _atomic_json(output / "fixed_lm2mh_rescore_status.json", status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--generation", type=Path, default=GENERATION)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        input_root=args.input,
        output=args.output,
        generation=args.generation,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
