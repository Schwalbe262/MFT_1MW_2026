"""Fail-closed, row-level simulation quality contract.

The campaign parquet is an audit log, not an automatically trusted training
set.  In particular, older solver revisions emitted ``result_valid_em=1`` for
rows whose adaptive energy error was above the requested tolerance.  This
module therefore recomputes validity from the numerical evidence on every
row.  Stored validity flags are necessary, but never sufficient.

The contract is intentionally independent of the scheduler and AEDT so it can
be reused by collection, checkpoint training, model training, optimization,
and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = HERE / "verify" / "profiles" / "standard.json"
MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15

MATRIX_REQUIRED_OUTPUTS = (
    "Ltx", "Lrx", "M", "k", "Lmt", "Lmr", "Llt", "Llr",
)
LOSS_REQUIRED_OUTPUTS = (
    "P_core_total", "P_core_plate_total", "P_wcp_total",
    "P_winding_total", "B_mean_core", "B_max_core",
)
MANDATORY_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
)
SIDE_TEMPERATURE_COLUMNS = (
    "T_max_Rx_side",
    "Tprobe_Rx_side_leeward_max",
)
FLOW_RESIDUAL_COLUMNS = (
    "thermal_residual_continuity",
    "thermal_residual_x_velocity",
    "thermal_residual_y_velocity",
    "thermal_residual_z_velocity",
)

# These settings change the analysis basis.  Bookkeeping settings such as
# keep_project and pass ceilings do not: a stricter pass ceiling is acceptable
# when the row still demonstrates convergence.
PROFILE_IDENTITY_KEYS = (
    "full_model",
    "matrix_on",
    "loss_on",
    "thermal_on",
    "loss_sym_on",
    "thermal_symmetry",
    "n_explicit_turns",
    "matrix_skin_mesh",
    "freq",
    "loss_from_copy",
    "P_target",
)


@dataclass(frozen=True)
class ValidationResult:
    em_valid: bool
    thermal_valid: bool
    full_valid: bool
    reasons: tuple[str, ...]


def load_profile(profile: str | Path | Mapping[str, Any] | None = None) -> dict:
    if profile is None:
        profile = DEFAULT_PROFILE_PATH
    if isinstance(profile, Mapping):
        return dict(profile)
    with open(profile, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("param_overrides"), dict):
        raise ValueError("simulation profile must contain param_overrides")
    return value


def _value(record: Mapping[str, Any], key: str) -> Any:
    try:
        value = record.get(key)
    except AttributeError:
        value = record[key] if key in record else None
    return value


def _number(record: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(_value(record, key))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _one(record: Mapping[str, Any], key: str) -> bool:
    value = _number(record, key)
    return value is not None and value == 1.0


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return str(actual or "").strip().lower() == expected.strip().lower()
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _profile_reasons(record: Mapping[str, Any], profile: dict) -> list[str]:
    expected = profile["param_overrides"]
    reasons: list[str] = []
    for key in PROFILE_IDENTITY_KEYS:
        if key in expected and not _same(_value(record, key), expected[key]):
            reasons.append(f"profile_mismatch:{key}")
    return reasons


def _provenance_reasons(
    record: Mapping[str, Any], expected_solver_revision=None,
    expected_library_revision=None,
) -> list[str]:
    reasons: list[str] = []
    for key in ("git_dirty", "pyaedt_library_git_dirty"):
        if _number(record, key) != 0.0:
            reasons.append(f"untrusted_provenance:{key}")
    for key in ("git_hash", "pyaedt_library_git_hash"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(_value(record, key) or "").strip()):
            reasons.append(f"untrusted_provenance:{key}")
    if expected_solver_revision is not None and str(
        _value(record, "git_hash") or ""
    ).strip().lower() != str(expected_solver_revision).strip().lower():
        reasons.append("untrusted_provenance:solver_revision_mismatch")
    if expected_library_revision is not None and str(
        _value(record, "pyaedt_library_git_hash") or ""
    ).strip().lower() != str(expected_library_revision).strip().lower():
        reasons.append("untrusted_provenance:library_revision_mismatch")
    return reasons


def _em_reasons(record: Mapping[str, Any], profile: dict) -> list[str]:
    reasons: list[str] = []
    if not _one(record, "result_valid_em"):
        reasons.append("stored_flag:result_valid_em")
    for key in ("matrix_solve_attempts", "loss_solve_attempts"):
        if _number(record, key) != 1.0:
            reasons.append(f"em_solve_attempt:{key}")

    expected = profile["param_overrides"]
    stages = (
        ("matrix", "matrix_percent_error", "matrix_min_converged", MATRIX_REQUIRED_OUTPUTS),
        ("loss", "percent_error", "min_converged", LOSS_REQUIRED_OUTPUTS),
    )
    for label, tolerance_key, minimum_key, outputs in stages:
        tolerance = _number(record, tolerance_key)
        configured = expected.get(tolerance_key)
        configured_tolerance = None
        try:
            configured_tolerance = float(configured)
        except (TypeError, ValueError, OverflowError):
            pass
        if tolerance is None or tolerance <= 0:
            reasons.append(f"{label}:missing_tolerance")
        elif configured_tolerance is not None and tolerance > configured_tolerance:
            reasons.append(f"{label}:tolerance_exceeds_profile")

        passes = _number(record, f"conv_passes_{label}")
        minimum_passes = _number(expected, minimum_key) or 1.0
        if passes is None or passes < minimum_passes:
            reasons.append(f"{label}:missing_pass_count")
        for metric in ("error", "delta"):
            key = f"conv_{metric}_pct_{label}"
            value = _number(record, key)
            if value is None or value < 0:
                reasons.append(f"{label}:missing_{metric}")
            elif tolerance is not None and tolerance > 0 and value > tolerance:
                reasons.append(f"{label}:{metric}_exceeds_tolerance")
            elif configured_tolerance is not None and value > configured_tolerance:
                reasons.append(f"{label}:{metric}_exceeds_profile")

        for key in outputs:
            if _number(record, key) is None:
                reasons.append(f"{label}:nonfinite_output:{key}")

    for key in ("P_core_total", "P_core_plate_total", "P_wcp_total", "P_winding_total"):
        value = _number(record, key)
        if value is not None and value < 0:
            reasons.append(f"loss:negative_output:{key}")
    for key in ("B_mean_core", "B_max_core"):
        value = _number(record, key)
        if value is not None and value < 0:
            reasons.append(f"loss:negative_output:{key}")
    llt = _number(record, "Llt")
    if llt is not None and llt <= 0:
        reasons.append("matrix:nonpositive_output:Llt")

    # Current production profile uses the skin-free matrix extraction path.
    # Requiring its readbacks keeps older, physically different data out of the
    # strict cohort even when those rows happen to converge numerically.
    skin = expected.get("matrix_skin_mesh")
    if skin == 0:
        checks = {
            "matrix_extraction_backend": "export_rl_matrix",
            "matrix_conductor_policy": "stranded_no_eddy_no_skin",
            "matrix_winding_stranded_count": 2,
            "matrix_conductor_mesh_operation_count": 0,
        }
        for key, wanted in checks.items():
            if not _same(_value(record, key), wanted):
                reasons.append(f"matrix_policy:{key}")
        matrix_plate = _number(record, "matrix_plate_eddy_off_readback_count")
        loss_plate = _number(record, "loss_plate_eddy_on_readback_count")
        if matrix_plate is None or matrix_plate < 0:
            reasons.append("matrix_policy:matrix_plate_eddy_off_readback_count")
        if loss_plate is None or matrix_plate is None or loss_plate != matrix_plate:
            reasons.append("loss_policy:loss_plate_eddy_on_readback_count")
        for key, wanted in (
            ("loss_winding_solid_update_count", 2),
            ("loss_winding_mesh_operation_count", 2),
        ):
            if _number(record, key) != wanted:
                reasons.append(f"loss_policy:{key}")
        loss_mesh = _number(record, "loss_conductor_mesh_operation_count")
        if matrix_plate is None or loss_mesh != 2.0 + float(matrix_plate > 0):
            reasons.append("loss_policy:loss_conductor_mesh_operation_count")

    return reasons


def _thermal_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "result_valid_thermal",
        "thermal_solved",
        "thermal_convergence_available",
        "thermal_converged",
        "thermal_extraction_complete",
        "thermal_rx_power_balance_ok",
    ):
        if not _one(record, key):
            reasons.append(f"thermal_flag:{key}")
    if _number(record, "thermal_required_missing_count") != 0.0:
        reasons.append("thermal_extraction:required_missing")
    iterations = _number(record, "thermal_iterations")
    if iterations is None or iterations <= 0:
        reasons.append("thermal_convergence:iterations")

    flow_limit = _number(record, "thermal_residual_flow_limit")
    energy_limit = _number(record, "thermal_residual_energy_limit")
    if flow_limit is None or not 0 < flow_limit <= 1e-3:
        reasons.append("thermal_convergence:flow_limit")
    if energy_limit is None or not 0 < energy_limit <= 1e-7:
        reasons.append("thermal_convergence:energy_limit")
    for key in FLOW_RESIDUAL_COLUMNS:
        value = _number(record, key)
        if value is None or flow_limit is None or not 0 <= value <= flow_limit:
            reasons.append(f"thermal_convergence:{key}")
    energy = _number(record, "thermal_residual_energy")
    if energy is None or energy_limit is None or not 0 <= energy <= energy_limit:
        reasons.append("thermal_convergence:thermal_residual_energy")

    n2_side = _number(record, "N2_side")
    if n2_side is None or n2_side < 0:
        reasons.append("thermal_extraction:N2_side")
        side_required = True
    else:
        side_required = n2_side > 0
    expected_mask = 15 if side_required else 11
    if _number(record, "thermal_required_group_mask") != float(expected_mask):
        reasons.append("thermal_extraction:required_group_mask")
    required = list(MANDATORY_TEMPERATURE_COLUMNS)
    if side_required:
        required.extend(SIDE_TEMPERATURE_COLUMNS)
    for key in required:
        value = _number(record, key)
        if value is None:
            reasons.append(f"thermal_temperature:nonfinite:{key}")
        elif not MIN_TRUSTED_TEMPERATURE_C < value < MAX_TRUSTED_TEMPERATURE_C:
            reasons.append(f"thermal_temperature:untrusted:{key}")

    explicit_turns = _number(record, "n_explicit_turns")
    if explicit_turns is None or explicit_turns < 0:
        reasons.append("thermal_power:n_explicit_turns")
        expected_model = None
    else:
        expected_model = "homogenized_blocks" if explicit_turns == 0 else "hybrid_explicit"
    model = str(_value(record, "thermal_rx_model") or "")
    if expected_model is None or model != expected_model:
        reasons.append("thermal_power:rx_model")
    group_count = _number(record, "thermal_rx_power_balance_group_count")
    expected = _number(record, "thermal_rx_expected_power_w")
    assigned = _number(record, "thermal_rx_assigned_power_w")
    balance = _number(record, "thermal_rx_power_balance_max_abs_w")
    if group_count is None or group_count < 1:
        reasons.append("thermal_power:group_count")
    if expected is None or expected < 0 or assigned is None or assigned < 0:
        reasons.append("thermal_power:nonfinite")
    elif not math.isclose(assigned, expected, rel_tol=1e-12, abs_tol=1e-9):
        reasons.append("thermal_power:assigned_mismatch")
    if balance is None or not 0 <= balance <= 1e-9:
        reasons.append("thermal_power:balance_error")
    return reasons


def validate_record(
    record: Mapping[str, Any],
    profile: str | Path | Mapping[str, Any] | None = None,
    *,
    require_profile: bool = True,
    require_clean_provenance: bool = True,
    expected_solver_revision: str | None = None,
    expected_library_revision: str | None = None,
) -> ValidationResult:
    profile_data = load_profile(profile)
    shared = []
    if require_profile:
        shared.extend(_profile_reasons(record, profile_data))
    if require_clean_provenance:
        shared.extend(_provenance_reasons(
            record, expected_solver_revision, expected_library_revision
        ))
    em_reasons = shared + _em_reasons(record, profile_data)
    em_valid = not em_reasons
    thermal_only = _thermal_reasons(record)
    thermal_valid = not thermal_only
    reasons = tuple(dict.fromkeys(em_reasons + thermal_only))
    return ValidationResult(
        em_valid=em_valid,
        thermal_valid=thermal_valid,
        full_valid=em_valid and thermal_valid,
        reasons=reasons,
    )


def annotate_validity(
    frame: pd.DataFrame,
    profile: str | Path | Mapping[str, Any] | None = None,
    *,
    require_profile: bool = True,
    require_clean_provenance: bool = True,
    expected_solver_revision: str | None = None,
    expected_library_revision: str | None = None,
) -> pd.DataFrame:
    """Return a copy with recomputed strict validity and quarantine reasons."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    profile_data = load_profile(profile)
    results = [
        validate_record(
            row,
            profile_data,
            require_profile=require_profile,
            require_clean_provenance=require_clean_provenance,
            expected_solver_revision=expected_solver_revision,
            expected_library_revision=expected_library_revision,
        )
        for row in frame.to_dict("records")
    ]
    out = frame.copy()
    out["_strict_valid_em"] = [result.em_valid for result in results]
    out["_strict_valid_thermal"] = [result.thermal_valid for result in results]
    out["_strict_valid_full"] = [result.full_valid for result in results]
    out["_strict_invalid_reasons"] = [";".join(result.reasons) for result in results]
    return out


def strict_valid_count(
    frame: pd.DataFrame,
    profile: str | Path | Mapping[str, Any] | None = None,
    *,
    expected_solver_revision: str | None = None,
    expected_library_revision: str | None = None,
) -> int:
    return int(annotate_validity(
        frame, profile,
        expected_solver_revision=expected_solver_revision,
        expected_library_revision=expected_library_revision,
    )["_strict_valid_full"].sum())
