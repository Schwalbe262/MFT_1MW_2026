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

# The strict contract is imported from repo-root scripts, the campaign
# package, and standalone training loops; make module.* resolvable from
# any of those working directories.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from module.core_material_contract import (
    PHYSICS_DATA_REVISION,
    solver_revision_matches_physics_cohort,
)
from module.thermal_probe_contract import (
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
)

try:
    from .model_targets import (
        CORE_REGION_TEMPERATURE_TARGETS,
        SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
    )
except ImportError:  # Script execution with regression_260707 on sys.path.
    from model_targets import (
        CORE_REGION_TEMPERATURE_TARGETS,
        SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = HERE / "verify" / "profiles" / "standard.json"
MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15
POOLED_OPTIONAL_SURFACE_CALCOP_REASON = (
    "pooled_optional_calcop_unavailable:"
    "center_leg_surface_flux_uses_equivalent_faraday_evidence"
)

# Evidence / approval record (operator decision 2026-07-15): the reviewed
# production-physics diff bffbb15..262574a is limited to the pooled AEDT
# adapter, pooled-only overlap-guard scoping, and pooled result-extraction
# plumbing.  It makes no geometry, setup, expression, or material changes.
PHYSICS_EQUIVALENT_SOLVER_REVISIONS: dict[str, frozenset[str]] = {
    "262574a886cef9e0f8f550d12571cf6d54c826e2": frozenset({
        "bffbb15fe2cdec74a72f47e7eb9bacbf0f4e95f7",
        "66ee6685859c207eafdca796120e2e1643f72f5c",
        "f0271da72ff4b9f085b3927769c583c163792adb",
        "f411bf5492669f87896eb657b9e5db2998d219a7",
        "1a5f904214fb39bc83e52f3cc5da6d30977ada34",
    }),
    # Reviewed 2026-07-15: 262574a..26afff8 changes only the pooled
    # Desktop transport/lifecycle, exact-project cache rebind, host-owned DSO
    # handling, and failure quarantine.  Geometry, materials, setups,
    # expressions, and PHYSICS_DATA_REVISION are unchanged.  Keep the runtime
    # solver pin at 26afff8 while reusing the already approved physical rows.
    "26afff8de2936f605783395fbff19d5f1d26b354": frozenset({
        "262574a886cef9e0f8f550d12571cf6d54c826e2",
        "bffbb15fe2cdec74a72f47e7eb9bacbf0f4e95f7",
        "66ee6685859c207eafdca796120e2e1643f72f5c",
        "f0271da72ff4b9f085b3927769c583c163792adb",
        "f411bf5492669f87896eb657b9e5db2998d219a7",
        "1a5f904214fb39bc83e52f3cc5da6d30977ada34",
    }),
}
# Explicitly not enrolled: these physics-adjacent diffs remain unreviewed.
# dba903eb671e37642168afc5578b8e6a93e9c046
# 22d715011a827a111ed32e40da1272b9d47251fe
# 4f585b0540dbe3b2828f991024fdb9f1f2d23b8b
# 513a6f321b997d1866f8c0da57cc27c285b29a5c

MATRIX_REQUIRED_OUTPUTS = (
    "Ltx", "Lrx", "M", "k", "Lmt", "Lmr", "Llt", "Llr",
)
LOSS_REQUIRED_OUTPUTS = (
    "P_core_total", "P_core_plate_total", "P_wcp_total",
    "P_winding_total", "B_mean_core", "B_max_core",
    *SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
)
MANDATORY_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
    *CORE_REGION_TEMPERATURE_TARGETS,
)
SIDE_TEMPERATURE_COLUMNS = (
    "T_max_Rx_side",
    "Tprobe_Rx_side_leeward_max",
)
STACKING_REVISION_SIDE_TEMPERATURE_COLUMNS = (
    "Tprobe_Rx_side_leeward_mean",
    "Tprobe_Rx_side_outer_max",
    "Tprobe_Rx_side_outer_mean",
    "Tprobe_Rx_side_inner_max",
    "Tprobe_Rx_side_inner_mean",
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
    "V1_rms",
    "I1_rated",
    "I2_rated",
    "I2_phase_deg",
    "loss_from_copy",
    "P_target",
    "V2_rms",
    "core_cm",
    "core_x",
    "core_y",
    "core_plate_on",
    "wcp_on",
    "round_corner",
    "plate_temp",
    "air_temp",
    "fan_velocity",
    "k_ins",
    "core_k_thermal",
    "rx_mesh_mode",
    "fan_config",
    "thermal_max_iterations",
    "conductor_temp_C",
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


def _optional_text(value: Any) -> str:
    """Normalize absent scalar values, including schema-union NaNs."""
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError, OverflowError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "<na>", "none"} else text


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


def _uses_explicit_solver_revision_equivalence(
    record: Mapping[str, Any], expected_solver_revision: object,
) -> bool:
    actual = _optional_text(_value(record, "git_hash")).lower()
    expected = _optional_text(expected_solver_revision).lower()
    return (
        actual != expected
        and actual in PHYSICS_EQUIVALENT_SOLVER_REVISIONS.get(
            expected, frozenset()
        )
        and _optional_text(_value(record, "physics_data_revision"))
        == PHYSICS_DATA_REVISION
    )


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
    if expected_solver_revision is not None and not (
        solver_revision_matches_physics_cohort(
            _value(record, "git_hash"),
            expected_solver_revision,
            _value(record, "physics_data_revision"),
        )
        or _uses_explicit_solver_revision_equivalence(
            record, expected_solver_revision
        )
    ):
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
        consecutive = _number(record, f"conv_consecutive_{label}")
        minimum_passes = _number(expected, minimum_key) or 1.0
        if passes is None or passes < 1:
            reasons.append(f"{label}:missing_pass_count")
        if (consecutive is None or consecutive < minimum_passes
                or consecutive != math.floor(consecutive)):
            reasons.append(f"{label}:insufficient_consecutive_convergence")
        elif passes is not None and consecutive > passes:
            reasons.append(f"{label}:invalid_consecutive_convergence")
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

    for key in (
        "P_core_total", "P_core_plate_total", "P_wcp_total",
        "P_winding_total", *SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
    ):
        value = _number(record, key)
        if value is not None and value < 0:
            reasons.append(f"loss:negative_output:{key}")
    winding_total = _number(record, "P_winding_total")
    winding_components = [
        _number(record, key)
        for key in SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
    ]
    if winding_total is not None and all(
        value is not None for value in winding_components
    ) and not math.isclose(
        winding_total,
        sum(winding_components),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        reasons.append("loss:winding_component_sum_mismatch")
    for key in ("B_mean_core", "B_max_core"):
        value = _number(record, key)
        if value is not None and value < 0:
            reasons.append(f"loss:negative_output:{key}")
    if str(_value(record, "physics_data_revision") or "").strip() == (
        PHYSICS_DATA_REVISION
    ):
        winding_readback_applicable = _number(
            record, "winding_flux_linkage_readback_applicable"
        )
        winding_readback_available = _number(
            record, "winding_flux_linkage_readback_available"
        )
        winding_readback_passed = _number(
            record, "winding_flux_linkage_readback_passed"
        )
        winding_readback_reason = _optional_text(_value(
            record, "winding_flux_linkage_readback_reason"
        ))
        winding_readback_status = _optional_text(_value(
            record, "winding_flux_linkage_readback_status"
        ))
        winding_readback_fail_soft = (
            winding_readback_applicable == 0.0
            and winding_readback_available == 0.0
            and winding_readback_passed == 1.0
            and winding_readback_status == "unavailable"
            and winding_readback_reason.startswith(
                "grpc_calcop_unavailable:"
            )
            and len(winding_readback_reason) > len(
                "grpc_calcop_unavailable:"
            )
        )
        winding_readback_available_valid = (
            winding_readback_applicable == 1.0
            and winding_readback_available == 1.0
            and winding_readback_passed == 1.0
            and winding_readback_status == "available"
            and not winding_readback_reason
        )
        winding_readback_evidence_declared = (
            winding_readback_applicable is not None
            or winding_readback_available is not None
            or winding_readback_passed is not None
            or bool(winding_readback_status)
            or bool(winding_readback_reason)
        )
        winding_readback_unavailable_declared = (
            winding_readback_applicable == 0.0
            or winding_readback_available == 0.0
            or winding_readback_status == "unavailable"
        )
        if (
            winding_readback_unavailable_declared
            and not winding_readback_fail_soft
        ):
            reasons.append(
                "native_lamination:"
                "winding_flux_linkage_readback_unavailability_invalid"
            )
        elif (
            winding_readback_evidence_declared
            and not winding_readback_fail_soft
            and not winding_readback_available_valid
        ):
            reasons.append(
                "native_lamination:"
                "winding_flux_linkage_readback_evidence_invalid"
            )
        for key in (
            "core_native_material_readback_attested",
            "core_loss_native_attested",
        ):
            if not _one(record, key):
                reasons.append(f"native_lamination:{key}")
        if (
            not winding_readback_fail_soft
            and not _one(record, "flux_linkage_attested")
        ):
            reasons.append("native_lamination:flux_linkage_attested")
        if not _one(record, "B_mean_faraday_attested"):
            reasons.append("native_lamination:B_mean_faraday_attested")
        loss_error = _number(record, "core_loss_native_rel_error")
        loss_tolerance = _number(record, "core_loss_native_tolerance_rel")
        if (
            loss_error is None or loss_tolerance is None
            or not 0 <= loss_error <= loss_tolerance <= 0.30
        ):
            reasons.append("native_lamination:core_loss_faraday_mass_attestation")
        b_error = _number(
            record, "B_mean_material_vs_sine_analytic_rel_error"
        )
        b_tolerance = _number(record, "B_mean_faraday_tolerance_rel")
        if (
            b_error is None or b_tolerance is None
            or not 0 <= b_error <= b_tolerance <= 0.15
        ):
            reasons.append("native_lamination:B_mean_faraday_attestation")
        if str(_value(record, "core_loss_reference_basis") or "") != (
            "sinusoidal_faraday_Bpack_then_Bmaterial_div_kf_then_"
            "POWERLITE_Wkg_times_effective_mass"
        ):
            reasons.append("native_lamination:core_loss_reference_basis")
        surface_applicable = _number(
            record, "center_leg_surface_flux_integral_applicable"
        )
        surface_reason = str(_value(
            record, "center_leg_surface_flux_integral_reason"
        ) or "")
        surface_reason_approved = (
            (
                surface_reason.startswith("grpc_calcop_unavailable:")
                and len(surface_reason) > len("grpc_calcop_unavailable:")
            )
            or surface_reason == POOLED_OPTIONAL_SURFACE_CALCOP_REASON
        )
        surface_fail_soft = surface_applicable == 0.0 and (
            _number(record, "center_leg_surface_flux_integral_available") == 0.0
            and _number(record, "center_leg_surface_flux_integral_passed") == 1.0
            and str(_value(
                record, "center_leg_surface_flux_integral_status"
            ) or "").strip() == "unavailable"
            and surface_reason_approved
        )
        if surface_applicable == 0.0 and not surface_fail_soft:
            reasons.append(
                "native_lamination:center_leg_surface_flux_unavailability_invalid"
            )
        if winding_readback_fail_soft:
            # The standard B-mean Faraday attestation above remains mandatory.
            # Only CalcOp-backed winding readback and comparisons derived from
            # it are informational when that readback is unavailable.
            pass
        elif surface_fail_soft:
            # Surface CalcOp evidence is informational when the cluster gRPC
            # calculator cannot execute it.  The independent Faraday flux
            # linkage check remains mandatory; the EMF-vs-source-peak
            # deviation is loaded-magnetizing-current circuit physics (I*Z
            # drop, kf-invariant per the approved gate's kf=1.00 control) —
            # recorded as advisory evidence: it must be present and finite,
            # but is not tolerance-gated per row.
            for key, limit in (
                ("Tx_flux_linkage_faraday_rel_error", 0.01),
            ):
                value = _number(record, key)
                if value is None or not 0 <= value <= limit:
                    reasons.append(f"native_lamination:{key}")
            advisory = _number(record, "Tx_induced_vs_source_peak_rel_error")
            if advisory is None or advisory < 0:
                reasons.append(
                    "native_lamination:Tx_induced_vs_source_peak_rel_error"
                )
        else:
            for key in (
                "core_surface_flux_vs_linkage_rel_error",
                "core_surface_flux_vs_induced_voltage_rel_error",
            ):
                value = _number(record, key)
                if value is None or not 0 <= value <= 0.05:
                    reasons.append(f"native_lamination:{key}")
        if str(_value(
            record, "core_native_model_approval_status"
        ) or "") != "approved_by_isolated_solved_kf_ab":
            reasons.append("native_lamination:solved_ab_not_approved")
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
    if str(_value(record, "physics_data_revision") or "").strip() == (
        PHYSICS_DATA_REVISION
    ):
        if str(_value(record, "thermal_core_loss_source") or "") != (
            "aedt_native_lamination_loss_attested_then_margin_adjusted"
        ):
            reasons.append("native_thermal:core_loss_source")
        count = _number(record, "thermal_core_native_readback_count")
        if count is None or count < 1 or count != math.floor(count):
            reasons.append("native_thermal:readback_count")
        balance = _number(record, "thermal_core_native_restored_rel_error")
        if balance is None or not 0 <= balance <= 1e-12:
            reasons.append("native_thermal:restored_power_balance")
        margin = _number(record, "thermal_core_loss_correction_factor")
        if margin is None or not math.isclose(
            margin, 1.15, rel_tol=0.0, abs_tol=1e-12
        ):
            reasons.append("native_thermal:loss_margin")
        expected = _number(
            record, "thermal_core_full_expected_margin_adjusted_w"
        )
        p_core = _number(record, "P_core_total")
        if (
            expected is None or p_core is None
            or not math.isclose(expected, p_core, rel_tol=1e-12, abs_tol=1e-9)
        ):
            reasons.append("native_thermal:em_to_icepak_core_power")
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
    new_probe_contract_required = (
        side_required
        and str(_value(record, "physics_data_revision") or "").strip()
        == PHYSICS_DATA_REVISION
    )
    if new_probe_contract_required:
        required.extend(STACKING_REVISION_SIDE_TEMPERATURE_COLUMNS)
    for key in required:
        value = _number(record, key)
        if value is None:
            reasons.append(f"thermal_temperature:nonfinite:{key}")
        elif not MIN_TRUSTED_TEMPERATURE_C < value < MAX_TRUSTED_TEMPERATURE_C:
            reasons.append(f"thermal_temperature:untrusted:{key}")

    if new_probe_contract_required:
        if str(_value(
            record, "thermal_rx_side_probe_contract_version"
        ) or "") != RX_SIDE_FACE_PROBE_CONTRACT_VERSION:
            reasons.append("thermal_probe:rx_side_contract_version")
        if str(_value(
            record, "thermal_rx_side_probe_max_rule"
        ) or "") != RX_SIDE_FACE_MAX_RULE:
            reasons.append("thermal_probe:rx_side_max_rule")
        if str(_value(
            record, "thermal_rx_side_probe_mean_rule"
        ) or "") != RX_SIDE_FACE_MEAN_RULE:
            reasons.append("thermal_probe:rx_side_mean_rule")
        selected_face = str(_value(
            record, "thermal_rx_side_probe_selected_face"
        ) or "")
        if selected_face not in {
            "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
            "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
        }:
            reasons.append("thermal_probe:rx_side_selected_face")
        mode = str(_value(record, "thermal_symmetry") or "").strip().lower()
        expected_face_count = 4.0 if mode == "full" else 2.0
        if _number(
            record, "thermal_rx_side_probe_face_count"
        ) != expected_face_count:
            reasons.append("thermal_probe:rx_side_face_count")

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
    records = frame.to_dict("records")
    results = [
        validate_record(
            row,
            profile_data,
            require_profile=require_profile,
            require_clean_provenance=require_clean_provenance,
            expected_solver_revision=expected_solver_revision,
            expected_library_revision=expected_library_revision,
        )
        for row in records
    ]
    out = frame.copy()
    out["_strict_valid_em"] = [result.em_valid for result in results]
    out["_strict_valid_thermal"] = [result.thermal_valid for result in results]
    out["_strict_valid_full"] = [result.full_valid for result in results]
    out["_strict_invalid_reasons"] = [";".join(result.reasons) for result in results]
    out.attrs["provenance_equivalent_rows"] = sum(
        result.full_valid
        and require_clean_provenance
        and _uses_explicit_solver_revision_equivalence(
            row, expected_solver_revision
        )
        for row, result in zip(records, results)
    )
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
