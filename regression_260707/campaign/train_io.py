"""Stable, human-readable training I/O projection of the raw campaign dataset.

The raw ``train.parquet`` remains the lossless audit dataset.  This module only
selects deterministic design inputs, geometry-derived inputs, aggregate physical
outputs, and the minimum quality/provenance needed to interpret each row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_IO_SCHEMA_VERSION = 9

CAPACITANCE_RECOVERY_CONTRACT = "mft-capacitance-lc-inverse-v1"
CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F = 5.1e-11
CAPACITANCE_RECOVERY_ATTR = "capacitance_recovery"
CAPACITANCE_RECOVERY_COLUMNS = (
    "capacitance_recovery_contract",
    "capacitance_recovery_required",
    "capacitance_recovered_from_resonance",
    "capacitance_recovery_max_abs_delta_F",
)
CAPACITANCE_RECOVERY_SPECS = (
    ("C_tx_tx_F", "f_res_tx_self_Hz", "cap_L_tx_self_H"),
    ("C_rx_rx_F", "f_res_rx_self_Hz", "cap_L_rx_self_H"),
    ("C_tx_rx_F", "f_res_interwinding_Hz", "cap_L_leakage_H"),
)

IDENTITY_COLUMNS = (
    "project_name",
    "saved_at",
    "task_id",
    "task_name",
    "source",
    "sample_weight",
)

DESIGN_INPUT_COLUMNS = (
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
    "core_plate_on",
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
    "wcp_t",
    "wcp_pad_t",
    "wcp_len_pct",
    "wcp_len_x",
    "wcp_on",
    "core_plate_pad_t",
    "round_corner",
    "corner_radius",
)

PHYSICAL_CONTEXT_COLUMNS = (
    "freq",
    "V1_rms",
    "I1_rated",
    "I2_rated",
    "I2_phase_deg",
    "P_target",
    "V2_rms",
    "core_cm",
    "core_x",
    "core_y",
    "core_lamination_factor",
    "core_loss_margin",
    "core_cm_base",
    "core_cm_assigned",
    "core_cm_equivalent_gross_unassigned",
    "core_x_contract",
    "core_y_contract",
    "core_loss_correction_factor",
    "core_B_average_to_material_factor",
    "core_B_macro_to_material_factor",
    "plate_temp",
    "air_temp",
    "fan_velocity",
    "fan_config",
    "k_ins",
    "core_k_thermal",
    "core_k_anisotropic",
    "core_k_alloy",
    "core_k_interlayer",
    "conductor_temp_C",
)

GEOMETRY_DERIVED_COLUMNS = (
    "N1",
    "N2",
    "core_depth_each",
    "nwl1_main",
    "nwl1_side",
    "nwl2_main",
    "nwl2_side",
    "wff1_main",
    "wff1_side",
    "wff2_main",
    "wff2_side",
    "coil_gap_layer1",
    "coil_gap_layer2",
    "nwb1_main_y",
    "h_gap1",
    "h_gap2",
    "sl2_main_x",
    "sl2_main_y",
    "sl1_main_x",
    "sl1_main_y",
    "sl1_side_x",
    "sl1_side_y",
    "sl2_side_x",
    "sl2_side_y",
    "w1c_w2s_gap_x_actual",
    "wcp_len_ref_x",
    "Ae_m2",
    "Ae_gross_m2",
    "Ae_effective_m2",
    "core_vol_m3",
    "core_vol_gross_m3",
    "core_vol_effective_m3",
    "core_mass_kg",
    "core_mass_gross_kg",
    "core_mass_effective_kg",
    "MLT_Tx_mm",
    "MLT_Rx_main_mm",
    "MLT_Rx_side_mm",
    "cu_mass_Tx_kg",
    "cu_mass_Rx_main_kg",
    "cu_mass_Rx_side_kg",
    "cu_mass_total_kg",
    "window_fill_x",
    "window_fill_z1",
    "aspect_h1_l2",
    "aspect_w1_l2",
)

ANALYSIS_BASIS_COLUMNS = (
    "full_model",
    "matrix_skin_mesh",
    "loss_sym_on",
    "thermal_symmetry",
    "n_explicit_turns",
    "thermal_rx_model",
    "thermal_core_conductivity_model",
    "matrix_on",
    "loss_on",
    "thermal_on",
    "cap_on",
)

INDUCTANCE_SOURCE_COLUMNS = (
    "Ltx",
    "Lrx",
    "M",
    "Lmt",
    "Lmr",
    "Llt",
    "Llr",
)
INDUCTANCE_PHYSICAL_COLUMNS = tuple(
    f"{column}_phys" for column in INDUCTANCE_SOURCE_COLUMNS
)
INDUCTANCE_BASIS_COLUMNS = (
    "inductance_source_basis",
    "inductance_to_physical_factor",
)

AGGREGATE_EM_OUTPUT_COLUMNS = (
    "k",
    "P_winding_total",
    "P_core_total",
    "P_core_total_native_raw_W",
    "P_core_total_expected_from_Bavg_integral",
    "P_core_total_expected_native_raw_W",
    "core_loss_native_rel_error",
    "core_loss_native_tolerance_rel",
    "core_loss_native_attested",
    "core_loss_margin_applied",
    "Bavg_power_volume_integral",
    "core_flux_section_retained_area_m2",
    "core_flux_integral_reported_Wb",
    "core_flux_integral_physical_Wb",
    "B_core_section_average",
    "B_core_section_material",
    "B_core_section_vs_volume_mean_rel_error",
    "P_core_plate_total",
    "P_wcp_total",
    "P_Rx_side_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_group",
    "B_mean_core",
    "B_max_core",
    "B_mean_core_average",
    "B_max_core_average_diagnostic",
    "B_mean_core_macro",
    "B_max_core_macro",
    "B_mean_core_material",
    "B_max_core_material",
    "B_design_square_material_analytic",
    "B_ac_sine_material_analytic",
    "B_ac_sine_average_analytic",
    "B_mean_material_vs_sine_analytic_rel_error",
    "Tx_induced_voltage_peak_V",
    "Tx_induced_voltage_reported_peak_V",
    "Tx_flux_linkage_peak_Wb_turn",
    "Tx_flux_linkage_reported_peak_Wb_turn",
    "Tx_flux_linkage_faraday_rel_error",
    "Tx_induced_vs_source_peak_rel_error",
    "B_flux_linkage_material",
    "core_surface_flux_vs_linkage_rel_error",
    "core_surface_flux_vs_induced_voltage_rel_error",
    "B_flux_linkage_vs_sine_analytic_rel_error",
    "flux_linkage_attested",
    "winding_flux_linkage_readback_status",
    "winding_flux_linkage_readback_applicable",
    "winding_flux_linkage_readback_available",
    "winding_flux_linkage_readback_passed",
    "winding_flux_linkage_readback_reason",
    "loss_report_to_physical_flux_factor",
    "I1_mag_peak",
    "I1_phase_deg",
    "phi_deg",
    "I2_phase_used_deg",
)

ELECTROSTATIC_OUTPUT_COLUMNS = (
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "f_res_tx_self_Hz",
    "f_res_rx_self_Hz",
    "f_res_interwinding_Hz",
)

AGGREGATE_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Tx_leeward_mean",
    "Tprobe_Tx_side_max",
    "Tprobe_Tx_side_mean",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_main_leeward_mean",
    "Tprobe_Rx_main_side_max",
    "Tprobe_Rx_main_side_mean",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_Rx_side_leeward_mean",
    "Tprobe_Rx_side_outer_max",
    "Tprobe_Rx_side_outer_mean",
    "Tprobe_Rx_side_inner_max",
    "Tprobe_Rx_side_inner_mean",
    "Tprobe_Rx_side_side_max",
    "Tprobe_Rx_side_side_mean",
    "Tprobe_Rx_side_flow_leeward_max",
    "Tprobe_Rx_side_flow_leeward_mean",
    "Tprobe_Rx_side1_inner_max",
    "Tprobe_Rx_side1_inner_mean",
    "Tprobe_Rx_side2_side_max",
    "Tprobe_Rx_side2_side_mean",
    "Tprobe_Rx_side2_inner_max",
    "Tprobe_Rx_side2_inner_mean",
    "Tprobe_Rx_side2_flow_leeward_max",
    "Tprobe_Rx_side2_flow_leeward_mean",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_center_leg_mean",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_side_leg_mean",
    "Tprobe_core_top_yoke_max",
    "Tprobe_core_top_yoke_mean",
    "Tprobe_core_center_max",
    "Tprobe_core_center_mean",
)

QUALITY_COLUMNS = (
    "result_valid_em",
    "result_valid_thermal",
    "conv_error_pct_matrix",
    "conv_error_pct_loss",
)

TIMING_PROVENANCE_COLUMNS = (
    "time_matrix",
    "time_loss",
    "time_thermal",
    "time",
)

PROVENANCE_COLUMNS = (
    "physics_data_revision",
    "core_material_contract_version",
    "core_material_identity",
    "core_geometry_material_basis",
    "core_stacking_model",
    "core_stacking_type",
    "core_stacking_direction_leg",
    "core_stacking_direction_yoke",
    "core_stacking_direction_basis",
    "core_corner_orientation_model",
    "core_native_model_approval_status",
    "core_native_material_readback_attested",
    "core_native_leg_material_name",
    "core_native_yoke_material_name",
    "core_native_leg_stacking_direction_readback",
    "core_native_yoke_stacking_direction_readback",
    "core_native_leg_stacking_factor_readback",
    "core_native_yoke_stacking_factor_readback",
    "core_native_cm_readback",
    "core_segmented_piece_count",
    "core_segmented_area_rel_error",
    "core_segmented_volume_rel_error",
    "core_segmented_mass_rel_error",
    "core_flux_section_full_area_rel_error",
    "core_flux_section_symmetry_area_rel_error",
    "core_lamination_factor_source",
    "core_lamination_factor_datasheet_guaranteed_min",
    "core_lamination_factor_user_conservative_candidate",
    "core_lamination_factor_ab_candidates",
    "core_loss_model_source",
    "core_loss_margin_source",
    "core_loss_application_basis",
    "core_loss_expected_integral_basis",
    "core_flux_density_output_basis",
    "maxwell_B_output_basis",
    "maxwell_B_peak_operator",
    "maxwell_voltage_source_basis",
    "loss_report_flux_basis",
    "B_max_core_usage",
    "core_mass_density_kg_m3",
    "core_mass_density_source",
    "core_mass_definition",
    "core_permeability_model",
    "core_bulk_conductivity_model",
    "core_loss_equiv_cut_depth_model",
    "Ae_m2_basis",
    "core_mass_kg_basis",
    "thermal_core_k_inplane",
    "thermal_core_k_throughstack",
    "thermal_core_loss_contract_version",
    "thermal_core_loss_source",
    "thermal_core_loss_correction_factor",
    "thermal_core_expected_injected_w",
    "thermal_core_requested_wrapper_echo_w",
    "thermal_core_native_readback_w",
    "thermal_core_restore_factor",
    "thermal_core_native_restored_full_w",
    "thermal_core_full_expected_margin_adjusted_w",
    "thermal_core_native_restored_rel_error",
    "thermal_core_native_readback_count",
    "thermal_core_power_balance_abs_error_w",
    "thermal_core_power_balance_rel_error",
    "thermal_rx_side_probe_contract_version",
    "thermal_rx_side_probe_max_rule",
    "thermal_rx_side_probe_mean_rule",
    "thermal_rx_side_probe_selected_face",
    "thermal_rx_side_probe_face_count",
    "git_hash",
    "git_dirty",
    "pyaedt_library_git_hash",
    "pyaedt_library_git_dirty",
)

TRAIN_IO_COLUMNS = (
    "train_io_schema_version",
    *IDENTITY_COLUMNS,
    *DESIGN_INPUT_COLUMNS,
    *PHYSICAL_CONTEXT_COLUMNS,
    *GEOMETRY_DERIVED_COLUMNS,
    *ANALYSIS_BASIS_COLUMNS,
    *INDUCTANCE_BASIS_COLUMNS,
    *INDUCTANCE_PHYSICAL_COLUMNS,
    *AGGREGATE_EM_OUTPUT_COLUMNS,
    *ELECTROSTATIC_OUTPUT_COLUMNS,
    *CAPACITANCE_RECOVERY_COLUMNS,
    *AGGREGATE_TEMPERATURE_COLUMNS,
    *QUALITY_COLUMNS,
    *TIMING_PROVENANCE_COLUMNS,
    *PROVENANCE_COLUMNS,
)


def _column_or_missing(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].copy()
    return pd.Series(np.nan, index=frame.index, name=column)


def recover_quantized_capacitance_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy RESULT_JSON capacitance rounding repaired.

    The solver emitted each physical LC resonance and its corresponding
    physical inductance independently of the rounded capacitance JSON field.
    Therefore ``C = 1 / ((2*pi*f)**2*L)`` recovers the pre-serialization value.

    Recovery is row-atomic across all three targets.  It applies only to
    ``cap_on=1`` rows whose L/f evidence is positive and finite and whose
    stored capacitances are finite and non-negative.  The legacy serializer
    can introduce at most half of one 1e-10-F step; a larger discrepancy is a
    semantic or unit mismatch and fails closed instead of rewriting evidence.
    Frames lacking the LC evidence columns pass through unchanged apart from
    explicit recovery audit fields.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    out = frame.copy()
    evidence_columns = set()
    for target, frequency, inductance in CAPACITANCE_RECOVERY_SPECS:
        evidence_columns.update((target, frequency, inductance))
    required = {"cap_on", *evidence_columns}
    missing = tuple(sorted(required.difference(out.columns)))
    present_evidence = tuple(sorted(evidence_columns.intersection(out.columns)))
    legacy_passthrough = not present_evidence
    if not legacy_passthrough and missing:
        raise ValueError(
            "partial capacitance recovery schema: "
            f"present_evidence={list(present_evidence)}, "
            f"missing={list(missing)}, "
            f"contract={CAPACITANCE_RECOVERY_CONTRACT}"
        )

    recovered_flag = pd.Series(0, index=out.index, dtype="int8")
    recovery_required = pd.Series(0, index=out.index, dtype="int8")
    row_max_delta = pd.Series(np.nan, index=out.index, dtype=float)
    contract_column = pd.Series(pd.NA, index=out.index, dtype="string")
    audit = {
        "contract": CAPACITANCE_RECOVERY_CONTRACT,
        "max_allowed_abs_delta_F": CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F,
        "row_count": int(len(out)),
        "cap_enabled_row_count": 0,
        "eligible_row_count": 0,
        "recovered_row_count": 0,
        "max_observed_abs_delta_F": None,
        "missing_columns": list(missing),
        "status": (
            "evidence_unavailable_legacy_passthrough"
            if legacy_passthrough else "applied"
        ),
    }

    if not legacy_passthrough and len(out):
        cap_enabled = pd.to_numeric(out["cap_on"], errors="coerce").eq(1)
        recovery_required.loc[cap_enabled] = 1
        audit["cap_enabled_row_count"] = int(cap_enabled.sum())
        eligible = cap_enabled.copy()
        reconstructed = {}
        stored = {}
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for target, frequency, inductance in CAPACITANCE_RECOVERY_SPECS:
                frequency_values = pd.to_numeric(
                    out[frequency], errors="coerce"
                ).astype(float)
                inductance_values = pd.to_numeric(
                    out[inductance], errors="coerce"
                ).astype(float)
                stored_values = pd.to_numeric(
                    out[target], errors="coerce"
                ).astype(float)
                recovered_values = 1.0 / (
                    (2.0 * np.pi * frequency_values) ** 2
                    * inductance_values
                )
                inputs_valid = (
                    np.isfinite(frequency_values)
                    & frequency_values.gt(0.0)
                    & np.isfinite(inductance_values)
                    & inductance_values.gt(0.0)
                )
                stored_valid = (
                    np.isfinite(stored_values) & stored_values.ge(0.0)
                )
                recovered_valid = (
                    np.isfinite(recovered_values) & recovered_values.gt(0.0)
                )
                eligible &= inputs_valid & stored_valid & recovered_valid
                reconstructed[target] = recovered_values
                stored[target] = stored_values

        deltas = pd.DataFrame(
            {
                target: (stored[target] - reconstructed[target]).abs()
                for target, _, _ in CAPACITANCE_RECOVERY_SPECS
            },
            index=out.index,
        )
        row_max_delta = deltas.max(axis=1, skipna=False)
        semantic_mismatch = eligible & row_max_delta.gt(
            CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F
        )
        if semantic_mismatch.any():
            mismatch_count = int(semantic_mismatch.sum())
            mismatch_max = float(row_max_delta.loc[semantic_mismatch].max())
            raise ValueError(
                "capacitance recovery semantic mismatch: "
                f"{mismatch_count} row(s), max_abs_delta_F={mismatch_max:.17g}, "
                f"allowed={CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F:.17g}, "
                f"contract={CAPACITANCE_RECOVERY_CONTRACT}"
            )

        recoverable = eligible
        for target, _, _ in CAPACITANCE_RECOVERY_SPECS:
            out.loc[recoverable, target] = reconstructed[target].loc[recoverable]
        recovered_flag.loc[recoverable] = 1
        contract_column.loc[recoverable] = CAPACITANCE_RECOVERY_CONTRACT
        row_max_delta = row_max_delta.where(eligible)
        audit["eligible_row_count"] = int(eligible.sum())
        audit["recovered_row_count"] = int(recoverable.sum())
        if recoverable.any():
            audit["max_observed_abs_delta_F"] = float(
                row_max_delta.loc[recoverable].max()
            )

    out["capacitance_recovery_contract"] = contract_column
    out["capacitance_recovery_required"] = recovery_required
    out["capacitance_recovered_from_resonance"] = recovered_flag
    out["capacitance_recovery_max_abs_delta_F"] = row_max_delta
    out.attrs.update(frame.attrs)
    out.attrs[CAPACITANCE_RECOVERY_ATTR] = audit
    return out


def capacitance_recovery_audit(frame: pd.DataFrame) -> dict:
    """Return stable JSON-safe recovery provenance for model artifacts."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    value = frame.attrs.get(CAPACITANCE_RECOVERY_ATTR)
    if isinstance(value, dict):
        return dict(value)
    return {
        "contract": CAPACITANCE_RECOVERY_CONTRACT,
        "max_allowed_abs_delta_F": CAPACITANCE_QUANTIZATION_MAX_ABS_DELTA_F,
        "row_count": int(len(frame)),
        "cap_enabled_row_count": None,
        "eligible_row_count": None,
        "recovered_row_count": 0,
        "max_observed_abs_delta_F": None,
        "missing_columns": None,
        "status": "audit_unavailable",
    }


def add_wcp_length_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Backfill winding cold-plate reference and percentage losslessly."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    out = frame.copy()
    length_mm = pd.to_numeric(
        _column_or_missing(out, "wcp_len_x"), errors="coerce"
    )
    supplied_reference_mm = pd.to_numeric(
        _column_or_missing(out, "wcp_len_ref_x"), errors="coerce"
    )
    straight_x = pd.to_numeric(
        _column_or_missing(out, "sl1_main_x"), errors="coerce"
    )
    rounded = pd.to_numeric(
        _column_or_missing(out, "round_corner"), errors="coerce"
    ).eq(1)
    radius = pd.to_numeric(
        _column_or_missing(out, "corner_radius"), errors="coerce"
    ).fillna(0.0)
    derived_reference_mm = straight_x - radius.where(rounded, 0.0) * 2.0
    # Geometry and exact mm are authoritative. Never preserve a supplied pct
    # (or a conflicting supplied reference) when it can be recomputed.
    reference_mm = derived_reference_mm.where(
        derived_reference_mm.gt(0),
        supplied_reference_mm.where(supplied_reference_mm.gt(0)),
    )
    derived_pct = length_mm * 100.0 / reference_mm.where(reference_mm.gt(0))
    out["wcp_len_ref_x"] = reference_mm
    out["wcp_len_pct"] = derived_pct
    return out


def build_train_io(master: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed-schema physical I/O view without mutating ``master``."""
    if not isinstance(master, pd.DataFrame):
        raise TypeError("master must be a pandas DataFrame")
    master = recover_quantized_capacitance_targets(master)
    master = add_wcp_length_features(master)

    data = {
        "train_io_schema_version": pd.Series(
            TRAIN_IO_SCHEMA_VERSION, index=master.index, dtype=int
        )
    }
    for column in (
        *IDENTITY_COLUMNS,
        *DESIGN_INPUT_COLUMNS,
        *PHYSICAL_CONTEXT_COLUMNS,
        *GEOMETRY_DERIVED_COLUMNS,
        *ANALYSIS_BASIS_COLUMNS,
    ):
        data[column] = _column_or_missing(master, column)

    full_model = pd.to_numeric(
        _column_or_missing(master, "full_model"), errors="coerce"
    )
    factor = pd.Series(np.nan, index=master.index, dtype=float)
    factor.loc[full_model.eq(0)] = 2.0
    factor.loc[full_model.eq(1)] = 1.0
    basis = pd.Series(pd.NA, index=master.index, dtype="string")
    basis.loc[full_model.eq(0)] = "eighth_symmetry"
    basis.loc[full_model.eq(1)] = "full_model"
    data["inductance_source_basis"] = basis
    data["inductance_to_physical_factor"] = factor
    for source, physical in zip(
        INDUCTANCE_SOURCE_COLUMNS, INDUCTANCE_PHYSICAL_COLUMNS
    ):
        values = pd.to_numeric(_column_or_missing(master, source), errors="coerce")
        data[physical] = values * factor

    for column in (
        *AGGREGATE_EM_OUTPUT_COLUMNS,
        *ELECTROSTATIC_OUTPUT_COLUMNS,
        *CAPACITANCE_RECOVERY_COLUMNS,
        *AGGREGATE_TEMPERATURE_COLUMNS,
        *QUALITY_COLUMNS,
        *TIMING_PROVENANCE_COLUMNS,
        *PROVENANCE_COLUMNS,
    ):
        data[column] = _column_or_missing(master, column)

    out = pd.DataFrame(data, index=master.index)
    out = out.loc[:, TRAIN_IO_COLUMNS].reset_index(drop=True)
    out.attrs.update(master.attrs)
    return out
