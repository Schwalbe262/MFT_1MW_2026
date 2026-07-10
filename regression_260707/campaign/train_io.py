"""Stable, human-readable training I/O projection of the raw campaign dataset.

The raw ``train.parquet`` remains the lossless audit dataset.  This module only
selects deterministic design inputs, geometry-derived inputs, aggregate physical
outputs, and the minimum quality/provenance needed to interpret each row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_IO_SCHEMA_VERSION = 1

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
    "plate_temp",
    "air_temp",
    "fan_velocity",
    "fan_config",
    "k_ins",
    "core_k_thermal",
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
    "Ae_m2",
    "core_vol_m3",
    "core_mass_kg",
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
    "loss_sym_on",
    "thermal_symmetry",
    "matrix_on",
    "loss_on",
    "thermal_on",
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
    "P_core_plate_total",
    "P_wcp_total",
    "P_Rx_side_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_group",
    "B_mean_core",
    "B_max_core",
    "I1_mag_peak",
    "I1_phase_deg",
    "phi_deg",
    "I2_phase_used_deg",
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
    "Tprobe_Rx_side_side_max",
    "Tprobe_Rx_side_side_mean",
    "Tprobe_core_center_max",
    "Tprobe_core_center_mean",
)

QUALITY_COLUMNS = (
    "result_valid_em",
    "result_valid_thermal",
    "conv_error_pct_matrix",
    "conv_error_pct_loss",
)

PROVENANCE_COLUMNS = (
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
    *AGGREGATE_TEMPERATURE_COLUMNS,
    *QUALITY_COLUMNS,
    *PROVENANCE_COLUMNS,
)


def _column_or_missing(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].copy()
    return pd.Series(np.nan, index=frame.index, name=column)


def build_train_io(master: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed-schema physical I/O view without mutating ``master``."""
    if not isinstance(master, pd.DataFrame):
        raise TypeError("master must be a pandas DataFrame")

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
        *AGGREGATE_TEMPERATURE_COLUMNS,
        *QUALITY_COLUMNS,
        *PROVENANCE_COLUMNS,
    ):
        data[column] = _column_or_missing(master, column)

    out = pd.DataFrame(data, index=master.index)
    return out.loc[:, TRAIN_IO_COLUMNS].reset_index(drop=True)
