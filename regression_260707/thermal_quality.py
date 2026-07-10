"""Shared fail-closed quality gates for thermal surrogate training rows."""

import numpy as np
import pandas as pd


THERMAL_STRICT_VALID_COLUMN = "thermal_strict_valid"
THERMAL_TIER_COLUMN = "thermal_training_tier"
THERMAL_TIER_STRICT = "strict_thermal"
THERMAL_TIER_EM_ONLY = "em_only"

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
VALID_RX_MODELS = frozenset(("homogenized_blocks", "hybrid_explicit"))


def _numeric(df, column):
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _finite(values):
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float, na_value=float("nan"))
    return pd.Series(np.isfinite(array), index=values.index, dtype=bool)


def _equals_one(df, column):
    return _numeric(df, column).eq(1).fillna(False)


def strict_thermal_mask(df):
    """Return rows carrying all current thermal evidence needed for training.

    This intentionally does not require an exact solver or profile revision so
    valid rows from mixed campaign profiles can coexist. Missing evidence fails
    closed. In particular, legacy ``thermal_solved=1`` alone is never enough.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    valid = pd.Series(True, index=df.index, dtype=bool)

    for column in (
        "result_valid_em",
        "result_valid_thermal",
        "thermal_solved",
        "thermal_extraction_complete",
        "thermal_convergence_available",
        "thermal_converged",
    ):
        valid &= _equals_one(df, column)

    required_missing = _numeric(df, "thermal_required_missing_count")
    valid &= _finite(required_missing) & required_missing.eq(0)

    iterations = _numeric(df, "thermal_iterations")
    valid &= _finite(iterations) & iterations.gt(0)

    n2_side = _numeric(df, "N2_side")
    n2_finite = _finite(n2_side)
    valid &= n2_finite & n2_side.ge(0).fillna(False)
    side_required = n2_finite & n2_side.gt(0).fillna(False)

    required_mask = _numeric(df, "thermal_required_group_mask")
    expected_mask = pd.Series(
        np.where(side_required.to_numpy(), 15, 11), index=df.index, dtype=float
    )
    valid &= _finite(required_mask) & required_mask.eq(expected_mask)

    for column in MANDATORY_TEMPERATURE_COLUMNS:
        valid &= _finite(_numeric(df, column))
    for column in SIDE_TEMPERATURE_COLUMNS:
        valid &= ~side_required | _finite(_numeric(df, column))

    flow_limit = _numeric(df, "thermal_residual_flow_limit")
    flow_limit_ok = (
        _finite(flow_limit)
        & flow_limit.gt(0).fillna(False)
        & flow_limit.le(1e-3).fillna(False)
    )
    valid &= flow_limit_ok
    for column in FLOW_RESIDUAL_COLUMNS:
        residual = _numeric(df, column)
        valid &= (
            _finite(residual)
            & residual.ge(0).fillna(False)
            & residual.le(flow_limit).fillna(False)
        )

    energy_limit = _numeric(df, "thermal_residual_energy_limit")
    energy = _numeric(df, "thermal_residual_energy")
    valid &= (
        _finite(energy_limit)
        & energy_limit.gt(0).fillna(False)
        & energy_limit.le(1e-7).fillna(False)
        & _finite(energy)
        & energy.ge(0).fillna(False)
        & energy.le(energy_limit).fillna(False)
    )

    balance_groups = _numeric(df, "thermal_rx_power_balance_group_count")
    balance_error = _numeric(df, "thermal_rx_power_balance_max_abs_w")
    expected_power = _numeric(df, "thermal_rx_expected_power_w")
    assigned_power = _numeric(df, "thermal_rx_assigned_power_w")
    powers_close = pd.Series(
        np.isclose(
            assigned_power.to_numpy(dtype=float, na_value=float("nan")),
            expected_power.to_numpy(dtype=float, na_value=float("nan")),
            rtol=1e-12,
            atol=1e-9,
            equal_nan=False,
        ),
        index=df.index,
        dtype=bool,
    )
    valid &= (
        _equals_one(df, "thermal_rx_power_balance_ok")
        & _finite(balance_groups)
        & balance_groups.ge(1).fillna(False)
        & _finite(balance_error)
        & balance_error.ge(0).fillna(False)
        & balance_error.le(1e-9).fillna(False)
        & _finite(expected_power)
        & expected_power.ge(0).fillna(False)
        & _finite(assigned_power)
        & assigned_power.ge(0).fillna(False)
        & powers_close
    )

    if "thermal_rx_model" not in df.columns:
        model = pd.Series("", index=df.index, dtype=object)
    else:
        model = df["thermal_rx_model"].fillna("").astype(str).str.strip()
    valid &= model.isin(VALID_RX_MODELS)
    n_explicit = _numeric(df, "n_explicit_turns")
    explicit_zero = _finite(n_explicit) & n_explicit.eq(0)
    valid &= ~explicit_zero | model.eq("homogenized_blocks")

    return valid.fillna(False).astype(bool)


def annotate_thermal_tier(df):
    """Recompute durable strict-valid and training-tier columns."""
    mask = strict_thermal_mask(df)
    out = df.copy()
    out[THERMAL_STRICT_VALID_COLUMN] = mask.astype("int8")
    out[THERMAL_TIER_COLUMN] = np.where(
        mask.to_numpy(), THERMAL_TIER_STRICT, THERMAL_TIER_EM_ONLY
    )
    return out


def is_temperature_target(target):
    value = str(target or "")
    return value.startswith(("Tprobe_", "T_max_", "T_mean_"))


def target_training_mask(df, target):
    """Return the common quality mask used by every surrogate trainer."""
    if is_temperature_target(target):
        return strict_thermal_mask(df)
    return pd.Series(True, index=df.index, dtype=bool)
