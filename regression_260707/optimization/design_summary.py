"""Human-readable physical summary fields for NSGA-II Pareto candidates."""
from __future__ import annotations

import math

try:
    from ..model_targets import SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
    from .geometry_metrics import bounding_box_lit
except ImportError:  # Direct script execution with regression root on sys.path.
    from model_targets import SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
    from optimization.geometry_metrics import bounding_box_lit


COMPONENT_LOSS_TARGETS = SURROGATE_WINDING_COMPONENT_LOSS_TARGETS
B_AREA_BASIS_GROSS_WITH_LAMINATION = (
    "gross_geometry_times_lamination_factor"
)
B_AREA_BASIS_DATASHEET_NET = "datasheet_net_Ac"


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _nonnegative(value, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise ValueError(f"{label} is negative")
    return number


def _turn_count(row, name: str) -> int:
    number = _nonnegative(row[name], name)
    integer = int(number)
    if not math.isclose(number, integer, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} is not an integer turn count")
    return integer


def rated_power_w(row) -> float:
    """Return the physical output-power basis used for efficiency reporting."""
    target = _nonnegative(row["P_target"], "P_target")
    if target > 0:
        return target
    primary = _nonnegative(row["V1_rms"], "V1_rms") * _nonnegative(
        row["I1_rated"], "I1_rated"
    )
    secondary = _nonnegative(row["V2_rms"], "V2_rms") * _nonnegative(
        row["I2_rated"], "I2_rated"
    )
    candidates = [value for value in (primary, secondary) if value > 0]
    if not candidates:
        raise ValueError("rated power is not positive")
    # The lower port rating is the transferable transformer power.
    return min(candidates)


def _primary_electrical_basis(row) -> tuple[int, float, float]:
    turns = _turn_count(row, "N1_main") + _turn_count(row, "N1_side")
    if turns <= 0:
        raise ValueError("primary turn count is not positive")
    voltage = _nonnegative(row["V1_rms"], "V1_rms")
    frequency = _nonnegative(row["freq"], "freq")
    if voltage <= 0 or frequency <= 0:
        raise ValueError("V1_rms and freq must be positive")
    return turns, voltage, frequency


def effective_core_area_m2(
    row,
    *,
    core_lamination_factor,
    area_basis,
) -> tuple[float, float]:
    """Return raw ``Ae_m2`` and the effective area used by the B equation.

    Decoded ``Ae_m2`` is gross geometric iron-pack area, so the production
    basis applies the datasheet lamination factor.  If a future input supplies
    datasheet net ``Ac`` directly, callers must explicitly select
    ``datasheet_net_Ac`` and the factor is not multiplied a second time.
    """
    raw_area_m2 = _nonnegative(row["Ae_m2"], "Ae_m2")
    if raw_area_m2 <= 0:
        raise ValueError("Ae_m2 is not positive")
    factor = _nonnegative(
        core_lamination_factor, "core_lamination_factor"
    )
    if not 0 < factor <= 1:
        raise ValueError("core_lamination_factor must be in (0, 1]")
    if area_basis == B_AREA_BASIS_GROSS_WITH_LAMINATION:
        return raw_area_m2, raw_area_m2 * factor
    if area_basis == B_AREA_BASIS_DATASHEET_NET:
        return raw_area_m2, raw_area_m2
    raise ValueError(f"unsupported B area basis: {area_basis!r}")


def design_analytical_b_field_t(
    row,
    *,
    core_lamination_factor,
    area_basis,
) -> float:
    """Bipolar-square design flux density using effective physical iron area."""
    turns, voltage, frequency = _primary_electrical_basis(row)
    _, effective_area_m2 = effective_core_area_m2(
        row,
        core_lamination_factor=core_lamination_factor,
        area_basis=area_basis,
    )
    return voltage / (4.0 * frequency * turns * effective_area_m2)


def legacy_0p7_b_field_t(row) -> float:
    """Audit-only legacy 0.7 stacking-factor flux-density calculation."""
    turns, voltage, frequency = _primary_electrical_basis(row)
    limb_width_m = _nonnegative(row["l1"], "l1") * 1e-3
    core_depth_m = _nonnegative(row["w1"], "w1") * 1e-3
    legacy_area_m2 = 2.0 * core_depth_m * limb_width_m * 0.7
    if legacy_area_m2 <= 0:
        raise ValueError("legacy 0.7 analytical core area is not positive")
    return voltage / (4.0 * frequency * turns * legacy_area_m2)


def pareto_design_summary(
    row,
    predictions: dict,
    total_loss_w,
    *,
    leakage_target_uH,
    core_lamination_factor,
    B_area_basis,
) -> dict:
    """Build physical/reporting columns for one decoded Pareto candidate.

    ``predictions`` contains full-physical surrogate outputs.  Separate Tx and
    Rx losses are preserved as their own predictions; the authoritative
    optimization loss remains the sum of the four production aggregate loss
    targets and is the value used for efficiency.
    """
    volume_l, dimensions = bounding_box_lit(row)
    width_mm, length_mm, height_mm = map(float, dimensions)
    total_loss = _nonnegative(total_loss_w, "total_loss_W")

    primary_loss = _nonnegative(
        predictions["P_Tx_main_group"], "P_Tx_main_group"
    )
    secondary_center_loss = _nonnegative(
        predictions["P_Rx_main_group"], "P_Rx_main_group"
    )
    secondary_side_loss = _nonnegative(
        predictions["P_Rx_side_total"], "P_Rx_side_total"
    )
    core_loss = _nonnegative(predictions["P_core_total"], "P_core_total")
    core_plate_loss = _nonnegative(
        predictions["P_core_plate_total"], "P_core_plate_total"
    )
    winding_plate_loss = _nonnegative(
        predictions["P_wcp_total"], "P_wcp_total"
    )
    winding_total = _nonnegative(
        predictions["P_winding_total"], "P_winding_total"
    )
    leakage = _nonnegative(predictions["Llt_phys"], "Llt_phys")
    design_b_field = design_analytical_b_field_t(
        row,
        core_lamination_factor=core_lamination_factor,
        area_basis=B_area_basis,
    )
    legacy_b_field = legacy_0p7_b_field_t(row)
    gross_iron_area_m2, effective_iron_area_m2 = effective_core_area_m2(
        row,
        core_lamination_factor=core_lamination_factor,
        area_basis=B_area_basis,
    )
    leakage_target = _nonnegative(leakage_target_uH, "leakage_target_uH")
    aggregate_total = (
        winding_total + core_loss + core_plate_loss + winding_plate_loss
    )
    if not math.isclose(
        total_loss, aggregate_total, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError(
            "total_loss_W does not match the production aggregate-loss "
            "surrogates"
        )

    primary_turns = _turn_count(row, "N1_main")
    primary_side_turns = _turn_count(row, "N1_side")
    if primary_side_turns:
        raise ValueError(
            "P_Tx_main_group cannot represent total primary loss when "
            "N1_side is nonzero"
        )
    core_group_count = _turn_count(row, "n_core_group")
    if core_group_count <= 0:
        raise ValueError("n_core_group is not positive")
    power = rated_power_w(row)
    efficiency = power / (power + total_loss) * 100.0

    return {
        "size_W_mm": width_mm,
        "size_L_mm": length_mm,
        "size_H_mm": height_mm,
        "size_WxLxH_mm": (
            f"{width_mm:.1f} × {length_mm:.1f} × {height_mm:.1f}"
        ),
        "volume_L": float(volume_l),
        # The reference table's cm² value is W x L footprint, not six-face
        # surface area (748 x 812 mm -> 6074 cm²).
        "footprint_cm2": width_mm * length_mm / 100.0,
        "turns_primary": primary_turns + primary_side_turns,
        "turns_secondary_center": _turn_count(row, "N2_main"),
        "turns_secondary_side": _turn_count(row, "N2_side"),
        "cw1_conductor_thickness_mm": _nonnegative(row["cw1"], "cw1"),
        "cw2_conductor_thickness_mm": _nonnegative(row["cw2"], "cw2"),
        "gap1_mm": _nonnegative(row["gap1"], "gap1"),
        "gap2_mm": _nonnegative(row["gap2"], "gap2"),
        "nwl1_main_pack_width_mm": _nonnegative(
            row["nwl1_main"], "nwl1_main"
        ),
        "nwl1_side_pack_width_mm": _nonnegative(
            row["nwl1_side"], "nwl1_side"
        ),
        "nwl2_main_pack_width_mm": _nonnegative(
            row["nwl2_main"], "nwl2_main"
        ),
        "nwl2_side_pack_width_mm": _nonnegative(
            row["nwl2_side"], "nwl2_side"
        ),
        "nwh1_winding_height_mm": _nonnegative(row["nwh1"], "nwh1"),
        "nwh2_winding_height_mm": _nonnegative(row["nwh2"], "nwh2"),
        "core_depth_each_mm": _nonnegative(
            row["core_depth_each"], "core_depth_each"
        ),
        "n_core_group": core_group_count,
        "core_cold_plate_thickness_mm": _nonnegative(
            row["core_plate_t"], "core_plate_t"
        ),
        "core_thermal_pad_thickness_mm": _nonnegative(
            row["core_plate_pad_t"], "core_plate_pad_t"
        ),
        "winding_cold_plate_thickness_mm": _nonnegative(
            row["wcp_t"], "wcp_t"
        ),
        "winding_thermal_pad_thickness_mm": _nonnegative(
            row["wcp_pad_t"], "wcp_pad_t"
        ),
        "wcp_len_pct": _nonnegative(row["wcp_len_pct"], "wcp_len_pct"),
        "wcp_len_x_mm": _nonnegative(row["wcp_len_x"], "wcp_len_x"),
        "leakage_target_uH": leakage_target,
        "pred_leakage_inductance_uH": leakage,
        "B_design_analytic_T": design_b_field,
        "B_legacy_0p7_T": legacy_b_field,
        "B_design_waveform": "bipolar_square",
        "B_denominator_coefficient": 4.0,
        "Ae_m2": gross_iron_area_m2,
        "Ae_gross_m2": gross_iron_area_m2,
        "Ae_effective_m2": effective_iron_area_m2,
        "core_lamination_factor": _nonnegative(
            core_lamination_factor, "core_lamination_factor"
        ),
        "B_area_basis": B_area_basis,
        "pred_core_loss_W": core_loss,
        "pred_core_cold_plate_loss_W": core_plate_loss,
        "pred_winding_cold_plate_loss_W": winding_plate_loss,
        "pred_primary_winding_loss_W": primary_loss,
        "pred_secondary_center_winding_loss_W": secondary_center_loss,
        "pred_secondary_side_winding_loss_W": secondary_side_loss,
        "pred_secondary_winding_loss_W": (
            secondary_center_loss + secondary_side_loss
        ),
        "pred_component_winding_loss_sum_W": (
            primary_loss + secondary_center_loss + secondary_side_loss
        ),
        "pred_total_winding_loss_W": winding_total,
        "pred_total_loss_W": total_loss,
        "rated_power_W": power,
        "pred_efficiency_pct": efficiency,
        "surrogate_output_basis": "full_physical",
    }
