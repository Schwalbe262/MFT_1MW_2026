"""Traceable 1K101/2605SA1 core-material physics contract.

The Maxwell solid is the *gross* amorphous ribbon pack.  For that geometry the
UU137 guaranteed lamination factor must be represented by Maxwell's native
``Lamination`` material model; it is not emulated by changing the POWERLITE
core-loss coefficient.  A wound rectangular core cannot use one global
stacking direction: the ribbon normal is X in the legs and Z in the yokes.

This module contains deterministic validation and unit-safe reference
formulae.  It intentionally does not read the source PDFs at run time.
"""

from __future__ import annotations

import math
import re


CORE_MATERIAL_CONTRACT_VERSION = "1k101-uu137-aedt-lamination-kf0p85-v3"
PHYSICS_DATA_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
# Governance: extend this tuple only for code-reviewed solver revisions proven
# to produce physically identical rows for PHYSICS_DATA_REVISION. Any solver
# change that touches physics MUST bump PHYSICS_DATA_REVISION instead of
# extending this tuple.
PHYSICS_EQUIVALENT_SOLVER_REVISIONS = (
    "dba903eb671e37642168afc5578b8e6a93e9c046",
    "bffbb15fe2cdec74a72f47e7eb9bacbf0f4e95f7",
)
CORE_MATERIAL_IDENTITY = "1K101_Fe_based_amorphous_2605SA1_equivalent"
CORE_LAMINATION_FACTOR_SOURCE = (
    "UU137_approval_sheet_2023-02-04_p6_guaranteed_minimum"
)
CORE_LAMINATION_FACTOR_DATASHEET_MIN = 0.85
CORE_LAMINATION_FACTOR_USER_CONSERVATIVE_CANDIDATE = 0.70
CORE_LAMINATION_FACTOR_AB_CANDIDATES = (1.0, 0.85, 0.70)
CORE_LOSS_MODEL_SOURCE = "POWERLITE_C_opt_p2_6p5_f_kHz_1p51_B_1p74"
CORE_SPECIFIC_LOSS_COEFFICIENT_W_KG = 6.5
CORE_LOSS_MARGIN_SOURCE = "production_conservative_margin_1p15"
CORE_MASS_DENSITY_KG_M3 = 7180.0
CORE_MASS_DENSITY_SOURCE = "POWERLITE_C_opt_p1_density_7p18_g_cm3"

NATIVE_STACKING_TYPE = "Lamination"
LEG_STACKING_DIRECTION = "V(1)"
YOKE_STACKING_DIRECTION = "V(3)"
NATIVE_STACKING_MODEL = "aedt_native_lamination_segmented_wound_core"
NATIVE_STACKING_DIRECTION_BASIS = (
    "global_cartesian_leg_normal_x_yoke_normal_z_flux_path_in_xz"
)
MAXWELL_B_OUTPUT_BASIS = "homogenized_gross_pack_average_air_plus_ribbon"
MAXWELL_VOLTAGE_SOURCE_BASIS = "peak_phasor_V1_rms_times_sqrt2"

AREA_BASIS_GROSS_HOMOGENEOUS = "gross_homogeneous_em_geometry"
AREA_BASIS_EXPLICIT_NET = "explicit_net_iron_geometry"
SUPPORTED_AREA_BASES = {
    AREA_BASIS_GROSS_HOMOGENEOUS,
    AREA_BASIS_EXPLICIT_NET,
}


def solver_revision_matches_physics_cohort(
    actual_revision: object,
    expected_revision: object,
    physics_data_revision: object,
) -> bool:
    """Match legacy exact pins or the reviewed current-physics solver cohort."""
    actual = str(
        "" if actual_revision is None else actual_revision
    ).strip().lower()
    expected = str(
        "" if expected_revision is None else expected_revision
    ).strip().lower()
    equivalent = PHYSICS_EQUIVALENT_SOLVER_REVISIONS
    if actual in equivalent or expected in equivalent:
        return (
            str(
                "" if physics_data_revision is None else physics_data_revision
            ).strip() == PHYSICS_DATA_REVISION
            and actual in equivalent
            and expected in equivalent
        )
    return actual == expected


def solver_revision_cohort_identity(solver_revision: object) -> str | None:
    """Return one checkpoint identity for all reviewed current-physics SHAs."""
    if solver_revision is None:
        return None
    revision = str(solver_revision).strip().lower()
    if revision in PHYSICS_EQUIVALENT_SOLVER_REVISIONS:
        return f"physics_data_revision:{PHYSICS_DATA_REVISION}"
    return f"solver_revision:{revision}"


def lamination_factor_policy_source(lamination_factor) -> str:
    """Label the meaning of an A/B factor without calling it measured data."""
    kf = _finite(lamination_factor, "core_lamination_factor")
    if math.isclose(kf, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "isolated_ab_reference_no_stacking_reduction"
    if math.isclose(
        kf, CORE_LAMINATION_FACTOR_DATASHEET_MIN,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        return CORE_LAMINATION_FACTOR_SOURCE
    if math.isclose(
        kf, CORE_LAMINATION_FACTOR_USER_CONSERVATIVE_CANDIDATE,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        return (
            "user_selected_conservative_native_lamination_candidate_0p70_"
            "not_a_datasheet_stacking_factor"
        )
    return "explicit_parameter_candidate_no_datasheet_attestation"


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _aedt_number(value, label: str) -> float:
    """Read the numeric prefix of an AEDT value such as ``1.377A_per_meter``."""
    if isinstance(value, (int, float)):
        return _finite(value, label)
    match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        str(value),
    )
    if not match:
        raise ValueError(f"{label} has no numeric AEDT value: {value!r}")
    return _finite(match.group(1), label)


def validate_core_material_contract(
    *, cm_base, core_x, core_y, lamination_factor, loss_margin, area_basis
) -> tuple[float, float, float, float, float, str]:
    """Validate and normalize numerical material-contract inputs."""
    cm = _finite(cm_base, "core_cm")
    x = _finite(core_x, "core_x")
    y = _finite(core_y, "core_y")
    kf = _finite(lamination_factor, "core_lamination_factor")
    margin = _finite(loss_margin, "core_loss_margin")
    basis = str(area_basis)
    if cm <= 0:
        raise ValueError("core_cm must be > 0")
    if x <= 0:
        raise ValueError("core_x must be > 0")
    if y <= 0:
        raise ValueError("core_y must be > 0")
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    if margin < 1:
        raise ValueError("core_loss_margin must be >= 1")
    if basis not in SUPPORTED_AREA_BASES:
        raise ValueError(f"unsupported core area basis: {basis!r}")
    return cm, x, y, kf, margin, basis


def native_lamination_material_specs(lamination_factor) -> dict[str, dict]:
    """Return the two native material orientations required by the XZ frame."""
    kf = _finite(lamination_factor, "core_lamination_factor")
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    return {
        "leg": {
            "stacking_type": NATIVE_STACKING_TYPE,
            "stacking_factor": kf,
            "stacking_direction": LEG_STACKING_DIRECTION,
        },
        "yoke": {
            "stacking_type": NATIVE_STACKING_TYPE,
            "stacking_factor": kf,
            "stacking_direction": YOKE_STACKING_DIRECTION,
        },
    }


def _choice(props: dict, key: str) -> str:
    value = props.get(key)
    if isinstance(value, dict):
        value = value.get("Choice")
    return str(value or "")


def validate_native_lamination_readback(
    props: dict,
    *,
    lamination_factor,
    stacking_direction,
    cm_base,
    core_x,
    core_y,
    permeability=3000.0,
) -> dict:
    """Fail closed unless raw AEDT material data matches the contract exactly.

    ``props`` must come from ``oMaterialManager.GetData`` rather than from the
    values just assigned through PyAEDT.  The latter would only attest Python
    state, not the native AEDT definition used by the solver.
    """
    if not isinstance(props, dict) or not props:
        raise RuntimeError("native AEDT material readback is empty")
    if str(props.get("CoordinateSystemType", "")) != "Cartesian":
        raise RuntimeError(
            "native material CoordinateSystemType must be Cartesian, got "
            f"{props.get('CoordinateSystemType')!r}"
        )
    for choice_key in ("stacking_type", "stacking_direction", "core_loss_type"):
        choice_props = props.get(choice_key)
        if not isinstance(choice_props, dict) or (
            choice_props.get("property_type") != "ChoiceProperty"
        ):
            raise RuntimeError(
                f"native {choice_key} is not a ChoiceProperty: {choice_props!r}"
            )
    expected_direction = str(stacking_direction)
    if expected_direction not in {LEG_STACKING_DIRECTION, YOKE_STACKING_DIRECTION}:
        raise ValueError(f"unsupported stacking direction: {expected_direction!r}")

    actual_type = _choice(props, "stacking_type")
    actual_direction = _choice(props, "stacking_direction")
    actual_factor = _aedt_number(props.get("stacking_factor"), "stacking_factor")
    if actual_type != NATIVE_STACKING_TYPE:
        raise RuntimeError(
            f"native stacking_type mismatch: {actual_type!r} != {NATIVE_STACKING_TYPE!r}"
        )
    if actual_direction != expected_direction:
        raise RuntimeError(
            "native stacking_direction mismatch: "
            f"{actual_direction!r} != {expected_direction!r}"
        )
    if not math.isclose(
        actual_factor,
        _finite(lamination_factor, "core_lamination_factor"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"native stacking_factor mismatch: {actual_factor!r} != {lamination_factor!r}"
        )

    actual_loss_type = _choice(props, "core_loss_type")
    if actual_loss_type != "Power Ferrite":
        raise RuntimeError(
            f"native core_loss_type mismatch: {actual_loss_type!r} != 'Power Ferrite'"
        )
    expected_numbers = {
        "core_loss_cm": cm_base,
        "core_loss_x": core_x,
        "core_loss_y": core_y,
        "permeability": permeability,
        "conductivity": 0.0,
        "core_loss_kdc": 0.0,
        "core_loss_equiv_cut_depth": 0.0,
    }
    actual_numbers = {}
    for key, expected in expected_numbers.items():
        actual = _aedt_number(props.get(key), key)
        expected = _finite(expected, key)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                f"native {key} mismatch: {actual!r} != {expected!r}"
            )
        actual_numbers[key] = actual

    return {
        "stacking_type": actual_type,
        "stacking_factor": actual_factor,
        "stacking_direction": actual_direction,
        "core_loss_type": actual_loss_type,
        "CoordinateSystemType": "Cartesian",
        **actual_numbers,
    }


def effective_steinmetz_cm(
    cm_base,
    core_y,
    lamination_factor,
    loss_margin,
    *,
    area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
) -> float:
    """Return a *linear-law diagnostic* gross-volume coefficient.

    For a prescribed B field and power law, the identity is
    ``Cm_equiv = Cm * margin * kf**(1-y)``.  Production does not assign this
    coefficient to Maxwell.  It is not equivalent to native stacking when a
    nonlinear BH curve changes the solved field distribution.
    """
    cm, _, y, kf, margin, basis = validate_core_material_contract(
        cm_base=cm_base,
        core_x=1.0,
        core_y=core_y,
        lamination_factor=lamination_factor,
        loss_margin=loss_margin,
        area_basis=area_basis,
    )
    if basis == AREA_BASIS_EXPLICIT_NET:
        return cm * margin
    return cm * margin * kf ** (1.0 - y)


def material_flux_density_t(
    b_average_t,
    lamination_factor,
    *,
    area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
) -> float:
    """Convert Maxwell homogenized pack-average B to ribbon-material B."""
    b_average = _finite(b_average_t, "B_average")
    kf = _finite(lamination_factor, "core_lamination_factor")
    if b_average < 0:
        raise ValueError("B_average must be >= 0")
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    if area_basis not in SUPPORTED_AREA_BASES:
        raise ValueError(f"unsupported core area basis: {area_basis!r}")
    return b_average if area_basis == AREA_BASIS_EXPLICIT_NET else b_average / kf


def effective_area_m2(
    geometry_area_m2,
    lamination_factor,
    *,
    area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
) -> float:
    """Return effective magnetic area without double-applying net area."""
    area = _finite(geometry_area_m2, "core geometry area")
    kf = _finite(lamination_factor, "core_lamination_factor")
    if area <= 0:
        raise ValueError("core geometry area must be > 0")
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    if area_basis not in SUPPORTED_AREA_BASES:
        raise ValueError(f"unsupported core area basis: {area_basis!r}")
    return area if area_basis == AREA_BASIS_EXPLICIT_NET else area * kf


def geometry_volume_and_masses(
    geometry_volume_m3,
    lamination_factor,
    *,
    density_kg_m3=CORE_MASS_DENSITY_KG_M3,
    area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
) -> tuple[float, float, float, float]:
    """Return gross/effective volumes and corresponding bare-alloy masses."""
    volume = _finite(geometry_volume_m3, "core geometry volume")
    density = _finite(density_kg_m3, "core mass density")
    kf = _finite(lamination_factor, "core_lamination_factor")
    if volume <= 0:
        raise ValueError("core geometry volume must be > 0")
    if density <= 0:
        raise ValueError("core mass density must be > 0")
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    if area_basis not in SUPPORTED_AREA_BASES:
        raise ValueError(f"unsupported core area basis: {area_basis!r}")
    effective_volume = (
        volume if area_basis == AREA_BASIS_EXPLICIT_NET else volume * kf
    )
    return (
        volume,
        effective_volume,
        volume * density,
        effective_volume * density,
    )


def square_wave_b_material_t(voltage_v, frequency_hz, turns, ae_effective_m2) -> float:
    """DAB bipolar square-wave peak B: ``V/(4 f N Ae_effective)``."""
    voltage = _finite(voltage_v, "voltage")
    frequency = _finite(frequency_hz, "frequency")
    turns = _finite(turns, "turns")
    area = _finite(ae_effective_m2, "Ae_effective")
    if min(voltage, frequency, turns, area) <= 0:
        raise ValueError("voltage, frequency, turns, and Ae_effective must be > 0")
    return voltage / (4.0 * frequency * turns * area)


def sinusoidal_b_peak_material_t(
    voltage_rms_v, frequency_hz, turns, ae_effective_m2
) -> float:
    """AC Magnetic sinusoidal peak B for the code's ``sqrt(2)*Vrms`` source."""
    voltage_rms = _finite(voltage_rms_v, "voltage_rms")
    frequency = _finite(frequency_hz, "frequency")
    turns = _finite(turns, "turns")
    area = _finite(ae_effective_m2, "Ae_effective")
    if min(voltage_rms, frequency, turns, area) <= 0:
        raise ValueError(
            "voltage_rms, frequency, turns, and Ae_effective must be > 0"
        )
    return math.sqrt(2.0) * voltage_rms / (
        2.0 * math.pi * frequency * turns * area
    )


def expected_specific_core_loss_w_kg(
    frequency_hz, b_material_t, *, coefficient=6.5, x=1.51, y=1.74
) -> float:
    """POWERLITE loss law in its published kHz/W/kg units."""
    frequency = _finite(frequency_hz, "frequency")
    b_material = _finite(b_material_t, "B_material")
    coefficient = _finite(coefficient, "specific loss coefficient")
    x = _finite(x, "frequency exponent")
    y = _finite(y, "flux exponent")
    if frequency <= 0 or b_material < 0 or coefficient <= 0 or x <= 0 or y <= 0:
        raise ValueError("invalid POWERLITE specific-loss inputs")
    return coefficient * (frequency / 1000.0) ** x * b_material**y


def faraday_lumped_core_reference(
    *,
    voltage_rms_v,
    frequency_hz,
    turns,
    gross_area_m2,
    lamination_factor,
    effective_mass_kg,
    loss_margin,
    coefficient=CORE_SPECIFIC_LOSS_COEFFICIENT_W_KG,
    x=1.51,
    y=1.74,
) -> dict:
    """Independent non-calculator reference for sinusoidal Maxwell results.

    The reference deliberately avoids a field-calculator ``B**y`` moment.
    Faraday's law first gives the homogenized gross-pack flux density.  The
    ribbon-material density is ``B_pack / kf`` and POWERLITE's published
    W/kg law is applied to the effective alloy mass.  Because this is a
    lumped estimate, production records compare it to both the standard AEDT
    B-average extraction and native CoreLoss with explicit engineering
    tolerances rather than claiming pointwise identity.
    """
    voltage = _finite(voltage_rms_v, "voltage_rms")
    frequency = _finite(frequency_hz, "frequency")
    turns = _finite(turns, "turns")
    gross_area = _finite(gross_area_m2, "gross_area")
    kf = _finite(lamination_factor, "core_lamination_factor")
    mass = _finite(effective_mass_kg, "effective_mass")
    margin = _finite(loss_margin, "core_loss_margin")
    if min(voltage, frequency, turns, gross_area, mass) <= 0:
        raise ValueError(
            "voltage, frequency, turns, gross area, and effective mass must be > 0"
        )
    if not 0 < kf <= 1:
        raise ValueError("core_lamination_factor must satisfy 0 < kf <= 1")
    if margin < 1:
        raise ValueError("core_loss_margin must be >= 1")
    b_pack = math.sqrt(2.0) * voltage / (
        2.0 * math.pi * frequency * turns * gross_area
    )
    b_material = b_pack / kf
    specific_loss = expected_specific_core_loss_w_kg(
        frequency,
        b_material,
        coefficient=coefficient,
        x=x,
        y=y,
    )
    native_raw = specific_loss * mass
    return {
        "B_pack_T": b_pack,
        "B_material_T": b_material,
        "specific_loss_W_kg": specific_loss,
        "effective_mass_kg": mass,
        "native_raw_loss_W": native_raw,
        "margin_adjusted_loss_W": native_raw * margin,
        "loss_margin": margin,
        "coefficient_W_kg": float(coefficient),
        "frequency_exponent": float(x),
        "flux_exponent": float(y),
        "basis": (
            "sinusoidal_faraday_Bpack_then_Bmaterial_div_kf_then_"
            "POWERLITE_Wkg_times_effective_mass"
        ),
    }


def expected_core_loss_from_bavg_moment_w(
    bavg_power_volume_integral,
    *,
    cm_base,
    frequency_hz,
    core_x,
    core_y,
    lamination_factor,
    loss_margin,
) -> float:
    """Independent native-stacking loss check from ``integral(Bavg**y dVgross)``."""
    moment = _finite(bavg_power_volume_integral, "Bavg power-volume integral")
    cm, x, y, kf, margin, _ = validate_core_material_contract(
        cm_base=cm_base,
        core_x=core_x,
        core_y=core_y,
        lamination_factor=lamination_factor,
        loss_margin=loss_margin,
        area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
    )
    frequency = _finite(frequency_hz, "frequency")
    if moment < 0 or frequency <= 0:
        raise ValueError("Bavg moment must be >= 0 and frequency must be > 0")
    return cm * frequency**x * margin * kf ** (1.0 - y) * moment


def build_core_material_contract_fields(
    *,
    cm_base,
    core_x,
    core_y,
    lamination_factor,
    loss_margin,
    area_basis=AREA_BASIS_GROSS_HOMOGENEOUS,
) -> dict:
    """Return result-row provenance and deterministic reference factors."""
    cm, x, y, kf, margin, basis = validate_core_material_contract(
        cm_base=cm_base,
        core_x=core_x,
        core_y=core_y,
        lamination_factor=lamination_factor,
        loss_margin=loss_margin,
        area_basis=area_basis,
    )
    cm_equivalent_gross = effective_steinmetz_cm(
        cm, y, kf, margin, area_basis=basis
    )
    gross_basis = basis == AREA_BASIS_GROSS_HOMOGENEOUS
    return {
        "physics_data_revision": PHYSICS_DATA_REVISION,
        "core_material_contract_version": CORE_MATERIAL_CONTRACT_VERSION,
        "core_material_identity": CORE_MATERIAL_IDENTITY,
        "core_geometry_material_basis": basis,
        "core_stacking_model": (
            NATIVE_STACKING_MODEL if gross_basis else "explicit_net_iron_geometry"
        ),
        "core_stacking_type": NATIVE_STACKING_TYPE if gross_basis else "Solid",
        "core_stacking_direction_leg": LEG_STACKING_DIRECTION if gross_basis else "",
        "core_stacking_direction_yoke": YOKE_STACKING_DIRECTION if gross_basis else "",
        "core_stacking_direction_basis": NATIVE_STACKING_DIRECTION_BASIS,
        "core_corner_orientation_model": (
            "abrupt_axis_aligned_leg_yoke_transition_unvalidated_no_curvilinear_corner"
        ),
        "core_native_model_approval_status": (
            # kf=0.85 approved by the user on 2026-07-13 from isolated solved
            # A/B evidence (tasks 30615/30616/30617 fresh at a83acf2, 30546
            # saved cross-check): B-ratio exact for all kf, B_avg vs Faraday
            # 1.97%, native loss vs POWERLITE 6.9% at kf=0.85. The 9.24%
            # induced-vs-source peak is kf-invariant (identical at kf=1.0)
            # and scoped out as test-design circuit drop.
            "approved_by_isolated_solved_kf_ab"
        ),
        "core_lamination_factor": kf,
        "core_lamination_factor_source": lamination_factor_policy_source(kf),
        "core_lamination_factor_datasheet_guaranteed_min": (
            CORE_LAMINATION_FACTOR_DATASHEET_MIN
        ),
        "core_lamination_factor_user_conservative_candidate": (
            CORE_LAMINATION_FACTOR_USER_CONSERVATIVE_CANDIDATE
        ),
        "core_lamination_factor_ab_candidates": "1.00,0.85,0.70",
        "core_loss_model_source": CORE_LOSS_MODEL_SOURCE,
        "core_loss_margin": margin,
        "core_loss_margin_source": CORE_LOSS_MARGIN_SOURCE,
        "core_cm_base": cm,
        "core_cm_assigned": cm,
        "core_cm_equivalent_gross_unassigned": cm_equivalent_gross,
        "core_x_contract": x,
        "core_y_contract": y,
        # Native stacking must account for kf.  Only the independent 1.15
        # engineering margin is applied after the solver loss is attested.
        "core_loss_correction_factor": margin,
        "core_loss_postprocess_factor": margin,
        "core_loss_application_basis": (
            "aedt_native_lamination_base_cm_then_margin_after_native_loss_attestation"
            if gross_basis
            else "explicit_net_geometry_base_cm_then_margin"
        ),
        "core_loss_expected_integral_basis": (
            "lumped_faraday_Bpack_then_Bmaterial_div_kf_then_"
            "POWERLITE_Wkg_times_effective_mass"
        ),
        "core_specific_loss_coefficient_W_kg": (
            CORE_SPECIFIC_LOSS_COEFFICIENT_W_KG
        ),
        "core_B_average_to_material_factor": 1.0 / kf if gross_basis else 1.0,
        # Compatibility name retained as an optional result column.
        "core_B_macro_to_material_factor": 1.0 / kf if gross_basis else 1.0,
        "core_area_gross_to_effective_factor": kf if gross_basis else 1.0,
        "core_volume_gross_to_effective_factor": kf if gross_basis else 1.0,
        "core_mass_density_kg_m3": CORE_MASS_DENSITY_KG_M3,
        "core_mass_density_source": CORE_MASS_DENSITY_SOURCE,
        "core_mass_definition": "bare_magnetic_alloy_excludes_impregnation",
        "core_permeability_model": (
            "constant_mu_r_3000_existing_assumption_not_datasheet_BH"
        ),
        "core_bulk_conductivity_model": (
            "zero_to_avoid_double_counting_empirical_power_ferrite_loss"
        ),
        "core_loss_equiv_cut_depth_model": (
            "zero_disabled_to_prevent_latent_cut_edge_loss_double_margin"
        ),
        "core_flux_density_output_basis": (
            "maxwell_B_average_and_derived_B_material_equals_Bavg_div_kf"
            if gross_basis
            else "material_ribbon_explicit_net_geometry"
        ),
        "maxwell_B_output_basis": MAXWELL_B_OUTPUT_BASIS,
        "maxwell_B_peak_operator": (
            "CmplxMag_then_Mag_complex_vector_norm_diagnostic_not_phase_peak"
        ),
        "maxwell_voltage_source_basis": MAXWELL_VOLTAGE_SOURCE_BASIS,
    }
