"""Conditional apparent charge from an assumed spherical gas void.

This reduced model uses a native *interior* host field at a 1 V electrode
difference as the host weighting field. It does not use global capacitance.
For an aligned small sphere in an infinite, locally uniform dielectric,
q = 3 eps_host Volume (E_inc - E_ext) E_weight_host. The corresponding change
in free surface charge is sigma0*cos(theta), with
sigma0=(eps_gas+2 eps_host)*DeltaE. Hemisphere charge fields are signed
increments for each event, not the accumulated wall charge; the latter is
represented by the retained wall-field memory.

Reference: Crichton, Karlsson and Pedersen (1989), equations 19 and 23:
https://backend.orbit.dtu.dk/ws/portalfiles/portal/4486981/Crichton.pdf
The optional air inception relation is an empirical streamer criterion, not
a Townsend/Paschen law. It presumes an available starting electron and has
no stochastic time lag. Neither equation establishes measured PD charge.

Input example::

    {"field_evidence": {
       "source": "native vector export + hash", "solution": "Setup1 : LastAdaptive",
       "object_name": "Pad", "material_name": "thermal_pad", "location_mm": [0,0,0],
       "point_in_dielectric": true, "model_basis": "full", "unit": "V/m",
       "normalization_voltage_V": 1, "local_field_at_1V_V_per_m": 500,
       "local_field_converged": false},
     "void": {"radii_m": [0.00005, 0.0001], "minimum_host_boundary_distance_m": 0.001,
       "relative_permittivity_host": 4, "relative_permittivity_gas": 1,
       "pressure_Pa": 101325, "extinction_ratio": 0.5,
       "containment_evidence": "native solid containment / nearest boundary"},
     "voltage_scenarios_rms_V": [5000,10000,15000,20000], "frequency_Hz": 60}

Use radius_m for one radius or radii_m for a grid. Explicit
inception_field_V_per_m may replace the assumed air relation. Frequency and
all voltage values remain assumed scenarios. One virgin sinusoidal cycle
starts with zero wall-charge memory; memory is retained with no relaxation.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

EPSILON_0_F_PER_M = 8.8541878128e-12
PD_REFERENCE_LIMIT_PC = 15.0
REFERENCE_URL = "https://backend.orbit.dtu.dk/ws/portalfiles/portal/4486981/Crichton.pdf"


class EventLimitExceeded(ValueError):
    """An event train exceeded its explicit bound; it was not truncated."""


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def sphere_apparent_charge_C(radius_m, epsr_host, field_drop_V_per_m, weighting_field_per_m):
    """Magnitude of the induced terminal charge for an aligned spherical void."""
    radius = _finite(radius_m, "radius_m", positive=True)
    epsr = _finite(epsr_host, "relative_permittivity_host", positive=True)
    drop = _finite(field_drop_V_per_m, "field_drop_V_per_m", nonnegative=True)
    weighting = _finite(weighting_field_per_m, "weighting_field_per_m", nonnegative=True)
    try:
        charge = 4 * math.pi * EPSILON_0_F_PER_M * epsr * radius ** 3 * drop * weighting
    except OverflowError as error:
        raise ValueError("apparent charge exceeds the finite numeric range") from error
    return _finite(charge, "apparent_charge_C", nonnegative=True)


def air_streamer_inception_field_V_per_m(pressure_Pa, radius_m):
    """Assumed air streamer onset: 24.2*p*(1+8.6/sqrt(p*diameter)).

    Pressure is Pa and diameter is m. The empirical 8.6 coefficient carries
    sqrt(Pa*m); 24.2 carries V/(Pa*m). Gas composition and calibration are
    external assumptions, not inferred from the native dielectric model.
    """
    pressure = _finite(pressure_Pa, "pressure_Pa", positive=True)
    radius = _finite(radius_m, "radius_m", positive=True)
    result = 24.2 * pressure * (1 + 8.6 / math.sqrt(pressure * 2 * radius))
    return _finite(result, "inception_field_V_per_m", positive=True)


def _next_crossing(phase, amplitude, memory, inception):
    candidates = []
    two_pi = 2 * math.pi
    for polarity in (1, -1):
        ratio = (polarity * inception - memory) / amplitude
        if abs(ratio) > 1 + 1e-12:
            continue
        angle = math.asin(max(-1.0, min(1.0, ratio)))
        for root in (angle % two_pi, (math.pi - angle) % two_pi):
            # Cross the inception surface outward; tangency at a peak also
            # counts as deterministic onset in the zero-delay hypothesis.
            if polarity * math.cos(root) < -1e-12:
                continue
            for candidate in (root, root + two_pi):
                if phase + 1e-12 < candidate < two_pi - 1e-12:
                    candidates.append((candidate, polarity))
    return min(candidates, default=None)


def first_cycle_events(*, void_peak_field_V_per_m, inception_field_V_per_m,
                       extinction_field_V_per_m, frequency_Hz, max_events=10000):
    """Exact threshold crossings of B*sin(phase)+wall-charge memory.

    No time step is used, so an event drops E_inc to E_ext rather than a
    sampled overshoot. Returns one cycle, starting at phase zero with virgin
    memory. A limit violation raises instead of returning a truncated train.
    """
    amplitude = _finite(void_peak_field_V_per_m, "void_peak_field_V_per_m", nonnegative=True)
    inception = _finite(inception_field_V_per_m, "inception_field_V_per_m", positive=True)
    extinction = _finite(extinction_field_V_per_m, "extinction_field_V_per_m", nonnegative=True)
    frequency = _finite(frequency_Hz, "frequency_Hz", positive=True)
    if extinction >= inception:
        raise ValueError("extinction field must be below inception field")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or not 1 <= max_events <= 1000000:
        raise ValueError("max_events must be an integer from 1 to 1000000")
    events, memory, phase = [], 0.0, 0.0
    if amplitude == 0:
        return events, memory
    # The virgin rising quarter-cycle alone has thresholds Einc+n*DeltaE.
    # This lower bound also catches tiny drops whose roots would be closer
    # than floating-point phase resolution; do not silently skip that train.
    if (amplitude - inception) / (inception - extinction) >= max_events:
        raise EventLimitExceeded(f"virgin rising-quarter event count exceeds max_events={max_events}; no truncated estimate returned")
    while True:
        crossing = _next_crossing(phase, amplitude, memory, inception)
        if crossing is None:
            return events, memory
        if len(events) >= max_events:
            raise EventLimitExceeded(f"first-cycle event count exceeds max_events={max_events}; no truncated estimate returned")
        phase, polarity = crossing
        applied = amplitude * math.sin(phase)
        before = applied + memory
        if not math.isclose(before, polarity * inception, rel_tol=1e-9, abs_tol=inception * 1e-9):
            raise RuntimeError("threshold crossing failed its field conservation check")
        after_memory = polarity * extinction - applied
        events.append({
            "phase_rad": phase, "phase_deg": math.degrees(phase),
            "time_s": phase / (2 * math.pi * frequency), "polarity": polarity,
            "applied_void_field_V_per_m": applied,
            "total_void_field_before_V_per_m": before,
            "total_void_field_after_V_per_m": polarity * extinction,
            "wall_memory_before_V_per_m": memory,
            "wall_memory_after_V_per_m": after_memory,
            "field_drop_magnitude_V_per_m": inception - extinction,
        })
        memory = after_memory


def _validated_input(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("PD void input must be a JSON object")
    payload = json.loads(json.dumps(dict(payload), allow_nan=False))
    field, void = payload.get("field_evidence"), payload.get("void")
    if not isinstance(field, dict) or not isinstance(void, dict):
        raise ValueError("field_evidence and void objects are required")
    for key in ("source", "solution", "object_name", "material_name"):
        if not isinstance(field.get(key), str) or not field[key].strip():
            raise ValueError(f"field_evidence.{key} is required")
    normalization = _finite(field.get("normalization_voltage_V"), "normalization_voltage_V", positive=True)
    if field.get("unit") != "V/m" or normalization != 1:
        raise ValueError("native interior field must be normalized to exactly 1 V and expressed in V/m")
    geometry_multiplier = _finite(field.get("local_field_geometry_multiplier", 1), "local_field_geometry_multiplier", positive=True)
    if field.get("model_basis") != "full" or geometry_multiplier != 1:
        raise ValueError("use the full-model local field without a geometry restoration multiplier")
    if field.get("point_in_dielectric") is not True:
        raise ValueError("native evaluation point must be inside the dielectric, not an interface maximum")
    location = field.get("location_mm")
    if not isinstance(location, list) or len(location) != 3:
        raise ValueError("native dielectric interior location_mm must have three coordinates")
    for coordinate in location:
        _finite(coordinate, "location_mm")
    weighting = _finite(field.get("local_field_at_1V_V_per_m"), "local_field_at_1V_V_per_m", nonnegative=True)
    if "charge_relaxation_time_s" in payload:
        raise ValueError("this model has no charge relaxation; a decay parameter cannot be silently ignored")
    if ("radius_m" in void) == ("radii_m" in void):
        raise ValueError("provide exactly one of void.radius_m or void.radii_m")
    radii = void.get("radii_m", [void.get("radius_m")])
    if not isinstance(radii, list) or not radii or len(radii) > 1000:
        raise ValueError("void.radii_m must be a nonempty bounded list")
    distance = _finite(void.get("minimum_host_boundary_distance_m"), "minimum_host_boundary_distance_m", positive=True)
    if not isinstance(void.get("containment_evidence"), str) or not void["containment_evidence"].strip():
        raise ValueError("native void containment_evidence is required")
    radii = [_finite(radius, "radius_m", positive=True) for radius in radii]
    if any(radius >= distance for radius in radii):
        raise ValueError("assumed spherical void must fit strictly inside the native dielectric solid")
    epsh = _finite(void.get("relative_permittivity_host"), "relative_permittivity_host", positive=True)
    epsg = _finite(void.get("relative_permittivity_gas"), "relative_permittivity_gas", positive=True)
    ratio = _finite(void.get("extinction_ratio"), "extinction_ratio", nonnegative=True)
    if ratio >= 1:
        raise ValueError("extinction_ratio must be below 1")
    if "inception_field_V_per_m" in void:
        _finite(void["inception_field_V_per_m"], "inception_field_V_per_m", positive=True)
    else:
        _finite(void.get("pressure_Pa"), "pressure_Pa", positive=True)
        if void.get("gas_is_air") is False or void.get("gas_species", "assumed_air") not in {"air", "assumed_air"}:
            raise ValueError("the air streamer criterion requires assumed/declared air; use an explicit inception field for other gases")
    if "pressure_Pa" in void:
        _finite(void["pressure_Pa"], "pressure_Pa", positive=True)
    voltages = payload.get("voltage_scenarios_rms_V")
    if not isinstance(voltages, list) or not voltages or len(voltages) > 1000:
        raise ValueError("voltage_scenarios_rms_V must be a nonempty bounded list")
    voltages = [_finite(voltage, "voltage_scenario_rms_V", positive=True) for voltage in voltages]
    frequency = _finite(payload.get("frequency_Hz"), "frequency_Hz", positive=True)
    return payload, field, void, radii, distance, epsh, epsg, ratio, voltages, frequency, weighting


def estimate_pd_void(payload):
    """Return conditional first-cycle PD charge scenarios, never qualification.

    Every input is retained. Local native field convergence, spherical-void
    validity, actual PD voltage, gas state, extinction, electron availability
    and finite-host corrections are not inferred from a completed FE solve.
    """
    (payload, field, void, radii, distance, epsh, epsg, ratio,
     voltages, frequency, weighting) = _validated_input(payload)
    k = 3 * epsh / (epsg + 2 * epsh)
    _finite(k, "spherical_embedding_factor", positive=True)
    max_events = payload.get("max_events", 10000)
    cases = []
    for radius in radii:
        inception = (void["inception_field_V_per_m"] if "inception_field_V_per_m" in void
                     else air_streamer_inception_field_V_per_m(void["pressure_Pa"], radius))
        extinction = ratio * inception
        drop = inception - extinction
        charge = sphere_apparent_charge_C(radius, epsh, drop, weighting)
        cavity_voltage_drop = _finite(2 * radius * drop, "cavity_voltage_drop_V", positive=True)
        coupling = _finite(2 * math.pi * EPSILON_0_F_PER_M * epsh * radius ** 2 * weighting,
                           "equivalent_terminal_coupling_capacitance_F", nonnegative=True)
        pdiv = inception / (k * weighting * math.sqrt(2)) if weighting > 0 else None
        if pdiv is not None:
            _finite(pdiv, "conditional_virgin_PDIV_rms_V", positive=True)
        sigma = EPSILON_0_F_PER_M * (epsg + 2 * epsh) * drop
        hemispheric_charge = math.pi * radius ** 2 * sigma
        scenarios = []
        for voltage in voltages:
            peak = k * weighting * voltage * math.sqrt(2)
            events, final_memory = first_cycle_events(
                void_peak_field_V_per_m=peak, inception_field_V_per_m=inception,
                extinction_field_V_per_m=extinction, frequency_Hz=frequency,
                max_events=max_events,
            )
            for event in events:
                event["q_app_C"] = event["polarity"] * charge
                event["q_app_pC"] = event["polarity"] * charge * 1e12
                event["hemisphere_free_charge_increment_north_C"] = event["polarity"] * hemispheric_charge
                event["hemisphere_free_charge_increment_south_C"] = -event["polarity"] * hemispheric_charge
                event["net_void_wall_free_charge_C"] = 0.0
            q_peak = charge * 1e12 if events else 0.0
            _finite(q_peak, "q_peak_pC", nonnegative=True)
            scenarios.append({
                "voltage_rms_V": voltage, "voltage_peak_V": voltage * math.sqrt(2),
                "voltage_is_assumed_scenario": True, "frequency_Hz": frequency,
                "frequency_is_assumed": True, "void_peak_field_without_wall_memory_V_per_m": peak,
                "q_peak_pC": q_peak, "event_count": len(events),
                "positive_event_count": sum(event["polarity"] > 0 for event in events),
                "negative_event_count": sum(event["polarity"] < 0 for event in events),
                "conditional_exceeds_15pC": q_peak > PD_REFERENCE_LIMIT_PC,
                "event_status": "conditional_events" if events else "no_events_under_assumptions",
                "physical_qualification": "insufficient_data", "does_not_establish_pd_free": True,
                "final_wall_memory_V_per_m": final_memory, "events": events,
            })
        cases.append({
            "radius_m": radius, "diameter_m": 2 * radius,
            "radius_to_host_boundary_distance_ratio": radius / distance,
            "inception_field_V_per_m": inception, "extinction_field_V_per_m": extinction,
            "threshold_field_drop_V_per_m": drop,
            "cavity_voltage_drop_V": cavity_voltage_drop,
            "equivalent_terminal_coupling_capacitance_F": coupling,
            "equivalent_terminal_coupling_capacitance_pF": coupling * 1e12,
            "threshold_pulse_q_app_pC_if_event": charge * 1e12,
            "conditional_virgin_PDIV_rms_V": pdiv, "scenarios": scenarios,
        })
    result = {
        "schema_version": "mft-conditional-spherical-void-pd-v1",
        "status": "conditional_estimate", "physical_qualification": "insufficient_data",
        "input": payload, "radius_cases": cases,
        "model": {
            "apparent_charge_formula": "4*pi*eps0*epsr_host*r_m^3*(Einc-Eext)*Ew_host_per_m",
            "spherical_embedding_factor": k, "host_weighting_field_per_m": weighting,
            "host_absolute_permittivity_F_per_m": EPSILON_0_F_PER_M * epsh,
            "local_field_geometry_multiplier": 1.0,
            "charge_polarity_convention": "same_sign_as_void_field_at_inception_aligned_with_host_weighting_field",
            "hemisphere_free_charge_semantics": "signed per-event increments, not accumulated wall charge; north is aligned with positive host weighting field",
            "inception_model": "explicit_assumed_field" if "inception_field_V_per_m" in void else "empirical_air_streamer_criterion",
            "inception_gas_assumption": void.get("gas_species", "assumed_air") if "inception_field_V_per_m" not in void else "explicit_inception_input",
            "equivalent_terminal_coupling_formula": "2*pi*eps0*epsr_host*r_m^2*Ew_host_per_m",
            "q_equals_C_times_V_interpretation": "effective_terminal_coupling_times_local_cavity_voltage_drop; not global transformer C times test voltage",
            "reference": REFERENCE_URL,
            "initial_phase_rad": 0.0, "initial_wall_memory_V_per_m": 0.0,
            "cycles": 1, "wall_charge_relaxation": "none", "event_solver": "exact_sinusoidal_threshold_crossings",
            "max_events_per_scenario": max_events,
        },
        "limitations": [
            "actual_void_presence_shape_radius_gas_pressure_and_extinction_not_measured",
            "actual_PD_voltage_and_frequency_not_established_by_the_voltage_scenarios",
            "finite_host_electrode_proximity_and_infinite_uniform_sphere_approximation_not_validated",
            "initial_electron_availability_and_statistical_time_lag_omitted",
            "no_wall_charge_decay_surface_conduction_or_multi_void_interactions",
            "induced_terminal_charge_model_not_calibrated_apparent_charge_measurement",
            "no_events_for_an_assumed_void_does_not_establish_PD_free_operation",
        ],
    }
    if field.get("local_field_converged") is not True:
        result["limitations"].append("native_interior_local_field_mesh_convergence_not_verified")
    json.dumps(result, allow_nan=False)
    return result


def estimate_pd_void_batch(payload):
    """Evaluate a bounded named model batch without losing source provenance."""
    if not isinstance(payload, Mapping):
        raise TypeError("PD model batch must be a JSON object")
    payload = json.loads(json.dumps(dict(payload), allow_nan=False))
    models = payload.get("models")
    if not isinstance(models, list) or not models or len(models) > 1000:
        raise ValueError("models must be a nonempty list with at most 1000 entries")
    identifiers, results = set(), []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("each batch model must be a JSON object")
        identifier = model.get("scenario_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("each batch model requires a nonempty scenario_id")
        if identifier.strip() in identifiers:
            raise ValueError(f"duplicate batch scenario_id: {identifier}")
        identifiers.add(identifier.strip())
        if model.get("case") not in {"primary", "secondary"}:
            raise ValueError("each batch model case must be primary or secondary")
        results.append({"case": model["case"], "scenario_id": identifier,
                        "result": estimate_pd_void(model)})
    return {
        "schema_version": "mft-pd-void-scenario-batch-v1",
        "status": "conditional_estimate", "physical_qualification": "insufficient_data",
        "batch_input": payload, "provenance": payload.get("provenance", {}),
        "model_results": results,
    }
