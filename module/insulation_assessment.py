"""Evidence-based assessment of the two grounded-hardware AC insulation tests.

This module never launches a solver. ``assess_insulation`` accepts a mapping,
JSON text, or a pathlib.Path to JSON and returns a strict JSON-serializable dict.
Missing evidence yields ``insufficient_data`` while diagnostic field values are
retained. Capacitance solve success, field stress, measured AC withstand, and
measured partial discharge are independent findings.

Minimal useful input (omitted evidence remains explicitly unresolved)::

    {
      "model": {
        "source_geometry_kind": "actual_cad",
        "cad_identity": "CAD revision/hash",
        "target_cad_identity": "CAD revision/hash",
        "verified_final_cad": true,
        "dielectric_complete": true,
        "conductive_hardware_complete": true,
        "material_assignments_verified": true,
        "model_basis": "full", "geometry_factor": 1
      },
      "provenance": {"project": "...", "design": "..."},
      "capacitance_solve": {"success": true, "source": "native export"},
      "materials": [{
        "name": "paper", "allowable_peak_field_V_per_m": 1000000,
        "limit_evidence": {
          "source": "applicable material qualification",
          "applicable_to_ac_withstand": true
        }
      }],
      "convergence": {
        "field_local_max_converged": true, "source": "mesh field history",
        "field_max_relative_change": 0.01,
        "field_max_relative_tolerance": 0.02
      },
      "tests": {
        "primaryHV_secondaryHWground": {
          "native_field_evidence": {
            "source": "native Maxwell field export", "solution": "Setup1",
            "normalization_voltage_V": 1, "unit": "V/m"
          },
          "boundary_conditions": {
            "terminals_within_each_winding_shorted": true,
            "primary_secondary_connected": false,
            "other_winding_grounded": true,
            "all_core_cooling_hardware_grounded": true,
            "excited_winding": "primary"
          },
          "dielectric_regions_covered": true,
          "hotspots": [{"object_name": "Paper1", "material": "paper",
                        "E_1V_V_per_m": 1, "location_mm": [0, 0, 0]}]
        }
      }
    }

Provide both TEST_NAMES. Each test may override ``convergence`` and include
``electrostatic_energy_1V_J``. Eighth geometry requires factor 8 and explicit
``electric_potential_symmetry_verified``. The factor restores energy only;
local E always scales by 20000*sqrt(2), with no geometry multiplier. Conductors
can be listed as ``{"name": "copper", "is_conductor": true}``.

Actual records are lists in ``withstand_measurements`` / ``pd_measurements``.
Both require test_name, source, specimen_cad_identity, applied_voltage_rms_V,
hold_time_s, frequency_Hz, and the same recorded boundary_conditions. Withstand
requires kind="measured_ac_withstand", breakdown=false, and calibration.source
plus calibration.date. PD requires kind="measured_apparent_charge",
apparent_charge_pC and calibration.source/date/charge_pC. PD voltage is specified
independently in ``pd_test_conditions`` (voltage_rms_V, hold_time_s,
frequency_Hz); it is never inferred from the withstand voltage. Withstand
required hold_time_s and frequency_Hz belong in ``test_conditions``. A measured
result lacking the requested hold/frequency cannot qualify a test.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


SCHEMA_VERSION = "mft-insulation-assessment-v1"
TEST_NAMES = (
    "primaryHV_secondaryHWground",
    "secondaryHV_primaryHWground",
)
WITHSTAND_RMS_V = 20000.0
PD_LIMIT_PC = 15.0


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value, *, positive=False):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        return None
    return result


def _present(value):
    return isinstance(value, str) and bool(value.strip())


def _finding(status, reasons, **details):
    return {"status": status, "reasons": list(dict.fromkeys(reasons)), **details}


def _model_reasons(model):
    reasons = []
    if model.get("source_geometry_kind") != "actual_cad":
        reasons.append("source_geometry_is_not_verified_actual_cad")
    for key in ("verified_final_cad", "dielectric_complete",
                "conductive_hardware_complete", "material_assignments_verified"):
        if model.get(key) is not True:
            reasons.append(key + "_not_verified")
    if not _present(model.get("cad_identity")) or not _present(model.get("target_cad_identity")):
        reasons.append("cad_identity_or_target_identity_missing")
    elif model["cad_identity"] != model["target_cad_identity"]:
        reasons.append("source_cad_identity_does_not_match_target")
    basis = model.get("model_basis")
    expected_factor = {"full": 1.0, "eighth": 8.0}.get(basis)
    factor = _number(model.get("geometry_factor"), positive=True)
    if expected_factor is None or factor != expected_factor:
        reasons.append("model_basis_or_geometry_factor_invalid")
        factor = None
    if basis == "eighth" and model.get("electric_potential_symmetry_verified") is not True:
        reasons.append("electric_potential_symmetry_not_verified")
    return reasons, factor


def _boundary_reasons(boundary, winding):
    reasons = []
    for key in ("terminals_within_each_winding_shorted", "other_winding_grounded",
                "all_core_cooling_hardware_grounded"):
        if boundary.get(key) is not True:
            reasons.append(key + "_not_verified")
    if boundary.get("primary_secondary_connected") is not False:
        reasons.append("primary_secondary_separation_not_verified")
    if boundary.get("excited_winding") != winding:
        reasons.append("excited_winding_not_verified")
    return reasons


def _convergence_reasons(convergence):
    reasons = []
    if convergence.get("field_local_max_converged") is not True:
        reasons.append("local_field_maximum_convergence_not_verified")
    if not _present(convergence.get("source")):
        reasons.append("local_field_convergence_source_missing")
    change = _number(convergence.get("field_max_relative_change"))
    tolerance = _number(convergence.get("field_max_relative_tolerance"), positive=True)
    if change is None or tolerance is None:
        reasons.append("local_field_convergence_change_or_tolerance_missing")
    elif change > tolerance:
        reasons.append("local_field_convergence_tolerance_not_met")
    return reasons


def _field_assessment(test, model_reasons, factor, materials, convergence, winding):
    reasons = list(model_reasons)
    reasons.extend(_boundary_reasons(_mapping(test.get("boundary_conditions")), winding))
    reasons.extend(_convergence_reasons(convergence))
    native = _mapping(test.get("native_field_evidence"))
    native_valid = True
    for key in ("source", "solution"):
        if not _present(native.get(key)):
            reasons.append("native_field_" + key + "_missing")
            native_valid = False
    if native.get("unit") != "V/m":
        reasons.append("native_field_unit_not_V_per_m")
        native_valid = False
    if _number(native.get("normalization_voltage_V"), positive=True) != 1.0:
        reasons.append("native_field_not_normalized_to_1V")
        native_valid = False
    if test.get("dielectric_regions_covered") is not True:
        reasons.append("all_dielectric_regions_not_verified_covered")
    raw_hotspots = test.get("hotspots")
    if not isinstance(raw_hotspots, list) or not raw_hotspots:
        reasons.append("native_dielectric_field_hotspots_missing")
        raw_hotspots = []
    hotspots = []
    exceeds = False
    for raw in raw_hotspots:
        raw = _mapping(raw)
        hotspot_reasons = []
        material = materials.get(raw.get("material"), {})
        if material.get("is_conductor") is True:
            continue
        if not _present(raw.get("object_name")):
            hotspot_reasons.append("hotspot_object_name_missing")
        if not _present(raw.get("material")):
            hotspot_reasons.append("hotspot_material_missing")
        location = raw.get("location_mm")
        valid_location = (isinstance(location, (list, tuple)) and len(location) == 3
                          and all(not isinstance(x, bool) and isinstance(x, (int, float))
                                  and math.isfinite(x) for x in location))
        if not valid_location:
            hotspot_reasons.append("hotspot_location_missing_or_invalid")
        normalized = _number(raw.get("E_1V_V_per_m"))
        peak = normalized * WITHSTAND_RMS_V * math.sqrt(2) if normalized is not None and native_valid else None
        if peak is not None and not math.isfinite(peak):
            peak = None
        if normalized is None or peak is None:
            hotspot_reasons.append("native_field_value_invalid_or_unusable")
        limit = _number(material.get("allowable_peak_field_V_per_m"), positive=True)
        evidence = _mapping(material.get("limit_evidence"))
        if (limit is None or not _present(evidence.get("source"))
                or evidence.get("applicable_to_ac_withstand") is not True):
            limit = None
            hotspot_reasons.append("applicable_material_allowable_peak_field_missing")
        margin = limit / peak if limit is not None and peak is not None and peak > 0 else None
        if margin is not None and not math.isfinite(margin):
            margin = None
        over_limit = peak is not None and limit is not None and peak > limit
        exceeds |= over_limit
        hotspots.append({
            "object_name": raw.get("object_name"), "material": raw.get("material"),
            "location_mm": list(location) if valid_location else None,
            "E_1V_V_per_m": normalized, "E_peak_20kV_V_per_m": peak,
            "allowable_peak_field_V_per_m": limit, "field_margin": margin,
            "limit_evidence": dict(evidence),
            "status": "insufficient_data" if hotspot_reasons else "exceeds_limit" if over_limit else "within_limit",
            "reasons": hotspot_reasons,
        })
        reasons.extend(hotspot_reasons)
    if not hotspots:
        reasons.append("no_dielectric_hotspots_available")
    peak_values = [h["E_peak_20kV_V_per_m"] for h in hotspots if h["E_peak_20kV_V_per_m"] is not None]
    energy = _number(test.get("electrostatic_energy_1V_J"))
    full_peak_energy = energy * (WITHSTAND_RMS_V * math.sqrt(2)) ** 2 * factor if energy is not None and factor is not None and native_valid else None
    if full_peak_energy is not None and not math.isfinite(full_peak_energy):
        full_peak_energy = None
        reasons.append("scaled_energy_not_finite")
    return _finding(
        "insufficient_data" if reasons else "exceeds_limit" if exceeds else "within_limit",
        reasons, hotspots=hotspots, native_field_evidence=dict(native),
        convergence=dict(convergence),
        maximum_E_peak_20kV_V_per_m=max(peak_values) if peak_values else None,
        E_voltage_scale=WITHSTAND_RMS_V * math.sqrt(2), local_E_geometry_factor=1.0,
        electrostatic_energy_1V_J=energy, energy_geometry_factor=factor,
        electrostatic_energy_full_at_voltage_peak_J=full_peak_energy,
    )


def _measurement_assessment(records, name, winding, target_identity, requested, *, pd):
    prefix = "pd" if pd else "withstand"
    records = records if isinstance(records, list) else []
    records = [_mapping(record) for record in records if _mapping(record).get("test_name") == name]
    required_voltage = _number(requested.get("voltage_rms_V"), positive=True)
    required_hold = _number(requested.get("hold_time_s"), positive=True)
    required_frequency = _number(requested.get("frequency_Hz"), positive=True)
    requirement_reasons = []
    for key, value in (("voltage", required_voltage), ("hold_time", required_hold), ("frequency", required_frequency)):
        if value is None:
            requirement_reasons.append(prefix + "_required_" + key + "_not_specified")
    if not records:
        return _finding("insufficient_data", requirement_reasons + [prefix + "_actual_measurement_missing"],
                        required_conditions=dict(requested), records=[])
    findings = []
    for record in records:
        reasons = list(requirement_reasons)
        if record.get("kind") != ("measured_apparent_charge" if pd else "measured_ac_withstand"):
            reasons.append(prefix + "_evidence_is_not_actual_measurement")
        if not _present(record.get("source")):
            reasons.append(prefix + "_measurement_source_missing")
        if not _present(target_identity) or record.get("specimen_cad_identity") != target_identity:
            reasons.append(prefix + "_specimen_identity_not_verified_for_target")
        reasons.extend(_boundary_reasons(_mapping(record.get("boundary_conditions")), winding))
        voltage = _number(record.get("applied_voltage_rms_V"), positive=True)
        hold = _number(record.get("hold_time_s"), positive=True)
        frequency = _number(record.get("frequency_Hz"), positive=True)
        for key, value in (("voltage", voltage), ("hold_time", hold), ("frequency", frequency)):
            if value is None:
                reasons.append(prefix + "_measurement_" + key + "_missing")
        if voltage is not None and required_voltage is not None:
            if (pd and not math.isclose(voltage, required_voltage, rel_tol=1e-6)) or (not pd and voltage < required_voltage):
                reasons.append(prefix + "_measurement_voltage_does_not_meet_requested_conditions")
        if hold is not None and required_hold is not None and hold < required_hold:
            reasons.append(prefix + "_measurement_hold_time_below_requirement")
        if frequency is not None and required_frequency is not None and not math.isclose(frequency, required_frequency, rel_tol=1e-6):
            reasons.append(prefix + "_measurement_frequency_does_not_match_requirement")
        calibration = _mapping(record.get("calibration"))
        if not _present(calibration.get("source")) or not _present(calibration.get("date")):
            reasons.append(prefix + "_calibration_evidence_missing")
        if pd:
            if _number(calibration.get("charge_pC"), positive=True) is None:
                reasons.append("pd_calibration_charge_missing")
            charge = _number(record.get("apparent_charge_pC"))
            if charge is None:
                reasons.append("measured_apparent_charge_missing_or_invalid")
            failed = charge is not None and charge > PD_LIMIT_PC
        else:
            if not isinstance(record.get("breakdown"), bool):
                reasons.append("withstand_breakdown_observation_missing")
            failed = record.get("breakdown") is True
        findings.append(_finding("insufficient_data" if reasons else "fail" if failed else "pass",
                                 reasons, measurement=dict(record)))
    statuses = [finding["status"] for finding in findings]
    status = "fail" if "fail" in statuses else "pass" if "pass" in statuses else "insufficient_data"
    reasons = [reason for finding in findings for reason in finding["reasons"]] if status == "insufficient_data" else []
    return _finding(status, reasons, required_conditions=dict(requested), records=findings)


def assess_insulation(payload):
    """Assess native normalized fields and independent physical test evidence.

    See the module docstring for the input schema. Numeric units are explicit
    SI except PD (pC) and hotspot coordinates (mm). Malformed input structure
    or non-JSON/nonfinite values raises TypeError/ValueError. Evidence gaps are
    returned as reason codes. ``within_limit`` describes a simulated field
    comparison; only complete physical measurement evidence can yield a
    withstand/PD ``pass``. The fixed AC test voltage is 20 kV RMS.
    """
    if isinstance(payload, Path):
        payload = payload.read_text(encoding="utf-8-sig")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("insulation assessment input must be a mapping, JSON object, or Path")
    # This also detaches the result from mutable caller-owned data.
    payload = json.loads(json.dumps(dict(payload), allow_nan=False))
    model = _mapping(payload.get("model"))
    model_reasons, factor = _model_reasons(model)
    materials_list = payload.get("materials", [])
    materials = {}
    if isinstance(materials_list, list):
        for material in materials_list:
            material = _mapping(material)
            if _present(material.get("name")):
                if material["name"] in materials:
                    model_reasons.append("duplicate_material_name")
                materials[material["name"]] = material
    cap = _mapping(payload.get("capacitance_solve"))
    cap_status = "success" if cap.get("success") is True and _present(cap.get("source")) else "failed" if cap.get("success") is False else "insufficient_data"
    conditions = dict(_mapping(payload.get("test_conditions")))
    conditions.setdefault("voltage_rms_V", WITHSTAND_RMS_V)
    if _number(conditions.get("voltage_rms_V"), positive=True) != WITHSTAND_RMS_V:
        raise ValueError("the insulation withstand assessment requires 20000 V RMS")
    pd_conditions = dict(_mapping(payload.get("pd_test_conditions")))
    tests = {}
    for name, winding in zip(TEST_NAMES, ("primary", "secondary")):
        test = _mapping(_mapping(payload.get("tests")).get(name))
        convergence = _mapping(test.get("convergence", payload.get("convergence")))
        field = _field_assessment(test, model_reasons, factor, materials, convergence, winding)
        withstand = _measurement_assessment(payload.get("withstand_measurements"), name, winding,
                                            model.get("target_cad_identity"), conditions, pd=False)
        pd = _measurement_assessment(payload.get("pd_measurements"), name, winding,
                                    model.get("target_cad_identity"), pd_conditions, pd=True)
        tests[name] = {
            "excited_winding": winding, "test_conditions": conditions,
            "boundary_conditions": dict(_mapping(test.get("boundary_conditions"))),
            "electric_field_assessment": field,
            "actual_20kV_withstand_qualification": withstand,
            "pd_apparent_charge_assessment": {**pd, "limit_pC": PD_LIMIT_PC},
        }
    statuses = []
    reasons = []
    for name, test in tests.items():
        for key in ("electric_field_assessment", "actual_20kV_withstand_qualification", "pd_apparent_charge_assessment"):
            statuses.append(test[key]["status"])
            reasons.extend(name + ":" + key + ":" + reason for reason in test[key]["reasons"])
    overall = "fail" if any(s in ("fail", "exceeds_limit") for s in statuses) else "insufficient_data" if "insufficient_data" in statuses else "qualified_by_measurement_and_supported_by_field_assessment"
    result = {
        "schema_version": SCHEMA_VERSION, "status": overall, "reasons": reasons,
        "model": dict(model), "provenance": dict(_mapping(payload.get("provenance"))),
        "materials": materials_list, "convergence": dict(_mapping(payload.get("convergence"))),
        "test_conditions": {**conditions, "voltage_peak_V": WITHSTAND_RMS_V * math.sqrt(2),
                            "waveform": "AC_sinusoidal", "each_winding_terminals_shorted": True,
                            "primary_secondary_connected": False,
                            "all_core_cooling_conductive_hardware_grounded": True},
        "capacitance_solve": _finding(cap_status, [] if cap_status == "success" else ["capacitance_solve_not_verified_successful"], evidence=dict(cap)),
        "tests": tests,
        "qualification_policy": {
            "field_simulation_alone_establishes_withstand_pass": False,
            "capacitance_times_voltage_is_pd_apparent_charge": False,
            "pd_voltage_inferred_from_withstand_voltage": False,
            "local_E_geometry_multiplier": 1.0,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


build_insulation_assessment = assess_insulation
