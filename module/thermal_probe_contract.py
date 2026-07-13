"""Versioned semantics for the Rx-side winding thermal face probes.

The side winding exists at negative x in symmetry models and at both negative
and positive x in a full model.  ``outward`` always means away from the
transformer centre (x=0), while ``inward`` means toward it.  The representative
temperature is deliberately selected from the hottest face; face means are
never averaged because a future geometry may give the faces different areas.
"""

from __future__ import annotations

import json
import math
import re


RX_SIDE_FACE_PROBE_CONTRACT_VERSION = "rx-side-transformer-inner-outer-v1"
RX_SIDE_FACE_MAX_RULE = "max_across_all_transformer_inner_and_outer_faces"
RX_SIDE_FACE_MEAN_RULE = "mean_of_face_selected_by_max_no_cross_face_average"

_TEMPERATURE_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"([A-Za-z°_ ]*)\s*$"
)
_CELSIUS_UNITS = {"", "c", "cel", "degc", "celsius"}
_KELVIN_UNITS = {"k", "kel", "degk", "kelvin"}
_RECTANGLE_ORIENTATIONS = {"XY", "XZ", "YZ", "ZX"}


class ProbeSheetCollection(list):
    """List-compatible probe result carrying the attempted geometry contract."""

    def __init__(self):
        super().__init__()
        self.expected_names = []
        self.failures = []

    def expect(self, name):
        name = str(name)
        if name not in self.expected_names:
            self.expected_names.append(name)

    def record_failure(self, name, stage, reason, detail=""):
        record = {
            "probe": str(name),
            "stage": str(stage),
            "reason": str(reason),
        }
        detail = str(detail or "").strip()
        if detail:
            record["detail"] = detail[:512]
        self.failures.append(record)
        return record


def validate_probe_rectangle(name, orientation, origin, sizes):
    """Validate a parameter-derived AEDT probe rectangle before creating it.

    A zero/negative span creates a sheet with no usable surface mesh.  Non-
    finite coordinates can also cross the gRPC boundary as strings and fail
    later, during Field Summary, which makes the geometry error look like an
    extraction error.  Reject both cases at their source.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("probe name is empty")
    orientation = str(orientation).strip().upper()
    if orientation not in _RECTANGLE_ORIENTATIONS:
        raise ValueError(f"unsupported rectangle orientation: {orientation!r}")
    if len(origin) != 3:
        raise ValueError(f"origin must have 3 coordinates, got {len(origin)}")
    if len(sizes) != 2:
        raise ValueError(f"rectangle must have 2 spans, got {len(sizes)}")
    try:
        numeric_origin = tuple(float(value) for value in origin)
        numeric_sizes = tuple(float(value) for value in sizes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probe coordinates are not numeric") from exc
    if not all(math.isfinite(value) for value in numeric_origin + numeric_sizes):
        raise ValueError("probe coordinates must be finite")
    if any(value <= 0.0 for value in numeric_sizes):
        raise ValueError(f"probe spans must be positive, got {numeric_sizes!r}")
    return orientation, numeric_origin, numeric_sizes


def parse_temperature_celsius(value, unit=None):
    """Return a finite Celsius scalar from numeric or unit-tagged AEDT data."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"temperature has no numeric value: {value!r}")
    embedded_unit = ""
    if isinstance(value, str):
        match = _TEMPERATURE_RE.match(value)
        if match is None:
            raise ValueError(f"invalid temperature value: {value!r}")
        number = float(match.group(1))
        embedded_unit = match.group(2)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid temperature value: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"temperature must be finite: {number!r}")

    def _normalize(raw):
        if raw is None:
            return ""
        if not isinstance(raw, str):
            try:
                if math.isnan(float(raw)):
                    return ""
            except (TypeError, ValueError, OverflowError):
                pass
        return str(raw).strip().lower().replace("°", "").replace("_", "").replace(" ", "")

    embedded_unit = _normalize(embedded_unit)
    supplied_unit = _normalize(unit)
    if embedded_unit and supplied_unit:
        same_celsius = (
            embedded_unit in _CELSIUS_UNITS and supplied_unit in _CELSIUS_UNITS
        )
        same_kelvin = (
            embedded_unit in _KELVIN_UNITS and supplied_unit in _KELVIN_UNITS
        )
        if not (same_celsius or same_kelvin):
            raise ValueError(
                f"conflicting temperature units: {embedded_unit!r} vs {supplied_unit!r}"
            )
    resolved_unit = embedded_unit or supplied_unit
    if resolved_unit in _CELSIUS_UNITS:
        return number
    if resolved_unit in _KELVIN_UNITS:
        return number - 273.15
    raise ValueError(f"unsupported temperature unit: {resolved_unit!r}")


def serialize_probe_failures(failures):
    """Serialize bounded per-probe diagnostics deterministically for result rows."""
    return json.dumps(list(failures), sort_keys=True, separators=(",", ":"))


def select_hottest_complete_face(temperatures, face_names):
    """Return ``(maximum, paired_mean, name)`` only for a complete face set.

    Requiring both max and mean for every requested face prevents a missing
    inward face from silently falling back to the historical outward-only
    result.  The returned mean belongs to the exact face that supplied the
    maximum; it is not an unweighted average across potentially unequal areas.
    """
    names = tuple(dict.fromkeys(str(name) for name in face_names))
    if not names:
        return None
    candidates = []
    for name in names:
        try:
            maximum = float(temperatures[f"{name}_max"])
            mean = float(temperatures[f"{name}_mean"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(maximum) and math.isfinite(mean)):
            return None
        candidates.append((maximum, mean, name))
    return max(candidates, key=lambda item: item[0])


def aggregate_rx_side_faces(temperatures, outward_names, inward_names):
    """Return complete aggregate fields plus the representative face name.

    An empty mapping and empty name are returned until both transformer-
    relative face sets are complete.  This makes partial extraction unusable
    as a deceptively low temperature target.
    """
    outward_names = tuple(outward_names)
    inward_names = tuple(inward_names)
    if not outward_names or len(outward_names) != len(inward_names):
        return {}, ""
    outward = select_hottest_complete_face(temperatures, outward_names)
    inward = select_hottest_complete_face(temperatures, inward_names)
    representative = select_hottest_complete_face(
        temperatures, outward_names + inward_names
    )
    if outward is None or inward is None or representative is None:
        return {}, ""
    return {
        "Tprobe_Rx_side_outer_max": outward[0],
        "Tprobe_Rx_side_outer_mean": outward[1],
        "Tprobe_Rx_side_inner_max": inward[0],
        "Tprobe_Rx_side_inner_mean": inward[1],
        "Tprobe_Rx_side_leeward_max": representative[0],
        "Tprobe_Rx_side_leeward_mean": representative[1],
    }, representative[2]
