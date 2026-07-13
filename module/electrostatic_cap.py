"""Pure-Python contract for Maxwell electrostatic capacitance results.

Maxwell's ``export_c_matrix`` output contains a source-unit header followed by
the signed Maxwell capacitance matrix.  This module parses that native text,
converts it to SI, restores the full-transformer basis used by result rows, and
computes deliberately simple first-order LC resonance estimates.

The production magnetic eighth model uses an effective doubled current, so its
reported inductances are one half of the full-transformer values.  Its
electrostatic solve uses unchanged voltages and retains one eighth of the field
energy/geometry, so capacitance is one eighth of the full-transformer value.
Consequently eighth-model payloads apply independent restoration factors of 2
to L and 8 to C.  Full-model payloads apply factor 1 to both.  Raw and restored
values are both emitted so this convention is auditable rather than implicit.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping


CAPACITANCE_EXPORT_SCHEMA_VERSION = "maxwell-capacitance-export-v1"
CAPACITANCE_PAYLOAD_SCHEMA_VERSION = "mft-electrostatic-capacitance-v1"
CAPACITANCE_TIMING_PAYLOAD_FIELDS = frozenset({
    "time_cap",
    "cap_solve_time_s",
    "cap_extraction_time_s",
    "cap_stage_added_time_s",
})
CAPACITANCE_PAYLOAD_FIELDS = frozenset({
    "cap_schema_version",
    "cap_model_basis",
    "cap_raw_capacitance_basis",
    "cap_raw_inductance_basis",
    "cap_output_basis",
    "cap_resonance_basis",
    "cap_matrix_source",
    "cap_matrix_source_unit",
    "cap_matrix_order",
    "cap_tx_conductor",
    "cap_rx_conductor",
    "cap_capacitance_restoration_factor",
    "cap_inductance_restoration_factor",
    "cap_inductance_source",
    "cap_inductance_source_unit",
    "cap_resonance_formula",
    "cap_diagonal_interpretation",
    "cap_ground_policy",
    "cap_interwinding_estimate_kind",
    "cap_region_basis",
    "cap_region_remote_padding_percent",
    "C_tx_tx_raw_F",
    "C_rx_rx_raw_F",
    "C_tx_rx_signed_raw_F",
    "C_tx_rx_raw_F",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_signed_F",
    "C_tx_rx_F",
    "cap_L_tx_self_raw_uH",
    "cap_L_rx_self_raw_uH",
    "cap_L_leakage_raw_uH",
    "cap_L_tx_self_H",
    "cap_L_rx_self_H",
    "cap_L_leakage_H",
    "f_res_tx_self_Hz",
    "f_res_rx_self_Hz",
    "f_res_interwinding_Hz",
})

EIGHTH_CAPACITANCE_RESTORATION_FACTOR = 8.0
EIGHTH_INDUCTANCE_RESTORATION_FACTOR = 2.0

_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_UNIT_HEADER_RE = re.compile(
    r"^\s*Capacitance\s+Unit\s*:\s*(\S+)\s*$", re.IGNORECASE
)
_CAPACITANCE_TITLE_RE = re.compile(r"^\s*Capacitance\s*$", re.IGNORECASE)
_COUPLING_TITLE_RE = re.compile(
    r"^\s*Capacitive\s+Coupling\s+Coefficient\s*$", re.IGNORECASE
)


def capacitance_unit_scale(unit):
    """Return the multiplier that converts an AEDT capacitance unit to F.

    AEDT's native capacitance units are fF, pF, nF, uF, mF, and F.  Both
    Unicode micro symbols are accepted because exported files can pass through
    editors that normalize one form to the other.
    """
    if unit is None or isinstance(unit, bool):
        raise ValueError(f"invalid capacitance unit: {unit!r}")
    raw = str(unit).strip()
    if len(raw) >= 2 and raw[0] == "[" and raw[-1] == "]":
        raw = raw[1:-1].strip()
    normalized = raw.replace("µ", "u").replace("μ", "u").lower()
    scales = {
        "ff": 1e-15,
        "pf": 1e-12,
        "nf": 1e-9,
        "uf": 1e-6,
        "mf": 1e-3,
        "f": 1.0,
        "farad": 1.0,
        "farads": 1.0,
    }
    try:
        return scales[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported capacitance unit: {unit!r}") from error


def _finite_matrix_number(token, *, row_name, column_name):
    token = str(token).strip()
    if _NUMBER_RE.fullmatch(token) is None:
        raise ValueError(
            "capacitance matrix contains a non-numeric value at "
            f"({row_name}, {column_name}): {token!r}"
        )
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(
            "capacitance matrix contains a non-finite value at "
            f"({row_name}, {column_name})"
        )
    return value


def _validate_symmetric(matrix, names):
    for row_index, row_name in enumerate(names):
        for column_index in range(row_index + 1, len(names)):
            column_name = names[column_index]
            forward = matrix[row_index][column_index]
            reverse = matrix[column_index][row_index]
            if not math.isclose(
                forward, reverse, rel_tol=1e-9, abs_tol=1e-15
            ):
                raise ValueError(
                    "capacitance matrix is not symmetric: "
                    f"C({row_name},{column_name})={forward!r}, "
                    f"C({column_name},{row_name})={reverse!r}"
                )


def parse_maxwell_capacitance_export(text):
    """Parse a native Maxwell ``export_c_matrix`` text result.

    Returns a dictionary containing the source unit, conductor order, signed
    source-unit matrix, and signed farad matrix.  The parser fails closed on an
    ambiguous report, duplicate/missing rows, non-finite values, or an
    asymmetric matrix.  The later ``Capacitive Coupling Coefficient`` table is
    intentionally ignored; coupling capacitance is the magnitude of signed C12.
    """
    if not isinstance(text, str):
        raise TypeError("Maxwell capacitance export must be text")
    if not text.strip():
        raise ValueError("Maxwell capacitance export is empty")
    lines = text.lstrip("\ufeff").splitlines()

    unit_matches = []
    for index, line in enumerate(lines):
        match = _UNIT_HEADER_RE.fullmatch(line)
        if match is not None:
            unit_matches.append((index, match.group(1)))
    if not unit_matches:
        raise ValueError("Maxwell capacitance export has no 'Capacitance Unit:' header")
    if len(unit_matches) != 1:
        raise ValueError("Maxwell capacitance export has ambiguous unit headers")
    unit_index, source_unit = unit_matches[0]
    unit_to_f = capacitance_unit_scale(source_unit)

    title_indexes = [
        index
        for index in range(unit_index + 1, len(lines))
        if _CAPACITANCE_TITLE_RE.fullmatch(lines[index]) is not None
    ]
    if not title_indexes:
        raise ValueError("Maxwell capacitance export has no 'Capacitance' table")
    if len(title_indexes) != 1:
        raise ValueError("Maxwell capacitance export has ambiguous capacitance tables")

    cursor = title_indexes[0] + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise ValueError("Maxwell capacitance table has no column header")
    names = tuple(lines[cursor].split())
    if not names:
        raise ValueError("Maxwell capacitance table has an empty column header")
    if len(set(names)) != len(names):
        raise ValueError("Maxwell capacitance table has duplicate column names")
    cursor += 1

    rows = {}
    while cursor < len(lines) and len(rows) < len(names):
        line = lines[cursor]
        cursor += 1
        if not line.strip():
            continue
        if _COUPLING_TITLE_RE.fullmatch(line) is not None:
            break
        fields = line.split()
        if len(fields) != len(names) + 1:
            raise ValueError(
                "malformed capacitance matrix row: expected "
                f"{len(names)} values, got {max(len(fields) - 1, 0)}"
            )
        row_name = fields[0]
        if row_name not in names:
            raise ValueError(
                f"unexpected capacitance matrix row name: {row_name!r}"
            )
        if row_name in rows:
            raise ValueError(
                f"duplicate capacitance matrix row name: {row_name!r}"
            )
        rows[row_name] = tuple(
            _finite_matrix_number(value, row_name=row_name, column_name=column_name)
            for column_name, value in zip(names, fields[1:])
        )

    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError(
            "Maxwell capacitance matrix is missing rows: " + ", ".join(missing)
        )
    matrix_raw = tuple(rows[name] for name in names)
    _validate_symmetric(matrix_raw, names)
    matrix_f = tuple(
        tuple(value * unit_to_f for value in row) for row in matrix_raw
    )
    return {
        "schema_version": CAPACITANCE_EXPORT_SCHEMA_VERSION,
        "source": "Maxwell Electrostatic export_c_matrix",
        "unit": source_unit,
        "unit_to_f": unit_to_f,
        "names": names,
        "matrix_raw": matrix_raw,
        "matrix_f": matrix_f,
    }


def _positive_finite(value, label):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def lc_resonance_hz(inductance_h, capacitance_f):
    """Return ``1 / (2*pi*sqrt(L*C))`` after strict SI validation."""
    inductance_h = _positive_finite(inductance_h, "inductance_h")
    capacitance_f = _positive_finite(capacitance_f, "capacitance_f")
    product = inductance_h * capacitance_f
    if not math.isfinite(product) or product <= 0.0:
        raise ValueError("L*C must be finite and representable")
    frequency = 1.0 / (2.0 * math.pi * math.sqrt(product))
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("LC resonance frequency is not finite")
    return frequency


def build_capacitance_timing_payload(
    solve_time_s, extraction_time_s, stage_added_time_s
):
    """Build the exact timing aliases appended by the optional stage."""
    values = {}
    for label, value in (
        ("cap_solve_time_s", solve_time_s),
        ("cap_extraction_time_s", extraction_time_s),
        ("cap_stage_added_time_s", stage_added_time_s),
    ):
        if value is None or isinstance(value, bool):
            raise ValueError(f"{label} must be a non-negative finite number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{label} must be a non-negative finite number"
            ) from error
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label} must be a non-negative finite number")
        values[label] = number
    return {
        "time_cap": values["cap_solve_time_s"],
        **values,
    }


def _normalize_full_model(full_model):
    if isinstance(full_model, bool):
        return full_model
    if isinstance(full_model, (int, float)) and not isinstance(full_model, bool):
        numeric = float(full_model)
        if math.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(int(numeric))
    raise ValueError("full_model must be bool or numeric 0/1")


def _validated_parsed_matrix(parsed):
    if not isinstance(parsed, Mapping):
        raise TypeError("parsed capacitance result must be a mapping")
    try:
        names = tuple(parsed["names"])
        matrix_raw = tuple(tuple(row) for row in parsed["matrix_raw"])
        matrix_f = tuple(tuple(row) for row in parsed["matrix_f"])
        source_unit = str(parsed["unit"])
        unit_to_f = float(parsed["unit_to_f"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("parsed capacitance result is incomplete") from error
    if not names or len(set(names)) != len(names):
        raise ValueError("parsed capacitance conductor names are invalid")
    if len(matrix_raw) != len(names) or len(matrix_f) != len(names):
        raise ValueError("parsed capacitance matrix is not square")
    if any(len(row) != len(names) for row in matrix_raw + matrix_f):
        raise ValueError("parsed capacitance matrix is not square")
    expected_scale = capacitance_unit_scale(source_unit)
    if (
        not math.isfinite(unit_to_f)
        or unit_to_f <= 0.0
        or not math.isclose(unit_to_f, expected_scale, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("parsed capacitance unit conversion is inconsistent")

    normalized_raw = []
    normalized_f = []
    for row_index, row_name in enumerate(names):
        raw_row = []
        farad_row = []
        for column_index, column_name in enumerate(names):
            raw_value = _finite_matrix_number(
                matrix_raw[row_index][column_index],
                row_name=row_name,
                column_name=column_name,
            )
            try:
                farad_value = float(matrix_f[row_index][column_index])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("parsed farad matrix contains a non-numeric value") from error
            if not math.isfinite(farad_value):
                raise ValueError("parsed farad matrix contains a non-finite value")
            expected_farad = raw_value * expected_scale
            if not math.isclose(
                farad_value,
                expected_farad,
                rel_tol=1e-12,
                abs_tol=max(abs(expected_farad) * 1e-15, 1e-300),
            ):
                raise ValueError("parsed raw and farad capacitance matrices disagree")
            raw_row.append(raw_value)
            farad_row.append(farad_value)
        normalized_raw.append(tuple(raw_row))
        normalized_f.append(tuple(farad_row))
    normalized_raw = tuple(normalized_raw)
    normalized_f = tuple(normalized_f)
    _validate_symmetric(normalized_raw, names)
    _validate_symmetric(normalized_f, names)
    return names, normalized_raw, normalized_f, source_unit


def build_capacitance_payload(
    parsed,
    ltx_uH,
    lrx_uH,
    llt_uH,
    *,
    full_model=False,
    tx_name="CapTx",
    rx_name="CapRx",
):
    """Build full-physical C/L resonance fields from a parsed Maxwell matrix.

    ``ltx_uH``, ``lrx_uH``, and ``llt_uH`` are the matrix-stage result values
    on that design's native basis.  For the eighth model they are multiplied by
    the established magnetic factor 2 before resonance calculations.  The
    signed Maxwell off-diagonal term is retained, while its absolute magnitude
    is used as the positive interwinding coupling capacitance.
    """
    names, _matrix_raw, matrix_f, source_unit = _validated_parsed_matrix(parsed)
    full_model = _normalize_full_model(full_model)
    tx_name = str(tx_name).strip()
    rx_name = str(rx_name).strip()
    if not tx_name or not rx_name or tx_name == rx_name:
        raise ValueError("Tx/Rx capacitance conductor names must be distinct")
    try:
        tx_index = names.index(tx_name)
        rx_index = names.index(rx_name)
    except ValueError as error:
        raise ValueError(
            f"capacitance matrix must contain {tx_name!r} and {rx_name!r}"
        ) from error

    c_tx_tx_raw_f = _positive_finite(
        matrix_f[tx_index][tx_index], "C_tx_tx_raw_F"
    )
    c_rx_rx_raw_f = _positive_finite(
        matrix_f[rx_index][rx_index], "C_rx_rx_raw_F"
    )
    c_tx_rx_signed_raw_f = float(matrix_f[tx_index][rx_index])
    if not math.isfinite(c_tx_rx_signed_raw_f):
        raise ValueError("C_tx_rx_signed_raw_F must be finite")
    if c_tx_rx_signed_raw_f >= 0.0:
        raise ValueError(
            "C_tx_rx_signed_raw_F must be negative for a passive Maxwell "
            "coefficient matrix"
        )
    c_tx_rx_raw_f = abs(c_tx_rx_signed_raw_f)
    coupling_limit = math.sqrt(c_tx_tx_raw_f * c_rx_rx_raw_f)
    if c_tx_rx_raw_f > coupling_limit * (1.0 + 1e-9):
        raise ValueError("selected Tx/Rx capacitance submatrix is not positive semidefinite")
    # Eliminating the explicit ground conductor yields a passive Maxwell
    # coefficient matrix: off-diagonals are negative and each row sum is the
    # non-negative partial capacitance from that signal net to ground.  This
    # catches a valid-looking symmetric/PSD table with the wrong source or sign
    # convention before it can generate a plausible but meaningless resonance.
    # Native export formatting can round separately printed coefficients, so
    # allow one part per million when a theoretically zero row sum straddles
    # zero in text.  Materially negative ground partials still fail closed.
    ground_tolerance_f = max(c_tx_tx_raw_f, c_rx_rx_raw_f) * 1e-6
    if (
        c_tx_tx_raw_f + c_tx_rx_signed_raw_f < -ground_tolerance_f
        or c_rx_rx_raw_f + c_tx_rx_signed_raw_f < -ground_tolerance_f
    ):
        raise ValueError(
            "selected Tx/Rx capacitance matrix has a negative ground partial"
        )

    ltx_raw_uh = _positive_finite(ltx_uH, "ltx_uH")
    lrx_raw_uh = _positive_finite(lrx_uH, "lrx_uH")
    llt_raw_uh = _positive_finite(llt_uH, "llt_uH")

    capacitance_factor = (
        1.0 if full_model else EIGHTH_CAPACITANCE_RESTORATION_FACTOR
    )
    inductance_factor = (
        1.0 if full_model else EIGHTH_INDUCTANCE_RESTORATION_FACTOR
    )
    c_tx_tx_f = c_tx_tx_raw_f * capacitance_factor
    c_rx_rx_f = c_rx_rx_raw_f * capacitance_factor
    c_tx_rx_signed_f = c_tx_rx_signed_raw_f * capacitance_factor
    c_tx_rx_f = c_tx_rx_raw_f * capacitance_factor
    ltx_h = ltx_raw_uh * 1e-6 * inductance_factor
    lrx_h = lrx_raw_uh * 1e-6 * inductance_factor
    llt_h = llt_raw_uh * 1e-6 * inductance_factor

    model_basis = "full" if full_model else "eighth"
    raw_geometry_basis = "full_geometry" if full_model else "retained_eighth_geometry"
    raw_inductance_basis = (
        "full_model_matrix" if full_model else "eighth_current_driven_matrix"
    )
    return {
        "cap_schema_version": CAPACITANCE_PAYLOAD_SCHEMA_VERSION,
        "cap_model_basis": model_basis,
        "cap_raw_capacitance_basis": raw_geometry_basis,
        "cap_raw_inductance_basis": raw_inductance_basis,
        "cap_output_basis": "full_physical",
        "cap_resonance_basis": "full_physical_restored_L_and_C",
        "cap_matrix_source": "Maxwell Electrostatic export_c_matrix",
        "cap_matrix_source_unit": source_unit,
        "cap_matrix_order": json.dumps(list(names), separators=(",", ":")),
        "cap_tx_conductor": tx_name,
        "cap_rx_conductor": rx_name,
        "cap_capacitance_restoration_factor": capacitance_factor,
        "cap_inductance_restoration_factor": inductance_factor,
        "cap_inductance_source": "matrix_stage:Ltx,Lrx,Llt",
        "cap_inductance_source_unit": "uH",
        "cap_resonance_formula": "1/(2*pi*sqrt(L_H*C_F))",
        "cap_diagonal_interpretation": (
            "grounded_other_signal_maxwell_coefficient"
        ),
        "cap_ground_policy": (
            "other_signal_core_plates_and_remote_region_at_0V"
        ),
        "cap_interwinding_estimate_kind": (
            "primary_referred_leakage_mutual_C_first_order_heuristic"
        ),
        "cap_region_basis": (
            "full_geometry_100pct_padding_matched_by_eighth_200pct_remote"
        ),
        "cap_region_remote_padding_percent": 100.0 if full_model else 200.0,
        "C_tx_tx_raw_F": c_tx_tx_raw_f,
        "C_rx_rx_raw_F": c_rx_rx_raw_f,
        "C_tx_rx_signed_raw_F": c_tx_rx_signed_raw_f,
        "C_tx_rx_raw_F": c_tx_rx_raw_f,
        "C_tx_tx_F": c_tx_tx_f,
        "C_rx_rx_F": c_rx_rx_f,
        "C_tx_rx_signed_F": c_tx_rx_signed_f,
        "C_tx_rx_F": c_tx_rx_f,
        "cap_L_tx_self_raw_uH": ltx_raw_uh,
        "cap_L_rx_self_raw_uH": lrx_raw_uh,
        "cap_L_leakage_raw_uH": llt_raw_uh,
        "cap_L_tx_self_H": ltx_h,
        "cap_L_rx_self_H": lrx_h,
        "cap_L_leakage_H": llt_h,
        "f_res_tx_self_Hz": lc_resonance_hz(ltx_h, c_tx_tx_f),
        "f_res_rx_self_Hz": lc_resonance_hz(lrx_h, c_rx_rx_f),
        "f_res_interwinding_Hz": lc_resonance_hz(llt_h, c_tx_rx_f),
    }
