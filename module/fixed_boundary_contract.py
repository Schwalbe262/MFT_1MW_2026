"""Authoritative thermal-boundary contract for production MFT results.

Deadline mesh experiments may deliberately vary fan speed, pad thickness, or
TIM conductivity.  Those runs are useful diagnostics, but they must never be
mistaken for the fixed-boundary production design.  This module is the single
source of truth shared by defaults, the thermal builder, recovery, and result
integration.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


FIXED_BOUNDARY_CONTRACT_SCHEMA = "mft-fixed-thermal-boundary-v1"
FIXED_FAN_VELOCITY_M_S = 1.5
FIXED_CORE_PLATE_PAD_THICKNESS_MM = 2.0
FIXED_WCP_PAD_THICKNESS_MM = 2.0
FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK = 0.2

_PARAMETER_EXPECTATIONS = {
    "fan_velocity": FIXED_FAN_VELOCITY_M_S,
    "core_plate_pad_t": FIXED_CORE_PLATE_PAD_THICKNESS_MM,
    "wcp_pad_t": FIXED_WCP_PAD_THICKNESS_MM,
}
_EXPECTED_VALUES = {
    **_PARAMETER_EXPECTATIONS,
    "thermal_pad_conductivity_W_mK": (
        FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
    ),
}
_UNITS = {
    "fan_velocity": "m/s",
    "core_plate_pad_t": "mm",
    "wcp_pad_t": "mm",
    "thermal_pad_conductivity_W_mK": "W/(m*K)",
}
_CONTRACT_PAYLOAD = {
    "schema": FIXED_BOUNDARY_CONTRACT_SCHEMA,
    "expected_values": _EXPECTED_VALUES,
    "units": _UNITS,
}
FIXED_BOUNDARY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _CONTRACT_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


class FixedBoundaryContractError(RuntimeError):
    """Raised when a result attempts to cross the fixed-boundary gate."""


def fixed_boundary_expected_values() -> dict[str, float]:
    """Return a copy of the exact authoritative values."""
    return dict(_EXPECTED_VALUES)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def evaluate_fixed_boundary(
    parameter_values: Mapping[str, Any],
    *,
    thermal_pad_conductivity_w_mk: Any,
) -> dict[str, Any]:
    """Classify a run without granting it authority.

    A mismatch is explicitly diagnostic-only and carries fail-closed promotion
    metadata.  Call :func:`attest_fixed_boundary` at every production boundary
    that must reject such a run.
    """
    if not isinstance(parameter_values, Mapping):
        raise TypeError("fixed-boundary parameters must be a mapping")

    observed: dict[str, float | None] = {}
    mismatches: list[dict[str, Any]] = []
    for name, expected in _PARAMETER_EXPECTATIONS.items():
        if name not in parameter_values:
            observed[name] = None
            mismatches.append({
                "field": name,
                "reason": "missing",
                "expected": expected,
                "observed": None,
            })
            continue
        try:
            actual = _finite_number(parameter_values[name], name)
        except ValueError as error:
            observed[name] = None
            mismatches.append({
                "field": name,
                "reason": str(error),
                "expected": expected,
                "observed": None,
            })
            continue
        observed[name] = actual
        if actual != expected:
            mismatches.append({
                "field": name,
                "reason": "not_exact",
                "expected": expected,
                "observed": actual,
            })

    tim_name = "thermal_pad_conductivity_W_mK"
    tim_expected = _EXPECTED_VALUES[tim_name]
    try:
        tim_actual = _finite_number(
            thermal_pad_conductivity_w_mk, tim_name
        )
    except ValueError as error:
        observed[tim_name] = None
        mismatches.append({
            "field": tim_name,
            "reason": str(error),
            "expected": tim_expected,
            "observed": None,
        })
    else:
        observed[tim_name] = tim_actual
        if tim_actual != tim_expected:
            mismatches.append({
                "field": tim_name,
                "reason": "not_exact",
                "expected": tim_expected,
                "observed": tim_actual,
            })

    attested = not mismatches
    return {
        "schema": FIXED_BOUNDARY_CONTRACT_SCHEMA,
        "contract_sha256": FIXED_BOUNDARY_CONTRACT_SHA256,
        "expected_values": fixed_boundary_expected_values(),
        "observed_values": observed,
        "units": dict(_UNITS),
        "mismatches": mismatches,
        "authoritative_fixed_boundary_attested": attested,
        "authority_class": (
            "authoritative_fixed_boundary"
            if attested else "diagnostic_override_only"
        ),
        "diagnostic_only": not attested,
        "promotion_forbidden_by_fixed_boundary": not attested,
        "canonical_dataset_mutation_forbidden_by_fixed_boundary": (
            not attested
        ),
    }


def attest_fixed_boundary(
    parameter_values: Mapping[str, Any],
    *,
    thermal_pad_conductivity_w_mk: Any,
) -> dict[str, Any]:
    """Fail closed unless every authoritative boundary value is exact."""
    evidence = evaluate_fixed_boundary(
        parameter_values,
        thermal_pad_conductivity_w_mk=(
            thermal_pad_conductivity_w_mk
        ),
    )
    if not evidence["authoritative_fixed_boundary_attested"]:
        detail = "; ".join(
            (
                f"{item['field']}: expected={item['expected']!r}, "
                f"observed={item['observed']!r}, reason={item['reason']}"
            )
            for item in evidence["mismatches"]
        )
        raise FixedBoundaryContractError(
            "non-authoritative fixed-boundary candidate rejected: "
            + detail
        )
    return evidence


def fixed_boundary_classification_metadata(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten authoritative or diagnostic evidence into durable columns."""
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("schema") != FIXED_BOUNDARY_CONTRACT_SCHEMA
        or evidence.get("contract_sha256")
        != FIXED_BOUNDARY_CONTRACT_SHA256
    ):
        raise FixedBoundaryContractError(
            "result metadata requires valid fixed-boundary evidence"
        )
    observed = evidence.get("observed_values")
    if not isinstance(observed, Mapping):
        raise FixedBoundaryContractError(
            "fixed-boundary observed values are unavailable"
        )
    recomputed = evaluate_fixed_boundary(
        observed,
        thermal_pad_conductivity_w_mk=observed.get(
            "thermal_pad_conductivity_W_mK"
        ),
    )
    evidence_mismatches = evidence.get("mismatches")
    if not isinstance(evidence_mismatches, list):
        raise FixedBoundaryContractError(
            "fixed-boundary mismatch evidence is unavailable"
        )

    def _mismatch_identity(items: list[dict[str, Any]]) -> list[tuple]:
        return [
            (
                item.get("field"),
                item.get("expected"),
                item.get("observed"),
            )
            for item in items
        ]

    if (
        _mismatch_identity(evidence_mismatches)
        != _mismatch_identity(recomputed["mismatches"])
        or evidence.get("authoritative_fixed_boundary_attested")
        is not recomputed["authoritative_fixed_boundary_attested"]
        or evidence.get("authority_class")
        != recomputed["authority_class"]
    ):
        raise FixedBoundaryContractError(
            "fixed-boundary evidence classification was altered"
        )
    attested = (
        evidence.get("authoritative_fixed_boundary_attested") is True
        and evidence.get("mismatches") == []
    )

    def _observed(name: str) -> float | None:
        value = observed.get(name)
        return None if value is None else float(value)

    return {
        "fixed_boundary_contract_schema": (
            FIXED_BOUNDARY_CONTRACT_SCHEMA
        ),
        "fixed_boundary_contract_sha256": (
            FIXED_BOUNDARY_CONTRACT_SHA256
        ),
        "fixed_boundary_authoritative_attested": int(attested),
        "fixed_boundary_authority_class": (
            "authoritative_fixed_boundary"
            if attested else "diagnostic_override_only"
        ),
        "fixed_boundary_diagnostic_only": int(not attested),
        "fixed_boundary_promotion_forbidden": int(not attested),
        "fixed_boundary_canonical_dataset_mutation_forbidden": int(
            not attested
        ),
        "fixed_boundary_fan_velocity_m_s": _observed(
            "fan_velocity"
        ),
        "fixed_boundary_core_plate_pad_t_mm": _observed(
            "core_plate_pad_t"
        ),
        "fixed_boundary_wcp_pad_t_mm": _observed("wcp_pad_t"),
        "fixed_boundary_thermal_pad_conductivity_W_mK": _observed(
            "thermal_pad_conductivity_W_mK"
        ),
        "fixed_boundary_mismatches_json": json.dumps(
            evidence.get("mismatches", []),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    }


def fixed_boundary_result_metadata(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten exact attestation into authoritative result columns."""
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("authoritative_fixed_boundary_attested") is not True
        or evidence.get("mismatches") != []
    ):
        raise FixedBoundaryContractError(
            "result metadata requires exact fixed-boundary attestation"
        )
    return fixed_boundary_classification_metadata(evidence)
