"""Fail-closed scientific validity contract for MFT thermal results.

The thermal solver can return finite temperatures even when Fluent has
isolated a homogenized Rx winding block behind an unpaired interface.  Such a
run is execution evidence, not physical truth.  This module centralizes the
terminal fields that distinguish a scientifically usable thermal result from
that failure mode so search acquisition, active-learning ingestion, and truth
promotion cannot drift apart.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


THERMAL_RX_INTERFACE_CONTRACT_FIELDS = (
    "thermal_rx_block_interface_contract_version",
    "thermal_rx_main_interface_coverage_passed",
    "thermal_rx_main_unpaired_interfaces",
    "thermal_temperature_limiter_triggered",
    "thermal_temperature_limiter_max_K",
    "thermal_result_scientific_valid",
)
THERMAL_RX_INTERFACE_CONTRACT_VERSIONS = frozenset(
    {"thermal-rx-block-interface-coverage-v1"}
)
THERMAL_INVALID_LOG_MARKERS = (
    "temperature limited to 5.000000e+03",
    "set unpaired interface zone",
)

# Fluent's emergency limiter is 5000 K.  Keep a small margin so formatting or
# rounding cannot turn a limiter-clamped result into acquisition truth.
THERMAL_LIMITER_FAIL_CLOSE_K = 4_990.0
THERMAL_LIMITER_FAIL_CLOSE_C = THERMAL_LIMITER_FAIL_CLOSE_K - 273.15


def _optional_finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def thermal_scientific_truth_contract(
    result: Mapping[str, Any],
    *,
    observed_temperatures_C: Mapping[str, float] | None = None,
    task_log_text: str = "",
) -> dict[str, Any]:
    """Return a deterministic fail-closed thermal scientific classification."""

    reasons: list[str] = []
    missing = [
        name
        for name in THERMAL_RX_INTERFACE_CONTRACT_FIELDS
        if name not in result
    ]
    reasons.extend(
        f"scientific_contract_field_missing:{name}" for name in missing
    )

    version = result.get("thermal_rx_block_interface_contract_version")
    if (
        version is not None
        and version not in THERMAL_RX_INTERFACE_CONTRACT_VERSIONS
    ):
        reasons.append(
            "scientific_contract_invalid:"
            "thermal_rx_block_interface_contract_version"
        )

    coverage = result.get("thermal_rx_main_interface_coverage_passed")
    if coverage is not None and coverage is not True:
        reasons.append(
            "scientific_contract_invalid:"
            "thermal_rx_main_interface_coverage_passed"
        )

    unpaired = result.get("thermal_rx_main_unpaired_interfaces")
    if unpaired is not None:
        if not isinstance(unpaired, list):
            reasons.append(
                "scientific_contract_invalid:"
                "thermal_rx_main_unpaired_interfaces_type"
            )
        elif unpaired:
            reasons.append(
                "scientific_invalid:"
                "thermal_rx_main_unpaired_interfaces_present"
            )

    limiter_triggered = result.get("thermal_temperature_limiter_triggered")
    if limiter_triggered is not None and limiter_triggered is not False:
        reasons.append(
            "scientific_invalid:thermal_temperature_limiter_triggered"
        )

    limiter_max_K = _optional_finite(
        result.get("thermal_temperature_limiter_max_K")
    )
    if (
        "thermal_temperature_limiter_max_K" in result
        and limiter_max_K is None
    ):
        reasons.append(
            "scientific_contract_invalid:thermal_temperature_limiter_max_K"
        )
    elif (
        limiter_max_K is not None
        and limiter_max_K >= THERMAL_LIMITER_FAIL_CLOSE_K
    ):
        reasons.append(
            "scientific_invalid:thermal_temperature_limiter_near_5000K"
        )

    result_scientific_valid = result.get("thermal_result_scientific_valid")
    if (
        result_scientific_valid is not None
        and result_scientific_valid is not True
    ):
        reasons.append(
            "scientific_invalid:thermal_result_scientific_valid_false"
        )

    observed = observed_temperatures_C or {}
    if any(
        value >= THERMAL_LIMITER_FAIL_CLOSE_C
        for value in observed.values()
    ):
        reasons.append(
            "scientific_invalid:observed_temperature_near_5000K_limiter"
        )

    normalized_log = task_log_text.lower()
    for marker in THERMAL_INVALID_LOG_MARKERS:
        if marker in normalized_log:
            reasons.append(
                "scientific_invalid:task_log_marker:"
                + marker.replace(" ", "_")
            )

    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "contract_fields": {
            name: copy.deepcopy(result.get(name))
            for name in THERMAL_RX_INTERFACE_CONTRACT_FIELDS
        },
        "accepted_contract_versions": sorted(
            THERMAL_RX_INTERFACE_CONTRACT_VERSIONS
        ),
        "log_markers_scanned": list(THERMAL_INVALID_LOG_MARKERS),
    }
