from __future__ import annotations

import json

import pytest

from tools.tier1_corrected_generation_preflight import (
    CURRENT_STAGE_SPEC,
    RESONANCE_MAXIMUM_CONSTRAINT,
    RESONANCE_MINIMUM_CONSTRAINT,
    canonical_sha256,
    half_magnetizing_resonance_band_violations,
    stage_constraint_names,
    stage_hard_constraint_contract,
    stage_spec_from_json_identity,
    stage_temperature_contract,
    validate_stage_spec,
)


def _final_stage() -> dict:
    return {
        **CURRENT_STAGE_SPEC,
        "T_limit_C": 100.0,
        "size_W_max_mm": 1_000.0,
        "size_L_max_mm": 1_000.0,
        "size_H_max_mm": 750.0,
        "resonance_min_Hz": 15_000.0,
        "resonance_max_Hz": 20_000.0,
    }


def test_final_stage_contract_seals_geometry_temperature_and_resonance_band():
    stage = validate_stage_spec(_final_stage())
    names = stage_constraint_names(stage)
    hard = stage_hard_constraint_contract(stage)
    thermal = stage_temperature_contract(stage)

    assert RESONANCE_MINIMUM_CONSTRAINT in names
    assert RESONANCE_MAXIMUM_CONSTRAINT in names
    assert hard["size_limits_mm"] == {
        "W": 1_000.0,
        "L": 1_000.0,
        "H": 750.0,
    }
    assert hard["self_resonance"]["lower_bound_inclusive_Hz"] == 15_000.0
    assert hard["self_resonance"]["upper_bound_exclusive_Hz"] == 20_000.0
    assert hard["stage_spec_sha256"] == canonical_sha256(stage)
    assert thermal["robust_upper_bound_C"] == 100.0


def test_resonance_upper_bound_is_strict_and_lower_bound_is_inclusive():
    stage = _final_stage()

    at_lower = half_magnetizing_resonance_band_violations(15_000.0, stage)
    inside = half_magnetizing_resonance_band_violations(19_999.0, stage)
    at_upper = half_magnetizing_resonance_band_violations(20_000.0, stage)

    assert at_lower[RESONANCE_MINIMUM_CONSTRAINT] == 0.0
    assert at_lower[RESONANCE_MAXIMUM_CONSTRAINT] < 0.0
    assert all(value <= 0.0 for value in inside.values())
    assert at_upper[RESONANCE_MAXIMUM_CONSTRAINT] > 0.0


def test_stage_spec_identity_and_fixed_physics_fail_closed():
    stage = validate_stage_spec(_final_stage())
    encoded = json.dumps(stage, sort_keys=True, separators=(",", ":"))
    assert stage_spec_from_json_identity(
        encoded, canonical_sha256(stage)
    ) == stage

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        stage_spec_from_json_identity(encoded, "0" * 64)

    mutated = {**stage, "primary_conductor_thickness_mm": 4.0}
    with pytest.raises(RuntimeError, match="forbids overriding"):
        validate_stage_spec(mutated)


def test_optional_resonance_lower_bound_omits_only_minimum_constraint():
    stage = {**_final_stage(), "resonance_min_Hz": None}
    names = stage_constraint_names(stage)
    assert RESONANCE_MINIMUM_CONSTRAINT not in names
    assert RESONANCE_MAXIMUM_CONSTRAINT in names
    violations = half_magnetizing_resonance_band_violations(10_000.0, stage)
    assert set(violations) == {RESONANCE_MAXIMUM_CONSTRAINT}
    assert violations[RESONANCE_MAXIMUM_CONSTRAINT] < 0.0
