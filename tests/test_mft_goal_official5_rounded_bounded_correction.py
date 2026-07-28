from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from module import input_parameter_260706 as input_parameter
from tools import (
    mft_goal_official5_rounded_bounded_correction_prepare as correction,
)


NOW = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)
ANCHOR_UNIT = (
    0.3750093752343808,
    0.15292715370004006,
    0.48125,
    0.4666666666666667,
    0.6957142857142857,
    0.414958384779362,
    0.4602635026118509,
    0.5000166672222407,
    0.255139453053436,
    0.0018806172878270441,
    0.3333333333333333,
    0.34075906178121596,
    0.3333333333333333,
    0.33499393810315864,
    0.34444444444444444,
    0.3333333333333333,
    0.370696711999252,
    0.4901916235673672,
    0.45297812062970055,
    0.014983351831298557,
    0.9148936170212765,
    0.8500000000000001,
    0.8588201390019485,
    0.0002824867940323661,
    0.8245937962206423,
)
EXACT_GEOMETRY = {
    "N1_main": 6,
    "N1_side": 0,
    "N2_main": 37,
    "N2_side": 23,
    "l1": 68,
    "l2": 357.5,
    "h1": 571,
    "w1": 476,
    "n_core_group": 5,
    "core_plate_t": 10.0,
    "wcp_t": 15.1,
    "wcp_len_x": 349.6,
    "cw1": 1.13,
    "gap1": 4.6,
    "cw2": 1.1,
    "gap2": 1.745,
    "nwh1": 520.6,
    "nwh2": 285.6,
    "cc_w2c_space_x": 40.0,
    "cc_w2c_space_y": 40.3,
    "w2c_w1c_space_x": 40.0,
    "w2c_w1c_space_y": 40.1,
    "w1c_w2s_space_x": 32.4,
    "w2s_w1s_space_x": 0.0,
    "w1s_w2s_space_y": 0.0,
    "w1s_cs_space_x": 40.0,
    "cs_w1s_space_y": 41.7,
}


def _base_params() -> dict[str, Any]:
    params = input_parameter.get_drawing_default_params()
    params.update(EXACT_GEOMETRY)
    params.update(
        {
            "core_lamination_factor": 0.85,
            "round_corner": 1,
            "corner_radius": 10.0,
            "corner_segments": 4,
            "full_model": 0,
            "thermal_symmetry": "eighth",
            "fan_config": "dual",
            "fan_velocity": 1.5,
            "k_ins": 0.2,
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
        }
    )
    return params


def _profile() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[1]
        / "regression_260707"
        / "verify"
        / "profiles"
        / "goal_diagnostic_standard_rounded_final.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement() -> dict[str, Any]:
    return {
        "actual_volume_L": 812.0,
        "actual_total_loss_W": 5_470.0,
        "actual_dimensions_mm": {
            "W": 1_194.38,
            "L": 961.4,
            "H": 707.0,
        },
        "actual_resonance_Hz": 14_950.0,
        "actual_temperature_targets_C": {},
        "actual_winding_max_C": 99.7,
        "actual_core_max_C": 119.5,
        "actual_physical_Llt_uH": 27.5,
    }


def _predicted() -> dict[str, Any]:
    return {
        "volume_L": 811.8317909240001,
        "total_loss_W": 5_461.276475877185,
        "dimensions_mm": {
            "W": 1_194.38,
            "L": 961.4,
            "H": 707.0,
        },
        "resonance_Hz": 15_078.457610818337,
        "winding_max_C": 99.63987976713237,
        "core_max_C": 119.39600472639508,
    }


def test_exact_anchor_replay_and_legacy_path_rejection() -> None:
    params = _base_params()
    problem = correction.exact_goal_repair_problem(
        core_lamination_factor=0.85
    )
    assert problem.geometry_constraint_profile is None
    assert problem.geometry_constraint_profile_sha256 is None
    assert problem.launch_eligible is False
    assert problem.legacy_repair_compatibility == (
        correction.LEGACY_REPAIR_COMPATIBILITY
    )
    assert problem.legacy_repair_compatibility_sha256 == (
        correction.LEGACY_REPAIR_COMPATIBILITY_SHA256
    )

    row, evidence = correction.attest_anchor_replay(
        problem=problem,
        anchor_coordinate=ANCHOR_UNIT,
        base_params=params,
    )
    rejection = correction.legacy_path_rejection(
        anchor_coordinate=ANCHOR_UNIT,
        base_params=params,
    )

    assert evidence["coordinate_fixed_point"] is True
    assert evidence["geometry_drift"] == {}
    assert row["cw1"] == pytest.approx(1.13)
    assert row["cw2"] == pytest.approx(1.1)
    assert row["wcp_len_x"] == pytest.approx(349.6)
    assert rejection["legacy_path_allowed"] is False
    assert rejection["legacy_path_reproduces_sealed_anchor"] is False
    assert set(rejection["observed_anchor_drift"]) == {
        "cw1",
        "cw2",
        "wcp_len_x",
    }
    assert rejection["observed_anchor_drift"]["cw1"] == {
        "sealed_candidate": 1.13,
        "legacy_raw_decode": 4.55,
    }


def test_legacy_repair_compatibility_fails_closed_on_profile_drift() -> None:
    problem = correction.exact_goal_repair_problem(
        core_lamination_factor=0.85
    )
    problem.geometry_constraint_profile = copy.deepcopy(
        correction.goal_repair.GOAL_OFFICIAL_GEOMETRY_CONSTRAINT_PROFILE
    )

    with pytest.raises(
        correction.ContractError,
        match="legacy repair compatibility",
    ):
        correction.attest_anchor_replay(
            problem=problem,
            anchor_coordinate=ANCHOR_UNIT,
            base_params=_base_params(),
        )


def test_a_b_c_d_templates_use_exact_goal_repair_and_frozen_contract() -> None:
    problem = correction.exact_goal_repair_problem(
        core_lamination_factor=0.85
    )

    templates = correction.generate_exact_templates(
        problem=problem,
        anchor_coordinate=ANCHOR_UNIT,
        base_params=_base_params(),
        profile=_profile(),
    )

    assert [item["template"] for item in templates] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert len({item["geometry_sha256"] for item in templates}) == 4
    assert all(
        0.0 < item["distance_linf"]
        <= correction.MAX_TRUST_RADIUS + 1.0e-12
        for item in templates
    )
    assert all(item["exact_goal_repair_used"] is True for item in templates)
    assert all(
        item["legacy_local_trust_generator_used"] is False
        for item in templates
    )
    assert all(
        item["frozen_contract"]["fan_velocity"] == 1.5
        and item["frozen_contract"]["k_ins"] == 0.2
        and item["frozen_contract"]["round_corner"] == 1
        and item["frozen_contract"]["corner_radius"] == 10.0
        and item["frozen_contract"]["corner_segments"] == 4
        and item["frozen_contract"]["full_model"] == 0
        and item["frozen_contract"]["thermal_symmetry"] == "eighth"
        for item in templates
    )
    decoded = {
        item["template"]: item["decoded_params"] for item in templates
    }
    assert decoded["A"]["cw1"] == pytest.approx(1.0)
    assert decoded["A"]["w1"] == pytest.approx(506.0)
    assert decoded["B"]["cw1"] == pytest.approx(1.13)
    assert decoded["B"]["w1"] == pytest.approx(506.0)
    assert decoded["C"]["cw1"] == pytest.approx(1.0)
    assert decoded["C"]["nwh2"] == pytest.approx(290.1)
    assert decoded["D"]["l1"] == pytest.approx(69.0)
    assert decoded["D"]["w1"] == pytest.approx(488.0)
    assert all(
        item["decoded_params"]["gap2"] == pytest.approx(1.745)
        for item in templates
    )


def test_narrow_miss_gate_accepts_only_small_local_misses() -> None:
    accepted = correction.narrow_miss_gate(
        measurement=_measurement(),
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert accepted["prepare_allowed"] is True
    assert accepted["failed_requested_constraints"] == ["resonance"]
    assert accepted["submission_allowed"] is False
    assert accepted["scheduler_post_calls"] == 0

    broad = _measurement()
    broad["actual_resonance_Hz"] = 14_899.0
    broad_gate = correction.narrow_miss_gate(
        measurement=broad,
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert broad_gate["prepare_allowed"] is False
    assert "resonance_miss_exceeds_100_Hz" in broad_gate["reasons"]

    dimension = _measurement()
    dimension["actual_dimensions_mm"]["W"] = 1_200.1
    dimension_gate = correction.narrow_miss_gate(
        measurement=dimension,
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert dimension_gate["prepare_allowed"] is False
    assert "dimension_miss_is_not_fea_error" in dimension_gate["reasons"]

    passing = _measurement()
    passing.update(
        {
            "actual_resonance_Hz": 15_001.0,
            "actual_winding_max_C": 99.9,
            "actual_core_max_C": 119.9,
        }
    )
    passing_gate = correction.narrow_miss_gate(
        measurement=passing,
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert passing_gate["prepare_allowed"] is False
    assert (
        "source_already_passes_requested_hard_constraints"
        in passing_gate["reasons"]
    )


def test_composite_miss_uses_stricter_limits() -> None:
    within = _measurement()
    within.update(
        {
            "actual_resonance_Hz": 14_940.0,
            "actual_winding_max_C": 100.20,
            "actual_core_max_C": 120.10,
        }
    )
    allowed = correction.narrow_miss_gate(
        measurement=within,
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert allowed["prepare_allowed"] is True
    assert len(allowed["failed_requested_constraints"]) == 3

    outside = copy.deepcopy(within)
    outside["actual_core_max_C"] = 120.21
    rejected = correction.narrow_miss_gate(
        measurement=outside,
        predicted=_predicted(),
        observed_at=NOW,
    )
    assert rejected["prepare_allowed"] is False
    assert (
        "composite_miss_exceeds_strict_local_limits"
        in rejected["reasons"]
    )


def test_scoring_caps_ranked_same_design_plan_at_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    templates = []
    for index, name in enumerate(("A", "B", "C", "D"), start=1):
        coordinate = np.asarray(ANCHOR_UNIT, dtype=float)
        coordinate[correction.ACTIVE_UNIT_INDICES[0]] += 0.005 * index
        templates.append(
            {
                "template": name,
                "unit_coordinate": coordinate.tolist(),
                "geometry_sha256": f"{index:064x}",
            }
        )
    predictions = iter(
        [
            {
                "volume_L": 812.0,
                "total_loss_W": 5_450.0,
                "resonance_Hz": resonance,
                "winding_max_C": 99.0,
                "core_max_C": 119.0,
                "dimensions_mm": {
                    "W": 1_190.0,
                    "L": 950.0,
                    "H": 700.0,
                },
                "temperature_targets_C": {},
            }
            for resonance in (15_060.0, 15_050.0, 15_040.0, 15_030.0)
        ]
    )
    monkeypatch.setattr(
        correction,
        "_prediction_metrics",
        lambda _values, _names: next(predictions),
    )
    output_names = (
        "objective_volume_L",
        "objective_total_loss_W",
        "normalized_G:half_magnetizing_resonance_minimum",
        "normalized_G:exterior_width_limit",
        "normalized_G:exterior_length_limit",
        "normalized_G:exterior_height_limit",
    )
    gate = correction.narrow_miss_gate(
        measurement=_measurement(),
        predicted=_predicted(),
        observed_at=NOW,
    )

    ranked = correction.score_templates(
        templates=templates,
        anchor_coordinate=ANCHOR_UNIT,
        measurement=_measurement(),
        gate=gate,
        coefficients=np.zeros(
            (len(correction.ACTIVE_UNIT_INDICES), len(output_names))
        ),
        anchor_output=np.zeros(len(output_names)),
        output_names=output_names,
    )

    assert [item["template"] for item in ranked] == ["A", "B", "C"]
    assert [item["correction_rank"] for item in ranked] == [1, 2, 3]
    assert all(item["guard_pass"].values() for item in ranked)


def test_module_has_prepare_only_cli_and_rejects_unsealed_authority(
    tmp_path: Path,
) -> None:
    assert not hasattr(correction, "submit")
    source_text = Path(correction.__file__).read_text(encoding="utf-8")
    assert ".post(" not in source_text.casefold()
    args = correction._parser().parse_args(  # noqa: SLF001
        [
            "prepare",
            "--standard-plan",
            "plan.json",
            "--standard-final",
            "final.json",
            "--standard-collection",
            "collection",
        ]
    )
    assert args.command == "prepare"

    authority = correction.sealed(
        {
            "schema_version": correction.SOURCE_SCHEMA,
            "source_task_id": correction.SOURCE_TASK_ID,
        }
    )
    authority["source_task_id"] = 1

    def unauthenticated(**_kwargs: Any) -> dict[str, Any]:
        return {
            "authority": authority,
            "params": {},
            "profile": {},
            "selected": {},
            "result": {},
        }

    output = tmp_path / "correction"
    with pytest.raises(correction.ContractError, match="seal"):
        correction.prepare(
            standard_plan_path=tmp_path / "plan.json",
            standard_final_path=tmp_path / "final.json",
            standard_collection_root=tmp_path / "collection",
            output=output,
            observed_at=NOW,
            collection_authenticator=unauthenticated,
        )
    assert not output.exists()
