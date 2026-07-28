from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from module.input_parameter_260706 import decode_unit_sample, unit_to_dims
from tools import mft_goal_local_trust_acquisition as local


ANCHOR_SHA256 = f"{12:064x}"
OTHER_SHA256 = f"{6:064x}"

# A deliberately roomy, decoder-valid interior point.  The discrete topology
# coordinates are fixed by the local acquisition workflow.
ANCHOR_UNIT = [
    0.375,
    0.5,
    0.48,
    0.30,
    0.90,
    0.65,
    0.50,
    0.50,
    0.50,
    0.50,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.35,
    0.50,
    0.30,
    0.30,
    0.50,
    0.50,
    0.50,
]

FIXED_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
}


def _measurement() -> dict[str, Any]:
    return {
        "terminal_authenticated": True,
        "artifact_authenticated": True,
        "solver_valid": True,
        "physics_valid": True,
        "candidate_physics_sha256": ANCHOR_SHA256,
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "predicted": {
            "objectives": {
                "exterior_volume_L": 803.2,
                "total_loss_W": 5319.5,
            },
            "metrics": {
                "resonance_frequency_kHz": 15.40,
                "winding_max_C": 99.5,
                "core_max_C": 117.0,
            },
            "normalized_constraints": {
                "g_resonance": -0.080,
                "g_winding_temperature": -0.025,
                "g_core_temperature": -0.150,
                "g_llt_robust": 0.40,
            },
        },
        "actual": {
            "objectives": {
                "exterior_volume_L": 807.0,
                "total_loss_W": 5410.0,
            },
            "metrics": {
                "resonance_frequency_kHz": 15.20,
                "winding_max_C": 100.2,
                "core_max_C": 118.0,
            },
            "normalized_constraints": {
                "g_resonance": -0.040,
                "g_winding_temperature": 0.010,
                "g_core_temperature": -0.100,
                "g_llt_robust": 0.45,
            },
        },
    }


def test_authenticated_small_residual_enters_local_correction() -> None:
    decision = local.classify_authenticated_residual(
        _measurement(),
        expected_candidate_sha256=ANCHOR_SHA256,
    )

    assert decision.classification is local.ResidualClass.SMALL_LOCAL_CORRECTION
    assert decision.eligible_for_local_acquisition is True
    assert decision.max_scaled_residual <= 1.0
    assert decision.reasons == ()


def test_authenticated_large_residual_stops_for_model_mismatch() -> None:
    measurement = _measurement()
    measurement["actual"]["metrics"]["winding_max_C"] = 112.0
    measurement["actual"]["objectives"]["total_loss_W"] = 6900.0

    decision = local.classify_authenticated_residual(
        measurement,
        expected_candidate_sha256=ANCHOR_SHA256,
    )

    assert decision.classification is local.ResidualClass.LARGE_MODEL_MISMATCH
    assert decision.eligible_for_local_acquisition is False
    assert decision.max_scaled_residual > 1.0
    assert decision.reasons


def test_resonance_residual_preserves_hz_constraint_units(
    tmp_path: Path,
) -> None:
    """Regression: physical_G is Hz even though display metrics are kHz."""

    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"Llt": 13.75}), encoding="utf-8")
    physical_residual_hz = -78.45761081833734
    anchor = local.AnchorEvidence(
        selection_order=5,
        role="backup",
        candidate_physics_sha256=ANCHOR_SHA256,
        row={
            "objective_volume_L": 811.8317909240001,
            "objective_total_loss_W": 5461.276475877185,
        },
        coordinate_unit=tuple(ANCHOR_UNIT),
        physical_constraints={
            "half_magnetizing_resonance_minimum": physical_residual_hz,
            "temperature_robust_limit:T_max_Tx": -0.3601202328676294,
            "temperature_robust_limit:T_max_Rx_main": -1.5,
            "temperature_robust_limit:T_max_core": -0.603995273604923,
        },
        normalized_constraints={
            "half_magnetizing_resonance_minimum": (
                physical_residual_hz
                / local.RESONANCE_CONSTRAINT_SCALE_HZ
            ),
            "temperature_robust_limit:T_max_Tx": -0.03601202328676294,
            "temperature_robust_limit:T_max_Rx_main": -0.15,
            "temperature_robust_limit:T_max_core": -0.0603995273604923,
            "Llt_robust_band": 0.0,
        },
    )
    actual_resonance_hz = 15_075.0
    observation = {
        "actual_resonance_Hz": actual_resonance_hz,
        "actual_temperature_targets": {
            "T_max_Tx": {"actual_C": 99.0},
            "T_max_Rx_main": {"actual_C": 119.0},
            "T_max_core": {"actual_C": 119.0},
        },
        "actual_dimensions_mm": {
            "W": 1194.38,
            "L": 961.4,
            "H": 707.0,
        },
        "source_result_json": local._file_record(result_path),  # noqa: SLF001
        "actual_volume_L": 811.8317909240001,
        "actual_total_loss_W": 5461.276475877185,
        "actual_winding_max_C": 119.0,
        "actual_core_max_C": 119.0,
    }

    measurement = local._observation_to_measurement(  # noqa: SLF001
        observation, anchor
    )

    assert measurement["predicted"]["metrics"][
        "resonance_frequency_kHz"
    ] == pytest.approx(15.078457610818337)
    assert measurement["actual"]["metrics"][
        "resonance_frequency_kHz"
    ] == pytest.approx(15.075)
    assert measurement["predicted"]["metrics"]["winding_max_C"] == pytest.approx(
        118.5
    )
    assert measurement["actual"]["metrics"]["winding_max_C"] == pytest.approx(
        119.0
    )
    assert measurement["actual"]["normalized_constraints"][
        "half_magnetizing_resonance_minimum"
    ] == pytest.approx(
        (15_000.0 - actual_resonance_hz)
        / local.RESONANCE_CONSTRAINT_SCALE_HZ
    )
    assert measurement["predicted"]["metrics"]["winding_max_C"] == pytest.approx(
        118.5
    )
    assert measurement["actual"]["metrics"]["winding_max_C"] == pytest.approx(
        119.0
    )


@pytest.mark.parametrize(
    ("mutator", "reason_fragment"),
    [
        (
            lambda value: value.__setitem__("terminal_authenticated", False),
            "terminal",
        ),
        (
            lambda value: value.__setitem__("artifact_authenticated", False),
            "artifact",
        ),
        (
            lambda value: value.__setitem__("solver_valid", False),
            "solver",
        ),
        (
            lambda value: value.__setitem__("physics_valid", False),
            "physics",
        ),
        (
            lambda value: value.__setitem__(
                "candidate_physics_sha256", OTHER_SHA256
            ),
            "candidate",
        ),
        (
            lambda value: value["fixed_boundary"].__setitem__(
                "fan_velocity", 1.6
            ),
            "fixed",
        ),
        (
            lambda value: value["actual"]["metrics"].__setitem__(
                "core_max_C", math.nan
            ),
            "finite",
        ),
    ],
)
def test_unauthenticated_or_invalid_measurement_fails_closed(
    mutator,
    reason_fragment: str,
) -> None:
    measurement = _measurement()
    mutator(measurement)

    decision = local.classify_authenticated_residual(
        measurement,
        expected_candidate_sha256=ANCHOR_SHA256,
    )

    assert (
        decision.classification
        is local.ResidualClass.SOLVER_OR_PHYSICS_INVALID
    )
    assert decision.eligible_for_local_acquisition is False
    assert any(
        reason_fragment in reason.lower() for reason in decision.reasons
    )


@pytest.mark.parametrize(
    ("selection_order", "anchor_role"),
    [(12, "primary"), (6, "primary"), (1, "backup"), (5, "backup")],
)
def test_local_neighbors_are_deterministic_bounded_and_topology_preserving(
    selection_order: int,
    anchor_role: str,
) -> None:
    kwargs = {
        "selection_order": selection_order,
        "trust_radius": 0.02,
        "max_pool": 24,
    }
    first = local.generate_local_neighbors(ANCHOR_UNIT, **kwargs)
    second = local.generate_local_neighbors(ANCHOR_UNIT, **kwargs)

    assert first == second
    assert 0 < len(first) <= 24
    assert len({item["geometry_sha256"] for item in first}) == len(first)

    anchor_decoded = decode_unit_sample(
        unit_to_dims(ANCHOR_UNIT),
        allow_space_shrink=False,
    )
    for item in first:
        assert item["anchor_role"] == anchor_role
        coordinate = np.asarray(item["unit_coordinate"], dtype=float)
        assert coordinate.shape == (len(ANCHOR_UNIT),)
        assert np.isfinite(coordinate).all()
        assert ((0.0 <= coordinate) & (coordinate <= 1.0)).all()
        distance = float(np.max(np.abs(coordinate - ANCHOR_UNIT)))
        assert 0.0 < distance <= 0.02 + 1e-12
        assert item["distance_linf"] == pytest.approx(distance)
        assert len(item["geometry_sha256"]) == 64
        assert item["geometry_sha256"] == item[
            "geometry_sha256"
        ].lower()

        decoded = item["decoded_params"]
        direct = decode_unit_sample(
            unit_to_dims(coordinate),
            allow_space_shrink=False,
        )
        for geometry_key in (
            "l1",
            "l2",
            "h1",
            "w1",
            "cw1",
            "cw2",
            "gap1",
            "gap2",
            "nwh1",
            "nwh2",
            "wcp_t",
            "core_plate_t",
            "wcp_len_x",
        ):
            assert decoded[geometry_key] == direct[geometry_key]
        assert (
            decoded["N1_main"],
            decoded["N1_side"],
            decoded["N2_main"] + decoded["N2_side"],
            decoded["n_core_group"],
        ) == (
            anchor_decoded["N1_main"],
            anchor_decoded["N1_side"],
            anchor_decoded["N2_main"] + anchor_decoded["N2_side"],
            anchor_decoded["n_core_group"],
        )
        assert decoded["fan_config"] == "dual"
        assert decoded["fan_velocity"] == 1.5
        assert decoded["k_ins"] == 0.2
        assert decoded["core_plate_pad_t"] == 2.0
        assert decoded["wcp_pad_t"] == 2.0
        assert decoded["full_model"] == 0
        assert decoded["thermal_symmetry"] == "eighth"


def test_neighbor_generation_rejects_broad_or_wrong_anchor_search() -> None:
    with pytest.raises(local.LocalTrustContractError, match="#12|#6|#1|#5"):
        local.generate_local_neighbors(
            ANCHOR_UNIT,
            selection_order=7,
            trust_radius=0.02,
            max_pool=24,
        )
    with pytest.raises(local.LocalTrustContractError, match="trust"):
        local.generate_local_neighbors(
            ANCHOR_UNIT,
            selection_order=12,
            trust_radius=0.051,
            max_pool=24,
        )
    with pytest.raises(local.LocalTrustContractError, match="pool"):
        local.generate_local_neighbors(
            ANCHOR_UNIT,
            selection_order=12,
            trust_radius=0.02,
            max_pool=65,
        )


def _scored_neighbors(count: int = 6) -> list[dict[str, Any]]:
    neighbors = local.generate_local_neighbors(
        ANCHOR_UNIT,
        selection_order=12,
        trust_radius=0.02,
        max_pool=count,
    )
    improvements = [0.08, 0.01, 0.06, 0.12, 0.04, -0.10]
    scores = [4.0, 9.0, 3.5, 3.0, 2.0, 20.0]
    return [
        {
            **item,
            "expected_constraint_improvement": improvements[index],
            "pareto_value": 1.0 / (index + 1),
            "acquisition_score": scores[index],
        }
        for index, item in enumerate(neighbors)
    ]


def test_finite_batch_filters_non_improvers_and_caps_parallel_fea() -> None:
    scored = _scored_neighbors()
    selected = local.select_finite_batch(
        scored,
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )

    assert selected.should_stop is False
    assert selected.stop_reason is None
    assert 0 < len(selected.selected) <= 3
    assert all(
        item["expected_constraint_improvement"] >= 0.02
        for item in selected.selected
    )
    assert [item["acquisition_score"] for item in selected.selected] == [
        4.0,
        3.5,
        3.0,
    ]


def test_finite_stop_criteria_prevent_endless_invalid_and_jump() -> None:
    scored = _scored_neighbors()
    exhausted = local.select_finite_batch(
        scored,
        current_round=2,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )
    no_improvement = local.select_finite_batch(
        [
            {
                **item,
                "expected_constraint_improvement": 0.0,
            }
            for item in scored
        ],
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )

    assert exhausted.should_stop is True
    assert exhausted.selected == ()
    assert exhausted.stop_reason == "max_correction_rounds_reached"
    assert no_improvement.should_stop is True
    assert no_improvement.selected == ()
    assert no_improvement.stop_reason == "no_expected_constraint_improvement"


def test_finite_batch_rejects_nonfinite_acquisition_evidence() -> None:
    scored = _scored_neighbors()
    scored[0]["pareto_value"] = math.nan

    with pytest.raises(local.LocalTrustContractError, match="finite"):
        local.select_finite_batch(
            scored,
            current_round=0,
            max_rounds=2,
            max_batch=3,
            min_expected_constraint_improvement=0.02,
        )


def test_exact_global_nds_hook_preserves_ties_and_all_fronts() -> None:
    objectives = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [5.0, 5.0],
            [1.0, 4.0],
        ]
    )

    ranks = local.exact_global_nds_ranks(objectives)

    assert np.array_equal(ranks, np.array([0, 0, 0, 0, 1, 1, 2, 0]))


def test_existing_passing_symmetric_result_stops_local_acquisition() -> None:
    anchor_table = [
        {
            "priority_index": 0,
            "selection_order": 12,
            "candidate_physics_sha256": ANCHOR_SHA256,
            "measurement_status": "authenticated_symmetric_standard",
            "hard_constraints_passed": True,
            "minimum_normalized_actual_margin": 0.01,
            "actual_total_loss_W": 5400.0,
            "actual_volume_L": 810.0,
            "task_id": 96331,
        },
        {
            "priority_index": 1,
            "selection_order": 6,
            "candidate_physics_sha256": OTHER_SHA256,
            "measurement_status": "authenticated_symmetric_standard",
            "hard_constraints_passed": True,
            "minimum_normalized_actual_margin": 0.02,
            "actual_total_loss_W": 5500.0,
            "actual_volume_L": 820.0,
            "task_id": 96340,
        },
    ]

    selected = local.select_current_symmetric_result(anchor_table)

    assert selected is not None
    assert selected["selection_order"] == 6
    assert selected["task_id"] == 96340
    assert selected["new_local_neighbors_required"] is False
    assert selected["status"] == "selected_existing_symmetric_result"


def test_actual_margin_tie_breaks_by_loss_then_volume() -> None:
    table = [
        {
            "priority_index": 0,
            "selection_order": 12,
            "candidate_physics_sha256": ANCHOR_SHA256,
            "measurement_status": "authenticated_symmetric_standard",
            "hard_constraints_passed": True,
            "minimum_normalized_actual_margin": 0.01,
            "actual_total_loss_W": 5400.0,
            "actual_volume_L": 810.0,
            "task_id": 96331,
        },
        {
            "priority_index": 1,
            "selection_order": 6,
            "candidate_physics_sha256": OTHER_SHA256,
            "measurement_status": "authenticated_symmetric_standard",
            "hard_constraints_passed": True,
            "minimum_normalized_actual_margin": 0.01,
            "actual_total_loss_W": 5350.0,
            "actual_volume_L": 830.0,
            "task_id": 96340,
        },
    ]

    selected = local.select_current_symmetric_result(table)

    assert selected is not None
    assert selected["selection_order"] == 6
    assert "actual_total_loss_W_asc" in selected["ranking"]


def test_sealed_manifest_is_symmetric_prepare_only_and_full_final_only() -> None:
    residual = local.classify_authenticated_residual(
        _measurement(),
        expected_candidate_sha256=ANCHOR_SHA256,
    )
    batch = local.select_finite_batch(
        _scored_neighbors(),
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )

    manifest = local.build_prepare_only_manifest(
        anchor_selection_order=12,
        anchor_candidate_sha256=ANCHOR_SHA256,
        residual_decision=residual,
        batch=batch,
        trust_radius=0.02,
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )
    validated = local.validate_seal(manifest, local.MANIFEST_SCHEMA)

    assert validated["prepare_only"] is True
    assert validated["stage"] == "symmetric_standard"
    assert validated["symmetric_model_primary"] is True
    assert validated["scheduler_post_calls"] == 0
    assert validated["scheduler_submission_performed"] is False
    assert validated["submission_capability_present"] is False
    assert validated["scheduler_methods_used"] == []
    assert validated["candidate_count"] == len(batch.selected) <= 3
    assert validated["full_model_policy"] == "final_one_only"
    assert validated["automatic_full_per_candidate"] is False
    assert validated["automatic_candidate_continuation"] is False
    assert validated["automatic_full_trigger"] is False
    assert validated["explicit_final_full_cap"] == 1
    assert validated["existing_full_reference_task_id"] == 96326
    assert validated["fixed_boundary"] == FIXED_BOUNDARY
    assert validated["finite_stop_criteria"] == {
        "current_round": 0,
        "max_correction_rounds": 2,
        "max_parallel_fea_batch": 3,
        "min_expected_constraint_improvement": 0.02,
    }
    assert validated["exact_global_nds_reevaluation_hook"] == {
        "callable": (
            "tools.mft_goal_global_pareto.nondominated_ranks_2d"
        ),
        "required_after_authenticated_measurements": True,
        "status": "pending",
    }
    assert all(
        item["decoded_params"]["full_model"] == 0
        for item in validated["candidates"]
    )
    assert not hasattr(local, "submit")


def test_manifest_seal_rejects_tampering() -> None:
    residual = local.classify_authenticated_residual(
        _measurement(),
        expected_candidate_sha256=ANCHOR_SHA256,
    )
    batch = local.select_finite_batch(
        _scored_neighbors(),
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )
    manifest = local.build_prepare_only_manifest(
        anchor_selection_order=12,
        anchor_candidate_sha256=ANCHOR_SHA256,
        residual_decision=residual,
        batch=batch,
        trust_radius=0.02,
        current_round=0,
        max_rounds=2,
        max_batch=3,
        min_expected_constraint_improvement=0.02,
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["scheduler_post_calls"] = 1

    with pytest.raises(local.LocalTrustContractError, match="SHA256|seal"):
        local.validate_seal(tampered, local.MANIFEST_SCHEMA)
