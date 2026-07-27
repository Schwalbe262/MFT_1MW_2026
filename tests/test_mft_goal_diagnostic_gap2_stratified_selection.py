from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from module.input_parameter_260706 import (
    create_input_parameter,
    get_drawing_default_params,
)
from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_diagnostic_gap2_stratified_selection as selection


def _diversity_candidate(
    token: str,
    *,
    physics_ucb: float,
    cw2: float,
    gap2: float,
    spread: float,
) -> dict:
    features = {
        name: spread for name in selection.DIVERSITY_FEATURES
    }
    features.update({"cw2": cw2, "gap2": gap2})
    return {
        "physical_geometry_sha256": token * 64,
        "physics_delta_prediction": {
            "physics_delta_Crx_q90_ucb_F": physics_ucb,
            "physics_delta_extrapolation_distance": 0.0,
        },
        "diversity_features": features,
        # A raw-like diagnostic must have no influence on selection.
        "diagnostic_raw_same_metric_C_rx_rx_F": 1e99,
    }


def _scientific_collection(task_id: int, seed: int) -> dict:
    identities = {
        name: character * 64
        for name, character in zip(selection.IDENTITY_COLUMNS, "abcde")
    }
    record = {
        "payload_sha256": f"{task_id:064x}",
        "identities": identities,
        "objective_columns": list(selection.OBJECTIVE_COLUMNS),
        "physical_constraint_columns": [
            selection.CORRECTED_CAP_PHYSICAL_COLUMN
        ],
        "normalized_constraint_columns": [
            selection.CORRECTED_CAP_NORMALIZED_COLUMN
        ],
    }
    return {
        "task_id": task_id,
        "seed": seed,
        "record": record,
        "record_path": Path(f"task-{task_id}/collection_record.json"),
        "record_file_sha256": "f" * 64,
        "profile_sha256": "1" * 64,
        "geometry_constraint_profile_sha256": "2" * 64,
        "physical_constraint_columns": record["physical_constraint_columns"],
        "normalized_constraint_columns": record[
            "normalized_constraint_columns"
        ],
        "capacitance_authority": {
            "schema_version": selection.CAP_AUTHORITY_SCHEMA,
        },
    }


def _fallback_candidate(
    token: str,
    *,
    stratum: str,
    gap2: float,
    violation: float,
    spread: float,
    f_lcb_hz: float = 12_000.0,
) -> dict:
    candidate = _diversity_candidate(
        token,
        physics_ucb=8.0e-10 + spread * 1.0e-12,
        cw2=0.7 + spread * 0.001,
        gap2=gap2,
        spread=spread,
    )
    candidate.update(
        {
            "gap2_stratum": stratum,
            "volume_L": 700.0 + spread,
            "total_loss_W": 7_000.0 + spread,
            "strict_latest_rerank_eligible": False,
            "strict_latest_rerank_failure_reasons": [
                "latest_hard_nonphysics_constraints_failed",
                "physics_delta_fRx_q90_lcb_below_15000Hz",
            ],
            "normalized_G": {
                "Llt_robust_band": violation,
                selection.CORRECTED_CAP_CONSTRAINT: 1.0e6,
                "half_magnetizing_resonance_minimum": 1.0e6,
            },
        }
    )
    candidate["physics_delta_prediction"].update(
        {
            "physics_delta_Crx_mean_F": 6.0e-10 + spread * 1.0e-12,
            "physics_delta_fRx_q90_lcb_Hz": f_lcb_hz,
        }
    )
    candidate["multi_violation_evidence"] = (
        selection._multi_violation_evidence(  # noqa: SLF001
            candidate,
            candidate["physics_delta_prediction"],
        )
    )
    return candidate


@pytest.mark.parametrize(
    ("gap2", "expected"),
    [
        (0.35, "low"),
        (0.899, "low"),
        (0.90, "mid"),
        (1.449, "mid"),
        (1.45, "high"),
        (2.0, "high"),
    ],
)
def test_gap_strata_have_explicit_nonoverlapping_boundaries(
    gap2: float, expected: str
) -> None:
    assert selection._gap_stratum(gap2) == expected


@pytest.mark.parametrize("gap2", [0.349999, 2.000001])
def test_gap_strata_reject_out_of_band_values(gap2: float) -> None:
    with pytest.raises(selection.StratifiedSelectionError):
        selection._gap_stratum(gap2)


def test_bounded_cap_contract_is_corrected_acquisition_only() -> None:
    record = {
        "raw_same_metric_C_rx_rx_F_UCB_gate_active": False,
        "provisional_turn_graded_C_acquisition_gate_active": True,
        "capacitance_screening_constraint_name": (
            selection.CORRECTED_CAP_CONSTRAINT
        ),
        "authenticated_turn_graded_transfer_ratio": (
            selection.CORRECTED_CAP_TRANSFER_RATIO
        ),
        "raw_two_net_C_physical_feasibility_authority": False,
        "final_turn_graded_symmetric_FEA_required": True,
        "raw_same_metric_C_rx_rx_F_front_classification": (
            "provisional_corrected_single_truth_screening_only"
        ),
    }
    authority = selection._validate_bounded_cap_contract(record)
    assert (
        authority["ratio_use"]
        == "legacy_search_acquisition_only_excluded_from_rerank"
    )
    assert (
        authority[
            "legacy_corrected_constraint_used_for_rerank_eligibility"
        ]
        is False
    )
    assert (
        authority["legacy_half_magnetizing_resonance_used_for_rerank"]
        is False
    )
    assert authority["raw_two_net_block_C_used_for_ranking"] is False
    assert authority["raw_two_net_block_C_used_for_physical_truth"] is False

    for drift in (
        {"raw_same_metric_C_rx_rx_F_UCB_gate_active": True},
        {"raw_two_net_C_physical_feasibility_authority": True},
        {"authenticated_turn_graded_transfer_ratio": 0.75},
    ):
        invalid = {**record, **drift}
        with pytest.raises(selection.StratifiedSelectionError):
            selection._validate_bounded_cap_contract(invalid)


def test_physics_ucb_anchor_and_diversity_ignore_legacy_cap_metric() -> None:
    candidates = [
        _diversity_candidate(
            "a", physics_ucb=1.0e-10, cw2=0.5, gap2=0.5, spread=0.0
        ),
        _diversity_candidate(
            "b", physics_ucb=1.1e-10, cw2=0.51, gap2=0.51, spread=0.05
        ),
        _diversity_candidate(
            "c", physics_ucb=1.2e-10, cw2=0.9, gap2=0.85, spread=1.0
        ),
    ]
    first = selection._select_diverse(copy.deepcopy(candidates), count=2)
    changed_raw = copy.deepcopy(candidates)
    for index, item in enumerate(changed_raw):
        item["diagnostic_raw_same_metric_C_rx_rx_F"] = -float(index + 1)
    second = selection._select_diverse(changed_raw, count=2)

    assert [item["physical_geometry_sha256"] for item in first] == [
        "a" * 64,
        "c" * 64,
    ]
    assert [item["physical_geometry_sha256"] for item in second] == [
        item["physical_geometry_sha256"] for item in first
    ]
    assert first[0]["physics_delta_ucb_rank_in_stratum"] == 1
    assert first[0]["selection_role"] == "minimum_physics_delta_q90_UCB"


def test_missing_stratum_is_fail_closed_without_partial_fea_candidates() -> None:
    eligible = [
        {
            **_diversity_candidate(
                character,
                physics_ucb=(index + 1) * 1e-10,
                cw2=0.5,
                gap2=0.5,
                spread=float(index),
            ),
            "gap2_stratum": "low",
        }
        for index, character in enumerate(("a", "b"))
    ]
    by_stratum, missing, proposed = selection._build_stratified_proposal(
        eligible,
        per_stratum=2,
    )
    assert len(by_stratum["low"]) == 2
    assert missing == ["mid", "high"]
    assert proposed == []


def test_latest_hard_acceptance_mapping_is_l900_and_110_130_130() -> None:
    assert selection.ACCEPTANCE_SIZE_LIMITS_MM == {
        "W": 1200.0,
        "L": 900.0,
        "H": 750.0,
    }
    family_limits = {
        target: selection.ACCEPTANCE_TEMPERATURE_LIMITS_C[target]
        for target in selection.ACCEPTANCE_TEMPERATURE_LIMITS_C
    }
    assert {
        family_limits[target]
        for target in selection.PRIMARY_WINDING_TEMPERATURE_TARGETS
    } == {110.0}
    assert {
        family_limits[target]
        for target in selection.SECONDARY_WINDING_TEMPERATURE_TARGETS
    } == {130.0}
    assert {
        family_limits[target]
        for target in selection.CORE_TEMPERATURE_TARGETS
    } == {130.0}


def test_active_search_temperature_matches_latest_acceptance() -> None:
    physical = {
        f"temperature_robust_limit:{target}": -10.0
        for target in selection.ACCEPTANCE_TEMPERATURE_LIMITS_C
    }
    evidence = selection._temperature_evidence(physical)
    assert all(
        item["acceptance_margin_C"] == pytest.approx(10.0)
        for item in evidence.values()
    )
    assert all(
        item["acceptance_limit_passed"] is True
        for item in evidence.values()
    )

    primary = next(iter(selection.PRIMARY_WINDING_TEMPERATURE_TARGETS))
    physical[f"temperature_robust_limit:{primary}"] = 0.001
    with pytest.raises(
        selection.StratifiedSelectionError, match="exceeds acceptance"
    ):
        selection._temperature_evidence(physical)


def test_temperature_evidence_can_retain_exploratory_overlimit_margin() -> None:
    physical = {
        f"temperature_robust_limit:{target}": 0.0
        for target in selection.ACCEPTANCE_TEMPERATURE_LIMITS_C
    }
    primary = next(iter(selection.PRIMARY_WINDING_TEMPERATURE_TARGETS))
    physical[f"temperature_robust_limit:{primary}"] = 2.5
    evidence = selection._temperature_evidence(physical, enforce=False)
    assert evidence[primary]["realized_robust_temperature_C"] == pytest.approx(
        112.5
    )
    assert evidence[primary]["acceptance_margin_C"] == pytest.approx(-2.5)
    assert evidence[primary]["acceptance_limit_passed"] is False


def test_legacy_cap_and_resonance_constraints_are_replaced() -> None:
    assert selection.CORRECTED_CAP_CONSTRAINT in (
        selection.REPLACED_LEGACY_CAPACITANCE_CONSTRAINTS
    )
    assert "half_magnetizing_resonance_minimum" in (
        selection.REPLACED_LEGACY_CAPACITANCE_CONSTRAINTS
    )
    assert "analytical_flux_density_limit" not in (
        selection.REPLACED_LEGACY_CAPACITANCE_CONSTRAINTS
    )


def test_physics_delta_resonance_constraint_uses_l2_0p2H_formula() -> None:
    threshold_c = 1.0 / (
        (2.0 * math.pi * selection.physics_reranker.RESONANCE_MIN_HZ)
        ** 2
        * selection.physics_reranker.FIXED_L2_H
    )
    boundary = {
        "physics_delta_Crx_q90_ucb_F": threshold_c,
        "physics_delta_fRx_q90_lcb_Hz": (
            selection.physics_reranker.RESONANCE_MIN_HZ
        ),
    }
    assert selection._physics_delta_provisional_resonance_G(  # noqa: SLF001
        boundary
    ) == pytest.approx(0.0, abs=1e-9)

    lower_capacitance = {
        "physics_delta_Crx_q90_ucb_F": threshold_c * 0.81,
        "physics_delta_fRx_q90_lcb_Hz": (
            selection.physics_reranker.RESONANCE_MIN_HZ / 0.9
        ),
    }
    assert selection._physics_delta_provisional_resonance_G(  # noqa: SLF001
        lower_capacitance
    ) < 0.0


def test_multi_violation_formula_excludes_legacy_gates() -> None:
    candidate = _fallback_candidate(
        "a",
        stratum="low",
        gap2=0.5,
        violation=0.1,
        spread=0.0,
    )
    evidence = candidate["multi_violation_evidence"]
    assert set(evidence["positive_normalized_components"]) == {
        "Llt_robust_band",
        selection.PHYSICS_RESONANCE_CONSTRAINT_NAME,
    }
    assert evidence["normalized_positive_sum"] == pytest.approx(
        0.1 + (15_000.0 / 12_000.0 - 1.0)
    )
    assert (
        selection.CORRECTED_CAP_CONSTRAINT
        not in evidence["positive_normalized_components"]
    )
    assert (
        "half_magnetizing_resonance_minimum"
        not in evidence["positive_normalized_components"]
    )


def test_nearest_fallback_selects_exactly_four_per_gap_stratum() -> None:
    candidates = []
    tokens = iter("abcdefghijklmnopqr")
    for stratum, gap2 in (("low", 0.5), ("mid", 1.1), ("high", 1.8)):
        for index in range(6):
            candidates.append(
                _fallback_candidate(
                    next(tokens),
                    stratum=stratum,
                    gap2=gap2 + index * 0.01,
                    violation=0.05 + index * 0.01,
                    spread=float(index),
                )
            )
    by_stratum, selected = selection._build_nearest_fallback_proposal(  # noqa: SLF001
        candidates
    )
    assert {name: len(pool) for name, pool in by_stratum.items()} == {
        "low": 6,
        "mid": 6,
        "high": 6,
    }
    assert len(selected) == 12
    assert {
        name: sum(item["gap2_stratum"] == name for item in selected)
        for name in ("low", "mid", "high")
    } == {"low": 4, "mid": 4, "high": 4}
    assert all(item["exploratory_nonpromotion"] is True for item in selected)
    assert all(item["strict_latest_rerank_eligible"] is False for item in selected)


def test_nearest_fallback_is_invariant_to_legacy_gate_values() -> None:
    candidates = [
        _fallback_candidate(
            token,
            stratum=stratum,
            gap2=gap2,
            violation=0.1 + index * 0.01,
            spread=float(index),
        )
        for index, (token, stratum, gap2) in enumerate(
            zip(
                "abcdefghijkl",
                ("low",) * 4 + ("mid",) * 4 + ("high",) * 4,
                (0.5,) * 4 + (1.1,) * 4 + (1.8,) * 4,
            )
        )
    ]
    _, first = selection._build_nearest_fallback_proposal(  # noqa: SLF001
        copy.deepcopy(candidates)
    )
    changed = copy.deepcopy(candidates)
    for index, candidate in enumerate(changed):
        candidate["normalized_G"][selection.CORRECTED_CAP_CONSTRAINT] = (
            -1.0e9 + index
        )
        candidate["normalized_G"]["half_magnetizing_resonance_minimum"] = (
            1.0e12 - index
        )
        candidate["multi_violation_evidence"] = (
            selection._multi_violation_evidence(  # noqa: SLF001
                candidate,
                candidate["physics_delta_prediction"],
            )
        )
    _, second = selection._build_nearest_fallback_proposal(  # noqa: SLF001
        changed
    )
    assert [item["physical_geometry_sha256"] for item in first] == [
        item["physical_geometry_sha256"] for item in second
    ]


def test_seed_retry_dedupe_keeps_highest_authenticated_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "collector"
    paths = {}
    for task_id, seed in ((11, 4300), (12, 4300), (13, 4301)):
        path = root / f"task-{task_id}" / "collection_record.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        collection = _scientific_collection(task_id, seed)
        collection["record_path"] = path.resolve()
        paths[path.resolve()] = collection

    monkeypatch.setattr(
        selection,
        "authenticate_collection",
        lambda path: copy.deepcopy(paths[path.resolve()]),
    )
    chosen, evidence = selection.discover_and_deduplicate_collections([root])
    assert [(item["seed"], item["task_id"]) for item in chosen] == [
        (4300, 12),
        (4301, 13),
    ]
    assert evidence["discarded_retry_tasks"] == [
        {
            "seed": 4300,
            "discarded_task_id": 11,
            "retained_task_id": 12,
            "reason": "higher_task_id_authenticated_retry_wins",
            "discarded_collection_record_payload_sha256": f"{11:064x}",
            "retained_collection_record_payload_sha256": f"{12:064x}",
        }
    ]


def test_seed_retry_dedupe_rejects_profile_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "collector"
    paths = {}
    for task_id in (21, 22):
        path = root / f"task-{task_id}" / "collection_record.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        collection = _scientific_collection(task_id, 4300)
        collection["record_path"] = path.resolve()
        paths[path.resolve()] = collection
    paths[(root / "task-22" / "collection_record.json").resolve()][
        "profile_sha256"
    ] = "9" * 64
    monkeypatch.setattr(
        selection,
        "authenticate_collection",
        lambda path: copy.deepcopy(paths[path.resolve()]),
    )
    with pytest.raises(selection.StratifiedSelectionError, match="mix scientific"):
        selection.discover_and_deduplicate_collections([root])


def test_reviewed_rx_params_are_nonrounded_symmetric_and_60_turn_compatible() -> None:
    params = get_drawing_default_params()
    params.update(
        {
            "N1_main": 6,
            "N1_side": 0,
            "N2_main": 35,
            "N2_side": 25,
            "cw1": 5.0,
            "gap1": 1.6,
            "cw2": 0.8,
            "gap2": 0.5,
            "core_plate_t": 20.0,
            "wcp_t": 20.0,
            "n_core_group": 6,
            "l2": 330.0,
            "cc_w2c_space_x": 40.0,
            "cc_w2c_space_y": 40.0,
            "w2c_w1c_space_x": 40.0,
            "w2c_w1c_space_y": 40.0,
            "w1c_w2s_space_x": 20.0,
            "w1s_cs_space_x": 40.0,
            "cs_w1s_space_y": 40.0,
        }
    )
    decoded = create_input_parameter(params).iloc[0].to_dict()
    base, rx, compatibility = selection._reviewed_rx_turn_graded_params(decoded)

    assert set(base) == set(selection.ALL_INPUT_KEYS)
    assert set(rx) == set(selection.ALL_INPUT_KEYS)
    assert rx["full_model"] == 0
    assert rx["round_corner"] == 0
    assert rx["matrix_on"] == 1
    assert rx["cap_turn_graded_active_winding"] == "Rx"
    assert rx["cap_turn_graded_section_order"] == "main,side"
    assert rx["N2_main"] + rx["N2_side"] == 60
    assert compatibility["explicit_turn_voltage_count"] == 60
    assert compatibility["n_explicit_turns_field_not_repurposed"] is True
    assert compatibility["profile_canonical_sha256"] == canonical_sha256(
        selection.REVIEWED_TURN_GRADED_PROFILE
    )
