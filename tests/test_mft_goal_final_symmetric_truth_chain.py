import json
import math

import pytest

from regression_260707.verify import scheduler_client
from tools import mft_goal_final_symmetric_truth_chain as final_chain
from tools import mft_goal_lm2mh_gap_tuner as gap_tuner
from tools import mft_goal_targeted_symmetric_fea_batch as targeted


def _actual_cap_gate():
    geometry = "6" * 64
    tuned = {"tuned_core_center_gap_mm": 0.65}
    campaign = {
        "candidate": {"physical_geometry_sha256": geometry}
    }
    cap_plan = {
        "payload_sha256": "7" * 64,
        "selected_variant_ids": [
            "tx-main-mid",
            "rx-main-side-mid",
        ],
        "source": {
            "physical_geometry_sha256": geometry,
            "core_center_gap_mm": 0.65,
        },
    }
    actual = {
        "Tx": {
            "variant_id": "tx-main-mid",
            "contract_valid": True,
            "resonance_spec_pass_15kHz": True,
            "C_terminal_F": 13.2e-9,
            "f_res_Hz": 30_000.0,
        },
        "Rx": {
            "variant_id": "rx-main-side-mid",
            "contract_valid": True,
            "resonance_spec_pass_15kHz": True,
            "C_terminal_F": 0.42e-9,
            "f_res_Hz": 17_000.0,
        },
    }
    collection = {
        "plan_payload_sha256": cap_plan["payload_sha256"],
        "all_terminal": True,
        "actual_connection_resonance_spec_pass": True,
        "actual_connection_results": actual,
    }
    return tuned, campaign, cap_plan, collection


@pytest.mark.parametrize("drift", ["gap", "geometry"])
def test_actual_cap_gate_rejects_different_gap_or_geometry(drift):
    tuned, campaign, cap_plan, collection = _actual_cap_gate()
    if drift == "gap":
        cap_plan["source"]["core_center_gap_mm"] += 0.001
    else:
        cap_plan["source"]["physical_geometry_sha256"] = "8" * 64
    with pytest.raises(final_chain.FinalTruthError):
        final_chain._validate_actual_cap_gate(
            tuned=tuned,
            campaign=campaign,
            cap_plan=cap_plan,
            cap_collection=collection,
        )


def test_actual_cap_gate_rejects_legacy_cap_only():
    tuned, campaign, cap_plan, collection = _actual_cap_gate()
    collection.pop("actual_connection_results")
    collection["actual_connection_resonance_spec_pass"] = False
    collection["legacy_equipotential_capacitance_F"] = {
        "C_tx_tx_F": 13.2e-9,
        "C_rx_rx_F": 0.42e-9,
    }
    with pytest.raises(final_chain.FinalTruthError):
        final_chain._validate_actual_cap_gate(
            tuned=tuned,
            campaign=campaign,
            cap_plan=cap_plan,
            cap_collection=collection,
        )


def test_same_candidate_tuned_and_actual_graded_builds_plan(
    tmp_path, monkeypatch
):
    tuned, campaign, cap_plan, collection = _actual_cap_gate()
    actual = collection["actual_connection_results"]
    collection["payload_sha256"] = "e" * 64
    tuned.update(
        {
            "payload_sha256": "9" * 64,
            "tuned_Lm_primary_referred_H": 0.002,
        }
    )
    campaign.update(
        {
            "payload_sha256": "a" * 64,
            "candidate": {
                **campaign["candidate"],
                "source_batch_rank": 1,
            },
        }
    )
    cap_plan.update(
        {
            "solver_revision": "b" * 40,
            "library_revision": "c" * 40,
        }
    )
    downstream = tmp_path / "gap" / "downstream_params"
    downstream.mkdir(parents=True)
    source_params = {
        "cw1": 5.0,
        "gap1": 1.6,
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 37,
        "N2_side": 23,
    }
    source_path = downstream / "symmetric_loss_thermal.json"
    source_path.write_text(json.dumps(source_params), encoding="utf-8")
    tuned["downstream_params"] = {
        "symmetric_loss_thermal": gap_tuner._file_record(
            source_path, relative_to=tmp_path / "gap"
        )
    }
    base_profile = {
        "schema_version": "source-profile",
        "cli_flags": (
            "--thermal --headless --symmetry-thermal-direct-analyze"
        ),
        "fixed_boundary_contract": {"fan_velocity_m_s": 1.5},
        "param_overrides": {
            "fan_velocity": 1.5,
            "fan_config": "dual",
            "core_plate_pad_t": 2.0,
            "wcp_pad_t": 2.0,
            "k_ins": 0.2,
        },
    }
    source_plan = {
        "payload_sha256": "d" * 64,
        "solver_revision": cap_plan["solver_revision"],
        "library_revision": cap_plan["library_revision"],
    }
    for name in ("gap.json", "cap-plan.json", "cap-collection.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        final_chain,
        "_load_upstream",
        lambda **_kwargs: (
            tuned,
            tmp_path / "gap",
            campaign,
            cap_plan,
            collection,
            actual,
        ),
    )
    monkeypatch.setattr(
        final_chain,
        "_source_batch",
        lambda _campaign: (
            source_plan,
            tmp_path,
            base_profile,
        ),
    )
    monkeypatch.setattr(
        "module.input_parameter_260706.create_input_parameter",
        lambda _params: object(),
    )
    monkeypatch.setattr(
        "module.input_parameter_260706.validation_check",
        lambda _frame, strict: (strict, None),
    )
    plan_path = final_chain.prepare(
        gap_manifest_path=tmp_path / "gap.json",
        cap_plan_path=tmp_path / "cap-plan.json",
        cap_collection_path=tmp_path / "cap-collection.json",
        output=tmp_path / "final",
        priority=100,
    )
    plan, _root, profile, params = final_chain._load_plan(plan_path)
    assert plan["scheduler_priority"] == 100
    assert plan["physical_geometry_sha256"] == "6" * 64
    assert plan["single_final_authority_task"] is True
    assert plan["rounded_FEA_used"] is False
    assert params["core_center_gap_mm"] == pytest.approx(0.65)
    assert profile["param_overrides"][
        "cap_turn_graded_active_winding"
    ] == "Rx"
    assert profile["param_overrides"]["cap_on"] == 0
    assert profile["param_overrides"]["loss_on"] == 1
    assert profile["param_overrides"]["thermal_on"] == 1


def _completed_result():
    native_lm_uH = 1000.0
    l11_uH = 1250.0
    coupling = math.sqrt(native_lm_uH / l11_uH)
    l22_uH = 125_000.0
    result = {
        "full_model": 0,
        "round_corner": 0,
        "matrix_on": 1,
        "cap_on": 0,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "cap_turn_graded_active_winding": "Rx",
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
        "core_center_gap_mm": 0.65,
        "core_center_gap_geometry_attested": 1,
        "core_center_gap_symmetry_geometry_attested": 1,
        "core_center_gap_topology": (
            "center_leg_bottom_top_physical_air_interval"
        ),
        "core_center_gap_readback_mm": 0.65,
        "Ltx": l11_uH,
        "Lrx": l22_uH,
        "M": coupling * math.sqrt(l11_uH * l22_uH),
        "k": coupling,
        "Lmt": native_lm_uH,
        "matrix_percent_error": 1.5,
        "matrix_min_converged": 1,
        "conv_passes_matrix": 6,
        "conv_consecutive_matrix": 1,
        "conv_error_pct_matrix": 1.0,
        "conv_delta_pct_matrix": 0.8,
        "C_rx_rx_turn_graded_F": 0.42e-9,
        "f_res_rx_turn_graded_Hz": 17_000.0,
        "N2_side": 23,
        "P_winding_total": 2500.0,
        "P_core_total": 3000.0,
        "thermal_iterations": 50,
    }
    for name in targeted.PRIMARY_WINDING_TEMPERATURES:
        result[name] = 95.0
    for name in targeted.SECONDARY_WINDING_TEMPERATURES:
        result[name] = 110.0
    for name in targeted.CORE_TEMPERATURES:
        result[name] = 115.0
    return result


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            lambda result: result.__setitem__("T_max_Tx", 100.01),
            "primary_winding_above_100C",
        ),
        (
            lambda result: result.__setitem__(
                "f_res_rx_turn_graded_Hz", 14_999.0
            ),
            "actual_turn_graded_resonance_below_15kHz",
        ),
        (
            lambda result: result.__setitem__("Lmt", 980.0),
            "physical_Lm_not_within_2mH_tolerance",
        ),
    ],
)
def test_collect_split_temperature_resonance_and_lm_fail_closed(
    tmp_path, monkeypatch, mutation, expected_reason
):
    result = _completed_result()
    mutation(result)
    plan = {
        "payload_sha256": "1" * 64,
        "solver_revision": "2" * 40,
        "library_revision": "3" * 40,
        "tuned_core_center_gap_mm": 0.65,
        "upstream_actual_connection_results": {
            "Tx": {"f_res_Hz": 30_000.0},
            "Rx": {"C_terminal_F": 0.42e-9},
        },
    }
    profile = {"param_overrides": {}}
    params = {}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    submission = gap_tuner._seal(
        {
            "schema_version": final_chain.SUBMISSION_SCHEMA,
            "complete": True,
            "plan_payload_sha256": plan["payload_sha256"],
            "task_id": 97014,
            "scheduler_url": final_chain.SCHEDULER_URL,
        }
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        json.dumps(submission), encoding="utf-8"
    )
    monkeypatch.setattr(
        final_chain,
        "_load_plan",
        lambda _path: (plan, tmp_path, profile, params),
    )
    monkeypatch.setattr(
        final_chain,
        "_task",
        lambda _base, _task_id: {
            "status": "completed",
            "result_json": result,
        },
    )
    monkeypatch.setattr(
        scheduler_client, "is_valid_result", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        scheduler_client,
        "result_matches_params",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        final_chain.geometry_metrics,
        "bounding_box_lit",
        lambda _params: (700.0, (1100.0, 900.0, 700.0)),
    )
    output = final_chain.collect(
        plan_path=plan_path,
        submission_path=submission_path,
        output=tmp_path / "collection.json",
    )
    collected = json.loads(output.read_text(encoding="utf-8"))
    assert collected["final_design_pass"] is False
    assert collected["production_eligible"] is False
    assert expected_reason in collected["row"]["reasons"]
