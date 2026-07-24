import json
import math
from pathlib import Path

import pytest

from module.input_parameter_260706 import KEYS
from run_simulation_260706 import standalone_core_contract_auth_sha256
from tools.prepare_fixed_thermal_replay import (
    HARD_SPEC,
    ReplayPreparationError,
    build_candidate_report,
    build_payload,
    fixed_execution_params,
    replay_geometry,
    standalone_8_core_auth,
)


FIXTURE = Path(__file__).parent / "fixtures" / (
    "b428_task95067_standard_candidate.json"
)


def _source_params():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["params"]


def _source_result():
    return {
        "Llt": 27.0978955275e-6,
        "Lmt": 1.2e-3,
        "Lmr": 1.1e-3,
        "C_tx_tx_F": 4.75e-8,
        "C_rx_rx_F": 6.0e-10,
        "C_tx_rx_F": 3.0e-10,
        "f_res_tx_self_Hz": 12704.732977,
        "f_res_rx_self_Hz": 11166.429837,
        "f_res_interwinding_Hz": 1844050.79,
    }


def test_exact_b428_fixed_boundary_replay_is_nonselectable_on_length():
    source = _source_params()
    assert set(source) == set(KEYS)
    assert replay_geometry(source)["dimensions_mm"] == pytest.approx(
        {"W": 1196.796, "L": 999.844, "H": 704.0}
    )

    standard = fixed_execution_params(source, "standard")
    full = fixed_execution_params(source, "full")
    fixed_geometry = replay_geometry(standard)
    assert fixed_geometry["dimensions_mm"] == pytest.approx(
        {"W": 1196.796, "L": 1007.844, "H": 704.0}
    )
    assert fixed_geometry["volume_L"] == pytest.approx(849.153302148096)
    assert fixed_geometry["B_design_square_material_analytic_T"] == (
        pytest.approx(1.037320294632158)
    )
    assert fixed_geometry["Ae_gross_m2"] == pytest.approx(0.047256)
    assert fixed_geometry["core_depth_each_mm"] == pytest.approx(89.5)
    assert fixed_geometry["tx_winding_y_pack_mm"] == pytest.approx(80.8)
    assert fixed_geometry["hard_checks"]["length"] is False
    assert fixed_geometry["hard_prescreen_pass"] is False
    assert fixed_geometry["superseded_size_contract"] == {
        "size_L_max_mm": 1200.0,
        "authoritative": False,
        "pass": True,
        "reason": "superseded_by_final_1200x1000x750_envelope",
    }

    assert standard["fan_velocity"] == 1.5
    assert standard["wcp_pad_t"] == 2.0
    assert standard["core_plate_pad_t"] == 2.0
    assert standard["full_model"] == 0
    assert full["full_model"] == 1
    assert full["thermal_symmetry"] == "full"

    report, _ = build_candidate_report(source, _source_result())
    assert report["candidate_digest"] == (
        "212dbc99d467ee93f4420841e708671b848580de3f567ccd6a960da36888cb26"
    )
    assert report["status"] == (
        "diagnostic_nonselectable_hard_prescreen_failed"
    )
    assert report["submission_eligible"] is False
    assert report["design_approved"] is False
    assert report["final_hard_gate_status"]["failed_prescreens"] == ["length"]
    assert report["fixed_boundary"]["attestation"][
        "authoritative_fixed_boundary_attested"
    ] is True
    assert report["source_candidate"]["fixed_boundary_evaluation"][
        "authority_class"
    ] == "diagnostic_override_only"
    assert report["source_candidate"]["fixed_boundary_evaluation"][
        "promotion_forbidden_by_fixed_boundary"
    ] is True
    assert (
        report["source_candidate"]["electrical"][
            "reusable_for_fixed_geometry"
        ]
        is False
    )
    assert report["fresh_fea_required"] == {
        "Llt": True,
        "capacitance": True,
        "resonance": True,
        "loss": True,
        "temperature": True,
        "reason": "physical pad stack and TIM boundary changed",
    }
    assert HARD_SPEC["size_L_max_mm"] == 1000.0
    assert HARD_SPEC["temperature_max_C"] == 100.0


def test_payload_builder_rejects_the_exact_nonselectable_candidate():
    params = fixed_execution_params(_source_params(), "standard")
    with pytest.raises(ReplayPreparationError, match="hard prescreen"):
        build_payload(
            fidelity="standard",
            params=params,
            candidate_digest="a" * 64,
            profile_sha256="b" * 64,
            revision="c" * 40,
            bundle_sha256="d" * 64,
            bundle_size=1,
            release_root="/gpfs/home1/wjddn5916/release",
            output_parent="/gpfs/home1/wjddn5916/output",
        )


def test_eligible_payload_is_offline_and_uses_attested_storage_and_core_auth():
    params = fixed_execution_params(_source_params(), "standard")
    params["w1"] = 420
    assert replay_geometry(params)["hard_prescreen_pass"] is True
    revision = "c" * 40
    payload = build_payload(
        fidelity="standard",
        params=params,
        candidate_digest="a" * 64,
        profile_sha256="b" * 64,
        revision=revision,
        bundle_sha256="d" * 64,
        bundle_size=123,
        release_root="/gpfs/home1/wjddn5916/release",
        output_parent="/gpfs/home1/wjddn5916/output",
    )
    assert payload["cpus"] == 8
    assert payload["account_name"] == "r1jae262"
    assert payload["node_name"] == "n107"
    assert payload["payload_json"]["storage_account"] == "wjddn5916"
    assert "/gpfs/home1/wjddn5916/release" in payload["command"]
    assert "/gpfs/home1/wjddn5916/output" in payload["command"]
    assert "http://" not in payload["command"]
    assert "https://" not in payload["command"]
    assert "curl " not in payload["command"]
    assert "MFT_STANDALONE_CORE_COUNT=\"8\"" in payload["command"]
    assert standalone_8_core_auth(revision) == (
        standalone_core_contract_auth_sha256(revision, 8)
    )
    assert math.isfinite(float(params["fan_velocity"]))
