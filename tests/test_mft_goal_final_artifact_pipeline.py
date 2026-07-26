import copy
import json
from pathlib import Path

import pytest

from module.input_parameter_260706 import get_drawing_default_params
from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY,
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
)
from module.fixed_boundary_contract import FIXED_BOUNDARY_CONTRACT_SHA256
from tools import mft_goal_final_artifact_pipeline as pipeline


def _params():
    params = get_drawing_default_params()
    params.update(FIXED_OPERATING_IDENTITY)
    params.update(
        {
            key: value
            for key, value in FIXED_COOLING_IDENTITY.items()
            if key != "thermal_pad_conductivity_W_mK"
        }
    )
    params.update(
        {
            "full_model": 0,
            "round_corner": 0,
            "matrix_on": 1,
            "cap_on": 1,
            "loss_on": 1,
            "thermal_on": 1,
            "loss_sym_on": 1,
            "thermal_symmetry": "eighth",
            "keep_project": 1,
        }
    )
    return params


def _authority(tmp_path: Path, *, params=None):
    params = copy.deepcopy(params if params is not None else _params())
    candidate_sha256 = "a" * 64
    symmetric = tmp_path / "symmetric.aedt"
    symmetric.write_bytes(b"synthetic-aedt")
    upstream = tmp_path / "selected-receipt.json"
    upstream.write_text('{"selected":true}\n', encoding="utf-8")
    tx_cap_receipt = tmp_path / "tx-graded-cap-receipt.json"
    tx_cap_receipt.write_text('{"authenticated":"Tx"}\n', encoding="utf-8")
    rx_cap_receipt = tmp_path / "rx-graded-cap-receipt.json"
    rx_cap_receipt.write_text('{"authenticated":"Rx"}\n', encoding="utf-8")
    topology_receipt = tmp_path / "graded-cap-topology-receipt.json"
    topology_receipt.write_text(
        '{"actual_connection_topology_attested":true}\n',
        encoding="utf-8",
    )
    return pipeline.seal(
        {
            "schema_version": pipeline.WINNER_AUTHORITY_SCHEMA,
            "source": {
                "task_id": 12345,
                "candidate_physics_sha256": candidate_sha256,
                "selection_performed_upstream": True,
                "candidate_promotion_performed_by_pipeline": False,
                "selected_symmetric_hard_pass": True,
                "upstream_selection_receipt": pipeline.file_record(upstream),
                "retained_symmetric_aedt": pipeline.file_record(symmetric),
            },
            "contracts": {
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "goal_stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "goal_temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "fixed_operating_identity_sha256": (
                    FIXED_OPERATING_IDENTITY_SHA256
                ),
                "fixed_cooling_identity_sha256": (
                    FIXED_COOLING_IDENTITY_SHA256
                ),
                "fixed_boundary_contract_sha256": (
                    FIXED_BOUNDARY_CONTRACT_SHA256
                ),
            },
            "params": params,
            "symmetric_verification": {
                "full_model": 0,
                "round_corner": 0,
                "matrix_solved": True,
                "capacitance_solved": True,
                "graded_capacitance_solved": True,
                "loss_solved": True,
                "thermal_solved": True,
                "measured_hard_constraints_passed": True,
                "rounded_fea_used": False,
                "legacy_two_net_capacitance_used_for_final_resonance": False,
                "actual_dimensions_mm": {
                    "W": 1190.0,
                    "L": 900.0,
                    "H": 740.0,
                },
                "actual_resonance_Hz": 15_100.0,
                "actual_Lm_primary_referred_H": 0.00199,
                "graded_capacitance_provenance": {
                    "schema_version": pipeline.GRADED_CAP_PROVENANCE_SCHEMA,
                    "provenance_authenticated": True,
                    "source_kind": (
                        "authenticated_same_geometry_tx_rx_pair"
                    ),
                    "geometry_and_gap_exact_match": True,
                    "actual_connection_topology_attested": True,
                    "actual_connection_topology_receipt": (
                        pipeline.file_record(topology_receipt)
                    ),
                    "legacy_two_net_result_used": False,
                    "candidate_physics_sha256": candidate_sha256,
                    "core_center_gap_mm": params["core_center_gap_mm"],
                    "minimum_resonance_Hz": 15_100.0,
                    "tx": {
                        "task_id": 2001,
                        "active_winding": "Tx",
                        "cap_turn_graded_schema_version": (
                            pipeline.TURN_GRADED_CAP_SCHEMA
                        ),
                        "candidate_physics_sha256": candidate_sha256,
                        "core_center_gap_mm": params["core_center_gap_mm"],
                        "solver_revision": "b" * 40,
                        "library_revision": "c" * 40,
                        "result_sha256": "d" * 64,
                        "terminal_capacitance_F": 1e-9,
                        "self_inductance_H": 0.002,
                        "resonance_Hz": 16_000.0,
                        "authenticated_result_receipt": (
                            pipeline.file_record(tx_cap_receipt)
                        ),
                    },
                    "rx": {
                        "task_id": 2002,
                        "active_winding": "Rx",
                        "cap_turn_graded_schema_version": (
                            pipeline.TURN_GRADED_CAP_SCHEMA
                        ),
                        "candidate_physics_sha256": candidate_sha256,
                        "core_center_gap_mm": params["core_center_gap_mm"],
                        "solver_revision": "b" * 40,
                        "library_revision": "c" * 40,
                        "result_sha256": "e" * 64,
                        "terminal_capacitance_F": 1e-9,
                        "self_inductance_H": 0.02,
                        "resonance_Hz": 15_100.0,
                        "authenticated_result_receipt": (
                            pipeline.file_record(rx_cap_receipt)
                        ),
                    },
                },
                "actual_temperature_family_max_C": {
                    "primary_winding": 99.0,
                    "secondary_winding": 119.0,
                    "core": 119.0,
                },
                "fixed_boundary": {
                    "fan_velocity_m_s": 1.5,
                    "fan_config": "dual",
                    "core_plate_pad_t_mm": 2.0,
                    "wcp_pad_t_mm": 2.0,
                    "thermal_pad_conductivity_W_mK": 0.2,
                    "TIM_mutated": False,
                },
            },
        }
    )


def test_winner_authority_authenticates_exact_nonrounded_symmetric_pass(
    tmp_path,
):
    authority = _authority(tmp_path)
    result = pipeline.validate_winner_authority(
        authority, authority_directory=tmp_path
    )
    assert result["turns"] == {"N1": 6, "N2": 60}
    assert result["source"]["task_id"] == 12345
    assert (
        result["fixed_boundary_attestation"][
            "authoritative_fixed_boundary_attested"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (
            "symmetric_verification",
            "rounded_fea_used",
            True,
            "rounded_fea_used drifted",
        ),
        (
            "symmetric_verification",
            "actual_resonance_Hz",
            14_999.0,
            "below 15 kHz",
        ),
    ],
)
def test_winner_authority_fails_closed_on_scientific_drift(
    tmp_path, section, key, value, message
):
    authority = _authority(tmp_path)
    body = copy.deepcopy(authority)
    body.pop("payload_sha256")
    body[section][key] = value
    with pytest.raises(pipeline.FinalArtifactPipelineError, match=message):
        pipeline.validate_winner_authority(
            pipeline.seal(body), authority_directory=tmp_path
        )


def test_winner_authority_rejects_turn_ratio_drift(tmp_path):
    params = _params()
    params["N2_side"] -= 1
    authority = _authority(tmp_path, params=params)
    with pytest.raises(
        pipeline.FinalArtifactPipelineError, match="required 1:10 ratio"
    ):
        pipeline.validate_winner_authority(
            authority, authority_directory=tmp_path
        )


def test_winner_authority_rejects_legacy_two_net_capacitance(
    tmp_path,
):
    authority = _authority(tmp_path)
    body = copy.deepcopy(authority)
    body.pop("payload_sha256")
    body["symmetric_verification"][
        "legacy_two_net_capacitance_used_for_final_resonance"
    ] = True
    with pytest.raises(
        pipeline.FinalArtifactPipelineError,
        match="legacy_two_net_capacitance_used_for_final_resonance drifted",
    ):
        pipeline.validate_winner_authority(
            pipeline.seal(body), authority_directory=tmp_path
        )


def test_winner_authority_requires_actual_graded_tx_rx_provenance(
    tmp_path,
):
    authority = _authority(tmp_path)
    body = copy.deepcopy(authority)
    body.pop("payload_sha256")
    del body["symmetric_verification"]["graded_capacitance_provenance"]
    with pytest.raises(
        pipeline.FinalArtifactPipelineError,
        match="actual turn-graded capacitance provenance is absent",
    ):
        pipeline.validate_winner_authority(
            pipeline.seal(body), authority_directory=tmp_path
        )


def test_final_resonance_must_equal_authenticated_graded_minimum(
    tmp_path,
):
    authority = _authority(tmp_path)
    body = copy.deepcopy(authority)
    body.pop("payload_sha256")
    body["symmetric_verification"]["actual_resonance_Hz"] = 15_200.0
    with pytest.raises(
        pipeline.FinalArtifactPipelineError,
        match="not the authenticated turn-graded minimum",
    ):
        pipeline.validate_winner_authority(
            pipeline.seal(body), authority_directory=tmp_path
        )


def test_model_variants_are_full_model_only_and_keep_fixed_boundary():
    variants = pipeline.derive_model_variants(_params())
    unrounded = variants["full_unrounded"]
    rounded = variants["full_rounded_drawing_only"]
    for value in variants.values():
        assert value["full_model"] == 1
        assert value["cap_on"] == 0
        assert value["loss_on"] == 0
        assert value["thermal_on"] == 0
        assert value["fan_velocity"] == 1.5
        assert value["fan_config"] == "dual"
        assert value["core_plate_pad_t"] == 2.0
        assert value["wcp_pad_t"] == 2.0
        assert value["k_ins"] == 0.2
    assert unrounded["round_corner"] == 0
    assert rounded["round_corner"] == 1


def test_prepare_writes_sealed_parallel_execution_plan_without_solving(
    tmp_path, monkeypatch
):
    authority = _authority(tmp_path)
    authority_path = tmp_path / "winner.json"
    authority_path.write_bytes(pipeline.canonical_bytes(authority) + b"\n")
    pdf = tmp_path / "reference.pdf"
    pptx = tmp_path / "reference.pptx"
    pdf.write_bytes(b"reference-pdf")
    pptx.write_bytes(b"reference-pptx")
    monkeypatch.setattr(
        pipeline, "REFERENCE_PDF_SHA256", pipeline.sha256_file(pdf)
    )
    monkeypatch.setattr(
        pipeline, "REFERENCE_PPTX_SHA256", pipeline.sha256_file(pptx)
    )
    output = tmp_path / "prepared"
    manifest_path = pipeline.prepare(
        winner_authority=authority_path,
        output=output,
        reference_pdf=pdf,
        reference_pptx=pptx,
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    pipeline.validate_seal(
        manifest, schema=pipeline.PIPELINE_SCHEMA, label="pipeline"
    )
    assert manifest["authority"]["selection_performed_by_pipeline"] is False
    assert manifest["scientific_contract"]["rounded_FEA_allowed"] is False
    assert len(manifest["drawing_deck"]["slide_contract"]) == 9
    rounded = manifest["model_outputs"][
        "full_rounded_drawing_only"
    ]
    assert rounded["solver_invoked"] is False
    assert rounded["rounded_FEA_prohibited"] is True
    assert "--model-only" in rounded["command"]["argv"]
    assert "--round" in rounded["command"]["argv"]
    assert (
        manifest["model_outputs"]["full_unrounded_model_only"][
            "command"
        ]["argv"]
    ).count("--model-only") == 1
    assert len(manifest["parallel_execution"][0]["parallel"]) == 4
    assert pipeline.prepare(
        winner_authority=authority_path,
        output=output,
        reference_pdf=pdf,
        reference_pptx=pptx,
    ) == manifest_path
