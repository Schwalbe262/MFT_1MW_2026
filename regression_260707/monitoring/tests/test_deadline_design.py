import copy
import hashlib
import json
from pathlib import Path

import pytest

from regression_260707.monitoring import deadline_design, readers


def _write(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _package(tmp_path, *, full=False, variant_label="timk3"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    identity = "a" * 64
    digest = "b" * 64
    variant = deadline_design.ALLOWED_TIM_COOLING_VARIANTS[variant_label]
    pad_mm = variant["pad_thickness_mm"]
    fan_m_s = variant["fan_velocity_m_s"]
    parameters = {
        key: float(index + 1)
        for index, key in enumerate(deadline_design.REQUIRED_PARAMETERS)
    }
    for key in ("N1_main", "N1_side", "N2_main", "N2_side", "n_core_group"):
        parameters[key] = int(parameters[key])
    parameters.update({
        "core_plate_pad_t": pad_mm,
        "wcp_pad_t": pad_mm,
        "fan_velocity": fan_m_s,
        "plate_temp": 50.0,
        "air_temp": 50.0,
        "fan_config": "dual",
    })
    evidence = {}
    for fidelity in ("standard", "full"):
        if fidelity == "full" and not full:
            evidence[fidelity] = None
            continue
        plan = {
            "target": dict(deadline_design.EXPECTED_HARD_SPEC),
            "candidate": {
                "candidate_id": variant_label,
                "decoded_params": dict(parameters),
            },
            "solver_contract": {
                "solver_variant": deadline_design.EXPECTED_TIM[
                    "solver_variant"
                ],
                "solver_revision": deadline_design.EXPECTED_TIM[
                    "solver_revision"
                ],
                "library_revision": deadline_design.EXPECTED_LOCAL_GUI_SOLVER[
                    "library_revision"
                ],
                "thermal_pad_native_readback_contract_version": (
                    deadline_design.EXPECTED_TIM[
                        "native_readback_contract_version"
                    ]
                ),
                "thermal_pad_native_readback_required": True,
            },
        }
        native = {
            **parameters,
            "git_hash": deadline_design.EXPECTED_TIM["solver_revision"],
            "pyaedt_library_git_hash": (
                deadline_design.EXPECTED_LOCAL_GUI_SOLVER[
                    "library_revision"
                ]
            ),
            "thermal_pad_conductivity_W_mK": (
                deadline_design.EXPECTED_TIM[
                    "thermal_conductivity_W_mK"
                ]
            ),
            "thermal_pad_material_policy": (
                deadline_design.EXPECTED_TIM["material_policy"]
            ),
            "thermal_pad_native_readback_contract_version": (
                deadline_design.EXPECTED_TIM[
                    "native_readback_contract_version"
                ]
            ),
            "thermal_pad_native_readback_attested": 1,
            "thermal_pad_native_thermal_conductivity_W_mK": (
                deadline_design.EXPECTED_TIM[
                    "native_thermal_conductivity_W_mK"
                ]
            ),
            "thermal_pad_native_electrical_conductivity_S_m": (
                deadline_design.EXPECTED_TIM[
                    "native_electrical_conductivity_S_m"
                ]
            ),
            "full_model": 1 if fidelity == "full" else 0,
        }
        result = {
            "schema_version": deadline_design.RESULT_SCHEMA,
            "fidelity": fidelity,
            "actual_hard_spec_pass": True,
            "candidate_identity_matches": True,
            "result_contract_valid": True,
            "solver_variant_contract_valid": True,
            "result_state": "valid",
            "task_status": "completed",
            "final_design_approved": False,
            "canonical_dataset_mutated": False,
            "scheduler_configuration_mutated": False,
            "task_cancellation_performed": False,
            "candidate_identity_sha256": identity,
            "candidate_digest": digest,
            "actual_hard_spec_gate": {
                "pass": True,
                "half_magnetizing_resonance_Hz": 16_000.0,
            },
            "result": native,
        }
        evidence[fidelity] = {
            "result": _write(tmp_path / f"{fidelity}-result.json", result),
            "plan": _write(
                tmp_path / f"{fidelity}-plan.json", plan
            ),
            "submission_receipt": _write(
                tmp_path / f"{fidelity}-submission.json",
                {"fidelity": fidelity},
            ),
        }

    numbers = {
        key: float(index + 1)
        for index, key in enumerate(
            deadline_design.REQUIRED_CANDIDATE_NUMBERS
        )
    }
    numbers.update({
        "size_W_mm": 1199.0,
        "size_L_mm": 988.0,
        "size_H_mm": 708.0,
        "volume_L": 839.0,
        "total_loss_W": 5600.0,
        "pred_Llt_phys": 27.5,
        "pred_max_temperature_C": 97.0,
        "min_insulation_mm": 40.0,
        "B_design_analytic_T": 0.75,
        "core_thermal_pad_thickness_mm": pad_mm,
        "winding_thermal_pad_thickness_mm": pad_mm,
        "thermal_pad_conductivity_W_mK": 3.0,
        "thermal_pad_native_readback_attested": 1.0,
        "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
        "thermal_pad_native_electrical_conductivity_S_m": 0.0,
        "fan_velocity_m_s": fan_m_s,
        "plate_temp_C": 50.0,
        "air_temp_C": 50.0,
        "pred_f_res_min_screen_Hz": 16_000.0,
        "resonance_minimum_required_Hz": 15_000.0,
        "resonance_margin_Hz": 1_000.0,
    })
    candidate = {
        "id": f"deadline-{variant_label}-{identity[:16]}",
        **numbers,
        "fan_config": "dual",
        "thermal_pad_material_policy": (
            deadline_design.EXPECTED_TIM["material_policy"]
        ),
        "thermal_pad_native_readback_contract_version": (
            deadline_design.EXPECTED_TIM[
                "native_readback_contract_version"
            ]
        ),
        "solver_revision": deadline_design.EXPECTED_TIM[
            "solver_revision"
        ],
        "spec_status": "pass",
        "artifact_hydrated": True,
        "embedded_authenticated_result": True,
        "fea_verified": True,
        "gui_launch_eligible": True,
        "gui_build_eligible": True,
        "gui_solve_eligible": full,
        "pareto_front_sha256": "c" * 64,
        "pareto_row_number": 1,
        "parameters": parameters,
        "local_gui_solver_contract": {
            **copy.deepcopy(deadline_design.EXPECTED_LOCAL_GUI_SOLVER),
            "required_result_echo_keys": list(
                deadline_design.REQUIRED_COOLING_ECHO_KEYS
            ),
        },
        "constraints": {
            "llt": {"pass": True},
            "temperature": {"pass": True},
            "resonance": {
                "value": 16_000.0,
                "limit": 15_000.0,
                "pass": True,
                "direction": "minimum",
                "operator": ">=",
                "minimum_Hz": 15_000.0,
            },
        },
    }
    hard_spec = dict(deadline_design.EXPECTED_HARD_SPEC)
    publication = {
        "schema_version": deadline_design.PUBLICATION_SCHEMA,
        "created_at": "2026-07-23T22:00:00+09:00",
        "ui_publication_ready": True,
        "engineering_handoff_ready": full,
        "standard_actual_hard_pass": True,
        "full_actual_hard_pass": full,
        "full_pending": not full,
        "final_design_approved": False,
        "engineering_review_required": True,
        "live_8010_mutated": False,
        "scheduler_8002_mutated": False,
        "task_mutation_performed": False,
        "canonical_dataset_mutated": False,
        "constraint_version": deadline_design.EXPECTED_CONSTRAINT_VERSION,
        "hard_spec": hard_spec,
        "hard_spec_sha256": deadline_design._canonical_sha256(hard_spec),
        "candidate_identity_sha256": identity,
        "candidate_digest": digest,
        "candidate": candidate,
        "candidate_sha256": deadline_design._canonical_sha256(candidate),
        "evidence": evidence,
        "tim_contract": dict(deadline_design.EXPECTED_TIM),
        "cooling_boundary_contract": {
            "variant": variant_label,
            "fan_config": "dual",
            "fan_velocity_m_s": fan_m_s,
            "plate_temp_C": 50.0,
            "air_temp_C": 50.0,
            "core_pad_thickness_mm": pad_mm,
            "winding_pad_thickness_mm": pad_mm,
            "core_pad_conductivity_W_mK": 3.0,
            "winding_pad_conductivity_W_mK": 3.0,
            "core_pad_material_policy": deadline_design.EXPECTED_TIM[
                "material_policy"
            ],
            "winding_pad_material_policy": deadline_design.EXPECTED_TIM[
                "material_policy"
            ],
            "native_readback_contract_version": deadline_design.EXPECTED_TIM[
                "native_readback_contract_version"
            ],
            "native_readback_required": True,
            "native_readback_attested": 1,
            "native_thermal_conductivity_W_mK": 3.0,
            "native_electrical_conductivity_S_m": 0.0,
            "fan_auxiliary_power_included_in_reported_loss": False,
        },
        "authority": {
            "read_only_ui_publication": True,
            "local_gui_geometry_build": True,
            "local_gui_solve": full,
            "automatic_production_promotion": False,
            "automatic_scheduler_submission": False,
            "engineering_signoff": False,
        },
    }
    publication["tim_contract"].update({
        "core_pad_thickness_mm": pad_mm,
        "winding_pad_thickness_mm": pad_mm,
    })
    path = tmp_path / "deadline-design-publication.json"
    _write(path, publication)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_loads_hash_authenticated_standard_design(tmp_path):
    path, sha256 = _package(tmp_path)

    loaded = deadline_design.load_publication(path, sha256)

    assert loaded["candidate"]["spec_status"] == "pass"
    assert loaded["candidate"]["gui_build_eligible"] is True
    assert loaded["candidate"]["gui_solve_eligible"] is False
    assert loaded["publication"]["full_pending"] is True
    assert loaded["generation"]["source_kind"] == "deadline_validated_design"
    assert loaded["payload"]["valid_candidate_count"] == 1
    assert loaded["payload"]["warnings"] == [
        "Full/fine actual FEA validation is still pending."
    ]
    assert (
        loaded["candidate"]["constraints"]["resonance"]["operator"]
        == ">="
    )
    assert "≥ 15 kHz" in loaded["generation"]["label"]


def test_adapter_rejects_old_max_only_hard_spec(tmp_path):
    path, _ = _package(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hard_spec"].pop("resonance_min_Hz")
    payload["hard_spec"]["resonance_max_Hz"] = 15_000.0
    payload["hard_spec_sha256"] = deadline_design._canonical_sha256(
        payload["hard_spec"]
    )
    _write(path, payload)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="hard-spec identity drifted",
    ):
        deadline_design.load_publication(path, sha256)


@pytest.mark.parametrize(
    ("resonance_Hz", "accepted"),
    ((15_000.0, True), (14_999.999, False)),
)
def test_adapter_enforces_inclusive_half_lm_minimum(
    tmp_path, resonance_Hz, accepted
):
    path, _ = _package(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_ref = payload["evidence"]["standard"]["result"]
    result_path = Path(result_ref["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["actual_hard_spec_gate"][
        "half_magnetizing_resonance_Hz"
    ] = resonance_Hz
    payload["evidence"]["standard"]["result"] = _write(
        result_path, result
    )
    candidate = payload["candidate"]
    candidate["pred_f_res_min_screen_Hz"] = resonance_Hz
    candidate["resonance_minimum_required_Hz"] = 15_000.0
    candidate["resonance_margin_Hz"] = resonance_Hz - 15_000.0
    candidate["constraints"]["resonance"].update({
        "value": resonance_Hz,
        "limit": 15_000.0,
        "pass": True,
        "direction": "minimum",
        "operator": ">=",
        "minimum_Hz": 15_000.0,
    })
    payload["candidate_sha256"] = deadline_design._canonical_sha256(
        candidate
    )
    _write(path, payload)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    if accepted:
        assert deadline_design.load_publication(
            path, sha256
        )["candidate"]["pred_f_res_min_screen_Hz"] == 15_000.0
    else:
        with pytest.raises(
            deadline_design.DeadlineDesignError,
            match=">=15 kHz",
        ):
            deadline_design.load_publication(path, sha256)


def test_adapter_rejects_maximum_direction_candidate_constraint(tmp_path):
    path, _ = _package(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    resonance = payload["candidate"]["constraints"]["resonance"]
    resonance["direction"] = "maximum"
    resonance["operator"] = "<"
    resonance["maximum_Hz"] = resonance.pop("minimum_Hz")
    payload["candidate_sha256"] = deadline_design._canonical_sha256(
        payload["candidate"]
    )
    _write(path, payload)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="candidate resonance",
    ):
        deadline_design.load_publication(path, sha256)


def test_loads_full_verified_design(tmp_path):
    path, sha256 = _package(tmp_path, full=True)

    loaded = deadline_design.load_publication(path, sha256)

    assert loaded["publication"]["engineering_handoff_ready"] is True
    assert loaded["publication"]["full_pending"] is False
    assert loaded["payload"]["warnings"] == []
    assert loaded["generation"]["state"] == "actual_standard_and_full_pass"
    assert loaded["candidate"]["gui_solve_eligible"] is True


def test_publication_replacement_fails_closed(tmp_path):
    path, sha256 = _package(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate"]["volume_L"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="publication SHA-256 mismatch",
    ):
        deadline_design.load_publication(path, sha256)


def test_nested_result_replacement_fails_closed(tmp_path):
    path, sha256 = _package(tmp_path)
    publication = json.loads(path.read_text(encoding="utf-8"))
    result_path = publication["evidence"]["standard"]["result"]["path"]
    with open(result_path, "a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="result SHA-256 mismatch",
    ):
        deadline_design.load_publication(path, sha256)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("thermal_pad_native_readback_attested", 0),
        ("thermal_pad_native_thermal_conductivity_W_mK", 0.2),
        ("thermal_pad_native_electrical_conductivity_S_m", 1.0),
        ("thermal_pad_native_readback_contract_version", "stale-v0"),
    ),
)
def test_native_tim_result_drift_is_rejected(tmp_path, field, value):
    path, _ = _package(tmp_path)
    publication = json.loads(path.read_text(encoding="utf-8"))
    result_ref = publication["evidence"]["standard"]["result"]
    result_path = Path(result_ref["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["result"][field] = value
    publication["evidence"]["standard"]["result"] = _write(
        result_path, result
    )
    _write(path, publication)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="solver/TIM result echo drifted",
    ):
        deadline_design.load_publication(path, sha256)


def test_old_solver_result_revision_is_rejected(tmp_path):
    path, _ = _package(tmp_path)
    publication = json.loads(path.read_text(encoding="utf-8"))
    result_ref = publication["evidence"]["standard"]["result"]
    result_path = Path(result_ref["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["result"]["git_hash"] = (
        "47191a5146e087c10bdcdfaab72043fce5d65b73"
    )
    publication["evidence"]["standard"]["result"] = _write(
        result_path, result
    )
    _write(path, publication)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="solver/TIM result echo drifted",
    ):
        deadline_design.load_publication(path, sha256)


def test_solve_authority_cannot_be_enabled_in_package(tmp_path):
    path, _ = _package(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate"]["gui_solve_eligible"] = True
    payload["candidate_sha256"] = deadline_design._canonical_sha256(
        payload["candidate"]
    )
    _write(path, payload)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="candidate authority drifted",
    ):
        deadline_design.load_publication(path, sha256)


def test_local_gui_one_megawatt_full_profile_drift_is_rejected(tmp_path):
    path, _ = _package(tmp_path, full=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload["candidate"]["local_gui_solver_contract"]
    contract["common_solver_parameters"]["P_target"] = 0.0
    contract["mode_solver_parameters"]["full"][
        "thermal_symmetry"
    ] = "eighth"
    payload["candidate_sha256"] = deadline_design._canonical_sha256(
        payload["candidate"]
    )
    _write(path, payload)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(
        deadline_design.DeadlineDesignError,
        match="local GUI solver contract drifted",
    ):
        deadline_design.load_publication(path, sha256)


def test_artifact_service_exposes_selectable_generation(
    tmp_path, monkeypatch
):
    path, sha256 = _package(tmp_path / "package", full=True)
    monkeypatch.setenv(deadline_design.PATH_ENV, str(path))
    monkeypatch.setenv(deadline_design.SHA256_ENV, sha256)

    class Scheduler:
        base_url = "http://127.0.0.1:8002"

        @staticmethod
        def mft_pipeline_status():
            return {}

    service = readers.ArtifactService(
        tmp_path / "regression",
        scheduler=Scheduler(),
        record_runtime=False,
    )

    snapshot = service._continuous_nsga2()

    assert snapshot is not None
    assert snapshot["available"] is True
    assert snapshot["source_kind"] == "deadline_validated_design"
    assert snapshot["candidate_count"] == 1
    assert snapshot["display_candidate_count"] == 1
    assert snapshot["valid_candidate_count"] == 1
    assert snapshot["summary"]["valid_candidate_count"] == 1
    assert len(snapshot["candidates"]) == 1
    generation = next(
        item for item in snapshot["pareto_generations"]
        if item["source_kind"] == "deadline_validated_design"
    )
    assert snapshot["selected_generation_id"] == generation["id"]
    detail = service.nsga2_generation(generation["id"])
    assert detail["integrity_verified"] is True
    assert detail["candidates"][0]["gui_launch_eligible"] is True
    assert (
        service.deadline_design_candidate(
            detail["candidates"][0]["id"]
        )["publication_sha256"]
        == sha256
    )


@pytest.mark.parametrize(
    ("variant_label", "pad_mm", "fan_m_s"),
    (
        ("timk3", 1.0, 1.5),
        ("tim05k3", 0.5, 1.5),
        ("timk3fan3", 1.0, 3.0),
        ("tim05k3fan3", 0.5, 3.0),
        ("tim1k3fan6", 1.0, 6.0),
        ("tim1k3fan7", 1.0, 7.0),
        ("tim1k3fan8", 1.0, 8.0),
        ("tim1k3fan9", 1.0, 9.0),
    ),
)
def test_adapter_accepts_only_exact_allowlisted_cooling_variants(
    tmp_path, variant_label, pad_mm, fan_m_s
):
    path, sha256 = _package(
        tmp_path, full=True, variant_label=variant_label
    )

    loaded = deadline_design.load_publication(path, sha256)

    candidate = loaded["candidate"]
    assert candidate["core_thermal_pad_thickness_mm"] == pad_mm
    assert candidate["fan_velocity_m_s"] == fan_m_s
    assert candidate["fan_config"] == "dual"
