import hashlib
import json
from datetime import timedelta
from pathlib import Path
from unittest import mock
from urllib.error import URLError

import pandas as pd
import pytest

from regression_260707.model_targets import (
    CORE_REGION_TEMPERATURE_TARGETS,
    SURROGATE_TEMPERATURE_TARGETS,
)
from regression_260707.monitoring.readers import (
    ArtifactService,
    CURRENT_PHYSICS_DATA_REVISION,
    CURRENT_SOLVER_REVISION,
    RuntimeRecorder,
    SafeArtifactCache,
    SchedulerReader,
    TARGET_META,
    TEMPERATURE_TARGETS,
    _campaign_frame_summary,
    _simulation_timing_summary,
    _zero_aware_percentage_metrics,
)


def _install_checkpoint_fixture(
        campaign_root: Path, *, metrics_hash_valid: bool = True,
        parity_profile_sha: str = "profile-sha") -> tuple[Path, Path]:
    pointer = campaign_root / "training" / "registry" / "current.json"
    pointer.unlink()
    run_root = campaign_root / "training" / "checkpoint_runs" / "current"
    metrics_path = run_root / "checkpoint_metrics" / "threshold_000500_attempt_000001.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "schema_version": 1,
        "completed_at": "2026-07-11T02:40:00+09:00",
        "checkpoint": 500,
        "dataset_sha256": "snapshot-sha",
        "profile_sha256": "profile-sha",
        "strict_full_rows": 100,
        "metrics": [
            {
                "time": "2026-07-11 02:40:00", "target": "Llt_phys",
                "n": 100, "r2": .95, "rmse": .15, "mape_pct": 1.0,
                "p90_ape_pct": 2.0, "slice": "global",
            },
            {
                "time": "2026-07-11 02:40:00", "target": "Llt_phys",
                "n": 100, "r2": .90, "rmse": .25, "mape_pct": 1.5,
                "p90_ape_pct": 3.0, "slice": "Llt20-40",
            },
        ],
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if not metrics_hash_valid:
        metrics_hash = "0" * 64
    state = {
        "schema_version": 2,
        "completed": [{
            "threshold": 500,
            "actual_strict_full_rows": 100,
            "snapshot_sha256": "snapshot-sha",
            "metrics_result": str(metrics_path),
            "metrics_result_sha256": metrics_hash,
            "profile_sha256": "profile-sha",
            "activation_minimum_strict_full_rows": 3000,
            "completed_at": "2026-07-11T02:41:00+09:00",
            "kind": "metrics_only",
        }],
        "identity": {
            "solver_revision": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "library_revision": "c" * 40,
            "profile_sha256": "profile-sha",
            "activation_minimum_strict_full_rows": 3000,
        },
    }
    state_path = run_root / "checkpoint_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    parity_path = metrics_path.with_suffix(".parity.json")
    parity_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "checkpoint_cv_oof_parity",
        "checkpoint": 500,
        "dataset_sha256": "snapshot-sha",
        "profile_sha256": parity_profile_sha,
        "strict_full_rows": 100,
        "prediction_kind": "out_of_fold",
        "cv": {"n_splits": 5, "shuffle": True, "seed": 42},
        "max_pairs_per_target": 400,
        "targets": {
            "Llt_phys": {
                "n": 100,
                "sample_count": 2,
                "sampling": {"method": "evenly_spaced_position", "limit": 400},
                "pairs": [
                    {"row_position": 0, "row_index": 10, "actual": 27.0, "predicted": 27.1},
                    {"row_position": 99, "row_index": 20, "actual": 28.0, "predicted": 27.9},
                ],
            },
        },
    }), encoding="utf-8")
    return metrics_path, parity_path


def test_data_counts_quality_throughput_and_revision(artifact_service):
    data = artifact_service.data()
    assert data["raw_total_rows"] == 2
    assert data["total_rows"] == 1
    assert data["em_valid_rows"] == 2
    assert data["thermal_valid_rows"] == 1
    assert data["complete_rows"] == 1
    assert data["throughput_1h"] == 1
    assert data["added_24h"] == 1
    assert data["collector"]["no_data_tasks"] == 1
    assert data["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert data["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert data["pinned_library_revision"] == "c" * 40
    assert data["rows_not_latest_revision"] == 1
    assert data["eta_3000"] is not None
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["window_rows"] == 2
    assert timing["window_limit_rows"] == 100
    assert timing["stages"]["matrix"] == {
        "source_field": "time_matrix",
        "sample_count": 2,
        "mean_seconds": 450.0,
        "median_seconds": 450.0,
    }
    assert timing["stages"]["loss"]["mean_seconds"] == 1750.0
    assert timing["stages"]["icepak"]["median_seconds"] == 1100.0
    assert timing["stages"]["total"]["mean_seconds"] == 3300.0


def test_data_separates_raw_rows_from_zero_b171_pinned_rows(
        campaign_root, artifact_service):
    manifest_path = Path(campaign_root, "data", "dataset", "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 436
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    strict_path = Path(campaign_root, "training", "strict_data_status.json")
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict["strict_em_rows"] = 0
    strict["strict_full_rows"] = 0
    strict_path.write_text(json.dumps(strict), encoding="utf-8")

    service = ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        clock=artifact_service.clock,
        record_runtime=False,
    )
    data = service.data()

    assert data["raw_total_rows"] == 436
    assert data["total_rows"] == 0
    assert data["em_valid_rows"] == 0
    assert data["thermal_valid_rows"] == 0
    assert data["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert data["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"


def test_models_include_planned_missing_targets_and_metrics(artifact_service):
    payload = artifact_service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    assert lookup["Llt_phys"]["r2"] == .91
    assert lookup["Llt_phys"]["trained"] is True
    assert lookup["Llt_phys"]["evaluated"] is True
    assert lookup["Llt_phys"]["evaluation_kind"] == "active_registry"
    assert lookup["Llt_phys"]["parity_available"] is False
    assert lookup["P_winding_total"]["status"] == "attention"
    assert lookup["Tprobe_Tx_leeward_max"]["status"] == "not_trained"
    assert lookup["Llt_phys"]["history"][-1]["n"] == 100


def test_models_use_only_sha_authorized_checkpoint_metrics_and_parity(
        campaign_root, artifact_service):
    metrics_path, parity_path = _install_checkpoint_fixture(campaign_root)

    payload = artifact_service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    model = lookup["Llt_phys"]

    assert payload["trained_count"] == 0
    assert payload["evaluated_count"] == 1
    assert payload["latest_checkpoint"] == 500
    assert payload["activation_minimum_strict_full_rows"] == 3000
    assert payload["checkpoint_evaluated_at"] == "2026-07-11T02:40:00+09:00"
    assert payload["activation_state"] == "preactivation_checkpoint"
    assert payload["source_kind"] == "checkpoint_cv"
    assert Path(payload["checkpoint_source"]).resolve() == metrics_path.resolve()
    assert not any("model pointer is unavailable" in warning
                   for warning in payload["warnings"])
    assert "배포 모델로 취급하지 않습니다" in payload["quality_note"]

    assert model["status"] == "checkpoint"
    assert model["trained"] is False
    assert model["evaluated"] is True
    assert model["deployable"] is False
    assert model["evaluation_kind"] == "checkpoint_cv"
    assert model["checkpoint"] == 500
    assert model["n_used"] == 100
    assert model["r2"] == .95
    assert model["rmse"] == .15
    assert model["mape_pct"] == 1.0
    assert model["parity_available"] is True
    assert model["parity_sample_count"] == 2
    assert Path(model["parity_source"]).resolve() == parity_path.resolve()
    assert "pairs" not in model

    parity = artifact_service.model_parity("Llt_phys")
    assert parity["available"] is True
    assert parity["checkpoint"] == 500
    assert parity["n"] == 100
    assert parity["sample_count"] == 2
    assert parity["pairs"] == [
        {"row_position": 0, "row_index": 10, "actual": 27.0, "predicted": 27.1},
        {"row_position": 99, "row_index": 20, "actual": 28.0, "predicted": 27.9},
    ]

    dashboard = artifact_service.dashboard(record=False)
    model_stage = next(
        stage for stage in dashboard["status"]["stages"]
        if stage["key"] == "models"
    )
    assert model_stage["state"] == "waiting"
    assert f"checkpoint 500 CV 1/{payload['target_count']}" in model_stage["detail"]
    assert "활성화 1/3,000" in model_stage["detail"]
    assert not any(
        warning.startswith("미학습 모델:")
        for warning in dashboard["status"]["warnings"]
    )


def test_models_reject_checkpoint_with_bad_metrics_hash(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root, metrics_hash_valid=False)

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")

    # learning_curve.csv still exists, but it is history only and cannot make
    # an unauthorised checkpoint appear in the table.
    assert model["history"]
    assert model["trained"] is False
    assert model["evaluated"] is False
    assert model["status"] == "not_trained"
    assert payload["evaluated_count"] == 0
    assert payload["latest_checkpoint"] is None
    assert any("checkpoint metrics hash mismatch" in warning
               for warning in payload["warnings"])


def test_bad_parity_identity_keeps_metrics_but_returns_empty_parity(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root, parity_profile_sha="wrong-profile")

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")

    assert model["status"] == "checkpoint"
    assert model["evaluated"] is True
    assert model["r2"] == .95
    assert model["parity_available"] is False
    assert model["parity_sample_count"] == 0
    assert any("checkpoint parity identity mismatch" in warning
               for warning in payload["warnings"])
    parity = artifact_service.model_parity("Llt_phys")
    assert parity["available"] is False
    assert parity["pairs"] == []
    assert any("checkpoint parity identity mismatch" in warning
               for warning in parity["warnings"])


def test_checkpoint_state_must_match_current_strict_solver_library_identity(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root)
    state_path = (
        campaign_root / "training" / "checkpoint_runs" / "current" /
        "checkpoint_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["identity"]["solver_revision"] = "f" * 40
    state_path.write_text(json.dumps(state), encoding="utf-8")

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")
    assert payload["latest_checkpoint"] is None
    assert payload["evaluated_count"] == 0
    assert model["evaluated"] is False


def test_missing_active_pointer_warns_once_activation_floor_is_reached(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root)

    payload = artifact_service.models(current_data_count=3000)

    assert payload["activation_state"] == "activation_due"
    assert any("accepted schema-v2 model pointer is unavailable" in warning
               for warning in payload["warnings"])


def test_core_region_temperature_models_and_predictions_are_independent(
        artifact_service):
    assert TEMPERATURE_TARGETS == SURROGATE_TEMPERATURE_TARGETS
    assert tuple(TEMPERATURE_TARGETS[-4:]) == (
        "Tprobe_core_center_max",
        *CORE_REGION_TEMPERATURE_TARGETS,
    )

    model_payload = artifact_service.models(current_data_count=100)
    model_lookup = {item["target"]: item for item in model_payload["models"]}
    expected_labels = {
        "Tprobe_core_center_max": "코어 최대 온도(3영역 최대)",
        "Tprobe_core_center_leg_max": "코어 중앙 레그 최대 온도",
        "Tprobe_core_side_leg_max": "코어 사이드 레그 최대 온도",
        "Tprobe_core_top_yoke_max": "코어 상부 요크 최대 온도",
    }
    for target, label in expected_labels.items():
        assert TARGET_META[target]["label"] == label
        assert model_lookup[target]["label"] == label
        assert model_lookup[target]["status"] == "not_trained"

    predictions = {
        target: 80.0 + index
        for index, target in enumerate(TEMPERATURE_TARGETS)
    }
    predictions["Tprobe_core_center_max"] = 95.0
    predictions["Tprobe_core_center_leg_max"] = 91.0
    predictions["Tprobe_core_side_leg_max"] = 98.0
    predictions["Tprobe_core_top_yoke_max"] = 101.0
    candidate = artifact_service._candidate(
        {f"pred_{target}": value for target, value in predictions.items()},
        round_number=2,
        index=0,
    )

    assert candidate["pred_temperatures_C"] == predictions
    assert candidate["pred_max_temperature_C"] == 101.0
    assert candidate["constraints"]["temperature"]["pass"] is False


def test_nsga_and_verification_are_joined(artifact_service):
    nsga = artifact_service.nsga2()
    assert nsga["round"] == 2
    assert nsga["candidate_count"] == 2
    assert nsga["summary"]["min_volume_L"] == 500
    assert nsga["comparison"]["min_volume_change_L"] == -100
    assert nsga["candidates"][0]["id"] == "r02-0000"
    assert nsga["candidates"][0]["spec_status"] == "unknown"  # temperature models are absent
    assert nsga["candidates"][0]["constraints"]["bfield"]["pass"] is True
    assert nsga["candidates"][0]["B_design_analytic_T"] == 1.0
    assert nsga["candidates"][0]["diagnostic_pred_B_max_core"] == 2.7
    assert nsga["candidates"][0]["report"]["cw1_conductor_thickness_mm"] == 5

    verification = artifact_service.verification(nsga)
    assert verification["counts"]["coverage"] == 1.0
    assert verification["standard_candidates"][0]["evaluation"]["computed_status"] == "pass"
    assert verification["standard_candidates"][0]["evaluation"]["timing_seconds"] == {
        "matrix": 353.31,
        "loss": 1720.78,
        "icepak": 1039.83,
        "total": 3113.92,
    }
    assert verification["final"]["status"] == "pass"
    assert verification["final"]["evaluation"]["checks"]["full_model"]["pass"] is True
    assert verification["final"]["evaluation"]["timing_seconds"]["total"] == 3113.92


def test_fea_timings_fail_closed_without_nonnegative_finite_result_fields(artifact_service):
    evaluation = artifact_service._evaluate_fea({
        "time_matrix": "12.5",
        "time_loss": -1,
        "time_thermal": "nan",
        "time": True,
    })
    assert evaluation["timing_seconds"] == {
        "matrix": 12.5,
        "loss": None,
        "icepak": None,
        "total": None,
    }

    missing = artifact_service._evaluate_fea({})
    assert missing["timing_seconds"] == {
        "matrix": None,
        "loss": None,
        "icepak": None,
        "total": None,
    }


def test_simulation_timing_summary_uses_only_most_recent_rows():
    summary = _simulation_timing_summary([
        {"saved_at": "2026-07-11 00:00:00", "time_matrix": "100"},
        {"saved_at": "2026-07-11 02:00:00", "time_matrix": "300"},
        {"saved_at": "2026-07-11 01:00:00", "time_matrix": "200"},
    ], limit=2)

    assert summary["window_rows"] == 2
    assert summary["stages"]["matrix"]["sample_count"] == 2
    assert summary["stages"]["matrix"]["mean_seconds"] == 250.0
    assert summary["stages"]["matrix"]["median_seconds"] == 250.0
    assert summary["stages"]["loss"]["sample_count"] == 0
    assert summary["stages"]["loss"]["mean_seconds"] is None


def test_zero_aware_percentage_metrics_exclude_structural_zero_targets():
    metrics = _zero_aware_percentage_metrics(
        [0.0, 10.0, 20.0],
        [5.0, 11.0, 18.0],
    )

    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["p90_ape_pct"] == pytest.approx(10.0)
    assert metrics["mape_n"] == 2
    assert metrics["mape_excluded_zero_count"] == 1
    assert metrics["mape_valid_pair_count"] == 3
    assert metrics["mape_zero_abs_tolerance"] == 1e-9

    all_zero = _zero_aware_percentage_metrics([0.0], [123.0])
    assert all_zero["mape_pct"] is None
    assert all_zero["p90_ape_pct"] is None
    assert all_zero["mape_n"] == 0
    assert all_zero["mape_excluded_zero_count"] == 1


def test_campaign_frame_summary_separates_v32_cohort_and_physics_panels():
    from .conftest import FIXED_NOW

    current = {
        "git_hash": CURRENT_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "saved_at": (FIXED_NOW - timedelta(minutes=20)).isoformat(),
        "thermal_core_conductivity_model": (
            "anisotropic_wound_rule_of_mixtures_v1"
        ),
        "thermal_core_k_inplane": 18.0,
        "thermal_core_k_throughstack": 2.0,
        "core_lamination_factor": 0.85,
    }
    frame = pd.DataFrame([
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 1e-9,
            "C_rx_rx_F": 4e-9,
            "C_tx_rx_F": 0.2e-9,
            "f_res_tx_self_Hz": 100_000.0,
            "f_res_rx_self_Hz": 200_000.0,
            "f_res_interwinding_Hz": 50_000.0,
            "winding_flux_linkage_readback_status": "available",
            "winding_flux_linkage_readback_applicable": 1,
            "winding_flux_linkage_readback_available": 1,
            "winding_flux_linkage_readback_passed": 1,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 3e-9,
            "C_rx_rx_F": 8e-9,
            "C_tx_rx_F": 0.6e-9,
            "f_res_tx_self_Hz": 300_000.0,
            "f_res_rx_self_Hz": 400_000.0,
            "f_res_interwinding_Hz": 150_000.0,
            "winding_flux_linkage_readback_status": "unavailable",
            "winding_flux_linkage_readback_applicable": 1,
            "winding_flux_linkage_readback_available": 0,
            "winding_flux_linkage_readback_passed": 0,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 0,
        },
        {
            **current,
            "_strict_valid_em": True,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": "thermal:required_group_missing",
            "cap_on": 1,
            "winding_flux_linkage_readback_status": "available",
            "winding_flux_linkage_readback_available": 1,
        },
        {
            **current,
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": "em:matrix_invalid",
            "cap_on": 1,
            "winding_flux_linkage_readback_status": "unavailable",
            "winding_flux_linkage_readback_available": 0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "_strict_invalid_reasons": (
                "untrusted_provenance:solver_revision_mismatch"
            ),
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 10.0,
            "thermal_core_k_throughstack": 10.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 10.0,
            "thermal_core_k_throughstack": 10.0,
        },
    ])
    history = {
        (CURRENT_SOLVER_REVISION, CURRENT_PHYSICS_DATA_REVISION): [
            (FIXED_NOW - timedelta(hours=1, minutes=5), 1),
        ],
    }

    summary = _campaign_frame_summary(frame, FIXED_NOW, history)

    assert len(summary["cohorts"]) == 2
    cohort = summary["cohorts"][0]
    assert cohort == {
        "git_hash": CURRENT_SOLVER_REVISION,
        "git_hash_short": CURRENT_SOLVER_REVISION[:10],
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "current": True,
        "raw_rows": 5,
        "strict_em_rows": 4,
        "strict_full_rows": 3,
        "growth_rate_per_hour": 2.0,
    }
    legacy = summary["cohorts"][1]
    assert legacy["current"] is False
    assert legacy["raw_rows"] == 2
    assert legacy["strict_em_rows"] == 0
    assert legacy["strict_full_rows"] == 0

    electrostatic = summary["electrostatic"]
    assert electrostatic["cohort_basis"] == "current_v3.2_strict_full"
    assert electrostatic["cohort_rows"] == 3
    assert electrostatic["cap_stage_present_rows"] == 2
    assert electrostatic["cap_stage_absent_rows"] == 1
    assert electrostatic["cap_stage_unknown_rows"] == 0
    expected_capacitance = {
        "tx_tx": ("C_tx_tx_F", 1.0, 2.0, 3.0),
        "rx_rx": ("C_rx_rx_F", 4.0, 6.0, 8.0),
        "tx_rx": ("C_tx_rx_F", 0.2, 0.4, 0.6),
    }
    for key, (source, minimum, median, maximum) in expected_capacitance.items():
        metric = electrostatic["capacitance"][key]
        assert metric["source_column"] == source
        assert metric["sample_count"] == 2
        assert metric["min_nF"] == pytest.approx(minimum)
        assert metric["median_nF"] == pytest.approx(median)
        assert metric["max_nF"] == pytest.approx(maximum)
    expected_resonance = {
        "tx_self": ("f_res_tx_self_Hz", 100.0, 200.0, 300.0),
        "rx_self": ("f_res_rx_self_Hz", 200.0, 300.0, 400.0),
        "interwinding": (
            "f_res_interwinding_Hz", 50.0, 100.0, 150.0
        ),
    }
    for key, (source, minimum, median, maximum) in expected_resonance.items():
        metric = electrostatic["resonance"][key]
        assert metric["source_column"] == source
        assert metric["sample_count"] == 2
        assert metric["min_kHz"] == pytest.approx(minimum)
        assert metric["median_kHz"] == pytest.approx(median)
        assert metric["max_kHz"] == pytest.approx(maximum)

    thermal = {
        item["model"]: item for item in summary["thermal_models"]["models"]
    }
    assert summary["thermal_models"]["tagged_rows"] == 7
    assert thermal["anisotropic_wound_rule_of_mixtures_v1"]["count"] == 5
    assert thermal["isotropic_legacy"]["count"] == 2
    assert thermal["anisotropic_wound_rule_of_mixtures_v1"][
        "thermal_core_k_inplane"
    ]["median"] == pytest.approx(18.0)

    quarantine = summary["quarantine"]
    assert quarantine["current"]["rows"] == 2
    assert quarantine["legacy"]["rows"] == 2
    current_reasons = {
        item["reason"]: item["count"]
        for item in quarantine["current"]["reasons"]
    }
    legacy_reasons = {
        item["reason"]: item["count"]
        for item in quarantine["legacy"]["reasons"]
    }
    assert current_reasons == {
        "em:matrix_invalid": 1,
        "thermal:required_group_missing": 1,
    }
    assert not any("solver_revision_mismatch" in reason
                   for reason in current_reasons)
    assert legacy_reasons[
        "untrusted_provenance:solver_revision_mismatch"
    ] == 2

    metadata = summary["current_cohort_metadata"]
    assert metadata["core_lamination_factor"] == {
        "source_column": "core_lamination_factor",
        "sample_count": 5,
        "min": 0.85,
        "median": 0.85,
        "max": 0.85,
    }
    readback = metadata["winding_flux_linkage_readback"]
    assert readback["cohort_rows"] == 5
    assert readback["available_rows"] == 2
    assert readback["unavailable_rows"] == 2
    assert readback["missing_rows"] == 1
    assert {item["status"]: item["count"] for item in readback["statuses"]} == {
        "available": 2,
        "missing": 1,
        "unavailable": 2,
    }


def test_campaign_frame_summary_tolerates_missing_v32_columns_and_uses_flags():
    from .conftest import FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": CURRENT_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
        },
        {
            "git_hash": CURRENT_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        },
    ])

    summary = _campaign_frame_summary(frame, FIXED_NOW)

    cohort = summary["cohorts"][0]
    assert cohort["raw_rows"] == 2
    assert cohort["strict_em_rows"] == 1
    assert cohort["strict_full_rows"] == 1
    assert cohort["growth_rate_per_hour"] == 0.0
    electrostatic = summary["electrostatic"]
    assert electrostatic["available"] is False
    assert electrostatic["cohort_rows"] == 1
    assert electrostatic["cap_stage_present_rows"] == 0
    assert electrostatic["cap_stage_absent_rows"] == 0
    assert electrostatic["cap_stage_unknown_rows"] == 1
    for metric in (
        *electrostatic["capacitance"].values(),
        *electrostatic["resonance"].values(),
    ):
        assert metric["sample_count"] == 0
        assert all(value is None for key, value in metric.items()
                   if key.startswith(("min_", "median_", "max_")))
    assert summary["thermal_models"] == {
        "available": False,
        "total_rows": 2,
        "tagged_rows": 0,
        "missing_rows": 2,
        "models": [],
    }
    assert summary["current_cohort_metadata"]["core_lamination_factor"][
        "sample_count"
    ] == 0
    assert summary["current_cohort_metadata"][
        "winding_flux_linkage_readback"
    ]["missing_rows"] == 2


def test_artifact_service_data_reads_lossless_campaign_parquet(campaign_root):
    from .conftest import DummyScheduler, FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": CURRENT_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": FIXED_NOW.isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 2e-9,
            "C_rx_rx_F": 4e-9,
            "C_tx_rx_F": 0.5e-9,
            "f_res_tx_self_Hz": 100_000.0,
            "f_res_rx_self_Hz": 200_000.0,
            "f_res_interwinding_Hz": 50_000.0,
            "thermal_core_conductivity_model": (
                "anisotropic_wound_rule_of_mixtures_v1"
            ),
            "thermal_core_k_inplane": 18.0,
            "thermal_core_k_throughstack": 2.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "thermal_core_conductivity_model": "isotropic_legacy",
        },
    ])
    parquet_path = campaign_root / "data" / "dataset" / "train.parquet"
    frame.to_parquet(parquet_path, index=False)
    service = ArtifactService(
        campaign_root,
        scheduler=DummyScheduler(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )

    def passthrough_audit(result, expected_library_revision):
        assert result.path == str(parquet_path)
        assert result.value is not None
        return result.value, None

    with mock.patch.object(
        service,
        "_audited_campaign_frame",
        side_effect=passthrough_audit,
    ):
        data = service.data()

    assert data["source"]["campaign_rows"] == str(parquet_path)
    assert data["cohorts"][0]["current"] is True
    assert data["cohorts"][0]["raw_rows"] == 1
    assert data["cohorts"][0]["strict_full_rows"] == 1
    assert data["electrostatic"]["cap_stage_present_rows"] == 1
    assert data["electrostatic"]["capacitance"]["tx_tx"][
        "median_nF"
    ] == pytest.approx(2.0)
    assert data["thermal_models"]["tagged_rows"] == 2
    assert data["quarantine"]["legacy"]["rows"] == 1


def test_runtime_recorder_recovers_from_winerror5_without_temp_collision(
        tmp_path, artifact_service):
    recorder = RuntimeRecorder(tmp_path / "runtime", min_interval_seconds=0)
    dashboard = artifact_service.dashboard(record=False)
    denied = PermissionError(13, "RaiDrive rename denied")
    denied.winerror = 5

    with mock.patch(
            "regression_260707.monitoring.readers.os.replace",
            side_effect=denied) as replace, mock.patch(
            "regression_260707.monitoring.readers.time.sleep"):
        recorder.record(dashboard)

    assert replace.call_count == 5
    snapshot = json.loads(recorder.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    history = recorder.history()["entries"]
    assert history[-1]["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    assert history[-1]["data"]["pinned_library_revision"] == dashboard["data"]["pinned_library_revision"]
    assert list(recorder.directory.glob(".monitor_snapshot.json.*.tmp")) == []


def test_runtime_snapshot_failure_warns_but_history_still_appends(
        campaign_root, artifact_service):
    service = ArtifactService(
        campaign_root,
        scheduler=artifact_service.scheduler,
        clock=artifact_service.clock,
        record_runtime=True,
    )
    with mock.patch.object(
            service.recorder, "_write_snapshot",
            side_effect=PermissionError(13, "snapshot blocked")):
        dashboard = service.dashboard()

    assert any("snapshot write failed" in warning for warning in dashboard["status"]["warnings"])
    history = service.recorder.history()["entries"]
    assert history[-1]["time"] == dashboard["generated_at"]
    assert history[-1]["data"]["pinned_solver_revision"] == dashboard["data"]["pinned_revision"]
    assert history[-1]["data"]["pinned_library_revision"] == dashboard["data"]["pinned_library_revision"]


def test_final_display_fails_closed_on_missing_or_negative_wcp_loss(
        artifact_service, campaign_root):
    artifact = json.loads(
        Path(campaign_root, "verify", "results", "final_verification.json")
        .read_text(encoding="utf-8")
    )
    result = dict(artifact["result"])
    result["P_wcp_total"] = -1.0
    negative = artifact_service._evaluate_fea(result, require_full_model=True)
    assert negative["computed_status"] == "fail"
    assert negative["checks"]["loss_components"]["pass"] is False

    result.pop("P_wcp_total")
    missing = artifact_service._evaluate_fea(result, require_full_model=True)
    assert missing["computed_status"] == "unknown"
    assert missing["checks"]["loss_components"]["pass"] is None


def test_corrupt_json_retains_last_good_value(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"stage": "WAIT"}', encoding="utf-8")
    cache = SafeArtifactCache()
    first = cache.json(path, {})
    assert first.value["stage"] == "WAIT"
    path.write_text("{broken", encoding="utf-8")
    second = cache.json(path, {})
    assert second.value["stage"] == "WAIT"
    assert "마지막 정상" in second.warning or "읽기 실패" in second.warning


def test_parquet_reader_retains_last_good_synthetic_frame(tmp_path):
    path = tmp_path / "train.parquet"
    expected = pd.DataFrame([{
        "git_hash": CURRENT_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "C_tx_tx_F": 1e-9,
    }])
    expected.to_parquet(path, index=False)
    cache = SafeArtifactCache()

    first = cache.parquet(path)
    assert first.exists is True
    assert first.warning is None
    assert first.value.to_dict("records") == expected.to_dict("records")

    path.write_bytes(b"partial parquet")
    second = cache.parquet(path)
    assert second.exists is True
    assert second.value.to_dict("records") == expected.to_dict("records")
    assert second.warning is not None


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_scheduler_uses_aggregate_summary_and_exact_project_status():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), timeout))
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({"name_prefix": "mft", "total": 9, "statuses": {"running": 4, "completed": 3, "failed": 2}})
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 300,
            "queued_count": 7,
            "attaching_count": 2,
            "executing_count": 4,
            "logical_active_count": 13,
            "updated_at": "2026-07-13T01:00:00+09:00",
        })

    result = SchedulerReader(base_url="http://example.test", opener=opener, ttl=0).snapshot()
    assert result["connected"] is True
    assert result["running"] == 4
    assert result["total"] == 9
    assert result["parallel_target"] == 300
    assert result["live_queued"] == 7
    assert result["live_attaching"] == 2
    assert result["live_running"] == 4
    assert result["logical_active"] == 13
    assert result["control_enabled"] is True
    assert calls[0][1] == "GET"
    assert "/api/tasks/summary?" in calls[0][0]
    assert "/api/tasks?" not in calls[0][0]
    assert calls[1][1] == "GET"
    assert calls[1][0].endswith("/api/projects/MFT_1MW_2026v1")


def test_scheduler_normalizes_aedt_pool_and_license_attach_status():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 9,
                "statuses": {"running": 4, "completed": 5},
            })
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 300,
                "queued_count": 7,
                "attaching_count": 2,
                "executing_count": 4,
                "logical_active_count": 13,
            })
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 2,
                    "target_project_concurrency": 16,
                    "projects_per_aedt": 2,
                },
                "plan": {
                    "idle_session_count": 2,
                    "hard_session_count": 3,
                    "warm_spare_status_reason": "target satisfied",
                },
                "sessions": [
                    {"id": 1, "state": "ready"},
                    {"id": 2, "state": "ready"},
                    {"id": 3, "state": "starting"},
                ],
                "leases": [
                    {"id": 11, "state": "active"},
                    {"id": 12, "state": "queued"},
                    {"id": 13, "state": "releasing"},
                ],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "label": "Electronics Desktop",
                    "total": 550,
                    "used": 344,
                }],
                # A conflicting fallback proves the complete feature list wins.
                "in_use": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 12,
                }],
                "error": "",
                "admission": {"snapshot_valid": True},
            })
        raise AssertionError(request.full_url)

    result = SchedulerReader(
        base_url="http://example.test",
        opener=opener,
        timeout=2.0,
        optional_timeout=0.25,
        ttl=0,
    ).snapshot()

    assert result["connected"] is True
    attach = result["aedt_attach"]
    assert attach["available"] is True
    assert attach["state"] == "operational"
    assert attach["errors"] == []
    assert attach["pool"] == {
        "available": True,
        "enabled": True,
        "adapter_ready": True,
        "validation_passed": True,
        "operational": True,
        "max_sessions": 8,
        "min_idle_sessions": 2,
        "idle_sessions": 2,
        "hard_sessions": 3,
        "warm_spare_deficit": None,
        "warm_spare_start_needed": None,
        "session_record_count": 3,
        "lease_record_count": 3,
        "live_leases": 3,
        "queued_leases": 1,
        "ready_sessions": 2,
        "busy_sessions": 0,
        "session_states": {"ready": 2, "starting": 1},
        "lease_states": {"active": 1, "queued": 1, "releasing": 1},
        "warm_spare_reason": "target satisfied",
        "error": None,
    }
    assert attach["license"] == {
        "available": True,
        "feature": "electronics_desktop",
        "label": "Electronics Desktop",
        "used": 344,
        "total": 550,
        "snapshot_valid": True,
        "checked_at": "2026-07-13T02:00:00+09:00",
        "error": None,
    }
    optional_calls = {
        url: timeout for url, timeout in calls
        if url.endswith(("/api/aedt-pool", "/api/licenses"))
    }
    assert optional_calls == {
        "http://example.test/api/aedt-pool": 0.25,
        "http://example.test/api/licenses": 0.25,
    }


@pytest.mark.parametrize("failed_endpoint", ["pool", "license"])
def test_scheduler_optional_attach_endpoint_failure_is_section_local(
        failed_endpoint):
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 4,
                "statuses": {"running": 4},
            })
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 300,
                "queued_count": 0,
                "attaching_count": 0,
                "executing_count": 4,
                "logical_active_count": 4,
            })
        if request.full_url.endswith("/api/aedt-pool"):
            if failed_endpoint == "pool":
                raise URLError("optional endpoint timed out")
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 0,
                },
                "plan": {
                    "idle_session_count": 0,
                    "hard_session_count": 0,
                },
                "sessions": [],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            if failed_endpoint == "license":
                return FakeResponse({"features": "malformed"})
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 0,
                }],
                "admission": {"snapshot_valid": True},
            })
        raise AssertionError(request.full_url)

    result = SchedulerReader(
        base_url="http://optional.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    ).snapshot()

    # Optional diagnostics never erase the authoritative summary/project state.
    assert result["connected"] is True
    assert result["control_enabled"] is True
    assert result["running"] == 4
    assert result["parallel_target"] == 300
    attach = result["aedt_attach"]
    assert len(attach["errors"]) == 1
    if failed_endpoint == "pool":
        assert attach["state"] == "pool_unavailable"
        assert attach["pool"]["available"] is False
        assert "/api/aedt-pool" in attach["pool"]["error"]
        assert attach["license"]["available"] is True
        assert attach["license"]["used"] == 0
    else:
        assert attach["state"] == "partial"
        assert attach["pool"]["available"] is True
        assert attach["license"]["available"] is False
        assert "/api/licenses" in attach["license"]["error"]


def test_scheduler_aedt_attach_reports_warm_spare_shortfall():
    def opener(request, timeout):
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 2,
                },
                "plan": {
                    "idle_session_count": 0,
                    "hard_session_count": 1,
                    "warm_spare_deficit": 2,
                    "warm_spare_start_needed": 2,
                    "warm_spare_status_reason": (
                        "warm-spare session startup is in progress"
                    ),
                    "state_counts": {"starting": 1},
                    "lease_counts": {},
                },
                # Historical terminal rows must not become pool capacity.
                "sessions": [
                    *[{"id": index, "state": "closed"} for index in range(20)],
                    {"id": 21, "state": "starting"},
                ],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 344,
                }],
                "admission": {"snapshot_valid": True},
            })
        raise AssertionError(request.full_url)

    attach = SchedulerReader(
        base_url="http://warm-spare.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    )._aedt_attach_snapshot()

    assert attach["state"] == "warming"
    assert attach["pool"]["hard_sessions"] == 1
    assert attach["pool"]["max_sessions"] == 8
    assert attach["pool"]["session_record_count"] == 21
    assert attach["pool"]["warm_spare_deficit"] == 2


def test_scheduler_aedt_attach_marks_stale_license_snapshot_degraded():
    def opener(request, timeout):
        if request.full_url.endswith("/api/aedt-pool"):
            return FakeResponse({
                "config": {
                    "enabled": True,
                    "adapter_ready": True,
                    "validation_passed": True,
                    "operational": True,
                    "max_aedt_sessions": 8,
                    "min_idle_aedt_sessions": 1,
                },
                "plan": {
                    "idle_session_count": 1,
                    "hard_session_count": 1,
                    "state_counts": {"ready": 1},
                    "lease_counts": {},
                },
                "sessions": [{"id": 1, "state": "ready"}],
                "leases": [],
            })
        if request.full_url.endswith("/api/licenses"):
            return FakeResponse({
                "checked_at": "2026-07-13T02:00:00+09:00",
                "server_up": True,
                "features": [{
                    "feature": "electronics_desktop",
                    "total": 550,
                    "used": 344,
                }],
                "error": "showing the last good snapshot",
                "admission": {"snapshot_valid": False},
            })
        raise AssertionError(request.full_url)

    attach = SchedulerReader(
        base_url="http://stale-license.test",
        opener=opener,
        optional_timeout=0.1,
        ttl=0,
    )._aedt_attach_snapshot()

    assert attach["state"] == "degraded"
    assert attach["license"]["available"] is True
    assert attach["license"]["used"] == 344
    assert attach["license"]["snapshot_valid"] is False
    assert attach["errors"] == ["showing the last good snapshot"]


def test_scheduler_parallel_target_uses_cap_only_patch_and_exact_readback():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data, timeout))
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 275,
            "queued_count": 200,
            "attaching_count": 5,
            "executing_count": 70,
            "logical_active_count": 275,
            "updated_at": "2026-07-13T01:05:00+09:00",
            "repos": [{"url": "must-not-be-sent"}],
            "setup": "must-not-be-sent",
            "entrypoints": [{"path": "must-not-be-sent"}],
        })

    reader = SchedulerReader(base_url="http://example.test", opener=opener, ttl=60)
    result = reader.set_parallel_target(275)

    assert result["parallel_target"] == 275
    assert result["logical_active"] == 275
    assert len(calls) == 1
    assert calls[0][0].endswith("/api/projects/MFT_1MW_2026v1/max-active-tasks")
    assert calls[0][1] == "PATCH"
    assert json.loads(calls[0][2].decode("utf-8")) == {"max_active_tasks": 275}


def test_scheduler_parallel_target_rejects_invalid_value_before_request():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise AssertionError("invalid target must not reach scheduler")

    reader = SchedulerReader(base_url="http://example.test", opener=opener)
    for invalid in (0, 301, -1, 1.5, True, "300"):
        with pytest.raises(ValueError):
            reader.set_parallel_target(invalid)
    assert calls == []


def test_scheduler_disables_control_when_runtime_lacks_exact_live_count_fields():
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({
                "name_prefix": "mft",
                "total": 9,
                "statuses": {"queued": 3, "attaching": 2, "running": 4},
            })
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 300,
        })

    result = SchedulerReader(
        base_url="http://old-runtime.test", opener=opener, ttl=0
    ).snapshot()

    assert result["connected"] is True
    assert result["control_enabled"] is False
    assert result["parallel_target"] is None
    assert result["logical_active"] == 9
    assert "live counts" in result["project_error"]


def test_dashboard_survives_missing_optional_artifacts(tmp_path):
    from regression_260707.monitoring.readers import ArtifactService
    from .conftest import DummyScheduler, FIXED_NOW

    service = ArtifactService(tmp_path / "empty", scheduler=DummyScheduler(), clock=lambda: FIXED_NOW, record_runtime=False)
    payload = service.dashboard()
    assert payload["data"]["total_rows"] == 0
    assert payload["models"]["trained_count"] == 0
    assert payload["nsga2"]["available"] is False
    assert payload["verification"]["final"]["status"] == "waiting"
    timing = payload["data"]["simulation_timing"]
    assert timing["available"] is False
    assert timing["window_rows"] == 0
    assert all(
        stage["sample_count"] == 0
        and stage["mean_seconds"] is None
        and stage["median_seconds"] is None
        for stage in timing["stages"].values()
    )
