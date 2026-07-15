import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707.model_targets import (
    CORE_REGION_TEMPERATURE_TARGETS,
    SURROGATE_TEMPERATURE_TARGETS,
)
from regression_260707.monitoring.readers import (
    ArtifactService,
    CURRENT_PHYSICS_DATA_REVISION,
    ReadResult,
    RefillControllerReader,
    RuntimeRecorder,
    SafeArtifactCache,
    SchedulerReader,
    SimulationPolicyConflict,
    TARGET_META,
    TEMPERATURE_TARGETS,
    _campaign_frame_summary,
    _simulation_timing_summary,
    _zero_aware_percentage_metrics,
)


OLDER_SOLVER_REVISION = "a" * 40
NEWER_SOLVER_REVISION = "b" * 40


def test_current_physics_revision_comes_from_repo_contract():
    assert CURRENT_PHYSICS_DATA_REVISION == PHYSICS_DATA_REVISION


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
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 1
    assert data["em_valid_rows"] == 1
    assert data["thermal_valid_rows"] == 1
    assert data["complete_rows"] == 1
    assert data["throughput_1h"] == 1
    assert data["added_24h"] == 1
    assert data["collector"]["no_data_tasks"] == 1
    assert data["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert data["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert data["pinned_library_revision"] == "c" * 40
    assert data["rows_not_latest_revision"] == 1
    assert data["rows_not_current_physics_revision"] == 0
    assert data["count_basis"] == "physics_revision_strict_full"
    assert data["member_git_hash_shorts"] == [
        "754923c", "bbbbbbb",
    ]
    assert data["eta_3000"] is not None
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["cohort_basis"] == "active_identity"
    assert timing["cohort_filter"] == {
        "git_hash": "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a",
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert timing["active_cohort"]["status"] == "active"
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["window_limit_rows"] == 100
    assert timing["stages"]["matrix"]["mean_seconds"] == 300.0
    assert timing["stages"]["total"]["mean_seconds"] == 3_000.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 0


def test_data_revision_aggregate_does_not_reset_with_zero_pinned_status(
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
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 1
    assert data["em_valid_rows"] == 1
    assert data["thermal_valid_rows"] == 1
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


def test_simulation_timing_summary_uses_newest_sha_for_current_physics():
    older = {
        "git_hash": OLDER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    newer = {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    summary = _simulation_timing_summary([
        {
            **older, "saved_at": "2026-07-11 00:00:00",
            "time_matrix": "100", "time_thermal": "100",
        },
        {
            "git_hash": "legacy", "physics_data_revision": "legacy",
            "saved_at": "2026-07-11 05:00:00", "thermal_on": 0,
            "time_matrix": "999", "time_thermal": "3",
        },
        {
            **newer, "git_hash": NEWER_SOLVER_REVISION.upper(),
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": "400", "time_thermal": "400",
        },
        {
            **newer, "saved_at": "2026-07-11 01:00:00",
            "time_matrix": "200", "time_thermal": "200",
        },
        {
            **older,
            "saved_at": "2026-07-11 03:00:00",
            "time_matrix": "300", "time_thermal": "300",
        },
    ], limit=2)

    assert summary["cohort_basis"] == "active_identity"
    assert summary["cohort_label"] == "활성 코호트 bbbbbbbbbb"
    assert summary["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert summary["active_cohort"]["available"] is True
    assert summary["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert summary["cohort_rows"] == 2
    assert summary["window_rows"] == 2
    assert summary["stages"]["matrix"]["sample_count"] == 2
    assert summary["stages"]["matrix"]["mean_seconds"] == 300.0
    assert summary["stages"]["matrix"]["median_seconds"] == 300.0
    assert summary["stages"]["icepak"]["mean_seconds"] == 300.0
    assert summary["stages"]["loss"]["sample_count"] == 0
    assert summary["stages"]["loss"]["mean_seconds"] is None


def test_simulation_timing_summary_does_not_fall_back_to_legacy_rows():
    summary = _simulation_timing_summary([
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": "2026-07-11 03:00:00", "thermal_on": 0,
            "time_matrix": 1, "time_loss": 2,
            "time_thermal": 3, "time": 6,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": "legacy_unspecified",
            "time_thermal": 3,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": "previous-physics-revision",
            "time_thermal": 3,
        },
    ])

    assert summary["available"] is False
    assert summary["cohort_filter"] == {
        "git_hash": None,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert "현재 revision 데이터 없음" in summary["cohort_label"]
    assert CURRENT_PHYSICS_DATA_REVISION in summary["cohort_label"]
    assert summary["cohort_rows"] == 0
    assert summary["window_rows"] == 0
    assert all(
        stage["sample_count"] == 0
        and stage["mean_seconds"] is None
        and stage["median_seconds"] is None
        for stage in summary["stages"].values()
    )


def test_simulation_timing_summary_uses_cap_solve_plus_extraction_only_when_on():
    base = {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    summary = _simulation_timing_summary([
        {
            **base, "saved_at": "2026-07-11 04:00:00", "cap_on": 1,
            "cap_solve_time_s": 10, "cap_extraction_time_s": 2,
            "time": 100,
        },
        {
            **base, "saved_at": "2026-07-11 03:00:00", "cap_on": "true",
            "cap_solve_time_s": 20, "cap_extraction_time_s": 4,
            "time": 110,
        },
        {
            **base, "saved_at": "2026-07-11 02:00:00", "cap_on": 0,
            "cap_solve_time_s": 999, "cap_extraction_time_s": 999,
            "time": 120,
        },
        {
            **base, "saved_at": "2026-07-11 01:00:00", "cap_on": 1,
            "cap_solve_time_s": 30,
            "time": 130,
        },
    ])

    stage = summary["stages"]["electrostatic"]
    assert stage["source_fields"] == [
        "cap_solve_time_s", "cap_extraction_time_s",
    ]
    assert stage["sample_count"] == 2
    assert stage["mean_seconds"] == pytest.approx(18.0)
    assert stage["median_seconds"] == pytest.approx(18.0)
    assert summary["stages"]["total"]["sample_count"] == 4


def test_simulation_timing_summary_tolerates_missing_columns():
    frame = pd.DataFrame([
        {"saved_at": "2026-07-11 03:00:00", "time_thermal": 3},
        {"git_hash": OLDER_SOLVER_REVISION, "time_matrix": 100},
        {
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "time_loss": 200,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
        },
    ])

    summary = _simulation_timing_summary(frame)

    assert summary["available"] is False
    assert summary["cohort_rows"] == 1
    assert summary["window_rows"] == 1
    assert all(stage["sample_count"] == 0
               for stage in summary["stages"].values())


def test_active_cohort_degrades_when_recency_or_newest_hash_is_missing():
    undated = _simulation_timing_summary([{
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "time_matrix": 100,
    }])
    assert undated["active_cohort"]["available"] is False
    assert undated["cohort_rows"] == 0

    newest_hash_missing = _simulation_timing_summary([
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 03:00:00",
            "time_matrix": 100,
        },
        {
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": 200,
        },
    ])
    assert newest_hash_missing["active_cohort"]["available"] is False
    assert newest_hash_missing["cohort_rows"] == 0


def test_simulation_timing_summary_degrades_if_physics_import_is_unavailable():
    summary = _simulation_timing_summary(
        [{
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": PHYSICS_DATA_REVISION,
            "saved_at": "2026-07-11 04:00:00",
            "time_matrix": 100,
        }],
        current_physics_revision=None,
    )

    assert summary["available"] is False
    assert summary["active_cohort"]["status"] == "physics_revision_unavailable"
    assert summary["cohort_filter"] == {
        "git_hash": None,
        "physics_data_revision": None,
    }
    assert "import 실패" in summary["cohort_label"]
    assert summary["cohort_rows"] == 0


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


def test_campaign_frame_summary_scopes_all_panels_to_newest_rolling_sha():
    from .conftest import FIXED_NOW

    current = {
        "git_hash": NEWER_SOLVER_REVISION,
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
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=30)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 1,
            "C_tx_tx_F": 99e-9,
            "C_rx_rx_F": 99e-9,
            "C_tx_rx_F": 99e-9,
            "f_res_tx_self_Hz": 9_900_000.0,
            "f_res_rx_self_Hz": 9_900_000.0,
            "f_res_interwinding_Hz": 9_900_000.0,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_core_k_inplane": 99.0,
            "thermal_core_k_throughstack": 99.0,
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
    summary = _campaign_frame_summary(frame, FIXED_NOW)

    assert summary["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert summary["active_cohort"]["physics_data_revision"] == (
        CURRENT_PHYSICS_DATA_REVISION
    )
    assert len(summary["cohorts"]) == 3
    cohorts = {
        (item["git_hash"], item["physics_data_revision"]): item
        for item in summary["cohorts"]
    }
    cohort = cohorts[(NEWER_SOLVER_REVISION, CURRENT_PHYSICS_DATA_REVISION)]
    assert cohort == {
        "git_hash": NEWER_SOLVER_REVISION,
        "git_hash_short": NEWER_SOLVER_REVISION[:10],
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        "latest_saved_at": (FIXED_NOW - timedelta(minutes=20)).isoformat(),
        "active": True,
        "current": True,
        "raw_rows": 5,
        "strict_em_rows": 4,
        "strict_full_rows": 3,
        "growth_rate_per_hour": 3.0,
    }
    older = cohorts[(OLDER_SOLVER_REVISION, CURRENT_PHYSICS_DATA_REVISION)]
    assert older["active"] is False
    assert older["current"] is False
    assert older["raw_rows"] == 1
    assert older["strict_em_rows"] == 1
    assert older["strict_full_rows"] == 1
    legacy = cohorts[(
        "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
        "legacy_unspecified",
    )]
    assert legacy["active"] is False
    assert legacy["current"] is False
    assert legacy["raw_rows"] == 2
    assert legacy["strict_em_rows"] == 0
    assert legacy["strict_full_rows"] == 0
    assert [item["git_hash"] for item in summary["cohorts"]] == [
        NEWER_SOLVER_REVISION,
        "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
        OLDER_SOLVER_REVISION,
    ]
    aggregate = summary["physics_revision_aggregate"]
    assert aggregate["physics_data_revision"] == CURRENT_PHYSICS_DATA_REVISION
    assert aggregate["raw_rows"] == 6
    assert aggregate["strict_em_rows"] == 5
    assert aggregate["strict_full_rows"] == 4
    assert aggregate["growth_rate_per_hour"] == 4.0
    assert aggregate["member_git_hashes"] == [
        NEWER_SOLVER_REVISION, OLDER_SOLVER_REVISION,
    ]
    assert aggregate["member_git_hash_shorts"] == ["bbbbbbb", "aaaaaaa"]

    electrostatic = summary["electrostatic"]
    assert electrostatic["cohort_basis"] == "active_strict_full"
    assert electrostatic["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
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
    assert summary["thermal_models"]["cohort_basis"] == "active_identity"
    assert summary["thermal_models"]["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert summary["thermal_models"]["total_rows"] == 5
    assert summary["thermal_models"]["tagged_rows"] == 5
    assert summary["thermal_models"]["missing_rows"] == 0
    assert thermal["anisotropic_wound_rule_of_mixtures_v1"]["count"] == 5
    assert "isotropic_legacy" not in thermal
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


def test_campaign_cohorts_sort_inactive_rows_by_latest_saved_row():
    from .conftest import FIXED_NOW

    recent_revision = "c" * 40
    undated_revision = "d" * 40
    inactive_second = FIXED_NOW - timedelta(minutes=10)
    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=1)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=100)
            ).isoformat(),
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=200)
            ).isoformat(),
        },
        {
            "git_hash": recent_revision,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (
                inactive_second + timedelta(microseconds=800)
            ).isoformat(),
        },
        {
            "git_hash": undated_revision,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
        },
    ])

    cohorts = _campaign_frame_summary(frame, FIXED_NOW)["cohorts"]

    assert [item["git_hash"] for item in cohorts] == [
        NEWER_SOLVER_REVISION,
        recent_revision,
        OLDER_SOLVER_REVISION,
        undated_revision,
    ]
    assert cohorts[1]["latest_saved_at"] == (
        inactive_second + timedelta(microseconds=800)
    ).isoformat()
    assert cohorts[2]["raw_rows"] == 2
    assert cohorts[2]["latest_saved_at"] == (
        inactive_second + timedelta(microseconds=200)
    ).isoformat()
    assert cohorts[3]["latest_saved_at"] is None


def test_campaign_frame_summary_tolerates_missing_columns_and_uses_flags():
    from .conftest import FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=2)).isoformat(),
            "result_valid_em": 1,
            "result_valid_thermal": 1,
        },
        {
            "git_hash": NEWER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(minutes=1)).isoformat(),
        },
    ])

    summary = _campaign_frame_summary(frame, FIXED_NOW)

    cohort = summary["cohorts"][0]
    assert cohort["active"] is True
    assert cohort["raw_rows"] == 2
    assert cohort["strict_em_rows"] == 1
    assert cohort["strict_full_rows"] == 1
    assert cohort["growth_rate_per_hour"] == 1.0
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
    thermal = summary["thermal_models"]
    assert thermal["available"] is False
    assert thermal["cohort_filter"] == {
        "git_hash": NEWER_SOLVER_REVISION,
        "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
    }
    assert thermal["total_rows"] == 2
    assert thermal["tagged_rows"] == 0
    assert thermal["missing_rows"] == 2
    assert thermal["models"] == []
    assert summary["current_cohort_metadata"]["core_lamination_factor"][
        "sample_count"
    ] == 0
    assert summary["current_cohort_metadata"][
        "winding_flux_linkage_readback"
    ]["missing_rows"] == 2


def test_campaign_audit_is_per_sha_and_preserves_malformed_rows(tmp_path):
    frame = pd.DataFrame({
        "git_hash": [
            NEWER_SOLVER_REVISION, None, OLDER_SOLVER_REVISION, "bad-sha",
        ],
        "physics_data_revision": [CURRENT_PHYSICS_DATA_REVISION] * 4,
    }, index=[7, 7, 3, 9])
    calls = []

    def fake_annotate(
            cohort, *, expected_solver_revision,
            expected_library_revision):
        calls.append((
            tuple(cohort["git_hash"]), expected_solver_revision,
            expected_library_revision,
        ))
        audited = cohort.copy()
        valid = expected_solver_revision is not None
        audited["_strict_valid_em"] = valid
        audited["_strict_valid_thermal"] = valid
        audited["_strict_valid_full"] = valid
        audited["_strict_invalid_reasons"] = "" if valid else "bad-provenance"
        return audited

    service = ArtifactService(tmp_path, record_runtime=False)
    with mock.patch(
        "regression_260707.quality_contract.annotate_validity",
        side_effect=fake_annotate,
    ):
        audited, warning = service._audited_campaign_frame(
            ReadResult(frame, "memory.parquet", True),
            NEWER_SOLVER_REVISION,
            "d" * 40,
        )

    assert warning is None
    assert audited.index.tolist() == [7, 7, 3, 9]
    assert audited["git_hash"].tolist()[:1] == [NEWER_SOLVER_REVISION]
    assert pd.isna(audited["git_hash"].iloc[1])
    assert audited["git_hash"].tolist()[2:] == [OLDER_SOLVER_REVISION, "bad-sha"]
    assert audited["_strict_valid_full"].tolist() == [True, False, True, False]
    assert calls == [
        ((NEWER_SOLVER_REVISION,), NEWER_SOLVER_REVISION, "d" * 40),
        ((None,), None, None),
        ((OLDER_SOLVER_REVISION,), OLDER_SOLVER_REVISION, None),
        (("bad-sha",), None, None),
    ]


def test_artifact_service_data_reads_lossless_campaign_parquet(campaign_root):
    from .conftest import DummyScheduler, FIXED_NOW

    frame = pd.DataFrame([
        {
            "git_hash": NEWER_SOLVER_REVISION,
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
            "time_matrix": 480.0,
            "time_loss": 1_800.0,
            "cap_solve_time_s": 40.0,
            "cap_extraction_time_s": 5.0,
            "time_thermal": 900.0,
            "time": 3_180.0,
        },
        {
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
            "cap_on": 0,
            "time_matrix": 300.0,
            "time_loss": 1_000.0,
            "time_thermal": 800.0,
            "time": 2_100.0,
        },
        {
            "git_hash": "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c",
            "physics_data_revision": "legacy_unspecified",
            "saved_at": (FIXED_NOW - timedelta(hours=2)).isoformat(),
            "_strict_valid_em": False,
            "_strict_valid_full": False,
            "thermal_core_conductivity_model": "isotropic_legacy",
            "thermal_on": 0,
            "time_matrix": 1.0,
            "time_loss": 2.0,
            "time_thermal": 3.0,
            "time": 6.0,
        },
        {
            "git_hash": "c" * 40,
            "physics_data_revision": "previous-physics-revision",
            "saved_at": (FIXED_NOW - timedelta(hours=3)).isoformat(),
            "_strict_valid_em": True,
            "_strict_valid_full": True,
        },
    ])
    manifest_path = campaign_root / "data" / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_rows"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parquet_path = campaign_root / "data" / "dataset" / "train.parquet"
    frame.to_parquet(parquet_path, index=False)
    history_path = (
        campaign_root / "monitoring" / "runtime" / "monitor_history.jsonl"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "time": (FIXED_NOW - timedelta(minutes=10)).isoformat(),
        "data": {"cohorts": [{
            "git_hash": OLDER_SOLVER_REVISION,
            "physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "strict_full_rows": 0,
        }]},
    }) + "\n", encoding="utf-8")
    service = ArtifactService(
        campaign_root,
        scheduler=DummyScheduler(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )

    audit_calls = []

    def audit_by_declared_sha(
            cohort, *, expected_solver_revision,
            expected_library_revision):
        declared = {
            str(value).strip().lower() for value in cohort["git_hash"]
        }
        assert declared == {expected_solver_revision}
        assert expected_library_revision is None
        audit_calls.append((expected_solver_revision, tuple(cohort.index)))
        audited = cohort.copy()
        valid = audited["physics_data_revision"].ne("legacy_unspecified")
        audited["_strict_valid_em"] = valid
        audited["_strict_valid_thermal"] = valid
        audited["_strict_valid_full"] = valid
        audited["_strict_invalid_reasons"] = [
            "" if item else "test:legacy" for item in valid
        ]
        return audited

    with mock.patch(
        "regression_260707.quality_contract.annotate_validity",
        side_effect=audit_by_declared_sha,
    ):
        data = service.data()

    assert audit_calls == [
        (NEWER_SOLVER_REVISION, (0,)),
        (OLDER_SOLVER_REVISION, (1,)),
        ("b171c7ce5f7a018be6a575a32b1a1f5b7caa980c", (2,)),
        ("c" * 40, (3,)),
    ]

    assert data["source"]["campaign_rows"] == str(parquet_path)
    assert data["active_cohort"]["git_hash"] == NEWER_SOLVER_REVISION
    assert data["cohorts"][0]["active"] is True
    assert data["cohorts"][0]["current"] is True
    assert data["cohorts"][0]["raw_rows"] == 1
    assert data["cohorts"][0]["strict_full_rows"] == 1
    assert data["count_basis"] == "physics_revision_strict_full"
    assert data["raw_total_rows"] == 4
    assert data["revision_raw_rows"] == 2
    assert data["total_rows"] == 2
    assert data["em_valid_rows"] == 2
    assert data["throughput_1h"] == 1
    assert data["member_git_hashes"] == [
        NEWER_SOLVER_REVISION, OLDER_SOLVER_REVISION,
    ]
    assert data["member_git_hash_shorts"] == ["bbbbbbb", "aaaaaaa"]
    older_cohort = next(
        item for item in data["cohorts"]
        if item["git_hash"] == OLDER_SOLVER_REVISION
    )
    # Old zero-based runtime snapshots must not turn a rolled SHA's existing
    # rows into false +/h growth.  saved_at is the authoritative basis.
    assert older_cohort["growth_rate_per_hour"] == 0
    assert data["rows_not_current_physics_revision"] == 2
    assert data["eta_3000"] is not None
    assert data["electrostatic"]["cap_stage_present_rows"] == 1
    assert data["electrostatic"]["capacitance"]["tx_tx"][
        "median_nF"
    ] == pytest.approx(2.0)
    assert data["thermal_models"]["total_rows"] == 1
    assert data["thermal_models"]["tagged_rows"] == 1
    assert data["quarantine"]["legacy"]["rows"] == 1
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["stages"]["matrix"]["mean_seconds"] == 480.0
    assert timing["stages"]["electrostatic"]["mean_seconds"] == 45.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 1
    assert timing["stages"]["icepak"]["mean_seconds"] == 900.0
    assert timing["stages"]["total"]["mean_seconds"] == 3_180.0


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
        refill_controller=artifact_service.refill_controller,
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
        "git_hash": NEWER_SOLVER_REVISION,
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
            "max_active_tasks": 510,
            "desired_simulations": 500,
            "effective_simulations": 480,
            "validated_concurrency_limit": 500,
            "min_desired_simulations": 0,
            "max_desired_simulations": 500,
            "policy_revision": 12,
            "scale_down_mode": "drain",
            "queued_count": 7,
            "attaching_count": 2,
            "executing_count": 4,
            "solving_count": 3,
            "logical_active_count": 13,
            "updated_at": "2026-07-13T01:00:00+09:00",
        })

    result = SchedulerReader(base_url="http://example.test", opener=opener, ttl=0).snapshot()
    assert result["connected"] is True
    assert result["running"] == 4
    assert result["total"] == 9
    assert result["parallel_target"] == 500
    assert result["effective_simulations"] == 480
    assert result["validated_concurrency_limit"] == 500
    assert result["policy_revision"] == 12
    assert result["live_queued"] == 7
    assert result["live_attaching"] == 2
    assert result["live_active"] == 4
    assert result["live_solving"] == 3
    assert result["logical_active"] == 6
    assert result["control_enabled"] is True
    assert calls[0][1] == "GET"
    assert "/api/tasks/summary?" in calls[0][0]
    assert "/api/tasks?" not in calls[0][0]
    assert calls[1][1] == "GET"
    assert calls[1][0].endswith("/api/projects/MFT_1MW_2026v1")
    assert calls[2][1] == "GET"
    assert calls[2][0].endswith(
        "/api/projects/MFT_1MW_2026v1/simulation-policy"
    )


def test_scheduler_legacy_cap_above_300_remains_visible_but_not_mutable():
    def opener(request, timeout):
        if "/api/tasks/summary?" in request.full_url:
            return FakeResponse({"total": 9, "statuses": {"running": 4}})
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "max_active_tasks": 510,
            "queued_count": 3,
            "attaching_count": 2,
            "executing_count": 4,
        })

    result = SchedulerReader(
        base_url="http://legacy.test", opener=opener, ttl=0
    ).snapshot()

    assert result["connected"] is True
    assert result["legacy_project_cap"] == 510
    assert result["logical_active"] == 6
    assert result["policy_supported"] is False
    assert result["control_enabled"] is False
    assert result["parallel_target"] is None
    assert "durable simulation-policy" in result["control_gate_reason"]


def test_scheduler_policy_allows_bounded_repair_of_legacy_desired_510():
    control = SchedulerReader(base_url="http://repair.test")._simulation_policy_control({
        "project": "MFT_1MW_2026v1",
        "desired_simulations": 510,
        "effective_simulations": 500,
        "validated_concurrency_limit": 510,
        "min_desired_simulations": 0,
        "max_desired_simulations": 500,
        "policy_revision": 4,
        "scale_down_mode": "drain",
        "queued_count": 10,
        "attaching_count": 2,
        "active_count": 6,
        "solving_count": 4,
        "logical_active_count": 6,
    })

    assert control["desired_simulations"] == 510
    assert control["parallel_target_max"] == 500
    assert control["live_active"] == 4
    assert control["logical_active"] == 6
    assert control["control_enabled"] is True


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
        if "/api/tasks?" in request.full_url and "_aedt_pool_hosts" in request.full_url:
            return FakeResponse([
                {
                    "task_id": 701, "name": "bundle-a-host",
                    "project": "_aedt_pool_hosts", "status": "running",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 702, "name": "bundle-b-host",
                    "project": "_aedt_pool_hosts", "status": "attaching",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 703, "name": "bundle-c-host",
                    "project": "_aedt_pool_hosts", "status": "queued",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 704, "name": "finished-host",
                    "project": "_aedt_pool_hosts", "status": "completed",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 705, "name": "unrelated-host",
                    "project": "other", "status": "running",
                    "entrypoint": "aedt_node_canary_host",
                },
                {
                    "task_id": 706, "name": "central-session-host",
                    "project": "_aedt_pool_hosts", "status": "running",
                    "entrypoint": "slurm_scheduler.aedt_session_host",
                },
            ])
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
    assert attach["node_local"] == {
        "available": True,
        "project": "_aedt_pool_hosts",
        "active_host_tasks": 3,
        "statuses": {"attaching": 1, "queued": 1, "running": 1},
        "bundle_count": 3,
        "bundle_ids": ["bundle-a", "bundle-b", "bundle-c"],
        "expected_projects": None,
        "hosts": [
            {
                "task_id": 701, "name": "bundle-a-host",
                "status": "running", "bundle_id": "bundle-a",
            },
            {
                "task_id": 702, "name": "bundle-b-host",
                "status": "attaching", "bundle_id": "bundle-b",
            },
            {
                "task_id": 703, "name": "bundle-c-host",
                "status": "queued", "bundle_id": "bundle-c",
            },
        ],
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
        if request.full_url.endswith(
                "/api/projects/MFT_1MW_2026v1/simulation-policy"):
            return FakeResponse({
                "project": "MFT_1MW_2026v1",
                "desired_simulations": 300,
                "effective_simulations": 300,
                "validated_concurrency_limit": 500,
                "min_desired_simulations": 0,
                "max_desired_simulations": 500,
                "policy_revision": 3,
                "scale_down_mode": "drain",
                "queued_count": 0,
                "attaching_count": 0,
                "active_count": 4,
                "solving_count": 4,
                "logical_active_count": 4,
                "control_enabled": True,
            })
        if request.full_url.endswith("/api/projects/MFT_1MW_2026v1"):
            return FakeResponse({
                "name": "MFT_1MW_2026v1",
                "max_active_tasks": 300,
                "desired_simulations": 300,
                "effective_simulations": 300,
                "validated_concurrency_limit": 500,
                "min_desired_simulations": 0,
                "max_desired_simulations": 500,
                "policy_revision": 3,
                "queued_count": 0,
                "attaching_count": 0,
                "executing_count": 4,
                "solving_count": 4,
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
    assert attach["node_local"]["available"] is False
    assert "/api/tasks?" in attach["node_local"]["error"]
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


def test_scheduler_simulation_policy_uses_versioned_drain_patch_and_exact_readback():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), request.data, timeout))
        return FakeResponse({
            "name": "MFT_1MW_2026v1",
            "desired_simulations": 500,
            "effective_simulations": 275,
            "validated_concurrency_limit": 500,
            "min_desired_simulations": 0,
            "max_desired_simulations": 500,
            "policy_revision": 18,
            "scale_down_mode": "drain",
            "queued_count": 200,
            "attaching_count": 5,
            "active_count": 70,
            "solving_count": 65,
            "updated_at": "2026-07-13T01:05:00+09:00",
            "repos": [{"url": "must-not-be-sent"}],
            "setup": "must-not-be-sent",
            "entrypoints": [{"path": "must-not-be-sent"}],
        })

    reader = SchedulerReader(base_url="http://example.test", opener=opener, ttl=60)
    result = reader.set_simulation_policy(500, expected_revision=17)

    assert result["parallel_target"] == 500
    assert result["effective_simulations"] == 275
    assert result["logical_active"] == 75
    assert len(calls) == 1
    assert calls[0][0].endswith("/api/projects/MFT_1MW_2026v1/simulation-policy")
    assert calls[0][1] == "PATCH"
    assert json.loads(calls[0][2].decode("utf-8")) == {
        "desired_simulations": 500,
        "expected_revision": 17,
        "scale_down_mode": "drain",
    }


def test_scheduler_simulation_policy_rejects_invalid_value_before_request():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        raise AssertionError("invalid target must not reach scheduler")

    reader = SchedulerReader(base_url="http://example.test", opener=opener)
    for invalid in (601, -1, 1.5, True, "300"):
        with pytest.raises(ValueError):
            reader.set_simulation_policy(invalid, expected_revision=1)
    with pytest.raises(ValueError):
        reader.set_simulation_policy(300, expected_revision=None)
    assert calls == []


def test_scheduler_simulation_policy_maps_revision_conflict():
    def opener(request, timeout):
        raise HTTPError(request.full_url, 409, "conflict", {}, None)

    reader = SchedulerReader(base_url="http://example.test", opener=opener)
    with pytest.raises(SimulationPolicyConflict):
        reader.set_simulation_policy(500, expected_revision=17)


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
    assert result["logical_active"] == 6
    assert "live counts" in result["project_error"]


@pytest.mark.parametrize(
    "state_payload",
    (
        {"policy": {"target": 275}},
        {"generation": {"identity": {"project_concurrency_target": 275}}},
    ),
    ids=("policy-target", "production-generation-identity"),
)
def test_refill_controller_reader_returns_latest_tick_and_state_target(
    tmp_path, monkeypatch, state_payload
):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            (
                json.dumps({"action": "older_tick"}),
                json.dumps({
                    "action": "rolling_refill_complete",
                    "active_project_tasks_before": 271,
                    "accepted_or_reconciled_count": 4,
                    "generation": {"id": "restart-v3-1234567890abcdef"},
                }),
                "",
            )
        ),
        encoding="utf-8",
    )
    tick_timestamp = 1_784_000_000
    os.utime(log_path, (tick_timestamp, tick_timestamp))
    monkeypatch.setenv("MFT_CONTROLLER_STATE_PATH", str(state_path))
    monkeypatch.setenv("MFT_CONTROLLER_LOG_PATH", str(log_path))

    result = RefillControllerReader().snapshot()

    assert result == {
        "available": True,
        "last_tick_at": result["last_tick_at"],
        "action": "rolling_refill_complete",
        "active_project_tasks_before": 271,
        "accepted_or_reconciled_count": 4,
        "generation_id": "restart-v3-1234567890abcdef",
        "concurrency_target": 275,
    }
    assert datetime.fromisoformat(result["last_tick_at"]).timestamp() == tick_timestamp


def test_refill_controller_reader_missing_files_is_unavailable(tmp_path):
    result = RefillControllerReader(
        tmp_path / "missing-state.json", tmp_path / "missing.log"
    ).snapshot()

    assert result == {"available": False}


def test_refill_controller_reader_rejects_malformed_last_log_line(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    log_path.write_text(
        json.dumps({"action": "no_refill_needed"}) + "\n{malformed\n\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_rejects_malformed_state(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text("{malformed", encoding="utf-8")
    log_path.write_text(
        json.dumps({"action": "no_refill_needed"}) + "\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_requires_tick_action(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    log_path.write_text(
        json.dumps({"active_project_tasks_before": 300}) + "\n",
        encoding="utf-8",
    )

    assert RefillControllerReader(state_path, log_path).snapshot() == {
        "available": False
    }


def test_refill_controller_reader_accepts_tick_larger_than_64_kib(tmp_path):
    state_path = tmp_path / "controller-state.json"
    log_path = tmp_path / "controller.log"
    state_path.write_text(json.dumps({"policy": {"target": 300}}), encoding="utf-8")
    tick_line = json.dumps({
        "action": "pooled_bundle_pending",
        "active_project_tasks_before": 298,
        "accepted_or_reconciled_count": 0,
        "padding": "x" * 70_000,
    })
    assert 64 * 1024 < len(tick_line.encode("utf-8")) < 128 * 1024
    log_path.write_text(
        json.dumps({"action": "older_tick"}) + "\n" + tick_line + "\n",
        encoding="utf-8",
    )

    result = RefillControllerReader(state_path, log_path).snapshot()

    assert result["available"] is True
    assert result["action"] == "pooled_bundle_pending"
    assert result["active_project_tasks_before"] == 298
    assert result["accepted_or_reconciled_count"] == 0


def test_dashboard_survives_missing_optional_artifacts(tmp_path):
    from regression_260707.monitoring.readers import ArtifactService
    from .conftest import DummyRefillController, DummyScheduler, FIXED_NOW

    service = ArtifactService(
        tmp_path / "empty",
        scheduler=DummyScheduler(),
        refill_controller=DummyRefillController(),
        clock=lambda: FIXED_NOW,
        record_runtime=False,
    )
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
