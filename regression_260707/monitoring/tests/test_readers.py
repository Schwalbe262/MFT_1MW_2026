import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from regression_260707.model_targets import (
    CORE_REGION_TEMPERATURE_TARGETS,
    SURROGATE_TEMPERATURE_TARGETS,
)
from regression_260707.monitoring.readers import (
    ArtifactService,
    RuntimeRecorder,
    SafeArtifactCache,
    SchedulerReader,
    TARGET_META,
    TEMPERATURE_TARGETS,
    _simulation_timing_summary,
    _zero_aware_percentage_metrics,
)


def _install_checkpoint_fixture(
        campaign_root: Path, *, metrics_hash_valid: bool = True,
        parity_hash_valid: bool = True,
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
        "dataset": "snapshot.parquet",
        "dataset_sha256": "snapshot-sha",
        "profile": "profile.json",
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
    parity_path = metrics_path.with_suffix(".parity.json")
    parity_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": "checkpoint_cv_oof_parity",
        "checkpoint": 500,
        "dataset": "snapshot.parquet",
        "dataset_sha256": "snapshot-sha",
        "profile": "profile.json",
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
    parity_hash = hashlib.sha256(parity_path.read_bytes()).hexdigest()
    if not parity_hash_valid:
        parity_hash = "0" * 64
    state["completed"][0].update({
        "parity_result": str(parity_path),
        "parity_result_sha256": parity_hash,
    })
    state_path = run_root / "checkpoint_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
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


def test_bad_parity_hash_keeps_metrics_but_returns_empty_parity(
        campaign_root, artifact_service):
    _install_checkpoint_fixture(campaign_root, parity_hash_valid=False)

    payload = artifact_service.models(current_data_count=100)
    model = next(item for item in payload["models"] if item["target"] == "Llt_phys")

    assert model["status"] == "checkpoint"
    assert model["evaluated"] is True
    assert model["r2"] == .95
    assert model["parity_available"] is False
    assert model["parity_sample_count"] == 0
    assert any("checkpoint parity hash mismatch" in warning
               for warning in payload["warnings"])


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
