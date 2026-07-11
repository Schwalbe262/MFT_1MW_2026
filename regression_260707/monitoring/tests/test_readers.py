import json
from pathlib import Path

from regression_260707.monitoring.readers import SafeArtifactCache, SchedulerReader


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
    assert data["latest_revision"] == "a" * 40
    assert data["rows_not_latest_revision"] == 1
    assert data["eta_3000"] is not None


def test_models_include_planned_missing_targets_and_metrics(artifact_service):
    payload = artifact_service.models(current_data_count=100)
    lookup = {item["target"]: item for item in payload["models"]}
    assert lookup["Llt_phys"]["r2"] == .91
    assert lookup["P_winding_total"]["status"] == "attention"
    assert lookup["Tprobe_Tx_leeward_max"]["status"] == "not_trained"
    assert lookup["Llt_phys"]["history"][-1]["n"] == 100


def test_nsga_and_verification_are_joined(artifact_service):
    nsga = artifact_service.nsga2()
    assert nsga["round"] == 2
    assert nsga["candidate_count"] == 2
    assert nsga["summary"]["min_volume_L"] == 500
    assert nsga["comparison"]["min_volume_change_L"] == -100
    assert nsga["candidates"][0]["id"] == "r02-0000"
    assert nsga["candidates"][0]["spec_status"] == "unknown"  # temperature models are absent

    verification = artifact_service.verification(nsga)
    assert verification["counts"]["coverage"] == 1.0
    assert verification["standard_candidates"][0]["evaluation"]["computed_status"] == "pass"
    assert verification["final"]["status"] == "pass"
    assert verification["final"]["evaluation"]["checks"]["full_model"]["pass"] is True


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


def test_scheduler_uses_get_summary_only():
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, request.get_method(), timeout))
        return FakeResponse({"name_prefix": "mft", "total": 9, "statuses": {"running": 4, "completed": 3, "failed": 2}})

    result = SchedulerReader(base_url="http://example.test", opener=opener, ttl=0).snapshot()
    assert result["connected"] is True
    assert result["running"] == 4
    assert result["total"] == 9
    assert calls[0][1] == "GET"
    assert "/api/tasks/summary?" in calls[0][0]
    assert "/api/tasks?" not in calls[0][0]


def test_dashboard_survives_missing_optional_artifacts(tmp_path):
    from regression_260707.monitoring.readers import ArtifactService
    from .conftest import DummyScheduler, FIXED_NOW

    service = ArtifactService(tmp_path / "empty", scheduler=DummyScheduler(), clock=lambda: FIXED_NOW, record_runtime=False)
    payload = service.dashboard()
    assert payload["data"]["total_rows"] == 0
    assert payload["models"]["trained_count"] == 0
    assert payload["nsga2"]["available"] is False
    assert payload["verification"]["final"]["status"] == "waiting"
