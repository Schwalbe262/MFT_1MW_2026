import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707.model_targets import CORE_REGION_TEMPERATURE_TARGETS
from regression_260707.monitoring.app import create_app


def test_dashboard_page_and_all_read_only_apis(artifact_service):
    client = TestClient(create_app(service=artifact_service))
    page = client.get("/")
    assert page.status_code == 200
    assert "최적설계 파이프라인" in page.text
    assert "data-chart" in page.text
    assert "현재 물리 revision 학습 가능" in page.text
    assert "코호트 상세 →" in page.text
    assert 'href="/cohorts"' in page.text
    assert 'id="data-member-shas"' in page.text
    assert "전체 수집 데이터" in page.text
    assert "data-raw-total" in page.text
    assert "최근 시뮬레이션 단계별 소요시간" in page.text
    assert "stage-time-matrix-mean" in page.text
    assert "stage-time-electrostatic-mean" in page.text
    assert "Electrostatic 평균" in page.text
    assert 'id="stage-timing-basis"' in page.text
    assert "활성 코호트 기준" in page.text
    assert "활성 코호트 확인 중" in page.text
    assert "활성 코호트 타이밍 데이터 없음" in page.text
    assert "단계 소요시간" in page.text
    assert "final-time-matrix" in page.text
    assert "MFT 병렬 실행 목표" in page.text
    assert "parallel-target-input" in page.text
    assert "parallel-logical-active" in page.text
    assert "parallel-attaching" in page.text
    assert 'id="model-history-metric"' in page.text
    assert 'value="mape_pct"' in page.text
    assert 'id="cohort-list"' not in page.text
    assert 'id="cohort-history"' not in page.text
    assert 'id="cohort-lamination-factor"' in page.text
    assert 'id="cohort-flux-availability"' in page.text
    assert 'id="quarantine-current-reasons"' in page.text
    assert 'id="quarantine-legacy-reasons"' in page.text
    assert 'id="capacitance-summary"' in page.text
    assert 'id="resonance-summary"' in page.text
    assert 'id="thermal-model-list"' in page.text
    assert 'id="thermal-model-basis"' in page.text
    assert 'id="aedt-attach-card"' in page.text
    assert 'id="aedt-license-usage"' in page.text
    assert 'id="aedt-pool-idle"' in page.text
    assert "중앙 풀(비활성·구조적 차단)" in page.text
    assert 'id="aedt-node-local-progress"' in page.text
    assert "학습 데이터 수 기준" in page.text

    cohorts_page = client.get("/cohorts")
    assert cohorts_page.status_code == 200
    assert "수집 데이터 코호트" in cohorts_page.text
    assert "코호트 상세" in cohorts_page.text
    assert 'href="/"' in cohorts_page.text
    assert '<th>SHA</th><th>revision</th><th>Raw</th>' in cohorts_page.text
    assert '<th>Strict EM</th><th>Strict full</th><th>+/h</th>' in cohorts_page.text
    assert 'id="cohorts-body"' in cohorts_page.text
    assert "/static/cohorts.js" in cohorts_page.text

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "function duration(value)" in script.text
    assert "data.current_physics_data_revision" in script.text
    assert "data.member_git_hash_shorts" in script.text
    assert "data.revision_raw_rows" in script.text
    assert "data.raw_total_rows" in script.text
    assert "격리" in script.text
    assert "data.simulation_timing" in script.text
    assert "timing.cohort_label" in script.text
    assert "timing.active_cohort" in script.text
    assert "data.active_cohort" in script.text
    assert "n=${number(timingWindowRows)}" in script.text
    assert "timingCell(evaluation.timing_seconds)" in script.text
    assert 'return "—"' in script.text
    assert 'fetch("/api/operator/parallel-target"' in script.text
    assert '"X-MFT-Operator-Control": "parallel-target-v1"' in script.text
    assert "scheduler.live_queued" in script.text
    assert "x: (item) => Number(item.n)" in script.text
    assert "historyPointTooltip" in script.text
    assert "CV P90 APE" in script.text
    assert "data.current_cohort_metadata" in script.text
    assert "data.quarantine" in script.text
    assert "electrostatic.cap_stage_present_rows" in script.text
    assert '"C_tx_tx"' in script.text
    assert "resonance.interwinding" in script.text
    assert "data.thermal_models" in script.text
    assert "scheduler.aedt_attach" in script.text
    assert "license.used" in script.text
    assert "pool.min_idle_sessions" in script.text
    assert "ratio(pool.hard_sessions, pool.max_sessions)" in script.text
    assert "ratio(pool.ready_sessions, pool.busy_sessions)" in script.text
    assert "attach.node_local" in script.text
    assert "nodeLocal.active_host_tasks" in script.text
    assert "노드 로컬: 활성 호스트" in script.text
    assert '["matrix", "loss", "electrostatic", "icepak", "total"]' in script.text

    cohorts_script = client.get("/static/cohorts.js")
    assert cohorts_script.status_code == 200
    assert "function compactCohorts(payload)" in cohorts_script.text
    assert "legacy_aggregate: true" in cohorts_script.text
    assert "레거시 (${number(cohort.cohort_count)}개 코호트)" in cohorts_script.text
    assert 'fetch("/api/data"' in cohorts_script.text

    stylesheet = client.get("/static/app.css")
    assert stylesheet.status_code == 200
    assert ".timing-grid" in stylesheet.text
    assert ".stage-timing-grid" in stylesheet.text
    assert ".stage-timing-empty" in stylesheet.text
    assert ".history-metric-control" in stylesheet.text
    assert ".chart-tooltip" in stylesheet.text
    assert ".cohort-detail-table tr.active" in stylesheet.text
    assert ".cohort-detail-table tr.legacy-aggregate" in stylesheet.text
    assert ".quarantine-legacy" in stylesheet.text
    assert ".electrostatic-presence-grid" in stylesheet.text
    assert ".thermal-model-row" in stylesheet.text
    assert ".aedt-attach-card" in stylesheet.text
    assert ".aedt-node-local-progress" in stylesheet.text

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["data"]["total_rows"] == 1
    assert dashboard.json()["data"]["raw_total_rows"] == 2
    assert dashboard.json()["data"]["count_basis"] == "physics_revision_strict_full"
    assert dashboard.json()["data"]["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert dashboard.json()["data"]["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    data = dashboard.json()["data"]
    active = data["active_cohort"]
    assert active["available"] is True
    assert active["status"] == "active"
    assert active["git_hash"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert active["expected_physics_data_revision"] == PHYSICS_DATA_REVISION
    assert data["revision_raw_rows"] == 2
    assert data["member_git_hash_shorts"] == ["754923c", "bbbbbbb"]
    timing = data["simulation_timing"]
    assert timing["available"] is True
    assert timing["active_cohort"] == active
    assert timing["cohort_rows"] == 1
    assert timing["window_rows"] == 1
    assert timing["stages"]["total"]["mean_seconds"] == 3000.0
    assert timing["stages"]["electrostatic"]["sample_count"] == 0

    assert client.get("/api/status").status_code == 200
    assert client.get("/api/data").json()["complete_rows"] == 1
    models = client.get("/api/models").json()
    assert models["trained_count"] == 2
    model_lookup = {item["target"]: item for item in models["models"]}
    expected_core_targets = (
        "Tprobe_core_center_max",
        *CORE_REGION_TEMPERATURE_TARGETS,
    )
    assert all(target in model_lookup for target in expected_core_targets)
    assert all(model_lookup[target]["status"] == "not_trained"
               for target in expected_core_targets)
    assert client.get("/api/models/Llt_phys/history").json()["target"] == "Llt_phys"
    parity = client.get("/api/models/Llt_phys/parity")
    assert parity.status_code == 200
    assert parity.json()["target"] == "Llt_phys"
    assert parity.json()["available"] is False  # active registry takes priority
    for target in expected_core_targets:
        history = client.get(f"/api/models/{target}/history")
        assert history.status_code == 200
        assert history.json()["target"] == target
    assert client.get("/api/models/not-a-target/history").status_code == 404
    assert client.get("/api/models/not-a-target/parity").status_code == 404
    assert client.get("/api/nsga2").json()["candidate_count"] == 2
    verification = client.get("/api/verification").json()
    assert verification["final"]["status"] == "pass"
    assert verification["standard_candidates"][0]["evaluation"]["timing_seconds"] == {
        "matrix": 353.31,
        "loss": 1720.78,
        "icepak": 1039.83,
        "total": 3113.92,
    }
    assert client.get("/api/history").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_cohorts_page_compacts_legacy_noise_and_keeps_active_first():
    cohorts_js = Path(__file__).resolve().parents[1] / "static" / "cohorts.js"
    script = r"""
const path = require("node:path");
const { compactCohorts } = require(path.resolve(process.argv[1]));
const revision = "mft1mw-1k101-native-lamination-kf0p85-v3";
const cohort = (sha, savedAt, raw = 1, physicsRevision = revision) => ({
  git_hash: sha.repeat(40), git_hash_short: sha.repeat(10),
  physics_data_revision: physicsRevision, latest_saved_at: savedAt,
  active: false, raw_rows: raw, strict_em_rows: 0,
  strict_full_rows: 0, growth_rate_per_hour: 0,
});
const cohorts = [
  { ...cohort("a", "2026-07-13T10:00:00+09:00", 8), active: true,
    strict_em_rows: 7, strict_full_rows: 6, growth_rate_per_hour: 2 },
  cohort("b", "2026-07-13T09:59:00.000900+09:00"),
  cohort("c", "2026-07-13T09:59:00.000700+09:00"),
  cohort("d", "2026-07-13T09:59:00.000600+09:00"),
  cohort("e", "2026-07-13T09:59:00.000800+09:00", 2, "legacy_unspecified"),
  cohort("f", "2026-07-13T09:59:00.000500+09:00", 3, "legacy_unspecified"),
];
process.stdout.write(JSON.stringify(compactCohorts(cohorts)));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(cohorts_js)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    rows = json.loads(completed.stdout)

    assert [row["git_hash_short"] for row in rows] == [
        "a" * 10,
        "b" * 10,
        "legacy",
        "c" * 10,
        "d" * 10,
    ]
    assert rows[0]["active"] is True
    aggregate = rows[2]
    assert aggregate["legacy_aggregate"] is True
    assert aggregate["cohort_count"] == 2
    assert aggregate["raw_rows"] == 5
    assert aggregate["latest_saved_at"] == "2026-07-13T09:59:00.000800+09:00"


def test_api_failure_is_section_local(campaign_root):
    class BrokenService:
        def dashboard(self):
            raise RuntimeError("broken artifact")

    client = TestClient(create_app(regression_root=campaign_root, service=BrokenService()))
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "broken artifact" in response.json()["error"]


def test_local_operator_can_set_exact_bounded_parallel_target(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    client = TestClient(
        create_app(service=artifact_service),
        base_url="http://127.0.0.1:8010",
        client=("127.0.0.1", 51000),
    )
    response = client.patch(
        "/api/operator/parallel-target",
        headers={
            "Content-Type": "application/json",
            "X-MFT-Operator-Control": "parallel-target-v1",
            "Origin": "http://127.0.0.1:8010",
        },
        json={"target": 275},
    )

    assert response.status_code == 200
    assert response.json()["updated"] is True
    assert response.json()["project"] == "MFT_1MW_2026v1"
    assert response.json()["parallel_target"] == 275
    assert artifact_service.scheduler.parallel_target == 275


def test_parallel_target_control_rejects_csrf_remote_and_invalid_requests(
        artifact_service, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "regression_260707.monitoring.app.CAMPAIGN_MUTATION_LOCK_PATH",
        tmp_path / "campaign-mutation.lock",
    )
    app = create_app(service=artifact_service)
    local = TestClient(
        app,
        base_url="http://127.0.0.1:8010",
        client=("127.0.0.1", 51001),
    )
    valid_headers = {
        "Content-Type": "application/json",
        "X-MFT-Operator-Control": "parallel-target-v1",
    }

    assert local.patch(
        "/api/operator/parallel-target",
        headers={"Content-Type": "application/json"},
        json={"target": 300},
    ).status_code == 403
    assert local.patch(
        "/api/operator/parallel-target",
        headers={**valid_headers, "Origin": "https://attacker.invalid"},
        json={"target": 300},
    ).status_code == 403
    assert local.patch(
        "/api/operator/parallel-target",
        headers={**valid_headers, "Host": "monitor.public.invalid"},
        json={"target": 300},
    ).status_code == 403
    for invalid in (0, 301, -1, 1.5, True, "300"):
        response = local.patch(
            "/api/operator/parallel-target",
            headers=valid_headers,
            json={"target": invalid},
        )
        assert response.status_code == 422

    remote = TestClient(
        app,
        base_url="http://127.0.0.1:8010",
        client=("192.0.2.10", 51002),
    )
    assert remote.patch(
        "/api/operator/parallel-target",
        headers=valid_headers,
        json={"target": 300},
    ).status_code == 403
