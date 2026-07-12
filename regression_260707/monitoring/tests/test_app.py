from fastapi.testclient import TestClient

from regression_260707.model_targets import CORE_REGION_TEMPERATURE_TARGETS
from regression_260707.monitoring.app import create_app


def test_dashboard_page_and_all_read_only_apis(artifact_service):
    client = TestClient(create_app(service=artifact_service))
    page = client.get("/")
    assert page.status_code == 200
    assert "최적설계 파이프라인" in page.text
    assert "data-chart" in page.text
    assert "최신 revision 학습 가능" in page.text
    assert "전체 수집 데이터" in page.text
    assert "신규 solver 결과 · 1시간" in page.text
    assert "strict-valid 누적" in page.text
    assert "정책 재판정에 따른 증감 포함" in page.text
    assert "data-raw-total" in page.text
    assert "최근 시뮬레이션 단계별 소요시간" in page.text
    assert "stage-time-matrix-mean" in page.text
    assert "단계 소요시간" in page.text
    assert "final-time-matrix" in page.text
    assert "MFT 병렬 실행 목표" in page.text
    assert "parallel-target-input" in page.text
    assert "parallel-logical-active" in page.text
    assert "parallel-attaching" in page.text
    assert 'id="model-history-metric"' in page.text
    assert 'value="mape_pct"' in page.text
    assert "학습 데이터 수 기준" in page.text

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "function duration(value)" in script.text
    assert "data.pinned_revision" in script.text
    assert "data.raw_total_rows" in script.text
    assert "data.new_solver_results_1h" in script.text
    assert "data.strict_valid_growth_1h" in script.text
    assert "격리" in script.text
    assert "data.simulation_timing" in script.text
    assert "timingCell(evaluation.timing_seconds)" in script.text
    assert 'return "—"' in script.text
    assert 'fetch("/api/operator/parallel-target"' in script.text
    assert '"X-MFT-Operator-Control": "parallel-target-v1"' in script.text
    assert "scheduler.live_queued" in script.text
    assert "x: (item) => Number(item.n)" in script.text
    assert "historyPointTooltip" in script.text
    assert "CV P90 APE" in script.text

    stylesheet = client.get("/static/app.css")
    assert stylesheet.status_code == 200
    assert ".timing-grid" in stylesheet.text
    assert ".stage-timing-grid" in stylesheet.text
    assert ".history-metric-control" in stylesheet.text
    assert ".chart-tooltip" in stylesheet.text

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["data"]["total_rows"] == 1
    assert dashboard.json()["data"]["raw_total_rows"] == 2
    assert dashboard.json()["data"]["new_solver_results_1h"] == 1
    assert dashboard.json()["data"]["strict_valid_growth_1h"] == 1
    assert dashboard.json()["data"]["count_basis"] == "pinned_strict_full"
    assert dashboard.json()["data"]["latest_revision"] == "754923cf1c97bc45bcd9d8c6ba60d98773a5c30a"
    assert dashboard.json()["data"]["pinned_revision"] == "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
    assert dashboard.json()["data"]["simulation_timing"]["stages"]["total"]["mean_seconds"] == 3300.0

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


def test_api_failure_is_section_local(campaign_root):
    class BrokenService:
        def dashboard(self):
            raise RuntimeError("broken artifact")

    client = TestClient(create_app(regression_root=campaign_root, service=BrokenService()))
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert "broken artifact" in response.json()["error"]


def test_local_operator_can_set_exact_bounded_parallel_target(artifact_service):
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


def test_parallel_target_control_rejects_csrf_remote_and_invalid_requests(artifact_service):
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
