from fastapi.testclient import TestClient

from regression_260707.monitoring.app import create_app


def test_dashboard_page_and_all_read_only_apis(artifact_service):
    client = TestClient(create_app(service=artifact_service))
    page = client.get("/")
    assert page.status_code == 200
    assert "최적설계 파이프라인" in page.text
    assert "data-chart" in page.text

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["data"]["total_rows"] == 2

    assert client.get("/api/status").status_code == 200
    assert client.get("/api/data").json()["complete_rows"] == 1
    assert client.get("/api/models").json()["trained_count"] == 2
    assert client.get("/api/models/Llt_phys/history").json()["target"] == "Llt_phys"
    assert client.get("/api/models/not-a-target/history").status_code == 404
    assert client.get("/api/nsga2").json()["candidate_count"] == 2
    assert client.get("/api/verification").json()["final"]["status"] == "pass"
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
