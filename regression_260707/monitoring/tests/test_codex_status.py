from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from regression_260707.monitoring.app import create_app
from regression_260707.monitoring.codex_status import (
    SCHEMA_VERSION,
    CodexWorkStatusReader,
)


def _payload(now: datetime) -> dict:
    timestamp = now.isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "goal_id": "mft-goal-20260726",
        "goal": "MFT 1 MW 설계 목표 작업 현황",
        "owner": "Codex",
        "generated_at": timestamp,
        "deadline_at": "2026-07-26T18:00:00+09:00",
        "summary": "다중 시드 NSGA-II와 최종 FEA 인계를 진행 중입니다.",
        "current": [{
            "id": "nsga-offload",
            "title": "NSGA-II Slurm 오프로드",
            "detail": "봉인된 번들을 스테이징하고 있습니다.",
            "state": "in_progress",
            "updated_at": timestamp,
            "progress_pct": 35,
            "evidence": ["offload_plan.json"],
        }],
        "completed": [{
            "id": "goal-contract",
            "title": "설계 계약 고정",
            "detail": "치수·공진·온도·냉각 계약을 고정했습니다.",
            "state": "completed",
            "updated_at": timestamp,
            "evidence": ["goal_contract.json"],
        }],
        "attention": [{
            "id": "surrogate-quality",
            "title": "대리모델 strict gate 미통과",
            "detail": "탐색 제안용으로만 사용하며 FEA 검증이 필수입니다.",
            "state": "attention",
            "updated_at": timestamp,
            "evidence": ["quality_status.json"],
        }],
    }


def test_reader_accepts_explicit_timestamped_artifact(tmp_path):
    now = datetime.now(timezone.utc)
    path = tmp_path / "codex-work-status.json"
    path.write_text(
        json.dumps(_payload(now), ensure_ascii=False),
        encoding="utf-8",
    )

    snapshot = CodexWorkStatusReader(
        path,
        stale_after_seconds=1800,
    ).snapshot()

    assert snapshot["available"] is True
    assert snapshot["integrity_verified"] is True
    assert snapshot["goal_id"] == "mft-goal-20260726"
    assert snapshot["counts"] == {
        "current": 1,
        "completed": 1,
        "attention": 1,
    }
    assert snapshot["current"][0]["state"] == "in_progress"
    assert snapshot["stale"] is False


def test_reader_fails_closed_for_cross_group_state(tmp_path):
    now = datetime.now(timezone.utc)
    payload = _payload(now)
    payload["current"][0]["state"] = "completed"
    path = tmp_path / "codex-work-status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = CodexWorkStatusReader(path).snapshot()

    assert snapshot["available"] is False
    assert snapshot["integrity_verified"] is False
    assert "invalid for current" in snapshot["error"]


def test_reader_marks_old_but_valid_artifact_stale(tmp_path):
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    path = tmp_path / "codex-work-status.json"
    path.write_text(json.dumps(_payload(now)), encoding="utf-8")

    snapshot = CodexWorkStatusReader(
        path,
        stale_after_seconds=300,
    ).snapshot()

    assert snapshot["available"] is True
    assert snapshot["stale"] is True
    assert snapshot["age_seconds"] >= 7100


def test_api_and_first_page_expose_codex_work_panel(tmp_path):
    now = datetime.now(timezone.utc)
    expected = _payload(now)
    expected.update({
        "available": True,
        "integrity_verified": True,
        "counts": {"current": 1, "completed": 1, "attention": 1},
        "stale": False,
    })

    class StubStatusReader:
        def snapshot(self):
            return expected

    client = TestClient(create_app(
        regression_root=tmp_path,
        service=object(),
        codex_status_reader=StubStatusReader(),
    ))

    page = client.get("/")
    assert page.status_code == 200
    assert 'id="codex-work-panel"' in page.text
    assert "현재 작업 중" in page.text
    assert "주의 / 차단" in page.text

    response = client.get("/api/codex-work")
    assert response.status_code == 200
    assert response.json()["goal_id"] == "mft-goal-20260726"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-type"] == "application/json; charset=utf-8"
