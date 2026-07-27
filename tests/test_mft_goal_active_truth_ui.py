from __future__ import annotations

import json

from tools import mft_goal_postdeadline_ui_updater as updater


def _sealed_receipt() -> dict:
    unsigned = {
        "schema_version": updater.ACTIVE_EXACT100_RECEIPT_SCHEMA,
        "bundle_id": updater.ACTIVE_EXACT100_BUNDLE_ID,
        "apply": True,
        "task_count": 100,
        "tasks": [
            {"task_id": 100_000 + index}
            for index in range(100)
        ],
        "submitted_count": 100,
        "existing_count": 0,
        "absent_count": 0,
        "campaign_authorized_post_count": 100,
        "first_clean_run_exact100_scheduler_posts": True,
        "scheduler_post_count": 100,
        "scheduler_endpoint": "POST /api/tasks",
    }
    return {
        **unsigned,
        "sha256": updater.canonical_sha256(unsigned),
    }


def test_active_exact100_ui_fails_closed_before_sealed_post(tmp_path):
    state = updater._active_exact100_post_state(tmp_path)

    assert state["phase"] == "preparing"
    assert state["submitted_count"] == 0
    assert state["receipt_path"] is None


def test_active_exact100_ui_accepts_only_complete_first_clean_receipt(tmp_path):
    receipt_path = tmp_path / "submission_receipt.json"
    receipt_path.write_text(
        json.dumps(_sealed_receipt(), sort_keys=True),
        encoding="utf-8",
    )

    state = updater._active_exact100_post_state(tmp_path)

    assert state["phase"] == "submitted"
    assert state["submitted_count"] == 100
    assert state["task_id_min"] == 100_000
    assert state["task_id_max"] == 100_099


def test_active_truth_cards_publish_current_contract_and_boundaries(tmp_path):
    state = updater._active_exact100_post_state(tmp_path)

    cards = updater._active_truth_ui_cards(
        observed_at="2026-07-27T16:20:00+09:00",
        exact100=state,
    )

    rendered = json.dumps(cards, ensure_ascii=False)
    assert "1200×900×750" in rendered
    assert "gap2 0.350" in rendered
    assert "1차≤110°C" in rendered
    assert "2차≤130°C" in rendered
    assert "20T/20T" in rendered
    assert "SUBMITTED 0/100" in rendered
    assert "RAW 2-NET METRIC IS PROVISIONAL" in rendered
    assert "turn-graded" in rendered
    assert "MAX 136.247°C" in rendered
    assert "legacy diagnostic geometry" in rendered
