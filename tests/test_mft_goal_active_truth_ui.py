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
    assert "gap2 0.35-2.00" in rendered
    assert "gap2=0.350" in rendered
    assert "temperature_C: primary<=110 / secondary<=130 / core<=130" in rendered
    assert "core cooling plate thickness=20.0mm exactly" in rendered
    assert "winding cold-plate thickness=20.0mm exactly" in rendered
    assert "LOGICAL 0" in rendered
    assert "ATTEMPTS 0" in rendered
    assert "RAW 2-NET METRIC IS PROVISIONAL" in rendered
    assert "turn-graded" in rendered
    assert "MAX 136.247°C" in rendered
    assert "legacy diagnostic geometry" in rendered


def _write_bound_receipts(tmp_path, monkeypatch):
    original_tasks = [
        {
            "task_id": 97145 + index,
            "seed": 2607264100 + index,
            "name": (
                f"{updater.ACTIVE_EXACT100_BUNDLE_ID}-"
                f"s{2607264100 + index}-n1-6"
            ),
            "dedupe_key": "mft-goal-20260726-diag-compact:" + f"{index:064x}",
        }
        for index in range(100)
    ]
    original_unsigned = {
        "schema_version": updater.ACTIVE_EXACT100_RECEIPT_SCHEMA,
        "bundle_id": updater.ACTIVE_EXACT100_BUNDLE_ID,
        "apply": True,
        "task_count": 100,
        "tasks": original_tasks,
        "submitted_count": 100,
        "existing_count": 0,
        "absent_count": 0,
        "campaign_authorized_post_count": 100,
        "first_clean_run_exact100_scheduler_posts": True,
        "scheduler_post_count": 100,
        "scheduler_endpoint": "POST /api/tasks",
    }
    original = {
        **original_unsigned,
        "sha256": updater.canonical_sha256(original_unsigned),
    }
    (tmp_path / "submission_receipt.json").write_text(
        json.dumps(original, sort_keys=True),
        encoding="utf-8",
    )

    allowed = ["dhj02", "jji0930", "r1jae262", "dw16", "wjddn5916"]
    retry_tasks = [
        {
            "task_id": 97245 + index,
            "seed": 2607264153 + index,
            "name": (
                f"{updater.ACTIVE_EXACT100_BUNDLE_ID}-retry1-"
                f"s{2607264153 + index}-n1-6-a{allowed[index % len(allowed)]}"
            ),
            "dedupe_key": (
                "mft-goal-20260726-diag-compact:"
                + f"{1000 + index:064x}"
            ),
            "requested_account": allowed[index % len(allowed)],
        }
        for index in range(47)
    ]
    retry_unsigned = {
        "schema_version": updater.ACTIVE_EXACT100_RETRY_RECEIPT_SCHEMA,
        "apply": True,
        "retry_task_count": 47,
        "tasks": retry_tasks,
        "submitted_count": 47,
        "existing_count": 0,
        "absent_count": 0,
        "scheduler_post_count": 47,
        "allowed_accounts": allowed,
        "failed_account_excluded": "harry261",
        "target_active_inventory": 100,
        "equal_three_leg_air_gap_FEA_required": True,
        "physical_Lm_2mH_symmetric_FEA_verification_required": True,
        "automatic_final_promotion_allowed": False,
        "final_promotion_allowed": False,
        "screening_only": True,
        "production_eligible": False,
    }
    retry = {
        **retry_unsigned,
        "payload_sha256": updater.canonical_sha256(retry_unsigned),
    }
    retry_root = tmp_path / "retry1_nonharry"
    retry_root.mkdir()
    (retry_root / "submission_receipt.json").write_text(
        json.dumps(retry, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        updater,
        "ACTIVE_EXACT100_ORIGINAL_PAYLOAD_SHA256",
        original["sha256"],
    )
    monkeypatch.setattr(
        updater,
        "ACTIVE_EXACT100_RETRY_PAYLOAD_SHA256",
        retry["payload_sha256"],
    )
    return original_tasks, retry_tasks


def test_active_retry_truth_reconciles_147_attempts_to_100_live_seeds(
    tmp_path,
    monkeypatch,
):
    original_tasks, retry_tasks = _write_bound_receipts(tmp_path, monkeypatch)
    observed = {}
    for index, task in enumerate(original_tasks):
        failed = index >= 53
        observed[task["task_id"]] = {
            **task,
            "id": task["task_id"],
            "state": "failed" if failed else "running",
            "account_name": "harry261" if failed else "r1jae262",
            "started_at": None if failed else "2026-07-27 07:47:53",
            "finished_at": "2026-07-27 07:51:45" if failed else None,
            "failure_message": "Failure" if failed else "",
            "remote_dir": "" if failed else "runs/task",
            "stdout_path": "" if failed else "runs/task/stdout.log",
            "stderr_path": "" if failed else "runs/task/stderr.log",
            "exit_code_path": "" if failed else "runs/task/exit_code",
        }
    for task in retry_tasks:
        observed[task["task_id"]] = {
            **task,
            "id": task["task_id"],
            "state": "queued",
            "account_name": "",
            "started_at": None,
            "finished_at": None,
            "failure_message": "",
            "remote_dir": "",
            "stdout_path": "",
            "stderr_path": "",
            "exit_code_path": "",
        }

    state = updater._active_exact100_retry_state(
        scheduler_url="http://scheduler.invalid",
        root=tmp_path,
        task_reader=lambda _url, task_id: observed[task_id],
    )

    assert state["logical_seed_count"] == 100
    assert state["submitted_attempt_count"] == 147
    assert state["original_infrastructure_failed_count"] == 47
    assert state["retry_identity_count"] == 47
    assert state["active_nonterminal_seed_count"] == 100
    assert state["scheduler_get_count"] == 147
    assert state["equal_three_leg_air_gap_FEA_required"] is True
    assert state["final_promotion_allowed"] is False
    assert state["scheduler_mutation_performed"] is False

    cards = updater._active_truth_ui_cards(
        observed_at="2026-07-27T17:10:00+09:00",
        exact100=updater._active_exact100_post_state(tmp_path),
        retry_state=state,
    )
    rendered = json.dumps(cards, ensure_ascii=False)
    assert "LOGICAL 100" in rendered
    assert "ATTEMPTS 147" in rendered
    assert "ACTIVE 100" in rendered
    assert "INFRA-FAILED 47" in rendered
    assert "RETRY 47" in rendered
    assert "equal_three_leg_air_gap_FEA_required=true" in rendered
    assert "final_promotion_allowed=false" in rendered
