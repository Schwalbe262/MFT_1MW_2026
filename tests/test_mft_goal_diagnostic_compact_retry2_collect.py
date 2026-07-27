from __future__ import annotations

import copy

import pytest

from tools import mft_goal_diagnostic_compact_retry2_collect as overlay2


def _prior_context_and_authority():
    entries = []
    selections = []
    for ordinal, seed in enumerate(overlay2.offload.EXACT_SEEDS):
        task_id = 10_000 + ordinal
        entries.append(
            {
                "seed": seed,
                "task_id": task_id,
                "dedupe_key": f"base-{seed}",
                "task": {"seed": seed},
                "scheduler_payload": {
                    "name": f"base-{seed}",
                    "dedupe_key": f"base-{seed}",
                },
            }
        )
        selections.append(
            {
                "seed": seed,
                "selected_task_id": task_id,
                "selected_dedupe_key": f"base-{seed}",
                "attempt_class": "first_clean_original",
                "ignored_original_task_id": None,
            }
        )
    return {"entries": entries}, {"selections": selections}


def _retry2_plan_and_receipt(prior_context):
    prior_by_seed = {
        int(row["seed"]): row for row in prior_context["entries"]
    }
    tasks = []
    receipts = []
    for ordinal, seed in enumerate(overlay2.retry2.EXPECTED_SEEDS):
        original_task_id = prior_by_seed[seed]["task_id"]
        account = overlay2.retry2.ALLOWED_ACCOUNTS[ordinal]
        payload = {
            "name": f"retry2-{seed}",
            "dedupe_key": f"retry2-dedupe-{seed}",
            "account_name": account,
        }
        tasks.append(
            {
                "seed": seed,
                "requested_account": account,
                "scheduler_payload": payload,
                "original_failure": {
                    "task_id": original_task_id,
                    "status": "failed",
                    "account_name": overlay2.retry2.FAILED_ACCOUNT,
                    "exit_code": 1,
                    "runtime_paths_nonempty": True,
                },
            }
        )
        receipts.append(
            {
                "seed": seed,
                "task_id": 20_000 + ordinal,
                "name": payload["name"],
                "dedupe_key": payload["dedupe_key"],
                "requested_account": account,
                "status": "queued",
            }
        )
    return {"tasks": tasks}, {"tasks": receipts}


def test_upgrade_replaces_exact_five_and_preserves_other_95():
    prior_context, prior_authority = _prior_context_and_authority()
    retry2_plan, retry2_receipt = _retry2_plan_and_receipt(prior_context)
    before = copy.deepcopy(prior_context)
    entries, selections = overlay2._upgrade_entries(
        prior_context=prior_context,
        prior_authority=prior_authority,
        retry2_plan=retry2_plan,
        retry2_receipt=retry2_receipt,
    )
    assert prior_context == before
    assert len(entries) == len(selections) == 100
    replaced = [
        row for row in entries
        if row.get("attempt_class") == "runtime_exit1_retry2"
    ]
    assert [row["seed"] for row in replaced] == list(
        overlay2.retry2.EXPECTED_SEEDS
    )
    assert [row["task_id"] for row in replaced] == list(
        range(20_000, 20_005)
    )
    preserved = [
        row for row in entries
        if row["seed"] not in overlay2.retry2.EXPECTED_SEEDS
    ]
    assert all(
        row["task_id"] == 10_000 + ordinal
        for ordinal, row in enumerate(entries)
        if row["seed"] not in overlay2.retry2.EXPECTED_SEEDS
    )
    assert len(preserved) == 95
    assert len({row["task_id"] for row in entries}) == 100


def test_upgrade_fails_closed_if_retry2_targets_retry1_selection():
    prior_context, prior_authority = _prior_context_and_authority()
    retry2_plan, retry2_receipt = _retry2_plan_and_receipt(prior_context)
    seed = overlay2.retry2.EXPECTED_SEEDS[0]
    selection = next(
        row for row in prior_authority["selections"] if row["seed"] == seed
    )
    selection["attempt_class"] = "quota_prestart_retry1"
    with pytest.raises(RuntimeError, match="runtime-failed original"):
        overlay2._upgrade_entries(
            prior_context=prior_context,
            prior_authority=prior_authority,
            retry2_plan=retry2_plan,
            retry2_receipt=retry2_receipt,
        )


def test_retry2_overlay_has_no_scheduler_mutation_client():
    assert not hasattr(overlay2, "SchedulerClient")
    assert "POST /api/tasks" not in (overlay2.__doc__ or "")
