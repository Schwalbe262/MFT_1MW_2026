from __future__ import annotations

import copy

import pytest

from tools import mft_goal_diagnostic_compact_retry2_collect as overlay2


def _prior_context_and_authority(
    retry_count=47,
    *,
    include_selected_retry_count=False,
):
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
    logical_count = len(entries)
    authority = {
        "logical_seed_count": logical_count,
        "physical_submitted_attempt_count": logical_count + retry_count,
        "selected_physical_attempt_count": logical_count,
        "ignored_infrastructure_prestart_attempt_count": retry_count,
        "selections": selections,
    }
    if include_selected_retry_count:
        authority["selected_retry_attempt_count"] = retry_count
    return {"entries": entries}, authority


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


def test_prior_retry_count_is_read_dynamically_from_sealed_authority():
    _, prior_authority = _prior_context_and_authority(retry_count=23)

    assert "selected_retry_attempt_count" not in prior_authority
    assert overlay2._validated_prior_retry_count(prior_authority) == 23


@pytest.mark.parametrize(
    "field",
    [
        "logical_seed_count",
        "physical_submitted_attempt_count",
        "selected_physical_attempt_count",
        "ignored_infrastructure_prestart_attempt_count",
    ],
)
def test_prior_retry_count_rejects_boolean_counts(field):
    _, prior_authority = _prior_context_and_authority()
    prior_authority[field] = True

    with pytest.raises(RuntimeError, match="count type mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


@pytest.mark.parametrize(
    "field",
    [
        "logical_seed_count",
        "physical_submitted_attempt_count",
        "selected_physical_attempt_count",
        "ignored_infrastructure_prestart_attempt_count",
    ],
)
def test_prior_retry_count_rejects_missing_required_counts(field):
    _, prior_authority = _prior_context_and_authority()
    prior_authority.pop(field)

    with pytest.raises(RuntimeError, match="count type mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


@pytest.mark.parametrize("retry_count", [0, 101])
def test_prior_retry_count_rejects_out_of_range_values(retry_count):
    _, prior_authority = _prior_context_and_authority()
    prior_authority[
        "ignored_infrastructure_prestart_attempt_count"
    ] = retry_count

    with pytest.raises(RuntimeError, match="count range mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


def test_prior_retry_count_rejects_boolean_optional_selected_count():
    _, prior_authority = _prior_context_and_authority(
        include_selected_retry_count=True
    )
    prior_authority["selected_retry_attempt_count"] = True

    with pytest.raises(RuntimeError, match="count type mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_submitted_attempt_count", 148),
        ("selected_physical_attempt_count", 99),
    ],
)
def test_prior_retry_count_rejects_cross_count_mismatch(field, value):
    _, prior_authority = _prior_context_and_authority()
    prior_authority[field] = value

    with pytest.raises(RuntimeError, match="count accounting mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


def test_prior_retry_count_rejects_optional_selected_count_mismatch():
    _, prior_authority = _prior_context_and_authority(
        include_selected_retry_count=True
    )
    prior_authority["selected_retry_attempt_count"] = 46

    with pytest.raises(RuntimeError, match="count accounting mismatch"):
        overlay2._validated_prior_retry_count(prior_authority)


def test_retry2_overlay_has_no_scheduler_mutation_client():
    assert not hasattr(overlay2, "SchedulerClient")
    assert "POST /api/tasks" not in (overlay2.__doc__ or "")
