"""Collect exact100 compact seeds through retry1 plus runtime retry2.

The existing base/retry1 logical overlay remains immutable.  This adapter
authenticates it, replaces only the exact five GET-proven runtime-failed
original attempts with their retry2 tasks, and passes one selected physical
attempt per logical seed to the read-only collector.

This module has no Scheduler mutation client.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import mft_goal_diagnostic_compact_collect as collector  # noqa: E402
from tools import mft_goal_diagnostic_compact_retry as retry1  # noqa: E402
from tools import mft_goal_diagnostic_compact_retry_collect as overlay1  # noqa: E402
from tools import mft_goal_diagnostic_compact_runtime_retry as retry2  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402


OVERLAY_SCHEMA = "mft-goal-diagnostic-compact-logical-seed-overlay-v2"
EXPECTED_RETRY1_COUNT = overlay1.EXPECTED_RETRY_COUNT
EXPECTED_RETRY2_COUNT = len(retry2.EXPECTED_SEEDS)


def _validate_retry2_receipt(
    *,
    retry2_plan_path: Path,
    retry2_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = retry2._validate_plan(retry2._read_json(retry2_plan_path))
    receipt = retry2._validate_seal(
        retry2._read_json(retry2_receipt_path),
        retry2.RECEIPT_SCHEMA,
    )
    plan_by_seed = {
        int(row["seed"]): row for row in plan["tasks"]
    }
    rows = receipt.get("tasks")
    if (
        receipt.get("apply") is not True
        or receipt.get("runtime_retry2_plan")
        != retry2._record(retry2_plan_path)
        or receipt.get("runtime_retry2_plan_payload_sha256")
        != plan["payload_sha256"]
        or receipt.get("retry_attempt") != retry2.RETRY_ATTEMPT
        or receipt.get("retry_task_count") != EXPECTED_RETRY2_COUNT
        or receipt.get("absent_count") != 0
        or receipt.get("one_selected_attempt_per_logical_seed") is not True
        or receipt.get("scheduler_cancel_count") != 0
        or receipt.get("scheduler_preempt_count") != 0
        or receipt.get("allowed_accounts") != list(retry2.ALLOWED_ACCOUNTS)
        or receipt.get("equal_three_leg_air_gap_FEA_required") is not True
        or receipt.get("final_promotion_allowed") is not False
        or receipt.get("screening_only") is not True
        or receipt.get("production_eligible") is not False
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_RETRY2_COUNT
    ):
        raise RuntimeError("retry2 collector receipt contract mismatch")
    observed_seeds = []
    task_ids = set()
    for row in rows:
        seed = int(row.get("seed", -1))
        expected = plan_by_seed.get(seed)
        task_id = row.get("task_id")
        if (
            expected is None
            or isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or task_id in task_ids
            or row.get("requested_account")
            != expected["requested_account"]
            or row.get("dedupe_key")
            != expected["scheduler_payload"]["dedupe_key"]
            or row.get("name")
            != expected["scheduler_payload"]["name"]
            or row.get("status")
            not in retry1.NONTERMINAL | retry1.TERMINAL
        ):
            raise RuntimeError("retry2 collector task mapping mismatch")
        observed_seeds.append(seed)
        task_ids.add(task_id)
    if (
        tuple(observed_seeds) != retry2.EXPECTED_SEEDS
        or set(plan_by_seed) != set(retry2.EXPECTED_SEEDS)
    ):
        raise RuntimeError("retry2 collector seed coverage mismatch")
    return plan, receipt


def _upgrade_entries(
    *,
    prior_context: Mapping[str, Any],
    prior_authority: Mapping[str, Any],
    retry2_plan: Mapping[str, Any],
    retry2_receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace exactly five original failures and preserve all other choices."""

    prior_entries = {
        int(entry["seed"]): copy.deepcopy(entry)
        for entry in prior_context["entries"]
    }
    prior_selections = {
        int(row["seed"]): copy.deepcopy(row)
        for row in prior_authority["selections"]
    }
    plan_by_seed = {
        int(row["seed"]): row for row in retry2_plan["tasks"]
    }
    receipt_by_seed = {
        int(row["seed"]): row for row in retry2_receipt["tasks"]
    }
    if (
        set(prior_entries) != set(offload.EXACT_SEEDS)
        or set(prior_selections) != set(offload.EXACT_SEEDS)
        or set(plan_by_seed) != set(retry2.EXPECTED_SEEDS)
        or set(receipt_by_seed) != set(retry2.EXPECTED_SEEDS)
    ):
        raise RuntimeError("retry2 logical overlay coverage mismatch")

    selected_entries = []
    selections = []
    for seed in offload.EXACT_SEEDS:
        prior_entry = prior_entries[seed]
        prior_selection = prior_selections[seed]
        plan_row = plan_by_seed.get(seed)
        receipt_row = receipt_by_seed.get(seed)
        if (plan_row is None) != (receipt_row is None):
            raise RuntimeError("retry2 plan and receipt coverage differs")
        if plan_row is None:
            selected = prior_entry
            selection = prior_selection
        else:
            original = plan_row.get("original_failure") or {}
            if (
                prior_selection.get("attempt_class")
                != "first_clean_original"
                or int(prior_entry["task_id"])
                != int(original.get("task_id", -1))
                or original.get("status") != "failed"
                or original.get("account_name") != retry2.FAILED_ACCOUNT
                or original.get("exit_code") != 1
                or original.get("runtime_paths_nonempty") is not True
            ):
                raise RuntimeError(
                    "retry2 did not replace its exact runtime-failed original"
                )
            selected = {
                **prior_entry,
                "task_id": int(receipt_row["task_id"]),
                "dedupe_key": str(receipt_row["dedupe_key"]),
                "scheduler_payload": copy.deepcopy(
                    plan_row["scheduler_payload"]
                ),
                "attempt_class": "runtime_exit1_retry2",
                "replaced_task_id": int(prior_entry["task_id"]),
            }
            selection = {
                "seed": seed,
                "selected_task_id": int(receipt_row["task_id"]),
                "selected_dedupe_key": str(receipt_row["dedupe_key"]),
                "attempt_class": "runtime_exit1_retry2",
                "ignored_original_task_id": int(prior_entry["task_id"]),
            }
        selected_entries.append(selected)
        selections.append(selection)

    selected_ids = [int(entry["task_id"]) for entry in selected_entries]
    if (
        len(selected_entries) != offload.EXACT_TASK_COUNT
        or [int(entry["seed"]) for entry in selected_entries]
        != list(offload.EXACT_SEEDS)
        or len(set(selected_ids)) != offload.EXACT_TASK_COUNT
    ):
        raise RuntimeError("retry2 selected physical attempts are invalid")
    return selected_entries, selections


def authenticate_overlay_context(
    *,
    plan_path: Path,
    base_receipt_path: Path,
    retry1_plan_path: Path,
    retry1_receipt_path: Path,
    prior_authority_path: Path,
    retry2_plan_path: Path,
    retry2_receipt_path: Path,
    authority_path: Path,
    scheduler_url: str,
) -> dict[str, Any]:
    """Authenticate the 152 submitted attempts and select exact logical100."""

    prior = overlay1.authenticate_overlay_context(
        plan_path=plan_path,
        base_receipt_path=base_receipt_path,
        retry_plan_path=retry1_plan_path,
        retry_receipt_path=retry1_receipt_path,
        authority_path=prior_authority_path,
        scheduler_url=scheduler_url,
    )
    prior_authority = collector._validate_sealed(
        collector._read_json(prior_authority_path),
        schema=overlay1.OVERLAY_SCHEMA,
    )
    retry2_plan, retry2_receipt = _validate_retry2_receipt(
        retry2_plan_path=retry2_plan_path,
        retry2_receipt_path=retry2_receipt_path,
    )
    if (
        retry2_plan.get("base_plan")
        != retry2._record(plan_path)
        or retry2_plan.get("base_receipt")
        != retry2._record(base_receipt_path)
        or retry2_plan.get("retry1_plan")
        != retry2._record(retry1_plan_path)
        or retry2_plan.get("retry1_receipt")
        != retry2._record(retry1_receipt_path)
        or prior_authority.get("physical_submitted_attempt_count")
        != offload.EXACT_TASK_COUNT + EXPECTED_RETRY1_COUNT
        or prior_authority.get("selected_physical_attempt_count")
        != offload.EXACT_TASK_COUNT
    ):
        raise RuntimeError("retry2 collector prior authority drifted")

    selected_entries, selections = _upgrade_entries(
        prior_context=prior,
        prior_authority=prior_authority,
        retry2_plan=retry2_plan,
        retry2_receipt=retry2_receipt,
    )
    runtime_ignored = [
        int(row["original_failure"]["task_id"])
        for row in retry2_plan["tasks"]
    ]
    authority_created_at = retry2_receipt["observed_at"]
    if authority_path.is_file():
        existing = collector._validate_sealed(
            collector._read_json(authority_path),
            schema=OVERLAY_SCHEMA,
        )
        authority_created_at = existing["created_at"]
    authority = collector._sealed(
        {
            "schema_version": OVERLAY_SCHEMA,
            "created_at": authority_created_at,
            "prior_overlay_authority": {
                "path": str(prior_authority_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    prior_authority_path.resolve(strict=True)
                ),
                "payload_sha256": prior_authority["payload_sha256"],
            },
            "runtime_retry2_plan": {
                "path": str(retry2_plan_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    retry2_plan_path.resolve(strict=True)
                ),
                "payload_sha256": retry2_plan["payload_sha256"],
            },
            "runtime_retry2_receipt": {
                "path": str(retry2_receipt_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    retry2_receipt_path.resolve(strict=True)
                ),
                "payload_sha256": retry2_receipt["payload_sha256"],
            },
            "logical_seed_count": offload.EXACT_TASK_COUNT,
            "physical_submitted_attempt_count": (
                offload.EXACT_TASK_COUNT
                + EXPECTED_RETRY1_COUNT
                + EXPECTED_RETRY2_COUNT
            ),
            "selected_physical_attempt_count": offload.EXACT_TASK_COUNT,
            "selected_retry1_attempt_count": EXPECTED_RETRY1_COUNT,
            "selected_retry2_attempt_count": EXPECTED_RETRY2_COUNT,
            "ignored_infrastructure_prestart_attempt_count": (
                EXPECTED_RETRY1_COUNT
            ),
            "ignored_runtime_exit1_attempt_count": EXPECTED_RETRY2_COUNT,
            "ignored_runtime_exit1_task_ids": runtime_ignored,
            "selections": selections,
            "one_selected_physical_attempt_per_logical_seed": True,
            "duplicate_physical_result_collection_allowed": False,
            "retry_account_set": list(retry2.ALLOWED_ACCOUNTS),
            "failed_account_excluded_from_retry": retry2.FAILED_ACCOUNT,
            "equal_three_leg_air_gap_FEA_required": True,
            "physical_Lm_2mH_symmetric_FEA_verification_required": True,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "final_promotion_allowed": False,
            "scheduler_access_mode": "GET_status_plus_read_only_SFTP",
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
        }
    )
    collector._immutable_json(authority_path, authority)
    physical_attempt_count = (
        offload.EXACT_TASK_COUNT
        + EXPECTED_RETRY1_COUNT
        + EXPECTED_RETRY2_COUNT
    )
    return {
        **prior,
        "entries": selected_entries,
        "receipt_path": str(authority_path.resolve(strict=True)),
        "receipt_file_sha256": collector._sha256_file(
            authority_path.resolve(strict=True)
        ),
        "logical_seed_overlay": {
            "schema_version": OVERLAY_SCHEMA,
            "authority_path": str(authority_path.resolve(strict=True)),
            "authority_file_sha256": collector._sha256_file(
                authority_path.resolve(strict=True)
            ),
            "authority_payload_sha256": authority["payload_sha256"],
            "logical_seed_count": offload.EXACT_TASK_COUNT,
            "physical_submitted_attempt_count": physical_attempt_count,
            "selected_retry_attempt_count": (
                EXPECTED_RETRY1_COUNT + EXPECTED_RETRY2_COUNT
            ),
            "selected_retry1_attempt_count": EXPECTED_RETRY1_COUNT,
            "selected_retry2_attempt_count": EXPECTED_RETRY2_COUNT,
            "one_selected_physical_attempt_per_logical_seed": True,
            "duplicate_physical_result_collection_allowed": False,
            "equal_three_leg_air_gap_FEA_required": True,
            "final_promotion_allowed": False,
        },
    }


def run(
    *,
    plan_path: Path,
    base_receipt_path: Path,
    retry1_plan_path: Path,
    retry1_receipt_path: Path,
    prior_authority_path: Path,
    retry2_plan_path: Path,
    retry2_receipt_path: Path,
    output_root: Path,
    scheduler_url: str = collector.DEFAULT_SCHEDULER_URL,
    accounts_path: Path = collector.DEFAULT_ACCOUNTS,
    scheduler_source: Path = collector.DEFAULT_SCHEDULER_SOURCE,
    retries: int = 3,
    max_polls: int = 1,
    poll_seconds: float = 30.0,
    require_final: bool = False,
    scheduler: collector.SchedulerReader | None = None,
) -> tuple[dict[str, Any], int]:
    if not 1 <= retries <= 10:
        raise ValueError("retries must be within 1..10")
    if not 1 <= max_polls <= 10_000:
        raise ValueError("max_polls must be within 1..10000")
    if not 0.0 <= poll_seconds <= 300.0:
        raise ValueError("poll_seconds must be within 0..300")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    authority_path = output_root / "logical_seed_overlay_authority.json"
    context = authenticate_overlay_context(
        plan_path=plan_path,
        base_receipt_path=base_receipt_path,
        retry1_plan_path=retry1_plan_path,
        retry1_receipt_path=retry1_receipt_path,
        prior_authority_path=prior_authority_path,
        retry2_plan_path=retry2_plan_path,
        retry2_receipt_path=retry2_receipt_path,
        authority_path=authority_path,
        scheduler_url=scheduler_url,
    )
    client = scheduler or collector.ReadOnlySchedulerClient(scheduler_url)
    status: dict[str, Any] = {}
    for poll in range(1, max_polls + 1):
        status = collector.collect_once(
            context=context,
            output_root=output_root,
            scheduler=client,
            accounts_path=accounts_path,
            scheduler_source=scheduler_source,
            retries=retries,
        )
        if status["final_exact100_reached"]:
            break
        if poll < max_polls:
            time.sleep(poll_seconds)
    return status, (
        2 if require_final and not status["final_exact100_reached"] else 0
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--retry1-plan", type=Path, required=True)
    parser.add_argument("--retry1-receipt", type=Path, required=True)
    parser.add_argument("--prior-authority", type=Path, required=True)
    parser.add_argument("--retry2-plan", type=Path, required=True)
    parser.add_argument("--retry2-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scheduler-url", default=collector.DEFAULT_SCHEDULER_URL
    )
    parser.add_argument(
        "--accounts", type=Path, default=collector.DEFAULT_ACCOUNTS
    )
    parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=collector.DEFAULT_SCHEDULER_SOURCE,
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--require-final", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status, exit_code = run(
        plan_path=args.plan,
        base_receipt_path=args.base_receipt,
        retry1_plan_path=args.retry1_plan,
        retry1_receipt_path=args.retry1_receipt,
        prior_authority_path=args.prior_authority,
        retry2_plan_path=args.retry2_plan,
        retry2_receipt_path=args.retry2_receipt,
        output_root=args.output,
        scheduler_url=args.scheduler_url,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        retries=args.retries,
        max_polls=args.max_polls,
        poll_seconds=args.poll_seconds,
        require_final=args.require_final,
    )
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
