"""Collect one exact100 logical seed set across base and retry attempts.

The immutable first-run receipt remains the authority for the exact100 seed
interval declared by its authenticated plan.  The separately sealed pre-start
quota retry receipt replaces only the failed physical attempts named by its
retry plan.  Exactly one physical Scheduler task is selected for every logical
seed before the standard GET/SFTP collector and global NDS pipeline are used.

This module has no Scheduler mutation method.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import mft_goal_diagnostic_compact_collect as collector  # noqa: E402
from tools import mft_goal_diagnostic_compact_retry as retry  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402


OVERLAY_SCHEMA = "mft-goal-diagnostic-compact-logical-seed-overlay-v1"


def _validate_retry_receipt(
    *,
    retry_plan_path: Path,
    retry_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = retry._validate_plan(retry._read_json(retry_plan_path))
    receipt = retry._validate_seal(
        retry._read_json(retry_receipt_path),
        retry.RECEIPT_SCHEMA,
    )
    plan_by_seed = {
        int(row["seed"]): row for row in plan["tasks"]
    }
    expected_retry_count = len(plan_by_seed)
    rows = receipt.get("tasks")
    if (
        not 1 <= expected_retry_count <= offload.EXACT_TASK_COUNT
        or receipt.get("apply") is not True
        or receipt.get("retry_plan_file_sha256")
        != retry._sha256_file(retry_plan_path.resolve(strict=True))
        or receipt.get("retry_plan_payload_sha256")
        != plan["payload_sha256"]
        or receipt.get("retry_task_count") != expected_retry_count
        or receipt.get("absent_count") != 0
        or receipt.get("scheduler_cancel_count") != 0
        or receipt.get("scheduler_preempt_count") != 0
        or receipt.get("allowed_accounts") != list(retry.ALLOWED_ACCOUNTS)
        or receipt.get("failed_account_excluded") != retry.FAILED_ACCOUNT
        or receipt.get("equal_three_leg_air_gap_FEA_required") is not True
        or receipt.get("final_promotion_allowed") is not False
        or receipt.get("screening_only") is not True
        or receipt.get("production_eligible") is not False
        or not isinstance(rows, list)
        or len(rows) != expected_retry_count
    ):
        raise RuntimeError("retry collector receipt contract mismatch")
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
            or row.get("name") != expected["scheduler_payload"]["name"]
        ):
            raise RuntimeError("retry collector task mapping mismatch")
        observed_seeds.append(seed)
        task_ids.add(task_id)
    if (
        observed_seeds != sorted(plan_by_seed)
        or len(plan_by_seed) != expected_retry_count
    ):
        raise RuntimeError("retry collector seed coverage mismatch")
    return plan, receipt


def authenticate_overlay_context(
    *,
    plan_path: Path,
    base_receipt_path: Path,
    retry_plan_path: Path,
    retry_receipt_path: Path,
    authority_path: Path,
    scheduler_url: str,
) -> dict[str, Any]:
    """Bind exactly one Scheduler attempt to each of the 100 logical seeds."""

    base = collector.authenticate_context(
        plan_path=plan_path,
        receipt_path=base_receipt_path,
        scheduler_url=scheduler_url,
    )
    retry_plan, retry_receipt = _validate_retry_receipt(
        retry_plan_path=retry_plan_path,
        retry_receipt_path=retry_receipt_path,
    )
    if (
        retry_plan.get("base_plan_file_sha256")
        != collector._sha256_file(plan_path.resolve(strict=True))
        or retry_plan.get("base_receipt_file_sha256")
        != collector._sha256_file(base_receipt_path.resolve(strict=True))
        or retry_plan.get("base_bundle_id") != base["plan"]["bundle_id"]
        or retry_plan.get("base_plan_contract_sha256")
        != base["plan"]["contract_sha256"]
        or retry_plan.get("base_receipt_payload_sha256")
        != base["receipt"]["sha256"]
    ):
        raise RuntimeError("retry collector base authority drifted")

    base_by_seed = {
        int(entry["seed"]): copy.deepcopy(entry)
        for entry in base["entries"]
    }
    retry_plan_by_seed = {
        int(row["seed"]): row for row in retry_plan["tasks"]
    }
    retry_receipt_by_seed = {
        int(row["seed"]): row for row in retry_receipt["tasks"]
    }
    authorized_seeds = tuple(int(seed) for seed in base["authorized_seeds"])
    retry_count = len(retry_plan_by_seed)
    selected_entries = []
    selections = []
    ignored_attempt_ids = []
    for seed in authorized_seeds:
        base_entry = base_by_seed.get(seed)
        if base_entry is None:
            raise RuntimeError("logical exact100 base seed is missing")
        retry_plan_row = retry_plan_by_seed.get(seed)
        retry_receipt_row = retry_receipt_by_seed.get(seed)
        if (retry_plan_row is None) != (retry_receipt_row is None):
            raise RuntimeError("logical retry plan/receipt coverage differs")
        if retry_plan_row is None:
            selected = base_entry
            attempt_class = "first_clean_original"
            ignored = None
        else:
            original = retry_plan_row.get("original") or {}
            if (
                int(original.get("task_id", -1)) != int(base_entry["task_id"])
                or original.get("status") != "failed"
                or original.get("started_at") is not None
                or original.get("remote_paths_empty") is not True
                or original.get("account_name") != retry.FAILED_ACCOUNT
            ):
                raise RuntimeError(
                    "logical retry did not replace its exact failed attempt"
                )
            selected = {
                **base_entry,
                "task_id": int(retry_receipt_row["task_id"]),
                "dedupe_key": str(retry_receipt_row["dedupe_key"]),
                "scheduler_payload": copy.deepcopy(
                    retry_plan_row["scheduler_payload"]
                ),
                "attempt_class": "quota_prestart_retry1",
                "replaced_task_id": int(base_entry["task_id"]),
            }
            attempt_class = "quota_prestart_retry1"
            ignored = int(base_entry["task_id"])
            ignored_attempt_ids.append(ignored)
        selected_entries.append(selected)
        selections.append(
            {
                "seed": seed,
                "selected_task_id": int(selected["task_id"]),
                "selected_dedupe_key": selected["dedupe_key"],
                "attempt_class": attempt_class,
                "ignored_original_task_id": ignored,
            }
        )
    selected_ids = [int(entry["task_id"]) for entry in selected_entries]
    if (
        len(selected_entries) != offload.EXACT_TASK_COUNT
        or [int(entry["seed"]) for entry in selected_entries]
        != list(authorized_seeds)
        or len(set(selected_ids)) != offload.EXACT_TASK_COUNT
        or set(selected_ids).intersection(ignored_attempt_ids)
    ):
        raise RuntimeError("logical exact100 physical-attempt selection is invalid")
    authority_created_at = retry_receipt["observed_at"]
    if authority_path.is_file():
        existing_authority = collector._validate_sealed(
            collector._read_json(authority_path),
            schema=OVERLAY_SCHEMA,
        )
        authority_created_at = existing_authority["created_at"]
    authority = collector._sealed(
        {
            "schema_version": OVERLAY_SCHEMA,
            "created_at": authority_created_at,
            "base_plan": {
                "path": str(plan_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    plan_path.resolve(strict=True)
                ),
                "contract_sha256": base["plan"]["contract_sha256"],
            },
            "base_receipt": {
                "path": str(base_receipt_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    base_receipt_path.resolve(strict=True)
                ),
                "receipt_sha256": base["receipt"]["sha256"],
            },
            "retry_plan": {
                "path": str(retry_plan_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    retry_plan_path.resolve(strict=True)
                ),
                "payload_sha256": retry_plan["payload_sha256"],
            },
            "retry_receipt": {
                "path": str(retry_receipt_path.resolve(strict=True)),
                "sha256": collector._sha256_file(
                    retry_receipt_path.resolve(strict=True)
                ),
                "payload_sha256": retry_receipt["payload_sha256"],
            },
            "logical_seed_count": offload.EXACT_TASK_COUNT,
            "physical_submitted_attempt_count": (
                offload.EXACT_TASK_COUNT + retry_count
            ),
            "selected_physical_attempt_count": offload.EXACT_TASK_COUNT,
            "ignored_infrastructure_prestart_attempt_count": (
                retry_count
            ),
            "ignored_infrastructure_prestart_task_ids": ignored_attempt_ids,
            "selections": selections,
            "one_selected_physical_attempt_per_logical_seed": True,
            "duplicate_physical_result_collection_allowed": False,
            "retry_account_set": list(retry.ALLOWED_ACCOUNTS),
            "failed_account_excluded_from_retry": retry.FAILED_ACCOUNT,
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
    return {
        **base,
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
            "physical_submitted_attempt_count": (
                offload.EXACT_TASK_COUNT + retry_count
            ),
            "selected_retry_attempt_count": retry_count,
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
    retry_plan_path: Path,
    retry_receipt_path: Path,
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
        retry_plan_path=retry_plan_path,
        retry_receipt_path=retry_receipt_path,
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
    return status, (2 if require_final and not status["final_exact100_reached"] else 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--retry-plan", type=Path, required=True)
    parser.add_argument("--retry-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scheduler-url", default=collector.DEFAULT_SCHEDULER_URL
    )
    parser.add_argument("--accounts", type=Path, default=collector.DEFAULT_ACCOUNTS)
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
        retry_plan_path=args.retry_plan,
        retry_receipt_path=args.retry_receipt,
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
