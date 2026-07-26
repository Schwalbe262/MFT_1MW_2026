"""Fail-closed exact-once submitter for the prepared rounded Full lane.

Importing this module and all commands except ``submit`` are mutation-free.
``submit`` requires the literal authorization token, reauthenticates task96340
success, requires the sealed license snapshot to remain fresh, repeats the
strict FEA-empty node gate, writes an immutable consumed-attempt ledger, and
then permits at most one Scheduler POST.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_official5_rounded_full_prepare as prepare_only  # noqa: E402


ContractError = prepare_only.ContractError
JsonReader = prepare_only.continuation.JsonReader
PostOnce = Callable[[str, Mapping[str, Any]], tuple[int, Any]]

POST_AUTHORIZATION = "authorize-rounded-full-one-post-v1"
SUBMISSION_DIRECTORY = "submission"
ATTEMPT_NAME = "scheduler_post_attempt.json"
RECEIPT_NAME = "scheduler_submission_receipt.json"
ATTEMPT_SCHEMA = "mft-goal-official5-rounded-full-submit-attempt-v1"
RECEIPT_SCHEMA = "mft-goal-official5-rounded-full-submit-receipt-v1"


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    try:
        return prepare_only.rounded.write_exclusive_json(path, value)
    except AttributeError:
        return prepare_only.rounded.direct.write_exclusive_json(path, value)


def _authority(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    record = plan.get("source_standard_success_authority")
    path = prepare_only.rounded._contained(  # noqa: SLF001
        plan_path.resolve().parent,
        record,
        "rounded Standard success authority",
    )
    return prepare_only.validate_seal(
        prepare_only.read_json(path), prepare_only.SOURCE_AUTHORITY_SCHEMA
    )


def _reauthenticate_source(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    authority = _authority(plan_path, plan)
    receipt_path = Path(
        authority["source_collection_receipt"]["path"]
    ).resolve(strict=True)
    refreshed = prepare_only.authenticate_standard_success(
        standard_plan_path=Path(
            authority["source_plan"]["path"]
        ).resolve(strict=True),
        standard_final_path=Path(
            authority["source_submission_final"]["path"]
        ).resolve(strict=True),
        standard_collection_root=receipt_path.parent,
    )
    ignored = {"authenticated_at_utc", "payload_sha256"}
    if {
        key: value
        for key, value in authority.items()
        if key not in ignored
    } != {
        key: value
        for key, value in refreshed.items()
        if key not in ignored
    }:
        raise ContractError(
            "rounded Standard success authority changed before Full POST"
        )
    return refreshed


def _license_path(plan: Mapping[str, Any]) -> Path:
    freshness = plan.get("fresh_license_snapshot")
    record = freshness.get("record") if isinstance(freshness, Mapping) else None
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
    ):
        raise ContractError("sealed Full license snapshot is absent")
    path = Path(record["path"]).resolve(strict=True)
    if prepare_only._file_record(path) != dict(record):  # noqa: SLF001
        raise ContractError("sealed Full license snapshot bytes drifted")
    return path


def submit(
    *,
    plan_path: Path,
    authorization: str,
    reader: JsonReader = prepare_only.continuation.get_json,
    post_once: PostOnce = prepare_only.continuation._post_json_once,  # noqa: SLF001
    lock_factory: Callable[..., Any] = (
        scheduler_client.campaign_mutation_lock
    ),
    observed_at: datetime | None = None,
) -> Path:
    """Consume the one Full attempt and POST only after every gate passes."""

    plan_path = plan_path.resolve(strict=True)
    plan = prepare_only.load_plan(plan_path)
    if authorization != POST_AUTHORIZATION:
        raise ContractError("rounded Full POST authorization is invalid")
    root = plan_path.parent / SUBMISSION_DIRECTORY
    attempt_path = root / ATTEMPT_NAME
    receipt_path = root / RECEIPT_NAME
    if receipt_path.exists():
        prepare_only.validate_seal(
            prepare_only.read_json(receipt_path), RECEIPT_SCHEMA
        )
        return receipt_path
    if attempt_path.exists():
        raise ContractError(
            "rounded Full POST attempt is already consumed; no retry"
        )
    with lock_factory():
        if receipt_path.exists():
            prepare_only.validate_seal(
                prepare_only.read_json(receipt_path), RECEIPT_SCHEMA
            )
            return receipt_path
        if attempt_path.exists():
            raise ContractError(
                "rounded Full POST attempt was consumed inside lock"
            )
        source = _reauthenticate_source(plan_path, plan)
        license_path = _license_path(plan)
        license_freshness = prepare_only.validate_license_freshness(
            license_path, observed_at=observed_at
        )
        lane = prepare_only.StrictLane(
            str(plan["target_lane"]["account_name"]),
            str(plan["target_lane"]["node_name"]),
        )
        live = prepare_only.continuation._lane_preflight(  # noqa: SLF001
            lane=lane,
            source_node=str(source["source_node_name"]),
            task_name=prepare_only.TASK_NAME,
            dedupe_key=str(plan["scheduler_payload"]["dedupe_key"]),
            source_task_id=prepare_only.SOURCE_STANDARD_TASK_ID,
            reader=reader,
        )
        attempt = prepare_only.sealed(
            {
                "schema_version": ATTEMPT_SCHEMA,
                "created_at_utc": (
                    observed_at or datetime.now(timezone.utc)
                ).astimezone(timezone.utc).isoformat(),
                "plan": prepare_only._file_record(plan_path),  # noqa: SLF001
                "plan_payload_sha256": plan["payload_sha256"],
                "source_standard_task_id": (
                    prepare_only.SOURCE_STANDARD_TASK_ID
                ),
                "source_candidate_physics_sha256": (
                    prepare_only.SOURCE_CANDIDATE_SHA256
                ),
                "task_name": prepare_only.TASK_NAME,
                "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
                "locked_pre_submit_live_preflight": live,
                "fresh_license_snapshot": license_freshness,
                "post_attempt_consumed": True,
                "maximum_scheduler_posts": 1,
                "scheduler_post_outcome_known": False,
                "scheduler_repository_modified": False,
                "mft_and_scheduler_functionality_mixed": False,
            }
        )
        _write_exclusive(attempt_path, attempt)
        status, response = post_once(
            f"{prepare_only.SCHEDULER_URL}/api/tasks",
            plan["scheduler_payload"],
        )
        if status not in {200, 201} or not isinstance(response, Mapping):
            raise ContractError(
                f"rounded Full Scheduler POST outcome unknown: HTTP {status}"
            )
        task_id = response.get("task_id", response.get("id"))
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
        ):
            raise ContractError(
                "rounded Full Scheduler POST returned no task ID"
            )
        task = reader(f"/api/tasks/{task_id}", None)
        expected = {
            "task_id": task_id,
            "name": prepare_only.TASK_NAME,
            "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
            "project": prepare_only.PROJECT,
            "requested_account_name": lane.account_name,
            "requested_node_name": lane.node_name,
            "requested_node_name_policy": "strict",
            "preferred_node_relaxed": False,
            "same_node_as_task_id": 0,
            "requested_allocation_id": 0,
            "cpus": prepare_only.CPUS,
            "memory_mb": prepare_only.MEMORY_MB,
            "timeout_seconds": prepare_only.SCHEDULER_TIMEOUT_SECONDS,
            "max_workers_per_node": (
                prepare_only.MAX_WORKERS_PER_NODE
            ),
            "aedt_backend": "standalone",
        }
        drift = {
            key: {"expected": value, "actual": task.get(key)}
            for key, value in expected.items()
            if not isinstance(task, Mapping) or task.get(key) != value
        }
        if drift:
            raise ContractError(
                f"rounded Full task readback drifted: {drift}"
            )
        receipt = prepare_only.sealed(
            {
                "schema_version": RECEIPT_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "rounded_full_submitted_pending_actual_result",
                "plan": prepare_only._file_record(plan_path),  # noqa: SLF001
                "plan_payload_sha256": plan["payload_sha256"],
                "attempt_ledger": prepare_only._file_record(  # noqa: SLF001
                    attempt_path
                ),
                "source_standard_task_id": (
                    prepare_only.SOURCE_STANDARD_TASK_ID
                ),
                "source_candidate_physics_sha256": (
                    prepare_only.SOURCE_CANDIDATE_SHA256
                ),
                "full_task_id": task_id,
                "task_name": prepare_only.TASK_NAME,
                "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
                "target_lane": copy.deepcopy(plan["target_lane"]),
                "task_readback": copy.deepcopy(dict(task)),
                "task_readback_sha256": prepare_only.payload_sha256(task),
                "scheduler_post_attempts_consumed": 1,
                "maximum_scheduler_posts": 1,
                "scheduler_mutation_performed": status == 201,
                "scheduler_repository_modified": False,
                "mft_and_scheduler_functionality_mixed": False,
            }
        )
        _write_exclusive(receipt_path, receipt)
        return receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=prepare_only.OUTPUT_ROOT / prepare_only.PLAN_NAME,
    )
    parser.add_argument("--authorize-post", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = submit(
        plan_path=args.plan,
        authorization=args.authorize_post,
    )
    value = prepare_only.validate_seal(
        prepare_only.read_json(path), RECEIPT_SCHEMA
    )
    print(
        json.dumps(
            {
                "event": "rounded_full_submitted",
                "receipt": str(path),
                "full_task_id": value["full_task_id"],
                "scheduler_post_attempts_consumed": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_full_submit_error",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
