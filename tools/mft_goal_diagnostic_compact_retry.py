"""Retry pre-start compact-scout failures on explicitly healthy accounts.

This adapter is deliberately isolated from the immutable exact100 first-run
receipt.  It can only retry exact100 tasks that the live Scheduler proves
failed before launch on the quota-full ``harry261`` account.  Retry identity,
account placement, warning suppression, and the equal-three-leg physical-gap
FEA obligation are all committed by a new dedupe key.

No Scheduler cancellation or preemption method exists in this module.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping
import urllib.parse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402


PLAN_SCHEMA = "mft-goal-diagnostic-compact-prestart-retry-plan-v1"
DRY_RUN_SCHEMA = "mft-goal-diagnostic-compact-prestart-retry-dry-run-v1"
RECEIPT_SCHEMA = "mft-goal-diagnostic-compact-prestart-retry-receipt-v1"
STATE_SCHEMA = "mft-goal-diagnostic-compact-prestart-retry-state-v1"
RETRY_ATTEMPT = 1
FAILED_ACCOUNT = "harry261"
ALLOWED_ACCOUNTS = (
    "dhj02",
    "jji0930",
    "r1jae262",
    "dw16",
    "wjddn5916",
)
TERMINAL = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)
NONTERMINAL = frozenset({"queued", "attaching", "running"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} seal mismatch")
    return result


class SchedulerClient:
    """Minimal GET/POST client; cancellation APIs are intentionally absent."""

    def __init__(
        self,
        base_url: str = offload.DEFAULT_SCHEDULER_URL,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.get_count = 0
        self.post_count = 0

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        self.get_count += method == "GET"
        self.post_count += method == "POST"
        return transport._api_json(
            self.base_url + path,
            method=method,
            payload=None if payload is None else dict(payload),
            timeout=self.timeout,
        )

    def get_task(self, task_id: int) -> dict[str, Any]:
        value = self._request(f"/api/tasks/{int(task_id)}")
        if not isinstance(value, dict):
            raise RuntimeError("Scheduler task detail is invalid")
        return value

    def list_tasks(self, name_prefix: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {
                    "name_prefix": name_prefix,
                    "sort_by": "id",
                    "sort_order": "asc",
                    "paged": "true",
                    "page": page,
                    "page_size": 10_000,
                }
            )
            value = self._request(f"/api/tasks?{query}")
            if not isinstance(value, Mapping) or not isinstance(
                value.get("items"), list
            ):
                raise RuntimeError("Scheduler retry inventory is invalid")
            result.extend(copy.deepcopy(value["items"]))
            if value.get("has_next") is not True:
                break
            page += 1
        return result

    def submit_task(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = self._request("/api/tasks", method="POST", payload=payload)
        if not isinstance(value, dict):
            raise RuntimeError("Scheduler retry POST response is invalid")
        return value


def _task_id(row: Mapping[str, Any]) -> int:
    task_id = row.get("task_id", row.get("id"))
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise RuntimeError("Scheduler task id is invalid")
    alias = row.get("id", task_id)
    if alias != task_id:
        raise RuntimeError("Scheduler task id aliases disagree")
    return task_id


def _status(row: Mapping[str, Any]) -> str:
    status = row.get("status")
    if not isinstance(status, str) or status not in TERMINAL | NONTERMINAL:
        raise RuntimeError("Scheduler task status is invalid")
    return status


def _prestart_quota_failure(
    detail: Mapping[str, Any],
    *,
    expected_task_id: int,
) -> dict[str, Any]:
    if (
        _task_id(detail) != expected_task_id
        or _status(detail) != "failed"
        or detail.get("account_name") != FAILED_ACCOUNT
        or detail.get("started_at") is not None
        or str(detail.get("remote_dir") or "")
        or str(detail.get("stdout_path") or "")
        or str(detail.get("stderr_path") or "")
        or str(detail.get("exit_code_path") or "")
        or detail.get("exit_code") is not None
        or detail.get("failure_message") != "Failure"
    ):
        raise RuntimeError(
            f"task {expected_task_id} is not the authorized pre-start quota failure"
        )
    return {
        "task_id": expected_task_id,
        "status": "failed",
        "account_name": FAILED_ACCOUNT,
        "allocation_id": detail.get("allocation_id"),
        "slurm_job_id": str(detail.get("slurm_job_id") or ""),
        "attached_at": detail.get("attached_at"),
        "launch_started_at": detail.get("launch_started_at"),
        "finished_at": detail.get("finished_at"),
        "started_at": None,
        "failure_message": "Failure",
        "remote_paths_empty": True,
    }


def _retry_prefix(base_plan: Mapping[str, Any]) -> str:
    return f"{offload._task_name_prefix(base_plan)}retry{RETRY_ATTEMPT}-"


def _retry_payload(
    *,
    base_plan: Mapping[str, Any],
    task: Mapping[str, Any],
    original_task_id: int,
    requested_account: str,
) -> dict[str, Any]:
    if requested_account not in ALLOWED_ACCOUNTS:
        raise RuntimeError("retry account is not authorized")
    base = offload.scheduler_payload(
        plan=base_plan,
        task=task,
        priority=None,
    )
    command = str(base["command"])
    marker = "exec python "
    if command.count(marker) != 1:
        raise RuntimeError("base worker command cannot be warning-capped")
    command = command.replace(marker, "exec python -W ignore ", 1)
    seed = int(task["seed"])
    name = (
        f"{_retry_prefix(base_plan)}s{seed}-n1-6-"
        f"a{requested_account}"
    )
    identity = {
        "schema_version": (
            "mft-goal-diagnostic-compact-prestart-retry-dedupe-v1"
        ),
        "retry_attempt": RETRY_ATTEMPT,
        "original_task_id": int(original_task_id),
        "original_dedupe_key": base["dedupe_key"],
        "seed": seed,
        "requested_account": requested_account,
        "warning_policy": "python_-W_ignore",
        "equal_three_leg_air_gap_FEA_required": True,
        "final_promotion_allowed": False,
        "immutable_retry_envelope": {
            **{key: value for key, value in base.items() if key != "dedupe_key"},
            "name": name,
            "command": command,
            "account_name": requested_account,
        },
    }
    return {
        **base,
        "name": name,
        "command": command,
        "account_name": requested_account,
        "dedupe_key": offload.DEDUPE_PREFIX + canonical_sha256(identity),
    }


def _base_context(
    *,
    plan_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan, _deployment, tasks, authentication = offload.authenticate_plan(
        plan_path
    )
    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            offload.scheduler_payload(
                plan=plan,
                task=task,
                priority=None,
            )
            for task in tasks
        )
    }
    raw_receipt = _read_json(receipt_path)
    receipt = offload._validate_receipt(
        raw_receipt,
        plan=plan,
        expected=expected,
        authentication=authentication,
        ready=raw_receipt.get("remote_ready") or {},
        scheduler_url=str(
            raw_receipt.get("scheduler_url") or offload.DEFAULT_SCHEDULER_URL
        ),
    )
    return plan, tasks, receipt


def prepare(
    *,
    plan_path: Path,
    receipt_path: Path,
    output_path: Path,
    scheduler: SchedulerClient | Any | None = None,
    scheduler_url: str = offload.DEFAULT_SCHEDULER_URL,
) -> dict[str, Any]:
    """Seal the exact live failed-seed retry set without Scheduler mutation."""

    base_plan, tasks, receipt = _base_context(
        plan_path=plan_path,
        receipt_path=receipt_path,
    )
    client = scheduler or SchedulerClient(scheduler_url)
    task_by_seed = {int(task["seed"]): task for task in tasks}
    receipt_by_seed = {
        int(row["seed"]): row for row in receipt["tasks"]
    }
    failures: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for seed in sorted(task_by_seed):
        receipt_row = receipt_by_seed.get(seed)
        if receipt_row is None:
            raise RuntimeError("base receipt seed mapping is incomplete")
        detail = client.get_task(int(receipt_row["task_id"]))
        status = _status(detail)
        if status == "failed":
            failure = _prestart_quota_failure(
                detail,
                expected_task_id=int(receipt_row["task_id"]),
            )
            failures.append((seed, task_by_seed[seed], failure))
    if not failures:
        raise RuntimeError("no pre-start quota failures require retry")
    rows = []
    for ordinal, (seed, task, failure) in enumerate(failures):
        account = ALLOWED_ACCOUNTS[ordinal % len(ALLOWED_ACCOUNTS)]
        payload = _retry_payload(
            base_plan=base_plan,
            task=task,
            original_task_id=failure["task_id"],
            requested_account=account,
        )
        rows.append(
            {
                "seed": seed,
                "original": failure,
                "requested_account": account,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": canonical_sha256(payload),
            }
        )
    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "created_at": _now(),
        "base_plan_path": str(plan_path.resolve(strict=True)),
        "base_plan_file_sha256": _sha256_file(plan_path.resolve(strict=True)),
        "base_receipt_path": str(receipt_path.resolve(strict=True)),
        "base_receipt_file_sha256": _sha256_file(
            receipt_path.resolve(strict=True)
        ),
        "retry_tool_source_sha256": _sha256_file(Path(__file__).resolve()),
        "base_bundle_id": base_plan["bundle_id"],
        "base_plan_contract_sha256": base_plan["contract_sha256"],
        "base_diagnostic_plan_sha256": base_plan["diagnostic_plan_sha256"],
        "base_receipt_payload_sha256": receipt["sha256"],
        "retry_attempt": RETRY_ATTEMPT,
        "retry_reason": "harry261_GPFS_hard_quota_prestart_task_sh_write_failure",
        "failed_account": FAILED_ACCOUNT,
        "allowed_accounts": list(ALLOWED_ACCOUNTS),
        "account_assignment": "sorted_seed_round_robin",
        "retry_task_count": len(rows),
        "warning_policy": "python_-W_ignore",
        "tasks": rows,
        "target_active_inventory": 100,
        "no_running_task_cancelled": True,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "equal_three_leg_air_gap_FEA_required": True,
        "equal_three_leg_air_gap_definition": (
            "identical_physical_air_gap_on_center_and_both_side_legs"
        ),
        "physical_Lm_2mH_symmetric_FEA_verification_required": True,
        "automatic_final_promotion_allowed": False,
        "final_promotion_allowed": False,
        "screening_only": True,
        "production_eligible": False,
        "scheduler_post_count": 0,
        "scheduler_get_count": int(client.get_count),
    }
    plan = _seal(unsigned)
    _atomic_json(output_path, plan)
    return plan


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_seal(value, PLAN_SCHEMA)
    rows = plan.get("tasks")
    if (
        plan.get("retry_attempt") != RETRY_ATTEMPT
        or plan.get("failed_account") != FAILED_ACCOUNT
        or plan.get("allowed_accounts") != list(ALLOWED_ACCOUNTS)
        or not isinstance(rows, list)
        or len(rows) != plan.get("retry_task_count")
        or len(rows) < 1
        or plan.get("warning_policy") != "python_-W_ignore"
        or plan.get("equal_three_leg_air_gap_FEA_required") is not True
        or plan.get("automatic_final_promotion_allowed") is not False
        or plan.get("final_promotion_allowed") is not False
        or plan.get("screening_only") is not True
        or plan.get("production_eligible") is not False
        or plan.get("retry_tool_source_sha256")
        != _sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("retry plan contract mismatch")
    seeds = []
    dedupes = set()
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RuntimeError("retry plan task row is invalid")
        seed = row.get("seed")
        payload = row.get("scheduler_payload")
        account = ALLOWED_ACCOUNTS[index % len(ALLOWED_ACCOUNTS)]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(payload, Mapping)
            or row.get("requested_account") != account
            or payload.get("account_name") != account
            or payload.get("name") in names
            or payload.get("dedupe_key") in dedupes
            or row.get("scheduler_payload_sha256")
            != canonical_sha256(payload)
            or "exec python -W ignore " not in str(payload.get("command") or "")
        ):
            raise RuntimeError("retry plan task identity mismatch")
        seeds.append(seed)
        names.add(payload["name"])
        dedupes.add(payload["dedupe_key"])
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        raise RuntimeError("retry plan seeds are not unique and ordered")
    return plan


def _inventory(
    client: SchedulerClient | Any,
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prefix = str(plan["tasks"][0]["scheduler_payload"]["name"]).split(
        "s", 1
    )[0]
    expected = {
        row["scheduler_payload"]["dedupe_key"]: row["scheduler_payload"]
        for row in plan["tasks"]
    }
    result = {}
    for row in client.list_tasks(prefix):
        dedupe = str(row.get("dedupe_key") or "")
        if dedupe not in expected:
            raise RuntimeError("foreign task entered the retry namespace")
        if dedupe in result:
            raise RuntimeError("duplicate retry dedupe identity")
        wanted = expected[dedupe]
        if (
            row.get("name") != wanted["name"]
            or row.get("requested_account_name") != wanted["account_name"]
        ):
            raise RuntimeError("live retry task placement identity drifted")
        _task_id(row)
        _status(row)
        result[dedupe] = dict(row)
    return result


def submit(
    *,
    plan_path: Path,
    output_dir: Path,
    apply: bool = False,
    scheduler: SchedulerClient | Any | None = None,
    scheduler_url: str = offload.DEFAULT_SCHEDULER_URL,
) -> dict[str, Any]:
    """Dry-run or idempotently submit every missing retry task."""

    plan = _validate_plan(_read_json(plan_path))
    client = scheduler or SchedulerClient(scheduler_url)
    inventory = _inventory(client, plan)
    expected = {
        row["scheduler_payload"]["dedupe_key"]: row
        for row in plan["tasks"]
    }
    rows = []
    submitted = 0
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dedupe, entry in expected.items():
        payload = entry["scheduler_payload"]
        observed = inventory.get(dedupe)
        source = "existing"
        if observed is None and apply:
            response = client.submit_task(payload)
            task_id = response.get("task_id", response.get("id"))
            if isinstance(task_id, bool) or not isinstance(task_id, int):
                raise RuntimeError("retry POST lacks a task id")
            observed = client.get_task(task_id)
            if (
                observed.get("name") != payload["name"]
                or observed.get("dedupe_key") != payload["dedupe_key"]
                or observed.get("requested_account_name")
                != payload["account_name"]
            ):
                raise RuntimeError("retry POST readback identity mismatch")
            submitted += 1
            source = "submitted"
        if observed is None:
            rows.append(
                {
                    "seed": entry["seed"],
                    "requested_account": entry["requested_account"],
                    "dedupe_key": dedupe,
                    "name": payload["name"],
                    "task_id": None,
                    "status": "absent",
                    "source": "absent",
                }
            )
        else:
            rows.append(
                {
                    "seed": entry["seed"],
                    "requested_account": entry["requested_account"],
                    "dedupe_key": dedupe,
                    "name": payload["name"],
                    "task_id": _task_id(observed),
                    "status": _status(observed),
                    "source": source,
                }
            )
        state = _seal(
            {
                "schema_version": STATE_SCHEMA,
                "updated_at": _now(),
                "retry_plan_payload_sha256": plan["payload_sha256"],
                "apply": bool(apply),
                "mapped_count": len(rows),
                "submitted_this_run": submitted,
                "tasks": copy.deepcopy(rows),
                "scheduler_post_count": int(client.post_count),
                "scheduler_get_count": int(client.get_count),
                "scheduler_cancel_count": 0,
                "scheduler_preempt_count": 0,
                "equal_three_leg_air_gap_FEA_required": True,
                "final_promotion_allowed": False,
            }
        )
        _atomic_json(output_dir / "submission_state.json", state)
    schema = RECEIPT_SCHEMA if apply else DRY_RUN_SCHEMA
    result = _seal(
        {
            "schema_version": schema,
            "observed_at": _now(),
            "apply": bool(apply),
            "retry_plan_path": str(plan_path.resolve(strict=True)),
            "retry_plan_file_sha256": _sha256_file(
                plan_path.resolve(strict=True)
            ),
            "retry_plan_payload_sha256": plan["payload_sha256"],
            "retry_task_count": len(rows),
            "tasks": rows,
            "submitted_count": submitted,
            "existing_count": sum(row["source"] == "existing" for row in rows),
            "absent_count": sum(row["status"] == "absent" for row in rows),
            "scheduler_post_count": int(client.post_count),
            "scheduler_get_count": int(client.get_count),
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "allowed_accounts": list(ALLOWED_ACCOUNTS),
            "failed_account_excluded": FAILED_ACCOUNT,
            "warning_policy": "python_-W_ignore",
            "target_active_inventory": 100,
            "equal_three_leg_air_gap_FEA_required": True,
            "physical_Lm_2mH_symmetric_FEA_verification_required": True,
            "automatic_final_promotion_allowed": False,
            "final_promotion_allowed": False,
            "screening_only": True,
            "production_eligible": False,
        }
    )
    filename = "submission_receipt.json" if apply else "submission_dry_run.json"
    _atomic_json(output_dir / filename, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--plan", type=Path, required=True)
    prepare_parser.add_argument("--receipt", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--scheduler-url", default=offload.DEFAULT_SCHEDULER_URL
    )
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--output-dir", type=Path, required=True)
    submit_parser.add_argument(
        "--scheduler-url", default=offload.DEFAULT_SCHEDULER_URL
    )
    submit_parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare(
            plan_path=args.plan,
            receipt_path=args.receipt,
            output_path=args.output,
            scheduler_url=args.scheduler_url,
        )
    else:
        value = submit(
            plan_path=args.plan,
            output_dir=args.output_dir,
            apply=args.apply,
            scheduler_url=args.scheduler_url,
        )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
