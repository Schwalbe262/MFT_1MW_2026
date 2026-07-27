"""Retry the exact five compact runtime exit-1 failures as retry2.

This path is isolated from both the immutable original exact100 receipt and
the retry1 pre-start-quota receipt.  It authenticates all known attempts by
GET, preserves completed logical seeds, and permits one new attempt only for
the exact original harry261 tasks that started, produced remote paths, and
terminated with exit code 1.

Scheduler mutation is limited to idempotent ``POST /api/tasks``.  There are
no cancel or preempt methods.
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_compact_retry as retry1  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402


PLAN_SCHEMA = "mft-goal-diagnostic-compact-runtime-retry2-plan-v1"
DRY_RUN_SCHEMA = "mft-goal-diagnostic-compact-runtime-retry2-dry-run-v1"
RECEIPT_SCHEMA = "mft-goal-diagnostic-compact-runtime-retry2-receipt-v1"
RETRY_ATTEMPT = 2
FAILED_ACCOUNT = "harry261"
EXPECTED_SEEDS = (2_607_264_113, 2_607_264_121, 2_607_264_129,
                  2_607_264_137, 2_607_264_145)
ALLOWED_ACCOUNTS = retry1.ALLOWED_ACCOUNTS
ACTIVE = retry1.NONTERMINAL
TERMINAL = retry1.TERMINAL
TARGET_ACTIVE = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"runtime-retry JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("runtime-retry JSON must be an object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha(resolved)}


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("runtime-retry value already sealed")
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


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value, stream, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _immutable(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _read_json(target) != dict(value):
            raise RuntimeError("immutable runtime-retry artifact differs")
        return
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value, stream, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            if _read_json(target) != dict(value):
                raise RuntimeError("concurrent runtime-retry artifact differs")
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _task_id(row: Mapping[str, Any]) -> int:
    return retry1._task_id(row)


def _status(row: Mapping[str, Any]) -> str:
    return retry1._status(row)


def _runtime_exit1_failure(
    detail: Mapping[str, Any], *, expected_task_id: int
) -> dict[str, Any]:
    failure = str(detail.get("failure_message") or "")
    paths = {
        name: str(detail.get(name) or "")
        for name in (
            "remote_dir", "stdout_path", "stderr_path", "exit_code_path"
        )
    }
    if (
        _task_id(detail) != expected_task_id
        or _status(detail) != "failed"
        or detail.get("account_name") != FAILED_ACCOUNT
        or not isinstance(detail.get("started_at"), str)
        or not detail["started_at"]
        or not isinstance(detail.get("finished_at"), str)
        or not detail["finished_at"]
        or detail.get("exit_code") != 1
        or any(not value for value in paths.values())
        or "Exited with exit code 1" not in failure
    ):
        raise RuntimeError(
            f"task {expected_task_id} is not an authorized runtime exit1 failure"
        )
    return {
        "task_id": expected_task_id,
        "status": "failed",
        "account_name": FAILED_ACCOUNT,
        "allocation_id": detail.get("allocation_id"),
        "slurm_job_id": str(detail.get("slurm_job_id") or ""),
        "started_at": detail["started_at"],
        "finished_at": detail["finished_at"],
        "exit_code": 1,
        **paths,
        "failure_message_sha256": hashlib.sha256(
            failure.encode("utf-8")
        ).hexdigest(),
        "runtime_paths_nonempty": True,
    }


def _load_retry1_receipt(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = retry1._validate_seal(
        retry1._read_json(path), retry1.RECEIPT_SCHEMA
    )
    plan_path = Path(str(receipt.get("retry_plan_path") or ""))
    plan = retry1._validate_plan(retry1._read_json(plan_path))
    expected = {
        row["scheduler_payload"]["dedupe_key"]: row
        for row in plan["tasks"]
    }
    observed: set[str] = set()
    for row in receipt.get("tasks") or []:
        source = expected.get(str(row.get("dedupe_key") or ""))
        if (
            source is None
            or row["dedupe_key"] in observed
            or row.get("seed") != source["seed"]
            or row.get("name") != source["scheduler_payload"]["name"]
            or row.get("requested_account") != source["requested_account"]
            or not isinstance(row.get("task_id"), int)
            or row.get("status") not in ACTIVE | TERMINAL
        ):
            raise RuntimeError("retry1 receipt task identity mismatch")
        observed.add(row["dedupe_key"])
    if (
        receipt.get("apply") is not True
        or receipt.get("absent_count") != 0
        or observed != set(expected)
        or receipt.get("retry_plan_file_sha256") != _sha(plan_path)
        or receipt.get("retry_plan_payload_sha256")
        != plan["payload_sha256"]
        or receipt.get("allowed_accounts") != list(ALLOWED_ACCOUNTS)
    ):
        raise RuntimeError("retry1 receipt authority mismatch")
    return plan, receipt


def _retry2_prefix(base_plan: Mapping[str, Any]) -> str:
    return f"{offload._task_name_prefix(base_plan)}retry2-"


def _retry2_payload(
    *,
    base_plan: Mapping[str, Any],
    task: Mapping[str, Any],
    original_task_id: int,
    failure_proof_sha256: str,
    requested_account: str,
) -> dict[str, Any]:
    if requested_account not in ALLOWED_ACCOUNTS:
        raise RuntimeError("runtime-retry account is unauthorized")
    base = offload.scheduler_payload(plan=base_plan, task=task)
    command = str(base["command"])
    if command.count("exec python ") != 1:
        raise RuntimeError("base worker command cannot be warning-capped")
    command = command.replace("exec python ", "exec python -W ignore ", 1)
    seed = int(task["seed"])
    name = (
        f"{_retry2_prefix(base_plan)}s{seed}-n1-6-"
        f"a{requested_account}"
    )
    envelope = {
        **{key: copy.deepcopy(value) for key, value in base.items()
           if key != "dedupe_key"},
        "name": name,
        "command": command,
        "account_name": requested_account,
    }
    dedupe = canonical_sha256(
        {
            "schema_version": (
                "mft-goal-diagnostic-compact-runtime-retry2-dedupe-v1"
            ),
            "retry_attempt": RETRY_ATTEMPT,
            "original_task_id": int(original_task_id),
            "original_dedupe_key": base["dedupe_key"],
            "failure_proof_sha256": failure_proof_sha256,
            "seed": seed,
            "requested_account": requested_account,
            "warning_policy": "python_-W_ignore",
            "equal_three_leg_air_gap_FEA_required": True,
            "final_promotion_allowed": False,
            "immutable_retry_envelope": envelope,
        }
    )
    return {**envelope, "dedupe_key": offload.DEDUPE_PREFIX + dedupe}


def _live_attempt(
    client: Any,
    *,
    binding: Mapping[str, Any],
    requested_account: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    detail = client.get_task(int(binding["task_id"]))
    status = _status(detail)
    if (
        detail.get("name") != binding["name"]
        or detail.get("dedupe_key") != binding["dedupe_key"]
        or (
            requested_account is not None
            and detail.get("requested_account_name") != requested_account
        )
    ):
        raise RuntimeError("live attempt identity drifted")
    compact = {
        "attempt": int(binding["attempt"]),
        "task_id": int(binding["task_id"]),
        "name": binding["name"],
        "dedupe_key": binding["dedupe_key"],
        "requested_account": requested_account,
        "status": status,
    }
    return compact, detail


def prepare(
    *,
    base_plan_path: Path,
    base_receipt_path: Path,
    retry1_receipt_path: Path,
    output_path: Path,
    scheduler: Any | None = None,
    scheduler_url: str = offload.DEFAULT_SCHEDULER_URL,
) -> dict[str, Any]:
    base_plan, tasks, base_receipt = retry1._base_context(
        plan_path=base_plan_path, receipt_path=base_receipt_path
    )
    retry1_plan, retry1_receipt = _load_retry1_receipt(
        retry1_receipt_path
    )
    if (
        retry1_plan["base_bundle_id"] != base_plan["bundle_id"]
        or retry1_plan["base_plan_contract_sha256"]
        != base_plan["contract_sha256"]
    ):
        raise RuntimeError("retry1 does not bind the exact base plan")
    client = scheduler or retry1.SchedulerClient(scheduler_url)
    task_by_seed = {int(task["seed"]): task for task in tasks}
    base_by_seed = {
        int(row["seed"]): {
            **row, "attempt": 0,
        }
        for row in base_receipt["tasks"]
    }
    retry1_by_seed = {
        int(row["seed"]): {
            **row, "attempt": 1,
        }
        for row in retry1_receipt["tasks"]
    }
    overlay = []
    candidates = []
    active_before = 0
    for seed in sorted(task_by_seed):
        attempts = []
        base_attempt, base_detail = _live_attempt(
            client, binding=base_by_seed[seed]
        )
        attempts.append(base_attempt)
        if seed in retry1_by_seed:
            row = retry1_by_seed[seed]
            attempt, _detail = _live_attempt(
                client,
                binding=row,
                requested_account=row["requested_account"],
            )
            attempts.append(attempt)
        active = [row for row in attempts if row["status"] in ACTIVE]
        completed = [
            row for row in attempts if row["status"] == "completed"
        ]
        if len(active) > 1 or len(completed) > 1 or (active and completed):
            raise RuntimeError("logical seed has multiple selected attempts")
        active_before += len(active)
        selected = completed[0] if completed else (active[0] if active else None)
        if (
            seed in EXPECTED_SEEDS
            and selected is None
            and seed not in retry1_by_seed
        ):
            proof = _runtime_exit1_failure(
                base_detail, expected_task_id=base_by_seed[seed]["task_id"]
            )
            proof_sha = canonical_sha256(proof)
            account = ALLOWED_ACCOUNTS[len(candidates) % len(ALLOWED_ACCOUNTS)]
            payload = _retry2_payload(
                base_plan=base_plan,
                task=task_by_seed[seed],
                original_task_id=proof["task_id"],
                failure_proof_sha256=proof_sha,
                requested_account=account,
            )
            planned = {
                "attempt": RETRY_ATTEMPT,
                "task_id": None,
                "name": payload["name"],
                "dedupe_key": payload["dedupe_key"],
                "requested_account": account,
                "status": "planned",
            }
            attempts.append(planned)
            candidates.append(
                {
                    "seed": seed,
                    "original_failure": proof,
                    "original_failure_proof_sha256": proof_sha,
                    "requested_account": account,
                    "scheduler_payload": payload,
                    "scheduler_payload_sha256": canonical_sha256(payload),
                }
            )
        overlay.append(
            {
                "logical_seed": seed,
                "attempts": attempts,
                "selected_attempt": (
                    None if selected is None else selected["attempt"]
                ),
                "selection_policy": (
                    "completed_then_single_active_then_none"
                ),
            }
        )
    if tuple(row["seed"] for row in candidates) != EXPECTED_SEEDS:
        raise RuntimeError("runtime-retry2 exact five failure set mismatch")
    projected = active_before + len(candidates)
    if projected > TARGET_ACTIVE:
        raise RuntimeError("runtime-retry2 would exceed active100")
    value = _seal(
        {
            "schema_version": PLAN_SCHEMA,
            "created_at": _now(),
            "base_plan": _record(base_plan_path),
            "base_receipt": _record(base_receipt_path),
            "retry1_plan": _record(
                Path(retry1_receipt["retry_plan_path"])
            ),
            "retry1_receipt": _record(retry1_receipt_path),
            "base_bundle_id": base_plan["bundle_id"],
            "base_plan_contract_sha256": base_plan["contract_sha256"],
            "retry1_plan_payload_sha256": retry1_plan["payload_sha256"],
            "retry1_receipt_payload_sha256": retry1_receipt[
                "payload_sha256"
            ],
            "retry_attempt": RETRY_ATTEMPT,
            "retry_name_prefix": _retry2_prefix(base_plan),
            "failed_account": FAILED_ACCOUNT,
            "allowed_accounts": list(ALLOWED_ACCOUNTS),
            "expected_runtime_failure_seeds": list(EXPECTED_SEEDS),
            "tasks": candidates,
            "retry_task_count": len(candidates),
            "multi_attempt_overlay": overlay,
            "logical_seed_count": 100,
            "known_attempt_count": 152,
            "active_before": active_before,
            "projected_active_after_retry2": projected,
            "reserved_continuation_refill_count": TARGET_ACTIVE - projected,
            "target_active_count": TARGET_ACTIVE,
            "one_selected_attempt_per_logical_seed": True,
            "completed_attempts_preserved": True,
            "warning_policy": "python_-W_ignore",
            "equal_three_leg_air_gap_FEA_required": True,
            "final_promotion_allowed": False,
            "screening_only": True,
            "production_eligible": False,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "scheduler_post_count": 0,
            "scheduler_get_count": int(client.get_count),
            "runtime_retry_tool_source_sha256": _sha(Path(__file__)),
        }
    )
    _immutable(output_path, value)
    return value


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_seal(value, PLAN_SCHEMA)
    rows = plan.get("tasks")
    if (
        plan.get("retry_attempt") != RETRY_ATTEMPT
        or plan.get("failed_account") != FAILED_ACCOUNT
        or plan.get("allowed_accounts") != list(ALLOWED_ACCOUNTS)
        or plan.get("expected_runtime_failure_seeds") != list(EXPECTED_SEEDS)
        or not isinstance(rows, list)
        or len(rows) != len(EXPECTED_SEEDS)
        or plan.get("retry_task_count") != len(EXPECTED_SEEDS)
        or plan.get("one_selected_attempt_per_logical_seed") is not True
        or plan.get("completed_attempts_preserved") is not True
        or plan.get("equal_three_leg_air_gap_FEA_required") is not True
        or plan.get("final_promotion_allowed") is not False
        or plan.get("runtime_retry_tool_source_sha256")
        != _sha(Path(__file__))
    ):
        raise RuntimeError("runtime-retry2 plan contract mismatch")
    prefix = plan.get("retry_name_prefix")
    if (
        not isinstance(prefix, str)
        or not prefix.startswith(offload.TASK_NAME_PREFIX)
        or not prefix.endswith("-retry2-")
    ):
        raise RuntimeError("runtime-retry2 task prefix mismatch")
    for index, row in enumerate(rows):
        payload = row.get("scheduler_payload") or {}
        if (
            row.get("seed") != EXPECTED_SEEDS[index]
            or row.get("requested_account")
            != ALLOWED_ACCOUNTS[index % len(ALLOWED_ACCOUNTS)]
            or payload.get("account_name") != row["requested_account"]
            or row.get("scheduler_payload_sha256")
            != canonical_sha256(payload)
            or not str(payload.get("name") or "").startswith(prefix)
            or "exec python -W ignore " not in str(payload.get("command") or "")
        ):
            raise RuntimeError("runtime-retry2 task identity mismatch")
    return plan


def _inventory(client: Any, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    prefix = str(plan["retry_name_prefix"])
    expected = {
        row["scheduler_payload"]["dedupe_key"]: row
        for row in plan["tasks"]
    }
    result = {}
    for row in client.list_tasks(prefix):
        dedupe = str(row.get("dedupe_key") or "")
        source = expected.get(dedupe)
        if (
            source is None
            or dedupe in result
            or row.get("name") != source["scheduler_payload"]["name"]
            or row.get("requested_account_name")
            != source["requested_account"]
        ):
            raise RuntimeError("runtime-retry2 inventory mismatch")
        _task_id(row)
        _status(row)
        result[dedupe] = dict(row)
    return result


def submit(
    *,
    plan_path: Path,
    output_dir: Path,
    apply: bool = False,
    scheduler: Any | None = None,
    scheduler_url: str = offload.DEFAULT_SCHEDULER_URL,
) -> dict[str, Any]:
    plan = _validate_plan(_read_json(plan_path))
    client = scheduler or retry1.SchedulerClient(scheduler_url)
    inventory = _inventory(client, plan)
    rows = []
    submitted = 0
    for entry in plan["tasks"]:
        payload = entry["scheduler_payload"]
        dedupe = payload["dedupe_key"]
        original = client.get_task(entry["original_failure"]["task_id"])
        proof = _runtime_exit1_failure(
            original,
            expected_task_id=entry["original_failure"]["task_id"],
        )
        if proof != entry["original_failure"]:
            raise RuntimeError("runtime failure proof changed")
        observed = inventory.get(dedupe)
        source = "existing"
        if observed is None and apply:
            response = client.submit_task(payload)
            task_id = response.get("task_id", response.get("id"))
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
            ):
                raise RuntimeError("runtime-retry2 POST lacks task id")
            observed = client.get_task(task_id)
            submitted += 1
            source = "submitted"
        if observed is None:
            rows.append(
                {
                    "seed": entry["seed"],
                    "requested_account": entry["requested_account"],
                    "name": payload["name"],
                    "dedupe_key": dedupe,
                    "task_id": None,
                    "status": "absent",
                    "source": "absent",
                }
            )
            continue
        if (
            observed.get("name") != payload["name"]
            or observed.get("dedupe_key") != dedupe
            or observed.get("requested_account_name")
            != entry["requested_account"]
        ):
            raise RuntimeError("runtime-retry2 readback identity mismatch")
        rows.append(
            {
                "seed": entry["seed"],
                "requested_account": entry["requested_account"],
                "name": payload["name"],
                "dedupe_key": dedupe,
                "task_id": _task_id(observed),
                "status": _status(observed),
                "source": source,
            }
        )
    schema = RECEIPT_SCHEMA if apply else DRY_RUN_SCHEMA
    result = _seal(
        {
            "schema_version": schema,
            "observed_at": _now(),
            "apply": bool(apply),
            "runtime_retry2_plan": _record(plan_path),
            "runtime_retry2_plan_payload_sha256": plan["payload_sha256"],
            "retry_attempt": RETRY_ATTEMPT,
            "retry_task_count": len(rows),
            "tasks": rows,
            "submitted_count": submitted,
            "existing_count": sum(row["source"] == "existing" for row in rows),
            "absent_count": sum(row["status"] == "absent" for row in rows),
            "projected_active_after_retry2": plan[
                "projected_active_after_retry2"
            ],
            "reserved_continuation_refill_count": plan[
                "reserved_continuation_refill_count"
            ],
            "multi_attempt_overlay_authority_sha256": canonical_sha256(
                plan["multi_attempt_overlay"]
            ),
            "one_selected_attempt_per_logical_seed": True,
            "scheduler_post_count": int(client.post_count),
            "scheduler_get_count": int(client.get_count),
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "allowed_accounts": list(ALLOWED_ACCOUNTS),
            "warning_policy": "python_-W_ignore",
            "equal_three_leg_air_gap_FEA_required": True,
            "final_promotion_allowed": False,
            "screening_only": True,
            "production_eligible": False,
        }
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "submission_receipt.json" if apply else "submission_dry_run.json"
    )
    if apply:
        if result["absent_count"] != 0:
            raise RuntimeError("runtime-retry2 apply left absent tasks")
        _immutable(output, result)
    else:
        _atomic(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--base-plan", type=Path, required=True)
    prepare_parser.add_argument("--base-receipt", type=Path, required=True)
    prepare_parser.add_argument("--retry1-receipt", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument(
        "--scheduler-url", default=offload.DEFAULT_SCHEDULER_URL
    )
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--output-dir", type=Path, required=True)
    submit_parser.add_argument("--apply", action="store_true")
    submit_parser.add_argument(
        "--scheduler-url", default=offload.DEFAULT_SCHEDULER_URL
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare(
            base_plan_path=args.base_plan,
            base_receipt_path=args.base_receipt,
            retry1_receipt_path=args.retry1_receipt,
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
