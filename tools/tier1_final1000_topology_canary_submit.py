"""Idempotently submit the sealed Final1000 topology pre-cutover canary.

The only mutating Scheduler operation in this tool is ``POST /api/tasks``.
Existing tasks are accepted only when the live compact and detail identities
match the sealed canary envelope exactly.  Every remote READY is re-read over
SSH immediately before Scheduler access.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tier1_final1000_slurm_controller import SchedulerApiClient
    from tier1_final1000_slurm_launch import (
        load_stage_bindings,
        validate_topology_pre_cutover_canary_plan,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tools.tier1_final1000_slurm_controller import SchedulerApiClient
    from tools.tier1_final1000_slurm_launch import (
        load_stage_bindings,
        validate_topology_pre_cutover_canary_plan,
    )


RECEIPT_SCHEMA = "mft-tier1-final1000-topology-canary-submission-v1"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable receipt already exists: {path}")
    staged = path.with_name(path.name + f".part.{os.getpid()}")
    try:
        staged.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _validate_existing_receipt(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    tasks = value.get("tasks")
    if (
        value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("sha256") != canonical_sha256(unsigned)
        or value.get("canary_plan_sha256") != plan["sha256"]
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or not isinstance(tasks, list)
        or len(tasks) != len(plan["tasks"])
        or {item.get("dedupe_key") for item in tasks}
        != {item["dedupe_key"] for item in plan["tasks"]}
    ):
        raise RuntimeError("existing topology canary receipt identity/SHA mismatch")
    return dict(value)


def _authenticate_task(
    row: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, str]:
    observation = scheduler_task_observation(row or {})
    if (
        observation is None
        or not scheduler_task_identity_matches(row or {}, expected)
    ):
        raise RuntimeError(f"{label} changed the sealed Scheduler task identity")
    return observation


def _live_ready_inventory(
    bindings: Mapping[str, Mapping[str, Any]], transport: Any
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage_id, binding in bindings.items():
        publication = binding["publication"]
        remote_bundle = str(binding["plan"]["remote_bundle"])
        try:
            live = json.loads(
                transport.read_bytes(remote_bundle.rstrip("/") + "/READY.json").decode(
                    "utf-8"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{stage_id} live READY is unavailable") from exc
        if (
            not isinstance(live, dict)
            or live != publication["ready"]
            or canonical_sha256(live) != publication["ready_sha256"]
        ):
            raise RuntimeError(f"{stage_id} live READY changed after publication")
        result[stage_id] = {
            "bundle_id": binding["plan"]["bundle_id"],
            "bundle_manifest_sha256": binding["plan"][
                "bundle_manifest_sha256"
            ],
            "remote_bundle": remote_bundle,
            "ready_sha256": publication["ready_sha256"],
            "publication_receipt_sha256": publication["receipt_sha256"],
        }
    return result


def run(
    *,
    canary_plan_path: Path,
    bindings_path: Path,
    scheduler_url: str,
    accounts_path: Path,
    scheduler_source: Path,
    publication_account: str,
    apply: bool,
    receipt_out: Path | None,
) -> dict[str, Any]:
    plan = validate_topology_pre_cutover_canary_plan(_read_json(canary_plan_path))
    bindings = load_stage_bindings(bindings_path)
    expected_by_dedupe = {task["dedupe_key"]: task for task in plan["tasks"]}
    if len(expected_by_dedupe) != 16:
        raise RuntimeError("topology canary dedupe identity is not unique")

    with scheduler_publication_transport(
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        account_name=publication_account,
    ) as transport:
        ready = _live_ready_inventory(bindings, transport)

    scheduler = SchedulerApiClient(scheduler_url)
    inventory = {
        str(row.get("dedupe_key") or ""): row
        for row in scheduler.list_complete_namespace_tasks()
    }

    existing_receipt = None
    if receipt_out is not None and receipt_out.is_file():
        existing_receipt = _validate_existing_receipt(
            _read_json(receipt_out), plan=plan
        )

    rows: list[dict[str, Any]] = []
    submitted_count = 0
    existing_count = 0
    for dedupe, expected in expected_by_dedupe.items():
        row = inventory.get(dedupe)
        source = "existing"
        if row is None:
            if existing_receipt is not None:
                raise RuntimeError("receipt task disappeared from Scheduler inventory")
            if not apply:
                rows.append(
                    {
                        "dedupe_key": dedupe,
                        "name": expected["name"],
                        "state": "absent",
                        "task_id": None,
                    }
                )
                continue
            row = scheduler.submit_task(expected)
            submitted_count += 1
            source = "submitted"
        else:
            existing_count += 1
        task_id, status = _authenticate_task(row, expected, label=source)
        detail = scheduler.get_task(task_id)
        detail_id, detail_status = _authenticate_task(
            detail, expected, label="Scheduler task detail"
        )
        if detail_id != task_id:
            raise RuntimeError("Scheduler list/detail task id mismatch")
        rows.append(
            {
                "dedupe_key": dedupe,
                "name": expected["name"],
                "source": source,
                "state": detail_status,
                "task_id": task_id,
            }
        )

    if existing_receipt is not None:
        sealed_ids = {
            item["dedupe_key"]: item["task_id"]
            for item in existing_receipt["tasks"]
        }
        observed_ids = {item["dedupe_key"]: item["task_id"] for item in rows}
        if sealed_ids != observed_ids:
            raise RuntimeError("existing receipt/Scheduler task id mapping changed")
        return existing_receipt

    unsigned = {
        "schema_version": RECEIPT_SCHEMA,
        "observed_at": _now(),
        "apply": apply,
        "canary_plan_sha256": plan["sha256"],
        "science_plan_sha256": plan["science_plan_sha256"],
        "logical_seed_count": len(plan["tasks"]),
        "live_ready": ready,
        "tasks": rows,
        "submitted_count": submitted_count,
        "existing_count": existing_count,
        "absent_count": sum(item["state"] == "absent" for item in rows),
        "scheduler_endpoint": "POST /api/tasks",
        "scheduler_post_count": scheduler.post_count,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "scheduler_inventory_snapshot": scheduler.inventory_snapshot_receipt,
        "aedt_used": False,
        "fea_submission_performed": False,
    }
    value = {**unsigned, "sha256": canonical_sha256(unsigned)}
    if apply:
        if receipt_out is None:
            raise RuntimeError("--apply requires --receipt-out")
        if value["absent_count"] != 0 or scheduler.post_count != submitted_count:
            raise RuntimeError("topology canary submission accounting mismatch")
        _atomic_json(receipt_out, value)
    elif scheduler.post_count != 0:
        raise RuntimeError("dry-run performed a Scheduler POST")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--publication-account", default="harry261")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    value = run(
        canary_plan_path=args.canary_plan,
        bindings_path=args.bindings,
        scheduler_url=args.scheduler_url,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        publication_account=args.publication_account,
        apply=args.apply,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
