"""Manifest-bound GET-only collector for the official #8 n111 failover."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import mft_goal_official_standard8_collector as base  # noqa: E402
from tools import mft_goal_official_standard8_n111_failover as failover  # noqa: E402


class FailoverCollectionError(base.Official8CollectionError):
    """Manifest or failover evidence drifted."""


def _manifest(path: Path) -> dict[str, Any]:
    value = failover.validate_seal(
        failover.read_json(path.resolve(strict=True)),
        failover.COLLECTOR_MANIFEST_SCHEMA,
    )
    if (
        any(
            value.get(key) is not expected
            for key, expected in failover.SAFETY_FLAGS.items()
        )
        or value.get("candidate_physics_sha256")
        != failover.CANDIDATE_SHA256
        or value.get("task_name") != failover.TASK_NAME
        or value.get("account_name") != failover.ACCOUNT_NAME
        or value.get("node_name") != failover.NODE_NAME
        or value.get("node_name_policy") != "strict"
        or value.get("preferred_node_relaxed") is not False
        or value.get("scheduler_get_only_collector") is not True
        or value.get("collector_scheduler_mutation_allowed") is not False
        or value.get("scheduler_post_call_budget") != 1
        or value.get("scheduler_post_calls_evidenced") != 1
        or value.get("retry_allowed") is not False
        or value.get("original_task_id")
        != failover.ORIGINAL_OFFICIAL8_TASK_ID
        or value.get("original_task_cancelled_or_modified") is not False
    ):
        raise FailoverCollectionError(
            "n111 failover collector manifest identity drifted"
        )
    for key in (
        "plan",
        "submission_intent",
        "attempt_ledger",
        "submission_receipt",
        "final_seal",
        "collector_intent",
    ):
        record = value.get(key)
        if not isinstance(record, Mapping):
            raise FailoverCollectionError(
                f"collector manifest {key} record is absent"
            )
        path_value = Path(str(record.get("path") or "")).resolve(strict=True)
        if failover.sha256_file(path_value) != record.get("sha256"):
            raise FailoverCollectionError(
                f"collector manifest {key} bytes drifted"
            )
    return value


@contextmanager
def _collector_contract(manifest: Mapping[str, Any]) -> Iterator[None]:
    plan = manifest["plan"]
    submission = manifest["submission_receipt"]
    intent = manifest["submission_intent"]
    attempt = manifest["attempt_ledger"]
    final = manifest["final_seal"]
    replacements = {
        "official": failover,
        "TASK_ID": int(manifest["task_id"]),
        "TASK_NAME": failover.TASK_NAME,
        "DEDUPE_KEY": str(manifest["dedupe_key"]),
        "CANDIDATE_PHYSICS_SHA256": failover.CANDIDATE_SHA256,
        "PLAN_FILE_SHA256": str(plan["sha256"]),
        "PLAN_PAYLOAD_SHA256": str(manifest["plan_payload_sha256"]),
        "INTENT_FILE_SHA256": str(intent["sha256"]),
        "INTENT_PAYLOAD_SHA256": str(
            manifest["submission_intent_payload_sha256"]
        ),
        "ATTEMPT_FILE_SHA256": str(attempt["sha256"]),
        "ATTEMPT_PAYLOAD_SHA256": str(
            manifest["attempt_ledger_payload_sha256"]
        ),
        "SUBMISSION_FILE_SHA256": str(submission["sha256"]),
        "SUBMISSION_PAYLOAD_SHA256": str(
            manifest["submission_receipt_payload_sha256"]
        ),
        "FINAL_FILE_SHA256": str(final["sha256"]),
        "FINAL_PAYLOAD_SHA256": str(
            manifest["final_seal_payload_sha256"]
        ),
        "EXPECTED_NODE": failover.NODE_NAME,
        "EXPECTED_ACCOUNT": failover.ACCOUNT_NAME,
        "RUNTIME_ROOT": failover.OUTPUT_ROOT,
    }
    old = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 10 <= args.interval <= 3_600:
        raise FailoverCollectionError(
            "watch interval must be between 10 and 3600 seconds"
        )
    manifest = _manifest(args.manifest)
    output = args.output or Path(str(manifest["collector_output"]))
    plan_path = Path(str(manifest["plan"]["path"]))
    submission_path = Path(str(manifest["submission_receipt"]["path"]))
    with _collector_contract(manifest):
        contract = base.load_contract(
            plan_path=plan_path,
            submission_path=submission_path,
            plan_loader=failover.load_plan,
        )
        sequence = 0
        while True:
            terminal, event = base.poll_once(
                contract=contract,
                output=output,
                sequence=sequence,
            )
            event["collector_manifest"] = str(
                args.manifest.resolve(strict=True)
            )
            print(json.dumps(event, sort_keys=True), flush=True)
            if terminal or args.once:
                return 0
            sequence += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FailoverCollectionError, base.base.CollectionError) as exc:
        print(
            json.dumps(
                {
                    "event": "official8_n111_failover_collector_error",
                    "error": str(exc),
                    "scheduler_get_only": True,
                    "scheduler_mutation_performed": False,
                    "scientific_pass_claimed": False,
                    "scientific_infeasible_claimed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from exc
