"""GET-only collector for the submitted rounded candidate-5 Standard lane.

The collector authenticates the prepare plan and one-shot submission receipt,
checks the exact ``dw16/n113`` allocation lineage, and reconstructs the
retained ``symmetric.aedt`` plus AEDT results from Scheduler GET endpoints.
It never POSTs, retries, cancels, or otherwise mutates Scheduler state.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import mft_goal_postdeadline_standard_collector as base  # noqa: E402
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_prepare as prepare_only,
)
from tools import (  # noqa: E402
    mft_goal_official5_rounded_final_submit as submit_only,
)


CollectionError = base.CollectionError
TASK_NAME = prepare_only.TASK_NAME
PROJECT = prepare_only.PROJECT
ACCOUNT_NAME = prepare_only.ACCOUNT_NAME
NODE_NAME = prepare_only.NODE_NAME
SAME_NODE_AS_TASK_ID = prepare_only.SAME_NODE_AS_TASK_ID
ALLOCATION_ID = prepare_only.SOURCE_ALLOCATION_ID
SLURM_JOB_ID = prepare_only.SOURCE_SLURM_JOB_ID
CPUS = prepare_only.CPUS
MEMORY_MB = prepare_only.MEMORY_MB
SCHEDULER_SECONDS = prepare_only.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = prepare_only.MAX_WORKERS_PER_NODE
SCHEDULER_URL = prepare_only.SCHEDULER_URL

DEFAULT_OUTPUT = (
    prepare_only.OUTPUT_ROOT / "authenticated_get_collection"
)
HANDOFF_NAME = "drawing_view_export_handoff.json"
FAILURE_NAME = "terminal_failure.json"
HANDOFF_SCHEMA = "mft-goal-rounded-final-drawing-view-handoff-v1"
FAILURE_SCHEMA = "mft-goal-rounded-final-terminal-failure-v1"
ACTIVE_STATES = {"queued", "attaching", "running"}
TERMINAL_SUCCESS = {("completed", "succeeded")}
TERMINAL_FAILURE_STATES = {
    "failed",
    "cancelled",
    "timed_out",
    "timeout",
}


def _resolve_record(record: Any, label: str) -> Path:
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise CollectionError(f"{label} record is malformed")
    path = Path(record["path"]).resolve(strict=True)
    if (
        not path.is_file()
        or submit_only.direct.sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["size_bytes"]
    ):
        raise CollectionError(f"{label} bytes drifted")
    return path


def load_contract(
    *,
    plan_path: Path | None = None,
    final_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate local immutable evidence and derive GET paths."""

    plan_file = (
        plan_path or (prepare_only.OUTPUT_ROOT / prepare_only.PLAN_NAME)
    ).resolve(strict=True)
    plan = prepare_only.load_plan(plan_file)
    final_file = (
        final_path
        or (
            prepare_only.OUTPUT_ROOT
            / submit_only.SUBMISSION_DIRECTORY_NAME
            / submit_only.FINAL_NAME
        )
    ).resolve(strict=True)
    final = submit_only.validate_seal(
        submit_only.read_json(final_file), submit_only.FINAL_SCHEMA
    )
    receipt_path = _resolve_record(
        final.get("submission_receipt"), "rounded submission receipt"
    )
    receipt = submit_only.validate_seal(
        submit_only.read_json(receipt_path), submit_only.RECEIPT_SCHEMA
    )
    receipt_plan = _resolve_record(
        receipt.get("plan"), "rounded submission plan"
    )
    payload = plan.get("scheduler_payload")
    retained = plan.get("retained_aedt_bundle")
    if (
        receipt_plan != plan_file
        or final.get("task_id") != receipt.get("task_id")
        or receipt.get("task_name") != TASK_NAME
        or receipt.get("candidate_physics_sha256")
        != prepare_only.SOURCE_CANDIDATE_SHA256
        or not isinstance(payload, Mapping)
        or not isinstance(retained, Mapping)
        or payload.get("name") != TASK_NAME
        or payload.get("dedupe_key") != receipt.get("dedupe_key")
        or payload.get("account_name") != ACCOUNT_NAME
        or payload.get("node_name") != NODE_NAME
        or payload.get("same_node_as_task_id") != SAME_NODE_AS_TASK_ID
        or "requested_allocation_id" in payload
        or payload.get("timeout_seconds") != SCHEDULER_SECONDS
        or retained.get("dedupe_key") != payload.get("dedupe_key")
        or retained.get("artifact_path", "").split("/")[-1]
        != "symmetric.aedt"
        or retained.get("stage") != "standard"
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
    ):
        raise CollectionError(
            "rounded plan/submission/retention identity drifted"
        )
    transport = retained.get("transport")
    if not isinstance(transport, Mapping):
        raise CollectionError("rounded retained transport is absent")
    return {
        "plan": copy.deepcopy(plan),
        "submission": copy.deepcopy(receipt),
        "plan_path": plan_file,
        "submission_path": receipt_path,
        "plan_file_sha256": submit_only.direct.sha256_file(plan_file),
        "submission_file_sha256": submit_only.direct.sha256_file(
            receipt_path
        ),
        "task_id": int(receipt["task_id"]),
        "task_name": TASK_NAME,
        "dedupe_key": str(payload["dedupe_key"]),
        "candidate_physics_sha256": (
            prepare_only.SOURCE_CANDIDATE_SHA256
        ),
        "solver_revision": str(plan["solver_revision"]),
        "library_revision": str(plan["library_revision"]),
        "profile_sha256": str(retained["profile_sha256"]),
        "parameter_digest": str(retained["parameter_digest"]),
        "scheduler_url": SCHEDULER_URL.rstrip("/"),
        "node_name": NODE_NAME,
        "retained": {
            "root": str(retained["relative_directory"]),
            "artifact_path": str(retained["artifact_path"]),
            "receipt_path": str(retained["receipt_path"]),
            "marker_path": str(retained["marker_path"]),
            "chunk_directory": str(transport["chunk_directory"]),
            "results_path": str(retained["results_path"]),
            "results_manifest_path": str(
                retained["results_manifest_path"]
            ),
            "marker_contract": copy.deepcopy(
                retained["marker_contract"]
            ),
            "marker_contract_sha256": str(
                retained["marker_contract_sha256"]
            ),
        },
    }


def get_task(
    contract: Mapping[str, Any],
    *,
    reader: submit_only.JsonReader = submit_only.get_json,
) -> dict[str, Any]:
    """GET and authenticate the exact rounded Scheduler task."""

    task = reader(f"/api/tasks/{contract['task_id']}", None)
    expected = {
        "task_id": contract["task_id"],
        "name": TASK_NAME,
        "dedupe_key": contract["dedupe_key"],
        "project": PROJECT,
        "requested_account_name": ACCOUNT_NAME,
        "requested_node_name": NODE_NAME,
        "requested_node_name_policy": "strict",
        "preferred_node_relaxed": False,
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "requested_allocation_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": expected_value, "actual": task.get(key)}
        for key, expected_value in expected.items()
        if not isinstance(task, Mapping)
        or task.get(key) != expected_value
    }
    status = str(task.get("status") or "")
    if drift or status not in (
        ACTIVE_STATES
        | TERMINAL_FAILURE_STATES
        | {"completed", "succeeded"}
    ):
        raise CollectionError(
            f"rounded Scheduler task identity drifted: {drift}"
        )
    if status not in {"queued"}:
        placement = {
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "actual_node_name": NODE_NAME,
            "allocation_node_name": NODE_NAME,
            "allocation_id": ALLOCATION_ID,
            "assigned_allocation": ALLOCATION_ID,
            "slurm_job_id": SLURM_JOB_ID,
        }
        placement_drift = {
            key: {"expected": value, "actual": task.get(key)}
            for key, value in placement.items()
            if task.get(key) != value
        }
        if placement_drift:
            raise CollectionError(
                f"rounded task placement drifted: {placement_drift}"
            )
    return copy.deepcopy(dict(task))


def _write_local_seal(path: Path, value: Mapping[str, Any]) -> Path:
    payload = base.seal(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = base.canonical_bytes(payload) + b"\n"
    if path.exists():
        existing = base._read_json(path, f"existing {path.name}")  # noqa: SLF001
        base._validate_seal(  # noqa: SLF001
            existing,
            str(value.get("schema_version") or ""),
            f"existing {path.name}",
        )
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _drawing_handoff(
    *,
    contract: Mapping[str, Any],
    output: Path,
) -> Path:
    artifact = output.resolve(strict=True) / "symmetric.aedt"
    if not artifact.is_file():
        raise CollectionError("collected symmetric.aedt is absent")
    handoff = output.resolve().parent / HANDOFF_NAME
    return _write_local_seal(
        handoff,
        {
            "schema_version": HANDOFF_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_id": contract["task_id"],
            "task_name": TASK_NAME,
            "candidate_physics_sha256": (
                contract["candidate_physics_sha256"]
            ),
            "rounded_model": True,
            "source_model": base.file_record(artifact),
            "view_exporter": (
                "tools/mft_goal_export_rounded_drawing_views.py"
            ),
            "export_command_argv": [
                "conda",
                "run",
                "-n",
                "pyaedt2026v1",
                "python",
                "tools/mft_goal_export_rounded_drawing_views.py",
                "--project",
                str(artifact),
                "--output",
                str(output.resolve().parent / "drawing_views"),
                "--background",
                "white",
            ],
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        },
    )


def collect(
    *,
    contract: Mapping[str, Any],
    output: Path,
    reader: submit_only.JsonReader = submit_only.get_json,
    getter: Any = base.http_get,
) -> dict[str, Any]:
    """Poll once and collect on terminal success using GET endpoints only."""

    task = get_task(contract, reader=reader)
    status = str(task.get("status") or "")
    state = str(task.get("state") or "")
    if status in ACTIVE_STATES:
        return {
            "event": "rounded_task_active",
            "task_id": contract["task_id"],
            "status": status,
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    if status in TERMINAL_FAILURE_STATES:
        failure = _write_local_seal(
            output.resolve().parent / FAILURE_NAME,
            {
                "schema_version": FAILURE_SCHEMA,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": contract["task_id"],
                "task_name": TASK_NAME,
                "status": status,
                "state": state,
                "exit_code": task.get("exit_code"),
                "failure_message": task.get("failure_message"),
                "scheduler_task": task,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
            },
        )
        return {
            "event": "rounded_task_failed",
            "task_id": contract["task_id"],
            "failure_ledger": str(failure),
            "scheduler_post_calls": 0,
        }
    if (status, state) not in TERMINAL_SUCCESS:
        raise CollectionError(
            f"rounded terminal state is not successful: {status}/{state}"
        )
    result, _stdout = base.get_result(contract, getter=getter)
    expected_rounding = {
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
    }
    rounding_drift: dict[str, dict[str, Any]] = {}
    for key, expected in expected_rounding.items():
        try:
            matches = float(result.get(key)) == float(expected)
        except (TypeError, ValueError, OverflowError):
            matches = False
        if not matches:
            rounding_drift[key] = {
                "expected": expected,
                "actual": result.get(key),
            }
    if rounding_drift:
        raise CollectionError(
            f"rounded RESULT_JSON identity drifted: {rounding_drift}"
        )
    collected = base.collect_success(
        contract=contract,
        task=task,
        output=output,
        getter=getter,
    )
    handoff = _drawing_handoff(contract=contract, output=output)
    return {
        **collected,
        "drawing_view_export_handoff": str(handoff),
        "scheduler_get_only": True,
        "scheduler_post_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=prepare_only.OUTPUT_ROOT / prepare_only.PLAN_NAME,
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=(
            prepare_only.OUTPUT_ROOT
            / submit_only.SUBMISSION_DIRECTORY_NAME
            / submit_only.FINAL_NAME
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_contract(
        plan_path=args.plan, final_path=args.final
    )
    result = collect(contract=contract, output=args.output)
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CollectionError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_final_collector_error",
                    "error": str(exc),
                    "scheduler_get_only": True,
                    "scheduler_post_calls": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
