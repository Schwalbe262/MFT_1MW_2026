"""GET-only watcher and collector for the provisional project-only Full AEDT.

This module is deliberately outside the canonical truth/promotion path.  It
authenticates one ``mft_goal_provisional_full_precompute`` plan and submission,
watches the exact Scheduler task with GET requests, and, only after strict
terminal success, reconstructs the retained ``full.aedt`` on local storage.

No Scheduler POST, cancellation, remote write, remote delete, claim mutation,
or truth promotion is implemented here.  Every terminal output remains
diagnostic/provisional and explicitly production-ineligible.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_campaign_atomic_claim as atomic_claim  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_provisional_full_precompute as provisional  # noqa: E402


WATCH_PLAN_SCHEMA = "mft-goal-provisional-full-collector-plan-v1"
SUCCESS_SCHEMA = "mft-goal-provisional-full-collection-receipt-v1"
FAILURE_SCHEMA = "mft-goal-provisional-full-collection-failure-v1"
STATE_SCHEMA = "mft-goal-provisional-full-collection-state-v1"
PID_SCHEMA = "mft-goal-provisional-full-collector-pid-v1"
CAMPAIGN_ID = provisional.CAMPAIGN_ID
DEFAULT_SCHEDULER_URL = provisional.SCHEDULER_URL
MIN_POLL_SECONDS = 10
MAX_POLL_SECONDS = 60
DEFAULT_POLL_SECONDS = 15
MAX_CONSECUTIVE_ERRORS = 3
REMOTE_CHUNK_READ_ATTEMPTS = 3
ACTIVE_STATUSES = frozenset(
    {"queued", "pending", "submitted", "attaching", "starting", "running"}
)
FAILED_STATUSES = frozenset({"failed", "cancelled", "canceled", "timed_out", "timeout"})
CAP_RECEIPT_FIELDS = frozenset(
    {
        "source_size_hard_cap_bytes",
        "source_size_hard_cap_contract",
        "source_size_hard_cap_enforced_before_destination_create",
    }
)
SOURCE_SIZE_HARD_CAP_CONTRACT = "pre-gpfs-destination-create-v1"

HandoffContractError = production.HandoffContractError


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).astimezone(timezone.utc).isoformat(timespec="microseconds")


def _flags() -> dict[str, bool]:
    return {
        "provisional": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
    }


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    return production._seal(copy.deepcopy(dict(value)))


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    return production._write_immutable_json(path.resolve(), dict(value))


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(production._json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def _aware(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffContractError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise HandoffContractError(f"{label} timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _assert_local_output(path: Path) -> Path:
    target = path.resolve()
    posix = target.as_posix()
    if (
        posix == "/gpfs"
        or posix.startswith("/gpfs/")
        or posix == "/enroot"
        or posix.startswith("/enroot/")
    ):
        raise HandoffContractError(
            "provisional Full collector output must be local, not GPFS/enroot"
        )
    return target


def _validate_embedded_admission(
    value: Any,
    *,
    plan: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    admission = production._validate_seal(value, provisional.GATE_SCHEMA)
    account, node = provisional._selected_placement(plan)
    if (
        admission.get("plan_payload_sha256") != plan["payload_sha256"]
        or admission.get("selected_account_name") != account
        or admission.get("requested_node_name") != node
        or admission.get("all_scheduler_reads_get_only") is not True
        or admission.get("fresh_capacity_passed") is not True
        or admission.get("fresh_license_passed") is not True
        or admission.get("fresh_storage_passed") is not True
        or admission.get("scheduler_repository_modified") is not False
        or admission.get("scheduler_submission_performed") is not False
        or any(
            admission.get(name) is not expected for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError(f"{label} admission identity drifted")
    production._require_sha(
        admission.get("license_snapshot_sha256"),
        f"{label} license snapshot SHA",
    )
    return admission


def _load_submission(
    path: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    submission = production._validate_seal(
        production._read_json(resolved),
        provisional.SUBMISSION_SCHEMA,
    )
    stage = plan["stage"]
    account, node = provisional._selected_placement(plan)
    task_id = _positive_int(
        submission.get("task_id"), "provisional Full submission task ID"
    )
    finalized = production._validate_seal(
        submission.get("atomic_claim_finalized"),
        atomic_claim.FINALIZED_CLAIM_SCHEMA,
    )
    initial = _validate_embedded_admission(
        submission.get("initial_fresh_admission"),
        plan=plan,
        label="initial",
    )
    locked = _validate_embedded_admission(
        submission.get("locked_fresh_admission"),
        plan=plan,
        label="locked",
    )
    retained = submission.get("retained_aedt")
    if (
        submission.get("plan") != production._file_record(plan_path)
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
        or submission.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or submission.get("logical_authority_task_id")
        != plan["logical_authority_task_id"]
        or submission.get("source_actual_standard_task_id")
        != plan["source_actual_standard_task_id"]
        or submission.get("selected_account_name") != account
        or submission.get("requested_node_name") != node
        or submission.get("global_exact_once_scope") != plan["global_exact_once_scope"]
        or submission.get("task_name") != stage["task_name"]
        or submission.get("dedupe_key") != stage["retained_aedt"]["dedupe_key"]
        or submission.get("resources") != provisional.RESOURCES
        or submission.get("full_model") != 1
        or submission.get("thermal_symmetry") != "full"
        or submission.get("profile") != plan["profile"]
        or retained != stage["retained_aedt"]
        or submission.get("retained_aedt_max_bytes")
        != plan["retention_storage_bound"]["expected_full_aedt_bytes"]
        or submission.get("retained_aedt_source_size_hard_cap_contract")
        != SOURCE_SIZE_HARD_CAP_CONTRACT
        or submission.get("retention_storage_bound") != plan["retention_storage_bound"]
        or submission.get("exact_sibling_count") != 1
        or finalized.get("task_id") != task_id
        or submission.get("maximum_scheduler_posts_lifetime") != 1
        or submission.get("scheduler_submission_performed") is not True
        or submission.get("scheduler_cancel_performed") is not False
        or submission.get("scheduler_repository_modified") is not False
        or submission.get("canonical_truth_gate_modified") is not False
        or submission.get("canonical_claim_modified") is not False
        or any(
            submission.get(name) is not expected for name, expected in _flags().items()
        )
    ):
        raise HandoffContractError("provisional Full submission identity drifted")
    if initial["payload_sha256"] == locked["payload_sha256"]:
        # Two independently captured fresh gates must not be represented by
        # one copied object.
        raise HandoffContractError(
            "provisional Full locked admission was not independently captured"
        )
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != scheduler_client.RETAINED_AEDT_SCHEMA
        or retained.get("stage") != "full"
        or PurePosixPath(str(retained.get("artifact_path") or "")).name != "full.aedt"
        or retained.get("source_size_hard_cap_bytes")
        != plan["retention_storage_bound"]["expected_full_aedt_bytes"]
        or retained.get("source_size_hard_cap_contract")
        != SOURCE_SIZE_HARD_CAP_CONTRACT
        or "results_path" in retained
        or "results_manifest_path" in retained
    ):
        raise HandoffContractError("provisional Full submission is not project-only")
    return submission


def _load_inputs(
    plan_path: Path,
    submission_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, params, profile = provisional.load_plan(plan_path)
    submission = _load_submission(
        submission_path,
        plan_path=plan_path,
        plan=plan,
    )
    return plan, params, profile, submission


def initialize_watch_plan(
    *,
    provisional_plan_path: Path,
    submission_path: Path,
    output_root: Path,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> Path:
    """Create an immutable GET-only collection authority."""
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not MIN_POLL_SECONDS <= poll_seconds <= MAX_POLL_SECONDS
    ):
        raise HandoffContractError("provisional Full poll interval is invalid")
    plan_path = provisional_plan_path.resolve(strict=True)
    receipt_path = submission_path.resolve(strict=True)
    plan, _params, _profile, submission = _load_inputs(plan_path, receipt_path)
    root = _assert_local_output(output_root)
    source_root = Path(plan["output_root"]).resolve()
    if root == source_root:
        raise HandoffContractError(
            "collector output must be separate from provisional submission root"
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "watch_plan.json"
    local_aedt = root / "full.aedt"
    success = root / "provisional_full_collection_receipt.json"
    failure = root / "failure_ledger.json"
    expected = {
        "schema_version": WATCH_PLAN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "provisional_plan": production._file_record(plan_path),
        "provisional_plan_payload_sha256": plan["payload_sha256"],
        "submission": production._file_record(receipt_path),
        "submission_payload_sha256": submission["payload_sha256"],
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "task_id": submission["task_id"],
        "task_name": submission["task_name"],
        "dedupe_key": submission["dedupe_key"],
        "selected_account_name": submission["selected_account_name"],
        "requested_node_name": submission["requested_node_name"],
        "output_root": str(root),
        "local_full_aedt_path": str(local_aedt),
        "success_receipt_path": str(success),
        "failure_ledger_path": str(failure),
        "sealed_source_size_hard_cap_bytes": plan["retention_storage_bound"][
            "expected_full_aedt_bytes"
        ],
        "scheduler_url": plan["scheduler_url"],
        "scheduler_project": plan["scheduler_project"],
        "terminal_deadline_utc": _aware(
            plan["target_finish_kst"], "target finish"
        ).isoformat(timespec="microseconds"),
        "poll_seconds": poll_seconds,
        "scheduler_methods_allowed": ["GET"],
        "scheduler_mutation_performed": False,
        "remote_artifact_mutation_performed": False,
        "gpfs_write_performed": False,
        "gpfs_delete_performed": False,
        "canonical_truth_gate_modified": False,
        "canonical_claim_modified": False,
        "canonical_promotion_modified": False,
        **_flags(),
    }
    if path.exists():
        existing = _load_watch_plan(path)[0]
        comparable = dict(existing)
        comparable.pop("created_at_utc")
        comparable.pop("payload_sha256")
        if comparable != expected:
            raise HandoffContractError(
                "existing provisional Full collector plan differs"
            )
        return path
    if local_aedt.exists() or success.exists() or failure.exists():
        raise HandoffContractError(
            "collector output pre-exists before immutable watch authority"
        )
    value = _sealed({**expected, "created_at_utc": _stamp()})
    return _write_immutable(path, value)


def _load_watch_plan(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = path.resolve(strict=True)
    watch = production._validate_seal(
        production._read_json(resolved), WATCH_PLAN_SCHEMA
    )
    root = Path(str(watch.get("output_root") or "")).resolve()
    if (
        resolved != root / "watch_plan.json"
        or watch.get("campaign_id") != CAMPAIGN_ID
        or Path(str(watch.get("local_full_aedt_path") or "")).resolve()
        != root / "full.aedt"
        or Path(str(watch.get("success_receipt_path") or "")).resolve()
        != root / "provisional_full_collection_receipt.json"
        or Path(str(watch.get("failure_ledger_path") or "")).resolve()
        != root / "failure_ledger.json"
        or watch.get("scheduler_url") != DEFAULT_SCHEDULER_URL
        or watch.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or watch.get("scheduler_methods_allowed") != ["GET"]
        or watch.get("scheduler_mutation_performed") is not False
        or watch.get("remote_artifact_mutation_performed") is not False
        or watch.get("gpfs_write_performed") is not False
        or watch.get("gpfs_delete_performed") is not False
        or watch.get("canonical_truth_gate_modified") is not False
        or watch.get("canonical_claim_modified") is not False
        or watch.get("canonical_promotion_modified") is not False
        or any(watch.get(name) is not expected for name, expected in _flags().items())
        or not MIN_POLL_SECONDS
        <= _positive_int(watch.get("poll_seconds"), "poll seconds")
        <= MAX_POLL_SECONDS
    ):
        raise HandoffContractError("provisional Full collector watch plan drifted")
    _assert_local_output(root)
    _aware(watch.get("created_at_utc"), "collector plan creation")
    deadline = _aware(watch.get("terminal_deadline_utc"), "terminal deadline")
    plan_record = watch.get("provisional_plan")
    submission_record = watch.get("submission")
    if not isinstance(plan_record, Mapping) or not isinstance(
        submission_record, Mapping
    ):
        raise HandoffContractError("collector authority records are absent")
    plan_path = Path(str(plan_record.get("path") or ""))
    submission_path = Path(str(submission_record.get("path") or ""))
    if (
        production._file_record(plan_path) != plan_record
        or production._file_record(submission_path) != submission_record
    ):
        raise HandoffContractError("collector input bytes drifted")
    plan, params, profile, submission = _load_inputs(plan_path, submission_path)
    if (
        watch.get("provisional_plan_payload_sha256") != plan["payload_sha256"]
        or watch.get("submission_payload_sha256") != submission["payload_sha256"]
        or watch.get("candidate_physics_sha256") != plan["candidate_physics_sha256"]
        or watch.get("task_id") != submission["task_id"]
        or watch.get("task_name") != submission["task_name"]
        or watch.get("dedupe_key") != submission["dedupe_key"]
        or watch.get("selected_account_name") != submission["selected_account_name"]
        or watch.get("requested_node_name") != submission["requested_node_name"]
        or watch.get("sealed_source_size_hard_cap_bytes")
        != plan["retention_storage_bound"]["expected_full_aedt_bytes"]
        or deadline != _aware(plan["target_finish_kst"], "provisional target finish")
    ):
        raise HandoffContractError("collector input lineage drifted")
    return watch, plan, params, profile, submission


def _normalize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    task_id = provisional._task_id(task)
    return {
        "task_id": task_id,
        "name": task.get("name"),
        "status": str(task.get("status") or "").strip().lower(),
        "state": str(task.get("state") or "").strip().lower(),
        "exit_code": task.get("exit_code"),
        "failure_message": task.get("failure_message"),
        "project": task.get("project"),
        "dedupe_key": task.get("dedupe_key"),
        "cpus": task.get("cpus"),
        "memory_mb": task.get("memory_mb"),
        "timeout_seconds": task.get("timeout_seconds"),
        "aedt_backend": task.get("aedt_backend"),
        "account_name": task.get("account_name"),
        "requested_account_name": task.get("requested_account_name"),
        "actual_node_name": task.get("actual_node_name"),
        "allocation_node_name": task.get("allocation_node_name"),
        "requested_node_name": task.get("requested_node_name"),
        "requested_node_name_policy": task.get(
            "requested_node_name_policy", task.get("node_name_policy")
        ),
        "allocation_id": task.get("allocation_id", task.get("assigned_allocation")),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "remote_cwd": task.get("remote_cwd"),
        "remote_dir": task.get("remote_dir"),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
    }


def _validate_task_identity(
    task: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    submission: Mapping[str, Any],
    terminal_success: bool,
) -> dict[str, Any]:
    row = _normalize_task(task)
    account, node = provisional._selected_placement(plan)
    actual_node = row["actual_node_name"] or row["allocation_node_name"]
    if (
        row["task_id"] != submission["task_id"]
        or row["name"] != submission["task_name"]
        or row["project"] != scheduler_client.MFT_PROJECT
        or row["dedupe_key"] != submission["dedupe_key"]
        or row["cpus"] != provisional.RESOURCES["cpus"]
        or row["memory_mb"] != provisional.RESOURCES["memory_mb"]
        or row["timeout_seconds"] != provisional.RESOURCES["timeout_seconds"]
        or row["aedt_backend"] != "standalone"
        or row["account_name"] != account
        or row["requested_account_name"] != account
        or row["requested_node_name"] != node
        or row["requested_node_name_policy"] != "strict"
        or (actual_node is not None and actual_node != node)
    ):
        raise HandoffContractError("provisional Full Scheduler task lineage drifted")
    if terminal_success:
        allocation = row["allocation_id"]
        if (
            row["status"] != "completed"
            or row["state"] != "succeeded"
            or row["exit_code"] != 0
            or row["failure_message"] not in {"", None}
            or actual_node != node
            or not row["slurm_job_id"].isdigit()
            or isinstance(allocation, bool)
            or not isinstance(allocation, int)
            or allocation <= 0
            or not str(row["remote_cwd"] or row["remote_dir"] or "")
            or not str(row["finished_at"] or "")
        ):
            raise HandoffContractError(
                "provisional Full terminal success evidence drifted"
            )
    return row


def _validate_result(
    result: Any,
    *,
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    submission: Mapping[str, Any],
    task: Mapping[str, Any],
    scheduler: Any,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise HandoffContractError("provisional Full RESULT_JSON is not an object")
    value = copy.deepcopy(dict(result))
    effective = production._effective_params(params, profile)
    if (
        not scheduler.result_matches_params(
            value,
            effective,
            required_keys=set(production.ALL_INPUT_KEYS),
        )
        or canonical_sha256(effective) != plan["effective_full_params_sha256"]
        or production._require_revision(
            value.get("git_hash"), "provisional Full solver revision"
        )
        != plan["solver_revision"]
        or production._require_revision(
            value.get("pyaedt_library_git_hash"),
            "provisional Full library revision",
        )
        != plan["library_revision"]
        or production._integer(value.get("full_model"), "provisional Full model mode")
        != 1
        or str(value.get("thermal_symmetry") or "") != "full"
        or str(value.get("solver_core_slurm_job_id_readback") or "")
        != task["slurm_job_id"]
    ):
        raise HandoffContractError("provisional Full RESULT_JSON identity drifted")
    core_submission = {
        "stage": "full",
        "task_id": submission["task_id"],
        "solver_revision": plan["solver_revision"],
        "core_policy": {
            "contract": production.FULL_CORE_CONTRACT,
            "requested_num_cores": 16,
        },
    }
    production._validate_result_core_policy(value, core_submission)
    return value


def _validate_source_cap(
    *,
    plan: Mapping[str, Any],
    watch: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> int:
    bound = plan.get("retention_storage_bound")
    if not isinstance(bound, Mapping):
        raise HandoffContractError(
            "provisional Full retained source hard cap is absent"
        )
    cap = _positive_int(
        bound.get("expected_full_aedt_bytes"),
        "sealed Full AEDT source hard cap",
    )
    size = _positive_int(receipt.get("artifact_size_bytes"), "retained Full AEDT size")
    raw_chunk = _positive_int(
        receipt.get("transport_raw_chunk_bytes"),
        "retained Full raw chunk size",
    )
    chunk_count = _positive_int(
        receipt.get("transport_chunk_count"),
        "retained Full chunk count",
    )
    if (
        cap != watch["sealed_source_size_hard_cap_bytes"]
        or cap > scheduler_client.RETAINED_AEDT_MAX_BYTES
        or size > cap
        or chunk_count != math.ceil(size / raw_chunk)
        or receipt.get("source_size_hard_cap_bytes") != cap
        or receipt.get("source_size_hard_cap_contract") != SOURCE_SIZE_HARD_CAP_CONTRACT
        or receipt.get("source_size_hard_cap_enforced_before_destination_create")
        is not True
        or plan["stage"]["retained_aedt"].get("source_size_hard_cap_bytes") != cap
        or plan["stage"]["retained_aedt"].get("source_size_hard_cap_contract")
        != SOURCE_SIZE_HARD_CAP_CONTRACT
        or "results_path" in plan["stage"]["retained_aedt"]
        or "results_manifest_path" in plan["stage"]["retained_aedt"]
    ):
        raise HandoffContractError(
            "retained Full AEDT exceeds or drifts from sealed project-only cap"
        )
    return cap


def _validated_capped_remote_artifact_receipt(
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
    scheduler_url: str,
    remote_reader: Callable[..., bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the explicit pre-GPFS hard-cap receipt schema extension."""
    expected = submission["retained_aedt"]
    task_id = int(submission["task_id"])
    receipt_raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=expected["receipt_path"],
        max_bytes=production.MAX_REMOTE_METADATA_BYTES,
    )
    marker_raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=expected["marker_path"],
        max_bytes=production.MAX_REMOTE_METADATA_BYTES,
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
        marker = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "capped retained Full metadata is invalid JSON"
        ) from exc
    if not isinstance(receipt, dict) or set(receipt) != set(
        production.REMOTE_RECEIPT_FIELDS
    ) | set(CAP_RECEIPT_FIELDS):
        raise HandoffContractError("capped retained Full receipt fields drifted")
    legacy = {name: receipt[name] for name in production.REMOTE_RECEIPT_FIELDS}
    production._validate_remote_receipt_payload(
        legacy,
        submission=submission,
        result=result,
    )
    marker = production._validate_marker_payload(marker, expected=expected)
    if (
        receipt.get("source_size_hard_cap_bytes")
        != expected.get("source_size_hard_cap_bytes")
        or receipt.get("source_size_hard_cap_contract")
        != expected.get("source_size_hard_cap_contract")
        or receipt.get("source_size_hard_cap_enforced_before_destination_create")
        is not True
        or production._sha256_bytes(marker_raw) != receipt["marker_sha256"]
    ):
        raise HandoffContractError("retained Full pre-GPFS hard-cap evidence drifted")
    return receipt, marker


def _read_validated_remote_chunk(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    expected_encoded_size: int,
    expected_raw_size: int,
    maximum_encoded_size: int,
) -> bytes:
    """Read and validate one base64 chunk with bounded GET retries."""
    pure = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure.is_absolute()
        or ".." in pure.parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or not 0 < expected_encoded_size <= maximum_encoded_size
        or maximum_encoded_size > production.MAX_REMOTE_TEXT_CHUNK_BYTES
        or not 0 < expected_raw_size <= scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
    ):
        raise HandoffContractError("unsafe retained Full chunk read contract")
    query = urllib.parse.urlencode(
        {
            "path": relative_path,
            "base": "remote_cwd",
            "max_bytes": production.MAX_REMOTE_TEXT_CHUNK_BYTES,
        }
    )
    url = scheduler_url.rstrip("/") + f"/api/tasks/{task_id}/remote-file?{query}"
    last_error: BaseException | None = None
    for _attempt in range(REMOTE_CHUNK_READ_ATTEMPTS):
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=120.0) as response:
                encoded = response.read(production.MAX_REMOTE_TEXT_CHUNK_BYTES + 1)
            if (
                len(encoded) != expected_encoded_size
                or len(encoded) > maximum_encoded_size
            ):
                raise HandoffContractError("retained Full encoded chunk size drifted")
            try:
                chunk = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise HandoffContractError(
                    "retained Full chunk is invalid base64"
                ) from exc
            if len(chunk) != expected_raw_size:
                raise HandoffContractError("retained Full raw chunk size drifted")
            return chunk
        except (
            OSError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            HandoffContractError,
        ) as exc:
            last_error = exc
    raise HandoffContractError(
        "retained Full chunk GET/validation exhausted bounded retries"
    ) from last_error


def _fetch_capped_remote_to_path(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    transport_chunk_directory: str,
    transport_raw_chunk_bytes: int,
    transport_max_encoded_chunk_bytes: int,
    transport_chunk_count: int,
    expected_size: int,
    expected_sha256: str,
    destination: Path,
    chunk_reader: Callable[..., bytes] = _read_validated_remote_chunk,
) -> None:
    """Atomically reconstruct one retained AEDT after per-chunk retries."""
    artifact = PurePosixPath(relative_path)
    chunks = PurePosixPath(transport_chunk_directory)
    if (
        not relative_path
        or artifact.is_absolute()
        or ".." in artifact.parts
        or not transport_chunk_directory
        or chunks.is_absolute()
        or ".." in chunks.parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 0 < expected_size <= scheduler_client.RETAINED_AEDT_MAX_BYTES
        or production._require_sha(expected_sha256, "retained Full AEDT SHA")
        != expected_sha256
        or transport_raw_chunk_bytes != scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
        or transport_max_encoded_chunk_bytes
        != scheduler_client.RETAINED_AEDT_MAX_ENCODED_CHUNK_BYTES
        or isinstance(transport_chunk_count, bool)
        or not isinstance(transport_chunk_count, int)
        or transport_chunk_count != math.ceil(expected_size / transport_raw_chunk_bytes)
        or destination.exists()
    ):
        raise HandoffContractError("unsafe capped retained Full fetch contract")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    count = 0
    try:
        with temporary.open("xb") as stream:
            for index in range(transport_chunk_count):
                raw_size = min(
                    transport_raw_chunk_bytes,
                    expected_size - count,
                )
                encoded_size = 4 * math.ceil(raw_size / 3)
                chunk = chunk_reader(
                    scheduler_url=scheduler_url,
                    task_id=task_id,
                    relative_path=(f"{transport_chunk_directory}/{index:08d}.b64"),
                    expected_encoded_size=encoded_size,
                    expected_raw_size=raw_size,
                    maximum_encoded_size=(transport_max_encoded_chunk_bytes),
                )
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if count != expected_size or digest.hexdigest() != expected_sha256:
            raise HandoffContractError(
                "reconstructed Full AEDT differs from sealed receipt"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            os.remove(temporary)
        raise


def _state(
    watch: Mapping[str, Any],
    *,
    status: str,
    observed_at: datetime,
    **extra: Any,
) -> dict[str, Any]:
    details = dict(extra)
    streak = details.pop("consecutive_error_streak", 0)
    if (
        isinstance(streak, bool)
        or not isinstance(streak, int)
        or not 0 <= streak <= MAX_CONSECUTIVE_ERRORS
    ):
        raise HandoffContractError("collector consecutive error streak is invalid")
    value = _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan_payload_sha256": watch["payload_sha256"],
            "task_id": watch["task_id"],
            "status": status,
            "observed_at_utc": _stamp(observed_at),
            "consecutive_error_streak": streak,
            "maximum_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "remote_artifact_mutation_performed": False,
            "gpfs_write_performed": False,
            "gpfs_delete_performed": False,
            **_flags(),
            **details,
        }
    )
    _write_atomic(Path(watch["output_root"]) / "state.json", value)
    return value


def _previous_error_streak(watch: Mapping[str, Any]) -> int:
    path = Path(watch["output_root"]) / "state.json"
    if not path.exists():
        return 0
    state = production._validate_seal(production._read_json(path), STATE_SCHEMA)
    streak = state.get("consecutive_error_streak")
    if (
        state.get("watch_plan_payload_sha256") != watch["payload_sha256"]
        or state.get("task_id") != watch["task_id"]
        or state.get("maximum_consecutive_errors") != MAX_CONSECUTIVE_ERRORS
        or isinstance(streak, bool)
        or not isinstance(streak, int)
        or not 0 <= streak <= MAX_CONSECUTIVE_ERRORS
        or state.get("scheduler_methods_used") != ["GET"]
        or state.get("scheduler_mutation_performed") is not False
        or state.get("remote_artifact_mutation_performed") is not False
        or state.get("gpfs_write_performed") is not False
        or state.get("gpfs_delete_performed") is not False
        or any(state.get(name) is not expected for name, expected in _flags().items())
    ):
        raise HandoffContractError("collector mutable state authority drifted")
    if state.get("status") != "retrying_after_collection_error":
        return 0
    return streak


def _failure(
    watch: Mapping[str, Any],
    *,
    category: str,
    reason: str,
    observed_status: str,
    task: Mapping[str, Any] | None,
    observed_at: datetime,
) -> dict[str, Any]:
    path = Path(watch["failure_ledger_path"])
    if path.exists():
        return production._validate_seal(production._read_json(path), FAILURE_SCHEMA)
    artifact = Path(watch["local_full_aedt_path"])
    value = _sealed(
        {
            "schema_version": FAILURE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan": production._file_record(
                Path(watch["output_root"]) / "watch_plan.json"
            ),
            "watch_plan_payload_sha256": watch["payload_sha256"],
            "task_id": watch["task_id"],
            "category": category,
            "reason": reason,
            "observed_status": observed_status,
            "scheduler_task": copy.deepcopy(dict(task)) if task else None,
            "local_full_aedt_exists": artifact.exists(),
            "success_receipt_created": False,
            "collection_blocked_fail_closed": True,
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "remote_artifact_mutation_performed": False,
            "gpfs_write_performed": False,
            "gpfs_delete_performed": False,
            "canonical_truth_gate_modified": False,
            "canonical_claim_modified": False,
            "canonical_promotion_modified": False,
            "created_at_utc": _stamp(observed_at),
            **_flags(),
        }
    )
    _write_immutable(path, value)
    return value


def _validate_success_replay(
    path: Path,
    *,
    watch: Mapping[str, Any],
    plan: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    value = production._validate_seal(
        production._read_json(path.resolve(strict=True)), SUCCESS_SCHEMA
    )
    local = Path(watch["local_full_aedt_path"])
    local_record = value.get("local_full_aedt")
    receipt = value.get("remote_retained_aedt_receipt")
    if (
        value.get("watch_plan")
        != production._file_record(Path(watch["output_root"]) / "watch_plan.json")
        or value.get("watch_plan_payload_sha256") != watch["payload_sha256"]
        or value.get("provisional_plan_payload_sha256") != plan["payload_sha256"]
        or value.get("submission_payload_sha256") != submission["payload_sha256"]
        or value.get("task_id") != submission["task_id"]
        or value.get("candidate_physics_sha256") != plan["candidate_physics_sha256"]
        or not isinstance(receipt, Mapping)
        or not isinstance(local_record, Mapping)
        or production._file_record(local) != local_record
        or local_record.get("sha256") != receipt.get("artifact_sha256")
        or local_record.get("size_bytes") != receipt.get("artifact_size_bytes")
        or value.get("scheduler_methods_used") != ["GET"]
        or value.get("scheduler_mutation_performed") is not False
        or value.get("remote_artifact_mutation_performed") is not False
        or value.get("gpfs_write_performed") is not False
        or value.get("gpfs_delete_performed") is not False
        or value.get("canonical_truth_gate_modified") is not False
        or value.get("canonical_claim_modified") is not False
        or value.get("canonical_promotion_modified") is not False
        or any(value.get(name) is not expected for name, expected in _flags().items())
    ):
        raise HandoffContractError(
            "provisional Full collection replay authority drifted"
        )
    _validate_source_cap(plan=plan, watch=watch, receipt=receipt)
    return value


def _default_status_reader(*, scheduler_url: str, task_id: int) -> Any:
    return scheduler_client.get_status(task_id, scheduler_url=scheduler_url)


def process_cycle(
    *,
    watch_plan_path: Path,
    scheduler: Any = scheduler_client,
    status_reader: Callable[..., Any] = _default_status_reader,
    task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    remote_reader: Callable[..., bytes] = production._remote_bytes,
    artifact_fetcher: Callable[..., None] = _fetch_capped_remote_to_path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one authenticated watch/collection cycle."""
    current = (now or _now()).astimezone(timezone.utc)
    watch, plan, params, profile, submission = _load_watch_plan(watch_plan_path)
    success_path = Path(watch["success_receipt_path"])
    failure_path = Path(watch["failure_ledger_path"])
    if success_path.exists():
        success = _validate_success_replay(
            success_path,
            watch=watch,
            plan=plan,
            submission=submission,
        )
        return _state(
            watch,
            status="completed_immutable_replay",
            observed_at=current,
            success_receipt=production._file_record(success_path),
            local_full_aedt=success["local_full_aedt"],
        )
    if failure_path.exists():
        failure = production._validate_seal(
            production._read_json(failure_path), FAILURE_SCHEMA
        )
        return _state(
            watch,
            status="blocked_by_immutable_failure",
            observed_at=current,
            failure_ledger=production._file_record(failure_path),
            failure_category=failure["category"],
        )

    task_id = int(submission["task_id"])
    previous_streak = _previous_error_streak(watch)
    deadline = _aware(watch["terminal_deadline_utc"], "terminal deadline")
    try:
        raw_status = status_reader(
            scheduler_url=watch["scheduler_url"], task_id=task_id
        )
        status = str(raw_status or "").strip().lower()
        if not status:
            raise HandoffContractError(
                "provisional Full Scheduler status is temporarily unavailable"
            )
        raw_task = task_reader(scheduler_url=watch["scheduler_url"], task_id=task_id)
        task = _validate_task_identity(
            raw_task,
            plan=plan,
            submission=submission,
            terminal_success=status == "completed",
        )
        if task["status"] != status:
            raise HandoffContractError("Scheduler status and task snapshot disagree")
        if status in ACTIVE_STATUSES:
            if current > deadline:
                failure = _failure(
                    watch,
                    category="deadline_exceeded",
                    reason="provisional Full remained active after deadline",
                    observed_status=status,
                    task=task,
                    observed_at=current,
                )
                return _state(
                    watch,
                    status="blocked_deadline_exceeded",
                    observed_at=current,
                    failure_ledger=production._file_record(failure_path),
                    failure_category=failure["category"],
                )
            return _state(
                watch,
                status="watching_active_task",
                observed_at=current,
                scheduler_status=status,
                scheduler_task=task,
            )
        if status in FAILED_STATUSES:
            failure = _failure(
                watch,
                category="terminal_task_failure",
                reason=f"provisional Full terminal status={status}",
                observed_status=status,
                task=task,
                observed_at=current,
            )
            return _state(
                watch,
                status="blocked_terminal_failure",
                observed_at=current,
                failure_ledger=production._file_record(failure_path),
                failure_category=failure["category"],
            )
        if status != "completed":
            raise HandoffContractError(
                f"unexpected provisional Full Scheduler status={status!r}"
            )
        finished_at = _aware(task["finished_at"], "provisional Full finished_at")
        if finished_at > deadline:
            raise HandoffContractError(
                "provisional Full terminal success missed sealed deadline"
            )
        fetched = scheduler.fetch_result(
            task_id,
            expected_revision=plan["solver_revision"],
            expected_library_revision=plan["library_revision"],
            expected_profile=profile["param_overrides"],
            scheduler_url=watch["scheduler_url"],
        )
        if fetched.state != scheduler.RESULT_VALID or not isinstance(
            fetched.result, Mapping
        ):
            raise HandoffContractError(
                "provisional Full RESULT_JSON is not strict-valid"
            )
        result = _validate_result(
            fetched.result,
            plan=plan,
            params=params,
            profile=profile,
            submission=submission,
            task=task,
            scheduler=scheduler,
        )
        remote_submission = {
            "stage": "full",
            "task_id": task_id,
            "retained_aedt": submission["retained_aedt"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "profile_sha256": plan["profile_canonical_sha256"],
        }
        remote_receipt, marker = _validated_capped_remote_artifact_receipt(
            submission=remote_submission,
            result=result,
            scheduler_url=watch["scheduler_url"],
            remote_reader=remote_reader,
        )
        cap = _validate_source_cap(plan=plan, watch=watch, receipt=remote_receipt)
        local_aedt = Path(watch["local_full_aedt_path"])
        if local_aedt.exists():
            if (
                not local_aedt.is_file()
                or production._sha256_file(local_aedt)
                != remote_receipt["artifact_sha256"]
                or local_aedt.stat().st_size != remote_receipt["artifact_size_bytes"]
            ):
                raise HandoffContractError(
                    "pre-existing local Full AEDT differs from remote receipt"
                )
        else:
            artifact_fetcher(
                scheduler_url=watch["scheduler_url"],
                task_id=task_id,
                relative_path=remote_receipt["artifact_path"],
                transport_chunk_directory=remote_receipt["transport_chunk_directory"],
                transport_raw_chunk_bytes=remote_receipt["transport_raw_chunk_bytes"],
                transport_max_encoded_chunk_bytes=remote_receipt[
                    "transport_max_encoded_chunk_bytes"
                ],
                transport_chunk_count=remote_receipt["transport_chunk_count"],
                expected_size=remote_receipt["artifact_size_bytes"],
                expected_sha256=remote_receipt["artifact_sha256"],
                destination=local_aedt,
            )
        local_record = production._file_record(local_aedt)
        if (
            local_record["sha256"] != remote_receipt["artifact_sha256"]
            or local_record["size_bytes"] != remote_receipt["artifact_size_bytes"]
        ):
            raise HandoffContractError(
                "reconstructed local Full AEDT verification failed"
            )
        result_sha = canonical_sha256(result)
        success = _sealed(
            {
                "schema_version": SUCCESS_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "watch_plan": production._file_record(watch_plan_path),
                "watch_plan_payload_sha256": watch["payload_sha256"],
                "provisional_plan": watch["provisional_plan"],
                "provisional_plan_payload_sha256": plan["payload_sha256"],
                "submission": watch["submission"],
                "submission_payload_sha256": submission["payload_sha256"],
                "candidate_physics_sha256": plan["candidate_physics_sha256"],
                "logical_authority_task_id": plan["logical_authority_task_id"],
                "source_actual_standard_task_id": plan[
                    "source_actual_standard_task_id"
                ],
                "task_id": task_id,
                "task_name": submission["task_name"],
                "dedupe_key": submission["dedupe_key"],
                "selected_account_name": submission["selected_account_name"],
                "requested_node_name": submission["requested_node_name"],
                "scheduler_task_execution": task,
                "result": result,
                "result_sha256": result_sha,
                "result_identity": {
                    "solver_revision": plan["solver_revision"],
                    "library_revision": plan["library_revision"],
                    "profile_canonical_sha256": plan["profile_canonical_sha256"],
                    "effective_full_params_sha256": plan[
                        "effective_full_params_sha256"
                    ],
                    "project_name": result["project_name"],
                    "full_model": 1,
                    "thermal_symmetry": "full",
                },
                "remote_retained_aedt_receipt": remote_receipt,
                "remote_prune_protection_marker": marker,
                "remote_prune_protection_marker_verified": True,
                "sealed_source_size_hard_cap_bytes": cap,
                "source_artifact_size_within_sealed_hard_cap": True,
                "all_declared_chunks_reconstructed": True,
                "local_full_aedt": local_record,
                "immutable_replay_required": True,
                "canonical_truth_authority_claimed": False,
                "canonical_promotion_allowed": False,
                "scheduler_methods_used": ["GET"],
                "scheduler_mutation_performed": False,
                "remote_artifact_mutation_performed": False,
                "gpfs_write_performed": False,
                "gpfs_delete_performed": False,
                "canonical_truth_gate_modified": False,
                "canonical_claim_modified": False,
                "canonical_promotion_modified": False,
                "created_at_utc": _stamp(current),
                **_flags(),
            }
        )
        _write_immutable(success_path, success)
        return _state(
            watch,
            status="completed_provisional_full_collected",
            observed_at=current,
            success_receipt=production._file_record(success_path),
            local_full_aedt=local_record,
        )
    except (
        HandoffContractError,
        OSError,
        scheduler_client.ResultFetchError,
    ) as exc:
        streak = previous_streak + 1
        if current <= deadline and streak < MAX_CONSECUTIVE_ERRORS:
            return _state(
                watch,
                status="retrying_after_collection_error",
                observed_at=current,
                consecutive_error_streak=streak,
                error=str(exc),
                last_observed_status=locals().get("status", ""),
                retry_allowed_before_deadline=True,
                immutable_failure_created=False,
            )
        category = (
            "collection_error_after_deadline"
            if current > deadline
            else "repeated_collection_error"
        )
        failure = _failure(
            watch,
            category=category,
            reason=str(exc),
            observed_status=locals().get("status", ""),
            task=locals().get("task"),
            observed_at=current,
        )
        return _state(
            watch,
            status="blocked_collection_error",
            observed_at=current,
            consecutive_error_streak=min(streak, MAX_CONSECUTIVE_ERRORS),
            failure_ledger=production._file_record(failure_path),
            failure_category=failure["category"],
            error=str(exc),
        )


def watch(
    *,
    watch_plan_path: Path,
    max_cycles: int | None = None,
    **cycle_kwargs: Any,
) -> dict[str, Any]:
    watch_plan = _load_watch_plan(watch_plan_path)[0]
    cycles = 0
    pid = _sealed(
        {
            "schema_version": PID_SCHEMA,
            "watch_plan": production._file_record(watch_plan_path),
            "watch_plan_payload_sha256": watch_plan["payload_sha256"],
            "pid": os.getpid(),
            "started_at_utc": _stamp(),
            "scheduler_methods_allowed": ["GET"],
            "scheduler_mutation_performed": False,
            **_flags(),
        }
    )
    _write_atomic(Path(watch_plan["output_root"]) / "watcher.pid.json", pid)
    while True:
        state = process_cycle(watch_plan_path=watch_plan_path, **cycle_kwargs)
        cycles += 1
        if state["status"].startswith(("completed_", "blocked_")):
            return state
        if max_cycles is not None and cycles >= max_cycles:
            return state
        time.sleep(int(watch_plan["poll_seconds"]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--plan", type=Path, required=True)
    init.add_argument("--submission", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    once = commands.add_parser("once")
    once.add_argument("--watch-plan", type=Path, required=True)
    run = commands.add_parser("watch")
    run.add_argument("--watch-plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result: Any = initialize_watch_plan(
            provisional_plan_path=args.plan,
            submission_path=args.submission,
            output_root=args.output_root,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "once":
        result = process_cycle(watch_plan_path=args.watch_plan)
    else:
        result = watch(watch_plan_path=args.watch_plan)
    if isinstance(result, Path):
        print(
            json.dumps(
                {"status": "initialized", "path": str(result)},
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(result, sort_keys=True))
    if isinstance(result, Mapping) and str(result.get("status", "")).startswith(
        "blocked_"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
