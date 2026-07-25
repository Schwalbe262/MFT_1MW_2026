"""Exact-once infrastructure-startup successor for logical slot 96223.

The source task is the immutable task-96303 terminal.  This module preserves
its candidate, physics, profile, solver/library revisions, resources, strict
r1jae262/n114 placement, and retained-result semantics.  The only operational
changes are a fresh generation/identity and task-local AEDT temporary paths.

Dry-run is the default.  A live cycle can issue at most one Scheduler POST for
the lifetime of the dedicated atomic claim and never calls a cancel endpoint.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_campaign_atomic_claim as atomic_claim  # noqa: E402
from tools import mft_goal_bounded_parallel_refill as bounded  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_safe_refill as refill  # noqa: E402
from tools import mft_goal_terminal_success_watcher as watcher  # noqa: E402
from tools import mft_goal_timeout12h_retry as timeout12h  # noqa: E402


PLAN_SCHEMA = timeout12h.PLAN_SCHEMA
SUBMISSION_SCHEMA = timeout12h.SUBMISSION_SCHEMA
REPLACEMENT_SCHEMA = "mft-goal-watcher-authorized-replacement-v1"
STARTUP_RECORD_SCHEMA = "mft-goal-startup-successor-lineage-v1"
FAILURE_EVIDENCE_SCHEMA = "mft-goal-startup-failure-evidence-v1"
SIBLING_SCHEMA = "mft-goal-startup-successor-sibling-guard-v1"
CLAIM_RECEIPT_SCHEMA = "mft-goal-startup-successor-claim-receipt-v1"
EVALUATION_SCHEMA = "mft-goal-startup-successor-evaluation-v1"
STATE_SCHEMA = "mft-goal-startup-successor-state-v1"

CAMPAIGN_ID = "mft-goal-20260726"
GENERATION = "startup-r1"
LOGICAL_ID = 96223
FAILED_TASK_ID = 96303
TARGET_ACCOUNT = "r1jae262"
TARGET_NODE = "n114"
PROJECT = refill.PROJECT
SCHEDULER_URL = refill.SCHEDULER_URL
RESOURCES = {"cpus": 8, "memory_mb": 32768, "timeout_seconds": 43200}
PROSPECTIVE_GRID_GIB = 19.15191717632115
EXPECTED_CANDIDATE_SHA256 = (
    "2a1bb6f2be79d5a9443538702b833e2b1918877a1137923c1a1ea0f606b46660"
)
EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256 = (
    "08748c42a0306c6cace6aebd91ff83e186ac0c190ef9d2a212447b26aae9d2d6"
)
EXPECTED_SOURCE_SUBMISSION_PAYLOAD_SHA256 = (
    "0f631c6ba8cda1b10f7311481e799497821e5e797eee53f7e2af593056d56a1b"
)
EXPECTED_FAILURE_MESSAGE = (
    "RuntimeError: AEDT desktop startup failed after 3 attempts: "
    "AttributeError: 'NoneType' object has no attribute 'EnableAutoSave'; "
    "AttributeError: 'NoneType' object has no attribute 'EnableAutoSave'; "
    "AttributeError: 'NoneType' object has no attribute 'EnableAutoSave'"
)
EXPECTED_FIXED_IDENTITY_SHA256 = refill.FIXED_IDENTITY_SHA256
WORKDIR_PREFIX = "/enroot/"
TEMP_ENVIRONMENT = {
    "ANS_TEMP_PATH": "$MFT_WORKDIR",
    "TEMP": "$MFT_WORKDIR",
    "TMP": "$MFT_WORKDIR",
    "TMPDIR": "$MFT_WORKDIR",
}
CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "startup_successor_claims_r1"
)
SUBMISSION_CUTOFF_KST = datetime(
    2026, 7, 26, 5, 30, tzinfo=timezone(timedelta(hours=9))
)
SUBMISSION_CUTOFF_UTC = SUBMISSION_CUTOFF_KST.astimezone(timezone.utc)
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": CAMPAIGN_ID,
        "retry_generation": GENERATION,
        "logical_authority_task_id": LOGICAL_ID,
        "failed_task_id": FAILED_TASK_ID,
        "candidate_physics_sha256": EXPECTED_CANDIDATE_SHA256,
        "submission_environment": TEMP_ENVIRONMENT,
        "required_workdir_prefix": WORKDIR_PREFIX,
        "resources": RESOURCES,
        "target_account": TARGET_ACCOUNT,
        "target_node": TARGET_NODE,
    }
)
MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_RECONCILIATION_READS = 8


class StartupRetryError(production.HandoffContractError):
    """Fail-closed startup-successor contract error."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise StartupRetryError("startup successor payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StartupRetryError(f"startup JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise StartupRetryError(f"startup JSON artifact is not an object: {path}")
    return value


def _validate_seal(
    value: Mapping[str, Any], schema: str
) -> dict[str, Any]:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise StartupRetryError(f"{schema} seal mismatch")
    return copy.deepcopy(dict(value))


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
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


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    try:
        return production._write_immutable_json(path.resolve(), value)
    except Exception as exc:
        raise StartupRetryError(
            f"startup immutable output exists or failed: {path}"
        ) from exc


def _file_record(path: Path) -> dict[str, Any]:
    return production._file_record(path.resolve(strict=True))


def _task_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("task_id", row.get("id"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _flags() -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "standard_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "full_submission_allowed": False,
        "production_package_allowed": False,
    }


def _task_identity(candidate_sha256: str) -> tuple[str, str]:
    if candidate_sha256 != EXPECTED_CANDIDATE_SHA256:
        raise StartupRetryError("startup successor candidate identity drifted")
    stem = candidate_sha256[:12]
    return (
        f"mft-goal-diag-standard-{GENERATION}-l{LOGICAL_ID}-{stem}",
        f"mft_goal_diag_standard_{GENERATION.replace('-', '_')}_"
        f"l{LOGICAL_ID}_{stem}",
    )


def _claim_authority() -> dict[str, Any]:
    try:
        authority = atomic_claim.load_claim_root(CLAIM_ROOT)
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError(
            "startup successor atomic claim root is unavailable"
        ) from exc
    if (
        authority.get("campaign_id") != CAMPAIGN_ID
        or authority.get("campaign_authority_sha256")
        != CLAIM_AUTHORITY_SHA256
    ):
        raise StartupRetryError("startup successor claim authority drifted")
    return authority


def initialize_claim_root() -> dict[str, Any]:
    try:
        return atomic_claim.initialize_claim_root(
            CLAIM_ROOT,
            campaign_id=CAMPAIGN_ID,
            campaign_authority_sha256=CLAIM_AUTHORITY_SHA256,
        )
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError(
            "startup successor atomic claim root cannot be initialized"
        ) from exc


def _claim_reference() -> dict[str, Any]:
    try:
        return atomic_claim.build_claim_reference(
            _claim_authority(),
            candidate_physics_sha256=EXPECTED_CANDIDATE_SHA256,
            logical_authority_task_id=LOGICAL_ID,
            retry_generation=GENERATION,
        )
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError(
            "startup successor claim reference is invalid"
        ) from exc


def _validate_claim_reference(value: Any) -> dict[str, Any]:
    try:
        reference = atomic_claim.validate_claim_reference(
            value, _claim_authority()
        )
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError(
            "startup successor claim reference drifted"
        ) from exc
    if (
        reference.get("candidate_physics_sha256")
        != EXPECTED_CANDIDATE_SHA256
        or reference.get("logical_authority_task_id") != LOGICAL_ID
        or reference.get("retry_generation") != GENERATION
    ):
        raise StartupRetryError("startup successor claim binding drifted")
    return reference


def _normalized_source_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "name",
        "status",
        "state",
        "exit_code",
        "failure_message",
        "timeout_seconds",
        "slurm_job_id",
        "allocation_id",
        "account_name",
        "requested_account_name",
        "actual_node_name",
        "cpus",
        "memory_mb",
        "aedt_backend",
        "project",
        "dedupe_key",
        "remote_cwd",
        "remote_dir",
        "started_at",
        "finished_at",
        "requested_node_name",
        "requested_node_name_policy",
        "placement_contract_satisfied",
    )
    result = {name: snapshot.get(name) for name in fields}
    result["task_id"] = _task_id(snapshot)
    return result


def _source_failure_evidence(
    *,
    snapshot: Mapping[str, Any],
    stdout: bytes | str,
    stderr: bytes | str,
    source_submission: Mapping[str, Any],
    retained_inventory: Mapping[str, Any],
    collection_exists: bool,
    success_receipt_exists: bool,
) -> dict[str, Any]:
    row = _normalized_source_snapshot(snapshot)
    out = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    err = stderr.encode("utf-8") if isinstance(stderr, str) else stderr
    if (
        not isinstance(out, bytes)
        or not isinstance(err, bytes)
        or not out
        or not err
        or len(out) > MAX_STREAM_BYTES
        or len(err) > MAX_STREAM_BYTES
    ):
        raise StartupRetryError("startup source stream evidence is invalid")
    try:
        out_text = out.decode("utf-8")
        err_text = err.decode("utf-8")
    except UnicodeError as exc:
        raise StartupRetryError(
            "startup source streams are not UTF-8"
        ) from exc
    attempts = [
        int(match.group(1))
        for match in re.finditer(
            r"(?m)^WARNING:root:AEDT session startup attempt "
            r"([1-3])/3 failed: AttributeError: 'NoneType' object has no "
            r"attribute 'EnableAutoSave'$",
            err_text,
        )
    ]
    workdirs = re.findall(r"(?m)^MFT_WORKDIR (/.+)$", out_text)
    files = retained_inventory.get("files")
    retained = source_submission["retained_aedt_bundle"]
    if not isinstance(files, list):
        raise StartupRetryError(
            "startup source retained-file inventory is malformed"
        )
    complete_paths = {
        retained["receipt_path"],
        retained["results_manifest_path"],
    }
    observed_paths = {
        str(item.get("path") or "")
        for item in files
        if isinstance(item, Mapping)
    }
    if (
        row["task_id"] != FAILED_TASK_ID
        or row["task_id"] != source_submission.get("task_id")
        or row["name"] != source_submission.get("task_name")
        or row["dedupe_key"] != source_submission.get("dedupe_key")
        or row["status"] != "failed"
        or row["state"] != "failed"
        or row["exit_code"] != 1
        or row["failure_message"] != EXPECTED_FAILURE_MESSAGE
        or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
        or row["cpus"] != RESOURCES["cpus"]
        or row["memory_mb"] != RESOURCES["memory_mb"]
        or row["aedt_backend"] != "standalone"
        or row["project"] != PROJECT
        or row["account_name"] != TARGET_ACCOUNT
        or row["requested_account_name"] != TARGET_ACCOUNT
        or row["actual_node_name"] != TARGET_NODE
        or row["requested_node_name"] != TARGET_NODE
        or row["requested_node_name_policy"] != "strict"
        or row["placement_contract_satisfied"] is not True
        or not str(row["slurm_job_id"] or "").isdigit()
        or not str(row["remote_cwd"] or "")
        or not str(row["remote_dir"] or "")
        or not str(row["started_at"] or "")
        or not str(row["finished_at"] or "")
        or attempts != [1, 2, 3]
        or EXPECTED_FAILURE_MESSAGE not in err_text
        or len(workdirs) != 1
        or not workdirs[0].startswith(WORKDIR_PREFIX)
        or "RESULT_JSON " in out_text
        or "RESULT_JSON:" in out_text
        or "RESULT_JSON " in err_text
        or "RESULT_JSON:" in err_text
        or complete_paths.intersection(observed_paths)
        or collection_exists
        or success_receipt_exists
    ):
        raise StartupRetryError(
            "startup retry requires exact task96303 exit1 Desktop-startup "
            "3/3 terminal with RESULT_JSON/collection/complete bundle absent"
        )
    return {
        "schema_version": FAILURE_EVIDENCE_SCHEMA,
        "scheduler_snapshot": row,
        "stdout_sha256": production._sha256_bytes(out),
        "stdout_size_bytes": len(out),
        "stderr_sha256": production._sha256_bytes(err),
        "stderr_size_bytes": len(err),
        "startup_attempts": attempts,
        "desktop_startup_failed_after_three_attempts": True,
        "result_json_absent": True,
        "authenticated_collection_absent": True,
        "authenticated_success_receipt_absent": True,
        "complete_retained_bundle_absent": True,
        "selected_mft_workdir": workdirs[0],
    }


def _failure_paths(
    safe_plan: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    base_watch = Path(safe_plan["watcher_plan"]["path"])
    watch_plan = watcher._load_watch_plan(base_watch)
    root = Path(watch_plan["output_root"])
    slot_name = f"l{LOGICAL_ID}-t{FAILED_TASK_ID}"
    return (
        root / "slots" / slot_name / "collection.json",
        root / "slots" / slot_name / "success_receipt.json",
        root / "state.json",
    )


def _read_source_failure(
    *,
    safe_plan: Mapping[str, Any],
    source_submission: Mapping[str, Any],
    task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    stdout_reader: Callable[..., bytes] | None = None,
    stderr_reader: Callable[..., bytes] | None = None,
    remote_files_reader: Callable[..., Mapping[str, Any]] = (
        timeout12h._scheduler_remote_files
    ),
) -> dict[str, Any]:
    read_stdout = stdout_reader or (
        lambda **kwargs: timeout12h._scheduler_stream(
            stream="stdout", **kwargs
        )
    )
    read_stderr = stderr_reader or (
        lambda **kwargs: timeout12h._scheduler_stream(
            stream="stderr", **kwargs
        )
    )
    collection, receipt, state_path = _failure_paths(safe_plan)
    evidence = _source_failure_evidence(
        snapshot=task_reader(
            scheduler_url=SCHEDULER_URL, task_id=FAILED_TASK_ID
        ),
        stdout=read_stdout(
            scheduler_url=SCHEDULER_URL, task_id=FAILED_TASK_ID
        ),
        stderr=read_stderr(
            scheduler_url=SCHEDULER_URL, task_id=FAILED_TASK_ID
        ),
        source_submission=source_submission,
        retained_inventory=remote_files_reader(
            scheduler_url=SCHEDULER_URL,
            task_id=FAILED_TASK_ID,
            glob=(
                source_submission["retained_aedt_bundle"][
                    "relative_directory"
                ]
                + "/**"
            ),
        ),
        collection_exists=collection.exists(),
        success_receipt_exists=receipt.exists(),
    )
    state = _validate_seal(
        _read_json(state_path), watcher.STATE_SCHEMA
    )
    matching = [
        item
        for item in state.get("slots", [])
        if isinstance(item, Mapping)
        and item.get("logical_authority_task_id") == LOGICAL_ID
    ]
    if (
        len(matching) != 1
        or matching[0].get("execution_task_id") != FAILED_TASK_ID
        or matching[0].get("state") != "terminal_failure"
    ):
        raise StartupRetryError(
            "watcher does not currently classify task96303 as terminal_failure"
        )
    evidence["watcher_state"] = "terminal_failure"
    evidence["watcher_execution_task_id"] = FAILED_TASK_ID
    return evidence


def _source_candidate(
    safe_plan: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        dict(item)
        for item in safe_plan.get("candidates", [])
        if item.get("logical_authority_task_id") == LOGICAL_ID
    ]
    if (
        len(rows) != 1
        or rows[0].get("candidate_physics_sha256")
        != EXPECTED_CANDIDATE_SHA256
        or rows[0].get("plan_payload_sha256")
        != EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256
        or rows[0].get("prospective_grid_gb") != PROSPECTIVE_GRID_GIB
        or rows[0].get("account_name") != TARGET_ACCOUNT
        or rows[0].get("fixed_identity_attestation_sha256")
        != EXPECTED_FIXED_IDENTITY_SHA256
    ):
        raise StartupRetryError("logical96223 SAFE REFILL source drifted")
    return rows[0]


def _extension_target(
    safe_plan: Mapping[str, Any],
) -> Path:
    return (
        Path(safe_plan["watcher_extension_directory"]).resolve()
        / f"l{LOGICAL_ID}.json"
    )


def create_plan(
    *,
    output_root: Path,
    safe_refill_plan_path: Path,
    extension_authority_path: Path,
    failure_ledger_path: Path,
    code_revision: str,
    task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    stdout_reader: Callable[..., bytes] | None = None,
    stderr_reader: Callable[..., bytes] | None = None,
    remote_files_reader: Callable[..., Mapping[str, Any]] = (
        timeout12h._scheduler_remote_files
    ),
) -> Path:
    safe_plan = refill.load_plan(safe_refill_plan_path)
    authority = refill.authenticate_extension_authority(
        extension_authority_path,
        watch_plan_path=Path(safe_plan["watcher_plan"]["path"]),
    )
    candidate = _source_candidate(safe_plan)
    source_plan_path = Path(candidate["plan"]["path"])
    source_plan, params, selected, _parent = timeout12h._load_plan(
        source_plan_path
    )
    if source_plan["payload_sha256"] != EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256:
        raise StartupRetryError("source timeout12h plan payload drifted")
    old_extension_path = _extension_target(safe_plan)
    old_slot = refill.authenticate_watcher_extension_receipt(
        old_extension_path, authority=authority
    )
    source_submission_path = Path(old_slot["submission"]["path"])
    source_submission = timeout12h.load_submission_for_probe(
        source_submission_path, plan=source_plan
    )
    if (
        source_submission["payload_sha256"]
        != EXPECTED_SOURCE_SUBMISSION_PAYLOAD_SHA256
        or source_submission["task_id"] != FAILED_TASK_ID
    ):
        raise StartupRetryError("source task96303 submission receipt drifted")
    failure_ledger = _validate_seal(
        _read_json(failure_ledger_path), watcher.FAILURE_LEDGER_SCHEMA
    )
    if (
        failure_ledger.get("logical_authority_task_id") != LOGICAL_ID
        or failure_ledger.get("execution_task_id") != FAILED_TASK_ID
        or failure_ledger.get("classification")
        != "terminal_execution_not_successful"
        or failure_ledger.get("collection_performed") is not False
        or failure_ledger.get("physical_truth_claimed") is not False
    ):
        raise StartupRetryError("task96303 watcher failure ledger drifted")
    failure = _read_source_failure(
        safe_plan=safe_plan,
        source_submission=source_submission,
        task_reader=task_reader,
        stdout_reader=stdout_reader,
        stderr_reader=stderr_reader,
        remote_files_reader=remote_files_reader,
    )
    if _normalized_source_snapshot(
        failure_ledger.get("scheduler_snapshot") or {}
    ) != failure["scheduler_snapshot"]:
        raise StartupRetryError(
            "task96303 watcher and fresh Scheduler snapshots differ"
        )
    revision = str(code_revision).strip().lower()
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise StartupRetryError("startup successor code revision is invalid")
    initialize_claim_root()
    reference = _claim_reference()
    task_name, workdir = _task_identity(EXPECTED_CANDIDATE_SHA256)
    source_root = source_plan_path.resolve(strict=True).parent
    profile = production._read_json(
        source_root / source_plan["profile"]["path"]
    )
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        source_plan["solver_revision"],
        source_plan["library_revision"],
    )
    if (
        retained is None
        or retained["dedupe_key"]
        == source_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or retained["relative_directory"]
        == source_plan["stage"]["retained_aedt_bundle"][
            "relative_directory"
        ]
    ):
        raise StartupRetryError(
            "startup successor retained identity is not fresh"
        )
    destination = output_root.resolve()
    if destination.exists():
        raise StartupRetryError(
            f"startup successor output root exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        params_path = staging / "fea_params.json"
        selected_path = staging / "selected_candidate.json"
        profile_path = staging / "profile.json"
        old_extension_archive = staging / "source_l96223_t96303_extension.json"
        params_path.write_bytes(production._json_bytes(params))
        selected_path.write_bytes(production._json_bytes(selected))
        profile_path.write_bytes(production._json_bytes(profile))
        shutil.copy2(old_extension_path, old_extension_archive)
        startup_record = {
            "schema_version": STARTUP_RECORD_SCHEMA,
            "retry_generation": GENERATION,
            "logical_authority_task_id": LOGICAL_ID,
            "retry_of_task_id": FAILED_TASK_ID,
            "source_plan": _file_record(source_plan_path),
            "source_plan_payload_sha256": source_plan["payload_sha256"],
            "source_submission": _file_record(source_submission_path),
            "source_submission_payload_sha256": source_submission[
                "payload_sha256"
            ],
            "failure_ledger": _file_record(failure_ledger_path),
            "source_failure_evidence": failure,
            "source_failure_evidence_sha256": canonical_sha256(failure),
            "safe_refill_plan": _file_record(safe_refill_plan_path),
            "extension_authority": _file_record(extension_authority_path),
            "source_watcher_extension_archive": {
                "path": old_extension_archive.name,
                "sha256": production._sha256_file(old_extension_archive),
                "size_bytes": old_extension_archive.stat().st_size,
            },
            "watcher_extension_target": str(old_extension_path.resolve()),
            "previous_execution_task_id": FAILED_TASK_ID,
            "maximum_lifetime_scheduler_posts": 1,
            "scheduler_cancel_allowed": False,
            "submission_cutoff_utc": SUBMISSION_CUTOFF_UTC.isoformat(),
            "prospective_grid_gib": PROSPECTIVE_GRID_GIB,
            "submission_environment": copy.deepcopy(TEMP_ENVIRONMENT),
            "ans_mw_inherit_tmp_omitted": True,
            "submission_environment_after_workdir_selection": True,
            "required_workdir_prefix": WORKDIR_PREFIX,
            "only_operational_changes": [
                "fresh_generation_task_dedupe_retention_and_claim",
                "task_local_aedt_temporary_paths",
                "fail_closed_enroot_workdir_assertion",
            ],
            "physics_profile_solver_library_resources_unchanged": True,
            "fixed_identity_attestation_sha256": (
                EXPECTED_FIXED_IDENTITY_SHA256
            ),
            "code_revision": revision,
        }
        unsigned = copy.deepcopy(source_plan)
        unsigned.pop("payload_sha256", None)
        unsigned.update(
            {
                "fea_params": {
                    "path": params_path.name,
                    "sha256": production._sha256_file(params_path),
                },
                "selected_candidate": {
                    "path": selected_path.name,
                    "sha256": production._sha256_file(selected_path),
                },
                "profile": {
                    "path": profile_path.name,
                    "sha256": production._sha256_file(profile_path),
                    "canonical_sha256": canonical_sha256(profile),
                    "source": copy.deepcopy(source_plan["profile"]["source"]),
                },
                "stage": {
                    **copy.deepcopy(source_plan["stage"]),
                    "task_name": task_name,
                    "workdir": workdir,
                    "retained_aedt_bundle": retained,
                    "retention_run_root": (
                        diagnostic._retention_run_root_evidence(retained)
                    ),
                },
                "available_submission_commands": ["submit-startup-retry"],
                "startup_successor": startup_record,
                "startup_successor_atomic_claim_reference": reference,
                "scheduler_submission_performed": False,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
            }
        )
        plan_path = production._write_immutable_json(
            staging / "startup_successor_plan.json",
            production._seal(unsigned),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise StartupRetryError(f"{label} record is absent")
    path = production._contained_file(root, record.get("path"), label)
    if _file_record(path)["sha256"] != record.get("sha256"):
        raise StartupRetryError(f"{label} bytes drifted")
    return path


def validate_plan_overlay(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    selected: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    record = plan.get("startup_successor")
    if not isinstance(record, Mapping):
        raise StartupRetryError("startup successor overlay is absent")
    required = {
        "schema_version",
        "retry_generation",
        "logical_authority_task_id",
        "retry_of_task_id",
        "source_plan",
        "source_plan_payload_sha256",
        "source_submission",
        "source_submission_payload_sha256",
        "failure_ledger",
        "source_failure_evidence",
        "source_failure_evidence_sha256",
        "safe_refill_plan",
        "extension_authority",
        "source_watcher_extension_archive",
        "watcher_extension_target",
        "previous_execution_task_id",
        "maximum_lifetime_scheduler_posts",
        "scheduler_cancel_allowed",
        "submission_cutoff_utc",
        "prospective_grid_gib",
        "submission_environment",
        "ans_mw_inherit_tmp_omitted",
        "submission_environment_after_workdir_selection",
        "required_workdir_prefix",
        "only_operational_changes",
        "physics_profile_solver_library_resources_unchanged",
        "fixed_identity_attestation_sha256",
        "code_revision",
    }
    failure = record.get("source_failure_evidence")
    if (
        set(record) != required
        or record.get("schema_version") != STARTUP_RECORD_SCHEMA
        or record.get("retry_generation") != GENERATION
        or record.get("logical_authority_task_id") != LOGICAL_ID
        or record.get("retry_of_task_id") != FAILED_TASK_ID
        or record.get("source_plan_payload_sha256")
        != EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256
        or record.get("source_submission_payload_sha256")
        != EXPECTED_SOURCE_SUBMISSION_PAYLOAD_SHA256
        or record.get("previous_execution_task_id") != FAILED_TASK_ID
        or record.get("maximum_lifetime_scheduler_posts") != 1
        or record.get("scheduler_cancel_allowed") is not False
        or record.get("submission_cutoff_utc")
        != SUBMISSION_CUTOFF_UTC.isoformat()
        or record.get("prospective_grid_gib") != PROSPECTIVE_GRID_GIB
        or record.get("submission_environment") != TEMP_ENVIRONMENT
        or "ANS_MW_INHERIT_TMP" in record.get(
            "submission_environment", {}
        )
        or record.get("ans_mw_inherit_tmp_omitted") is not True
        or record.get("submission_environment_after_workdir_selection")
        is not True
        or record.get("required_workdir_prefix") != WORKDIR_PREFIX
        or record.get(
            "physics_profile_solver_library_resources_unchanged"
        )
        is not True
        or record.get("fixed_identity_attestation_sha256")
        != EXPECTED_FIXED_IDENTITY_SHA256
        or not isinstance(failure, Mapping)
        or record.get("source_failure_evidence_sha256")
        != canonical_sha256(failure)
        or failure.get("schema_version") != FAILURE_EVIDENCE_SCHEMA
        or failure.get("desktop_startup_failed_after_three_attempts")
        is not True
        or failure.get("result_json_absent") is not True
        or failure.get("authenticated_collection_absent") is not True
        or failure.get("watcher_state") != "terminal_failure"
    ):
        raise StartupRetryError("startup successor overlay drifted")
    root = plan_path.resolve(strict=True).parent
    source_plan_path = Path(record["source_plan"]["path"])
    if _file_record(source_plan_path) != record["source_plan"]:
        raise StartupRetryError("startup source plan bytes drifted")
    source_plan, source_params, source_selected, _parent = (
        timeout12h._load_plan(source_plan_path)
    )
    source_submission_path = Path(record["source_submission"]["path"])
    if _file_record(source_submission_path) != record["source_submission"]:
        raise StartupRetryError("startup source submission bytes drifted")
    source_submission = timeout12h.load_submission_for_probe(
        source_submission_path, plan=source_plan
    )
    safe_plan_path = Path(record["safe_refill_plan"]["path"])
    if _file_record(safe_plan_path) != record["safe_refill_plan"]:
        raise StartupRetryError("startup SAFE REFILL plan bytes drifted")
    safe_plan = refill.load_plan(safe_plan_path)
    authority_path = Path(record["extension_authority"]["path"])
    if _file_record(authority_path) != record["extension_authority"]:
        raise StartupRetryError("startup watcher authority bytes drifted")
    authority = refill.authenticate_extension_authority(
        authority_path,
        watch_plan_path=Path(safe_plan["watcher_plan"]["path"]),
    )
    old_extension = _artifact(
        root,
        record["source_watcher_extension_archive"],
        "startup archived source extension",
    )
    old_slot = refill.authenticate_watcher_extension_receipt(
        old_extension, authority=authority
    )
    failure_ledger_path = Path(record["failure_ledger"]["path"])
    if (
        _file_record(failure_ledger_path) != record["failure_ledger"]
        or _validate_seal(
            _read_json(failure_ledger_path), watcher.FAILURE_LEDGER_SCHEMA
        ).get("execution_task_id")
        != FAILED_TASK_ID
    ):
        raise StartupRetryError("startup failure ledger bytes drifted")
    task_name, workdir = _task_identity(EXPECTED_CANDIDATE_SHA256)
    expected_retained = scheduler_client.retained_aedt_identity(
        task_name,
        dict(params),
        dict(profile),
        plan["solver_revision"],
        plan["library_revision"],
    )
    if (
        source_plan["payload_sha256"]
        != EXPECTED_SOURCE_PLAN_PAYLOAD_SHA256
        or source_submission["payload_sha256"]
        != EXPECTED_SOURCE_SUBMISSION_PAYLOAD_SHA256
        or old_slot["execution_task_id"] != FAILED_TASK_ID
        or old_slot["logical_authority_task_id"] != LOGICAL_ID
        or old_slot["candidate_physics_sha256"]
        != EXPECTED_CANDIDATE_SHA256
        or source_params != params
        or source_selected != selected
        or production._read_json(
            source_plan_path.parent / source_plan["profile"]["path"]
        )
        != profile
        or any(
            plan.get(name) != source_plan.get(name)
            for name in (
                "campaign_id",
                "goal_contract_schema",
                "hard_spec",
                "hard_spec_sha256",
                "temperature_contract_sha256",
                "solver_revision",
                "library_revision",
                "candidate_physics_sha256",
                "search_authority_sha256",
                "fea_params_sha256",
                "retry_of_timeout12h",
                "scheduler_strict_node_contract",
            )
        )
        or plan["candidate_physics_sha256"]
        != EXPECTED_CANDIDATE_SHA256
        or plan["stage"]["task_name"] != task_name
        or plan["stage"]["workdir"] != workdir
        or plan["stage"]["resources"] != timeout12h.RESOURCES
        or plan["stage"]["retained_aedt_bundle"] != expected_retained
        or plan["stage"]["profile_sha256"]
        != source_plan["stage"]["profile_sha256"]
        or plan["stage"]["effective_params_sha256"]
        != source_plan["stage"]["effective_params_sha256"]
        or plan["available_submission_commands"]
        != ["submit-startup-retry"]
        or plan.get("physics_override_allowed") is not False
        or any(plan.get(name) is not expected for name, expected in _flags().items())
        or record["watcher_extension_target"]
        != str(_extension_target(safe_plan))
        or scheduler_client.LOCAL_SCRATCH_ROOT != "/enroot"
    ):
        raise StartupRetryError(
            "startup successor changes physics/profile/solver/library/"
            "resources or its fresh execution identity drifted"
        )
    _validate_claim_reference(
        plan.get("startup_successor_atomic_claim_reference")
    )


def load_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return timeout12h._load_plan(path)


def _all_project_tasks(*, scheduler_url: str) -> list[dict[str, Any]]:
    value = diagnostic._scheduler_project_tasks(
        scheduler_url=scheduler_url,
        project=PROJECT,
        task_name="mft-goal-diag-standard-",
    )
    if not isinstance(value, list):
        raise StartupRetryError("startup project task inventory is malformed")
    return [copy.deepcopy(dict(row)) for row in value]


def _successor_siblings(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stage = plan["stage"]
    name = stage["task_name"]
    dedupe = stage["retained_aedt_bundle"]["dedupe_key"]
    prefix = name.rsplit("-", 1)[0] + "-"
    matches = []
    for raw in rows:
        raw_name = str(raw.get("name") or "")
        raw_dedupe = str(raw.get("dedupe_key") or "")
        if not (
            raw_name == name
            or raw_dedupe == dedupe
            or raw_name.startswith(prefix)
        ):
            continue
        row = timeout12h._normalized_task(raw)
        if (
            row["name"] != name
            or row["dedupe_key"] != dedupe
            or row["project"] != PROJECT
            or row["cpus"] != RESOURCES["cpus"]
            or row["memory_mb"] != RESOURCES["memory_mb"]
            or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
            or row["aedt_backend"] != "standalone"
            or row["requested_node_name"] != TARGET_NODE
            or row["requested_node_name_policy"] != "strict"
            or row["same_node_as_task_id"] != 0
        ):
            raise StartupRetryError(
                "startup successor sibling identity collision"
            )
        matches.append(row)
    matches.sort(key=lambda item: item["task_id"])
    if len(matches) > 1:
        raise StartupRetryError("more than one startup successor sibling exists")
    return matches


def _claim_state(plan: Mapping[str, Any]) -> str:
    reference = _validate_claim_reference(
        plan["startup_successor_atomic_claim_reference"]
    )
    path = (
        Path(reference["claim_root_authority"]["resolved_root"])
        / reference["relative_claim_directory"]
    )
    return "claimed" if path.exists() else "unsubmitted"


def _deadline_open() -> bool:
    return _deadline_open_at(datetime.now(timezone.utc))


def _deadline_open_at(observed: datetime) -> bool:
    if observed.tzinfo is None:
        raise StartupRetryError(
            "startup successor cutoff observation lacks timezone"
        )
    return observed.astimezone(timezone.utc) < SUBMISSION_CUTOFF_UTC


def _effective_slots(
    *,
    safe_plan: Mapping[str, Any],
    authority_path: Path,
) -> dict[int, dict[str, Any]]:
    base_path = Path(safe_plan["watcher_plan"]["path"])
    effective = watcher._plan_with_authorized_extensions(
        watcher._load_watch_plan(base_path),
        watch_plan_path=base_path,
        extension_authority_path=authority_path,
    )
    return {
        int(slot["execution_task_id"]): copy.deepcopy(dict(slot))
        for slot in effective["slots"]
    }


def evaluate(
    plan: Mapping[str, Any],
    *,
    task_reader: Callable[..., list[dict[str, Any]]] = (
        bounded._all_active_task_inventory
    ),
    project_task_reader: Callable[..., list[dict[str, Any]]] = (
        _all_project_tasks
    ),
    capacity_reader: Callable[..., dict[str, Any]] = refill._capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = (
        refill._fresh_storage
    ),
    source_task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    stdout_reader: Callable[..., bytes] | None = None,
    stderr_reader: Callable[..., bytes] | None = None,
    remote_files_reader: Callable[..., Mapping[str, Any]] = (
        timeout12h._scheduler_remote_files
    ),
    validate_cutover: bool = True,
    ignore_claim: bool = False,
    allow_existing_successor: bool = False,
    deadline_reader: Callable[[], bool] = _deadline_open,
) -> dict[str, Any]:
    record = plan["startup_successor"]
    safe_plan = refill.load_plan(Path(record["safe_refill_plan"]["path"]))
    source_plan = timeout12h._load_plan(
        Path(record["source_plan"]["path"])
    )[0]
    source_submission = timeout12h.load_submission_for_probe(
        Path(record["source_submission"]["path"]), plan=source_plan
    )
    if validate_cutover:
        diagnostic._validate_scheduler_cutover_receipt(
            Path(safe_plan["cutover_receipt"]["path"]),
            verify_live_launcher=True,
            require_strict_node=True,
            strict_node_contract=plan["scheduler_strict_node_contract"],
            require_active_strict=True,
        )
    source_failure = _read_source_failure(
        safe_plan=safe_plan,
        source_submission=source_submission,
        task_reader=source_task_reader,
        stdout_reader=stdout_reader,
        stderr_reader=stderr_reader,
        remote_files_reader=remote_files_reader,
    )
    if source_failure != record["source_failure_evidence"]:
        raise StartupRetryError(
            "task96303 startup failure evidence changed before admission"
        )
    tasks = task_reader(scheduler_url=SCHEDULER_URL)
    all_project = project_task_reader(scheduler_url=SCHEDULER_URL)
    siblings = _successor_siblings(all_project, plan=plan)
    capacity = capacity_reader(
        scheduler_url=SCHEDULER_URL, account_name=TARGET_ACCOUNT
    )
    storage = storage_reader(safe_plan)
    slots = _effective_slots(
        safe_plan=safe_plan,
        authority_path=Path(record["extension_authority"]["path"]),
    )
    candidates = {
        int(item["logical_authority_task_id"]): dict(item)
        for item in safe_plan["candidates"]
    }
    active_bounds = []
    unapproved = []
    for task in tasks:
        if task.get("project") != PROJECT or not refill._is_fea(task):
            continue
        execution = _task_id(task)
        slot = slots.get(execution or -1)
        if (
            slot is None
            and allow_existing_successor
            and task.get("name") == plan["stage"]["task_name"]
            and task.get("dedupe_key")
            == plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        ):
            slot = {
                "logical_authority_task_id": LOGICAL_ID,
                "execution_task_id": execution,
                "task_name": plan["stage"]["task_name"],
                "dedupe_key": plan["stage"]["retained_aedt_bundle"][
                    "dedupe_key"
                ],
                "candidate_physics_sha256": (
                    EXPECTED_CANDIDATE_SHA256
                ),
            }
        if slot is None:
            unapproved.append(execution or -1)
            continue
        logical = int(slot["logical_authority_task_id"])
        candidate = candidates.get(logical)
        if (
            candidate is None
            or task.get("name") != slot["task_name"]
            or task.get("dedupe_key") != slot["dedupe_key"]
            or task.get("account_name") != TARGET_ACCOUNT
            or task.get("requested_account_name") != TARGET_ACCOUNT
            or task.get("cpus") != RESOURCES["cpus"]
            or task.get("memory_mb") != RESOURCES["memory_mb"]
            or task.get("timeout_seconds") != RESOURCES["timeout_seconds"]
            or task.get("requested_node_name") != TARGET_NODE
            or task.get("requested_node_name_policy") != "strict"
            or slot["candidate_physics_sha256"]
            != candidate["candidate_physics_sha256"]
        ):
            raise StartupRetryError(
                "active MFT FEA task escaped authenticated r1/n114 bounds"
            )
        active_bounds.append(
            {
                "logical_authority_task_id": logical,
                "execution_task_id": execution,
                "prospective_grid_gib": candidate["prospective_grid_gb"],
            }
        )
    active_bound = sum(
        float(item["prospective_grid_gib"]) for item in active_bounds
    )
    limiting = storage["limiting_quota"]
    effective_free = (
        float(limiting["raw_free_gb"])
        - float(limiting["in_doubt_gb"])
    )
    prospective_charge = (
        0.0 if allow_existing_successor and siblings else PROSPECTIVE_GRID_GIB
    )
    remaining = (
        effective_free
        - active_bound
        - prospective_charge
        - refill.SAFETY_FLOOR_GB
    )
    allocations = capacity.get("allocations")
    allocation_rows = allocations if isinstance(allocations, list) else []
    placement = bool(allocation_rows) and all(
        row.get("account_name") == TARGET_ACCOUNT
        and row.get("node_name") == TARGET_NODE
        and row.get("state") in {"warm", "active"}
        and int(row.get("fit_slots") or 0) > 0
        for row in allocation_rows
    )
    available = capacity.get("standalone_aedt_available")
    license_ready = (
        not isinstance(available, bool)
        and isinstance(available, int)
        and available > 0
    )
    claim = _claim_state(plan)
    extension_path = Path(record["watcher_extension_target"])
    current_extension = _read_json(extension_path)
    source_extension_present = (
        current_extension.get("schema_version")
        == refill.EXTENSION_RECEIPT_SCHEMA
    )
    gates = {
        "exact_task96303_startup_failure_reauthenticated": True,
        "watcher_currently_terminal_failure": (
            source_failure["watcher_state"] == "terminal_failure"
        ),
        "result_json_collection_absent": (
            source_failure["result_json_absent"]
            and source_failure["authenticated_collection_absent"]
        ),
        "active_4fac_cutover_reauthenticated": True,
        "all_active_mft_fea_tasks_approved_and_bounded": not unapproved,
        "scheduler_ready_fit_slot_positive": (
            int(capacity.get("ready_fit_slots") or 0) > 0
        ),
        "strict_r1_n114_placement_available": placement,
        "standalone_aedt_license_available": license_ready,
        "cumulative_active_plus_prospective_storage_floor_passed": (
            remaining >= 0
        ),
        "successor_sibling_gate": (
            len(siblings) <= 1
            if allow_existing_successor
            else len(siblings) == 0
        ),
        "fresh_atomic_claim": claim == "unsubmitted" or ignore_claim,
        "source_watcher_extension_still_authoritative": (
            source_extension_present
        ),
        "submission_cutoff_open": deadline_reader(),
        "exact_temp_environment_and_enroot_contract": (
            record["submission_environment"] == TEMP_ENVIRONMENT
            and record["required_workdir_prefix"] == WORKDIR_PREFIX
            and "ANS_MW_INHERIT_TMP"
            not in record["submission_environment"]
        ),
        "fixed_physics_profile_solver_library_resources_preserved": True,
    }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "observed_at_utc": _now(),
        "eligible": all(gates.values()),
        "gates": gates,
        "active_tasks_sha256": canonical_sha256(tasks),
        "active_authenticated_bounds": active_bounds,
        "unapproved_active_mft_fea_task_ids": sorted(unapproved),
        "active_authenticated_storage_bound_gib": active_bound,
        "prospective_grid_gib": PROSPECTIVE_GRID_GIB,
        "prospective_storage_charge_gib": prospective_charge,
        "effective_free_gib": effective_free,
        "remaining_after_active_prospective_floor_gib": remaining,
        "capacity": copy.deepcopy(capacity),
        "capacity_sha256": canonical_sha256(capacity),
        "storage": copy.deepcopy(storage),
        "storage_sha256": canonical_sha256(storage),
        "successor_sibling_task_ids": [
            item["task_id"] for item in siblings
        ],
        "claim_state": claim,
        "submission_cutoff_utc": SUBMISSION_CUTOFF_UTC.isoformat(),
        "maximum_lifetime_scheduler_posts": 1,
        "scheduler_cancel_allowed": False,
        "all_scheduler_reads_get_only": True,
    }


def _claim_winner(
    plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "immediate_task_id": FAILED_TASK_ID,
        "immediate_retry_kind": "timeout",
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": production._sha256_file(plan_path),
        "profile_sha256": plan["stage"]["profile_sha256"],
        "resources": copy.deepcopy(RESOURCES),
        "task_name": plan["stage"]["task_name"],
        "dedupe_key": plan["stage"]["retained_aedt_bundle"][
            "dedupe_key"
        ],
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    row = timeout12h._normalized_task(task)
    winner = pending.get("winner")
    if (
        not isinstance(winner, Mapping)
        or row["task_id"] is None
        or row["task_id"] in {LOGICAL_ID, FAILED_TASK_ID}
        or row["name"] != winner.get("task_name")
        or row["dedupe_key"] != winner.get("dedupe_key")
        or row["project"] != PROJECT
        or row["cpus"] != RESOURCES["cpus"]
        or row["memory_mb"] != RESOURCES["memory_mb"]
        or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
        or row["aedt_backend"] != "standalone"
        or row["requested_node_name"] != TARGET_NODE
        or row["requested_node_name_policy"] != "strict"
        or row["same_node_as_task_id"] != 0
        or row["account_name"] != TARGET_ACCOUNT
        or row["requested_account_name"] != TARGET_ACCOUNT
        or not str(row["status"] or "")
        or not str(row["state"] or "")
    ):
        raise atomic_claim.ClaimContractError(
            "startup successor claim task identity drifted"
        )
    return row


def _sibling_receipt(
    *,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    task_id: int,
) -> dict[str, Any]:
    if (
        len(before) not in {0, 1}
        or len(after) != 1
        or after[0].get("task_id") != task_id
        or (
            len(before) == 1
            and before[0].get("task_id") != task_id
        )
    ):
        raise StartupRetryError(
            "startup successor exactly-one sibling invariant failed"
        )
    return {
        "schema_version": SIBLING_SCHEMA,
        "before_submission": copy.deepcopy(list(before)),
        "after_submission": copy.deepcopy(list(after)),
        "exactly_one_successor_after_submission": True,
    }


def _claim_receipt(
    *, status: str, finalized: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": CLAIM_RECEIPT_SCHEMA,
        "acquisition_status": status,
        "fresh_claim_authorized_scheduler_post": status == "fresh_pending",
        "recovered_without_scheduler_post": status != "fresh_pending",
        "finalized_claim": copy.deepcopy(dict(finalized)),
    }


def _validate_claim_receipt(
    value: Any, *, plan_path: Path, plan: Mapping[str, Any], task_id: int
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != CLAIM_RECEIPT_SCHEMA
        or value.get("acquisition_status")
        not in {"fresh_pending", "existing_pending", "existing_finalized"}
    ):
        raise StartupRetryError("startup claim receipt is malformed")
    status = value["acquisition_status"]
    if (
        value.get("fresh_claim_authorized_scheduler_post")
        is not (status == "fresh_pending")
        or value.get("recovered_without_scheduler_post")
        is not (status != "fresh_pending")
    ):
        raise StartupRetryError("startup claim receipt semantics drifted")
    try:
        finalized = atomic_claim.validate_finalized_claim(
            CLAIM_ROOT,
            _validate_claim_reference(
                plan["startup_successor_atomic_claim_reference"]
            ),
            claim=value.get("finalized_claim"),
            expected_winner=_claim_winner(plan_path, plan),
        )
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError("startup finalized claim drifted") from exc
    if finalized.get("task_id") != task_id:
        raise StartupRetryError("startup finalized claim task drifted")
    return copy.deepcopy(dict(value))


def _build_submission_receipt(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    task_id: int,
    reauthentication: Mapping[str, Any],
    core_policy: Mapping[str, Any],
    sibling_receipt: Mapping[str, Any],
    claim_receipt: Mapping[str, Any],
    submission_result: Mapping[str, Any],
    durable: Mapping[str, Any],
    cutover_path: Path,
    cutover: Mapping[str, Any],
    launcher: Mapping[str, Any],
    admission: Mapping[str, Any],
    fresh_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    stage = plan["stage"]
    strict_trace = {
        "plan_contract": copy.deepcopy(
            plan["scheduler_strict_node_contract"]
        ),
        "submission_source": submission_result["submission_source"],
        "scheduler_mutation_performed": submission_result[
            "scheduler_mutation_performed"
        ],
        "api_pre_submission_readback": copy.deepcopy(
            submission_result["api_pre_submission_readback"]
        ),
        "api_post_submission_response": copy.deepcopy(
            submission_result["api_post_submission_response"]
        ),
        "api_durable_get_readback": copy.deepcopy(dict(durable)),
        "direct_strict_node_policy_authenticated": True,
        "same_node_as_task_id": 0,
    }
    return production._seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "stage": "standard",
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "search_authority_reauthentication": copy.deepcopy(
                dict(reauthentication)
            ),
            "task_id": task_id,
            "task_name": stage["task_name"],
            "workdir": stage["workdir"],
            "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "profile_sha256": stage["profile_sha256"],
            "effective_params_sha256": stage[
                "effective_params_sha256"
            ],
            "resources": copy.deepcopy(timeout12h.RESOURCES),
            "aedt_backend": "standalone",
            "core_policy": copy.deepcopy(dict(core_policy)),
            "retained_aedt_bundle": copy.deepcopy(
                stage["retained_aedt_bundle"]
            ),
            "retention_run_root": copy.deepcopy(
                stage["retention_run_root"]
            ),
            "retry_of_timeout12h": copy.deepcopy(
                plan["retry_of_timeout12h"]
            ),
            "startup_successor": copy.deepcopy(
                plan["startup_successor"]
            ),
            "startup_successor_sibling_guard": copy.deepcopy(
                dict(sibling_receipt)
            ),
            "startup_successor_atomic_claim": copy.deepcopy(
                dict(claim_receipt)
            ),
            "startup_submission_environment": copy.deepcopy(
                TEMP_ENVIRONMENT
            ),
            "startup_workdir_contract": {
                "submission_environment_after_workdir_selection": True,
                "required_workdir_prefix": WORKDIR_PREFIX,
                "fail_closed_exit_code": 86,
                "ans_mw_inherit_tmp_omitted": True,
            },
            "fresh_gate_evaluation": copy.deepcopy(
                dict(fresh_evaluation)
            ),
            "scheduler_strict_node_contract": strict_trace,
            "scheduler_cutover_receipt": _file_record(cutover_path),
            "scheduler_cutover_payload_sha256": cutover[
                "payload_sha256"
            ],
            "scheduler_live_launcher_identity": copy.deepcopy(
                dict(launcher)
            ),
            "scheduler_admission_snapshot": copy.deepcopy(
                dict(admission)
            ),
            "scheduler_url": SCHEDULER_URL,
            "scheduler_project": PROJECT,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": True,
            "retention_required": True,
            "prune_protection_required": True,
            "fresh_pre_submit_storage_reauthentication_required": True,
            **_flags(),
        }
    )


def load_submission_for_probe(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    receipt = production._validate_seal(
        production._read_json(resolved), SUBMISSION_SCHEMA
    )
    expected = {
        "schema_version",
        "stage",
        "plan",
        "plan_payload_sha256",
        "candidate_physics_sha256",
        "search_authority_reauthentication",
        "task_id",
        "task_name",
        "workdir",
        "dedupe_key",
        "solver_revision",
        "library_revision",
        "profile_sha256",
        "effective_params_sha256",
        "resources",
        "aedt_backend",
        "core_policy",
        "retained_aedt_bundle",
        "retention_run_root",
        "retry_of_timeout12h",
        "startup_successor",
        "startup_successor_sibling_guard",
        "startup_successor_atomic_claim",
        "startup_submission_environment",
        "startup_workdir_contract",
        "fresh_gate_evaluation",
        "scheduler_strict_node_contract",
        "scheduler_cutover_receipt",
        "scheduler_cutover_payload_sha256",
        "scheduler_live_launcher_identity",
        "scheduler_admission_snapshot",
        "scheduler_url",
        "scheduler_project",
        "scheduler_project_mutation_performed",
        "scheduler_repository_modified",
        "scheduler_submission_performed",
        "retention_required",
        "prune_protection_required",
        "fresh_pre_submit_storage_reauthentication_required",
        *_flags(),
        "payload_sha256",
    }
    if set(receipt) != expected:
        raise StartupRetryError("startup submission fields drifted")
    if _file_record(Path(receipt["plan"]["path"])) != receipt["plan"]:
        raise StartupRetryError("startup submission plan bytes drifted")
    task_id = _task_id(receipt)
    sibling = receipt.get("startup_successor_sibling_guard")
    strict = receipt.get("scheduler_strict_node_contract")
    reauth = receipt.get("search_authority_reauthentication")
    core = receipt.get("core_policy")
    workdir = receipt.get("startup_workdir_contract")
    if (
        task_id is None
        or not isinstance(sibling, Mapping)
        or sibling.get("schema_version") != SIBLING_SCHEMA
        or sibling.get("exactly_one_successor_after_submission") is not True
        or len(sibling.get("after_submission", [])) != 1
        or sibling["after_submission"][0].get("task_id") != task_id
        or not isinstance(strict, Mapping)
        or strict.get("plan_contract")
        != plan["scheduler_strict_node_contract"]
        or strict.get("direct_strict_node_policy_authenticated") is not True
        or strict.get("same_node_as_task_id") != 0
        or strict.get("api_durable_get_readback", {}).get("task_id")
        != task_id
        or strict.get("api_durable_get_readback", {}).get(
            "requested_node_name"
        )
        != TARGET_NODE
        or strict.get("api_durable_get_readback", {}).get(
            "requested_node_name_policy"
        )
        != "strict"
        or not isinstance(reauth, Mapping)
        or reauth.get("fresh_candidate_reauthenticated") is not True
        or reauth.get("search_authority_sha256")
        != plan["search_authority_sha256"]
        or not isinstance(core, Mapping)
        or core.get("contract") != production.STANDARD_CORE_CONTRACT
        or core.get("requested_num_cores") != 8
        or receipt.get("startup_submission_environment")
        != TEMP_ENVIRONMENT
        or "ANS_MW_INHERIT_TMP"
        in receipt.get("startup_submission_environment", {})
        or workdir
        != {
            "submission_environment_after_workdir_selection": True,
            "required_workdir_prefix": WORKDIR_PREFIX,
            "fail_closed_exit_code": 86,
            "ans_mw_inherit_tmp_omitted": True,
        }
        or receipt.get("stage") != "standard"
        or receipt.get("plan_payload_sha256")
        != plan["payload_sha256"]
        or receipt.get("candidate_physics_sha256")
        != EXPECTED_CANDIDATE_SHA256
        or receipt.get("task_name") != plan["stage"]["task_name"]
        or receipt.get("workdir") != plan["stage"]["workdir"]
        or receipt.get("dedupe_key")
        != plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or receipt.get("solver_revision") != plan["solver_revision"]
        or receipt.get("library_revision") != plan["library_revision"]
        or receipt.get("profile_sha256")
        != plan["stage"]["profile_sha256"]
        or receipt.get("effective_params_sha256")
        != plan["stage"]["effective_params_sha256"]
        or receipt.get("resources") != timeout12h.RESOURCES
        or receipt.get("aedt_backend") != "standalone"
        or receipt.get("retained_aedt_bundle")
        != plan["stage"]["retained_aedt_bundle"]
        or receipt.get("retention_run_root")
        != plan["stage"]["retention_run_root"]
        or receipt.get("retry_of_timeout12h")
        != plan["retry_of_timeout12h"]
        or receipt.get("startup_successor")
        != plan["startup_successor"]
        or receipt.get("scheduler_url") != SCHEDULER_URL
        or receipt.get("scheduler_project") != PROJECT
        or receipt.get("scheduler_submission_performed") is not True
        or receipt.get("scheduler_project_mutation_performed") is not False
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("retention_required") is not True
        or receipt.get("prune_protection_required") is not True
        or any(
            receipt.get(name) is not value
            for name, value in _flags().items()
        )
    ):
        raise StartupRetryError("startup submission identity drifted")
    _validate_claim_receipt(
        receipt["startup_successor_atomic_claim"],
        plan_path=Path(receipt["plan"]["path"]),
        plan=plan,
        task_id=task_id,
    )
    return receipt


def _replacement_receipt(
    *,
    plan_path: Path,
    submission_path: Path,
) -> dict[str, Any]:
    plan, _params, _selected, _parent = load_plan(plan_path)
    load_submission_for_probe(submission_path, plan=plan)
    record = plan["startup_successor"]
    safe_plan = refill.load_plan(Path(record["safe_refill_plan"]["path"]))
    authority_path = Path(record["extension_authority"]["path"])
    authority = refill.authenticate_extension_authority(
        authority_path,
        watch_plan_path=Path(safe_plan["watcher_plan"]["path"]),
    )
    archive = _artifact(
        plan_path.parent,
        record["source_watcher_extension_archive"],
        "startup source watcher extension archive",
    )
    previous = refill.authenticate_watcher_extension_receipt(
        archive, authority=authority
    )
    successor = watcher._load_slot_source(
        LOGICAL_ID,
        submission_path,
        scheduler_url=SCHEDULER_URL,
        project=PROJECT,
    )
    if (
        previous["execution_task_id"] != FAILED_TASK_ID
        or successor["execution_task_id"] == FAILED_TASK_ID
        or successor["candidate_physics_sha256"]
        != previous["candidate_physics_sha256"]
        or successor["fixed_identity_attestation_sha256"]
        != previous["fixed_identity_attestation_sha256"]
    ):
        raise StartupRetryError("watcher replacement slot identity drifted")
    return _sealed(
        {
            "schema_version": REPLACEMENT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "extension_authority": _file_record(authority_path),
            "startup_plan": _file_record(plan_path),
            "startup_submission": _file_record(submission_path),
            "source_extension_archive": _file_record(archive),
            "logical_authority_task_id": LOGICAL_ID,
            "previous_execution_task_id": FAILED_TASK_ID,
            "successor_execution_task_id": successor[
                "execution_task_id"
            ],
            "candidate_physics_sha256": successor[
                "candidate_physics_sha256"
            ],
            "fixed_identity_attestation_sha256": successor[
                "fixed_identity_attestation_sha256"
            ],
            "replacement_after_authenticated_submission_only": True,
            "watcher_scheduler_methods": ["GET"],
            "watcher_scheduler_mutation_performed": False,
            "producer_scheduler_submission_performed": True,
            "authorized_at_utc": _now(),
        }
    )


def authenticate_watcher_replacement_receipt(
    path: Path, *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), REPLACEMENT_SCHEMA)
    authority_path = Path(value["extension_authority"]["path"])
    if (
        _file_record(authority_path) != value["extension_authority"]
        or refill.authenticate_extension_authority(authority_path)
        != authority
    ):
        raise StartupRetryError("watcher replacement authority drifted")
    plan_path = Path(value["startup_plan"]["path"])
    submission_path = Path(value["startup_submission"]["path"])
    if (
        _file_record(plan_path) != value["startup_plan"]
        or _file_record(submission_path) != value["startup_submission"]
    ):
        raise StartupRetryError("watcher replacement source bytes drifted")
    expected = _replacement_receipt(
        plan_path=plan_path, submission_path=submission_path
    )
    stable = (
        "extension_authority",
        "startup_plan",
        "startup_submission",
        "source_extension_archive",
        "logical_authority_task_id",
        "previous_execution_task_id",
        "successor_execution_task_id",
        "candidate_physics_sha256",
        "fixed_identity_attestation_sha256",
        "replacement_after_authenticated_submission_only",
        "watcher_scheduler_methods",
        "watcher_scheduler_mutation_performed",
        "producer_scheduler_submission_performed",
    )
    if any(value.get(name) != expected.get(name) for name in stable):
        raise StartupRetryError("watcher replacement receipt drifted")
    plan = load_plan(plan_path)[0]
    load_submission_for_probe(submission_path, plan=plan)
    return watcher._load_slot_source(
        LOGICAL_ID,
        submission_path,
        scheduler_url=SCHEDULER_URL,
        project=PROJECT,
    )


def _ensure_replacement(
    *, plan_path: Path, submission_path: Path
) -> Path:
    plan = load_plan(plan_path)[0]
    record = plan["startup_successor"]
    target = Path(record["watcher_extension_target"])
    authority = refill.authenticate_extension_authority(
        Path(record["extension_authority"]["path"])
    )
    current = _read_json(target)
    if current.get("schema_version") == REPLACEMENT_SCHEMA:
        authenticate_watcher_replacement_receipt(
            target, authority=authority
        )
        return target
    archive = _artifact(
        plan_path.parent,
        record["source_watcher_extension_archive"],
        "startup archived extension",
    )
    if (
        current.get("schema_version") != refill.EXTENSION_RECEIPT_SCHEMA
        or production._sha256_file(target)
        != production._sha256_file(archive)
        or target.stat().st_size != archive.stat().st_size
    ):
        raise StartupRetryError(
            "watcher extension changed before atomic replacement"
        )
    replacement = _replacement_receipt(
        plan_path=plan_path, submission_path=submission_path
    )
    _write_atomic(target, replacement)
    authenticate_watcher_replacement_receipt(
        target, authority=authority
    )
    return target


def submit(
    *,
    plan_path: Path,
    output: Path,
    scheduler: Any = scheduler_client,
    task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    project_task_reader: Callable[..., list[dict[str, Any]]] = (
        _all_project_tasks
    ),
    live_reader: Any = diagnostic._default_scheduler_live_reader,
    evaluator: Callable[..., dict[str, Any]] = evaluate,
    reconciliation_waiter: Callable[[], None] | None = None,
) -> tuple[Path, int]:
    target = output.resolve()
    if target.exists():
        raise StartupRetryError(
            f"startup successor submission receipt exists: {target}"
        )
    plan, params, selected, _parent = load_plan(plan_path)
    record = plan["startup_successor"]
    recovering = _claim_state(plan) == "claimed"
    initial = evaluator(
        plan,
        ignore_claim=recovering,
        allow_existing_successor=recovering,
    )
    if initial.get("eligible") is not True:
        raise StartupRetryError("startup successor admission gates are closed")
    reauthentication = diagnostic._fresh_selection_reauthentication(
        plan=plan, selected=selected
    )
    cutover_path = Path(
        refill.load_plan(Path(record["safe_refill_plan"]["path"]))[
            "cutover_receipt"
        ]["path"]
    )
    cutover, launcher_before = (
        diagnostic._validate_scheduler_cutover_receipt(
            cutover_path,
            verify_live_launcher=True,
            require_strict_node=True,
            strict_node_contract=plan["scheduler_strict_node_contract"],
            require_active_strict=True,
        )
    )
    admission = diagnostic._live_scheduler_admission_snapshot(
        scheduler_url=SCHEDULER_URL, reader=live_reader
    )
    strict_pin = diagnostic._strict_node_scheduler_pin(
        plan["scheduler_strict_node_contract"], require_active=True
    )
    launcher_after = diagnostic._live_launcher_identity(
        cutover, expected_sha256=strict_pin["launcher_sha256"]
    )
    if launcher_after != launcher_before:
        raise StartupRetryError(
            "Scheduler launcher changed during startup admission"
        )
    reference = _validate_claim_reference(
        plan["startup_successor_atomic_claim_reference"]
    )
    winner = _claim_winner(plan_path, plan)
    try:
        acquisition = atomic_claim.acquire_claim(
            CLAIM_ROOT, reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise StartupRetryError(
            "startup successor atomic claim acquisition failed"
        ) from exc
    status = acquisition["status"]
    before = _successor_siblings(
        project_task_reader(scheduler_url=SCHEDULER_URL), plan=plan
    )
    finalized: Mapping[str, Any]
    post_calls = 0
    if status == "fresh_pending":
        if before:
            raise StartupRetryError(
                "fresh startup claim found an existing successor"
            )
        locked_count = 0

        def guard() -> None:
            nonlocal locked_count
            latest = evaluator(plan, ignore_claim=True)
            if latest.get("eligible") is not True:
                raise StartupRetryError(
                    "immediate pre-POST startup gates failed"
                )
            if _successor_siblings(
                project_task_reader(scheduler_url=SCHEDULER_URL),
                plan=plan,
            ):
                raise StartupRetryError(
                    "startup sibling appeared under locked guard"
                )
            locked_count += 1

        profile = production._read_json(
            plan_path.resolve(strict=True).parent / plan["profile"]["path"]
        )
        environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
        environment.update(TEMP_ENVIRONMENT)
        environment.pop("ANS_MW_INHERIT_TMP", None)
        submission_result = scheduler.submit_verification(
            plan["stage"]["task_name"],
            plan["stage"]["workdir"],
            params,
            profile,
            mem_mb=RESOURCES["memory_mb"],
            cpus=RESOURCES["cpus"],
            solver_revision=plan["solver_revision"],
            library_revision=plan["library_revision"],
            priority=100,
            aedt_backend="standalone",
            submission_env=environment,
            submission_env_after_workdir=True,
            required_workdir_prefix=WORKDIR_PREFIX,
            required_project_cap=diagnostic.GOAL_FEA_PROJECT_CAP,
            max_project_active_tasks=diagnostic.GOAL_FEA_PROJECT_CAP,
            scheduler_url=SCHEDULER_URL,
            pre_submit_guard=guard,
            account_name=TARGET_ACCOUNT,
            node_name=TARGET_NODE,
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        if (
            locked_count != 1
            or not isinstance(submission_result, Mapping)
            or submission_result.get("scheduler_mutation_performed")
            is not True
        ):
            raise StartupRetryError(
                "fresh startup claim lacks exactly one authenticated POST"
            )
        post_calls = 1
        waiter = reconciliation_waiter or (lambda: time.sleep(0.25))
        after = []
        for index in range(MAX_RECONCILIATION_READS):
            after = _successor_siblings(
                project_task_reader(scheduler_url=SCHEDULER_URL),
                plan=plan,
            )
            if len(after) == 1:
                break
            if index + 1 < MAX_RECONCILIATION_READS:
                waiter()
        if len(after) != 1:
            raise StartupRetryError(
                "startup POST did not become a unique durable sibling"
            )
        new_task_id = int(after[0]["task_id"])
        try:
            finalized = atomic_claim.finalize_claim(
                CLAIM_ROOT,
                reference,
                acquisition["claim"],
                task_id=new_task_id,
                task_readback=after[0],
                sibling_snapshot={
                    "schema_version": SIBLING_SCHEMA,
                    "matching_task_count": 1,
                    "matching_tasks": copy.deepcopy(after),
                },
                evidence_validator=_claim_task_evidence,
            )
        except atomic_claim.ClaimContractError as exc:
            raise StartupRetryError(
                "startup successor atomic claim finalization failed"
            ) from exc
    else:
        if len(before) != 1:
            raise StartupRetryError(
                "startup claim recovery requires exactly one successor; "
                "it never re-POSTs"
            )
        new_task_id = int(before[0]["task_id"])
        try:
            if status == "existing_pending":
                finalized = atomic_claim.recover_pending_claim(
                    CLAIM_ROOT,
                    reference,
                    acquisition["claim"],
                    matching_tasks=before,
                    sibling_snapshot={
                        "schema_version": SIBLING_SCHEMA,
                        "matching_task_count": 1,
                        "matching_tasks": copy.deepcopy(before),
                    },
                    evidence_validator=_claim_task_evidence,
                )
            else:
                finalized = atomic_claim.validate_finalized_claim(
                    CLAIM_ROOT,
                    reference,
                    claim=acquisition["claim"],
                    expected_winner=winner,
                )
                _claim_task_evidence(
                    before[0], finalized["pending_claim"]
                )
        except atomic_claim.ClaimContractError as exc:
            raise StartupRetryError(
                "startup successor claim recovery failed closed"
            ) from exc
        after = copy.deepcopy(before)
        submission_result = {
            "task_id": new_task_id,
            "submission_source": "pre_submission_reconciliation",
            "scheduler_mutation_performed": False,
            "api_pre_submission_readback": copy.deepcopy(before[0]),
            "api_post_submission_response": None,
        }
        _environment, core_policy = production._submission_environment(
            stage="standard",
            solver_revision=plan["solver_revision"],
            license_snapshot_path=None,
        )
    durable = task_reader(
        scheduler_url=SCHEDULER_URL, task_id=new_task_id
    )
    _claim_task_evidence(durable, finalized["pending_claim"])
    sibling_receipt = _sibling_receipt(
        before=[] if status == "fresh_pending" else before,
        after=after,
        task_id=new_task_id,
    )
    receipt = _build_submission_receipt(
        plan_path=plan_path,
        plan=plan,
        task_id=new_task_id,
        reauthentication=reauthentication,
        core_policy=core_policy,
        sibling_receipt=sibling_receipt,
        claim_receipt=_claim_receipt(status=status, finalized=finalized),
        submission_result=submission_result,
        durable=durable,
        cutover_path=cutover_path,
        cutover=cutover,
        launcher=launcher_after,
        admission=admission,
        fresh_evaluation=initial,
    )
    result = _write_immutable(target, receipt)
    load_submission_for_probe(result, plan=plan)
    return result, post_calls


def cycle(
    *,
    plan_path: Path,
    authorize_submit: bool = False,
    evaluator: Callable[..., dict[str, Any]] = evaluate,
    submitter: Callable[..., tuple[Path, int]] = submit,
) -> dict[str, Any]:
    plan = load_plan(plan_path)[0]
    root = plan_path.resolve(strict=True).parent
    submission_path = root / "submission.json"
    with watcher.SingleInstanceLock(root / "startup_retry.lock"):
        post_calls = 0
        replacement: Path | None = None
        if submission_path.exists():
            load_submission_for_probe(submission_path, plan=plan)
            replacement = _ensure_replacement(
                plan_path=plan_path, submission_path=submission_path
            )
            action = "submission_recovered_and_watcher_replaced"
            evaluation = None
        else:
            evaluation = evaluator(plan)
            recovering = evaluation.get("claim_state") == "claimed"
            if recovering:
                evaluation = evaluator(
                    plan,
                    ignore_claim=True,
                    allow_existing_successor=True,
                )
            action = (
                (
                    "recovery_pending_authorization"
                    if recovering
                    else "dry_run_would_submit"
                )
                if evaluation.get("eligible")
                else "blocked"
            )
            if authorize_submit and evaluation.get("eligible"):
                submission_path, post_calls = submitter(
                    plan_path=plan_path, output=submission_path
                )
                replacement = _ensure_replacement(
                    plan_path=plan_path, submission_path=submission_path
                )
                action = "submitted_and_watcher_replaced"
        state = _sealed(
            {
                "schema_version": STATE_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "observed_at_utc": _now(),
                "mode": "live_authorized" if authorize_submit else "dry_run",
                "action": action,
                "scheduler_post_calls_this_cycle": post_calls,
                "scheduler_cancel_calls_this_cycle": 0,
                "maximum_lifetime_scheduler_posts": 1,
                "evaluation": evaluation,
                "submission": (
                    _file_record(submission_path)
                    if submission_path.exists()
                    else None
                ),
                "watcher_replacement": (
                    _file_record(replacement)
                    if replacement is not None
                    else None
                ),
            }
        )
        _write_atomic(root / "state.json", state)
        return state


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={REPOSITORY_ROOT}", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed exact-once logical96223/task96303 AEDT-startup "
            "infrastructure successor"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-claim-root")
    init = commands.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--safe-refill-plan", type=Path, required=True)
    init.add_argument("--extension-authority", type=Path, required=True)
    init.add_argument("--failure-ledger", type=Path, required=True)
    init.add_argument("--code-revision", default="")
    run = commands.add_parser("cycle")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument(
        "--authorize-submit",
        action="store_true",
        help="allow the one lifetime guarded POST; absent is GET-only",
    )
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-claim-root":
        result: Any = initialize_claim_root()
    elif args.command == "init":
        result = create_plan(
            output_root=args.output_root,
            safe_refill_plan_path=args.safe_refill_plan,
            extension_authority_path=args.extension_authority,
            failure_ledger_path=args.failure_ledger,
            code_revision=args.code_revision or _git_revision(),
        )
    elif args.command == "validate":
        load_plan(args.plan)
        result = args.plan.resolve(strict=True)
    else:
        result = cycle(
            plan_path=args.plan, authorize_submit=args.authorize_submit
        )
    if isinstance(result, Mapping):
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
