"""GET-only Full-success collection and final-package orchestrator.

This module begins where ``mft_goal_full_fastlane_watcher`` stops.  It
consumes that watcher's immutable receipt through a deliberately small file
schema, proves that exactly one matching Full task exists, waits for strict
terminal success, and delegates all result authentication to
``mft_goal_truth_promotion.collect_full``.

The final package is created only after a same-volume local-disk admission
based on the exact retained Standard/Full receipt byte totals.  The required
free space is the retained byte total plus ten percent plus five GiB.  Final
packaging delegates to ``mft_goal_truth_promotion.package_results`` and a
success receipt is emitted only after a second complete package
authentication and an independent hash inventory of every package file.

No Scheduler POST, PATCH, DELETE, or cancellation operation exists here.
The only Scheduler-facing operations are GETs performed by the strict
collector and the task readers.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_terminal_success_watcher as terminal  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


CAMPAIGN_ID = "mft-goal-20260726"
FAST_LANE_RECEIPT_SCHEMA = (
    "mft-goal-first-truth-full-fast-lane-receipt-v1"
)
WATCH_PLAN_SCHEMA = "mft-goal-full-postsuccess-watch-plan-v1"
DISK_ADMISSION_SCHEMA = "mft-goal-full-postsuccess-disk-admission-v1"
PACKAGE_INTENT_SCHEMA = "mft-goal-full-postsuccess-package-intent-v1"
PACKAGE_REHASH_SCHEMA = "mft-goal-full-postsuccess-package-rehash-v1"
SUCCESS_RECEIPT_SCHEMA = "mft-goal-full-postsuccess-success-receipt-v1"
STATE_SCHEMA = "mft-goal-full-postsuccess-state-v1"
PID_SCHEMA = "mft-goal-full-postsuccess-pid-v1"

DEFAULT_SCHEDULER_URL = diagnostic.DIAGNOSTIC_SCHEDULER_URL
DEFAULT_FINAL_OUTPUT_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "final_deliverable_20260726"
)
FULL_RESOURCES = {
    "cpus": 16,
    "memory_mb": 98304,
    "timeout_seconds": 43200,
}
FIXED_BOUNDARY = {
    "fan_velocity_m_s": 1.5,
    "fan_config": "dual",
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
}
GIB = 1024**3
LOCAL_FIXED_RESERVE_BYTES = 5 * GIB
LOCAL_PERCENT_NUMERATOR = 10
LOCAL_PERCENT_DENOMINATOR = 100
MIN_POLL_SECONDS = 10
MAX_POLL_SECONDS = 60
DEFAULT_POLL_SECONDS = 15
ACTIVE_STATUSES = frozenset(
    {"queued", "pending", "submitted", "attaching", "starting", "running"}
)
FAILED_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "timed_out", "timeout"}
)
COMPLETED_STATUSES = frozenset({"completed"})

HandoffContractError = production.HandoffContractError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _aware(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffContractError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise HandoffContractError(f"{label} timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise HandoffContractError("post-success payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(
    value: Any,
    schema: str,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError(f"{label} is not an object")
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise HandoffContractError(f"{label} payload seal drifted")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            f"JSON artifact is unavailable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise HandoffContractError(f"JSON artifact is not an object: {path}")
    return value


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


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _status(value: Mapping[str, Any]) -> str:
    return str(value.get("status") or value.get("state") or "").strip().lower()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def initialize_watch_plan(
    *,
    fast_lane_receipt_path: Path,
    output_root: Path,
    final_output_root: Path = DEFAULT_FINAL_OUTPUT_ROOT,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> Path:
    """Create the immutable operational plan without creating final output."""
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not MIN_POLL_SECONDS <= poll_seconds <= MAX_POLL_SECONDS
    ):
        raise HandoffContractError(
            "Full post-success poll interval is invalid"
        )
    scheduler_origin = scheduler_url.rstrip("/")
    if scheduler_origin != DEFAULT_SCHEDULER_URL:
        raise HandoffContractError(
            "Full post-success collection requires Scheduler port 8002"
        )
    fast_lane_path = fast_lane_receipt_path.resolve()
    if (
        not fast_lane_path.is_absolute()
        or not fast_lane_path.parent.is_dir()
    ):
        raise HandoffContractError(
            "Full fast-lane receipt parent is unavailable"
        )
    final_root = final_output_root.resolve()
    final_parent = final_root.parent
    if (
        not final_root.is_absolute()
        or not final_parent.is_dir()
        or final_root == final_parent
    ):
        raise HandoffContractError(
            "final package parent must already exist"
        )
    operational_root = output_root.resolve()
    if (
        operational_root == final_root
        or _path_is_within(operational_root, final_root)
        or _path_is_within(final_root, operational_root)
    ):
        raise HandoffContractError(
            "operational and final package roots must be separate"
        )
    operational_root.mkdir(parents=True, exist_ok=True)
    plan_path = operational_root / "watch_plan.json"
    if plan_path.exists():
        plan = _load_watch_plan(plan_path)
        if (
            plan["fast_lane_receipt_path"] != str(fast_lane_path)
            or plan["final_output_root"] != str(final_root)
            or plan["scheduler_url"] != scheduler_origin
            or plan["poll_seconds"] != poll_seconds
        ):
            raise HandoffContractError(
                "existing Full post-success watch plan differs"
            )
        return plan_path
    if final_root.exists():
        raise HandoffContractError(
            "final package root pre-exists before watch authority"
        )
    plan = _sealed(
        {
            "schema_version": WATCH_PLAN_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "fast_lane_receipt_path": str(fast_lane_path),
            "output_root": str(operational_root),
            "final_output_root": str(final_root),
            "final_output_parent": str(final_parent.resolve(strict=True)),
            "canonical_final_output_root": (
                final_root == DEFAULT_FINAL_OUTPUT_ROOT.resolve()
            ),
            "scheduler_url": scheduler_origin,
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "full_resources": copy.deepcopy(FULL_RESOURCES),
            "full_model": 1,
            "thermal_symmetry": "full",
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "maximum_full_tasks": 1,
            "disk_admission_formula": {
                "receipt_byte_components": [
                    "standard_artifact_size_bytes",
                    "standard_results_size_bytes",
                    "full_artifact_size_bytes",
                    "full_results_size_bytes",
                ],
                "additional_percent_numerator": LOCAL_PERCENT_NUMERATOR,
                "additional_percent_denominator": (
                    LOCAL_PERCENT_DENOMINATOR
                ),
                "fixed_reserve_bytes": LOCAL_FIXED_RESERVE_BYTES,
                "same_volume_required": True,
            },
            "poll_seconds": poll_seconds,
            "stable_root_created_during_init": False,
            "path_existence_is_completion_authority": False,
            "remote_retained_markers_pruned": False,
            "scheduler_methods_allowed": ["GET"],
            "scheduler_submission_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
            "created_at_utc": _now(),
        }
    )
    _write_immutable(plan_path, plan)
    if final_root.exists():
        raise HandoffContractError(
            "watch initialization unexpectedly created final package root"
        )
    return plan_path


WATCH_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "fast_lane_receipt_path",
        "output_root",
        "final_output_root",
        "final_output_parent",
        "canonical_final_output_root",
        "scheduler_url",
        "scheduler_project",
        "full_resources",
        "full_model",
        "thermal_symmetry",
        "fixed_boundary",
        "maximum_full_tasks",
        "disk_admission_formula",
        "poll_seconds",
        "stable_root_created_during_init",
        "path_existence_is_completion_authority",
        "remote_retained_markers_pruned",
        "scheduler_methods_allowed",
        "scheduler_submission_performed",
        "scheduler_cancel_performed",
        "scheduler_mutation_performed",
        "created_at_utc",
    }
)


def _load_watch_plan(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    plan = _validate_seal(
        _read_json(resolved),
        WATCH_PLAN_SCHEMA,
        label="Full post-success watch plan",
    )
    root = Path(str(plan.get("output_root") or "")).resolve()
    final_root = Path(str(plan.get("final_output_root") or "")).resolve()
    final_parent = Path(str(plan.get("final_output_parent") or "")).resolve(
        strict=True
    )
    receipt = Path(str(plan.get("fast_lane_receipt_path") or "")).resolve()
    if (
        set(plan) != WATCH_PLAN_FIELDS
        or resolved != root / "watch_plan.json"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or not root.is_dir()
        or not receipt.parent.is_dir()
        or final_parent != final_root.parent
        or not final_parent.is_dir()
        or root == final_root
        or _path_is_within(root, final_root)
        or _path_is_within(final_root, root)
        or plan.get("scheduler_url") != DEFAULT_SCHEDULER_URL
        or plan.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or plan.get("full_resources") != FULL_RESOURCES
        or plan.get("full_model") != 1
        or plan.get("thermal_symmetry") != "full"
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("maximum_full_tasks") != 1
        or plan.get("canonical_final_output_root")
        is not (final_root == DEFAULT_FINAL_OUTPUT_ROOT.resolve())
        or plan.get("disk_admission_formula")
        != {
            "receipt_byte_components": [
                "standard_artifact_size_bytes",
                "standard_results_size_bytes",
                "full_artifact_size_bytes",
                "full_results_size_bytes",
            ],
            "additional_percent_numerator": LOCAL_PERCENT_NUMERATOR,
            "additional_percent_denominator": LOCAL_PERCENT_DENOMINATOR,
            "fixed_reserve_bytes": LOCAL_FIXED_RESERVE_BYTES,
            "same_volume_required": True,
        }
        or not MIN_POLL_SECONDS
        <= _positive_int(plan.get("poll_seconds"), "poll seconds")
        <= MAX_POLL_SECONDS
        or plan.get("stable_root_created_during_init") is not False
        or plan.get("path_existence_is_completion_authority") is not False
        or plan.get("remote_retained_markers_pruned") is not False
        or plan.get("scheduler_methods_allowed") != ["GET"]
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("scheduler_cancel_performed") is not False
        or plan.get("scheduler_mutation_performed") is not False
    ):
        raise HandoffContractError(
            "Full post-success watch plan contract drifted"
        )
    _aware(plan.get("created_at_utc"), "Full post-success watch plan")
    return plan


def _load_fast_lane_handoff(path: Path) -> dict[str, Any]:
    """Authenticate the stable cross-commit file boundary from fast lane."""
    receipt_path = path.resolve(strict=True)
    receipt = _validate_seal(
        _read_json(receipt_path),
        FAST_LANE_RECEIPT_SCHEMA,
        label="Full fast-lane receipt",
    )
    submission_record = receipt.get("full_submission")
    if not isinstance(submission_record, Mapping):
        raise HandoffContractError(
            "Full fast-lane submission record is absent"
        )
    submission_path = Path(str(submission_record.get("path") or ""))
    if production._file_record(submission_path) != submission_record:
        raise HandoffContractError(
            "Full fast-lane submission bytes drifted"
        )
    raw_submission = production._validate_seal(
        production._read_json(submission_path),
        promotion.FULL_SUBMISSION_SCHEMA,
    )
    plan_record = raw_submission.get("plan")
    if not isinstance(plan_record, Mapping):
        raise HandoffContractError("Full submission plan record is absent")
    plan_path = Path(str(plan_record.get("path") or ""))
    if production._file_record(plan_path) != plan_record:
        raise HandoffContractError("Full submission plan bytes drifted")
    full_plan, _params, _profile, _view, standard_truth = (
        promotion._load_full_plan(plan_path)
    )
    submission = promotion._load_full_submission(
        submission_path,
        plan=full_plan,
        standard_truth=standard_truth,
    )
    if (
        receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("candidate_physics_sha256")
        != full_plan["candidate_physics_sha256"]
        or receipt.get("task_id") != submission["task_id"]
        or receipt.get("task_name") != full_plan["stage"]["task_name"]
        or receipt.get("dedupe_key")
        != full_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or receipt.get("resources") != FULL_RESOURCES
        or receipt.get("full_model") != 1
        or receipt.get("thermal_symmetry") != "full"
        or receipt.get("fixed_boundary") != FIXED_BOUNDARY
        or receipt.get("exact_full_sibling_count") != 1
        or receipt.get("scheduler_submission_performed") is not True
        or receipt.get("scheduler_repository_modified") is not False
        or submission.get("scheduler_url") != DEFAULT_SCHEDULER_URL
        or submission.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or submission.get("resources")
        != {"cpus": 16, "timeout_seconds": 43200}
    ):
        raise HandoffContractError(
            "Full fast-lane handoff contract drifted"
        )
    return {
        "receipt_path": receipt_path,
        "receipt": receipt,
        "submission_path": submission_path.resolve(strict=True),
        "submission": submission,
        "plan_path": plan_path.resolve(strict=True),
        "plan": full_plan,
        "standard_truth": standard_truth,
    }


def _default_sibling_reader(
    *,
    scheduler_url: str,
    project: str,
    task_name: str,
) -> list[dict[str, Any]]:
    return diagnostic._scheduler_project_tasks(
        scheduler_url=scheduler_url,
        project=project,
        task_name=task_name,
    )


def _normalize_task(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _positive_int(
        row.get("task_id", row.get("id")), "Scheduler task ID"
    )
    return {
        "task_id": task_id,
        "name": str(row.get("name") or ""),
        "status": str(row.get("status") or "").strip().lower(),
        "state": str(row.get("state") or "").strip().lower(),
        "project": str(row.get("project") or ""),
        "dedupe_key": str(row.get("dedupe_key") or ""),
        "cpus": row.get("cpus"),
        "memory_mb": row.get("memory_mb"),
        "timeout_seconds": row.get("timeout_seconds"),
        "aedt_backend": str(row.get("aedt_backend") or ""),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "exit_code": row.get("exit_code"),
    }


def _exact_sibling_inventory(
    rows: Any,
    *,
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError("Full sibling inventory is absent")
    plan = handoff["plan"]
    submission = handoff["submission"]
    expected_name = plan["stage"]["task_name"]
    expected_dedupe = plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    matching = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise HandoffContractError("Full sibling row is malformed")
        name = str(raw.get("name") or "")
        dedupe = str(raw.get("dedupe_key") or "")
        if name != expected_name and dedupe != expected_dedupe:
            continue
        normalized = _normalize_task(raw)
        if (
            normalized["name"] != expected_name
            or normalized["dedupe_key"] != expected_dedupe
            or normalized["project"] != scheduler_client.MFT_PROJECT
        ):
            raise HandoffContractError(
                "Full sibling partially collided with exact identity"
            )
        matching.append(normalized)
    matching.sort(key=lambda item: item["task_id"])
    if (
        len(matching) != 1
        or matching[0]["task_id"] != submission["task_id"]
    ):
        raise HandoffContractError(
            "exactly one submitted Full sibling is required"
        )
    unsigned = {
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "task_name": expected_name,
        "dedupe_key": expected_dedupe,
        "matching_task_count": 1,
        "matching_tasks": matching,
    }
    return {**unsigned, "snapshot_sha256": canonical_sha256(unsigned)}


def _terminal_snapshot(
    *,
    handoff: Mapping[str, Any],
    scheduler_url: str,
    sibling_reader: Any,
    task_reader: Any,
) -> dict[str, Any]:
    plan = handoff["plan"]
    submission = handoff["submission"]
    rows = sibling_reader(
        scheduler_url=scheduler_url,
        project=scheduler_client.MFT_PROJECT,
        task_name=plan["stage"]["task_name"],
    )
    inventory = _exact_sibling_inventory(rows, handoff=handoff)
    raw = task_reader(
        scheduler_url=scheduler_url,
        task_id=int(submission["task_id"]),
    )
    if not isinstance(raw, Mapping):
        raise HandoffContractError("Full task snapshot is absent")
    task = _normalize_task(raw)
    inventory_task = inventory["matching_tasks"][0]
    task_status = _status(task)
    inventory_status = _status(inventory_task)
    if (
        task["task_id"] != submission["task_id"]
        or task["name"] != plan["stage"]["task_name"]
        or task["project"] != scheduler_client.MFT_PROJECT
        or task["dedupe_key"]
        != plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
        or not task_status
        or (inventory_status and inventory_status != task_status)
    ):
        raise HandoffContractError("Full live task identity drifted")
    return {
        "inventory": inventory,
        "task": task,
        "status": task_status,
        "scheduler_methods_used": ["GET"],
        "scheduler_mutation_performed": False,
    }


def _load_or_collect(
    *,
    root: Path,
    handoff: Mapping[str, Any],
    scheduler_url: str,
    full_collector: Any,
) -> tuple[Path, dict[str, Any]]:
    collection_path = root / "full_collection.json"
    if not collection_path.exists():
        full_collector(
            plan_path=handoff["plan_path"],
            submission_path=handoff["submission_path"],
            scheduler_url=scheduler_url,
            output=collection_path,
        )
    view = promotion.authenticate_full_collection(collection_path)
    collection = view["collection"]
    if (
        collection.get("submission")
        != production._file_record(handoff["submission_path"])
        or collection.get("plan")
        != production._file_record(handoff["plan_path"])
        or collection.get("task_id") != handoff["submission"]["task_id"]
        or collection.get("candidate_physics_sha256")
        != handoff["plan"]["candidate_physics_sha256"]
        or collection.get("scheduler_get_only_collection") is not True
        or collection.get("scheduler_mutation_performed") is not False
        or collection.get("retention_required_until_package") is not True
    ):
        raise HandoffContractError(
            "Full collection ancestry or GET-only contract drifted"
        )
    return collection_path, collection


def _receipt_byte_components(
    collection: Mapping[str, Any],
) -> dict[str, int]:
    standard = collection.get("standard_remote_aedt_bundle_receipt")
    full = collection.get("remote_full_aedt_bundle_receipt")
    if not isinstance(standard, Mapping) or not isinstance(full, Mapping):
        raise HandoffContractError(
            "Standard/Full retained receipts are absent"
        )
    components = {
        "standard_artifact_size_bytes": _positive_int(
            standard.get("artifact_size_bytes"),
            "Standard retained AEDT size",
        ),
        "standard_results_size_bytes": _nonnegative_int(
            standard.get("results_size_bytes"),
            "Standard retained results size",
        ),
        "full_artifact_size_bytes": _positive_int(
            full.get("artifact_size_bytes"),
            "Full retained AEDT size",
        ),
        "full_results_size_bytes": _nonnegative_int(
            full.get("results_size_bytes"),
            "Full retained results size",
        ),
    }
    if (
        standard.get("retention_required") is not True
        or standard.get("prune_protection_required") is not True
        or full.get("retention_required") is not True
        or full.get("prune_protection_required") is not True
        or collection.get("standard_retained_bundle_reauthenticated")
        is not True
        or collection.get("full_prune_protection_marker_verified")
        is not True
    ):
        raise HandoffContractError(
            "retained Standard/Full prune protection is not authenticated"
        )
    return components


def _default_disk_usage(path: Path) -> Any:
    return shutil.disk_usage(path)


def _default_device(path: Path) -> int:
    return int(path.stat().st_dev)


def _disk_usage_values(value: Any) -> tuple[int, int, int]:
    if isinstance(value, Mapping):
        raw = (value.get("total"), value.get("used"), value.get("free"))
    else:
        try:
            raw = (value.total, value.used, value.free)
        except AttributeError:
            try:
                raw = tuple(value)
            except TypeError as exc:
                raise HandoffContractError(
                    "local disk usage response is malformed"
                ) from exc
    if len(raw) != 3:
        raise HandoffContractError("local disk usage response is malformed")
    total, used, free = (
        _nonnegative_int(item, f"local disk {label}")
        for item, label in zip(raw, ("total", "used", "free"), strict=True)
    )
    if total <= 0 or used + free > total:
        raise HandoffContractError("local disk usage values are invalid")
    return total, used, free


def _capture_disk_admission(
    *,
    plan: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
    disk_usage_reader: Callable[[Path], Any],
    device_reader: Callable[[Path], int],
    observed_at_utc: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    final_root = Path(plan["final_output_root"])
    parent = final_root.parent.resolve(strict=True)
    if parent != Path(plan["final_output_parent"]):
        raise HandoffContractError("final package parent drifted")
    if final_root.exists():
        raise HandoffContractError(
            "final package root appeared before local disk admission"
        )
    components = _receipt_byte_components(collection)
    retained_bytes = sum(components.values())
    ten_percent = (
        retained_bytes * LOCAL_PERCENT_NUMERATOR
        + LOCAL_PERCENT_DENOMINATOR
        - 1
    ) // LOCAL_PERCENT_DENOMINATOR
    required = retained_bytes + ten_percent + LOCAL_FIXED_RESERVE_BYTES
    total, used, free = _disk_usage_values(disk_usage_reader(parent))
    source_device = _nonnegative_int(
        device_reader(parent), "local disk device"
    )
    destination_device = _nonnegative_int(
        device_reader(final_root.parent), "final package device"
    )
    if source_device != destination_device:
        raise HandoffContractError(
            "local disk admission was not measured on final package volume"
        )
    observed = str(observed_at_utc or _now())
    admission = _sealed(
        {
            "schema_version": DISK_ADMISSION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "collection": production._file_record(collection_path),
            "collection_payload_sha256": collection["payload_sha256"],
            "candidate_physics_sha256": collection[
                "candidate_physics_sha256"
            ],
            "full_task_id": collection["task_id"],
            "final_output_root": str(final_root),
            "disk_usage_path": str(parent),
            "disk_volume_anchor": str(parent.anchor),
            "disk_volume_device": source_device,
            "same_volume_verified": True,
            "receipt_byte_components": components,
            "retained_receipt_bytes_total": retained_bytes,
            "ten_percent_reserve_bytes": ten_percent,
            "fixed_reserve_bytes": LOCAL_FIXED_RESERVE_BYTES,
            "required_free_bytes": required,
            "observed_total_bytes": total,
            "observed_used_bytes": used,
            "observed_free_bytes": free,
            "admission_passed": free >= required,
            "stable_root_existed_at_admission": False,
            "remote_retained_markers_pruned": False,
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "observed_at_utc": observed,
        }
    )
    admission_root = Path(plan["output_root"]) / "disk_admissions"
    name = f"disk_admission_{admission['payload_sha256'][:20]}.json"
    path = admission_root / name
    if path.exists():
        existing = _validate_disk_admission(
            path,
            plan=plan,
            collection_path=collection_path,
            collection=collection,
        )
        if existing != admission:
            raise HandoffContractError(
                "disk admission digest collision drifted"
            )
    else:
        _write_immutable(path, admission)
    return path, admission


def _validate_disk_admission(
    path: Path,
    *,
    plan: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        DISK_ADMISSION_SCHEMA,
        label="Full package local disk admission",
    )
    components = _receipt_byte_components(collection)
    retained = sum(components.values())
    reserve = (
        retained * LOCAL_PERCENT_NUMERATOR
        + LOCAL_PERCENT_DENOMINATOR
        - 1
    ) // LOCAL_PERCENT_DENOMINATOR
    required = retained + reserve + LOCAL_FIXED_RESERVE_BYTES
    if (
        set(value) != DISK_ADMISSION_FIELDS
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("collection")
        != production._file_record(collection_path)
        or value.get("collection_payload_sha256")
        != collection["payload_sha256"]
        or value.get("candidate_physics_sha256")
        != collection["candidate_physics_sha256"]
        or value.get("full_task_id") != collection["task_id"]
        or value.get("final_output_root") != plan["final_output_root"]
        or value.get("disk_usage_path") != plan["final_output_parent"]
        or value.get("disk_volume_anchor")
        != str(Path(plan["final_output_parent"]).anchor)
        or value.get("same_volume_verified") is not True
        or value.get("receipt_byte_components") != components
        or value.get("retained_receipt_bytes_total") != retained
        or value.get("ten_percent_reserve_bytes") != reserve
        or value.get("fixed_reserve_bytes") != LOCAL_FIXED_RESERVE_BYTES
        or value.get("required_free_bytes") != required
        or value.get("admission_passed")
        is not (value.get("observed_free_bytes", -1) >= required)
        or value.get("stable_root_existed_at_admission") is not False
        or value.get("remote_retained_markers_pruned") is not False
        or value.get("scheduler_methods_used") != ["GET"]
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise HandoffContractError(
            "Full package local disk admission drifted"
        )
    _nonnegative_int(value.get("disk_volume_device"), "disk device")
    total = _nonnegative_int(
        value.get("observed_total_bytes"), "observed total bytes"
    )
    used = _nonnegative_int(
        value.get("observed_used_bytes"), "observed used bytes"
    )
    free = _nonnegative_int(
        value.get("observed_free_bytes"), "observed free bytes"
    )
    if total <= 0 or used + free > total:
        raise HandoffContractError(
            "Full package disk admission capacity drifted"
        )
    _aware(value.get("observed_at_utc"), "local disk admission")
    return value


DISK_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "collection",
        "collection_payload_sha256",
        "candidate_physics_sha256",
        "full_task_id",
        "final_output_root",
        "disk_usage_path",
        "disk_volume_anchor",
        "disk_volume_device",
        "same_volume_verified",
        "receipt_byte_components",
        "retained_receipt_bytes_total",
        "ten_percent_reserve_bytes",
        "fixed_reserve_bytes",
        "required_free_bytes",
        "observed_total_bytes",
        "observed_used_bytes",
        "observed_free_bytes",
        "admission_passed",
        "stable_root_existed_at_admission",
        "remote_retained_markers_pruned",
        "scheduler_methods_used",
        "scheduler_mutation_performed",
        "observed_at_utc",
    }
)


def _package_intent(
    *,
    plan: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
    admission_path: Path,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": PACKAGE_INTENT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "collection": production._file_record(collection_path),
            "collection_payload_sha256": collection["payload_sha256"],
            "candidate_physics_sha256": collection[
                "candidate_physics_sha256"
            ],
            "full_task_id": collection["task_id"],
            "disk_admission": production._file_record(admission_path),
            "disk_admission_payload_sha256": admission["payload_sha256"],
            "required_free_bytes": admission["required_free_bytes"],
            "observed_free_bytes": admission["observed_free_bytes"],
            "same_volume_verified": True,
            "final_output_root": plan["final_output_root"],
            "stable_root_existed_before_package": False,
            "package_results_atomic_rename_required": True,
            "path_existence_is_completion_authority": False,
            "remote_retained_markers_pruned": False,
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "created_at_utc": _now(),
        }
    )


def _load_package_intent(
    path: Path,
    *,
    plan: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        PACKAGE_INTENT_SCHEMA,
        label="Full final package intent",
    )
    admission_record = value.get("disk_admission")
    if not isinstance(admission_record, Mapping):
        raise HandoffContractError("package intent disk admission is absent")
    admission_path = Path(str(admission_record.get("path") or ""))
    if production._file_record(admission_path) != admission_record:
        raise HandoffContractError("package intent disk admission drifted")
    admission = _validate_disk_admission(
        admission_path,
        plan=plan,
        collection_path=collection_path,
        collection=collection,
    )
    if (
        set(value) != PACKAGE_INTENT_FIELDS
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("collection")
        != production._file_record(collection_path)
        or value.get("collection_payload_sha256")
        != collection["payload_sha256"]
        or value.get("candidate_physics_sha256")
        != collection["candidate_physics_sha256"]
        or value.get("full_task_id") != collection["task_id"]
        or value.get("disk_admission_payload_sha256")
        != admission["payload_sha256"]
        or value.get("required_free_bytes")
        != admission["required_free_bytes"]
        or value.get("observed_free_bytes")
        != admission["observed_free_bytes"]
        or admission.get("admission_passed") is not True
        or value.get("same_volume_verified") is not True
        or value.get("final_output_root") != plan["final_output_root"]
        or value.get("stable_root_existed_before_package") is not False
        or value.get("package_results_atomic_rename_required") is not True
        or value.get("path_existence_is_completion_authority") is not False
        or value.get("remote_retained_markers_pruned") is not False
        or value.get("scheduler_methods_used") != ["GET"]
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise HandoffContractError("Full final package intent drifted")
    if _aware(
        value.get("created_at_utc"), "Full package intent"
    ) < _aware(admission.get("observed_at_utc"), "local disk admission"):
        raise HandoffContractError(
            "Full package intent predates local disk admission"
        )
    return value, admission_path, admission


PACKAGE_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "collection",
        "collection_payload_sha256",
        "candidate_physics_sha256",
        "full_task_id",
        "disk_admission",
        "disk_admission_payload_sha256",
        "required_free_bytes",
        "observed_free_bytes",
        "same_volume_verified",
        "final_output_root",
        "stable_root_existed_before_package",
        "package_results_atomic_rename_required",
        "path_existence_is_completion_authority",
        "remote_retained_markers_pruned",
        "scheduler_methods_used",
        "scheduler_mutation_performed",
        "created_at_utc",
    }
)


def _expected_package_files(
    package: Mapping[str, Any],
    root: Path,
) -> set[str]:
    expected = {"package_manifest.json"}
    artifacts = package["aedt_artifacts"]
    trees = package["aedtresults_trees"]
    evidence = package["evidence"]
    for record in artifacts.values():
        path = Path(record["local_absolute_path"]).resolve(strict=True)
        expected.add(path.relative_to(root).as_posix())
    for record in evidence.values():
        path = Path(record["local_absolute_path"]).resolve(strict=True)
        expected.add(path.relative_to(root).as_posix())
    for tree in trees.values():
        for record in tree["files"]:
            path = Path(record["local_absolute_path"]).resolve(strict=True)
            expected.add(path.relative_to(root).as_posix())
    return expected


def _rehash_package(
    *,
    manifest_path: Path,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    root = manifest_path.resolve(strict=True).parent
    nodes = list(root.rglob("*"))
    if any(node.is_symlink() for node in nodes):
        raise HandoffContractError("final package contains a symlink")
    files = sorted(
        (node for node in nodes if node.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    expected = _expected_package_files(package, root)
    observed_names = {
        item.relative_to(root).as_posix() for item in files
    }
    observed_directories = {
        item.relative_to(root).as_posix()
        for item in nodes
        if item.is_dir()
    }
    expected_directories = {
        parent.as_posix()
        for name in expected
        for parent in Path(name).parents
        if parent != Path(".")
    }
    if (
        observed_names != expected
        or observed_directories != expected_directories
    ):
        raise HandoffContractError(
            "final package file inventory differs from manifest authority"
        )
    inventory = []
    total = 0
    for item in files:
        size = item.stat().st_size
        total += size
        inventory.append(
            {
                "relative_path": item.relative_to(root).as_posix(),
                "sha256": production._sha256_file(item),
                "size_bytes": size,
            }
        )
    return _sealed(
        {
            "schema_version": PACKAGE_REHASH_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest": production._file_record(manifest_path),
            "package_payload_sha256": package["payload_sha256"],
            "candidate_physics_sha256": package[
                "candidate_physics_sha256"
            ],
            "full_task_id": package["result_identities"]["full"]["task_id"],
            "file_count": len(inventory),
            "total_size_bytes": total,
            "files": inventory,
            "package_tree_sha256": canonical_sha256(inventory),
            "full_validate_package_reauthenticated": True,
            "every_package_file_rehashed": True,
            "created_at_utc": _now(),
        }
    )


def _ensure_rehash_receipt(
    *,
    root: Path,
    manifest_path: Path,
    package: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    current = _rehash_package(
        manifest_path=manifest_path,
        package=package,
    )
    path = root / "package_rehash_manifest.json"
    if path.exists():
        previous = _validate_seal(
            _read_json(path),
            PACKAGE_REHASH_SCHEMA,
            label="Full final package rehash manifest",
        )
        previous_without_time = dict(previous)
        current_without_time = dict(current)
        previous_without_time.pop("created_at_utc", None)
        previous_without_time.pop("payload_sha256", None)
        current_without_time.pop("created_at_utc", None)
        current_without_time.pop("payload_sha256", None)
        if previous_without_time != current_without_time:
            raise HandoffContractError(
                "Full final package rehash inventory drifted"
            )
        _aware(
            previous.get("created_at_utc"),
            "Full final package rehash manifest",
        )
        return path, previous
    _write_immutable(path, current)
    return path, current


def _validate_package_for_collection(
    *,
    plan: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    final_root = Path(plan["final_output_root"])
    if not final_root.is_dir() or final_root.is_symlink():
        raise HandoffContractError(
            "final package root is absent or not a regular directory"
        )
    manifest_path = final_root / "package_manifest.json"
    package = promotion.authenticate_package(manifest_path)
    full_source = package.get("evidence", {}).get("full_collection", {}).get(
        "source"
    )
    if (
        full_source != production._file_record(collection_path)
        or package.get("candidate_physics_sha256")
        != collection["candidate_physics_sha256"]
        or package.get("result_identities", {}).get("full", {}).get("task_id")
        != collection["task_id"]
        or package.get("scheduler_get_only_package_fetch") is not True
        or package.get("scheduler_mutation_performed") is not False
        or package.get("both_aedt_projects_present") is not True
        or package.get("both_aedtresults_trees_present") is not True
    ):
        raise HandoffContractError(
            "validated final package does not match collected Full truth"
        )
    return manifest_path, package


def _success_payload(
    *,
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    terminal_snapshot: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
    intent_path: Path,
    intent: Mapping[str, Any],
    admission_path: Path,
    admission: Mapping[str, Any],
    manifest_path: Path,
    package: Mapping[str, Any],
    rehash_path: Path,
    rehash: Mapping[str, Any],
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": SUCCESS_RECEIPT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan_payload_sha256": plan["payload_sha256"],
            "fast_lane_receipt": production._file_record(
                handoff["receipt_path"]
            ),
            "fast_lane_receipt_payload_sha256": handoff["receipt"][
                "payload_sha256"
            ],
            "full_submission": production._file_record(
                handoff["submission_path"]
            ),
            "full_submission_payload_sha256": handoff["submission"][
                "payload_sha256"
            ],
            "candidate_physics_sha256": collection[
                "candidate_physics_sha256"
            ],
            "full_task_id": collection["task_id"],
            "terminal_status": terminal_snapshot["status"],
            "terminal_sibling_evidence": {
                "matching_task_count": terminal_snapshot["inventory"][
                    "matching_task_count"
                ],
                "task_id": terminal_snapshot["inventory"][
                    "matching_tasks"
                ][0]["task_id"],
                "task_name": terminal_snapshot["inventory"]["task_name"],
                "dedupe_key": terminal_snapshot["inventory"]["dedupe_key"],
                "status": terminal_snapshot["status"],
            },
            "exactly_one_full_terminal_success": True,
            "full_collection": production._file_record(collection_path),
            "full_collection_payload_sha256": collection["payload_sha256"],
            "package_intent": production._file_record(intent_path),
            "package_intent_payload_sha256": intent["payload_sha256"],
            "disk_admission": production._file_record(admission_path),
            "disk_admission_payload_sha256": admission["payload_sha256"],
            "retained_receipt_bytes_total": admission[
                "retained_receipt_bytes_total"
            ],
            "required_free_bytes": admission["required_free_bytes"],
            "observed_free_bytes": admission["observed_free_bytes"],
            "same_volume_disk_admission_passed": True,
            "package_manifest": production._file_record(manifest_path),
            "package_payload_sha256": package["payload_sha256"],
            "package_rehash_manifest": production._file_record(rehash_path),
            "package_rehash_payload_sha256": rehash["payload_sha256"],
            "package_tree_sha256": rehash["package_tree_sha256"],
            "package_file_count": rehash["file_count"],
            "package_total_size_bytes": rehash["total_size_bytes"],
            "full_validate_package_reauthenticated": True,
            "every_package_file_rehashed": True,
            "path_existence_used_as_completion_authority": False,
            "standard_retained_remote_marker_pruned": False,
            "full_retained_remote_marker_pruned": False,
            "remote_retention_required_through_validation": True,
            "scheduler_methods_used": ["GET"],
            "scheduler_post_performed": False,
            "scheduler_patch_performed": False,
            "scheduler_delete_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
            "completed_at_utc": _now(),
        }
    )


def _load_success_receipt(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        SUCCESS_RECEIPT_SCHEMA,
        label="Full post-success final receipt",
    )
    ignored = {"completed_at_utc", "payload_sha256"}
    if {
        key: item for key, item in value.items() if key not in ignored
    } != {
        key: item for key, item in expected.items() if key not in ignored
    }:
        raise HandoffContractError(
            "Full post-success final receipt drifted"
        )
    _aware(value.get("completed_at_utc"), "Full post-success final receipt")
    return value


def _finalize_package(
    *,
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    terminal_snapshot: Mapping[str, Any],
    collection_path: Path,
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(plan["output_root"])
    final_root = Path(plan["final_output_root"])
    intent_path = root / "package_intent.json"
    if final_root.exists():
        if not intent_path.is_file():
            raise HandoffContractError(
                "final package exists without a pre-package disk intent"
            )
        intent, admission_path, admission = _load_package_intent(
            intent_path,
            plan=plan,
            collection_path=collection_path,
            collection=collection,
        )
    else:
        raise HandoffContractError(
            "final package cannot be finalized before package creation"
        )
    manifest_path, package = _validate_package_for_collection(
        plan=plan,
        collection_path=collection_path,
        collection=collection,
    )
    rehash_path, rehash = _ensure_rehash_receipt(
        root=root,
        manifest_path=manifest_path,
        package=package,
    )
    expected = _success_payload(
        plan=plan,
        handoff=handoff,
        terminal_snapshot=terminal_snapshot,
        collection_path=collection_path,
        collection=collection,
        intent_path=intent_path,
        intent=intent,
        admission_path=admission_path,
        admission=admission,
        manifest_path=manifest_path,
        package=package,
        rehash_path=rehash_path,
        rehash=rehash,
    )
    success_path = root / "success_receipt.json"
    if success_path.exists():
        success = _load_success_receipt(success_path, expected=expected)
    else:
        _write_immutable(success_path, expected)
        success = expected
    return {
        "success_path": success_path,
        "success": success,
        "manifest_path": manifest_path,
        "package": package,
        "rehash_path": rehash_path,
        "rehash": rehash,
        "admission_path": admission_path,
        "admission": admission,
    }


def _state(
    *,
    plan: Mapping[str, Any],
    status: str,
    handoff: Mapping[str, Any] | None = None,
    terminal_snapshot: Mapping[str, Any] | None = None,
    collection_path: Path | None = None,
    admission_path: Path | None = None,
    admission: Mapping[str, Any] | None = None,
    final: Mapping[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan_payload_sha256": plan["payload_sha256"],
            "observed_at_utc": _now(),
            "status": status,
            "fast_lane_receipt_present": handoff is not None,
            "full_task_id": (
                handoff["submission"]["task_id"]
                if handoff is not None
                else None
            ),
            "full_task_status": (
                terminal_snapshot["status"]
                if terminal_snapshot is not None
                else None
            ),
            "exact_full_sibling_count": (
                terminal_snapshot["inventory"]["matching_task_count"]
                if terminal_snapshot is not None
                else None
            ),
            "full_collection": (
                production._file_record(collection_path)
                if collection_path is not None and collection_path.is_file()
                else None
            ),
            "disk_admission": (
                production._file_record(admission_path)
                if admission_path is not None and admission_path.is_file()
                else None
            ),
            "disk_admission_passed": (
                admission["admission_passed"]
                if admission is not None
                else None
            ),
            "final_output_root": plan["final_output_root"],
            "final_package_manifest": (
                production._file_record(final["manifest_path"])
                if final is not None
                else None
            ),
            "success_receipt": (
                production._file_record(final["success_path"])
                if final is not None
                else None
            ),
            "completion_authority": (
                "validated_success_receipt"
                if final is not None
                else None
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "path_existence_used_as_completion_authority": False,
            "remote_retained_markers_pruned": False,
            "scheduler_methods_used": ["GET"],
            "scheduler_post_performed": False,
            "scheduler_patch_performed": False,
            "scheduler_delete_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
        }
    )


def process_cycle(
    *,
    watch_plan_path: Path,
    sibling_reader: Any = _default_sibling_reader,
    task_reader: Any = diagnostic._scheduler_task_snapshot,
    full_collector: Any = promotion.collect_full,
    package_builder: Any = promotion.package_results,
    disk_usage_reader: Callable[[Path], Any] = _default_disk_usage,
    device_reader: Callable[[Path], int] = _default_device,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run one restart-safe cycle and persist a sealed mutable state."""
    plan = _load_watch_plan(watch_plan_path)
    root = Path(plan["output_root"])
    handoff = None
    terminal_snapshot = None
    collection_path = None
    admission_path = None
    admission = None
    final = None
    try:
        fast_lane_path = Path(plan["fast_lane_receipt_path"])
        if not fast_lane_path.exists():
            if Path(plan["final_output_root"]).exists():
                raise HandoffContractError(
                    "final package root exists before fast-lane authority"
                )
            state = _state(
                plan=plan,
                status="waiting_for_full_fast_lane_receipt",
            )
        else:
            handoff = _load_fast_lane_handoff(fast_lane_path)
            terminal_snapshot = _terminal_snapshot(
                handoff=handoff,
                scheduler_url=plan["scheduler_url"],
                sibling_reader=sibling_reader,
                task_reader=task_reader,
            )
            live_status = terminal_snapshot["status"]
            collection_path = root / "full_collection.json"
            if live_status in ACTIVE_STATUSES:
                if (
                    collection_path.exists()
                    or Path(plan["final_output_root"]).exists()
                ):
                    raise HandoffContractError(
                        "Full collection/package exists before terminal success"
                    )
                state = _state(
                    plan=plan,
                    status="waiting_for_full_terminal_success",
                    handoff=handoff,
                    terminal_snapshot=terminal_snapshot,
                )
            elif live_status in FAILED_STATUSES:
                if (
                    collection_path.exists()
                    or Path(plan["final_output_root"]).exists()
                ):
                    raise HandoffContractError(
                        "Full collection/package exists for terminal-failed task"
                    )
                state = _state(
                    plan=plan,
                    status="blocked_full_terminal_failure",
                    handoff=handoff,
                    terminal_snapshot=terminal_snapshot,
                )
            elif live_status not in COMPLETED_STATUSES:
                raise HandoffContractError(
                    f"Full task status is unsupported: {live_status!r}"
                )
            else:
                collection_path, collection = _load_or_collect(
                    root=root,
                    handoff=handoff,
                    scheduler_url=plan["scheduler_url"],
                    full_collector=full_collector,
                )
                if (
                    collection.get("goal_physical_spec_passed") is not True
                    or collection.get("actual_truth_evidence", {}).get(
                        "goal_physical_spec_passed"
                    )
                    is not True
                ):
                    if Path(plan["final_output_root"]).exists():
                        raise HandoffContractError(
                            "final package exists for failed Full truth"
                        )
                    state = _state(
                        plan=plan,
                        status="blocked_full_actual_constraints_failed",
                        handoff=handoff,
                        terminal_snapshot=terminal_snapshot,
                        collection_path=collection_path,
                    )
                elif Path(plan["final_output_root"]).exists():
                    final = _finalize_package(
                        plan=plan,
                        handoff=handoff,
                        terminal_snapshot=terminal_snapshot,
                        collection_path=collection_path,
                        collection=collection,
                    )
                    state = _state(
                        plan=plan,
                        status="completed_validated_final_package",
                        handoff=handoff,
                        terminal_snapshot=terminal_snapshot,
                        collection_path=collection_path,
                        admission_path=final["admission_path"],
                        admission=final["admission"],
                        final=final,
                    )
                else:
                    admission_path, admission = _capture_disk_admission(
                        plan=plan,
                        collection_path=collection_path,
                        collection=collection,
                        disk_usage_reader=disk_usage_reader,
                        device_reader=device_reader,
                        observed_at_utc=observed_at_utc,
                    )
                    if admission["admission_passed"] is not True:
                        state = _state(
                            plan=plan,
                            status="waiting_for_same_volume_local_disk",
                            handoff=handoff,
                            terminal_snapshot=terminal_snapshot,
                            collection_path=collection_path,
                            admission_path=admission_path,
                            admission=admission,
                        )
                    else:
                        intent_path = root / "package_intent.json"
                        intent = _package_intent(
                            plan=plan,
                            collection_path=collection_path,
                            collection=collection,
                            admission_path=admission_path,
                            admission=admission,
                        )
                        _write_atomic(intent_path, intent)
                        package_builder(
                            full_collection_path=collection_path,
                            scheduler_url=plan["scheduler_url"],
                            output=Path(plan["final_output_root"]),
                        )
                        final = _finalize_package(
                            plan=plan,
                            handoff=handoff,
                            terminal_snapshot=terminal_snapshot,
                            collection_path=collection_path,
                            collection=collection,
                        )
                        state = _state(
                            plan=plan,
                            status="completed_validated_final_package",
                            handoff=handoff,
                            terminal_snapshot=terminal_snapshot,
                            collection_path=collection_path,
                            admission_path=final["admission_path"],
                            admission=final["admission"],
                            final=final,
                        )
    except Exception as exc:
        state = _state(
            plan=plan,
            status="blocked_fail_closed",
            handoff=handoff,
            terminal_snapshot=terminal_snapshot,
            collection_path=collection_path,
            admission_path=admission_path,
            admission=admission,
            error=exc,
        )
    _write_atomic(root / "state.json", state)
    return state


def run_watcher(*, watch_plan_path: Path, once: bool = False) -> None:
    plan = _load_watch_plan(watch_plan_path)
    root = Path(plan["output_root"])
    with terminal.SingleInstanceLock(root / "postsuccess_watcher.lock"):
        _write_atomic(
            root / "watcher.pid.json",
            _sealed(
                {
                    "schema_version": PID_SCHEMA,
                    "pid": os.getpid(),
                    "watch_plan": production._file_record(
                        watch_plan_path.resolve(strict=True)
                    ),
                    "scheduler_methods_allowed": ["GET"],
                    "scheduler_mutation_performed": False,
                    "started_at_utc": _now(),
                }
            ),
        )
        while True:
            state = process_cycle(watch_plan_path=watch_plan_path)
            if (
                once
                or state["status"] == "completed_validated_final_package"
            ):
                return
            time.sleep(plan["poll_seconds"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GET-only Full terminal-success collector and final packager"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--fast-lane-receipt", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument(
        "--final-output-root",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_ROOT,
    )
    init.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    for name in ("run", "once"):
        command = commands.add_parser(name)
        command.add_argument("--watch-plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result: Any = initialize_watch_plan(
            fast_lane_receipt_path=args.fast_lane_receipt,
            output_root=args.output_root,
            final_output_root=args.final_output_root,
            poll_seconds=args.poll_seconds,
        )
    else:
        run_watcher(
            watch_plan_path=args.watch_plan,
            once=args.command == "once",
        )
        result = Path(args.watch_plan).resolve().parent / "state.json"
    print(json.dumps({"status": "ok", "result": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
