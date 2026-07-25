"""Restart-safe first-feasible-truth to exactly-one Full fast lane.

The terminal watcher state is discovery-only.  Submission authority comes
exclusively from its immutable actual-truth Pareto manifest and paired
non-dominated-sort receipt.  The first feasible rank-0 candidate is frozen in
an immutable selection receipt and planned with
``mft_goal_truth_promotion.create_full_plans(limit=1)``.

No Scheduler mutation is possible without a separately created, immutable
root-review activation receipt.  Even then, the existing
``mft_goal_truth_promotion.submit_full`` implementation performs the one
allowed POST behind an atomic singleton claim, an exact-sibling scan, and
fresh Scheduler, license, capacity, and GPFS gates.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    canonical_sha256,
)
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_campaign_atomic_claim as atomic_claim  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_terminal_success_watcher as terminal  # noqa: E402
from tools import mft_goal_timeout12h_retry as timeout12h  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


WATCH_PLAN_SCHEMA = "mft-goal-first-truth-full-fast-lane-watch-plan-v1"
ACTIVATION_SCHEMA = "mft-goal-first-truth-full-fast-lane-activation-v1"
SELECTION_SCHEMA = "mft-goal-first-truth-full-fast-lane-selection-v1"
GPFS_AUDIT_SCHEMA = "mft-goal-first-truth-full-fast-lane-gpfs-audit-v1"
STANDARD_STORAGE_BOUND_SCHEMA = (
    "mft-goal-first-truth-full-fast-lane-standard-storage-bound-v1"
)
SUCCESS_GRID_AUTHORITY_SCHEMA = (
    "mft-goal-first-truth-full-fast-lane-success-grid-authority-v1"
)
FULL_STORAGE_AUTHORITY_SCHEMA = (
    "mft-goal-first-truth-full-fast-lane-full-storage-authority-v1"
)
POST_INTENT_SCHEMA = "mft-goal-first-truth-full-fast-lane-post-intent-v1"
POST_RESULT_SCHEMA = "mft-goal-first-truth-full-fast-lane-post-result-v1"
RECEIPT_SCHEMA = "mft-goal-first-truth-full-fast-lane-receipt-v1"
STATE_SCHEMA = "mft-goal-first-truth-full-fast-lane-state-v1"
PID_SCHEMA = "mft-goal-first-truth-full-fast-lane-pid-v1"
PRIORITY_FENCE_SCHEMA = (
    "mft-goal-first-truth-full-fast-lane-priority-fence-v1"
)

CAMPAIGN_ID = "mft-goal-20260726"
FULL_RESOURCES = {
    "cpus": 16,
    "memory_mb": 98304,
    "timeout_seconds": 43200,
}
MINIMUM_GPFS_FREE_GB = 10.0
# Standard is an authenticated one-eighth thermal model.  The Full envelope
# reserves all eight physical sectors of the exact successful Standard grid.
# The established 10 GiB post-reservation floor remains the non-grid working
# allowance.  Neither retained-result bytes nor a generic per-task estimate
# is admissible storage authority.
FULL_SYMMETRY_EXPANSION_FACTOR = 8
STANDARD_SYMMETRY_DENOMINATOR = 8
SOLVER_TRANSIENT_ROOT = "/enroot"
SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB = 200.0
MAX_GPFS_AUDIT_AGE_SECONDS = 120
MIN_POLL_SECONDS = 10
MAX_POLL_SECONDS = 60
DEFAULT_POLL_SECONDS = 15
CLAIM_GENERATION = "first-truth-full-fast-lane-v1"
SINGLETON_LOGICAL_ID = 1
SINGLETON_CANDIDATE_SHA256 = canonical_sha256(
    {
        "campaign_id": CAMPAIGN_ID,
        "authority": "first-authenticated-feasible-rank0-full",
        "maximum_full_siblings": 1,
    }
)
DEFAULT_CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "first_truth_full_fast_lane_claims_v1"
)
DEFAULT_PRIORITY_FENCE_PATH = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "first_truth_full_priority_v1/full_priority_fence.json"
)
SAFE_REFILL_CANDIDATE_STEMS = {
    96223: "2a1bb6f2be79",
    96224: "436565e3f360",
    96230: "b7c30cb70b95",
}
ACTIVE_STATUSES = frozenset(
    {
        "queued",
        "pending",
        "submitted",
        "attaching",
        "starting",
        "running",
    }
)
SCHEDULER_ACTIVE_STATUS_FILTERS = ("queued", "attaching", "running")
CLAIM_AUTHORITY_SHA256 = canonical_sha256(
    {
        "campaign_id": CAMPAIGN_ID,
        "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "temperature_contract_sha256": GOAL_TEMPERATURE_CONTRACT_SHA256,
        "claim_generation": CLAIM_GENERATION,
        "singleton_candidate_sha256": SINGLETON_CANDIDATE_SHA256,
        "full_resources": FULL_RESOURCES,
        "full_model": 1,
        "thermal_symmetry": "full",
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "minimum_gpfs_free_gb": MINIMUM_GPFS_FREE_GB,
        "full_symmetry_expansion_factor": FULL_SYMMETRY_EXPANSION_FACTOR,
        "standard_symmetry_denominator": STANDARD_SYMMETRY_DENOMINATOR,
        "storage_bound_source": "authenticated_fresh_grid_output_bytes",
        "solver_transient_root": SOLVER_TRANSIENT_ROOT,
        "solver_transient_route_minimum_gib": (
            SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        ),
        "retained_bytes_are_storage_authority": False,
        "generic_per_task_storage_reservation_allowed": False,
        "maximum_scheduler_posts": 1,
        "safe_refill_priority_fence_path": str(DEFAULT_PRIORITY_FENCE_PATH),
    }
)

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


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandoffContractError(f"{label} is invalid")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise HandoffContractError(f"{label} is invalid")
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HandoffContractError(f"{label} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise HandoffContractError(f"{label} is invalid")
    return result


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise HandoffContractError("fast-lane payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(
    value: Any, schema: str, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError(f"{label} is not an object")
    result = copy.deepcopy(dict(value))
    observed = result.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or observed != canonical_sha256(result)
    ):
        raise HandoffContractError(f"{label} payload seal drifted")
    return copy.deepcopy(dict(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(f"JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffContractError(f"JSON artifact is not an object: {path}")
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    return production._write_immutable_json(path.resolve(), dict(value))


def _write_immutable_bytes(path: Path, value: bytes) -> Path:
    if not isinstance(value, bytes) or not value:
        raise HandoffContractError("immutable stream bytes are absent")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != value:
            raise HandoffContractError(
                f"existing immutable stream bytes differ: {target}"
            )
        return target
    try:
        with target.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if target.read_bytes() != value:
            raise HandoffContractError(
                f"raced immutable stream bytes differ: {target}"
            )
    return target


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                dict(value),
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def _git_revision() -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPOSITORY_ROOT.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return production._require_revision(
        completed.stdout.strip(), "fast-lane code revision"
    )


def _initialize_claim_root(root: Path) -> dict[str, Any]:
    try:
        return atomic_claim.initialize_claim_root(
            root,
            campaign_id=CAMPAIGN_ID,
            campaign_authority_sha256=CLAIM_AUTHORITY_SHA256,
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "Full fast-lane claim root could not be initialized"
        ) from exc


def _claim_reference(authority: Mapping[str, Any]) -> dict[str, Any]:
    return atomic_claim.build_claim_reference(
        authority,
        candidate_physics_sha256=SINGLETON_CANDIDATE_SHA256,
        logical_authority_task_id=SINGLETON_LOGICAL_ID,
        retry_generation=CLAIM_GENERATION,
    )


def initialize_watch_plan(
    *,
    terminal_state_path: Path,
    scheduler_cutover_receipt_path: Path,
    license_snapshot_directory: Path,
    gpfs_audit_directory: Path,
    output_root: Path,
    full_storage_authority_directory: Path | None = None,
    claim_root: Path = DEFAULT_CLAIM_ROOT,
    priority_fence_path: Path = DEFAULT_PRIORITY_FENCE_PATH,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> Path:
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int)
        or not MIN_POLL_SECONDS <= poll_seconds <= MAX_POLL_SECONDS
    ):
        raise HandoffContractError("Full fast-lane poll interval is invalid")
    state = terminal_state_path.resolve()
    if not state.is_absolute() or not state.parent.is_dir():
        raise HandoffContractError("terminal watcher state parent is unavailable")
    license_root = license_snapshot_directory.resolve()
    gpfs_root = gpfs_audit_directory.resolve()
    full_storage_root = (
        full_storage_authority_directory.resolve()
        if full_storage_authority_directory is not None
        else (gpfs_root / "full_authorities").resolve()
    )
    license_root.mkdir(parents=True, exist_ok=True)
    gpfs_root.mkdir(parents=True, exist_ok=True)
    full_storage_root.mkdir(parents=True, exist_ok=True)
    destination = output_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fence_path = priority_fence_path.resolve()
    if fence_path != DEFAULT_PRIORITY_FENCE_PATH.resolve():
        raise HandoffContractError(
            "Full fast-lane priority fence must use the shared fixed path"
        )
    path = destination / "watch_plan.json"
    if path.exists():
        existing = _load_watch_plan(path)
        if (
            existing["terminal_state_path"] != str(state)
            or existing["scheduler_cutover_receipt"]
            != production._file_record(
                scheduler_cutover_receipt_path.resolve(strict=True)
            )
            or existing["license_snapshot_directory"]
            != str(license_root)
            or existing["gpfs_audit_directory"] != str(gpfs_root)
            or existing["full_storage_authority_directory"]
            != str(full_storage_root)
            or existing["priority_fence_path"] != str(fence_path)
            or Path(
                existing["claim_root_authority"]["resolved_root"]
            ).resolve()
            != claim_root.resolve()
            or existing["poll_seconds"] != poll_seconds
        ):
            raise HandoffContractError(
                "existing Full fast-lane watch plan differs"
            )
        return path
    authority = _initialize_claim_root(claim_root.resolve())
    reference = _claim_reference(authority)
    cutover, _unused = diagnostic._validate_scheduler_cutover_receipt(
        scheduler_cutover_receipt_path,
        verify_live_launcher=False,
        require_strict_node=True,
        require_active_strict=True,
    )
    if (
        cutover["candidate_revision"]
        != diagnostic.SCHEDULER_STRICT_NODE_REVISION
        or cutover["pin_generation"] != "scheduler-strict-node-4fac-v4"
    ):
        raise HandoffContractError("Full fast lane requires active Scheduler 4fac")
    plan = _sealed(
        {
            "schema_version": WATCH_PLAN_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "terminal_state_path": str(state),
            "scheduler_url": diagnostic.DIAGNOSTIC_SCHEDULER_URL,
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "scheduler_cutover_receipt": production._file_record(
                scheduler_cutover_receipt_path.resolve(strict=True)
            ),
            "scheduler_cutover_payload_sha256": cutover["payload_sha256"],
            "scheduler_revision": diagnostic.SCHEDULER_STRICT_NODE_REVISION,
            "license_snapshot_directory": str(license_root),
            "gpfs_audit_directory": str(gpfs_root),
            "full_storage_authority_directory": str(full_storage_root),
            "output_root": str(destination),
            "priority_fence_path": str(fence_path),
            "claim_root_authority": authority,
            "claim_reference": reference,
            "full_resources": copy.deepcopy(FULL_RESOURCES),
            "full_plan_limit": 1,
            "full_model": 1,
            "thermal_symmetry": "full",
            "fixed_boundary": {
                "fan_velocity_m_s": 1.5,
                "fan_config": "dual",
                "thermal_pad_conductivity_W_mK": 0.2,
                "core_plate_pad_t_mm": 2.0,
                "wcp_pad_t_mm": 2.0,
            },
            "minimum_gpfs_free_gb": MINIMUM_GPFS_FREE_GB,
            "full_symmetry_expansion_factor": (
                FULL_SYMMETRY_EXPANSION_FACTOR
            ),
            "standard_symmetry_denominator": (
                STANDARD_SYMMETRY_DENOMINATOR
            ),
            "storage_bound_source": (
                "authenticated_fresh_grid_output_bytes"
            ),
            "solver_transient_root": SOLVER_TRANSIENT_ROOT,
            "solver_transient_route_minimum_gib": (
                SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
            ),
            "retained_bytes_are_storage_authority": False,
            "generic_per_task_storage_reservation_allowed": False,
            "maximum_scheduler_posts": 1,
            "poll_seconds": poll_seconds,
            "activation_receipt_required": True,
            "live_post_default_enabled": False,
            "scheduler_submission_performed": False,
            "scheduler_repository_modified": False,
            "created_by_revision": _git_revision(),
            "created_at_utc": _now(),
        }
    )
    _write_immutable(path, plan)
    return path


WATCH_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "terminal_state_path",
        "scheduler_url",
        "scheduler_project",
        "scheduler_cutover_receipt",
        "scheduler_cutover_payload_sha256",
        "scheduler_revision",
        "license_snapshot_directory",
        "gpfs_audit_directory",
        "full_storage_authority_directory",
        "output_root",
        "priority_fence_path",
        "claim_root_authority",
        "claim_reference",
        "full_resources",
        "full_plan_limit",
        "full_model",
        "thermal_symmetry",
        "fixed_boundary",
        "minimum_gpfs_free_gb",
        "full_symmetry_expansion_factor",
        "standard_symmetry_denominator",
        "storage_bound_source",
        "solver_transient_root",
        "solver_transient_route_minimum_gib",
        "retained_bytes_are_storage_authority",
        "generic_per_task_storage_reservation_allowed",
        "maximum_scheduler_posts",
        "poll_seconds",
        "activation_receipt_required",
        "live_post_default_enabled",
        "scheduler_submission_performed",
        "scheduler_repository_modified",
        "created_by_revision",
        "created_at_utc",
    }
)


def _load_watch_plan(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    plan = _validate_seal(
        _read_json(resolved), WATCH_PLAN_SCHEMA, label="Full fast-lane watch plan"
    )
    root = Path(str(plan.get("output_root") or "")).resolve()
    full_storage_root = Path(
        str(plan.get("full_storage_authority_directory") or "")
    ).resolve()
    if (
        set(plan) != WATCH_PLAN_FIELDS
        or plan.get("campaign_id") != CAMPAIGN_ID
        or resolved != root / "watch_plan.json"
        or plan.get("scheduler_url") != diagnostic.DIAGNOSTIC_SCHEDULER_URL
        or plan.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or plan.get("scheduler_revision")
        != diagnostic.SCHEDULER_STRICT_NODE_REVISION
        or scheduler_client.LOCAL_SCRATCH_ROOT != SOLVER_TRANSIENT_ROOT
        or scheduler_client.LOCAL_SCRATCH_MIN_FREE_KB
        != int(SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB * 1024 * 1024)
        or not full_storage_root.is_dir()
        or plan.get("full_resources") != FULL_RESOURCES
        or Path(str(plan.get("priority_fence_path") or "")).resolve()
        != DEFAULT_PRIORITY_FENCE_PATH.resolve()
        or plan.get("full_plan_limit") != 1
        or plan.get("full_model") != 1
        or plan.get("thermal_symmetry") != "full"
        or plan.get("fixed_boundary")
        != {
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "thermal_pad_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
        }
        or plan.get("minimum_gpfs_free_gb") != MINIMUM_GPFS_FREE_GB
        or plan.get("full_symmetry_expansion_factor")
        != FULL_SYMMETRY_EXPANSION_FACTOR
        or plan.get("standard_symmetry_denominator")
        != STANDARD_SYMMETRY_DENOMINATOR
        or plan.get("storage_bound_source")
        != "authenticated_fresh_grid_output_bytes"
        or plan.get("solver_transient_root") != SOLVER_TRANSIENT_ROOT
        or plan.get("solver_transient_route_minimum_gib")
        != SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        or plan.get("retained_bytes_are_storage_authority") is not False
        or plan.get("generic_per_task_storage_reservation_allowed")
        is not False
        or plan.get("maximum_scheduler_posts") != 1
        or plan.get("activation_receipt_required") is not True
        or plan.get("live_post_default_enabled") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("scheduler_repository_modified") is not False
    ):
        raise HandoffContractError("Full fast-lane watch plan contract drifted")
    cutover_record = plan.get("scheduler_cutover_receipt")
    if not isinstance(cutover_record, Mapping):
        raise HandoffContractError("Full fast-lane cutover record is absent")
    cutover_path = Path(str(cutover_record.get("path") or ""))
    if production._file_record(cutover_path) != cutover_record:
        raise HandoffContractError("Full fast-lane cutover bytes drifted")
    cutover, _unused = diagnostic._validate_scheduler_cutover_receipt(
        cutover_path,
        verify_live_launcher=False,
        require_strict_node=True,
        require_active_strict=True,
    )
    if (
        cutover["payload_sha256"]
        != plan["scheduler_cutover_payload_sha256"]
    ):
        raise HandoffContractError("Full fast-lane cutover payload drifted")
    authority = atomic_claim.load_claim_root(
        Path(plan["claim_root_authority"]["resolved_root"]),
        expected_authority=plan["claim_root_authority"],
    )
    try:
        reference = atomic_claim.validate_claim_reference(
            plan["claim_reference"], authority
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError("Full fast-lane claim reference drifted") from exc
    if reference != _claim_reference(authority):
        raise HandoffContractError("Full fast-lane singleton claim drifted")
    return plan


def create_activation_receipt(
    *, watch_plan_path: Path, reviewer: str, output: Path
) -> Path:
    plan = _load_watch_plan(watch_plan_path)
    if reviewer != "root-reviewed":
        raise HandoffContractError(
            "Full fast-lane activation requires explicit root review"
        )
    receipt = _sealed(
        {
            "schema_version": ACTIVATION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan": production._file_record(
                watch_plan_path.resolve(strict=True)
            ),
            "watch_plan_payload_sha256": plan["payload_sha256"],
            "reviewer": reviewer,
            "authorization_scope": (
                "at-most-one Scheduler Full POST for the immutable first "
                "authenticated feasible rank-0 truth selection"
            ),
            "maximum_scheduler_posts": 1,
            "live_post_authorized": True,
            "scheduler_repository_modification_authorized": False,
            "created_at_utc": _now(),
        }
    )
    return _write_immutable(output.resolve(), receipt)


def _load_activation(
    path: Path, *, watch_plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        ACTIVATION_SCHEMA,
        label="Full fast-lane activation receipt",
    )
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("watch_plan")
        != production._file_record(watch_plan_path.resolve(strict=True))
        or value.get("watch_plan_payload_sha256") != plan["payload_sha256"]
        or value.get("reviewer") != "root-reviewed"
        or value.get("maximum_scheduler_posts") != 1
        or value.get("live_post_authorized") is not True
        or value.get("scheduler_repository_modification_authorized") is not False
    ):
        raise HandoffContractError("Full fast-lane activation authority drifted")
    return value


def _terminal_receipt_path(manifest_path: Path) -> Path:
    directory = manifest_path.resolve(strict=True).parent
    return directory.parent / f"{directory.name}.receipt.json"


def _authenticate_terminal_authority(
    *, manifest_path: Path, receipt_path: Path
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    receipt = terminal._validate_seal(
        terminal._read_json(receipt_path), terminal.NDS_RECEIPT_SCHEMA
    )
    manifest, ranked, _by_candidate = promotion._load_truth_manifest(
        manifest_path
    )
    rank0 = [
        copy.deepcopy(row)
        for row in ranked
        if row["truth_non_dominated_rank"] == 0
    ]
    if (
        receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("truth_manifest")
        != production._file_record(manifest_path)
        or receipt.get("global_nondominated_sort_performed") is not True
        or receipt.get("scheduler_mutation_performed") is not False
        or not rank0
        or manifest.get("rank0_count") != len(rank0)
    ):
        raise HandoffContractError(
            "terminal feasible rank-0 promotion authority drifted"
        )
    return {
        "truth_manifest": production._file_record(manifest_path),
        "truth_manifest_payload_sha256": manifest["payload_sha256"],
        "promotion_receipt": production._file_record(receipt_path),
        "promotion_receipt_payload_sha256": receipt["payload_sha256"],
        "rank0_rows": rank0,
    }


def _discover_terminal_authority(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    state_path = Path(plan["terminal_state_path"]).resolve(strict=True)
    state = terminal._validate_seal(
        terminal._read_json(state_path), terminal.STATE_SCHEMA
    )
    if (
        state.get("campaign_id") != CAMPAIGN_ID
        or state.get("scheduler_methods_used") != ["GET"]
        or state.get("scheduler_submission_performed") is not False
        or state.get("scheduler_cancel_performed") is not False
        or state.get("scheduler_mutation_performed") is not False
    ):
        raise HandoffContractError("terminal watcher state authority drifted")
    latest = state.get("latest_truth_manifest")
    if latest is None:
        return None
    manifest_path = Path(str(latest)).resolve(strict=True)
    truth_root = (state_path.parent / "truth_snapshots").resolve(strict=True)
    if truth_root not in manifest_path.parents:
        raise HandoffContractError(
            "terminal latest truth manifest escaped truth snapshots"
        )
    receipt_path = _terminal_receipt_path(manifest_path)
    return _authenticate_terminal_authority(
        manifest_path=manifest_path, receipt_path=receipt_path
    )


def _selection_plan_path(selection: Mapping[str, Any]) -> Path:
    record = selection["full_plan"]
    path = Path(str(record.get("path") or "")).resolve(strict=True)
    if production._file_record(path) != record:
        raise HandoffContractError("Full fast-lane selected plan bytes drifted")
    return path


def _load_selection(path: Path) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        SELECTION_SCHEMA,
        label="Full fast-lane selection",
    )
    manifest_path = Path(value["truth_manifest"]["path"])
    receipt_path = Path(value["promotion_receipt"]["path"])
    authority = _authenticate_terminal_authority(
        manifest_path=manifest_path, receipt_path=receipt_path
    )
    plan_set_path = Path(value["full_plan_set"]["path"]).resolve(strict=True)
    if production._file_record(plan_set_path) != value["full_plan_set"]:
        raise HandoffContractError("Full fast-lane plan-set bytes drifted")
    plan_set, plans = promotion._load_full_plan_set(plan_set_path)
    plan_path = _selection_plan_path(value)
    full_plan = promotion._load_full_plan(plan_path)[0]
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("truth_manifest") != authority["truth_manifest"]
        or value.get("truth_manifest_payload_sha256")
        != authority["truth_manifest_payload_sha256"]
        or value.get("promotion_receipt") != authority["promotion_receipt"]
        or value.get("promotion_receipt_payload_sha256")
        != authority["promotion_receipt_payload_sha256"]
        or plan_set.get("requested_limit") != 1
        or plan_set.get("selected_plan_count") != 1
        or len(plans) != 1
        or value.get("full_plan_payload_sha256")
        != full_plan["payload_sha256"]
        or value.get("candidate_physics_sha256")
        != full_plan["candidate_physics_sha256"]
        or value.get("standard_task_id") != full_plan["standard_task_id"]
        or value.get("truth_non_dominated_rank") != 0
        or value.get("first_feasible_truth_frozen") is not True
        or value.get("full_plan_limit") != 1
        or value.get("scheduler_submission_performed") is not False
    ):
        raise HandoffContractError("Full fast-lane selection authority drifted")
    return value


def _ensure_priority_fence(
    *, plan: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(plan["priority_fence_path"]).resolve()
    selection_path = (
        Path(plan["output_root"]).resolve(strict=True) / "selection.json"
    )
    expected = _sealed(
        {
            "schema_version": PRIORITY_FENCE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "priority": "full_before_safe_refill",
            "truth_manifest": copy.deepcopy(selection["truth_manifest"]),
            "truth_manifest_payload_sha256": selection[
                "truth_manifest_payload_sha256"
            ],
            "feasible_rank0_promotion_receipt": copy.deepcopy(
                selection["promotion_receipt"]
            ),
            "feasible_rank0_promotion_receipt_payload_sha256": selection[
                "promotion_receipt_payload_sha256"
            ],
            "selection": production._file_record(selection_path),
            "selection_payload_sha256": selection["payload_sha256"],
            "candidate_physics_sha256": selection[
                "candidate_physics_sha256"
            ],
            "standard_task_id": selection["standard_task_id"],
            "truth_non_dominated_rank": 0,
            "safe_refill_logical_authority_task_ids": sorted(
                SAFE_REFILL_CANDIDATE_STEMS
            ),
            "safe_refill_next_post_blocked": True,
            "scheduler_cancel_performed": False,
            "scheduler_submission_performed": False,
            "scheduler_mutation_performed": False,
        }
    )
    if path.exists():
        observed = _validate_seal(
            _read_json(path),
            PRIORITY_FENCE_SCHEMA,
            label="Full-before-refill priority fence",
        )
        if observed != expected:
            raise HandoffContractError(
                "Full-before-refill priority fence belongs to another selection"
            )
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(path, expected)
    return expected


def _prepare_selection(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    root = Path(plan["output_root"])
    selection_path = root / "selection.json"
    if selection_path.exists():
        selected = _load_selection(selection_path)
        _ensure_priority_fence(plan=plan, selection=selected)
        return selected
    authority = _discover_terminal_authority(plan)
    if authority is None:
        return None
    manifest_path = Path(authority["truth_manifest"]["path"])
    plan_root = root / (
        f"full-plan-{authority['truth_manifest_payload_sha256'][:16]}"
    )
    plan_set_path = plan_root / "full_plan_set.json"
    if not plan_set_path.exists():
        manifest, _ranked, _by_candidate = promotion._load_truth_manifest(
            manifest_path
        )
        promotion.create_full_plans(
            truth_manifest_path=manifest_path,
            solver_revision=manifest["solver_revision"],
            library_revision=manifest["library_revision"],
            limit=1,
            output=plan_root,
        )
    plan_set, plans = promotion._load_full_plan_set(plan_set_path)
    if plan_set["selected_plan_count"] != 1 or len(plans) != 1:
        raise HandoffContractError("Full fast lane requires exactly one plan")
    record = plan_set["plans"][0]["plan"]
    full_plan_path = production._contained_file(
        plan_set_path.parent, record["path"], "Full fast-lane selected plan"
    )
    full_plan = promotion._load_full_plan(full_plan_path)[0]
    selection = _sealed(
        {
            "schema_version": SELECTION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "truth_manifest": authority["truth_manifest"],
            "truth_manifest_payload_sha256": authority[
                "truth_manifest_payload_sha256"
            ],
            "promotion_receipt": authority["promotion_receipt"],
            "promotion_receipt_payload_sha256": authority[
                "promotion_receipt_payload_sha256"
            ],
            "full_plan_set": production._file_record(plan_set_path),
            "full_plan_set_payload_sha256": plan_set["payload_sha256"],
            "full_plan": production._file_record(full_plan_path),
            "full_plan_payload_sha256": full_plan["payload_sha256"],
            "candidate_physics_sha256": full_plan[
                "candidate_physics_sha256"
            ],
            "standard_task_id": full_plan["standard_task_id"],
            "truth_non_dominated_rank": 0,
            "first_feasible_truth_frozen": True,
            "full_plan_limit": 1,
            "scheduler_submission_performed": False,
            "created_at_utc": _now(),
        }
    )
    _write_immutable(selection_path, selection)
    selected = _load_selection(selection_path)
    _ensure_priority_fence(plan=plan, selection=selected)
    return selected


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("state") or row.get("status") or "").strip().lower()


def _normalize_task(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _positive_int(
        row.get("task_id", row.get("id")), "Scheduler task ID"
    )
    return {
        "task_id": task_id,
        "id": task_id,
        "name": str(row.get("name") or ""),
        "status": str(row.get("status") or "").strip().lower(),
        "state": str(row.get("state") or "").strip().lower(),
        "project": str(row.get("project") or ""),
        "dedupe_key": str(row.get("dedupe_key") or ""),
        "account_name": str(
            row.get("account_name")
            or row.get("requested_account_name")
            or ""
        ),
        "cpus": row.get("cpus"),
        "memory_mb": row.get("memory_mb"),
        "timeout_seconds": row.get("timeout_seconds"),
        "aedt_backend": str(row.get("aedt_backend") or ""),
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
    }


def _active_project_tasks(rows: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError("active Scheduler inventory is absent")
    active = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise HandoffContractError("active Scheduler row is malformed")
        if (
            str(row.get("project") or "") == scheduler_client.MFT_PROJECT
            and _status(row) in ACTIVE_STATUSES
        ):
            active.append(_normalize_task(row))
    active.sort(key=lambda item: item["task_id"])
    return active


def _bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def _successful_standard_grid_stream_evidence(
    *,
    stdout: bytes,
    stderr: bytes,
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    limit = diagnostic.OPERATIONAL_ATTESTATION_MAX_STREAM_BYTES
    if (
        not isinstance(stdout, bytes)
        or not isinstance(stderr, bytes)
        or not stdout
        or not stderr
        or len(stdout) > limit
        or len(stderr) > limit
    ):
        raise HandoffContractError(
            "successful Standard terminal streams are absent or oversized"
        )
    try:
        out_text = stdout.decode("utf-8")
        err_text = stderr.decode("utf-8")
    except UnicodeError as exc:
        raise HandoffContractError(
            "successful Standard terminal streams are not UTF-8"
        ) from exc
    result_lines = [
        line
        for line in out_text.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    if len(result_lines) != 1:
        raise HandoffContractError(
            "successful Standard RESULT_JSON stream evidence is ambiguous"
        )
    try:
        result = json.loads(result_lines[0][len("RESULT_JSON ") :])
    except json.JSONDecodeError as exc:
        raise HandoffContractError(
            "successful Standard RESULT_JSON stream evidence is invalid"
        ) from exc
    marker = "[thermal] native mesh preflight: "
    preflight_lines = [
        line for line in err_text.splitlines() if marker in line
    ]
    if len(preflight_lines) != 1:
        raise HandoffContractError(
            "successful Standard native premesh evidence is ambiguous"
        )
    try:
        preflight = json.loads(preflight_lines[0].split(marker, 1)[1])
    except json.JSONDecodeError as exc:
        raise HandoffContractError(
            "successful Standard native premesh JSON is invalid"
        ) from exc
    artifacts = preflight.get("fresh_mesh_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HandoffContractError(
            "successful Standard fresh mesh inventory is absent"
        )
    grid_bytes = 0
    normalized = []
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise HandoffContractError(
                "successful Standard fresh mesh artifact is malformed"
            )
        size = _positive_int(
            raw.get("grid_output_size"),
            "successful Standard grid output size",
        )
        directory = str(raw.get("directory") or "")
        name = str(raw.get("name") or "")
        if (
            not directory.startswith(f"{SOLVER_TRANSIENT_ROOT}/")
            or not name
            or "/" in name
            or "\\" in name
            or production._require_sha(
                raw.get("grid_output_sha256_sample"),
                "successful Standard grid output sample SHA",
            )
            != raw.get("grid_output_sha256_sample")
        ):
            raise HandoffContractError(
                "successful Standard grid artifact identity drifted"
            )
        grid_bytes += size
        normalized.append(copy.deepcopy(dict(raw)))
    task_id = _positive_int(
        collection.get("task_id"), "successful Standard task ID"
    )
    if (
        collection.get("scheduler_status") != "completed"
        or collection.get("scheduler_task_execution", {}).get("state")
        != "succeeded"
        or collection.get("scheduler_task_execution", {}).get("exit_code")
        != 0
        or result != collection.get("result")
        or canonical_sha256(result) != collection.get("result_sha256")
        or preflight.get("schema") != "thermal-mesh-preflight-v2"
        or preflight.get("passed") is not True
        or preflight.get("status")
        != "passed_standalone_native_premesh"
        or preflight.get("mesh_mapping_coverage_passed") is not True
        or preflight.get("mesh_artifact_readback_passed") is not True
        or preflight.get("native_operation_readback_passed") is not True
        or preflight.get("standalone_idle_barrier_passed") is not True
        or preflight.get("postflight_error") != ""
        or preflight.get("native_errors") != []
        or preflight.get("fresh_mesh_artifact_count") != len(normalized)
        or grid_bytes <= 0
        or str(result.get("solver_core_scheduler_task_id_readback") or "")
        != str(task_id)
        or result.get("solver_num_cores_requested") != 8
        or result.get("solver_num_cores_effective") != 8
        or result.get("full_model") != 0
        or result.get("thermal_symmetry") != "eighth"
    ):
        raise HandoffContractError(
            "successful Standard fresh-grid stream authority drifted"
        )
    return {
        "schema_version": (
            "mft-goal-successful-standard-grid-stream-evidence-v1"
        ),
        "task_id": task_id,
        "stdout_sha256": production._sha256_bytes(stdout),
        "stdout_size_bytes": len(stdout),
        "stderr_sha256": production._sha256_bytes(stderr),
        "stderr_size_bytes": len(stderr),
        "result_sha256": collection["result_sha256"],
        "native_premesh_sha256": canonical_sha256(preflight),
        "native_premesh_schema": "thermal-mesh-preflight-v2",
        "fresh_mesh_artifact_count": len(normalized),
        "fresh_grid_output_bytes": grid_bytes,
        "solver_transient_root": SOLVER_TRANSIENT_ROOT,
        "source_full_model": 0,
        "source_thermal_symmetry": "eighth",
    }


def _successful_grid_authority_payload(
    *,
    collection_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    resolved_collection = collection_path.resolve(strict=True)
    authenticated = diagnostic.authenticate_collection(
        resolved_collection
    )
    collection = authenticated["collection"]
    plan = authenticated["plan"]
    submission = authenticated["submission"]
    stdout_path = stdout_path.resolve(strict=True)
    stderr_path = stderr_path.resolve(strict=True)
    evidence = _successful_standard_grid_stream_evidence(
        stdout=stdout_path.read_bytes(),
        stderr=stderr_path.read_bytes(),
        collection=collection,
    )
    if (
        collection.get("plan") != production._file_record(
            Path(collection["plan"]["path"])
        )
        or collection.get("submission") != production._file_record(
            Path(collection["submission"]["path"])
        )
        or collection.get("task_id") != submission["task_id"]
        or collection.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
    ):
        raise HandoffContractError(
            "successful Standard grid collection lineage drifted"
        )
    return {
        "schema_version": SUCCESS_GRID_AUTHORITY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "standard_collection": production._file_record(
            resolved_collection
        ),
        "standard_collection_payload_sha256": collection["payload_sha256"],
        "source_standard_plan": copy.deepcopy(collection["plan"]),
        "source_standard_plan_payload_sha256": plan["payload_sha256"],
        "source_standard_submission": copy.deepcopy(
            collection["submission"]
        ),
        "source_standard_submission_payload_sha256": submission[
            "payload_sha256"
        ],
        "standard_task_id": collection["task_id"],
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "stdout": production._file_record(stdout_path),
        "stderr": production._file_record(stderr_path),
        "stream_evidence": evidence,
        "stream_evidence_sha256": canonical_sha256(evidence),
        "fresh_grid_output_bytes": evidence["fresh_grid_output_bytes"],
        "source": "terminal_success_scheduler_GET_native_premesh_streams",
        "scheduler_methods_used": ["GET"],
        "scheduler_submission_performed": False,
        "scheduler_cancel_performed": False,
        "scheduler_mutation_performed": False,
        "retained_bytes_used": False,
    }


def create_successful_standard_grid_authority(
    *,
    collection_path: Path,
    output: Path,
    stream_reader: Any = diagnostic._scheduler_operational_stream,
) -> Path:
    authenticated = diagnostic.authenticate_collection(
        collection_path.resolve(strict=True)
    )
    collection = authenticated["collection"]
    task_id = _positive_int(
        collection.get("task_id"), "successful Standard task ID"
    )
    scheduler_url = str(collection.get("scheduler_url") or "")
    first_stdout = stream_reader(
        scheduler_url=scheduler_url, task_id=task_id, stream="stdout"
    )
    first_stderr = stream_reader(
        scheduler_url=scheduler_url, task_id=task_id, stream="stderr"
    )
    second_stdout = stream_reader(
        scheduler_url=scheduler_url, task_id=task_id, stream="stdout"
    )
    second_stderr = stream_reader(
        scheduler_url=scheduler_url, task_id=task_id, stream="stderr"
    )
    if (
        first_stdout != second_stdout
        or first_stderr != second_stderr
        or not isinstance(first_stdout, bytes)
        or not isinstance(first_stderr, bytes)
    ):
        raise HandoffContractError(
            "successful Standard terminal streams changed during GET capture"
        )
    path = output.resolve()
    stdout_path = path.with_suffix(".stdout.log")
    stderr_path = path.with_suffix(".stderr.log")
    _write_immutable_bytes(stdout_path, first_stdout)
    _write_immutable_bytes(stderr_path, first_stderr)
    value = _sealed(
        {
            **_successful_grid_authority_payload(
                collection_path=collection_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ),
            "created_at_utc": _now(),
        }
    )
    if path.exists():
        observed = _load_successful_standard_grid_authority(path)
        comparable = {
            key: item
            for key, item in observed.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        expected = {
            key: item
            for key, item in value.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        if comparable != expected:
            raise HandoffContractError(
                "existing successful Standard grid authority differs"
            )
        return path
    return _write_immutable(path, value)


def _load_successful_standard_grid_authority(
    path: Path,
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        SUCCESS_GRID_AUTHORITY_SCHEMA,
        label="successful Standard grid authority",
    )
    for name in ("standard_collection", "stdout", "stderr"):
        record = value.get(name)
        if not isinstance(record, Mapping):
            raise HandoffContractError(
                f"successful Standard grid {name} record is absent"
            )
        if production._file_record(
            Path(str(record.get("path") or ""))
        ) != record:
            raise HandoffContractError(
                f"successful Standard grid {name} bytes drifted"
            )
    expected = _successful_grid_authority_payload(
        collection_path=Path(value["standard_collection"]["path"]),
        stdout_path=Path(value["stdout"]["path"]),
        stderr_path=Path(value["stderr"]["path"]),
    )
    if any(value.get(key) != item for key, item in expected.items()):
        raise HandoffContractError(
            "successful Standard grid authority drifted"
        )
    return value


def _authenticated_standard_grid_evidence(
    source_plan_path: Path,
    *,
    successful_grid_authority_path: Path | None = None,
) -> dict[str, Any]:
    """Reauthenticate an actual fresh Standard grid, never retained bytes."""
    resolved = source_plan_path.resolve(strict=True)
    raw = _read_json(resolved)
    if raw.get("schema_version") != timeout12h.PLAN_SCHEMA:
        source_plan, _params, _selected = (
            diagnostic._load_collectible_plan(resolved)
        )
        stage = source_plan.get("stage")
        retained = stage.get("retained_aedt_bundle") if isinstance(
            stage, Mapping
        ) else None
        if successful_grid_authority_path is None:
            raise HandoffContractError(
                "source Standard successor has no embedded fresh-grid "
                "evidence and no terminal-success stream authority"
            )
        authority = _load_successful_standard_grid_authority(
            successful_grid_authority_path
        )
        try:
            logical_id = terminal._logical_authority_id(source_plan)
        except Exception as exc:
            raise HandoffContractError(
                "source Standard successor logical lineage is unsupported"
            ) from exc
        if (
            not isinstance(stage, Mapping)
            or stage.get("name") != "standard"
            or stage.get("full_model") != 0
            or stage.get("thermal_symmetry") != "eighth"
            or not isinstance(retained, Mapping)
            or authority.get("source_standard_plan")
            != production._file_record(resolved)
            or authority.get("source_standard_plan_payload_sha256")
            != source_plan["payload_sha256"]
            or authority.get("candidate_physics_sha256")
            != source_plan["candidate_physics_sha256"]
            or authority.get("standard_task_id")
            != authority.get("stream_evidence", {}).get("task_id")
        ):
            raise HandoffContractError(
                "source Standard terminal-success grid lineage drifted"
            )
        grid_bytes = _positive_int(
            authority.get("fresh_grid_output_bytes"),
            "successful Standard fresh grid bytes",
        )
        return {
            "source_standard_plan": production._file_record(resolved),
            "source_standard_plan_payload_sha256": source_plan[
                "payload_sha256"
            ],
            "candidate_physics_sha256": source_plan[
                "candidate_physics_sha256"
            ],
            "logical_authority_task_id": logical_id,
            "task_name": stage["task_name"],
            "dedupe_key": retained["dedupe_key"],
            "fresh_grid_output_bytes": grid_bytes,
            "fresh_grid_output_gib": _bytes_to_gib(grid_bytes),
            "stream_evidence_sha256": authority[
                "stream_evidence_sha256"
            ],
            "storage_audit_sha256": None,
            "fresh_grid_provenance": {
                "kind": "terminal_success_native_premesh_stream",
                "authority": production._file_record(
                    successful_grid_authority_path.resolve(strict=True)
                ),
                "authority_payload_sha256": authority["payload_sha256"],
                "scheduler_methods_used": ["GET"],
                "retained_bytes_used": False,
            },
            "source_stage_full_model": 0,
            "source_stage_thermal_symmetry": "eighth",
        }
    source_plan, _params, _selected, _parent = timeout12h._load_plan(resolved)
    retry = source_plan.get("retry_of_timeout12h")
    stage = source_plan.get("stage")
    if not isinstance(retry, Mapping) or not isinstance(stage, Mapping):
        raise HandoffContractError(
            "source Standard fresh-grid lineage is absent"
        )
    stream = retry.get("stream_evidence")
    storage = retry.get("storage_audit")
    if not isinstance(stream, Mapping) or not isinstance(storage, Mapping):
        raise HandoffContractError(
            "source Standard fresh-grid/storage evidence is absent"
        )
    grid_bytes = _positive_int(
        stream.get("fresh_grid_output_bytes"),
        "source Standard fresh grid bytes",
    )
    prospective = _nonnegative_float(
        storage.get("prospective_grid_gb"),
        "source Standard prospective grid",
    )
    retained = stage.get("retained_aedt_bundle")
    logical_id = retry.get("logical_authority_task_id")
    if (
        stage.get("name") != "standard"
        or stage.get("full_model") != 0
        or stage.get("thermal_symmetry") != "eighth"
        or not isinstance(retained, Mapping)
        or not str(retained.get("dedupe_key") or "")
        or isinstance(logical_id, bool)
        or not isinstance(logical_id, int)
        or logical_id <= 0
        or retry.get("stream_evidence_sha256")
        != canonical_sha256(stream)
        or retry.get("storage_audit_sha256")
        != canonical_sha256(storage)
        or not math.isclose(
            prospective,
            _bytes_to_gib(grid_bytes),
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise HandoffContractError(
            "source Standard fresh-grid authority drifted"
        )
    return {
        "source_standard_plan": production._file_record(resolved),
        "source_standard_plan_payload_sha256": source_plan["payload_sha256"],
        "candidate_physics_sha256": source_plan[
            "candidate_physics_sha256"
        ],
        "logical_authority_task_id": logical_id,
        "task_name": stage["task_name"],
        "dedupe_key": retained["dedupe_key"],
        "fresh_grid_output_bytes": grid_bytes,
        "fresh_grid_output_gib": _bytes_to_gib(grid_bytes),
        "stream_evidence_sha256": retry["stream_evidence_sha256"],
        "storage_audit_sha256": retry["storage_audit_sha256"],
        "fresh_grid_provenance": {
            "kind": "timeout12h_embedded_native_premesh_stream",
            "source_plan": production._file_record(resolved),
            "stream_evidence_sha256": retry[
                "stream_evidence_sha256"
            ],
            "storage_audit_sha256": retry["storage_audit_sha256"],
            "retained_bytes_used": False,
        },
        "source_stage_full_model": 0,
        "source_stage_thermal_symmetry": "eighth",
    }


def create_standard_storage_bound_authority(
    *,
    source_plan_path: Path,
    task_id: int,
    output: Path,
) -> Path:
    """Seal a live Standard task bound from its authenticated source grid."""
    task_id = _positive_int(task_id, "active Standard task ID")
    evidence = _authenticated_standard_grid_evidence(source_plan_path)
    value = _sealed(
        {
            "schema_version": STANDARD_STORAGE_BOUND_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "task_id": task_id,
            **evidence,
            "standard_storage_bound_bytes": evidence[
                "fresh_grid_output_bytes"
            ],
            "standard_storage_bound_gib": evidence[
                "fresh_grid_output_gib"
            ],
            "bound_method": (
                "authenticated_same-geometry_standard_fresh_grid_bytes"
            ),
            "retained_bytes_used": False,
            "generic_reservation_used": False,
            "created_at_utc": _now(),
        }
    )
    path = output.resolve()
    if path.exists():
        observed = _load_standard_storage_bound(path)
        comparable = {
            key: item
            for key, item in observed.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        expected = {
            key: item
            for key, item in value.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        if comparable != expected:
            raise HandoffContractError(
                "existing Standard storage-bound authority differs"
            )
        return path
    return _write_immutable(path, value)


def _load_standard_storage_bound(
    path: Path,
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        STANDARD_STORAGE_BOUND_SCHEMA,
        label="Standard active storage-bound authority",
    )
    source_record = value.get("source_standard_plan")
    if not isinstance(source_record, Mapping):
        raise HandoffContractError(
            "Standard active storage source plan is absent"
        )
    evidence = _authenticated_standard_grid_evidence(
        Path(str(source_record.get("path") or ""))
    )
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or any(value.get(key) != item for key, item in evidence.items())
        or value.get("standard_storage_bound_bytes")
        != evidence["fresh_grid_output_bytes"]
        or value.get("standard_storage_bound_gib")
        != evidence["fresh_grid_output_gib"]
        or value.get("bound_method")
        != "authenticated_same-geometry_standard_fresh_grid_bytes"
        or value.get("retained_bytes_used") is not False
        or value.get("generic_reservation_used") is not False
    ):
        raise HandoffContractError(
            "Standard active storage-bound authority drifted"
        )
    task_id = _positive_int(value.get("task_id"), "active Standard task ID")
    if task is not None and (
        task_id != task.get("task_id")
        or value.get("task_name") != task.get("name")
        or value.get("dedupe_key") != task.get("dedupe_key")
    ):
        raise HandoffContractError(
            "Standard storage bound does not match the active task"
        )
    return value


def _full_storage_authority_payload(
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
    successful_grid_authority_path: Path | None = None,
) -> dict[str, Any]:
    authenticated_selection = _load_selection(selection_path)
    if (
        authenticated_selection.get("payload_sha256")
        != selection.get("payload_sha256")
    ):
        raise HandoffContractError(
            "Full storage authority selection bytes drifted"
        )
    full_plan_path = _selection_plan_path(selection)
    full_plan, _params, _profile, _view, _truth = (
        promotion._load_full_plan(full_plan_path)
    )
    collection_record = full_plan["standard_collection"]
    collection_path = Path(str(collection_record.get("path") or ""))
    if production._file_record(collection_path) != collection_record:
        raise HandoffContractError(
            "Full storage source collection bytes drifted"
        )
    authenticated = diagnostic.authenticate_collection(collection_path)
    collection = authenticated["collection"]
    source_plan = authenticated["plan"]
    source_plan_record = collection.get("plan")
    if not isinstance(source_plan_record, Mapping):
        raise HandoffContractError(
            "Full storage source Standard plan is absent"
        )
    source_plan_path = Path(str(source_plan_record.get("path") or ""))
    evidence = _authenticated_standard_grid_evidence(
        source_plan_path,
        successful_grid_authority_path=successful_grid_authority_path,
    )
    source_bytes = evidence["fresh_grid_output_bytes"]
    full_bound_bytes = source_bytes * FULL_SYMMETRY_EXPANSION_FACTOR
    if (
        collection.get("task_id") != full_plan["standard_task_id"]
        or collection.get("payload_sha256")
        != full_plan["standard_collection_payload_sha256"]
        or collection.get("plan") != evidence["source_standard_plan"]
        or source_plan.get("payload_sha256")
        != evidence["source_standard_plan_payload_sha256"]
        or source_plan.get("candidate_physics_sha256")
        != full_plan["candidate_physics_sha256"]
        or evidence["candidate_physics_sha256"]
        != selection["candidate_physics_sha256"]
    ):
        raise HandoffContractError(
            "Full storage authority escaped selected Standard lineage"
        )
    return {
        "schema_version": FULL_STORAGE_AUTHORITY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "selection": production._file_record(
            selection_path.resolve(strict=True)
        ),
        "selection_payload_sha256": selection["payload_sha256"],
        "full_plan": production._file_record(full_plan_path),
        "full_plan_payload_sha256": full_plan["payload_sha256"],
        "standard_collection": production._file_record(collection_path),
        "standard_collection_payload_sha256": full_plan[
            "standard_collection_payload_sha256"
        ],
        "standard_task_id": full_plan["standard_task_id"],
        **evidence,
        "full_symmetry_expansion_factor": (
            FULL_SYMMETRY_EXPANSION_FACTOR
        ),
        "standard_symmetry_denominator": STANDARD_SYMMETRY_DENOMINATOR,
        "full_prospective_storage_bound_bytes": full_bound_bytes,
        "full_prospective_storage_bound_gib": _bytes_to_gib(
            full_bound_bytes
        ),
        "minimum_post_reservation_free_floor_gib": MINIMUM_GPFS_FREE_GB,
        "solver_transient_root": SOLVER_TRANSIENT_ROOT,
        "solver_transient_route_minimum_gib": (
            SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        ),
        "full_bound_plus_working_floor_gib": (
            _bytes_to_gib(full_bound_bytes) + MINIMUM_GPFS_FREE_GB
        ),
        "static_transient_route_bound_passed": (
            _bytes_to_gib(full_bound_bytes) + MINIMUM_GPFS_FREE_GB
            <= SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        ),
        "bound_method": (
            "authenticated_standard_fresh_grid_bytes_times_"
            "full_symmetry_expansion_factor"
        ),
        "retained_bytes_used": False,
        "generic_reservation_used": False,
    }


def create_full_storage_authority(
    *,
    selection_path: Path,
    output: Path,
    stream_reader: Any = diagnostic._scheduler_operational_stream,
) -> Path:
    selection_path = selection_path.resolve(strict=True)
    selection = _load_selection(selection_path)
    successful_grid_path: Path | None = None
    try:
        payload = _full_storage_authority_payload(
            selection_path=selection_path, selection=selection
        )
    except HandoffContractError:
        full_plan_path = _selection_plan_path(selection)
        full_plan = promotion._load_full_plan(full_plan_path)[0]
        collection_path = Path(full_plan["standard_collection"]["path"])
        successful_grid_path = output.resolve().with_name(
            f"{output.resolve().stem}.successful-standard-grid.json"
        )
        try:
            create_successful_standard_grid_authority(
                collection_path=collection_path,
                output=successful_grid_path,
                stream_reader=stream_reader,
            )
            payload = _full_storage_authority_payload(
                selection_path=selection_path,
                selection=selection,
                successful_grid_authority_path=successful_grid_path,
            )
        except Exception as stream_error:
            raise HandoffContractError(
                "Full storage source has neither authenticated embedded "
                "fresh-grid evidence nor a valid terminal-success native "
                "premesh stream authority"
            ) from stream_error
    value = _sealed(
        {
            **payload,
            "created_at_utc": _now(),
        }
    )
    path = output.resolve()
    if path.exists():
        observed = _load_full_storage_authority(
            path, selection=selection
        )
        comparable = {
            key: item
            for key, item in observed.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        expected = {
            key: item
            for key, item in value.items()
            if key not in {"payload_sha256", "created_at_utc"}
        }
        if comparable != expected:
            raise HandoffContractError(
                "existing Full storage authority differs"
            )
        return path
    return _write_immutable(path, value)


def _load_full_storage_authority(
    path: Path, *, selection: Mapping[str, Any]
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        FULL_STORAGE_AUTHORITY_SCHEMA,
        label="Full prospective-storage authority",
    )
    selection_record = value.get("selection")
    if not isinstance(selection_record, Mapping):
        raise HandoffContractError(
            "Full prospective-storage selection is absent"
        )
    selection_path = Path(str(selection_record.get("path") or ""))
    provenance = value.get("fresh_grid_provenance")
    successful_grid_path = None
    if (
        isinstance(provenance, Mapping)
        and provenance.get("kind")
        == "terminal_success_native_premesh_stream"
    ):
        authority_record = provenance.get("authority")
        if not isinstance(authority_record, Mapping):
            raise HandoffContractError(
                "Full prospective-storage stream authority is absent"
            )
        successful_grid_path = Path(
            str(authority_record.get("path") or "")
        )
    expected = _full_storage_authority_payload(
        selection_path=selection_path,
        selection=selection,
        successful_grid_authority_path=successful_grid_path,
    )
    if (
        value.get("selection_payload_sha256")
        != selection["payload_sha256"]
        or any(value.get(key) != item for key, item in expected.items())
    ):
        raise HandoffContractError(
            "Full prospective-storage authority drifted"
        )
    return value


def _latest_full_storage_authority(
    directory: Path, *, selection: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in directory.resolve(strict=True).rglob("*.json"):
        try:
            value = _load_full_storage_authority(path, selection=selection)
        except HandoffContractError:
            continue
        candidates.append((path.resolve(strict=True), value))
    if len(candidates) != 1:
        raise HandoffContractError(
            "exactly one matching Full prospective-storage authority "
            "is required"
        )
    return candidates[0]


def _ensure_full_storage_authority(
    plan: Mapping[str, Any], selection: Mapping[str, Any]
) -> Path:
    directory = Path(plan["full_storage_authority_directory"]).resolve(
        strict=True
    )
    path = directory / (
        "full-"
        f"{selection['candidate_physics_sha256'][:12]}-"
        f"{selection['payload_sha256'][:16]}.json"
    )
    return create_full_storage_authority(
        selection_path=Path(plan["output_root"]) / "selection.json",
        output=path,
    )


def build_gpfs_audit(
    *,
    observed_at_utc: str,
    account_observations: Sequence[Mapping[str, Any]],
    rows: Any,
    active_task_storage_authority_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    _aware(observed_at_utc, "GPFS audit")
    if not account_observations:
        raise HandoffContractError("GPFS account observations are absent")
    observations: dict[str, dict[str, Any]] = {}
    for raw in account_observations:
        if not isinstance(raw, Mapping):
            raise HandoffContractError("GPFS account observation is malformed")
        account = str(raw.get("account_name") or "").strip()
        used = _nonnegative_float(raw.get("block_used_gb"), "GPFS block used")
        in_doubt = _nonnegative_float(
            raw.get("block_in_doubt_gb"), "GPFS block in-doubt"
        )
        limit = _nonnegative_float(raw.get("block_limit_gb"), "GPFS block limit")
        if (
            not account
            or account in observations
            or str(raw.get("filesystem_type") or "").lower() != "gpfs"
            or not str(raw.get("fileset_name") or "").strip()
            or raw.get("user_quota_scope") != "filesystem"
            or raw.get("quota_type") != "USR"
            or limit <= 0
            or used + in_doubt > limit
        ):
            raise HandoffContractError("GPFS account identity drifted")
        observations[account] = {
            "account_name": account,
            "filesystem_type": "gpfs",
            "fileset_name": str(raw["fileset_name"]),
            "user_quota_scope": "filesystem",
            "quota_type": "USR",
            "block_used_gb": used,
            "block_in_doubt_gb": in_doubt,
            "block_limit_gb": limit,
        }
    active = _active_project_tasks(rows)
    authority_by_task: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in active_task_storage_authority_paths:
        path = Path(raw_path).resolve(strict=True)
        authority = _load_standard_storage_bound(path)
        task_id = _positive_int(
            authority.get("task_id"), "active Standard bound task ID"
        )
        if task_id in authority_by_task:
            raise HandoffContractError(
                "duplicate active Standard storage-bound authority"
            )
        authority_by_task[task_id] = (path, authority)
    active_ids = {item["task_id"] for item in active}
    if set(authority_by_task) - active_ids:
        raise HandoffContractError(
            "active Standard storage-bound authority has no live task"
        )
    conflicts = []
    bounded = []
    reservations = {account: 0 for account in observations}
    for task in active:
        account = task["account_name"]
        timeout = task["timeout_seconds"]
        if (
            not account
            or account not in observations
            or isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout <= 0
        ):
            conflicts.append(
                {
                    "reason": "active_task_account_or_timeout_unbounded",
                    "task": copy.deepcopy(task),
                }
            )
            continue
        authority_item = authority_by_task.get(task["task_id"])
        if authority_item is None:
            conflicts.append(
                {
                    "reason": "active_task_storage_authority_missing",
                    "task": copy.deepcopy(task),
                }
            )
            continue
        authority_path, authority = authority_item
        try:
            _load_standard_storage_bound(authority_path, task=task)
        except HandoffContractError:
            conflicts.append(
                {
                    "reason": "active_task_storage_authority_mismatch",
                    "task": copy.deepcopy(task),
                    "authority": production._file_record(authority_path),
                }
            )
            continue
        bound_bytes = authority["standard_storage_bound_bytes"]
        reservations[account] += bound_bytes
        bounded.append(
            {
                "task_id": task["task_id"],
                "account_name": account,
                "storage_bound_bytes": bound_bytes,
                "storage_bound_gib": _bytes_to_gib(bound_bytes),
                "authority": production._file_record(authority_path),
                "authority_payload_sha256": authority["payload_sha256"],
            }
        )
    accounts = []
    for account, observation in sorted(observations.items()):
        active_bytes = reservations[account]
        active_gib = _bytes_to_gib(active_bytes)
        observed_free = (
            observation["block_limit_gb"]
            - observation["block_used_gb"]
            - observation["block_in_doubt_gb"]
        )
        free_after_active = observed_free - active_gib
        bounded_count = sum(
            1 for item in bounded if item["account_name"] == account
        )
        accounts.append(
            {
                **observation,
                "observed_free_gb": observed_free,
                "active_bounded_task_count": bounded_count,
                "active_storage_bound_bytes": active_bytes,
                "active_storage_bound_gib": active_gib,
                "free_after_active_gb": free_after_active,
                "minimum_free_floor_gb": MINIMUM_GPFS_FREE_GB,
                "active_only_arithmetic_passed": (
                    free_after_active >= MINIMUM_GPFS_FREE_GB
                ),
            }
        )
    passed = (
        not conflicts
        and any(item["active_only_arithmetic_passed"] for item in accounts)
    )
    return _sealed(
        {
            "schema_version": GPFS_AUDIT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "observed_at_utc": observed_at_utc,
            "source": "read_only_gpfs_user_quota_probe",
            "account_observations": accounts,
            "active_project_tasks": active,
            "active_project_task_count": len(active),
            "active_task_storage_bounds": bounded,
            "active_task_storage_authorities": [
                {
                    "task_id": task_id,
                    "authority": production._file_record(path),
                    "authority_payload_sha256": authority[
                        "payload_sha256"
                    ],
                }
                for task_id, (path, authority) in sorted(
                    authority_by_task.items()
                )
            ],
            "active_unbounded_storage_conflicts": conflicts,
            "active_unbounded_storage_conflict_count": len(conflicts),
            "active_storage_bound_source": (
                "per-task_authenticated_fresh_grid_output_bytes"
            ),
            "candidate_storage_bound_included": False,
            "generic_per_task_storage_reservation_used": False,
            "retained_bytes_used": False,
            "minimum_free_floor_gb": MINIMUM_GPFS_FREE_GB,
            "arithmetic_passed": passed,
            "scheduler_mutation_performed": False,
        }
    )


def _load_gpfs_audit(
    path: Path,
    *,
    require_fresh: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    audit = _validate_seal(
        _read_json(path.resolve(strict=True)),
        GPFS_AUDIT_SCHEMA,
        label="Full fast-lane GPFS audit",
    )
    observations = audit.get("account_observations")
    if not isinstance(observations, list):
        raise HandoffContractError("Full fast-lane GPFS accounts are absent")
    raw_observations = [
        {
            key: item[key]
            for key in (
                "account_name",
                "filesystem_type",
                "fileset_name",
                "user_quota_scope",
                "quota_type",
                "block_used_gb",
                "block_in_doubt_gb",
                "block_limit_gb",
            )
        }
        for item in observations
        if isinstance(item, Mapping)
    ]
    rebuilt = build_gpfs_audit(
        observed_at_utc=str(audit.get("observed_at_utc") or ""),
        account_observations=raw_observations,
        rows=audit.get("active_project_tasks"),
        active_task_storage_authority_paths=[
            Path(str(item["authority"]["path"]))
            for item in audit.get("active_task_storage_authorities") or []
            if isinstance(item, Mapping)
            and isinstance(item.get("authority"), Mapping)
        ],
    )
    if rebuilt != audit:
        raise HandoffContractError("Full fast-lane GPFS audit drifted")
    if (
        audit.get("arithmetic_passed") is not True
        or audit.get("active_unbounded_storage_conflict_count") != 0
    ):
        raise HandoffContractError(
            "Full fast-lane GPFS audit has an unbounded storage conflict"
        )
    if require_fresh:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age = (
            current - _aware(audit["observed_at_utc"], "GPFS audit")
        ).total_seconds()
        if age < -30 or age > MAX_GPFS_AUDIT_AGE_SECONDS:
            raise HandoffContractError("Full fast-lane GPFS audit is stale")
    return audit


def _latest_gpfs_audit(
    directory: Path, *, now: datetime | None = None
) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in directory.resolve(strict=True).rglob("*.json"):
        try:
            value = _load_gpfs_audit(path, require_fresh=True, now=now)
        except HandoffContractError:
            continue
        candidates.append(
            (_aware(value["observed_at_utc"], "GPFS audit"), path, value)
        )
    if not candidates:
        raise HandoffContractError("no fresh passing GPFS audit is available")
    _stamp, path, value = max(candidates, key=lambda item: item[0])
    return path, value


def _latest_license_snapshot(
    directory: Path, *, now: datetime | None = None
) -> Path:
    candidates = []
    for path in directory.resolve(strict=True).rglob("snapshot.json"):
        try:
            production._validate_license_snapshot(path, now=now)
            checked = _aware(
                _read_json(path).get("checked_at"), "16-core license snapshot"
            )
        except HandoffContractError:
            continue
        candidates.append((checked, path.resolve(strict=True)))
    if not candidates:
        raise HandoffContractError(
            "no fresh passing 16-core license snapshot is available"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _default_live_reader(*, scheduler_url: str, endpoint: str) -> Any:
    return diagnostic._default_scheduler_live_reader(
        scheduler_url=scheduler_url, endpoint=endpoint
    )


def _default_task_list_reader(*, scheduler_url: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        [
            ("paged", "true"),
            ("page", "1"),
            ("page_size", "10000"),
            ("project", scheduler_client.MFT_PROJECT),
            *[
                ("status", status)
                for status in SCHEDULER_ACTIVE_STATUS_FILTERS
            ],
        ]
    )
    endpoint = f"/api/tasks?{query}"
    value = _default_live_reader(
        scheduler_url=scheduler_url, endpoint=endpoint
    )
    if not isinstance(value, Mapping) or not isinstance(
        value.get("items"), list
    ):
        raise HandoffContractError("paged active Scheduler inventory is absent")
    items = value["items"]
    if (
        value.get("filtered_total") != len(items)
        or value.get("page_count") != 1
        or value.get("has_next") is not False
    ):
        raise HandoffContractError("active Scheduler inventory is incomplete")
    return copy.deepcopy(items)


def _capacity_endpoint(account_name: str) -> str:
    query = urllib.parse.urlencode(
        {
            "cpus": FULL_RESOURCES["cpus"],
            "memory_mb": FULL_RESOURCES["memory_mb"],
            "scheduling_profile": "standard",
            "aedt_backend": "standalone",
            "project": scheduler_client.MFT_PROJECT,
            "account_name": account_name,
            "partition": "auto",
        }
    )
    return f"/api/task-capacity?{query}"


def _require_account_capacity(
    *,
    scheduler_url: str,
    account_name: str,
    live_reader: Any = _default_live_reader,
) -> dict[str, Any]:
    value = live_reader(
        scheduler_url=scheduler_url,
        endpoint=_capacity_endpoint(account_name),
    )
    if not isinstance(value, Mapping):
        raise HandoffContractError("Full fast-lane capacity response is absent")
    fit = value.get("fit_slots")
    ready = value.get("ready_fit_slots")
    allocations = value.get("allocations")
    matching = (
        [
            item
            for item in allocations
            if isinstance(item, Mapping)
            and item.get("account_name") == account_name
            and item.get("state") in {"warm", "active"}
            and item.get("memory_pressure_state") == "ok"
            and isinstance(item.get("fit_slots"), int)
            and not isinstance(item.get("fit_slots"), bool)
            and item["fit_slots"] >= 1
            and isinstance(item.get("free_cpus"), int)
            and item["free_cpus"] >= FULL_RESOURCES["cpus"]
            and isinstance(item.get("free_memory_mb"), int)
            and item["free_memory_mb"] >= FULL_RESOURCES["memory_mb"]
        ]
        if isinstance(allocations, list)
        else []
    )
    if (
        value.get("memory_pressure_state") != "ok"
        or isinstance(fit, bool)
        or not isinstance(fit, int)
        or fit < 1
        or isinstance(ready, bool)
        or not isinstance(ready, int)
        or ready < 1
        or not matching
    ):
        raise HandoffContractError(
            f"no ready 16-core/98304MB capacity on audited account {account_name}"
        )
    return copy.deepcopy(dict(value))


def _assert_audit_matches_live(
    audit: Mapping[str, Any], rows: Any
) -> list[dict[str, Any]]:
    live = _active_project_tasks(rows)
    if live != audit.get("active_project_tasks"):
        raise HandoffContractError(
            "active storage inventory changed after GPFS audit"
        )
    return live


def _active_safe_refill_tasks(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    matches = []
    for row in rows:
        name = str(row.get("name") or "")
        if any(
            stem in name or f"-l{logical}-" in name or f"-t{logical}-" in name
            for logical, stem in SAFE_REFILL_CANDIDATE_STEMS.items()
        ):
            matches.append(copy.deepcopy(dict(row)))
    return matches


def _storage_admission_accounts(
    *,
    audit: Mapping[str, Any],
    full_storage: Mapping[str, Any],
    required_account: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_bound_bytes = _positive_int(
        full_storage.get("full_prospective_storage_bound_bytes"),
        "Full prospective storage bound",
    )
    full_bound_gib = _nonnegative_float(
        full_storage.get("full_prospective_storage_bound_gib"),
        "Full prospective storage bound",
    )
    source_bytes = _positive_int(
        full_storage.get("fresh_grid_output_bytes"),
        "source Standard fresh grid bytes",
    )
    if (
        full_storage.get("full_symmetry_expansion_factor")
        != FULL_SYMMETRY_EXPANSION_FACTOR
        or full_storage.get("standard_symmetry_denominator")
        != STANDARD_SYMMETRY_DENOMINATOR
        or full_storage.get("minimum_post_reservation_free_floor_gib")
        != MINIMUM_GPFS_FREE_GB
        or full_storage.get("solver_transient_root")
        != SOLVER_TRANSIENT_ROOT
        or full_storage.get("solver_transient_route_minimum_gib")
        != SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        or full_storage.get("full_bound_plus_working_floor_gib")
        != full_bound_gib + MINIMUM_GPFS_FREE_GB
        or full_storage.get("static_transient_route_bound_passed")
        is not (
            full_bound_gib + MINIMUM_GPFS_FREE_GB
            <= SOLVER_TRANSIENT_ROUTE_MINIMUM_GIB
        )
        or full_bound_bytes
        != source_bytes * FULL_SYMMETRY_EXPANSION_FACTOR
        or not math.isclose(
            full_bound_gib,
            _bytes_to_gib(full_bound_bytes),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or full_storage.get("retained_bytes_used") is not False
        or full_storage.get("generic_reservation_used") is not False
    ):
        raise HandoffContractError(
            "Full prospective-storage arithmetic drifted"
        )
    accounts = audit.get("account_observations")
    if not isinstance(accounts, list):
        raise HandoffContractError("GPFS admission accounts are absent")
    evaluated = []
    for item in accounts:
        if not isinstance(item, Mapping):
            raise HandoffContractError("GPFS admission account is malformed")
        free_after_active = _finite_float(
            item.get("free_after_active_gb"),
            "GPFS free space after active bounds",
        )
        free_after_full = free_after_active - full_bound_gib
        evaluated.append(
            {
                **copy.deepcopy(dict(item)),
                "full_prospective_storage_bound_gib": full_bound_gib,
                "free_after_active_and_full_gib": free_after_full,
                "full_arithmetic_passed": (
                    item.get("active_only_arithmetic_passed") is True
                    and full_storage[
                        "static_transient_route_bound_passed"
                    ]
                    is True
                    and free_after_full >= MINIMUM_GPFS_FREE_GB
                ),
            }
        )
    safe = [
        item
        for item in evaluated
        if item["full_arithmetic_passed"] is True
        and (
            required_account is None
            or item["account_name"] == required_account
        )
    ]
    safe.sort(
        key=lambda item: (
            -item["free_after_active_and_full_gib"],
            item["account_name"],
        )
    )
    return evaluated, safe


def _fresh_gates(
    plan: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    required_account: str | None = None,
    required_gpfs_audit_path: Path | None = None,
    required_full_storage_authority_path: Path | None = None,
    required_license_snapshot_path: Path | None = None,
    live_reader: Any = _default_live_reader,
    task_list_reader: Any = _default_task_list_reader,
    now: datetime | None = None,
) -> dict[str, Any]:
    cutover_path = Path(plan["scheduler_cutover_receipt"]["path"])
    cutover, launcher_before = diagnostic._validate_scheduler_cutover_receipt(
        cutover_path,
        verify_live_launcher=True,
        require_strict_node=True,
        require_active_strict=True,
    )
    admission = diagnostic._live_scheduler_admission_snapshot(
        scheduler_url=plan["scheduler_url"], reader=live_reader
    )
    license_path = (
        required_license_snapshot_path.resolve(strict=True)
        if required_license_snapshot_path is not None
        else _latest_license_snapshot(
            Path(plan["license_snapshot_directory"]), now=now
        )
    )
    production._validate_license_snapshot(license_path, now=now)
    if required_gpfs_audit_path is None:
        audit_path, audit = _latest_gpfs_audit(
            Path(plan["gpfs_audit_directory"]), now=now
        )
    else:
        audit_path = required_gpfs_audit_path.resolve(strict=True)
        audit = _load_gpfs_audit(audit_path, require_fresh=True, now=now)
    if required_full_storage_authority_path is None:
        full_storage_path, full_storage = (
            _latest_full_storage_authority(
                Path(plan["full_storage_authority_directory"]),
                selection=selection,
            )
        )
    else:
        full_storage_path = (
            required_full_storage_authority_path.resolve(strict=True)
        )
        full_storage = _load_full_storage_authority(
            full_storage_path, selection=selection
        )
    rows = task_list_reader(scheduler_url=plan["scheduler_url"])
    active = _assert_audit_matches_live(audit, rows)
    active_refill = _active_safe_refill_tasks(active)
    if active_refill:
        raise HandoffContractError(
            "active safe-refill work must yield before the priority Full POST"
        )
    storage_admission_accounts, safe_accounts = (
        _storage_admission_accounts(
            audit=audit,
            full_storage=full_storage,
            required_account=required_account,
        )
    )
    full_bound_gib = full_storage["full_prospective_storage_bound_gib"]
    selected = None
    capacity = None
    errors = []
    for item in safe_accounts:
        try:
            capacity = _require_account_capacity(
                scheduler_url=plan["scheduler_url"],
                account_name=item["account_name"],
                live_reader=live_reader,
            )
            selected = item["account_name"]
            break
        except HandoffContractError as exc:
            errors.append(str(exc))
    if selected is None or capacity is None:
        suffix = f": {'; '.join(errors)}" if errors else ""
        raise HandoffContractError(
            f"Full fast lane has no ready audited account{suffix}"
        )
    launcher_after = diagnostic._live_launcher_identity(
        cutover,
        expected_sha256=diagnostic.SCHEDULER_STRICT_NODE_LAUNCHER_SHA256,
    )
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler launcher changed during Full fast-lane admission"
        )
    return {
        "captured_at_utc": _now(),
        "selected_account_name": selected,
        "scheduler_cutover_receipt": production._file_record(cutover_path),
        "scheduler_cutover_payload_sha256": cutover["payload_sha256"],
        "scheduler_live_launcher_identity": launcher_after,
        "scheduler_admission_snapshot": admission,
        "scheduler_capacity_snapshot": capacity,
        "license_snapshot": production._file_record(license_path),
        "gpfs_audit": production._file_record(audit_path),
        "gpfs_audit_payload_sha256": audit["payload_sha256"],
        "full_storage_authority": production._file_record(
            full_storage_path
        ),
        "full_storage_authority_payload_sha256": full_storage[
            "payload_sha256"
        ],
        "source_standard_fresh_grid_output_bytes": full_storage[
            "fresh_grid_output_bytes"
        ],
        "full_symmetry_expansion_factor": (
            FULL_SYMMETRY_EXPANSION_FACTOR
        ),
        "full_prospective_storage_bound_bytes": full_storage[
            "full_prospective_storage_bound_bytes"
        ],
        "full_prospective_storage_bound_gib": full_bound_gib,
        "storage_admission_accounts": storage_admission_accounts,
        "active_project_tasks": active,
        "active_safe_refill_task_count": 0,
        "active_safe_refill_tasks": [],
        "full_priority_fence": production._file_record(
            Path(plan["priority_fence_path"]).resolve(strict=True)
        ),
        "active_unbounded_storage_conflict_count": 0,
        "minimum_gpfs_free_gb": MINIMUM_GPFS_FREE_GB,
    }


def _default_sibling_reader(
    *, scheduler_url: str, project: str, task_name: str
) -> list[dict[str, Any]]:
    return diagnostic._scheduler_project_tasks(
        scheduler_url=scheduler_url,
        project=project,
        task_name=task_name,
    )


def _sibling_inventory(
    rows: Any, *, full_plan: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
    ):
        raise HandoffContractError("Full sibling inventory is absent")
    stage = full_plan["stage"]
    expected_name = stage["task_name"]
    expected_dedupe = stage["retained_aedt_bundle"]["dedupe_key"]
    matching = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if (
            str(row.get("name") or "") != expected_name
            and str(row.get("dedupe_key") or "") != expected_dedupe
        ):
            continue
        normalized = _normalize_task(row)
        if normalized["project"] != scheduler_client.MFT_PROJECT:
            raise HandoffContractError("Full sibling crossed project boundary")
        matching.append(normalized)
    matching.sort(key=lambda item: item["task_id"])
    if len(matching) > 1:
        raise HandoffContractError("more than one Full sibling exists")
    unsigned = {
        "candidate_physics_sha256": full_plan[
            "candidate_physics_sha256"
        ],
        "task_name": expected_name,
        "dedupe_key": expected_dedupe,
        "matching_task_count": len(matching),
        "matching_tasks": matching,
    }
    return {**unsigned, "snapshot_sha256": canonical_sha256(unsigned)}


def _claim_winner(plan_path: Path, full_plan: Mapping[str, Any]) -> dict[str, Any]:
    stage = full_plan["stage"]
    return {
        "immediate_task_id": SINGLETON_LOGICAL_ID,
        "immediate_retry_kind": "none",
        "plan_payload_sha256": full_plan["payload_sha256"],
        "plan_file_sha256": production._sha256_file(
            plan_path.resolve(strict=True)
        ),
        "profile_sha256": stage["profile_sha256"],
        "resources": copy.deepcopy(FULL_RESOURCES),
        "task_name": stage["task_name"],
        "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = _normalize_task(task)
    winner = pending.get("winner")
    if (
        not isinstance(winner, Mapping)
        or normalized["name"] != winner.get("task_name")
        or normalized["dedupe_key"] != winner.get("dedupe_key")
        or normalized["project"] != scheduler_client.MFT_PROJECT
        or normalized["cpus"] != FULL_RESOURCES["cpus"]
        or normalized["memory_mb"] != FULL_RESOURCES["memory_mb"]
        or normalized["timeout_seconds"] != FULL_RESOURCES["timeout_seconds"]
        or normalized["aedt_backend"] != "standalone"
        or not normalized["account_name"]
    ):
        raise atomic_claim.ClaimContractError(
            "Full fast-lane task readback identity drifted"
        )
    return normalized


class _SubmissionAdapter:
    """Inject one audited account and a locked pre-POST guard."""

    def __init__(
        self,
        *,
        delegate: Any,
        account_name: str,
        guard: Callable[[], None],
        post_result_path: Path,
        recovered_task_id: int | None = None,
    ):
        self.delegate = delegate
        self.account_name = account_name
        self.guard = guard
        self.post_result_path = post_result_path
        self.recovered_task_id = recovered_task_id
        self.scheduler_post_count = 0

    def submit_verification(self, *args: Any, **kwargs: Any) -> int:
        if self.recovered_task_id is not None:
            return self.recovered_task_id
        if self.scheduler_post_count != 0:
            raise HandoffContractError("Full fast-lane adapter attempted a re-POST")
        if kwargs.get("account_name") not in {None, ""}:
            raise HandoffContractError("Full plan unexpectedly fixed an account")
        kwargs["account_name"] = self.account_name
        kwargs["pre_submit_guard"] = self.guard
        task_id = self.delegate.submit_verification(*args, **kwargs)
        task_id = _positive_int(task_id, "Full fast-lane submitted task ID")
        self.scheduler_post_count = 1
        _write_immutable(
            self.post_result_path,
            _sealed(
                {
                    "schema_version": POST_RESULT_SCHEMA,
                    "task_id": task_id,
                    "account_name": self.account_name,
                    "scheduler_post_count": 1,
                    "scheduler_submission_performed": True,
                    "created_at_utc": _now(),
                }
            ),
        )
        return task_id


def _write_post_intent(
    *,
    path: Path,
    selection: Mapping[str, Any],
    activation_path: Path,
    activation: Mapping[str, Any],
    gates: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> None:
    value = _sealed(
        {
            "schema_version": POST_INTENT_SCHEMA,
            "selection_payload_sha256": selection["payload_sha256"],
            "candidate_physics_sha256": selection[
                "candidate_physics_sha256"
            ],
            "activation_receipt": production._file_record(
                activation_path.resolve(strict=True)
            ),
            "activation_payload_sha256": activation["payload_sha256"],
            "selected_account_name": gates["selected_account_name"],
            "gpfs_audit": gates["gpfs_audit"],
            "full_storage_authority": gates["full_storage_authority"],
            "full_prospective_storage_bound_bytes": gates[
                "full_prospective_storage_bound_bytes"
            ],
            "license_snapshot": gates["license_snapshot"],
            "pending_claim_payload_sha256": pending["payload_sha256"],
            "maximum_scheduler_posts": 1,
            "scheduler_post_may_follow": True,
            "created_at_utc": _now(),
        }
    )
    _write_immutable(path, value)


def _load_post_intent(
    path: Path,
    *,
    selection: Mapping[str, Any],
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        POST_INTENT_SCHEMA,
        label="Full fast-lane POST intent",
    )
    storage_record = value.get("full_storage_authority")
    if not isinstance(storage_record, Mapping):
        raise HandoffContractError(
            "Full fast-lane POST intent storage authority is absent"
        )
    storage_path = Path(str(storage_record.get("path") or ""))
    if production._file_record(storage_path) != storage_record:
        raise HandoffContractError(
            "Full fast-lane POST intent storage bytes drifted"
        )
    storage = _load_full_storage_authority(
        storage_path, selection=selection
    )
    if (
        value.get("selection_payload_sha256") != selection["payload_sha256"]
        or value.get("candidate_physics_sha256")
        != selection["candidate_physics_sha256"]
        or value.get("activation_payload_sha256")
        != activation["payload_sha256"]
        or value.get("maximum_scheduler_posts") != 1
        or value.get("scheduler_post_may_follow") is not True
        or value.get("full_prospective_storage_bound_bytes")
        != storage["full_prospective_storage_bound_bytes"]
    ):
        raise HandoffContractError("Full fast-lane POST intent drifted")
    return value


def _load_fast_lane_receipt(
    path: Path,
    *,
    watch_plan_path: Path,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(
        _read_json(path.resolve(strict=True)),
        RECEIPT_SCHEMA,
        label="Full fast-lane receipt",
    )
    root = Path(plan["output_root"])
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or value.get("selection")
        != production._file_record(root / "selection.json")
        or value.get("selection_payload_sha256")
        != selection["payload_sha256"]
        or value.get("full_priority_fence")
        != production._file_record(
            Path(plan["priority_fence_path"]).resolve(strict=True)
        )
        or value.get("candidate_physics_sha256")
        != selection["candidate_physics_sha256"]
        or value.get("resources") != FULL_RESOURCES
        or value.get("full_model") != 1
        or value.get("thermal_symmetry") != "full"
        or value.get("fixed_boundary") != plan["fixed_boundary"]
        or value.get("exact_full_sibling_count") != 1
        or value.get("scheduler_submission_performed") is not True
        or value.get("scheduler_repository_modified") is not False
    ):
        raise HandoffContractError("Full fast-lane receipt contract drifted")
    for gate_name in ("initial_fresh_gates", "locked_fresh_gates"):
        gate = value.get(gate_name)
        if not isinstance(gate, Mapping):
            raise HandoffContractError(
                f"Full fast-lane receipt {gate_name} is absent"
            )
        storage_record = gate.get("full_storage_authority")
        if not isinstance(storage_record, Mapping):
            raise HandoffContractError(
                "Full fast-lane receipt storage authority is absent"
            )
        storage_path = Path(str(storage_record.get("path") or ""))
        if production._file_record(storage_path) != storage_record:
            raise HandoffContractError(
                "Full fast-lane receipt storage bytes drifted"
            )
        storage = _load_full_storage_authority(
            storage_path, selection=selection
        )
        if (
            gate.get("full_storage_authority_payload_sha256")
            != storage["payload_sha256"]
            or gate.get("full_prospective_storage_bound_bytes")
            != storage["full_prospective_storage_bound_bytes"]
            or gate.get("full_symmetry_expansion_factor")
            != FULL_SYMMETRY_EXPANSION_FACTOR
        ):
            raise HandoffContractError(
                "Full fast-lane receipt storage gate drifted"
            )
    activation_record = value.get("activation_receipt")
    submission_record = value.get("full_submission")
    if not isinstance(activation_record, Mapping) or not isinstance(
        submission_record, Mapping
    ):
        raise HandoffContractError(
            "Full fast-lane receipt authority records are absent"
        )
    activation_path = Path(str(activation_record.get("path") or ""))
    if production._file_record(activation_path) != activation_record:
        raise HandoffContractError(
            "Full fast-lane activation receipt bytes drifted"
        )
    activation = _load_activation(
        activation_path, watch_plan_path=watch_plan_path, plan=plan
    )
    if value.get("activation_payload_sha256") != activation["payload_sha256"]:
        raise HandoffContractError(
            "Full fast-lane activation payload drifted"
        )
    submission_path = Path(str(submission_record.get("path") or ""))
    if production._file_record(submission_path) != submission_record:
        raise HandoffContractError("Full submission receipt bytes drifted")
    full_plan_path = _selection_plan_path(selection)
    full_plan, _params, _profile, _view, standard_truth = (
        promotion._load_full_plan(full_plan_path)
    )
    submission = promotion._load_full_submission(
        submission_path,
        plan=full_plan,
        standard_truth=standard_truth,
    )
    if (
        value.get("full_submission_payload_sha256")
        != submission["payload_sha256"]
        or value.get("task_id") != submission["task_id"]
        or value.get("task_name") != full_plan["stage"]["task_name"]
        or value.get("dedupe_key")
        != full_plan["stage"]["retained_aedt_bundle"]["dedupe_key"]
    ):
        raise HandoffContractError("Full fast-lane submission ancestry drifted")
    authority = atomic_claim.load_claim_root(
        Path(plan["claim_root_authority"]["resolved_root"]),
        expected_authority=plan["claim_root_authority"],
    )
    reference = atomic_claim.validate_claim_reference(
        plan["claim_reference"], authority
    )
    finalized = atomic_claim.validate_finalized_claim(
        Path(authority["resolved_root"]),
        reference,
        claim=value.get("atomic_claim_finalized"),
        expected_winner=_claim_winner(full_plan_path, full_plan),
    )
    if finalized["task_id"] != submission["task_id"]:
        raise HandoffContractError("Full fast-lane finalized task drifted")
    return value


def _execute_submission(
    *,
    watch_plan_path: Path,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    activation_path: Path,
    activation: Mapping[str, Any],
    initial_gates: Mapping[str, Any],
    scheduler: Any = scheduler_client,
    live_reader: Any = _default_live_reader,
    task_list_reader: Any = _default_task_list_reader,
    sibling_reader: Any = _default_sibling_reader,
    full_submitter: Any = promotion.submit_full,
    waiter: Callable[[], None] | None = None,
) -> dict[str, Any]:
    root = Path(plan["output_root"])
    final_path = root / "fast_lane_receipt.json"
    if final_path.exists():
        return _load_fast_lane_receipt(
            final_path,
            watch_plan_path=watch_plan_path,
            plan=plan,
            selection=selection,
        )
    full_plan_path = _selection_plan_path(selection)
    full_plan, _params, _profile, _view, standard_truth = (
        promotion._load_full_plan(full_plan_path)
    )

    def read_siblings() -> dict[str, Any]:
        rows = sibling_reader(
            scheduler_url=plan["scheduler_url"],
            project=plan["scheduler_project"],
            task_name=full_plan["stage"]["task_name"],
        )
        return _sibling_inventory(rows, full_plan=full_plan)

    inventory_before = read_siblings()
    authority = atomic_claim.load_claim_root(
        Path(plan["claim_root_authority"]["resolved_root"]),
        expected_authority=plan["claim_root_authority"],
    )
    reference = atomic_claim.validate_claim_reference(
        plan["claim_reference"], authority
    )
    winner = _claim_winner(full_plan_path, full_plan)
    try:
        acquisition = atomic_claim.acquire_claim(
            Path(authority["resolved_root"]), reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "Full fast-lane singleton claim acquisition failed"
        ) from exc
    claim_status = acquisition["status"]
    pending = (
        acquisition["claim"]["pending_claim"]
        if claim_status == "existing_finalized"
        else acquisition["claim"]
    )
    intent_path = root / "post_intent.json"
    submission_path = root / "full_submission.json"
    post_result_path = root / "post_result.json"
    own = inventory_before["matching_tasks"]
    recovered_task_id = own[0]["task_id"] if len(own) == 1 else None
    if len(own) > 1:
        raise HandoffContractError("Full fast-lane sibling count exceeded one")
    if claim_status == "existing_finalized" and recovered_task_id is None:
        raise HandoffContractError(
            "finalized Full fast-lane claim has no durable sibling"
        )
    if (
        claim_status == "existing_pending"
        and recovered_task_id is None
        and intent_path.exists()
    ):
        _load_post_intent(
            intent_path, selection=selection, activation=activation
        )
        raise HandoffContractError(
            "pending Full POST intent has no sibling; manual reconciliation "
            "is required and re-POST is forbidden"
        )
    locked_guard_count = 0
    latest_gates = copy.deepcopy(dict(initial_gates))

    def locked_guard() -> None:
        nonlocal locked_guard_count, latest_gates
        if read_siblings()["matching_task_count"] != 0:
            raise HandoffContractError(
                "Full fast-lane locked sibling slot is not empty"
            )
        latest_gates = _fresh_gates(
            plan,
            selection=selection,
            required_account=initial_gates["selected_account_name"],
            required_gpfs_audit_path=Path(
                initial_gates["gpfs_audit"]["path"]
            ),
            required_full_storage_authority_path=Path(
                initial_gates["full_storage_authority"]["path"]
            ),
            required_license_snapshot_path=Path(
                initial_gates["license_snapshot"]["path"]
            ),
            live_reader=live_reader,
            task_list_reader=task_list_reader,
        )
        _write_post_intent(
            path=intent_path,
            selection=selection,
            activation_path=activation_path,
            activation=activation,
            gates=latest_gates,
            pending=pending,
        )
        locked_guard_count += 1

    if submission_path.exists():
        submission = promotion._load_full_submission(
            submission_path,
            plan=full_plan,
            standard_truth=standard_truth,
        )
    else:
        adapter = _SubmissionAdapter(
            delegate=scheduler,
            account_name=initial_gates["selected_account_name"],
            guard=locked_guard,
            post_result_path=post_result_path,
            recovered_task_id=recovered_task_id,
        )
        full_submitter(
            plan_path=full_plan_path,
            scheduler_cutover_receipt_path=Path(
                plan["scheduler_cutover_receipt"]["path"]
            ),
            license_snapshot_path=Path(
                initial_gates["license_snapshot"]["path"]
            ),
            priority=100,
            output=submission_path,
            scheduler=adapter,
            live_reader=live_reader,
        )
        submission = promotion._load_full_submission(
            submission_path,
            plan=full_plan,
            standard_truth=standard_truth,
        )
        if recovered_task_id is None and locked_guard_count != 1:
            raise HandoffContractError(
                "Full fast-lane POST lacks exactly one locked guard"
            )
        if recovered_task_id is not None and adapter.scheduler_post_count != 0:
            raise HandoffContractError("Full recovery mutated Scheduler")
    task_id = _positive_int(submission["task_id"], "Full submission task ID")
    wait = waiter or (lambda: time.sleep(0.25))
    inventory_after = None
    for attempt in range(8):
        latest = read_siblings()
        if latest["matching_task_count"] == 1:
            inventory_after = latest
            break
        if attempt < 7:
            wait()
    if inventory_after is None:
        raise HandoffContractError("Full POST did not produce one durable sibling")
    durable = inventory_after["matching_tasks"][0]
    if durable["task_id"] != task_id:
        raise HandoffContractError("Full submission and sibling task IDs differ")
    try:
        if claim_status == "fresh_pending":
            finalized = atomic_claim.finalize_claim(
                Path(authority["resolved_root"]),
                reference,
                acquisition["claim"],
                task_id=task_id,
                task_readback=durable,
                sibling_snapshot=inventory_after,
                evidence_validator=_claim_task_evidence,
            )
        elif claim_status == "existing_pending":
            finalized = atomic_claim.recover_pending_claim(
                Path(authority["resolved_root"]),
                reference,
                acquisition["claim"],
                matching_tasks=[durable],
                sibling_snapshot=inventory_after,
                evidence_validator=_claim_task_evidence,
            )
        else:
            finalized = atomic_claim.validate_finalized_claim(
                Path(authority["resolved_root"]),
                reference,
                claim=acquisition["claim"],
                expected_winner=winner,
            )
            _claim_task_evidence(durable, finalized["pending_claim"])
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "Full fast-lane claim finalization failed"
        ) from exc
    receipt = _sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "selection": production._file_record(root / "selection.json"),
            "selection_payload_sha256": selection["payload_sha256"],
            "full_priority_fence": production._file_record(
                Path(plan["priority_fence_path"]).resolve(strict=True)
            ),
            "activation_receipt": production._file_record(
                activation_path.resolve(strict=True)
            ),
            "activation_payload_sha256": activation["payload_sha256"],
            "full_submission": production._file_record(submission_path),
            "full_submission_payload_sha256": submission["payload_sha256"],
            "candidate_physics_sha256": selection[
                "candidate_physics_sha256"
            ],
            "task_id": task_id,
            "task_name": full_plan["stage"]["task_name"],
            "dedupe_key": full_plan["stage"]["retained_aedt_bundle"][
                "dedupe_key"
            ],
            "resources": copy.deepcopy(FULL_RESOURCES),
            "full_model": 1,
            "thermal_symmetry": "full",
            "fixed_boundary": copy.deepcopy(plan["fixed_boundary"]),
            "selected_account_name": initial_gates[
                "selected_account_name"
            ],
            "initial_fresh_gates": copy.deepcopy(dict(initial_gates)),
            "locked_fresh_gates": latest_gates,
            "sibling_inventory_before": inventory_before,
            "sibling_inventory_after": inventory_after,
            "exact_full_sibling_count": 1,
            "atomic_claim_acquisition_status": claim_status,
            "atomic_claim_finalized": finalized,
            "scheduler_post_count_this_run": (
                0 if recovered_task_id is not None else 1
            ),
            "scheduler_submission_performed": True,
            "scheduler_repository_modified": False,
            "created_at_utc": _now(),
        }
    )
    _write_immutable(final_path, receipt)
    return receipt


def _state(
    *,
    plan: Mapping[str, Any],
    status: str,
    selection: Mapping[str, Any] | None,
    gates: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
    error: Exception | None = None,
    activation_present: bool = False,
) -> dict[str, Any]:
    fence_record = None
    fence_path = Path(str(plan.get("priority_fence_path") or ""))
    if selection is not None and fence_path.is_file():
        fence_record = production._file_record(fence_path)
    return _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "watch_plan_payload_sha256": plan["payload_sha256"],
            "observed_at_utc": _now(),
            "status": status,
            "selection_payload_sha256": (
                selection["payload_sha256"] if selection is not None else None
            ),
            "candidate_physics_sha256": (
                selection["candidate_physics_sha256"]
                if selection is not None
                else None
            ),
            "full_priority_fence": fence_record,
            "activation_receipt_present": activation_present,
            "live_post_default_enabled": False,
            "fresh_gates": (
                copy.deepcopy(dict(gates)) if gates is not None else None
            ),
            "receipt_payload_sha256": (
                receipt["payload_sha256"] if receipt is not None else None
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "maximum_scheduler_posts": 1,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": receipt is not None,
        }
    )


def process_cycle(
    *,
    watch_plan_path: Path,
    activation_receipt_path: Path | None = None,
    live_reader: Any = _default_live_reader,
    task_list_reader: Any = _default_task_list_reader,
    sibling_reader: Any = _default_sibling_reader,
    scheduler: Any = scheduler_client,
    full_submitter: Any = promotion.submit_full,
    now: datetime | None = None,
) -> dict[str, Any]:
    plan = _load_watch_plan(watch_plan_path)
    root = Path(plan["output_root"])
    selection = None
    gates = None
    try:
        selection = _prepare_selection(plan)
        if selection is None:
            state = _state(
                plan=plan,
                status="waiting_for_first_authenticated_feasible_truth",
                selection=None,
            )
        elif (root / "fast_lane_receipt.json").exists():
            receipt = _load_fast_lane_receipt(
                root / "fast_lane_receipt.json",
                watch_plan_path=watch_plan_path,
                plan=plan,
                selection=selection,
            )
            state = _state(
                plan=plan,
                status="full_submitted_exactly_once",
                selection=selection,
                receipt=receipt,
                activation_present=True,
            )
        else:
            _ensure_full_storage_authority(plan, selection)
            gates = _fresh_gates(
                plan,
                selection=selection,
                live_reader=live_reader,
                task_list_reader=task_list_reader,
                now=now,
            )
            if activation_receipt_path is None:
                state = _state(
                    plan=plan,
                    status="ready_live_post_disabled_pending_root_review",
                    selection=selection,
                    gates=gates,
                )
            else:
                activation = _load_activation(
                    activation_receipt_path,
                    watch_plan_path=watch_plan_path,
                    plan=plan,
                )
                receipt = _execute_submission(
                    watch_plan_path=watch_plan_path,
                    plan=plan,
                    selection=selection,
                    activation_path=activation_receipt_path,
                    activation=activation,
                    initial_gates=gates,
                    scheduler=scheduler,
                    live_reader=live_reader,
                    task_list_reader=task_list_reader,
                    sibling_reader=sibling_reader,
                    full_submitter=full_submitter,
                )
                state = _state(
                    plan=plan,
                    status="full_submitted_exactly_once",
                    selection=selection,
                    gates=gates,
                    receipt=receipt,
                    activation_present=True,
                )
    except Exception as exc:
        state = _state(
            plan=plan,
            status="blocked_fail_closed",
            selection=selection,
            gates=gates,
            error=exc,
            activation_present=activation_receipt_path is not None,
        )
    _write_atomic(root / "state.json", state)
    return state


def run_watcher(
    *,
    watch_plan_path: Path,
    activation_receipt_path: Path | None = None,
    once: bool = False,
) -> None:
    plan = _load_watch_plan(watch_plan_path)
    root = Path(plan["output_root"])
    with terminal.SingleInstanceLock(root / "full_fast_lane.lock"):
        _write_atomic(
            root / "watcher.pid.json",
            _sealed(
                {
                    "schema_version": PID_SCHEMA,
                    "pid": os.getpid(),
                    "watch_plan": production._file_record(
                        watch_plan_path.resolve(strict=True)
                    ),
                    "activation_receipt": (
                        production._file_record(
                            activation_receipt_path.resolve(strict=True)
                        )
                        if activation_receipt_path is not None
                        else None
                    ),
                    "live_post_enabled": activation_receipt_path is not None,
                    "started_at_utc": _now(),
                }
            ),
        )
        while True:
            process_cycle(
                watch_plan_path=watch_plan_path,
                activation_receipt_path=activation_receipt_path,
            )
            if once:
                return
            time.sleep(plan["poll_seconds"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restart-safe first truth to exactly-one Full fast lane"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--terminal-state", type=Path, required=True)
    init.add_argument("--scheduler-cutover-receipt", type=Path, required=True)
    init.add_argument("--license-snapshot-directory", type=Path, required=True)
    init.add_argument("--gpfs-audit-directory", type=Path, required=True)
    init.add_argument("--full-storage-authority-directory", type=Path)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--claim-root", type=Path, default=DEFAULT_CLAIM_ROOT)
    init.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    audit = commands.add_parser("capture-gpfs-audit")
    audit.add_argument("--observations", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    standard_bound = commands.add_parser(
        "create-standard-storage-bound"
    )
    standard_bound.add_argument("--source-plan", type=Path, required=True)
    standard_bound.add_argument("--task-id", type=int, required=True)
    standard_bound.add_argument("--output", type=Path, required=True)
    full_bound = commands.add_parser("create-full-storage-authority")
    full_bound.add_argument("--selection", type=Path, required=True)
    full_bound.add_argument("--output", type=Path, required=True)
    activate = commands.add_parser("activate")
    activate.add_argument("--watch-plan", type=Path, required=True)
    activate.add_argument("--reviewer", required=True)
    activate.add_argument("--output", type=Path, required=True)
    for name in ("run", "once"):
        command = commands.add_parser(name)
        command.add_argument("--watch-plan", type=Path, required=True)
        command.add_argument("--activation-receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result: Any = initialize_watch_plan(
            terminal_state_path=args.terminal_state,
            scheduler_cutover_receipt_path=args.scheduler_cutover_receipt,
            license_snapshot_directory=args.license_snapshot_directory,
            gpfs_audit_directory=args.gpfs_audit_directory,
            output_root=args.output_root,
            full_storage_authority_directory=(
                args.full_storage_authority_directory
            ),
            claim_root=args.claim_root,
            poll_seconds=args.poll_seconds,
        )
    elif args.command == "capture-gpfs-audit":
        observations = _read_json(args.observations)
        rows = _default_task_list_reader(
            scheduler_url=diagnostic.DIAGNOSTIC_SCHEDULER_URL
        )
        audit = build_gpfs_audit(
            observed_at_utc=str(observations.get("observed_at_utc") or ""),
            account_observations=observations.get("accounts") or [],
            rows=rows,
            active_task_storage_authority_paths=[
                Path(str(item))
                for item in (
                    observations.get("active_task_storage_authorities")
                    or []
                )
            ],
        )
        result = _write_immutable(args.output, audit)
    elif args.command == "create-standard-storage-bound":
        result = create_standard_storage_bound_authority(
            source_plan_path=args.source_plan,
            task_id=args.task_id,
            output=args.output,
        )
    elif args.command == "create-full-storage-authority":
        result = create_full_storage_authority(
            selection_path=args.selection,
            output=args.output,
        )
    elif args.command == "activate":
        result = create_activation_receipt(
            watch_plan_path=args.watch_plan,
            reviewer=args.reviewer,
            output=args.output,
        )
    else:
        run_watcher(
            watch_plan_path=args.watch_plan,
            activation_receipt_path=args.activation_receipt,
            once=args.command == "once",
        )
        result = Path(args.watch_plan).resolve().parent / "state.json"
    print(json.dumps({"status": "ok", "result": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
