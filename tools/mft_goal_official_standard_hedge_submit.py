"""Fail-closed one-shot submitter for the sealed official Standard hedge3.

This module is intentionally separate from
``mft_goal_official_standard_hedge_prepare``.  The prepare-only tool retains
no Scheduler mutation entry point.  This submitter authenticates that exact
three-lane prepare package, repeats all live GET gates, and consumes a
separate immutable per-lane attempt ledger before each possible POST.

Every lane has a lifetime POST budget of one.  A missing or ambiguous HTTP
response consumes that budget and is only reconciled by GET; it is never
retried.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

prepare_only = importlib.import_module(
    "tools.mft_goal_official_standard_hedge_prepare"
)


reviewed = prepare_only.reviewed
PostdeadlineContractError = prepare_only.PostdeadlineContractError
CandidateSpec = prepare_only.CandidateSpec
JsonReader = prepare_only.JsonReader

SCHEDULER_URL = prepare_only.SCHEDULER_URL
PROJECT = prepare_only.PROJECT
CPUS = prepare_only.CPUS
MEMORY_MB = prepare_only.MEMORY_MB
SCHEDULER_SECONDS = prepare_only.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = prepare_only.MAX_WORKERS_PER_NODE
FIXED_BOUNDARY = copy.deepcopy(prepare_only.FIXED_BOUNDARY)
SAFETY_FLAGS = copy.deepcopy(prepare_only.SAFETY_FLAGS)
CANDIDATES = prepare_only.CANDIDATES

PREPARE_ROOT = prepare_only.OUTPUT_ROOT
PREPARE_MANIFEST = PREPARE_ROOT / prepare_only.MANIFEST_NAME
PREPARE_MANIFEST_SHA256 = (
    "42f2e96c22c09fbf78f8730ac6c21dabc40f98917bfb8480af63aecf838cd3c4"
)
PREPARE_MANIFEST_PAYLOAD_SHA256 = (
    "4af033eb36a64b44a2e0d788c8af11d9f35b6807fb5bed5fe9a73be229d0e9a8"
)

OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official_hedge3_submit_260726_v1"
)
AUTHORIZATION = (
    "authorize-official-hedge3-strict-n110-n112-n115-one-post-each-v1"
)
RUN_COMMAND_ARGV = [
    "python",
    "tools/mft_goal_official_standard_hedge_submit.py",
    "submit",
    "--authorize-post",
    AUTHORIZATION,
]

MIN_ALLOCATION_WALLTIME_SECONDS = 14 * 60 * 60
REVIEWED_ALLOCATION_WALLTIME_SECONDS = 48 * 60 * 60
ALLOCATION_DRAIN_AFTER_SECONDS = 129_600
ALLOCATION_ATTACH_STOP_BEFORE_DRAIN_SECONDS = 1_800
ALLOCATION_PENDING_TIMEOUT_SECONDS = 1_800
ALLOCATION_ATTACH_WINDOW_SECONDS = (
    ALLOCATION_DRAIN_AFTER_SECONDS
    - ALLOCATION_ATTACH_STOP_BEFORE_DRAIN_SECONDS
)
MIN_STORAGE_FREE_GB = 10.0
LICENSE_FEATURE = "electronics_desktop"
LICENSE_RESERVE_FLOOR = 24
NODE_STATE_ALLOWLIST = frozenset({"idle", "mix"})
MAX_NODE_METRICS_AGE_SECONDS = 10 * 60

BATCH_INTENT_NAME = "batch_submission_intent.json"
BATCH_FINAL_NAME = "batch_final_seal.json"
BATCH_FAILURE_NAME = "batch_failure.json"
AGGREGATE_SUBMIT_MANIFEST_NAME = "aggregate_submit_manifest.json"
LANE_INTENT_NAME = "lane_submission_intent.json"
ATTEMPT_NAME = "scheduler_post_attempt.json"
RECEIPT_NAME = "submission_receipt.json"
LANE_FINAL_NAME = "lane_final_seal.json"
AMBIGUOUS_NAME = "post_ambiguous.json"

BATCH_INTENT_SCHEMA = "mft-goal-official-standard-hedge3-submit-intent-v1"
BATCH_FINAL_SCHEMA = "mft-goal-official-standard-hedge3-final-seal-v1"
BATCH_FAILURE_SCHEMA = "mft-goal-official-standard-hedge3-failure-v1"
AGGREGATE_SUBMIT_MANIFEST_SCHEMA = (
    "mft-goal-official-standard-hedge3-submit-manifest-v1"
)
LANE_INTENT_SCHEMA = "mft-goal-official-standard-hedge-lane-intent-v1"
ATTEMPT_SCHEMA = "mft-goal-official-standard-hedge-post-attempt-v1"
RECEIPT_SCHEMA = "mft-goal-official-standard-hedge-submission-v1"
LANE_FINAL_SCHEMA = "mft-goal-official-standard-hedge-lane-final-v1"
AMBIGUOUS_SCHEMA = "mft-goal-official-standard-hedge-post-ambiguous-v1"
LIVE_GATE_SCHEMA = "mft-goal-official-standard-hedge-submit-live-gate-v1"

LIVE_LAUNCHER = Path(r"Y:\runtime\slurm_scheduler\start_web_y.cmd")
LIVE_LAUNCHER_SHA256 = (
    "62eb3c29048b4ffc80469e975c8dbb0671feb7fc4dfc4e4b18792be08471e01f"
)
LIVE_CONFIG = Path(r"Y:\runtime\slurm_scheduler\config\app.yaml")
LIVE_CONFIG_SHA256 = (
    "5b420f97db63b23a40985d49988e3b202c66575832a99d8fff1f3cd186a79aba"
)
DEPLOYED_COMMIT = "4facdfe36f74ff0679a477f6c25e1dddf26ce791"
DEPLOYED_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\deployments\4facdfe36f74"
)
DEPLOYED_CONFIG_PY = DEPLOYED_ROOT / "slurm_scheduler" / "config.py"
DEPLOYED_CONFIG_PY_SHA256 = (
    "77e8d14bff644e52357ba40dbde2629982a5706c1dbb1adb830373dcec4b1912"
)
DEPLOYED_SCHEDULER_PY = DEPLOYED_ROOT / "slurm_scheduler" / "scheduler.py"
DEPLOYED_SCHEDULER_PY_SHA256 = (
    "43609ae622246ca106392747895e8036b9a3ac960effd00c2b0dc9f805abac01"
)
CUTOVER_RECEIPT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\deployment_candidates"
    r"\4facdfe36f74-strict-demand-pool-retention-20260725"
    r"\cutover_receipt_20260725T221419.json"
)
CUTOVER_RECEIPT_SHA256 = (
    "45cf323de69a9a9c926c528d905ee4d09715b1a3d40c68349b9fa511646486bc"
)

canonical_bytes = prepare_only.canonical_bytes
payload_sha256 = prepare_only.payload_sha256
sealed = prepare_only.sealed
validate_seal = prepare_only.validate_seal
sha256_file = prepare_only.sha256_file
file_record = prepare_only.file_record
read_json = prepare_only.read_json
write_immutable_json = prepare_only.write_immutable_json
write_exclusive_json = reviewed.write_exclusive_json
get_json = prepare_only.get_json
scheduler_campaign_lock = reviewed.scheduler_campaign_lock

Poster = Callable[
    [str, Mapping[str, Any]],
    tuple[int | None, dict[str, Any] | None, str | None],
]
LaneReauthenticator = Callable[
    [CandidateSpec],
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
]


def _contained_record(root: Path, record: Any, label: str) -> Path:
    """Authenticate one relative file record without accepting path escape."""

    if (
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise PostdeadlineContractError(f"{label} record is malformed")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / record["path"]).resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise PostdeadlineContractError(
            f"{label} escapes the sealed root"
        ) from exc
    if (
        not path.is_file()
        or sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["size_bytes"]
    ):
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return path


def _default_reauthenticate(
    spec: CandidateSpec,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    selected, params = prepare_only.authenticate_candidate(spec)
    profile = prepare_only.candidate8.reviewed_profile()
    payload, environment, retained = prepare_only.derive_scheduler_payload(
        spec, params, profile
    )
    return selected, params, profile, payload, environment, retained


def _authenticate_lane_plan(
    root: Path,
    spec: CandidateSpec,
    manifest_lane: Mapping[str, Any],
    *,
    reauthenticate: LaneReauthenticator,
) -> dict[str, Any]:
    expected_manifest_lane = {
        "selection_order": spec.selection_order,
        "candidate_physics_sha256": spec.candidate_sha256,
        "account_name": spec.account_name,
        "node_name": spec.node_name,
        "task_name": spec.task_name,
    }
    drift = {
        key: {"expected": expected, "actual": manifest_lane.get(key)}
        for key, expected in expected_manifest_lane.items()
        if manifest_lane.get(key) != expected
    }
    if drift or manifest_lane.get("scheduler_post_calls") != 0:
        raise PostdeadlineContractError(
            f"prepare manifest lane drifted: {drift}"
        )
    plan_path = _contained_record(
        root, manifest_lane.get("plan"), f"official{spec.selection_order} plan"
    )
    plan_record = manifest_lane["plan"]
    plan = validate_seal(
        read_json(plan_path), prepare_only.PLAN_SCHEMA
    )
    if (
        plan.get("payload_sha256")
        != manifest_lane.get("plan_payload_sha256")
        or plan.get("prepare_only") is not True
        or plan.get("submission_capability_present") is not False
        or plan.get("future_submission_implementation_present") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("scheduler_post_calls") != 0
        or plan.get("campaign_id") != prepare_only.CAMPAIGN_ID
        or any(
            plan.get(key) is not value for key, value in SAFETY_FLAGS.items()
        )
        or plan.get("candidate_physics_sha256") != spec.candidate_sha256
        or plan.get("canonical_official_row_json_sha256")
        != spec.canonical_row_json_sha256
        or plan.get("canonical_physical_params_sha256")
        != spec.canonical_physical_params_sha256
        or plan.get("source_seed") != spec.source_seed
        or plan.get("source_task_id") != spec.source_task_id
        or plan.get("official_standard_selection_order")
        != spec.selection_order
        or plan.get("task_name") != spec.task_name
        or plan.get("workdir") != spec.workdir
        or plan.get("scheduler_url") != SCHEDULER_URL
        or plan.get("scheduler_project") != PROJECT
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("fixed_physics_unchanged") is not True
    ):
        raise PostdeadlineContractError("sealed hedge lane plan drifted")

    placement = plan.get("placement")
    resources = plan.get("resources")
    opening = plan.get("opening_demand_pool_contract")
    payload = plan.get("scheduler_payload")
    if (
        not isinstance(placement, dict)
        or placement.get("account_name") != spec.account_name
        or placement.get("node_name") != spec.node_name
        or placement.get("node_name_policy") != "strict"
        or placement.get("preferred_node_relaxed_allowed") is not False
        or not isinstance(resources, dict)
        or resources.get("cpus") != CPUS
        or resources.get("memory_mb") != MEMORY_MB
        or resources.get("scheduler_timeout_seconds") != SCHEDULER_SECONDS
        or resources.get("max_workers_per_node") != MAX_WORKERS_PER_NODE
        or not isinstance(opening, dict)
        or opening.get("allowed_pre_submit_queue_state") != "opening"
        or opening.get("relaxed_allocation_allowed") is not False
        or not isinstance(payload, dict)
        or payload.get("name") != spec.task_name
        or payload.get("account_name") != spec.account_name
        or payload.get("node_name") != spec.node_name
        or payload.get("node_name_policy") != "strict"
        or payload.get("timeout_seconds") != SCHEDULER_SECONDS
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(payload)
        or plan.get("dedupe_key") != payload.get("dedupe_key")
        or manifest_lane.get("dedupe_key") != payload.get("dedupe_key")
    ):
        raise PostdeadlineContractError(
            "sealed placement/resource/payload contract drifted"
        )
    if (
        REVIEWED_ALLOCATION_WALLTIME_SECONDS
        < MIN_ALLOCATION_WALLTIME_SECONDS
        or SCHEDULER_SECONDS > MIN_ALLOCATION_WALLTIME_SECONDS
    ):
        raise PostdeadlineContractError(
            "reviewed >=14h allocation walltime contract is invalid"
        )

    lane_root = plan_path.parent
    selected_path = _contained_record(
        lane_root, plan.get("selected_candidate"), "selected candidate"
    )
    params_path = _contained_record(
        lane_root, plan.get("fea_params"), "FEA params"
    )
    profile_path = _contained_record(
        lane_root, plan.get("execution_profile"), "execution profile"
    )
    selected = validate_seal(
        read_json(selected_path), prepare_only.CANDIDATE_SCHEMA
    )
    params = read_json(params_path)
    profile = read_json(profile_path)
    (
        fresh_selected,
        fresh_params,
        fresh_profile,
        fresh_payload,
        fresh_environment,
        fresh_retained,
    ) = reauthenticate(spec)
    if (
        selected != fresh_selected
        or params != fresh_params
        or profile != fresh_profile
        or payload != fresh_payload
        or plan.get("raw_fea_params_sha256") != payload_sha256(params)
        or plan.get("profiled_fea_params_sha256")
        != payload_sha256(prepare_only._profiled_params(params, profile))
        or plan.get("execution_profile_canonical_sha256")
        != payload_sha256(profile)
        or plan.get("submission_environment") != fresh_environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(fresh_environment)
        or plan.get("retained_aedt_bundle") != fresh_retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(fresh_retained)
    ):
        raise PostdeadlineContractError(
            "lane no longer matches reauthenticated candidate and payload"
        )
    return {
        "spec": spec,
        "plan": plan,
        "plan_path": plan_path,
        "plan_file_record": copy.deepcopy(plan_record),
        "lane_nonce": payload_sha256(
            {
                "schema_version": (
                    "mft-goal-official-standard-hedge-lane-nonce-v1"
                ),
                "prepare_manifest_sha256": PREPARE_MANIFEST_SHA256,
                "prepare_manifest_payload_sha256": (
                    PREPARE_MANIFEST_PAYLOAD_SHA256
                ),
                "selection_order": spec.selection_order,
                "candidate_physics_sha256": spec.candidate_sha256,
                "task_name": spec.task_name,
                "dedupe_key": payload["dedupe_key"],
                "account_name": spec.account_name,
                "node_name": spec.node_name,
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
            }
        ),
    }


def authenticate_prepare_package(
    manifest_path: Path = PREPARE_MANIFEST,
    *,
    reauthenticate: LaneReauthenticator = _default_reauthenticate,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate the fixed manifest and all three rederived lane payloads."""

    resolved = manifest_path.resolve(strict=True)
    if resolved != PREPARE_MANIFEST.resolve(strict=True):
        raise PostdeadlineContractError("prepare manifest path is not fixed")
    if sha256_file(resolved) != PREPARE_MANIFEST_SHA256:
        raise PostdeadlineContractError("prepare manifest bytes drifted")
    manifest = validate_seal(
        read_json(resolved), prepare_only.MANIFEST_SCHEMA
    )
    lanes = manifest.get("lanes")
    if (
        manifest.get("payload_sha256")
        != PREPARE_MANIFEST_PAYLOAD_SHA256
        or manifest.get("campaign_id") != prepare_only.CAMPAIGN_ID
        or manifest.get("prepare_only") is not True
        or manifest.get("submission_capability_present") is not False
        or manifest.get("scheduler_submission_performed") is not False
        or manifest.get("scheduler_post_calls") != 0
        or manifest.get("official_standard_candidate_count") != 3
        or manifest.get("official_standard_selection_orders")
        != [spec.selection_order for spec in CANDIDATES]
        or any(
            manifest.get(key) is not value
            for key, value in SAFETY_FLAGS.items()
        )
        or manifest.get("fixed_boundary") != FIXED_BOUNDARY
        or manifest.get("fixed_physics_unchanged") is not True
        or not isinstance(lanes, list)
        or len(lanes) != len(CANDIDATES)
    ):
        raise PostdeadlineContractError("sealed prepare manifest drifted")
    authenticated = [
        _authenticate_lane_plan(
            resolved.parent,
            spec,
            lane,
            reauthenticate=reauthenticate,
        )
        for spec, lane in zip(CANDIDATES, lanes, strict=True)
    ]
    return manifest, authenticated


def _license_gate(value: Any, *, required_headroom: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PostdeadlineContractError(
            "Scheduler license inventory is malformed"
        )
    admission = value.get("admission")
    features = admission.get("features") if isinstance(admission, dict) else None
    feature = (
        features.get(LICENSE_FEATURE)
        if isinstance(features, dict)
        else None
    )
    reserve = (
        admission.get("reserve_by_feature", {}).get(LICENSE_FEATURE)
        if isinstance(admission, dict)
        and isinstance(admission.get("reserve_by_feature"), dict)
        else None
    )
    if (
        value.get("server_up") is not True
        or not isinstance(admission, dict)
        or admission.get("enabled") is not True
        or admission.get("snapshot_valid") is not True
        or float(admission.get("snapshot_age_seconds") or 10**9)
        > float(admission.get("snapshot_max_age_seconds") or 0)
        or admission.get("blocked_reason") not in {"", None}
        or not isinstance(feature, dict)
        or int(feature.get("admit_headroom") or 0) < required_headroom
        or int(reserve or 0) < LICENSE_RESERVE_FLOOR
    ):
        raise PostdeadlineContractError(
            "Scheduler electronics_desktop license gate failed"
        )
    return {
        "feature": LICENSE_FEATURE,
        "admit_headroom": int(feature["admit_headroom"]),
        "reserve": int(reserve),
        "snapshot_age_seconds": float(admission["snapshot_age_seconds"]),
        "snapshot_max_age_seconds": float(
            admission["snapshot_max_age_seconds"]
        ),
    }


def _parse_scheduler_time(value: Any) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            raise PostdeadlineContractError(
                "node metrics timestamp is malformed"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _node_gate(
    value: Any,
    *,
    account_name: str,
    node_name: str,
    observed_at: datetime,
) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("allocations"), list):
        rows = value["allocations"]
    elif isinstance(value, list):
        rows = value
    else:
        raise PostdeadlineContractError(
            "Scheduler node inventory is malformed"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("node_name") == node_name
    ]
    if not matches:
        raise PostdeadlineContractError(
            f"fresh Scheduler node evidence is absent for {node_name}"
        )
    matches.sort(
        key=lambda row: (
            str(row.get("node_metrics_observed_at") or ""),
            int(row.get("id") or 0),
        ),
        reverse=True,
    )
    node = matches[0]
    live_states = {"pending", "warm", "active", "draining", "closing"}
    live_exact = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("account_name") == account_name
        and row.get("node_name") == node_name
        and row.get("state") in live_states
    ]
    metrics_at = _parse_scheduler_time(node.get("node_metrics_observed_at"))
    age = (observed_at - metrics_at).total_seconds()
    if (
        node.get("node_pestat_state") not in NODE_STATE_ALLOWLIST
        or int(node.get("node_cpu_total") or 0) < CPUS
        or int(node.get("node_memory_total_mb") or 0) < MEMORY_MB
        or int(node.get("node_memory_free_mb") or 0) < MEMORY_MB
        or age < -60
        or age > MAX_NODE_METRICS_AGE_SECONDS
        or live_exact
    ):
        raise PostdeadlineContractError(
            f"fresh Scheduler node capacity gate failed for {node_name}"
        )
    return {
        "node_name": node_name,
        "state": node["node_pestat_state"],
        "cpu_total": int(node["node_cpu_total"]),
        "cpu_used": int(node.get("node_cpu_used") or 0),
        "memory_total_mb": int(node["node_memory_total_mb"]),
        "memory_free_mb": int(node["node_memory_free_mb"]),
        "metrics_observed_at": metrics_at.isoformat(),
        "metrics_age_seconds": age,
        "exact_account_node_live_allocation_count": 0,
        "exact_account_node_live_allocation_must_be_absent": True,
    }


def _account_storage_gate(account: Any) -> dict[str, Any]:
    if not isinstance(account, dict):
        raise PostdeadlineContractError(
            "Scheduler account evidence is malformed"
        )
    running = int(account.get("running") or 0)
    pending = int(account.get("pending") or 0)
    max_running = int(account.get("max_running") or 0)
    max_pending = int(account.get("max_pending") or 0)
    max_total = int(account.get("max_total") or 0)
    if (
        running >= max_running
        or pending >= max_pending
        or running + pending >= max_total
    ):
        raise PostdeadlineContractError(
            "Scheduler account run/pending/total cap gate failed"
        )
    used = account.get("storage_used_gb")
    quota = account.get("storage_quota_gb")
    if (used is None) is not (quota is None):
        raise PostdeadlineContractError(
            "Scheduler account storage cap evidence is incomplete"
        )
    if used is not None:
        free = float(quota) - float(used)
        if free < MIN_STORAGE_FREE_GB:
            raise PostdeadlineContractError(
                "Scheduler account storage cap gate failed"
            )
        storage = {
            "mode": "declared_quota",
            "used_gb": float(used),
            "quota_gb": float(quota),
            "free_gb": free,
        }
    else:
        if account.get("storage_path") != "slurm_scheduler":
            raise PostdeadlineContractError(
                "Scheduler account storage guard path drifted"
            )
        storage = {
            "mode": "scheduler_fail_closed_runtime_guard",
            "storage_path": "slurm_scheduler",
            "minimum_free_gb": MIN_STORAGE_FREE_GB,
            "quota_fields_unset": True,
            "live_quota_headroom_available": False,
            "storage_headroom_claimed": False,
        }
    return {
        "running": running,
        "pending": pending,
        "max_running": max_running,
        "max_pending": max_pending,
        "max_total": max_total,
        "storage": storage,
    }


def _authority_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PostdeadlineContractError(
            f"deployed Scheduler {label} is unavailable"
        ) from exc
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise PostdeadlineContractError(
            f"deployed Scheduler {label} bytes drifted"
        )
    return file_record(resolved)


def authenticate_runtime_walltime_contract() -> dict[str, Any]:
    """Bind the live launcher and deployed 4facdfe walltime source bytes."""

    launcher = _authority_file(
        LIVE_LAUNCHER, LIVE_LAUNCHER_SHA256, "launcher"
    )
    config = _authority_file(LIVE_CONFIG, LIVE_CONFIG_SHA256, "live config")
    config_py = _authority_file(
        DEPLOYED_CONFIG_PY,
        DEPLOYED_CONFIG_PY_SHA256,
        "deployed config.py",
    )
    scheduler_py = _authority_file(
        DEPLOYED_SCHEDULER_PY,
        DEPLOYED_SCHEDULER_PY_SHA256,
        "deployed scheduler.py",
    )
    cutover = _authority_file(
        CUTOVER_RECEIPT, CUTOVER_RECEIPT_SHA256, "cutover receipt"
    )
    try:
        launcher_text = LIVE_LAUNCHER.read_text(
            encoding="utf-8", errors="strict"
        )
        config_text = LIVE_CONFIG.read_text(
            encoding="utf-8", errors="strict"
        )
        config_py_text = DEPLOYED_CONFIG_PY.read_text(
            encoding="utf-8", errors="strict"
        )
        scheduler_text = DEPLOYED_SCHEDULER_PY.read_text(
            encoding="utf-8", errors="strict"
        )
        cutover_value = json.loads(
            CUTOVER_RECEIPT.read_text(encoding="utf-8-sig")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PostdeadlineContractError(
            "deployed Scheduler walltime authority cannot be read"
        ) from exc
    forbidden_live_overrides = (
        "allocation_time_limit:",
        "allocation_drain_after_seconds:",
        "allocation_attach_stop_before_drain_seconds:",
        "allocation_force_cancel_after_seconds:",
    )
    launcher_contract = (
        DEPLOYED_COMMIT in launcher_text
        and str(DEPLOYED_ROOT) in launcher_text
        and str(LIVE_CONFIG) in launcher_text
        and "8002" in launcher_text
    )
    defaults_contract = all(
        text in config_py_text
        for text in (
            'allocation_time_limit: str = "48:00:00"',
            "allocation_drain_after_seconds: int = 129600",
            "allocation_attach_stop_before_drain_seconds: int = 1800",
            "allocation_force_cancel_after_seconds: int = 140400",
            "allocation_pending_timeout_seconds: int = 1800",
        )
    )
    scheduler_contract = all(
        text in scheduler_text
        for text in (
            "cutoff = max(0, self.allocation_drain_after_seconds - stop_before)",
            "return self._age_seconds(allocation) < cutoff",
            "A failed or unavailable GPFS probe fails closed for new work",
            'return True, False',
            'if (allocation.get("resource_pool") or "") != "gpu:a6000":',
        )
    )
    cutover_contract = (
        isinstance(cutover_value, dict)
        and cutover_value.get("to_commit") == DEPLOYED_COMMIT
        and cutover_value.get("launcher_sha256") == LIVE_LAUNCHER_SHA256
        and cutover_value.get("scheduler_ok") is True
        and cutover_value.get("scheduler_thread_alive") is True
        and cutover_value.get("scheduler_stalled") is False
    )
    if (
        any(key in config_text for key in forbidden_live_overrides)
        or "storage_guard_min_free_gb: 10" not in config_text
        or "aedt_storage_reservation_per_project_gb: 4" not in config_text
        or not launcher_contract
        or not defaults_contract
        or not scheduler_contract
        or not cutover_contract
        or REVIEWED_ALLOCATION_WALLTIME_SECONDS != 172_800
        or ALLOCATION_ATTACH_WINDOW_SECONDS != 127_800
        or ALLOCATION_ATTACH_WINDOW_SECONDS
        < MIN_ALLOCATION_WALLTIME_SECONDS
    ):
        raise PostdeadlineContractError(
            "deployed Scheduler >=14h walltime source contract drifted"
        )
    return sealed(
        {
            "schema_version": (
                "mft-goal-scheduler-runtime-walltime-authority-v1"
            ),
            "deployed_commit": DEPLOYED_COMMIT,
            "launcher": launcher,
            "live_config": config,
            "deployed_config_py": config_py,
            "deployed_scheduler_py": scheduler_py,
            "cutover_receipt": cutover,
            "live_config_lifecycle_overrides_absent": True,
            "allocation_walltime_seconds": (
                REVIEWED_ALLOCATION_WALLTIME_SECONDS
            ),
            "allocation_drain_after_seconds": (
                ALLOCATION_DRAIN_AFTER_SECONDS
            ),
            "allocation_attach_stop_before_drain_seconds": (
                ALLOCATION_ATTACH_STOP_BEFORE_DRAIN_SECONDS
            ),
            "fresh_pool_attach_window_seconds": (
                ALLOCATION_ATTACH_WINDOW_SECONDS
            ),
            "ordinary_cpu_pending_timeout_seconds": (
                ALLOCATION_PENDING_TIMEOUT_SECONDS
            ),
            "ordinary_cpu_pending_timeout_exempt": False,
            "minimum_required_residual_seconds": (
                MIN_ALLOCATION_WALLTIME_SECONDS
            ),
            "nominal_fresh_pool_attach_window_exceeds_minimum": True,
            "hard_residual_seconds_api_proven": False,
            "hard_residual_guarantee": False,
            "operational_source_inference": True,
            "storage_guard_min_free_gb": MIN_STORAGE_FREE_GB,
            "storage_guard_probe_unavailable_fails_closed": True,
        }
    )


def live_submission_gate(
    lane: Mapping[str, Any],
    *,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
    required_license_headroom: int = 1,
    runtime_authenticator: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repeat strict opening, account, license, and node GET gates."""

    spec = lane["spec"]
    plan = lane["plan"]
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError("live gate time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    base = prepare_only.live_preflight(
        spec,
        expected_dedupe_key=plan["dedupe_key"],
        reader=reader,
        observed_at=now,
    )
    license_evidence = _license_gate(
        reader("/api/licenses", None),
        required_headroom=required_license_headroom,
    )
    node_evidence = _node_gate(
        reader("/api/allocations", [("limit", 10000)]),
        account_name=spec.account_name,
        node_name=spec.node_name,
        observed_at=now,
    )
    account_evidence = _account_storage_gate(base.get("account_status"))
    runtime_walltime = (
        runtime_authenticator or authenticate_runtime_walltime_contract
    )()
    if (
        base.get("queue_state") != "opening"
        or base.get("ready_fit_slots") != 0
        or base.get("pending_fit_slots") != 0
        or base.get("inflight_fit_slots") != 0
        or base.get("preferred_node_relaxed") is not False
        or base.get("active_target_fea_count") != 0
        or base.get("collision_count") != 0
        or spec.account_name in prepare_only.EXCLUDED_ACCOUNT_GUARDS
    ):
        raise PostdeadlineContractError(
            "strict opening/collision/relaxation gate drifted"
        )
    return sealed(
        {
            "schema_version": LIVE_GATE_SCHEMA,
            **SAFETY_FLAGS,
            "observed_at_utc": now.isoformat(),
            "selection_order": spec.selection_order,
            "task_name": spec.task_name,
            "dedupe_key": plan["dedupe_key"],
            "account_name": spec.account_name,
            "node_name": spec.node_name,
            "node_name_policy": "strict",
            "strict_opening_preflight": base,
            "account_cap_and_storage_gate": account_evidence,
            "license_gate": license_evidence,
            "node_gate": node_evidence,
            "allocation_walltime_contract": {
                "minimum_seconds": MIN_ALLOCATION_WALLTIME_SECONDS,
                "fresh_pool_attach_window_seconds": (
                    ALLOCATION_ATTACH_WINDOW_SECONDS
                ),
                "task_timeout_seconds": SCHEDULER_SECONDS,
                "fresh_strict_opening_requires_new_pool": True,
                "capacity_allocations_empty": (
                    base.get("capacity", {}).get("allocations") == []
                ),
                "residual_proof_basis": (
                    "fresh_opening_runtime_source_contract"
                ),
                "api_residual_seconds_proven": False,
                "source_contract_derived_not_api_residual": True,
                "hard_residual_guarantee": False,
                "operational_source_inference": True,
                "runtime_source_authority": runtime_walltime,
            },
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    )


def _lane_output(root: Path, spec: CandidateSpec) -> Path:
    return root / spec.lane_name


def _intent_payload(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    lanes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return sealed(
        {
            "schema_version": BATCH_INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepare_manifest": file_record(manifest_path),
            "prepare_manifest_payload_sha256": manifest["payload_sha256"],
            "authorization_contract": AUTHORIZATION,
            "run_command_argv": copy.deepcopy(RUN_COMMAND_ARGV),
            "lane_count": len(lanes),
            "lane_nonces": [
                {
                    "selection_order": lane["spec"].selection_order,
                    "lane_name": lane["spec"].lane_name,
                    "lane_nonce": lane["lane_nonce"],
                    "task_name": lane["spec"].task_name,
                    "dedupe_key": lane["plan"]["dedupe_key"],
                    "account_name": lane["spec"].account_name,
                    "node_name": lane["spec"].node_name,
                    "post_call_budget": 1,
                }
                for lane in lanes
            ],
            "total_post_call_budget": len(lanes),
            "attempt_ledger_consumed_before_each_network_call": True,
            "ambiguous_post_retry_allowed": False,
            "node_relaxation_allowed": False,
            "minimum_allocation_walltime_seconds": (
                MIN_ALLOCATION_WALLTIME_SECONDS
            ),
            "reviewed_allocation_walltime_seconds": (
                REVIEWED_ALLOCATION_WALLTIME_SECONDS
            ),
            "walltime_evidence_classification": (
                "fresh_opening_runtime_source_contract"
            ),
            "api_residual_seconds_proven": False,
        }
    )


def _load_or_create_intent(
    root: Path,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    lanes: Sequence[Mapping[str, Any]],
) -> Path:
    expected = _intent_payload(manifest_path, manifest, lanes)
    path = root / BATCH_INTENT_NAME
    if root.exists():
        if not root.is_dir() or not path.is_file():
            raise PostdeadlineContractError(
                "existing submission root lacks its immutable intent"
            )
        current = validate_seal(read_json(path), BATCH_INTENT_SCHEMA)
        comparable = dict(current)
        comparable["created_at_utc"] = expected["created_at_utc"]
        comparable.pop("payload_sha256", None)
        expected_unsigned = dict(expected)
        expected_unsigned.pop("payload_sha256", None)
        if comparable != expected_unsigned:
            raise PostdeadlineContractError(
                "existing batch submission intent drifted"
            )
        return path
    root.mkdir(parents=True)
    return write_immutable_json(path, expected)


def _inventory_exact(
    reader: JsonReader, lane: Mapping[str, Any]
) -> list[dict[str, Any]]:
    spec = lane["spec"]
    rows = reviewed._task_rows(
        reader(
            "/api/tasks",
            [
                ("limit", 10000),
                ("project", PROJECT),
                ("name_prefix", spec.task_name),
            ],
        )
    )
    return [
        row
        for row in rows
        if row.get("name") == spec.task_name
        and row.get("dedupe_key") == lane["plan"]["dedupe_key"]
    ]


def _authenticated_readback(
    reader: JsonReader,
    lane: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    spec = lane["spec"]
    plan = lane["plan"]
    readback = reader(f"/api/tasks/{task_id}", None)
    expected = {
        "name": spec.task_name,
        "dedupe_key": plan["dedupe_key"],
        "project": PROJECT,
        "requested_account_name": spec.account_name,
        "requested_node_name": spec.node_name,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "timeout_seconds": SCHEDULER_SECONDS,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "aedt_backend": "standalone",
    }
    drift = {
        key: {"expected": value, "actual": readback.get(key)}
        for key, value in expected.items()
        if not isinstance(readback, dict) or readback.get(key) != value
    }
    if (
        drift
        or readback.get("status") not in {"queued", "attaching", "running"}
        or readback.get("preferred_node_relaxed") is not False
        or readback.get("node_name_policy") not in {None, "", "strict"}
        or (
            readback.get("account_name")
            and readback.get("account_name") != spec.account_name
        )
        or (
            readback.get("node_name")
            and readback.get("node_name") != spec.node_name
        )
        or (
            readback.get("actual_node_name")
            and readback.get("actual_node_name") != spec.node_name
        )
        or (
            readback.get("allocation_node_name")
            and readback.get("allocation_node_name") != spec.node_name
        )
    ):
        raise PostdeadlineContractError(
            f"durable strict task readback drifted: {drift}"
        )
    return copy.deepcopy(readback)


def _existing_lane_result(
    lane_root: Path,
    lane: Mapping[str, Any],
    *,
    reader: JsonReader,
) -> Path | None:
    attempt_path = lane_root / ATTEMPT_NAME
    final_path = lane_root / LANE_FINAL_NAME
    if not attempt_path.exists():
        if any(
            (lane_root / name).exists()
            for name in (RECEIPT_NAME, LANE_FINAL_NAME, AMBIGUOUS_NAME)
        ):
            raise PostdeadlineContractError(
                "lane output exists without its consumed attempt ledger"
            )
        return None
    attempt = validate_seal(read_json(attempt_path), ATTEMPT_SCHEMA)
    if (
        attempt.get("lane_nonce") != lane["lane_nonce"]
        or attempt.get("post_call_consumed_before_network") is not True
        or attempt.get("post_call_budget") != 1
    ):
        raise PostdeadlineContractError(
            "existing lane attempt ledger drifted"
        )
    if final_path.is_file():
        final = validate_seal(read_json(final_path), LANE_FINAL_SCHEMA)
        receipt_path = _contained_record(
            lane_root, final.get("receipt"), "existing lane receipt"
        )
        receipt = validate_seal(read_json(receipt_path), RECEIPT_SCHEMA)
        if (
            final.get("lane_nonce") != lane["lane_nonce"]
            or receipt.get("lane_nonce") != lane["lane_nonce"]
            or receipt.get("scheduler_post_calls") != 1
        ):
            raise PostdeadlineContractError(
                "existing lane final evidence drifted"
            )
        _authenticated_readback(reader, lane, int(receipt["task_id"]))
        return final_path
    exact = _inventory_exact(reader, lane)
    if len(exact) == 1:
        task_id = exact[0].get("task_id") or exact[0].get("id")
        if task_id is not None:
            readback = _authenticated_readback(reader, lane, int(task_id))
            return _finish_lane(
                lane_root,
                lane,
                attempt_path=attempt_path,
                task_id=int(task_id),
                readback=readback,
                http_status=None,
                response=None,
                error="reconciled consumed attempt by exact GET",
                reconciled=True,
            )
    if not (lane_root / AMBIGUOUS_NAME).exists():
        _write_ambiguous(
            lane_root,
            lane,
            attempt_path=attempt_path,
            http_status=None,
            response=None,
            error="consumed attempt has no exact GET reconciliation",
            exact_count=len(exact),
        )
    raise PostdeadlineContractError(
        f"official{lane['spec'].selection_order} POST attempt is "
        "already consumed ambiguously; retry is forbidden"
    )


def _load_or_create_lane_intent(
    lane_root: Path,
    lane: Mapping[str, Any],
    *,
    batch_intent_path: Path,
    initial_gate: Mapping[str, Any],
) -> Path:
    path = lane_root / LANE_INTENT_NAME
    if path.exists():
        intent = validate_seal(read_json(path), LANE_INTENT_SCHEMA)
        if (
            intent.get("lane_nonce") != lane["lane_nonce"]
            or intent.get("selection_order")
            != lane["spec"].selection_order
            or intent.get("post_call_budget") != 1
            or intent.get("retry_after_ambiguity_allowed") is not False
        ):
            raise PostdeadlineContractError(
                "existing lane submission intent drifted"
            )
        return path
    if not initial_gate:
        raise PostdeadlineContractError(
            "fresh lane submission intent lacks its initial live gate"
        )
    intent = sealed(
        {
            "schema_version": LANE_INTENT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_order": lane["spec"].selection_order,
            "candidate_physics_sha256": (
                lane["spec"].candidate_sha256
            ),
            "lane_nonce": lane["lane_nonce"],
            "batch_intent": file_record(batch_intent_path),
            "prepare_plan": file_record(lane["plan_path"]),
            "prepare_plan_payload_sha256": lane["plan"][
                "payload_sha256"
            ],
            "scheduler_payload_sha256": lane["plan"][
                "scheduler_payload_sha256"
            ],
            "task_name": lane["spec"].task_name,
            "dedupe_key": lane["plan"]["dedupe_key"],
            "account_name": lane["spec"].account_name,
            "node_name": lane["spec"].node_name,
            "node_name_policy": "strict",
            "initial_live_gate": copy.deepcopy(initial_gate),
            "post_call_budget": 1,
            "attempt_ledger_must_precede_network": True,
            "retry_after_ambiguity_allowed": False,
        }
    )
    return write_immutable_json(path, intent)


def _write_ambiguous(
    lane_root: Path,
    lane: Mapping[str, Any],
    *,
    attempt_path: Path,
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    exact_count: int,
) -> Path:
    artifact = sealed(
        {
            "schema_version": AMBIGUOUS_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_order": lane["spec"].selection_order,
            "lane_nonce": lane["lane_nonce"],
            "attempt_ledger": file_record(attempt_path),
            "http_status": http_status,
            "http_response": copy.deepcopy(response),
            "http_error": error,
            "exact_get_reconciliation_count": exact_count,
            "scheduler_post_calls": 1,
            "attempt_consumed": True,
            "retry_allowed": False,
        }
    )
    return write_immutable_json(lane_root / AMBIGUOUS_NAME, artifact)


def _finish_lane(
    lane_root: Path,
    lane: Mapping[str, Any],
    *,
    attempt_path: Path,
    task_id: int,
    readback: Mapping[str, Any],
    http_status: int | None,
    response: Mapping[str, Any] | None,
    error: str | None,
    reconciled: bool,
) -> Path:
    receipt = sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_order": lane["spec"].selection_order,
            "candidate_physics_sha256": (
                lane["spec"].candidate_sha256
            ),
            "lane_nonce": lane["lane_nonce"],
            "plan": file_record(lane["plan_path"]),
            "plan_payload_sha256": lane["plan"]["payload_sha256"],
            "attempt_ledger": file_record(attempt_path),
            "scheduler_payload_sha256": lane["plan"][
                "scheduler_payload_sha256"
            ],
            "scheduler_post_calls": 1,
            "scheduler_post_http_status": http_status,
            "scheduler_post_response": copy.deepcopy(response),
            "scheduler_post_error": error,
            "exact_get_reconciled": reconciled,
            "task_id": task_id,
            "task_name": lane["spec"].task_name,
            "dedupe_key": lane["plan"]["dedupe_key"],
            "account_name": lane["spec"].account_name,
            "node_name": lane["spec"].node_name,
            "node_name_policy": "strict",
            "task_readback": copy.deepcopy(readback),
            "task_readback_sha256": payload_sha256(readback),
            "preferred_node_relaxed": False,
            "scheduler_repository_modified": False,
            "scheduler_project_mutation_performed": False,
        }
    )
    receipt_path = write_immutable_json(
        lane_root / RECEIPT_NAME, receipt
    )
    final = sealed(
        {
            "schema_version": LANE_FINAL_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_order": lane["spec"].selection_order,
            "lane_nonce": lane["lane_nonce"],
            "attempt_ledger": file_record(attempt_path),
            "receipt": file_record(receipt_path),
            "receipt_payload_sha256": receipt["payload_sha256"],
            "task_id": task_id,
            "scheduler_post_calls": 1,
            "immutable_evidence_complete": True,
        }
    )
    return write_immutable_json(lane_root / LANE_FINAL_NAME, final)


def _submit_lane(
    root: Path,
    lane: Mapping[str, Any],
    *,
    intent_path: Path,
    initial_gate: Mapping[str, Any],
    reader: JsonReader,
    poster: Poster,
    observed_at: datetime | None,
    lock_factory: Callable[[], Any],
) -> Path:
    spec = lane["spec"]
    lane_root = _lane_output(root, spec)
    lane_root.mkdir(exist_ok=True)
    existing = _existing_lane_result(lane_root, lane, reader=reader)
    if existing is not None:
        return existing
    lane_intent_path = _load_or_create_lane_intent(
        lane_root,
        lane,
        batch_intent_path=intent_path,
        initial_gate=initial_gate,
    )
    with lock_factory():
        existing = _existing_lane_result(lane_root, lane, reader=reader)
        if existing is not None:
            return existing
        locked_gate = live_submission_gate(
            lane,
            reader=reader,
            observed_at=observed_at,
            required_license_headroom=1,
        )
        attempt = sealed(
            {
                "schema_version": ATTEMPT_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_order": spec.selection_order,
                "lane_nonce": lane["lane_nonce"],
                "batch_intent": file_record(intent_path),
                "lane_intent": file_record(lane_intent_path),
                "prepare_plan": file_record(lane["plan_path"]),
                "prepare_plan_payload_sha256": lane["plan"][
                    "payload_sha256"
                ],
                "scheduler_payload_sha256": lane["plan"][
                    "scheduler_payload_sha256"
                ],
                "initial_live_gate": copy.deepcopy(initial_gate),
                "locked_pre_submit_live_gate": locked_gate,
                "campaign_mutation_lock_acquired": True,
                "post_call_budget": 1,
                "post_call_consumed_before_network": True,
                "scheduler_post_calls_before": 0,
                "scheduler_post_calls_authorized": 1,
                "ambiguous_retry_allowed": False,
            }
        )
        attempt_path = write_exclusive_json(
            lane_root / ATTEMPT_NAME, attempt
        )
        try:
            status, response, error = poster(
                f"{SCHEDULER_URL}/api/tasks",
                lane["plan"]["scheduler_payload"],
            )
        except Exception as exc:
            status, response, error = None, None, str(exc)

    task_id = (
        response.get("task_id") or response.get("id")
        if isinstance(response, dict)
        else None
    )
    reconciled = False
    exact: list[dict[str, Any]] = []
    if task_id is None or status != 201:
        exact = _inventory_exact(reader, lane)
        if len(exact) == 1:
            task_id = exact[0].get("task_id") or exact[0].get("id")
            reconciled = task_id is not None
    if task_id is None:
        _write_ambiguous(
            lane_root,
            lane,
            attempt_path=attempt_path,
            http_status=status,
            response=response,
            error=error,
            exact_count=len(exact),
        )
        raise PostdeadlineContractError(
            f"official{spec.selection_order} POST outcome is ambiguous; "
            "attempt is consumed and retry is forbidden"
        )
    readback = _authenticated_readback(reader, lane, int(task_id))
    return _finish_lane(
        lane_root,
        lane,
        attempt_path=attempt_path,
        task_id=int(task_id),
        readback=readback,
        http_status=status,
        response=response,
        error=error,
        reconciled=reconciled,
    )


def submit_batch(
    *,
    authorize_post: str,
    manifest_path: Path = PREPARE_MANIFEST,
    output_root: Path = OUTPUT_ROOT,
    reader: JsonReader = get_json,
    poster: Poster = reviewed._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    package_authenticator: Callable[
        [Path], tuple[dict[str, Any], list[dict[str, Any]]]
    ] = authenticate_prepare_package,
) -> Path:
    """Submit each authenticated lane at most once, resuming safe partials."""

    if authorize_post != AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit hedge3 one-shot POST authorization is absent"
        )
    target = output_root.resolve()
    if target != OUTPUT_ROOT.resolve():
        raise PostdeadlineContractError("submission output root is not fixed")
    manifest, lanes = package_authenticator(manifest_path)
    if [lane["spec"] for lane in lanes] != list(CANDIDATES):
        raise PostdeadlineContractError(
            "authenticated lane order/specification drifted"
        )
    intent_path = _load_or_create_intent(
        target,
        manifest_path=manifest_path.resolve(strict=True),
        manifest=manifest,
        lanes=lanes,
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for lane in lanes:
        spec = lane["spec"]
        lane_root = _lane_output(target, spec)
        try:
            if lane_root.exists() and (lane_root / ATTEMPT_NAME).exists():
                initial_gate: dict[str, Any] = {}
            else:
                initial_gate = live_submission_gate(
                    lane,
                    reader=reader,
                    observed_at=observed_at,
                    required_license_headroom=1,
                )
            final_path = _submit_lane(
                target,
                lane,
                intent_path=intent_path,
                initial_gate=initial_gate,
                reader=reader,
                poster=poster,
                observed_at=observed_at,
                lock_factory=lock_factory,
            )
            final = validate_seal(
                read_json(final_path), LANE_FINAL_SCHEMA
            )
            lane_intent_path = lane_root / LANE_INTENT_NAME
            attempt_path = lane_root / ATTEMPT_NAME
            receipt_path = lane_root / RECEIPT_NAME
            validate_seal(read_json(lane_intent_path), LANE_INTENT_SCHEMA)
            validate_seal(read_json(attempt_path), ATTEMPT_SCHEMA)
            receipt = validate_seal(
                read_json(receipt_path), RECEIPT_SCHEMA
            )
            if receipt.get("task_id") != final.get("task_id"):
                raise PostdeadlineContractError(
                    "lane receipt/final task identity drifted"
                )
            results.append(
                {
                    "selection_order": spec.selection_order,
                    "candidate_physics_sha256": spec.candidate_sha256,
                    "lane_nonce": lane["lane_nonce"],
                    "task_name": spec.task_name,
                    "dedupe_key": lane["plan"]["dedupe_key"],
                    "account_name": spec.account_name,
                    "node_name": spec.node_name,
                    "node_name_policy": "strict",
                    "task_id": final["task_id"],
                    "prepare_plan": file_record(lane["plan_path"]),
                    "lane_intent": file_record(lane_intent_path),
                    "attempt_ledger": file_record(attempt_path),
                    "submission_receipt": file_record(receipt_path),
                    "lane_final_seal": file_record(final_path),
                }
            )
        except PostdeadlineContractError as exc:
            failures.append(
                {
                    "selection_order": spec.selection_order,
                    "lane_nonce": lane["lane_nonce"],
                    "error": str(exc),
                    "attempt_consumed": (
                        lane_root / ATTEMPT_NAME
                    ).is_file(),
                }
            )
    if failures:
        failure_path = target / BATCH_FAILURE_NAME
        failure = sealed(
            {
                "schema_version": BATCH_FAILURE_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "batch_intent": file_record(intent_path),
                "successful_lanes": results,
                "failed_lanes": failures,
                "successful_lane_count": len(results),
                "failed_lane_count": len(failures),
                "ambiguous_retry_allowed": False,
            }
        )
        if not failure_path.exists():
            write_immutable_json(failure_path, failure)
        raise PostdeadlineContractError(
            f"hedge3 submission incomplete: {failures}"
        )
    aggregate = sealed(
        {
            "schema_version": AGGREGATE_SUBMIT_MANIFEST_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepare_manifest": file_record(
                manifest_path.resolve(strict=True)
            ),
            "prepare_manifest_payload_sha256": manifest["payload_sha256"],
            "batch_intent": file_record(intent_path),
            "authorization_contract": AUTHORIZATION,
            "run_command_argv": copy.deepcopy(RUN_COMMAND_ARGV),
            "lanes": results,
            "lane_count": len(results),
            "selection_orders": [
                result["selection_order"] for result in results
            ],
            "task_ids": [result["task_id"] for result in results],
            "task_names": [result["task_name"] for result in results],
            "strict_nodes": [result["node_name"] for result in results],
            "scheduler_post_call_budget": len(CANDIDATES),
            "scheduler_post_calls_evidenced": len(results),
            "maximum_one_post_per_lane": True,
            "node_relaxation_allowed": False,
            "get_only_collector_ready": True,
        }
    )
    aggregate_path = target / AGGREGATE_SUBMIT_MANIFEST_NAME
    if aggregate_path.exists():
        current_aggregate = validate_seal(
            read_json(aggregate_path), AGGREGATE_SUBMIT_MANIFEST_SCHEMA
        )
        if current_aggregate.get("lanes") != results:
            raise PostdeadlineContractError(
                "aggregate submit manifest drifted"
            )
        aggregate = current_aggregate
    else:
        aggregate_path = write_immutable_json(aggregate_path, aggregate)
    final = sealed(
        {
            "schema_version": BATCH_FINAL_SCHEMA,
            **SAFETY_FLAGS,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_intent": file_record(intent_path),
            "prepare_manifest": file_record(
                manifest_path.resolve(strict=True)
            ),
            "prepare_manifest_payload_sha256": manifest["payload_sha256"],
            "aggregate_submit_manifest": file_record(aggregate_path),
            "aggregate_submit_manifest_payload_sha256": aggregate[
                "payload_sha256"
            ],
            "lanes": results,
            "lane_count": len(results),
            "scheduler_post_call_budget": len(CANDIDATES),
            "maximum_one_post_per_lane": True,
            "strict_nodes": [spec.node_name for spec in CANDIDATES],
            "node_relaxation_allowed": False,
            "immutable_evidence_complete": True,
        }
    )
    final_path = target / BATCH_FINAL_NAME
    if final_path.exists():
        current = validate_seal(read_json(final_path), BATCH_FINAL_SCHEMA)
        if current.get("lanes") != results:
            raise PostdeadlineContractError("batch final seal drifted")
        return final_path
    return write_immutable_json(final_path, final)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate and one-shot submit the sealed official hedge3"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect",
        help="reauthenticate the package and repeat all GET-only gates",
    )
    submit = commands.add_parser(
        "submit", help="submit each lane with a lifetime budget of one POST"
    )
    submit.add_argument("--authorize-post", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "inspect":
        manifest, lanes = authenticate_prepare_package()
        gates = [live_submission_gate(lane) for lane in lanes]
        result = {
            "prepare_manifest": str(PREPARE_MANIFEST),
            "prepare_manifest_sha256": sha256_file(PREPARE_MANIFEST),
            "prepare_manifest_payload_sha256": manifest["payload_sha256"],
            "lane_count": len(lanes),
            "gates": gates,
            "scheduler_post_calls": 0,
            "authorization_contract": AUTHORIZATION,
            "run_command_argv": copy.deepcopy(RUN_COMMAND_ARGV),
        }
    else:
        final_path = submit_batch(authorize_post=args.authorize_post)
        final = validate_seal(read_json(final_path), BATCH_FINAL_SCHEMA)
        result = {
            "final_seal": str(final_path),
            "final_seal_sha256": sha256_file(final_path),
            "task_ids": [lane["task_id"] for lane in final["lanes"]],
            "lane_count": final["lane_count"],
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
