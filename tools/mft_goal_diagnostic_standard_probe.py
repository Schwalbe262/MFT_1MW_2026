"""Diagnostic-only Standard FEA truth probes for the 2026-07-26 MFT goal.

This tool is intentionally separate from ``mft_goal_fea_handoff.py``.  It
accepts only authenticated terminal candidates whose non-Llt hard constraints
pass, replays the pinned Llt model to prove that the surrogate mean is in-band,
and can submit only an eighth-symmetry Standard solve.

There is no Full command, no production package command, and no automatic
promotion.  Even a physically passing Standard result remains diagnostic
evidence until a separate production authority consumes it.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_N1_MAX_TURNS,
    GOAL_N1_MIN_TURNS,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_TARGETS,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    attest_fixed_identity,
    canonical_sha256,
    dynamic_core_group_violation,
    validate_cw1_mm,
    validate_goal_stage_spec,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_20260726_launch as launch  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


SELECTION_SCHEMA = "mft-goal-diagnostic-standard-selection-v1"
PLAN_SCHEMA = "mft-goal-diagnostic-standard-plan-v1"
SELECTED_SCHEMA = "mft-goal-diagnostic-standard-selected-candidate-v1"
SUBMISSION_SCHEMA = "mft-goal-diagnostic-standard-submission-v1"
COLLECTION_SCHEMA = "mft-goal-diagnostic-standard-collection-v1"
AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-diagnostic-standard-authenticated-collection-v1"
)
SCHEDULER_CUTOVER_SCHEMA = (
    "slurm-scheduler-prune-protection-cutover-receipt-v1"
)
DIAGNOSTIC_SCHEDULER_URL = "http://127.0.0.1:8002"
# The live project policy is explicitly validated for 500 concurrent tasks.
# Bind every diagnostic POST to that exact reviewed capacity instead of the
# scheduler client's legacy 300-task default.
GOAL_FEA_PROJECT_CAP = 500
SCHEDULER_MARKER_AWARE_REVISION = (
    "0800a8d204da3cf772c4848814008b5755c92739"
)
SCHEDULER_MARKER_AWARE_TREE = (
    "7377e526e81af6d171c136e9cc7346d9f96e140f"
)
SCHEDULER_RELEASE_MANIFEST_SHA256 = (
    "2b666a8fc6552cf2c18021d9a39fe6db1e3b76579bfd877852b5176d50e2a2c8"
)
SCHEDULER_LIVE_LAUNCHER_SHA256 = (
    "e26c3eeb0453cd5f049e9f3280213b46e9c194d149e139b34dc5d9947c137c3d"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_REVISION = (
    "e542c8a6350d0b7101aa786e61f79df88958f40d"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_TREE = (
    "2ff7b60477cb8ad1d09470740bf504226c676224"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_FROM_REVISION = (
    "22f6fb93d71f9e703b7169bb2ac2c460ca80ec99"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_LAUNCHER_SHA256 = (
    "e1c327bd986ca2f86dfa5e02fbb9a9c40bb7e3347bfafcd6aea2acc182da325a"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_ROLLBACK_LAUNCHER_SHA256 = (
    "3cea73b022b8bfda1bdfc935751a3874c508498c8e1de463dfe554433d3425e5"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_SHA256 = (
    "28fc55c60281cb9690eb84a442f149aed57dfb79623a83672f930f17f8c68348"
)
SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_SCHEMA = (
    "slurm-scheduler-cutover-receipt-v2"
)
# Only this generation may be used to create or submit a new strict retry.
# The e542 generation above remains accepted solely for authenticating
# historical plans, submissions, and terminal collections.
SCHEDULER_STRICT_NODE_REVISION = (
    "41b3b939368423f0fa586dbe06328c5e37add179"
)
SCHEDULER_STRICT_NODE_TREE = (
    "80427698045d24addaac363c9bd138ce032b44af"
)
SCHEDULER_STRICT_NODE_FROM_REVISION = (
    SCHEDULER_STRICT_NODE_LEGACY_E542_REVISION
)
SCHEDULER_STRICT_NODE_LAUNCHER_SHA256 = (
    "f6ee9de8ab4434c54bba65f8fe696de23266b99e2fcc6d06faf169a5648bf472"
)
SCHEDULER_STRICT_NODE_ROLLBACK_LAUNCHER_SHA256 = (
    SCHEDULER_STRICT_NODE_LEGACY_E542_LAUNCHER_SHA256
)
SCHEDULER_STRICT_NODE_CUTOVER_SHA256 = (
    "e3b795526bb7a7eddd4888467c8551dd27785988c29d76783c0bce5db36e14bc"
)
SCHEDULER_STRICT_NODE_LIVE_LAUNCHER = Path(
    "Y:/runtime/slurm_scheduler/start_web_y.cmd"
)
SCHEDULER_STRICT_NODE_CUTOVER_SCHEMA = (
    "slurm-scheduler-cutover-receipt-v3"
)
RESULTS_MANIFEST_SCHEMA = (
    scheduler_client.RETAINED_AEDT_RESULTS_MANIFEST_SCHEMA
)
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard.json"
)
TIMEOUT_RETRY_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_timeout_retry.json"
)
STANDARD_RESOURCES = {"cpus": 8, "timeout_seconds": 4 * 3600}
TIMEOUT_RETRY_RESOURCES = {"cpus": 8, "timeout_seconds": 8 * 3600}
TIMEOUT_RETRY_PROFILE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout-retry-profile-v1"
)
TIMEOUT_RETRY_EVIDENCE_SCHEMA = (
    "mft-goal-diagnostic-standard-timeout-retry-evidence-v1"
)
SAME_ALLOCATION_PLACEMENT_SCHEMA = (
    "mft-goal-diagnostic-same-allocation-placement-v1"
)
STRICT_NODE_PLACEMENT_SCHEMA = (
    "mft-goal-diagnostic-strict-node-placement-v1"
)
STRICT_NODE_SUBMISSION_SCHEMA = (
    "mft-goal-diagnostic-strict-node-submission-v1"
)
LLT_UNCERTAINTY_CONSTRAINTS = frozenset(
    {"Llt_robust_band", "Llt_ensemble_disagreement"}
)
SELECTION_ALGORITHM = (
    "eligible_rows_sorted_by_Llt_uncertainty_then_objectives; "
    "one_best_per_available_N1_stratum_before_deterministic_farthest_fill"
)
MAX_SELECTION_COUNT = 12
MAX_REMOTE_METADATA_BYTES = production.MAX_REMOTE_METADATA_BYTES
HandoffContractError = production.HandoffContractError


def _diagnostic_flags() -> dict[str, bool]:
    return {
        "diagnostic_only": True,
        "standard_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "full_submission_allowed": False,
        "production_package_allowed": False,
    }


COLLECTION_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "stage",
        "plan",
        "plan_payload_sha256",
        "submission",
        "submission_payload_sha256",
        "selection_manifest",
        "source_result",
        "task_payload",
        "candidate_physics_sha256",
        "selected_candidate_identity",
        "task_id",
        "scheduler_url",
        "scheduler_status",
        "scheduler_task_execution",
        "result",
        "result_sha256",
        "result_identity",
        "remote_aedt_bundle_receipt",
        "aedtresults_manifest",
        "aedtresults_manifest_sha256",
        "prune_protection_marker",
        "prune_protection_marker_verified",
        "active_temperature_targets",
        "actual_body_probe_temperatures",
        "actual_body_probe_temperature_gate_passed",
        "goal_physical_spec_reasons",
        "goal_physical_spec_passed",
        "truth_evidence",
        "diagnostic_truth_observation_available",
        "scheduler_get_only_collection",
        "scheduler_mutation_performed",
        "retention_required",
        "prune_protection_required",
        *_diagnostic_flags(),
    }
)


SCHEDULER_CUTOVER_FIELDS = frozenset(
    {
        "schema_version",
        "scheduler_url",
        "candidate_revision",
        "candidate_tree",
        "release_manifest",
        "deployed_path",
        "live_launcher_path",
        "live_launcher_post_cutover_sha256",
        "cutover_at",
        "post_health",
        "marker",
        "preflight",
        "rollback_launcher_sha256",
        "created_at",
        "payload_sha256",
    }
)
SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_FIELDS = frozenset(
    {
        "schema_version",
        "cutover_at",
        "from_commit",
        "to_commit",
        "tree",
        "launcher_sha256",
        "database_backup",
        "database_backup_sha256",
        "pre_snapshot",
        "post_snapshot",
        "cohort_tasks_preserved",
        "active_tasks_pre",
        "active_tasks_post",
        "allowed_terminal_transitions",
        "scheduler_ok",
        "scheduler_thread_alive",
        "pressure_episode_migration_smoke",
        "database_quick_check",
        "rollback_launcher",
        "rollback_launcher_sha256",
    }
)
SCHEDULER_STRICT_NODE_CUTOVER_FIELDS = frozenset(
    {
        "schema_version",
        "cutover_at",
        "from_commit",
        "to_commit",
        "tree",
        "archive_sha256",
        "launcher_sha256",
        "cutover_guard_sha256",
        "database_migration",
        "configuration_change",
        "database_backup",
        "database_backup_sha256",
        "pre_snapshot",
        "pre_snapshot_sha256",
        "immediate_snapshot",
        "final_snapshot",
        "dynamic_campaign_selection",
        "campaign_tasks_preserved",
        "active_tasks_pre",
        "active_tasks_final",
        "allowed_transitions",
        "protected_cancelled_tasks",
        "allocation_14616_immediate_requested_owned",
        "allocation_14619_immediate_requested_owned",
        "extra_attach_to_14616_or_14619",
        "strict_same_node_cpu_gate",
        "final_fea_storage_admission_gate",
        "pressure_episode_preservation",
        "n114_pressure_episode_preserved",
        "database_quick_check",
        "scheduler_ok",
        "scheduler_thread_alive",
        "rollback_launcher",
        "rollback_launcher_sha256",
    }
)


def _aware_timestamp(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffContractError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HandoffContractError(f"{label} lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _absolute_regular_file(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
    ):
        raise HandoffContractError(
            f"{label} is not an absolute regular file"
        )
    return path.resolve(strict=True)


def _absolute_directory(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if (
        not path.is_absolute()
        or not path.is_dir()
        or path.is_symlink()
    ):
        raise HandoffContractError(
            f"{label} is not an absolute real directory"
        )
    return path.resolve(strict=True)


def _same_absolute_regular_file_identity(left: Any, right: Any) -> bool:
    """Compare two absolute file spellings after strict local resolution.

    Windows mapped-drive paths can resolve to the backing UNC path.  Submission
    records intentionally store that resolved identity, while the cutover
    receipt retains the operator-facing mapped-drive spelling.  Both inputs
    must still name an existing non-symlink regular file; byte identity is
    checked separately by the sealed launcher SHA.
    """

    try:
        left_path = _absolute_regular_file(
            left, "recorded Scheduler live launcher"
        )
        right_path = _absolute_regular_file(
            right, "cutover Scheduler live launcher"
        )
    except (HandoffContractError, OSError):
        return False
    return os.path.normcase(str(left_path)) == os.path.normcase(
        str(right_path)
    )


def _live_launcher_identity(
    receipt: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_sha256 is None:
        expected_sha256 = SCHEDULER_LIVE_LAUNCHER_SHA256
    launcher = _absolute_regular_file(
        receipt.get("live_launcher_path"),
        "Scheduler live launcher",
    )
    digest = production._sha256_file(launcher)
    if digest != expected_sha256:
        raise HandoffContractError(
            "Scheduler live launcher is not the reviewed cutover"
        )
    return {
        "path": str(launcher),
        "sha256": digest,
        "size_bytes": launcher.stat().st_size,
    }


def _legacy_e542_strict_node_scheduler_pin() -> dict[str, Any]:
    return {
        "pin_generation": "scheduler-strict-node-e542-v2",
        "revision": SCHEDULER_STRICT_NODE_LEGACY_E542_REVISION,
        "tree": SCHEDULER_STRICT_NODE_LEGACY_E542_TREE,
        "from_revision": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_FROM_REVISION
        ),
        "launcher_sha256": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_LAUNCHER_SHA256
        ),
        "rollback_launcher_sha256": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_ROLLBACK_LAUNCHER_SHA256
        ),
        "cutover_receipt_sha256": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_SHA256
        ),
        "cutover_receipt_schema": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_SCHEMA
        ),
        "cutover_fields": (
            SCHEDULER_STRICT_NODE_LEGACY_E542_CUTOVER_FIELDS
        ),
    }


def _active_strict_node_scheduler_pin() -> dict[str, Any]:
    return {
        "pin_generation": "scheduler-strict-node-41b-v3",
        "revision": SCHEDULER_STRICT_NODE_REVISION,
        "tree": SCHEDULER_STRICT_NODE_TREE,
        "from_revision": SCHEDULER_STRICT_NODE_FROM_REVISION,
        "launcher_sha256": SCHEDULER_STRICT_NODE_LAUNCHER_SHA256,
        "rollback_launcher_sha256": (
            SCHEDULER_STRICT_NODE_ROLLBACK_LAUNCHER_SHA256
        ),
        "cutover_receipt_sha256": (
            SCHEDULER_STRICT_NODE_CUTOVER_SHA256
        ),
        "cutover_receipt_schema": (
            SCHEDULER_STRICT_NODE_CUTOVER_SCHEMA
        ),
        "cutover_fields": SCHEDULER_STRICT_NODE_CUTOVER_FIELDS,
    }


def _accepted_strict_node_scheduler_pins() -> tuple[dict[str, Any], ...]:
    legacy = _legacy_e542_strict_node_scheduler_pin()
    active = _active_strict_node_scheduler_pin()
    if legacy["revision"] == active["revision"]:
        return (active,)
    return (legacy, active)


def _validate_strict_scheduler_cutover_receipt(
    path: Path,
    *,
    verify_live_launcher: bool,
    strict_node_contract: Mapping[str, Any] | None = None,
    require_active: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    pin = (
        _active_strict_node_scheduler_pin()
        if strict_node_contract is None
        else _strict_node_scheduler_pin(
            strict_node_contract, require_active=require_active
        )
    )
    resolved = path.resolve(strict=True)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "Scheduler strict-node cutover receipt is invalid JSON"
        ) from exc
    if (
        production._sha256_file(resolved)
        != pin["cutover_receipt_sha256"]
        or not isinstance(raw, dict)
        or set(raw) != pin["cutover_fields"]
        or raw.get("schema_version")
        != pin["cutover_receipt_schema"]
        or raw.get("from_commit") != pin["from_revision"]
        or raw.get("to_commit") != pin["revision"]
        or raw.get("tree") != pin["tree"]
        or raw.get("launcher_sha256")
        != pin["launcher_sha256"]
        or raw.get("rollback_launcher_sha256")
        != pin["rollback_launcher_sha256"]
    ):
        raise HandoffContractError(
            "Scheduler strict-node cutover identity drifted"
        )
    if pin["pin_generation"] == "scheduler-strict-node-e542-v2":
        if (
            raw.get("cohort_tasks_preserved") != 30
            or isinstance(raw.get("active_tasks_pre"), bool)
            or not isinstance(raw.get("active_tasks_pre"), int)
            or raw.get("active_tasks_pre") < 0
            or raw.get("active_tasks_pre") > 30
            or raw.get("active_tasks_post") != raw.get("active_tasks_pre")
            or raw.get("allowed_terminal_transitions")
            != "running->completed/0|failed/124"
            or raw.get("scheduler_ok") is not True
            or raw.get("scheduler_thread_alive") is not True
            or raw.get("pressure_episode_migration_smoke") != "pass"
            or raw.get("database_quick_check") != "ok"
        ):
            raise HandoffContractError(
                "Scheduler strict-node e542 cutover evidence drifted"
            )
    elif pin["pin_generation"] == "scheduler-strict-node-41b-v3":
        if (
            production._require_sha(
                raw.get("archive_sha256"),
                "Scheduler cutover archive SHA",
            )
            != raw.get("archive_sha256")
            or production._require_sha(
                raw.get("cutover_guard_sha256"),
                "Scheduler cutover guard SHA",
            )
            != raw.get("cutover_guard_sha256")
            or raw.get("database_migration") != "none"
            or raw.get("configuration_change") != "none"
            or raw.get("dynamic_campaign_selection")
            != "every existing task id in inclusive range 96208..96280"
            or raw.get("campaign_tasks_preserved") != 73
            or raw.get("active_tasks_pre") != 26
            or raw.get("active_tasks_final") != 26
            or raw.get("allowed_transitions")
            != (
                "running->completed/0|failed/124; "
                "attaching->running|terminal; "
                "queued->attaching|running"
            )
            or raw.get("protected_cancelled_tasks")
            != [96260, 96276, 96277]
            or raw.get("allocation_14616_immediate_requested_owned")
            != "64/64"
            or raw.get("allocation_14619_immediate_requested_owned")
            != "64/64"
            or raw.get("extra_attach_to_14616_or_14619") is not False
            or raw.get("strict_same_node_cpu_gate") != "pass"
            or raw.get("final_fea_storage_admission_gate") != "pass"
            or raw.get("pressure_episode_preservation") != "pass"
            or raw.get("n114_pressure_episode_preserved") is not True
            or raw.get("database_quick_check") != "ok"
            or raw.get("scheduler_ok") is not True
            or raw.get("scheduler_thread_alive") is not True
        ):
            raise HandoffContractError(
                "Scheduler strict-node 41b cutover evidence drifted"
            )
    else:
        raise HandoffContractError(
            "Scheduler strict-node cutover generation is unsupported"
        )
    _aware_timestamp(raw.get("cutover_at"), "Scheduler cutover timestamp")
    backup = _absolute_regular_file(
        raw.get("database_backup"), "Scheduler cutover database backup"
    )
    if (
        production._sha256_file(backup)
        != production._require_sha(
            raw.get("database_backup_sha256"),
            "Scheduler database backup SHA",
        )
    ):
        raise HandoffContractError(
            "Scheduler cutover database backup bytes drifted"
        )
    pre_snapshot = _absolute_regular_file(
        raw.get("pre_snapshot"), "Scheduler pre-cutover snapshot"
    )
    if pin["pin_generation"] == "scheduler-strict-node-e542-v2":
        _absolute_regular_file(
            raw.get("post_snapshot"), "Scheduler post-cutover snapshot"
        )
    else:
        if (
            production._sha256_file(pre_snapshot)
            != production._require_sha(
                raw.get("pre_snapshot_sha256"),
                "Scheduler pre-cutover snapshot SHA",
            )
        ):
            raise HandoffContractError(
                "Scheduler pre-cutover snapshot bytes drifted"
            )
        _absolute_regular_file(
            raw.get("immediate_snapshot"),
            "Scheduler immediate cutover snapshot",
        )
        _absolute_regular_file(
            raw.get("final_snapshot"), "Scheduler final cutover snapshot"
        )
    rollback_launcher = _absolute_regular_file(
        raw.get("rollback_launcher"), "Scheduler rollback launcher"
    )
    if (
        production._sha256_file(rollback_launcher)
        != pin["rollback_launcher_sha256"]
    ):
        raise HandoffContractError(
            "Scheduler rollback launcher bytes drifted"
        )
    launcher = _absolute_regular_file(
        SCHEDULER_STRICT_NODE_LIVE_LAUNCHER,
        "Scheduler strict-node live launcher",
    )
    normalized = {
        **copy.deepcopy(raw),
        "scheduler_url": DIAGNOSTIC_SCHEDULER_URL,
        "candidate_revision": pin["revision"],
        "candidate_tree": pin["tree"],
        "pin_generation": pin["pin_generation"],
        "live_launcher_path": str(launcher),
        "payload_sha256": canonical_sha256(raw),
        "cutover_receipt_file_sha256": production._sha256_file(resolved),
    }
    launcher_identity = (
        _live_launcher_identity(
            normalized,
            expected_sha256=pin["launcher_sha256"],
        )
        if verify_live_launcher
        else None
    )
    return normalized, launcher_identity


def _validate_scheduler_cutover_receipt(
    path: Path,
    *,
    verify_live_launcher: bool,
    require_strict_node: bool = False,
    strict_node_contract: Mapping[str, Any] | None = None,
    require_active_strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if require_strict_node:
        return _validate_strict_scheduler_cutover_receipt(
            path,
            verify_live_launcher=verify_live_launcher,
            strict_node_contract=strict_node_contract,
            require_active=require_active_strict,
        )
    resolved = path.resolve(strict=True)
    receipt = production._validate_seal(
        production._read_json(resolved), SCHEDULER_CUTOVER_SCHEMA
    )
    if set(receipt) != SCHEDULER_CUTOVER_FIELDS:
        raise HandoffContractError(
            "Scheduler prune-protection cutover receipt fields drifted"
        )
    release = receipt.get("release_manifest")
    if (
        not isinstance(release, dict)
        or set(release) != {"path", "sha256"}
    ):
        raise HandoffContractError(
            "Scheduler cutover release manifest record is malformed"
        )
    release_path = _absolute_regular_file(
        release.get("path"), "Scheduler release manifest"
    )
    release_sha = production._sha256_file(release_path)
    deployed = _absolute_directory(
        receipt.get("deployed_path"), "Scheduler deployed path"
    )
    launcher_path = Path(str(receipt.get("live_launcher_path") or ""))
    post_health = receipt.get("post_health")
    marker = receipt.get("marker")
    preflight = receipt.get("preflight")
    cutover_at = _aware_timestamp(
        receipt.get("cutover_at"), "Scheduler cutover timestamp"
    )
    created_at = _aware_timestamp(
        receipt.get("created_at"), "Scheduler cutover receipt timestamp"
    )
    health_at = (
        _aware_timestamp(
            post_health.get("captured_at"),
            "Scheduler post-cutover health timestamp",
        )
        if isinstance(post_health, dict)
        else None
    )
    if (
        receipt.get("scheduler_url") != DIAGNOSTIC_SCHEDULER_URL
        or receipt.get("candidate_revision")
        != SCHEDULER_MARKER_AWARE_REVISION
        or receipt.get("candidate_tree") != SCHEDULER_MARKER_AWARE_TREE
        or release.get("sha256") != SCHEDULER_RELEASE_MANIFEST_SHA256
        or release_sha != SCHEDULER_RELEASE_MANIFEST_SHA256
        or not launcher_path.is_absolute()
        or receipt.get("live_launcher_post_cutover_sha256")
        != SCHEDULER_LIVE_LAUNCHER_SHA256
        or not isinstance(post_health, dict)
        or set(post_health) != {"status", "captured_at"}
        or post_health.get("status") != "ok"
        or not isinstance(marker, dict)
        or set(marker)
        != {"filename", "schema", "semantics_verified"}
        or marker.get("filename")
        != scheduler_client.SCHEDULER_PRESERVE_MARKER
        or marker.get("schema")
        != scheduler_client.SCHEDULER_PRESERVE_SCHEMA
        or marker.get("semantics_verified") is not True
        or not isinstance(preflight, dict)
        or set(preflight)
        != {"goal_active_count", "all_nonterminal_count"}
        or preflight.get("goal_active_count") != 0
        or isinstance(preflight.get("goal_active_count"), bool)
        or preflight.get("all_nonterminal_count") != 0
        or isinstance(preflight.get("all_nonterminal_count"), bool)
        or production._require_sha(
            receipt.get("rollback_launcher_sha256"),
            "Scheduler rollback launcher SHA",
        )
        != receipt.get("rollback_launcher_sha256")
        or health_at is None
        or health_at < cutover_at
        or created_at < health_at
    ):
        raise HandoffContractError(
            "Scheduler marker-aware cutover identity drifted"
        )
    # Resolve these values to prove the receipt references exact, live local
    # deployment objects rather than relative paths interpreted by the caller.
    del deployed
    launcher_identity = (
        _live_launcher_identity(receipt) if verify_live_launcher else None
    )
    return receipt, launcher_identity


def _default_scheduler_live_reader(
    *, scheduler_url: str, endpoint: str
) -> dict[str, Any]:
    try:
        response = scheduler_client.requests.get(
            f"{scheduler_url.rstrip('/')}{endpoint}", timeout=30
        )
        response.raise_for_status()
        value = response.json()
    except Exception as exc:
        raise HandoffContractError(
            f"failed to read live Scheduler endpoint {endpoint}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HandoffContractError(
            f"live Scheduler endpoint {endpoint} returned no object"
        )
    return value


def _scheduler_admission_snapshot_from_values(
    *,
    scheduler_url: str,
    health: Any,
    license_snapshot: Any,
    captured_at: str,
) -> dict[str, Any]:
    if scheduler_url.rstrip("/") != DIAGNOSTIC_SCHEDULER_URL:
        raise HandoffContractError(
            "diagnostic Standard probes require Scheduler port 8002"
        )
    if not isinstance(health, dict):
        raise HandoffContractError(
            "live Scheduler health response is not an object"
        )
    if (
        health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
        or isinstance(health.get("consecutive_tick_failures"), bool)
        or not isinstance(health.get("consecutive_tick_failures"), int)
        or health.get("consecutive_tick_failures") != 0
    ):
        raise HandoffContractError(
            "live Scheduler health is not admission-safe"
        )
    if not isinstance(license_snapshot, dict):
        raise HandoffContractError(
            "live Scheduler license response is not an object"
        )
    admission = license_snapshot.get("admission")
    if not isinstance(admission, dict):
        raise HandoffContractError(
            "live Scheduler license admission diagnostics are absent"
        )
    features = admission.get("features")
    costs_by_project = admission.get("persistent_cost_by_project")
    project_costs = (
        costs_by_project.get(scheduler_client.MFT_PROJECT)
        if isinstance(costs_by_project, dict)
        else None
    )
    age = admission.get("snapshot_age_seconds")
    max_age = admission.get("snapshot_max_age_seconds")
    try:
        age_value = float(age)
        max_age_value = float(max_age)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HandoffContractError(
            "live Scheduler license snapshot freshness is invalid"
        ) from exc
    checked_at = _aware_timestamp(
        license_snapshot.get("checked_at"),
        "Scheduler license checked_at",
    )
    captured = _aware_timestamp(
        captured_at, "Scheduler admission captured_at"
    )
    wall_age = (captured - checked_at).total_seconds()
    if (
        license_snapshot.get("server_up") is not True
        or str(license_snapshot.get("server") or "").strip() == ""
        or str(license_snapshot.get("error") or "") != ""
        or admission.get("enabled") is not True
        or admission.get("snapshot_valid") is not True
        or str(admission.get("blocked_reason") or "") != ""
        or not math.isfinite(age_value)
        or not math.isfinite(max_age_value)
        or age_value < 0
        or max_age_value <= 0
        or age_value > max_age_value
        or wall_age < -5
        or wall_age > max_age_value + 5
        or not isinstance(features, dict)
        or not isinstance(project_costs, dict)
        or not project_costs
    ):
        raise HandoffContractError(
            "live Scheduler license snapshot is not admission-safe"
        )
    for feature, raw_cost in project_costs.items():
        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, int)
            or raw_cost <= 0
        ):
            raise HandoffContractError(
                "Scheduler MFT license cost contract is invalid"
            )
        feature_status = features.get(feature)
        headroom = (
            feature_status.get("admit_headroom")
            if isinstance(feature_status, dict)
            else None
        )
        if (
            isinstance(headroom, bool)
            or not isinstance(headroom, int)
            or headroom < raw_cost
        ):
            raise HandoffContractError(
                f"Scheduler license headroom is insufficient for {feature}"
            )
    return {
        "schema_version": (
            "mft-goal-diagnostic-scheduler-admission-snapshot-v1"
        ),
        "scheduler_url": scheduler_url.rstrip("/"),
        "captured_at": captured_at,
        "health": copy.deepcopy(health),
        "health_sha256": canonical_sha256(health),
        "license": copy.deepcopy(license_snapshot),
        "license_sha256": canonical_sha256(license_snapshot),
        "license_project": scheduler_client.MFT_PROJECT,
        "license_project_costs": copy.deepcopy(project_costs),
        "health_verified": True,
        "license_admission_verified": True,
        "license_headroom_verified": True,
    }


def _live_scheduler_admission_snapshot(
    *,
    scheduler_url: str,
    reader: Any = _default_scheduler_live_reader,
) -> dict[str, Any]:
    if scheduler_url.rstrip("/") != DIAGNOSTIC_SCHEDULER_URL:
        raise HandoffContractError(
            "diagnostic Standard probes require Scheduler port 8002"
        )
    health = reader(scheduler_url=scheduler_url, endpoint="/api/health")
    license_snapshot = reader(
        scheduler_url=scheduler_url, endpoint="/api/licenses"
    )
    captured_at = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )
    return _scheduler_admission_snapshot_from_values(
        scheduler_url=scheduler_url,
        health=health,
        license_snapshot=license_snapshot,
        captured_at=captured_at,
    )


def _validate_recorded_admission_snapshot(value: Any) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "scheduler_url",
        "captured_at",
        "health",
        "health_sha256",
        "license",
        "license_sha256",
        "license_project",
        "license_project_costs",
        "health_verified",
        "license_admission_verified",
        "license_headroom_verified",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise HandoffContractError(
            "recorded Scheduler admission snapshot fields drifted"
        )
    if (
        value.get("schema_version")
        != "mft-goal-diagnostic-scheduler-admission-snapshot-v1"
        or value.get("health_sha256")
        != canonical_sha256(value.get("health"))
        or value.get("license_sha256")
        != canonical_sha256(value.get("license"))
    ):
        raise HandoffContractError(
            "recorded Scheduler admission snapshot hashes drifted"
        )
    expected = _scheduler_admission_snapshot_from_values(
        scheduler_url=str(value.get("scheduler_url") or ""),
        health=value.get("health"),
        license_snapshot=value.get("license"),
        captured_at=str(value.get("captured_at") or ""),
    )
    if expected != value:
        raise HandoffContractError(
            "recorded Scheduler admission evidence drifted"
        )
    return copy.deepcopy(value)


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    return {
        "path": relative,
        "sha256": production._sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _artifact(
    root: Path, record: Any, label: str, *, expect_file: bool = True
) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise HandoffContractError(f"{label} record is malformed")
    target = production._contained_file(root, record["path"], label)
    if (
        (expect_file and not target.is_file())
        or target.stat().st_size != production._integer(
            record["size_bytes"], f"{label} size"
        )
        or production._sha256_file(target)
        != production._require_sha(record["sha256"], f"{label} SHA")
    ):
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def _profile_content(
    *, timeout_retry: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (
        TIMEOUT_RETRY_PROFILE_PATH if timeout_retry else PROFILE_PATH
    ).resolve(strict=True)
    profile = production._read_json(path)
    _validate_profile(profile, timeout_retry=timeout_retry)
    return profile, production._file_record(path)


def _validate_profile(
    profile: Mapping[str, Any], *, timeout_retry: bool = False
) -> None:
    production_profile, _source = production._profile_content("standard")
    expected_schema = (
        TIMEOUT_RETRY_PROFILE_SCHEMA
        if timeout_retry
        else "mft-goal-diagnostic-standard-profile-v1"
    )
    expected_resources = (
        TIMEOUT_RETRY_RESOURCES if timeout_retry else STANDARD_RESOURCES
    )
    expected_comment = (
        "Diagnostic-only eighth-symmetry Standard FEA timeout retry with "
        "retained AEDT project and AEDT results"
        if timeout_retry
        else "Diagnostic-only eighth-symmetry Standard FEA with retained "
        "AEDT project and AEDT results"
    )
    if (
        set(profile)
        != {
            "schema_version",
            "stage",
            "comment",
            "reviewed_solver_path",
            "cli_flags",
            "param_overrides",
            "fixed_boundary_contract",
            "artifact_retention",
            "mem_mb",
            "cpus",
            "timeout_seconds",
        }
        or profile.get("schema_version")
        != expected_schema
        or profile.get("stage") != "standard"
        or profile.get("comment") != expected_comment
        or profile.get("reviewed_solver_path")
        != production.PROFILE_REVIEWED_PATH["standard"]
        or profile.get("cli_flags")
        != production.PROFILE_CLI_FLAGS["standard"]
        or profile.get("param_overrides")
        != production_profile["param_overrides"]
        or profile.get("fixed_boundary_contract")
        != production_profile["fixed_boundary_contract"]
        or profile.get("mem_mb") != 32768
        or profile.get("cpus") != expected_resources["cpus"]
        or profile.get("timeout_seconds")
        != expected_resources["timeout_seconds"]
        or profile.get("artifact_retention")
        != {
            "schema_version": (
                scheduler_client.RETAINED_AEDT_BUNDLE_SCHEMA
            ),
            "stage": "standard",
            "artifact_filename": "symmetric.aedt",
            "results_directory": "symmetric.aedtresults",
            "results_manifest_filename": (
                "symmetric.aedtresults.manifest.json"
            ),
            "receipt_filename": "symmetric.aedt.receipt.json",
            "marker_filename": (
                scheduler_client.SCHEDULER_PRESERVE_MARKER
            ),
            "retention_required": True,
            "prune_protection_required": True,
        }
    ):
        raise HandoffContractError(
            "diagnostic Standard profile contract drifted"
        )
    effective_identity = dict(profile["param_overrides"])
    effective_identity["thermal_pad_conductivity_W_mK"] = 0.2
    attest_fixed_identity(effective_identity)


def _generation_identity(
    generation_path: Path,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generation = generation_path.resolve(strict=True)
    if not generation.is_dir() or generation.is_symlink():
        raise HandoffContractError("surrogate generation is not a real directory")
    report_path = generation / "train_report.json"
    model_path = generation / "Llt_phys" / "models.pkl"
    meta_path = generation / "Llt_phys" / "meta.json"
    report = production._read_json(report_path.resolve(strict=True))
    artifacts = report.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("Llt_phys/models.pkl")
        != production._sha256_file(model_path.resolve(strict=True))
        or artifacts.get("Llt_phys/meta.json")
        != production._sha256_file(meta_path.resolve(strict=True))
    ):
        raise HandoffContractError("pinned Llt model artifacts drifted")
    report_sha = production._sha256_file(report_path.resolve(strict=True))
    if not tasks:
        raise HandoffContractError("surrogate generation has no bound tasks")
    source_identities = [task.get("source_identity") for task in tasks]
    if any(not isinstance(identity, dict) for identity in source_identities):
        raise HandoffContractError("task surrogate identity is absent")
    if any(
        identity.get("train_report_sha256") != report_sha
        or identity.get("dataset_sha256") != report.get("dataset_sha256")
        for identity in source_identities
    ):
        raise HandoffContractError(
            "surrogate generation differs from selected task authority"
        )
    evaluation_model_sha = {
        str(identity.get("evaluation_model_sha256"))
        for identity in source_identities
    }
    if (
        len(evaluation_model_sha) != 1
        or not production.HEX64.fullmatch(next(iter(evaluation_model_sha)))
    ):
        raise HandoffContractError("task evaluation-model identity is mixed")
    meta = production._read_json(meta_path.resolve(strict=True))
    if (
        meta.get("target") != "Llt_phys"
        or meta.get("training_run_id") != report.get("training_run_id")
        or meta.get("dataset_sha256") != report.get("dataset_sha256")
    ):
        raise HandoffContractError("pinned Llt model metadata drifted")
    return {
        "path": str(generation),
        "training_run_id": report["training_run_id"],
        "dataset_sha256": report["dataset_sha256"],
        "evaluation_model_sha256": next(iter(evaluation_model_sha)),
        "train_report": production._file_record(report_path),
        "Llt_phys_model": production._file_record(model_path),
        "Llt_phys_meta": production._file_record(meta_path),
    }


def _validate_generation_identity(
    identity: Any, tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise HandoffContractError("selection surrogate generation is absent")
    path = Path(str(identity.get("path") or ""))
    refreshed = _generation_identity(path, tasks)
    if refreshed != identity:
        raise HandoffContractError("selection surrogate generation bytes drifted")
    return refreshed


def _load_llt_predictor(identity: Mapping[str, Any]) -> Any:
    training_path = REPOSITORY_ROOT / "regression_260707" / "training"
    if str(training_path) not in sys.path:
        sys.path.insert(0, str(training_path))
    from regression_260707.training.predictor import EnsemblePredictor

    report = production._read_json(
        Path(str(identity["train_report"]["path"]))
    )
    record = {"generation": identity["path"], "report": report}
    predictor = EnsemblePredictor._load_record("Llt_phys", record)
    configured = predictor.configure_inference_threads(1)
    if (
        not isinstance(configured, dict)
        or configured.get("threads") != 1
    ):
        raise HandoffContractError("Llt inference thread binding failed")
    return predictor


def _prediction_from_values(
    mean_value: Any,
    q90_value: Any,
    disagreement_value: Any,
) -> dict[str, float | bool]:
    mean = production._finite(mean_value, "Llt mean")
    q90 = production._finite(q90_value, "Llt q90 half-width")
    spread = production._finite(
        disagreement_value, "Llt disagreement"
    )
    target = float(GOAL_STAGE_SPEC["Llt_target_uH"])
    tolerance = float(GOAL_STAGE_SPEC["Llt_tol_uH"])
    return {
        "mean_uH": mean,
        "q90_half_width_uH": q90,
        "robust_low_uH": mean - q90,
        "robust_high_uH": mean + q90,
        "ensemble_disagreement_uH": spread,
        "recomputed_Llt_robust_G_uH": (
            abs(mean - target) + q90 - tolerance
        ),
        "recomputed_Llt_disagreement_G_uH": (
            spread - 2.0 * tolerance
        ),
        "mean_band_low_uH": target - tolerance,
        "mean_band_high_uH": target + tolerance,
        "mean_in_target_band": abs(mean - target) <= tolerance,
    }


def _predict_llt_batch(
    predictor: Any,
    decoded_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | bool]]:
    if not decoded_rows:
        return []
    frame = pd.DataFrame([dict(decoded) for decoded in decoded_rows])
    mu, half_width = predictor.predict_mu_sigma(frame)
    disagreement = predictor.disagreement(frame)
    means = np.asarray(mu).reshape(-1)
    q90_values = np.asarray(half_width).reshape(-1)
    spreads = np.asarray(disagreement).reshape(-1)
    if not (
        len(means)
        == len(q90_values)
        == len(spreads)
        == len(decoded_rows)
    ):
        raise HandoffContractError(
            "Llt predictor returned a row-count mismatch"
        )
    return [
        _prediction_from_values(mean, q90, spread)
        for mean, q90, spread in zip(means, q90_values, spreads)
    ]


def _predict_llt(
    predictor: Any, decoded: Mapping[str, Any]
) -> dict[str, float | bool]:
    return _predict_llt_batch(predictor, [decoded])[0]


def _diagnostic_prefilter(row: Mapping[str, Any]) -> None:
    if (
        not production._csv_bool(row.get("decoder_valid"), "decoder_valid")
        or not production._csv_bool(
            row.get("surrogate_physical_valid"),
            "surrogate_physical_valid",
        )
        or not production._csv_bool(
            row.get("surrogate_physicality_passed"),
            "surrogate_physicality_passed",
        )
        or production._csv_bool(
            row.get("physical_feasible"), "physical_feasible"
        )
        or production._csv_bool(
            row.get("physical_constraint_feasible"),
            "physical_constraint_feasible",
        )
    ):
        raise HandoffContractError(
            "diagnostic row is not a surrogate-valid near-feasible row"
        )
    physical_g = production._parse_json_cell(
        row.get("physical_G_json"), "physical_G_json"
    )
    names = tuple(preflight.GOAL_CONSTRAINT_NAMES)
    if set(physical_g) != set(names):
        raise HandoffContractError(
            "diagnostic constraint vector is incomplete"
        )
    values = {
        name: production._finite(physical_g[name], name)
        for name in names
    }
    if any(
        value > 0.0
        for name, value in values.items()
        if name not in LLT_UNCERTAINTY_CONSTRAINTS
    ):
        raise HandoffContractError(
            "diagnostic row violates a non-Llt goal constraint"
        )
    if not any(
        values[name] > 0.0 for name in LLT_UNCERTAINTY_CONSTRAINTS
    ):
        raise HandoffContractError(
            "diagnostic row has no Llt uncertainty blocker"
        )


def _diagnostic_row_contract(
    row: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    geometry_sha = production._require_sha(
        row.get("physical_geometry_sha256"), "physical_geometry_sha256"
    )
    if (
        production._require_sha(
            row.get("candidate_physics_sha"), "candidate_physics_sha"
        )
        != geometry_sha
        or not production._csv_bool(row.get("decoder_valid"), "decoder_valid")
        or not production._csv_bool(
            row.get("surrogate_physical_valid"),
            "surrogate_physical_valid",
        )
        or not production._csv_bool(
            row.get("surrogate_physicality_passed"),
            "surrogate_physicality_passed",
        )
        or production._csv_bool(
            row.get("physical_feasible"), "physical_feasible"
        )
        or production._csv_bool(
            row.get("physical_constraint_feasible"),
            "physical_constraint_feasible",
        )
    ):
        raise HandoffContractError(
            "diagnostic row is not a surrogate-valid near-feasible row"
        )
    decoded = production._parse_json_cell(
        row.get("decoded_physical_params_json"),
        "decoded_physical_params_json",
    )
    if canonical_sha256(decoded) != production._require_sha(
        row.get("canonical_physical_params_sha256"),
        "canonical_physical_params_sha256",
    ):
        raise HandoffContractError("diagnostic decoded parameter SHA drifted")
    geometry = {
        name: decoded.get(name)
        for name in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
    }
    if (
        any(value is None for value in geometry.values())
        or canonical_sha256(geometry) != geometry_sha
    ):
        raise HandoffContractError("diagnostic geometry identity drifted")
    names = tuple(preflight.GOAL_CONSTRAINT_NAMES)
    physical_g = production._parse_json_cell(
        row.get("physical_G_json"), "physical_G_json"
    )
    normalized_g = production._parse_json_cell(
        row.get("normalized_G_json"), "normalized_G_json"
    )
    if set(physical_g) != set(names) or set(normalized_g) != set(names):
        raise HandoffContractError("diagnostic constraint vector is incomplete")
    physical_values = {
        name: production._finite(physical_g[name], name) for name in names
    }
    normalized_values = {
        name: production._finite(
            normalized_g[name], f"normalized:{name}"
        )
        for name in names
    }
    for name in names:
        if not math.isclose(
            physical_values[name],
            production._finite(
                row.get(f"physical_G:{name}"), f"physical_G:{name}"
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            normalized_values[name],
            production._finite(
                row.get(f"normalized_G:{name}"), f"normalized_G:{name}"
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise HandoffContractError(
                f"diagnostic flattened constraint drifted: {name}"
            )
    non_llt_violations = {
        name: value
        for name, value in physical_values.items()
        if name not in LLT_UNCERTAINTY_CONSTRAINTS and value > 0.0
    }
    if non_llt_violations:
        raise HandoffContractError(
            "diagnostic row violates a non-Llt goal constraint: "
            f"{non_llt_violations}"
        )
    if not any(
        physical_values[name] > 0.0 for name in LLT_UNCERTAINTY_CONSTRAINTS
    ):
        raise HandoffContractError(
            "diagnostic row has no Llt uncertainty blocker"
        )
    if (
        prediction.get("mean_in_target_band") is not True
        or not math.isclose(
            production._finite(
                prediction.get("recomputed_Llt_robust_G_uH"),
                "recomputed Llt robust G",
            ),
            physical_values["Llt_robust_band"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        )
        or not math.isclose(
            production._finite(
                prediction.get("recomputed_Llt_disagreement_G_uH"),
                "recomputed Llt disagreement G",
            ),
            physical_values["Llt_ensemble_disagreement"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        )
    ):
        raise HandoffContractError(
            "diagnostic Llt replay differs from terminal evidence or mean band"
        )
    primary_turns = production._integer(
        decoded.get("N1_main"), "N1_main"
    ) + production._integer(decoded.get("N1_side"), "N1_side")
    if not GOAL_N1_MIN_TURNS <= primary_turns <= GOAL_N1_MAX_TURNS:
        raise HandoffContractError("diagnostic primary turns are outside 5..8")
    cw1_mm = validate_cw1_mm(decoded.get("cw1"))
    if dynamic_core_group_violation(decoded) > 0:
        raise HandoffContractError("diagnostic dynamic core-group contract failed")
    if (
        production._finite(
            decoded.get("core_plate_pad_t"), "core_plate_pad_t"
        )
        != 2.0
        or production._finite(decoded.get("wcp_pad_t"), "wcp_pad_t")
        != 2.0
    ):
        raise HandoffContractError("diagnostic cooling pad thickness drifted")
    identity = dict(decoded)
    observed_tim = identity.get("thermal_pad_conductivity_W_mK")
    if (
        observed_tim is not None
        and production._finite(observed_tim, "TIM conductivity") != 0.2
    ):
        raise HandoffContractError("diagnostic TIM conductivity drifted")
    identity["thermal_pad_conductivity_W_mK"] = 0.2
    fixed_identity = attest_fixed_identity(identity)
    volume_l, dimensions = geometry_metrics.bounding_box_lit(decoded)
    width, length, height = (
        production._finite(value, "exterior dimension")
        for value in dimensions
    )
    for axis, value in zip(("W", "L", "H"), (width, length, height)):
        if value > GOAL_SIZE_LIMITS_MM[axis]:
            raise HandoffContractError(
                f"diagnostic exterior {axis} exceeds goal"
            )
    missing = [key for key in ALL_INPUT_KEYS if decoded.get(key) is None]
    if missing:
        raise HandoffContractError(
            f"diagnostic FEA parameter schema is incomplete: {missing}"
        )
    params = {key: decoded[key] for key in ALL_INPUT_KEYS}
    return {
        "physical_geometry_sha256": geometry_sha,
        "canonical_physical_params_sha256": canonical_sha256(decoded),
        "decoded_params": decoded,
        "fea_params": params,
        "fea_params_sha256": canonical_sha256(params),
        "fixed_identity_attestation": fixed_identity,
        "primary_turns": primary_turns,
        "cw1_mm": cw1_mm,
        "n_core_group": production._integer(
            decoded["n_core_group"], "n_core_group"
        ),
        "exterior_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "volume_L": float(volume_l),
        "physical_G": physical_values,
        "normalized_G": normalized_values,
        "surrogate_Llt_replay": copy.deepcopy(dict(prediction)),
        "goal_non_Llt_hard_spec_passed": True,
        "surrogate_mean_Llt_in_band": True,
        "surrogate_robust_feasible": False,
        **_diagnostic_flags(),
    }


def _authenticated_result_frames(
    *,
    bundle_manifest_path: Path,
    result_paths: Sequence[Path],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, Any],
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    (
        bundle,
        tasks,
        task_paths,
        code_authentication,
    ) = production._authenticate_bundle(
        bundle_manifest_path, allow_search_only=True
    )
    resolved_results = sorted(
        path.resolve(strict=True) for path in result_paths
    )
    if (
        not resolved_results
        or len(set(resolved_results)) != len(resolved_results)
    ):
        raise HandoffContractError(
            "diagnostic source results are empty or duplicated"
        )
    result_records = []
    frames = []
    observed_tasks: set[str] = set()
    for result_path in resolved_results:
        envelope = launch._validate_seal(
            production._read_json(result_path),
            schema=launch.SEARCH_RESULT_SCHEMA,
        )
        task_sha = production._require_sha(
            envelope.get("task_payload_sha256"),
            "diagnostic result task payload SHA",
        )
        task = tasks.get(task_sha)
        if task is None or task_sha in observed_tasks:
            raise HandoffContractError(
                "diagnostic result is outside or duplicates the bundle ledger"
            )
        try:
            result, terminal = launch._validated_seed_table(
                result_path, expected_task=task
            )
        except RuntimeError as exc:
            raise HandoffContractError(
                "diagnostic terminal result authentication failed"
            ) from exc
        if (
            production._integer(
                result.get("evaluated_generations"),
                "diagnostic evaluated generations",
            )
            != launch.GENERATIONS
            or production._integer(
                result.get("completed_generations"),
                "diagnostic completed generations",
            )
            != launch.EXPECTED_ALGORITHM_N_GEN_COUNTER
            or result.get("fea_submission_performed") is not False
            or not isinstance(result.get("search_only_proposal"), bool)
            or result.get("search_only_proposal")
            != bundle.get("search_only_proposal")
            or result.get("production_eligible") is not False
            or result.get("automatic_promotion_allowed") is not False
            or result.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
            or result.get("temperature_contract_sha256")
            != GOAL_TEMPERATURE_CONTRACT_SHA256
        ):
            raise HandoffContractError(
                "diagnostic source result campaign contract drifted"
            )
        observed_tasks.add(task_sha)
        result_records.append(
            {
                "result": production._file_record(result_path),
                "task_payload": production._file_record(
                    task_paths[task_sha]
                ),
                "task_payload_sha256": task_sha,
                "seed": production._integer(result["seed"], "result seed"),
                "fixed_primary_turns": production._integer(
                    result["fixed_primary_turns"], "result primary turns"
                ),
            }
        )
        frames.append(terminal)
    return (
        bundle,
        tasks,
        task_paths,
        code_authentication,
        result_records,
        frames,
    )


def _ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    robust = production._finite(
        row["physical_G:Llt_robust_band"], "Llt robust G"
    )
    disagreement = production._finite(
        row["physical_G:Llt_ensemble_disagreement"],
        "Llt disagreement G",
    )
    return (
        max(robust, 0.0) + max(disagreement, 0.0),
        max(robust, disagreement),
        robust,
        disagreement,
        production._finite(row["objective_volume_L"], "objective volume"),
        production._finite(
            row["objective_total_loss_W"], "objective total loss"
        ),
        str(row["physical_geometry_sha256"]).lower(),
    )


def _diversity_vector(contract: Mapping[str, Any], row: Mapping[str, Any]) -> list[float]:
    dimensions = contract["exterior_dimensions_mm"]
    return [
        float(contract["primary_turns"]),
        float(contract["n_core_group"]),
        float(contract["cw1_mm"]),
        float(dimensions["W"]),
        float(dimensions["L"]),
        float(dimensions["H"]),
        production._finite(row["objective_volume_L"], "objective volume"),
        production._finite(
            row["objective_total_loss_W"], "objective total loss"
        ),
    ]


def _select_diverse(
    eligible: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    ranked = sorted(eligible, key=lambda item: _ranking_key(item["row"]))
    selected: list[dict[str, Any]] = []
    selected_sha: set[str] = set()
    by_turns: dict[int, dict[str, Any]] = {}
    for item in ranked:
        by_turns.setdefault(
            int(item["contract"]["primary_turns"]), item
        )
    for turns in sorted(by_turns):
        if len(selected) >= count:
            break
        item = by_turns[turns]
        selected.append(item)
        selected_sha.add(
            str(item["row"]["physical_geometry_sha256"]).lower()
        )
    if len(selected) >= count:
        return sorted(selected, key=lambda item: _ranking_key(item["row"]))

    vectors = np.asarray(
        [_diversity_vector(item["contract"], item["row"]) for item in ranked],
        dtype=float,
    )
    minimum = vectors.min(axis=0)
    span = vectors.max(axis=0) - minimum
    span[span == 0.0] = 1.0
    normalized = (vectors - minimum) / span
    index_by_sha = {
        str(item["row"]["physical_geometry_sha256"]).lower(): index
        for index, item in enumerate(ranked)
    }
    while len(selected) < count:
        candidates = [
            item
            for item in ranked
            if str(item["row"]["physical_geometry_sha256"]).lower()
            not in selected_sha
        ]
        if not candidates:
            break
        selected_indices = [
            index_by_sha[
                str(item["row"]["physical_geometry_sha256"]).lower()
            ]
            for item in selected
        ]

        def distance_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            sha = str(item["row"]["physical_geometry_sha256"]).lower()
            index = index_by_sha[sha]
            distance = min(
                float(np.linalg.norm(normalized[index] - normalized[other]))
                for other in selected_indices
            )
            ranking = _ranking_key(item["row"])
            return (-distance, *ranking)

        chosen = min(candidates, key=distance_key)
        selected.append(chosen)
        selected_sha.add(
            str(chosen["row"]["physical_geometry_sha256"]).lower()
        )
    return sorted(selected, key=lambda item: _ranking_key(item["row"]))


def create_selection(
    *,
    bundle_manifest_path: Path,
    generation_path: Path,
    result_paths: Sequence[Path] | None,
    aggregate_manifest_path: Path | None = None,
    count: int,
    output: Path,
    predictor: Any | None = None,
) -> Path:
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= MAX_SELECTION_COUNT
    ):
        raise HandoffContractError(
            f"diagnostic selection count must be 1..{MAX_SELECTION_COUNT}"
        )
    supplied_results = list(result_paths or [])
    if (aggregate_manifest_path is None) == (not supplied_results):
        raise HandoffContractError(
            "choose exactly one final aggregate or repeated source results"
        )
    aggregate_source = None
    if aggregate_manifest_path is not None:
        aggregate_path = aggregate_manifest_path.resolve(strict=True)
        aggregate = launch._validate_seal(
            production._read_json(aggregate_path),
            schema=launch.GLOBAL_PARETO_SCHEMA,
        )
        if (
            aggregate.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
            or aggregate.get("hard_spec") != GOAL_STAGE_SPEC
            or aggregate.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
            or aggregate.get("stage_spec_sha256")
            != GOAL_STAGE_SPEC_SHA256
            or aggregate.get("temperature_contract_sha256")
            != GOAL_TEMPERATURE_CONTRACT_SHA256
            or aggregate.get("seed_local_pareto_merge_used") is not False
            or "all_authenticated_terminal_rows"
            not in str(aggregate.get("sorting_authority") or "")
        ):
            raise HandoffContractError(
                "diagnostic final aggregate goal contract drifted"
            )
        aggregate_authority = production._authenticate_aggregate_authority(
            aggregate=aggregate,
            manifest_path=aggregate_path,
            bundle_manifest_path=bundle_manifest_path,
            allow_search_only=True,
        )
        supplied_results = [
            Path(str(record["path"]))
            for record in aggregate.get("inputs") or []
        ]
        aggregate_source = {
            "manifest": production._file_record(aggregate_path),
            "payload_sha256": aggregate["payload_sha256"],
            "seed_count": aggregate_authority["seed_count"],
            "minimum_seed_count": aggregate_authority[
                "minimum_seed_count"
            ],
            "all_bundle_seed_results_reauthenticated": True,
            "global_nds_recomputed": True,
            "input_result_sha256": sorted(
                aggregate_authority["result_paths_by_sha256"]
            ),
        }
    (
        bundle,
        tasks,
        _task_paths,
        code_authentication,
        result_records,
        frames,
    ) = _authenticated_result_frames(
        bundle_manifest_path=bundle_manifest_path,
        result_paths=supplied_results,
    )
    selected_tasks = [
        tasks[record["task_payload_sha256"]] for record in result_records
    ]
    generation = _generation_identity(generation_path, selected_tasks)
    model = predictor or _load_llt_predictor(generation)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        [
            "source_seed",
            "terminal_population_index",
            "physical_geometry_sha256",
        ],
        kind="stable",
    )
    deduplicated = merged.drop_duplicates(
        "physical_geometry_sha256", keep="first"
    )
    eligible: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    prefiltered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in deduplicated.to_dict(orient="records"):
        try:
            _diagnostic_prefilter(raw)
            decoded = production._parse_json_cell(
                raw.get("decoded_physical_params_json"),
                "diagnostic decoded parameters",
            )
        except HandoffContractError as exc:
            reason = str(exc).split(":", 1)[0]
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        prefiltered.append((raw, decoded))
    predictions = _predict_llt_batch(
        model, [decoded for _raw, decoded in prefiltered]
    )
    for (raw, _decoded), prediction in zip(prefiltered, predictions):
        try:
            contract = _diagnostic_row_contract(raw, prediction)
        except HandoffContractError as exc:
            reason = str(exc).split(":", 1)[0]
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        eligible.append(
            {"row": raw, "prediction": prediction, "contract": contract}
        )
    if len(eligible) < count:
        raise HandoffContractError(
            "authenticated diagnostic near-feasible population is too small: "
            f"eligible={len(eligible)}, requested={count}"
        )
    selected = _select_diverse(eligible, count)
    table_rows = []
    for diagnostic_rank, item in enumerate(selected):
        row = copy.deepcopy(item["row"])
        row.update(
            {
                "diagnostic_selection_rank": diagnostic_rank,
                "diagnostic_selection_key_json": json.dumps(
                    list(_ranking_key(row)),
                    separators=(",", ":"),
                ),
                "diagnostic_primary_turns": item["contract"][
                    "primary_turns"
                ],
                "diagnostic_llt_mean_uH": item["prediction"]["mean_uH"],
                "diagnostic_llt_q90_half_width_uH": item["prediction"][
                    "q90_half_width_uH"
                ],
                "diagnostic_llt_robust_low_uH": item["prediction"][
                    "robust_low_uH"
                ],
                "diagnostic_llt_robust_high_uH": item["prediction"][
                    "robust_high_uH"
                ],
                "diagnostic_llt_disagreement_uH": item["prediction"][
                    "ensemble_disagreement_uH"
                ],
                "diagnostic_llt_mean_in_target_band": True,
                **_diagnostic_flags(),
            }
        )
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"diagnostic selection output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        table_path = staging / "diagnostic_candidates.csv"
        table.to_csv(table_path, index=False)
        manifest = production._seal(
            {
                "schema_version": SELECTION_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "bundle_manifest": production._file_record(
                    bundle_manifest_path.resolve(strict=True)
                ),
                "bundle_payload_sha256": bundle["payload_sha256"],
                "bundle_task_count": len(tasks),
                "bundle_code_authentication": code_authentication,
                "surrogate_generation": generation,
                "source_results": result_records,
                "source_mode": (
                    "final_authenticated_aggregate"
                    if aggregate_source is not None
                    else "authenticated_terminal_results"
                ),
                "aggregate_source": aggregate_source,
                "authenticated_partial_seed_count": len(result_records),
                "authenticated_terminal_row_count": len(merged),
                "deduplicated_geometry_count": len(deduplicated),
                "eligible_near_feasible_count": len(eligible),
                "selected_count": len(table),
                "selected_geometry_sha256": list(
                    table["physical_geometry_sha256"].astype(str).str.lower()
                ),
                "candidate_table": _relative_record(table_path, staging),
                "eligibility_contract": {
                    "decoder_valid": True,
                    "surrogate_physical_valid": True,
                    "all_non_Llt_goal_constraints_G_le_zero": True,
                    "allowed_positive_constraints": sorted(
                        LLT_UNCERTAINTY_CONSTRAINTS
                    ),
                    "at_least_one_Llt_uncertainty_G_gt_zero": True,
                    "replayed_surrogate_mean_Llt_in_target_band": True,
                    "surrogate_robust_feasible": False,
                },
                "selection_algorithm": SELECTION_ALGORITHM,
                "rejection_counts": rejection_counts,
                "scheduler_submission_performed": False,
                "scheduler_project_modified": False,
                **_diagnostic_flags(),
            }
        )
        manifest_path = production._write_immutable_json(
            staging / "selection_manifest.json", manifest
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / manifest_path.name


def _load_selection(
    path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved = path.resolve(strict=True)
    selection = production._validate_seal(
        production._read_json(resolved), SELECTION_SCHEMA
    )
    flags = _diagnostic_flags()
    if (
        selection.get("campaign_id") != "mft-goal-20260726"
        or selection.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or selection.get("hard_spec") != GOAL_STAGE_SPEC
        or selection.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or selection.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or any(selection.get(name) is not value for name, value in flags.items())
        or selection.get("scheduler_submission_performed") is not False
        or selection.get("scheduler_project_modified") is not False
        or selection.get("selection_algorithm") != SELECTION_ALGORITHM
        or selection.get("source_mode")
        not in {
            "final_authenticated_aggregate",
            "authenticated_terminal_results",
        }
        or (
            selection.get("source_mode")
            == "final_authenticated_aggregate"
            and not isinstance(selection.get("aggregate_source"), dict)
        )
        or (
            selection.get("source_mode")
            == "authenticated_terminal_results"
            and selection.get("aggregate_source") is not None
        )
    ):
        raise HandoffContractError("diagnostic selection contract drifted")
    table_path = _artifact(
        resolved.parent,
        selection.get("candidate_table"),
        "diagnostic candidate table",
    )
    table = pd.read_csv(table_path)
    selected_sha = list(
        table["physical_geometry_sha256"].astype(str).str.lower()
    )
    if (
        len(table)
        != production._integer(
            selection.get("selected_count"), "diagnostic selected_count"
        )
        or not 1 <= len(table) <= MAX_SELECTION_COUNT
        or len(set(selected_sha)) != len(table)
        or selected_sha != selection.get("selected_geometry_sha256")
        or table["diagnostic_selection_rank"].tolist()
        != list(range(len(table)))
        or not table["diagnostic_llt_mean_in_target_band"].map(
            lambda value: production._csv_bool(
                value, "diagnostic mean-band flag"
            )
        ).all()
    ):
        raise HandoffContractError("diagnostic candidate table drifted")
    for name, expected in flags.items():
        if name not in table or not all(
            production._csv_bool(value, name) is expected
            for value in table[name]
        ):
            raise HandoffContractError(
                f"diagnostic candidate flag drifted: {name}"
            )
    return selection, table


def _result_record_for_sha(
    selection: Mapping[str, Any], digest: str
) -> Mapping[str, Any]:
    matches = [
        record
        for record in selection.get("source_results") or []
        if isinstance(record, dict)
        and (record.get("result") or {}).get("sha256") == digest
    ]
    if len(matches) != 1:
        raise HandoffContractError(
            "diagnostic candidate source result is missing or ambiguous"
        )
    return matches[0]


def authenticate_candidate(
    *,
    selection_manifest_path: Path,
    candidate_physics_sha256: str,
    predictor: Any | None = None,
) -> dict[str, Any]:
    candidate_sha = production._require_sha(
        candidate_physics_sha256, "diagnostic candidate geometry SHA"
    )
    selection, table = _load_selection(selection_manifest_path)
    matches = table.loc[
        table["physical_geometry_sha256"].astype(str).str.lower().eq(
            candidate_sha
        )
    ]
    if len(matches) != 1:
        raise HandoffContractError(
            "diagnostic candidate geometry is missing or ambiguous"
        )
    selected = matches.iloc[0].to_dict()
    bundle_record = selection.get("bundle_manifest")
    if not isinstance(bundle_record, dict):
        raise HandoffContractError("diagnostic bundle record is absent")
    bundle_path = Path(str(bundle_record.get("path") or ""))
    if production._file_record(bundle_path) != bundle_record:
        raise HandoffContractError("diagnostic bundle manifest bytes drifted")
    (
        bundle,
        tasks,
        task_paths,
        code_authentication,
    ) = production._authenticate_bundle(
        bundle_path, allow_search_only=True
    )
    if (
        bundle.get("payload_sha256")
        != selection.get("bundle_payload_sha256")
        or len(tasks)
        != production._integer(
            selection.get("bundle_task_count"),
            "diagnostic bundle task_count",
        )
        or code_authentication
        != selection.get("bundle_code_authentication")
    ):
        raise HandoffContractError("diagnostic bundle authority drifted")
    aggregate_source = selection.get("aggregate_source")
    if selection.get("source_mode") == "final_authenticated_aggregate":
        if not isinstance(aggregate_source, dict):
            raise HandoffContractError(
                "diagnostic aggregate source authority is absent"
            )
        aggregate_record = aggregate_source.get("manifest")
        if not isinstance(aggregate_record, dict):
            raise HandoffContractError(
                "diagnostic aggregate manifest record is absent"
            )
        aggregate_path = Path(str(aggregate_record.get("path") or ""))
        if production._file_record(aggregate_path) != aggregate_record:
            raise HandoffContractError(
                "diagnostic aggregate manifest bytes drifted"
            )
        aggregate = launch._validate_seal(
            production._read_json(aggregate_path),
            schema=launch.GLOBAL_PARETO_SCHEMA,
        )
        aggregate_authority = production._authenticate_aggregate_authority(
            aggregate=aggregate,
            manifest_path=aggregate_path,
            bundle_manifest_path=bundle_path,
            allow_search_only=True,
        )
        refreshed_aggregate_source = {
            "manifest": production._file_record(aggregate_path),
            "payload_sha256": aggregate["payload_sha256"],
            "seed_count": aggregate_authority["seed_count"],
            "minimum_seed_count": aggregate_authority[
                "minimum_seed_count"
            ],
            "all_bundle_seed_results_reauthenticated": True,
            "global_nds_recomputed": True,
            "input_result_sha256": sorted(
                aggregate_authority["result_paths_by_sha256"]
            ),
        }
        if refreshed_aggregate_source != aggregate_source:
            raise HandoffContractError(
                "diagnostic final aggregate authority drifted"
            )
    task_sha = production._require_sha(
        selected.get("source_bundle_id"),
        "diagnostic selected task payload SHA",
    )
    task = tasks.get(task_sha)
    task_path = task_paths.get(task_sha)
    if task is None or task_path is None:
        raise HandoffContractError(
            "diagnostic selected task is outside the bundle"
        )
    source_result_sha = production._require_sha(
        selected.get("source_result_sha256"),
        "diagnostic source result SHA",
    )
    source_record = _result_record_for_sha(selection, source_result_sha)
    result_file = source_record.get("result")
    task_file = source_record.get("task_payload")
    if not isinstance(result_file, dict) or not isinstance(task_file, dict):
        raise HandoffContractError(
            "diagnostic source result/task record is malformed"
        )
    source_path = Path(str(result_file.get("path") or ""))
    if (
        production._file_record(source_path) != result_file
        or production._file_record(task_path) != task_file
        or task_file != production._file_record(task_path)
        or source_record.get("task_payload_sha256") != task_sha
    ):
        raise HandoffContractError(
            "diagnostic source result/task bytes drifted"
        )
    try:
        source_result, terminal = launch._validated_seed_table(
            source_path, expected_task=task
        )
    except RuntimeError as exc:
        raise HandoffContractError(
            "diagnostic source result failed fresh task authentication"
        ) from exc
    terminal_index = production._integer(
        selected.get("terminal_population_index"),
        "diagnostic terminal index",
    )
    if not 0 <= terminal_index < launch.POPULATION:
        raise HandoffContractError(
            "diagnostic terminal index is outside 0..319"
        )
    source_row = terminal.iloc[terminal_index].to_dict()
    production._rows_equivalent(selected, source_row)
    if (
        source_result.get("task_payload_sha256") != task_sha
        or production._integer(source_result.get("seed"), "result seed")
        != production._integer(task.get("seed"), "task seed")
        or production._integer(selected.get("source_seed"), "source seed")
        != production._integer(task.get("seed"), "task seed")
        or str(selected.get("source_island_id"))
        != f"n1-{production._integer(task['fixed_primary_turns'], 'task turns')}"
        or source_result.get("stage_spec_sha256") != GOAL_STAGE_SPEC_SHA256
    ):
        raise HandoffContractError(
            "diagnostic selected row/result/task identity drifted"
        )
    generation = _validate_generation_identity(
        selection.get("surrogate_generation"), [task]
    )
    model = predictor or _load_llt_predictor(generation)
    decoded = production._parse_json_cell(
        selected.get("decoded_physical_params_json"),
        "diagnostic selected decoded params",
    )
    prediction = _predict_llt(model, decoded)
    contract = _diagnostic_row_contract(selected, prediction)
    if contract["primary_turns"] != production._integer(
        task["fixed_primary_turns"], "task primary turns"
    ):
        raise HandoffContractError(
            "diagnostic row escaped its task N1 stratum"
        )
    echoed_prediction = {
        "mean_uH": selected.get("diagnostic_llt_mean_uH"),
        "q90_half_width_uH": selected.get(
            "diagnostic_llt_q90_half_width_uH"
        ),
        "robust_low_uH": selected.get(
            "diagnostic_llt_robust_low_uH"
        ),
        "robust_high_uH": selected.get(
            "diagnostic_llt_robust_high_uH"
        ),
        "ensemble_disagreement_uH": selected.get(
            "diagnostic_llt_disagreement_uH"
        ),
    }
    for name, observed in echoed_prediction.items():
        if not math.isclose(
            production._finite(observed, f"selected {name}"),
            production._finite(prediction[name], f"replayed {name}"),
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise HandoffContractError(
                f"diagnostic selected Llt replay drifted: {name}"
            )
    return {
        "selection_source": {
            "kind": "diagnostic_partial_terminal_Llt_truth_probe",
            "selection_manifest": production._file_record(
                selection_manifest_path.resolve(strict=True)
            ),
            "bundle_manifest": production._file_record(bundle_path),
            "surrogate_generation": generation,
            "partial_seed_count": production._integer(
                selection["authenticated_partial_seed_count"],
                "diagnostic partial seed count",
            ),
            "source_mode": selection["source_mode"],
            "aggregate_source": copy.deepcopy(aggregate_source),
            **_diagnostic_flags(),
        },
        "selected_row": selected,
        "row_contract": contract,
        "source_result": production._file_record(source_path),
        "source_result_identity": {
            "schema_version": source_result["schema_version"],
            "payload_sha256": source_result["payload_sha256"],
            "task_payload_sha256": source_result["task_payload_sha256"],
            "seed": production._integer(
                source_result["seed"], "source result seed"
            ),
            "fixed_primary_turns": production._integer(
                source_result["fixed_primary_turns"],
                "source result primary turns",
            ),
            "dataset_sha256": source_result["dataset_sha256"],
            "evaluation_model_sha256": source_result[
                "evaluation_model_sha256"
            ],
            "stage_spec_sha256": source_result["stage_spec_sha256"],
            "temperature_contract_sha256": source_result[
                "temperature_contract_sha256"
            ],
            "hard_constraint_contract_sha256": source_result[
                "hard_constraint_contract_sha256"
            ],
        },
        "task_payload": production._file_record(task_path),
        "task_identity": {
            "payload_sha256": task["payload_sha256"],
            "task_name": task["task_name"],
            "seed": production._integer(task["seed"], "task seed"),
            "fixed_primary_turns": production._integer(
                task["fixed_primary_turns"], "task primary turns"
            ),
            "source_code_revision": task["source_identity"][
                "code_revision"
            ],
            "source_code_manifest_payload_sha256": task[
                "source_identity"
            ]["code_manifest_payload_sha256"],
            "source_code_inventory_sha256": task["source_identity"][
                "code_inventory_sha256"
            ],
        },
    }


AUTHENTICATION_FIELDS = (
    "selection_source",
    "selected_row",
    "row_contract",
    "source_result",
    "source_result_identity",
    "task_payload",
    "task_identity",
)


def _authentication_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if any(name not in value for name in AUTHENTICATION_FIELDS):
        raise HandoffContractError(
            "diagnostic candidate authentication is incomplete"
        )
    return {
        name: copy.deepcopy(value[name]) for name in AUTHENTICATION_FIELDS
    }


def _retention_run_root_evidence(
    retained: Mapping[str, Any],
) -> dict[str, Any]:
    path_fields = (
        "artifact_path",
        "results_path",
        "receipt_path",
        "results_manifest_path",
        "marker_path",
    )
    paths = {
        name: PurePosixPath(str(retained.get(name) or ""))
        for name in path_fields
    }
    parents = {path.parent.as_posix() for path in paths.values()}
    marker = retained.get("marker_contract")
    if (
        len(parents) != 1
        or "." in parents
        or any(
            path.is_absolute() or ".." in path.parts
            for path in paths.values()
        )
        or not isinstance(marker, dict)
        or marker.get("schema")
        != scheduler_client.SCHEDULER_PRESERVE_SCHEMA
        or marker.get("preserve") is not True
        or str(marker.get("reason") or "").strip() == ""
        or marker.get("owner") != scheduler_client.MFT_PROJECT
        or canonical_sha256(marker)
        != retained.get("marker_contract_sha256")
    ):
        raise HandoffContractError(
            "diagnostic retained run-root preservation marker drifted"
        )
    run_root = next(iter(parents))
    if (
        paths["marker_path"].name
        != scheduler_client.SCHEDULER_PRESERVE_MARKER
    ):
        raise HandoffContractError(
            "diagnostic preservation marker filename drifted"
        )
    return {
        "run_root": run_root,
        "marker_path": paths["marker_path"].as_posix(),
        "marker_contract": copy.deepcopy(marker),
        "marker_contract_sha256": retained["marker_contract_sha256"],
        "marker_created_by_task_before_terminal_cleanup": True,
        "scheduler_marker_semantics_required": True,
    }


def create_plan(
    *,
    selection_manifest_path: Path,
    candidate_physics_sha256: str,
    solver_revision: str,
    library_revision: str,
    output: Path,
    predictor: Any | None = None,
) -> Path:
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    solver = production._require_revision(
        solver_revision, "solver_revision"
    )
    library = production._require_revision(
        library_revision, "library_revision"
    )
    authenticated = authenticate_candidate(
        selection_manifest_path=selection_manifest_path,
        candidate_physics_sha256=candidate_physics_sha256,
        predictor=predictor,
    )
    params = {
        key: authenticated["row_contract"]["fea_params"][key]
        for key in sorted(ALL_INPUT_KEYS)
    }
    profile, profile_source = _profile_content()
    effective = production._effective_params(params, profile)
    stem = authenticated["row_contract"]["physical_geometry_sha256"][:12]
    task_name = f"mft-goal-diag-standard-{stem}"
    workdir = f"mft_goal_diag_standard_{stem}"
    retained = scheduler_client.retained_aedt_identity(
        task_name, params, profile, solver, library
    )
    if (
        retained is None
        or retained.get("schema_version")
        != scheduler_client.RETAINED_AEDT_BUNDLE_SCHEMA
        or not retained.get("artifact_path", "").endswith(
            "/symmetric.aedt"
        )
        or not retained.get("results_path", "").endswith(
            "/symmetric.aedtresults"
        )
        or not retained.get("marker_path", "").endswith(
            f"/{scheduler_client.SCHEDULER_PRESERVE_MARKER}"
        )
    ):
        raise HandoffContractError(
            "diagnostic Standard retained artifact identity is incomplete"
        )
    retention_run_root = _retention_run_root_evidence(retained)
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"diagnostic plan output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_payload = production._seal(
            {
                "schema_version": SELECTED_SCHEMA,
                **_authentication_payload(authenticated),
                **_diagnostic_flags(),
            }
        )
        selected_path = production._write_immutable_json(
            staging / "selected_candidate.json", selected_payload
        )
        params_path = production._write_immutable_json(
            staging / "fea_params.json", params
        )
        profile_path = production._write_immutable_json(
            staging / "diagnostic_standard_profile.json", profile
        )
        plan = production._seal(
            {
                "schema_version": PLAN_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "solver_revision": solver,
                "library_revision": library,
                "candidate_physics_sha256": authenticated[
                    "row_contract"
                ]["physical_geometry_sha256"],
                "search_authority_sha256": canonical_sha256(
                    _authentication_payload(authenticated)
                ),
                "fea_params_sha256": canonical_sha256(params),
                "selected_candidate": {
                    "path": selected_path.name,
                    "sha256": production._sha256_file(selected_path),
                },
                "fea_params": {
                    "path": params_path.name,
                    "sha256": production._sha256_file(params_path),
                },
                "profile": {
                    "path": profile_path.name,
                    "sha256": production._sha256_file(profile_path),
                    "canonical_sha256": canonical_sha256(profile),
                    "source": profile_source,
                },
                "stage": {
                    "name": "standard",
                    "task_name": task_name,
                    "workdir": workdir,
                    "profile_sha256": canonical_sha256(profile),
                    "effective_params_sha256": canonical_sha256(effective),
                    "resources": copy.deepcopy(STANDARD_RESOURCES),
                    "retained_aedt_bundle": retained,
                    "retention_run_root": retention_run_root,
                    "scheduler_project": scheduler_client.MFT_PROJECT,
                    "scheduler_url": DIAGNOSTIC_SCHEDULER_URL,
                    "aedt_backend": "standalone",
                    "full_model": 0,
                    "thermal_symmetry": "eighth",
                },
                "available_submission_commands": ["submit-standard"],
                "physics_override_allowed": False,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                "retention_required": True,
                "prune_protection_required": True,
                **_diagnostic_flags(),
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_plan.json", plan
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _plan_is_timeout_retry(plan: Mapping[str, Any]) -> bool:
    return "retry_of_timeout" in plan


def _strict_node_name(value: Any) -> str:
    if not isinstance(value, str):
        raise HandoffContractError(
            "strict timeout retry node name is unsafe or empty"
        )
    node_name = str(value or "").strip()
    if (
        not node_name
        or len(node_name) > 64
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node_name) is None
    ):
        raise HandoffContractError(
            "strict timeout retry node name is unsafe or empty"
        )
    return node_name


def _strict_node_plan_contract(
    node_name: str,
    *,
    scheduler_pin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pin = (
        _active_strict_node_scheduler_pin()
        if scheduler_pin is None
        else scheduler_pin
    )
    return {
        "schema_version": STRICT_NODE_PLACEMENT_SCHEMA,
        "requested_node_name": _strict_node_name(node_name),
        "node_name_policy": "strict",
        "scheduler_revision": pin["revision"],
        "scheduler_tree": pin["tree"],
        "scheduler_launcher_sha256": pin["launcher_sha256"],
        "scheduler_cutover_receipt_schema": pin[
            "cutover_receipt_schema"
        ],
        "scheduler_cutover_receipt_sha256": pin[
            "cutover_receipt_sha256"
        ],
        "task_identity_generation": "timeout-strict-r2-node-bound",
        "fallback_allocation_allowed": False,
        "api_submission_readback_required": True,
        "durable_get_readback_required": True,
        "terminal_readback_required": True,
    }


def _strict_node_scheduler_pin(
    value: Mapping[str, Any],
    *,
    require_active: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HandoffContractError(
            "strict timeout retry placement contract is absent"
        )
    for pin in _accepted_strict_node_scheduler_pins():
        expected = _strict_node_plan_contract(
            value.get("requested_node_name"), scheduler_pin=pin
        )
        if value == expected:
            active = _active_strict_node_scheduler_pin()
            if require_active and pin["revision"] != active["revision"]:
                raise HandoffContractError(
                    "historical strict-node generation cannot submit"
                )
            return pin
    raise HandoffContractError(
        "strict timeout retry placement contract drifted"
    )


def _validate_strict_node_plan_contract(value: Any) -> dict[str, Any]:
    pin = _strict_node_scheduler_pin(value)
    return _strict_node_plan_contract(
        value.get("requested_node_name"), scheduler_pin=pin
    )


def _plan_strict_node_contract(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = plan.get("scheduler_strict_node_contract")
    if value is None:
        return None
    if not _plan_is_timeout_retry(plan):
        raise HandoffContractError(
            "strict node placement is restricted to timeout retries"
        )
    return _validate_strict_node_plan_contract(value)


def _plan_resources(plan: Mapping[str, Any]) -> dict[str, int]:
    return (
        TIMEOUT_RETRY_RESOURCES
        if _plan_is_timeout_retry(plan)
        else STANDARD_RESOURCES
    )


def _recorded_external_file(record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise HandoffContractError(f"{label} record is malformed")
    path = Path(str(record.get("path") or "")).resolve(strict=True)
    if production._file_record(path) != record:
        raise HandoffContractError(f"{label} bytes drifted")
    return path


def _timeout_failure_evidence(
    snapshot: Mapping[str, Any],
    *,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema_version": TIMEOUT_RETRY_EVIDENCE_SCHEMA,
        "task_id": snapshot.get("task_id", snapshot.get("id")),
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "exit_code": snapshot.get("exit_code"),
        "failure_message": snapshot.get("failure_message"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": snapshot.get("actual_node_name"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "remote_cwd": snapshot.get("remote_cwd"),
        "remote_dir": snapshot.get("remote_dir"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    allocation_id = evidence["allocation_id"]
    expected_failure = (
        f"task timed out after {STANDARD_RESOURCES['timeout_seconds']}s"
    )
    if (
        evidence["task_id"] != submission["task_id"]
        or evidence["name"] != submission["task_name"]
        or evidence["status"] != "failed"
        or evidence["state"] != "failed"
        or evidence["exit_code"] != 124
        or evidence["failure_message"] != expected_failure
        or evidence["timeout_seconds"]
        != STANDARD_RESOURCES["timeout_seconds"]
        or not str(evidence["slurm_job_id"]).isdigit()
        or isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
        or not str(evidence["account_name"] or "").strip()
        or not str(evidence["actual_node_name"] or "").strip()
        or evidence["cpus"] != 8
        or evidence["memory_mb"] != 32768
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["dedupe_key"] != submission["dedupe_key"]
        or not str(evidence["remote_cwd"] or "").strip()
        or not str(evidence["remote_dir"] or "").strip()
        or not str(evidence["started_at"] or "").strip()
        or not str(evidence["finished_at"] or "").strip()
    ):
        raise HandoffContractError(
            "diagnostic Standard timeout retry requires the exact terminal "
            "failed/124 Scheduler task"
        )
    return evidence


def _validate_timeout_retry_record(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    record = plan.get("retry_of_timeout")
    expected_fields = {
        "schema_version",
        "retry_of_task_id",
        "original_plan",
        "original_plan_payload_sha256",
        "original_submission",
        "original_submission_payload_sha256",
        "original_task_execution",
        "original_task_execution_sha256",
        "scheduler_url",
        "timeout_reason",
    }
    if (
        not isinstance(record, dict)
        or set(record) != expected_fields
        or record.get("schema_version")
        != TIMEOUT_RETRY_EVIDENCE_SCHEMA
        or record.get("scheduler_url") != DIAGNOSTIC_SCHEDULER_URL
        or record.get("timeout_reason")
        != f"task timed out after {STANDARD_RESOURCES['timeout_seconds']}s"
    ):
        raise HandoffContractError(
            "diagnostic Standard timeout retry record drifted"
        )
    original_plan_path = _recorded_external_file(
        record["original_plan"], "original diagnostic plan"
    )
    original_plan, original_params, original_selected = _load_plan(
        original_plan_path
    )
    if _plan_is_timeout_retry(original_plan):
        raise HandoffContractError(
            "a timeout retry cannot be based on another timeout retry"
        )
    original_submission_path = _recorded_external_file(
        record["original_submission"], "original diagnostic submission"
    )
    original_submission = _load_submission(
        original_submission_path, plan=original_plan
    )
    execution = record.get("original_task_execution")
    if (
        record.get("retry_of_task_id") != original_submission["task_id"]
        or record.get("original_plan_payload_sha256")
        != original_plan["payload_sha256"]
        or record.get("original_submission_payload_sha256")
        != original_submission["payload_sha256"]
        or not isinstance(execution, dict)
        or _timeout_failure_evidence(
            execution, submission=original_submission
        )
        != execution
        or canonical_sha256(execution)
        != record.get("original_task_execution_sha256")
        or any(
            plan.get(name) != original_plan.get(name)
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
            )
        )
    ):
        raise HandoffContractError(
            "diagnostic Standard timeout retry ancestry drifted"
        )
    return original_plan, original_submission, execution


def create_timeout_retry_plan(
    *,
    original_plan_path: Path,
    original_submission_path: Path,
    output: Path,
    scheduler_url: str = DIAGNOSTIC_SCHEDULER_URL,
    task_reader: Any = None,
    strict_node_name: str = "",
) -> Path:
    original_plan, params, selected = _load_plan(original_plan_path)
    if _plan_is_timeout_retry(original_plan):
        raise HandoffContractError(
            "a timeout retry cannot be based on another timeout retry"
        )
    original_submission = _load_submission(
        original_submission_path, plan=original_plan
    )
    normalized_scheduler_url = scheduler_url.rstrip("/")
    if normalized_scheduler_url != original_submission["scheduler_url"]:
        raise HandoffContractError(
            "timeout retry Scheduler origin differs from original submission"
        )
    reader = task_reader or _scheduler_task_snapshot
    execution = _timeout_failure_evidence(
        reader(
            scheduler_url=normalized_scheduler_url,
            task_id=int(original_submission["task_id"]),
        ),
        submission=original_submission,
    )
    original_profile = production._read_json(
        original_plan_path.resolve(strict=True).parent
        / original_plan["profile"]["path"]
    )
    retry_profile, retry_profile_source = _profile_content(
        timeout_retry=True
    )
    if (
        retry_profile["param_overrides"]
        != original_profile["param_overrides"]
        or retry_profile["fixed_boundary_contract"]
        != original_profile["fixed_boundary_contract"]
        or production._effective_params(params, retry_profile)
        != production._effective_params(params, original_profile)
    ):
        raise HandoffContractError(
            "timeout retry profile changes fixed physics"
        )
    stem = str(original_plan["candidate_physics_sha256"])[:12]
    strict_contract = None
    if strict_node_name not in (None, ""):
        strict_contract = _strict_node_plan_contract(strict_node_name)
        node_slug = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            strict_contract["requested_node_name"],
        )[:16]
        task_name = (
            f"mft-goal-diag-standard-timeout-strict-r2-"
            f"{node_slug}-{stem}"
        )
        workdir = (
            f"mft_goal_diag_standard_timeout_strict_r2_"
            f"{node_slug}_{stem}"
        )
    else:
        task_name = f"mft-goal-diag-standard-timeout-r1-{stem}"
        workdir = f"mft_goal_diag_standard_timeout_r1_{stem}"
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        retry_profile,
        original_plan["solver_revision"],
        original_plan["library_revision"],
    )
    original_retained = original_plan["stage"]["retained_aedt_bundle"]
    if (
        retained is None
        or retained["dedupe_key"] == original_retained["dedupe_key"]
        or retained["relative_directory"]
        == original_retained["relative_directory"]
        or retained["profile_sha256"]
        == original_retained["profile_sha256"]
    ):
        raise HandoffContractError(
            "timeout retry did not derive a distinct immutable identity"
        )
    retry_record = {
        "schema_version": TIMEOUT_RETRY_EVIDENCE_SCHEMA,
        "retry_of_task_id": original_submission["task_id"],
        "original_plan": production._file_record(
            original_plan_path.resolve(strict=True)
        ),
        "original_plan_payload_sha256": original_plan["payload_sha256"],
        "original_submission": production._file_record(
            original_submission_path.resolve(strict=True)
        ),
        "original_submission_payload_sha256": original_submission[
            "payload_sha256"
        ],
        "original_task_execution": execution,
        "original_task_execution_sha256": canonical_sha256(execution),
        "scheduler_url": normalized_scheduler_url,
        "timeout_reason": (
            f"task timed out after {STANDARD_RESOURCES['timeout_seconds']}s"
        ),
    }
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"diagnostic timeout retry plan output already exists: "
            f"{destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_path = production._write_immutable_json(
            staging / "selected_candidate.json", selected
        )
        params_path = production._write_immutable_json(
            staging / "fea_params.json", params
        )
        profile_path = production._write_immutable_json(
            staging / "diagnostic_standard_timeout_retry_profile.json",
            retry_profile,
        )
        unsigned_plan = copy.deepcopy(original_plan)
        unsigned_plan.pop("payload_sha256", None)
        unsigned_plan.update(
            {
                "selected_candidate": {
                    "path": selected_path.name,
                    "sha256": production._sha256_file(selected_path),
                },
                "fea_params": {
                    "path": params_path.name,
                    "sha256": production._sha256_file(params_path),
                },
                "profile": {
                    "path": profile_path.name,
                    "sha256": production._sha256_file(profile_path),
                    "canonical_sha256": canonical_sha256(retry_profile),
                    "source": retry_profile_source,
                },
                "stage": {
                    **copy.deepcopy(original_plan["stage"]),
                    "task_name": task_name,
                    "workdir": workdir,
                    "profile_sha256": canonical_sha256(retry_profile),
                    "resources": copy.deepcopy(TIMEOUT_RETRY_RESOURCES),
                    "retained_aedt_bundle": retained,
                    "retention_run_root": _retention_run_root_evidence(
                        retained
                    ),
                },
                "available_submission_commands": [
                    "submit-timeout-retry"
                ],
                "retry_of_timeout": retry_record,
                **(
                    {"scheduler_strict_node_contract": strict_contract}
                    if strict_contract is not None
                    else {}
                ),
            }
        )
        plan_path = production._write_immutable_json(
            staging / "diagnostic_timeout_retry_plan.json",
            production._seal(unsigned_plan),
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / plan_path.name


def _plan_artifact(root: Path, record: Any, label: str) -> Path:
    if (
        not isinstance(record, dict)
        or not {"path", "sha256"}.issubset(record)
    ):
        raise HandoffContractError(f"{label} plan record is malformed")
    target = production._contained_file(root, record["path"], label)
    if production._sha256_file(target) != production._require_sha(
        record["sha256"], f"{label} SHA"
    ):
        raise HandoffContractError(f"{label} plan bytes drifted")
    return target


def _load_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(
        production._read_json(resolved), PLAN_SCHEMA
    )
    timeout_retry = _plan_is_timeout_retry(plan)
    expected_commands = (
        ["submit-timeout-retry"]
        if timeout_retry
        else ["submit-standard"]
    )
    flags = _diagnostic_flags()
    if (
        plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or any(plan.get(name) is not value for name, value in flags.items())
        or plan.get("available_submission_commands") != expected_commands
        or plan.get("physics_override_allowed") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("retention_required") is not True
        or plan.get("prune_protection_required") is not True
    ):
        raise HandoffContractError("diagnostic Standard plan contract drifted")
    root = resolved.parent
    params = production._read_json(
        _plan_artifact(root, plan.get("fea_params"), "diagnostic FEA params")
    )
    if (
        set(params) != set(ALL_INPUT_KEYS)
        or canonical_sha256(params) != plan.get("fea_params_sha256")
    ):
        raise HandoffContractError("diagnostic FEA parameter identity drifted")
    selected = production._validate_seal(
        production._read_json(
            _plan_artifact(
                root,
                plan.get("selected_candidate"),
                "diagnostic selected candidate",
            )
        ),
        SELECTED_SCHEMA,
    )
    if (
        any(
            selected.get(name) is not value for name, value in flags.items()
        )
        or selected.get("row_contract", {}).get(
            "physical_geometry_sha256"
        )
        != plan.get("candidate_physics_sha256")
        or selected.get("row_contract", {}).get("fea_params_sha256")
        != plan.get("fea_params_sha256")
        or canonical_sha256(_authentication_payload(selected))
        != plan.get("search_authority_sha256")
    ):
        raise HandoffContractError(
            "diagnostic selected candidate identity drifted"
        )
    profile_record = plan.get("profile")
    if not isinstance(profile_record, dict):
        raise HandoffContractError("diagnostic profile record is absent")
    profile_path = _plan_artifact(
        root, profile_record, "diagnostic Standard profile"
    )
    profile = production._read_json(profile_path)
    _validate_profile(profile, timeout_retry=timeout_retry)
    effective = production._effective_params(params, profile)
    stage = plan.get("stage")
    if not isinstance(stage, dict):
        raise HandoffContractError("diagnostic Standard stage is absent")
    retained = scheduler_client.retained_aedt_identity(
        stage.get("task_name"),
        params,
        profile,
        plan.get("solver_revision"),
        plan.get("library_revision"),
    )
    if (
        profile_record.get("canonical_sha256")
        != canonical_sha256(profile)
        or stage.get("name") != "standard"
        or stage.get("profile_sha256") != canonical_sha256(profile)
        or stage.get("effective_params_sha256")
        != canonical_sha256(effective)
        or stage.get("resources") != _plan_resources(plan)
        or stage.get("retained_aedt_bundle") != retained
        or stage.get("retention_run_root")
        != _retention_run_root_evidence(retained)
        or stage.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or stage.get("scheduler_url") != DIAGNOSTIC_SCHEDULER_URL
        or stage.get("aedt_backend") != "standalone"
        or stage.get("full_model") != 0
        or stage.get("thermal_symmetry") != "eighth"
    ):
        raise HandoffContractError(
            "diagnostic Standard execution identity drifted"
        )
    if timeout_retry:
        _validate_timeout_retry_record(plan)
        strict_contract = _plan_strict_node_contract(plan)
        if strict_contract is not None:
            node_slug = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                strict_contract["requested_node_name"],
            )[:16]
            stem = str(plan["candidate_physics_sha256"])[:12]
            if (
                stage.get("task_name")
                != (
                    "mft-goal-diag-standard-timeout-strict-r2-"
                    f"{node_slug}-{stem}"
                )
                or stage.get("workdir")
                != (
                    "mft_goal_diag_standard_timeout_strict_r2_"
                    f"{node_slug}_{stem}"
                )
            ):
                raise HandoffContractError(
                    "strict timeout retry r2 identity drifted"
                )
    elif "scheduler_strict_node_contract" in plan:
        raise HandoffContractError(
            "strict node placement is restricted to timeout retries"
        )
    return plan, params, selected


def _fresh_selection_reauthentication(
    *,
    plan: Mapping[str, Any],
    selected: Mapping[str, Any],
    predictor: Any | None = None,
) -> dict[str, Any]:
    source = selected.get("selection_source")
    if (
        not isinstance(source, dict)
        or source.get("kind")
        != "diagnostic_partial_terminal_Llt_truth_probe"
        or any(
            source.get(name) is not value
            for name, value in _diagnostic_flags().items()
        )
    ):
        raise HandoffContractError(
            "diagnostic submission selection source drifted"
        )
    selection_record = source.get("selection_manifest")
    if not isinstance(selection_record, dict):
        raise HandoffContractError(
            "diagnostic selection manifest record is absent"
        )
    selection_path = Path(str(selection_record.get("path") or ""))
    if production._file_record(selection_path) != selection_record:
        raise HandoffContractError(
            "diagnostic selection manifest bytes drifted"
        )
    authenticated = authenticate_candidate(
        selection_manifest_path=selection_path,
        candidate_physics_sha256=str(
            plan.get("candidate_physics_sha256") or ""
        ),
        predictor=predictor,
    )
    if _authentication_payload(authenticated) != _authentication_payload(
        selected
    ):
        raise HandoffContractError(
            "diagnostic candidate differs from fresh source authentication"
        )
    identity_sha = canonical_sha256(_authentication_payload(authenticated))
    if identity_sha != plan.get("search_authority_sha256"):
        raise HandoffContractError(
            "diagnostic fresh search authority differs from plan"
        )
    return {
        "schema_version": (
            "mft-goal-diagnostic-standard-submit-reauth-v1"
        ),
        "search_authority_sha256": identity_sha,
        "selection_manifest_sha256": selection_record["sha256"],
        "source_result_sha256": authenticated["source_result"]["sha256"],
        "task_payload_sha256": authenticated["task_payload"]["sha256"],
        "fresh_candidate_reauthenticated": True,
        **_diagnostic_flags(),
    }


def _same_allocation_anchor_evidence(
    snapshot: Mapping[str, Any],
    *,
    task_id: int,
    allocation_id: int,
    slurm_job_id: str,
    account_name: str,
    node_name: str,
) -> dict[str, Any]:
    evidence = {
        "task_id": snapshot.get("task_id", snapshot.get("id")),
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": (
            snapshot.get("actual_node_name")
            or snapshot.get("allocation_node_name")
        ),
        "scheduling_profile": snapshot.get("scheduling_profile"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
        or not str(slurm_job_id).isdigit()
        or not str(account_name).strip()
        or not str(node_name).strip()
        or evidence["task_id"] != task_id
        or not str(evidence["name"] or "").strip()
        or evidence["status"] != "running"
        or evidence["state"] != "running"
        or evidence["allocation_id"] != allocation_id
        or evidence["slurm_job_id"] != str(slurm_job_id)
        or evidence["account_name"] != account_name
        or evidence["actual_node_name"] != node_name
        or evidence["scheduling_profile"] != "fea_bursty"
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["cpus"] != TIMEOUT_RETRY_RESOURCES["cpus"]
        or evidence["memory_mb"] != 32768
        or evidence["timeout_seconds"]
        != TIMEOUT_RETRY_RESOURCES["timeout_seconds"]
        or not str(evidence["dedupe_key"] or "").strip()
        or not str(evidence["started_at"] or "").strip()
        or evidence["finished_at"] not in (None, "")
    ):
        raise HandoffContractError(
            "diagnostic timeout retry same-allocation anchor drifted"
        )
    return evidence


def _same_allocation_submitted_task_evidence(
    snapshot: Mapping[str, Any],
    *,
    task_id: int,
    task_name: str,
    dedupe_key: str,
    anchor_task_id: int,
    allocation_id: int,
    slurm_job_id: str,
    account_name: str,
    node_name: str,
) -> dict[str, Any]:
    evidence = {
        "task_id": snapshot.get("task_id", snapshot.get("id")),
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": (
            snapshot.get("actual_node_name")
            or snapshot.get("allocation_node_name")
            or ""
        ),
        "scheduling_profile": snapshot.get("scheduling_profile"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "timeout_seconds": snapshot.get("timeout_seconds"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "same_node_as_task_id": snapshot.get("same_node_as_task_id"),
        "requested_allocation_id": snapshot.get(
            "requested_allocation_id", 0
        ),
        "finished_at": snapshot.get("finished_at"),
    }
    status = evidence["status"]
    state = evidence["state"]
    assigned = evidence["allocation_id"] not in (None, 0)
    assigned_identity_valid = (
        evidence["allocation_id"] == allocation_id
        and evidence["slurm_job_id"] == str(slurm_job_id)
        and evidence["account_name"] == account_name
        and evidence["actual_node_name"] == node_name
        and status in ("attaching", "running")
        and state in ("attaching", "running")
    )
    queued_identity_valid = (
        not assigned
        and status == "queued"
        and state == "queued"
        and evidence["slurm_job_id"] == ""
        and evidence["actual_node_name"] == ""
    )
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or evidence["task_id"] != task_id
        or evidence["name"] != task_name
        or evidence["dedupe_key"] != dedupe_key
        or evidence["same_node_as_task_id"] != anchor_task_id
        or evidence["requested_allocation_id"] not in (None, 0)
        or evidence["scheduling_profile"] != "fea_bursty"
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["cpus"] != TIMEOUT_RETRY_RESOURCES["cpus"]
        or evidence["memory_mb"] != 32768
        or evidence["timeout_seconds"]
        != TIMEOUT_RETRY_RESOURCES["timeout_seconds"]
        or evidence["finished_at"] not in (None, "")
        or not (assigned_identity_valid or queued_identity_valid)
    ):
        raise HandoffContractError(
            "diagnostic timeout retry same-allocation task binding drifted"
        )
    return evidence


def _same_allocation_placement_contract(
    *,
    anchor_before: Mapping[str, Any],
    anchor_after: Mapping[str, Any],
    submitted_task: Mapping[str, Any],
    anchor_task_id: int,
    allocation_id: int,
    slurm_job_id: str,
    account_name: str,
    node_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": SAME_ALLOCATION_PLACEMENT_SCHEMA,
        "same_node_as_task_id": anchor_task_id,
        "expected_allocation_id": allocation_id,
        "expected_slurm_job_id": str(slurm_job_id),
        "expected_account_name": account_name,
        "expected_node_name": node_name,
        "anchor_before_submission": copy.deepcopy(anchor_before),
        "anchor_after_submission": copy.deepcopy(anchor_after),
        "submitted_task_after_submission": copy.deepcopy(submitted_task),
        "same_node_reference_allocation_enforced": True,
        "fallback_allocation_allowed": False,
        "requested_allocation_id_used": False,
    }


def _validate_same_allocation_placement_contract(
    value: Any,
    *,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "same_node_as_task_id",
        "expected_allocation_id",
        "expected_slurm_job_id",
        "expected_account_name",
        "expected_node_name",
        "anchor_before_submission",
        "anchor_after_submission",
        "submitted_task_after_submission",
        "same_node_reference_allocation_enforced",
        "fallback_allocation_allowed",
        "requested_allocation_id_used",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise HandoffContractError(
            "diagnostic same-allocation placement contract is malformed"
        )
    anchor_task_id = value.get("same_node_as_task_id")
    allocation_id = value.get("expected_allocation_id")
    slurm_job_id = value.get("expected_slurm_job_id")
    account_name = value.get("expected_account_name")
    node_name = value.get("expected_node_name")
    if (
        value.get("schema_version") != SAME_ALLOCATION_PLACEMENT_SCHEMA
        or value.get("same_node_reference_allocation_enforced") is not True
        or value.get("fallback_allocation_allowed") is not False
        or value.get("requested_allocation_id_used") is not False
    ):
        raise HandoffContractError(
            "diagnostic same-allocation placement policy drifted"
        )
    before = _same_allocation_anchor_evidence(
        value.get("anchor_before_submission") or {},
        task_id=anchor_task_id,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        account_name=account_name,
        node_name=node_name,
    )
    after = _same_allocation_anchor_evidence(
        value.get("anchor_after_submission") or {},
        task_id=anchor_task_id,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        account_name=account_name,
        node_name=node_name,
    )
    submitted = _same_allocation_submitted_task_evidence(
        value.get("submitted_task_after_submission") or {},
        task_id=submission.get("task_id"),
        task_name=str(submission.get("task_name") or ""),
        dedupe_key=str(submission.get("dedupe_key") or ""),
        anchor_task_id=anchor_task_id,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        account_name=account_name,
        node_name=node_name,
    )
    normalized = _same_allocation_placement_contract(
        anchor_before=before,
        anchor_after=after,
        submitted_task=submitted,
        anchor_task_id=anchor_task_id,
        allocation_id=allocation_id,
        slurm_job_id=slurm_job_id,
        account_name=account_name,
        node_name=node_name,
    )
    if normalized != value:
        raise HandoffContractError(
            "diagnostic same-allocation placement evidence drifted"
        )
    return normalized


def _strict_node_task_evidence(
    snapshot: Mapping[str, Any],
    *,
    task_id: int,
    task_name: str,
    dedupe_key: str,
    node_name: str,
    same_node_as_task_id: int = 0,
    expected_allocation_id: int = 0,
    expected_slurm_job_id: str = "",
    expected_account_name: str = "",
) -> dict[str, Any]:
    requested_node = _strict_node_name(node_name)
    evidence = {
        "task_id": snapshot.get("task_id", snapshot.get("id")),
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "project": snapshot.get("project"),
        "scheduling_profile": snapshot.get("scheduling_profile"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "node_name": snapshot.get("node_name"),
        "requested_node_name": snapshot.get("requested_node_name"),
        "node_name_policy": snapshot.get("node_name_policy"),
        "requested_node_name_policy": snapshot.get(
            "requested_node_name_policy"
        ),
        "strict_node_placement": snapshot.get("strict_node_placement"),
        "placement_contract_satisfied": snapshot.get(
            "placement_contract_satisfied"
        ),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "assigned_allocation": snapshot.get(
            "assigned_allocation", snapshot.get("allocation_id")
        ),
        "allocation_node_name": snapshot.get("allocation_node_name"),
        "actual_node_name": snapshot.get("actual_node_name"),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "account_name": snapshot.get("account_name"),
        "requested_account_name": snapshot.get("requested_account_name"),
        "same_node_as_task_id": snapshot.get("same_node_as_task_id", 0),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
    }
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or evidence["task_id"] != task_id
        or evidence["name"] != task_name
        or evidence["dedupe_key"] != dedupe_key
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["scheduling_profile"] != "fea_bursty"
        or evidence["aedt_backend"] != "standalone"
        or evidence["cpus"] != TIMEOUT_RETRY_RESOURCES["cpus"]
        or evidence["memory_mb"] != 32768
        or evidence["node_name"] != requested_node
        or evidence["requested_node_name"] != requested_node
        or evidence["node_name_policy"] != "strict"
        or evidence["requested_node_name_policy"] != "strict"
        or evidence["strict_node_placement"] is not True
        or evidence["same_node_as_task_id"] != same_node_as_task_id
    ):
        raise HandoffContractError(
            "diagnostic strict-node Scheduler identity drifted"
        )
    status = evidence["status"]
    state = evidence["state"]
    allocation_id = evidence["allocation_id"]
    assigned_allocation = evidence["assigned_allocation"]
    if status == "queued" and state == "queued":
        valid_state = (
            allocation_id in (None, 0)
            and assigned_allocation in (None, 0)
            and evidence["slurm_job_id"] == ""
            and not str(evidence["allocation_node_name"] or "").strip()
            and not str(evidence["actual_node_name"] or "").strip()
            and evidence["placement_contract_satisfied"] is False
            and not str(evidence["started_at"] or "").strip()
            and evidence["finished_at"] in (None, "")
        )
    elif status == "attaching" and state == "attaching":
        valid_state = (
            isinstance(allocation_id, int)
            and not isinstance(allocation_id, bool)
            and allocation_id > 0
            and assigned_allocation == allocation_id
            and evidence["allocation_node_name"] == requested_node
            and evidence["actual_node_name"] == requested_node
            and evidence["slurm_job_id"].isdigit()
            and str(evidence["account_name"] or "").strip() != ""
            and evidence["placement_contract_satisfied"] is False
            and not str(evidence["started_at"] or "").strip()
            and evidence["finished_at"] in (None, "")
        )
    elif (
        (status == "running" and state == "running")
        or (status == "completed" and state == "succeeded")
    ):
        valid_state = (
            isinstance(allocation_id, int)
            and not isinstance(allocation_id, bool)
            and allocation_id > 0
            and assigned_allocation == allocation_id
            and evidence["allocation_node_name"] == requested_node
            and evidence["actual_node_name"] == requested_node
            and evidence["slurm_job_id"].isdigit()
            and str(evidence["account_name"] or "").strip() != ""
            and evidence["placement_contract_satisfied"] is True
            and str(evidence["started_at"] or "").strip() != ""
            and (
                evidence["finished_at"] in (None, "")
                if status == "running"
                else str(evidence["finished_at"] or "").strip() != ""
            )
        )
    else:
        valid_state = False
    if not valid_state:
        raise HandoffContractError(
            "diagnostic strict-node Scheduler placement readback drifted"
        )
    if expected_allocation_id and allocation_id not in (
        None,
        0,
        expected_allocation_id,
    ):
        raise HandoffContractError(
            "diagnostic strict-node allocation fell back or drifted"
        )
    if expected_slurm_job_id and evidence["slurm_job_id"] not in (
        "",
        str(expected_slurm_job_id),
    ):
        raise HandoffContractError(
            "diagnostic strict-node Slurm job drifted"
        )
    if expected_account_name:
        requested_account = str(
            evidence["requested_account_name"] or ""
        ).strip()
        actual_account = str(evidence["account_name"] or "").strip()
        if requested_account != expected_account_name or actual_account not in (
            "",
            expected_account_name,
        ):
            raise HandoffContractError(
                "diagnostic strict-node account drifted"
            )
    return evidence


def _strict_node_submission_contract(
    *,
    plan_contract: Mapping[str, Any],
    submission_trace: Mapping[str, Any],
    durable_readback: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STRICT_NODE_SUBMISSION_SCHEMA,
        "plan_contract": copy.deepcopy(plan_contract),
        "submission_source": submission_trace["submission_source"],
        "scheduler_mutation_performed": submission_trace[
            "scheduler_mutation_performed"
        ],
        "api_pre_submission_readback": copy.deepcopy(
            submission_trace.get("api_pre_submission_readback")
        ),
        "api_post_submission_response": copy.deepcopy(
            submission_trace.get("api_post_submission_response")
        ),
        "api_durable_get_readback": copy.deepcopy(durable_readback),
        "fallback_allocation_observed": False,
        "strict_policy_authenticated": True,
    }


def _validate_strict_node_submission_contract(
    value: Any,
    *,
    submission: Mapping[str, Any],
    plan_contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "plan_contract",
        "submission_source",
        "scheduler_mutation_performed",
        "api_pre_submission_readback",
        "api_post_submission_response",
        "api_durable_get_readback",
        "fallback_allocation_observed",
        "strict_policy_authenticated",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version") != STRICT_NODE_SUBMISSION_SCHEMA
        or value.get("plan_contract") != plan_contract
        or value.get("fallback_allocation_observed") is not False
        or value.get("strict_policy_authenticated") is not True
        or not isinstance(value.get("scheduler_mutation_performed"), bool)
    ):
        raise HandoffContractError(
            "diagnostic strict-node submission contract drifted"
        )
    placement = submission.get("scheduler_placement_contract")
    if placement is None:
        same_node_id = 0
        expected_allocation_id = 0
        expected_slurm_job_id = ""
        expected_account_name = ""
    else:
        same_node_id = placement.get("same_node_as_task_id")
        expected_allocation_id = placement.get("expected_allocation_id")
        expected_slurm_job_id = placement.get("expected_slurm_job_id")
        expected_account_name = placement.get("expected_account_name")
    evidence_options = {
        "task_id": submission.get("task_id"),
        "task_name": str(submission.get("task_name") or ""),
        "dedupe_key": str(submission.get("dedupe_key") or ""),
        "node_name": plan_contract["requested_node_name"],
        "same_node_as_task_id": same_node_id,
        "expected_allocation_id": expected_allocation_id,
        "expected_slurm_job_id": expected_slurm_job_id,
        "expected_account_name": expected_account_name,
    }
    pre_raw = value.get("api_pre_submission_readback")
    post_raw = value.get("api_post_submission_response")
    durable_raw = value.get("api_durable_get_readback")
    pre = (
        _strict_node_task_evidence(pre_raw, **evidence_options)
        if isinstance(pre_raw, dict)
        else None
    )
    post = (
        _strict_node_task_evidence(post_raw, **evidence_options)
        if isinstance(post_raw, dict)
        else None
    )
    if pre is None and post is None:
        raise HandoffContractError(
            "diagnostic strict-node submission has no API submission readback"
        )
    durable = _strict_node_task_evidence(
        durable_raw or {}, **evidence_options
    )
    source = value.get("submission_source")
    if (
        source
        not in {
            "post_created",
            "post_deduped",
            "pre_submission_reconciliation",
            "post_rejection_reconciliation",
            "post_response_reconciliation",
        }
        or (
            source == "post_created"
            and (
                value["scheduler_mutation_performed"] is not True
                or post is None
            )
        )
        or (
            source == "pre_submission_reconciliation"
            and (
                value["scheduler_mutation_performed"] is not False
                or pre is None
            )
        )
    ):
        raise HandoffContractError(
            "diagnostic strict-node submission source drifted"
        )
    normalized_trace = {
        "submission_source": source,
        "scheduler_mutation_performed": value[
            "scheduler_mutation_performed"
        ],
        "api_pre_submission_readback": pre,
        "api_post_submission_response": post,
    }
    normalized = _strict_node_submission_contract(
        plan_contract=plan_contract,
        submission_trace=normalized_trace,
        durable_readback=durable,
    )
    if normalized != value:
        raise HandoffContractError(
            "diagnostic strict-node API readback evidence drifted"
        )
    return normalized


def _submit_standard_plan(
    *,
    plan_path: Path,
    scheduler_cutover_receipt_path: Path,
    output: Path,
    expected_timeout_retry: bool,
    priority: int = 0,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    live_reader: Any = _default_scheduler_live_reader,
    task_reader: Any = None,
    same_node_as_task_id: int = 0,
    expected_allocation_id: int = 0,
    expected_slurm_job_id: str = "",
    expected_account_name: str = "",
    expected_node_name: str = "",
) -> Path:
    plan, params, selected = _load_plan(plan_path)
    timeout_retry = _plan_is_timeout_retry(plan)
    strict_node_contract = _plan_strict_node_contract(plan)
    strict_node_pin = (
        _strict_node_scheduler_pin(
            strict_node_contract, require_active=True
        )
        if strict_node_contract is not None
        else None
    )
    if timeout_retry is not expected_timeout_retry:
        command = (
            "submit-timeout-retry"
            if timeout_retry
            else "submit-standard"
        )
        raise HandoffContractError(
            f"diagnostic plan requires {command}"
        )
    placement_requested = (
        same_node_as_task_id != 0
        or expected_allocation_id != 0
        or bool(str(expected_slurm_job_id).strip())
        or bool(str(expected_account_name).strip())
        or bool(str(expected_node_name).strip())
    )
    if placement_requested and (
        not timeout_retry
        or isinstance(same_node_as_task_id, bool)
        or not isinstance(same_node_as_task_id, int)
        or same_node_as_task_id <= 0
        or isinstance(expected_allocation_id, bool)
        or not isinstance(expected_allocation_id, int)
        or expected_allocation_id <= 0
        or not str(expected_slurm_job_id).isdigit()
        or not str(expected_account_name).strip()
        or not str(expected_node_name).strip()
    ):
        raise HandoffContractError(
            "same-allocation placement requires a timeout retry and the "
            "complete positive anchor identity"
        )
    if (
        strict_node_contract is not None
        and placement_requested
        and strict_node_contract["requested_node_name"]
        != str(expected_node_name).strip()
    ):
        raise HandoffContractError(
            "strict-node plan and same-allocation node differ"
        )
    reauthentication = _fresh_selection_reauthentication(
        plan=plan, selected=selected, predictor=predictor
    )
    cutover, launcher_before = _validate_scheduler_cutover_receipt(
        scheduler_cutover_receipt_path,
        verify_live_launcher=True,
        require_strict_node=strict_node_contract is not None,
        strict_node_contract=strict_node_contract,
        require_active_strict=strict_node_contract is not None,
    )
    stage = plan["stage"]
    if cutover["scheduler_url"] != stage["scheduler_url"]:
        raise HandoffContractError(
            "Scheduler cutover endpoint differs from diagnostic plan"
        )
    admission = _live_scheduler_admission_snapshot(
        scheduler_url=stage["scheduler_url"],
        reader=live_reader,
    )
    launcher_after = _live_launcher_identity(
        cutover,
        expected_sha256=(
            strict_node_pin["launcher_sha256"]
            if strict_node_pin is not None
            else SCHEDULER_LIVE_LAUNCHER_SHA256
        ),
    )
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler live launcher changed during admission checks"
        )
    retry_record = None
    reader = task_reader or _scheduler_task_snapshot
    anchor_before = None
    if timeout_retry:
        (
            _original_plan,
            original_submission,
            stored_execution,
        ) = _validate_timeout_retry_record(plan)
        live_execution = _timeout_failure_evidence(
            reader(
                scheduler_url=stage["scheduler_url"],
                task_id=int(original_submission["task_id"]),
            ),
            submission=original_submission,
        )
        if live_execution != stored_execution:
            raise HandoffContractError(
                "diagnostic timeout failure evidence changed before retry "
                "submission"
            )
        retry_record = copy.deepcopy(plan["retry_of_timeout"])
        if placement_requested:
            anchor_before = _same_allocation_anchor_evidence(
                reader(
                    scheduler_url=stage["scheduler_url"],
                    task_id=same_node_as_task_id,
                ),
                task_id=same_node_as_task_id,
                allocation_id=expected_allocation_id,
                slurm_job_id=str(expected_slurm_job_id),
                account_name=str(expected_account_name),
                node_name=str(expected_node_name),
            )
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"diagnostic submission receipt already exists: {target}"
        )
    profile = production._read_json(
        plan_path.resolve(strict=True).parent / plan["profile"]["path"]
    )
    environment, core_evidence = production._submission_environment(
        stage="standard",
        solver_revision=plan["solver_revision"],
        license_snapshot_path=None,
    )
    placement_submission_options = (
        {
            "account_name": str(expected_account_name),
            "node_name": str(expected_node_name),
            "same_node_as_task_id": same_node_as_task_id,
        }
        if placement_requested
        else {}
    )
    if strict_node_contract is not None:
        placement_submission_options.update(
            {
                "node_name": strict_node_contract[
                    "requested_node_name"
                ],
                "node_name_policy": "strict",
                "return_submission_evidence": True,
            }
        )
    submission_result = scheduler.submit_verification(
        stage["task_name"],
        stage["workdir"],
        params,
        profile,
        mem_mb=int(profile["mem_mb"]),
        cpus=int(profile["cpus"]),
        solver_revision=plan["solver_revision"],
        library_revision=plan["library_revision"],
        priority=priority,
        aedt_backend="standalone",
        submission_env=environment,
        required_project_cap=GOAL_FEA_PROJECT_CAP,
        max_project_active_tasks=GOAL_FEA_PROJECT_CAP,
        scheduler_url=stage["scheduler_url"],
        **placement_submission_options,
    )
    if strict_node_contract is not None:
        if (
            not isinstance(submission_result, dict)
            or set(submission_result)
            != {
                "task_id",
                "submission_source",
                "scheduler_mutation_performed",
                "api_pre_submission_readback",
                "api_post_submission_response",
            }
        ):
            raise HandoffContractError(
                "strict-node Scheduler submission returned no API evidence"
            )
        task_id = submission_result.get("task_id")
    else:
        task_id = submission_result
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError(
            "diagnostic Scheduler submission returned no durable task ID"
        )
    if (
        timeout_retry
        and task_id == int(plan["retry_of_timeout"]["retry_of_task_id"])
    ):
        raise HandoffContractError(
            "diagnostic timeout retry resolved to the original task ID"
        )
    placement_contract = None
    submitted_task_snapshot = None
    if placement_requested:
        anchor_after = _same_allocation_anchor_evidence(
            reader(
                scheduler_url=stage["scheduler_url"],
                task_id=same_node_as_task_id,
            ),
            task_id=same_node_as_task_id,
            allocation_id=expected_allocation_id,
            slurm_job_id=str(expected_slurm_job_id),
            account_name=str(expected_account_name),
            node_name=str(expected_node_name),
        )
        submitted_task_snapshot = reader(
            scheduler_url=stage["scheduler_url"],
            task_id=task_id,
        )
        submitted_task = _same_allocation_submitted_task_evidence(
            submitted_task_snapshot,
            task_id=task_id,
            task_name=stage["task_name"],
            dedupe_key=stage["retained_aedt_bundle"]["dedupe_key"],
            anchor_task_id=same_node_as_task_id,
            allocation_id=expected_allocation_id,
            slurm_job_id=str(expected_slurm_job_id),
            account_name=str(expected_account_name),
            node_name=str(expected_node_name),
        )
        placement_contract = _same_allocation_placement_contract(
            anchor_before=anchor_before,
            anchor_after=anchor_after,
            submitted_task=submitted_task,
            anchor_task_id=same_node_as_task_id,
            allocation_id=expected_allocation_id,
            slurm_job_id=str(expected_slurm_job_id),
            account_name=str(expected_account_name),
            node_name=str(expected_node_name),
        )
    strict_submission_contract = None
    if strict_node_contract is not None:
        if submitted_task_snapshot is None:
            submitted_task_snapshot = reader(
                scheduler_url=stage["scheduler_url"],
                task_id=task_id,
            )
        same_node_id = same_node_as_task_id if placement_requested else 0
        expected_allocation = (
            expected_allocation_id if placement_requested else 0
        )
        expected_job = (
            str(expected_slurm_job_id) if placement_requested else ""
        )
        expected_account = (
            str(expected_account_name) if placement_requested else ""
        )
        evidence_options = {
            "task_id": task_id,
            "task_name": stage["task_name"],
            "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
            "node_name": strict_node_contract["requested_node_name"],
            "same_node_as_task_id": same_node_id,
            "expected_allocation_id": expected_allocation,
            "expected_slurm_job_id": expected_job,
            "expected_account_name": expected_account,
        }
        raw_pre = submission_result.get("api_pre_submission_readback")
        raw_post = submission_result.get("api_post_submission_response")
        normalized_trace = {
            "submission_source": submission_result["submission_source"],
            "scheduler_mutation_performed": submission_result[
                "scheduler_mutation_performed"
            ],
            "api_pre_submission_readback": (
                _strict_node_task_evidence(
                    raw_pre, **evidence_options
                )
                if isinstance(raw_pre, dict)
                else None
            ),
            "api_post_submission_response": (
                _strict_node_task_evidence(
                    raw_post, **evidence_options
                )
                if isinstance(raw_post, dict)
                else None
            ),
        }
        if (
            normalized_trace["api_pre_submission_readback"] is None
            and normalized_trace["api_post_submission_response"] is None
        ):
            raise HandoffContractError(
                "strict-node Scheduler submission has no API readback"
            )
        durable_readback = _strict_node_task_evidence(
            submitted_task_snapshot, **evidence_options
        )
        strict_submission_contract = _strict_node_submission_contract(
            plan_contract=strict_node_contract,
            submission_trace=normalized_trace,
            durable_readback=durable_readback,
        )
    receipt = production._seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "stage": "standard",
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "search_authority_reauthentication": reauthentication,
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
            "resources": stage["resources"],
            "aedt_backend": "standalone",
            "core_policy": core_evidence,
            "retained_aedt_bundle": stage["retained_aedt_bundle"],
            "retention_run_root": stage["retention_run_root"],
            "scheduler_cutover_receipt": production._file_record(
                scheduler_cutover_receipt_path.resolve(strict=True)
            ),
            "scheduler_cutover_payload_sha256": cutover[
                "payload_sha256"
            ],
            "scheduler_live_launcher_identity": launcher_after,
            "scheduler_admission_snapshot": admission,
            "scheduler_url": stage["scheduler_url"],
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "scheduler_project_mutation_performed": False,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": True,
            "retention_required": True,
            "prune_protection_required": True,
            **(
                {"retry_of_timeout": retry_record}
                if timeout_retry
                else {}
            ),
            **(
                {"scheduler_placement_contract": placement_contract}
                if placement_contract is not None
                else {}
            ),
            **(
                {
                    "scheduler_strict_node_contract": (
                        strict_submission_contract
                    )
                }
                if strict_submission_contract is not None
                else {}
            ),
            **_diagnostic_flags(),
        }
    )
    return production._write_immutable_json(target, receipt)


def submit_standard(
    *,
    plan_path: Path,
    scheduler_cutover_receipt_path: Path,
    output: Path,
    priority: int = 0,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    live_reader: Any = _default_scheduler_live_reader,
) -> Path:
    return _submit_standard_plan(
        plan_path=plan_path,
        scheduler_cutover_receipt_path=scheduler_cutover_receipt_path,
        output=output,
        expected_timeout_retry=False,
        priority=priority,
        scheduler=scheduler,
        predictor=predictor,
        live_reader=live_reader,
    )


def submit_timeout_retry(
    *,
    plan_path: Path,
    scheduler_cutover_receipt_path: Path,
    output: Path,
    priority: int = 0,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    live_reader: Any = _default_scheduler_live_reader,
    task_reader: Any = None,
    same_node_as_task_id: int = 0,
    expected_allocation_id: int = 0,
    expected_slurm_job_id: str = "",
    expected_account_name: str = "",
    expected_node_name: str = "",
) -> Path:
    return _submit_standard_plan(
        plan_path=plan_path,
        scheduler_cutover_receipt_path=scheduler_cutover_receipt_path,
        output=output,
        expected_timeout_retry=True,
        priority=priority,
        scheduler=scheduler,
        predictor=predictor,
        live_reader=live_reader,
        task_reader=task_reader,
        same_node_as_task_id=same_node_as_task_id,
        expected_allocation_id=expected_allocation_id,
        expected_slurm_job_id=expected_slurm_job_id,
        expected_account_name=expected_account_name,
        expected_node_name=expected_node_name,
    )


def _load_submission(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = production._validate_seal(
        production._read_json(path.resolve(strict=True)), SUBMISSION_SCHEMA
    )
    stage = plan["stage"]
    timeout_retry = _plan_is_timeout_retry(plan)
    strict_node_contract = _plan_strict_node_contract(plan)
    strict_node_pin = (
        _strict_node_scheduler_pin(strict_node_contract)
        if strict_node_contract is not None
        else None
    )
    cutover_record = receipt.get("scheduler_cutover_receipt")
    if not isinstance(cutover_record, dict):
        raise HandoffContractError(
            "diagnostic Scheduler cutover receipt record is absent"
        )
    cutover_path = Path(str(cutover_record.get("path") or ""))
    if production._file_record(cutover_path) != cutover_record:
        raise HandoffContractError(
            "diagnostic Scheduler cutover receipt bytes drifted"
        )
    cutover, _unused_live_launcher = (
        _validate_scheduler_cutover_receipt(
            cutover_path,
            verify_live_launcher=False,
            require_strict_node=strict_node_contract is not None,
            strict_node_contract=strict_node_contract,
        )
    )
    admission = _validate_recorded_admission_snapshot(
        receipt.get("scheduler_admission_snapshot")
    )
    launcher = receipt.get("scheduler_live_launcher_identity")
    if (
        not isinstance(launcher, dict)
        or set(launcher) != {"path", "sha256", "size_bytes"}
        or not _same_absolute_regular_file_identity(
            launcher.get("path"), cutover["live_launcher_path"]
        )
        or launcher.get("sha256")
        != (
            strict_node_pin["launcher_sha256"]
            if strict_node_pin is not None
            else SCHEDULER_LIVE_LAUNCHER_SHA256
        )
        or isinstance(launcher.get("size_bytes"), bool)
        or not isinstance(launcher.get("size_bytes"), int)
        or launcher.get("size_bytes") <= 0
    ):
        raise HandoffContractError(
            "diagnostic live Scheduler launcher evidence drifted"
        )
    reauth = receipt.get("search_authority_reauthentication")
    expected_reauth_fields = {
        "schema_version",
        "search_authority_sha256",
        "selection_manifest_sha256",
        "source_result_sha256",
        "task_payload_sha256",
        "fresh_candidate_reauthenticated",
        *_diagnostic_flags(),
    }
    if (
        not isinstance(reauth, dict)
        or set(reauth) != expected_reauth_fields
        or reauth.get("schema_version")
        != "mft-goal-diagnostic-standard-submit-reauth-v1"
        or reauth.get("search_authority_sha256")
        != plan.get("search_authority_sha256")
        or reauth.get("fresh_candidate_reauthenticated") is not True
        or any(
            reauth.get(name) is not value
            for name, value in _diagnostic_flags().items()
        )
        or any(
            production._require_sha(reauth.get(name), name)
            != reauth.get(name)
            for name in (
                "selection_manifest_sha256",
                "source_result_sha256",
                "task_payload_sha256",
            )
        )
    ):
        raise HandoffContractError(
            "diagnostic submission source reauthentication drifted"
        )
    if (
        receipt.get("stage") != "standard"
        or receipt.get("plan_payload_sha256") != plan.get("payload_sha256")
        or receipt.get("candidate_physics_sha256")
        != plan.get("candidate_physics_sha256")
        or receipt.get("task_name") != stage["task_name"]
        or receipt.get("workdir") != stage["workdir"]
        or receipt.get("dedupe_key")
        != stage["retained_aedt_bundle"]["dedupe_key"]
        or receipt.get("retained_aedt_bundle")
        != stage["retained_aedt_bundle"]
        or receipt.get("retention_run_root")
        != stage["retention_run_root"]
        or receipt.get("scheduler_cutover_payload_sha256")
        != cutover["payload_sha256"]
        or admission.get("scheduler_url") != stage["scheduler_url"]
        or receipt.get("resources") != _plan_resources(plan)
        or receipt.get("solver_revision") != plan["solver_revision"]
        or receipt.get("library_revision") != plan["library_revision"]
        or receipt.get("profile_sha256") != stage["profile_sha256"]
        or receipt.get("effective_params_sha256")
        != stage["effective_params_sha256"]
        or receipt.get("aedt_backend") != "standalone"
        or receipt.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or receipt.get("scheduler_url")
        != stage["scheduler_url"]
        or receipt.get("scheduler_project_mutation_performed") is not False
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("scheduler_submission_performed") is not True
        or receipt.get("retention_required") is not True
        or receipt.get("prune_protection_required") is not True
        or (
            timeout_retry
            and receipt.get("retry_of_timeout")
            != plan.get("retry_of_timeout")
        )
        or (
            not timeout_retry
            and "retry_of_timeout" in receipt
        )
        or (
            strict_node_contract is not None
            and "scheduler_strict_node_contract" not in receipt
        )
        or (
            strict_node_contract is None
            and "scheduler_strict_node_contract" in receipt
        )
        or any(
            receipt.get(name) is not value
            for name, value in _diagnostic_flags().items()
        )
    ):
        raise HandoffContractError(
            "diagnostic Standard submission identity drifted"
        )
    if "scheduler_placement_contract" in receipt:
        if not timeout_retry:
            raise HandoffContractError(
                "same-allocation placement is restricted to timeout retries"
            )
        _validate_same_allocation_placement_contract(
            receipt.get("scheduler_placement_contract"),
            submission=receipt,
        )
    if strict_node_contract is not None:
        _validate_strict_node_submission_contract(
            receipt.get("scheduler_strict_node_contract"),
            submission=receipt,
            plan_contract=strict_node_contract,
        )
    if receipt.get("core_policy") != {
        "contract": production.STANDARD_CORE_CONTRACT,
        "requested_num_cores": 8,
        "auth_sha256": production._core_auth(
            plan["solver_revision"], 8
        ),
    }:
        raise HandoffContractError(
            "diagnostic Standard core policy drifted"
        )
    task_id = receipt.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError(
            "diagnostic submission task ID is invalid"
        )
    return receipt


BUNDLE_RECEIPT_EXTRA_FIELDS = {
    "results_path",
    "results_manifest_path",
    "results_manifest_schema_version",
    "results_manifest_sha256",
    "results_tree_sha256",
    "results_file_count",
    "results_size_bytes",
    "source_results_directory_name",
}
BUNDLE_RECEIPT_FIELDS = (
    production.REMOTE_RECEIPT_FIELDS | BUNDLE_RECEIPT_EXTRA_FIELDS
)


def _validate_bundle_receipt(
    value: Any,
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != BUNDLE_RECEIPT_FIELDS:
        raise HandoffContractError(
            "diagnostic retained artifact-bundle receipt is malformed"
        )
    receipt = copy.deepcopy(value)
    base = {
        name: receipt[name] for name in production.REMOTE_RECEIPT_FIELDS
    }
    base["schema_version"] = production.REMOTE_RECEIPT_SCHEMA
    shadow_submission = dict(submission)
    shadow_submission["retained_aedt"] = submission[
        "retained_aedt_bundle"
    ]
    production._validate_remote_receipt_payload(
        base, submission=shadow_submission, result=result
    )
    expected = submission["retained_aedt_bundle"]
    file_count = receipt.get("results_file_count")
    results_size = receipt.get("results_size_bytes")
    if (
        receipt.get("schema_version")
        != scheduler_client.RETAINED_AEDT_BUNDLE_RECEIPT_SCHEMA
        or receipt.get("results_path") != expected["results_path"]
        or receipt.get("results_manifest_path")
        != expected["results_manifest_path"]
        or receipt.get("results_manifest_schema_version")
        != RESULTS_MANIFEST_SCHEMA
        or production._require_sha(
            receipt.get("results_manifest_sha256"),
            "diagnostic results manifest SHA",
        )
        != receipt.get("results_manifest_sha256")
        or production._require_sha(
            receipt.get("results_tree_sha256"),
            "diagnostic results tree SHA",
        )
        != receipt.get("results_tree_sha256")
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or not 1 <= file_count <= (
            scheduler_client.RETAINED_AEDT_RESULTS_MAX_FILES
        )
        or isinstance(results_size, bool)
        or not isinstance(results_size, int)
        or not 0 <= results_size <= (
            scheduler_client.RETAINED_AEDT_RESULTS_MAX_BYTES
        )
        or receipt.get("source_results_directory_name")
        != f"{receipt['source_project_name']}.aedtresults"
    ):
        raise HandoffContractError(
            "diagnostic retained AEDT results receipt drifted"
        )
    return receipt


def _validate_results_manifest(
    value: Any,
    *,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "source_project_name",
            "source_results_directory_name",
            "retained_results_directory_name",
            "file_count",
            "size_bytes",
            "tree_sha256",
            "files",
        }
        or value.get("schema_version") != RESULTS_MANIFEST_SCHEMA
        or value.get("source_project_name") != result.get("project_name")
        or value.get("source_results_directory_name")
        != f"{result.get('project_name')}.aedtresults"
        or value.get("retained_results_directory_name")
        != PurePosixPath(str(receipt["results_path"])).name
        or not isinstance(value.get("files"), list)
    ):
        raise HandoffContractError(
            "diagnostic AEDT results manifest identity drifted"
        )
    files = value["files"]
    if (
        len(files) != receipt["results_file_count"]
        or value.get("file_count") != len(files)
        or not 1
        <= len(files)
        <= scheduler_client.RETAINED_AEDT_RESULTS_MAX_FILES
    ):
        raise HandoffContractError(
            "diagnostic AEDT results manifest file count drifted"
        )
    paths = []
    total = 0
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise HandoffContractError(
                "diagnostic AEDT results file record is malformed"
            )
        relative = str(record.get("path") or "")
        pure = PurePosixPath(relative)
        size = record.get("size_bytes")
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in relative
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or production._require_sha(
                record.get("sha256"), "diagnostic results file SHA"
            )
            != record.get("sha256")
        ):
            raise HandoffContractError(
                "diagnostic AEDT results file identity is unsafe"
            )
        paths.append(relative)
        total += size
    tree_sha = hashlib.sha256(
        json.dumps(
            files,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if (
        paths != sorted(paths)
        or len(set(paths)) != len(paths)
        or total != value.get("size_bytes")
        or total != receipt["results_size_bytes"]
        or tree_sha != value.get("tree_sha256")
        or tree_sha != receipt["results_tree_sha256"]
    ):
        raise HandoffContractError(
            "diagnostic AEDT results content inventory drifted"
        )
    return copy.deepcopy(value)


def _remote_manifest_bytes(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    max_bytes: int,
) -> bytes:
    pure = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure.is_absolute()
        or ".." in pure.parts
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or not 1
        <= max_bytes
        <= scheduler_client.RETAINED_AEDT_RESULTS_MANIFEST_MAX_BYTES
    ):
        raise HandoffContractError(
            "unsafe Scheduler diagnostic manifest request"
        )
    query = production.urllib.parse.urlencode(
        {
            "path": relative_path,
            "base": "remote_cwd",
            "max_bytes": max_bytes,
        }
    )
    url = (
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/remote-file?"
        f"{query}"
    )
    request = production.urllib.request.Request(url, method="GET")
    try:
        with production.urllib.request.urlopen(
            request, timeout=120.0
        ) as response:
            raw = response.read(max_bytes + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            "Scheduler diagnostic manifest fetch failed"
        ) from exc
    if len(raw) > max_bytes:
        raise HandoffContractError(
            "Scheduler diagnostic manifest exceeds byte bound"
        )
    return raw


def _validated_remote_bundle(
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
    scheduler_url: str,
    remote_reader: Any,
    manifest_reader: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    task_id = int(submission["task_id"])
    retained = submission["retained_aedt_bundle"]
    receipt_raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=retained["receipt_path"],
        max_bytes=MAX_REMOTE_METADATA_BYTES,
    )
    try:
        receipt_value = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "diagnostic retained bundle receipt is invalid JSON"
        ) from exc
    receipt = _validate_bundle_receipt(
        receipt_value, submission=submission, result=result
    )
    marker_raw = remote_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=retained["marker_path"],
        max_bytes=MAX_REMOTE_METADATA_BYTES,
    )
    if production._sha256_bytes(marker_raw) != receipt["marker_sha256"]:
        raise HandoffContractError(
            "diagnostic prune-protection marker SHA drifted"
        )
    try:
        marker_value = json.loads(marker_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "diagnostic prune-protection marker is invalid JSON"
        ) from exc
    marker = production._validate_marker_payload(
        marker_value, expected=retained
    )
    manifest_raw = manifest_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=retained["results_manifest_path"],
        max_bytes=(
            scheduler_client.RETAINED_AEDT_RESULTS_MANIFEST_MAX_BYTES
        ),
    )
    if (
        not manifest_raw
        or production._sha256_bytes(manifest_raw)
        != receipt["results_manifest_sha256"]
    ):
        raise HandoffContractError(
            "diagnostic AEDT results manifest SHA drifted"
        )
    try:
        manifest_value = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "diagnostic AEDT results manifest is invalid JSON"
        ) from exc
    manifest = _validate_results_manifest(
        manifest_value, receipt=receipt, result=result
    )
    return (
        receipt,
        marker,
        manifest,
        production._sha256_bytes(manifest_raw),
    )


def _result_identity_valid(
    *,
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    params: Mapping[str, Any],
    submission: Mapping[str, Any],
    profile: Mapping[str, Any],
    scheduler: Any = scheduler_client,
) -> None:
    effective = production._effective_params(params, profile)
    if (
        not scheduler.result_matches_params(
            result, effective, required_keys=set(ALL_INPUT_KEYS)
        )
        or production._require_revision(
            result.get("git_hash"), "diagnostic result git_hash"
        )
        != plan["solver_revision"]
        or production._require_revision(
            result.get("pyaedt_library_git_hash"),
            "diagnostic result library hash",
        )
        != plan["library_revision"]
        or production._integer(
            result.get("full_model"), "diagnostic result full_model"
        )
        != 0
    ):
        raise HandoffContractError(
            "diagnostic Standard result runtime identity drifted"
        )
    production._validate_result_core_policy(result, submission)


def _scheduler_task_snapshot(
    *, scheduler_url: str, task_id: int
) -> dict[str, Any]:
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError("diagnostic Scheduler task ID is invalid")
    request = production.urllib.request.Request(
        f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}",
        method="GET",
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=30.0
        ) as response:
            raw = response.read(1024 * 1024 + 1)
    except (OSError, production.urllib.error.URLError) as exc:
        raise HandoffContractError(
            "diagnostic Scheduler task snapshot fetch failed"
        ) from exc
    if len(raw) > 1024 * 1024:
        raise HandoffContractError(
            "diagnostic Scheduler task snapshot exceeds byte bound"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "diagnostic Scheduler task snapshot is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HandoffContractError(
            "diagnostic Scheduler task snapshot is not an object"
        )
    return value


def _task_execution_evidence(
    snapshot: Mapping[str, Any],
    *,
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = {
        "task_id": snapshot.get("task_id", snapshot.get("id")),
        "name": snapshot.get("name"),
        "status": snapshot.get("status"),
        "state": snapshot.get("state"),
        "exit_code": snapshot.get("exit_code"),
        "failure_message": snapshot.get("failure_message"),
        "slurm_job_id": str(snapshot.get("slurm_job_id") or ""),
        "allocation_id": snapshot.get(
            "allocation_id", snapshot.get("assigned_allocation")
        ),
        "account_name": snapshot.get("account_name"),
        "actual_node_name": snapshot.get("actual_node_name"),
        "cpus": snapshot.get("cpus"),
        "memory_mb": snapshot.get("memory_mb"),
        "aedt_backend": snapshot.get("aedt_backend"),
        "project": snapshot.get("project"),
        "dedupe_key": snapshot.get("dedupe_key"),
        "remote_cwd": snapshot.get("remote_cwd"),
        "remote_dir": snapshot.get("remote_dir"),
        "finished_at": snapshot.get("finished_at"),
    }
    allocation_id = evidence["allocation_id"]
    if (
        evidence["task_id"] != submission["task_id"]
        or evidence["name"] != submission["task_name"]
        or evidence["status"] != "completed"
        or evidence["state"] != "succeeded"
        or evidence["exit_code"] != 0
        or evidence["failure_message"] not in {"", None}
        or not str(evidence["slurm_job_id"]).isdigit()
        or isinstance(allocation_id, bool)
        or not isinstance(allocation_id, int)
        or allocation_id <= 0
        or evidence["cpus"] != 8
        or evidence["memory_mb"] != 32768
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["dedupe_key"] != submission["dedupe_key"]
        or not str(evidence["actual_node_name"] or "")
        or not str(evidence["remote_cwd"] or "")
        or not str(evidence["remote_dir"] or "")
        or not str(evidence["finished_at"] or "")
    ):
        raise HandoffContractError(
            "diagnostic Scheduler terminal execution evidence drifted"
        )
    placement = submission.get("scheduler_placement_contract")
    if placement is not None:
        validated = _validate_same_allocation_placement_contract(
            placement, submission=submission
        )
        if (
            evidence["allocation_id"]
            != validated["expected_allocation_id"]
            or evidence["slurm_job_id"]
            != validated["expected_slurm_job_id"]
            or evidence["account_name"]
            != validated["expected_account_name"]
            or evidence["actual_node_name"]
            != validated["expected_node_name"]
        ):
            raise HandoffContractError(
                "diagnostic terminal execution escaped its sealed "
                "same-allocation placement"
            )
    strict = submission.get("scheduler_strict_node_contract")
    if strict is not None:
        plan_contract = strict.get("plan_contract")
        if not isinstance(plan_contract, dict):
            raise HandoffContractError(
                "diagnostic strict-node terminal plan contract is absent"
            )
        same_node_id = (
            placement.get("same_node_as_task_id") if placement else 0
        )
        expected_allocation_id = (
            placement.get("expected_allocation_id") if placement else 0
        )
        expected_slurm_job_id = (
            placement.get("expected_slurm_job_id") if placement else ""
        )
        expected_account_name = (
            placement.get("expected_account_name") if placement else ""
        )
        strict_terminal = _strict_node_task_evidence(
            snapshot,
            task_id=submission["task_id"],
            task_name=submission["task_name"],
            dedupe_key=submission["dedupe_key"],
            node_name=plan_contract["requested_node_name"],
            same_node_as_task_id=same_node_id,
            expected_allocation_id=expected_allocation_id,
            expected_slurm_job_id=expected_slurm_job_id,
            expected_account_name=expected_account_name,
        )
        if (
            strict_terminal["allocation_id"] != evidence["allocation_id"]
            or strict_terminal["actual_node_name"]
            != evidence["actual_node_name"]
            or strict_terminal["slurm_job_id"] != evidence["slurm_job_id"]
        ):
            raise HandoffContractError(
                "diagnostic terminal strict-node evidence drifted"
            )
    return evidence


def _selected_candidate_identity(
    selected: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "source_task_payload_sha256": selected["task_identity"][
            "payload_sha256"
        ],
        "source_result_sha256": selected["source_result"]["sha256"],
        "seed": production._integer(
            selected["task_identity"]["seed"],
            "selected candidate seed",
        ),
        "fixed_primary_turns": production._integer(
            selected["task_identity"]["fixed_primary_turns"],
            "selected candidate primary turns",
        ),
    }


def _truth_evidence(
    *,
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    plan: Mapping[str, Any],
    submission: Mapping[str, Any],
    receipt: Mapping[str, Any],
    temperatures: Mapping[str, Any],
) -> dict[str, Any]:
    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        production._finite(value, "actual exterior dimension")
        for value in dimensions
    )
    raw_llt = production._finite(result.get("Llt"), "actual Standard Llt")
    physical_llt = raw_llt * 2.0
    losses = {
        name: production._finite(result.get(name), f"actual {name}")
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    actual_identity_values = dict(result)
    actual_identity_values["thermal_pad_conductivity_W_mK"] = 0.2
    actual_fixed_identity = attest_fixed_identity(actual_identity_values)
    all_temperatures = {
        name: production._finite(
            result.get(name), f"actual temperature {name}"
        )
        for name in GOAL_TEMPERATURE_TARGETS
        if name in result and not pd.isna(result.get(name))
    }
    return {
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "canonical_physical_params_sha256": selected["row_contract"][
            "canonical_physical_params_sha256"
        ],
        "fea_params_sha256": selected["row_contract"][
            "fea_params_sha256"
        ],
        "source_result_sha256": selected["source_result"]["sha256"],
        "source_task_payload_sha256": selected["task_identity"][
            "payload_sha256"
        ],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "scheduler_task_id": submission["task_id"],
        "actual_volume_L": float(volume_l),
        "actual_exterior_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_loss_components_W": losses,
        "actual_total_loss_W": sum(losses.values()),
        "actual_symmetric_Llt_uH": raw_llt,
        "actual_physical_Llt_uH": physical_llt,
        "actual_resonance_Hz": production._finite(
            result.get("f_res_min_tx_rx_only_Hz"),
            "actual resonance",
        ),
        "actual_temperature_targets_C": all_temperatures,
        "actual_active_temperature_targets_C": {
            name: production._finite(
                evidence["actual_C"], f"actual temperature {name}"
            )
            for name, evidence in temperatures.items()
        },
        "temperature_targets_not_reported": sorted(
            set(GOAL_TEMPERATURE_TARGETS) - set(all_temperatures)
        ),
        "selected_fixed_identity_attestation": copy.deepcopy(
            selected["row_contract"]["fixed_identity_attestation"]
        ),
        "actual_fixed_identity_attestation": actual_fixed_identity,
        "retained_symmetric_aedt": {
            "path": receipt["artifact_path"],
            "sha256": receipt["artifact_sha256"],
            "size_bytes": receipt["artifact_size_bytes"],
        },
        "retained_symmetric_aedtresults": {
            "path": receipt["results_path"],
            "manifest_path": receipt["results_manifest_path"],
            "manifest_sha256": receipt["results_manifest_sha256"],
            "tree_sha256": receipt["results_tree_sha256"],
            "file_count": receipt["results_file_count"],
            "size_bytes": receipt["results_size_bytes"],
        },
        **_diagnostic_flags(),
    }


def collect_standard(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str = DIAGNOSTIC_SCHEDULER_URL,
    scheduler: Any = scheduler_client,
    remote_reader: Any = production._remote_bytes,
    manifest_reader: Any = _remote_manifest_bytes,
    task_reader: Any = _scheduler_task_snapshot,
) -> Path:
    plan, params, selected = _load_plan(plan_path)
    submission = _load_submission(submission_path, plan=plan)
    normalized_scheduler_url = scheduler_url.rstrip("/")
    if normalized_scheduler_url != submission["scheduler_url"]:
        raise HandoffContractError(
            "diagnostic collection Scheduler origin differs from submission"
        )
    status = scheduler.get_status(
        int(submission["task_id"]),
        scheduler_url=scheduler_url,
    )
    if status != "completed":
        raise HandoffContractError(
            "diagnostic Standard task is not completed: "
            f"status={status!r}"
        )
    task_execution = _task_execution_evidence(
        task_reader(
            scheduler_url=scheduler_url,
            task_id=int(submission["task_id"]),
        ),
        submission=submission,
    )
    profile = production._read_json(
        plan_path.resolve(strict=True).parent / plan["profile"]["path"]
    )
    fetched = scheduler.fetch_result(
        int(submission["task_id"]),
        expected_revision=plan["solver_revision"],
        expected_library_revision=plan["library_revision"],
        expected_profile=profile["param_overrides"],
        scheduler_url=scheduler_url,
    )
    if (
        fetched.state != scheduler.RESULT_VALID
        or not isinstance(fetched.result, dict)
    ):
        raise HandoffContractError(
            "diagnostic Standard Scheduler result is not valid"
        )
    result = copy.deepcopy(fetched.result)
    _result_identity_valid(
        result=result,
        plan=plan,
        params=params,
        submission=submission,
        profile=profile,
        scheduler=scheduler,
    )
    (
        remote_receipt,
        marker,
        results_manifest,
        results_manifest_sha,
    ) = _validated_remote_bundle(
        submission=submission,
        result=result,
        scheduler_url=scheduler_url,
        remote_reader=remote_reader,
        manifest_reader=manifest_reader,
    )
    reasons = production._goal_result_reasons(result, selected)
    active, temperatures, body_probe_pass = (
        production._temperature_gate_evidence(result)
    )
    truth = _truth_evidence(
        result=result,
        selected=selected,
        plan=plan,
        submission=submission,
        receipt=remote_receipt,
        temperatures=temperatures,
    )
    result_sha = canonical_sha256(result)
    collection = production._seal(
        {
            "schema_version": COLLECTION_SCHEMA,
            "stage": "standard",
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission": production._file_record(submission_path),
            "submission_payload_sha256": submission["payload_sha256"],
            "selection_manifest": selected["selection_source"][
                "selection_manifest"
            ],
            "source_result": selected["source_result"],
            "task_payload": selected["task_payload"],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "selected_candidate_identity": (
                _selected_candidate_identity(selected, plan)
            ),
            "task_id": submission["task_id"],
            "scheduler_url": submission["scheduler_url"],
            "scheduler_status": status,
            "scheduler_task_execution": task_execution,
            "result": result,
            "result_sha256": result_sha,
            "result_identity": {
                "project_name": result["project_name"],
                "solver_revision": plan["solver_revision"],
                "library_revision": plan["library_revision"],
                "effective_params_sha256": submission[
                    "effective_params_sha256"
                ],
                "full_model": 0,
                "thermal_symmetry": "eighth",
                "solver_core_auth_sha256": result[
                    "solver_core_auth_sha256"
                ],
                "solver_num_cores_effective": 8,
            },
            "remote_aedt_bundle_receipt": remote_receipt,
            "aedtresults_manifest": results_manifest,
            "aedtresults_manifest_sha256": results_manifest_sha,
            "prune_protection_marker": marker,
            "prune_protection_marker_verified": True,
            "active_temperature_targets": active,
            "actual_body_probe_temperatures": temperatures,
            "actual_body_probe_temperature_gate_passed": body_probe_pass,
            "goal_physical_spec_reasons": reasons,
            "goal_physical_spec_passed": not reasons,
            "truth_evidence": truth,
            "diagnostic_truth_observation_available": True,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "retention_required": True,
            "prune_protection_required": True,
            **_diagnostic_flags(),
        }
    )
    return production._write_immutable_json(output.resolve(), collection)


def _load_collection(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = path.resolve(strict=True)
    value = production._validate_seal(
        production._read_json(resolved), COLLECTION_SCHEMA
    )
    if set(value) != COLLECTION_FIELDS:
        raise HandoffContractError(
            "diagnostic collection fields drifted"
        )
    if any(
        value.get(name) is not expected
        for name, expected in _diagnostic_flags().items()
    ):
        raise HandoffContractError(
            "diagnostic collection no-promotion flags drifted"
        )
    plan_record = value.get("plan")
    submission_record = value.get("submission")
    if not isinstance(plan_record, dict) or not isinstance(
        submission_record, dict
    ):
        raise HandoffContractError(
            "diagnostic collection references are absent"
        )
    plan_path = Path(str(plan_record.get("path") or ""))
    submission_path = Path(str(submission_record.get("path") or ""))
    if (
        production._file_record(plan_path) != plan_record
        or production._file_record(submission_path) != submission_record
    ):
        raise HandoffContractError(
            "diagnostic collection reference bytes drifted"
        )
    plan, params, selected = _load_plan(plan_path)
    submission = _load_submission(submission_path, plan=plan)
    refreshed = _fresh_selection_reauthentication(
        plan=plan, selected=selected, predictor=predictor
    )
    if refreshed != submission["search_authority_reauthentication"]:
        raise HandoffContractError(
            "diagnostic collection source reauthentication drifted"
        )
    result = value.get("result")
    if not isinstance(result, dict):
        raise HandoffContractError(
            "diagnostic collection result is absent"
        )
    profile = production._read_json(
        plan_path.parent / plan["profile"]["path"]
    )
    _result_identity_valid(
        result=result,
        plan=plan,
        params=params,
        submission=submission,
        profile=profile,
    )
    result_sha = canonical_sha256(result)
    receipt = _validate_bundle_receipt(
        value.get("remote_aedt_bundle_receipt"),
        submission=submission,
        result=result,
    )
    _validated_results_manifest = _validate_results_manifest(
        value.get("aedtresults_manifest"),
        receipt=receipt,
        result=result,
    )
    marker = production._validate_marker_payload(
        value.get("prune_protection_marker"),
        expected=submission["retained_aedt_bundle"],
    )
    marker_bytes = production._json_bytes(marker)
    reasons = production._goal_result_reasons(result, selected)
    active, temperatures, body_probe_pass = (
        production._temperature_gate_evidence(result)
    )
    task_execution = _task_execution_evidence(
        value.get("scheduler_task_execution") or {},
        submission=submission,
    )
    truth = _truth_evidence(
        result=result,
        selected=selected,
        plan=plan,
        submission=submission,
        receipt=receipt,
        temperatures=temperatures,
    )
    expected_result_identity = {
        "project_name": result["project_name"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "effective_params_sha256": submission[
            "effective_params_sha256"
        ],
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "solver_core_auth_sha256": result["solver_core_auth_sha256"],
        "solver_num_cores_effective": 8,
    }
    if (
        value.get("stage") != "standard"
        or value.get("plan_payload_sha256") != plan["payload_sha256"]
        or value.get("submission_payload_sha256")
        != submission["payload_sha256"]
        or value.get("selection_manifest")
        != selected["selection_source"]["selection_manifest"]
        or value.get("source_result") != selected["source_result"]
        or value.get("task_payload") != selected["task_payload"]
        or value.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or value.get("selected_candidate_identity")
        != _selected_candidate_identity(selected, plan)
        or value.get("task_id") != submission["task_id"]
        or value.get("scheduler_url") != submission["scheduler_url"]
        or value.get("scheduler_status") != "completed"
        or value.get("scheduler_task_execution") != task_execution
        or value.get("result_sha256") != result_sha
        or value.get("result_identity") != expected_result_identity
        or value.get("aedtresults_manifest_sha256")
        != receipt["results_manifest_sha256"]
        or production._sha256_bytes(marker_bytes)
        != receipt["marker_sha256"]
        or value.get("prune_protection_marker_verified") is not True
        or value.get("active_temperature_targets") != active
        or value.get("actual_body_probe_temperatures") != temperatures
        or value.get("actual_body_probe_temperature_gate_passed")
        is not body_probe_pass
        or value.get("goal_physical_spec_reasons") != reasons
        or value.get("goal_physical_spec_passed") is not (not reasons)
        or value.get("truth_evidence") != truth
        or value.get("diagnostic_truth_observation_available") is not True
        or value.get("scheduler_get_only_collection") is not True
        or value.get("scheduler_mutation_performed") is not False
        or value.get("retention_required") is not True
        or value.get("prune_protection_required") is not True
    ):
        raise HandoffContractError(
            "diagnostic collection evidence drifted"
        )
    return value, plan, params, selected, submission


def authenticate_collection(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Return a stable view over a fully reauthenticated collection."""
    collection, plan, params, selected, submission = _load_collection(
        path, predictor=predictor
    )
    return {
        "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
        "collection": collection,
        "plan": plan,
        "params": params,
        "selected": selected,
        "submission": submission,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticated diagnostic-only Standard FEA truth probes"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--bundle-manifest", type=Path, required=True)
    select.add_argument("--generation", type=Path, required=True)
    selection_source = select.add_mutually_exclusive_group(required=True)
    selection_source.add_argument(
        "--aggregate-manifest",
        type=Path,
        help="final aggregate manifest over every authenticated seed",
    )
    selection_source.add_argument(
        "--source-result",
        type=Path,
        action="append",
        help="repeat for arbitrary sealed terminal result.json inputs",
    )
    select.add_argument(
        "--limit",
        dest="count",
        type=int,
        default=8,
        help="bounded independent Standard probes (default 8, maximum 12)",
    )
    select.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--selection-manifest", type=Path, required=True)
    plan.add_argument("--candidate-physics-sha256", required=True)
    plan.add_argument("--solver-revision", required=True)
    plan.add_argument("--library-revision", required=True)
    plan.add_argument("--output", type=Path, required=True)

    retry_plan = commands.add_parser("plan-timeout-retry")
    retry_plan.add_argument("--original-plan", type=Path, required=True)
    retry_plan.add_argument(
        "--original-submission", type=Path, required=True
    )
    retry_plan.add_argument(
        "--scheduler-url", default=DIAGNOSTIC_SCHEDULER_URL
    )
    retry_plan.add_argument(
        "--strict-node-name",
        default="",
        help=(
            "opt into Scheduler fail-closed exact-node placement and a "
            "distinct timeout-strict-r2 identity"
        ),
    )
    retry_plan.add_argument("--output", type=Path, required=True)

    submit = commands.add_parser("submit-standard")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument(
        "--scheduler-cutover-receipt",
        type=Path,
        required=True,
    )
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--output", type=Path, required=True)

    retry_submit = commands.add_parser("submit-timeout-retry")
    retry_submit.add_argument("--plan", type=Path, required=True)
    retry_submit.add_argument(
        "--scheduler-cutover-receipt",
        type=Path,
        required=True,
    )
    retry_submit.add_argument("--priority", type=int, default=0)
    retry_submit.add_argument(
        "--same-node-as-task-id", type=int, default=0
    )
    retry_submit.add_argument(
        "--expected-allocation-id", type=int, default=0
    )
    retry_submit.add_argument("--expected-slurm-job-id", default="")
    retry_submit.add_argument("--expected-account-name", default="")
    retry_submit.add_argument("--expected-node-name", default="")
    retry_submit.add_argument("--output", type=Path, required=True)

    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--submission", type=Path, required=True)
    collect.add_argument(
        "--scheduler-url", default=DIAGNOSTIC_SCHEDULER_URL
    )
    collect.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-collection")
    validate.add_argument("--collection", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        result = create_selection(
            bundle_manifest_path=args.bundle_manifest,
            generation_path=args.generation,
            result_paths=args.source_result,
            aggregate_manifest_path=args.aggregate_manifest,
            count=args.count,
            output=args.output,
        )
    elif args.command == "plan":
        result = create_plan(
            selection_manifest_path=args.selection_manifest,
            candidate_physics_sha256=args.candidate_physics_sha256,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            output=args.output,
        )
    elif args.command == "plan-timeout-retry":
        result = create_timeout_retry_plan(
            original_plan_path=args.original_plan,
            original_submission_path=args.original_submission,
            scheduler_url=args.scheduler_url,
            strict_node_name=args.strict_node_name,
            output=args.output,
        )
    elif args.command == "submit-standard":
        result = submit_standard(
            plan_path=args.plan,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            priority=args.priority,
            output=args.output,
        )
    elif args.command == "submit-timeout-retry":
        result = submit_timeout_retry(
            plan_path=args.plan,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            priority=args.priority,
            same_node_as_task_id=args.same_node_as_task_id,
            expected_allocation_id=args.expected_allocation_id,
            expected_slurm_job_id=args.expected_slurm_job_id,
            expected_account_name=args.expected_account_name,
            expected_node_name=args.expected_node_name,
            output=args.output,
        )
    elif args.command == "collect":
        result = collect_standard(
            plan_path=args.plan,
            submission_path=args.submission,
            scheduler_url=args.scheduler_url,
            output=args.output,
        )
    else:
        view = authenticate_collection(args.collection)
        collection = view["collection"]
        result = args.collection.resolve(strict=True)
        if collection.get("schema_version") != COLLECTION_SCHEMA:
            raise HandoffContractError(
                "diagnostic collection schema drifted after validation"
            )
    print(json.dumps({"status": "ok", "path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
