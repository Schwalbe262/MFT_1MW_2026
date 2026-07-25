"""Exact-once provisional Full precompute for logical candidate 96230.

This is an intentionally separate, diagnostic-only hedge.  It does not alter
the canonical Standard-to-Full truth gate, and its output can never be
promoted automatically.  ``dry-run`` performs GET/read-only admission checks;
``submit`` is the only command that can issue one Scheduler POST.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Callable, Mapping, Sequence

import paramiko


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
from tools import mft_goal_full_fastlane_watcher as fastlane  # noqa: E402
from tools import mft_goal_safe_refill as safe_refill  # noqa: E402
from tools import mft_goal_startup_retry as startup_retry  # noqa: E402
from tools import mft_goal_timeout12h_retry as timeout12h  # noqa: E402


PLAN_SCHEMA = "mft-goal-provisional-full-precompute-plan-v1"
GATE_SCHEMA = "mft-goal-provisional-full-precompute-admission-v1"
POST_INTENT_SCHEMA = "mft-goal-provisional-full-precompute-post-intent-v1"
POST_RESULT_SCHEMA = "mft-goal-provisional-full-precompute-post-result-v1"
SUBMISSION_SCHEMA = "mft-goal-provisional-full-precompute-submission-v1"
CLAIM_GENERATION = "provisional-full-precompute-v1"
CAMPAIGN_ID = "mft-goal-20260726"
SCHEDULER_URL = "http://127.0.0.1:8002"
SCHEDULER_PROJECT = scheduler_client.MFT_PROJECT
ACCOUNT_NAME = "r1jae262"
NODE_NAME = "n114"
LOGICAL_AUTHORITY_TASK_ID = 96230
SOURCE_STANDARD_TASK_ID = 96304
SCHEDULER_CUTOVER_SHA256 = (
    "45cf323de69a9a9c926c528d905ee4d09715b1a3d40c68349b9fa511646486bc"
)
TARGET_CANDIDATE_SHA256 = (
    "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
)
COMPARISON_CANDIDATE_SHA256 = (
    "2a1bb6f2be79d5a9443538702b833e2b1918877a1137923c1a1ea0f606b46660"
)
TARGET_SOURCE_PLAN_PAYLOAD_SHA256 = (
    "d5eccb7c52ccb6642000a8adc9bd0f791abdb1692bb514686534c8b8639c32f6"
)
COMPARISON_SOURCE_PLAN_PAYLOAD_SHA256 = (
    "08748c42a0306c6cace6aebd91ff83e186ac0c190ef9d2a212447b26aae9d2d6"
)
SOLVER_REVISION = "a1e4f70cefa1af04673c73a6131bf490c0cc14b5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
RESOURCES = {"cpus": 16, "memory_mb": 98304, "timeout_seconds": 43200}
TARGET_FINISH_KST = datetime.fromisoformat("2026-07-26T18:00:00+09:00")
# Thirty minutes of explicit launch/retention slack precedes the mathematical
# 06:00 KST latest start for a 12-hour timeout.
SUBMISSION_CUTOFF_KST = datetime.fromisoformat("2026-07-26T05:30:00+09:00")
ENROOT_PATH = "/enroot"
ENROOT_MINIMUM_FREE_KIB = 200 * 1024 * 1024
GPFS_POST_RETENTION_FLOOR_GIB = 10.0
RETENTION_METADATA_RESERVE_BYTES = 64 * 1024 * 1024
MAX_EXPECTED_AEDT_BYTES = scheduler_client.RETAINED_AEDT_MAX_BYTES
MAX_POST_RECONCILIATION_READS = 8
DEFAULT_CLAIM_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "provisional_full_precompute_claims_v1"
)

HandoffContractError = production.HandoffContractError


def _flags() -> dict[str, bool]:
    return {
        "provisional": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
    }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).astimezone(timezone.utc).isoformat(timespec="microseconds")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HandoffContractError(f"{label} is invalid")
    return value


def _task_id(value: Mapping[str, Any]) -> int:
    observed = [value[key] for key in ("task_id", "id") if value.get(key) is not None]
    if not observed or any(item != observed[0] for item in observed):
        raise HandoffContractError("Scheduler task IDs are absent or disagree")
    return _positive_int(observed[0], "Scheduler task ID")


def _retention_storage_bound(expected_aedt_bytes: int) -> dict[str, Any]:
    expected = _positive_int(expected_aedt_bytes, "expected Full AEDT bytes")
    if expected > MAX_EXPECTED_AEDT_BYTES:
        raise HandoffContractError("expected Full AEDT exceeds transport maximum")
    raw_chunk = scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
    chunk_count = math.ceil(expected / raw_chunk)
    encoded = 4 * math.ceil(expected / 3)
    total = expected + encoded + RETENTION_METADATA_RESERVE_BYTES
    return {
        "schema_version": "mft-goal-project-only-aedt-storage-bound-v1",
        "scope": "retained_full_aedt_and_base64_chunks_only",
        "results_directory_included": False,
        "expected_full_aedt_bytes": expected,
        "raw_chunk_bytes": raw_chunk,
        "expected_chunk_count": chunk_count,
        "base64_encoded_bytes_upper_bound": encoded,
        "metadata_reserve_bytes": RETENTION_METADATA_RESERVE_BYTES,
        "gpfs_retention_bound_bytes": total,
        "gpfs_retention_bound_gib": total / (1024**3),
        "minimum_post_retention_free_floor_gib": (GPFS_POST_RETENTION_FLOOR_GIB),
    }


def _positive_violation_sum(row_contract: Mapping[str, Any]) -> float:
    values = row_contract.get("normalized_G")
    if not isinstance(values, Mapping):
        raise HandoffContractError("candidate normalized violations are absent")
    result = sum(max(0.0, float(value)) for value in values.values())
    if not math.isfinite(result):
        raise HandoffContractError("candidate violation sum is non-finite")
    return result


def _selection_summary(
    plan: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    row = selected.get("selected_row")
    contract = selected.get("row_contract")
    if not isinstance(row, Mapping) or not isinstance(contract, Mapping):
        raise HandoffContractError("source selected-candidate row is absent")
    dimensions = contract.get("exterior_dimensions_mm")
    normalized = contract.get("normalized_G")
    if not isinstance(dimensions, Mapping) or not isinstance(normalized, Mapping):
        raise HandoffContractError("candidate geometry/constraint evidence is absent")
    temperatures = {
        key: float(value)
        for key, value in normalized.items()
        if str(key).startswith("temperature_robust_limit:")
    }
    summary = {
        "logical_authority_task_id": plan["retry_of_timeout12h"][
            "logical_authority_task_id"
        ],
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "diagnostic_selection_rank": row["diagnostic_selection_rank"],
        "objective_volume_L": row["objective_volume_L"],
        "objective_total_loss_W": row["objective_total_loss_W"],
        "normalized_positive_violation_sum": _positive_violation_sum(contract),
        "exterior_dimensions_mm": copy.deepcopy(dict(dimensions)),
        "surrogate_mean_Llt_in_band": contract["surrogate_mean_Llt_in_band"],
        "goal_non_Llt_hard_spec_passed": contract["goal_non_Llt_hard_spec_passed"],
        "surrogate_robust_feasible": contract["surrogate_robust_feasible"],
        "normalized_half_magnetizing_resonance_margin": float(
            normalized["half_magnetizing_resonance_minimum"]
        ),
        "normalized_temperature_margins": temperatures,
    }
    return summary


def _load_reviewed_sources(
    target_path: Path, comparison_path: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    target, params, target_selected, _target_parent = timeout12h._load_plan(target_path)
    comparison, _comparison_params, comparison_selected, _comparison_parent = (
        timeout12h._load_plan(comparison_path)
    )
    target_record = target["retry_of_timeout12h"]
    comparison_record = comparison["retry_of_timeout12h"]
    if (
        target.get("payload_sha256") != TARGET_SOURCE_PLAN_PAYLOAD_SHA256
        or target.get("candidate_physics_sha256") != TARGET_CANDIDATE_SHA256
        or target_record.get("logical_authority_task_id") != LOGICAL_AUTHORITY_TASK_ID
        or target_record.get("retry_generation") != "timeout12h-r4"
        or comparison.get("payload_sha256") != COMPARISON_SOURCE_PLAN_PAYLOAD_SHA256
        or comparison.get("candidate_physics_sha256") != COMPARISON_CANDIDATE_SHA256
        or comparison_record.get("logical_authority_task_id") != 96223
        or comparison_record.get("retry_generation") != "timeout12h-r3"
        or target.get("solver_revision") != SOLVER_REVISION
        or target.get("library_revision") != LIBRARY_REVISION
        or comparison.get("solver_revision") != SOLVER_REVISION
        or comparison.get("library_revision") != LIBRARY_REVISION
    ):
        raise HandoffContractError("reviewed provisional Full source drifted")
    return target, params, target_selected, comparison, comparison_selected


def _selection_rationale(
    *,
    target_plan: Mapping[str, Any],
    target_selected: Mapping[str, Any],
    comparison_plan: Mapping[str, Any],
    comparison_selected: Mapping[str, Any],
) -> dict[str, Any]:
    target = _selection_summary(target_plan, target_selected)
    comparison = _selection_summary(comparison_plan, comparison_selected)
    dimensions = target["exterior_dimensions_mm"]
    expected_target = {
        "diagnostic_selection_rank": 1,
        "objective_volume_L": 803.9135169656241,
        "objective_total_loss_W": 5291.746327152702,
        "normalized_positive_violation_sum": 2.245223135539715,
        "exterior_dimensions_mm": {
            "W": 1186.354,
            "L": 970.8220000000001,
            "H": 698.0,
        },
    }
    expected_comparison = {
        "diagnostic_selection_rank": 9,
        "objective_volume_L": 860.394184155,
        "objective_total_loss_W": 5496.370392297698,
        "normalized_positive_violation_sum": 3.7557855470303148,
    }
    for key, expected in expected_target.items():
        observed = target[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12):
                raise HandoffContractError(f"target rationale {key} drifted")
        elif observed != expected:
            raise HandoffContractError(f"target rationale {key} drifted")
    for key, expected in expected_comparison.items():
        observed = comparison[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12):
                raise HandoffContractError(f"comparison rationale {key} drifted")
        elif observed != expected:
            raise HandoffContractError(f"comparison rationale {key} drifted")
    if (
        float(dimensions["W"]) > 1200
        or float(dimensions["L"]) > 1000
        or float(dimensions["H"]) > 750
        or target["surrogate_mean_Llt_in_band"] is not True
        or target["goal_non_Llt_hard_spec_passed"] is not True
        or target["surrogate_robust_feasible"] is not False
        or target["normalized_half_magnetizing_resonance_margin"] > 0
        or not target["normalized_temperature_margins"]
        or any(
            margin > 0 for margin in target["normalized_temperature_margins"].values()
        )
    ):
        raise HandoffContractError("target provisional rationale gates drifted")
    return {
        "selection_policy": (
            "prefer_rank1_lower_volume_lower_loss_lower_positive_violation_"
            "sum_over_rank9_comparison"
        ),
        "target": target,
        "comparison": comparison,
        "all_target_dimensions_within_limits": True,
        "mean_Llt_and_non_Llt_hard_spec_pass": True,
        "half_resonance_and_temperature_normalized_margins_nonpositive": True,
        "robust_Llt_constraint_passed": False,
        "provisional_diagnostic_only_required": True,
        "source_actual_standard": {
            "task_id": SOURCE_STANDARD_TASK_ID,
            "logical_authority_task_id": LOGICAL_AUTHORITY_TASK_ID,
            "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
            "review_observation": "furthest_in_Icepak_thermal",
            "authenticated_standard_pass_gate_present": False,
        },
    }


def _task_identity() -> tuple[str, str]:
    stem = TARGET_CANDIDATE_SHA256[:12]
    return (
        f"mft-goal-provisional-full-l{LOGICAL_AUTHORITY_TASK_ID}-{stem}-v1",
        f"mft_goal_provisional_full_l{LOGICAL_AUTHORITY_TASK_ID}_{stem}_v1",
    )


def _active_standard_storage_authorities(
    refill_plan: Mapping[str, Any],
    startup_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates = refill_plan.get("candidates")
    if not isinstance(candidates, list):
        raise HandoffContractError("parallel Standard storage authority is absent")
    authorities = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise HandoffContractError(
                "parallel Standard storage authority is malformed"
            )
        logical = item.get("logical_authority_task_id")
        grid_bytes = item.get("fresh_grid_output_bytes")
        if (
            logical not in {96223, 96224, 96230}
            or isinstance(grid_bytes, bool)
            or not isinstance(grid_bytes, int)
            or grid_bytes <= 0
            or item.get("account_name") != ACCOUNT_NAME
        ):
            raise HandoffContractError("parallel Standard storage authority drifted")
        authorities.append(
            {
                "logical_authority_task_id": logical,
                "candidate_physics_sha256": item["candidate_physics_sha256"],
                "task_name": item["task_name"],
                "dedupe_key": item["dedupe_key"],
                "account_name": ACCOUNT_NAME,
                "fresh_grid_output_bytes": grid_bytes,
                "fresh_grid_output_gib": grid_bytes / (1024**3),
                "bound_source": ("authenticated_timeout_native_premesh_grid_bytes"),
            }
        )
    startup = startup_plan.get("startup_successor")
    stage = startup_plan.get("stage")
    if (
        not isinstance(startup, Mapping)
        or not isinstance(stage, Mapping)
        or startup.get("logical_authority_task_id") != 96223
        or startup.get("prospective_grid_gib") != 19.15191717632115
        or startup_plan.get("candidate_physics_sha256") != COMPARISON_CANDIDATE_SHA256
        or stage.get("task_name")
        != "mft-goal-diag-standard-startup-r1-l96223-2a1bb6f2be79"
        or not isinstance(stage.get("retained_aedt_bundle"), Mapping)
    ):
        raise HandoffContractError("startup successor storage authority drifted")
    startup_bytes = next(
        item["fresh_grid_output_bytes"]
        for item in authorities
        if item["logical_authority_task_id"] == 96223
    )
    if not math.isclose(
        startup_bytes / (1024**3),
        float(startup["prospective_grid_gib"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise HandoffContractError(
            "startup successor storage bytes disagree with source grid"
        )
    authorities.append(
        {
            "logical_authority_task_id": 96223,
            "candidate_physics_sha256": COMPARISON_CANDIDATE_SHA256,
            "task_name": stage["task_name"],
            "dedupe_key": stage["retained_aedt_bundle"]["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "fresh_grid_output_bytes": startup_bytes,
            "fresh_grid_output_gib": startup_bytes / (1024**3),
            "bound_source": ("authenticated_startup_successor_source_grid_bytes"),
        }
    )
    authorities.sort(key=lambda item: item["logical_authority_task_id"])
    if [item["logical_authority_task_id"] for item in authorities] != [
        96223,
        96223,
        96224,
        96230,
    ]:
        raise HandoffContractError(
            "parallel Standard storage authority allowlist is incomplete"
        )
    return authorities


def initialize_plan(
    *,
    output_root: Path,
    target_source_plan: Path,
    comparison_source_plan: Path,
    scheduler_cutover_receipt: Path,
    active_storage_authority_plan: Path,
    startup_successor_plan: Path,
    expected_full_aedt_bytes: int,
    claim_root: Path,
    ssh_host: str,
    ssh_port: int,
    ssh_private_key: Path,
    ssh_known_hosts: Path,
    ssh_gpfs_path: str,
) -> Path:
    root = output_root.resolve()
    path = root / "provisional_full_precompute_plan.json"
    if path.exists():
        existing, _params, _profile = load_plan(path)
        requested_ssh = {
            "host": str(ssh_host).strip(),
            "port": _positive_int(ssh_port, "SSH port"),
            "username": ACCOUNT_NAME,
            "gpfs_path": str(ssh_gpfs_path).strip(),
            "private_key": production._file_record(ssh_private_key),
            "known_hosts": production._file_record(ssh_known_hosts),
        }
        requested_claim_root = claim_root.resolve(strict=True)
        if (
            existing["target_source_plan"]
            != production._file_record(target_source_plan)
            or existing["comparison_source_plan"]
            != production._file_record(comparison_source_plan)
            or existing["scheduler_cutover_receipt"]
            != production._file_record(scheduler_cutover_receipt)
            or existing["active_storage_authority_plan"]
            != production._file_record(active_storage_authority_plan)
            or existing["startup_successor_plan"]
            != production._file_record(startup_successor_plan)
            or existing["retention_storage_bound"]
            != _retention_storage_bound(expected_full_aedt_bytes)
            or existing["ssh_storage_authority"] != requested_ssh
            or Path(existing["claim_root_authority"]["resolved_root"]).resolve(
                strict=True
            )
            != requested_claim_root
        ):
            raise HandoffContractError(
                "existing immutable provisional Full plan differs"
            )
        return path
    (
        target,
        params,
        target_selected,
        comparison,
        comparison_selected,
    ) = _load_reviewed_sources(target_source_plan, comparison_source_plan)
    rationale = _selection_rationale(
        target_plan=target,
        target_selected=target_selected,
        comparison_plan=comparison,
        comparison_selected=comparison_selected,
    )
    profile, profile_record = production._profile_content("full")
    name, workdir = _task_identity()
    retained = scheduler_client.retained_aedt_identity(
        name, params, profile, SOLVER_REVISION, LIBRARY_REVISION
    )
    if (
        not isinstance(retained, Mapping)
        or retained.get("schema_version") != scheduler_client.RETAINED_AEDT_SCHEMA
        or retained.get("stage") != "full"
        or retained.get("artifact_path", "").endswith("/full.aedt") is not True
        or "results_path" in retained
        or "results_manifest_path" in retained
    ):
        raise HandoffContractError(
            "reviewed Full profile is not project-only retained AEDT"
        )
    storage_bound = _retention_storage_bound(expected_full_aedt_bytes)
    cutover_record = production._file_record(scheduler_cutover_receipt)
    if cutover_record["sha256"] != SCHEDULER_CUTOVER_SHA256:
        raise HandoffContractError("reviewed Scheduler cutover receipt drifted")
    refill_plan = safe_refill.load_plan(active_storage_authority_plan)
    startup_plan = startup_retry.load_plan(startup_successor_plan)[0]
    active_storage_authorities = _active_standard_storage_authorities(
        refill_plan, startup_plan
    )
    ssh_authority = {
        "host": str(ssh_host).strip(),
        "port": _positive_int(ssh_port, "SSH port"),
        "username": ACCOUNT_NAME,
        "gpfs_path": str(ssh_gpfs_path).strip(),
        "private_key": production._file_record(ssh_private_key),
        "known_hosts": production._file_record(ssh_known_hosts),
    }
    if not ssh_authority["host"] or not ssh_authority["gpfs_path"]:
        raise HandoffContractError("SSH storage authority is incomplete")
    claim_authority_sha = canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
            "logical_authority_task_id": LOGICAL_AUTHORITY_TASK_ID,
            "claim_generation": CLAIM_GENERATION,
            "profile_sha256": production.canonical_sha256(profile),
            "resources": RESOURCES,
            "task_name": name,
            "retained_dedupe_key": retained["dedupe_key"],
            "storage_bound": storage_bound,
            "flags": _flags(),
            "maximum_scheduler_posts": 1,
        }
    )
    authority = atomic_claim.initialize_claim_root(
        claim_root,
        campaign_id=CAMPAIGN_ID,
        campaign_authority_sha256=claim_authority_sha,
    )
    reference = atomic_claim.build_claim_reference(
        authority,
        candidate_physics_sha256=TARGET_CANDIDATE_SHA256,
        logical_authority_task_id=LOGICAL_AUTHORITY_TASK_ID,
        retry_generation=CLAIM_GENERATION,
    )
    plan = production._seal(
        {
            "schema_version": PLAN_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "goal_hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (GOAL_TEMPERATURE_CONTRACT_SHA256),
            "output_root": str(root),
            "target_source_plan": production._file_record(target_source_plan),
            "target_source_plan_payload_sha256": target["payload_sha256"],
            "comparison_source_plan": production._file_record(comparison_source_plan),
            "comparison_source_plan_payload_sha256": comparison["payload_sha256"],
            "scheduler_cutover_receipt": cutover_record,
            "active_storage_authority_plan": production._file_record(
                active_storage_authority_plan
            ),
            "startup_successor_plan": production._file_record(startup_successor_plan),
            "active_standard_storage_authorities": (active_storage_authorities),
            "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
            "logical_authority_task_id": LOGICAL_AUTHORITY_TASK_ID,
            "source_actual_standard_task_id": SOURCE_STANDARD_TASK_ID,
            "selection_rationale": rationale,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
            "profile": profile_record,
            "profile_canonical_sha256": production.canonical_sha256(profile),
            "source_fea_params_sha256": canonical_sha256(params),
            "effective_full_params_sha256": canonical_sha256(
                production._effective_params(params, profile)
            ),
            "stage": {
                "name": "full",
                "task_name": name,
                "workdir": workdir,
                "resources": copy.deepcopy(RESOURCES),
                "full_model": 1,
                "thermal_symmetry": "full",
                "aedt_backend": "standalone",
                "requested_account_name": ACCOUNT_NAME,
                "requested_node_name": NODE_NAME,
                "node_name_policy": "strict",
                "retained_aedt": copy.deepcopy(dict(retained)),
            },
            "fixed_boundary": {
                "fan_velocity_m_s": 1.5,
                "fan_config": "dual",
                "thermal_pad_conductivity_W_mK": 0.2,
                "core_plate_pad_t_mm": 2.0,
                "wcp_pad_t_mm": 2.0,
                "mutable": False,
            },
            "retention_storage_bound": storage_bound,
            "solver_transient_storage": {
                "path": ENROOT_PATH,
                "filesystem_type": "xfs",
                "minimum_free_kib": ENROOT_MINIMUM_FREE_KIB,
                "submission_time_fresh_probe_required": True,
                "task_start_fresh_probe_required": True,
                "gpfs_fallback_allowed": False,
                "scratch_environment_after_workdir_selection": {
                    "ANS_TEMP_PATH": "$MFT_WORKDIR",
                    "TMPDIR": "$MFT_WORKDIR",
                    "TMP": "$MFT_WORKDIR",
                    "TEMP": "$MFT_WORKDIR",
                },
                "ANS_MW_INHERIT_TMP_present": False,
            },
            "scheduler_url": SCHEDULER_URL,
            "scheduler_project": SCHEDULER_PROJECT,
            "scheduler_repository_modified": False,
            "scheduler_cancel_allowed": False,
            "maximum_scheduler_posts_lifetime": 1,
            "live_submission_default": False,
            "target_finish_kst": TARGET_FINISH_KST.isoformat(),
            "submission_cutoff_kst": SUBMISSION_CUTOFF_KST.isoformat(),
            "cutoff_safety_margin_seconds": 1800,
            "fresh_admission_required": [
                "active_tasks",
                "capacity_16cpu_98304mb_r1_n114",
                "license_16core",
                "gpfs_project_only_retention",
                "enroot_xfs_200gib",
            ],
            "canonical_truth_gate_modified": False,
            "canonical_claim_modified": False,
            "canonical_promotion_requires_authenticated_standard_and_full": True,
            "claim_root_authority": authority,
            "claim_reference": reference,
            "ssh_storage_authority": ssh_authority,
            "created_at_utc": _stamp(),
            **_flags(),
        }
    )
    if path.exists():
        raise HandoffContractError(
            "provisional Full plan appeared during initialization"
        )
    return production._write_immutable_json(path, plan)


def load_plan(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    plan = production._validate_seal(production._read_json(resolved), PLAN_SCHEMA)
    if (
        resolved
        != Path(plan.get("output_root", "")).resolve()
        / "provisional_full_precompute_plan.json"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("candidate_physics_sha256") != TARGET_CANDIDATE_SHA256
        or plan.get("logical_authority_task_id") != LOGICAL_AUTHORITY_TASK_ID
        or plan.get("source_actual_standard_task_id") != SOURCE_STANDARD_TASK_ID
        or plan.get("solver_revision") != SOLVER_REVISION
        or plan.get("library_revision") != LIBRARY_REVISION
        or plan.get("scheduler_url") != SCHEDULER_URL
        or plan.get("scheduler_project") != SCHEDULER_PROJECT
        or plan.get("scheduler_cutover_receipt", {}).get("sha256")
        != SCHEDULER_CUTOVER_SHA256
        or plan.get("scheduler_cancel_allowed") is not False
        or plan.get("maximum_scheduler_posts_lifetime") != 1
        or plan.get("live_submission_default") is not False
        or plan.get("canonical_truth_gate_modified") is not False
        or plan.get("canonical_claim_modified") is not False
        or any(plan.get(key) is not value for key, value in _flags().items())
    ):
        raise HandoffContractError("provisional Full plan contract drifted")
    (
        target,
        params,
        target_selected,
        comparison,
        comparison_selected,
    ) = _load_reviewed_sources(
        Path(plan["target_source_plan"]["path"]),
        Path(plan["comparison_source_plan"]["path"]),
    )
    if (
        production._file_record(Path(plan["target_source_plan"]["path"]))
        != plan["target_source_plan"]
        or production._file_record(Path(plan["comparison_source_plan"]["path"]))
        != plan["comparison_source_plan"]
        or _selection_rationale(
            target_plan=target,
            target_selected=target_selected,
            comparison_plan=comparison,
            comparison_selected=comparison_selected,
        )
        != plan["selection_rationale"]
    ):
        raise HandoffContractError("provisional Full source bytes drifted")
    if (
        production._file_record(Path(plan["scheduler_cutover_receipt"]["path"]))
        != plan["scheduler_cutover_receipt"]
    ):
        raise HandoffContractError("Scheduler cutover receipt bytes drifted")
    active_authority_path = Path(plan["active_storage_authority_plan"]["path"])
    startup_authority_path = Path(plan["startup_successor_plan"]["path"])
    if (
        production._file_record(active_authority_path)
        != plan["active_storage_authority_plan"]
        or production._file_record(startup_authority_path)
        != plan["startup_successor_plan"]
        or _active_standard_storage_authorities(
            safe_refill.load_plan(active_authority_path),
            startup_retry.load_plan(startup_authority_path)[0],
        )
        != plan.get("active_standard_storage_authorities")
    ):
        raise HandoffContractError("parallel Standard storage authority bytes drifted")
    profile, record = production._profile_content("full")
    name, workdir = _task_identity()
    retained = scheduler_client.retained_aedt_identity(
        name, params, profile, SOLVER_REVISION, LIBRARY_REVISION
    )
    stage = plan.get("stage")
    if (
        record != plan.get("profile")
        or plan.get("profile_canonical_sha256") != production.canonical_sha256(profile)
        or plan.get("source_fea_params_sha256") != canonical_sha256(params)
        or plan.get("effective_full_params_sha256")
        != canonical_sha256(production._effective_params(params, profile))
        or not isinstance(stage, Mapping)
        or stage.get("task_name") != name
        or stage.get("workdir") != workdir
        or stage.get("resources") != RESOURCES
        or stage.get("full_model") != 1
        or stage.get("thermal_symmetry") != "full"
        or stage.get("retained_aedt") != retained
        or retained.get("schema_version") != scheduler_client.RETAINED_AEDT_SCHEMA
        or "results_path" in retained
    ):
        raise HandoffContractError("provisional Full execution identity drifted")
    bound = plan.get("retention_storage_bound")
    if not isinstance(bound, Mapping) or dict(bound) != _retention_storage_bound(
        bound.get("expected_full_aedt_bytes")
    ):
        raise HandoffContractError("project-only storage bound drifted")
    for name in ("private_key", "known_hosts"):
        ssh_record = plan["ssh_storage_authority"][name]
        if production._file_record(Path(ssh_record["path"])) != ssh_record:
            raise HandoffContractError(f"SSH {name} bytes drifted")
    authority = atomic_claim.load_claim_root(
        Path(plan["claim_root_authority"]["resolved_root"]),
        expected_authority=plan["claim_root_authority"],
    )
    atomic_claim.validate_claim_reference(plan["claim_reference"], authority)
    return plan, params, profile


def _normalize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "name",
        "status",
        "state",
        "project",
        "dedupe_key",
        "cpus",
        "memory_mb",
        "timeout_seconds",
        "aedt_backend",
        "account_name",
        "requested_account_name",
        "actual_node_name",
        "allocation_node_name",
        "requested_node_name",
        "requested_node_name_policy",
        "node_name_policy",
        "same_node_as_task_id",
        "allocation_id",
        "slurm_job_id",
        "remote_dir",
    )
    result = {field: task.get(field) for field in fields}
    result["task_id"] = _task_id(task)
    result["requested_node_name_policy"] = task.get(
        "requested_node_name_policy", task.get("node_name_policy")
    )
    result["same_node_as_task_id"] = task.get("same_node_as_task_id", 0)
    return result


def _source_standard_evidence(
    task: Mapping[str, Any], *, target_plan: Mapping[str, Any]
) -> dict[str, Any]:
    row = _normalize_task(task)
    source_stage = target_plan["stage"]
    if (
        row["task_id"] != SOURCE_STANDARD_TASK_ID
        or row["name"] != source_stage["task_name"]
        or row["dedupe_key"] != source_stage["retained_aedt_bundle"]["dedupe_key"]
        or row["project"] != SCHEDULER_PROJECT
        or row["cpus"] != 8
        or row["memory_mb"] != 32768
        or row["timeout_seconds"] != 43200
        or row["aedt_backend"] != "standalone"
        or row["requested_account_name"] != ACCOUNT_NAME
        or row["requested_node_name"] != NODE_NAME
        or row["requested_node_name_policy"] != "strict"
    ):
        raise HandoffContractError("actual Standard source task 96304 drifted")
    return row


def _capacity_endpoint() -> str:
    return fastlane._capacity_endpoint(ACCOUNT_NAME)


def _exact_capacity(
    *,
    live_reader: Callable[..., Any],
    scheduler_url: str,
) -> dict[str, Any]:
    value = fastlane._require_account_capacity(
        scheduler_url=scheduler_url,
        account_name=ACCOUNT_NAME,
        live_reader=live_reader,
    )
    allocations = value.get("allocations")
    exact = [
        dict(item)
        for item in allocations
        if isinstance(item, Mapping)
        and item.get("account_name") == ACCOUNT_NAME
        and (
            item.get("node_name") == NODE_NAME
            or item.get("actual_node_name") == NODE_NAME
            or item.get("allocation_node_name") == NODE_NAME
        )
        and isinstance(item.get("free_cpus"), int)
        and item["free_cpus"] >= RESOURCES["cpus"]
        and isinstance(item.get("free_memory_mb"), int)
        and item["free_memory_mb"] >= RESOURCES["memory_mb"]
        and re.fullmatch(r"[0-9]+", str(item.get("slurm_job_id") or ""))
    ]
    if not exact:
        raise HandoffContractError(
            "fresh capacity has no exact ready r1/n114 allocation"
        )
    exact.sort(key=lambda item: int(str(item["slurm_job_id"])))
    return {
        "response": value,
        "selected_allocation": exact[0],
        "capacity_endpoint": _capacity_endpoint(),
    }


def _ssh_client(authority: Mapping[str, Any]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_host_keys(authority["known_hosts"]["path"])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        hostname=authority["host"],
        port=authority["port"],
        username=authority["username"],
        key_filename=authority["private_key"]["path"],
        allow_agent=False,
        look_for_keys=False,
        timeout=20,
        auth_timeout=20,
        banner_timeout=20,
    )
    return client


def _fresh_enroot(
    plan: Mapping[str, Any], *, allocation: Mapping[str, Any]
) -> dict[str, Any]:
    authority = plan["ssh_storage_authority"]
    job_id = str(allocation.get("slurm_job_id") or "")
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise HandoffContractError("n114 capacity has no Slurm allocation ID")
    probe = (
        "set -eu; "
        "node=$(hostname -s); fs=$(findmnt -n -o FSTYPE -T /enroot); "
        "free=$(df -Pk /enroot | awk 'NR==2 {print $4}'); "
        'printf "__NODE__:%s\\n__FS__:%s\\n__FREE_KIB__:%s\\n" '
        '"$node" "$fs" "$free"'
    )
    command = (
        f"srun --jobid={job_id} --overlap --nodes=1 --ntasks=1 "
        f"--cpus-per-task=1 --nodelist={NODE_NAME} --unbuffered "
        f"bash -lc {shlex.quote(probe)}"
    )
    client = _ssh_client(authority)
    try:
        _stdin, stdout, stderr = client.exec_command(command, timeout=60)
        output = stdout.read().decode("utf-8", "strict")
        error = stderr.read().decode("utf-8", "replace")
        exit_code = stdout.channel.recv_exit_status()
    except Exception as exc:
        raise HandoffContractError("fresh n114 /enroot probe failed") from exc
    finally:
        client.close()
    if exit_code != 0:
        raise HandoffContractError(
            f"fresh n114 /enroot probe exit={exit_code}: {error[:200]}"
        )
    parsed = {}
    for line in output.splitlines():
        if line.startswith("__") and ":" in line:
            key, value = line.split(":", 1)
            parsed[key] = value.strip()
    try:
        free_kib = int(parsed["__FREE_KIB__"])
    except (KeyError, ValueError) as exc:
        raise HandoffContractError("fresh /enroot free-space value is absent") from exc
    if (
        parsed.get("__NODE__") != NODE_NAME
        or parsed.get("__FS__") != "xfs"
        or free_kib < ENROOT_MINIMUM_FREE_KIB
    ):
        raise HandoffContractError("fresh n114 /enroot XFS 200GiB gate failed")
    return {
        "observed_at_utc": _stamp(),
        "node_name": NODE_NAME,
        "slurm_job_id": job_id,
        "path": ENROOT_PATH,
        "filesystem_type": "xfs",
        "free_kib": free_kib,
        "minimum_free_kib": ENROOT_MINIMUM_FREE_KIB,
        "passed": True,
        "probe_read_only": True,
    }


def _active_storage_reservation(
    plan: Mapping[str, Any], active_tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    known = plan.get("active_standard_storage_authorities")
    if not isinstance(known, list):
        raise HandoffContractError("active Standard storage authorities are absent")
    reservations = []
    conflicts = []
    for raw in active_tasks:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("account_name") != ACCOUNT_NAME:
            continue
        is_fea = raw.get("aedt_backend") in {"standalone", "pooled"} or str(
            raw.get("required_capability") or ""
        ).startswith("conda:pyaedt")
        if not is_fea:
            continue
        match = next(
            (
                item
                for item in known
                if raw.get("name") == item["task_name"]
                and raw.get("dedupe_key") == item["dedupe_key"]
            ),
            None,
        )
        if match is None:
            conflicts.append(_normalize_task(raw))
            continue
        reservations.append(
            {
                "task_id": _task_id(raw),
                "task_name": raw["name"],
                "dedupe_key": raw["dedupe_key"],
                "logical_authority_task_id": match["logical_authority_task_id"],
                "storage_bound_bytes": match["fresh_grid_output_bytes"],
                "storage_bound_gib": match["fresh_grid_output_gib"],
                "bound_source": match["bound_source"],
            }
        )
    if conflicts:
        raise HandoffContractError(
            "active r1 standalone task lacks authenticated storage bound"
        )
    reservations.sort(key=lambda item: item["task_id"])
    total = sum(item["storage_bound_bytes"] for item in reservations)
    return {
        "active_bounded_task_count": len(reservations),
        "active_task_storage_reservations": reservations,
        "active_storage_bound_bytes": total,
        "active_storage_bound_gib": total / (1024**3),
        "active_unbounded_storage_conflict_count": 0,
    }


def _fresh_gpfs(
    plan: Mapping[str, Any],
    *,
    active_tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    compatibility_plan = {
        "ssh_storage_authority": {
            **copy.deepcopy(plan["ssh_storage_authority"]),
            "storage_path": plan["ssh_storage_authority"]["gpfs_path"],
        }
    }
    evidence = safe_refill._fresh_storage(compatibility_plan)
    bound = plan["retention_storage_bound"]
    active = _active_storage_reservation(plan, active_tasks)
    available = float(evidence["limiting_quota"]["effective_free_gb"])
    required = (
        float(active["active_storage_bound_gib"])
        + float(bound["gpfs_retention_bound_gib"])
        + GPFS_POST_RETENTION_FLOOR_GIB
    )
    if available < required:
        raise HandoffContractError(
            "fresh GPFS space does not cover project-only AEDT/chunks bound"
        )
    return {
        **evidence,
        **active,
        "project_only_retention_bound_gib": bound["gpfs_retention_bound_gib"],
        "required_free_including_floor_gib": required,
        "free_after_active_and_retention_gib": (
            available
            - float(active["active_storage_bound_gib"])
            - float(bound["gpfs_retention_bound_gib"])
        ),
        "results_directory_reserved": False,
        "passed": True,
    }


def _default_active_tasks(*, scheduler_url: str) -> list[dict[str, Any]]:
    return fastlane._default_task_list_reader(scheduler_url=scheduler_url)


def fresh_gates(
    *,
    plan_path: Path,
    license_snapshot_path: Path,
    now: datetime | None = None,
    live_reader: Callable[..., Any] = fastlane._default_live_reader,
    active_task_reader: Callable[..., list[dict[str, Any]]] = (_default_active_tasks),
    task_reader: Callable[..., Mapping[str, Any]] = (
        diagnostic._scheduler_task_snapshot
    ),
    gpfs_reader: Callable[..., dict[str, Any]] = _fresh_gpfs,
    enroot_reader: Callable[..., dict[str, Any]] = _fresh_enroot,
    validate_cutover: bool = True,
) -> dict[str, Any]:
    plan, _params, _profile = load_plan(plan_path)
    current = (now or _now()).astimezone(TARGET_FINISH_KST.tzinfo)
    if current > SUBMISSION_CUTOFF_KST:
        raise HandoffContractError(
            "provisional Full submission cutoff 05:30 KST has passed"
        )
    if current + timedelta(seconds=RESOURCES["timeout_seconds"]) > (
        TARGET_FINISH_KST - timedelta(seconds=1800)
    ):
        raise HandoffContractError("12-hour Full cannot retain the safety margin")
    target_plan = timeout12h._load_plan(Path(plan["target_source_plan"]["path"]))[0]
    if validate_cutover:
        diagnostic._validate_scheduler_cutover_receipt(
            Path(plan["scheduler_cutover_receipt"]["path"]),
            verify_live_launcher=True,
            require_strict_node=True,
            strict_node_contract=target_plan["scheduler_strict_node_contract"],
            require_active_strict=True,
        )
    admission = diagnostic._live_scheduler_admission_snapshot(
        scheduler_url=plan["scheduler_url"], reader=live_reader
    )
    active = active_task_reader(scheduler_url=plan["scheduler_url"])
    if not isinstance(active, list):
        raise HandoffContractError("fresh active task inventory is absent")
    source = _source_standard_evidence(
        task_reader(
            scheduler_url=plan["scheduler_url"],
            task_id=SOURCE_STANDARD_TASK_ID,
        ),
        target_plan=target_plan,
    )
    capacity = _exact_capacity(
        live_reader=live_reader, scheduler_url=plan["scheduler_url"]
    )
    license_path = license_snapshot_path.resolve(strict=True)
    _snapshot, license_sha = production._validate_license_snapshot(
        license_path, now=(now or _now())
    )
    gpfs = gpfs_reader(plan, active_tasks=active)
    enroot = enroot_reader(plan, allocation=capacity["selected_allocation"])
    result = production._seal(
        {
            "schema_version": GATE_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "captured_at_utc": _stamp(now),
            "submission_cutoff_kst": SUBMISSION_CUTOFF_KST.isoformat(),
            "target_finish_kst": TARGET_FINISH_KST.isoformat(),
            "scheduler_admission": admission,
            "active_project_tasks": copy.deepcopy(active),
            "active_project_task_inventory_sha256": canonical_sha256(active),
            "source_actual_standard_task": source,
            "capacity": capacity,
            "license_snapshot": production._file_record(license_path),
            "license_snapshot_sha256": license_sha,
            "runtime_license_refresh_required": True,
            "gpfs": gpfs,
            "enroot": enroot,
            "all_scheduler_reads_get_only": True,
            "fresh_capacity_passed": True,
            "fresh_license_passed": True,
            "fresh_storage_passed": True,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": False,
            **_flags(),
        }
    )
    return result


def _sibling_inventory(rows: Any, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise HandoffContractError("provisional Full sibling inventory is absent")
    stage = plan["stage"]
    matches = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        if (
            raw.get("name") != stage["task_name"]
            and raw.get("dedupe_key") != stage["retained_aedt"]["dedupe_key"]
        ):
            continue
        row = _normalize_task(raw)
        if (
            row["name"] != stage["task_name"]
            or row["dedupe_key"] != stage["retained_aedt"]["dedupe_key"]
            or row["project"] != SCHEDULER_PROJECT
            or row["cpus"] != RESOURCES["cpus"]
            or row["memory_mb"] != RESOURCES["memory_mb"]
            or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
            or row["aedt_backend"] != "standalone"
            or row["requested_node_name"] != NODE_NAME
            or row["requested_node_name_policy"] != "strict"
            or row["same_node_as_task_id"] != 0
        ):
            raise HandoffContractError("provisional Full sibling identity collision")
        matches.append(row)
    matches.sort(key=lambda row: row["task_id"])
    if len(matches) > 1:
        raise HandoffContractError("more than one provisional Full sibling exists")
    unsigned = {
        "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
        "task_name": stage["task_name"],
        "dedupe_key": stage["retained_aedt"]["dedupe_key"],
        "matching_task_count": len(matches),
        "matching_tasks": matches,
    }
    return {
        **unsigned,
        "snapshot_sha256": canonical_sha256(unsigned),
    }


def _default_sibling_reader(
    *, scheduler_url: str, project: str, task_name: str
) -> list[dict[str, Any]]:
    return diagnostic._scheduler_project_tasks(
        scheduler_url=scheduler_url, project=project, task_name=task_name
    )


def _claim_winner(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "immediate_task_id": LOGICAL_AUTHORITY_TASK_ID,
        "immediate_retry_kind": "none",
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": production._sha256_file(plan_path.resolve(strict=True)),
        "profile_sha256": plan["profile_canonical_sha256"],
        "resources": copy.deepcopy(RESOURCES),
        "task_name": plan["stage"]["task_name"],
        "dedupe_key": plan["stage"]["retained_aedt"]["dedupe_key"],
    }


def _claim_task_evidence(
    task: Mapping[str, Any], pending: Mapping[str, Any]
) -> dict[str, Any]:
    row = _normalize_task(task)
    winner = pending.get("winner")
    if (
        not isinstance(winner, Mapping)
        or row["name"] != winner.get("task_name")
        or row["dedupe_key"] != winner.get("dedupe_key")
        or row["project"] != SCHEDULER_PROJECT
        or row["cpus"] != RESOURCES["cpus"]
        or row["memory_mb"] != RESOURCES["memory_mb"]
        or row["timeout_seconds"] != RESOURCES["timeout_seconds"]
        or row["aedt_backend"] != "standalone"
        or row["requested_account_name"] != ACCOUNT_NAME
        or row["requested_node_name"] != NODE_NAME
        or row["requested_node_name_policy"] != "strict"
    ):
        raise atomic_claim.ClaimContractError(
            "provisional Full claim task evidence drifted"
        )
    return row


def _post_intent(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    initial_admission_path: Path,
    locked_admission_path: Path,
    gates: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    return production._seal(
        {
            "schema_version": POST_INTENT_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
            "pending_claim_payload_sha256": pending["payload_sha256"],
            "initial_fresh_admission": production._file_record(initial_admission_path),
            "locked_fresh_admission": production._file_record(locked_admission_path),
            "fresh_admission_payload_sha256": gates["payload_sha256"],
            "selected_account_name": ACCOUNT_NAME,
            "requested_node_name": NODE_NAME,
            "retention_storage_bound": copy.deepcopy(plan["retention_storage_bound"]),
            "maximum_scheduler_posts_lifetime": 1,
            "scheduler_post_may_follow": True,
            "scheduler_cancel_allowed": False,
            "created_at_utc": _stamp(),
            **_flags(),
        }
    )


def _load_durable_admission(
    path: Path, *, plan_path: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    value = production._validate_seal(
        production._read_json(path.resolve(strict=True)), GATE_SCHEMA
    )
    if (
        value.get("plan") != production._file_record(plan_path)
        or value.get("plan_payload_sha256") != plan["payload_sha256"]
        or any(value.get(key) is not expected for key, expected in _flags().items())
    ):
        raise HandoffContractError("durable provisional Full admission drifted")
    return value


def _load_post_intent(
    path: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    pending: Mapping[str, Any],
    initial_admission_path: Path,
    locked_admission_path: Path,
) -> dict[str, Any]:
    value = production._validate_seal(
        production._read_json(path.resolve(strict=True)), POST_INTENT_SCHEMA
    )
    initial = _load_durable_admission(
        initial_admission_path, plan_path=plan_path, plan=plan
    )
    locked = _load_durable_admission(
        locked_admission_path, plan_path=plan_path, plan=plan
    )
    if (
        value.get("plan") != production._file_record(plan_path)
        or value.get("plan_payload_sha256") != plan["payload_sha256"]
        or value.get("candidate_physics_sha256") != TARGET_CANDIDATE_SHA256
        or value.get("pending_claim_payload_sha256") != pending["payload_sha256"]
        or value.get("initial_fresh_admission")
        != production._file_record(initial_admission_path)
        or value.get("locked_fresh_admission")
        != production._file_record(locked_admission_path)
        or value.get("fresh_admission_payload_sha256") != locked["payload_sha256"]
        or value.get("maximum_scheduler_posts_lifetime") != 1
        or value.get("scheduler_post_may_follow") is not True
        or value.get("scheduler_cancel_allowed") is not False
        or any(value.get(key) is not expected for key, expected in _flags().items())
    ):
        raise HandoffContractError("durable provisional Full POST intent drifted")
    return {"intent": value, "initial": initial, "locked": locked}


def submit(
    *,
    plan_path: Path,
    license_snapshot_path: Path,
    output: Path,
    priority: int = 100,
    scheduler: Any = scheduler_client,
    gate_reader: Callable[..., dict[str, Any]] = fresh_gates,
    sibling_reader: Callable[..., list[dict[str, Any]]] = (_default_sibling_reader),
    waiter: Callable[[], None] | None = None,
) -> Path:
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"provisional Full submission receipt exists: {target}"
        )
    plan, params, profile = load_plan(plan_path)
    root = Path(plan["output_root"])
    intent_path = root / "post_intent.json"
    initial_admission_path = root / "initial_admission.json"
    locked_admission_path = root / "locked_admission.json"
    post_result_path = root / "post_result.json"
    claim_directory = (
        Path(plan["claim_root_authority"]["resolved_root"])
        / plan["claim_reference"]["relative_claim_directory"]
    )
    recovery_state_present = (
        intent_path.exists()
        or initial_admission_path.exists()
        or locked_admission_path.exists()
        or (claim_directory / atomic_claim.PENDING_CLAIM_NAME).exists()
        or (claim_directory / atomic_claim.FINALIZED_CLAIM_NAME).exists()
    )
    initial_gates = None
    if not recovery_state_present:
        initial_gates = gate_reader(
            plan_path=plan_path, license_snapshot_path=license_snapshot_path
        )

    def read_siblings() -> dict[str, Any]:
        return _sibling_inventory(
            sibling_reader(
                scheduler_url=plan["scheduler_url"],
                project=plan["scheduler_project"],
                task_name=plan["stage"]["task_name"],
            ),
            plan=plan,
        )

    before = read_siblings()
    authority = atomic_claim.load_claim_root(
        Path(plan["claim_root_authority"]["resolved_root"]),
        expected_authority=plan["claim_root_authority"],
    )
    reference = atomic_claim.validate_claim_reference(
        plan["claim_reference"], authority
    )
    winner = _claim_winner(plan_path, plan)
    try:
        acquisition = atomic_claim.acquire_claim(
            Path(authority["resolved_root"]), reference, winner
        )
    except atomic_claim.ClaimContractError as exc:
        raise HandoffContractError(
            "provisional Full atomic claim acquisition failed"
        ) from exc
    status = acquisition["status"]
    pending = (
        acquisition["claim"]["pending_claim"]
        if status == "existing_finalized"
        else acquisition["claim"]
    )
    own = before["matching_tasks"]
    if status != "fresh_pending":
        if len(own) != 1:
            if intent_path.exists():
                _load_post_intent(
                    intent_path,
                    plan_path=plan_path,
                    plan=plan,
                    pending=pending,
                    initial_admission_path=initial_admission_path,
                    locked_admission_path=locked_admission_path,
                )
            raise HandoffContractError(
                "existing provisional Full claim has no unique sibling; "
                "manual reconciliation is required and re-POST is forbidden"
            )
        try:
            if status == "existing_pending":
                finalized = atomic_claim.recover_pending_claim(
                    Path(authority["resolved_root"]),
                    reference,
                    acquisition["claim"],
                    matching_tasks=own,
                    sibling_snapshot=before,
                    evidence_validator=_claim_task_evidence,
                )
            else:
                finalized = atomic_claim.validate_finalized_claim(
                    Path(authority["resolved_root"]),
                    reference,
                    claim=acquisition["claim"],
                    expected_winner=winner,
                )
                _claim_task_evidence(own[0], finalized["pending_claim"])
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "provisional Full claim recovery failed"
            ) from exc
        durable = _load_post_intent(
            intent_path,
            plan_path=plan_path,
            plan=plan,
            pending=pending,
            initial_admission_path=initial_admission_path,
            locked_admission_path=locked_admission_path,
        )
        submission_result = {
            "task_id": finalized["task_id"],
            "submission_source": "pre_submission_reconciliation",
            "scheduler_mutation_performed": False,
            "api_pre_submission_readback": own[0],
            "api_post_submission_response": None,
        }
        initial_gates = durable["initial"]
        locked_gates = durable["locked"]
        after = before
        post_count = 0
    else:
        if recovery_state_present or initial_gates is None:
            raise HandoffContractError(
                "fresh provisional Full claim collided with durable recovery state"
            )
        if own:
            raise HandoffContractError(
                "fresh provisional Full claim found an existing sibling"
            )
        production._write_immutable_json(initial_admission_path, initial_gates)
        locked_gates: dict[str, Any] | None = None
        guard_calls = 0

        def locked_guard() -> None:
            nonlocal locked_gates, guard_calls
            if read_siblings()["matching_task_count"] != 0:
                raise HandoffContractError(
                    "provisional Full locked sibling slot is not empty"
                )
            locked_gates = gate_reader(
                plan_path=plan_path,
                license_snapshot_path=license_snapshot_path,
            )
            production._write_immutable_json(locked_admission_path, locked_gates)
            production._write_immutable_json(
                intent_path,
                _post_intent(
                    plan_path=plan_path,
                    plan=plan,
                    initial_admission_path=initial_admission_path,
                    locked_admission_path=locked_admission_path,
                    gates=locked_gates,
                    pending=pending,
                ),
            )
            guard_calls += 1

        environment, _core_policy = production._submission_environment(
            stage="full",
            solver_revision=SOLVER_REVISION,
            license_snapshot_path=license_snapshot_path,
        )
        environment.update(
            {
                "ANS_TEMP_PATH": "$MFT_WORKDIR",
                "TMPDIR": "$MFT_WORKDIR",
                "TMP": "$MFT_WORKDIR",
                "TEMP": "$MFT_WORKDIR",
            }
        )
        if "ANS_MW_INHERIT_TMP" in environment:
            raise HandoffContractError("ANS_MW_INHERIT_TMP must remain omitted")
        submission_result = scheduler.submit_verification(
            plan["stage"]["task_name"],
            plan["stage"]["workdir"],
            params,
            profile,
            mem_mb=RESOURCES["memory_mb"],
            cpus=RESOURCES["cpus"],
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            priority=priority,
            aedt_backend="standalone",
            submission_env=environment,
            submission_env_after_workdir=True,
            required_workdir_prefix="/enroot/",
            required_project_cap=diagnostic.GOAL_FEA_PROJECT_CAP,
            max_project_active_tasks=diagnostic.GOAL_FEA_PROJECT_CAP,
            scheduler_url=plan["scheduler_url"],
            pre_submit_guard=locked_guard,
            account_name=ACCOUNT_NAME,
            node_name=NODE_NAME,
            node_name_policy="strict",
            return_submission_evidence=True,
        )
        if guard_calls != 1 or locked_gates is None:
            raise HandoffContractError(
                "provisional Full POST lacks exactly one locked fresh gate"
            )
        task_id = _positive_int(
            submission_result.get("task_id"),
            "provisional Full submitted task ID",
        )
        production._write_immutable_json(
            post_result_path,
            production._seal(
                {
                    "schema_version": POST_RESULT_SCHEMA,
                    "task_id": task_id,
                    "scheduler_post_count": 1,
                    "scheduler_submission_performed": True,
                    "created_at_utc": _stamp(),
                    **_flags(),
                }
            ),
        )
        wait = waiter or (lambda: None)
        after = None
        for attempt in range(MAX_POST_RECONCILIATION_READS):
            latest = read_siblings()
            if latest["matching_task_count"] == 1:
                after = latest
                break
            if attempt + 1 < MAX_POST_RECONCILIATION_READS:
                wait()
        if after is None:
            raise HandoffContractError(
                "provisional Full POST did not produce one durable sibling"
            )
        try:
            finalized = atomic_claim.finalize_claim(
                Path(authority["resolved_root"]),
                reference,
                acquisition["claim"],
                task_id=task_id,
                task_readback=after["matching_tasks"][0],
                sibling_snapshot=after,
                evidence_validator=_claim_task_evidence,
            )
        except atomic_claim.ClaimContractError as exc:
            raise HandoffContractError(
                "provisional Full claim finalization failed"
            ) from exc
        post_count = 1
    if not isinstance(submission_result, Mapping):
        raise HandoffContractError("Scheduler submission evidence is absent")
    task_id = _positive_int(
        submission_result.get("task_id"), "provisional Full task ID"
    )
    if after["matching_tasks"][0]["task_id"] != task_id:
        raise HandoffContractError("provisional Full task readback drifted")
    receipt = production._seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": TARGET_CANDIDATE_SHA256,
            "logical_authority_task_id": LOGICAL_AUTHORITY_TASK_ID,
            "source_actual_standard_task_id": SOURCE_STANDARD_TASK_ID,
            "task_id": task_id,
            "task_name": plan["stage"]["task_name"],
            "dedupe_key": plan["stage"]["retained_aedt"]["dedupe_key"],
            "resources": copy.deepcopy(RESOURCES),
            "full_model": 1,
            "thermal_symmetry": "full",
            "profile": copy.deepcopy(plan["profile"]),
            "retained_aedt": copy.deepcopy(plan["stage"]["retained_aedt"]),
            "retention_storage_bound": copy.deepcopy(plan["retention_storage_bound"]),
            "initial_fresh_admission": initial_gates,
            "locked_fresh_admission": locked_gates,
            "sibling_inventory_before": before,
            "sibling_inventory_after": after,
            "exact_sibling_count": 1,
            "atomic_claim_acquisition_status": status,
            "atomic_claim_finalized": finalized,
            "scheduler_post_count_this_run": post_count,
            "maximum_scheduler_posts_lifetime": 1,
            "scheduler_submission_performed": True,
            "scheduler_cancel_performed": False,
            "scheduler_repository_modified": False,
            "canonical_truth_gate_modified": False,
            "canonical_claim_modified": False,
            "canonical_promotion_requires_authenticated_standard_and_full": True,
            "created_at_utc": _stamp(),
            **_flags(),
        }
    )
    return production._write_immutable_json(target, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-plan")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--target-source-plan", type=Path, required=True)
    init.add_argument("--comparison-source-plan", type=Path, required=True)
    init.add_argument("--scheduler-cutover-receipt", type=Path, required=True)
    init.add_argument("--active-storage-authority-plan", type=Path, required=True)
    init.add_argument("--startup-successor-plan", type=Path, required=True)
    init.add_argument("--expected-full-aedt-bytes", type=int, required=True)
    init.add_argument("--claim-root", type=Path, default=DEFAULT_CLAIM_ROOT)
    init.add_argument("--ssh-host", required=True)
    init.add_argument("--ssh-port", type=int, default=22)
    init.add_argument("--ssh-private-key", type=Path, required=True)
    init.add_argument("--ssh-known-hosts", type=Path, required=True)
    init.add_argument("--ssh-gpfs-path", default="slurm_scheduler")
    dry = commands.add_parser("dry-run")
    dry.add_argument("--plan", type=Path, required=True)
    dry.add_argument("--license-snapshot", type=Path, required=True)
    live = commands.add_parser("submit")
    live.add_argument("--plan", type=Path, required=True)
    live.add_argument("--license-snapshot", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--priority", type=int, default=100)
    live.add_argument("--authorize-single-post", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-plan":
        path = initialize_plan(
            output_root=args.output_root,
            target_source_plan=args.target_source_plan,
            comparison_source_plan=args.comparison_source_plan,
            scheduler_cutover_receipt=args.scheduler_cutover_receipt,
            active_storage_authority_plan=args.active_storage_authority_plan,
            startup_successor_plan=args.startup_successor_plan,
            expected_full_aedt_bytes=args.expected_full_aedt_bytes,
            claim_root=args.claim_root,
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            ssh_private_key=args.ssh_private_key,
            ssh_known_hosts=args.ssh_known_hosts,
            ssh_gpfs_path=args.ssh_gpfs_path,
        )
        print(path)
        return 0
    if args.command == "dry-run":
        print(
            json.dumps(
                fresh_gates(
                    plan_path=args.plan,
                    license_snapshot_path=args.license_snapshot,
                ),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        return 0
    if not args.authorize_single_post:
        raise HandoffContractError("live submit requires --authorize-single-post")
    path = submit(
        plan_path=args.plan,
        license_snapshot_path=args.license_snapshot,
        output=args.output,
        priority=args.priority,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
