"""Prepare, but never submit, the single rounded candidate-5 Full lane.

The only scientific authority accepted by this module is the authenticated
terminal-success collection of rounded Standard task 96340.  The Full model
keeps that exact candidate geometry and changes only the reviewed execution
stage from eighth symmetry to Full:

* ``full_model=1`` and ``thermal_symmetry=full``;
* ``round_corner=1``, ``corner_radius=10 mm``, ``corner_segments=4``;
* dual 1.5 m/s fans, 0.2 W/(m K) TIM/insulation, and both 2 mm pads; and
* one retained ``full_model.aedt`` plus its native AEDT results manifest.

``dry-run`` derives and audits the payload with zero Scheduler I/O.  ``prepare``
performs GET-only capacity checks and seals a plan, but this file deliberately
has no POST implementation.  Scheduler remains an external service consumed
through its API; its repository is never imported or modified.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_official5_rounded_final_collector as rounded_collect  # noqa: E402
from tools import mft_goal_official5_rounded_final_prepare as rounded  # noqa: E402
from tools import mft_goal_postdeadline_standard_collector as standard_collect  # noqa: E402
from tools import mft_goal_standard_full_continuation as continuation  # noqa: E402


ContractError = rounded.PostdeadlineContractError
StrictLane = continuation.StrictLane

SOURCE_STANDARD_TASK_ID = 96_340
SOURCE_CANDIDATE_SHA256 = rounded.SOURCE_CANDIDATE_SHA256
EXPECTED_SOURCE_SOLVER_REVISION = (
    "623a5345ae1b85b974cb248bcb0c8dfaadf1a867"
)
CAMPAIGN_ID = rounded.CAMPAIGN_ID
PROJECT = rounded.PROJECT
SCHEDULER_URL = rounded.SCHEDULER_URL

CPUS = continuation.CPUS
MEMORY_MB = continuation.MEMORY_MB
MAX_WORKERS_PER_NODE = continuation.MAX_WORKERS_PER_NODE
SOLVER_TIMEOUT_SECONDS = continuation.SOLVER_TIMEOUT_SECONDS
SCHEDULER_TIMEOUT_SECONDS = continuation.SCHEDULER_TIMEOUT_SECONDS
KILL_GRACE_SECONDS = continuation.KILL_GRACE_SECONDS
PRIORITY = continuation.PRIORITY

TASK_NAME = (
    "mft-goal-final-full-official5-rounded-r10-s4-"
    f"from-t{SOURCE_STANDARD_TASK_ID}-{SOURCE_CANDIDATE_SHA256[:12]}-v1"
)
WORKDIR = (
    "mft_goal_final_full_official5_rounded_r10_s4_"
    f"from_t{SOURCE_STANDARD_TASK_ID}_{SOURCE_CANDIDATE_SHA256[:12]}_v1"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\final_full_official5_rounded_r10_s4_prepare_v1"
)

FULL_PROFILE_PATH = (
    REPOSITORY
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_truth_promotion_full_rounded_final.json"
)
FULL_PROFILE_SCHEMA = (
    "mft-goal-truth-promotion-full-rounded-final-profile-v1"
)
PLAN_NAME = "rounded_full_prepare_plan.json"
DRY_RUN_NAME = "rounded_full_dry_run.json"
PARAMS_NAME = "full_params.json"
PROFILE_NAME = "full_profile.json"
SOURCE_AUTHORITY_NAME = "rounded_standard_success_authority.json"

PLAN_SCHEMA = "mft-goal-official5-rounded-full-prepare-plan-v1"
DRY_RUN_SCHEMA = "mft-goal-official5-rounded-full-dry-run-v1"
SOURCE_AUTHORITY_SCHEMA = (
    "mft-goal-official5-rounded-standard-success-authority-v1"
)
OUTPUT_MANIFEST_SCHEMA = (
    "mft-goal-official5-rounded-full-output-contract-v1"
)
CHECKPOINT_SCHEMA = (
    "mft-goal-rounded-full-diagnostic-checkpoint-contract-v1"
)
CHECKPOINT_RECEIPT_SCHEMA = (
    "mft-goal-rounded-full-diagnostic-checkpoint-receipt-v1"
)
CHECKPOINT_MARKER_SCHEMA = (
    scheduler_client.SCHEDULER_PRESERVE_SCHEMA
)
MAX_LICENSE_SNAPSHOT_AGE_SECONDS = 180

FIXED_BOUNDARY = copy.deepcopy(rounded.FIXED_BOUNDARY)
ROUNDING_POLICY = copy.deepcopy(rounded.ROUNDING_POLICY)

canonical_bytes = rounded.canonical_bytes
payload_sha256 = rounded.payload_sha256
sealed = rounded.sealed
validate_seal = rounded.validate_seal
sha256_file = rounded.sha256_file
read_json = rounded.read_json
write_immutable_json = rounded.write_immutable_json


def _now(value: datetime | None = None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None:
        raise ContractError("Full prepare time must be timezone-aware")
    return result.astimezone(timezone.utc)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{label} is not finite") from exc
    if not math.isfinite(result):
        raise ContractError(f"{label} is not finite")
    return result


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("prepared Full artifact escaped its root") from exc
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_relative(
    root: Path, record: Any, label: str
) -> Path:
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise ContractError(f"{label} record is malformed")
    base = root.resolve(strict=True)
    path = (base / str(record["path"])).resolve(strict=True)
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ContractError(f"{label} escaped collection root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ContractError(f"{label} bytes drifted")
    return path


def _load_full_profile() -> dict[str, Any]:
    profile = read_json(FULL_PROFILE_PATH.resolve(strict=True))
    retention = profile.get("artifact_retention")
    expected_retention = {
        "schema_version": (
            scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_SCHEMA
        ),
        "stage": "full",
        "artifact_filename": "full_model.aedt",
        "results_directory": "full_model.aedtresults",
        "results_manifest_filename": (
            "full_model.aedtresults.manifest.json"
        ),
        "receipt_filename": "full_model.aedt.receipt.json",
        "marker_filename": scheduler_client.SCHEDULER_PRESERVE_MARKER,
        "retention_required": True,
        "prune_protection_required": True,
    }
    if (
        profile.get("schema_version") != FULL_PROFILE_SCHEMA
        or profile.get("stage") != "full"
        or profile.get("cpus") != CPUS
        or profile.get("mem_mb") != MEMORY_MB
        or profile.get("timeout_seconds") != 43_200
        or retention != expected_retention
        or profile.get("fixed_boundary_contract")
        != {
            **FIXED_BOUNDARY,
            "core_plate_on": 1,
            "wcp_on": 1,
        }
    ):
        raise ContractError("reviewed rounded Full profile drifted")
    return profile


def validate_license_freshness(
    path: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate the reviewed 16-core snapshot and bound its admission age."""

    resolved = path.resolve(strict=True)
    now = _now(observed_at)
    try:
        production._validate_license_snapshot(  # noqa: SLF001
            resolved, now=now
        )
    except Exception as exc:
        raise ContractError("Full 16-core license snapshot is invalid") from exc
    value = read_json(resolved)
    try:
        checked = datetime.fromisoformat(
            str(value.get("checked_at") or "").replace("Z", "+00:00")
        ).astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise ContractError("Full license timestamp is invalid") from exc
    age = (now - checked).total_seconds()
    if not 0 <= age <= MAX_LICENSE_SNAPSHOT_AGE_SECONDS:
        raise ContractError(
            f"Full license snapshot is stale or future-dated: age={age}"
        )
    return {
        "record": _file_record(resolved),
        "checked_at_utc": checked.isoformat(),
        "observed_at_utc": now.isoformat(),
        "age_seconds": age,
        "maximum_age_seconds": MAX_LICENSE_SNAPSHOT_AGE_SECONDS,
        "fresh": True,
    }


def full_inputs(
    standard_plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reauthenticate the rounded Standard plan and derive exact Full inputs."""

    standard_plan = rounded.load_plan(standard_plan_path)
    if (
        standard_plan.get("source_candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or standard_plan.get("solver_revision")
        != EXPECTED_SOURCE_SOLVER_REVISION
        or standard_plan.get("rounding_policy") != ROUNDING_POLICY
        or standard_plan.get("fixed_boundary") != FIXED_BOUNDARY
    ):
        raise ContractError("rounded Standard plan identity drifted")
    params_path = rounded._contained(  # noqa: SLF001
        standard_plan_path.resolve().parent,
        standard_plan.get("rounded_fea_params"),
        "rounded Standard params",
    )
    params = read_json(params_path)
    profile = _load_full_profile()
    effective = scheduler_client.effective_verification_params(
        params, profile
    )
    if set(effective) != set(ALL_INPUT_KEYS):
        raise ContractError("rounded Full profile changed parameter schema")
    observed = {
        "full_model": effective.get("full_model"),
        "thermal_symmetry": effective.get("thermal_symmetry"),
        "loss_sym_on": effective.get("loss_sym_on"),
        "round_corner": effective.get("round_corner"),
        "corner_radius": effective.get("corner_radius"),
        "corner_segments": effective.get("corner_segments"),
        "fan_config": effective.get("fan_config"),
        "fan_velocity": effective.get("fan_velocity"),
        "k_ins": effective.get("k_ins"),
        "core_plate_pad_t": effective.get("core_plate_pad_t"),
        "wcp_pad_t": effective.get("wcp_pad_t"),
        "core_plate_on": effective.get("core_plate_on"),
        "wcp_on": effective.get("wcp_on"),
        "keep_project": effective.get("keep_project"),
    }
    expected = {
        "full_model": 1,
        "thermal_symmetry": "full",
        "loss_sym_on": 0,
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "core_plate_on": 1,
        "wcp_on": 1,
        "keep_project": 1,
    }
    if observed != expected:
        raise ContractError(
            f"rounded Full effective contract drifted: {observed}"
        )
    return params, profile, standard_plan


def _source_collection_record(
    root: Path, record: Any, label: str
) -> Path:
    try:
        return _resolve_relative(root, record, label)
    except (OSError, ContractError) as exc:
        raise ContractError(f"{label} is not authenticated") from exc


def authenticate_standard_success(
    *,
    standard_plan_path: Path,
    standard_final_path: Path,
    standard_collection_root: Path,
) -> dict[str, Any]:
    """Require the exact task96340 rounded symmetric terminal success."""

    contract = rounded_collect.load_contract(
        plan_path=standard_plan_path,
        final_path=standard_final_path,
    )
    if (
        contract.get("task_id") != SOURCE_STANDARD_TASK_ID
        or contract.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or contract.get("solver_revision")
        != EXPECTED_SOURCE_SOLVER_REVISION
    ):
        raise ContractError("rounded Standard submission is not task96340")
    root = standard_collection_root.resolve(strict=True)
    receipt_path = (root / "collection_receipt.json").resolve(strict=True)
    seal_path = (root / "collection_seal.json").resolve(strict=True)
    receipt = standard_collect._validate_seal(  # noqa: SLF001
        standard_collect._read_json(  # noqa: SLF001
            receipt_path, "rounded Standard collection receipt"
        ),
        standard_collect.COLLECTION_SCHEMA,
        "rounded Standard collection receipt",
    )
    collection_seal = standard_collect._validate_seal(  # noqa: SLF001
        standard_collect._read_json(  # noqa: SLF001
            seal_path, "rounded Standard collection seal"
        ),
        standard_collect.COLLECTION_SEAL_SCHEMA,
        "rounded Standard collection seal",
    )
    receipt_record = _source_collection_record(
        root,
        collection_seal.get("collection_receipt"),
        "rounded Standard collection receipt",
    )
    if (
        receipt_record != receipt_path
        or receipt.get("task_id") != SOURCE_STANDARD_TASK_ID
        or receipt.get("task_name") != contract["task_name"]
        or receipt.get("dedupe_key") != contract["dedupe_key"]
        or receipt.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or receipt.get("source_plan_file_sha256")
        != contract["plan_file_sha256"]
        or receipt.get("source_submission_file_sha256")
        != contract["submission_file_sha256"]
        or receipt.get("scheduler_get_only_collection") is not True
        or receipt.get("scheduler_mutation_performed") is not False
    ):
        raise ContractError("rounded Standard collection lineage drifted")
    result_path = _source_collection_record(
        root, receipt.get("result_json"), "rounded Standard result"
    )
    result = read_json(result_path)
    if receipt.get("result_sha256") != payload_sha256(result):
        raise ContractError("rounded Standard result bytes drifted")
    artifact_path = _source_collection_record(
        root,
        receipt.get("retained_symmetric_aedt"),
        "rounded symmetric AEDT",
    )
    if artifact_path.name != "symmetric.aedt":
        raise ContractError("rounded Standard AEDT filename drifted")
    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ContractError("rounded Standard source files are absent")
    terminal_path = _source_collection_record(
        root,
        source_files.get("scheduler_terminal_task.json"),
        "rounded Standard terminal task",
    )
    terminal = read_json(terminal_path)
    expected_terminal = {
        "task_id": SOURCE_STANDARD_TASK_ID,
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": PROJECT,
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "actual_node_name": rounded.NODE_NAME,
        "requested_node_name_policy": "strict",
        "strict_node_placement": True,
        "placement_contract_satisfied": True,
    }
    terminal_drift = {
        key: {"expected": value, "actual": terminal.get(key)}
        for key, value in expected_terminal.items()
        if terminal.get(key) != value
    }
    if terminal_drift:
        raise ContractError(
            f"rounded Standard terminal identity drifted: {terminal_drift}"
        )

    params, standard_profile, _plan = rounded.derive_rounded_params_and_profile(
        rounded.direct.authenticate_source_authority()
    )
    effective = scheduler_client.effective_verification_params(
        params, standard_profile
    )
    if not scheduler_client.result_matches_params(
        result, effective, required_keys=set(ALL_INPUT_KEYS)
    ):
        raise ContractError("rounded Standard RESULT_JSON parameters drifted")
    if (
        _finite(result.get("full_model"), "Standard full_model") != 0.0
        or str(result.get("thermal_symmetry") or "").lower() != "eighth"
        or _finite(result.get("round_corner"), "round_corner") != 1.0
        or _finite(result.get("corner_radius"), "corner_radius") != 10.0
        or _finite(result.get("corner_segments"), "corner_segments") != 4.0
    ):
        raise ContractError("rounded Standard result model identity drifted")

    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        _finite(value, "actual exterior dimension") for value in dimensions
    )
    resonance_hz = _finite(
        result.get("f_res_min_tx_rx_only_Hz"), "actual resonance"
    )
    active_targets, temperature_evidence, temperature_passed = (
        production._temperature_gate_evidence(result)  # noqa: SLF001
    )
    hard_constraints = {
        "width_mm": {
            "actual": width,
            "limit": float(GOAL_SIZE_LIMITS_MM["W"]),
            "passed": width <= float(GOAL_SIZE_LIMITS_MM["W"]),
        },
        "length_mm": {
            "actual": length,
            "limit": float(GOAL_SIZE_LIMITS_MM["L"]),
            "passed": length <= float(GOAL_SIZE_LIMITS_MM["L"]),
        },
        "height_mm": {
            "actual": height,
            "limit": float(GOAL_SIZE_LIMITS_MM["H"]),
            "passed": height <= float(GOAL_SIZE_LIMITS_MM["H"]),
        },
        "resonance_Hz": {
            "actual": resonance_hz,
            "limit": float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
            "passed": resonance_hz
            >= float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
        },
        "body_temperatures": {
            "active_targets": active_targets,
            "evidence": temperature_evidence,
            "passed": temperature_passed,
        },
    }
    if not all(item["passed"] for item in hard_constraints.values()):
        raise ContractError(
            "task96340 rounded Standard actual constraints did not all pass"
        )
    fixed = {
        "fan_config": result.get("fan_config"),
        "fan_velocity_m_s": _finite(
            result.get("fan_velocity"), "fan_velocity"
        ),
        "thermal_pad_conductivity_W_mK": _finite(
            result.get("k_ins"), "k_ins"
        ),
        "core_plate_pad_t_mm": _finite(
            result.get("core_plate_pad_t"), "core_plate_pad_t"
        ),
        "wcp_pad_t_mm": _finite(
            result.get("wcp_pad_t"), "wcp_pad_t"
        ),
    }
    if fixed != FIXED_BOUNDARY:
        raise ContractError(
            f"task96340 fixed thermal conditions drifted: {fixed}"
        )
    return sealed(
        {
            "schema_version": SOURCE_AUTHORITY_SCHEMA,
            "authenticated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_task_id": SOURCE_STANDARD_TASK_ID,
            "source_task_name": contract["task_name"],
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "source_solver_revision": contract["solver_revision"],
            "source_library_revision": contract["library_revision"],
            "source_profile_sha256": contract["profile_sha256"],
            "source_parameter_digest": contract["parameter_digest"],
            "source_node_name": rounded.NODE_NAME,
            "source_plan": _file_record(standard_plan_path),
            "source_submission_final": _file_record(standard_final_path),
            "source_collection_receipt": _file_record(receipt_path),
            "source_collection_seal": _file_record(seal_path),
            "source_symmetric_aedt": _file_record(artifact_path),
            "source_terminal_task": terminal,
            "result_sha256": payload_sha256(result),
            "actual_volume_L": float(volume_l),
            "actual_dimensions_mm": {
                "W": width,
                "L": length,
                "H": height,
            },
            "actual_resonance_Hz": resonance_hz,
            "hard_constraint_evidence": hard_constraints,
            "measured_hard_constraints_passed": True,
            "fixed_thermal_boundary": fixed,
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "same_candidate_full_authorized": True,
        }
    )


def _validate_full_payload(
    *,
    payload: Mapping[str, Any],
    retained: Mapping[str, Any],
    core_evidence: Mapping[str, Any],
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    solver_revision: str,
    lane: StrictLane,
) -> None:
    effective = scheduler_client.effective_verification_params(
        dict(params), dict(profile)
    )
    if (
        effective.get("full_model") != 1
        or effective.get("thermal_symmetry") != "full"
        or effective.get("round_corner") != 1
        or effective.get("corner_radius") != 10.0
        or effective.get("corner_segments") != 4
    ):
        raise ContractError("Full payload effective physics drifted")
    expected = {
        "name": TASK_NAME,
        "project": PROJECT,
        "account_name": lane.account_name,
        "node_name": lane.node_name,
        "node_name_policy": "strict",
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    command = str(payload.get("command") or "")
    transport = retained.get("transport")
    checkpoint = diagnostic_checkpoint_contract(
        payload=payload,
        retained=retained,
        solver_revision=solver_revision,
    )
    if (
        drift
        or "same_node_as_task_id" in payload
        or "requested_allocation_id" in payload
        or command.count(
            "timeout --signal=TERM --kill-after="
            f"{KILL_GRACE_SECONDS}s {SOLVER_TIMEOUT_SECONDS}s "
            "python run_simulation_260706.py --fixed --thermal "
            "--headless --full --params cand.json"
        )
        != 1
        or command.count(
            "python run_simulation_260706.py --fixed --thermal "
            "--headless --full --model-only --params cand.json"
        )
        != 1
        or command.count("MFT_ROUNDED_FULL_CHECKPOINT_CREATED ") != 1
        or command.count(
            "python tools/mft_runtime_license_snapshot.py "
            '--output-dir "$MFT_WORKDIR/.goal-license-runtime" '
            f"--solver-revision {solver_revision}"
        )
        != 2
        or checkpoint["artifact_path"] not in command
        or checkpoint["receipt_path"] not in command
        or checkpoint["marker_path"] not in command
        or "--symmetry-thermal-direct-analyze" in command
        or solver_revision not in command
        or scheduler_client.RUNTIME_LICENSE_REFRESH_ENV not in command
        or retained.get("stage") != "full"
        or retained.get("solver_revision") != solver_revision
        or retained.get("dedupe_key") != payload.get("dedupe_key")
        or not str(retained.get("artifact_path") or "").endswith(
            "/full_model.aedt"
        )
        or not str(retained.get("results_path") or "").endswith(
            "/full_model.aedtresults"
        )
        or not str(retained.get("results_manifest_path") or "").endswith(
            "/full_model.aedtresults.manifest.json"
        )
        or not isinstance(transport, Mapping)
        or not str(transport.get("chunk_directory") or "").endswith(
            "/full_model.aedt.chunks"
        )
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
        or core_evidence.get("contract") != production.FULL_CORE_CONTRACT
        or core_evidence.get("requested_num_cores") != CPUS
        or core_evidence.get("runtime_license_refresh_required") is not True
    ):
        raise ContractError(
            f"rounded Full payload/retention contract drifted: {drift}"
        )


def diagnostic_checkpoint_contract(
    *,
    payload: Mapping[str, Any],
    retained: Mapping[str, Any],
    solver_revision: str,
) -> dict[str, Any]:
    """Derive the immutable pre-solve diagnostic checkpoint identity."""

    dedupe = str(payload.get("dedupe_key") or "")
    parameter_digest = str(retained.get("parameter_digest") or "")
    profile_sha256 = str(retained.get("profile_sha256") or "")
    library_revision = str(retained.get("library_revision") or "")
    if (
        not dedupe
        or not parameter_digest
        or not profile_sha256
        or retained.get("solver_revision") != solver_revision
        or not library_revision
    ):
        raise ContractError(
            "rounded Full checkpoint source identity is incomplete"
        )
    identity = payload_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "dedupe_key": dedupe,
            "source_standard_task_id": SOURCE_STANDARD_TASK_ID,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "profile_sha256": profile_sha256,
            "parameter_digest": parameter_digest,
            "rounding_policy": ROUNDING_POLICY,
            "fixed_boundary": FIXED_BOUNDARY,
        }
    )[:16]
    root = f"goal-fea-checkpoints/{identity}"
    marker_contract = {
        "schema": CHECKPOINT_MARKER_SCHEMA,
        "preserve": True,
        "owner": PROJECT,
        "reason": (
            "Diagnostic rounded Full geometry/setup checkpoint; "
            "not scientific or thermal PASS; "
            f"dedupe_key={dedupe}"
        ),
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "stage": "full_geometry_setup_pre_solve",
        "relative_directory": root,
        "artifact_path": (
            f"{root}/full_model_geometry_setup_checkpoint.aedt"
        ),
        "receipt_path": (
            f"{root}/full_model_geometry_setup_checkpoint.receipt.json"
        ),
        "marker_path": (
            f"{root}/{scheduler_client.SCHEDULER_PRESERVE_MARKER}"
        ),
        "chunk_directory": (
            f"{root}/full_model_geometry_setup_checkpoint.aedt.chunks"
        ),
        "transport_schema_version": (
            scheduler_client.RETAINED_AEDT_TEXT_CHUNK_SCHEMA
        ),
        "transport_encoding": "base64",
        "transport_raw_chunk_bytes": (
            scheduler_client.RETAINED_AEDT_RAW_CHUNK_BYTES
        ),
        "transport_max_encoded_chunk_bytes": (
            scheduler_client.RETAINED_AEDT_MAX_ENCODED_CHUNK_BYTES
        ),
        "receipt_schema_version": CHECKPOINT_RECEIPT_SCHEMA,
        "marker_contract": marker_contract,
        "marker_contract_sha256": payload_sha256(marker_contract),
        "source_standard_task_id": SOURCE_STANDARD_TASK_ID,
        "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "profile_sha256": profile_sha256,
        "parameter_digest": parameter_digest,
        "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "checkpoint_created_before_full_solve": True,
        "terminal_success_required_for_collection": False,
        "scientific_pass": False,
        "thermal_pass": False,
        "production_promotion_eligible": False,
        "diagnostic_only": True,
    }


def _checkpoint_python() -> str:
    return r"""
import base64
import datetime
import hashlib
import json
import os
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
artifact = pathlib.Path(sys.argv[2])
receipt = pathlib.Path(sys.argv[3])
marker = pathlib.Path(sys.argv[4])
chunks = pathlib.Path(sys.argv[5])
contract = json.loads(sys.argv[6])
if not src.is_file() or src.suffix.lower() != ".aedt":
    raise RuntimeError("rounded Full model-only checkpoint source is absent")
root = artifact.parent
if (
    receipt.parent != root
    or marker.parent != root
    or chunks.parent != root
    or root.exists()
):
    raise RuntimeError("rounded Full checkpoint destination is unsafe")
root.parent.mkdir(parents=True, exist_ok=True)
tmp = root.with_name("." + root.name + ".tmp")
if tmp.exists():
    raise RuntimeError("rounded Full checkpoint staging already exists")
tmp.mkdir()
tmp_artifact = tmp / artifact.name
tmp_receipt = tmp / receipt.name
tmp_marker = tmp / marker.name
tmp_chunks = tmp / chunks.name
tmp_chunks.mkdir()
digest = hashlib.sha256()
size = 0
count = 0
with src.open("rb") as source, tmp_artifact.open("xb") as target:
    while True:
        block = source.read(contract["transport_raw_chunk_bytes"])
        if not block:
            break
        target.write(block)
        digest.update(block)
        size += len(block)
        encoded = base64.b64encode(block)
        if len(encoded) > contract["transport_max_encoded_chunk_bytes"]:
            raise RuntimeError("rounded Full checkpoint chunk exceeds bound")
        (tmp_chunks / ("%08d.b64" % count)).write_bytes(encoded)
        count += 1
    target.flush()
    os.fsync(target.fileno())
if not 0 < size <= 68719476736 or count < 1:
    raise RuntimeError("rounded Full checkpoint size is outside bounds")
marker_payload = dict(contract["marker_contract"])
marker_payload["created_at"] = (
    datetime.datetime.now(datetime.timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z")
)
marker_data = (
    json.dumps(marker_payload, sort_keys=True, separators=(",", ":")) + "\n"
).encode("utf-8")
tmp_marker.write_bytes(marker_data)
receipt_payload = {
    "schema_version": contract["receipt_schema_version"],
    "stage": contract["stage"],
    "artifact_path": contract["artifact_path"],
    "artifact_sha256": digest.hexdigest(),
    "artifact_size_bytes": size,
    "transport_schema_version": contract["transport_schema_version"],
    "transport_encoding": contract["transport_encoding"],
    "transport_chunk_directory": contract["chunk_directory"],
    "transport_raw_chunk_bytes": contract["transport_raw_chunk_bytes"],
    "transport_max_encoded_chunk_bytes": contract[
        "transport_max_encoded_chunk_bytes"
    ],
    "transport_chunk_count": count,
    "marker_path": contract["marker_path"],
    "marker_sha256": hashlib.sha256(marker_data).hexdigest(),
    "marker_contract_sha256": contract["marker_contract_sha256"],
    "source_project_filename": src.name,
    "source_standard_task_id": contract["source_standard_task_id"],
    "source_candidate_physics_sha256": contract[
        "source_candidate_physics_sha256"
    ],
    "solver_revision": contract["solver_revision"],
    "library_revision": contract["library_revision"],
    "profile_sha256": contract["profile_sha256"],
    "parameter_digest": contract["parameter_digest"],
    "rounding_policy": contract["rounding_policy"],
    "fixed_boundary": contract["fixed_boundary"],
    "checkpoint_created_before_full_solve": True,
    "terminal_success_required_for_collection": False,
    "scientific_pass": False,
    "thermal_pass": False,
    "production_promotion_eligible": False,
    "diagnostic_only": True,
}
tmp_receipt.write_text(
    json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
for path in tmp.rglob("*"):
    os.chmod(path, 0o555 if path.is_dir() else 0o444)
os.chmod(tmp, 0o555)
os.replace(tmp, root)
print("MFT_ROUNDED_FULL_CHECKPOINT_CREATED " + digest.hexdigest(), flush=True)
""".strip()


def _inject_diagnostic_checkpoint(
    payload: dict[str, Any],
    retained: Mapping[str, Any],
    *,
    solver_revision: str,
) -> dict[str, Any]:
    """Prepend model-only generation and immutable diagnostic transport."""

    output = copy.deepcopy(payload)
    checkpoint = diagnostic_checkpoint_contract(
        payload=output,
        retained=retained,
        solver_revision=solver_revision,
    )
    actual = (
        "timeout --signal=TERM --kill-after="
        f"{KILL_GRACE_SECONDS}s {SOLVER_TIMEOUT_SECONDS}s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--full --params cand.json; simulation_rc=$?;"
    )
    if str(output.get("command") or "").count(actual) != 1:
        raise ContractError(
            "rounded Full solver command is unavailable for checkpoint wrap"
        )
    model_only = (
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--full --model-only --params cand.json"
    )
    context = json.dumps(
        checkpoint,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    checkpoint_step = (
        f"{model_only}; checkpoint_model_rc=$?; "
        '[ "$checkpoint_model_rc" -eq 0 ] || '
        'exit "$checkpoint_model_rc"; '
        "mapfile -d '' MFT_CHECKPOINT_PROJECTS < <("
        'find "$MFT_WORKDIR" -type f -name \'*.aedt\' -print0); '
        '[ "${#MFT_CHECKPOINT_PROJECTS[@]}" -eq 1 ] || { '
        "printf 'rounded Full checkpoint project count mismatch: %s\\n' "
        '"${#MFT_CHECKPOINT_PROJECTS[@]}" >&2; exit 94; }; '
        "python -c "
        + shlex.quote(_checkpoint_python())
        + ' "${MFT_CHECKPOINT_PROJECTS[0]}" '
        + shlex.quote(f"$MFT_TASK_ROOT/{checkpoint['artifact_path']}")
        + " "
        + shlex.quote(f"$MFT_TASK_ROOT/{checkpoint['receipt_path']}")
        + " "
        + shlex.quote(f"$MFT_TASK_ROOT/{checkpoint['marker_path']}")
        + " "
        + shlex.quote(f"$MFT_TASK_ROOT/{checkpoint['chunk_directory']}")
        + " "
        + shlex.quote(context)
        + "; checkpoint_rc=$?; "
        '[ "$checkpoint_rc" -eq 0 ] || exit "$checkpoint_rc"; '
        'rm -rf -- "${MFT_CHECKPOINT_PROJECTS[0]}" '
        '"${MFT_CHECKPOINT_PROJECTS[0]%.aedt}.aedtresults" '
        '"$MFT_WORKDIR/.goal-license-runtime"; '
        "python tools/mft_runtime_license_snapshot.py "
        '--output-dir "$MFT_WORKDIR/.goal-license-runtime" '
        f"--solver-revision {solver_revision} && "
        "export MFT_STANDALONE_CORE_LICENSE_CONTRACT="
        f"{production.FULL_LICENSE_CONTRACT} && "
        "export MFT_STANDALONE_CORE_LICENSE_SNAPSHOT_JSON="
        '"$(cat "$MFT_WORKDIR/.goal-license-runtime"/snapshot.json)" '
        "&& export MFT_STANDALONE_CORE_LICENSE_SNAPSHOT_SHA256="
        '"$(cat "$MFT_WORKDIR/.goal-license-runtime"/snapshot.sha256)" '
        "&& export MFT_STANDALONE_CORE_AUTH_SHA256="
        '"$(cat "$MFT_WORKDIR/.goal-license-runtime"/auth.sha256)"; '
    )
    # shlex.quote treats '$' literally.  The five destination arguments must
    # expand on the remote shell, so restore their double-quoted form.
    for key in (
        "artifact_path",
        "receipt_path",
        "marker_path",
        "chunk_directory",
    ):
        quoted = shlex.quote(f"$MFT_TASK_ROOT/{checkpoint[key]}")
        checkpoint_step = checkpoint_step.replace(
            quoted, f'"$MFT_TASK_ROOT/{checkpoint[key]}"'
        )
    output["command"] = str(output["command"]).replace(
        actual, checkpoint_step + actual, 1
    )
    return output


def derive_payload(
    *,
    standard_plan_path: Path,
    lane: StrictLane,
    license_snapshot_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    params, profile, standard_plan = full_inputs(standard_plan_path)
    solver_revision = str(standard_plan["solver_revision"])
    try:
        payload, retained, core_evidence = (
            continuation._capture_full_payload(  # noqa: SLF001
                params,
                profile,
                TASK_NAME,
                WORKDIR,
                solver_revision,
                str(standard_plan["library_revision"]),
                lane,
                license_snapshot_path.resolve(strict=True),
            )
        )
    except Exception as exc:
        raise ContractError("rounded Full payload derivation failed") from exc
    payload = _inject_diagnostic_checkpoint(
        payload,
        retained,
        solver_revision=solver_revision,
    )
    _validate_full_payload(
        payload=payload,
        retained=retained,
        core_evidence=core_evidence,
        params=params,
        profile=profile,
        solver_revision=solver_revision,
        lane=lane,
    )
    return params, profile, payload, retained, core_evidence


def _output_contract(retained: Mapping[str, Any]) -> dict[str, Any]:
    transport = retained["transport"]
    return {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "stage": "full",
        "source_task_id": SOURCE_STANDARD_TASK_ID,
        "rounded_full_model": True,
        "remote": {
            "artifact_path": retained["artifact_path"],
            "receipt_path": retained["receipt_path"],
            "marker_path": retained["marker_path"],
            "chunk_directory": transport["chunk_directory"],
            "results_path": retained["results_path"],
            "results_manifest_path": retained[
                "results_manifest_path"
            ],
        },
        "required_local_collection": {
            "artifact_filename": "full_model.aedt",
            "results_manifest_filename": (
                "full_model.aedtresults.manifest.json"
            ),
            "result_filename": "result.json",
            "terminal_task_filename": "scheduler_terminal_task.json",
            "collection_receipt_filename": "collection_receipt.json",
            "collection_seal_filename": "collection_seal.json",
        },
        "authenticity_checks": [
            "solver/library/profile/parameter identity",
            "full_model=1 and thermal_symmetry=full",
            "round_corner=1 radius=10mm segments=4",
            "fixed fan/TIM/pad identity",
            "artifact SHA-256 reconstructed from authenticated chunks",
            "AEDT results manifest tree SHA-256",
        ],
        "synthetic_aedt_or_results_forbidden": True,
    }


def dry_run(
    *,
    standard_plan_path: Path,
    lane: StrictLane,
    license_snapshot_path: Path,
    output: Path,
    observed_at: datetime | None = None,
) -> Path:
    """Derive a no-GET/no-POST audit; it is never submission authority."""

    params, profile, payload, retained, core = derive_payload(
        standard_plan_path=standard_plan_path,
        lane=lane,
        license_snapshot_path=license_snapshot_path,
    )
    checkpoint = diagnostic_checkpoint_contract(
        payload=payload,
        retained=retained,
        solver_revision=(
            rounded.load_plan(standard_plan_path)["solver_revision"]
        ),
    )
    value = sealed(
        {
            "schema_version": DRY_RUN_SCHEMA,
            "created_at_utc": _now(observed_at).isoformat(),
            "dry_run_only": True,
            "scientific_source_success_checked": False,
            "submission_authority": False,
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
            "scheduler_submission_performed": False,
            "source_standard_task_id_required": SOURCE_STANDARD_TASK_ID,
            "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
            "source_standard_plan": _file_record(standard_plan_path),
            "solver_revision": (
                rounded.load_plan(standard_plan_path)["solver_revision"]
            ),
            "library_revision": (
                rounded.load_plan(standard_plan_path)["library_revision"]
            ),
            "target_lane_template": {
                "account_name": lane.account_name,
                "node_name": lane.node_name,
                "node_name_policy": "strict",
                "same_node_as_task_id": 0,
                "requires_fea_empty_node": True,
            },
            "resources": {
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
                "scheduler_timeout_seconds": SCHEDULER_TIMEOUT_SECONDS,
                "kill_grace_seconds": KILL_GRACE_SECONDS,
            },
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "full_params_sha256": payload_sha256(params),
            "full_profile": _file_record(FULL_PROFILE_PATH),
            "scheduler_payload": payload,
            "scheduler_payload_sha256": payload_sha256(payload),
            "retained_full_aedt_bundle": retained,
            "retained_full_aedt_bundle_sha256": payload_sha256(retained),
            "diagnostic_geometry_setup_checkpoint": checkpoint,
            "diagnostic_geometry_setup_checkpoint_sha256": (
                payload_sha256(checkpoint)
            ),
            "core_policy": core,
            "output_contract": _output_contract(retained),
            "next_gate": (
                "task96340 authenticated rounded Standard terminal success "
                "and all actual hard constraints pass"
            ),
        }
    )
    target = output.resolve()
    if target.exists():
        raise ContractError(f"dry-run output already exists: {target}")
    return write_immutable_json(target, value)


def prepare(
    *,
    standard_plan_path: Path,
    standard_final_path: Path,
    standard_collection_root: Path,
    lane: StrictLane,
    license_snapshot_path: Path,
    output_root: Path = OUTPUT_ROOT,
    reader: continuation.JsonReader = continuation.get_json,
    observed_at: datetime | None = None,
) -> Path:
    """Seal the GET-only Full plan after task96340 actual success."""

    target = output_root.resolve()
    if target.exists():
        raise ContractError(f"Full prepare output already exists: {target}")
    license_freshness = validate_license_freshness(
        license_snapshot_path, observed_at=observed_at
    )
    authority = authenticate_standard_success(
        standard_plan_path=standard_plan_path,
        standard_final_path=standard_final_path,
        standard_collection_root=standard_collection_root,
    )
    params, profile, payload, retained, core = derive_payload(
        standard_plan_path=standard_plan_path,
        lane=lane,
        license_snapshot_path=license_snapshot_path,
    )
    checkpoint = diagnostic_checkpoint_contract(
        payload=payload,
        retained=retained,
        solver_revision=str(authority["source_solver_revision"]),
    )
    preflight = continuation._lane_preflight(  # noqa: SLF001
        lane=lane,
        source_node=str(authority["source_node_name"]),
        task_name=TASK_NAME,
        dedupe_key=str(payload["dedupe_key"]),
        source_task_id=SOURCE_STANDARD_TASK_ID,
        reader=reader,
    )
    solver_revision = str(authority["source_solver_revision"])
    revision_attestation = rounded.attest_rounded_solver_revision(
        solver_revision
    )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent)
    )
    try:
        params_path = write_immutable_json(staging / PARAMS_NAME, params)
        profile_path = write_immutable_json(staging / PROFILE_NAME, profile)
        authority_path = write_immutable_json(
            staging / SOURCE_AUTHORITY_NAME, authority
        )
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": _now(observed_at).isoformat(),
                "prepare_only": True,
                "scheduler_get_only": True,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "submission_capability_present": False,
                "maximum_full_tasks": 1,
                "source_standard_task_id": SOURCE_STANDARD_TASK_ID,
                "source_candidate_physics_sha256": (
                    SOURCE_CANDIDATE_SHA256
                ),
                "source_standard_success_authority": _relative_record(
                    staging, authority_path
                ),
                "measured_standard_hard_constraints_passed": True,
                "same_candidate_geometry": True,
                "full_params": _relative_record(staging, params_path),
                "full_params_sha256": payload_sha256(params),
                "full_profile": _relative_record(staging, profile_path),
                "full_profile_sha256": payload_sha256(profile),
                "solver_revision": solver_revision,
                "library_revision": authority[
                    "source_library_revision"
                ],
                "solver_revision_attestation": revision_attestation,
                "source_and_full_solver_revision_identical": True,
                "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "target_lane": {
                    "account_name": lane.account_name,
                    "node_name": lane.node_name,
                    "node_name_policy": "strict",
                    "same_node_as_task_id": 0,
                    "different_from_source_node": True,
                    "requires_fea_empty_node": True,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                },
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "solver_timeout_seconds": SOLVER_TIMEOUT_SECONDS,
                    "scheduler_timeout_seconds": (
                        SCHEDULER_TIMEOUT_SECONDS
                    ),
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                    "aedt_backend": "standalone",
                },
                "fresh_license_snapshot": license_freshness,
                "core_policy": core,
                "prepare_live_preflight": preflight,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "retained_full_aedt_bundle": retained,
                "retained_full_aedt_bundle_sha256": (
                    payload_sha256(retained)
                ),
                "diagnostic_geometry_setup_checkpoint": checkpoint,
                "diagnostic_geometry_setup_checkpoint_sha256": (
                    payload_sha256(checkpoint)
                ),
                "output_contract": _output_contract(retained),
                "collector_required": (
                    "tools/mft_goal_official5_rounded_full_collector.py"
                ),
                "scheduler_project_access": "public_api_only",
                "scheduler_repository_modified": False,
                "mft_and_scheduler_functionality_mixed": False,
                "automatic_full_trigger": False,
            }
        )
        write_immutable_json(staging / PLAN_NAME, plan)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def load_plan(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    plan = validate_seal(read_json(resolved), PLAN_SCHEMA)
    root = resolved.parent
    params_path = rounded._contained(  # noqa: SLF001
        root, plan.get("full_params"), "Full params"
    )
    profile_path = rounded._contained(  # noqa: SLF001
        root, plan.get("full_profile"), "Full profile"
    )
    authority_path = rounded._contained(  # noqa: SLF001
        root,
        plan.get("source_standard_success_authority"),
        "rounded Standard success authority",
    )
    params = read_json(params_path)
    profile = read_json(profile_path)
    authority = validate_seal(
        read_json(authority_path), SOURCE_AUTHORITY_SCHEMA
    )
    payload = plan.get("scheduler_payload")
    retained = plan.get("retained_full_aedt_bundle")
    checkpoint = plan.get("diagnostic_geometry_setup_checkpoint")
    lane = StrictLane(
        str(plan.get("target_lane", {}).get("account_name") or ""),
        str(plan.get("target_lane", {}).get("node_name") or ""),
    )
    if (
        plan.get("prepare_only") is not True
        or plan.get("scheduler_post_calls") != 0
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("submission_capability_present") is not False
        or plan.get("maximum_full_tasks") != 1
        or plan.get("source_standard_task_id")
        != SOURCE_STANDARD_TASK_ID
        or authority.get("same_candidate_full_authorized") is not True
        or plan.get("full_params_sha256") != payload_sha256(params)
        or profile != _load_full_profile()
        or plan.get("full_profile_sha256") != payload_sha256(profile)
        or not isinstance(payload, Mapping)
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(payload)
        or not isinstance(retained, Mapping)
        or plan.get("retained_full_aedt_bundle_sha256")
        != payload_sha256(retained)
        or not isinstance(checkpoint, Mapping)
        or checkpoint
        != diagnostic_checkpoint_contract(
            payload=payload,
            retained=retained,
            solver_revision=str(plan["solver_revision"]),
        )
        or plan.get("diagnostic_geometry_setup_checkpoint_sha256")
        != payload_sha256(checkpoint)
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("mft_and_scheduler_functionality_mixed") is not False
    ):
        raise ContractError("rounded Full prepare plan drifted")
    _validate_full_payload(
        payload=payload,
        retained=retained,
        core_evidence=plan["core_policy"],
        params=params,
        profile=profile,
        solver_revision=str(plan["solver_revision"]),
        lane=lane,
    )
    return plan


def _parse_lane(value: str) -> StrictLane:
    try:
        return continuation._parse_lane(value)  # noqa: SLF001
    except argparse.ArgumentTypeError:
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser(
        "dry-run", help="derive payload with zero Scheduler I/O"
    )
    dry.add_argument("--standard-plan", type=Path, required=True)
    dry.add_argument("--strict-lane", type=_parse_lane, required=True)
    dry.add_argument("--license-snapshot", type=Path, required=True)
    dry.add_argument("--output", type=Path, required=True)
    prep = commands.add_parser(
        "prepare", help="seal after authenticated task96340 success"
    )
    prep.add_argument("--standard-plan", type=Path, required=True)
    prep.add_argument("--standard-final", type=Path, required=True)
    prep.add_argument("--standard-collection", type=Path, required=True)
    prep.add_argument("--strict-lane", type=_parse_lane, required=True)
    prep.add_argument("--license-snapshot", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    inspect = commands.add_parser(
        "inspect", help="reauthenticate an existing prepare plan"
    )
    inspect.add_argument(
        "--plan", type=Path, default=OUTPUT_ROOT / PLAN_NAME
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        path = dry_run(
            standard_plan_path=args.standard_plan,
            lane=args.strict_lane,
            license_snapshot_path=args.license_snapshot,
            output=args.output,
        )
        value = validate_seal(read_json(path), DRY_RUN_SCHEMA)
        event = "rounded_full_dry_run_complete"
    elif args.command == "prepare":
        path = prepare(
            standard_plan_path=args.standard_plan,
            standard_final_path=args.standard_final,
            standard_collection_root=args.standard_collection,
            lane=args.strict_lane,
            license_snapshot_path=args.license_snapshot,
            output_root=args.output_root,
        )
        value = load_plan(path)
        event = "rounded_full_prepare_complete"
    else:
        path = args.plan.resolve(strict=True)
        value = load_plan(path)
        event = "rounded_full_prepare_authenticated"
    print(
        json.dumps(
            {
                "event": event,
                "path": str(path),
                "payload_sha256": value["payload_sha256"],
                "source_standard_task_id": SOURCE_STANDARD_TASK_ID,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_full_prepare_error",
                    "error": str(exc),
                    "scheduler_post_calls": 0,
                    "scheduler_submission_performed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
