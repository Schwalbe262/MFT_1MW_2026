"""Explicit, read-only bridge from a provisional project-only Full run.

The provisional Full precompute deliberately does not impersonate the
canonical truth-promotion Full submission/collection schemas.  This module
keeps that boundary intact.  It consumes:

* a canonical ``mft_goal_truth_promotion`` Full plan, which proves that the
  exact Standard candidate passed the actual constraints and is rank 0; and
* a terminal provisional Full collector receipt, which proves the Full result
  and local project bytes without claiming a retained ``.aedtresults`` tree.

Only an exact candidate/parameter/revision/cooling/Slurm-lineage match can
produce a bridge receipt.  The receipt is suitable authority for an explicit
future project-only package consumer.  It is intentionally *not* a
``FULL_COLLECTION_SCHEMA`` or ``PACKAGE_SCHEMA`` v1 object.

No Scheduler call, POST, cancellation, claim mutation, or remote mutation is
implemented here.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    attest_fixed_identity,
    canonical_sha256,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_provisional_full_collector as collector  # noqa: E402
from tools import mft_goal_terminal_success_watcher as terminal  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


BRIDGE_SCHEMA = "mft-goal-provisional-full-explicit-promotion-bridge-v1"
AUTHENTICATED_BRIDGE_SCHEMA = (
    "mft-goal-provisional-full-explicit-promotion-bridge-authenticated-v1"
)
CAMPAIGN_ID = "mft-goal-20260726"
SOURCE_RETRY_SCHEMA = "mft-goal-diagnostic-standard-timeout12h-evidence-v1"
FIXED_BOUNDARY = {
    "fan_velocity_m_s": 1.5,
    "fan_config": "dual",
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
}
PHYSICS_PROFILE_FIELDS = (
    "reviewed_solver_path",
    "cli_flags",
    "param_overrides",
    "fixed_boundary_contract",
    "mem_mb",
    "cpus",
    "timeout_seconds",
)

HandoffContractError = production.HandoffContractError


def _static_flags() -> dict[str, bool]:
    return {
        "explicit_adapter": True,
        "automatic_promotion": False,
        "canonical_full_collection_schema_impersonated": False,
        "canonical_package_v1_schema_impersonated": False,
        "canonical_truth_gate_modified": False,
        "canonical_claim_modified": False,
        "scheduler_mutation_performed": False,
        "scheduler_post_performed": False,
        "scheduler_cancel_performed": False,
        "remote_artifact_mutation_performed": False,
        "full_aedtresults_available": False,
        "drop_in_truth_full_collection_v1_compatible": False,
        "drop_in_truth_full_package_v1_compatible": False,
        "explicit_project_only_package_consumer_required": True,
    }


def _same_float(left: Any, right: Any, *, absolute: float = 1e-9) -> bool:
    return math.isclose(
        production._finite(left, "bridge numeric identity"),
        production._finite(right, "bridge numeric identity"),
        rel_tol=0.0,
        abs_tol=absolute,
    )


def _profile_physics_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in PHYSICS_PROFILE_FIELDS if name not in profile]
    if missing:
        raise HandoffContractError(f"Full physics profile fields are absent: {missing}")
    return {name: copy.deepcopy(profile[name]) for name in PHYSICS_PROFILE_FIELDS}


def _source_logical_authority(
    standard_plan: Mapping[str, Any],
    standard_submission: Mapping[str, Any],
) -> int:
    plan_retry = standard_plan.get("retry_of_timeout12h")
    submission_retry = standard_submission.get("retry_of_timeout12h")
    if (
        not isinstance(plan_retry, Mapping)
        or dict(plan_retry) != submission_retry
        or plan_retry.get("schema_version") != SOURCE_RETRY_SCHEMA
        or plan_retry.get("fixed_physics_unchanged") is not True
        or plan_retry.get("timeout_change_only") is not True
    ):
        raise HandoffContractError(
            "Standard timeout12h logical-authority lineage drifted"
        )
    logical = plan_retry.get("logical_authority_task_id")
    if isinstance(logical, bool) or not isinstance(logical, int) or logical <= 0:
        raise HandoffContractError("Standard logical-authority task ID is invalid")
    return logical


def _strict_standard_execution(
    execution: Mapping[str, Any],
    *,
    source_authority: Mapping[str, Any],
    standard_submission: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = source_authority.get("task_id")
    account = source_authority.get("account_name")
    node = source_authority.get("node_name")
    if (
        source_authority.get("node_name_policy") != "strict"
        or execution.get("task_id") != task_id
        or execution.get("task_id") != standard_submission.get("task_id")
        or execution.get("account_name") != account
        or execution.get("requested_account_name") != account
        or execution.get("actual_node_name") != node
        or execution.get("node_name") != node
        or execution.get("requested_node_name") != node
        or execution.get("allocation_node_name") != node
        or execution.get("node_name_policy") != "strict"
        or execution.get("requested_node_name_policy") != "strict"
        or execution.get("strict_node_placement") is not True
        or execution.get("placement_contract_satisfied") is not True
        or execution.get("same_node_as_task_id") != 0
        or execution.get("status") != "completed"
        or execution.get("state") != "succeeded"
        or execution.get("exit_code") != 0
        or execution.get("project") != scheduler_client.MFT_PROJECT
        or execution.get("cpus") != 8
        or execution.get("memory_mb") != 32768
        or execution.get("timeout_seconds") != 43200
        or execution.get("aedt_backend") != "standalone"
        or not str(execution.get("slurm_job_id") or "").isdigit()
    ):
        raise HandoffContractError(
            "Standard terminal strict-node execution lineage drifted"
        )
    return copy.deepcopy(dict(execution))


def _authenticate_provisional_success(
    path: Path,
) -> dict[str, Any]:
    success_path = path.resolve(strict=True)
    success = production._validate_seal(
        production._read_json(success_path), collector.SUCCESS_SCHEMA
    )
    watch_record = success.get("watch_plan")
    if not isinstance(watch_record, Mapping):
        raise HandoffContractError(
            "provisional Full success watch-plan record is absent"
        )
    watch_path = Path(str(watch_record.get("path") or ""))
    if production._file_record(watch_path) != dict(watch_record):
        raise HandoffContractError("provisional Full success watch-plan bytes drifted")
    watch, plan, params, profile, submission = collector._load_watch_plan(watch_path)
    if success_path != Path(watch["success_receipt_path"]).resolve(strict=True):
        raise HandoffContractError(
            "provisional Full success path escaped its watch authority"
        )
    success = collector._validate_success_replay(
        success_path,
        watch=watch,
        plan=plan,
        submission=submission,
    )
    task = collector._validate_task_identity(
        success.get("scheduler_task_execution") or {},
        plan=plan,
        submission=submission,
        terminal_success=True,
    )
    result = collector._validate_result(
        success.get("result"),
        plan=plan,
        params=params,
        profile=profile,
        submission=submission,
        task=task,
        scheduler=scheduler_client,
    )
    receipt = success.get("remote_retained_aedt_receipt")
    if not isinstance(receipt, Mapping):
        raise HandoffContractError("provisional Full retained AEDT receipt is absent")
    legacy = {
        name: copy.deepcopy(receipt[name]) for name in production.REMOTE_RECEIPT_FIELDS
    }
    remote_submission = {
        "stage": "full",
        "retained_aedt": submission["retained_aedt"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "profile_sha256": plan["profile_canonical_sha256"],
    }
    production._validate_remote_receipt_payload(
        legacy,
        submission=remote_submission,
        result=result,
    )
    cap = collector._validate_source_cap(
        plan=plan,
        watch=watch,
        receipt=receipt,
    )
    marker = production._validate_marker_payload(
        success.get("remote_prune_protection_marker"),
        expected=submission["retained_aedt"],
    )
    if (
        production._sha256_bytes(production._json_bytes(marker))
        != receipt["marker_sha256"]
        or canonical_sha256(result) != success.get("result_sha256")
        or success.get("result_identity")
        != {
            "solver_revision": plan["solver_revision"],
            "library_revision": plan["library_revision"],
            "profile_canonical_sha256": plan["profile_canonical_sha256"],
            "effective_full_params_sha256": plan["effective_full_params_sha256"],
            "project_name": result["project_name"],
            "full_model": 1,
            "thermal_symmetry": "full",
        }
        or success.get("sealed_source_size_hard_cap_bytes") != cap
        or success.get("source_artifact_size_within_sealed_hard_cap") is not True
        or success.get("all_declared_chunks_reconstructed") is not True
    ):
        raise HandoffContractError(
            "provisional Full success result/artifact identity drifted"
        )
    return {
        "success_path": success_path,
        "success": success,
        "watch_path": watch_path,
        "watch": watch,
        "plan": plan,
        "params": params,
        "profile": profile,
        "submission": submission,
        "task": task,
        "result": result,
        "receipt": copy.deepcopy(dict(receipt)),
        "marker": marker,
    }


def _full_actual_truth(
    *,
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    plan: Mapping[str, Any],
    task_id: int,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = production._goal_result_reasons(result, selected)
    active, temperatures, temperature_passed = promotion._temperature_evidence(result)
    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        production._finite(value, "provisional Full exterior dimension")
        for value in dimensions
    )
    resonance = production._finite(
        result.get("f_res_min_tx_rx_only_Hz"),
        "provisional Full resonance",
    )
    losses = {
        name: production._finite(result.get(name), f"provisional Full {name}")
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    identity_values = dict(result)
    identity_values["thermal_pad_conductivity_W_mK"] = 0.2
    fixed = attest_fixed_identity(identity_values)
    if fixed != selected["row_contract"]["fixed_identity_attestation"]:
        raise HandoffContractError(
            "provisional Full fixed cooling/operating identity drifted"
        )
    passed = (
        not reasons
        and temperature_passed
        and width <= float(GOAL_SIZE_LIMITS_MM["W"])
        and length <= float(GOAL_SIZE_LIMITS_MM["L"])
        and height <= float(GOAL_SIZE_LIMITS_MM["H"])
        and resonance >= float(GOAL_STAGE_SPEC["resonance_min_Hz"])
    )
    return {
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "scheduler_task_id": task_id,
        "actual_volume_L": float(volume_l),
        "actual_total_loss_W": sum(losses.values()),
        "actual_loss_components_W": losses,
        "actual_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_resonance_Hz": resonance,
        "actual_physical_Llt_uH": production._finite(
            result.get("Llt"), "provisional Full Llt"
        ),
        "active_temperature_targets": active,
        "actual_temperatures_C": {
            name: production._finite(item["actual_C"], name)
            for name, item in temperatures.items()
        },
        "actual_constraint_margins": {
            "width_mm": float(GOAL_SIZE_LIMITS_MM["W"]) - width,
            "length_mm": float(GOAL_SIZE_LIMITS_MM["L"]) - length,
            "height_mm": float(GOAL_SIZE_LIMITS_MM["H"]) - height,
            "resonance_Hz": resonance - float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
            "temperature_C": {
                name: float(item["limit_C"])
                - production._finite(item["actual_C"], name)
                for name, item in temperatures.items()
            },
        },
        "goal_physical_spec_reasons": list(reasons),
        "actual_body_probe_temperature_gate_passed": temperature_passed,
        "goal_physical_spec_passed": passed,
        "fixed_identity_attestation": fixed,
        "retained_full_model": {
            "path": receipt["artifact_path"],
            "sha256": receipt["artifact_sha256"],
            "size_bytes": receipt["artifact_size_bytes"],
        },
        "retained_full_results": None,
        "project_only_retention": True,
    }


def _geometry_matches(
    standard_truth: Mapping[str, Any],
    full_truth: Mapping[str, Any],
) -> bool:
    return all(
        _same_float(
            standard_truth["actual_dimensions_mm"][axis],
            full_truth["actual_dimensions_mm"][axis],
        )
        for axis in ("W", "L", "H")
    ) and _same_float(
        standard_truth["actual_volume_L"],
        full_truth["actual_volume_L"],
    )


def _terminal_nds_receipt(
    *,
    truth_manifest_record: Mapping[str, Any],
    standard_collection_record: Mapping[str, Any],
    logical_authority_task_id: int,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(str(truth_manifest_record.get("path") or ""))
    if production._file_record(manifest_path) != dict(truth_manifest_record):
        raise HandoffContractError("terminal global-NDS truth manifest bytes drifted")
    snapshot_directory = manifest_path.parent
    receipt_path = snapshot_directory.parent / (
        f"{snapshot_directory.name}.receipt.json"
    )
    receipt = production._validate_seal(
        production._read_json(receipt_path.resolve(strict=True)),
        terminal.NDS_RECEIPT_SCHEMA,
    )
    logical_ids = receipt.get("logical_authority_task_ids")
    collections = receipt.get("input_collections")
    if (
        set(receipt)
        != {
            "schema_version",
            "payload_sha256",
            "campaign_id",
            "authority_key",
            "logical_authority_task_ids",
            "input_collections",
            "truth_manifest",
            "global_nondominated_sort_performed",
            "scheduler_mutation_performed",
            "created_at_utc",
        }
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(logical_ids, list)
        or not logical_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in logical_ids
        )
        or logical_ids != sorted(set(logical_ids))
        or not isinstance(collections, list)
        or len(collections) != len(logical_ids)
        or receipt.get("truth_manifest") != dict(truth_manifest_record)
        or receipt.get("global_nondominated_sort_performed") is not True
        or receipt.get("scheduler_mutation_performed") is not False
        or not str(receipt.get("created_at_utc") or "")
    ):
        raise HandoffContractError("terminal global-NDS receipt contract drifted")
    authenticated_collections = []
    for record in collections:
        if not isinstance(record, Mapping):
            raise HandoffContractError(
                "terminal global-NDS collection record is malformed"
            )
        path = Path(str(record.get("path") or ""))
        if production._file_record(path) != dict(record):
            raise HandoffContractError("terminal global-NDS collection bytes drifted")
        authenticated_collections.append(dict(record))
    authority = [
        {
            "logical_authority_task_id": task_id,
            "collection_payload_sha256": record["sha256"],
        }
        for task_id, record in zip(logical_ids, authenticated_collections, strict=True)
    ]
    key = canonical_sha256(authority)[:16]
    if (
        receipt.get("authority_key") != key
        or snapshot_directory.name != f"n{len(authenticated_collections):02d}-{key}"
        or not any(
            task_id == logical_authority_task_id
            and record == dict(standard_collection_record)
            for task_id, record in zip(
                logical_ids, authenticated_collections, strict=True
            )
        )
    ):
        raise HandoffContractError("terminal global-NDS authority mapping drifted")
    return receipt_path, receipt


def _build_bridge(
    *,
    canonical_full_plan_path: Path,
    provisional_success_path: Path,
    predictor: Any | None = None,
) -> dict[str, Any]:
    full_plan_path = canonical_full_plan_path.resolve(strict=True)
    (
        full_plan,
        canonical_params,
        canonical_profile,
        standard_view,
        standard_truth,
    ) = promotion._load_full_plan(full_plan_path, predictor=predictor)
    provisional = _authenticate_provisional_success(provisional_success_path)
    success = provisional["success"]
    provisional_plan = provisional["plan"]
    provisional_submission = provisional["submission"]
    standard_collection = standard_view["collection"]
    standard_plan = standard_view["plan"]
    standard_submission = standard_view["submission"]
    candidate = full_plan["candidate_physics_sha256"]
    source_authority = provisional_plan.get("source_standard_execution_authority")
    if not isinstance(source_authority, Mapping):
        raise HandoffContractError(
            "provisional source Standard execution authority is absent"
        )
    logical_task_id = _source_logical_authority(standard_plan, standard_submission)
    standard_execution = _strict_standard_execution(
        standard_collection["scheduler_task_execution"],
        source_authority=source_authority,
        standard_submission=standard_submission,
    )
    if (
        standard_collection.get("plan") != provisional_plan.get("target_source_plan")
        or standard_plan.get("payload_sha256")
        != provisional_plan.get("target_source_plan_payload_sha256")
        or canonical_params != provisional["params"]
        or canonical_sha256(canonical_params)
        != provisional_plan.get("source_fea_params_sha256")
        or canonical_sha256(canonical_params) != full_plan.get("fea_params_sha256")
        or candidate != provisional_plan.get("candidate_physics_sha256")
        or candidate != success.get("candidate_physics_sha256")
    ):
        raise HandoffContractError(
            "canonical/provisional candidate parameter ancestry drifted"
        )
    if (
        full_plan.get("standard_collection") != standard_truth["collection"]
        or standard_truth["collection"]
        != production._file_record(Path(full_plan["standard_collection"]["path"]))
        or full_plan.get("standard_task_id")
        != provisional_plan.get("source_actual_standard_task_id")
        or standard_collection.get("task_id")
        != provisional_plan.get("source_actual_standard_task_id")
        or logical_task_id != provisional_plan.get("logical_authority_task_id")
        or full_plan.get("solver_revision") != provisional_plan.get("solver_revision")
        or full_plan.get("library_revision") != provisional_plan.get("library_revision")
        or full_plan.get("solver_revision")
        != success["result_identity"]["solver_revision"]
        or full_plan.get("library_revision")
        != success["result_identity"]["library_revision"]
        or standard_collection.get("candidate_physics_sha256") != candidate
        or standard_truth.get("candidate_physics_sha256") != candidate
        or provisional_submission.get("task_id") != success.get("task_id")
        or provisional_plan.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or standard_collection.get("scheduler_url")
        != provisional_plan.get("scheduler_url")
    ):
        raise HandoffContractError(
            "canonical/provisional truth or Slurm ancestry drifted"
        )
    nds_receipt_path, nds_receipt = _terminal_nds_receipt(
        truth_manifest_record=full_plan["truth_promotion_manifest"],
        standard_collection_record=standard_truth["collection"],
        logical_authority_task_id=logical_task_id,
    )
    canonical_physics = _profile_physics_identity(canonical_profile)
    provisional_physics = _profile_physics_identity(provisional["profile"])
    canonical_effective = production._effective_params(
        canonical_params, canonical_profile
    )
    provisional_effective = production._effective_params(
        provisional["params"], provisional["profile"]
    )
    if (
        canonical_physics != provisional_physics
        or canonical_effective != provisional_effective
        or canonical_sha256(canonical_effective)
        != full_plan["stage"]["effective_params_sha256"]
        or canonical_sha256(provisional_effective)
        != provisional_plan["effective_full_params_sha256"]
        or provisional_plan.get("fixed_boundary")
        != {**FIXED_BOUNDARY, "mutable": False}
    ):
        raise HandoffContractError(
            "canonical/provisional Full physics or fixed cooling drifted"
        )
    full_truth = _full_actual_truth(
        result=provisional["result"],
        selected=standard_view["selected"],
        plan=provisional_plan,
        task_id=success["task_id"],
        receipt=provisional["receipt"],
    )
    if (
        full_truth.get("candidate_physics_sha256") != candidate
        or full_truth.get("solver_revision") != full_plan["solver_revision"]
        or full_truth.get("library_revision") != full_plan["library_revision"]
        or full_truth.get("scheduler_task_id") != success["task_id"]
        or full_truth.get("fixed_identity_attestation")
        != standard_truth["fixed_identity_attestation"]
    ):
        raise HandoffContractError("provisional Full actual-truth identity drifted")
    geometry_match = _geometry_matches(standard_truth, full_truth)
    standard_pass = (
        standard_collection.get("goal_physical_spec_passed") is True
        and standard_collection.get("actual_body_probe_temperature_gate_passed") is True
        and standard_truth.get("actual_constraint_margins", {}).get("temperature_C")
        is not None
    )
    full_pass = full_truth["goal_physical_spec_passed"] is True
    rank0_pass = full_plan["truth_row"]["truth_non_dominated_rank"] == 0
    production_eligible = bool(
        standard_pass and full_pass and geometry_match and rank0_pass
    )
    receipt = production._seal(
        {
            "schema_version": BRIDGE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (GOAL_TEMPERATURE_CONTRACT_SHA256),
            "canonical_full_plan": production._file_record(full_plan_path),
            "canonical_full_plan_payload_sha256": full_plan["payload_sha256"],
            "truth_promotion_manifest": copy.deepcopy(
                full_plan["truth_promotion_manifest"]
            ),
            "truth_promotion_payload_sha256": full_plan[
                "truth_promotion_payload_sha256"
            ],
            "truth_nds_receipt": production._file_record(nds_receipt_path),
            "truth_nds_receipt_payload_sha256": nds_receipt["payload_sha256"],
            "truth_row_sha256": full_plan["truth_row"]["truth_row_sha256"],
            "truth_non_dominated_rank": full_plan["truth_row"][
                "truth_non_dominated_rank"
            ],
            "candidate_physics_sha256": candidate,
            "logical_authority_task_id": logical_task_id,
            "standard_collection": copy.deepcopy(standard_truth["collection"]),
            "standard_collection_payload_sha256": standard_truth[
                "collection_payload_sha256"
            ],
            "standard_result_sha256": standard_truth["standard_result_sha256"],
            "standard_task_id": standard_collection["task_id"],
            "provisional_full_success": production._file_record(
                provisional["success_path"]
            ),
            "provisional_full_success_payload_sha256": success["payload_sha256"],
            "provisional_full_plan": copy.deepcopy(success["provisional_plan"]),
            "provisional_full_plan_payload_sha256": provisional_plan["payload_sha256"],
            "provisional_full_submission": copy.deepcopy(success["submission"]),
            "provisional_full_submission_payload_sha256": (
                provisional_submission["payload_sha256"]
            ),
            "provisional_full_task_id": success["task_id"],
            "solver_revision": full_plan["solver_revision"],
            "library_revision": full_plan["library_revision"],
            "fea_params_sha256": full_plan["fea_params_sha256"],
            "effective_full_params_sha256": canonical_sha256(canonical_effective),
            "canonical_full_profile_physics_sha256": canonical_sha256(
                canonical_physics
            ),
            "provisional_full_profile_physics_sha256": canonical_sha256(
                provisional_physics
            ),
            "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
            "fixed_identity_attestation": copy.deepcopy(
                standard_truth["fixed_identity_attestation"]
            ),
            "standard_actual_truth": copy.deepcopy(standard_truth),
            "provisional_full_actual_truth": full_truth,
            "standard_scheduler_task_execution": standard_execution,
            "provisional_full_scheduler_task_execution": copy.deepcopy(
                provisional["task"]
            ),
            "standard_source_execution_authority": copy.deepcopy(
                dict(source_authority)
            ),
            "provisional_full_execution_route": {
                "selected_account_name": provisional_plan["selected_account_name"],
                "requested_node_name": provisional_plan["requested_node_name"],
                "node_name_policy": "strict",
            },
            "symmetric_model": {
                "retained_remote_receipt": copy.deepcopy(
                    standard_collection["remote_aedt_bundle_receipt"]
                ),
                "full_model": 0,
                "thermal_symmetry": "eighth",
            },
            "full_model": {
                "local_artifact": copy.deepcopy(success["local_full_aedt"]),
                "retained_remote_receipt": copy.deepcopy(provisional["receipt"]),
                "full_model": 1,
                "thermal_symmetry": "full",
                "aedtresults": None,
            },
            "authenticated_standard_constraint_pass": standard_pass,
            "authenticated_provisional_full_constraint_pass": full_pass,
            "exact_same_candidate_parameters_revisions": True,
            "exact_fixed_cooling_identity": True,
            "exact_geometry_identity": geometry_match,
            "rank0_standard_truth_authority": (rank0_pass),
            "project_only_model_package_eligible": production_eligible,
            "bridge_passed": production_eligible,
            "production_eligible": production_eligible,
            "canonical_v1_promotion_allowed": False,
            "explicit_operator_action_required": True,
            "consumer_contract": {
                "schema_version": ("mft-goal-project-only-full-package-consumer-v1"),
                "bridge_reauthentication_required": True,
                "production_eligible_true_required": True,
                "allowed_primary_outputs": [
                    "symmetric_model.aedt",
                    "full_model.aedt",
                ],
                "full_aedtresults_must_not_be_synthesized": True,
                "truth_full_collection_v1_must_not_be_emitted": True,
                "truth_full_package_v1_must_not_be_emitted": True,
            },
            **_static_flags(),
        }
    )
    return receipt


BRIDGE_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "canonical_full_plan",
        "canonical_full_plan_payload_sha256",
        "truth_promotion_manifest",
        "truth_promotion_payload_sha256",
        "truth_nds_receipt",
        "truth_nds_receipt_payload_sha256",
        "truth_row_sha256",
        "truth_non_dominated_rank",
        "candidate_physics_sha256",
        "logical_authority_task_id",
        "standard_collection",
        "standard_collection_payload_sha256",
        "standard_result_sha256",
        "standard_task_id",
        "provisional_full_success",
        "provisional_full_success_payload_sha256",
        "provisional_full_plan",
        "provisional_full_plan_payload_sha256",
        "provisional_full_submission",
        "provisional_full_submission_payload_sha256",
        "provisional_full_task_id",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "effective_full_params_sha256",
        "canonical_full_profile_physics_sha256",
        "provisional_full_profile_physics_sha256",
        "fixed_boundary",
        "fixed_identity_attestation",
        "standard_actual_truth",
        "provisional_full_actual_truth",
        "standard_scheduler_task_execution",
        "provisional_full_scheduler_task_execution",
        "standard_source_execution_authority",
        "provisional_full_execution_route",
        "symmetric_model",
        "full_model",
        "authenticated_standard_constraint_pass",
        "authenticated_provisional_full_constraint_pass",
        "exact_same_candidate_parameters_revisions",
        "exact_fixed_cooling_identity",
        "exact_geometry_identity",
        "rank0_standard_truth_authority",
        "project_only_model_package_eligible",
        "bridge_passed",
        "production_eligible",
        "canonical_v1_promotion_allowed",
        "explicit_operator_action_required",
        "consumer_contract",
        *_static_flags(),
    }
)


def authenticate_bridge(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Reauthenticate all source bytes and recompute a bridge receipt."""
    resolved = path.resolve(strict=True)
    value = production._validate_seal(production._read_json(resolved), BRIDGE_SCHEMA)
    if set(value) != BRIDGE_FIELDS:
        raise HandoffContractError("provisional Full bridge fields drifted")
    plan_record = value.get("canonical_full_plan")
    success_record = value.get("provisional_full_success")
    if not isinstance(plan_record, Mapping) or not isinstance(success_record, Mapping):
        raise HandoffContractError("provisional Full bridge source records are absent")
    plan_path = Path(str(plan_record.get("path") or ""))
    success_path = Path(str(success_record.get("path") or ""))
    if production._file_record(plan_path) != dict(
        plan_record
    ) or production._file_record(success_path) != dict(success_record):
        raise HandoffContractError("provisional Full bridge source bytes drifted")
    expected = _build_bridge(
        canonical_full_plan_path=plan_path,
        provisional_success_path=success_path,
        predictor=predictor,
    )
    if value != expected:
        raise HandoffContractError(
            "provisional Full bridge authority recomputation drifted"
        )
    return {
        "schema_version": AUTHENTICATED_BRIDGE_SCHEMA,
        "bridge": value,
        "canonical_full_plan_path": str(plan_path),
        "provisional_full_success_path": str(success_path),
        "production_eligible": value["production_eligible"],
    }


def create_bridge(
    *,
    canonical_full_plan_path: Path,
    provisional_success_path: Path,
    output: Path,
    predictor: Any | None = None,
) -> Path:
    """Create an immutable bridge after complete local reauthentication."""
    target = output.resolve()
    expected = _build_bridge(
        canonical_full_plan_path=canonical_full_plan_path,
        provisional_success_path=provisional_success_path,
        predictor=predictor,
    )
    if target.exists():
        authenticated = authenticate_bridge(target, predictor=predictor)
        if authenticated["bridge"] != expected:
            raise HandoffContractError("existing provisional Full bridge differs")
        return target
    production._write_immutable_json(target, expected)
    authenticate_bridge(target, predictor=predictor)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explicit read-only canonical-plan/provisional-Full promotion bridge"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--canonical-full-plan", type=Path, required=True)
    create.add_argument("--provisional-success", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--bridge", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        path = create_bridge(
            canonical_full_plan_path=args.canonical_full_plan,
            provisional_success_path=args.provisional_success,
            output=args.output,
        )
        value: Any = {
            "path": str(path),
            "payload_sha256": authenticate_bridge(path)["bridge"]["payload_sha256"],
        }
    else:
        value = authenticate_bridge(args.bridge)
    print(production._json_bytes(value).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
