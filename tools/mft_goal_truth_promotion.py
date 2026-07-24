"""Actual-truth promotion from diagnostic Standard FEA to bounded Full FEA.

This is deliberately separate from both ``mft_goal_diagnostic_standard_probe``
and the production Pareto handoff.  The v2 path exact-binds 24 diagnostic
submissions, reauthenticates and classifies all 24 collections, and ranks only
the feasible measured Standard volume/loss observations.  At most three
global rank-0 candidates may receive explicit Full plans.  No surrogate
constraint is relaxed and no automatic promotion exists.  The bounded v1
reader remains available for existing evidence.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    TEMPERATURE_TARGET_LIMITS_C,
    attest_fixed_identity,
    canonical_sha256,
    validate_goal_stage_spec,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402


TRUTH_MANIFEST_SCHEMA = "mft-goal-actual-truth-pareto-v1"
TRUTH_MANIFEST_SCHEMA_V2 = "mft-goal-actual-truth-pareto-v2"
COHORT_INVENTORY_SCHEMA = "mft-goal-diagnostic-standard-cohort-v1"
FULL_PLAN_SET_SCHEMA = "mft-goal-truth-full-plan-set-v1"
FULL_PLAN_SCHEMA = "mft-goal-truth-full-plan-v1"
FULL_SUBMISSION_SCHEMA = "mft-goal-truth-full-submission-v1"
FULL_COLLECTION_SCHEMA = "mft-goal-truth-full-collection-v1"
PACKAGE_SCHEMA = "mft-goal-truth-full-package-v1"
FULL_PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_truth_promotion_full.json"
)
FULL_RESOURCES = {"cpus": 16, "timeout_seconds": 12 * 3600}
MAX_PROMOTION_INPUTS = 12
EXACT_COHORT_SIZE = 24
MAX_FULL_PLANS = 3
DIAGNOSTIC_COLLECTION_SCHEMA = diagnostic.COLLECTION_SCHEMA
HandoffContractError = production.HandoffContractError


def _authority_flags() -> dict[str, bool]:
    return {
        "actual_truth_only_promotion": True,
        "surrogate_authority_used": False,
        "surrogate_constraint_relaxed": False,
        "automatic_promotion": False,
        "diagnostic_cli_full_enabled": False,
        "production_handoff_schema_used": False,
        "no_mixed_model_authority": True,
    }


def _file_record(path: Path) -> dict[str, Any]:
    return production._file_record(path.resolve(strict=True))


def _immutable_csv(path: Path, frame: pd.DataFrame) -> Path:
    if path.exists():
        raise HandoffContractError(f"CSV output already exists: {path}")
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    os.chmod(path, 0o444)
    return path


def _full_profile() -> tuple[dict[str, Any], dict[str, Any]]:
    profile = production._read_json(FULL_PROFILE_PATH.resolve(strict=True))
    reviewed, _record = production._profile_content("full")
    retention = profile.get("artifact_retention")
    if (
        set(profile) != set(reviewed)
        or profile.get("schema_version")
        != "mft-goal-truth-promotion-full-profile-v1"
        or profile.get("stage") != "full"
        or profile.get("reviewed_solver_path")
        != reviewed["reviewed_solver_path"]
        or profile.get("cli_flags") != reviewed["cli_flags"]
        or profile.get("param_overrides") != reviewed["param_overrides"]
        or profile.get("fixed_boundary_contract")
        != reviewed["fixed_boundary_contract"]
        or profile.get("mem_mb") != reviewed["mem_mb"]
        or profile.get("cpus") != 16
        or profile.get("timeout_seconds") != 12 * 3600
        or retention
        != {
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
            "marker_filename": (
                scheduler_client.SCHEDULER_PRESERVE_MARKER
            ),
            "retention_required": True,
            "prune_protection_required": True,
        }
    ):
        raise HandoffContractError(
            "truth-promotion Full profile drifted from reviewed Full physics"
        )
    identity = dict(profile["param_overrides"])
    identity["thermal_pad_conductivity_W_mK"] = 0.2
    attest_fixed_identity(identity)
    return profile, _file_record(FULL_PROFILE_PATH)


def _temperature_evidence(
    result: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], bool]:
    active, evidence, passed = production._temperature_gate_evidence(result)
    required = {
        "T_max_Tx",
        "T_max_Rx_main",
        "T_max_core",
    }
    if production._finite(result.get("N2_side"), "N2_side") > 0:
        required.add("T_max_Rx_side")
    if not required.issubset(active):
        raise HandoffContractError(
            "actual temperature evidence omits an active body target"
        )
    for name, item in evidence.items():
        if (
            item.get("limit_C")
            != float(TEMPERATURE_TARGET_LIMITS_C[name])
            or item.get("passed")
            is not (
                production._finite(item.get("actual_C"), name)
                <= float(TEMPERATURE_TARGET_LIMITS_C[name])
            )
        ):
            raise HandoffContractError(
                f"actual temperature evidence drifted: {name}"
            )
    return active, evidence, passed


def _actual_standard_observation(
    view: Mapping[str, Any],
    *,
    collection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(view) != {
        "schema_version",
        "collection",
        "plan",
        "params",
        "selected",
        "submission",
    }:
        raise HandoffContractError(
            "diagnostic authenticated collection view fields drifted"
        )
    collection = view["collection"]
    plan = view["plan"]
    selected = view["selected"]
    submission = view["submission"]
    result = collection.get("result")
    truth = collection.get("truth_evidence")
    if (
        view.get("schema_version")
        != diagnostic.AUTHENTICATED_COLLECTION_SCHEMA
        or collection.get("schema_version")
        != DIAGNOSTIC_COLLECTION_SCHEMA
        or not isinstance(result, dict)
        or not isinstance(truth, dict)
        or any(
            collection.get(name) is not value
            for name, value in diagnostic._diagnostic_flags().items()
        )
    ):
        raise HandoffContractError(
            "diagnostic Standard collection truth fields drifted"
        )
    reasons = production._goal_result_reasons(result, selected)
    if (
        reasons != collection.get("goal_physical_spec_reasons")
        or collection.get("goal_physical_spec_passed") is not (not reasons)
    ):
        raise HandoffContractError(
            "Standard actual physical goal recomputation failed"
        )
    active, temperatures, temperature_passed = _temperature_evidence(result)
    if (
        active != collection.get("active_temperature_targets")
        or temperatures
        != collection.get("actual_body_probe_temperatures")
        or collection.get(
            "actual_body_probe_temperature_gate_passed"
        )
        is not temperature_passed
    ):
        raise HandoffContractError(
            "Standard actual temperature gate recomputation failed"
        )
    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        production._finite(value, "actual exterior dimension")
        for value in dimensions
    )
    limits = GOAL_SIZE_LIMITS_MM
    resonance = production._finite(
        result.get("f_res_min_tx_rx_only_Hz"), "actual resonance"
    )
    losses = {
        name: production._finite(result.get(name), f"actual {name}")
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    }
    total_loss = sum(losses.values())
    identity_values = dict(result)
    identity_values["thermal_pad_conductivity_W_mK"] = 0.2
    fixed = attest_fixed_identity(identity_values)
    expected_dimensions = truth.get("actual_exterior_dimensions_mm")
    if (
        not isinstance(expected_dimensions, dict)
        or any(
            not math.isclose(
                value,
                production._finite(
                    expected_dimensions.get(axis),
                    f"truth dimension {axis}",
                ),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for axis, value in (
                ("W", width),
                ("L", length),
                ("H", height),
            )
        )
        or not math.isclose(
            float(volume_l),
            production._finite(
                truth.get("actual_volume_L"), "truth actual volume"
            ),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not math.isclose(
            total_loss,
            production._finite(
                truth.get("actual_total_loss_W"), "truth actual loss"
            ),
            rel_tol=1e-12,
            abs_tol=1e-6,
        )
        or not math.isclose(
            resonance,
            production._finite(
                truth.get("actual_resonance_Hz"),
                "truth actual resonance",
            ),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or fixed != truth.get("actual_fixed_identity_attestation")
        or fixed
        != selected["row_contract"]["fixed_identity_attestation"]
        or plan.get("solver_revision")
        != submission.get("solver_revision")
        or plan.get("solver_revision")
        != collection["result_identity"]["solver_revision"]
        or plan.get("library_revision")
        != submission.get("library_revision")
        or plan.get("library_revision")
        != collection["result_identity"]["library_revision"]
        or result.get("git_hash") != plan.get("solver_revision")
        or result.get("pyaedt_library_git_hash")
        != plan.get("library_revision")
        or production._integer(result.get("full_model"), "full_model") != 0
        or str(result.get("thermal_symmetry") or "") != "eighth"
    ):
        raise HandoffContractError(
            "Standard actual truth/provenance identity drifted"
        )
    margins = {
        "width_mm": float(limits["W"]) - width,
        "length_mm": float(limits["L"]) - length,
        "height_mm": float(limits["H"]) - height,
        "resonance_Hz": resonance
        - float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
        "temperature_C": {
            name: float(item["limit_C"])
            - production._finite(item["actual_C"], name)
            for name, item in temperatures.items()
        },
    }
    truth_row = {
        "collection": _file_record(collection_path),
        "collection_payload_sha256": collection["payload_sha256"],
        "candidate_physics_sha256": collection[
            "candidate_physics_sha256"
        ],
        "source_task_payload_sha256": collection[
            "selected_candidate_identity"
        ]["source_task_payload_sha256"],
        "standard_result_sha256": collection["result_sha256"],
        "standard_task_id": collection["task_id"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "fea_params_sha256": plan["fea_params_sha256"],
        "actual_volume_L": float(volume_l),
        "actual_total_loss_W": total_loss,
        "actual_loss_components_W": losses,
        "actual_dimensions_mm": {
            "W": width,
            "L": length,
            "H": height,
        },
        "actual_resonance_Hz": resonance,
        "actual_temperatures_C": {
            name: production._finite(item["actual_C"], name)
            for name, item in temperatures.items()
        },
        "actual_constraint_margins": margins,
        "fixed_identity_attestation": fixed,
        "standard_retained_bundle": copy.deepcopy(
            collection["remote_aedt_bundle_receipt"]
        ),
        "standard_results_manifest_sha256": collection[
            "aedtresults_manifest_sha256"
        ],
        "standard_prune_marker_sha256": collection[
            "remote_aedt_bundle_receipt"
        ]["marker_sha256"],
        **_authority_flags(),
    }
    exclusion_reasons = list(reasons)
    if not temperature_passed:
        exclusion_reasons.append(
            "actual_body_probe_temperature_gate_failed"
        )
    status = {
        "goal_physical_spec_passed": not reasons,
        "goal_physical_spec_reasons": list(reasons),
        "actual_body_probe_temperature_gate_passed": temperature_passed,
        "failed_temperature_targets": sorted(
            name
            for name, item in temperatures.items()
            if item["passed"] is not True
        ),
        "actual_truth_feasible": not reasons and temperature_passed,
        "promotion_exclusion_reasons": list(
            dict.fromkeys(exclusion_reasons)
        ),
    }
    return truth_row, status


def _actual_standard_truth(
    view: Mapping[str, Any],
    *,
    collection_path: Path,
) -> dict[str, Any]:
    truth, status = _actual_standard_observation(
        view, collection_path=collection_path
    )
    if status["actual_truth_feasible"] is not True:
        raise HandoffContractError(
            "only passing diagnostic Standard collections may be promoted"
        )
    return truth


COHORT_ENTRY_FIELDS = frozenset(
    {
        "entry_sha256",
        "task_id",
        "task_name",
        "candidate_physics_sha256",
        "source_task_payload_sha256",
        "source_result_sha256",
        "selection_manifest_sha256",
        "plan",
        "plan_payload_sha256",
        "submission",
        "submission_payload_sha256",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "search_authority_sha256",
    }
)

COHORT_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "exact_cohort_size",
        "entry_count",
        "unique_candidate_count",
        "solver_revision",
        "library_revision",
        "cohort_entries_sha256",
        "entries",
        "scheduler_mutation_performed",
        "collection_performed",
        "truth_promotion_performed",
    }
)


def _authenticated_submission_entry(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    unsigned = production._validate_seal(
        production._read_json(resolved), diagnostic.SUBMISSION_SCHEMA
    )
    plan_record = unsigned.get("plan")
    if not isinstance(plan_record, dict):
        raise HandoffContractError(
            "diagnostic cohort submission plan record is absent"
        )
    plan_path = Path(str(plan_record.get("path") or ""))
    if _file_record(plan_path) != plan_record:
        raise HandoffContractError(
            "diagnostic cohort submission plan bytes drifted"
        )
    plan, _params, selected = diagnostic._load_plan(plan_path)
    submission = diagnostic._load_submission(resolved, plan=plan)
    refreshed = diagnostic._fresh_selection_reauthentication(
        plan=plan, selected=selected, predictor=predictor
    )
    if refreshed != submission["search_authority_reauthentication"]:
        raise HandoffContractError(
            "diagnostic cohort submission source reauthentication drifted"
        )
    entry = {
        "task_id": submission["task_id"],
        "task_name": submission["task_name"],
        "candidate_physics_sha256": plan[
            "candidate_physics_sha256"
        ],
        "source_task_payload_sha256": selected["task_identity"][
            "payload_sha256"
        ],
        "source_result_sha256": selected["source_result"]["sha256"],
        "selection_manifest_sha256": selected["selection_source"][
            "selection_manifest"
        ]["sha256"],
        "plan": _file_record(plan_path),
        "plan_payload_sha256": plan["payload_sha256"],
        "submission": _file_record(resolved),
        "submission_payload_sha256": submission["payload_sha256"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "fea_params_sha256": plan["fea_params_sha256"],
        "search_authority_sha256": plan["search_authority_sha256"],
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def _validated_cohort_entries(
    submission_paths: Sequence[Path],
    *,
    predictor: Any | None = None,
) -> list[dict[str, Any]]:
    if len(submission_paths) != EXACT_COHORT_SIZE:
        raise HandoffContractError(
            "diagnostic Standard cohort requires exactly 24 submissions"
        )
    try:
        resolved = [path.resolve(strict=True) for path in submission_paths]
    except (OSError, RuntimeError) as exc:
        raise HandoffContractError(
            "diagnostic Standard cohort submission is unavailable"
        ) from exc
    if len(set(resolved)) != EXACT_COHORT_SIZE:
        raise HandoffContractError(
            "diagnostic Standard cohort submissions must be unique"
        )
    entries = sorted(
        (
            _authenticated_submission_entry(path, predictor=predictor)
            for path in resolved
        ),
        key=lambda item: (
            item["task_id"],
            item["candidate_physics_sha256"],
            item["submission_payload_sha256"],
        ),
    )
    if len({entry["task_id"] for entry in entries}) != EXACT_COHORT_SIZE:
        raise HandoffContractError(
            "diagnostic Standard cohort task IDs must be unique"
        )
    revisions = {
        (entry["solver_revision"], entry["library_revision"])
        for entry in entries
    }
    if len(revisions) != 1:
        raise HandoffContractError(
            "diagnostic Standard cohort cannot mix solver/library provenance"
        )
    return entries


def create_cohort_inventory(
    *,
    standard_submission_paths: Sequence[Path],
    output: Path,
    predictor: Any | None = None,
) -> Path:
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    entries = _validated_cohort_entries(
        standard_submission_paths, predictor=predictor
    )
    solver_revision, library_revision = next(
        iter(
            {
                (entry["solver_revision"], entry["library_revision"])
                for entry in entries
            }
        )
    )
    inventory = production._seal(
        {
            "schema_version": COHORT_INVENTORY_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (
                GOAL_TEMPERATURE_CONTRACT_SHA256
            ),
            "exact_cohort_size": EXACT_COHORT_SIZE,
            "entry_count": len(entries),
            "unique_candidate_count": len(
                {
                    entry["candidate_physics_sha256"]
                    for entry in entries
                }
            ),
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "cohort_entries_sha256": canonical_sha256(entries),
            "entries": entries,
            "scheduler_mutation_performed": False,
            "collection_performed": False,
            "truth_promotion_performed": False,
        }
    )
    return production._write_immutable_json(output.resolve(), inventory)


def _load_cohort_inventory(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = path.resolve(strict=True)
    inventory = production._validate_seal(
        production._read_json(resolved), COHORT_INVENTORY_SCHEMA
    )
    records = inventory.get("entries")
    if (
        set(inventory) != COHORT_INVENTORY_FIELDS
        or inventory.get("campaign_id") != "mft-goal-20260726"
        or inventory.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or inventory.get("hard_spec") != GOAL_STAGE_SPEC
        or inventory.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or inventory.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or inventory.get("exact_cohort_size") != EXACT_COHORT_SIZE
        or inventory.get("entry_count") != EXACT_COHORT_SIZE
        or not isinstance(records, list)
        or len(records) != EXACT_COHORT_SIZE
        or inventory.get("scheduler_mutation_performed") is not False
        or inventory.get("collection_performed") is not False
        or inventory.get("truth_promotion_performed") is not False
    ):
        raise HandoffContractError(
            "diagnostic Standard cohort inventory contract drifted"
        )
    submission_paths = []
    for record in records:
        if not isinstance(record, dict) or set(record) != COHORT_ENTRY_FIELDS:
            raise HandoffContractError(
                "diagnostic Standard cohort entry fields drifted"
            )
        unsigned = {
            name: value
            for name, value in record.items()
            if name != "entry_sha256"
        }
        if record.get("entry_sha256") != canonical_sha256(unsigned):
            raise HandoffContractError(
                "diagnostic Standard cohort entry seal drifted"
            )
        submission_record = record.get("submission")
        if not isinstance(submission_record, dict):
            raise HandoffContractError(
                "diagnostic Standard cohort submission record is malformed"
            )
        submission_path = Path(
            str(submission_record.get("path") or "")
        )
        if _file_record(submission_path) != submission_record:
            raise HandoffContractError(
                "diagnostic Standard cohort submission bytes drifted"
            )
        submission_paths.append(submission_path)
    refreshed = _validated_cohort_entries(
        submission_paths, predictor=predictor
    )
    if (
        records != refreshed
        or inventory.get("cohort_entries_sha256")
        != canonical_sha256(refreshed)
        or inventory.get("unique_candidate_count")
        != len(
            {
                entry["candidate_physics_sha256"]
                for entry in refreshed
            }
        )
        or inventory.get("solver_revision")
        != refreshed[0]["solver_revision"]
        or inventory.get("library_revision")
        != refreshed[0]["library_revision"]
    ):
        raise HandoffContractError(
            "diagnostic Standard cohort inventory reauthentication drifted"
        )
    return inventory, refreshed


def _authenticate_inputs(
    paths: Sequence[Path],
    *,
    predictor: Any | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if (
        not 1 <= len(paths) <= MAX_PROMOTION_INPUTS
        or len({path.resolve(strict=True) for path in paths}) != len(paths)
    ):
        raise HandoffContractError(
            "truth promotion requires 1..12 unique diagnostic collections"
        )
    authenticated = []
    for path in sorted(
        (item.resolve(strict=True) for item in paths),
        key=lambda item: str(item).casefold(),
    ):
        view = diagnostic.authenticate_collection(path, predictor=predictor)
        truth = _actual_standard_truth(view, collection_path=path)
        authenticated.append((view, truth))
    revisions = {
        (truth["solver_revision"], truth["library_revision"])
        for _view, truth in authenticated
    }
    if len(revisions) != 1:
        raise HandoffContractError(
            "truth promotion cannot mix solver/library provenance"
        )
    return authenticated


COHORT_CLASSIFICATION_FIELDS = frozenset(
    {
        "classification_sha256",
        "cohort_entry_sha256",
        "task_id",
        "candidate_physics_sha256",
        "source_task_payload_sha256",
        "plan_payload_sha256",
        "submission_payload_sha256",
        "collection",
        "collection_payload_sha256",
        "standard_result_sha256",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "goal_physical_spec_passed",
        "goal_physical_spec_reasons",
        "actual_body_probe_temperature_gate_passed",
        "failed_temperature_targets",
        "actual_truth_feasible",
        "promotion_exclusion_reasons",
        "actual_volume_L",
        "actual_total_loss_W",
        "actual_dimensions_mm",
        "actual_resonance_Hz",
        "actual_temperatures_C",
        "actual_constraint_margins",
    }
)


def _cohort_classification_row(
    *,
    entry: Mapping[str, Any],
    truth: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "cohort_entry_sha256": entry["entry_sha256"],
        "task_id": truth["standard_task_id"],
        "candidate_physics_sha256": truth[
            "candidate_physics_sha256"
        ],
        "source_task_payload_sha256": truth[
            "source_task_payload_sha256"
        ],
        "plan_payload_sha256": entry["plan_payload_sha256"],
        "submission_payload_sha256": entry[
            "submission_payload_sha256"
        ],
        "collection": copy.deepcopy(truth["collection"]),
        "collection_payload_sha256": truth[
            "collection_payload_sha256"
        ],
        "standard_result_sha256": truth["standard_result_sha256"],
        "solver_revision": truth["solver_revision"],
        "library_revision": truth["library_revision"],
        "fea_params_sha256": truth["fea_params_sha256"],
        "goal_physical_spec_passed": status[
            "goal_physical_spec_passed"
        ],
        "goal_physical_spec_reasons": copy.deepcopy(
            status["goal_physical_spec_reasons"]
        ),
        "actual_body_probe_temperature_gate_passed": status[
            "actual_body_probe_temperature_gate_passed"
        ],
        "failed_temperature_targets": copy.deepcopy(
            status["failed_temperature_targets"]
        ),
        "actual_truth_feasible": status["actual_truth_feasible"],
        "promotion_exclusion_reasons": copy.deepcopy(
            status["promotion_exclusion_reasons"]
        ),
        "actual_volume_L": truth["actual_volume_L"],
        "actual_total_loss_W": truth["actual_total_loss_W"],
        "actual_dimensions_mm": copy.deepcopy(
            truth["actual_dimensions_mm"]
        ),
        "actual_resonance_Hz": truth["actual_resonance_Hz"],
        "actual_temperatures_C": copy.deepcopy(
            truth["actual_temperatures_C"]
        ),
        "actual_constraint_margins": copy.deepcopy(
            truth["actual_constraint_margins"]
        ),
    }
    row["classification_sha256"] = canonical_sha256(row)
    return row


def _cohort_authority_task_id(
    view: Mapping[str, Any],
    *,
    expected_by_task: Mapping[int, Mapping[str, Any]],
) -> int:
    """Resolve one execution onto its immutable original cohort slot.

    A normal Standard execution is its own cohort authority.  A reviewed
    timeout retry may stand in for the original task only when the retry's
    sealed ancestry reauthenticates the exact original plan, submission, and
    task already frozen in the cohort inventory.
    """
    collection = view.get("collection")
    plan = view.get("plan")
    submission = view.get("submission")
    if (
        not isinstance(collection, Mapping)
        or not isinstance(plan, Mapping)
        or not isinstance(submission, Mapping)
    ):
        raise HandoffContractError(
            "truth promotion v2 collection is outside or duplicates the "
            "exact cohort because its authenticated view is malformed"
        )
    execution_task_id = collection.get("task_id")
    if (
        isinstance(execution_task_id, bool)
        or not isinstance(execution_task_id, int)
        or execution_task_id <= 0
    ):
        raise HandoffContractError(
            "truth promotion v2 execution task ID is invalid"
        )
    if not diagnostic._plan_is_timeout_retry(plan):
        return execution_task_id

    retry_record = plan.get("retry_of_timeout")
    if not isinstance(retry_record, Mapping):
        raise HandoffContractError(
            "truth promotion v2 timeout retry ancestry is absent"
        )
    authority_task_id = retry_record.get("retry_of_task_id")
    entry = expected_by_task.get(authority_task_id)
    if entry is None:
        raise HandoffContractError(
            "truth promotion v2 timeout retry is outside the exact cohort"
        )
    original_plan, original_submission, _execution = (
        diagnostic._validate_timeout_retry_record(plan)
    )
    if (
        execution_task_id == authority_task_id
        or submission.get("task_id") != execution_task_id
        or retry_record.get("original_plan") != entry["plan"]
        or retry_record.get("original_plan_payload_sha256")
        != entry["plan_payload_sha256"]
        or retry_record.get("original_submission") != entry["submission"]
        or retry_record.get("original_submission_payload_sha256")
        != entry["submission_payload_sha256"]
        or original_plan.get("payload_sha256")
        != entry["plan_payload_sha256"]
        or original_submission.get("payload_sha256")
        != entry["submission_payload_sha256"]
        or original_submission.get("task_id") != entry["task_id"]
        or original_submission.get("task_name") != entry["task_name"]
        or plan.get("candidate_physics_sha256")
        != entry["candidate_physics_sha256"]
        or plan.get("solver_revision") != entry["solver_revision"]
        or plan.get("library_revision") != entry["library_revision"]
        or plan.get("fea_params_sha256") != entry["fea_params_sha256"]
        or plan.get("search_authority_sha256")
        != entry["search_authority_sha256"]
    ):
        raise HandoffContractError(
            "truth promotion v2 timeout retry/cohort ancestry drifted"
        )
    return int(authority_task_id)


def _authenticate_cohort_inputs(
    *,
    cohort_entries: Sequence[Mapping[str, Any]],
    collection_paths: Sequence[Path],
    predictor: Any | None = None,
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]
]:
    if (
        len(cohort_entries) != EXACT_COHORT_SIZE
        or len(collection_paths) != EXACT_COHORT_SIZE
    ):
        raise HandoffContractError(
            "truth promotion v2 requires exactly 24 cohort collections"
        )
    try:
        resolved = [path.resolve(strict=True) for path in collection_paths]
    except (OSError, RuntimeError) as exc:
        raise HandoffContractError(
            "truth promotion v2 collection is unavailable"
        ) from exc
    if len(set(resolved)) != EXACT_COHORT_SIZE:
        raise HandoffContractError(
            "truth promotion v2 collections must be unique"
        )
    expected_by_task = {
        entry["task_id"]: dict(entry) for entry in cohort_entries
    }
    if len(expected_by_task) != EXACT_COHORT_SIZE:
        raise HandoffContractError(
            "truth promotion v2 cohort task inventory is ambiguous"
        )
    observed_by_task = {}
    observed_execution_task_ids = set()
    for path in resolved:
        view = diagnostic.authenticate_collection(
            path, predictor=predictor
        )
        collection = view["collection"]
        execution_task_id = collection.get("task_id")
        authority_task_id = _cohort_authority_task_id(
            view, expected_by_task=expected_by_task
        )
        if (
            authority_task_id not in expected_by_task
            or authority_task_id in observed_by_task
            or execution_task_id in observed_execution_task_ids
        ):
            raise HandoffContractError(
                "truth promotion v2 collection is outside or duplicates "
                "the exact cohort"
            )
        entry = expected_by_task[authority_task_id]
        timeout_retry = diagnostic._plan_is_timeout_retry(view["plan"])
        truth, status = _actual_standard_observation(
            view, collection_path=path
        )
        if (
            collection.get("candidate_physics_sha256")
            != entry["candidate_physics_sha256"]
            or collection.get("selected_candidate_identity", {}).get(
                "source_task_payload_sha256"
            )
            != entry["source_task_payload_sha256"]
            or collection.get("selection_manifest", {}).get("sha256")
            != entry["selection_manifest_sha256"]
            or view["plan"].get("candidate_physics_sha256")
            != entry["candidate_physics_sha256"]
            or view["plan"].get("solver_revision")
            != entry["solver_revision"]
            or view["plan"].get("library_revision")
            != entry["library_revision"]
            or view["plan"].get("fea_params_sha256")
            != entry["fea_params_sha256"]
            or view["plan"].get("search_authority_sha256")
            != entry["search_authority_sha256"]
            or truth["standard_task_id"] != execution_task_id
            or truth["solver_revision"] != entry["solver_revision"]
            or truth["library_revision"] != entry["library_revision"]
            or truth["fea_params_sha256"]
            != entry["fea_params_sha256"]
            or (
                not timeout_retry
                and (
                    collection.get("plan") != entry["plan"]
                    or collection.get("plan_payload_sha256")
                    != entry["plan_payload_sha256"]
                    or collection.get("submission") != entry["submission"]
                    or collection.get("submission_payload_sha256")
                    != entry["submission_payload_sha256"]
                    or view["plan"].get("payload_sha256")
                    != entry["plan_payload_sha256"]
                    or view["submission"].get("payload_sha256")
                    != entry["submission_payload_sha256"]
                    or view["submission"].get("task_name")
                    != entry["task_name"]
                )
            )
        ):
            raise HandoffContractError(
                "truth promotion v2 collection/cohort identity drifted"
            )
        classification = _cohort_classification_row(
            entry=entry, truth=truth, status=status
        )
        observed_by_task[authority_task_id] = (
            view,
            truth,
            classification,
            entry,
        )
        observed_execution_task_ids.add(execution_task_id)
    if set(observed_by_task) != set(expected_by_task):
        raise HandoffContractError(
            "truth promotion v2 exact cohort is incomplete"
        )
    authenticated = [
        observed_by_task[entry["task_id"]] for entry in cohort_entries
    ]
    revisions = {
        (truth["solver_revision"], truth["library_revision"])
        for _view, truth, _classification, _entry in authenticated
    }
    if len(revisions) != 1:
        raise HandoffContractError(
            "truth promotion v2 cannot mix solver/library provenance"
        )
    return authenticated


def _duplicate_observation_contract(
    truth: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_physics_sha256": truth[
            "candidate_physics_sha256"
        ],
        "solver_revision": truth["solver_revision"],
        "library_revision": truth["library_revision"],
        "fea_params_sha256": truth["fea_params_sha256"],
        "actual_volume_L": truth["actual_volume_L"],
        "actual_total_loss_W": truth["actual_total_loss_W"],
        "actual_loss_components_W": truth["actual_loss_components_W"],
        "actual_dimensions_mm": truth["actual_dimensions_mm"],
        "actual_resonance_Hz": truth["actual_resonance_Hz"],
        "actual_temperatures_C": truth["actual_temperatures_C"],
        "actual_constraint_margins": truth[
            "actual_constraint_margins"
        ],
        "fixed_identity_attestation": truth[
            "fixed_identity_attestation"
        ],
        "goal_physical_spec_passed": classification[
            "goal_physical_spec_passed"
        ],
        "goal_physical_spec_reasons": classification[
            "goal_physical_spec_reasons"
        ],
        "actual_body_probe_temperature_gate_passed": classification[
            "actual_body_probe_temperature_gate_passed"
        ],
        "failed_temperature_targets": classification[
            "failed_temperature_targets"
        ],
        "actual_truth_feasible": classification[
            "actual_truth_feasible"
        ],
        "promotion_exclusion_reasons": classification[
            "promotion_exclusion_reasons"
        ],
    }


def _validate_duplicate_cohort_observations(
    authenticated: Sequence[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ],
) -> None:
    groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for _view, truth, classification, _entry in authenticated:
        groups.setdefault(
            truth["candidate_physics_sha256"], []
        ).append((truth, classification))
    for candidate, group in groups.items():
        contracts = {
            canonical_sha256(
                _duplicate_observation_contract(truth, classification)
            )
            for truth, classification in group
        }
        if len(contracts) != 1:
            raise HandoffContractError(
                "duplicate cohort candidate has mixed actual truth: "
                f"{candidate}"
            )


def _deduplicate_truth(
    authenticated: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for _view, truth in authenticated:
        groups.setdefault(truth["candidate_physics_sha256"], []).append(
            truth
        )
    deduplicated = []
    for candidate in sorted(groups):
        group = groups[candidate]
        params = {item["fea_params_sha256"] for item in group}
        revisions = {
            (item["solver_revision"], item["library_revision"])
            for item in group
        }
        if len(params) != 1 or len(revisions) != 1:
            raise HandoffContractError(
                "duplicate candidate has mixed physics/provenance"
            )
        chosen = min(
            group,
            key=lambda item: (
                item["collection_payload_sha256"],
                item["standard_result_sha256"],
            ),
        )
        row = copy.deepcopy(chosen)
        row["duplicate_collection_count"] = len(group)
        row["deduplicated_collection_payload_sha256"] = sorted(
            item["collection_payload_sha256"] for item in group
        )
        deduplicated.append(row)
    return deduplicated


def _rank_truth(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    remaining = set(range(len(rows)))
    rank = 0
    ranked: list[dict[str, Any]] = []
    while remaining:
        front = []
        for index in sorted(remaining):
            candidate = rows[index]
            dominated = False
            for other_index in remaining:
                if other_index == index:
                    continue
                other = rows[other_index]
                no_worse = (
                    other["actual_volume_L"]
                    <= candidate["actual_volume_L"]
                    and other["actual_total_loss_W"]
                    <= candidate["actual_total_loss_W"]
                )
                strictly_better = (
                    other["actual_volume_L"]
                    < candidate["actual_volume_L"]
                    or other["actual_total_loss_W"]
                    < candidate["actual_total_loss_W"]
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(index)
        if not front:
            raise HandoffContractError("truth non-dominated sorting stalled")
        ordered = sorted(
            front,
            key=lambda index: (
                rows[index]["actual_volume_L"],
                rows[index]["actual_total_loss_W"],
                rows[index]["candidate_physics_sha256"],
            ),
        )
        for order, index in enumerate(ordered):
            row = copy.deepcopy(dict(rows[index]))
            row["truth_non_dominated_rank"] = rank
            row["truth_front_order"] = order
            row["truth_rank0"] = rank == 0
            row["truth_row_sha256"] = canonical_sha256(row)
            ranked.append(row)
        remaining.difference_update(front)
        rank += 1
    return sorted(
        ranked,
        key=lambda row: (
            row["truth_non_dominated_rank"],
            row["truth_front_order"],
            row["candidate_physics_sha256"],
        ),
    )


TRUTH_CSV_COLUMNS = (
    "truth_non_dominated_rank",
    "truth_front_order",
    "truth_rank0",
    "candidate_physics_sha256",
    "collection_payload_sha256",
    "standard_result_sha256",
    "standard_task_id",
    "solver_revision",
    "library_revision",
    "fea_params_sha256",
    "actual_volume_L",
    "actual_total_loss_W",
    "duplicate_collection_count",
    "actual_dimensions_mm_json",
    "actual_resonance_Hz",
    "actual_temperatures_C_json",
    "actual_constraint_margins_json",
    "truth_row_sha256",
)


def _truth_csv_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    output = []
    for row in rows:
        output.append(
            {
                name: (
                    json.dumps(
                        row[name[:-5]],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if name.endswith("_json")
                    else row[name]
                )
                for name in TRUTH_CSV_COLUMNS
            }
        )
    return pd.DataFrame(output, columns=TRUTH_CSV_COLUMNS)


def create_truth_promotion(
    *,
    standard_collection_paths: Sequence[Path],
    output: Path,
    predictor: Any | None = None,
    cohort_inventory_path: Path | None = None,
) -> Path:
    if cohort_inventory_path is not None:
        return _create_truth_promotion_v2(
            cohort_inventory_path=cohort_inventory_path,
            standard_collection_paths=standard_collection_paths,
            output=output,
            predictor=predictor,
        )
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    authenticated = _authenticate_inputs(
        standard_collection_paths, predictor=predictor
    )
    deduplicated = _deduplicate_truth(authenticated)
    ranked = _rank_truth(deduplicated)
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"truth promotion output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        csv_path = _immutable_csv(
            staging / "truth_validated_pareto_front.csv",
            _truth_csv_frame(ranked),
        )
        solver_revision = ranked[0]["solver_revision"]
        library_revision = ranked[0]["library_revision"]
        manifest = production._seal(
            {
                "schema_version": TRUTH_MANIFEST_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "input_collection_count": len(authenticated),
                "deduplicated_candidate_count": len(ranked),
                "rank0_count": sum(
                    row["truth_non_dominated_rank"] == 0 for row in ranked
                ),
                "solver_revision": solver_revision,
                "library_revision": library_revision,
                "input_collections": [
                    truth["collection"]
                    for _view, truth in authenticated
                ],
                "input_collection_payload_sha256": [
                    truth["collection_payload_sha256"]
                    for _view, truth in authenticated
                ],
                "ranked_rows": ranked,
                "truth_validated_pareto_front": {
                    "path": csv_path.name,
                    "sha256": production._sha256_file(csv_path),
                    "size_bytes": csv_path.stat().st_size,
                    "row_count": len(ranked),
                    "columns": list(TRUTH_CSV_COLUMNS),
                },
                "sorting_authority": (
                    "all_reauthenticated_passing_diagnostic_standard_actual_"
                    "truth_rows_deduplicated_then_combined_nondominated_sort"
                ),
                "full_plan_default_limit": MAX_FULL_PLANS,
                "full_plan_hard_limit": MAX_FULL_PLANS,
                "scheduler_submission_performed": False,
                "full_submission_performed": False,
                **_authority_flags(),
            }
        )
        path = production._write_immutable_json(
            staging / "truth_pareto_manifest.json", manifest
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / path.name


def _create_truth_promotion_v2(
    *,
    cohort_inventory_path: Path,
    standard_collection_paths: Sequence[Path],
    output: Path,
    predictor: Any | None = None,
) -> Path:
    validate_goal_stage_spec(GOAL_STAGE_SPEC)
    inventory, cohort_entries = _load_cohort_inventory(
        cohort_inventory_path, predictor=predictor
    )
    authenticated = _authenticate_cohort_inputs(
        cohort_entries=cohort_entries,
        collection_paths=standard_collection_paths,
        predictor=predictor,
    )
    _validate_duplicate_cohort_observations(authenticated)
    feasible = [
        (view, truth)
        for view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is True
    ]
    deduplicated = _deduplicate_truth(feasible)
    ranked = _rank_truth(deduplicated)
    classifications = [
        copy.deepcopy(classification)
        for _view, _truth, classification, _entry in authenticated
    ]
    feasible_payloads = [
        truth["collection_payload_sha256"]
        for _view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is True
    ]
    infeasible_payloads = [
        truth["collection_payload_sha256"]
        for _view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is False
    ]
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"truth promotion output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        csv_path = _immutable_csv(
            staging / "truth_validated_pareto_front.csv",
            _truth_csv_frame(ranked),
        )
        manifest = production._seal(
            {
                "schema_version": TRUTH_MANIFEST_SCHEMA_V2,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "cohort_inventory": _file_record(
                    cohort_inventory_path
                ),
                "cohort_inventory_payload_sha256": inventory[
                    "payload_sha256"
                ],
                "expected_collection_count": EXACT_COHORT_SIZE,
                "authenticated_collection_count": len(authenticated),
                "feasible_collection_count": len(feasible_payloads),
                "infeasible_collection_count": len(infeasible_payloads),
                "deduplicated_feasible_candidate_count": len(ranked),
                "rank0_count": sum(
                    row["truth_non_dominated_rank"] == 0
                    for row in ranked
                ),
                "solver_revision": inventory["solver_revision"],
                "library_revision": inventory["library_revision"],
                "authenticated_collections": [
                    copy.deepcopy(truth["collection"])
                    for _view, truth, _classification, _entry
                    in authenticated
                ],
                "authenticated_collection_payload_sha256": [
                    truth["collection_payload_sha256"]
                    for _view, truth, _classification, _entry
                    in authenticated
                ],
                "classification_rows": classifications,
                "feasible_collection_payload_sha256": feasible_payloads,
                "infeasible_collection_payload_sha256": (
                    infeasible_payloads
                ),
                "ranked_rows": ranked,
                "truth_validated_pareto_front": {
                    "path": csv_path.name,
                    "sha256": production._sha256_file(csv_path),
                    "size_bytes": csv_path.stat().st_size,
                    "row_count": len(ranked),
                    "columns": list(TRUTH_CSV_COLUMNS),
                },
                "sorting_authority": (
                    "exact_24_cohort_all_reauthenticated_and_classified_"
                    "then_all_feasible_actual_truth_rows_deduplicated_"
                    "and_combined_nondominated_sort"
                ),
                "full_plan_eligible": bool(ranked),
                "zero_feasible_audited": not ranked,
                "full_plan_default_limit": MAX_FULL_PLANS,
                "full_plan_hard_limit": MAX_FULL_PLANS,
                "scheduler_submission_performed": False,
                "full_submission_performed": False,
                **_authority_flags(),
            }
        )
        path = production._write_immutable_json(
            staging / "truth_pareto_manifest.json", manifest
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / path.name


TRUTH_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "input_collection_count",
        "deduplicated_candidate_count",
        "rank0_count",
        "solver_revision",
        "library_revision",
        "input_collections",
        "input_collection_payload_sha256",
        "ranked_rows",
        "truth_validated_pareto_front",
        "sorting_authority",
        "full_plan_default_limit",
        "full_plan_hard_limit",
        "scheduler_submission_performed",
        "full_submission_performed",
        *_authority_flags(),
    }
)


TRUTH_MANIFEST_V2_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "cohort_inventory",
        "cohort_inventory_payload_sha256",
        "expected_collection_count",
        "authenticated_collection_count",
        "feasible_collection_count",
        "infeasible_collection_count",
        "deduplicated_feasible_candidate_count",
        "rank0_count",
        "solver_revision",
        "library_revision",
        "authenticated_collections",
        "authenticated_collection_payload_sha256",
        "classification_rows",
        "feasible_collection_payload_sha256",
        "infeasible_collection_payload_sha256",
        "ranked_rows",
        "truth_validated_pareto_front",
        "sorting_authority",
        "full_plan_eligible",
        "zero_feasible_audited",
        "full_plan_default_limit",
        "full_plan_hard_limit",
        "scheduler_submission_performed",
        "full_submission_performed",
        *_authority_flags(),
    }
)


def _load_truth_manifest_v1(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    resolved = path.resolve(strict=True)
    manifest = production._validate_seal(
        production._read_json(resolved), TRUTH_MANIFEST_SCHEMA
    )
    if (
        set(manifest) != TRUTH_MANIFEST_FIELDS
        or manifest.get("campaign_id") != "mft-goal-20260726"
        or manifest.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or manifest.get("hard_spec") != GOAL_STAGE_SPEC
        or manifest.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or manifest.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or manifest.get("full_plan_default_limit") != MAX_FULL_PLANS
        or manifest.get("full_plan_hard_limit") != MAX_FULL_PLANS
        or manifest.get("scheduler_submission_performed") is not False
        or manifest.get("full_submission_performed") is not False
        or any(
            manifest.get(name) is not value
            for name, value in _authority_flags().items()
        )
    ):
        raise HandoffContractError("truth Pareto manifest contract drifted")
    records = manifest.get("input_collections")
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= MAX_PROMOTION_INPUTS
    ):
        raise HandoffContractError(
            "truth Pareto input collection inventory drifted"
        )
    paths = []
    for record in records:
        if not isinstance(record, dict):
            raise HandoffContractError(
                "truth Pareto collection record is malformed"
            )
        source = Path(str(record.get("path") or ""))
        if _file_record(source) != record:
            raise HandoffContractError(
                "truth Pareto source collection bytes drifted"
            )
        paths.append(source)
    authenticated = _authenticate_inputs(paths, predictor=predictor)
    truths = [truth for _view, truth in authenticated]
    ranked = _rank_truth(_deduplicate_truth(authenticated))
    by_collection_path = {
        truth["collection"]["path"]: {
            "view": view,
            "truth": truth,
        }
        for view, truth in authenticated
    }
    by_candidate = {
        row["candidate_physics_sha256"]: by_collection_path[
            row["collection"]["path"]
        ]
        for row in ranked
    }
    artifact = manifest.get("truth_validated_pareto_front")
    if (
        manifest.get("input_collection_count") != len(authenticated)
        or manifest.get("deduplicated_candidate_count") != len(ranked)
        or manifest.get("rank0_count")
        != sum(row["truth_non_dominated_rank"] == 0 for row in ranked)
        or manifest.get("solver_revision") != ranked[0]["solver_revision"]
        or manifest.get("library_revision") != ranked[0]["library_revision"]
        or manifest.get("input_collection_payload_sha256")
        != [truth["collection_payload_sha256"] for truth in truths]
        or manifest.get("ranked_rows") != ranked
        or not isinstance(artifact, dict)
        or set(artifact)
        != {"path", "sha256", "size_bytes", "row_count", "columns"}
    ):
        raise HandoffContractError(
            "truth Pareto authority recomputation drifted"
        )
    csv_path = production._contained_file(
        resolved.parent,
        artifact.get("path"),
        "truth-validated Pareto CSV",
    )
    frame = pd.read_csv(csv_path)
    if (
        _file_record(csv_path)["sha256"] != artifact.get("sha256")
        or csv_path.stat().st_size != artifact.get("size_bytes")
        or artifact.get("row_count") != len(ranked)
        or artifact.get("columns") != list(TRUTH_CSV_COLUMNS)
        or list(frame.columns) != list(TRUTH_CSV_COLUMNS)
        or frame["candidate_physics_sha256"].astype(str).tolist()
        != [row["candidate_physics_sha256"] for row in ranked]
        or frame["truth_row_sha256"].astype(str).tolist()
        != [row["truth_row_sha256"] for row in ranked]
        or frame["truth_non_dominated_rank"].astype(int).tolist()
        != [row["truth_non_dominated_rank"] for row in ranked]
    ):
        raise HandoffContractError(
            "truth-validated Pareto CSV drifted"
        )
    return manifest, ranked, by_candidate


def _load_truth_manifest_v2(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    resolved = path.resolve(strict=True)
    manifest = production._validate_seal(
        production._read_json(resolved), TRUTH_MANIFEST_SCHEMA_V2
    )
    if (
        set(manifest) != TRUTH_MANIFEST_V2_FIELDS
        or manifest.get("campaign_id") != "mft-goal-20260726"
        or manifest.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or manifest.get("hard_spec") != GOAL_STAGE_SPEC
        or manifest.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or manifest.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or manifest.get("expected_collection_count")
        != EXACT_COHORT_SIZE
        or manifest.get("full_plan_default_limit") != MAX_FULL_PLANS
        or manifest.get("full_plan_hard_limit") != MAX_FULL_PLANS
        or manifest.get("scheduler_submission_performed") is not False
        or manifest.get("full_submission_performed") is not False
        or any(
            manifest.get(name) is not value
            for name, value in _authority_flags().items()
        )
    ):
        raise HandoffContractError(
            "truth Pareto v2 manifest contract drifted"
        )
    inventory_record = manifest.get("cohort_inventory")
    if not isinstance(inventory_record, dict):
        raise HandoffContractError(
            "truth Pareto v2 cohort inventory record is absent"
        )
    inventory_path = Path(
        str(inventory_record.get("path") or "")
    )
    if _file_record(inventory_path) != inventory_record:
        raise HandoffContractError(
            "truth Pareto v2 cohort inventory bytes drifted"
        )
    inventory, cohort_entries = _load_cohort_inventory(
        inventory_path, predictor=predictor
    )
    records = manifest.get("authenticated_collections")
    if (
        not isinstance(records, list)
        or len(records) != EXACT_COHORT_SIZE
    ):
        raise HandoffContractError(
            "truth Pareto v2 collection inventory drifted"
        )
    collection_paths = []
    for record in records:
        if not isinstance(record, dict):
            raise HandoffContractError(
                "truth Pareto v2 collection record is malformed"
            )
        source = Path(str(record.get("path") or ""))
        if _file_record(source) != record:
            raise HandoffContractError(
                "truth Pareto v2 source collection bytes drifted"
            )
        collection_paths.append(source)
    authenticated = _authenticate_cohort_inputs(
        cohort_entries=cohort_entries,
        collection_paths=collection_paths,
        predictor=predictor,
    )
    _validate_duplicate_cohort_observations(authenticated)
    classifications = [
        copy.deepcopy(classification)
        for _view, _truth, classification, _entry in authenticated
    ]
    for classification in classifications:
        if (
            set(classification) != COHORT_CLASSIFICATION_FIELDS
            or classification.get("classification_sha256")
            != canonical_sha256(
                {
                    name: value
                    for name, value in classification.items()
                    if name != "classification_sha256"
                }
            )
        ):
            raise HandoffContractError(
                "truth Pareto v2 classification row drifted"
            )
    feasible = [
        (view, truth)
        for view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is True
    ]
    ranked = _rank_truth(_deduplicate_truth(feasible))
    feasible_payloads = [
        truth["collection_payload_sha256"]
        for _view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is True
    ]
    infeasible_payloads = [
        truth["collection_payload_sha256"]
        for _view, truth, classification, _entry in authenticated
        if classification["actual_truth_feasible"] is False
    ]
    by_collection_path = {
        truth["collection"]["path"]: {
            "view": view,
            "truth": truth,
        }
        for view, truth, _classification, _entry in authenticated
    }
    by_candidate = {
        row["candidate_physics_sha256"]: by_collection_path[
            row["collection"]["path"]
        ]
        for row in ranked
    }
    artifact = manifest.get("truth_validated_pareto_front")
    if (
        manifest.get("cohort_inventory_payload_sha256")
        != inventory["payload_sha256"]
        or manifest.get("authenticated_collection_count")
        != len(authenticated)
        or manifest.get("feasible_collection_count")
        != len(feasible_payloads)
        or manifest.get("infeasible_collection_count")
        != len(infeasible_payloads)
        or manifest.get("deduplicated_feasible_candidate_count")
        != len(ranked)
        or manifest.get("rank0_count")
        != sum(
            row["truth_non_dominated_rank"] == 0 for row in ranked
        )
        or manifest.get("solver_revision")
        != inventory["solver_revision"]
        or manifest.get("library_revision")
        != inventory["library_revision"]
        or manifest.get("authenticated_collections")
        != [
            truth["collection"]
            for _view, truth, _classification, _entry in authenticated
        ]
        or manifest.get("authenticated_collection_payload_sha256")
        != [
            truth["collection_payload_sha256"]
            for _view, truth, _classification, _entry in authenticated
        ]
        or manifest.get("classification_rows") != classifications
        or manifest.get("feasible_collection_payload_sha256")
        != feasible_payloads
        or manifest.get("infeasible_collection_payload_sha256")
        != infeasible_payloads
        or manifest.get("ranked_rows") != ranked
        or manifest.get("full_plan_eligible") is not bool(ranked)
        or manifest.get("zero_feasible_audited") is not (not ranked)
        or manifest.get("sorting_authority")
        != (
            "exact_24_cohort_all_reauthenticated_and_classified_"
            "then_all_feasible_actual_truth_rows_deduplicated_"
            "and_combined_nondominated_sort"
        )
        or not isinstance(artifact, dict)
        or set(artifact)
        != {"path", "sha256", "size_bytes", "row_count", "columns"}
    ):
        raise HandoffContractError(
            "truth Pareto v2 authority recomputation drifted"
        )
    csv_path = production._contained_file(
        resolved.parent,
        artifact.get("path"),
        "truth-validated Pareto v2 CSV",
    )
    frame = pd.read_csv(csv_path)
    if (
        _file_record(csv_path)["sha256"] != artifact.get("sha256")
        or csv_path.stat().st_size != artifact.get("size_bytes")
        or artifact.get("row_count") != len(ranked)
        or artifact.get("columns") != list(TRUTH_CSV_COLUMNS)
        or list(frame.columns) != list(TRUTH_CSV_COLUMNS)
        or frame["candidate_physics_sha256"].astype(str).tolist()
        != [row["candidate_physics_sha256"] for row in ranked]
        or frame["truth_row_sha256"].astype(str).tolist()
        != [row["truth_row_sha256"] for row in ranked]
        or frame["truth_non_dominated_rank"].astype(int).tolist()
        != [row["truth_non_dominated_rank"] for row in ranked]
    ):
        raise HandoffContractError(
            "truth-validated Pareto v2 CSV drifted"
        )
    return manifest, ranked, by_candidate


def _load_truth_manifest(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    resolved = path.resolve(strict=True)
    value = production._read_json(resolved)
    schema = value.get("schema_version")
    if schema == TRUTH_MANIFEST_SCHEMA:
        return _load_truth_manifest_v1(
            resolved, predictor=predictor
        )
    if schema == TRUTH_MANIFEST_SCHEMA_V2:
        return _load_truth_manifest_v2(
            resolved, predictor=predictor
        )
    raise HandoffContractError(
        "truth Pareto manifest schema is unsupported"
    )


def create_full_plans(
    *,
    truth_manifest_path: Path,
    solver_revision: str,
    library_revision: str,
    output: Path,
    limit: int = MAX_FULL_PLANS,
    predictor: Any | None = None,
) -> Path:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_FULL_PLANS
    ):
        raise HandoffContractError("Full plan limit must be 1..3")
    manifest, ranked, by_candidate = _load_truth_manifest(
        truth_manifest_path, predictor=predictor
    )
    solver = production._require_revision(
        solver_revision, "solver_revision"
    )
    library = production._require_revision(
        library_revision, "library_revision"
    )
    if (
        solver != manifest["solver_revision"]
        or library != manifest["library_revision"]
    ):
        raise HandoffContractError(
            "Full plan solver/library differs from Standard truth provenance"
        )
    selected_rows = [
        row for row in ranked if row["truth_non_dominated_rank"] == 0
    ][:limit]
    if not selected_rows:
        raise HandoffContractError("truth Pareto has no rank-0 candidate")
    profile, profile_source = _full_profile()
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"Full plan-set output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    plan_records = []
    try:
        for row in selected_rows:
            candidate = row["candidate_physics_sha256"]
            source = by_candidate[candidate]
            view = source["view"]
            params = {
                key: view["params"][key] for key in sorted(ALL_INPUT_KEYS)
            }
            effective = production._effective_params(params, profile)
            stem = candidate[:12]
            task_name = f"mft-goal-truth-full-{stem}"
            workdir = f"mft_goal_truth_full_{stem}"
            retained = scheduler_client.retained_aedt_identity(
                task_name, params, profile, solver, library
            )
            if (
                retained is None
                or retained.get("schema_version")
                != scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_SCHEMA
                or retained.get("stage") != "full"
            ):
                raise HandoffContractError(
                    "truth Full retained artifact identity is absent"
                )
            run_root = diagnostic._retention_run_root_evidence(retained)
            candidate_dir = staging / stem
            candidate_dir.mkdir()
            params_path = production._write_immutable_json(
                candidate_dir / "full_params.json", params
            )
            profile_path = production._write_immutable_json(
                candidate_dir / "full_profile.json", profile
            )
            plan = production._seal(
                {
                    "schema_version": FULL_PLAN_SCHEMA,
                    "campaign_id": "mft-goal-20260726",
                    "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                    "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                    "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                    "temperature_contract_sha256": (
                        GOAL_TEMPERATURE_CONTRACT_SHA256
                    ),
                    "truth_promotion_manifest": _file_record(
                        truth_manifest_path
                    ),
                    "truth_promotion_payload_sha256": manifest[
                        "payload_sha256"
                    ],
                    "truth_row": copy.deepcopy(row),
                    "candidate_physics_sha256": candidate,
                    "standard_collection": copy.deepcopy(
                        source["truth"]["collection"]
                    ),
                    "standard_collection_payload_sha256": source["truth"][
                        "collection_payload_sha256"
                    ],
                    "standard_result_sha256": source["truth"][
                        "standard_result_sha256"
                    ],
                    "standard_task_id": source["truth"][
                        "standard_task_id"
                    ],
                    "solver_revision": solver,
                    "library_revision": library,
                    "fea_params_sha256": canonical_sha256(params),
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
                        "name": "full",
                        "task_name": task_name,
                        "workdir": workdir,
                        "profile_sha256": canonical_sha256(profile),
                        "effective_params_sha256": canonical_sha256(
                            effective
                        ),
                        "resources": copy.deepcopy(FULL_RESOURCES),
                        "retained_aedt_bundle": retained,
                        "retention_run_root": run_root,
                        "scheduler_project": scheduler_client.MFT_PROJECT,
                        "scheduler_url": (
                            diagnostic.DIAGNOSTIC_SCHEDULER_URL
                        ),
                        "aedt_backend": "standalone",
                        "full_model": 1,
                        "thermal_symmetry": "full",
                    },
                    "available_submission_commands": ["submit-full"],
                    "explicit_full_submission_allowed": True,
                    "scheduler_submission_performed": False,
                    "physics_override_allowed": False,
                    "retention_required": True,
                    "prune_protection_required": True,
                    **_authority_flags(),
                }
            )
            plan_path = production._write_immutable_json(
                candidate_dir / "full_plan.json", plan
            )
            plan_records.append(
                {
                    "candidate_physics_sha256": candidate,
                    "truth_non_dominated_rank": 0,
                    "truth_front_order": row["truth_front_order"],
                    "plan": {
                        "path": f"{stem}/{plan_path.name}",
                        "sha256": production._sha256_file(plan_path),
                        "size_bytes": plan_path.stat().st_size,
                    },
                    "plan_payload_sha256": plan["payload_sha256"],
                }
            )
        plan_set = production._seal(
            {
                "schema_version": FULL_PLAN_SET_SCHEMA,
                "truth_promotion_manifest": _file_record(
                    truth_manifest_path
                ),
                "truth_promotion_payload_sha256": manifest[
                    "payload_sha256"
                ],
                "solver_revision": solver,
                "library_revision": library,
                "requested_limit": limit,
                "selected_plan_count": len(plan_records),
                "rank0_only": True,
                "plans": plan_records,
                "scheduler_submission_performed": False,
                **_authority_flags(),
            }
        )
        set_path = production._write_immutable_json(
            staging / "full_plan_set.json", plan_set
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / set_path.name


FULL_PLAN_SET_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "truth_promotion_manifest",
        "truth_promotion_payload_sha256",
        "solver_revision",
        "library_revision",
        "requested_limit",
        "selected_plan_count",
        "rank0_only",
        "plans",
        "scheduler_submission_performed",
        *_authority_flags(),
    }
)


def _load_full_plan_set(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved = path.resolve(strict=True)
    value = production._validate_seal(
        production._read_json(resolved), FULL_PLAN_SET_SCHEMA
    )
    if set(value) != FULL_PLAN_SET_FIELDS:
        raise HandoffContractError("truth Full plan-set fields drifted")
    truth_record = value.get("truth_promotion_manifest")
    if not isinstance(truth_record, dict):
        raise HandoffContractError(
            "truth Full plan-set promotion authority is absent"
        )
    truth_path = Path(str(truth_record.get("path") or ""))
    if _file_record(truth_path) != truth_record:
        raise HandoffContractError(
            "truth Full plan-set promotion bytes drifted"
        )
    manifest, ranked, _by_candidate = _load_truth_manifest(
        truth_path, predictor=predictor
    )
    limit = value.get("requested_limit")
    records = value.get("plans")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_FULL_PLANS
        or not isinstance(records, list)
    ):
        raise HandoffContractError(
            "truth Full plan-set bound drifted"
        )
    expected_rows = [
        row
        for row in ranked
        if row["truth_non_dominated_rank"] == 0
    ][:limit]
    loaded = []
    if (
        value.get("truth_promotion_payload_sha256")
        != manifest["payload_sha256"]
        or value.get("solver_revision") != manifest["solver_revision"]
        or value.get("library_revision") != manifest["library_revision"]
        or value.get("selected_plan_count") != len(expected_rows)
        or len(records) != len(expected_rows)
        or value.get("rank0_only") is not True
        or value.get("scheduler_submission_performed") is not False
        or any(
            value.get(name) is not expected
            for name, expected in _authority_flags().items()
        )
    ):
        raise HandoffContractError(
            "truth Full plan-set authority drifted"
        )
    for record, row in zip(records, expected_rows, strict=True):
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "candidate_physics_sha256",
                "truth_non_dominated_rank",
                "truth_front_order",
                "plan",
                "plan_payload_sha256",
            }
        ):
            raise HandoffContractError(
                "truth Full plan-set entry fields drifted"
            )
        plan_record = record.get("plan")
        if (
            not isinstance(plan_record, dict)
            or set(plan_record)
            != {"path", "sha256", "size_bytes"}
        ):
            raise HandoffContractError(
                "truth Full plan-set plan record drifted"
            )
        plan_path = production._contained_file(
            resolved.parent,
            plan_record["path"],
            "truth Full plan-set plan",
        )
        plan, *_rest = _load_full_plan(
            plan_path, predictor=predictor
        )
        if (
            production._sha256_file(plan_path)
            != plan_record["sha256"]
            or plan_path.stat().st_size != plan_record["size_bytes"]
            or record.get("candidate_physics_sha256")
            != row["candidate_physics_sha256"]
            or record.get("truth_non_dominated_rank") != 0
            or record.get("truth_front_order")
            != row["truth_front_order"]
            or record.get("plan_payload_sha256")
            != plan["payload_sha256"]
        ):
            raise HandoffContractError(
                "truth Full plan-set plan identity drifted"
            )
        loaded.append(plan)
    return value, loaded


FULL_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "truth_promotion_manifest",
        "truth_promotion_payload_sha256",
        "truth_row",
        "candidate_physics_sha256",
        "standard_collection",
        "standard_collection_payload_sha256",
        "standard_result_sha256",
        "standard_task_id",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "fea_params",
        "profile",
        "stage",
        "available_submission_commands",
        "explicit_full_submission_allowed",
        "scheduler_submission_performed",
        "physics_override_allowed",
        "retention_required",
        "prune_protection_required",
        *_authority_flags(),
    }
)


def _plan_artifact(
    root: Path, record: Any, label: str
) -> Path:
    if (
        not isinstance(record, dict)
        or frozenset(record)
        not in {
            frozenset({"path", "sha256"}),
            frozenset(
                {"path", "sha256", "canonical_sha256", "source"}
            ),
        }
    ):
        raise HandoffContractError(f"{label} record is malformed")
    target = production._contained_file(root, record["path"], label)
    if production._sha256_file(target) != record["sha256"]:
        raise HandoffContractError(f"{label} bytes drifted")
    return target


def _load_full_plan(
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
    plan = production._validate_seal(
        production._read_json(resolved), FULL_PLAN_SCHEMA
    )
    if (
        set(plan) != FULL_PLAN_FIELDS
        or plan.get("campaign_id") != "mft-goal-20260726"
        or plan.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or plan.get("hard_spec") != GOAL_STAGE_SPEC
        or plan.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or plan.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or plan.get("available_submission_commands") != ["submit-full"]
        or plan.get("explicit_full_submission_allowed") is not True
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("physics_override_allowed") is not False
        or plan.get("retention_required") is not True
        or plan.get("prune_protection_required") is not True
        or any(
            plan.get(name) is not value
            for name, value in _authority_flags().items()
        )
    ):
        raise HandoffContractError("truth Full plan contract drifted")
    truth_record = plan.get("truth_promotion_manifest")
    if not isinstance(truth_record, dict):
        raise HandoffContractError(
            "truth Full plan promotion manifest is absent"
        )
    truth_path = Path(str(truth_record.get("path") or ""))
    if _file_record(truth_path) != truth_record:
        raise HandoffContractError(
            "truth Full plan promotion bytes drifted"
        )
    truth_manifest, ranked, by_candidate = _load_truth_manifest(
        truth_path, predictor=predictor
    )
    candidate = plan.get("candidate_physics_sha256")
    matches = [
        row
        for row in ranked
        if row["candidate_physics_sha256"] == candidate
    ]
    if (
        len(matches) != 1
        or matches[0]["truth_non_dominated_rank"] != 0
        or plan.get("truth_row") != matches[0]
        or plan.get("truth_promotion_payload_sha256")
        != truth_manifest["payload_sha256"]
        or plan.get("solver_revision")
        != truth_manifest["solver_revision"]
        or plan.get("library_revision")
        != truth_manifest["library_revision"]
    ):
        raise HandoffContractError(
            "truth Full plan is not rank-0 actual-truth authority"
        )
    source = by_candidate[candidate]
    view = source["view"]
    standard_truth = source["truth"]
    if (
        plan.get("standard_collection")
        != standard_truth["collection"]
        or plan.get("standard_collection_payload_sha256")
        != standard_truth["collection_payload_sha256"]
        or plan.get("standard_result_sha256")
        != standard_truth["standard_result_sha256"]
        or plan.get("standard_task_id")
        != standard_truth["standard_task_id"]
    ):
        raise HandoffContractError(
            "truth Full plan Standard authority drifted"
        )
    root = resolved.parent
    params = production._read_json(
        _plan_artifact(root, plan.get("fea_params"), "Full params")
    )
    profile = production._read_json(
        _plan_artifact(root, plan.get("profile"), "Full profile")
    )
    reviewed_profile, source_profile = _full_profile()
    profile_record = plan["profile"]
    stage = plan.get("stage")
    if not isinstance(stage, dict):
        raise HandoffContractError("truth Full execution stage is absent")
    effective = production._effective_params(params, profile)
    retained = scheduler_client.retained_aedt_identity(
        stage.get("task_name"),
        params,
        profile,
        plan.get("solver_revision"),
        plan.get("library_revision"),
    )
    if (
        set(params) != set(ALL_INPUT_KEYS)
        or canonical_sha256(params) != plan.get("fea_params_sha256")
        or params != {
            key: view["params"][key] for key in sorted(ALL_INPUT_KEYS)
        }
        or profile != reviewed_profile
        or profile_record.get("canonical_sha256")
        != canonical_sha256(profile)
        or profile_record.get("source") != source_profile
        or stage.get("name") != "full"
        or stage.get("profile_sha256") != canonical_sha256(profile)
        or stage.get("effective_params_sha256")
        != canonical_sha256(effective)
        or stage.get("resources") != FULL_RESOURCES
        or stage.get("retained_aedt_bundle") != retained
        or stage.get("retention_run_root")
        != diagnostic._retention_run_root_evidence(retained)
        or stage.get("scheduler_project") != scheduler_client.MFT_PROJECT
        or stage.get("scheduler_url")
        != diagnostic.DIAGNOSTIC_SCHEDULER_URL
        or stage.get("aedt_backend") != "standalone"
        or stage.get("full_model") != 1
        or stage.get("thermal_symmetry") != "full"
    ):
        raise HandoffContractError(
            "truth Full plan execution identity drifted"
        )
    return plan, params, profile, view, standard_truth


def _truth_reauthentication(
    *,
    plan: Mapping[str, Any],
    standard_truth: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "mft-goal-truth-full-submit-reauth-v1",
        "truth_promotion_payload_sha256": plan[
            "truth_promotion_payload_sha256"
        ],
        "truth_row_sha256": plan["truth_row"]["truth_row_sha256"],
        "candidate_physics_sha256": plan[
            "candidate_physics_sha256"
        ],
        "standard_collection_payload_sha256": standard_truth[
            "collection_payload_sha256"
        ],
        "standard_result_sha256": standard_truth[
            "standard_result_sha256"
        ],
        "standard_retained_artifact_sha256": standard_truth[
            "standard_retained_bundle"
        ]["artifact_sha256"],
        "standard_retained_results_tree_sha256": standard_truth[
            "standard_retained_bundle"
        ]["results_tree_sha256"],
        "rank0_actual_truth_reauthenticated": True,
        **_authority_flags(),
    }


def submit_full(
    *,
    plan_path: Path,
    scheduler_cutover_receipt_path: Path,
    license_snapshot_path: Path,
    output: Path,
    priority: int = 0,
    scheduler: Any = scheduler_client,
    predictor: Any | None = None,
    live_reader: Any = diagnostic._default_scheduler_live_reader,
) -> Path:
    plan, params, profile, _view, standard_truth = _load_full_plan(
        plan_path, predictor=predictor
    )
    reauthentication = _truth_reauthentication(
        plan=plan, standard_truth=standard_truth
    )
    cutover, launcher_before = (
        diagnostic._validate_scheduler_cutover_receipt(
            scheduler_cutover_receipt_path,
            verify_live_launcher=True,
        )
    )
    stage = plan["stage"]
    if cutover["scheduler_url"] != stage["scheduler_url"]:
        raise HandoffContractError(
            "Full Scheduler cutover endpoint differs from plan"
        )
    admission = diagnostic._live_scheduler_admission_snapshot(
        scheduler_url=stage["scheduler_url"],
        reader=live_reader,
    )
    environment, core_evidence = production._submission_environment(
        stage="full",
        solver_revision=plan["solver_revision"],
        license_snapshot_path=license_snapshot_path,
    )
    launcher_after = diagnostic._live_launcher_identity(cutover)
    if launcher_after != launcher_before:
        raise HandoffContractError(
            "Scheduler live launcher changed during Full admission"
        )
    target = output.resolve()
    if target.exists():
        raise HandoffContractError(
            f"Full submission receipt already exists: {target}"
        )
    task_id = scheduler.submit_verification(
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
        required_project_cap=diagnostic.GOAL_FEA_PROJECT_CAP,
        max_project_active_tasks=diagnostic.GOAL_FEA_PROJECT_CAP,
        scheduler_url=stage["scheduler_url"],
    )
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError(
            "Full Scheduler submission returned no durable task ID"
        )
    submission = production._seal(
        {
            "schema_version": FULL_SUBMISSION_SCHEMA,
            "stage": "full",
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "truth_authority_reauthentication": reauthentication,
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "standard_collection": standard_truth["collection"],
            "standard_collection_payload_sha256": standard_truth[
                "collection_payload_sha256"
            ],
            "standard_result_sha256": standard_truth[
                "standard_result_sha256"
            ],
            "standard_task_id": standard_truth["standard_task_id"],
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
            "scheduler_cutover_receipt": _file_record(
                scheduler_cutover_receipt_path
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
            **_authority_flags(),
        }
    )
    return production._write_immutable_json(target, submission)


FULL_SUBMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "stage",
        "plan",
        "plan_payload_sha256",
        "truth_authority_reauthentication",
        "candidate_physics_sha256",
        "standard_collection",
        "standard_collection_payload_sha256",
        "standard_result_sha256",
        "standard_task_id",
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
        *_authority_flags(),
    }
)


def _historical_license_snapshot_sha(record: Any) -> str:
    if not isinstance(record, dict):
        raise HandoffContractError(
            "Full admission license snapshot record is absent"
        )
    path = Path(str(record.get("path") or ""))
    if _file_record(path) != record:
        raise HandoffContractError(
            "Full admission license snapshot bytes drifted"
        )
    value = production._read_json(path)
    required = {
        "anshpc": 16,
        "elec_solve_maxwell": 1,
        "electronics_desktop": 1,
        "electronics3d_gui": 1,
    }
    features = value.get("features") if isinstance(value, dict) else None
    try:
        checked = datetime.fromisoformat(
            str(value.get("checked_at") or "").replace("Z", "+00:00")
        )
    except (AttributeError, ValueError) as exc:
        raise HandoffContractError(
            "Full admission license snapshot timestamp drifted"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "server_up", "server", "checked_at", "features"}
        or value.get("schema")
        != "mft-aedt-license-headroom-snapshot-v1"
        or value.get("server_up") is not True
        or value.get("server") != "1055@172.16.10.81"
        or checked.tzinfo is None
        or not isinstance(features, dict)
        or set(features) != set(required)
    ):
        raise HandoffContractError(
            "Full admission license snapshot identity drifted"
        )
    for name, minimum in required.items():
        item = features[name]
        if (
            not isinstance(item, dict)
            or set(item) != {"total", "used"}
            or isinstance(item.get("total"), bool)
            or not isinstance(item.get("total"), int)
            or isinstance(item.get("used"), bool)
            or not isinstance(item.get("used"), int)
            or item["total"] < 0
            or item["used"] < 0
            or item["used"] > item["total"]
            or item["total"] - item["used"] < minimum
        ):
            raise HandoffContractError(
                f"Full historical license headroom drifted: {name}"
            )
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return production._sha256_bytes(canonical.encode("utf-8"))


def _load_full_submission(
    path: Path,
    *,
    plan: Mapping[str, Any],
    standard_truth: Mapping[str, Any],
) -> dict[str, Any]:
    submission = production._validate_seal(
        production._read_json(path.resolve(strict=True)),
        FULL_SUBMISSION_SCHEMA,
    )
    stage = plan["stage"]
    if set(submission) != FULL_SUBMISSION_FIELDS:
        raise HandoffContractError("Full submission fields drifted")
    cutover_record = submission.get("scheduler_cutover_receipt")
    if not isinstance(cutover_record, dict):
        raise HandoffContractError(
            "Full Scheduler cutover record is absent"
        )
    cutover_path = Path(str(cutover_record.get("path") or ""))
    if _file_record(cutover_path) != cutover_record:
        raise HandoffContractError(
            "Full Scheduler cutover receipt bytes drifted"
        )
    cutover, _launcher = diagnostic._validate_scheduler_cutover_receipt(
        cutover_path, verify_live_launcher=False
    )
    admission = diagnostic._validate_recorded_admission_snapshot(
        submission.get("scheduler_admission_snapshot")
    )
    expected_reauth = _truth_reauthentication(
        plan=plan, standard_truth=standard_truth
    )
    core = submission.get("core_policy")
    if not isinstance(core, dict):
        raise HandoffContractError("Full core policy is absent")
    license_sha = _historical_license_snapshot_sha(
        core.get("admission_license_snapshot")
    )
    launcher = submission.get("scheduler_live_launcher_identity")
    if (
        submission.get("stage") != "full"
        or submission.get("plan_payload_sha256")
        != plan["payload_sha256"]
        or submission.get("truth_authority_reauthentication")
        != expected_reauth
        or submission.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or submission.get("standard_collection")
        != standard_truth["collection"]
        or submission.get("standard_collection_payload_sha256")
        != standard_truth["collection_payload_sha256"]
        or submission.get("standard_result_sha256")
        != standard_truth["standard_result_sha256"]
        or submission.get("standard_task_id")
        != standard_truth["standard_task_id"]
        or submission.get("task_name") != stage["task_name"]
        or submission.get("workdir") != stage["workdir"]
        or submission.get("dedupe_key")
        != stage["retained_aedt_bundle"]["dedupe_key"]
        or submission.get("solver_revision")
        != plan["solver_revision"]
        or submission.get("library_revision")
        != plan["library_revision"]
        or submission.get("profile_sha256")
        != stage["profile_sha256"]
        or submission.get("effective_params_sha256")
        != stage["effective_params_sha256"]
        or submission.get("resources") != FULL_RESOURCES
        or submission.get("aedt_backend") != "standalone"
        or submission.get("retained_aedt_bundle")
        != stage["retained_aedt_bundle"]
        or submission.get("retention_run_root")
        != stage["retention_run_root"]
        or submission.get("scheduler_cutover_payload_sha256")
        != cutover["payload_sha256"]
        or not isinstance(launcher, dict)
        or set(launcher) != {"path", "sha256", "size_bytes"}
        or not diagnostic._same_absolute_regular_file_identity(
            launcher.get("path"), cutover["live_launcher_path"]
        )
        or launcher.get("sha256")
        != diagnostic.SCHEDULER_LIVE_LAUNCHER_SHA256
        or admission.get("scheduler_url") != stage["scheduler_url"]
        or submission.get("scheduler_url") != stage["scheduler_url"]
        or submission.get("scheduler_project")
        != scheduler_client.MFT_PROJECT
        or submission.get("scheduler_project_mutation_performed")
        is not False
        or submission.get("scheduler_repository_modified") is not False
        or submission.get("scheduler_submission_performed") is not True
        or submission.get("retention_required") is not True
        or submission.get("prune_protection_required") is not True
        or any(
            submission.get(name) is not value
            for name, value in _authority_flags().items()
        )
        or core.get("contract") != production.FULL_CORE_CONTRACT
        or core.get("requested_num_cores") != 16
        or core.get("runtime_license_refresh_required") is not True
        or core.get("admission_license_snapshot_sha256") != license_sha
        or core.get("admission_auth_sha256")
        != production._core_auth(
            plan["solver_revision"],
            16,
            license_contract=production.FULL_LICENSE_CONTRACT,
            license_snapshot_sha256=license_sha,
        )
    ):
        raise HandoffContractError("Full submission identity drifted")
    task_id = submission.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        raise HandoffContractError("Full submission task ID is invalid")
    return submission


FULL_BUNDLE_RECEIPT_FIELDS = diagnostic.BUNDLE_RECEIPT_FIELDS


def _validate_full_bundle_receipt(
    value: Any,
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != FULL_BUNDLE_RECEIPT_FIELDS
    ):
        raise HandoffContractError(
            "Full retained artifact-bundle receipt is malformed"
        )
    receipt = copy.deepcopy(value)
    base = {
        name: receipt[name] for name in production.REMOTE_RECEIPT_FIELDS
    }
    base["schema_version"] = production.REMOTE_RECEIPT_SCHEMA
    shadow = dict(submission)
    shadow["retained_aedt"] = submission["retained_aedt_bundle"]
    production._validate_remote_receipt_payload(
        base, submission=shadow, result=result
    )
    expected = submission["retained_aedt_bundle"]
    file_count = receipt.get("results_file_count")
    results_size = receipt.get("results_size_bytes")
    if (
        receipt.get("schema_version")
        != scheduler_client.RETAINED_AEDT_TRUTH_FULL_BUNDLE_RECEIPT_SCHEMA
        or receipt.get("stage") != "full"
        or receipt.get("results_path") != expected["results_path"]
        or receipt.get("results_manifest_path")
        != expected["results_manifest_path"]
        or receipt.get("results_manifest_schema_version")
        != scheduler_client.RETAINED_AEDT_TRUTH_FULL_RESULTS_MANIFEST_SCHEMA
        or production._require_sha(
            receipt.get("results_manifest_sha256"),
            "Full results manifest SHA",
        )
        != receipt.get("results_manifest_sha256")
        or production._require_sha(
            receipt.get("results_tree_sha256"),
            "Full results tree SHA",
        )
        != receipt.get("results_tree_sha256")
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or not 1
        <= file_count
        <= scheduler_client.RETAINED_AEDT_RESULTS_MAX_FILES
        or isinstance(results_size, bool)
        or not isinstance(results_size, int)
        or not 0
        <= results_size
        <= scheduler_client.RETAINED_AEDT_RESULTS_MAX_BYTES
        or receipt.get("source_results_directory_name")
        != f"{receipt['source_project_name']}.aedtresults"
    ):
        raise HandoffContractError(
            "Full retained AEDT results receipt drifted"
        )
    return receipt


def _validate_full_results_manifest(
    value: Any,
    *,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "source_project_name",
        "source_results_directory_name",
        "retained_results_directory_name",
        "file_count",
        "size_bytes",
        "tree_sha256",
        "files",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version")
        != scheduler_client.RETAINED_AEDT_TRUTH_FULL_RESULTS_MANIFEST_SCHEMA
        or value.get("source_project_name") != result.get("project_name")
        or value.get("source_results_directory_name")
        != f"{result.get('project_name')}.aedtresults"
        or value.get("retained_results_directory_name")
        != PurePosixPath(str(receipt["results_path"])).name
        or value.get("file_count") != receipt["results_file_count"]
        or value.get("size_bytes") != receipt["results_size_bytes"]
        or value.get("tree_sha256") != receipt["results_tree_sha256"]
        or not isinstance(value.get("files"), list)
        or len(value["files"]) != value["file_count"]
    ):
        raise HandoffContractError(
            "Full retained AEDT results manifest drifted"
        )
    paths = []
    size = 0
    normalized = []
    for item in value["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size_bytes"}
        ):
            raise HandoffContractError(
                "Full results manifest file record is malformed"
            )
        pure = PurePosixPath(str(item.get("path") or ""))
        item_size = item.get("size_bytes")
        if (
            pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or isinstance(item_size, bool)
            or not isinstance(item_size, int)
            or item_size < 0
            or production._require_sha(
                item.get("sha256"), "Full results file SHA"
            )
            != item.get("sha256")
        ):
            raise HandoffContractError(
                "Full results manifest file identity drifted"
            )
        path_text = pure.as_posix()
        paths.append(path_text)
        size += item_size
        normalized.append(
            {
                "path": path_text,
                "sha256": item["sha256"],
                "size_bytes": item_size,
            }
        )
    tree = production._sha256_bytes(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or size != value["size_bytes"]
        or tree != value["tree_sha256"]
    ):
        raise HandoffContractError(
            "Full results manifest tree identity drifted"
        )
    return copy.deepcopy(value)


def _validated_full_remote_bundle(
    *,
    submission: Mapping[str, Any],
    result: Mapping[str, Any],
    scheduler_url: str,
    remote_reader: Any,
    manifest_reader: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    expected = submission["retained_aedt_bundle"]
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
    manifest_raw = manifest_reader(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=expected["results_manifest_path"],
        max_bytes=(
            scheduler_client.RETAINED_AEDT_RESULTS_MANIFEST_MAX_BYTES + 1
        ),
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
        marker = json.loads(marker_raw.decode("utf-8"))
        results_manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffContractError(
            "Full retained bundle metadata is invalid JSON"
        ) from exc
    receipt = _validate_full_bundle_receipt(
        receipt, submission=submission, result=result
    )
    marker = production._validate_marker_payload(
        marker, expected=expected
    )
    results_manifest = _validate_full_results_manifest(
        results_manifest, receipt=receipt, result=result
    )
    manifest_sha = production._sha256_bytes(manifest_raw)
    if (
        production._sha256_bytes(marker_raw) != receipt["marker_sha256"]
        or manifest_sha != receipt["results_manifest_sha256"]
    ):
        raise HandoffContractError(
            "Full retained bundle metadata SHA drifted"
        )
    return receipt, marker, results_manifest, manifest_sha


def _full_task_execution(
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
    allocation = evidence["allocation_id"]
    if (
        evidence["task_id"] != submission["task_id"]
        or evidence["name"] != submission["task_name"]
        or evidence["status"] != "completed"
        or evidence["state"] != "succeeded"
        or evidence["exit_code"] != 0
        or evidence["failure_message"] not in {"", None}
        or not evidence["slurm_job_id"].isdigit()
        or isinstance(allocation, bool)
        or not isinstance(allocation, int)
        or allocation <= 0
        or evidence["cpus"] != 16
        or evidence["memory_mb"] != 98304
        or evidence["aedt_backend"] != "standalone"
        or evidence["project"] != scheduler_client.MFT_PROJECT
        or evidence["dedupe_key"] != submission["dedupe_key"]
        or not str(evidence["actual_node_name"] or "")
        or not str(evidence["remote_cwd"] or "")
        or not str(evidence["remote_dir"] or "")
        or not str(evidence["finished_at"] or "")
    ):
        raise HandoffContractError(
            "Full Scheduler terminal execution evidence drifted"
        )
    return evidence


def _full_actual_truth(
    *,
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    plan: Mapping[str, Any],
    submission: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = production._goal_result_reasons(result, selected)
    active, temperatures, temperature_passed = _temperature_evidence(result)
    volume_l, dimensions = geometry_metrics.bounding_box_lit(result)
    width, length, height = (
        production._finite(value, "Full exterior dimension")
        for value in dimensions
    )
    resonance = production._finite(
        result.get("f_res_min_tx_rx_only_Hz"), "Full resonance"
    )
    losses = {
        name: production._finite(result.get(name), f"Full {name}")
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
    if (
        fixed
        != selected["row_contract"]["fixed_identity_attestation"]
    ):
        raise HandoffContractError(
            "Full fixed cooling/operating identity drifted"
        )
    direct_pass = (
        width <= float(GOAL_SIZE_LIMITS_MM["W"])
        and length <= float(GOAL_SIZE_LIMITS_MM["L"])
        and height <= float(GOAL_SIZE_LIMITS_MM["H"])
        and resonance >= float(GOAL_STAGE_SPEC["resonance_min_Hz"])
        and temperature_passed
        and not reasons
    )
    margins = {
        "width_mm": float(GOAL_SIZE_LIMITS_MM["W"]) - width,
        "length_mm": float(GOAL_SIZE_LIMITS_MM["L"]) - length,
        "height_mm": float(GOAL_SIZE_LIMITS_MM["H"]) - height,
        "resonance_Hz": resonance
        - float(GOAL_STAGE_SPEC["resonance_min_Hz"]),
        "temperature_C": {
            name: float(item["limit_C"])
            - production._finite(item["actual_C"], name)
            for name, item in temperatures.items()
        },
    }
    return {
        "candidate_physics_sha256": plan[
            "candidate_physics_sha256"
        ],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "scheduler_task_id": submission["task_id"],
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
            result.get("Llt"), "Full Llt"
        ),
        "active_temperature_targets": active,
        "actual_temperatures_C": {
            name: production._finite(item["actual_C"], name)
            for name, item in temperatures.items()
        },
        "actual_constraint_margins": margins,
        "goal_physical_spec_reasons": reasons,
        "goal_physical_spec_passed": direct_pass,
        "fixed_identity_attestation": fixed,
        "retained_full_model": {
            "path": receipt["artifact_path"],
            "sha256": receipt["artifact_sha256"],
            "size_bytes": receipt["artifact_size_bytes"],
        },
        "retained_full_results": {
            "path": receipt["results_path"],
            "manifest_path": receipt["results_manifest_path"],
            "manifest_sha256": receipt["results_manifest_sha256"],
            "tree_sha256": receipt["results_tree_sha256"],
            "file_count": receipt["results_file_count"],
            "size_bytes": receipt["results_size_bytes"],
        },
        **_authority_flags(),
    }


def collect_full(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str = diagnostic.DIAGNOSTIC_SCHEDULER_URL,
    scheduler: Any = scheduler_client,
    remote_reader: Any = production._remote_bytes,
    manifest_reader: Any = diagnostic._remote_manifest_bytes,
    task_reader: Any = diagnostic._scheduler_task_snapshot,
    predictor: Any | None = None,
) -> Path:
    plan, params, profile, view, standard_truth = _load_full_plan(
        plan_path, predictor=predictor
    )
    submission = _load_full_submission(
        submission_path,
        plan=plan,
        standard_truth=standard_truth,
    )
    if scheduler_url.rstrip("/") != submission["scheduler_url"]:
        raise HandoffContractError(
            "Full collection Scheduler origin differs from submission"
        )
    status = scheduler.get_status(
        int(submission["task_id"]), scheduler_url=scheduler_url
    )
    if status != "completed":
        raise HandoffContractError(
            f"Full task is not completed: status={status!r}"
        )
    task_execution = _full_task_execution(
        task_reader(
            scheduler_url=scheduler_url,
            task_id=int(submission["task_id"]),
        ),
        submission=submission,
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
            "Full Scheduler result is not strict-valid"
        )
    result = copy.deepcopy(fetched.result)
    effective = production._effective_params(params, profile)
    if (
        not scheduler.result_matches_params(
            result, effective, required_keys=set(ALL_INPUT_KEYS)
        )
        or production._require_revision(
            result.get("git_hash"), "Full result solver revision"
        )
        != plan["solver_revision"]
        or production._require_revision(
            result.get("pyaedt_library_git_hash"),
            "Full result library revision",
        )
        != plan["library_revision"]
        or production._integer(
            result.get("full_model"), "Full result full_model"
        )
        != 1
        or str(result.get("thermal_symmetry") or "") != "full"
    ):
        raise HandoffContractError("Full result runtime identity drifted")
    production._validate_result_core_policy(result, submission)
    (
        full_receipt,
        full_marker,
        full_results_manifest,
        full_results_manifest_sha,
    ) = _validated_full_remote_bundle(
        submission=submission,
        result=result,
        scheduler_url=scheduler_url,
        remote_reader=remote_reader,
        manifest_reader=manifest_reader,
    )
    standard_collection = view["collection"]
    (
        standard_receipt,
        standard_marker,
        standard_results_manifest,
        standard_results_manifest_sha,
    ) = diagnostic._validated_remote_bundle(
        submission=view["submission"],
        result=standard_collection["result"],
        scheduler_url=scheduler_url,
        remote_reader=remote_reader,
        manifest_reader=manifest_reader,
    )
    if (
        standard_receipt
        != standard_collection["remote_aedt_bundle_receipt"]
        or standard_marker
        != standard_collection["prune_protection_marker"]
        or standard_results_manifest
        != standard_collection["aedtresults_manifest"]
        or standard_results_manifest_sha
        != standard_collection["aedtresults_manifest_sha256"]
        or standard_collection["candidate_physics_sha256"]
        != plan["candidate_physics_sha256"]
        or standard_collection["result_sha256"]
        != plan["standard_result_sha256"]
    ):
        raise HandoffContractError(
            "Standard retained symmetric evidence drifted before Full collect"
        )
    selected = view["selected"]
    actual = _full_actual_truth(
        result=result,
        selected=selected,
        plan=plan,
        submission=submission,
        receipt=full_receipt,
    )
    result_sha = canonical_sha256(result)
    collection = production._seal(
        {
            "schema_version": FULL_COLLECTION_SCHEMA,
            "stage": "full",
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission": _file_record(submission_path),
            "submission_payload_sha256": submission["payload_sha256"],
            "truth_promotion_manifest": plan[
                "truth_promotion_manifest"
            ],
            "truth_promotion_payload_sha256": plan[
                "truth_promotion_payload_sha256"
            ],
            "truth_row_sha256": plan["truth_row"][
                "truth_row_sha256"
            ],
            "candidate_physics_sha256": plan[
                "candidate_physics_sha256"
            ],
            "standard_collection": plan["standard_collection"],
            "standard_collection_payload_sha256": plan[
                "standard_collection_payload_sha256"
            ],
            "standard_result_sha256": plan[
                "standard_result_sha256"
            ],
            "standard_task_id": plan["standard_task_id"],
            "standard_retained_bundle_reauthenticated": True,
            "standard_remote_aedt_bundle_receipt": standard_receipt,
            "standard_remote_aedtresults_manifest": (
                standard_results_manifest
            ),
            "standard_remote_aedtresults_manifest_sha256": (
                standard_results_manifest_sha
            ),
            "standard_remote_prune_marker": standard_marker,
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
                "full_model": 1,
                "thermal_symmetry": "full",
                "solver_core_auth_sha256": result[
                    "solver_core_auth_sha256"
                ],
                "solver_num_cores_effective": 16,
                "solver_core_license_snapshot_sha256": result[
                    "solver_core_license_snapshot_sha256"
                ],
            },
            "remote_full_aedt_bundle_receipt": full_receipt,
            "full_aedtresults_manifest": full_results_manifest,
            "full_aedtresults_manifest_sha256": (
                full_results_manifest_sha
            ),
            "full_prune_protection_marker": full_marker,
            "full_prune_protection_marker_verified": True,
            "actual_truth_evidence": actual,
            "goal_physical_spec_reasons": actual[
                "goal_physical_spec_reasons"
            ],
            "goal_physical_spec_passed": actual[
                "goal_physical_spec_passed"
            ],
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "retention_required_until_package": True,
            **_authority_flags(),
        }
    )
    return production._write_immutable_json(output.resolve(), collection)


FULL_COLLECTION_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "stage",
        "plan",
        "plan_payload_sha256",
        "submission",
        "submission_payload_sha256",
        "truth_promotion_manifest",
        "truth_promotion_payload_sha256",
        "truth_row_sha256",
        "candidate_physics_sha256",
        "standard_collection",
        "standard_collection_payload_sha256",
        "standard_result_sha256",
        "standard_task_id",
        "standard_retained_bundle_reauthenticated",
        "standard_remote_aedt_bundle_receipt",
        "standard_remote_aedtresults_manifest",
        "standard_remote_aedtresults_manifest_sha256",
        "standard_remote_prune_marker",
        "task_id",
        "scheduler_url",
        "scheduler_status",
        "scheduler_task_execution",
        "result",
        "result_sha256",
        "result_identity",
        "remote_full_aedt_bundle_receipt",
        "full_aedtresults_manifest",
        "full_aedtresults_manifest_sha256",
        "full_prune_protection_marker",
        "full_prune_protection_marker_verified",
        "actual_truth_evidence",
        "goal_physical_spec_reasons",
        "goal_physical_spec_passed",
        "scheduler_get_only_collection",
        "scheduler_mutation_performed",
        "retention_required_until_package",
        *_authority_flags(),
    }
)


def _load_full_collection(
    path: Path,
    *,
    predictor: Any | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Fully reauthenticate one collected Full result without Scheduler I/O."""
    resolved = path.resolve(strict=True)
    collection = production._validate_seal(
        production._read_json(resolved), FULL_COLLECTION_SCHEMA
    )
    if set(collection) != FULL_COLLECTION_FIELDS:
        raise HandoffContractError("truth Full collection fields drifted")
    plan_record = collection.get("plan")
    submission_record = collection.get("submission")
    if not isinstance(plan_record, dict) or not isinstance(
        submission_record, dict
    ):
        raise HandoffContractError(
            "truth Full collection authority references are absent"
        )
    plan_path = Path(str(plan_record.get("path") or ""))
    submission_path = Path(str(submission_record.get("path") or ""))
    if (
        _file_record(plan_path) != plan_record
        or _file_record(submission_path) != submission_record
    ):
        raise HandoffContractError(
            "truth Full collection authority bytes drifted"
        )
    plan, params, profile, view, standard_truth = _load_full_plan(
        plan_path, predictor=predictor
    )
    submission = _load_full_submission(
        submission_path,
        plan=plan,
        standard_truth=standard_truth,
    )
    result = collection.get("result")
    if not isinstance(result, dict):
        raise HandoffContractError("truth Full collection result is absent")
    effective = production._effective_params(params, profile)
    if (
        not scheduler_client.result_matches_params(
            result, effective, required_keys=set(ALL_INPUT_KEYS)
        )
        or production._require_revision(
            result.get("git_hash"), "Full collection solver revision"
        )
        != plan["solver_revision"]
        or production._require_revision(
            result.get("pyaedt_library_git_hash"),
            "Full collection library revision",
        )
        != plan["library_revision"]
        or production._integer(
            result.get("full_model"), "Full collection full_model"
        )
        != 1
        or str(result.get("thermal_symmetry") or "") != "full"
    ):
        raise HandoffContractError(
            "truth Full collection result identity drifted"
        )
    production._validate_result_core_policy(result, submission)
    receipt = _validate_full_bundle_receipt(
        collection.get("remote_full_aedt_bundle_receipt"),
        submission=submission,
        result=result,
    )
    results_manifest = _validate_full_results_manifest(
        collection.get("full_aedtresults_manifest"),
        receipt=receipt,
        result=result,
    )
    marker = production._validate_marker_payload(
        collection.get("full_prune_protection_marker"),
        expected=submission["retained_aedt_bundle"],
    )
    standard_collection = view["collection"]
    standard_receipt = standard_collection["remote_aedt_bundle_receipt"]
    standard_manifest = standard_collection["aedtresults_manifest"]
    standard_marker = standard_collection["prune_protection_marker"]
    task_execution = _full_task_execution(
        collection.get("scheduler_task_execution") or {},
        submission=submission,
    )
    actual = _full_actual_truth(
        result=result,
        selected=view["selected"],
        plan=plan,
        submission=submission,
        receipt=receipt,
    )
    expected_result_identity = {
        "project_name": result["project_name"],
        "solver_revision": plan["solver_revision"],
        "library_revision": plan["library_revision"],
        "effective_params_sha256": submission[
            "effective_params_sha256"
        ],
        "full_model": 1,
        "thermal_symmetry": "full",
        "solver_core_auth_sha256": result[
            "solver_core_auth_sha256"
        ],
        "solver_num_cores_effective": 16,
        "solver_core_license_snapshot_sha256": result[
            "solver_core_license_snapshot_sha256"
        ],
    }
    marker_sha = production._sha256_bytes(production._json_bytes(marker))
    standard_marker_sha = production._sha256_bytes(
        production._json_bytes(standard_marker)
    )
    if (
        collection.get("stage") != "full"
        or collection.get("plan_payload_sha256")
        != plan["payload_sha256"]
        or collection.get("submission_payload_sha256")
        != submission["payload_sha256"]
        or collection.get("truth_promotion_manifest")
        != plan["truth_promotion_manifest"]
        or collection.get("truth_promotion_payload_sha256")
        != plan["truth_promotion_payload_sha256"]
        or collection.get("truth_row_sha256")
        != plan["truth_row"]["truth_row_sha256"]
        or collection.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or collection.get("standard_collection")
        != standard_truth["collection"]
        or collection.get("standard_collection_payload_sha256")
        != standard_truth["collection_payload_sha256"]
        or collection.get("standard_result_sha256")
        != standard_truth["standard_result_sha256"]
        or collection.get("standard_task_id")
        != standard_truth["standard_task_id"]
        or collection.get(
            "standard_retained_bundle_reauthenticated"
        )
        is not True
        or collection.get("standard_remote_aedt_bundle_receipt")
        != standard_receipt
        or collection.get("standard_remote_aedtresults_manifest")
        != standard_manifest
        or collection.get(
            "standard_remote_aedtresults_manifest_sha256"
        )
        != standard_collection["aedtresults_manifest_sha256"]
        or collection.get("standard_remote_prune_marker")
        != standard_marker
        or standard_marker_sha != standard_receipt["marker_sha256"]
        or collection.get("task_id") != submission["task_id"]
        or collection.get("scheduler_url")
        != submission["scheduler_url"]
        or collection.get("scheduler_status") != "completed"
        or collection.get("scheduler_task_execution") != task_execution
        or collection.get("result_sha256")
        != canonical_sha256(result)
        or collection.get("result_identity")
        != expected_result_identity
        or collection.get("full_aedtresults_manifest")
        != results_manifest
        or collection.get("full_aedtresults_manifest_sha256")
        != receipt["results_manifest_sha256"]
        or marker_sha != receipt["marker_sha256"]
        or collection.get("full_prune_protection_marker_verified")
        is not True
        or collection.get("actual_truth_evidence") != actual
        or collection.get("goal_physical_spec_reasons")
        != actual["goal_physical_spec_reasons"]
        or collection.get("goal_physical_spec_passed")
        is not actual["goal_physical_spec_passed"]
        or collection.get("scheduler_get_only_collection") is not True
        or collection.get("scheduler_mutation_performed") is not False
        or collection.get("retention_required_until_package") is not True
        or any(
            collection.get(name) is not value
            for name, value in _authority_flags().items()
        )
    ):
        raise HandoffContractError(
            "truth Full collection evidence drifted"
        )
    return collection, plan, params, profile, view, submission


def authenticate_full_collection(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Return a stable public view over a reauthenticated Full collection."""
    collection, plan, params, profile, view, submission = (
        _load_full_collection(path, predictor=predictor)
    )
    return {
        "schema_version": "mft-goal-truth-full-authenticated-view-v1",
        "collection": collection,
        "plan": plan,
        "params": params,
        "profile": profile,
        "standard_authenticated_collection": view,
        "submission": submission,
    }


def _fetch_remote_result_file_to_path(
    *,
    scheduler_url: str,
    task_id: int,
    relative_path: str,
    expected_size: int,
    expected_sha256: str,
    destination: Path,
) -> None:
    """GET one sealed results-tree member and verify it while streaming."""
    pure = PurePosixPath(str(relative_path or ""))
    if (
        scheduler_url.rstrip("/") != diagnostic.DIAGNOSTIC_SCHEDULER_URL
        or isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or not 0 <= expected_size
        <= scheduler_client.RETAINED_AEDT_RESULTS_MAX_BYTES
        or production._require_sha(
            expected_sha256, "results member SHA"
        )
        != expected_sha256
        or destination.exists()
    ):
        raise HandoffContractError(
            "unsafe retained AEDT results member fetch contract"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    query = urllib.parse.urlencode(
        {
            "path": pure.as_posix(),
            "base": "remote_cwd",
            "max_bytes": max(1, expected_size + 1),
        }
    )
    request = urllib.request.Request(
        scheduler_url.rstrip("/")
        + f"/api/tasks/{task_id}/remote-file?{query}",
        method="GET",
    )
    digest = hashlib.sha256()
    observed = 0
    try:
        with urllib.request.urlopen(request, timeout=300.0) as response:
            with temporary.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > expected_size:
                        raise HandoffContractError(
                            "remote results member exceeds sealed size"
                        )
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if (
            observed != expected_size
            or digest.hexdigest() != expected_sha256
        ):
            raise HandoffContractError(
                "remote results member differs from sealed manifest"
            )
        os.replace(temporary, destination)
    except (
        OSError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        if temporary.exists():
            temporary.unlink()
        raise HandoffContractError(
            "Scheduler results member GET failed"
        ) from exc
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _materialize_results_tree(
    *,
    scheduler_url: str,
    task_id: int,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    destination: Path,
    result_fetcher: Any,
    final_root: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise HandoffContractError(
            f"results-tree output already exists: {destination}"
        )
    destination.mkdir(parents=True)
    local_files = []
    total = 0
    for item in manifest["files"]:
        pure = PurePosixPath(item["path"])
        target = destination.joinpath(*pure.parts)
        resolved_parent = target.parent.resolve()
        try:
            resolved_parent.relative_to(destination.resolve())
        except ValueError as exc:
            raise HandoffContractError(
                "results-tree member escapes package"
            ) from exc
        remote_path = (
            PurePosixPath(receipt["results_path"]) / pure
        ).as_posix()
        result_fetcher(
            scheduler_url=scheduler_url,
            task_id=task_id,
            relative_path=remote_path,
            expected_size=item["size_bytes"],
            expected_sha256=item["sha256"],
            destination=target,
        )
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != item["size_bytes"]
            or production._sha256_file(target) != item["sha256"]
        ):
            raise HandoffContractError(
                f"packaged results member verification failed: {pure}"
            )
        os.chmod(target, 0o444)
        total += item["size_bytes"]
        local_files.append(
            {
                "relative_path": pure.as_posix(),
                "local_absolute_path": str(
                    final_root.joinpath(*pure.parts)
                ),
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
    normalized = [
        {
            "path": item["relative_path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in local_files
    ]
    tree_sha = production._sha256_bytes(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    if (
        len(local_files) != receipt["results_file_count"]
        or total != receipt["results_size_bytes"]
        or tree_sha != receipt["results_tree_sha256"]
    ):
        raise HandoffContractError(
            "packaged results tree differs from sealed inventory"
        )
    return {
        "local_absolute_root": str(final_root),
        "source_task_id": task_id,
        "source_remote_path": receipt["results_path"],
        "source_manifest_path": receipt["results_manifest_path"],
        "source_manifest_sha256": receipt[
            "results_manifest_sha256"
        ],
        "tree_sha256": tree_sha,
        "file_count": len(local_files),
        "size_bytes": total,
        "files": local_files,
    }


def _package_artifact(
    *,
    scheduler_url: str,
    task_id: int,
    receipt: Mapping[str, Any],
    destination: Path,
    final_path: Path,
    remote_fetcher: Any,
    full_model: int,
    thermal_symmetry: str,
) -> dict[str, Any]:
    remote_fetcher(
        scheduler_url=scheduler_url,
        task_id=task_id,
        relative_path=receipt["artifact_path"],
        transport_chunk_directory=receipt[
            "transport_chunk_directory"
        ],
        transport_raw_chunk_bytes=receipt[
            "transport_raw_chunk_bytes"
        ],
        transport_max_encoded_chunk_bytes=receipt[
            "transport_max_encoded_chunk_bytes"
        ],
        transport_chunk_count=receipt["transport_chunk_count"],
        expected_size=receipt["artifact_size_bytes"],
        expected_sha256=receipt["artifact_sha256"],
        destination=destination,
    )
    if (
        not destination.is_file()
        or destination.is_symlink()
        or destination.suffix.lower() != ".aedt"
        or destination.stat().st_size != receipt["artifact_size_bytes"]
        or production._sha256_file(destination)
        != receipt["artifact_sha256"]
    ):
        raise HandoffContractError(
            "packaged AEDT differs from retained receipt"
        )
    os.chmod(destination, 0o444)
    return {
        "local_absolute_path": str(final_path),
        "sha256": receipt["artifact_sha256"],
        "size_bytes": receipt["artifact_size_bytes"],
        "source_task_id": task_id,
        "source_remote_path": receipt["artifact_path"],
        "source_project_name": receipt["source_project_name"],
        "full_model": full_model,
        "thermal_symmetry": thermal_symmetry,
    }


def package_results(
    *,
    full_collection_path: Path,
    output: Path,
    scheduler_url: str = diagnostic.DIAGNOSTIC_SCHEDULER_URL,
    remote_fetcher: Any = production._fetch_remote_to_path,
    result_fetcher: Any = _fetch_remote_result_file_to_path,
    predictor: Any | None = None,
) -> Path:
    collection, plan, params, profile, view, submission = (
        _load_full_collection(
            full_collection_path, predictor=predictor
        )
    )
    standard = view["collection"]
    standard_truth = _actual_standard_truth(
        view,
        collection_path=Path(plan["standard_collection"]["path"]),
    )
    full_truth = collection["actual_truth_evidence"]
    if scheduler_url.rstrip("/") != diagnostic.DIAGNOSTIC_SCHEDULER_URL:
        raise HandoffContractError(
            "truth Full package requires Scheduler port 8002"
        )
    if (
        scheduler_url.rstrip("/") != collection["scheduler_url"]
        or scheduler_url.rstrip("/") != standard["scheduler_url"]
        or standard.get("goal_physical_spec_passed") is not True
        or standard.get(
            "actual_body_probe_temperature_gate_passed"
        )
        is not True
        or collection.get("goal_physical_spec_passed") is not True
        or full_truth.get("goal_physical_spec_passed") is not True
        or plan["candidate_physics_sha256"]
        != standard["candidate_physics_sha256"]
        or plan["candidate_physics_sha256"]
        != full_truth["candidate_physics_sha256"]
        or standard_truth["fixed_identity_attestation"]
        != full_truth["fixed_identity_attestation"]
        or canonical_sha256(params) != plan["fea_params_sha256"]
        or profile["param_overrides"]["full_model"] != 1
        or profile["param_overrides"]["thermal_symmetry"] != "full"
    ):
        raise HandoffContractError(
            "Standard/Full actual-truth package release gates failed"
        )
    for axis in ("W", "L", "H"):
        if not math.isclose(
            standard_truth["actual_dimensions_mm"][axis],
            full_truth["actual_dimensions_mm"][axis],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise HandoffContractError(
                "Standard/Full exact geometry identity drifted"
            )
    if not math.isclose(
        standard_truth["actual_volume_L"],
        full_truth["actual_volume_L"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HandoffContractError(
            "Standard/Full exact geometry volume drifted"
        )
    destination = output.resolve()
    if destination.exists():
        raise HandoffContractError(
            f"truth Full package output already exists: {destination}"
        )
    staging = destination.with_name(
        f".{destination.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        symmetric_path = staging / "symmetric_model.aedt"
        full_path = staging / "full_model.aedt"
        symmetric_final = destination / symmetric_path.name
        full_final = destination / full_path.name
        standard_receipt = standard[
            "remote_aedt_bundle_receipt"
        ]
        full_receipt = collection[
            "remote_full_aedt_bundle_receipt"
        ]
        artifacts = {
            "symmetric": _package_artifact(
                scheduler_url=scheduler_url,
                task_id=int(standard["task_id"]),
                receipt=standard_receipt,
                destination=symmetric_path,
                final_path=symmetric_final,
                remote_fetcher=remote_fetcher,
                full_model=0,
                thermal_symmetry="eighth",
            ),
            "full": _package_artifact(
                scheduler_url=scheduler_url,
                task_id=int(collection["task_id"]),
                receipt=full_receipt,
                destination=full_path,
                final_path=full_final,
                remote_fetcher=remote_fetcher,
                full_model=1,
                thermal_symmetry="full",
            ),
        }
        symmetric_results = staging / "symmetric_model.aedtresults"
        full_results = staging / "full_model.aedtresults"
        results_trees = {
            "symmetric": _materialize_results_tree(
                scheduler_url=scheduler_url,
                task_id=int(standard["task_id"]),
                receipt=standard_receipt,
                manifest=standard["aedtresults_manifest"],
                destination=symmetric_results,
                result_fetcher=result_fetcher,
                final_root=destination / symmetric_results.name,
            ),
            "full": _materialize_results_tree(
                scheduler_url=scheduler_url,
                task_id=int(collection["task_id"]),
                receipt=full_receipt,
                manifest=collection["full_aedtresults_manifest"],
                destination=full_results,
                result_fetcher=result_fetcher,
                final_root=destination / full_results.name,
            ),
        }
        truth_path = Path(plan["truth_promotion_manifest"]["path"])
        standard_path = Path(plan["standard_collection"]["path"])
        plan_path = Path(collection["plan"]["path"])
        submission_path = Path(collection["submission"]["path"])
        evidence: dict[str, Any] = {}
        for label, source, value in (
            (
                "truth_promotion_manifest",
                truth_path,
                production._read_json(truth_path),
            ),
            ("full_plan", plan_path, plan),
            ("full_params", None, params),
            ("full_profile", None, profile),
            ("standard_collection", standard_path, standard),
            ("full_submission", submission_path, submission),
            ("full_collection", full_collection_path, collection),
        ):
            target = production._write_immutable_json(
                staging / f"{label}.json", value
            )
            evidence[label] = {
                "local_absolute_path": str(destination / target.name),
                "sha256": production._sha256_file(target),
                "size_bytes": target.stat().st_size,
                "source": (
                    None if source is None else _file_record(source)
                ),
            }
        manifest = production._seal(
            {
                "schema_version": PACKAGE_SCHEMA,
                "campaign_id": "mft-goal-20260726",
                "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
                "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
                "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
                "temperature_contract_sha256": (
                    GOAL_TEMPERATURE_CONTRACT_SHA256
                ),
                "truth_promotion_manifest": _file_record(truth_path),
                "truth_promotion_payload_sha256": plan[
                    "truth_promotion_payload_sha256"
                ],
                "truth_row_sha256": plan["truth_row"][
                    "truth_row_sha256"
                ],
                "candidate_physics_sha256": plan[
                    "candidate_physics_sha256"
                ],
                "solver_revision": plan["solver_revision"],
                "library_revision": plan["library_revision"],
                "fea_params_sha256": plan["fea_params_sha256"],
                "aedt_artifacts": artifacts,
                "aedtresults_trees": results_trees,
                "actual_constraints": {
                    "standard_symmetric": standard_truth,
                    "full": copy.deepcopy(full_truth),
                },
                "fixed_cooling_identity": {
                    "fan_velocity_m_s": 1.5,
                    "fan_config": "dual",
                    "core_plate_pad_t_mm": 2.0,
                    "wcp_pad_t_mm": 2.0,
                    "thermal_pad_conductivity_W_mK": 0.2,
                    "standard_attestation": standard_truth[
                        "fixed_identity_attestation"
                    ],
                    "full_attestation": full_truth[
                        "fixed_identity_attestation"
                    ],
                },
                "task_execution_evidence": {
                    "standard_symmetric": copy.deepcopy(
                        standard["scheduler_task_execution"]
                    ),
                    "full": copy.deepcopy(
                        collection["scheduler_task_execution"]
                    ),
                },
                "result_identities": {
                    "standard_symmetric": {
                        **copy.deepcopy(standard["result_identity"]),
                        "result_sha256": standard["result_sha256"],
                        "task_id": standard["task_id"],
                    },
                    "full": {
                        **copy.deepcopy(collection["result_identity"]),
                        "result_sha256": collection["result_sha256"],
                        "task_id": collection["task_id"],
                    },
                },
                "evidence": evidence,
                "standard_actual_truth_passed": True,
                "full_actual_truth_passed": True,
                "exact_same_candidate_and_geometry": True,
                "both_aedt_projects_present": True,
                "both_aedtresults_trees_present": True,
                "scheduler_get_only_package_fetch": True,
                "scheduler_mutation_performed": False,
                "scheduler_repository_modified": False,
                **_authority_flags(),
            }
        )
        manifest_path = production._write_immutable_json(
            staging / "package_manifest.json", manifest
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    final_manifest = destination / manifest_path.name
    _load_package(final_manifest, predictor=predictor)
    return final_manifest


PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "truth_promotion_manifest",
        "truth_promotion_payload_sha256",
        "truth_row_sha256",
        "candidate_physics_sha256",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "aedt_artifacts",
        "aedtresults_trees",
        "actual_constraints",
        "fixed_cooling_identity",
        "task_execution_evidence",
        "result_identities",
        "evidence",
        "standard_actual_truth_passed",
        "full_actual_truth_passed",
        "exact_same_candidate_and_geometry",
        "both_aedt_projects_present",
        "both_aedtresults_trees_present",
        "scheduler_get_only_package_fetch",
        "scheduler_mutation_performed",
        "scheduler_repository_modified",
        *_authority_flags(),
    }
)

PACKAGE_ARTIFACT_FIELDS = frozenset(
    {
        "local_absolute_path",
        "sha256",
        "size_bytes",
        "source_task_id",
        "source_remote_path",
        "source_project_name",
        "full_model",
        "thermal_symmetry",
    }
)

PACKAGE_RESULTS_TREE_FIELDS = frozenset(
    {
        "local_absolute_root",
        "source_task_id",
        "source_remote_path",
        "source_manifest_path",
        "source_manifest_sha256",
        "tree_sha256",
        "file_count",
        "size_bytes",
        "files",
    }
)

PACKAGE_EVIDENCE_FIELDS = frozenset(
    {"local_absolute_path", "sha256", "size_bytes", "source"}
)


def _contained_absolute(
    root: Path,
    value: Any,
    label: str,
    *,
    expected_name: str | None = None,
    expect_directory: bool = False,
) -> Path:
    text = str(value or "")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise HandoffContractError(f"{label} path is not absolute")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise HandoffContractError(
            f"{label} escapes the package root"
        ) from exc
    if (
        (expect_directory and not resolved.is_dir())
        or (not expect_directory and not resolved.is_file())
        or resolved.is_symlink()
        or (expected_name is not None and resolved.name != expected_name)
    ):
        raise HandoffContractError(f"{label} local object drifted")
    return resolved


def _validate_packaged_artifact(
    value: Any,
    *,
    root: Path,
    expected_name: str,
    receipt: Mapping[str, Any],
    task_id: int,
    full_model: int,
    thermal_symmetry: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PACKAGE_ARTIFACT_FIELDS:
        raise HandoffContractError("packaged AEDT record fields drifted")
    path = _contained_absolute(
        root,
        value.get("local_absolute_path"),
        f"packaged {expected_name}",
        expected_name=expected_name,
    )
    if (
        value.get("sha256") != receipt["artifact_sha256"]
        or value.get("size_bytes") != receipt["artifact_size_bytes"]
        or value.get("source_task_id") != task_id
        or value.get("source_remote_path") != receipt["artifact_path"]
        or value.get("source_project_name")
        != receipt["source_project_name"]
        or value.get("full_model") != full_model
        or value.get("thermal_symmetry") != thermal_symmetry
        or path.stat().st_size != receipt["artifact_size_bytes"]
        or production._sha256_file(path) != receipt["artifact_sha256"]
    ):
        raise HandoffContractError("packaged AEDT identity drifted")
    return copy.deepcopy(value)


def _validate_packaged_results_tree(
    value: Any,
    *,
    root: Path,
    expected_name: str,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    task_id: int,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != PACKAGE_RESULTS_TREE_FIELDS
    ):
        raise HandoffContractError(
            "packaged AEDT results-tree record fields drifted"
        )
    tree_root = _contained_absolute(
        root,
        value.get("local_absolute_root"),
        f"packaged {expected_name}",
        expected_name=expected_name,
        expect_directory=True,
    )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(manifest["files"]):
        raise HandoffContractError(
            "packaged AEDT results-tree inventory drifted"
        )
    observed = []
    for packaged, expected in zip(files, manifest["files"], strict=True):
        if (
            not isinstance(packaged, dict)
            or set(packaged)
            != {
                "relative_path",
                "local_absolute_path",
                "sha256",
                "size_bytes",
            }
            or packaged.get("relative_path") != expected["path"]
            or packaged.get("sha256") != expected["sha256"]
            or packaged.get("size_bytes") != expected["size_bytes"]
        ):
            raise HandoffContractError(
                "packaged AEDT results member record drifted"
            )
        pure = PurePosixPath(expected["path"])
        local = _contained_absolute(
            tree_root,
            packaged.get("local_absolute_path"),
            f"packaged results member {pure}",
        )
        expected_local = tree_root.joinpath(*pure.parts).resolve(
            strict=True
        )
        if (
            local != expected_local
            or local.stat().st_size != expected["size_bytes"]
            or production._sha256_file(local) != expected["sha256"]
        ):
            raise HandoffContractError(
                "packaged AEDT results member bytes drifted"
            )
        observed.append(
            {
                "path": expected["path"],
                "sha256": expected["sha256"],
                "size_bytes": expected["size_bytes"],
            }
        )
    tree_nodes = list(tree_root.rglob("*"))
    if any(item.is_symlink() for item in tree_nodes):
        raise HandoffContractError(
            "packaged AEDT results tree contains a symlink"
        )
    inventory = sorted(
        str(item.relative_to(tree_root)).replace("\\", "/")
        for item in tree_nodes
        if item.is_file()
    )
    tree_sha = production._sha256_bytes(
        json.dumps(
            observed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    if (
        inventory != [item["path"] for item in manifest["files"]]
        or value.get("source_task_id") != task_id
        or value.get("source_remote_path") != receipt["results_path"]
        or value.get("source_manifest_path")
        != receipt["results_manifest_path"]
        or value.get("source_manifest_sha256")
        != receipt["results_manifest_sha256"]
        or value.get("tree_sha256") != receipt["results_tree_sha256"]
        or tree_sha != receipt["results_tree_sha256"]
        or value.get("file_count") != receipt["results_file_count"]
        or value.get("size_bytes") != receipt["results_size_bytes"]
    ):
        raise HandoffContractError(
            "packaged AEDT results-tree identity drifted"
        )
    return copy.deepcopy(value)


def _validate_package_evidence(
    record: Any,
    *,
    root: Path,
    label: str,
    expected_value: Mapping[str, Any],
    expected_source: Path | None,
) -> None:
    if not isinstance(record, dict) or set(record) != PACKAGE_EVIDENCE_FIELDS:
        raise HandoffContractError(
            f"packaged evidence fields drifted: {label}"
        )
    path = _contained_absolute(
        root,
        record.get("local_absolute_path"),
        f"packaged evidence {label}",
        expected_name=f"{label}.json",
    )
    if (
        production._sha256_file(path) != record.get("sha256")
        or path.stat().st_size != record.get("size_bytes")
        or production._read_json(path) != expected_value
        or record.get("source")
        != (
            None
            if expected_source is None
            else _file_record(expected_source)
        )
    ):
        raise HandoffContractError(
            f"packaged evidence bytes drifted: {label}"
        )


def _load_package(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = resolved.parent
    package = production._validate_seal(
        production._read_json(resolved), PACKAGE_SCHEMA
    )
    if set(package) != PACKAGE_FIELDS:
        raise HandoffContractError("truth Full package fields drifted")
    evidence = package.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "truth_promotion_manifest",
        "full_plan",
        "full_params",
        "full_profile",
        "standard_collection",
        "full_submission",
        "full_collection",
    }:
        raise HandoffContractError(
            "truth Full package evidence inventory drifted"
        )
    full_source = evidence["full_collection"].get("source")
    if not isinstance(full_source, dict):
        raise HandoffContractError(
            "truth Full package collection source is absent"
        )
    full_collection_path = Path(str(full_source.get("path") or ""))
    if _file_record(full_collection_path) != full_source:
        raise HandoffContractError(
            "truth Full package collection source bytes drifted"
        )
    collection, plan, params, profile, view, submission = (
        _load_full_collection(
            full_collection_path, predictor=predictor
        )
    )
    standard = view["collection"]
    standard_path = Path(plan["standard_collection"]["path"])
    plan_path = Path(collection["plan"]["path"])
    submission_path = Path(collection["submission"]["path"])
    truth_path = Path(plan["truth_promotion_manifest"]["path"])
    standard_truth = _actual_standard_truth(
        view, collection_path=standard_path
    )
    full_truth = collection["actual_truth_evidence"]
    artifacts = package.get("aedt_artifacts")
    trees = package.get("aedtresults_trees")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"symmetric", "full"}
        or not isinstance(trees, dict)
        or set(trees) != {"symmetric", "full"}
    ):
        raise HandoffContractError(
            "truth Full package model inventory drifted"
        )
    _validate_packaged_artifact(
        artifacts["symmetric"],
        root=root,
        expected_name="symmetric_model.aedt",
        receipt=standard["remote_aedt_bundle_receipt"],
        task_id=int(standard["task_id"]),
        full_model=0,
        thermal_symmetry="eighth",
    )
    _validate_packaged_artifact(
        artifacts["full"],
        root=root,
        expected_name="full_model.aedt",
        receipt=collection["remote_full_aedt_bundle_receipt"],
        task_id=int(collection["task_id"]),
        full_model=1,
        thermal_symmetry="full",
    )
    _validate_packaged_results_tree(
        trees["symmetric"],
        root=root,
        expected_name="symmetric_model.aedtresults",
        receipt=standard["remote_aedt_bundle_receipt"],
        manifest=standard["aedtresults_manifest"],
        task_id=int(standard["task_id"]),
    )
    _validate_packaged_results_tree(
        trees["full"],
        root=root,
        expected_name="full_model.aedtresults",
        receipt=collection["remote_full_aedt_bundle_receipt"],
        manifest=collection["full_aedtresults_manifest"],
        task_id=int(collection["task_id"]),
    )
    expected_evidence = {
        "truth_promotion_manifest": (
            production._read_json(truth_path),
            truth_path,
        ),
        "full_plan": (plan, plan_path),
        "full_params": (params, None),
        "full_profile": (profile, None),
        "standard_collection": (standard, standard_path),
        "full_submission": (submission, submission_path),
        "full_collection": (collection, full_collection_path),
    }
    for label, (expected_value, source) in expected_evidence.items():
        _validate_package_evidence(
            evidence[label],
            root=root,
            label=label,
            expected_value=expected_value,
            expected_source=source,
        )
    expected_fixed = {
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "thermal_pad_conductivity_W_mK": 0.2,
        "standard_attestation": standard_truth[
            "fixed_identity_attestation"
        ],
        "full_attestation": full_truth[
            "fixed_identity_attestation"
        ],
    }
    expected_tasks = {
        "standard_symmetric": standard["scheduler_task_execution"],
        "full": collection["scheduler_task_execution"],
    }
    expected_identities = {
        "standard_symmetric": {
            **standard["result_identity"],
            "result_sha256": standard["result_sha256"],
            "task_id": standard["task_id"],
        },
        "full": {
            **collection["result_identity"],
            "result_sha256": collection["result_sha256"],
            "task_id": collection["task_id"],
        },
    }
    if (
        package.get("campaign_id") != "mft-goal-20260726"
        or package.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or package.get("hard_spec") != GOAL_STAGE_SPEC
        or package.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or package.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or package.get("truth_promotion_manifest")
        != _file_record(truth_path)
        or package.get("truth_promotion_payload_sha256")
        != plan["truth_promotion_payload_sha256"]
        or package.get("truth_row_sha256")
        != plan["truth_row"]["truth_row_sha256"]
        or package.get("candidate_physics_sha256")
        != plan["candidate_physics_sha256"]
        or package.get("solver_revision") != plan["solver_revision"]
        or package.get("library_revision") != plan["library_revision"]
        or package.get("fea_params_sha256") != plan["fea_params_sha256"]
        or package.get("actual_constraints")
        != {
            "standard_symmetric": standard_truth,
            "full": full_truth,
        }
        or package.get("fixed_cooling_identity") != expected_fixed
        or package.get("task_execution_evidence") != expected_tasks
        or package.get("result_identities") != expected_identities
        or any(
            package.get(name) is not True
            for name in (
                "standard_actual_truth_passed",
                "full_actual_truth_passed",
                "exact_same_candidate_and_geometry",
                "both_aedt_projects_present",
                "both_aedtresults_trees_present",
                "scheduler_get_only_package_fetch",
            )
        )
        or package.get("scheduler_mutation_performed") is not False
        or package.get("scheduler_repository_modified") is not False
        or any(
            package.get(name) is not value
            for name, value in _authority_flags().items()
        )
    ):
        raise HandoffContractError(
            "truth Full package authority drifted"
        )
    return package


def authenticate_package(
    path: Path,
    *,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Reauthenticate a self-contained actual-truth model package."""
    return _load_package(path, predictor=predictor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic Standard actual-truth Pareto promotion to bounded Full"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cohort = commands.add_parser("create-cohort")
    cohort.add_argument(
        "--standard-submission",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly 24 authenticated Standard submissions",
    )
    cohort.add_argument("--output", type=Path, required=True)

    promote = commands.add_parser("promote")
    promote.add_argument(
        "--standard-collection",
        type=Path,
        action="append",
        required=True,
        help=(
            "repeat 1..12 collections for legacy v1, or exactly 24 "
            "collections with --cohort-inventory for v2"
        ),
    )
    promote.add_argument("--cohort-inventory", type=Path)
    promote.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("plan-full")
    plan.add_argument("--truth-manifest", type=Path, required=True)
    plan.add_argument("--solver-revision", required=True)
    plan.add_argument("--library-revision", required=True)
    plan.add_argument("--limit", type=int, default=MAX_FULL_PLANS)
    plan.add_argument("--output", type=Path, required=True)

    submit = commands.add_parser("submit-full")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument(
        "--scheduler-cutover-receipt", type=Path, required=True
    )
    submit.add_argument("--license-snapshot", type=Path, required=True)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--output", type=Path, required=True)

    collect = commands.add_parser("collect-full")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--submission", type=Path, required=True)
    collect.add_argument(
        "--scheduler-url",
        default=diagnostic.DIAGNOSTIC_SCHEDULER_URL,
    )
    collect.add_argument("--output", type=Path, required=True)

    package = commands.add_parser("package")
    package.add_argument(
        "--full-collection", type=Path, required=True
    )
    package.add_argument(
        "--scheduler-url",
        default=diagnostic.DIAGNOSTIC_SCHEDULER_URL,
    )
    package.add_argument("--output", type=Path, required=True)

    validate_truth = commands.add_parser("validate-truth")
    validate_truth.add_argument("--truth-manifest", type=Path, required=True)

    validate_cohort = commands.add_parser("validate-cohort")
    validate_cohort.add_argument(
        "--cohort-inventory", type=Path, required=True
    )

    validate_plan = commands.add_parser("validate-full-plan")
    validate_plan.add_argument("--plan", type=Path, required=True)

    validate_plan_set = commands.add_parser("validate-full-plan-set")
    validate_plan_set.add_argument(
        "--plan-set", type=Path, required=True
    )

    validate_collection = commands.add_parser(
        "validate-full-collection"
    )
    validate_collection.add_argument(
        "--full-collection", type=Path, required=True
    )

    validate_package = commands.add_parser("validate-package")
    validate_package.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create-cohort":
        result: Any = create_cohort_inventory(
            standard_submission_paths=args.standard_submission,
            output=args.output,
        )
    elif args.command == "promote":
        result: Any = create_truth_promotion(
            standard_collection_paths=args.standard_collection,
            output=args.output,
            cohort_inventory_path=args.cohort_inventory,
        )
    elif args.command == "plan-full":
        result = create_full_plans(
            truth_manifest_path=args.truth_manifest,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            limit=args.limit,
            output=args.output,
        )
    elif args.command == "submit-full":
        result = submit_full(
            plan_path=args.plan,
            scheduler_cutover_receipt_path=(
                args.scheduler_cutover_receipt
            ),
            license_snapshot_path=args.license_snapshot,
            priority=args.priority,
            output=args.output,
        )
    elif args.command == "collect-full":
        result = collect_full(
            plan_path=args.plan,
            submission_path=args.submission,
            scheduler_url=args.scheduler_url,
            output=args.output,
        )
    elif args.command == "package":
        result = package_results(
            full_collection_path=args.full_collection,
            scheduler_url=args.scheduler_url,
            output=args.output,
        )
    elif args.command == "validate-truth":
        result = _load_truth_manifest(args.truth_manifest)[0][
            "payload_sha256"
        ]
    elif args.command == "validate-cohort":
        result = _load_cohort_inventory(args.cohort_inventory)[0][
            "payload_sha256"
        ]
    elif args.command == "validate-full-plan":
        result = _load_full_plan(args.plan)[0]["payload_sha256"]
    elif args.command == "validate-full-plan-set":
        result = _load_full_plan_set(args.plan_set)[0][
            "payload_sha256"
        ]
    elif args.command == "validate-full-collection":
        result = authenticate_full_collection(
            args.full_collection
        )["collection"]["payload_sha256"]
    elif args.command == "validate-package":
        result = authenticate_package(args.manifest)["payload_sha256"]
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
