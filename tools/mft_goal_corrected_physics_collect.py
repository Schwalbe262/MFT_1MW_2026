"""Authenticate, collect, and globally sort corrected-physics exact60 results.

The corrected campaign is a separate 60-seed NSGA-II lane.  This collector
binds itself to that lane's sealed plan and submission receipt, observes
Scheduler state with GET only, and downloads immutable worker evidence with
read-only SFTP.  It never submits, cancels, preempts, or requeues work.

Every authenticated seed contributes all 320 terminal individuals.  The
aggregate is deduplicated by ``physical_geometry_sha256`` and globally
non-dominated-sorted across the resulting full population; seed-local fronts
are never unioned.  The physics-delta capacitance gate remains screening
evidence until turn-graded symmetric FEA and dielectric-stack sensitivity
close the physical gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import html
import json
import math
import os
from pathlib import Path
import posixpath
import time
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_corrected_physics_nsga_lane as lane  # noqa: E402
from tools import mft_goal_diagnostic_compact_collect as common  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import mft_goal_global_pareto as global_pareto  # noqa: E402
from tools import slurm_nsga_offload as transport  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


COLLECTION_RECORD_SCHEMA = (
    "mft-goal-corrected-physics-delta-terminal-collection-v1"
)
COLLECTOR_STATUS_SCHEMA = (
    "mft-goal-corrected-physics-delta-collector-status-v1"
)
SCREENING_AUTHORITY_SCHEMA = (
    "mft-goal-corrected-physics-delta-global-nds-authority-v1"
)
REPORT_MANIFEST_SCHEMA = (
    "mft-goal-corrected-physics-delta-global-pareto-report-v1"
)
EARLY_SUCCESS_MINIMUM = 10
FINAL_SUCCESS_COUNT = lane.SEED_COUNT
TERMINAL_POPULATION_COUNT = scout.POPULATION
OBJECTIVE_COLUMNS = ("objective_volume_L", "objective_total_loss_W")
IDENTITY_COLUMNS = tuple(global_pareto.REQUIRED_IDENTITIES)
CORRECTED_PHYSICAL_COLUMN = f"physical_G:{lane.PHYSICS_CONSTRAINT_NAME}"
CORRECTED_NORMALIZED_COLUMN = (
    f"normalized_G:{lane.PHYSICS_CONSTRAINT_NAME}"
)
FORBIDDEN_CONSTRAINT_NAMES = frozenset(
    {
        scout.RAW_CRX_CONSTRAINT_NAME,
        scout.PROVISIONAL_TURN_GRADED_C_ACQUISITION_CONSTRAINT_NAME,
        preflight.RESONANCE_MINIMUM_CONSTRAINT,
    }
)
MAX_RESULT_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_TERMINAL_TABLE_BYTES = 256 * 1024 * 1024
DEFAULT_SCHEDULER_URL = lane.offload.DEFAULT_SCHEDULER_URL
DEFAULT_ACCOUNTS = lane.offload.DEFAULT_ACCOUNTS
DEFAULT_SCHEDULER_SOURCE = lane.offload.DEFAULT_SCHEDULER_SOURCE
DEFAULT_OUTPUT_ROOT = (
    Path(r"C:\Users\peets\slurm_scheduler_runtime")
    / "mft_goal_20260726"
    / "corrected_physics_delta_exact60_collection"
)
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})
FAIL_CLOSED_FLAGS = {
    "screening_only": True,
    "production_eligible": False,
    "final_design_claim_allowed": False,
    "automatic_promotion_allowed": False,
}


class SchedulerReader(Protocol):
    get_count: int

    def get_task(self, task_id: int) -> dict[str, Any]: ...


def _hash(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64:
        raise RuntimeError(f"{label} is not a SHA256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a SHA256 digest") from exc
    return text


def _configure_from_plan(plan_path: Path) -> dict[str, Any]:
    """Install the corrected exact60 runtime before authenticating transport."""

    return lane._configure_from_plan(plan_path.resolve(strict=True))


def authenticate_context(
    *,
    plan_path: Path,
    receipt_path: Path,
    scheduler_url: str,
) -> dict[str, Any]:
    """Bind collection to one sealed corrected plan/receipt/task ledger."""

    model = _configure_from_plan(plan_path)
    plan, _deployment, tasks, authentication = lane.offload.authenticate_plan(
        plan_path
    )
    authorized_seeds = tuple(lane.offload._authorized_plan_seeds(plan))
    expected_seeds = tuple(range(lane.SEED_START, lane.SEED_END + 1))
    if (
        plan.get("task_count") != FINAL_SUCCESS_COUNT
        or len(tasks) != FINAL_SUCCESS_COUNT
        or authorized_seeds != expected_seeds
    ):
        raise RuntimeError(
            "collector plan is not corrected exact60 seeds 2607264400..4459"
        )
    for task, seed in zip(tasks, expected_seeds):
        profile = lane._validate_search_profile(
            task.get("manufacturing_search_profile")
            or (task.get("activation") or {}).get(
                "manufacturing_search_profile"
            )
            or {}
        )
        if (
            int(task.get("seed", -1)) != seed
            or task.get("campaign_id") != lane.CAMPAIGN_ID
            or profile.get("authorized_seed_count") != FINAL_SUCCESS_COUNT
            or profile.get("physics_delta_rx_resonance_gate", {}).get(
                "model_payload_sha256"
            )
            != lane.MODEL_PAYLOAD_SHA256
        ):
            raise RuntimeError("corrected exact60 task inventory drifted")

    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            lane.offload.scheduler_payload(
                plan=plan,
                task=task,
                priority=None,
            )
            for task in tasks
        )
    }
    receipt_raw = common._read_json(receipt_path)
    receipt = lane.offload._validate_receipt(
        receipt_raw,
        plan=plan,
        expected=expected,
        authentication=authentication,
        ready=receipt_raw.get("remote_ready") or {},
        scheduler_url=scheduler_url,
    )
    task_by_seed = {int(task["seed"]): task for task in tasks}
    entries: list[dict[str, Any]] = []
    for row in sorted(receipt["tasks"], key=lambda item: int(item["seed"])):
        seed = int(row["seed"])
        task = task_by_seed.get(seed)
        expected_payload = expected.get(str(row["dedupe_key"]))
        if (
            task is None
            or expected_payload is None
            or int(expected_payload["payload_json"]["seed"]) != seed
            or not str(row["dedupe_key"]).startswith(lane.DEDUPE_PREFIX)
            or not str(expected_payload["name"]).startswith(
                lane.TASK_NAME_PREFIX
            )
        ):
            raise RuntimeError("corrected receipt/task exact60 mapping changed")
        entries.append(
            {
                "seed": seed,
                "task_id": int(row["task_id"]),
                "dedupe_key": str(row["dedupe_key"]),
                "task": task,
                "scheduler_payload": expected_payload,
            }
        )
    if (
        [entry["seed"] for entry in entries] != list(expected_seeds)
        or len({entry["task_id"] for entry in entries})
        != FINAL_SUCCESS_COUNT
    ):
        raise RuntimeError("corrected receipt is not a unique exact60 mapping")
    return {
        "plan": plan,
        "authentication": authentication,
        "receipt": receipt,
        "entries": entries,
        "authorized_seeds": list(expected_seeds),
        "scheduler_url": scheduler_url.rstrip("/"),
        "plan_path": str(plan_path.resolve(strict=True)),
        "receipt_path": str(receipt_path.resolve(strict=True)),
        "plan_file_sha256": common._sha256_file(
            plan_path.resolve(strict=True)
        ),
        "receipt_file_sha256": common._sha256_file(
            receipt_path.resolve(strict=True)
        ),
        "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
        "physics_delta_model_payload_sha256": model["payload_sha256"],
    }


def observe_tasks(
    context: Mapping[str, Any],
    scheduler: SchedulerReader,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in context["entries"]:
        try:
            detail = scheduler.get_task(int(entry["task_id"]))
            expected_payload = entry["scheduler_payload"]
            requested_account = expected_payload.get("account_name")
            if requested_account is not None:
                expected_payload = {
                    key: copy.deepcopy(value)
                    for key, value in expected_payload.items()
                    if key != "account_name"
                }
            task_id, status = lane.offload._task_authentication(
                detail,
                expected_payload,
                label="corrected exact60 collector GET",
            )
            if (
                requested_account is not None
                and detail.get("requested_account_name") != requested_account
            ):
                raise RuntimeError("corrected task account identity changed")
            raw_exit = detail.get("exit_code")
            if isinstance(raw_exit, bool):
                raise RuntimeError("Scheduler exit_code has invalid type")
            exit_code = None if raw_exit is None else int(raw_exit)
            account_name = str(detail.get("account_name") or "")
            success = status == "completed" and exit_code == 0
            if success and not account_name:
                raise RuntimeError("terminal-success task has no owning account")
            rows.append(
                {
                    **entry,
                    "task_id": task_id,
                    "scheduler_status": status,
                    "exit_code": exit_code,
                    "account_name": account_name or None,
                    "actual_node_name": (
                        str(
                            detail.get("actual_node_name")
                            or detail.get("allocation_node_name")
                            or ""
                        )
                        or None
                    ),
                    "finished_at": detail.get("finished_at"),
                    "terminal": status in TERMINAL_STATES,
                    "terminal_success": success,
                    "observation_error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **entry,
                    "scheduler_status": "query_error",
                    "exit_code": None,
                    "account_name": None,
                    "actual_node_name": None,
                    "finished_at": None,
                    "terminal": False,
                    "terminal_success": False,
                    "observation_error": f"{type(exc).__name__}:{exc}",
                }
            )
    return rows


def _validate_result(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result = lane.goal_launch._validate_seal(
        common._read_json(path),
        schema=lane.RESULT_SCHEMA,
    )
    task = entry["task"]
    activation = task["activation"]
    profile = lane._validate_search_profile(
        activation["manufacturing_search_profile"]
    )
    forbidden_result_keys = {
        "raw_same_metric_C_rx_rx_F_UCB_gate_active",
        "provisional_turn_graded_C_acquisition_gate_active",
        "authenticated_turn_graded_transfer_ratio",
    }
    if (
        forbidden_result_keys.intersection(result)
        or result.get("campaign_id") != lane.CAMPAIGN_ID
        or result.get("task_payload_sha256") != task["payload_sha256"]
        or int(result.get("seed", -1)) != int(entry["seed"])
        or result.get("fixed_primary_turns") != 6
        or result.get("population") != TERMINAL_POPULATION_COUNT
        or result.get("generations") != scout.GENERATIONS
        or result.get("terminal_population_count")
        != TERMINAL_POPULATION_COUNT
        or result.get("manufacturing_search_profile_payload_sha256")
        != profile["payload_sha256"]
        or result.get("geometry_constraint_profile_sha256")
        != profile["geometry_constraint_profile_sha256"]
        or result.get("fixed_lm2mh_resonance_contract_sha256")
        != activation["fixed_lm2mh_resonance_contract_sha256"]
        or result.get("physics_delta_Crx_q90_ucb_gate_active") is not True
        or result.get("physics_delta_fRx_q90_lcb_gate_active") is not True
        or result.get("physics_delta_model_file_sha256")
        != lane.MODEL_FILE_SHA256
        or result.get("physics_delta_model_payload_sha256")
        != lane.MODEL_PAYLOAD_SHA256
        or result.get(
            "raw_two_net_C_optimizer_objective_constraint_authority"
        )
        is not False
        or result.get("raw_two_net_C_terminal_eligibility_authority")
        is not False
        or result.get("raw_two_net_C_physical_feasibility_authority")
        is not False
        or result.get("single_0p759701_transfer_ratio_used") is not False
        or result.get("legacy_half_magnetizing_resonance_G_present")
        is not False
        or result.get("feature_extrapolation_penalty_active") is not True
        or result.get("original_split_early_hard_rejection_allowed")
        is not False
        or result.get("selected_split_Llt_G_remains_hard") is not True
        or result.get(
            "terminal_selected_and_neighbor_splits_in_decoded_params"
        )
        is not True
        or result.get("fixed20T_turn_graded_FEA_retraining_required")
        is not True
        or result.get("approved_dielectric_stack_sensitivity_required")
        is not True
        or result.get("final_turn_graded_symmetric_FEA_required") is not True
        or result.get("screening_only") is not True
        or result.get("production_eligible") is not False
        or result.get("final_design_claim_allowed") is not False
        or result.get("fresh512_activation_evidence") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("fea_submission_performed") is not False
        or result.get("symmetric_FEA_validation_still_required") is not True
        or result.get("scheduler_write_performed") is not False
        or result.get("scheduler_submission_performed") is not False
    ):
        raise RuntimeError("corrected terminal result identity mismatch")
    inventory = result.get("artifact_inventory")
    if (
        not isinstance(inventory, Mapping)
        or result.get("artifact_inventory_sha256")
        != canonical_sha256(inventory)
    ):
        raise RuntimeError("corrected result artifact inventory seal mismatch")
    selected = {
        "terminal_physical_candidates": common._artifact_record(
            inventory,
            "terminal_physical_candidates",
            "terminal_physical_candidates.csv",
        ),
        "terminal_physical_candidates_manifest": common._artifact_record(
            inventory,
            "terminal_physical_candidates_manifest",
            "terminal_physical_candidates.manifest.json",
        ),
    }
    return result, selected


def _validate_terminal_table(
    *,
    csv_path: Path,
    manifest_path: Path,
    entry: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = lane.goal_launch._validate_seal(
        common._read_json(manifest_path),
        schema=preflight.GOAL_TERMINAL_TABLE_SCHEMA,
    )
    task = entry["task"]
    activation = task["activation"]
    source = manifest.get("source_identity") or {}
    csv_record = manifest.get("csv") or {}
    if (
        manifest.get("row_count") != TERMINAL_POPULATION_COUNT
        or manifest.get("terminal_population_index_min") != 0
        or manifest.get("terminal_population_index_max")
        != TERMINAL_POPULATION_COUNT - 1
        or manifest.get("one_row_per_terminal_individual") is not True
        or manifest.get("physical_deduplication_key")
        != "physical_geometry_sha256"
        or manifest.get("global_pareto_provenance_ready") is not True
        or csv_record.get("path") != csv_path.name
        or csv_record.get("sha256") != common._sha256_file(csv_path)
        or int(csv_record.get("size_bytes", -1)) != csv_path.stat().st_size
        or csv_record.get("sha256")
        != selected["terminal_physical_candidates"]["sha256"]
        or int(csv_record.get("size_bytes", -1))
        != selected["terminal_physical_candidates"]["size_bytes"]
        or int(source.get("seed", -1)) != int(entry["seed"])
        or str(source.get("task_id") or "") != str(entry["task_id"])
        or source.get("bundle_id") != task["payload_sha256"]
        or source.get("island_id") != "diagnostic-n1-6-compact"
        or source.get("dataset_sha256")
        != activation["source_identity"]["dataset_sha256"]
        or source.get("model_artifacts_sha256")
        != activation["source_identity"]["evaluation_model_sha256"]
        or source.get("temperature_contract_sha256")
        != task["temperature_contract_sha256"]
        or source.get("hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
        or manifest.get("temperature_contract_sha256")
        != task["temperature_contract_sha256"]
        or manifest.get("hard_constraint_contract_sha256")
        != task["hard_constraint_contract_sha256"]
        or manifest.get("fixed_lm2mh_resonance_contract_sha256")
        != activation["fixed_lm2mh_resonance_contract_sha256"]
        or manifest.get("resonance_contract_schema") != lane.RESONANCE_SCHEMA
    ):
        raise RuntimeError("corrected terminal table manifest mismatch")

    string_columns = {
        *IDENTITY_COLUMNS,
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
        "source_task_id",
        "source_bundle_id",
        "source_island_id",
        "evaluation_temperature_contract_sha256",
        "evaluation_hard_constraint_contract_sha256",
    }
    frame = pd.read_csv(
        csv_path,
        dtype={name: "string" for name in string_columns},
    )
    if (
        len(frame) != TERMINAL_POPULATION_COUNT
        or list(frame.columns) != list(manifest.get("columns") or [])
    ):
        raise RuntimeError("corrected terminal table shape/columns mismatch")
    indices = pd.to_numeric(
        frame["terminal_population_index"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.array_equal(
        indices, np.arange(TERMINAL_POPULATION_COUNT, dtype=float)
    ):
        raise RuntimeError("terminal population index is not exactly 0..319")
    decoder_valid = common._canonical_bool(
        frame["decoder_valid"], "decoder_valid"
    )
    surrogate_valid = common._canonical_bool(
        frame["surrogate_physical_valid"], "surrogate_physical_valid"
    )
    if not decoder_valid.all():
        raise RuntimeError("terminal population contains decoder-invalid rows")
    for name, expected in (
        ("source_seed", str(entry["seed"])),
        ("source_task_id", str(entry["task_id"])),
        ("source_bundle_id", task["payload_sha256"]),
        ("source_island_id", "diagnostic-n1-6-compact"),
    ):
        if {str(value) for value in frame[name]} != {expected}:
            raise RuntimeError(f"terminal table {name} identity mismatch")

    identities = {
        name: _hash(common._one_string(frame, name), name)
        for name in IDENTITY_COLUMNS
    }
    if (
        identities["dataset_sha256"]
        != activation["source_identity"]["dataset_sha256"]
        or identities["evaluation_model_sha256"]
        != activation["source_identity"]["evaluation_model_sha256"]
        or common._one_string(
            frame, "evaluation_temperature_contract_sha256"
        )
        != task["temperature_contract_sha256"]
        or common._one_string(
            frame, "evaluation_hard_constraint_contract_sha256"
        )
        != task["hard_constraint_contract_sha256"]
    ):
        raise RuntimeError("corrected terminal table scientific identity mismatch")

    physical = sorted(
        column for column in frame if column.startswith("physical_G:")
    )
    normalized = sorted(
        column for column in frame if column.startswith("normalized_G:")
    )
    physical_names = [
        name.removeprefix("physical_G:") for name in physical
    ]
    normalized_names = [
        name.removeprefix("normalized_G:") for name in normalized
    ]
    if (
        not physical
        or physical_names != normalized_names
        or CORRECTED_PHYSICAL_COLUMN not in physical
        or CORRECTED_NORMALIZED_COLUMN not in normalized
        or FORBIDDEN_CONSTRAINT_NAMES.intersection(physical_names)
    ):
        raise RuntimeError(
            "terminal table corrected physics resonance authority missing"
        )
    numeric_columns = (*OBJECTIVE_COLUMNS, *physical, *normalized)
    numeric = frame[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("terminal table contains non-finite numeric evidence")
    physical_values = numeric[physical].to_numpy(dtype=float)
    constraint_feasible = decoder_valid & np.all(
        physical_values <= 0.0, axis=1
    )
    declared_constraint = common._canonical_bool(
        frame["physical_constraint_feasible"],
        "physical_constraint_feasible",
    )
    declared_feasible = common._canonical_bool(
        frame["physical_feasible"],
        "physical_feasible",
    )
    expected_feasible = constraint_feasible & surrogate_valid
    if (
        not np.array_equal(declared_constraint, constraint_feasible)
        or not np.array_equal(declared_feasible, expected_feasible)
    ):
        raise RuntimeError("terminal physical feasibility flags contradict G")
    for column in (
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
    ):
        for index, value in enumerate(frame[column]):
            _hash(value, f"{column}[{index}]")
    if not (
        frame["physical_geometry_sha256"].astype(str)
        == frame["candidate_physics_sha"].astype(str)
    ).all():
        raise RuntimeError("candidate physics hash differs from geometry hash")
    return {
        "manifest_payload_sha256": manifest["payload_sha256"],
        "identities": identities,
        "physical_constraint_columns": physical,
        "normalized_constraint_columns": normalized,
        "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
        "physics_delta_model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
        "corrected_resonance_constraint_name": lane.PHYSICS_CONSTRAINT_NAME,
    }


def _validate_existing_collection(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    record = common._validate_sealed(
        common._read_json(path), schema=COLLECTION_RECORD_SCHEMA
    )
    task_dir = path.parent
    artifacts = record.get("artifacts") or {}
    for name in (
        "result.json",
        "terminal_physical_candidates.csv",
        "terminal_physical_candidates.manifest.json",
    ):
        artifact = artifacts.get(name) or {}
        local = task_dir / name
        if (
            not local.is_file()
            or artifact.get("sha256") != common._sha256_file(local)
            or int(artifact.get("size_bytes", -1)) != local.stat().st_size
        ):
            raise RuntimeError("existing corrected collection bytes changed")
    if (
        record.get("task_id") != int(entry["task_id"])
        or record.get("seed") != int(entry["seed"])
        or record.get("task_payload_sha256")
        != entry["task"]["payload_sha256"]
        or record.get("physics_delta_model_file_sha256")
        != lane.MODEL_FILE_SHA256
        or record.get("physics_delta_model_payload_sha256")
        != lane.MODEL_PAYLOAD_SHA256
        or any(
            record.get(name) != expected
            for name, expected in FAIL_CLOSED_FLAGS.items()
        )
    ):
        raise RuntimeError("existing corrected collection identity changed")
    return record


def collect_terminal_success(
    *,
    context: Mapping[str, Any],
    observed: Mapping[str, Any],
    connection: transport._PersistentAccountConnection,
    output_root: Path,
    retries: int = 3,
) -> dict[str, Any]:
    """Download and authenticate one completed task without Scheduler writes."""

    if (
        observed.get("terminal_success") is not True
        or observed.get("scheduler_status") != "completed"
        or observed.get("exit_code") != 0
    ):
        raise RuntimeError("only terminal-success tasks may be collected")
    task_id = int(observed["task_id"])
    task_dir = output_root / f"task-{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    collection_path = task_dir / "collection_record.json"
    if collection_path.is_file():
        return _validate_existing_collection(collection_path, entry=observed)

    remote_root = posixpath.join(
        str(context["plan"]["remote_bundle"]).rstrip("/"),
        "runs",
        f"task-{task_id}",
    )
    result_path = task_dir / "result.json"
    result_download = transport._download_verified_sftp(
        connection,
        posixpath.join(remote_root, "result.json"),
        result_path,
        retries=retries,
        max_bytes=MAX_RESULT_BYTES,
    )
    result, selected = _validate_result(result_path, entry=observed)
    downloads: dict[str, Mapping[str, Any]] = {
        "result.json": result_download
    }
    limits = {
        "terminal_physical_candidates.csv": MAX_TERMINAL_TABLE_BYTES,
        "terminal_physical_candidates.manifest.json": MAX_MANIFEST_BYTES,
    }
    for record in selected.values():
        name = record["path"]
        downloads[name] = transport._download_verified_sftp(
            connection,
            posixpath.join(remote_root, name),
            task_dir / name,
            retries=retries,
            expected_identity={
                "bytes": record["size_bytes"],
                "sha256": record["sha256"],
            },
            max_bytes=limits[name],
        )
    table_evidence = _validate_terminal_table(
        csv_path=task_dir / "terminal_physical_candidates.csv",
        manifest_path=(
            task_dir / "terminal_physical_candidates.manifest.json"
        ),
        entry=observed,
        selected=selected,
    )
    artifacts = {
        name: {
            "path": name,
            "sha256": common._sha256_file(task_dir / name),
            "size_bytes": (task_dir / name).stat().st_size,
            "transport": downloads[name]["transport"],
        }
        for name in (
            "result.json",
            "terminal_physical_candidates.csv",
            "terminal_physical_candidates.manifest.json",
        )
    }
    record = common._sealed(
        {
            "schema_version": COLLECTION_RECORD_SCHEMA,
            "campaign_id": lane.CAMPAIGN_ID,
            "bundle_id": context["plan"]["bundle_id"],
            "task_id": task_id,
            "seed": int(observed["seed"]),
            "task_payload_sha256": observed["task"]["payload_sha256"],
            "scheduler_status": "completed",
            "exit_code": 0,
            "account_name": observed["account_name"],
            "actual_node_name": observed["actual_node_name"],
            "result_payload_sha256": result["payload_sha256"],
            "terminal_manifest_payload_sha256": table_evidence[
                "manifest_payload_sha256"
            ],
            "terminal_population_count": TERMINAL_POPULATION_COUNT,
            "identities": table_evidence["identities"],
            "objective_columns": list(OBJECTIVE_COLUMNS),
            "physical_constraint_columns": table_evidence[
                "physical_constraint_columns"
            ],
            "normalized_constraint_columns": table_evidence[
                "normalized_constraint_columns"
            ],
            "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
            "corrected_resonance_constraint_name": (
                lane.PHYSICS_CONSTRAINT_NAME
            ),
            "raw_two_net_C_optimizer_objective_constraint_authority": False,
            "raw_two_net_C_terminal_eligibility_authority": False,
            "single_0p759701_transfer_ratio_used": False,
            "legacy_half_magnetizing_resonance_G_present": False,
            "feature_extrapolation_penalty_active": True,
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "approved_dielectric_stack_sensitivity_required": True,
            "final_turn_graded_symmetric_FEA_required": True,
            "artifacts": artifacts,
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_access_mode": "GET_status_plus_read_only_SFTP",
            "scheduler_get_only": True,
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "collected_at": common._now(),
        }
    )
    common._atomic_json(collection_path, record)
    return record


def terminal_population_manifest(
    *,
    records: Iterable[Mapping[str, Any]],
    output_root: Path,
    snapshot_class: str,
    expected_seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Adapt authenticated corrected collections to the global NDS contract."""

    ordered = sorted(
        (copy.deepcopy(dict(record)) for record in records),
        key=lambda record: int(record["seed"]),
    )
    if not ordered:
        raise RuntimeError("global NDS requires terminal-success evidence")
    allowed = {
        "provisional_first_threshold",
        "final_integrated_exact60",
    }
    if snapshot_class not in allowed:
        raise RuntimeError("unknown corrected NDS snapshot class")
    final_seed_authority = tuple(
        range(lane.SEED_START, lane.SEED_END + 1)
        if expected_seeds is None
        else (int(seed) for seed in expected_seeds)
    )
    if (
        len(final_seed_authority) != FINAL_SUCCESS_COUNT
        or len(set(final_seed_authority)) != FINAL_SUCCESS_COUNT
    ):
        raise RuntimeError("final integrated NDS seed authority is not exact60")
    observed_seeds = [int(record["seed"]) for record in ordered]
    if (
        len(set(observed_seeds)) != len(observed_seeds)
        or not set(observed_seeds).issubset(final_seed_authority)
    ):
        raise RuntimeError("corrected NDS records contain unauthorized seeds")
    if snapshot_class == "final_integrated_exact60" and (
        len(ordered) != FINAL_SUCCESS_COUNT
        or observed_seeds != list(final_seed_authority)
    ):
        raise RuntimeError("final integrated NDS requires exact60 seed coverage")

    first = ordered[0]
    for record in ordered:
        if (
            record.get("campaign_id") != lane.CAMPAIGN_ID
            or record.get("terminal_population_count")
            != TERMINAL_POPULATION_COUNT
            or record.get("identities") != first["identities"]
            or record.get("objective_columns") != list(OBJECTIVE_COLUMNS)
            or record.get("physical_constraint_columns")
            != first["physical_constraint_columns"]
            or record.get("normalized_constraint_columns")
            != first["normalized_constraint_columns"]
            or record.get("physics_delta_model_file_sha256")
            != lane.MODEL_FILE_SHA256
            or record.get("physics_delta_model_payload_sha256")
            != lane.MODEL_PAYLOAD_SHA256
            or record.get("corrected_resonance_constraint_name")
            != lane.PHYSICS_CONSTRAINT_NAME
            or CORRECTED_PHYSICAL_COLUMN
            not in record.get("physical_constraint_columns", [])
            or CORRECTED_NORMALIZED_COLUMN
            not in record.get("normalized_constraint_columns", [])
            or any(
                record.get(name) != expected
                for name, expected in FAIL_CLOSED_FLAGS.items()
            )
        ):
            raise RuntimeError("corrected terminal collections mix authority")

    population_records: list[dict[str, Any]] = []
    root = output_root.resolve()
    for record in ordered:
        task_id = int(record["task_id"])
        table = root / f"task-{task_id}" / "terminal_physical_candidates.csv"
        artifact = record["artifacts"][
            "terminal_physical_candidates.csv"
        ]
        if (
            not table.is_file()
            or common._sha256_file(table) != artifact["sha256"]
            or table.stat().st_size != int(artifact["size_bytes"])
        ):
            raise RuntimeError("corrected terminal table collection changed")
        relative = table.resolve().relative_to(root).as_posix()
        population_records.append(
            {
                "seed": int(record["seed"]),
                "task_id": task_id,
                "bundle_id": record["task_payload_sha256"],
                "terminal_authenticated": True,
                "terminal_population_count": TERMINAL_POPULATION_COUNT,
                "table_path": relative,
                "table_sha256": common._sha256_file(table),
                "source_model_sha256": record["identities"][
                    "evaluation_model_sha256"
                ],
                "identities": copy.deepcopy(record["identities"]),
                "collection_record_payload_sha256": record[
                    "payload_sha256"
                ],
            }
        )
    value = {
        "schema_version": global_pareto.MANIFEST_SCHEMA,
        "snapshot_class": snapshot_class,
        "authenticated_seed_count": len(ordered),
        "expected_final_seed_count": FINAL_SUCCESS_COUNT,
        "seeds": observed_seeds,
        "identities": copy.deepcopy(first["identities"]),
        "objective_columns": list(OBJECTIVE_COLUMNS),
        "physical_constraint_columns": copy.deepcopy(
            first["physical_constraint_columns"]
        ),
        "normalized_constraint_columns": copy.deepcopy(
            first["normalized_constraint_columns"]
        ),
        "records": population_records,
        "global_sort_scope": (
            "all_authenticated_terminal_population_rows_then_physical_"
            "geometry_deduplication_then_exact_global_non_dominated_sorting"
        ),
        "seed_local_front_union_used": False,
        "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
        "physics_delta_model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
        "corrected_resonance_constraint_name": (
            lane.PHYSICS_CONSTRAINT_NAME
        ),
        "capacitance_screening_authority": (
            "turn_voltage_energy_network_plus_calibrated_delta_q90_ucb"
        ),
        "raw_two_net_C_optimizer_objective_constraint_authority": False,
        "raw_two_net_C_terminal_eligibility_authority": False,
        "single_0p759701_transfer_ratio_used": False,
        "legacy_half_magnetizing_resonance_G_present": False,
        "feature_extrapolation_penalty_active": True,
        "fixed20T_turn_graded_FEA_retraining_required": True,
        "approved_dielectric_stack_sensitivity_required": True,
        "final_turn_graded_symmetric_FEA_required": True,
        **FAIL_CLOSED_FLAGS,
        "symmetric_FEA_validation_still_required": True,
        "integrated_seed_scope_complete": (
            snapshot_class == "final_integrated_exact60"
        ),
    }
    value["payload_sha256"] = canonical_sha256(value)
    return value


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return str(value)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _json_scalar(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _stable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if common._read_json(path) != dict(value):
            raise RuntimeError(f"immutable JSON report changed: {path}")
        return
    common._atomic_json(path, value)


def _report_html(
    *,
    summary: Mapping[str, Any],
    front_records: list[dict[str, Any]],
    standard_records: list[dict[str, Any]],
    plotted_front: str,
) -> str:
    payload = json.dumps(
        {
            "summary": dict(summary),
            "front": front_records,
            "standard": standard_records,
            "plotted_front": plotted_front,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("<", "\\u003c")
    title = "Corrected physics exact60 · global Pareto audit"
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#141b2f; --line:#263452;
--text:#edf4ff; --muted:#9fb0cd; --cyan:#4dd7e9; --amber:#ffbd59; --green:#6ee7a8; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1440px; margin:auto; padding:28px; }} h1 {{ margin:0 0 5px; font-size:26px }}
.sub {{ color:var(--muted); margin-bottom:22px }} .cards {{ display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px }}
.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
box-shadow:0 12px 32px #0005 }} .card {{ padding:15px }} .card b {{ display:block;
font-size:24px; color:var(--cyan) }} .card span {{ color:var(--muted) }}
.grid {{ display:grid; grid-template-columns:minmax(500px,1.7fr) minmax(360px,1fr);
gap:16px }} .panel {{ padding:16px; overflow:hidden }} h2 {{ margin:0 0 12px; font-size:17px }}
svg {{ width:100%; height:520px; display:block; background:#0d1426; border-radius:8px }}
.axis {{ stroke:#687a9b; stroke-width:1 }} .gridline {{ stroke:#263452; stroke-width:1 }}
.point {{ fill:var(--cyan); opacity:.72 }} .point.standard {{ fill:var(--amber); opacity:1 }}
table {{ border-collapse:collapse; width:100%; font-size:12px }} th,td {{ padding:7px 8px;
border-bottom:1px solid var(--line); text-align:right; white-space:nowrap }} th:first-child,
td:first-child {{ text-align:left }} th {{ color:var(--muted); position:sticky; top:0;
background:var(--panel) }} .scroll {{ max-height:520px; overflow:auto }} code {{ color:var(--green) }}
.note {{ margin-top:16px; color:var(--muted) }} @media(max-width:900px) {{
.grid {{ grid-template-columns:1fr }} main {{ padding:16px }} }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<div class="sub">모든 인증 seed의 terminal 320행을 합친 뒤 physical hash 중복 제거와 전역 non-dominated sorting을 수행한 screening 결과</div>
<section class="cards" id="cards"></section>
<section class="grid">
 <div class="panel"><h2 id="plot-title"></h2><svg id="plot" viewBox="0 0 900 520" role="img"></svg></div>
 <div class="panel"><h2>표준 후보</h2><div class="scroll"><table><thead><tr>
 <th>역할</th><th>Volume L</th><th>Loss W</th><th>Seed</th><th>Hash</th>
 </tr></thead><tbody id="rows"></tbody></table></div></div>
</section>
<p class="note">이 front는 corrected physics-delta surrogate screening 증거입니다.
최종 물리 적합성은 non-rounded turn-graded symmetric FEA, 3-leg equal-gap Lm=2 mH 검증,
승인된 dielectric-stack sensitivity로 닫아야 합니다. Seed-local Pareto front 합집합은 사용하지 않았습니다.</p>
<script type="application/json" id="payload">{payload}</script>
<script>
const d=JSON.parse(document.getElementById("payload").textContent),s=d.summary;
const cards=[["인증 seed",s.authenticated_seed_count],["terminal 행",s.terminal_row_count],
["고유 형상",s.unique_physical_candidate_count],["중복 발생",s.duplicate_occurrence_count],
["screening feasible",s.hard_feasible_count],["feasible front-0",s.feasible_front0_count]];
document.getElementById("cards").innerHTML=cards.map(x=>`<div class="card"><b>${{x[1].toLocaleString()}}</b><span>${{x[0]}}</span></div>`).join("");
document.getElementById("plot-title").textContent=(d.plotted_front==="feasible_front0"?"Feasible":"Objective")+" global front-0";
const svg=document.getElementById("plot"),pts=d.front,W=900,H=520,p=58;
const ns="http://www.w3.org/2000/svg", add=(tag,a)=>{{const e=document.createElementNS(ns,tag);Object.entries(a).forEach(x=>e.setAttribute(x[0],x[1]));svg.appendChild(e);return e}};
const xs=pts.map(x=>x.objective_volume_L),ys=pts.map(x=>x.objective_total_loss_W);
const xmin=Math.min(...xs,0),xmax=Math.max(...xs,1),ymin=Math.min(...ys,0),ymax=Math.max(...ys,1);
const sx=x=>p+(x-xmin)/(xmax-xmin||1)*(W-2*p),sy=y=>H-p-(y-ymin)/(ymax-ymin||1)*(H-2*p);
for(let i=0;i<=5;i++){{let x=p+i*(W-2*p)/5,y=p+i*(H-2*p)/5;add("line",{{x1:x,y1:p,x2:x,y2:H-p,class:"gridline"}});add("line",{{x1:p,y1:y,x2:W-p,y2:y,class:"gridline"}})}}
add("line",{{x1:p,y1:H-p,x2:W-p,y2:H-p,class:"axis"}});add("line",{{x1:p,y1:p,x2:p,y2:H-p,class:"axis"}});
pts.slice(0,4000).forEach(x=>{{const c=add("circle",{{cx:sx(x.objective_volume_L),cy:sy(x.objective_total_loss_W),r:3.2,class:"point"}});c.appendChild(document.createElementNS(ns,"title")).textContent=`${{x.objective_volume_L.toFixed(2)}} L · ${{x.objective_total_loss_W.toFixed(1)}} W · seed ${{x.source_seed}}`; }});
d.standard.forEach(x=>add("circle",{{cx:sx(x.objective_volume_L),cy:sy(x.objective_total_loss_W),r:6,class:"point standard"}}));
document.getElementById("rows").innerHTML=d.standard.map(x=>`<tr><td>${{x.standard_selection_roles||""}}</td><td>${{Number(x.objective_volume_L).toFixed(2)}}</td><td>${{Number(x.objective_total_loss_W).toFixed(1)}}</td><td>${{x.source_seed}}</td><td><code>${{String(x.physical_geometry_sha256).slice(0,10)}}</code></td></tr>`).join("");
</script></main></body></html>
"""


def publish_report(
    *,
    nds_dir: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Add stable JSON and one-file HTML views to global Pareto CSV output."""

    nds = nds_dir.resolve(strict=True)
    feasible = pd.read_csv(nds / "feasible_front0.csv")
    objective = pd.read_csv(nds / "objective_front0.csv")
    standard = pd.read_csv(nds / "standard_candidates.csv")
    plotted = feasible if not feasible.empty else objective
    plotted_name = (
        "feasible_front0" if not feasible.empty else "objective_front0"
    )
    feasible_value = {
        "schema_version": "mft-goal-corrected-feasible-front0-v1",
        "row_count": int(len(feasible)),
        "records": _frame_records(feasible),
    }
    standard_value = {
        "schema_version": "mft-goal-corrected-standard-candidates-v1",
        "row_count": int(len(standard)),
        "records": _frame_records(standard),
    }
    feasible_json = nds / "feasible_front0.json"
    standard_json = nds / "standard_candidates.json"
    _stable_json(feasible_json, feasible_value)
    _stable_json(standard_json, standard_value)
    report_path = nds / "global-pareto-audit.html"
    report_text = _report_html(
        summary=summary,
        front_records=_frame_records(plotted),
        standard_records=standard_value["records"],
        plotted_front=plotted_name,
    )
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != report_text:
            raise RuntimeError("immutable global Pareto HTML changed")
    else:
        report_path.write_text(report_text, encoding="utf-8", newline="\n")
    inventory = {
        path.name: {
            "sha256": common._sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (
            nds / "summary.json",
            nds / "ranked_unique_candidates.csv",
            nds / "feasible_front0.csv",
            feasible_json,
            standard_json,
            report_path,
        )
    }
    value = common._sealed(
        {
            "schema_version": REPORT_MANIFEST_SCHEMA,
            "global_sort_scope": summary["global_sort_scope"],
            "seed_local_front_union_used": False,
            "authenticated_seed_count": summary["authenticated_seed_count"],
            "terminal_row_count": summary["terminal_row_count"],
            "artifacts": inventory,
            **FAIL_CLOSED_FLAGS,
            "final_turn_graded_symmetric_FEA_required": True,
        }
    )
    manifest_path = nds / "global-pareto-audit.manifest.json"
    _stable_json(manifest_path, value)
    return {
        "html": str(report_path),
        "feasible_front_json": str(feasible_json),
        "standard_candidates_json": str(standard_json),
        "report_manifest": str(manifest_path),
    }


def publish_screening_nds(
    *,
    records: Iterable[Mapping[str, Any]],
    output_root: Path,
    snapshot_class: str,
    expected_seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest = terminal_population_manifest(
        records=records,
        output_root=output_root,
        snapshot_class=snapshot_class,
        expected_seeds=expected_seeds,
    )
    if snapshot_class == "provisional_first_threshold":
        manifest_path = (
            output_root / "provisional_terminal_population_manifest.json"
        )
        nds_dir = output_root / "nds" / "provisional-first-threshold"
    else:
        manifest_path = (
            output_root / "final_exact60_terminal_population_manifest.json"
        )
        nds_dir = output_root / "nds" / "final-integrated-exact60"
    common._immutable_json(manifest_path, manifest)
    if nds_dir.exists():
        summary = common._read_json(nds_dir / "summary.json")
        expected_manifest_sha = common._sha256_file(manifest_path)
        if (
            summary.get("input_manifest", {}).get("sha256")
            != expected_manifest_sha
        ):
            raise RuntimeError("existing global NDS input manifest changed")
    else:
        summary = global_pareto.build_global_pareto(
            manifest_path=manifest_path,
            output_dir=nds_dir,
        )
    expected_rows = (
        int(manifest["authenticated_seed_count"])
        * TERMINAL_POPULATION_COUNT
    )
    if (
        int(summary.get("authenticated_seed_count", -1))
        != int(manifest["authenticated_seed_count"])
        or int(summary.get("terminal_row_count", -1)) != expected_rows
        or summary.get("global_sort_scope")
        != "all_terminal_population_rows_after_physical_hash_dedupe"
    ):
        raise RuntimeError("global NDS summary coverage mismatch")
    report = publish_report(nds_dir=nds_dir, summary=summary)
    authority = common._sealed(
        {
            "schema_version": SCREENING_AUTHORITY_SCHEMA,
            "snapshot_class": snapshot_class,
            "terminal_population_manifest": {
                "path": str(manifest_path),
                "sha256": common._sha256_file(manifest_path),
                "payload_sha256": manifest["payload_sha256"],
            },
            "global_nds_summary": {
                "path": str(nds_dir / "summary.json"),
                "sha256": common._sha256_file(nds_dir / "summary.json"),
                "content_sha256": summary["content_sha256"],
            },
            "report": report,
            "authenticated_seed_count": manifest[
                "authenticated_seed_count"
            ],
            "terminal_row_count": summary["terminal_row_count"],
            "unique_physical_candidate_count": summary[
                "unique_physical_candidate_count"
            ],
            "hard_feasible_count": summary["hard_feasible_count"],
            "screening_feasible_front0_count": summary[
                "feasible_front0_count"
            ],
            "screening_objective_front0_count": summary[
                "objective_front0_count"
            ],
            "integrated_seed_scope_complete": manifest[
                "integrated_seed_scope_complete"
            ],
            "global_sort_scope": manifest["global_sort_scope"],
            "seed_local_front_union_used": False,
            "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": (
                lane.MODEL_PAYLOAD_SHA256
            ),
            "corrected_resonance_constraint_name": (
                lane.PHYSICS_CONSTRAINT_NAME
            ),
            "front_files_are_surrogate_screening_only": True,
            "raw_two_net_C_optimizer_objective_constraint_authority": False,
            "raw_two_net_C_terminal_eligibility_authority": False,
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "approved_dielectric_stack_sensitivity_required": True,
            "final_turn_graded_symmetric_FEA_required": True,
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_get_only": True,
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "published_at": common._now(),
        }
    )
    authority_path = nds_dir / "screening_authority.json"
    if authority_path.exists():
        prior = common._validate_sealed(
            common._read_json(authority_path),
            schema=SCREENING_AUTHORITY_SCHEMA,
        )
        stable = (
            "snapshot_class",
            "terminal_population_manifest",
            "global_nds_summary",
            "report",
            "authenticated_seed_count",
            "terminal_row_count",
            "unique_physical_candidate_count",
            "hard_feasible_count",
            "screening_feasible_front0_count",
            "screening_objective_front0_count",
            "integrated_seed_scope_complete",
        )
        if any(prior[name] != authority[name] for name in stable):
            raise RuntimeError("existing corrected NDS authority changed")
        authority = prior
    else:
        common._atomic_json(authority_path, authority)
    return {
        "manifest": str(manifest_path),
        "nds_output": str(nds_dir),
        "screening_authority": str(authority_path),
        **report,
        "authenticated_seed_count": manifest["authenticated_seed_count"],
        "integrated_seed_scope_complete": manifest[
            "integrated_seed_scope_complete"
        ],
    }


def _existing_collection_records(
    output_root: Path,
    entries_by_id: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not output_root.exists():
        return records
    for path in output_root.glob("task-*/collection_record.json"):
        try:
            task_id = int(path.parent.name.removeprefix("task-"))
        except ValueError:
            continue
        entry = entries_by_id.get(task_id)
        if entry is None:
            raise RuntimeError("foreign task collection exists in output root")
        records[task_id] = _validate_existing_collection(path, entry=entry)
    return records


def collect_once(
    *,
    context: Mapping[str, Any],
    output_root: Path,
    scheduler: SchedulerReader,
    accounts_path: Path,
    scheduler_source: Path,
    retries: int = 3,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    observed = observe_tasks(context, scheduler)
    entries_by_id = {
        int(entry["task_id"]): entry for entry in context["entries"]
    }
    collections = _existing_collection_records(output_root, entries_by_id)
    connections: dict[str, transport._PersistentAccountConnection] = {}
    collection_errors: dict[int, str] = {}
    try:
        for row in observed:
            task_id = int(row["task_id"])
            if row.get("terminal_success") is not True or task_id in collections:
                continue
            account_name = str(row["account_name"])
            connection = connections.get(account_name)
            if connection is None:
                account, ssh_session = transport._account(
                    accounts_path.resolve(strict=True),
                    scheduler_source.resolve(strict=True),
                    account_name,
                )
                connection = transport._PersistentAccountConnection(
                    account, ssh_session
                )
                connections[account_name] = connection
            try:
                collections[task_id] = collect_terminal_success(
                    context=context,
                    observed=row,
                    connection=connection,
                    output_root=output_root,
                    retries=retries,
                )
            except Exception as exc:
                collection_errors[task_id] = f"{type(exc).__name__}:{exc}"
    finally:
        for connection in connections.values():
            connection.close()

    ordered_records = sorted(
        collections.values(), key=lambda record: int(record["seed"])
    )
    provisional = None
    provisional_authority = (
        output_root
        / "nds"
        / "provisional-first-threshold"
        / "screening_authority.json"
    )
    if provisional_authority.is_file():
        authority = common._validate_sealed(
            common._read_json(provisional_authority),
            schema=SCREENING_AUTHORITY_SCHEMA,
        )
        provisional = {
            "manifest": str(
                output_root / "provisional_terminal_population_manifest.json"
            ),
            "nds_output": str(provisional_authority.parent),
            "screening_authority": str(provisional_authority),
            **authority["report"],
            "authenticated_seed_count": authority[
                "authenticated_seed_count"
            ],
            "integrated_seed_scope_complete": False,
        }
    elif len(ordered_records) >= EARLY_SUCCESS_MINIMUM:
        provisional = publish_screening_nds(
            records=ordered_records,
            output_root=output_root,
            snapshot_class="provisional_first_threshold",
            expected_seeds=context["authorized_seeds"],
        )

    final = None
    if len(ordered_records) == FINAL_SUCCESS_COUNT:
        final = publish_screening_nds(
            records=ordered_records,
            output_root=output_root,
            snapshot_class="final_integrated_exact60",
            expected_seeds=context["authorized_seeds"],
        )
    statuses = Counter(str(row["scheduler_status"]) for row in observed)
    status = common._sealed(
        {
            "schema_version": COLLECTOR_STATUS_SCHEMA,
            "campaign_id": lane.CAMPAIGN_ID,
            "bundle_id": context["plan"]["bundle_id"],
            "plan_path": context["plan_path"],
            "plan_file_sha256": context["plan_file_sha256"],
            "receipt_path": context["receipt_path"],
            "receipt_file_sha256": context["receipt_file_sha256"],
            "expected_task_count": FINAL_SUCCESS_COUNT,
            "scheduler_status_counts": dict(sorted(statuses.items())),
            "scheduler_terminal_count": sum(
                bool(row["terminal"]) for row in observed
            ),
            "scheduler_terminal_success_count": sum(
                bool(row["terminal_success"]) for row in observed
            ),
            "authenticated_collection_count": len(ordered_records),
            "authenticated_seeds": [
                int(record["seed"]) for record in ordered_records
            ],
            "collection_errors": {
                str(task_id): message
                for task_id, message in sorted(collection_errors.items())
            },
            "provisional_global_nds": provisional,
            "final_integrated_global_nds": final,
            "early_threshold_reached": (
                len(ordered_records) >= EARLY_SUCCESS_MINIMUM
            ),
            "final_exact60_reached": (
                len(ordered_records) == FINAL_SUCCESS_COUNT
            ),
            "global_sort_scope": (
                "all_terminal_rows_after_physical_hash_dedupe"
            ),
            "seed_local_front_union_used": False,
            "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": (
                lane.MODEL_PAYLOAD_SHA256
            ),
            "raw_two_net_C_optimizer_objective_constraint_authority": False,
            "raw_two_net_C_terminal_eligibility_authority": False,
            "fixed20T_turn_graded_FEA_retraining_required": True,
            "approved_dielectric_stack_sensitivity_required": True,
            "final_turn_graded_symmetric_FEA_required": True,
            **FAIL_CLOSED_FLAGS,
            "symmetric_FEA_validation_still_required": True,
            "scheduler_access_mode": "GET_status_plus_read_only_SFTP",
            "scheduler_get_only": True,
            "scheduler_get_count": int(scheduler.get_count),
            "scheduler_post_count": 0,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "updated_at": common._now(),
        }
    )
    common._atomic_json(output_root / "collector_status.json", status)
    return status


def run(
    *,
    plan_path: Path,
    receipt_path: Path,
    output_root: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    retries: int = 3,
    max_polls: int = 1,
    poll_seconds: float = 30.0,
    require_early: bool = False,
    require_final: bool = False,
    scheduler: SchedulerReader | None = None,
) -> tuple[dict[str, Any], int]:
    if not 1 <= retries <= 10:
        raise ValueError("retries must be within 1..10")
    if not 1 <= max_polls <= 10_000:
        raise ValueError("max_polls must be within 1..10000")
    if not 0.0 <= poll_seconds <= 300.0:
        raise ValueError("poll_seconds must be within 0..300")
    context = authenticate_context(
        plan_path=plan_path,
        receipt_path=receipt_path,
        scheduler_url=scheduler_url,
    )
    client = scheduler or common.ReadOnlySchedulerClient(scheduler_url)
    status: dict[str, Any] = {}
    for poll in range(1, max_polls + 1):
        status = collect_once(
            context=context,
            output_root=output_root,
            scheduler=client,
            accounts_path=accounts_path,
            scheduler_source=scheduler_source,
            retries=retries,
        )
        if status["final_exact60_reached"]:
            break
        if (
            require_early
            and not require_final
            and status["early_threshold_reached"]
        ):
            break
        if poll < max_polls:
            time.sleep(poll_seconds)
    exit_code = 0
    if require_final and not status["final_exact60_reached"]:
        exit_code = 2
    elif require_early and not status["early_threshold_reached"]:
        exit_code = 2
    return status, exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source",
        type=Path,
        default=DEFAULT_SCHEDULER_SOURCE,
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--require-early", action="store_true")
    parser.add_argument("--require-final", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status, exit_code = run(
        plan_path=args.plan,
        receipt_path=args.receipt,
        output_root=args.output,
        scheduler_url=args.scheduler_url,
        accounts_path=args.accounts,
        scheduler_source=args.scheduler_source,
        retries=args.retries,
        max_polls=args.max_polls,
        poll_seconds=args.poll_seconds,
        require_early=args.require_early,
        require_final=args.require_final,
    )
    print(json.dumps(status, indent=2, sort_keys=True, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
