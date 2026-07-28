"""Sealed post-deadline Standard probe for official candidate number eight.

This module is intentionally bound to one frozen candidate and one strict
``jji0930/n114`` demand-pool request.  Preparation is GET-only.  Submission
uses the already reviewed official-candidate transaction implementation for
the mutation lock, pre-lock and in-lock revalidation, consume-before-network
single-attempt ledger, exact reconciliation, and durable task readback.

Unlike the ready-allocation official candidate-six lane, this lane permits
only the exact Scheduler ``opening`` response.  It never accepts preferred
node relaxation, an already active n114 FEA task, or a different output root.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

reviewed = importlib.import_module(
    "tools.mft_goal_official_standard_postdeadline"
)

CAMPAIGN_ID = "mft-goal-20260726"
SCHEDULER_URL = "http://127.0.0.1:8002"
PROJECT = "MFT_1MW_2026v1"
ACCOUNT_NAME = "jji0930"
NODE_NAME = "n114"

CANDIDATE_SHA256 = (
    "622097dde126e12540a9aad52cb992d880ddb47a7d881f722858f624fd8878f4"
)
SOURCE_SEED = 2607262461
SOURCE_TASK_ID = 96141
SOURCE_TASK_NAME = "mft-goal-nsga-s2607262461-n1-6"
SOURCE_TASK_DEDUPE_KEY = (
    "mft-goal-20260726-nsga:"
    "6d3bec1ea9438ee868047f4d061ed6ebcbd90fbbb8f29e3cc73e080559a1a6f9"
)
SOURCE_BUNDLE_ID = (
    "c071f0f991ad0edd058f1ef97652b1ba5b36d07c45bcda850386ed8d0f1b7cf2"
)
SOURCE_RESULT_SHA256 = (
    "faf20ef2b0cd1f45b832b73f4dd16501d4250760733cc1f5310b0ad71b86c698"
)

AGGREGATE_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\aggregate_rolling512_d4e4d60"
)
STANDARD_CANDIDATES = AGGREGATE_ROOT / "standard_candidates.csv"
AGGREGATE_MANIFEST = AGGREGATE_ROOT / "aggregate_manifest.json"
STANDARD_CANDIDATES_SHA256 = (
    "f99ee4b7fba62a83e5b958d24c31260a5ba48c6fa77e6d25f2f189896df3ff6d"
)
AGGREGATE_MANIFEST_SHA256 = (
    "0ea4feb43f9eb37c8b057e7f06720f0c23590b463111443f99d8ffa6d08c4a63"
)
AGGREGATE_PAYLOAD_SHA256 = (
    "246417be7fb19629c0bb576999ebe711396f9774cebf83996be9dc83538bd96b"
)

PROFILE_PATH = reviewed.PROFILE_PATH
PROFILE_SHA256 = reviewed.PROFILE_SHA256
INPUT_PARAMETER_PATH = reviewed.INPUT_PARAMETER_PATH
SOLVER_REVISION = reviewed.SOLVER_REVISION
LIBRARY_REVISION = reviewed.LIBRARY_REVISION
CORE_AUTH_SHA256 = reviewed.CORE_AUTH_SHA256

CPUS = 8
MEMORY_MB = 98_304
SOLVER_SECONDS = 43_200
KILL_GRACE_SECONDS = 300
RETENTION_SECONDS = 1_800
SCHEDULER_SECONDS = SOLVER_SECONDS + KILL_GRACE_SECONDS + RETENTION_SECONDS
MAX_WORKERS_PER_NODE = 1
PRIORITY = 100
HISTORICAL_SCHEDULER_PAYLOAD_SHA256 = (
    "e340507d361d35b2f445ba4bca5ae6593dfc524c1e4022d0aa6f977d2cb217c0"
)
PYAEDT_LIBRARY_BINDING_COMMAND = (
    'export MFT_PYAEDT_LIBRARY_ROOT="$MFT_WORKDIR/pyaedt_library"; '
    "printf 'MFT_PYAEDT_LIBRARY_ROOT %s\\n' "
    '"$MFT_PYAEDT_LIBRARY_ROOT"; '
)

TASK_NAME = (
    "mft-goal-diag-standard-postdeadline-official8-"
    "s96141-622097dde126-n114"
)
WORKDIR = (
    "mft_goal_diag_standard_postdeadline_official8_"
    "s96141_622097dde126_n114"
)
POST_AUTHORIZATION = "official8-jji0930-n114-opening"
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official8_622097dde126_n114_260726_v1"
)
PLAN_NAME = "diagnostic_postdeadline_plan.json"
ATTEMPT_LEDGER_NAME = "scheduler_post_attempt.json"
SUBMISSION_DIRECTORY_NAME = "submission"
ORIGINAL_DEADLINE_UTC = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)

PLAN_SCHEMA = "mft-goal-official-standard8-postdeadline-plan-v1"
CANDIDATE_SCHEMA = (
    "mft-goal-official-standard8-postdeadline-selected-candidate-v1"
)
PREPARE_SCHEMA = "mft-goal-official-standard8-postdeadline-prepare-v1"
INTENT_SCHEMA = "mft-goal-official-standard8-postdeadline-submit-intent-v1"
SUBMISSION_SCHEMA = (
    "mft-goal-official-standard8-postdeadline-submission-v1"
)
FINAL_SEAL_SCHEMA = (
    "mft-goal-official-standard8-postdeadline-final-seal-v1"
)
PREFLIGHT_SCHEMA = (
    "mft-goal-official-standard8-opening-live-preflight-v1"
)
ATTEMPT_NONCE_SCHEMA = (
    "mft-goal-official-standard8-postdeadline-attempt-nonce-v1"
)
SINGLE_ATTEMPT_SCHEMA = (
    "mft-goal-official-standard8-postdeadline-single-attempt-contract-v1"
)

OPENING_QUEUE_REASON = (
    "no single ready pool has 8 free CPUs; opening demand pools"
)
SAFETY_FLAGS = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "noncanonical": True,
    "production_eligible": False,
    "automatic_promotion": False,
    "scientific_pass_claimed": False,
    "original_deadline_missed": True,
}
FIXED_BOUNDARY = {
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
    "fan_velocity_m_s": 1.5,
    "fan_config": "dual",
    "core_plate_on": 1,
    "wcp_on": 1,
}
BOUNDARY_PROJECTION = {
    "k_ins": 0.2,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
    "fan_velocity": 1.5,
    "fan_config": "dual",
    "core_plate_on": 1,
    "wcp_on": 1,
}

PostdeadlineContractError = reviewed.PostdeadlineContractError
JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]
PayloadBuilder = Callable[
    [dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]
PlanLoader = Callable[
    [Path],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]
Poster = Callable[
    [str, Mapping[str, Any]],
    tuple[int | None, dict[str, Any] | None, str | None],
]

canonical_bytes = reviewed.canonical_bytes
payload_sha256 = reviewed.payload_sha256
sealed = reviewed.sealed
validate_seal = reviewed.validate_seal
sha256_file = reviewed.sha256_file
file_record = reviewed.file_record
relative_record = reviewed.relative_record
read_json = reviewed.read_json
write_immutable_json = reviewed.write_immutable_json
get_json = reviewed.get_json
scheduler_campaign_lock = reviewed.scheduler_campaign_lock


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _one_candidate(
    rows: Sequence[Mapping[str, str]], *, label: str
) -> dict[str, str]:
    exact = [
        dict(row)
        for row in rows
        if row.get("candidate_physics_sha") == CANDIDATE_SHA256
    ]
    if len(exact) != 1:
        raise PostdeadlineContractError(
            f"{label} must contain candidate exactly once"
        )
    return exact[0]


def authenticate_official_candidate(
    *,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate frozen Standard12 row eight through source task bytes."""

    if sha256_file(standard_candidates_path) != STANDARD_CANDIDATES_SHA256:
        raise PostdeadlineContractError("official Standard12 CSV bytes drifted")
    if sha256_file(aggregate_manifest_path) != AGGREGATE_MANIFEST_SHA256:
        raise PostdeadlineContractError("aggregate manifest bytes drifted")
    manifest = validate_seal(
        read_json(aggregate_manifest_path),
        "mft-goal-20260726-global-pareto-v1",
    )
    standard_record = manifest.get("artifacts", {}).get(
        "standard_candidates"
    )
    if (
        manifest.get("payload_sha256") != AGGREGATE_PAYLOAD_SHA256
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("seed_count") != 512
        or manifest.get("standard_candidate_count") != 12
        or manifest.get("production_eligible") is not False
        or manifest.get("automatic_promotion_allowed") is not False
        or manifest.get("search_only_proposal") is not True
        or not isinstance(standard_record, dict)
        or standard_record.get("path") != "standard_candidates.csv"
        or standard_record.get("row_count") != 12
        or standard_record.get("sha256") != STANDARD_CANDIDATES_SHA256
    ):
        raise PostdeadlineContractError(
            "official aggregate Standard12 authority drifted"
        )
    official_rows = _csv_rows(standard_candidates_path)
    if len(official_rows) != 12:
        raise PostdeadlineContractError("official Standard12 row count drifted")
    row = _one_candidate(official_rows, label="official Standard12")
    expected_row = {
        "terminal_population_index": "15",
        "source_seed": str(SOURCE_SEED),
        "source_task_id": str(SOURCE_TASK_ID),
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "standard_selection_order": "8",
        "standard_selection_roles": "objective_space_maximin",
        "standard_selection_basis": "near_feasible_fallback",
        "physical_feasible": "False",
        "hard_feasible": "False",
        "feasible_rank": "-1",
        "global_non_dominated_rank": "-1",
    }
    drift = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in expected_row.items()
        if row.get(key) != expected
    }
    if drift:
        raise PostdeadlineContractError(
            f"official candidate-eight lineage/rank drifted: {drift}"
        )
    source_result_path = Path(str(row.get("source_result_path") or ""))
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise PostdeadlineContractError("source task result bytes drifted")
    source_result = read_json(source_result_path)
    unsigned_result = dict(source_result)
    source_result_payload = unsigned_result.pop("payload_sha256", None)
    if source_result_payload != payload_sha256(unsigned_result):
        raise PostdeadlineContractError("source task result seal drifted")
    terminal_manifest = source_result.get(
        "terminal_physical_candidates_manifest"
    )
    source_identity = (
        terminal_manifest.get("source_identity")
        if isinstance(terminal_manifest, dict)
        else None
    )
    if (
        source_result.get("campaign_id") != CAMPAIGN_ID
        or source_result.get("seed") != SOURCE_SEED
        or source_result.get("task_payload_sha256") != SOURCE_BUNDLE_ID
        or source_result.get("search_only_proposal") is not True
        or source_result.get("production_eligible") is not False
        or source_result.get("automatic_promotion_allowed") is not False
        or not isinstance(source_identity, dict)
        or source_identity.get("seed") != SOURCE_SEED
        or str(source_identity.get("task_id")) != str(SOURCE_TASK_ID)
        or source_identity.get("bundle_id") != SOURCE_BUNDLE_ID
    ):
        raise PostdeadlineContractError("source task result identity drifted")
    source_csv_record = source_result.get("artifact_inventory", {}).get(
        "terminal_physical_candidates"
    )
    if not isinstance(source_csv_record, dict):
        raise PostdeadlineContractError("source terminal CSV record is absent")
    source_csv_path = source_result_path.parent / str(
        source_csv_record.get("path") or ""
    )
    if (
        not source_csv_path.is_file()
        or sha256_file(source_csv_path) != source_csv_record.get("sha256")
    ):
        raise PostdeadlineContractError("source terminal CSV bytes drifted")
    source_row = _one_candidate(
        _csv_rows(source_csv_path), label="source task terminal CSV"
    )
    exact_columns = {
        "terminal_population_index",
        "decoder_valid",
        "surrogate_physical_valid",
        "surrogate_physicality_passed",
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
        "objective_volume_L",
        "objective_total_loss_W",
        "physical_constraint_feasible",
        "physical_feasible",
        "source_seed",
        "source_task_id",
        "source_bundle_id",
        "source_island_id",
        "dataset_sha256",
        "evaluation_model_sha256",
        "constraint_spec_sha256",
        "cooling_contract_sha256",
        "operating_point_sha256",
        "evaluation_model_artifacts_sha256",
        "evaluation_model_generation_sha256",
        "evaluation_spec_sha256",
        "evaluation_temperature_contract_sha256",
        "evaluation_hard_constraint_contract_sha256",
    }
    if any(row.get(key) != source_row.get(key) for key in exact_columns):
        raise PostdeadlineContractError(
            "official row differs from authenticated source terminal row"
        )
    parsed_columns: dict[str, Any] = {}
    for column in (
        "decoded_physical_params_json",
        "physical_G_json",
        "normalized_G_json",
        "coordinate_unit_json",
    ):
        try:
            official_value = json.loads(str(row.get(column)))
            source_value = json.loads(str(source_row.get(column)))
        except json.JSONDecodeError as exc:
            raise PostdeadlineContractError(
                f"candidate {column} is malformed"
            ) from exc
        if official_value != source_value:
            raise PostdeadlineContractError(
                f"official candidate {column} differs from source"
            )
        parsed_columns[column] = official_value
    decoded = parsed_columns["decoded_physical_params_json"]
    projection = {
        key: decoded.get(key) for key in sorted(BOUNDARY_PROJECTION)
    }
    expected_projection = {
        key: BOUNDARY_PROJECTION[key] for key in sorted(BOUNDARY_PROJECTION)
    }
    if projection != expected_projection:
        raise PostdeadlineContractError("candidate fixed cooling physics drifted")
    hard_cooling = source_result.get("hard_spec", {}).get(
        "fixed_cooling_identity"
    )
    if hard_cooling != {
        "core_k_thermal": 2.0,
        "core_plate_on": 1,
        "core_plate_pad_t": 2.0,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "thermal_pad_conductivity_W_mK": 0.2,
        "wcp_on": 1,
        "wcp_pad_t": 2.0,
    }:
        raise PostdeadlineContractError(
            "source hard-spec cooling identity drifted"
        )
    keys, key_contract = reviewed.all_input_keys(source_reader)
    missing = [key for key in keys if key not in decoded]
    if missing:
        raise PostdeadlineContractError(
            f"candidate lacks solver input keys: {missing}"
        )
    params = {key: decoded[key] for key in keys}
    authentication = {
        "official_standard_candidates": file_record(standard_candidates_path),
        "aggregate_manifest": file_record(aggregate_manifest_path),
        "aggregate_manifest_payload_sha256": manifest["payload_sha256"],
        "aggregate_seed_count": 512,
        "aggregate_standard_candidate_count": 12,
        "candidate_physics_sha256": CANDIDATE_SHA256,
        "candidate_standard_selection_order": 8,
        "candidate_standard_selection_role": "objective_space_maximin",
        "source_seed": SOURCE_SEED,
        "source_task_id": SOURCE_TASK_ID,
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "source_result": file_record(source_result_path),
        "source_result_payload_sha256": source_result_payload,
        "source_terminal_candidates": file_record(source_csv_path),
        "source_terminal_population_index": 15,
        "input_parameter_contract": key_contract,
        "candidate_boundary_projection": projection,
        "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "candidate_reauthenticated": True,
        "source_reauthenticated": True,
        "fixed_physics_unchanged": True,
    }
    selected = {
        "schema_version": CANDIDATE_SCHEMA,
        **SAFETY_FLAGS,
        "campaign_id": CAMPAIGN_ID,
        "authentication": authentication,
        "official_row": row,
        "source_row": source_row,
        "decoded_physical_params": decoded,
        "fea_params_sha256": payload_sha256(params),
    }
    return sealed(selected), params


def reviewed_profile() -> dict[str, Any]:
    if (
        PROFILE_PATH != reviewed.PROFILE_PATH
        or PROFILE_SHA256 != reviewed.PROFILE_SHA256
        or FIXED_BOUNDARY != reviewed.FIXED_BOUNDARY
    ):
        raise PostdeadlineContractError(
            "reviewed Standard profile authority drifted"
        )
    return reviewed.reviewed_profile()


@contextmanager
def _reviewed_contract_patch(*, submit: bool = False) -> Iterator[None]:
    """Temporarily bind reviewed transaction code to candidate-eight values."""

    replacements: dict[str, Any] = {
        "SCHEDULER_URL": SCHEDULER_URL,
        "PROJECT": PROJECT,
        "ACCOUNT_NAME": ACCOUNT_NAME,
        "NODE_NAME": NODE_NAME,
        "TASK_NAME": TASK_NAME,
        "WORKDIR": WORKDIR,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
        "POST_AUTHORIZATION": POST_AUTHORIZATION,
        "ATTEMPT_LEDGER_NAME": ATTEMPT_LEDGER_NAME,
        "SUBMISSION_DIRECTORY_NAME": SUBMISSION_DIRECTORY_NAME,
    }
    if submit:
        replacements.update(
            {
                "PLAN_SCHEMA": PLAN_SCHEMA,
                "CANDIDATE_SCHEMA": CANDIDATE_SCHEMA,
                "INTENT_SCHEMA": INTENT_SCHEMA,
                "SUBMISSION_SCHEMA": SUBMISSION_SCHEMA,
                "FINAL_SEAL_SCHEMA": FINAL_SEAL_SCHEMA,
                "SAFETY_FLAGS": SAFETY_FLAGS,
                "live_preflight": live_preflight,
            }
        )
    old = {name: getattr(reviewed, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(reviewed, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(reviewed, name, value)


def _capture_scheduler_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with _reviewed_contract_patch():
        return reviewed._capture_scheduler_payload(params, profile)


def validate_scheduler_payload(
    payload: Mapping[str, Any], retained: Mapping[str, Any]
) -> None:
    with _reviewed_contract_patch():
        reviewed.validate_scheduler_payload(payload, retained)


def _scheduler_payload_matches_reviewed_plan(
    plan: Mapping[str, Any],
    fresh_payload: Mapping[str, Any],
) -> bool:
    """Authenticate the sealed payload across one exact command hardening."""

    sealed_payload = plan.get("scheduler_payload")
    sealed_sha256 = plan.get("scheduler_payload_sha256")
    if not isinstance(sealed_payload, dict):
        return False
    if sealed_payload == fresh_payload:
        return sealed_sha256 == payload_sha256(fresh_payload)
    if (
        sealed_sha256 != HISTORICAL_SCHEDULER_PAYLOAD_SHA256
        or payload_sha256(sealed_payload)
        != HISTORICAL_SCHEDULER_PAYLOAD_SHA256
    ):
        return False
    fresh_command = str(fresh_payload.get("command") or "")
    if fresh_command.count(PYAEDT_LIBRARY_BINDING_COMMAND) != 1:
        return False
    reviewed_payload = copy.deepcopy(dict(fresh_payload))
    reviewed_payload["command"] = fresh_command.replace(
        PYAEDT_LIBRARY_BINDING_COMMAND,
        "",
        1,
    )
    return reviewed_payload == sealed_payload


def capacity_query() -> list[tuple[str, Any]]:
    return [
        ("cpus", CPUS),
        ("memory_mb", MEMORY_MB),
        ("scheduling_profile", "fea_bursty"),
        ("aedt_backend", "standalone"),
        ("required_capability", "conda:pyaedt2026v1"),
        ("env_profile", "pyaedt2026v1"),
        ("project", PROJECT),
        ("max_workers_per_node", MAX_WORKERS_PER_NODE),
        ("account_name", ACCOUNT_NAME),
        ("node_name", NODE_NAME),
        ("node_name_policy", "strict"),
    ]


def live_preflight(
    *,
    reader: JsonReader = get_json,
    expected_dedupe_key: str,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Require the exact strict n114 opening path and no active n114 FEA."""

    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError("preflight time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if now <= ORIGINAL_DEADLINE_UTC:
        raise PostdeadlineContractError(
            "post-deadline probe cannot precede original deadline"
        )
    health = reader("/api/health", None)
    source_task = reader(f"/api/tasks/{SOURCE_TASK_ID}", None)
    query = capacity_query()
    capacity = reader("/api/task-capacity", query)
    active = reader(
        "/api/tasks",
        [
            ("status", "queued"),
            ("status", "attaching"),
            ("status", "running"),
            ("limit", 10000),
        ],
    )
    collision_inventory = reader(
        "/api/tasks",
        [
            ("limit", 10000),
            ("project", PROJECT),
            ("name_prefix", TASK_NAME),
        ],
    )
    if (
        not isinstance(health, dict)
        or health.get("ok") is not True
        or health.get("scheduler_ok") is not True
        or health.get("scheduler_thread_alive") is not True
        or health.get("scheduler_stalled") is not False
    ):
        raise PostdeadlineContractError("Scheduler health gate failed")
    expected_source = {
        "task_id": SOURCE_TASK_ID,
        "name": SOURCE_TASK_NAME,
        "status": "completed",
        "exit_code": 0,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "dedupe_key": SOURCE_TASK_DEDUPE_KEY,
    }
    source_drift = {
        key: {"expected": expected, "actual": source_task.get(key)}
        for key, expected in expected_source.items()
        if not isinstance(source_task, dict)
        or source_task.get(key) != expected
    }
    if source_drift:
        raise PostdeadlineContractError(
            f"source Scheduler task identity drifted: {source_drift}"
        )
    if (
        not isinstance(capacity, dict)
        or capacity.get("queue_state") != "opening"
        or capacity.get("queue_reason") != OPENING_QUEUE_REASON
        or int(capacity.get("fit_slots") or 0) != 0
        or int(capacity.get("ready_fit_slots") or 0) != 0
        or int(capacity.get("pending_fit_slots") or 0) != 0
        or int(capacity.get("inflight_fit_slots") or 0) != 0
        or capacity.get("memory_pressure_state") != "ok"
        or capacity.get("preferred_node_relaxed") is not False
        or int(capacity.get("standalone_aedt_available") or 0) < 1
        or capacity.get("allocations") != []
    ):
        raise PostdeadlineContractError(
            "exact jji0930/n114 strict opening capacity gate failed"
        )
    active_rows = reviewed._task_rows(active)
    active_fea = [
        row
        for row in active_rows
        if row.get("status") in {"queued", "attaching", "running"}
        and (
            row.get("node_name") == NODE_NAME
            or row.get("requested_node_name") == NODE_NAME
        )
        and (
            row.get("aedt_backend") == "standalone"
            or row.get("scheduling_profile") == "fea_bursty"
        )
    ]
    if active_fea:
        raise PostdeadlineContractError("strict n114 lane is not FEA-empty")
    collisions = [
        row
        for row in reviewed._task_rows(collision_inventory)
        if row.get("name") == TASK_NAME
        or row.get("dedupe_key") == expected_dedupe_key
    ]
    if collisions:
        raise PostdeadlineContractError(
            "fresh official #8 task name or dedupe already exists"
        )
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "observed_at_utc": now.isoformat(),
        "scheduler_health": health,
        "source_task_identity": source_task,
        "capacity_query": query,
        "capacity": capacity,
        "queue_state": "opening",
        "queue_reason": OPENING_QUEUE_REASON,
        "ready_fit_slots": 0,
        "pending_fit_slots": 0,
        "inflight_fit_slots": 0,
        "preferred_node_relaxed": False,
        "standalone_aedt_available": int(
            capacity["standalone_aedt_available"]
        ),
        "active_n114_fea_count": 0,
        "collision_count": 0,
        "exact_account_node_opening": True,
        "strict_node_required": True,
        "fixed_physics_unchanged": True,
    }


def _attempt_nonce(payload: Mapping[str, Any]) -> str:
    return payload_sha256(
        {
            "schema_version": ATTEMPT_NONCE_SCHEMA,
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": payload["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "node_name_policy": "strict",
            "scheduler_payload_sha256": payload_sha256(payload),
        }
    )


def prepare(
    *,
    output: Path | None = None,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    reader: JsonReader = get_json,
    payload_builder: PayloadBuilder = _capture_scheduler_payload,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
    observed_at: datetime | None = None,
) -> Path:
    target = (output or OUTPUT_ROOT).resolve()
    if target != OUTPUT_ROOT.resolve():
        raise PostdeadlineContractError("prepare output root is not fixed")
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable prepare output already exists: {target}"
        )
    selected, params = authenticate_official_candidate(
        standard_candidates_path=standard_candidates_path,
        aggregate_manifest_path=aggregate_manifest_path,
        source_reader=source_reader,
    )
    profile = reviewed_profile()
    payload, environment, retained = payload_builder(params, profile)
    validate_scheduler_payload(payload, retained)
    preflight = live_preflight(
        reader=reader,
        expected_dedupe_key=str(payload["dedupe_key"]),
        observed_at=observed_at,
    )
    now = (
        observed_at.astimezone(timezone.utc)
        if observed_at is not None
        else datetime.now(timezone.utc)
    )
    nonce = _attempt_nonce(payload)
    attempt_path = target / ATTEMPT_LEDGER_NAME
    submission_path = target / SUBMISSION_DIRECTORY_NAME
    staging = target.with_name(
        f".{target.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    staging.mkdir(parents=True)
    try:
        selected_path = write_immutable_json(
            staging / "selected_candidate.json", selected
        )
        params_path = write_immutable_json(staging / "fea_params.json", params)
        profile_path = write_immutable_json(
            staging / "execution_profile.json", profile
        )
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "campaign_id": CAMPAIGN_ID,
                "standard_only": True,
                "symmetric_model": True,
                "full_model": False,
                "thermal_symmetry": "eighth",
                "candidate_physics_sha256": CANDIDATE_SHA256,
                "source_seed": SOURCE_SEED,
                "source_task_id": SOURCE_TASK_ID,
                "official_standard_selection_order": 8,
                "selected_candidate": relative_record(selected_path),
                "selected_candidate_payload_sha256": selected[
                    "payload_sha256"
                ],
                "fea_params": relative_record(params_path),
                "fea_params_sha256": payload_sha256(params),
                "execution_profile": relative_record(profile_path),
                "execution_profile_canonical_sha256": payload_sha256(
                    profile
                ),
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "task_name": TASK_NAME,
                "workdir": WORKDIR,
                "dedupe_key": payload["dedupe_key"],
                "resources": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "solver_seconds": SOLVER_SECONDS,
                    "kill_grace_seconds": KILL_GRACE_SECONDS,
                    "retention_seconds": RETENTION_SECONDS,
                    "scheduler_timeout_seconds": SCHEDULER_SECONDS,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                },
                "placement": {
                    "account_name": ACCOUNT_NAME,
                    "node_name": NODE_NAME,
                    "node_name_policy": "strict",
                    "same_node_as_task_id": 0,
                    "dependency_task_id": 0,
                    "allocation_state_at_prepare": "opening",
                    "allocation_id_at_prepare": None,
                    "slurm_job_id_at_prepare": None,
                    "demand_pool_opening_required": True,
                    "preferred_node_relaxed_allowed": False,
                },
                "opening_demand_pool_contract": {
                    "allowed_pre_submit_queue_state": "opening",
                    "required_queue_reason": OPENING_QUEUE_REASON,
                    "ready_fit_slots": 0,
                    "pending_fit_slots": 0,
                    "inflight_fit_slots": 0,
                    "active_target_fea_count": 0,
                    "require_reget_inside_mutation_lock": True,
                    "preferred_node_relaxed_allowed": False,
                    "relaxed_allocation_allowed": False,
                },
                "scheduler_url": SCHEDULER_URL,
                "scheduler_project": PROJECT,
                "scheduler_repository_modified": False,
                "scheduler_project_mutation_performed": False,
                "scheduler_submission_performed": False,
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "fixed_physics_unchanged": True,
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(environment),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "fresh_retention_identity": True,
                "fresh_chunk_identity": True,
                "preflight": preflight,
                "submit_authorization_token": POST_AUTHORIZATION,
                "single_attempt_contract": {
                    "schema_version": SINGLE_ATTEMPT_SCHEMA,
                    "attempt_ledger_path": str(attempt_path),
                    "submission_output_path": str(submission_path),
                    "attempt_nonce": nonce,
                    "post_call_budget": 1,
                    "ledger_consumed_before_network": True,
                    "output_override_allowed": False,
                },
            }
        )
        plan_path = write_immutable_json(staging / PLAN_NAME, plan)
        submit_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "submit",
            "--authorize-post",
            POST_AUTHORIZATION,
        ]
        receipt = sealed(
            {
                "schema_version": PREPARE_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "plan": relative_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_payload_sha256": plan[
                    "scheduler_payload_sha256"
                ],
                "task_name": TASK_NAME,
                "dedupe_key": payload["dedupe_key"],
                "account_name": ACCOUNT_NAME,
                "node_name": NODE_NAME,
                "node_name_policy": "strict",
                "queue_state_at_prepare": "opening",
                "allocation_id_at_prepare": None,
                "slurm_job_id_at_prepare": None,
                "submit_command_argv": submit_command,
                "scheduler_get_preflight_performed": True,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "attempt_ledger_path": str(attempt_path),
                "attempt_nonce": nonce,
                "ready_for_explicit_submit": True,
                "strict_opening_only": True,
                "relaxed_allocation_allowed": False,
            }
        )
        write_immutable_json(staging / "prepare_receipt.json", receipt)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def load_plan(
    plan_path: Path,
    *,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    payload_builder: PayloadBuilder = _capture_scheduler_payload,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = plan_path.resolve(strict=True)
    expected_plan = OUTPUT_ROOT.resolve() / PLAN_NAME
    if resolved != expected_plan:
        raise PostdeadlineContractError("plan path escaped the fixed output root")
    plan = validate_seal(read_json(resolved), PLAN_SCHEMA)
    single_attempt = plan.get("single_attempt_contract")
    expected_attempt_path = resolved.parent / ATTEMPT_LEDGER_NAME
    expected_submission_path = resolved.parent / SUBMISSION_DIRECTORY_NAME
    payload = plan.get("scheduler_payload")
    expected_nonce = (
        _attempt_nonce(payload) if isinstance(payload, dict) else None
    )
    preflight = plan.get("preflight")
    opening = plan.get("opening_demand_pool_contract")
    placement = plan.get("placement")
    if (
        any(plan.get(key) is not value for key, value in SAFETY_FLAGS.items())
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("candidate_physics_sha256") != CANDIDATE_SHA256
        or plan.get("source_seed") != SOURCE_SEED
        or plan.get("source_task_id") != SOURCE_TASK_ID
        or plan.get("official_standard_selection_order") != 8
        or plan.get("task_name") != TASK_NAME
        or plan.get("workdir") != WORKDIR
        or plan.get("scheduler_url") != SCHEDULER_URL
        or plan.get("scheduler_project") != PROJECT
        or not isinstance(placement, dict)
        or placement.get("account_name") != ACCOUNT_NAME
        or placement.get("node_name") != NODE_NAME
        or placement.get("node_name_policy") != "strict"
        or placement.get("allocation_state_at_prepare") != "opening"
        or placement.get("allocation_id_at_prepare") is not None
        or placement.get("slurm_job_id_at_prepare") is not None
        or placement.get("preferred_node_relaxed_allowed") is not False
        or plan.get("resources", {}).get("cpus") != CPUS
        or plan.get("resources", {}).get("memory_mb") != MEMORY_MB
        or plan.get("resources", {}).get("scheduler_timeout_seconds")
        != SCHEDULER_SECONDS
        or plan.get("resources", {}).get("max_workers_per_node")
        != MAX_WORKERS_PER_NODE
        or plan.get("fixed_boundary") != FIXED_BOUNDARY
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_mutation_performed") is not False
        or plan.get("scheduler_submission_performed") is not False
        or not isinstance(preflight, dict)
        or preflight.get("schema_version") != PREFLIGHT_SCHEMA
        or preflight.get("queue_state") != "opening"
        or preflight.get("active_n114_fea_count") != 0
        or preflight.get("collision_count") != 0
        or preflight.get("preferred_node_relaxed") is not False
        or not isinstance(opening, dict)
        or opening.get("allowed_pre_submit_queue_state") != "opening"
        or opening.get("require_reget_inside_mutation_lock") is not True
        or opening.get("relaxed_allocation_allowed") is not False
        or not isinstance(single_attempt, dict)
        or single_attempt.get("schema_version") != SINGLE_ATTEMPT_SCHEMA
        or Path(
            str(single_attempt.get("attempt_ledger_path") or "")
        ).resolve()
        != expected_attempt_path
        or Path(
            str(single_attempt.get("submission_output_path") or "")
        ).resolve()
        != expected_submission_path
        or single_attempt.get("attempt_nonce") != expected_nonce
        or single_attempt.get("post_call_budget") != 1
        or single_attempt.get("ledger_consumed_before_network") is not True
        or single_attempt.get("output_override_allowed") is not False
    ):
        raise PostdeadlineContractError("sealed candidate-eight plan drifted")
    root = resolved.parent
    selected_path = reviewed._contained_artifact(
        root, plan.get("selected_candidate"), "selected candidate"
    )
    params_path = reviewed._contained_artifact(
        root, plan.get("fea_params"), "FEA params"
    )
    profile_path = reviewed._contained_artifact(
        root, plan.get("execution_profile"), "execution profile"
    )
    selected = validate_seal(read_json(selected_path), CANDIDATE_SCHEMA)
    params = read_json(params_path)
    profile = read_json(profile_path)
    fresh_selected, fresh_params = authenticate_official_candidate(
        standard_candidates_path=standard_candidates_path,
        aggregate_manifest_path=aggregate_manifest_path,
        source_reader=source_reader,
    )
    fresh_profile = reviewed_profile()
    fresh_payload, environment, retained = payload_builder(
        fresh_params, fresh_profile
    )
    validate_scheduler_payload(fresh_payload, retained)
    if (
        selected != fresh_selected
        or params != fresh_params
        or profile != fresh_profile
        or plan.get("fea_params_sha256") != payload_sha256(params)
        or plan.get("execution_profile_canonical_sha256")
        != payload_sha256(profile)
        or not _scheduler_payload_matches_reviewed_plan(
            plan,
            fresh_payload,
        )
        or plan.get("dedupe_key") != fresh_payload["dedupe_key"]
        or plan.get("submission_environment") != environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(environment)
        or plan.get("retained_aedt_bundle") != retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(retained)
    ):
        raise PostdeadlineContractError(
            "sealed plan no longer matches reauthenticated inputs"
        )
    return plan, params, profile


def _strict_readback_reader(reader: JsonReader) -> JsonReader:
    def strict(
        path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        value = reader(path, query)
        prefix = "/api/tasks/"
        suffix = path[len(prefix) :] if path.startswith(prefix) else ""
        if suffix.isdigit() and int(suffix) != SOURCE_TASK_ID:
            if (
                not isinstance(value, dict)
                or value.get("requested_account_name") != ACCOUNT_NAME
                or value.get("requested_node_name") != NODE_NAME
                or value.get("requested_node_name_policy") != "strict"
                or value.get("preferred_node_relaxed") is not False
                or (
                    value.get("node_name")
                    and value.get("node_name") != NODE_NAME
                )
                or (
                    value.get("actual_node_name")
                    and value.get("actual_node_name") != NODE_NAME
                )
                or (
                    value.get("allocation_node_name")
                    and value.get("allocation_node_name") != NODE_NAME
                )
                or value.get("node_name_policy") not in {None, "strict"}
            ):
                raise PostdeadlineContractError(
                    "durable strict n114 readback relaxed or drifted"
                )
        return value

    return strict


def submit(
    *,
    plan_path: Path | None = None,
    output: Path | None = None,
    authorize_post: str,
    reader: JsonReader = get_json,
    poster: Poster = reviewed._post_json_once,
    observed_at: datetime | None = None,
    lock_factory: Callable[[], Any] = scheduler_campaign_lock,
    plan_loader: PlanLoader = load_plan,
) -> Path:
    """Perform at most one POST using the reviewed locked transaction."""

    if authorize_post != POST_AUTHORIZATION:
        raise PostdeadlineContractError(
            "explicit official8 jji0930/n114 POST authorization is absent"
        )
    resolved_plan = (plan_path or (OUTPUT_ROOT / PLAN_NAME)).resolve()
    resolved_output = (
        output or (OUTPUT_ROOT / SUBMISSION_DIRECTORY_NAME)
    ).resolve()
    if resolved_plan != (OUTPUT_ROOT.resolve() / PLAN_NAME):
        raise PostdeadlineContractError("submit plan path is not fixed")
    if resolved_output != (
        OUTPUT_ROOT.resolve() / SUBMISSION_DIRECTORY_NAME
    ):
        raise PostdeadlineContractError("submission output path is not fixed")
    strict_reader = _strict_readback_reader(reader)
    with _reviewed_contract_patch(submit=True):
        result = reviewed.submit(
            plan_path=resolved_plan,
            output=resolved_output,
            authorize_post=authorize_post,
            reader=strict_reader,
            poster=poster,
            observed_at=observed_at,
            lock_factory=lock_factory,
            plan_loader=plan_loader,
        )
    validate_seal(read_json(result), FINAL_SEAL_SCHEMA)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sealed official Standard12 candidate #8 strict n114 probe"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare",
        help="authenticate and seal the fixed-root opening plan without POST",
    )
    commands.add_parser(
        "inspect",
        help="reauthenticate the fixed-root plan and repeat GET preflight",
    )
    submit_parser = commands.add_parser(
        "submit",
        help="reauthenticate and perform at most one strict Scheduler POST",
    )
    submit_parser.add_argument("--authorize-post", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan_path = OUTPUT_ROOT / PLAN_NAME
    if args.command == "prepare":
        plan_path = prepare()
        plan = validate_seal(read_json(plan_path), PLAN_SCHEMA)
        receipt = validate_seal(
            read_json(plan_path.parent / "prepare_receipt.json"),
            PREPARE_SCHEMA,
        )
        result = {
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "queue_state": "opening",
            "scheduler_post_calls": 0,
            "submit_command_argv": receipt["submit_command_argv"],
        }
    elif args.command == "inspect":
        plan, _params, _profile = load_plan(plan_path)
        preflight = live_preflight(
            expected_dedupe_key=plan["dedupe_key"]
        )
        result = {
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": CANDIDATE_SHA256,
            "task_name": TASK_NAME,
            "dedupe_key": plan["dedupe_key"],
            "account_name": ACCOUNT_NAME,
            "node_name": NODE_NAME,
            "queue_state": preflight["queue_state"],
            "active_n114_fea_count": preflight[
                "active_n114_fea_count"
            ],
            "collision_count": preflight["collision_count"],
            "preferred_node_relaxed": preflight[
                "preferred_node_relaxed"
            ],
            "scheduler_post_calls": 0,
            "ready_for_explicit_submit": True,
        }
    else:
        seal_path = submit(authorize_post=args.authorize_post)
        final_seal = validate_seal(
            read_json(seal_path), FINAL_SEAL_SCHEMA
        )
        result = {
            "final_seal": str(seal_path),
            "final_seal_sha256": sha256_file(seal_path),
            "task_id": final_seal["task_id"],
            "scheduler_post_calls": final_seal[
                "scheduler_post_calls"
            ],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
