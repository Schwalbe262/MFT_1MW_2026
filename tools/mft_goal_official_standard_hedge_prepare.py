"""Prepare three sealed post-deadline Standard hedge lanes without POST.

The frozen 512-seed aggregate has no production-feasible row.  This tool
therefore prepares diagnostic/search-only Standard FEA plans for three
complementary official rows.  It reuses the reviewed candidate-eight
strict-opening GET gate and the reviewed Standard payload derivation, but it
deliberately exposes no submit command or Scheduler mutation function.

The three placements are fixed:

* official row 1 -> ``dhj02/n110`` (minimum-volume extreme)
* official row 12 -> ``r1jae262/n112`` (low-loss thermal hedge)
* official row 5 -> ``jji0930/n115`` (largest resonance-margin hedge)

All lanes retain the fixed 1.5 m/s dual-fan and 0.2 W/(m K), 2 mm TIM
contract.  Prepared plans remain noncanonical and cannot support a production
PASS claim.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

reviewed = importlib.import_module(
    "tools.mft_goal_official_standard_postdeadline"
)
candidate8 = importlib.import_module(
    "tools.mft_goal_official_standard8_postdeadline"
)

PostdeadlineContractError = reviewed.PostdeadlineContractError
JsonReader = Callable[[str, Sequence[tuple[str, Any]] | None], Any]

CAMPAIGN_ID = candidate8.CAMPAIGN_ID
SCHEDULER_URL = candidate8.SCHEDULER_URL
PROJECT = candidate8.PROJECT
AGGREGATE_ROOT = candidate8.AGGREGATE_ROOT
STANDARD_CANDIDATES = candidate8.STANDARD_CANDIDATES
AGGREGATE_MANIFEST = candidate8.AGGREGATE_MANIFEST
STANDARD_CANDIDATES_SHA256 = candidate8.STANDARD_CANDIDATES_SHA256
AGGREGATE_MANIFEST_SHA256 = candidate8.AGGREGATE_MANIFEST_SHA256
AGGREGATE_PAYLOAD_SHA256 = candidate8.AGGREGATE_PAYLOAD_SHA256

SOLVER_REVISION = candidate8.SOLVER_REVISION
LIBRARY_REVISION = candidate8.LIBRARY_REVISION
CPUS = candidate8.CPUS
MEMORY_MB = candidate8.MEMORY_MB
SOLVER_SECONDS = candidate8.SOLVER_SECONDS
KILL_GRACE_SECONDS = candidate8.KILL_GRACE_SECONDS
RETENTION_SECONDS = candidate8.RETENTION_SECONDS
SCHEDULER_SECONDS = candidate8.SCHEDULER_SECONDS
MAX_WORKERS_PER_NODE = candidate8.MAX_WORKERS_PER_NODE
PRIORITY = candidate8.PRIORITY
ORIGINAL_DEADLINE_UTC = candidate8.ORIGINAL_DEADLINE_UTC
OPENING_QUEUE_REASON = candidate8.OPENING_QUEUE_REASON
FIXED_BOUNDARY = copy.deepcopy(candidate8.FIXED_BOUNDARY)
BOUNDARY_PROJECTION = copy.deepcopy(candidate8.BOUNDARY_PROJECTION)
SAFETY_FLAGS = copy.deepcopy(candidate8.SAFETY_FLAGS)

OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_official_hedge3_prepare_260726_v1"
)
MANIFEST_NAME = "prepare_manifest.json"
PLAN_NAME = "diagnostic_prepare_only_plan.json"
SELECTED_NAME = "selected_candidate.json"
PARAMS_NAME = "fea_params.json"
PROFILE_NAME = "execution_profile.json"
RECEIPT_NAME = "prepare_receipt.json"

MANIFEST_SCHEMA = "mft-goal-official-standard-hedge3-prepare-manifest-v1"
PLAN_SCHEMA = "mft-goal-official-standard-hedge-prepare-only-plan-v1"
CANDIDATE_SCHEMA = (
    "mft-goal-official-standard-hedge-selected-candidate-v1"
)
PREPARE_SCHEMA = "mft-goal-official-standard-hedge-prepare-receipt-v1"
PREFLIGHT_SCHEMA = (
    "mft-goal-official-standard-hedge-opening-live-preflight-v1"
)

ALLOWED_ACCOUNTS = frozenset({"dhj02", "r1jae262", "jji0930"})
EXCLUDED_ACCOUNT_GUARDS = {
    "harry261": "Scheduler-authoritative storage guard exclusion",
}

canonical_bytes = reviewed.canonical_bytes
payload_sha256 = reviewed.payload_sha256
sealed = reviewed.sealed
validate_seal = reviewed.validate_seal
sha256_file = reviewed.sha256_file
file_record = reviewed.file_record
read_json = reviewed.read_json
write_immutable_json = reviewed.write_immutable_json
get_json = reviewed.get_json


@dataclass(frozen=True)
class CandidateSpec:
    """Immutable authority and placement for one official Standard row."""

    selection_order: int
    selection_role: str
    terminal_population_index: int
    candidate_sha256: str
    canonical_row_json_sha256: str
    canonical_physical_params_sha256: str
    raw_fea_params_sha256: str
    profiled_fea_params_sha256: str
    source_seed: int
    source_task_id: int
    source_task_name: str
    source_task_dedupe_key: str
    source_bundle_id: str
    source_result_sha256: str
    source_result_payload_sha256: str
    source_terminal_csv_sha256: str
    account_name: str
    node_name: str

    @property
    def short_sha(self) -> str:
        return self.candidate_sha256[:12]

    @property
    def lane_name(self) -> str:
        return (
            f"official{self.selection_order}_{self.short_sha}_"
            f"{self.node_name}"
        )

    @property
    def task_name(self) -> str:
        return (
            "mft-goal-diag-standard-postdeadline-"
            f"official{self.selection_order}-s{self.source_task_id}-"
            f"{self.short_sha}-{self.node_name}"
        )

    @property
    def workdir(self) -> str:
        return (
            "mft_goal_diag_standard_postdeadline_"
            f"official{self.selection_order}_s{self.source_task_id}_"
            f"{self.short_sha}_{self.node_name}"
        )


CANDIDATES = (
    CandidateSpec(
        selection_order=1,
        selection_role="minimum_volume",
        terminal_population_index=136,
        candidate_sha256=(
            "896084a5979344670a70a65484d985f8b244de43f0975fe1023cf9020ebb5860"
        ),
        canonical_row_json_sha256=(
            "040a67d10c9fbb2093a652e851eaf83abf06a7d06adebd464d4869b4f064ad34"
        ),
        canonical_physical_params_sha256=(
            "8da8b0bc958dc58ec24e01221673d3c035831e5497438ec313cffc2449cf6407"
        ),
        raw_fea_params_sha256=(
            "195b1c93d97ca594579f51d06cbe70c6e4ed64fc1c3078a1a78843855e8b3420"
        ),
        profiled_fea_params_sha256=(
            "b0dbb3cde8e570b4fbf0294a97444e7d0b9bab1a9bffd755686c4f11a00ebd49"
        ),
        source_seed=2607262329,
        source_task_id=96009,
        source_task_name="mft-goal-nsga-s2607262329-n1-6",
        source_task_dedupe_key=(
            "mft-goal-20260726-nsga:"
            "c9e2ae0f4deaed283fb81e2c32e7ae28845da42bb4308b7dada642f3b6b33ef6"
        ),
        source_bundle_id=(
            "86199db6789876e3d1b02005ddf2e72fdae452d9800bb6481f30962f49a599ec"
        ),
        source_result_sha256=(
            "ddc597f26e505adb2f30abfe3e591b96ff34b32ee51fa6a1747ba3ee5f0a76b5"
        ),
        source_result_payload_sha256=(
            "f89bc43d522f9ee64f11ca6f41ad3dd3c86a53692b570a84046efbbb9f1b2cca"
        ),
        source_terminal_csv_sha256=(
            "e11d0a2abd56e30cc31a04d18687ed58f544e828bbe947954627beb2a39c11c4"
        ),
        account_name="dhj02",
        node_name="n110",
    ),
    CandidateSpec(
        selection_order=12,
        selection_role="objective_space_maximin",
        terminal_population_index=128,
        candidate_sha256=(
            "828cb282cf4febdabd8ad3db58153cefbc737da5b0826e5620a3012960aa22ab"
        ),
        canonical_row_json_sha256=(
            "0d87ee81a49527d95bc4e2218e1867b65080d3a2e43fa8287d4f5ad02e2fad92"
        ),
        canonical_physical_params_sha256=(
            "1312ffc56c50cbfcb0be434f2c417f00bb04bccef8f5a4dea74f84180810453f"
        ),
        raw_fea_params_sha256=(
            "9ff8aa71aa979c1f2614024a73b73da713f18385a9c5839ae13629a706c8c125"
        ),
        profiled_fea_params_sha256=(
            "9c12f71706927f41470d38aecbdbdac9fd3ab10b33ac1bdf61901a01da715771"
        ),
        source_seed=2607262505,
        source_task_id=96185,
        source_task_name="mft-goal-nsga-s2607262505-n1-6",
        source_task_dedupe_key=(
            "mft-goal-20260726-nsga:"
            "264eb24fbb8e283eee75da263af89812bd543a611ec8dfd7e92cdc5b0df54a44"
        ),
        source_bundle_id=(
            "7e6276c58b44f5f2065ca80a16093617d121dc26cbf268406dacaa49b895ae38"
        ),
        source_result_sha256=(
            "7cb9aab49a7cfde376f4aa7e10345d0c587271b82e9a0f34f72b9813206dd2b7"
        ),
        source_result_payload_sha256=(
            "25c7f47eb8f44b1bd60a475630fa1fb74f05ecf2c3b4149eef9746cd089c9c5b"
        ),
        source_terminal_csv_sha256=(
            "a15ba2458981bf7b01183f9f7db70b162536e0402bc8fc7353757e3d13d8042b"
        ),
        account_name="r1jae262",
        node_name="n112",
    ),
    CandidateSpec(
        selection_order=5,
        selection_role="objective_space_maximin",
        terminal_population_index=215,
        candidate_sha256=(
            "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
        ),
        canonical_row_json_sha256=(
            "782f561d097576bd1c63314ae77b9773a18440f3604df2f8c1c767f50fc5205b"
        ),
        canonical_physical_params_sha256=(
            "b1a938197d4144b50f81e4c3ec5a30e86125cc41ec9ccf687d60176bef10a8a0"
        ),
        raw_fea_params_sha256=(
            "e2960709bb77ded8521caed8bec4fd0eb594d1a059c65a92fd8e5fcae72e85a9"
        ),
        profiled_fea_params_sha256=(
            "12aa95d67bc957bc057b19399b3721e88fe433ae61a91a893d7a8fa833a120bb"
        ),
        source_seed=2607262233,
        source_task_id=95913,
        source_task_name="mft-goal-nsga-s2607262233-n1-6",
        source_task_dedupe_key=(
            "mft-goal-20260726-nsga:"
            "d12d7dc26cf937cdb69c25165a51e57cad96cd0f18abd33d1570a53b337e2da7"
        ),
        source_bundle_id=(
            "8c9e57fe36eda1712ced3bb299a1a53bf4f398c6cee795c4dd5a60c86a6af1af"
        ),
        source_result_sha256=(
            "9ec3b4dd1bfa27af09a0935e4d70020d74db253be9da60d437ea12f8761caa29"
        ),
        source_result_payload_sha256=(
            "f448b21a69b8785c34826eb9e2e62ef7432df4779344b10cfd725f8744827e18"
        ),
        source_terminal_csv_sha256=(
            "ef7079060359de4087417fad812fda831c93e2fedf471aaf58e8c661d65baa2a"
        ),
        account_name="jji0930",
        node_name="n115",
    ),
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.resolve(strict=True).open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PostdeadlineContractError(
            f"CSV is unavailable: {path}"
        ) from exc


def _one_candidate(
    rows: Sequence[Mapping[str, str]],
    spec: CandidateSpec,
    *,
    label: str,
) -> dict[str, str]:
    matches = [
        dict(row)
        for row in rows
        if row.get("candidate_physics_sha") == spec.candidate_sha256
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            f"{label} must contain candidate exactly once"
        )
    return matches[0]


def _profiled_params(
    params: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, dict):
        raise PostdeadlineContractError(
            "reviewed Standard profile overrides are absent"
        )
    result = copy.deepcopy(dict(params))
    result.update(copy.deepcopy(overrides))
    return result


def authenticate_candidate(
    spec: CandidateSpec,
    *,
    standard_candidates_path: Path = STANDARD_CANDIDATES,
    aggregate_manifest_path: Path = AGGREGATE_MANIFEST,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate one frozen official row through its source task bytes."""

    if spec.account_name not in ALLOWED_ACCOUNTS:
        raise PostdeadlineContractError("candidate account is not allowed")
    if spec.account_name in EXCLUDED_ACCOUNT_GUARDS:
        raise PostdeadlineContractError(
            "candidate account is excluded by Scheduler guard"
        )
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
        or standard_record.get("path") != STANDARD_CANDIDATES.name
        or standard_record.get("row_count") != 12
        or standard_record.get("sha256") != STANDARD_CANDIDATES_SHA256
    ):
        raise PostdeadlineContractError(
            "official aggregate Standard12 authority drifted"
        )
    official_rows = _csv_rows(standard_candidates_path)
    if len(official_rows) != 12:
        raise PostdeadlineContractError("official Standard12 row count drifted")
    row = _one_candidate(official_rows, spec, label="official Standard12")
    expected_row = {
        "terminal_population_index": str(spec.terminal_population_index),
        "canonical_physical_params_sha256": (
            spec.canonical_physical_params_sha256
        ),
        "source_seed": str(spec.source_seed),
        "source_task_id": str(spec.source_task_id),
        "source_bundle_id": spec.source_bundle_id,
        "source_result_sha256": spec.source_result_sha256,
        "standard_selection_order": str(spec.selection_order),
        "standard_selection_roles": spec.selection_role,
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
    if drift or payload_sha256(row) != spec.canonical_row_json_sha256:
        raise PostdeadlineContractError(
            f"official candidate row drifted: {drift}"
        )

    source_result_path = Path(str(row.get("source_result_path") or ""))
    if sha256_file(source_result_path) != spec.source_result_sha256:
        raise PostdeadlineContractError("source task result bytes drifted")
    source_result = read_json(source_result_path)
    unsigned_result = dict(source_result)
    source_result_payload = unsigned_result.pop("payload_sha256", None)
    terminal_manifest = source_result.get(
        "terminal_physical_candidates_manifest"
    )
    source_identity = (
        terminal_manifest.get("source_identity")
        if isinstance(terminal_manifest, dict)
        else None
    )
    if (
        source_result_payload != spec.source_result_payload_sha256
        or source_result_payload != payload_sha256(unsigned_result)
        or source_result.get("campaign_id") != CAMPAIGN_ID
        or source_result.get("seed") != spec.source_seed
        or source_result.get("task_payload_sha256")
        != spec.source_bundle_id
        or source_result.get("search_only_proposal") is not True
        or source_result.get("production_eligible") is not False
        or source_result.get("automatic_promotion_allowed") is not False
        or not isinstance(source_identity, dict)
        or source_identity.get("seed") != spec.source_seed
        or str(source_identity.get("task_id"))
        != str(spec.source_task_id)
        or source_identity.get("bundle_id") != spec.source_bundle_id
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
        or source_csv_record.get("sha256")
        != spec.source_terminal_csv_sha256
        or sha256_file(source_csv_path) != spec.source_terminal_csv_sha256
    ):
        raise PostdeadlineContractError("source terminal CSV bytes drifted")
    source_row = _one_candidate(
        _csv_rows(source_csv_path),
        spec,
        label="source task terminal CSV",
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
                f"candidate {column} differs from source"
            )
        parsed_columns[column] = official_value

    decoded = parsed_columns["decoded_physical_params_json"]
    projection = {
        key: decoded.get(key) for key in sorted(BOUNDARY_PROJECTION)
    }
    if projection != {
        key: BOUNDARY_PROJECTION[key] for key in sorted(BOUNDARY_PROJECTION)
    }:
        raise PostdeadlineContractError("candidate cooling physics drifted")
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
    if payload_sha256(params) != spec.raw_fea_params_sha256:
        raise PostdeadlineContractError("candidate FEA params drifted")
    profile = candidate8.reviewed_profile()
    if (
        payload_sha256(_profiled_params(params, profile))
        != spec.profiled_fea_params_sha256
    ):
        raise PostdeadlineContractError(
            "candidate profiled FEA params drifted"
        )

    selected = sealed(
        {
            "schema_version": CANDIDATE_SCHEMA,
            **SAFETY_FLAGS,
            "campaign_id": CAMPAIGN_ID,
            "prepare_only": True,
            "submission_capability_present": False,
            "authentication": {
                "official_standard_candidates": file_record(
                    standard_candidates_path
                ),
                "aggregate_manifest": file_record(aggregate_manifest_path),
                "aggregate_manifest_payload_sha256": manifest[
                    "payload_sha256"
                ],
                "aggregate_seed_count": 512,
                "aggregate_standard_candidate_count": 12,
                "candidate_physics_sha256": spec.candidate_sha256,
                "candidate_canonical_row_json_sha256": (
                    spec.canonical_row_json_sha256
                ),
                "candidate_canonical_physical_params_sha256": (
                    spec.canonical_physical_params_sha256
                ),
                "candidate_standard_selection_order": (
                    spec.selection_order
                ),
                "candidate_standard_selection_role": spec.selection_role,
                "source_seed": spec.source_seed,
                "source_task_id": spec.source_task_id,
                "source_bundle_id": spec.source_bundle_id,
                "source_result": file_record(source_result_path),
                "source_result_payload_sha256": source_result_payload,
                "source_terminal_candidates": file_record(source_csv_path),
                "source_terminal_population_index": (
                    spec.terminal_population_index
                ),
                "input_parameter_contract": key_contract,
                "candidate_boundary_projection": projection,
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "candidate_reauthenticated": True,
                "source_reauthenticated": True,
                "fixed_physics_unchanged": True,
            },
            "official_row": row,
            "source_row": source_row,
            "decoded_physical_params": decoded,
            "raw_fea_params_sha256": spec.raw_fea_params_sha256,
            "profiled_fea_params_sha256": (
                spec.profiled_fea_params_sha256
            ),
        }
    )
    return selected, params


@contextmanager
def _reviewed_payload_patch(spec: CandidateSpec) -> Iterator[None]:
    replacements = {
        "SCHEDULER_URL": SCHEDULER_URL,
        "PROJECT": PROJECT,
        "ACCOUNT_NAME": spec.account_name,
        "NODE_NAME": spec.node_name,
        "TASK_NAME": spec.task_name,
        "WORKDIR": spec.workdir,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
    }
    previous = {name: getattr(reviewed, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(reviewed, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(reviewed, name, value)


def derive_scheduler_payload(
    spec: CandidateSpec,
    params: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive a payload using an in-process fake POST; no network mutation."""

    with _reviewed_payload_patch(spec):
        payload, environment, retained = reviewed._capture_scheduler_payload(
            params, profile
        )
        reviewed.validate_scheduler_payload(payload, retained)
    return payload, environment, retained


@contextmanager
def _candidate8_get_patch(spec: CandidateSpec) -> Iterator[None]:
    replacements = {
        "ACCOUNT_NAME": spec.account_name,
        "NODE_NAME": spec.node_name,
        "CANDIDATE_SHA256": spec.candidate_sha256,
        "SOURCE_SEED": spec.source_seed,
        "SOURCE_TASK_ID": spec.source_task_id,
        "SOURCE_TASK_NAME": spec.source_task_name,
        "SOURCE_TASK_DEDUPE_KEY": spec.source_task_dedupe_key,
        "SOURCE_BUNDLE_ID": spec.source_bundle_id,
        "SOURCE_RESULT_SHA256": spec.source_result_sha256,
        "TASK_NAME": spec.task_name,
    }
    previous = {name: getattr(candidate8, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(candidate8, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(candidate8, name, value)


def _account_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PostdeadlineContractError(
            "Scheduler account status inventory is malformed"
        )
    return [row for row in value if isinstance(row, dict)]


def _capability_accounts(value: Any, capability: str) -> set[str]:
    if not isinstance(value, list):
        raise PostdeadlineContractError(
            "Scheduler capability inventory is malformed"
        )
    for row in value:
        if (
            isinstance(row, dict)
            and row.get("capability") == capability
            and isinstance(row.get("accounts"), list)
        ):
            return {str(account) for account in row["accounts"]}
    return set()


def live_preflight(
    spec: CandidateSpec,
    *,
    expected_dedupe_key: str,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Run only GET gates for one exact account/node strict opening lane."""

    if spec.account_name not in ALLOWED_ACCOUNTS:
        raise PostdeadlineContractError("candidate account is not allowed")
    with _candidate8_get_patch(spec):
        base = candidate8.live_preflight(
            reader=reader,
            expected_dedupe_key=expected_dedupe_key,
            observed_at=observed_at,
        )
    accounts = _account_rows(reader("/api/accounts/status/live", None))
    matches = [
        row for row in accounts if row.get("account_name") == spec.account_name
    ]
    if len(matches) != 1:
        raise PostdeadlineContractError(
            "target Scheduler account status is absent or ambiguous"
        )
    account = matches[0]
    running = int(account.get("running") or 0)
    pending = int(account.get("pending") or 0)
    max_running = int(account.get("max_running") or 0)
    max_pending = int(account.get("max_pending") or 0)
    max_total = int(account.get("max_total") or 0)
    if (
        max_running < 1
        or max_pending < 1
        or max_total < 1
        or running >= max_running
        or pending >= max_pending
        or running + pending >= max_total
    ):
        raise PostdeadlineContractError(
            "target Scheduler account has no submission headroom"
        )
    capabilities = reader("/api/capabilities", None)
    capability = "conda:pyaedt2026v1"
    if spec.account_name not in _capability_accounts(
        capabilities, capability
    ):
        raise PostdeadlineContractError(
            "target Scheduler account lacks pyaedt2026v1"
        )

    result = copy.deepcopy(base)
    active_count = result.pop("active_n114_fea_count", None)
    if (
        result.get("queue_state") != "opening"
        or result.get("queue_reason") != OPENING_QUEUE_REASON
        or result.get("preferred_node_relaxed") is not False
        or active_count != 0
    ):
        raise PostdeadlineContractError(
            "generic strict-opening preflight drifted"
        )
    result.update(
        {
            "schema_version": PREFLIGHT_SCHEMA,
            "account_name": spec.account_name,
            "node_name": spec.node_name,
            "node_name_policy": "strict",
            "active_target_fea_count": 0,
            "account_status": copy.deepcopy(account),
            "required_capability": capability,
            "capability_present": True,
            "excluded_account_guards": copy.deepcopy(
                EXCLUDED_ACCOUNT_GUARDS
            ),
            "scheduler_get_only": True,
            "scheduler_post_calls": 0,
        }
    )
    return result


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise PostdeadlineContractError(
            "prepared artifact escapes output root"
        ) from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


Authenticator = Callable[
    [CandidateSpec], tuple[dict[str, Any], dict[str, Any]]
]
ProfileReader = Callable[[], dict[str, Any]]
PayloadBuilder = Callable[
    [CandidateSpec, dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]
Preflight = Callable[
    [CandidateSpec, str, datetime | None], dict[str, Any]
]


def prepare(
    *,
    output: Path = OUTPUT_ROOT,
    specs: Sequence[CandidateSpec] = CANDIDATES,
    authenticator: Authenticator | None = None,
    profile_reader: ProfileReader = candidate8.reviewed_profile,
    payload_builder: PayloadBuilder = derive_scheduler_payload,
    preflight: Preflight | None = None,
    reader: JsonReader = get_json,
    observed_at: datetime | None = None,
) -> Path:
    """Prepare all lanes atomically after every authentication and GET gate."""

    target = output.resolve()
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable prepare output already exists: {target}"
        )
    if tuple(specs) != CANDIDATES:
        raise PostdeadlineContractError(
            "prepare requires the exact reviewed three-lane specification"
        )
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError("prepare time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if now <= ORIGINAL_DEADLINE_UTC:
        raise PostdeadlineContractError(
            "post-deadline preparation cannot precede original deadline"
        )

    authenticate = authenticator or (
        lambda spec: authenticate_candidate(spec)
    )
    run_preflight = preflight or (
        lambda spec, dedupe, when: live_preflight(
            spec,
            expected_dedupe_key=dedupe,
            reader=reader,
            observed_at=when,
        )
    )
    profile = profile_reader()
    prepared: list[dict[str, Any]] = []
    for spec in specs:
        selected, params = authenticate(spec)
        if (
            validate_seal(selected, CANDIDATE_SCHEMA) is not selected
            or payload_sha256(params) != spec.raw_fea_params_sha256
            or payload_sha256(_profiled_params(params, profile))
            != spec.profiled_fea_params_sha256
        ):
            raise PostdeadlineContractError(
                "authenticated candidate payload drifted"
            )
        payload, environment, retained = payload_builder(
            spec, params, profile
        )
        if (
            payload.get("name") != spec.task_name
            or payload.get("project") != PROJECT
            or payload.get("account_name") != spec.account_name
            or payload.get("node_name") != spec.node_name
            or payload.get("node_name_policy") != "strict"
            or payload.get("cpus") != CPUS
            or payload.get("memory_mb") != MEMORY_MB
            or payload.get("max_workers_per_node")
            != MAX_WORKERS_PER_NODE
            or payload.get("aedt_backend") != "standalone"
            or payload.get("required_capability")
            != "conda:pyaedt2026v1"
            or payload.get("timeout_seconds") != SCHEDULER_SECONDS
            or not isinstance(payload.get("dedupe_key"), str)
            or not payload["dedupe_key"]
            or retained.get("dedupe_key") != payload["dedupe_key"]
        ):
            raise PostdeadlineContractError(
                "derived prepare-only Scheduler payload drifted"
            )
        gate = run_preflight(
            spec,
            str(payload["dedupe_key"]),
            observed_at,
        )
        if (
            gate.get("schema_version") != PREFLIGHT_SCHEMA
            or gate.get("account_name") != spec.account_name
            or gate.get("node_name") != spec.node_name
            or gate.get("node_name_policy") != "strict"
            or gate.get("queue_state") != "opening"
            or gate.get("ready_fit_slots") != 0
            or gate.get("pending_fit_slots") != 0
            or gate.get("inflight_fit_slots") != 0
            or gate.get("active_target_fea_count") != 0
            or gate.get("preferred_node_relaxed") is not False
            or gate.get("scheduler_get_only") is not True
            or gate.get("scheduler_post_calls") != 0
        ):
            raise PostdeadlineContractError(
                "prepare-only strict-opening gate drifted"
            )
        prepared.append(
            {
                "spec": spec,
                "selected": selected,
                "params": params,
                "profile": copy.deepcopy(profile),
                "payload": payload,
                "environment": environment,
                "retained": retained,
                "preflight": gate,
            }
        )

    staging = target.with_name(
        f".{target.name}.{os.getpid()}."
        f"{next(tempfile._get_candidate_names())}.tmp"
    )
    if staging.exists():
        raise PostdeadlineContractError(
            f"staging output already exists: {staging}"
        )
    staging.mkdir(parents=True)
    lane_records: list[dict[str, Any]] = []
    try:
        for item in prepared:
            spec = item["spec"]
            lane_root = staging / spec.lane_name
            lane_root.mkdir()
            selected_path = write_immutable_json(
                lane_root / SELECTED_NAME, item["selected"]
            )
            params_path = write_immutable_json(
                lane_root / PARAMS_NAME, item["params"]
            )
            profile_path = write_immutable_json(
                lane_root / PROFILE_NAME, item["profile"]
            )
            plan = sealed(
                {
                    "schema_version": PLAN_SCHEMA,
                    **SAFETY_FLAGS,
                    "created_at_utc": now.isoformat(),
                    "campaign_id": CAMPAIGN_ID,
                    "prepare_only": True,
                    "submission_capability_present": False,
                    "standard_only": True,
                    "symmetric_model": True,
                    "full_model": False,
                    "thermal_symmetry": "eighth",
                    "candidate_physics_sha256": spec.candidate_sha256,
                    "canonical_official_row_json_sha256": (
                        spec.canonical_row_json_sha256
                    ),
                    "canonical_physical_params_sha256": (
                        spec.canonical_physical_params_sha256
                    ),
                    "source_seed": spec.source_seed,
                    "source_task_id": spec.source_task_id,
                    "official_standard_selection_order": (
                        spec.selection_order
                    ),
                    "official_standard_selection_role": (
                        spec.selection_role
                    ),
                    "selected_candidate": _relative_record(
                        lane_root, selected_path
                    ),
                    "selected_candidate_payload_sha256": item[
                        "selected"
                    ]["payload_sha256"],
                    "fea_params": _relative_record(lane_root, params_path),
                    "raw_fea_params_sha256": spec.raw_fea_params_sha256,
                    "profiled_fea_params_sha256": (
                        spec.profiled_fea_params_sha256
                    ),
                    "execution_profile": _relative_record(
                        lane_root, profile_path
                    ),
                    "execution_profile_canonical_sha256": payload_sha256(
                        item["profile"]
                    ),
                    "solver_revision": SOLVER_REVISION,
                    "library_revision": LIBRARY_REVISION,
                    "task_name": spec.task_name,
                    "workdir": spec.workdir,
                    "dedupe_key": item["payload"]["dedupe_key"],
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
                        "account_name": spec.account_name,
                        "node_name": spec.node_name,
                        "node_name_policy": "strict",
                        "same_node_as_task_id": 0,
                        "dependency_task_id": 0,
                        "allocation_state_at_prepare": "opening",
                        "allocation_id_at_prepare": None,
                        "slurm_job_id_at_prepare": None,
                        "preferred_node_relaxed_allowed": False,
                    },
                    "scheduler_url": SCHEDULER_URL,
                    "scheduler_project": PROJECT,
                    "scheduler_repository_modified": False,
                    "scheduler_project_mutation_performed": False,
                    "scheduler_submission_performed": False,
                    "scheduler_post_calls": 0,
                    "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                    "fixed_physics_unchanged": True,
                    "scheduler_payload": item["payload"],
                    "scheduler_payload_sha256": payload_sha256(
                        item["payload"]
                    ),
                    "submission_environment": item["environment"],
                    "submission_environment_sha256": payload_sha256(
                        item["environment"]
                    ),
                    "retained_aedt_bundle": item["retained"],
                    "retained_aedt_bundle_sha256": payload_sha256(
                        item["retained"]
                    ),
                    "fresh_retention_identity": True,
                    "fresh_chunk_identity": True,
                    "preflight": item["preflight"],
                    "opening_demand_pool_contract": {
                        "allowed_pre_submit_queue_state": "opening",
                        "required_queue_reason": OPENING_QUEUE_REASON,
                        "ready_fit_slots": 0,
                        "pending_fit_slots": 0,
                        "inflight_fit_slots": 0,
                        "active_target_fea_count": 0,
                        "preferred_node_relaxed_allowed": False,
                        "relaxed_allocation_allowed": False,
                    },
                    "future_submission_review_required": True,
                    "future_submission_implementation_present": False,
                }
            )
            plan_path = write_immutable_json(
                lane_root / PLAN_NAME, plan
            )
            receipt = sealed(
                {
                    "schema_version": PREPARE_SCHEMA,
                    **SAFETY_FLAGS,
                    "created_at_utc": now.isoformat(),
                    "campaign_id": CAMPAIGN_ID,
                    "prepare_only": True,
                    "submission_capability_present": False,
                    "candidate_physics_sha256": spec.candidate_sha256,
                    "official_standard_selection_order": (
                        spec.selection_order
                    ),
                    "plan": _relative_record(lane_root, plan_path),
                    "plan_payload_sha256": plan["payload_sha256"],
                    "task_name": spec.task_name,
                    "dedupe_key": item["payload"]["dedupe_key"],
                    "account_name": spec.account_name,
                    "node_name": spec.node_name,
                    "node_name_policy": "strict",
                    "queue_state_at_prepare": "opening",
                    "allocation_id_at_prepare": None,
                    "slurm_job_id_at_prepare": None,
                    "scheduler_get_preflight_performed": True,
                    "scheduler_post_calls": 0,
                    "scheduler_submission_performed": False,
                    "submit_command_argv": None,
                    "ready_for_submission": False,
                    "future_submission_review_required": True,
                }
            )
            receipt_path = write_immutable_json(
                lane_root / RECEIPT_NAME, receipt
            )
            lane_records.append(
                {
                    "selection_order": spec.selection_order,
                    "candidate_physics_sha256": spec.candidate_sha256,
                    "account_name": spec.account_name,
                    "node_name": spec.node_name,
                    "task_name": spec.task_name,
                    "dedupe_key": item["payload"]["dedupe_key"],
                    "plan": _relative_record(staging, plan_path),
                    "plan_payload_sha256": plan["payload_sha256"],
                    "receipt": _relative_record(staging, receipt_path),
                    "scheduler_post_calls": 0,
                }
            )
        manifest = sealed(
            {
                "schema_version": MANIFEST_SCHEMA,
                **SAFETY_FLAGS,
                "created_at_utc": now.isoformat(),
                "campaign_id": CAMPAIGN_ID,
                "prepare_only": True,
                "submission_capability_present": False,
                "official_standard_candidate_count": len(lane_records),
                "official_standard_selection_orders": [
                    item["selection_order"] for item in lane_records
                ],
                "official_standard_candidates_sha256": (
                    STANDARD_CANDIDATES_SHA256
                ),
                "aggregate_manifest_sha256": (
                    AGGREGATE_MANIFEST_SHA256
                ),
                "aggregate_manifest_payload_sha256": (
                    AGGREGATE_PAYLOAD_SHA256
                ),
                "solver_revision": SOLVER_REVISION,
                "library_revision": LIBRARY_REVISION,
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "fixed_physics_unchanged": True,
                "excluded_account_guards": copy.deepcopy(
                    EXCLUDED_ACCOUNT_GUARDS
                ),
                "lanes": lane_records,
                "scheduler_get_preflights": len(lane_records),
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "future_submission_review_required": True,
            }
        )
        manifest_path = write_immutable_json(
            staging / MANIFEST_NAME, manifest
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / manifest_path.name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare three sealed official Standard hedge lanes; GET-only "
            "and no Scheduler submission command"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "prepare",
        help="authenticate, GET-preflight, and write immutable local plans",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command != "prepare":
        raise PostdeadlineContractError("unsupported prepare-only command")
    manifest_path = prepare()
    manifest = validate_seal(
        read_json(manifest_path), MANIFEST_SCHEMA
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_payload_sha256": manifest["payload_sha256"],
                "candidate_count": manifest[
                    "official_standard_candidate_count"
                ],
                "selection_orders": manifest[
                    "official_standard_selection_orders"
                ],
                "scheduler_get_preflights": manifest[
                    "scheduler_get_preflights"
                ],
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
