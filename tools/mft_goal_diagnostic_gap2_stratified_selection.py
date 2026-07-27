"""Build a fail-closed first-terminal compact-search FEA selection.

This tool consumes only locally collected, authenticated terminal populations.
It has no Scheduler client and never submits, cancels, or modifies a job.  A
single authenticated terminal seed is enough to publish the merged population
NDS.  Retry collections are deduplicated by seed before populations are
merged.

The bounded-gap scout's corrected, turn-graded acquisition constraint is the
only capacitance quantity used for eligibility or ranking.  The raw two-net
block-capacitance metric is explicitly rejected as physical truth.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    ALL_INPUT_KEYS,
    create_input_parameter,
    validation_check,
)
from module.mft_goal_20260726_contract import (  # noqa: E402
    CORE_TEMPERATURE_TARGETS,
    FIXED_COOLING_IDENTITY,
    FIXED_OPERATING_IDENTITY,
    GOAL_SIZE_LIMITS_MM,
    PRIMARY_WINDING_TEMPERATURE_TARGETS,
    SECONDARY_WINDING_TEMPERATURE_TARGETS,
    TEMPERATURE_TARGET_LIMITS_C,
    attest_fixed_identity,
    canonical_sha256,
)
from regression_260707.optimization import geometry_metrics  # noqa: E402
from tools import mft_goal_diagnostic_compact_collect as collector  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import mft_goal_global_pareto as global_pareto  # noqa: E402
from tools import mft_goal_turn_graded_physics_reranker as physics_reranker  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


SELECTION_SCHEMA = "mft-goal-diagnostic-gap2-stratified-selection-v1"
NDS_AUTHORITY_SCHEMA = "mft-goal-diagnostic-gap2-initial-nds-authority-v1"
CAP_AUTHORITY_SCHEMA = (
    "mft-goal-diagnostic-corrected-capacitance-acquisition-authority-v1"
)
TERMINAL_POPULATION_COUNT = scout.POPULATION
HASH64 = re.compile(r"[0-9a-f]{64}")
OBJECTIVE_COLUMNS = collector.OBJECTIVE_COLUMNS
CORRECTED_CAP_CONSTRAINT = (
    scout.PROVISIONAL_TURN_GRADED_C_ACQUISITION_CONSTRAINT_NAME
)
CORRECTED_CAP_PHYSICAL_COLUMN = f"physical_G:{CORRECTED_CAP_CONSTRAINT}"
CORRECTED_CAP_NORMALIZED_COLUMN = f"normalized_G:{CORRECTED_CAP_CONSTRAINT}"
RAW_CAP_PHYSICAL_COLUMN = f"physical_G:{scout.RAW_CRX_CONSTRAINT_NAME}"
RAW_CAP_NORMALIZED_COLUMN = f"normalized_G:{scout.RAW_CRX_CONSTRAINT_NAME}"
CORRECTED_CAP_MAXIMUM_F = scout.RAW_CRX_UCB_MAXIMUM_F
CORRECTED_CAP_TRANSFER_RATIO = (
    scout.AUTHENTICATED_TURN_GRADED_TRANSFER_RATIO
)
FIXED_PRIMARY_TURNS = scout.FIXED_PRIMARY_TURNS
FIXED_SECONDARY_TURNS = scout.FIXED_SECONDARY_TURNS
FIXED_PRIMARY_CONDUCTOR_MM = scout.FIXED_PRIMARY_CONDUCTOR_THICKNESS_MM
FIXED_PRIMARY_GAP_MM = scout.FIXED_PRIMARY_INTERTURN_GAP_MM
GAP2_MIN_MM = scout.VARIABLE_SECONDARY_INTERTURN_GAP_MINIMUM_MM
GAP2_MAX_MM = scout.VARIABLE_SECONDARY_INTERTURN_GAP_MAXIMUM_MM
GAP2_STEP_MM = scout.VARIABLE_SECONDARY_INTERTURN_GAP_STEP_MM
CW2_MIN_MM = scout.SECONDARY_CONDUCTOR_THICKNESS_MINIMUM_MM
CW2_MAX_MM = scout.SECONDARY_CONDUCTOR_THICKNESS_MAXIMUM_MM
GAP_STRATA = (
    ("low", 0.35, 0.90, False),
    ("mid", 0.90, 1.45, False),
    ("high", 1.45, 2.00, True),
)
DIAGNOSTIC_TEMPERATURE_LIMITS_C = {
    **{
        target: scout.EFFECTIVE_TEMPERATURE_FAMILY_LIMITS_C[
            "primary_winding"
        ]
        for target in PRIMARY_WINDING_TEMPERATURE_TARGETS
    },
    **{
        target: scout.EFFECTIVE_TEMPERATURE_FAMILY_LIMITS_C[
            "secondary_winding"
        ]
        for target in SECONDARY_WINDING_TEMPERATURE_TARGETS
    },
    **{
        target: scout.EFFECTIVE_TEMPERATURE_FAMILY_LIMITS_C["core"]
        for target in CORE_TEMPERATURE_TARGETS
    },
}
REQUIRED_PHYSICAL_CONSTRAINTS = frozenset(
    {
        "analytical_flux_density_limit",
        "exterior_width_limit",
        "exterior_length_limit",
        "exterior_height_limit",
        "half_magnetizing_resonance_minimum",
        CORRECTED_CAP_CONSTRAINT,
        *(
            f"temperature_robust_limit:{target}"
            for target in TEMPERATURE_TARGET_LIMITS_C
        ),
    }
)
CANONICAL_TRUE_COLUMNS = (
    "decoder_valid",
    "surrogate_physical_valid",
    "surrogate_physicality_passed",
    "physical_constraint_feasible",
    "physical_feasible",
)
IDENTITY_COLUMNS = collector.IDENTITY_COLUMNS
FAIL_CLOSED_FLAGS = collector.FAIL_CLOSED_FLAGS
DIVERSITY_FEATURES = (
    "cw2",
    "gap2",
    "W",
    "L",
    "H",
    "l1",
    "l2",
    "h1",
    "w1",
    "n_core_group",
    "N2_main",
    "N2_side",
    "nwh1",
    "nwh2",
    "wcp_len_x",
)
REVIEWED_TURN_GRADED_TOOL = (
    REPOSITORY_ROOT / "tools" / "mft_goal_turn_graded_cap_batch.py"
)
PHYSICS_RERANKER_TOOL = (
    REPOSITORY_ROOT / "tools" / "mft_goal_turn_graded_physics_reranker.py"
)
REVIEWED_TURN_GRADED_PROFILE_SCHEMA = "mft-goal-turn-graded-cap-profile-v1"
REVIEWED_TURN_GRADED_PROFILE = {
    "schema_version": REVIEWED_TURN_GRADED_PROFILE_SCHEMA,
    "stage": "diagnostic-turn-graded-capacitance",
    "comment": (
        "Symmetric non-rounded matrix plus one turn-graded "
        "electrostatic active winding"
    ),
    "reviewed_solver_path": "run_simulation_260706.py --fixed --headless",
    "cli_flags": "--headless",
    "param_overrides": {
        "full_model": 0,
        "round_corner": 0,
        "matrix_on": 1,
        "cap_on": 0,
        "loss_on": 0,
        "thermal_on": 0,
        "matrix_skin_mesh": 0,
        "matrix_percent_error": 0.5,
        "matrix_max_passes": 30,
        "matrix_min_converged": 1,
        "cap_percent_error": 1.0,
        "cap_max_passes": 10,
        "keep_project": 0,
        "core_equal_three_leg_air_gap": 1,
    },
    "mem_mb": 65_536,
    "cpus": 8,
    "timeout_seconds": 7_200,
}
REVIEWED_ACTUAL_RX_VARIANT = {
    "id": "rx-main-side-mid",
    "active": "Rx",
    "order": "main,side",
    "reverse": "none",
    "terminal_reverse": 0,
    "side_polarity": 1,
    "policy": "turn_midpoint",
}
RX_ACTUAL_VARIANT_ID = REVIEWED_ACTUAL_RX_VARIANT["id"]


class StratifiedSelectionError(RuntimeError):
    """An authenticated input or output contract failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StratifiedSelectionError(f"JSON input is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise StratifiedSelectionError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    text = str(value or "").strip().lower()
    if not HASH64.fullmatch(text):
        raise StratifiedSelectionError(f"{label} must be exact SHA-256 hex")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise StratifiedSelectionError(f"{label} must be finite numeric evidence")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StratifiedSelectionError(
            f"{label} must be finite numeric evidence"
        ) from exc
    if not math.isfinite(number):
        raise StratifiedSelectionError(f"{label} must be finite numeric evidence")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    if not number.is_integer():
        raise StratifiedSelectionError(f"{label} must be an integer")
    return int(number)


def _canonical_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise StratifiedSelectionError(f"{label} must be a canonical CSV boolean")


def _json_object_cell(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise StratifiedSelectionError(f"{label} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise StratifiedSelectionError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise StratifiedSelectionError(f"{label} must contain a JSON object")
    return parsed


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise StratifiedSelectionError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_compatible(dict(value)),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise StratifiedSelectionError(f"artifact is not a regular file: {resolved}")
    display = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": display,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_bounded_cap_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the acquisition-only capacitance authority."""

    ratio = _finite(
        record.get("authenticated_turn_graded_transfer_ratio"),
        "authenticated_turn_graded_transfer_ratio",
    )
    if (
        record.get("raw_same_metric_C_rx_rx_F_UCB_gate_active") is not False
        or record.get("provisional_turn_graded_C_acquisition_gate_active")
        is not True
        or record.get("capacitance_screening_constraint_name")
        != CORRECTED_CAP_CONSTRAINT
        or ratio != CORRECTED_CAP_TRANSFER_RATIO
        or record.get("raw_two_net_C_physical_feasibility_authority")
        is not False
        or record.get("final_turn_graded_symmetric_FEA_required") is not True
        or record.get("raw_same_metric_C_rx_rx_F_front_classification")
        != "provisional_corrected_single_truth_screening_only"
    ):
        raise StratifiedSelectionError(
            "bounded collection does not have corrected acquisition-only "
            "capacitance authority"
        )
    return {
        "schema_version": CAP_AUTHORITY_SCHEMA,
        "constraint_name": CORRECTED_CAP_CONSTRAINT,
        "physical_column": CORRECTED_CAP_PHYSICAL_COLUMN,
        "normalized_column": CORRECTED_CAP_NORMALIZED_COLUMN,
        "maximum_F": CORRECTED_CAP_MAXIMUM_F,
        "authenticated_turn_graded_transfer_ratio": ratio,
        "ratio_use": "surrogate_acquisition_only",
        "corrected_acquisition_formula": (
            "corrected_UCB_F=maximum_F*(1+physical_G_corrected)"
        ),
        "raw_two_net_block_C_used_for_physical_truth": False,
        "raw_two_net_block_C_used_for_ranking": False,
        "raw_two_net_block_C_used_for_FEA_promotion": False,
        "final_turn_graded_symmetric_FEA_required": True,
    }


def _artifact_path(
    task_dir: Path,
    record: Mapping[str, Any],
    name: str,
) -> Path:
    artifacts = record.get("artifacts")
    artifact = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    local = task_dir / name
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("path") != name
        or not local.is_file()
        or local.is_symlink()
        or _require_sha(artifact.get("sha256"), f"{name}.sha256")
        != _sha256_file(local)
        or _integer(artifact.get("size_bytes"), f"{name}.size_bytes")
        != local.stat().st_size
    ):
        raise StratifiedSelectionError(
            f"collection artifact bytes changed or are incomplete: {name}"
        )
    return local


def _one_text(frame: pd.DataFrame, column: str, label: str) -> str:
    if column not in frame:
        raise StratifiedSelectionError(f"{label} column is missing")
    values = {str(value) for value in frame[column]}
    if len(values) != 1:
        raise StratifiedSelectionError(f"{label} column contains mixed values")
    return next(iter(values))


def _validate_terminal_frame(
    frame: pd.DataFrame,
    *,
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    columns = list(manifest.get("columns") or [])
    if (
        len(frame) != TERMINAL_POPULATION_COUNT
        or list(frame.columns) != columns
        or record.get("terminal_population_count") != TERMINAL_POPULATION_COUNT
        or record.get("objective_columns") != list(OBJECTIVE_COLUMNS)
    ):
        raise StratifiedSelectionError("terminal population shape/columns drifted")
    indices = pd.to_numeric(
        frame.get("terminal_population_index"), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.array_equal(
        indices, np.arange(TERMINAL_POPULATION_COUNT, dtype=float)
    ):
        raise StratifiedSelectionError(
            "terminal_population_index is not exactly 0..319"
        )
    for column in CANONICAL_TRUE_COLUMNS:
        if column not in frame:
            raise StratifiedSelectionError(f"terminal table misses {column}")
        values = frame[column]
        if not pd.api.types.is_bool_dtype(values):
            raise StratifiedSelectionError(
                f"terminal table {column} is not canonical boolean evidence"
            )
    expected_source = {
        "source_seed": str(record["seed"]),
        "source_task_id": str(record["task_id"]),
        "source_bundle_id": str(record["task_payload_sha256"]),
        "source_island_id": "diagnostic-n1-6-compact",
    }
    for column, expected in expected_source.items():
        if _one_text(frame, column, column) != expected:
            raise StratifiedSelectionError(f"terminal table {column} drifted")
    identities = record.get("identities")
    if not isinstance(identities, Mapping):
        raise StratifiedSelectionError("collection scientific identities are absent")
    for name in IDENTITY_COLUMNS:
        expected = _require_sha(identities.get(name), f"identities.{name}")
        if _one_text(frame, name, name).lower() != expected:
            raise StratifiedSelectionError(
                f"terminal table scientific identity drifted: {name}"
            )

    physical = sorted(
        column for column in frame if column.startswith("physical_G:")
    )
    normalized = sorted(
        column for column in frame if column.startswith("normalized_G:")
    )
    record_physical = list(record.get("physical_constraint_columns") or [])
    record_normalized = list(record.get("normalized_constraint_columns") or [])
    if (
        physical != record_physical
        or normalized != record_normalized
        or [name.removeprefix("physical_G:") for name in physical]
        != [name.removeprefix("normalized_G:") for name in normalized]
        or CORRECTED_CAP_PHYSICAL_COLUMN not in physical
        or CORRECTED_CAP_NORMALIZED_COLUMN not in normalized
        or RAW_CAP_PHYSICAL_COLUMN in physical
        or RAW_CAP_NORMALIZED_COLUMN in normalized
        or not REQUIRED_PHYSICAL_CONSTRAINTS.issubset(
            name.removeprefix("physical_G:") for name in physical
        )
    ):
        raise StratifiedSelectionError(
            "terminal constraint schema is not the bounded corrected-C schema"
        )
    numeric_columns = [*OBJECTIVE_COLUMNS, *physical, *normalized]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise StratifiedSelectionError("terminal table has non-finite evidence")
    frame.loc[:, numeric_columns] = numeric
    for column in (
        "physical_geometry_sha256",
        "canonical_physical_params_sha256",
        "candidate_physics_sha",
    ):
        if column not in frame:
            raise StratifiedSelectionError(f"terminal table misses {column}")
        for value in frame[column]:
            _require_sha(value, column)
    return physical, normalized


def authenticate_collection(path: Path) -> dict[str, Any]:
    """Authenticate one standalone collection without its exact100 plan."""

    record_path = path.resolve(strict=True)
    match = re.fullmatch(r"task-(\d+)", record_path.parent.name)
    if record_path.name != "collection_record.json" or match is None:
        raise StratifiedSelectionError(
            "collection record must be task-<id>/collection_record.json"
        )
    record = collector._validate_sealed(  # noqa: SLF001
        _read_json(record_path),
        schema=collector.COLLECTION_RECORD_SCHEMA,
    )
    task_id = int(match.group(1))
    if (
        record.get("campaign_id") != scout.CAMPAIGN_ID
        or record.get("task_id") != task_id
        or isinstance(record.get("seed"), bool)
        or not isinstance(record.get("seed"), int)
        or record.get("scheduler_status") != "completed"
        or record.get("exit_code") != 0
        or record.get("terminal_population_count") != TERMINAL_POPULATION_COUNT
        or any(
            record.get(name) is not expected
            for name, expected in FAIL_CLOSED_FLAGS.items()
        )
        or record.get("symmetric_FEA_validation_still_required") is not True
        or record.get("scheduler_post_count") != 0
        or record.get("scheduler_cancel_count") != 0
        or record.get("scheduler_preempt_count") != 0
    ):
        raise StratifiedSelectionError("collection terminal/fail-closed identity drifted")
    cap_authority = _validate_bounded_cap_contract(record)
    task_payload_sha = _require_sha(
        record.get("task_payload_sha256"), "task_payload_sha256"
    )
    result_path = _artifact_path(record_path.parent, record, "result.json")
    csv_path = _artifact_path(
        record_path.parent, record, "terminal_physical_candidates.csv"
    )
    manifest_path = _artifact_path(
        record_path.parent,
        record,
        "terminal_physical_candidates.manifest.json",
    )

    result = scout.goal_launch._validate_seal(  # noqa: SLF001
        _read_json(result_path),
        schema=scout.RESULT_SCHEMA,
    )
    profile_sha = _require_sha(
        result.get("manufacturing_search_profile_payload_sha256"),
        "manufacturing_search_profile_payload_sha256",
    )
    geometry_profile_sha = _require_sha(
        result.get("geometry_constraint_profile_sha256"),
        "geometry_constraint_profile_sha256",
    )
    if (
        result.get("campaign_id") != scout.CAMPAIGN_ID
        or result.get("task_payload_sha256") != task_payload_sha
        or result.get("payload_sha256")
        != record.get("result_payload_sha256")
        or result.get("seed") != record.get("seed")
        or result.get("fixed_primary_turns") != FIXED_PRIMARY_TURNS
        or result.get("population") != TERMINAL_POPULATION_COUNT
        or result.get("generations") != scout.GENERATIONS
        or result.get("terminal_population_count") != TERMINAL_POPULATION_COUNT
        or result.get("raw_same_metric_C_rx_rx_F_UCB_gate_active") is not False
        or result.get("provisional_turn_graded_C_acquisition_gate_active")
        is not True
        or _finite(
            result.get("authenticated_turn_graded_transfer_ratio"),
            "result authenticated_turn_graded_transfer_ratio",
        )
        != CORRECTED_CAP_TRANSFER_RATIO
        or result.get("raw_two_net_C_physical_feasibility_authority")
        is not False
        or result.get("final_turn_graded_symmetric_FEA_required") is not True
        or result.get("screening_only") is not True
        or result.get("production_eligible") is not False
        or result.get("final_design_claim_allowed") is not False
        or result.get("fea_submission_approved") is not False
        or result.get("fea_submission_performed") is not False
        or result.get("symmetric_FEA_validation_still_required") is not True
        or result.get("scheduler_write_performed") is not False
        or result.get("scheduler_submission_performed") is not False
    ):
        raise StratifiedSelectionError("bounded diagnostic result authority drifted")
    inventory = result.get("artifact_inventory")
    if (
        not isinstance(inventory, Mapping)
        or result.get("artifact_inventory_sha256") != canonical_sha256(inventory)
    ):
        raise StratifiedSelectionError("result artifact inventory seal drifted")
    for role, name, local in (
        (
            "terminal_physical_candidates",
            "terminal_physical_candidates.csv",
            csv_path,
        ),
        (
            "terminal_physical_candidates_manifest",
            "terminal_physical_candidates.manifest.json",
            manifest_path,
        ),
    ):
        artifact = inventory.get(role)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("path") != name
            or artifact.get("sha256") != _sha256_file(local)
            or artifact.get("size_bytes") != local.stat().st_size
        ):
            raise StratifiedSelectionError(f"result inventory drifted: {role}")

    manifest = scout.goal_launch._validate_seal(  # noqa: SLF001
        _read_json(manifest_path),
        schema=preflight.GOAL_TERMINAL_TABLE_SCHEMA,
    )
    source = manifest.get("source_identity")
    csv_record = manifest.get("csv")
    if (
        not isinstance(source, Mapping)
        or not isinstance(csv_record, Mapping)
        or manifest.get("payload_sha256")
        != record.get("terminal_manifest_payload_sha256")
        or manifest.get("row_count") != TERMINAL_POPULATION_COUNT
        or manifest.get("terminal_population_index_min") != 0
        or manifest.get("terminal_population_index_max")
        != TERMINAL_POPULATION_COUNT - 1
        or manifest.get("one_row_per_terminal_individual") is not True
        or manifest.get("physical_deduplication_key")
        != "physical_geometry_sha256"
        or manifest.get("global_pareto_provenance_ready") is not True
        or csv_record.get("path") != csv_path.name
        or csv_record.get("sha256") != _sha256_file(csv_path)
        or csv_record.get("size_bytes") != csv_path.stat().st_size
        or source.get("seed") != record.get("seed")
        or str(source.get("task_id")) != str(task_id)
        or source.get("bundle_id") != task_payload_sha
        or source.get("island_id") != "diagnostic-n1-6-compact"
    ):
        raise StratifiedSelectionError("terminal manifest identity drifted")
    identities = record.get("identities")
    if (
        source.get("dataset_sha256") != identities.get("dataset_sha256")
        or source.get("model_artifacts_sha256")
        != identities.get("evaluation_model_sha256")
        or source.get("evaluation_spec_sha256")
        != identities.get("constraint_spec_sha256")
    ):
        raise StratifiedSelectionError(
            "terminal manifest scientific identity drifted"
        )
    frame = pd.read_csv(csv_path)
    physical, normalized = _validate_terminal_frame(
        frame,
        record=record,
        manifest=manifest,
    )
    return {
        "record_path": record_path,
        "record_file_sha256": _sha256_file(record_path),
        "record": record,
        "result": result,
        "manifest": manifest,
        "frame": frame,
        "task_id": task_id,
        "seed": int(record["seed"]),
        "profile_sha256": profile_sha,
        "geometry_constraint_profile_sha256": geometry_profile_sha,
        "physical_constraint_columns": physical,
        "normalized_constraint_columns": normalized,
        "capacitance_authority": cap_authority,
    }


def _scientific_signature(collection: Mapping[str, Any]) -> str:
    record = collection["record"]
    return canonical_sha256(
        {
            "identities": record["identities"],
            "objective_columns": record["objective_columns"],
            "physical_constraint_columns": collection[
                "physical_constraint_columns"
            ],
            "normalized_constraint_columns": collection[
                "normalized_constraint_columns"
            ],
            "manufacturing_search_profile_payload_sha256": collection[
                "profile_sha256"
            ],
            "geometry_constraint_profile_sha256": collection[
                "geometry_constraint_profile_sha256"
            ],
            "capacitance_authority": collection["capacitance_authority"],
        }
    )


def discover_and_deduplicate_collections(
    roots: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_roots: list[Path] = []
    paths: list[Path] = []
    for raw in roots:
        root = raw.resolve(strict=True)
        if not root.is_dir():
            raise StratifiedSelectionError(f"collector root is not a directory: {root}")
        if root in resolved_roots:
            raise StratifiedSelectionError(f"duplicate collector root: {root}")
        resolved_roots.append(root)
        paths.extend(sorted(root.glob("task-*/collection_record.json")))
    if not paths:
        raise StratifiedSelectionError(
            "no authenticated terminal collection record is available yet"
        )

    by_task: dict[int, dict[str, Any]] = {}
    duplicate_task_paths: list[dict[str, Any]] = []
    for path in paths:
        collection = authenticate_collection(path)
        task_id = int(collection["task_id"])
        prior = by_task.get(task_id)
        if prior is None:
            by_task[task_id] = collection
            continue
        if (
            prior["record"]["payload_sha256"]
            != collection["record"]["payload_sha256"]
            or prior["record_file_sha256"] != collection["record_file_sha256"]
        ):
            raise StratifiedSelectionError(
                f"task {task_id} has conflicting collection records"
            )
        duplicate_task_paths.append(
            {
                "task_id": task_id,
                "retained_path": str(prior["record_path"]),
                "duplicate_path": str(collection["record_path"]),
                "collection_record_payload_sha256": prior["record"][
                    "payload_sha256"
                ],
            }
        )

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for collection in by_task.values():
        by_seed.setdefault(int(collection["seed"]), []).append(collection)
    selected: list[dict[str, Any]] = []
    discarded_retries: list[dict[str, Any]] = []
    for seed, group in sorted(by_seed.items()):
        signatures = {_scientific_signature(item) for item in group}
        if len(signatures) != 1:
            raise StratifiedSelectionError(
                f"retry tasks for seed {seed} mix scientific authority"
            )
        ordered = sorted(group, key=lambda item: int(item["task_id"]))
        winner = ordered[-1]
        selected.append(winner)
        for discarded in ordered[:-1]:
            discarded_retries.append(
                {
                    "seed": seed,
                    "discarded_task_id": int(discarded["task_id"]),
                    "retained_task_id": int(winner["task_id"]),
                    "reason": "higher_task_id_authenticated_retry_wins",
                    "discarded_collection_record_payload_sha256": discarded[
                        "record"
                    ]["payload_sha256"],
                    "retained_collection_record_payload_sha256": winner["record"][
                        "payload_sha256"
                    ],
                }
            )
    signatures = {_scientific_signature(item) for item in selected}
    if len(signatures) != 1:
        raise StratifiedSelectionError(
            "selected seed populations mix scientific/profile authority"
        )
    return selected, {
        "collector_roots": [str(path) for path in resolved_roots],
        "discovered_collection_record_count": len(paths),
        "unique_task_count": len(by_task),
        "selected_unique_seed_count": len(selected),
        "duplicate_task_paths": duplicate_task_paths,
        "discarded_retry_tasks": discarded_retries,
        "seed_deduplication_rule": (
            "authenticate_all; require identical scientific/profile/"
            "constraint authority; retain highest task_id per seed"
        ),
    }


def _gap_stratum(gap2_mm: float) -> str:
    value = _finite(gap2_mm, "gap2")
    for name, lower, upper, upper_inclusive in GAP_STRATA:
        if value >= lower and (
            value <= upper if upper_inclusive else value < upper
        ):
            return name
    raise StratifiedSelectionError(
        f"realized gap2={value} mm is outside [{GAP2_MIN_MM},{GAP2_MAX_MM}]"
    )


def _temperature_evidence(
    physical_g: Mapping[str, float],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for target, strict_limit in TEMPERATURE_TARGET_LIMITS_C.items():
        name = f"temperature_robust_limit:{target}"
        residual = _finite(physical_g.get(name), name)
        diagnostic_limit = DIAGNOSTIC_TEMPERATURE_LIMITS_C[target]
        robust_temperature = diagnostic_limit + residual
        if robust_temperature > strict_limit + 1e-12:
            raise StratifiedSelectionError(
                f"{target} robust temperature {robust_temperature:.9g} C "
                f"exceeds strict {strict_limit:.9g} C"
            )
        evidence[target] = {
            "diagnostic_profile_limit_C": diagnostic_limit,
            "physical_G_C": residual,
            "realized_robust_temperature_C": robust_temperature,
            "strict_limit_C": strict_limit,
            "strict_margin_C": strict_limit - robust_temperature,
        }
    return evidence


def _validate_fixed_values(decoded: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(decoded)
    observed_tim = identity.get("thermal_pad_conductivity_W_mK")
    if observed_tim is not None and _finite(
        observed_tim, "thermal_pad_conductivity_W_mK"
    ) != FIXED_COOLING_IDENTITY["thermal_pad_conductivity_W_mK"]:
        raise StratifiedSelectionError("TIM conductivity was modified")
    identity["thermal_pad_conductivity_W_mK"] = FIXED_COOLING_IDENTITY[
        "thermal_pad_conductivity_W_mK"
    ]
    try:
        fixed_attestation = attest_fixed_identity(identity)
    except RuntimeError as exc:
        raise StratifiedSelectionError("fixed operating/cooling identity drifted") from exc
    for key, expected in FIXED_OPERATING_IDENTITY.items():
        if key not in decoded or _finite(decoded[key], key) != float(expected):
            raise StratifiedSelectionError(f"fixed operating value drifted: {key}")
    for key, expected in FIXED_COOLING_IDENTITY.items():
        if key == "thermal_pad_conductivity_W_mK":
            continue
        observed = decoded.get(key)
        if isinstance(expected, str):
            if str(observed) != expected:
                raise StratifiedSelectionError(f"fixed cooling value drifted: {key}")
        elif _finite(observed, key) != float(expected):
            raise StratifiedSelectionError(f"fixed cooling value drifted: {key}")
    return fixed_attestation


def _candidate_contract(
    row: Mapping[str, Any],
    *,
    physical_constraint_columns: Sequence[str],
) -> dict[str, Any]:
    for column in CANONICAL_TRUE_COLUMNS:
        if not _canonical_bool(row.get(column), column):
            raise StratifiedSelectionError(f"{column} is false")
    geometry_sha = _require_sha(
        row.get("physical_geometry_sha256"), "physical_geometry_sha256"
    )
    if (
        _require_sha(row.get("candidate_physics_sha"), "candidate_physics_sha")
        != geometry_sha
    ):
        raise StratifiedSelectionError("candidate physics/geometry SHA drifted")
    decoded = _json_object_cell(
        row.get("decoded_physical_params_json"),
        "decoded_physical_params_json",
    )
    if canonical_sha256(decoded) != _require_sha(
        row.get("canonical_physical_params_sha256"),
        "canonical_physical_params_sha256",
    ):
        raise StratifiedSelectionError("decoded physical parameter SHA drifted")
    geometry = {
        name: decoded.get(name)
        for name in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
    }
    if any(value is None for value in geometry.values()):
        raise StratifiedSelectionError("decoded geometry identity is incomplete")
    if canonical_sha256(geometry) != geometry_sha:
        raise StratifiedSelectionError("decoded geometry SHA drifted")

    physical_g = _json_object_cell(row.get("physical_G_json"), "physical_G_json")
    flattened_names = [
        column.removeprefix("physical_G:")
        for column in physical_constraint_columns
    ]
    if set(physical_g) != set(flattened_names):
        raise StratifiedSelectionError("physical_G_json schema drifted")
    physical_values: dict[str, float] = {}
    for name in flattened_names:
        value = _finite(physical_g[name], f"physical_G_json:{name}")
        flattened = _finite(row.get(f"physical_G:{name}"), f"physical_G:{name}")
        if not math.isclose(value, flattened, rel_tol=0.0, abs_tol=1e-12):
            raise StratifiedSelectionError(
                f"flattened physical constraint drifted: {name}"
            )
        physical_values[name] = value
    violations = {
        name: value
        for name, value in physical_values.items()
        if value > 1e-12
    }
    if violations:
        raise StratifiedSelectionError(
            f"search physical constraints are positive: {violations}"
        )
    temperature_evidence = _temperature_evidence(physical_values)

    if (
        _integer(decoded.get("N1_main"), "N1_main") != FIXED_PRIMARY_TURNS
        or _integer(decoded.get("N1_side"), "N1_side") != 0
        or _integer(decoded.get("N2_main"), "N2_main") <= 0
        or _integer(decoded.get("N2_side"), "N2_side") <= 0
        or _integer(decoded.get("N2_main"), "N2_main")
        + _integer(decoded.get("N2_side"), "N2_side")
        != FIXED_SECONDARY_TURNS
    ):
        raise StratifiedSelectionError("candidate is not explicit 6/60 turns")
    if (
        _finite(decoded.get("cw1"), "cw1") != FIXED_PRIMARY_CONDUCTOR_MM
        or _finite(decoded.get("gap1"), "gap1") != FIXED_PRIMARY_GAP_MM
    ):
        raise StratifiedSelectionError("fixed primary 5T/1.6 mm contract drifted")
    gap2 = _finite(decoded.get("gap2"), "gap2")
    cw2 = _finite(decoded.get("cw2"), "cw2")
    if not GAP2_MIN_MM <= gap2 <= GAP2_MAX_MM:
        raise StratifiedSelectionError("realized gap2 is outside bounded search")
    grid_index = round((gap2 - GAP2_MIN_MM) / GAP2_STEP_MM)
    realized_grid = GAP2_MIN_MM + grid_index * GAP2_STEP_MM
    if not math.isclose(gap2, realized_grid, rel_tol=0.0, abs_tol=1e-9):
        raise StratifiedSelectionError("realized gap2 is not on the 0.001 mm grid")
    if not CW2_MIN_MM <= cw2 <= CW2_MAX_MM:
        raise StratifiedSelectionError("cw2 is outside [0.3,1.0] mm")
    if (
        _finite(decoded.get("core_plate_t"), "core_plate_t")
        != scout.FIXED_CORE_PLATE_THICKNESS_MM
        or _finite(decoded.get("wcp_t"), "wcp_t")
        != scout.FIXED_WINDING_COLD_PLATE_THICKNESS_MM
    ):
        raise StratifiedSelectionError("20 mm plate thickness contract drifted")
    fixed_attestation = _validate_fixed_values(decoded)

    volume_l, dimensions = geometry_metrics.bounding_box_lit(decoded)
    width, length, height = (
        _finite(value, "exterior dimension") for value in dimensions
    )
    for axis, value in zip(("W", "L", "H"), (width, length, height)):
        if value > GOAL_SIZE_LIMITS_MM[axis] + 1e-9:
            raise StratifiedSelectionError(f"decoded exterior {axis} exceeds goal")
    objective_volume = _finite(row.get("objective_volume_L"), "objective_volume_L")
    if not math.isclose(
        objective_volume, float(volume_l), rel_tol=0.0, abs_tol=1e-8
    ):
        raise StratifiedSelectionError("objective/recomputed volume drifted")

    corrected_g = physical_values[CORRECTED_CAP_CONSTRAINT]
    if corrected_g <= -1.0 or corrected_g > 0.0:
        raise StratifiedSelectionError(
            "corrected acquisition C residual is outside physical [-1,0]"
        )
    corrected_c = CORRECTED_CAP_MAXIMUM_F * (1.0 + corrected_g)
    if not 0.0 < corrected_c <= CORRECTED_CAP_MAXIMUM_F:
        raise StratifiedSelectionError("corrected acquisition C is nonphysical")
    missing_inputs = [key for key in ALL_INPUT_KEYS if decoded.get(key) is None]
    if missing_inputs:
        raise StratifiedSelectionError(
            "decoded FEA input schema is incomplete: " + ",".join(missing_inputs)
        )
    diversity = {
        "cw2": cw2,
        "gap2": gap2,
        "W": width,
        "L": length,
        "H": height,
        **{
            name: _finite(decoded.get(name), name)
            for name in DIVERSITY_FEATURES
            if name not in {"cw2", "gap2", "W", "L", "H"}
        },
    }
    try:
        physics_network = physics_reranker.physics_energy_network(decoded)
    except physics_reranker.RerankError as exc:
        raise StratifiedSelectionError(
            "physics-informed turn-voltage energy network failed"
        ) from exc
    if (
        physics_network.get("raw_two_net_capacitance_used") is not False
        or physics_network.get("single_truth_transfer_ratio_used") is not False
        or physics_network.get("physical_feasibility_authority") is not False
        or physics_network.get("dielectric_material_basis") != "air_only"
        or physics_network.get("dielectric_stack_sensitivity_required") is not True
        or physics_network.get("turn_voltage_schedule", {}).get("turn_count")
        != FIXED_SECONDARY_TURNS
        or physics_network.get("component_breakdown", {}).get(
            "section_physical_multiplicity"
        )
        != {"main": 1.0, "side": 2.0}
    ):
        raise StratifiedSelectionError(
            "physics-informed energy network authority/multiplicity drifted"
        )
    physics_c = _finite(
        physics_network.get("physics_Ceq_F"), "physics_network.physics_Ceq_F"
    )
    physics_air_f = 1.0 / (
        2.0 * math.pi * math.sqrt(physics_reranker.FIXED_L2_H * physics_c)
    )
    return {
        "physical_geometry_sha256": geometry_sha,
        "decoded": decoded,
        "fixed_identity_attestation": fixed_attestation,
        "physical_G": physical_values,
        "temperature_evidence": temperature_evidence,
        "realized_gap2_mm": gap2,
        "gap2_grid_index": int(grid_index),
        "gap2_stratum": _gap_stratum(gap2),
        "cw2_mm": cw2,
        "corrected_acquisition_C_rx_rx_F_UCB_F": corrected_c,
        "corrected_acquisition_physical_G": corrected_g,
        "exterior_dimensions_mm": {"W": width, "L": length, "H": height},
        "volume_L": float(volume_l),
        "total_loss_W": _finite(
            row.get("objective_total_loss_W"), "objective_total_loss_W"
        ),
        "diversity_features": diversity,
        "physics_network": physics_network,
        "physics_network_air_Ceq_F": physics_c,
        "physics_network_air_f_at_L2_0p2H_Hz": physics_air_f,
        "physics_delta_prediction": None,
    }


def _select_diverse(
    candidates: Sequence[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Select corrected-C anchor then farthest feature-space candidates."""

    if count < 1 or len(candidates) < count:
        raise StratifiedSelectionError("diversity pool is smaller than selection")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["corrected_acquisition_C_rx_rx_F_UCB_F"],
            item["physical_geometry_sha256"],
        ),
    )
    for rank, item in enumerate(ordered, start=1):
        item["corrected_acquisition_rank_in_stratum"] = rank
    pool_size = min(len(ordered), max(16, count * 8))
    pool = ordered[:pool_size]
    matrix = np.array(
        [
            [float(item["diversity_features"][name]) for name in DIVERSITY_FEATURES]
            for item in pool
        ],
        dtype=float,
    )
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    normalized = (matrix - low) / np.where(span > 0.0, span, 1.0)
    normalized[:, span <= 0.0] = 0.0
    chosen = [0]
    pool[0]["selection_role"] = "minimum_corrected_acquisition_C"
    pool[0]["minimum_diversity_distance_at_selection"] = None
    while len(chosen) < count:
        remaining = [index for index in range(pool_size) if index not in chosen]
        scored = []
        for index in remaining:
            distances = np.linalg.norm(
                normalized[index] - normalized[chosen],
                axis=1,
            )
            scored.append(
                (
                    -float(distances.min()),
                    float(pool[index]["corrected_acquisition_C_rx_rx_F_UCB_F"]),
                    str(pool[index]["physical_geometry_sha256"]),
                    index,
                    float(distances.min()),
                )
            )
        _, _, _, selected_index, distance = min(scored)
        pool[selected_index]["selection_role"] = (
            "farthest_geometry_cw2_diversity"
        )
        pool[selected_index][
            "minimum_diversity_distance_at_selection"
        ] = distance
        chosen.append(selected_index)
    return [copy.deepcopy(pool[index]) for index in chosen]


def _build_stratified_proposal(
    eligible: Sequence[dict[str, Any]],
    *,
    per_stratum: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[dict[str, Any]]]:
    eligible_by_stratum = {
        name: [item for item in eligible if item["gap2_stratum"] == name]
        for name, *_ in GAP_STRATA
    }
    missing_strata = [
        name
        for name, candidates in eligible_by_stratum.items()
        if len(candidates) < per_stratum
    ]
    if missing_strata:
        return eligible_by_stratum, missing_strata, []
    selected: list[dict[str, Any]] = []
    for name, *_ in GAP_STRATA:
        chosen = _select_diverse(
            eligible_by_stratum[name],
            count=per_stratum,
        )
        for order, item in enumerate(chosen, start=1):
            item["selection_order_in_stratum"] = order
        selected.extend(chosen)
    return eligible_by_stratum, [], selected


def _reviewed_rx_turn_graded_params(
    decoded: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    missing = [name for name in ALL_INPUT_KEYS if decoded.get(name) is None]
    if missing:
        raise StratifiedSelectionError(
            "candidate cannot project to ALL_INPUT_KEYS: " + ",".join(missing)
        )
    projected = {name: copy.deepcopy(decoded[name]) for name in ALL_INPUT_KEYS}
    try:
        base_frame = create_input_parameter(projected)
        validation_check(base_frame, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise StratifiedSelectionError("base FEA params fail strict validation") from exc
    base = _json_compatible(base_frame.iloc[0].to_dict())

    profile = copy.deepcopy(REVIEWED_TURN_GRADED_PROFILE)
    expected_overrides = copy.deepcopy(profile["param_overrides"])
    variant = copy.deepcopy(REVIEWED_ACTUAL_RX_VARIANT)
    rx_params = copy.deepcopy(base)
    rx_params.update(expected_overrides)
    rx_params.update(
        {
            "cap_turn_graded_active_winding": variant["active"],
            "cap_turn_graded_voltage_policy": variant["policy"],
            "cap_turn_graded_section_order": variant["order"],
            "cap_turn_graded_reverse_sections": variant["reverse"],
            "cap_turn_graded_reverse_terminal_polarity": variant[
                "terminal_reverse"
            ],
            "cap_turn_graded_side_polarity": variant["side_polarity"],
            "cap_turn_graded_side2_polarity": 1,
        }
    )
    try:
        rx_frame = create_input_parameter(rx_params)
        validation_check(rx_frame, strict=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise StratifiedSelectionError(
            "Rx turn-graded FEA params fail strict validation"
        ) from exc
    rx_params = _json_compatible(rx_frame.iloc[0].to_dict())
    return base, rx_params, {
        "profile": _json_compatible(profile),
        "profile_canonical_sha256": canonical_sha256(profile),
        "reviewed_batch_tool": _file_record(REVIEWED_TURN_GRADED_TOOL),
        "actual_rx_variant": variant,
        "electrostatic_model_fraction": "1/8",
        "active_winding": "Rx",
        "explicit_turn_voltage_count": FIXED_SECONDARY_TURNS,
        "n_explicit_turns_field_not_repurposed": True,
        "matrix_on_required": True,
        "rounded_geometry_used": False,
        "full_model_used": False,
    }


def _merge_and_rank(
    collections: Sequence[Mapping[str, Any]],
    *,
    physics_delta_model: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for collection in collections:
        frame = collection["frame"].copy()
        frame["source_seed"] = int(collection["seed"])
        frame["source_task_id"] = str(collection["task_id"])
        frame["source_model_sha256"] = collection["record"]["identities"][
            "evaluation_model_sha256"
        ]
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True, copy=False)
    first = collections[0]
    physical = first["physical_constraint_columns"]
    normalized = first["normalized_constraint_columns"]
    comparison = [*OBJECTIVE_COLUMNS, *physical, *normalized]
    try:
        unique = global_pareto.deduplicate_physical_candidates(
            merged,
            comparison_columns=comparison,
        )
        unique["objective_nds_rank_all"] = global_pareto.nondominated_ranks_2d(
            unique[list(OBJECTIVE_COLUMNS)].to_numpy(dtype=float)
        )
    except global_pareto.ParetoContractError as exc:
        raise StratifiedSelectionError(
            "merged physical deduplication/NDS failed"
        ) from exc

    eligible: list[dict[str, Any]] = []
    eligibility = np.zeros(len(unique), dtype=bool)
    rejection: list[str] = []
    gap_values = np.full(len(unique), np.nan, dtype=float)
    cw2_values = np.full(len(unique), np.nan, dtype=float)
    corrected_values = np.full(len(unique), np.nan, dtype=float)
    physics_air_c_values = np.full(len(unique), np.nan, dtype=float)
    physics_air_f_values = np.full(len(unique), np.nan, dtype=float)
    physics_delta_mean_values = np.full(len(unique), np.nan, dtype=float)
    physics_delta_ucb_values = np.full(len(unique), np.nan, dtype=float)
    physics_delta_lcb_f_values = np.full(len(unique), np.nan, dtype=float)
    stratum_values = np.full(len(unique), "", dtype=object)
    for index, row in unique.iterrows():
        try:
            candidate = _candidate_contract(
                row.to_dict(),
                physical_constraint_columns=physical,
            )
        except StratifiedSelectionError as exc:
            rejection.append(str(exc))
            continue
        if physics_delta_model is not None:
            try:
                prediction = physics_reranker._candidate_prediction(  # noqa: SLF001
                    candidate["decoded"],
                    physics_delta_model,
                )
            except physics_reranker.RerankError as exc:
                rejection.append(
                    "physics log-delta prediction failed: " + str(exc)
                )
                continue
            candidate["physics_delta_prediction"] = prediction
        candidate["merged_unique_row_index"] = int(index)
        candidate["source_seed"] = int(row["source_seed"])
        candidate["source_task_id"] = str(row["source_task_id"])
        candidate["terminal_population_index"] = int(
            row["terminal_population_index"]
        )
        candidate["objective_nds_rank_all"] = int(
            row["objective_nds_rank_all"]
        )
        candidate["canonical_physical_params_sha256"] = str(
            row["canonical_physical_params_sha256"]
        )
        eligible.append(candidate)
        eligibility[index] = True
        rejection.append("")
        gap_values[index] = candidate["realized_gap2_mm"]
        cw2_values[index] = candidate["cw2_mm"]
        corrected_values[index] = candidate[
            "corrected_acquisition_C_rx_rx_F_UCB_F"
        ]
        physics_air_c_values[index] = candidate["physics_network_air_Ceq_F"]
        physics_air_f_values[index] = candidate[
            "physics_network_air_f_at_L2_0p2H_Hz"
        ]
        if candidate["physics_delta_prediction"] is not None:
            physics_delta_mean_values[index] = candidate[
                "physics_delta_prediction"
            ]["physics_delta_Crx_mean_F"]
            physics_delta_ucb_values[index] = candidate[
                "physics_delta_prediction"
            ]["physics_delta_Crx_q90_ucb_F"]
            physics_delta_lcb_f_values[index] = candidate[
                "physics_delta_prediction"
            ]["physics_delta_fRx_q90_lcb_Hz"]
        stratum_values[index] = candidate["gap2_stratum"]
    unique["strict_candidate_eligible"] = eligibility
    unique["strict_rejection_reason"] = rejection
    unique["realized_gap2_mm"] = gap_values
    unique["cw2_mm"] = cw2_values
    unique["gap2_stratum"] = stratum_values
    unique["corrected_acquisition_C_rx_rx_F_UCB_F"] = corrected_values
    unique["physics_network_air_Ceq_F"] = physics_air_c_values
    unique["physics_network_air_f_at_L2_0p2H_Hz"] = physics_air_f_values
    unique["physics_delta_Crx_mean_F"] = physics_delta_mean_values
    unique["physics_delta_Crx_q90_ucb_F"] = physics_delta_ucb_values
    unique["physics_delta_fRx_q90_lcb_Hz"] = physics_delta_lcb_f_values
    strict_ranks = np.full(len(unique), -1, dtype=np.int64)
    if eligible:
        eligible_indices = np.array(
            [item["merged_unique_row_index"] for item in eligible],
            dtype=np.int64,
        )
        strict_ranks[eligible_indices] = global_pareto.nondominated_ranks_2d(
            unique.loc[
                eligible_indices,
                list(OBJECTIVE_COLUMNS),
            ].to_numpy(dtype=float)
        )
        for item in eligible:
            item["strict_feasible_nds_rank"] = int(
                strict_ranks[item["merged_unique_row_index"]]
            )
    unique["strict_feasible_nds_rank"] = strict_ranks
    unique = unique.sort_values(
        [
            "strict_candidate_eligible",
            "strict_feasible_nds_rank",
            "objective_nds_rank_all",
            *OBJECTIVE_COLUMNS,
            "physical_geometry_sha256",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return unique, eligible, {
        "terminal_row_count": len(merged),
        "unique_physical_candidate_count": len(unique),
        "strict_candidate_eligible_count": len(eligible),
        "objective_front0_all_count": int(
            np.count_nonzero(unique["objective_nds_rank_all"].to_numpy() == 0)
        ),
        "strict_feasible_front0_count": int(
            np.count_nonzero(
                unique["strict_feasible_nds_rank"].to_numpy() == 0
            )
        ),
        "seed_local_front_union_used": False,
        "global_sort_scope": (
            "all_selected_seed_terminal_population_rows_then_seed-deduplicated_"
            "physical_geometry_then_exact_2d_non_dominated_sort"
        ),
    }


def _write_nds(
    output: Path,
    frame: pd.DataFrame,
    *,
    summary: Mapping[str, Any],
    collections: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    csv_path = output / "merged_initial_nds.csv"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    authority = _seal(
        {
            "schema_version": NDS_AUTHORITY_SCHEMA,
            "snapshot_class": "first_terminal_incremental",
            "authenticated_unique_seed_count": len(collections),
            "authenticated_task_ids": [
                int(item["task_id"]) for item in collections
            ],
            "authenticated_seeds": [int(item["seed"]) for item in collections],
            "source_collection_record_payload_sha256": [
                item["record"]["payload_sha256"] for item in collections
            ],
            "manufacturing_search_profile_payload_sha256": collections[0][
                "profile_sha256"
            ],
            "scientific_identities": copy.deepcopy(
                collections[0]["record"]["identities"]
            ),
            "objective_columns": list(OBJECTIVE_COLUMNS),
            "physical_constraint_columns": list(
                collections[0]["physical_constraint_columns"]
            ),
            "normalized_constraint_columns": list(
                collections[0]["normalized_constraint_columns"]
            ),
            "merged_initial_nds": _file_record(csv_path, relative_to=output),
            **copy.deepcopy(dict(summary)),
            "integrated_exact100_scope_complete": len(collections) == 100,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "scheduler_submission_performed": False,
            "created_at": _now(),
        }
    )
    authority_path = output / "merged_initial_nds.authority.json"
    _write_json(authority_path, authority)
    return csv_path, authority_path, authority


def prepare_selection(
    *,
    collector_roots: Sequence[Path],
    output: Path,
    per_stratum: int = 4,
    physics_delta_model_path: Path | None = None,
    approved_dielectric_stack_path: Path | None = None,
) -> dict[str, Any]:
    if isinstance(per_stratum, bool) or not 3 <= int(per_stratum) <= 4:
        raise StratifiedSelectionError(
            "per_stratum must be 3 (minimum 9) or 4 (recommended 12)"
        )
    destination = output.resolve()
    if destination.exists():
        raise StratifiedSelectionError(f"immutable output already exists: {destination}")

    physics_delta_model: dict[str, Any] | None = None
    physics_delta_model_record: dict[str, Any] | None = None
    if physics_delta_model_path is not None:
        try:
            physics_delta_model = physics_reranker._load_model(  # noqa: SLF001
                physics_delta_model_path
            )
        except physics_reranker.RerankError as exc:
            raise StratifiedSelectionError(
                "physics log-delta calibration model is invalid"
            ) from exc
        model_path = physics_delta_model_path.resolve(strict=True)
        physics_delta_model_record = {
            **_file_record(model_path),
            "payload_sha256": physics_delta_model["payload_sha256"],
            "schema_version": physics_delta_model["schema_version"],
            "calibration_row_count": physics_delta_model[
                "calibration_row_count"
            ],
        }
    approved_dielectric_stack: dict[str, Any] | None = None
    approved_dielectric_stack_record: dict[str, Any] | None = None
    if approved_dielectric_stack_path is not None:
        stack_path = approved_dielectric_stack_path.resolve(strict=True)
        raw_stack = _read_json(stack_path)
        try:
            approved_dielectric_stack = (
                physics_reranker._validated_approved_dielectric_stack(  # noqa: SLF001
                    raw_stack
                )
            )
        except physics_reranker.RerankError as exc:
            raise StratifiedSelectionError(
                "approved dielectric-stack artifact is invalid"
            ) from exc
        approved_dielectric_stack_record = {
            **_file_record(stack_path),
            "payload_sha256": approved_dielectric_stack["payload_sha256"],
            "schema_version": approved_dielectric_stack["schema_version"],
            "case_id": approved_dielectric_stack["case_id"],
            "approval": copy.deepcopy(approved_dielectric_stack["approval"]),
        }
    collections, deduplication = discover_and_deduplicate_collections(
        collector_roots
    )
    destination.mkdir(parents=True)
    try:
        ranked, eligible, nds_summary = _merge_and_rank(
            collections,
            physics_delta_model=physics_delta_model,
        )
        _, nds_authority_path, nds_authority = _write_nds(
            destination,
            ranked,
            summary=nds_summary,
            collections=collections,
        )
        (
            eligible_by_stratum,
            missing_strata,
            selected,
        ) = _build_stratified_proposal(
            eligible,
            per_stratum=per_stratum,
        )
        ready = not missing_strata

        profile_evidence: dict[str, Any] | None = None
        candidate_records: list[dict[str, Any]] = []
        if ready:
            physics_priority = sorted(
                selected,
                key=lambda item: (
                    -float(
                        item["physics_delta_prediction"][
                            "physics_delta_fRx_q90_lcb_Hz"
                        ]
                        if item["physics_delta_prediction"] is not None
                        else item["physics_network_air_f_at_L2_0p2H_Hz"]
                    ),
                    item["physical_geometry_sha256"],
                ),
            )
            priority_by_geometry = {
                item["physical_geometry_sha256"]: rank
                for rank, item in enumerate(physics_priority, start=1)
            }
            for candidate in selected:
                token = (
                    f"{candidate['gap2_stratum']}-"
                    f"{candidate['selection_order_in_stratum']:02d}-"
                    f"{candidate['physical_geometry_sha256'][:12]}"
                )
                base, rx_params, compatibility = _reviewed_rx_turn_graded_params(
                    candidate["decoded"]
                )
                if profile_evidence is None:
                    profile_evidence = compatibility
                elif profile_evidence != compatibility:
                    raise StratifiedSelectionError(
                        "turn-graded compatibility evidence changed within selection"
                    )
                base_path = destination / "params" / f"{token}-base.json"
                rx_path = (
                    destination
                    / "params"
                    / f"{token}-rx-turn-graded-eighth.json"
                )
                _write_json(base_path, base)
                _write_json(rx_path, rx_params)
                dielectric_stack_sensitivity = None
                if (
                    physics_delta_model is not None
                    and approved_dielectric_stack is not None
                ):
                    try:
                        dielectric_stack_sensitivity = (
                            physics_reranker.dielectric_stack_sensitivity(
                                candidate["decoded"],
                                physics_delta_model,
                                approved_dielectric_stack,
                            )
                        )
                    except physics_reranker.RerankError as exc:
                        raise StratifiedSelectionError(
                            "approved dielectric-stack sensitivity failed "
                            f"for {candidate['physical_geometry_sha256']}"
                        ) from exc
                candidate_records.append(
                    {
                        key: copy.deepcopy(candidate[key])
                        for key in (
                            "physical_geometry_sha256",
                            "canonical_physical_params_sha256",
                            "source_seed",
                            "source_task_id",
                            "terminal_population_index",
                            "objective_nds_rank_all",
                            "strict_feasible_nds_rank",
                            "gap2_stratum",
                            "selection_order_in_stratum",
                            "selection_role",
                            "minimum_diversity_distance_at_selection",
                            "corrected_acquisition_rank_in_stratum",
                            "corrected_acquisition_C_rx_rx_F_UCB_F",
                            "corrected_acquisition_physical_G",
                            "realized_gap2_mm",
                            "gap2_grid_index",
                            "cw2_mm",
                            "exterior_dimensions_mm",
                            "volume_L",
                            "total_loss_W",
                            "temperature_evidence",
                            "fixed_identity_attestation",
                            "physics_network",
                            "physics_network_air_Ceq_F",
                            "physics_network_air_f_at_L2_0p2H_Hz",
                            "physics_delta_prediction",
                        )
                    }
                    | {
                        "parallel_FEA_execution_priority": priority_by_geometry[
                            candidate["physical_geometry_sha256"]
                        ],
                        "parallel_FEA_execution_priority_basis": (
                            "calibrated_log_delta_q90_resonance_LCB"
                            if physics_delta_model is not None
                            else "uncalibrated_air_network_resonance_diagnostic"
                        ),
                        "approved_dielectric_stack_sensitivity": (
                            dielectric_stack_sensitivity
                        ),
                        "base_params": _file_record(
                            base_path, relative_to=destination
                        ),
                        "base_params_canonical_sha256": canonical_sha256(base),
                        "rx_turn_graded_eighth_params": _file_record(
                            rx_path, relative_to=destination
                        ),
                        "rx_turn_graded_eighth_params_canonical_sha256": (
                            canonical_sha256(rx_params)
                        ),
                        "physical_equal_three_leg_air_gap_mm_in_params": (
                            _finite(
                                rx_params["core_center_gap_mm"],
                                "core_center_gap_mm",
                            )
                        ),
                        "Lm_2mH_physical_gap_tuning_still_required": True,
                        "final_turn_graded_symmetric_FEA_required": True,
                    }
                )

        cap_authority = collections[0]["capacitance_authority"]
        selection = _seal(
            {
                "schema_version": SELECTION_SCHEMA,
                "status": (
                    "ready_for_parallel_rx_turn_graded_eighth_FEA"
                    if ready
                    else "waiting_for_authenticated_gap2_strata"
                ),
                "fea_handoff_ready": ready,
                "collector_input": {
                    **deduplication,
                    "authenticated_collection_records": [
                        {
                            "path": str(item["record_path"]),
                            "file_sha256": item["record_file_sha256"],
                            "collection_record_payload_sha256": item["record"][
                                "payload_sha256"
                            ],
                            "result_payload_sha256": item["result"][
                                "payload_sha256"
                            ],
                            "terminal_manifest_payload_sha256": item["manifest"][
                                "payload_sha256"
                            ],
                            "task_payload_sha256": item["record"][
                                "task_payload_sha256"
                            ],
                            "task_id": int(item["task_id"]),
                            "seed": int(item["seed"]),
                        }
                        for item in collections
                    ],
                    "selected_task_ids": [
                        int(item["task_id"]) for item in collections
                    ],
                    "selected_seeds": [
                        int(item["seed"]) for item in collections
                    ],
                    "manufacturing_search_profile_payload_sha256": collections[
                        0
                    ]["profile_sha256"],
                    "geometry_constraint_profile_sha256": collections[0][
                        "geometry_constraint_profile_sha256"
                    ],
                    "scientific_identities": copy.deepcopy(
                        collections[0]["record"]["identities"]
                    ),
                },
                "merged_initial_nds_authority": {
                    **_file_record(nds_authority_path, relative_to=destination),
                    "payload_sha256": nds_authority["payload_sha256"],
                },
                "nds_summary": nds_summary,
                "candidate_contract": {
                    "turns": {"N1": 6, "N2": 60},
                    "primary": {"cw1_mm": 5.0, "gap1_mm": 1.6},
                    "secondary": {
                        "cw2_min_mm": CW2_MIN_MM,
                        "cw2_max_mm": CW2_MAX_MM,
                        "gap2_min_mm": GAP2_MIN_MM,
                        "gap2_max_mm": GAP2_MAX_MM,
                        "gap2_step_mm": GAP2_STEP_MM,
                    },
                    "size_limits_mm": copy.deepcopy(GOAL_SIZE_LIMITS_MM),
                    "strict_temperature_limits_C": copy.deepcopy(
                        TEMPERATURE_TARGET_LIMITS_C
                    ),
                    "diagnostic_temperature_limits_C": copy.deepcopy(
                        DIAGNOSTIC_TEMPERATURE_LIMITS_C
                    ),
                    "all_search_physical_constraints_must_be_nonpositive": True,
                    "analytical_flux_density_constraint_preserved": True,
                    "Lm_2mH_half_magnetizing_resonance_15kHz_constraint_preserved": (
                        True
                    ),
                    "fixed_cooling_identity": copy.deepcopy(
                        FIXED_COOLING_IDENTITY
                    ),
                    "core_plate_t_mm": 20.0,
                    "wcp_t_mm": 20.0,
                },
                "capacitance_authority": cap_authority,
                "strata": [
                    {
                        "name": name,
                        "lower_gap2_mm": lower,
                        "upper_gap2_mm": upper,
                        "upper_inclusive": upper_inclusive,
                        "authenticated_strict_candidate_count": len(
                            eligible_by_stratum[name]
                        ),
                        "required_selection_count": per_stratum,
                    }
                    for name, lower, upper, upper_inclusive in GAP_STRATA
                ],
                "missing_or_underfilled_strata": missing_strata,
                "selection_method": {
                    "primary_rank": (
                        "corrected_turn_graded_acquisition_C_UCB_ascending_only"
                    ),
                    "first_per_stratum": "minimum_corrected_acquisition_C",
                    "remaining_per_stratum": (
                        "farthest_normalized_geometry_and_cw2_within_top_"
                        "corrected_acquisition_pool"
                    ),
                    "diversity_features": list(DIVERSITY_FEATURES),
                    "per_stratum": per_stratum,
                    "raw_two_net_block_C_ranked": False,
                },
                "physics_informed_reranker": {
                    "tool": _file_record(PHYSICS_RERANKER_TOOL),
                    "energy_network_schema": (
                        "mft-goal-turn-graded-voltage-energy-network-v1"
                    ),
                    "network_terms": (
                        "adjacent_turn_epsA_over_d_plus_external_core_plate_"
                        "and_ground_terms_evaluated_as_sum_Cij_deltaVij_squared"
                    ),
                    "turn_voltage_schedule": (
                        "explicit_60_turn_main_side_additive_midpoint"
                    ),
                    "full_equivalent_section_physical_multiplicity": {
                        "main": 1,
                        "side": 2,
                    },
                    "feature_mapping": list(
                        physics_reranker.DELTA_FEATURE_NAMES
                    ),
                    "log_delta_calibration_model": physics_delta_model_record,
                    "log_delta_calibration_status": (
                        "authenticated_current24_model_applied"
                        if physics_delta_model is not None
                        else "not_supplied_network_only_no_final_rerank_authority"
                    ),
                    "approved_dielectric_stack_artifact": (
                        approved_dielectric_stack_record
                    ),
                    "approved_dielectric_stack_API": (
                        "mft_goal_turn_graded_physics_reranker."
                        "dielectric_stack_sensitivity"
                    ),
                    "dielectric_stack_sensitivity_status": (
                        "authenticated_stack_and_log_delta_model_applied"
                        if (
                            approved_dielectric_stack is not None
                            and physics_delta_model is not None
                        )
                        else (
                            "fail_closed_missing_log_delta_model"
                            if approved_dielectric_stack is not None
                            else "fail_closed_missing_approved_stack_artifact"
                        )
                    ),
                    "dielectric_stack_sensitivity_ready": (
                        approved_dielectric_stack is not None
                        and physics_delta_model is not None
                    ),
                    "selection_influence": (
                        "post-selection_parallel_FEA_execution_priority_only"
                    ),
                    "corrected_0p759701_ratio_role": (
                        "surrogate_acquisition_gate_only_not_network_or_final_truth"
                    ),
                    "raw_two_net_block_C_role": "never_physical_or_final_truth",
                    "prediction_role": "next_FEA_batch_reranking_only",
                    "physical_feasibility_authority": False,
                    "final_turn_graded_symmetric_FEA_required": True,
                },
                "parallel_turn_graded_FEA_proposal": {
                    "minimum_initial_unique_geometry_count": 9,
                    "recommended_initial_unique_geometry_count": 12,
                    "requested_initial_unique_geometry_count": (
                        3 * per_stratum
                    ),
                    "adaptive_followup_Rx_geometry_count": 6,
                    "final_confirmation_geometry_count": 3,
                    "initial_stage": {
                        "model": "symmetric_eighth",
                        "rounded": False,
                        "active_winding": "Rx",
                        "turn_count": 60,
                        "section_order": ["main", "side"],
                        "voltage_policy": "turn_midpoint",
                        "parallel_lanes": len(candidate_records),
                    },
                    "air_only_FEA_boundary": {
                        "required": True,
                        "role": (
                            "lower-bound_and_log-delta_calibration_not_final_"
                            "dielectric_truth"
                        ),
                        "material_basis": "air_only_solver_baseline",
                        "cooling_or_thermal_boundary_modified": False,
                    },
                    "final_dielectric_stack_sensitivity": {
                        "required": True,
                        "geometry_selection": (
                            "top3_after_authenticated_Rx_turn_graded_FEA"
                        ),
                        "cases": [
                            "air_only_solver_baseline",
                            "explicit_approved_dielectric_stack_artifact",
                        ],
                        "approved_dielectric_stack_artifact_required": True,
                        "approved_dielectric_stack_artifact": (
                            approved_dielectric_stack_record
                        ),
                        "unapproved_example_relative_permittivity_allowed": False,
                        "implicit_bulk_relative_permittivity_allowed": False,
                        "minimum_case_count": 6,
                        "role": (
                            "bound_gap_dielectric_effect_before_final_"
                            "resonance_claim"
                        ),
                        "automatic_material_promotion_allowed": False,
                        "sensitivity_API_ready": (
                            approved_dielectric_stack is not None
                            and physics_delta_model is not None
                        ),
                    },
                    "final_top3_paired_Tx_Rx_required": True,
                    "scheduler_submission_performed": False,
                },
                "turn_graded_fea_compatibility": profile_evidence,
                "selected_candidates": candidate_records,
                "selected_candidate_count": len(candidate_records),
                "parallel_FEA_lane_count": len(candidate_records),
                "no_exact100_wait_required": True,
                "incremental_first_terminal_snapshot": True,
                "screening_only": True,
                "production_eligible": False,
                "final_design_claim_allowed": False,
                "automatic_promotion_allowed": False,
                "final_turn_graded_symmetric_FEA_required": True,
                "scheduler_access_performed": False,
                "scheduler_write_performed": False,
                "scheduler_submission_performed": False,
                "scheduler_cancel_performed": False,
                "scheduler_project_modified": False,
                "created_at": _now(),
            }
        )
        selection_path = destination / "selection.json"
        _write_json(selection_path, selection)
        return {
            "output": str(destination),
            "selection": str(selection_path),
            "selection_payload_sha256": selection["payload_sha256"],
            "status": selection["status"],
            "authenticated_unique_seed_count": len(collections),
            "selected_task_ids": [
                int(item["task_id"]) for item in collections
            ],
            "terminal_row_count": nds_summary["terminal_row_count"],
            "unique_physical_candidate_count": nds_summary[
                "unique_physical_candidate_count"
            ],
            "strict_candidate_eligible_count": nds_summary[
                "strict_candidate_eligible_count"
            ],
            "stratum_counts": {
                name: len(candidates)
                for name, candidates in eligible_by_stratum.items()
            },
            "selected_candidate_count": len(candidate_records),
            "scheduler_submission_performed": False,
        }
    except Exception:
        # Outputs are immutable snapshots.  A failed construction must not leave
        # a directory that could be mistaken for a completed authority.
        for child in sorted(
            destination.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        destination.rmdir()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate first-terminal bounded-gap populations, merge/NDS "
            "them, and prepare a corrected-C-only stratified FEA selection"
        )
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument(
        "--collector-root",
        action="append",
        type=Path,
        required=True,
        help="Repeat for the original collector and every retry overlay",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--per-stratum", type=int, default=4)
    prepare.add_argument(
        "--physics-delta-model",
        type=Path,
        help=(
            "Optional sealed current24 log-delta model; it changes only the "
            "parallel FEA execution priority"
        ),
    )
    prepare.add_argument(
        "--approved-dielectric-stack",
        type=Path,
        help=(
            "Optional sealed, project-approved secondary interturn stack; "
            "without it dielectric sensitivity remains fail-closed"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "prepare":
        raise StratifiedSelectionError(f"unsupported command: {args.command}")
    result = prepare_selection(
        collector_roots=args.collector_root,
        output=args.output,
        per_stratum=args.per_stratum,
        physics_delta_model_path=args.physics_delta_model,
        approved_dielectric_stack_path=args.approved_dielectric_stack,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
