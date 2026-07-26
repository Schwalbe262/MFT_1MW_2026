"""File-only symmetric FEA selection and bounded local-acquisition watcher.

The watcher consumes only the sealed post-success ``state.json`` and the
immutable measured-observation files referenced by that state.  It has no
Scheduler client, HTTP client, submit, cancel, restart, or Full-model
continuation capability.

An authenticated symmetric result that passes every actual hard constraint is
selected immediately by actual minimum normalized margin, then actual loss,
then actual volume.  While any effective lane is pending, non-passing results
can only produce an awaiting manifest.  Once every lane is terminal,
operational failures are excluded and a small authenticated residual may
prepare at most three symmetric candidates for one of two finite correction
rounds.  The candidates are never submitted by this tool.
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
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import mft_goal_local_trust_acquisition as acquisition  # noqa: E402


STATE_SCHEMA = "mft-goal-local-symmetric-selection-watch-state-v1"
HEARTBEAT_SCHEMA = (
    "mft-goal-local-symmetric-selection-watch-heartbeat-v1"
)
PID_SCHEMA = "mft-goal-local-symmetric-selection-watch-pid-v1"
NDS_SCHEMA = "mft-goal-local-symmetric-selection-exact-measured-nds-v1"
SOURCE_STATE_SCHEMA = (
    "mft-goal-postdeadline-standard-postsuccess-state-v1"
)
CAMPAIGN_ID = "mft-goal-20260726"

DEFAULT_SOURCE_STATE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\postdeadline_standard_postsuccess_96325_96327_96328_96330_"
    r"96331_96332_96333_96337_96338_v6\state.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_symmetric_selection_watch_v1"
)
DEFAULT_AGGREGATE_MANIFEST = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\aggregate_rolling512_d4e4d60\aggregate_manifest.json"
)
EXPECTED_TASK_IDS = (96325, 96327, 96328, 96330, 96331, 96338, 96333)
EXPECTED_LIFECYCLE_TASK_IDS = (
    96325,
    96327,
    96328,
    96330,
    96331,
    96332,
    96333,
    96337,
    96338,
)
SELECTION_SUPERSEDED_TASK_IDS = (96332, 96337)
ALLOWED_LANE_STATUSES = frozenset(
    ("pending", "collection_ready", "terminal_failure")
)
MAX_LOCAL_ROUNDS = 2
MAX_CANDIDATES_PER_ROUND = 3
MAX_LOCAL_CANDIDATES_TOTAL = (
    MAX_LOCAL_ROUNDS * MAX_CANDIDATES_PER_ROUND
)
DEFAULT_INTERVAL_SECONDS = 60
MIN_INTERVAL_SECONDS = 10
MAX_INTERVAL_SECONDS = 300

SAFETY_FLAGS = {
    "diagnostic_only": True,
    "search_only": True,
    "canonical": False,
    "production_eligible": False,
    "original_deadline_missed": True,
    "symmetric_model_primary": True,
    "prepare_only": True,
    "scheduler_methods_used": [],
    "scheduler_mutation_performed": False,
    "scheduler_submission_performed": False,
    "scheduler_cancel_performed": False,
    "scheduler_restart_performed": False,
    "automatic_full_trigger": False,
    "automatic_full_continuation": False,
}

OBSERVATION_COMPARISON_FIELDS = (
    "task_id",
    "candidate_physics_sha256",
    "actual_volume_L",
    "actual_total_loss_W",
    "actual_dimensions_mm",
    "actual_resonance_Hz",
    "actual_winding_max_C",
    "actual_core_max_C",
    "hard_constraint_evidence",
    "measured_hard_constraints_passed",
)


class SymmetricSelectionWatchError(RuntimeError):
    """Raised when file evidence cannot be authenticated fail-closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SymmetricSelectionWatchError(
            "payload is not canonical finite JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    if "payload_sha256" in output:
        raise SymmetricSelectionWatchError("payload is already sealed")
    output["payload_sha256"] = _canonical_sha256(output)
    return output


def _validate_seal(
    value: Mapping[str, Any],
    *,
    expected_schema: str,
    label: str,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    observed = output.pop("payload_sha256", None)
    if (
        value.get("schema_version") != expected_schema
        or not isinstance(observed, str)
        or observed != _canonical_sha256(output)
    ):
        raise SymmetricSelectionWatchError(f"{label} seal is invalid")
    output["payload_sha256"] = observed
    return output


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return target


def _immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    payload = _canonical_bytes(value) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise SymmetricSelectionWatchError(
                f"immutable output bytes differ: {target}"
            )
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return target


def _file_record(path: Path) -> dict[str, Any]:
    target = path.resolve(strict=True)
    if not target.is_file() or target.is_symlink():
        raise SymmetricSelectionWatchError(
            f"artifact is not a regular file: {target}"
        )
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _verify_file_record(value: Any, label: str) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
    ):
        raise SymmetricSelectionWatchError(
            f"{label} file record is invalid"
        )
    try:
        target = Path(str(value["path"])).resolve(strict=True)
    except OSError as exc:
        raise SymmetricSelectionWatchError(
            f"{label} file is absent"
        ) from exc
    if (
        not target.is_file()
        or target.is_symlink()
        or _sha256_file(target) != value["sha256"]
        or (
            "size_bytes" in value
            and (
                isinstance(value["size_bytes"], bool)
                or int(value["size_bytes"]) != target.stat().st_size
            )
        )
    ):
        raise SymmetricSelectionWatchError(
            f"{label} artifact bytes drifted"
        )
    return target


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        target = path.resolve(strict=True)
        if not target.is_file() or target.is_symlink():
            raise OSError("not a regular file")
        payload = target.read_bytes()
        parsed = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SymmetricSelectionWatchError(
            "effective post-success state is not readable JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise SymmetricSelectionWatchError(
            "effective post-success state is not a JSON object"
        )
    return parsed, {
        "path": str(target),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _validate_failure_record(value: Any, task_id: int) -> dict[str, Any]:
    path = _verify_file_record(value, f"task {task_id} failure ledger")
    try:
        parsed = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SymmetricSelectionWatchError(
            f"task {task_id} failure ledger is not readable JSON"
        ) from exc
    if not isinstance(parsed, dict) or parsed.get("task_id") != task_id:
        raise SymmetricSelectionWatchError(
            f"task {task_id} failure ledger identity drifted"
        )
    return _file_record(path)


def _load_effective_state(
    path: Path,
    *,
    expected_task_ids: Sequence[int] = EXPECTED_TASK_IDS,
    expected_lifecycle_task_ids: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, source_record = _read_json_snapshot(path)
    state = _validate_seal(
        raw,
        expected_schema=SOURCE_STATE_SCHEMA,
        label="effective post-success state",
    )
    expected = tuple(int(value) for value in expected_task_ids)
    lifecycle_expected = tuple(
        int(value)
        for value in (
            expected_lifecycle_task_ids
            if expected_lifecycle_task_ids is not None
            else (
                EXPECTED_LIFECYCLE_TASK_IDS
                if expected == EXPECTED_TASK_IDS
                else expected
            )
        )
    )
    expected_set = set(expected)
    lifecycle_set = set(lifecycle_expected)
    superseded_set = lifecycle_set - expected_set
    lanes = state.get("lanes")
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not lifecycle_expected
        or len(lifecycle_set) != len(lifecycle_expected)
        or not expected_set.issubset(lifecycle_set)
        or not isinstance(lanes, list)
        or state.get("expected_lane_count") != len(expected)
        or state.get("lifecycle_lane_count") != len(lifecycle_expected)
        or len(lanes) != len(lifecycle_expected)
        or set(state.get("effective_task_ids") or []) != expected_set
        or set(state.get("lifecycle_task_ids") or []) != lifecycle_set
        or set(state.get("selection_superseded_task_ids") or [])
        != superseded_set
        or state.get("scheduler_mutation_performed") is not False
        or state.get("orchestrator_scheduler_methods_used") != []
    ):
        raise SymmetricSelectionWatchError(
            "effective post-success state authority drifted"
        )
    by_task: dict[int, dict[str, Any]] = {}
    for raw_lane in lanes:
        if not isinstance(raw_lane, Mapping):
            raise SymmetricSelectionWatchError("effective lane is invalid")
        try:
            task_id = int(raw_lane["task_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SymmetricSelectionWatchError(
                "effective lane task identity is invalid"
            ) from exc
        status = raw_lane.get("status")
        selection_effective = raw_lane.get("selection_effective")
        expected_effective = task_id in expected_set
        if (
            task_id in by_task
            or task_id not in lifecycle_set
            or status not in ALLOWED_LANE_STATUSES
            or selection_effective is not expected_effective
        ):
            raise SymmetricSelectionWatchError(
                "effective lane identity/status drifted"
            )
        lane = copy.deepcopy(dict(raw_lane))
        if status == "collection_ready":
            _verify_file_record(
                lane.get("observation"),
                f"task {task_id} measured observation",
            )
        elif status == "terminal_failure":
            lane["failure_ledger"] = _validate_failure_record(
                lane.get("failure_ledger"), task_id
            )
        by_task[task_id] = lane
    ordered = [
        by_task[task_id] for task_id in expected
    ] + [
        by_task[task_id]
        for task_id in lifecycle_expected
        if task_id not in expected_set
    ]
    effective_lanes = [
        lane for lane in ordered if lane["selection_effective"] is True
    ]
    counts = {
        status: sum(lane["status"] == status for lane in effective_lanes)
        for status in ALLOWED_LANE_STATUSES
    }
    lifecycle_counts = {
        status: sum(lane["status"] == status for lane in ordered)
        for status in ALLOWED_LANE_STATUSES
    }
    if (
        counts["collection_ready"] != state.get("collection_count")
        or counts["pending"] != state.get("pending_count")
        or counts["terminal_failure"]
        != state.get("terminal_failure_count")
        or lifecycle_counts["collection_ready"]
        != state.get("lifecycle_collection_count")
        or lifecycle_counts["pending"]
        != state.get("lifecycle_pending_count")
        or lifecycle_counts["terminal_failure"]
        != state.get("lifecycle_terminal_failure_count")
    ):
        raise SymmetricSelectionWatchError(
            "effective lane counts are inconsistent"
        )
    state["lanes"] = ordered
    return state, source_record


def _authenticate_observation(
    record: Mapping[str, Any],
    *,
    expected_task_id: int,
) -> tuple[Path, dict[str, Any]]:
    """Reauthenticate a referenced observation without any network access."""

    from tools import (  # noqa: PLC0415
        mft_goal_postdeadline_standard_postsuccess as postsuccess,
    )

    path = _verify_file_record(
        record, f"task {expected_task_id} measured observation"
    )
    try:
        observation = postsuccess._load_observation(path)  # noqa: SLF001
        collection_record = observation.get(
            "source_collection_receipt"
        )
        if not isinstance(collection_record, Mapping):
            raise SymmetricSelectionWatchError(
                "observation collection lineage is absent"
            )
        collection_path = _verify_file_record(
            collection_record,
            f"task {expected_task_id} source collection",
        )
        view = postsuccess.authenticate_collection(collection_path)
        measured = postsuccess._measured_classification(view)  # noqa: SLF001
    except SymmetricSelectionWatchError:
        raise
    except Exception as exc:
        raise SymmetricSelectionWatchError(
            f"task {expected_task_id} observation authentication failed"
        ) from exc
    if (
        observation.get("task_id") != expected_task_id
        or any(
            observation.get(name) != measured.get(name)
            for name in OBSERVATION_COMPARISON_FIELDS
        )
    ):
        raise SymmetricSelectionWatchError(
            f"task {expected_task_id} measured observation drifted"
        )
    return path, observation


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SymmetricSelectionWatchError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SymmetricSelectionWatchError(
            f"{label} must be finite"
        ) from exc
    if not math.isfinite(result):
        raise SymmetricSelectionWatchError(f"{label} must be finite")
    return result


def _actual_margin(observation: Mapping[str, Any]) -> float:
    try:
        return float(
            acquisition._minimum_normalized_actual_margin(  # noqa: SLF001
                observation
            )
        )
    except Exception as exc:
        raise SymmetricSelectionWatchError(
            "actual normalized hard-constraint margin is invalid"
        ) from exc


def _select_passing(
    observations: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    passing: list[dict[str, Any]] = []
    for _path, observation in observations:
        if observation.get("measured_hard_constraints_passed") is not True:
            continue
        passing.append(
            {
                "task_id": int(observation["task_id"]),
                "candidate_physics_sha256": str(
                    observation["candidate_physics_sha256"]
                ),
                "minimum_normalized_actual_margin": _actual_margin(
                    observation
                ),
                "actual_total_loss_W": _finite(
                    observation.get("actual_total_loss_W"),
                    "actual total loss",
                ),
                "actual_volume_L": _finite(
                    observation.get("actual_volume_L"),
                    "actual volume",
                ),
                "actual_dimensions_mm": copy.deepcopy(
                    observation.get("actual_dimensions_mm")
                ),
                "actual_resonance_Hz": _finite(
                    observation.get("actual_resonance_Hz"),
                    "actual resonance",
                ),
                "actual_winding_max_C": _finite(
                    observation.get("actual_winding_max_C"),
                    "actual winding maximum",
                ),
                "actual_core_max_C": _finite(
                    observation.get("actual_core_max_C"),
                    "actual core maximum",
                ),
            }
        )
    if not passing:
        return None
    passing.sort(
        key=lambda item: (
            -item["minimum_normalized_actual_margin"],
            item["actual_total_loss_W"],
            item["actual_volume_L"],
            item["task_id"],
        )
    )
    winner = passing[0]
    winner.update(
        {
            "status": "selected_authenticated_symmetric_hard_pass",
            "ranking": (
                "actual_minimum_normalized_constraint_margin_desc,"
                "actual_total_loss_W_asc,actual_volume_L_asc,task_id_asc"
            ),
            "new_local_neighbors_required": False,
        }
    )
    return winner


def _scientific_snapshot_id(
    state: Mapping[str, Any],
    observations: Mapping[int, Mapping[str, Any]],
    *,
    current_round: int,
) -> str:
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "current_round": current_round,
        "lanes": [
            {
                "task_id": int(lane["task_id"]),
                "status": lane["status"],
                "selection_effective": lane["selection_effective"],
                "observation_payload_sha256": (
                    observations[int(lane["task_id"])][
                        "payload_sha256"
                    ]
                    if int(lane["task_id"]) in observations
                    else None
                ),
                "failure_ledger_sha256": (
                    lane.get("failure_ledger", {}).get("sha256")
                    if lane["status"] == "terminal_failure"
                    else None
                ),
            }
            for lane in state["lanes"]
        ],
    }
    return _canonical_sha256(identity)[:20]


def _write_exact_measured_nds(
    observations: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    output_path: Path,
) -> dict[str, Any]:
    if not observations:
        raise SymmetricSelectionWatchError(
            "exact measured NDS requires an observation"
        )
    objectives = np.asarray(
        [
            [
                _finite(item["actual_volume_L"], "actual volume"),
                _finite(item["actual_total_loss_W"], "actual total loss"),
            ]
            for _path, item in observations
        ],
        dtype=float,
    )
    ranks = acquisition.exact_global_nds_ranks(objectives)
    rows = []
    for (_path, item), rank in zip(observations, ranks, strict=True):
        rows.append(
            {
                "task_id": int(item["task_id"]),
                "candidate_physics_sha256": str(
                    item["candidate_physics_sha256"]
                ),
                "actual_volume_L": float(item["actual_volume_L"]),
                "actual_total_loss_W": float(
                    item["actual_total_loss_W"]
                ),
                "measured_hard_constraints_passed": bool(
                    item["measured_hard_constraints_passed"]
                ),
                "minimum_normalized_actual_margin": _actual_margin(item),
                "exact_measured_non_dominated_rank": int(rank),
            }
        )
    result = _seal(
        {
            "schema_version": NDS_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "diagnostic_measured_truth_only": True,
            "production_pareto_claimed": False,
            "all_available_authenticated_symmetric_observations_used": True,
            "objective_columns": [
                "actual_volume_L",
                "actual_total_loss_W",
            ],
            "row_count": len(rows),
            "rank0_count": sum(
                row["exact_measured_non_dominated_rank"] == 0
                for row in rows
            ),
            "rows": rows,
            "sorting_callable": (
                "tools.mft_goal_global_pareto.nondominated_ranks_2d"
            ),
            "scheduler_methods_used": [],
            "scheduler_mutation_performed": False,
            "automatic_full_trigger": False,
        }
    )
    _immutable_json(output_path, result)
    return result


def _validate_acquisition_result(
    result: Mapping[str, Any],
    *,
    current_round: int,
) -> dict[str, Any]:
    try:
        manifest_path = Path(str(result["manifest_path"]))
        candidate_path = Path(str(result["candidate_set_path"]))
        manifest = acquisition.validate_seal(
            json.loads(manifest_path.read_text("utf-8")),
            acquisition.MANIFEST_SCHEMA,
        )
        candidate_set = acquisition.validate_seal(
            json.loads(candidate_path.read_text("utf-8")),
            acquisition.CANDIDATE_SET_SCHEMA,
        )
    except Exception as exc:
        raise SymmetricSelectionWatchError(
            "local acquisition output authentication failed"
        ) from exc
    count = int(candidate_set.get("candidate_count", -1))
    finite = manifest.get("finite_stop_criteria")
    if (
        not isinstance(finite, Mapping)
        or finite.get("current_round") != current_round
        or finite.get("max_correction_rounds") != MAX_LOCAL_ROUNDS
        or finite.get("max_parallel_fea_batch")
        != MAX_CANDIDATES_PER_ROUND
        or count < 0
        or count > MAX_CANDIDATES_PER_ROUND
        or count != manifest.get("candidate_count")
        or candidate_set.get("prepare_only") is not True
        or candidate_set.get("stage") != "symmetric_standard"
        or candidate_set.get("scheduler_post_calls") != 0
        or candidate_set.get("automatic_full_trigger") is not False
        or manifest.get("scheduler_submission_performed") is not False
        or manifest.get("submission_capability_present") is not False
        or manifest.get("scheduler_methods_used") != []
        or manifest.get("automatic_full_trigger") is not False
        or manifest.get("automatic_candidate_continuation") is not False
    ):
        raise SymmetricSelectionWatchError(
            "local acquisition safety/cap contract drifted"
        )
    return {
        "manifest": _file_record(manifest_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "candidate_set": _file_record(candidate_path),
        "candidate_set_payload_sha256": candidate_set["payload_sha256"],
        "candidate_count": count,
        "stop_reason": result.get("stop_reason"),
        "measured_selection": copy.deepcopy(
            result.get("measured_selection")
        ),
    }


def _run_acquisition(
    *,
    aggregate_manifest: Path,
    observation_paths: Sequence[Path],
    output_directory: Path,
    current_results_complete: bool,
    current_round: int,
    failed_task_ids: Sequence[int],
    pending_task_ids: Sequence[int],
) -> dict[str, Any]:
    try:
        result = acquisition.prepare_local_acquisition(
            aggregate_manifest_path=aggregate_manifest,
            observation_paths=observation_paths,
            output_directory=output_directory,
            current_results_complete=current_results_complete,
            current_round=current_round,
            max_rounds=MAX_LOCAL_ROUNDS,
            max_batch=MAX_CANDIDATES_PER_ROUND,
            excluded_invalid_task_ids=failed_task_ids,
            pending_official_task_ids=pending_task_ids,
        )
    except Exception as exc:
        raise SymmetricSelectionWatchError(
            "bounded local acquisition failed closed"
        ) from exc
    return _validate_acquisition_result(
        result, current_round=current_round
    )


def _strict_cycle(
    *,
    source_state_path: Path,
    aggregate_manifest: Path,
    output_root: Path,
    current_round: int,
    expected_task_ids: Sequence[int],
) -> dict[str, Any]:
    state, source_record = _load_effective_state(
        source_state_path,
        expected_task_ids=expected_task_ids,
    )
    authenticated: list[tuple[Path, Mapping[str, Any]]] = []
    observations_by_task: dict[int, Mapping[str, Any]] = {}
    failed: list[int] = []
    pending: list[int] = []
    lane_summary: list[dict[str, Any]] = []
    for lane in state["lanes"]:
        task_id = int(lane["task_id"])
        status = str(lane["status"])
        selection_effective = lane["selection_effective"] is True
        summary: dict[str, Any] = {
            "task_id": task_id,
            "effective_status": status,
            "selection_effective": selection_effective,
        }
        if not selection_effective:
            summary.update(
                {
                    "selection_eligibility": (
                        "superseded_lifecycle_only"
                    ),
                    "selection_exclusion_reason": (
                        "explicit_effective_lane_replacement"
                    ),
                    "authenticated_for_selection": False,
                    "included_in_exact_measured_nds": False,
                    "observation": copy.deepcopy(
                        lane.get("observation")
                    ),
                    "failure_ledger": copy.deepcopy(
                        lane.get("failure_ledger")
                    ),
                    "full_model_continuation_allowed": False,
                }
            )
        elif status == "collection_ready":
            path, observation = _authenticate_observation(
                lane["observation"],
                expected_task_id=task_id,
            )
            authenticated.append((path, observation))
            observations_by_task[task_id] = observation
            summary.update(
                {
                    "selection_eligibility": (
                        "authenticated_measured_symmetric"
                    ),
                    "observation": _file_record(path),
                    "candidate_physics_sha256": observation[
                        "candidate_physics_sha256"
                    ],
                    "measured_hard_constraints_passed": observation[
                        "measured_hard_constraints_passed"
                    ],
                }
            )
        elif status == "terminal_failure":
            failed.append(task_id)
            summary.update(
                {
                    "selection_eligibility": "excluded_operational_failure",
                    "failure_ledger": copy.deepcopy(
                        lane["failure_ledger"]
                    ),
                    "authenticated_temperature_residual_available": False,
                    "local_neighbor_generation_from_failure_allowed": False,
                }
            )
        else:
            pending.append(task_id)
            summary.update(
                {
                    "selection_eligibility": "pending_no_classification",
                    "authenticated_temperature_residual_available": False,
                }
            )
        lane_summary.append(summary)

    snapshot_id = _scientific_snapshot_id(
        state,
        observations_by_task,
        current_round=current_round,
    )
    run_root = output_root.resolve() / "runs" / snapshot_id
    nds_record = None
    if authenticated:
        nds_path = run_root / "exact_measured_nds.json"
        nds = _write_exact_measured_nds(
            authenticated, output_path=nds_path
        )
        nds_record = {
            "artifact": _file_record(nds_path),
            "payload_sha256": nds["payload_sha256"],
            "row_count": nds["row_count"],
            "rank0_count": nds["rank0_count"],
        }

    selected = _select_passing(authenticated)
    acquisition_record = None
    excluded_nonlocal: list[dict[str, Any]] = []
    candidate_count = 0
    if selected is None and authenticated:
        try:
            _aggregate, anchors, _standard, _front = (
                acquisition.load_authenticated_anchor_set(
                    aggregate_manifest
                )
            )
        except Exception as exc:
            raise SymmetricSelectionWatchError(
                "official local anchor set authentication failed"
            ) from exc
        local_hashes = {
            anchor.candidate_physics_sha256 for anchor in anchors
        }
        local_paths = []
        for path, observation in authenticated:
            candidate = str(observation["candidate_physics_sha256"])
            if candidate in local_hashes:
                local_paths.append(path)
            else:
                excluded_nonlocal.append(
                    {
                        "task_id": int(observation["task_id"]),
                        "candidate_physics_sha256": candidate,
                        "reason": "outside_bounded_local_anchor_policy",
                        "excluded_from_local_delta_fit": True,
                        "included_in_exact_measured_nds": True,
                    }
                )
        acquisition_record = _run_acquisition(
            aggregate_manifest=aggregate_manifest,
            observation_paths=local_paths,
            output_directory=run_root / "local_acquisition",
            current_results_complete=not pending,
            current_round=current_round,
            failed_task_ids=failed,
            pending_task_ids=pending,
        )
        candidate_count = int(acquisition_record["candidate_count"])

    current_results_complete = not pending
    if selected is not None:
        status = "selected_symmetric_hard_pass"
        watch_complete = True
        stop_reason = "authenticated_symmetric_result_passes_actual_hard_gates"
    elif pending:
        status = (
            "partial_measured_waiting"
            if authenticated
            else "awaiting_symmetric_results"
        )
        watch_complete = False
        stop_reason = None
    elif candidate_count:
        status = "local_prepare_only_batch_ready"
        watch_complete = True
        stop_reason = "bounded_local_symmetric_batch_prepared"
    else:
        status = "terminal_no_passing_or_small_local_correction"
        watch_complete = True
        stop_reason = (
            "all_effective_lanes_terminal_without_passing_result_or_"
            "eligible_small_residual"
        )

    return _seal(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "observed_at_utc": _now(),
            **SAFETY_FLAGS,
            "status": status,
            "watch_complete": watch_complete,
            "stop_reason": stop_reason,
            "scientific_snapshot_id": snapshot_id,
            "source_effective_state": source_record,
            "source_effective_state_payload_sha256": state[
                "payload_sha256"
            ],
            "expected_task_ids": list(expected_task_ids),
            "lifecycle_task_ids": [
                int(lane["task_id"]) for lane in state["lanes"]
            ],
            "selection_superseded_task_ids": [
                int(lane["task_id"])
                for lane in state["lanes"]
                if lane["selection_effective"] is False
            ],
            "effective_lane_count": len(expected_task_ids),
            "lifecycle_lane_count": len(state["lanes"]),
            "authenticated_observation_count": len(authenticated),
            "pending_count": len(pending),
            "terminal_failure_count": len(failed),
            "current_results_complete": current_results_complete,
            "lanes": lane_summary,
            "selected_symmetric_result": selected,
            "selection_policy": (
                "actual_minimum_normalized_constraint_margin_desc,"
                "actual_total_loss_W_asc,actual_volume_L_asc,task_id_asc"
            ),
            "exact_measured_nds": nds_record,
            "local_acquisition": acquisition_record,
            "nonlocal_measured_observations": excluded_nonlocal,
            "finite_local_budget": {
                "current_round": current_round,
                "max_rounds": MAX_LOCAL_ROUNDS,
                "max_candidates_per_round": MAX_CANDIDATES_PER_ROUND,
                "max_candidates_total": MAX_LOCAL_CANDIDATES_TOTAL,
                "candidate_count_this_round": candidate_count,
            },
            "terminal_failures_are_physics_observations": False,
            "pending_allows_local_candidate_generation": False,
            "full_model_policy": "explicit_final_one_only",
            "full_model_started_by_watcher": False,
        }
    )


def _blocked_state(
    *,
    source_state_path: Path,
    current_round: int,
    expected_task_ids: Sequence[int],
    exc: Exception,
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "observed_at_utc": _now(),
            **SAFETY_FLAGS,
            "status": "blocked_fail_closed",
            "watch_complete": False,
            "stop_reason": None,
            "source_effective_state_path": str(
                source_state_path.resolve()
            ),
            "expected_task_ids": list(expected_task_ids),
            "authenticated_observation_count": 0,
            "pending_count": len(expected_task_ids),
            "terminal_failure_count": 0,
            "current_results_complete": False,
            "selected_symmetric_result": None,
            "exact_measured_nds": None,
            "local_acquisition": None,
            "finite_local_budget": {
                "current_round": current_round,
                "max_rounds": MAX_LOCAL_ROUNDS,
                "max_candidates_per_round": MAX_CANDIDATES_PER_ROUND,
                "max_candidates_total": MAX_LOCAL_CANDIDATES_TOTAL,
                "candidate_count_this_round": 0,
            },
            "fail_closed_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "terminal_failures_are_physics_observations": False,
            "pending_allows_local_candidate_generation": False,
            "full_model_policy": "explicit_final_one_only",
            "full_model_started_by_watcher": False,
        }
    )


def process_cycle(
    *,
    source_state_path: Path = DEFAULT_SOURCE_STATE,
    aggregate_manifest: Path = DEFAULT_AGGREGATE_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    current_round: int = 0,
    expected_task_ids: Sequence[int] = EXPECTED_TASK_IDS,
) -> dict[str, Any]:
    """Run one file-only cycle and atomically publish state plus heartbeat."""

    if (
        isinstance(current_round, bool)
        or not isinstance(current_round, int)
        or current_round < 0
        or current_round >= MAX_LOCAL_ROUNDS
    ):
        raise SymmetricSelectionWatchError(
            f"current round must be 0..{MAX_LOCAL_ROUNDS - 1}"
        )
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        state = _strict_cycle(
            source_state_path=source_state_path,
            aggregate_manifest=aggregate_manifest,
            output_root=root,
            current_round=current_round,
            expected_task_ids=expected_task_ids,
        )
    except Exception as exc:
        state = _blocked_state(
            source_state_path=source_state_path,
            current_round=current_round,
            expected_task_ids=expected_task_ids,
            exc=exc,
        )
    _atomic_json(root / "state.json", state)
    heartbeat = _seal(
        {
            "schema_version": HEARTBEAT_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "updated_at_utc": _now(),
            "watcher_pid": os.getpid(),
            "state": {
                "path": str((root / "state.json").resolve()),
                "payload_sha256": state["payload_sha256"],
                "status": state["status"],
                "watch_complete": state["watch_complete"],
            },
            **SAFETY_FLAGS,
        }
    )
    _atomic_json(root / "heartbeat.json", heartbeat)
    return state


class _SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream: Any = None

    def __enter__(self) -> _SingleInstanceLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(
                    self.stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise SymmetricSelectionWatchError(
                "another symmetric selection watcher holds the lock"
            ) from exc
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.stream is None:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


def run_watch(
    *,
    source_state_path: Path = DEFAULT_SOURCE_STATE,
    aggregate_manifest: Path = DEFAULT_AGGREGATE_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    current_round: int = 0,
    expected_task_ids: Sequence[int] = EXPECTED_TASK_IDS,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    once: bool = False,
) -> dict[str, Any]:
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, int)
        or not MIN_INTERVAL_SECONDS <= interval_seconds <= MAX_INTERVAL_SECONDS
    ):
        raise SymmetricSelectionWatchError("watch interval is invalid")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _SingleInstanceLock(root / "watcher.lock"):
        pid = _seal(
            {
                "schema_version": PID_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": _now(),
                "pid": os.getpid(),
                "source_effective_state_path": str(
                    source_state_path.resolve()
                ),
                "aggregate_manifest_path": str(
                    aggregate_manifest.resolve()
                ),
                "output_root": str(root),
                "expected_task_ids": list(expected_task_ids),
                "current_round": current_round,
                "interval_seconds": interval_seconds,
                **SAFETY_FLAGS,
            }
        )
        _atomic_json(root / "watcher.pid.json", pid)
        while True:
            state = process_cycle(
                source_state_path=source_state_path,
                aggregate_manifest=aggregate_manifest,
                output_root=root,
                current_round=current_round,
                expected_task_ids=expected_task_ids,
            )
            print(
                json.dumps(
                    {
                        "status": state["status"],
                        "watch_complete": state["watch_complete"],
                        "authenticated_observation_count": state[
                            "authenticated_observation_count"
                        ],
                        "pending_count": state["pending_count"],
                        "terminal_failure_count": state[
                            "terminal_failure_count"
                        ],
                        "scheduler_methods_used": [],
                        "automatic_full_trigger": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if once or state["watch_complete"]:
                return state
            time.sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-state",
        type=Path,
        default=DEFAULT_SOURCE_STATE,
    )
    parser.add_argument(
        "--aggregate-manifest",
        type=Path,
        default=DEFAULT_AGGREGATE_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--current-round",
        type=int,
        choices=range(MAX_LOCAL_ROUNDS),
        default=0,
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    parser.add_argument("--watch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = (
        run_watch(
            source_state_path=args.source_state,
            aggregate_manifest=args.aggregate_manifest,
            output_root=args.output_root,
            current_round=args.current_round,
            interval_seconds=args.interval,
        )
        if args.watch
        else process_cycle(
            source_state_path=args.source_state,
            aggregate_manifest=args.aggregate_manifest,
            output_root=args.output_root,
            current_round=args.current_round,
        )
    )
    print(
        json.dumps(
            {
                "status": state["status"],
                "watch_complete": state["watch_complete"],
                "state": str(
                    (args.output_root.resolve() / "state.json")
                ),
                "scheduler_methods_used": [],
                "automatic_full_trigger": False,
            },
            sort_keys=True,
        )
    )
    return 0 if state["status"] != "blocked_fail_closed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SymmetricSelectionWatchError as exc:
        print(
            json.dumps(
                {
                    "event": "symmetric_selection_watch_error",
                    "error": str(exc),
                    "scheduler_methods_used": [],
                    "automatic_full_trigger": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
