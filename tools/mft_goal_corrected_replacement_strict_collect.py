#!/usr/bin/env python3
"""Collect corrected replacement FEA truth without legacy-row contamination.

The only Scheduler executions admitted by this adapter are the sealed
clean-library replacement tasks 97116..97139.  The adapter performs read-only
Scheduler GETs, stores the exact task/stdout/stderr responses, extracts
RESULT_JSON, and fail-closes the terminal Rx-interface scientific contract.

Each scientifically valid, unique physical geometry is emitted as an
individually sealed Standard collection.  A sealed aggregate manifest lists
only those collections and reports the 8-geometry/4-execution-task active
learning trigger.  The legacy ``collection_live_detailed_v1.json`` artifact
is neither an input nor an accepted reference.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module import thermal_truth_contract as thermal_truth  # noqa: E402
from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_TEMPERATURE_TARGETS,
)


RUNTIME_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
)
SOURCE_REPLACEMENT_AUTHORITY_ROOT = (
    RUNTIME_ROOT / "rx_interface_corrected24_cutover_v1"
)
DEFAULT_AUTHORITY_ROOT = (
    RUNTIME_ROOT / "clean_library_thermal24_replay_v1"
)
DEFAULT_RECEIPT = DEFAULT_AUTHORITY_ROOT / "submission_receipt.json"
DEFAULT_REPLAY_PLAN = DEFAULT_AUTHORITY_ROOT / "replay_plan.json"
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"

CLEAN_LIBRARY_SUBMISSION_SCHEMA = (
    "mft-goal-clean-library-thermal24-replay-submission-v1"
)
CLEAN_LIBRARY_REPLAY_PLAN_SCHEMA = (
    "mft-goal-clean-library-thermal24-replay-plan-v1"
)
SOURCE_SUBMISSION_RECEIPT_SCHEMA = (
    "mft-corrected-rx-interface-submission-receipt-v1"
)
SOURCE_REPLACEMENT_PLAN_SCHEMA = (
    "mft-corrected-rx-interface-replacement-plan-v1"
)
SOURCE_PLAN_SCHEMA = "mft-goal-targeted-symmetric-fea-batch-plan-v1"
COLLECTION_SCHEMA = (
    "mft-goal-corrected-replacement-strict-collection-v1"
)
AUTHENTICATED_COLLECTION_SCHEMA = (
    "mft-goal-corrected-replacement-strict-authenticated-collection-v1"
)
MANIFEST_SCHEMA = "mft-goal-corrected-replacement-strict-manifest-v1"
AUTHENTICATED_MANIFEST_SCHEMA = (
    "mft-goal-corrected-replacement-strict-authenticated-manifest-v1"
)
TASK_IDENTITY_SCHEMA = (
    "mft-goal-corrected-replacement-task-identity-v1"
)

EXPECTED_TASK_IDS = tuple(range(97116, 97140))
SOURCE_REPLACEMENT_TASK_IDS = tuple(range(97042, 97066))
EXPECTED_CAMPAIGN_COUNTS = {"cooler": 16, "lastmile": 8}
EXPECTED_PROJECT = "MFT_1MW_2026v1"
EXPECTED_PRIMARY_TURNS = 6
EXPECTED_SECONDARY_TURNS = 60
MINIMUM_UNIQUE_GEOMETRIES = 8
MINIMUM_EXECUTION_TASKS = 4
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
FORBIDDEN_LEGACY_BASENAME = "collection_live_detailed_v1.json"
MAX_STREAM_BYTES = 64 * 1024 * 1024
STANDARD_MODE = {
    "full_model": 0,
    "matrix_on": 1,
    "loss_on": 1,
    "thermal_on": 1,
    "loss_sym_on": 1,
    "thermal_symmetry": "eighth",
    "n_explicit_turns": 0,
    "matrix_skin_mesh": 0,
    "keep_project": 1,
    "round_corner": 0,
}


class CorrectedReplacementCollectionError(RuntimeError):
    """Fail-closed corrected replacement collection contract violation."""


@dataclass(frozen=True)
class FetchedResponse:
    """Exact response bytes plus the decoded Scheduler value."""

    raw: bytes
    value: Any


@dataclass(frozen=True)
class AuthorityLane:
    """Authenticated lineage for one corrected Scheduler execution."""

    task_id: int
    campaign: str
    rank: int
    name: str
    dedupe_key: str
    physical_geometry_sha256: str
    old_task_id: int
    source_search_task_id: int
    source_seed: int
    source_fixed_primary_turns: int
    source_secondary_turns: int
    effective_params_sha256: str
    replacement_plan_path: Path
    replacement_plan: Mapping[str, Any]
    source_plan_path: Path
    source_plan: Mapping[str, Any]
    source_params_path: Path
    source_params: Mapping[str, Any]
    submission_receipt_path: Path
    submission_receipt: Mapping[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_legacy_reference(value: Any, label: str) -> None:
    """Reject the known contaminated collection path at every public edge."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_legacy_reference(key, label)
            _reject_legacy_reference(item, label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_legacy_reference(item, label)
        return
    if (
        isinstance(value, (str, os.PathLike))
        and FORBIDDEN_LEGACY_BASENAME
        in str(value).replace("\\", "/").casefold()
    ):
        raise CorrectedReplacementCollectionError(
            f"{label} references forbidden legacy collection "
            f"{FORBIDDEN_LEGACY_BASENAME}"
        )


def _regular_file(path: Path, label: str) -> Path:
    _reject_legacy_reference(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CorrectedReplacementCollectionError(
            f"{label} is unavailable: {path}"
        ) from exc
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or path.is_symlink()
    ):
        raise CorrectedReplacementCollectionError(
            f"{label} is not a regular file: {path}"
        )
    return resolved


def _read_json(path: Path, label: str = "JSON artifact") -> dict[str, Any]:
    resolved = _regular_file(path, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectedReplacementCollectionError(
            f"{label} cannot be decoded: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise CorrectedReplacementCollectionError(
            f"{label} is not a JSON object: {resolved}"
        )
    return value


def _seal(
    value: Mapping[str, Any],
    *,
    schema: str,
    schema_field: str = "schema_version",
) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    if (
        unsigned.get(schema_field) != schema
        or "payload_sha256" in unsigned
    ):
        raise CorrectedReplacementCollectionError(
            f"cannot seal malformed {schema}"
        )
    result = dict(unsigned)
    result["payload_sha256"] = _canonical_sha256(unsigned)
    return result


def _validate_seal(
    value: Mapping[str, Any],
    *,
    schema: str,
    schema_field: str = "schema_version",
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrectedReplacementCollectionError(
            f"{schema} must be an object"
        )
    unsigned = copy.deepcopy(dict(value))
    observed = unsigned.pop("payload_sha256", None)
    if (
        unsigned.get(schema_field) != schema
        or observed != _canonical_sha256(unsigned)
    ):
        raise CorrectedReplacementCollectionError(
            f"{schema} seal mismatch"
        )
    _reject_legacy_reference(unsigned, schema)
    return copy.deepcopy(dict(value))


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CorrectedReplacementCollectionError(
            f"{label} is not a SHA-256"
        )
    return digest


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CorrectedReplacementCollectionError(f"{label} is invalid")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedReplacementCollectionError(
            f"{label} is invalid"
        ) from exc
    if number <= 0:
        raise CorrectedReplacementCollectionError(f"{label} is invalid")
    return number


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CorrectedReplacementCollectionError(f"{label} is invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedReplacementCollectionError(
            f"{label} is invalid"
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise CorrectedReplacementCollectionError(f"{label} is invalid")
    return int(number)


def _file_record(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    resolved = _regular_file(path, "artifact")
    rendered = str(resolved)
    if relative_to is not None:
        root = relative_to.resolve(strict=True)
        try:
            rendered = str(resolved.relative_to(root))
        except ValueError as exc:
            raise CorrectedReplacementCollectionError(
                "artifact is outside the requested relative root"
            ) from exc
    return {
        "path": rendered,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _resolve_file_record(
    record: Any,
    *,
    owner_path: Path,
    label: str,
    relative_root: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CorrectedReplacementCollectionError(
            f"{label} file record is malformed"
        )
    raw_path = Path(str(record["path"]))
    _reject_legacy_reference(raw_path, label)
    if raw_path.is_absolute():
        target = _regular_file(raw_path, label)
    else:
        root = (
            relative_root.resolve(strict=True)
            if relative_root is not None
            else owner_path.resolve(strict=True).parent
        )
        target = _regular_file(root / raw_path, label)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise CorrectedReplacementCollectionError(
                f"{label} relative path escapes its artifact root"
            ) from exc
    try:
        size = int(record["size_bytes"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectedReplacementCollectionError(
            f"{label} size is invalid"
        ) from exc
    if (
        size != target.stat().st_size
        or str(record["sha256"]).lower() != _sha256_file(target)
    ):
        raise CorrectedReplacementCollectionError(
            f"{label} bytes drifted"
        )
    return target


def _read_source_plan(path: Path) -> dict[str, Any]:
    value = _read_json(path, "source targeted plan")
    return _validate_seal(value, schema=SOURCE_PLAN_SCHEMA)


def _source_replacement_plan_paths(
    receipt_path: Path,
    explicit: Sequence[Path] | None,
) -> tuple[Path, ...]:
    if explicit:
        paths = tuple(
            _regular_file(path, "replacement plan") for path in explicit
        )
    else:
        paths = tuple(
            _regular_file(
                receipt_path.parent / f"{campaign}_plan.json",
                "replacement plan",
            )
            for campaign in ("cooler", "lastmile")
        )
    if len(paths) != 2 or len(set(paths)) != 2:
        raise CorrectedReplacementCollectionError(
            "exactly two unique corrected replacement plans are required"
        )
    return paths


def _source_lane(
    source_plan: Mapping[str, Any],
    *,
    rank: int,
    geometry: str,
) -> Mapping[str, Any]:
    matches = [
        lane
        for lane in source_plan.get("lanes") or []
        if isinstance(lane, Mapping) and int(lane.get("rank") or -1) == rank
    ]
    if len(matches) != 1:
        raise CorrectedReplacementCollectionError(
            f"source plan rank {rank} is not unique"
        )
    lane = matches[0]
    candidate = lane.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("physical_geometry_sha256") != geometry
    ):
        raise CorrectedReplacementCollectionError(
            f"source plan rank {rank} geometry drifted"
        )
    return lane


def _source_params_path(
    replacement_lane: Mapping[str, Any],
    *,
    source_lane: Mapping[str, Any],
    source_plan_path: Path,
) -> Path:
    replacement_record = replacement_lane.get("source_params")
    if not isinstance(replacement_record, Mapping):
        raise CorrectedReplacementCollectionError(
            "replacement source parameter record is absent"
        )
    replacement_path = _regular_file(
        Path(str(replacement_record.get("path") or "")),
        "replacement source parameters",
    )
    if (
        replacement_record.get("sha256") != _sha256_file(replacement_path)
        or int(replacement_record.get("size_bytes") or -1)
        != replacement_path.stat().st_size
    ):
        raise CorrectedReplacementCollectionError(
            "replacement source parameter bytes drifted"
        )
    source_record = source_lane.get("params")
    if not isinstance(source_record, Mapping):
        raise CorrectedReplacementCollectionError(
            "source plan parameter record is absent"
        )
    source_path = _regular_file(
        source_plan_path.parent / str(source_record.get("path") or ""),
        "source plan parameters",
    )
    if (
        source_record.get("sha256") != _sha256_file(source_path)
        or int(source_record.get("size_bytes") or -1)
        != source_path.stat().st_size
        or source_path != replacement_path
    ):
        raise CorrectedReplacementCollectionError(
            "replacement/source plan parameter authority differs"
        )
    return source_path


def _load_source_replacement_authority(
    receipt_path: Path,
    *,
    replacement_plan_paths: Sequence[Path] | None = None,
    expected_task_ids: Sequence[int] = SOURCE_REPLACEMENT_TASK_IDS,
) -> dict[int, AuthorityLane]:
    """Authenticate the superseded task set as lineage, never as truth."""

    receipt_file = _regular_file(receipt_path, "submission receipt")
    receipt = _validate_seal(
        _read_json(receipt_file, "submission receipt"),
        schema=SOURCE_SUBMISSION_RECEIPT_SCHEMA,
        schema_field="schema",
    )
    expected_ids = tuple(int(value) for value in expected_task_ids)
    if (
        expected_ids != SOURCE_REPLACEMENT_TASK_IDS
        or receipt.get("replacement_count")
        != len(SOURCE_REPLACEMENT_TASK_IDS)
        or receipt.get("scheduler_post_calls")
        != len(SOURCE_REPLACEMENT_TASK_IDS)
        or receipt.get("all_get_identities_valid") is not True
        or receipt.get("all_tasks_accepted_active") is not True
    ):
        raise CorrectedReplacementCollectionError(
            "source submission receipt is not exact 97042..97065 lineage"
        )
    submissions = receipt.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != len(
        SOURCE_REPLACEMENT_TASK_IDS
    ):
        raise CorrectedReplacementCollectionError(
            "submission receipt row count drifted"
        )
    observed_ids = tuple(
        sorted(_positive_int(row.get("task_id"), "receipt task_id")
               for row in submissions if isinstance(row, Mapping))
    )
    if observed_ids != SOURCE_REPLACEMENT_TASK_IDS:
        raise CorrectedReplacementCollectionError(
            "source receipt task set is not exactly 97042..97065"
        )

    plans: dict[str, tuple[Path, dict[str, Any]]] = {}
    for plan_path in _source_replacement_plan_paths(
        receipt_file, replacement_plan_paths
    ):
        plan = _validate_seal(
            _read_json(plan_path, "replacement plan"),
            schema=SOURCE_REPLACEMENT_PLAN_SCHEMA,
            schema_field="schema",
        )
        campaign = str(plan.get("campaign") or "")
        if (
            campaign not in EXPECTED_CAMPAIGN_COUNTS
            or campaign in plans
            or len(plan.get("lanes") or [])
            != EXPECTED_CAMPAIGN_COUNTS[campaign]
        ):
            raise CorrectedReplacementCollectionError(
                "replacement campaign plan set drifted"
            )
        plans[campaign] = (plan_path, plan)
    if set(plans) != set(EXPECTED_CAMPAIGN_COUNTS):
        raise CorrectedReplacementCollectionError(
            "cooler and lastmile replacement plans are both required"
        )

    replacement_by_key: dict[
        tuple[str, int], tuple[Path, Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    source_cache: dict[
        Path, tuple[dict[str, Any], dict[int, Mapping[str, Any]]]
    ] = {}
    for campaign, (plan_path, plan) in plans.items():
        source_reference = plan.get("source_plan")
        if not isinstance(source_reference, Mapping):
            raise CorrectedReplacementCollectionError(
                "replacement source plan record is absent"
            )
        source_plan_path = _regular_file(
            Path(str(source_reference.get("path") or "")),
            "source targeted plan",
        )
        if (
            source_reference.get("sha256")
            != _sha256_file(source_plan_path)
        ):
            raise CorrectedReplacementCollectionError(
                "source targeted plan bytes drifted"
            )
        source_plan = _read_source_plan(source_plan_path)
        if (
            source_reference.get("payload_sha256")
            != source_plan["payload_sha256"]
        ):
            raise CorrectedReplacementCollectionError(
                "source targeted plan payload drifted"
            )
        source_cache[source_plan_path] = (
            source_plan,
            {
                int(lane["rank"]): lane
                for lane in source_plan.get("lanes") or []
                if isinstance(lane, Mapping)
            },
        )
        for lane in plan["lanes"]:
            rank = _positive_int(lane.get("rank"), "replacement rank")
            key = (campaign, rank)
            if key in replacement_by_key:
                raise CorrectedReplacementCollectionError(
                    "replacement campaign/rank is duplicated"
                )
            replacement_by_key[key] = (plan_path, plan, lane)

    authority: dict[int, AuthorityLane] = {}
    geometry_seen: set[str] = set()
    for submission in submissions:
        if not isinstance(submission, Mapping):
            raise CorrectedReplacementCollectionError(
                "submission receipt row is malformed"
            )
        task_id = _positive_int(submission.get("task_id"), "task_id")
        campaign = str(submission.get("campaign") or "")
        rank = _positive_int(submission.get("rank"), "rank")
        plan_path, plan, lane = replacement_by_key.get(
            (campaign, rank), (None, None, None)
        )
        if not all(
            value is not None for value in (plan_path, plan, lane)
        ):
            raise CorrectedReplacementCollectionError(
                f"task {task_id} has no replacement plan lane"
            )
        geometry = _require_sha256(
            submission.get("physical_geometry_sha256"),
            "physical geometry SHA",
        )
        if (
            lane.get("physical_geometry_sha256") != geometry
            or submission.get("name") != lane.get("name")
            or submission.get("dedupe_key") != lane.get("dedupe_key")
            or submission.get("old_task_id") != lane.get("old_task_id")
            or geometry in geometry_seen
        ):
            raise CorrectedReplacementCollectionError(
                f"task {task_id} replacement identity drifted"
            )
        geometry_seen.add(geometry)

        source_reference = plan["source_plan"]
        source_plan_path = _regular_file(
            Path(str(source_reference["path"])),
            "source targeted plan",
        )
        source_plan = source_cache[source_plan_path][0]
        source_lane = _source_lane(
            source_plan, rank=rank, geometry=geometry
        )
        source_params_path = _source_params_path(
            lane,
            source_lane=source_lane,
            source_plan_path=source_plan_path,
        )
        source_params = _read_json(
            source_params_path, "source parameters"
        )
        source = source_lane.get("candidate", {}).get("source")
        if not isinstance(source, Mapping):
            raise CorrectedReplacementCollectionError(
                f"task {task_id} upstream source identity is absent"
            )
        source_n1 = _positive_int(
            source.get("fixed_primary_turns_stratum"),
            "source N1 stratum",
        )
        params_n1 = _exact_int(
            source_params.get("N1_main"), "N1_main"
        ) + _exact_int(source_params.get("N1_side"), "N1_side")
        params_n2 = _exact_int(
            source_params.get("N2_main"), "N2_main"
        ) + _exact_int(source_params.get("N2_side"), "N2_side")
        if (
            source_n1 != EXPECTED_PRIMARY_TURNS
            or params_n1 != EXPECTED_PRIMARY_TURNS
            or params_n2 != EXPECTED_SECONDARY_TURNS
        ):
            raise CorrectedReplacementCollectionError(
                f"task {task_id} is not the targeted 6/60 topology"
            )
        authority[task_id] = AuthorityLane(
            task_id=task_id,
            campaign=campaign,
            rank=rank,
            name=str(lane["name"]),
            dedupe_key=str(lane["dedupe_key"]),
            physical_geometry_sha256=geometry,
            old_task_id=_positive_int(lane["old_task_id"], "old task ID"),
            source_search_task_id=_positive_int(
                source.get("scheduler_task_id"),
                "source search task ID",
            ),
            source_seed=_positive_int(source.get("seed"), "source seed"),
            source_fixed_primary_turns=source_n1,
            source_secondary_turns=params_n2,
            effective_params_sha256=_require_sha256(
                lane.get("effective_params_sha256"),
                "effective parameter SHA",
            ),
            replacement_plan_path=Path(plan_path),
            replacement_plan=plan,
            source_plan_path=source_plan_path,
            source_plan=source_plan,
            source_params_path=source_params_path,
            source_params=source_params,
            submission_receipt_path=receipt_file,
            submission_receipt=receipt,
        )
    if set(authority) != set(SOURCE_REPLACEMENT_TASK_IDS):
        raise CorrectedReplacementCollectionError(
            "authenticated source-lineage task set drifted"
        )
    return authority


def _referenced_file(
    record: Any,
    *,
    label: str,
    payload_schema: str | None = None,
    payload_schema_field: str = "schema",
) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(record, Mapping) or not {
        "path",
        "sha256",
    }.issubset(record):
        raise CorrectedReplacementCollectionError(
            f"{label} reference is malformed"
        )
    path = _regular_file(Path(str(record["path"])), label)
    if record.get("sha256") != _sha256_file(path):
        raise CorrectedReplacementCollectionError(
            f"{label} bytes drifted"
        )
    if payload_schema is None:
        return path, None
    value = _validate_seal(
        _read_json(path, label),
        schema=payload_schema,
        schema_field=payload_schema_field,
    )
    if (
        "payload_sha256" in record
        and record.get("payload_sha256") != value["payload_sha256"]
    ):
        raise CorrectedReplacementCollectionError(
            f"{label} payload drifted"
        )
    return path, value


def load_authority(
    receipt_path: Path,
    *,
    replay_plan_path: Path | None = None,
    expected_task_ids: Sequence[int] = EXPECTED_TASK_IDS,
) -> dict[int, AuthorityLane]:
    """Authenticate only clean-library executions 97116..97139."""

    receipt_file = _regular_file(
        receipt_path, "clean-library submission receipt"
    )
    receipt = _validate_seal(
        _read_json(receipt_file, "clean-library submission receipt"),
        schema=CLEAN_LIBRARY_SUBMISSION_SCHEMA,
        schema_field="schema",
    )
    expected_ids = tuple(int(value) for value in expected_task_ids)
    submissions = receipt.get("submissions")
    if (
        expected_ids != EXPECTED_TASK_IDS
        or receipt.get("complete") is not True
        or receipt.get("submitted_count") != len(EXPECTED_TASK_IDS)
        or receipt.get(
            "all_generated_commands_clean_library_preflight_passed"
        )
        is not True
        or receipt.get("scheduler_repository_modified") is not False
        or receipt.get("mft_solver_repository_modified") is not False
        or receipt.get("source_tasks_cancelled_or_modified") is not False
        or not isinstance(submissions, list)
        or len(submissions) != len(EXPECTED_TASK_IDS)
    ):
        raise CorrectedReplacementCollectionError(
            "clean-library receipt is not the exact 97116..97139 authority"
        )
    observed_ids = tuple(sorted(
        _positive_int(row.get("task_id"), "clean-library task_id")
        for row in submissions
        if isinstance(row, Mapping)
    ))
    if observed_ids != EXPECTED_TASK_IDS:
        raise CorrectedReplacementCollectionError(
            "clean-library task set is not exactly 97116..97139"
        )

    plan_file = _regular_file(
        replay_plan_path or receipt_file.parent / "replay_plan.json",
        "clean-library replay plan",
    )
    plan = _validate_seal(
        _read_json(plan_file, "clean-library replay plan"),
        schema=CLEAN_LIBRARY_REPLAY_PLAN_SCHEMA,
        schema_field="schema",
    )
    plan_flags = {
        "lane_count": len(EXPECTED_TASK_IDS),
        "all_generated_commands_clean_library_preflight_passed": True,
        "fixed_cooling_unchanged": True,
        "gap0_acquisition_only": True,
        "same_params_and_geometry_as_source": True,
        "symmetric_nonrounded": True,
        "source_tasks_cancelled_or_modified": False,
        "scheduler_repository_modified": False,
        "mft_solver_repository_modified": False,
    }
    if (
        receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or any(
            plan.get(key) != expected
            for key, expected in plan_flags.items()
        )
        or len(plan.get("lanes") or []) != len(EXPECTED_TASK_IDS)
    ):
        raise CorrectedReplacementCollectionError(
            "clean-library replay plan safety boundary drifted"
        )

    source_receipt_path, source_receipt = _referenced_file(
        plan.get("source_submission"),
        label="source replacement receipt",
        payload_schema=SOURCE_SUBMISSION_RECEIPT_SCHEMA,
    )
    source_cooler_path, _ = _referenced_file(
        plan.get("source_cooler_plan"),
        label="source cooler replacement plan",
        payload_schema=SOURCE_REPLACEMENT_PLAN_SCHEMA,
    )
    source_lastmile_path, _ = _referenced_file(
        plan.get("source_lastmile_plan"),
        label="source lastmile replacement plan",
        payload_schema=SOURCE_REPLACEMENT_PLAN_SCHEMA,
    )
    if source_receipt is None:
        raise CorrectedReplacementCollectionError(
            "source replacement receipt is absent"
        )
    source_authority = _load_source_replacement_authority(
        source_receipt_path,
        replacement_plan_paths=(
            source_cooler_path,
            source_lastmile_path,
        ),
    )
    if plan.get("source_old_task_ids") != list(
        SOURCE_REPLACEMENT_TASK_IDS
    ):
        raise CorrectedReplacementCollectionError(
            "clean-library source task lineage drifted"
        )

    plan_by_lane_index = {}
    for lane in plan["lanes"]:
        if not isinstance(lane, Mapping):
            raise CorrectedReplacementCollectionError(
                "clean-library replay lane is malformed"
            )
        lane_index = _positive_int(
            lane.get("lane_index"), "clean-library lane index"
        )
        if lane_index in plan_by_lane_index:
            raise CorrectedReplacementCollectionError(
                "clean-library lane index is duplicated"
            )
        plan_by_lane_index[lane_index] = lane
    receipt_by_lane_index = {}
    for row in submissions:
        if not isinstance(row, Mapping):
            raise CorrectedReplacementCollectionError(
                "clean-library submission row is malformed"
            )
        lane_index = _positive_int(
            row.get("lane_index"), "submission lane index"
        )
        if lane_index in receipt_by_lane_index:
            raise CorrectedReplacementCollectionError(
                "submission lane index is duplicated"
            )
        receipt_by_lane_index[lane_index] = row
    if set(plan_by_lane_index) != set(range(1, 25)) or set(
        receipt_by_lane_index
    ) != set(range(1, 25)):
        raise CorrectedReplacementCollectionError(
            "clean-library lane index set drifted"
        )

    authority: dict[int, AuthorityLane] = {}
    geometries: set[str] = set()
    for lane_index in range(1, 25):
        lane = plan_by_lane_index[lane_index]
        submission = receipt_by_lane_index[lane_index]
        task_id = _positive_int(submission.get("task_id"), "task_id")
        source_old_task_id = _positive_int(
            lane.get("source_old_task_id"), "source old task ID"
        )
        source = source_authority.get(source_old_task_id)
        if source is None:
            raise CorrectedReplacementCollectionError(
                f"clean-library lane {lane_index} source is absent"
            )
        geometry = _require_sha256(
            lane.get("physical_geometry_sha256"),
            "clean-library geometry SHA",
        )
        preflight = lane.get("generated_command_preflight")
        resources = lane.get("resources")
        topology = lane.get("topology")
        source_params_record = lane.get("source_params")
        if not all(
            isinstance(value, Mapping)
            for value in (
                preflight,
                resources,
                topology,
                source_params_record,
            )
        ):
            raise CorrectedReplacementCollectionError(
                f"clean-library lane {lane_index} is incomplete"
            )
        expected_resources = {
            "aedt_backend": "standalone",
            "cpus": 8,
            "memory_mb": 65536,
            "priority": 99,
            "project": EXPECTED_PROJECT,
            "timeout_seconds": 43200,
            "max_workers_per_node": 1,
        }
        expected_topology = {
            "N1_main": 6,
            "N1_side": 0,
            "N2_main": 37,
            "N2_side": 23,
        }
        if (
            task_id not in EXPECTED_TASK_IDS
            or lane.get("observed_geometry_sha256") != geometry
            or geometry != source.physical_geometry_sha256
            or geometry in geometries
            or lane.get("effective_params_sha256")
            != source.effective_params_sha256
            or lane.get("source_params_path")
            != str(source.source_params_path)
            or lane.get("source_params_file_sha256")
            != _sha256_file(source.source_params_path)
            or source_params_record.get("path")
            != str(source.source_params_path)
            or source_params_record.get("sha256")
            != _sha256_file(source.source_params_path)
            or int(source_params_record.get("size_bytes") or -1)
            != source.source_params_path.stat().st_size
            or topology != expected_topology
            or any(
                resources.get(key) != expected
                for key, expected in expected_resources.items()
            )
            or preflight.get("passed") is not True
            or preflight.get(
                "export_precedes_clone_checkout_test_and_python"
            )
            is not True
            or preflight.get("library_revision")
            != plan.get("library_revision")
            or submission.get("name") != lane.get("name")
            or submission.get("dedupe_key") != lane.get("dedupe_key")
            or submission.get("physical_geometry_sha256") != geometry
            or submission.get("source_old_task_id") != source_old_task_id
            or submission.get("generated_command_preflight") != preflight
            or submission.get("scheduler_mutation_performed") is not True
        ):
            raise CorrectedReplacementCollectionError(
                f"clean-library lane {lane_index} identity drifted"
            )
        readback = submission.get("get_readback")
        if (
            not isinstance(readback, Mapping)
            or submission.get("get_readback_sha256")
            != _canonical_sha256(readback)
            or readback.get("id") != task_id
            or readback.get("name") != lane.get("name")
            or readback.get("dedupe_key") != lane.get("dedupe_key")
            or readback.get("project") != EXPECTED_PROJECT
        ):
            raise CorrectedReplacementCollectionError(
                f"clean-library task {task_id} submission GET drifted"
            )
        geometries.add(geometry)
        authority[task_id] = AuthorityLane(
            task_id=task_id,
            campaign=source.campaign,
            rank=source.rank,
            name=str(lane["name"]),
            dedupe_key=str(lane["dedupe_key"]),
            physical_geometry_sha256=geometry,
            old_task_id=source_old_task_id,
            source_search_task_id=source.source_search_task_id,
            source_seed=source.source_seed,
            source_fixed_primary_turns=source.source_fixed_primary_turns,
            source_secondary_turns=source.source_secondary_turns,
            effective_params_sha256=source.effective_params_sha256,
            replacement_plan_path=plan_file,
            replacement_plan=plan,
            source_plan_path=source.source_plan_path,
            source_plan=source.source_plan,
            source_params_path=source.source_params_path,
            source_params=source.source_params,
            submission_receipt_path=receipt_file,
            submission_receipt=receipt,
        )
    if set(authority) != set(EXPECTED_TASK_IDS):
        raise CorrectedReplacementCollectionError(
            "authenticated clean-library task set drifted"
        )
    return authority


def _fetch_url(url: str, *, accept: str) -> FetchedResponse:
    request = urllib.request.Request(
        url, headers={"Accept": accept}, method="GET"
    )
    raw = None
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 502, 503, 504} or attempt == 7:
                break
            retry_after = exc.headers.get("Retry-After")
            try:
                requested_delay = float(retry_after)
            except (TypeError, ValueError):
                requested_delay = 0.25 * (2**attempt)
            time.sleep(min(max(requested_delay, 0.1), 4.0))
        except OSError as exc:
            last_error = exc
            if attempt == 7:
                break
            time.sleep(min(0.25 * (2**attempt), 4.0))
    if raw is None:
        raise CorrectedReplacementCollectionError(
            f"GET {url} failed: {last_error}"
        ) from last_error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = raw.decode("utf-8", errors="replace")
    return FetchedResponse(raw=raw, value=value)


def _scheduler_json_getter(url: str) -> FetchedResponse:
    response = _fetch_url(url, accept="application/json")
    if not isinstance(response.value, Mapping):
        raise CorrectedReplacementCollectionError(
            f"Scheduler task GET returned non-object: {url}"
        )
    return response


def _scheduler_stream_getter(url: str) -> FetchedResponse:
    return _fetch_url(url, accept="application/json, text/plain")


def _stream_text(value: Any, stream: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in (stream, "stdout", "stderr", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    raise CorrectedReplacementCollectionError(
        f"Scheduler {stream} response cannot be normalized"
    )


def _extract_result(
    task: Mapping[str, Any], stdout: str
) -> dict[str, Any] | None:
    raw_result = task.get("result_json")
    if isinstance(raw_result, Mapping):
        return copy.deepcopy(dict(raw_result))
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            encoded = line.split("=", 1)[1]
        elif line.startswith("RESULT_JSON "):
            encoded = line.split(" ", 1)[1]
        else:
            continue
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _same_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return actual == expected
    if isinstance(actual, bool):
        return False
    try:
        number = float(actual)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number == float(expected)


def _observed_temperatures(
    result: Mapping[str, Any],
) -> dict[str, float]:
    observed = {}
    for target in GOAL_TEMPERATURE_TARGETS:
        try:
            value = float(result.get(target))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            observed[target] = value
    return observed


def _validate_task_identity(
    task: Mapping[str, Any], lane: AuthorityLane
) -> None:
    task_id = _positive_int(
        task.get("id", task.get("task_id")), "Scheduler task ID"
    )
    resources = lane.replacement_plan.get("lanes") or []
    replacement_lane = next(
        (
            value
            for value in resources
            if value.get("name") == lane.name
        ),
        None,
    )
    if replacement_lane is None:
        replacement_lane = next(
            (
                value
                for value in resources
                if int(value.get("rank") or -1) == lane.rank
            ),
            None,
        )
    expected_resources = (
        replacement_lane.get("resources")
        if isinstance(replacement_lane, Mapping)
        else None
    )
    if not isinstance(expected_resources, Mapping):
        raise CorrectedReplacementCollectionError(
            f"task {lane.task_id} resource authority is absent"
        )
    expected = {
        "name": lane.name,
        "dedupe_key": lane.dedupe_key,
        "project": EXPECTED_PROJECT,
        "aedt_backend": expected_resources["aedt_backend"],
        "cpus": expected_resources["cpus"],
        "memory_mb": expected_resources["memory_mb"],
        "gpus": expected_resources.get("gpus", 0),
        "priority": expected_resources["priority"],
        "timeout_seconds": expected_resources["timeout_seconds"],
        "max_workers_per_node": expected_resources[
            "max_workers_per_node"
        ],
    }
    if task_id != lane.task_id or any(
        task.get(key) != expected_value
        for key, expected_value in expected.items()
    ):
        raise CorrectedReplacementCollectionError(
            f"Scheduler task {lane.task_id} identity drifted"
        )


def _task_identity(lane: AuthorityLane) -> dict[str, Any]:
    unsigned = {
        "schema_version": TASK_IDENTITY_SCHEMA,
        "execution_task_id": lane.task_id,
        "name": lane.name,
        "dedupe_key": lane.dedupe_key,
        "project": EXPECTED_PROJECT,
        "physical_geometry_sha256": lane.physical_geometry_sha256,
        "effective_params_sha256": lane.effective_params_sha256,
        "replacement_plan_payload_sha256": lane.replacement_plan[
            "payload_sha256"
        ],
        "source_plan_payload_sha256": lane.source_plan["payload_sha256"],
        "source_search_task_id": lane.source_search_task_id,
        "source_seed": lane.source_seed,
        "source_fixed_primary_turns": lane.source_fixed_primary_turns,
    }
    result = dict(unsigned)
    result["payload_sha256"] = _canonical_sha256(unsigned)
    return result


def _classify(
    *,
    task: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    stdout: str,
    stderr: str,
    lane: AuthorityLane,
) -> dict[str, Any]:
    state = str(task.get("status") or task.get("state") or "unknown")
    reasons: list[str] = []
    if state not in TERMINAL_STATES:
        reasons.append("task_not_terminal")
    elif state != "completed":
        reasons.append(f"task_terminal_state:{state}")
    exit_code = task.get("exit_code")
    if state == "completed" and exit_code != 0:
        reasons.append(f"task_exit_code:{exit_code!r}")
    if result is None:
        reasons.append("result_json_absent")
        scientific = {
            "valid": False,
            "reasons": ["result_json_absent"],
            "contract_fields": {},
            "accepted_contract_versions": sorted(
                thermal_truth.THERMAL_RX_INTERFACE_CONTRACT_VERSIONS
            ),
            "log_markers_scanned": list(
                thermal_truth.THERMAL_INVALID_LOG_MARKERS
            ),
        }
    else:
        scientific = thermal_truth.thermal_scientific_truth_contract(
            result,
            observed_temperatures_C=_observed_temperatures(result),
            task_log_text=f"{stdout}\n{stderr}",
        )
        reasons.extend(scientific["reasons"])
        for key, expected in STANDARD_MODE.items():
            if not _same_exact(result.get(key), expected):
                reasons.append(f"standard_mode_mismatch:{key}")
        if str(result.get("git_hash") or "").strip().lower() != str(
            lane.replacement_plan.get("solver_revision") or ""
        ).strip().lower():
            reasons.append("solver_revision_mismatch")
        if str(
            result.get("pyaedt_library_git_hash") or ""
        ).strip().lower() != str(
            lane.replacement_plan.get("library_revision") or ""
        ).strip().lower():
            reasons.append("library_revision_mismatch")
        if str(result.get("physics_data_revision") or "").strip() != str(
            lane.source_params.get("physics_data_revision") or ""
        ).strip():
            reasons.append("physics_data_revision_mismatch")
        try:
            result_n1 = _exact_int(result.get("N1"), "result N1")
            result_n2 = _exact_int(result.get("N2"), "result N2")
        except CorrectedReplacementCollectionError:
            reasons.append("result_turn_identity_invalid")
        else:
            if result_n1 != EXPECTED_PRIMARY_TURNS:
                reasons.append("result_N1_not_6")
            if result_n2 != EXPECTED_SECONDARY_TURNS:
                reasons.append("result_N2_not_60")
        result_task_id = result.get("task_id")
        if result_task_id is not None:
            try:
                parsed_task_id = _exact_int(
                    result_task_id, "result task_id"
                )
            except CorrectedReplacementCollectionError:
                reasons.append("result_task_id_invalid")
            else:
                if parsed_task_id != lane.task_id:
                    reasons.append("result_task_id_mismatch")
    unique_reasons = sorted(set(reasons))
    return {
        "state": state,
        "terminal": state in TERMINAL_STATES,
        "result_available": result is not None,
        "thermal_scientific_contract": scientific,
        "scientific_valid": (
            result is not None
            and scientific.get("valid") is True
            and not unique_reasons
        ),
        "reasons": unique_reasons,
    }


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes(path, encoded)


def _trigger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    geometries = {
        str(row["physical_geometry_sha256"]) for row in rows
    }
    execution_tasks = {int(row["task_id"]) for row in rows}
    upstream_tasks = {
        int(row["source_search_task_id"]) for row in rows
    }
    reasons = []
    if len(geometries) < MINIMUM_UNIQUE_GEOMETRIES:
        reasons.append(
            f"unique_scientific_geometries<{MINIMUM_UNIQUE_GEOMETRIES}"
        )
    if len(execution_tasks) < MINIMUM_EXECUTION_TASKS:
        reasons.append(
            f"unique_corrected_execution_tasks<{MINIMUM_EXECUTION_TASKS}"
        )
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "scientific_valid_unique_rows": len(rows),
        "unique_scientific_geometries": len(geometries),
        "minimum_unique_scientific_geometries": (
            MINIMUM_UNIQUE_GEOMETRIES
        ),
        "unique_corrected_execution_tasks": len(execution_tasks),
        "minimum_corrected_execution_tasks": MINIMUM_EXECUTION_TASKS,
        "unique_source_tasks": len(execution_tasks),
        "minimum_source_tasks": MINIMUM_EXECUTION_TASKS,
        "unique_upstream_search_tasks_informational": len(
            upstream_tasks
        ),
        "trigger_task_identity": (
            "corrected FEA execution task; upstream NSGA search task is "
            "reported separately"
        ),
        "all_rows_targeted_N1_6_N2_60": all(
            int(row["source_fixed_primary_turns"])
            == EXPECTED_PRIMARY_TURNS
            and int(row["source_secondary_turns"])
            == EXPECTED_SECONDARY_TURNS
            for row in rows
        ),
    }


def _collection_value(
    *,
    lane: AuthorityLane,
    result: Mapping[str, Any],
    row_dir: Path,
    replay_plan_path: Path,
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _task_identity(lane)
    unsigned = {
        "schema_version": COLLECTION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "task_id": lane.task_id,
        "campaign": lane.campaign,
        "rank": lane.rank,
        "candidate_physics_sha256": lane.physical_geometry_sha256,
        "result": copy.deepcopy(dict(result)),
        "result_sha256": _canonical_sha256(result),
        "source_task_payload_sha256": identity["payload_sha256"],
        "source_task_identity": identity,
        "source_search_task_id": lane.source_search_task_id,
        "source_seed": lane.source_seed,
        "source_fixed_primary_turns": lane.source_fixed_primary_turns,
        "source_secondary_turns": lane.source_secondary_turns,
        "source_replacement_task_id": lane.old_task_id,
        "plan": _file_record(lane.replacement_plan_path),
        "clean_library_replay_plan": _file_record(replay_plan_path),
        "source_plan": _file_record(lane.source_plan_path),
        "source_params": _file_record(lane.source_params_path),
        "submission_receipt": _file_record(
            lane.submission_receipt_path
        ),
        "scheduler_evidence": {
            "task_get": _file_record(
                row_dir / "task_get.response.json",
                relative_to=row_dir,
            ),
            "stdout": _file_record(
                row_dir / "stdout.response",
                relative_to=row_dir,
            ),
            "stderr": _file_record(
                row_dir / "stderr.response",
                relative_to=row_dir,
            ),
            "result": _file_record(
                row_dir / "result.json",
                relative_to=row_dir,
            ),
        },
        "thermal_scientific_contract": copy.deepcopy(
            classification["thermal_scientific_contract"]
        ),
        "scheduler_status": "completed",
        "scheduler_get_only_collection": True,
        "scheduler_mutation_performed": False,
        "scientific_valid": True,
        "unique_geometry_admitted": True,
        "truth_use_only": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "goal_physical_spec_passed": False,
        "physical_air_gap_attestation_pending": True,
        "legacy_collection_reused": False,
        "legacy_collection_rows_reused": 0,
        "source_kind": (
            "clean-library corrected Scheduler "
            "GET/stdout/stderr/result only"
        ),
    }
    return _seal(unsigned, schema=COLLECTION_SCHEMA)


def collect(
    *,
    receipt_path: Path,
    output_dir: Path,
    replay_plan_path: Path | None = None,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    json_getter: Callable[[str], FetchedResponse] = (
        _scheduler_json_getter
    ),
    stream_getter: Callable[[str], FetchedResponse] = (
        _scheduler_stream_getter
    ),
) -> Path:
    """Publish one immutable corrected-replacement collection bundle."""

    authority = load_authority(
        receipt_path,
        replay_plan_path=replay_plan_path,
    )
    receipt_file = next(iter(authority.values())).submission_receipt_path
    plan_paths = tuple({
        lane.replacement_plan_path for lane in authority.values()
    })
    if len(plan_paths) != 1:
        raise CorrectedReplacementCollectionError(
            "clean-library authority must have exactly one replay plan"
        )
    plan_path = plan_paths[0]
    output = output_dir.resolve()
    _reject_legacy_reference(output, "output directory")
    if output.exists():
        raise CorrectedReplacementCollectionError(
            f"immutable output already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    task_audit = []
    prospective = []
    try:
        origin = scheduler_url.rstrip("/")
        stream_query = urllib.parse.urlencode(
            {
                "tail_lines": "1000000",
                "max_bytes": str(MAX_STREAM_BYTES),
            }
        )
        for task_id in EXPECTED_TASK_IDS:
            lane = authority[task_id]
            row_dir = staging / "rows" / f"task-{task_id}"
            row_dir.mkdir(parents=True)
            task_response = json_getter(
                f"{origin}/api/tasks/{task_id}"
            )
            stdout_response = stream_getter(
                f"{origin}/api/tasks/{task_id}/stdout?{stream_query}"
            )
            stderr_response = stream_getter(
                f"{origin}/api/tasks/{task_id}/stderr?{stream_query}"
            )
            if not isinstance(task_response.value, Mapping):
                raise CorrectedReplacementCollectionError(
                    f"task {task_id} GET returned non-object"
                )
            task = copy.deepcopy(dict(task_response.value))
            _validate_task_identity(task, lane)
            stdout = _stream_text(stdout_response.value, "stdout")
            stderr = _stream_text(stderr_response.value, "stderr")
            result = _extract_result(task, stdout)

            _write_bytes(
                row_dir / "task_get.response.json", task_response.raw
            )
            _write_bytes(
                row_dir / "stdout.response", stdout_response.raw
            )
            _write_bytes(
                row_dir / "stderr.response", stderr_response.raw
            )
            if result is not None:
                _write_json(row_dir / "result.json", result)
            classification = _classify(
                task=task,
                result=result,
                stdout=stdout,
                stderr=stderr,
                lane=lane,
            )
            evidence = {
                "task_get": _file_record(
                    row_dir / "task_get.response.json",
                    relative_to=staging,
                ),
                "stdout": _file_record(
                    row_dir / "stdout.response",
                    relative_to=staging,
                ),
                "stderr": _file_record(
                    row_dir / "stderr.response",
                    relative_to=staging,
                ),
            }
            if result is not None:
                evidence["result"] = _file_record(
                    row_dir / "result.json",
                    relative_to=staging,
                )
            audit = {
                "task_id": task_id,
                "campaign": lane.campaign,
                "rank": lane.rank,
                "physical_geometry_sha256": (
                    lane.physical_geometry_sha256
                ),
                "source_search_task_id": lane.source_search_task_id,
                "source_seed": lane.source_seed,
                "source_fixed_primary_turns": (
                    lane.source_fixed_primary_turns
                ),
                "source_secondary_turns": lane.source_secondary_turns,
                "scheduler_state": classification["state"],
                "terminal": classification["terminal"],
                "result_available": classification["result_available"],
                "scientific_valid": classification[
                    "scientific_valid"
                ],
                "reasons": classification["reasons"],
                "evidence": evidence,
            }
            task_audit.append(audit)
            if classification["scientific_valid"]:
                if result is None:
                    raise CorrectedReplacementCollectionError(
                        "scientific-valid result unexpectedly absent"
                    )
                prospective.append(
                    (lane, result, row_dir, classification, audit)
                )

        accepted = []
        accepted_geometries: set[str] = set()
        for lane, result, row_dir, classification, audit in sorted(
            prospective, key=lambda item: item[0].task_id
        ):
            geometry = lane.physical_geometry_sha256
            if geometry in accepted_geometries:
                audit["scientific_valid"] = False
                audit["reasons"] = ["duplicate_physical_geometry"]
                continue
            accepted_geometries.add(geometry)
            collection = _collection_value(
                lane=lane,
                result=result,
                row_dir=row_dir,
                replay_plan_path=plan_path,
                classification=classification,
            )
            collection_path = row_dir / "collection.json"
            _write_json(collection_path, collection)
            accepted.append(
                {
                    "task_id": lane.task_id,
                    "physical_geometry_sha256": geometry,
                    "source_search_task_id": lane.source_search_task_id,
                    "source_seed": lane.source_seed,
                    "source_fixed_primary_turns": (
                        lane.source_fixed_primary_turns
                    ),
                    "source_secondary_turns": lane.source_secondary_turns,
                    "source_task_payload_sha256": _task_identity(lane)[
                        "payload_sha256"
                    ],
                    "collection": _file_record(
                        collection_path, relative_to=staging
                    ),
                    "collection_payload_sha256": collection[
                        "payload_sha256"
                    ],
                }
            )

        trigger = _trigger(accepted)
        unsigned_manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "scheduler_url": origin,
            "expected_task_ids": list(EXPECTED_TASK_IDS),
            "source_submission_receipt": _file_record(receipt_file),
            "source_clean_library_replay_plan": _file_record(plan_path),
            "task_audit": task_audit,
            "accepted_collections": accepted,
            "accepted_collection_count": len(accepted),
            "scientifically_invalid_or_pending_count": (
                len(EXPECTED_TASK_IDS) - len(accepted)
            ),
            "retraining_trigger": trigger,
            "strict_al_ingest_direct_consumption": True,
            "strict_al_ingest_argument": "--collection-manifest",
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "legacy_collection_reused": False,
            "legacy_collection_rows_reused": 0,
            "source_kind": (
                "clean-library corrected Scheduler "
                "GET/stdout/stderr/result only"
            ),
            "scientific_validity_is_not_thermal_feasibility": True,
            "production_promotion_authority_granted": False,
        }
        manifest = _seal(
            unsigned_manifest, schema=MANIFEST_SCHEMA
        )
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "manifest.json"


def _authority_matches_collection(
    collection: Mapping[str, Any],
    lane: AuthorityLane,
) -> None:
    expected = {
        "task_id": lane.task_id,
        "campaign": lane.campaign,
        "rank": lane.rank,
        "candidate_physics_sha256": lane.physical_geometry_sha256,
        "source_search_task_id": lane.source_search_task_id,
        "source_seed": lane.source_seed,
        "source_fixed_primary_turns": lane.source_fixed_primary_turns,
        "source_secondary_turns": lane.source_secondary_turns,
        "source_task_payload_sha256": _task_identity(lane)[
            "payload_sha256"
        ],
    }
    if any(collection.get(key) != value for key, value in expected.items()):
        raise CorrectedReplacementCollectionError(
            f"collection task {lane.task_id} authority drifted"
        )
    if collection.get("source_task_identity") != _task_identity(lane):
        raise CorrectedReplacementCollectionError(
            f"collection task {lane.task_id} identity seal drifted"
        )


def authenticate_collection(path: Path) -> dict[str, Any]:
    """Authenticate one emitted corrected replacement collection."""

    collection_path = _regular_file(path, "corrected collection")
    collection = _validate_seal(
        _read_json(collection_path, "corrected collection"),
        schema=COLLECTION_SCHEMA,
    )
    required_flags = {
        "scheduler_get_only_collection": True,
        "scheduler_mutation_performed": False,
        "scientific_valid": True,
        "unique_geometry_admitted": True,
        "truth_use_only": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "automatic_promotion": False,
        "goal_physical_spec_passed": False,
        "legacy_collection_reused": False,
        "legacy_collection_rows_reused": 0,
    }
    if any(
        collection.get(key) != expected
        for key, expected in required_flags.items()
    ):
        raise CorrectedReplacementCollectionError(
            "corrected collection safety flags drifted"
        )
    receipt_path = _resolve_file_record(
        collection.get("submission_receipt"),
        owner_path=collection_path,
        label="submission receipt",
    )
    replay_plan_path = _resolve_file_record(
        collection.get("clean_library_replay_plan"),
        owner_path=collection_path,
        label="clean-library replay plan",
    )
    authority = load_authority(
        receipt_path, replay_plan_path=replay_plan_path
    )
    task_id = _positive_int(collection.get("task_id"), "task_id")
    lane = authority.get(task_id)
    if lane is None:
        raise CorrectedReplacementCollectionError(
            "collection task is outside corrected authority"
        )
    _authority_matches_collection(collection, lane)
    for key, expected_path in (
        ("plan", lane.replacement_plan_path),
        ("clean_library_replay_plan", lane.replacement_plan_path),
        ("source_plan", lane.source_plan_path),
        ("source_params", lane.source_params_path),
    ):
        observed_path = _resolve_file_record(
            collection.get(key),
            owner_path=collection_path,
            label=key,
        )
        if observed_path != expected_path:
            raise CorrectedReplacementCollectionError(
                f"collection {key} authority differs"
            )

    evidence = collection.get("scheduler_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "task_get",
        "stdout",
        "stderr",
        "result",
    }:
        raise CorrectedReplacementCollectionError(
            "collection Scheduler evidence set is incomplete"
        )
    evidence_paths = {
        key: _resolve_file_record(
            record,
            owner_path=collection_path,
            label=f"Scheduler {key} evidence",
        )
        for key, record in evidence.items()
    }
    try:
        task = json.loads(
            evidence_paths["task_get"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectedReplacementCollectionError(
            "Scheduler task GET evidence is invalid"
        ) from exc
    if not isinstance(task, dict):
        raise CorrectedReplacementCollectionError(
            "Scheduler task GET evidence is not an object"
        )
    _validate_task_identity(task, lane)
    stdout_raw = evidence_paths["stdout"].read_bytes()
    stderr_raw = evidence_paths["stderr"].read_bytes()
    try:
        stdout_value = json.loads(stdout_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        stdout_value = stdout_raw.decode("utf-8", errors="replace")
    try:
        stderr_value = json.loads(stderr_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        stderr_value = stderr_raw.decode("utf-8", errors="replace")
    stdout = _stream_text(stdout_value, "stdout")
    stderr = _stream_text(stderr_value, "stderr")
    result = _read_json(evidence_paths["result"], "result evidence")
    if (
        collection.get("result") != result
        or collection.get("result_sha256") != _canonical_sha256(result)
    ):
        raise CorrectedReplacementCollectionError(
            "collection result evidence drifted"
        )
    classification = _classify(
        task=task,
        result=result,
        stdout=stdout,
        stderr=stderr,
        lane=lane,
    )
    if (
        classification["scientific_valid"] is not True
        or classification["reasons"]
        or collection.get("thermal_scientific_contract")
        != classification["thermal_scientific_contract"]
    ):
        raise CorrectedReplacementCollectionError(
            "collection no longer passes terminal scientific truth"
        )
    return {
        "schema_version": AUTHENTICATED_COLLECTION_SCHEMA,
        "collection": collection,
        "plan": copy.deepcopy(dict(lane.replacement_plan)),
        "params": copy.deepcopy(dict(lane.source_params)),
        "selected": {
            "candidate_physics_sha256": (
                lane.physical_geometry_sha256
            ),
            "task_identity": {
                "payload_sha256": _task_identity(lane)[
                    "payload_sha256"
                ],
                "seed": lane.source_seed,
                "fixed_primary_turns": (
                    lane.source_fixed_primary_turns
                ),
            },
            "source_search_task_id": lane.source_search_task_id,
        },
        "submission": copy.deepcopy(dict(lane.submission_receipt)),
    }


def authenticate_manifest(path: Path) -> dict[str, Any]:
    """Authenticate the aggregate manifest and every accepted collection."""

    manifest_path = _regular_file(path, "corrected collection manifest")
    manifest = _validate_seal(
        _read_json(manifest_path, "corrected collection manifest"),
        schema=MANIFEST_SCHEMA,
    )
    required_flags = {
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "strict_al_ingest_direct_consumption": True,
        "strict_al_ingest_argument": "--collection-manifest",
        "scheduler_get_only_collection": True,
        "scheduler_mutation_performed": False,
        "legacy_collection_reused": False,
        "legacy_collection_rows_reused": 0,
        "production_promotion_authority_granted": False,
    }
    if any(
        manifest.get(key) != expected
        for key, expected in required_flags.items()
    ):
        raise CorrectedReplacementCollectionError(
            "corrected manifest safety flags drifted"
        )
    receipt_path = _resolve_file_record(
        manifest.get("source_submission_receipt"),
        owner_path=manifest_path,
        label="manifest submission receipt",
    )
    replay_plan_path = _resolve_file_record(
        manifest.get("source_clean_library_replay_plan"),
        owner_path=manifest_path,
        label="manifest clean-library replay plan",
    )
    authority = load_authority(
        receipt_path, replay_plan_path=replay_plan_path
    )
    audits = manifest.get("task_audit")
    if (
        not isinstance(audits, list)
        or len(audits) != len(EXPECTED_TASK_IDS)
        or {
            _positive_int(row.get("task_id"), "audit task_id")
            for row in audits
            if isinstance(row, Mapping)
        }
        != set(EXPECTED_TASK_IDS)
    ):
        raise CorrectedReplacementCollectionError(
            "manifest task audit set drifted"
        )
    for row in audits:
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping):
            raise CorrectedReplacementCollectionError(
                "manifest task evidence is absent"
            )
        for key, record in evidence.items():
            if key not in {"task_get", "stdout", "stderr", "result"}:
                raise CorrectedReplacementCollectionError(
                    "manifest task evidence key is unsupported"
                )
            _resolve_file_record(
                record,
                owner_path=manifest_path,
                label=f"manifest task {key} evidence",
                relative_root=manifest_path.parent,
            )

    accepted = manifest.get("accepted_collections")
    if not isinstance(accepted, list):
        raise CorrectedReplacementCollectionError(
            "manifest accepted collection list is absent"
        )
    authenticated_paths = []
    trigger_rows = []
    seen_tasks: set[int] = set()
    seen_geometries: set[str] = set()
    for record in accepted:
        if not isinstance(record, Mapping):
            raise CorrectedReplacementCollectionError(
                "manifest accepted collection record is malformed"
            )
        collection_path = _resolve_file_record(
            record.get("collection"),
            owner_path=manifest_path,
            label="accepted corrected collection",
            relative_root=manifest_path.parent,
        )
        view = authenticate_collection(collection_path)
        collection = view["collection"]
        task_id = _positive_int(collection.get("task_id"), "task_id")
        geometry = _require_sha256(
            collection.get("candidate_physics_sha256"),
            "candidate geometry SHA",
        )
        lane = authority.get(task_id)
        if (
            lane is None
            or task_id in seen_tasks
            or geometry in seen_geometries
            or record.get("task_id") != task_id
            or record.get("physical_geometry_sha256") != geometry
            or record.get("collection_payload_sha256")
            != collection["payload_sha256"]
            or record.get("source_task_payload_sha256")
            != collection["source_task_payload_sha256"]
        ):
            raise CorrectedReplacementCollectionError(
                "manifest accepted collection identity drifted"
            )
        seen_tasks.add(task_id)
        seen_geometries.add(geometry)
        authenticated_paths.append(collection_path)
        trigger_rows.append(record)
    expected_trigger = _trigger(trigger_rows)
    if (
        manifest.get("accepted_collection_count") != len(accepted)
        or manifest.get("scientifically_invalid_or_pending_count")
        != len(EXPECTED_TASK_IDS) - len(accepted)
        or manifest.get("retraining_trigger") != expected_trigger
    ):
        raise CorrectedReplacementCollectionError(
            "manifest retraining trigger drifted"
        )
    return {
        "schema_version": AUTHENTICATED_MANIFEST_SCHEMA,
        "manifest": manifest,
        "collection_paths": tuple(authenticated_paths),
        "retraining_trigger": expected_trigger,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--replay-plan",
        type=Path,
        default=DEFAULT_REPLAY_PLAN,
        help="Sealed clean-library replay plan for tasks 97116..97139.",
    )
    parser.add_argument(
        "--scheduler-url", default=DEFAULT_SCHEDULER_URL
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    manifest_path = collect(
        receipt_path=args.receipt,
        output_dir=args.output_dir,
        replay_plan_path=args.replay_plan,
        scheduler_url=args.scheduler_url,
    )
    authenticated = authenticate_manifest(manifest_path)
    output = {
        "status": "ok",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "accepted_collection_count": len(
            authenticated["collection_paths"]
        ),
        "retraining_trigger": authenticated["retraining_trigger"],
        "scheduler_mutation_performed": False,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorrectedReplacementCollectionError as exc:
        print(
            f"[corrected-replacement-collect] ERROR: {exc}",
            file=os.sys.stderr,
        )
        raise SystemExit(2)
