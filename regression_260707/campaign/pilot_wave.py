"""Plan, submit, and report a revision-pinned restart pilot wave.

The wave combines newly sampled campaign designs with deterministic replays
from the trusted b171c7c strict-full cohort.  Planning is entirely local;
only the ``submit`` subcommand mutates scheduler state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
for search_path in (REPO_ROOT, REGRESSION_ROOT, REGRESSION_ROOT / "verify"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import pinned_pilot  # noqa: E402
import quality_contract  # noqa: E402
import scheduler_client  # noqa: E402
from module.input_parameter_260706 import KEYS  # noqa: E402


SCHEMA_VERSION = "mft-restart-pilot-wave-v1"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
TRUSTED_SOLVER_REVISION = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
DEFAULT_LEGACY_DATASET = Path(
    "Y:/git/MFT_1MW_2026/regression_260707/data/dataset/train.parquet"
)
DEFAULT_RESULTS_DATASET = REGRESSION_ROOT / "data" / "dataset" / "train.parquet"
DEFAULT_PROFILE = REGRESSION_ROOT / "verify" / "profiles" / "standard.json"
DEFAULT_SEED = 260710
YIELD_THRESHOLD = 0.85

REPLAY_TARGETS = (
    "Llt_phys",
    "k",
    "P_winding_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
)
TARGET_THRESHOLDS = {
    "Llt_phys": 0.005,
    "k": 0.005,
    "P_winding_total": 0.03,
    "P_Tx_main_group": 0.03,
    "P_Rx_main_group": 0.03,
    "P_Rx_side_total": 0.03,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_safe(value: Any) -> Any:
    """Convert a report to strict JSON without changing its in-memory API."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    value = _json_scalar(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload), stream, ensure_ascii=False, indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return target


def _full_sha(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise ValueError(f"{label} must be a full 40-character git SHA")
    return normalized


def _data_revision(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("data_revision must be a non-empty token without whitespace")
    return normalized


def _lamination_factor(value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("core_lamination_factor must be finite in (0, 1]") from exc
    if not math.isfinite(normalized) or not 0.0 < normalized <= 1.0:
        raise ValueError("core_lamination_factor must be finite in (0, 1]")
    return normalized


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _physical_llt(record: Mapping[str, Any]) -> float:
    try:
        stored = float(record.get("Llt_phys"))
    except (TypeError, ValueError, OverflowError):
        stored = math.nan
    if math.isfinite(stored):
        return stored
    try:
        raw = float(record["Llt"])
        full_model = float(record.get("full_model", 0))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("result has no finite Llt/Llt_phys physical target") from exc
    physical = raw * (2.0 if full_model == 0.0 else 1.0)
    if not math.isfinite(physical):
        raise ValueError("result has no finite Llt/Llt_phys physical target")
    return physical


def _finite_target(record: Mapping[str, Any], target: str) -> float:
    if target == "Llt_phys":
        return _physical_llt(record)
    try:
        value = float(record[target])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"result has no finite {target}") from exc
    if not math.isfinite(value):
        raise ValueError(f"result has no finite {target}")
    return value


def _source_row_id(record: Mapping[str, Any]) -> int | str:
    value = _json_scalar(record.get("task_id"))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if math.isfinite(number) and number > 0 and number == math.floor(number):
        return int(number)
    for key in ("row_id", "task_name", "project_name"):
        candidate = str(record.get(key) or "").strip()
        if candidate:
            return candidate
    raise ValueError("legacy replay row has no stable task_id/task_name/project_name")


def _resolve_trusted_revision(frame: pd.DataFrame, requested: str) -> str:
    requested = str(requested or "").strip().lower()
    if not requested or not re.fullmatch(r"[0-9a-f]{7,40}", requested):
        raise ValueError("trusted solver revision must be a 7-40 character SHA")
    if "git_hash" not in frame.columns:
        raise ValueError("legacy dataset is missing git_hash")
    revisions = sorted({
        str(value).strip().lower()
        for value in frame["git_hash"].dropna().tolist()
        if str(value).strip().lower().startswith(requested)
    })
    if len(revisions) != 1 or not re.fullmatch(r"[0-9a-f]{40}", revisions[0]):
        raise ValueError(
            f"trusted solver revision {requested!r} is missing or ambiguous: {revisions}"
        )
    return revisions[0]


def _stratified_positions(row_count: int, count: int) -> list[int]:
    if count == 0:
        return []
    if row_count < count:
        raise ValueError(
            f"requested {count} legacy replays but only {row_count} eligible rows exist"
        )
    # Inclusive rank points retain both tails and cover the interior Llt range.
    return np.rint(np.linspace(0, row_count - 1, count)).astype(int).tolist()


def select_legacy_replays(
    legacy_dataset: str | Path,
    replay_count: int,
    *,
    trusted_solver_revision: str = TRUSTED_SOLVER_REVISION,
    library_revision: str = LIBRARY_REVISION,
    profile: str | Path | Mapping[str, Any] = DEFAULT_PROFILE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic Llt-stratified rows from the trusted strict cohort."""
    replay_count = _nonnegative_count(replay_count, "replay_count")
    dataset = Path(legacy_dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"legacy dataset not found: {dataset}")
    frame = pd.read_parquet(dataset)
    resolved_solver = _resolve_trusted_revision(frame, trusted_solver_revision)
    library_revision = _full_sha(library_revision, "library_revision")
    if "pyaedt_library_git_hash" not in frame.columns:
        raise ValueError("legacy dataset is missing pyaedt_library_git_hash")
    solver_mask = frame["git_hash"].astype(str).str.lower().eq(resolved_solver)
    library_mask = (
        frame["pyaedt_library_git_hash"].astype(str).str.lower().eq(library_revision)
    )
    pinned = frame.loc[solver_mask & library_mask].copy()
    if pinned.empty:
        raise ValueError("legacy dataset has no rows at the trusted solver/library pins")
    missing_params = [key for key in KEYS if key not in pinned.columns]
    missing_targets = [
        key for key in REPLAY_TARGETS if key != "Llt_phys" and key not in pinned.columns
    ]
    if "Llt_phys" not in pinned.columns and "Llt" not in pinned.columns:
        missing_targets.append("Llt_phys (or Llt)")
    if missing_params or missing_targets:
        raise ValueError(
            "legacy dataset schema is incomplete: "
            f"params={missing_params}, targets={missing_targets}"
        )

    audited = quality_contract.annotate_validity(
        pinned,
        profile,
        expected_solver_revision=resolved_solver,
        expected_library_revision=library_revision,
    )
    strict = audited.loc[audited["_strict_valid_full"].astype(bool)].copy()
    strict["_pilot_Llt_phys"] = [
        _physical_llt(record) for record in strict.to_dict("records")
    ]
    finite = np.isfinite(strict["_pilot_Llt_phys"].to_numpy(dtype=float))
    for target in REPLAY_TARGETS[1:]:
        finite &= np.isfinite(pd.to_numeric(strict[target], errors="coerce"))
    strict = strict.loc[finite].sort_values(
        ["_pilot_Llt_phys", "task_id"], kind="stable"
    )
    positions = _stratified_positions(len(strict), replay_count)
    selected = strict.iloc[positions]

    rows: list[dict[str, Any]] = []
    seen_ids: set[int | str] = set()
    for record in selected.to_dict("records"):
        replay_of = _source_row_id(record)
        if replay_of in seen_ids:
            raise ValueError(f"legacy replay identity is not unique: {replay_of!r}")
        seen_ids.add(replay_of)
        params = {key: _json_scalar(record[key]) for key in KEYS}
        source_targets = {
            target: float(record["_pilot_Llt_phys"])
            if target == "Llt_phys" else float(record[target])
            for target in REPLAY_TARGETS
        }
        rows.append({
            "replay_of": replay_of,
            "source_task_name": str(record.get("task_name") or ""),
            "source_project_name": str(record.get("project_name") or ""),
            "source_saved_at": str(record.get("saved_at") or ""),
            "params": params,
            "source_targets": source_targets,
        })
    metadata = {
        "path": str(dataset.resolve()),
        "trusted_solver_revision": resolved_solver,
        "trusted_library_revision": library_revision,
        "pinned_rows": int(len(pinned)),
        "strict_full_rows": int(len(audited.loc[audited["_strict_valid_full"]])),
        "eligible_rows": int(len(strict)),
        "selection": "inclusive_even_rank_points_sorted_by_Llt_phys",
    }
    return rows, metadata


def _profile_payload(profile: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        payload = dict(profile)
    else:
        with open(profile, encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload.get("param_overrides"), dict):
        raise ValueError("profile must contain param_overrides")
    return payload


def _manifest_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for entry in manifest.get("entries", []):
        entries.append({
            key: value for key, value in entry.items()
            if key not in {
                "name", "workdir", "params_sha256", "dedupe_key", "task_id"
            }
        })
    return {
        "schema_version": manifest.get("schema_version"),
        "solver_revision": manifest.get("solver_revision"),
        "library_revision": manifest.get("library_revision"),
        "data_revision": manifest.get("data_revision"),
        "core_lamination_factor": manifest.get("core_lamination_factor"),
        "seed": manifest.get("seed"),
        "fresh_offset": manifest.get("fresh_offset"),
        "yield_threshold": manifest.get("yield_threshold"),
        "profile": manifest.get("profile"),
        "legacy_source": manifest.get("legacy_source"),
        "entries": entries,
    }


def build_manifest(
    *,
    solver_revision: str,
    data_revision: str,
    core_lamination_factor: float,
    legacy_dataset: str | Path = DEFAULT_LEGACY_DATASET,
    fresh_count: int = 25,
    replay_count: int = 25,
    seed: int = DEFAULT_SEED,
    library_revision: str = LIBRARY_REVISION,
    profile: str | Path | Mapping[str, Any] = DEFAULT_PROFILE,
    trusted_solver_revision: str = TRUSTED_SOLVER_REVISION,
    fresh_offset: int = pinned_pilot.PILOT_RESERVED_VALID_CANDIDATES,
) -> dict[str, Any]:
    """Construct a no-submission restart-pilot manifest."""
    solver_revision = _full_sha(solver_revision, "solver_revision")
    library_revision = _full_sha(library_revision, "library_revision")
    if library_revision != LIBRARY_REVISION:
        raise ValueError(f"library_revision must use the campaign pin {LIBRARY_REVISION}")
    data_revision = _data_revision(data_revision)
    kf = _lamination_factor(core_lamination_factor)
    fresh_count = _nonnegative_count(fresh_count, "fresh_count")
    replay_count = _nonnegative_count(replay_count, "replay_count")
    fresh_offset = _nonnegative_count(fresh_offset, "fresh_offset")
    if fresh_count + replay_count == 0:
        raise ValueError("pilot manifest must contain at least one entry")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    profile_payload = _profile_payload(profile)

    replay_rows: list[dict[str, Any]] = []
    legacy_source = None
    if replay_count:
        replay_rows, legacy_source = select_legacy_replays(
            legacy_dataset,
            replay_count,
            trusted_solver_revision=trusted_solver_revision,
            library_revision=library_revision,
            profile=profile_payload,
        )

    pending: list[dict[str, Any]] = []
    cursor = pinned_pilot.cursor_after_valid_candidates(fresh_offset, seed=seed)
    for index in range(fresh_count):
        cursor, raw_index, candidate = pinned_pilot.next_valid_candidate(cursor, seed)
        params = {key: _json_scalar(candidate[key]) for key in KEYS}
        params["core_lamination_factor"] = kf
        params["physics_data_revision"] = data_revision
        pending.append({
            "entry_id": f"fresh-{index:03d}",
            "kind": "fresh",
            "index": index,
            "candidate_raw_index": int(raw_index),
            "params": params,
        })
    for index, replay in enumerate(replay_rows):
        params = dict(replay["params"])
        params["core_lamination_factor"] = kf
        params["physics_data_revision"] = data_revision
        pending.append({
            "entry_id": f"replay-{index:03d}",
            "kind": "legacy_replay",
            "index": index,
            "params": params,
            "replay_of": replay["replay_of"],
            "source_task_name": replay["source_task_name"],
            "source_project_name": replay["source_project_name"],
            "source_saved_at": replay["source_saved_at"],
            "source_targets": replay["source_targets"],
        })

    identity = _manifest_identity_payload({
        "schema_version": SCHEMA_VERSION,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "data_revision": data_revision,
        "core_lamination_factor": kf,
        "seed": seed,
        "fresh_offset": fresh_offset,
        "yield_threshold": YIELD_THRESHOLD,
        "profile": profile_payload,
        "legacy_source": legacy_source,
        "entries": pending,
    })
    manifest_id = _sha256(identity)
    prefix = f"mft-restart-{solver_revision[:7]}-{manifest_id[:10]}"
    entries = []
    for entry in pending:
        code = "f" if entry["kind"] == "fresh" else "r"
        name = f"{prefix}-{code}{entry['index']:03d}"
        entry["name"] = name
        entry["workdir"] = name.replace("-", "_")
        entry["params_sha256"] = _sha256(entry["params"])
        entry["dedupe_key"] = scheduler_client.verification_dedupe_key(
            name,
            entry["params"],
            profile_payload,
            solver_revision,
            library_revision,
        )
        entry["task_id"] = None
        entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "task_prefix": prefix,
        "created_at": _utc_now(),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "data_revision": data_revision,
        "core_lamination_factor": kf,
        "seed": seed,
        "fresh_offset": fresh_offset,
        "fresh_count": fresh_count,
        "replay_count": replay_count,
        "entry_count": len(entries),
        "yield_threshold": YIELD_THRESHOLD,
        "profile": profile_payload,
        "legacy_source": legacy_source,
        "submission": {"executed": False, "completed_at": None},
        "entries": entries,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_manifest(
    manifest: Mapping[str, Any], *, require_submitted: bool = False
) -> dict[str, Any]:
    """Fail closed on revision, payload, replay, and dedupe identity drift."""
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema must be {SCHEMA_VERSION}")
    solver = _full_sha(manifest.get("solver_revision"), "solver_revision")
    library = _full_sha(manifest.get("library_revision"), "library_revision")
    if library != LIBRARY_REVISION:
        raise ValueError(f"manifest library revision must be {LIBRARY_REVISION}")
    revision = _data_revision(manifest.get("data_revision"))
    kf = _lamination_factor(manifest.get("core_lamination_factor"))
    if manifest.get("yield_threshold") != YIELD_THRESHOLD:
        raise ValueError(f"manifest yield threshold must be exactly {YIELD_THRESHOLD}")
    profile = _profile_payload(manifest.get("profile") or {})
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest entries must be a non-empty list")
    if manifest.get("entry_count") != len(entries):
        raise ValueError("manifest entry_count does not match entries")
    if manifest.get("fresh_count") != sum(
        entry.get("kind") == "fresh" for entry in entries if isinstance(entry, dict)
    ) or manifest.get("replay_count") != sum(
        entry.get("kind") == "legacy_replay"
        for entry in entries if isinstance(entry, dict)
    ):
        raise ValueError("manifest fresh/replay counts do not match entries")
    expected_manifest_id = _sha256(_manifest_identity_payload(manifest))
    if manifest.get("manifest_id") != expected_manifest_id:
        raise ValueError("manifest identity drift")
    expected_prefix = f"mft-restart-{solver[:7]}-{expected_manifest_id[:10]}"
    if manifest.get("task_prefix") != expected_prefix:
        raise ValueError("manifest task prefix drift")
    names: set[str] = set()
    entry_ids: set[str] = set()
    replay_ids: set[int | str] = set()
    task_ids: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest entry must be an object")
        name = str(entry.get("name") or "")
        entry_id = str(entry.get("entry_id") or "")
        if not name or name in names or not entry_id or entry_id in entry_ids:
            raise ValueError("manifest entry names/IDs must be present and unique")
        names.add(name)
        entry_ids.add(entry_id)
        kind = entry.get("kind")
        if kind not in {"fresh", "legacy_replay"}:
            raise ValueError(f"invalid manifest entry kind: {kind!r}")
        index = entry.get("index")
        expected_code = "f" if kind == "fresh" else "r"
        expected_name = (
            f"{expected_prefix}-{expected_code}{index:03d}"
            if isinstance(index, int) and not isinstance(index, bool) else None
        )
        if name != expected_name or entry.get("workdir") != name.replace("-", "_"):
            raise ValueError(f"manifest entry {entry_id} name/workdir identity drift")
        params = entry.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"manifest entry {entry_id} has no params")
        if params.get("physics_data_revision") != revision:
            raise ValueError(f"manifest entry {entry_id} data revision drift")
        try:
            entry_kf = float(params.get("core_lamination_factor"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"manifest entry {entry_id} has invalid kf") from exc
        if not math.isclose(entry_kf, kf, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"manifest entry {entry_id} kf drift")
        if "replay_of" in params:
            raise ValueError(f"manifest entry {entry_id} leaks replay_of into physics params")
        if entry.get("params_sha256") != _sha256(params):
            raise ValueError(f"manifest entry {entry_id} parameter digest drift")
        expected_dedupe = scheduler_client.verification_dedupe_key(
            name, params, profile, solver, library
        )
        if entry.get("dedupe_key") != expected_dedupe:
            raise ValueError(f"manifest entry {entry_id} dedupe identity drift")
        if kind == "legacy_replay":
            replay_of = entry.get("replay_of")
            targets = entry.get("source_targets")
            if replay_of is None or replay_of in replay_ids:
                raise ValueError("replay_of identities must be present and unique")
            replay_ids.add(replay_of)
            if not isinstance(targets, dict) or set(targets) != set(REPLAY_TARGETS):
                raise ValueError(f"manifest entry {entry_id} source target schema drift")
            for target in REPLAY_TARGETS:
                _finite_target(targets, target)
        task_id = entry.get("task_id")
        if task_id is not None:
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
                raise ValueError(f"manifest entry {entry_id} has invalid task_id")
            if task_id in task_ids:
                raise ValueError(f"manifest has duplicate task_id {task_id}")
            task_ids.add(task_id)
    if require_submitted:
        submission = manifest.get("submission")
        if (
            not isinstance(submission, dict)
            or submission.get("executed") is not True
            or not str(submission.get("completed_at") or "").strip()
            or len(task_ids) != len(entries)
        ):
            raise ValueError(
                "report requires a completely submitted manifest with every task ID"
            )
    return dict(manifest)


def submit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Idempotently submit all entries and update the manifest after each task."""
    path = Path(manifest_path)
    manifest = validate_manifest(_load_json(path))
    solver = manifest["solver_revision"]
    library = manifest["library_revision"]
    profile = manifest["profile"]
    with scheduler_client.campaign_mutation_lock():
        for entry in manifest["entries"]:
            task_id = scheduler_client.submit_verification(
                name=entry["name"],
                workdir=entry["workdir"],
                params=entry["params"],
                profile=profile,
                mem_mb=int(profile.get("mem_mb", 32768)),
                cpus=int(profile.get("cpus", 4)),
                solver_revision=solver,
                library_revision=library,
            )
            if task_id is None:
                raise RuntimeError(f"scheduler did not return a task ID for {entry['name']}")
            task_id = int(task_id)
            prior = entry.get("task_id")
            if prior is not None and prior != task_id:
                raise RuntimeError(
                    f"dedupe reconciliation changed task ID for {entry['name']}: "
                    f"{prior} -> {task_id}"
                )
            entry["task_id"] = task_id
            manifest["submission"]["executed"] = True
            manifest["submission"]["last_updated_at"] = _utc_now()
            _atomic_json(path, manifest)
    validate_manifest(manifest)
    manifest["submission"]["completed_at"] = _utc_now()
    _atomic_json(path, manifest)
    return manifest


def _matching_result(entry: Mapping[str, Any], results: pd.DataFrame) -> tuple[dict | None, str | None]:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame")
    task_id = entry.get("task_id")
    if task_id is None:
        return None, "missing_result"
    matches = results.iloc[0:0]
    if "task_id" in results.columns:
        ids = pd.to_numeric(results["task_id"], errors="coerce")
        matches = results.loc[ids.eq(float(task_id))]
    elif "task_name" in results.columns:
        matches = results.loc[results["task_name"].astype(str).eq(str(entry.get("name")))]
    if len(matches) == 0:
        return None, "missing_result"
    if len(matches) != 1:
        return None, "ambiguous_result"
    return matches.iloc[0].to_dict(), None


def relative_error(actual: Any, expected: Any) -> float:
    """Absolute paired relative error, with an explicit zero-reference rule."""
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if not math.isfinite(actual) or not math.isfinite(expected):
        return math.inf
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual - expected) / abs(expected)


def _result_identity_reasons(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    """Validate replay identity independently from strict thermal validity."""
    reasons = []
    expected_name = str(entry.get("name") or "")
    if expected_name and str(result.get("task_name") or "") != expected_name:
        reasons.append("identity:task_name")
    expected_solver = str(manifest.get("solver_revision") or "").lower()
    if expected_solver and str(result.get("git_hash") or "").lower() != expected_solver:
        reasons.append("identity:solver_revision")
    expected_library = str(manifest.get("library_revision") or "").lower()
    if expected_library and str(
        result.get("pyaedt_library_git_hash") or ""
    ).lower() != expected_library:
        reasons.append("identity:library_revision")
    expected_revision = str(manifest.get("data_revision") or "")
    if expected_revision and str(
        result.get("physics_data_revision") or ""
    ) != expected_revision:
        reasons.append("identity:physics_data_revision")
    if manifest.get("core_lamination_factor") is not None:
        try:
            result_kf = float(result.get("core_lamination_factor"))
            expected_kf = float(manifest.get("core_lamination_factor"))
            kf_matches = math.isclose(
                result_kf, expected_kf, rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError, OverflowError):
            kf_matches = False
        if not kf_matches:
            reasons.append("identity:core_lamination_factor")
    params = entry.get("params")
    profile = manifest.get("profile")
    if isinstance(params, dict) and isinstance(profile, dict):
        effective_params = scheduler_client.effective_verification_params(
            params, profile
        )
        if not scheduler_client.result_matches_params(result, effective_params):
            reasons.append("identity:parameter_echo")
    return reasons


def build_salvage_report(
    manifest: Mapping[str, Any], results: pd.DataFrame
) -> dict[str, Any]:
    """Build per-pair comparisons and fail-closed target-level gates."""
    entries = [
        entry for entry in manifest.get("entries", [])
        if entry.get("kind") == "legacy_replay"
    ]
    pairs = []
    target_errors: dict[str, list[float]] = {target: [] for target in REPLAY_TARGETS}
    target_passes: dict[str, list[bool]] = {target: [] for target in REPLAY_TARGETS}
    for entry in entries:
        result, match_error = _matching_result(entry, results)
        identity_reasons = (
            _result_identity_reasons(manifest, entry, result)
            if result is not None else []
        )
        if match_error is None and identity_reasons:
            match_error = ";".join(identity_reasons)
        comparisons = {}
        for target in REPLAY_TARGETS:
            expected = entry.get("source_targets", {}).get(target)
            try:
                actual = _finite_target(result or {}, target)
            except ValueError:
                actual = None
            error = relative_error(actual, expected)
            threshold = TARGET_THRESHOLDS[target]
            passed = match_error is None and math.isfinite(error) and error <= threshold
            comparisons[target] = {
                "source": expected,
                "actual": actual,
                "relative_error": error,
                "threshold": threshold,
                "passed": passed,
            }
            target_errors[target].append(error)
            target_passes[target].append(passed)
        pairs.append({
            "entry_id": entry.get("entry_id"),
            "task_id": entry.get("task_id"),
            "task_name": entry.get("name"),
            "replay_of": entry.get("replay_of"),
            "match_error": match_error,
            "identity_reasons": identity_reasons,
            "targets": comparisons,
            "passed": all(item["passed"] for item in comparisons.values()),
        })
    target_reports = {}
    for target in REPLAY_TARGETS:
        errors = target_errors[target]
        passes = target_passes[target]
        target_reports[target] = {
            "threshold": TARGET_THRESHOLDS[target],
            "pair_count": len(entries),
            "passing_pairs": sum(passes),
            "max_relative_error": max(errors, default=math.inf),
            "passed": bool(entries) and all(passes),
        }
    return {
        "pair_count": len(entries),
        "pairs": pairs,
        "targets": target_reports,
        "passed": bool(entries) and all(
            report["passed"] for report in target_reports.values()
        ),
    }


def _entry_validity(
    manifest: Mapping[str, Any], results: pd.DataFrame
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        result, match_error = _matching_result(entry, results)
        report = {
            "entry_id": entry.get("entry_id"),
            "kind": entry.get("kind"),
            "task_id": entry.get("task_id"),
            "task_name": entry.get("name"),
            "strict_full": False,
            "reasons": [match_error] if match_error else [],
        }
        reports.append(report)
        if result is not None:
            result["_pilot_entry_id"] = entry.get("entry_id")
            found.append(result)
    if not found:
        return reports

    annotated = quality_contract.annotate_validity(
        pd.DataFrame(found),
        manifest.get("profile"),
        expected_solver_revision=manifest.get("solver_revision"),
        expected_library_revision=manifest.get("library_revision"),
    )
    by_id = {
        row["_pilot_entry_id"]: row for row in annotated.to_dict("records")
    }
    entries_by_id = {
        entry.get("entry_id"): entry for entry in manifest.get("entries", [])
    }
    for report in reports:
        row = by_id.get(report["entry_id"])
        if row is None:
            continue
        reasons = [
            reason for reason in str(row.get("_strict_invalid_reasons") or "").split(";")
            if reason
        ]
        entry = entries_by_id[report["entry_id"]]
        reasons.extend(_result_identity_reasons(manifest, entry, row))
        report["reasons"] = list(dict.fromkeys(reasons))
        report["strict_full"] = bool(row.get("_strict_valid_full")) and not reasons
    return reports


def build_report(
    manifest: Mapping[str, Any], results: pd.DataFrame
) -> dict[str, Any]:
    """Build strict-full yield/quarantine and replay-salvage gates."""
    entries = list(manifest.get("entries", []))
    validity = _entry_validity(manifest, results)
    quarantines = Counter(
        reason for entry in validity for reason in entry["reasons"]
    )
    strict_count = sum(bool(entry["strict_full"]) for entry in validity)
    total = len(entries)
    rate = strict_count / total if total else 0.0
    threshold = YIELD_THRESHOLD
    kind_reports = {}
    for kind in ("fresh", "legacy_replay"):
        rows = [entry for entry in validity if entry["kind"] == kind]
        valid = sum(bool(entry["strict_full"]) for entry in rows)
        kind_reports[kind] = {
            "planned": len(rows),
            "strict_full": valid,
            "yield": valid / len(rows) if rows else None,
        }
    salvage = build_salvage_report(manifest, results)
    submission = manifest.get("submission")
    submission_complete = bool(
        isinstance(submission, dict)
        and submission.get("executed") is True
        and str(submission.get("completed_at") or "").strip()
        and entries
        and all(
            isinstance(entry.get("task_id"), int)
            and not isinstance(entry.get("task_id"), bool)
            and entry["task_id"] > 0
            for entry in entries
        )
    )
    yield_report = {
        "planned": total,
        "strict_full": strict_count,
        "quarantined": total - strict_count,
        "yield": rate,
        "threshold": threshold,
        "passed": rate >= threshold,
        "by_kind": kind_reports,
        "quarantine_reasons": dict(sorted(quarantines.items())),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest.get("manifest_id"),
        "generated_at": _utc_now(),
        "solver_revision": manifest.get("solver_revision"),
        "library_revision": manifest.get("library_revision"),
        "data_revision": manifest.get("data_revision"),
        "core_lamination_factor": manifest.get("core_lamination_factor"),
        "yield": yield_report,
        "salvage": salvage,
        "submission_complete": submission_complete,
        "entry_validity": validity,
        "passed": submission_complete and yield_report["passed"] and salvage["passed"],
    }


def _print_report(report: Mapping[str, Any]) -> None:
    yield_report = report["yield"]
    print(
        "strict-full yield: "
        f"{yield_report['strict_full']}/{yield_report['planned']} "
        f"({yield_report['yield']:.1%}) / gate {yield_report['threshold']:.1%} "
        f"=> {'PASS' if yield_report['passed'] else 'FAIL'}"
    )
    print("quarantine reasons:")
    if yield_report["quarantine_reasons"]:
        for reason, count in yield_report["quarantine_reasons"].items():
            print(f"  {count:3d}  {reason}")
    else:
        print("    0  none")

    salvage = report["salvage"]
    print("\nlegacy replay paired relative errors:")
    labels = ["replay_of", *REPLAY_TARGETS, "gate"]
    print("  ".join(f"{label:>14}" for label in labels))
    for pair in salvage["pairs"]:
        values = [str(pair["replay_of"])]
        for target in REPLAY_TARGETS:
            error = pair["targets"][target]["relative_error"]
            values.append(f"{error:.3%}" if math.isfinite(error) else "missing")
        values.append("PASS" if pair["passed"] else "FAIL")
        print("  ".join(f"{value:>14}" for value in values))
    print("\npaired target gates:")
    for target, gate in salvage["targets"].items():
        maximum = gate["max_relative_error"]
        maximum_text = f"{maximum:.3%}" if math.isfinite(maximum) else "missing"
        print(
            f"  {target:22s} {gate['passing_pairs']:3d}/{gate['pair_count']:<3d} "
            f"max={maximum_text:>9s} limit={gate['threshold']:.1%} "
            f"{'PASS' if gate['passed'] else 'FAIL'}"
        )
    print(f"\noverall restart pilot: {'PASS' if report['passed'] else 'FAIL'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write a pilot manifest without submitting")
    plan.add_argument(
        "--solver-sha", "--solver-revision", dest="solver_revision", required=True
    )
    plan.add_argument("--data-revision", required=True)
    plan.add_argument(
        "--core-lamination-factor", "--kf", dest="core_lamination_factor",
        type=float, required=True,
    )
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--legacy-dataset", type=Path, default=DEFAULT_LEGACY_DATASET)
    plan.add_argument("--fresh-count", type=int, default=25)
    plan.add_argument("--replay-count", type=int, default=25)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument("--fresh-offset", type=int, default=pinned_pilot.PILOT_RESERVED_VALID_CANDIDATES)
    plan.add_argument("--trusted-solver-revision", default=TRUSTED_SOLVER_REVISION)

    submit = subparsers.add_parser("submit", help="submit every task in a planned manifest")
    submit.add_argument("--manifest", type=Path, required=True)

    report = subparsers.add_parser("report", help="report collected yield and replay gates")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DATASET)
    report.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        manifest = build_manifest(
            solver_revision=args.solver_revision,
            data_revision=args.data_revision,
            core_lamination_factor=args.core_lamination_factor,
            legacy_dataset=args.legacy_dataset,
            fresh_count=args.fresh_count,
            replay_count=args.replay_count,
            seed=args.seed,
            trusted_solver_revision=args.trusted_solver_revision,
            fresh_offset=args.fresh_offset,
        )
        path = _atomic_json(args.manifest, manifest)
        source_rows = (manifest.get("legacy_source") or {}).get("strict_full_rows", 0)
        print(
            f"planned {manifest['fresh_count']} fresh + {manifest['replay_count']} replay "
            f"entries from {source_rows} trusted strict-full rows -> {path}\n"
            f"collection prefix: {manifest['task_prefix']}"
        )
        return 0
    if args.command == "submit":
        manifest = submit_manifest(args.manifest)
        print(
            f"submitted/reconciled {len(manifest['entries'])} tasks at "
            f"solver {manifest['solver_revision']} -> {args.manifest}"
        )
        return 0

    manifest = validate_manifest(_load_json(args.manifest), require_submitted=True)
    if not args.results.is_file():
        raise FileNotFoundError(f"collected results parquet not found: {args.results}")
    results = pd.read_parquet(args.results)
    report = build_report(manifest, results)
    output = args.output or args.manifest.with_name(
        f"{args.manifest.stem}.report.json"
    )
    _atomic_json(output, report)
    _print_report(report)
    print(f"report JSON: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
