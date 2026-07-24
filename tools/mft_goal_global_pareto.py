"""Authenticate and globally sort pooled MFT NSGA-II terminal populations.

The deadline campaign must aggregate every terminal population row, rather
than unioning seed-local Pareto fronts.  This module intentionally accepts a
small, sealed manifest and fail-closes on mixed scientific identities,
incomplete populations, duplicate seeds, or inconsistent physical duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MANIFEST_SCHEMA = "mft-goal-terminal-population-manifest-v1"
SUMMARY_SCHEMA = "mft-goal-global-pareto-summary-v1"
REQUIRED_IDENTITIES = (
    "dataset_sha256",
    "evaluation_model_sha256",
    "constraint_spec_sha256",
    "cooling_contract_sha256",
    "operating_point_sha256",
)
REQUIRED_ROW_COLUMNS = (
    "terminal_population_index",
    "physical_geometry_sha256",
)
HASH_HEX_LENGTH = 64


class ParetoContractError(RuntimeError):
    """Raised when terminal evidence does not satisfy the aggregate contract."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != HASH_HEX_LENGTH:
        raise ParetoContractError(f"{label} is not a SHA256 hex digest")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise ParetoContractError(
            f"{label} is not a SHA256 hex digest"
        ) from exc
    return text


def _require_string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ParetoContractError(f"{label} must be a non-empty unique string list")
    return list(value)


def _resolve_contained(base: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ParetoContractError(f"{label} is missing")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ParetoContractError(f"{label} escapes manifest directory") from exc
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ParetoContractError(f"unsupported terminal table suffix: {suffix}")


def validate_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value = _read_json(resolved)
    if not isinstance(value, dict):
        raise ParetoContractError("manifest must be a JSON object")
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ParetoContractError("manifest schema mismatch")

    identities = value.get("identities")
    if not isinstance(identities, dict) or set(identities) != set(
        REQUIRED_IDENTITIES
    ):
        raise ParetoContractError("manifest identities are incomplete or contain extras")
    normalized_identities = {
        name: _require_hash(identities[name], f"identities.{name}")
        for name in REQUIRED_IDENTITIES
    }

    objectives = _require_string_list(
        value.get("objective_columns"), "objective_columns"
    )
    if len(objectives) != 2:
        raise ParetoContractError("exactly two objective columns are required")
    physical_constraints = _require_string_list(
        value.get("physical_constraint_columns"),
        "physical_constraint_columns",
    )
    normalized_constraints = _require_string_list(
        value.get("normalized_constraint_columns"),
        "normalized_constraint_columns",
    )
    if len(physical_constraints) != len(normalized_constraints):
        raise ParetoContractError(
            "physical and normalized constraint counts differ"
        )

    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ParetoContractError("manifest records must be non-empty")

    return {
        **value,
        "path": resolved,
        "identities": normalized_identities,
        "objective_columns": objectives,
        "physical_constraint_columns": physical_constraints,
        "normalized_constraint_columns": normalized_constraints,
        "records": records,
    }


def _load_terminal_record(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    seen_seeds: set[int],
) -> pd.DataFrame:
    if record.get("terminal_authenticated") is not True:
        raise ParetoContractError("record is not terminal_authenticated=true")
    try:
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParetoContractError("record seed is invalid") from exc
    if seed in seen_seeds:
        raise ParetoContractError(f"duplicate seed: {seed}")
    seen_seeds.add(seed)

    try:
        population_count = int(record["terminal_population_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParetoContractError(
            f"seed {seed} terminal_population_count is invalid"
        ) from exc
    if population_count != 320:
        raise ParetoContractError(
            f"seed {seed} terminal population is {population_count}, expected 320"
        )

    record_identities = record.get("identities")
    if not isinstance(record_identities, dict):
        raise ParetoContractError(f"seed {seed} identities are missing")
    normalized_record_identities = {
        name: _require_hash(record_identities.get(name), f"seed {seed}.{name}")
        for name in REQUIRED_IDENTITIES
    }
    if normalized_record_identities != manifest["identities"]:
        raise ParetoContractError(f"seed {seed} scientific identity mismatch")

    manifest_dir = manifest["path"].parent
    table_path = _resolve_contained(
        manifest_dir, record.get("table_path"), f"seed {seed}.table_path"
    )
    expected_sha = _require_hash(
        record.get("table_sha256"), f"seed {seed}.table_sha256"
    )
    actual_sha = _sha256_file(table_path)
    if actual_sha != expected_sha:
        raise ParetoContractError(f"seed {seed} terminal table SHA256 mismatch")

    frame = _read_table(table_path)
    if len(frame) != population_count:
        raise ParetoContractError(
            f"seed {seed} table has {len(frame)} rows, expected {population_count}"
        )
    required = set(REQUIRED_ROW_COLUMNS)
    required.update(manifest["objective_columns"])
    required.update(manifest["physical_constraint_columns"])
    required.update(manifest["normalized_constraint_columns"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ParetoContractError(
            f"seed {seed} table misses columns: {','.join(missing)}"
        )

    indices = pd.to_numeric(
        frame["terminal_population_index"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(indices).all()
        or not np.array_equal(indices, np.arange(population_count, dtype=float))
    ):
        raise ParetoContractError(
            f"seed {seed} terminal_population_index is not exactly 0..319"
        )

    numeric_columns = (
        manifest["objective_columns"]
        + manifest["physical_constraint_columns"]
        + manifest["normalized_constraint_columns"]
    )
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ParetoContractError(f"seed {seed} contains non-finite numeric evidence")
    frame.loc[:, numeric_columns] = numeric

    hashes = [
        _require_hash(value, f"seed {seed}.physical_geometry_sha256")
        for value in frame["physical_geometry_sha256"]
    ]
    frame = frame.copy()
    frame["physical_geometry_sha256"] = hashes
    frame["source_seed"] = seed
    frame["source_task_id"] = str(record.get("task_id", ""))
    frame["source_bundle_id"] = str(record.get("bundle_id", ""))
    frame["source_model_sha256"] = _require_hash(
        record.get(
            "source_model_sha256",
            manifest["identities"]["evaluation_model_sha256"],
        ),
        f"seed {seed}.source_model_sha256",
    )
    return frame


def load_terminal_population(manifest: Mapping[str, Any]) -> pd.DataFrame:
    seen_seeds: set[int] = set()
    frames = [
        _load_terminal_record(
            record,
            manifest=manifest,
            seen_seeds=seen_seeds,
        )
        for record in manifest["records"]
    ]
    return pd.concat(frames, ignore_index=True, copy=False)


def _equal_numeric_rows(
    values: np.ndarray, *, absolute_tolerance: float = 1e-12
) -> bool:
    if len(values) <= 1:
        return True
    first = values[0]
    return bool(
        np.allclose(
            values,
            np.broadcast_to(first, values.shape),
            rtol=0.0,
            atol=absolute_tolerance,
            equal_nan=False,
        )
    )


def deduplicate_physical_candidates(
    frame: pd.DataFrame,
    *,
    comparison_columns: Sequence[str],
) -> pd.DataFrame:
    ordered_all = frame.sort_values(
        [
            "physical_geometry_sha256",
            "source_seed",
            "terminal_population_index",
        ],
        kind="stable",
    )
    duplicate_mask = ordered_all.duplicated(
        "physical_geometry_sha256", keep=False
    )
    unique = ordered_all.loc[~duplicate_mask].copy()
    unique["source_occurrence_count"] = 1
    # Direct source columns are the complete provenance for unique rows. JSON
    # is reserved for the uncommon many-to-one case, avoiding millions of
    # tiny JSON allocations in a 4,096-seed campaign.
    unique["source_provenance_json"] = ""

    duplicate_rows: list[pd.Series] = []
    for physical_hash, group in ordered_all.loc[duplicate_mask].groupby(
        "physical_geometry_sha256", sort=True, dropna=False
    ):
        values = group[list(comparison_columns)].to_numpy(dtype=float)
        if not _equal_numeric_rows(values):
            raise ParetoContractError(
                "same physical geometry has inconsistent final evaluation: "
                f"{physical_hash}"
            )
        ordered = group.sort_values(
            ["source_seed", "terminal_population_index"],
            kind="stable",
        )
        representative = ordered.iloc[0].copy()
        provenance = [
            {
                "seed": int(row.source_seed),
                "task_id": str(row.source_task_id),
                "bundle_id": str(row.source_bundle_id),
                "terminal_population_index": int(row.terminal_population_index),
                "source_model_sha256": str(row.source_model_sha256),
            }
            for row in ordered.itertuples(index=False)
        ]
        representative["source_occurrence_count"] = len(provenance)
        representative["source_provenance_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        duplicate_rows.append(representative)
    if duplicate_rows:
        unique = pd.concat(
            [unique, pd.DataFrame(duplicate_rows)],
            ignore_index=True,
            copy=False,
        )
    if unique.empty:
        raise ParetoContractError("no unique candidates remain")
    return unique.sort_values(
        "physical_geometry_sha256", kind="stable"
    ).reset_index(drop=True)


class _FenwickMax:
    def __init__(self, size: int) -> None:
        self.values = np.full(size + 1, -1, dtype=np.int64)

    def update(self, index: int, value: int) -> None:
        while index < len(self.values):
            if value > self.values[index]:
                self.values[index] = value
            index += index & -index

    def query(self, index: int) -> int:
        result = -1
        while index > 0:
            if self.values[index] > result:
                result = int(self.values[index])
            index -= index & -index
        return result


def nondominated_ranks_2d(objectives: np.ndarray) -> np.ndarray:
    """Return exact strict-Pareto ranks for two minimization objectives.

    The sweep is O(n log n). Equal points receive the same rank and do not
    dominate one another.
    """

    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ParetoContractError("objectives must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ParetoContractError("objectives contain non-finite values")
    count = len(values)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((np.arange(count), values[:, 1], values[:, 0]))
    unique_y = np.unique(values[:, 1])
    y_indices = np.searchsorted(unique_y, values[:, 1], side="left") + 1
    tree = _FenwickMax(len(unique_y))
    ranks = np.empty(count, dtype=np.int64)

    cursor = 0
    while cursor < count:
        x_value = values[order[cursor], 0]
        x_end = cursor + 1
        while x_end < count and values[order[x_end], 0] == x_value:
            x_end += 1

        within = cursor
        while within < x_end:
            y_value = values[order[within], 1]
            y_end = within + 1
            while y_end < x_end and values[order[y_end], 1] == y_value:
                y_end += 1
            y_index = int(y_indices[order[within]])
            rank = tree.query(y_index) + 1
            group_indices = order[within:y_end]
            ranks[group_indices] = rank
            tree.update(y_index, rank)
            within = y_end
        cursor = x_end
    return ranks


def crowding_distance(
    objectives: np.ndarray, ranks: np.ndarray
) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    rank_values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) != len(rank_values):
        raise ParetoContractError("crowding inputs have incompatible shapes")
    distances = np.zeros(len(values), dtype=float)
    for rank in np.unique(rank_values):
        members = np.flatnonzero(rank_values == rank)
        if len(members) <= 2:
            distances[members] = math.inf
            continue
        for objective in range(2):
            ordered = members[
                np.argsort(values[members, objective], kind="stable")
            ]
            distances[ordered[0]] = math.inf
            distances[ordered[-1]] = math.inf
            span = (
                values[ordered[-1], objective]
                - values[ordered[0], objective]
            )
            if span <= 0:
                continue
            interior = ordered[1:-1]
            contribution = (
                values[ordered[2:], objective]
                - values[ordered[:-2], objective]
            ) / span
            finite = np.isfinite(distances[interior])
            distances[interior[finite]] += contribution[finite]
    return distances


def rank_candidates(
    frame: pd.DataFrame,
    *,
    objective_columns: Sequence[str],
    physical_constraint_columns: Sequence[str],
    normalized_constraint_columns: Sequence[str],
) -> pd.DataFrame:
    ranked = frame.copy()
    objectives = ranked[list(objective_columns)].to_numpy(dtype=float)
    physical_g = ranked[list(physical_constraint_columns)].to_numpy(dtype=float)
    normalized_g = ranked[list(normalized_constraint_columns)].to_numpy(
        dtype=float
    )
    constraint_feasible = np.all(physical_g <= 0.0, axis=1)
    if "physical_feasible" in ranked.columns:
        declared = ranked["physical_feasible"]
        if not pd.api.types.is_bool_dtype(declared):
            raise ParetoContractError(
                "physical_feasible must be a canonical boolean column"
            )
        feasible = declared.to_numpy(dtype=bool)
        if np.any(feasible & ~constraint_feasible):
            raise ParetoContractError(
                "physical_feasible contradicts positive physical constraints"
            )
    else:
        feasible = constraint_feasible
    violation = np.maximum(normalized_g, 0.0).sum(axis=1)

    objective_rank = nondominated_ranks_2d(objectives)
    ranked["objective_rank_all"] = objective_rank
    ranked["objective_crowding"] = crowding_distance(
        objectives, objective_rank
    )
    ranked["hard_feasible"] = feasible
    ranked["normalized_constraint_violation"] = violation
    feasible_rank = np.full(len(ranked), -1, dtype=np.int64)
    feasible_crowding = np.full(len(ranked), np.nan, dtype=float)
    if np.any(feasible):
        local_objectives = objectives[feasible]
        local_rank = nondominated_ranks_2d(local_objectives)
        local_crowding = crowding_distance(local_objectives, local_rank)
        feasible_rank[feasible] = local_rank
        feasible_crowding[feasible] = local_crowding
    ranked["feasible_rank"] = feasible_rank
    ranked["feasible_crowding"] = feasible_crowding
    return ranked.sort_values(
        [
            "hard_feasible",
            "feasible_rank",
            "normalized_constraint_violation",
            "objective_rank_all",
            *objective_columns,
            "physical_geometry_sha256",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_standard_candidates(
    ranked: pd.DataFrame,
    *,
    objective_columns: Sequence[str],
    normalized_constraint_columns: Sequence[str],
    limit: int = 12,
) -> pd.DataFrame:
    """Select deterministic anchors plus objective-space spread points."""

    if limit <= 0:
        raise ParetoContractError("standard candidate limit must be positive")
    feasible_front = ranked[
        ranked["hard_feasible"] & (ranked["feasible_rank"] == 0)
    ].copy()
    if not feasible_front.empty:
        pool = feasible_front
        selection_basis = "robust_feasible_front0"
    else:
        pool = ranked.sort_values(
            [
                "normalized_constraint_violation",
                "objective_rank_all",
                *objective_columns,
                "physical_geometry_sha256",
            ],
            kind="stable",
        ).head(max(256, limit))
        selection_basis = "near_feasible_fallback"
    pool = pool.reset_index(drop=True)
    objectives = pool[list(objective_columns)].to_numpy(dtype=float)
    low = objectives.min(axis=0)
    span = objectives.max(axis=0) - low
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (objectives - low) / safe_span
    normalized[:, span <= 0.0] = 0.0

    selected: list[int] = []
    roles: dict[int, list[str]] = {}

    def add(index: int, role: str) -> None:
        index = int(index)
        roles.setdefault(index, []).append(role)
        if index not in selected:
            selected.append(index)

    minimum_volume_index = int(
        np.lexsort(
            (
                pool["physical_geometry_sha256"].to_numpy(),
                objectives[:, 1],
                objectives[:, 0],
            )
        )[0]
    )
    minimum_loss_index = int(
        np.lexsort(
            (
                pool["physical_geometry_sha256"].to_numpy(),
                objectives[:, 0],
                objectives[:, 1],
            )
        )[0]
    )
    add(minimum_volume_index, "minimum_volume")
    add(minimum_loss_index, "minimum_total_loss")

    end_a = normalized[minimum_volume_index]
    end_b = normalized[minimum_loss_index]
    chord = end_b - end_a
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm > 0.0:
        relative = normalized - end_a
        distance = np.abs(
            relative[:, 0] * chord[1] - relative[:, 1] * chord[0]
        ) / chord_norm
        knee_index = int(
            sorted(
                range(len(pool)),
                key=lambda index: (
                    -float(distance[index]),
                    float(np.linalg.norm(normalized[index])),
                    str(pool.iloc[index]["physical_geometry_sha256"]),
                ),
            )[0]
        )
    else:
        knee_index = int(
            sorted(
                range(len(pool)),
                key=lambda index: (
                    float(np.linalg.norm(normalized[index])),
                    str(pool.iloc[index]["physical_geometry_sha256"]),
                ),
            )[0]
        )
    add(knee_index, "pareto_knee")

    normalized_g = pool[list(normalized_constraint_columns)].to_numpy(
        dtype=float
    )
    minimum_margin = np.min(-normalized_g, axis=1)
    margin_index = int(
        sorted(
            range(len(pool)),
            key=lambda index: (
                -float(minimum_margin[index]),
                float(np.linalg.norm(normalized[index])),
                str(pool.iloc[index]["physical_geometry_sha256"]),
            ),
        )[0]
    )
    add(margin_index, "maximum_minimum_constraint_margin")

    target_count = min(limit, len(pool))
    while len(selected) < target_count:
        selected_objectives = normalized[np.asarray(selected, dtype=int)]
        remaining = [index for index in range(len(pool)) if index not in selected]
        best = sorted(
            remaining,
            key=lambda index: (
                -float(
                    np.linalg.norm(
                        selected_objectives - normalized[index], axis=1
                    ).min()
                ),
                -float(
                    pool.iloc[index]["feasible_crowding"]
                    if np.isfinite(pool.iloc[index]["feasible_crowding"])
                    else math.inf
                ),
                str(pool.iloc[index]["physical_geometry_sha256"]),
            ),
        )[0]
        add(best, "objective_space_maximin")

    selection = pool.iloc[selected].copy()
    selection["standard_selection_order"] = np.arange(1, len(selection) + 1)
    selection["standard_selection_roles"] = [
        ",".join(roles[index]) for index in selected
    ]
    selection["standard_selection_basis"] = selection_basis
    return selection.reset_index(drop=True)


def _write_table(frame: pd.DataFrame, stem: Path) -> dict[str, Any]:
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    return {
        "csv": {
            "path": csv_path.name,
            "sha256": _sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        },
        "parquet": {
            "path": parquet_path.name,
            "sha256": _sha256_file(parquet_path),
            "size_bytes": parquet_path.stat().st_size,
        },
        "row_count": int(len(frame)),
    }


def build_global_pareto(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    terminal = load_terminal_population(manifest)
    comparison_columns = (
        manifest["objective_columns"]
        + manifest["physical_constraint_columns"]
        + manifest["normalized_constraint_columns"]
    )
    unique = deduplicate_physical_candidates(
        terminal,
        comparison_columns=comparison_columns,
    )
    ranked = rank_candidates(
        unique,
        objective_columns=manifest["objective_columns"],
        physical_constraint_columns=manifest["physical_constraint_columns"],
        normalized_constraint_columns=manifest["normalized_constraint_columns"],
    )

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    feasible_front = ranked[
        ranked["hard_feasible"] & (ranked["feasible_rank"] == 0)
    ].copy()
    objective_front = ranked[ranked["objective_rank_all"] == 0].copy()
    near_feasible = ranked[~ranked["hard_feasible"]].sort_values(
        [
            "normalized_constraint_violation",
            "objective_rank_all",
            *manifest["objective_columns"],
            "physical_geometry_sha256",
        ],
        kind="stable",
    ).head(256)
    standard_candidates = select_standard_candidates(
        ranked,
        objective_columns=manifest["objective_columns"],
        normalized_constraint_columns=manifest[
            "normalized_constraint_columns"
        ],
    )

    artifacts = {
        "ranked_unique_candidates": _write_table(
            ranked, output / "ranked_unique_candidates"
        ),
        "objective_front0": _write_table(
            objective_front, output / "objective_front0"
        ),
        "feasible_front0": _write_table(
            feasible_front, output / "feasible_front0"
        ),
        "near_feasible": _write_table(
            near_feasible, output / "near_feasible"
        ),
        "standard_candidates": _write_table(
            standard_candidates, output / "standard_candidates"
        ),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "input_manifest": {
            "path": str(manifest["path"]),
            "sha256": _sha256_file(manifest["path"]),
        },
        "identities": manifest["identities"],
        "objective_columns": manifest["objective_columns"],
        "physical_constraint_columns": manifest[
            "physical_constraint_columns"
        ],
        "normalized_constraint_columns": manifest[
            "normalized_constraint_columns"
        ],
        "authenticated_seed_count": len(manifest["records"]),
        "terminal_row_count": int(len(terminal)),
        "unique_physical_candidate_count": int(len(ranked)),
        "duplicate_occurrence_count": int(len(terminal) - len(ranked)),
        "hard_feasible_count": int(ranked["hard_feasible"].sum()),
        "objective_front0_count": int(len(objective_front)),
        "feasible_front0_count": int(len(feasible_front)),
        "standard_candidate_count": int(len(standard_candidates)),
        "global_sort_scope": "all_terminal_population_rows_after_physical_hash_dedupe",
        "artifacts": artifacts,
    }
    summary["content_sha256"] = _sha256_bytes(_canonical_json_bytes(summary))
    summary_path = output / "summary.json"
    summary_path.write_bytes(_canonical_json_bytes(summary) + b"\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an authenticated global two-objective MFT Pareto front"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_global_pareto(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
