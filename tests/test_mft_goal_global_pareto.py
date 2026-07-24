from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.mft_goal_global_pareto import (
    MANIFEST_SCHEMA,
    ParetoContractError,
    build_global_pareto,
    nondominated_ranks_2d,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(number: int) -> str:
    return f"{number:064x}"


def _brute_ranks(values: np.ndarray) -> np.ndarray:
    count = len(values)
    dominates = [
        [
            j
            for j in range(count)
            if j != i
            and np.all(values[j] <= values[i])
            and np.any(values[j] < values[i])
        ]
        for i in range(count)
    ]
    ranks = np.full(count, -1, dtype=int)
    unresolved = set(range(count))
    rank = 0
    while unresolved:
        front = [
            i
            for i in unresolved
            if not any(j in unresolved for j in dominates[i])
        ]
        assert front
        ranks[front] = rank
        unresolved.difference_update(front)
        rank += 1
    return ranks


def test_nondominated_ranks_matches_brute_force_with_ties() -> None:
    rng = np.random.default_rng(260726)
    values = rng.integers(0, 12, size=(500, 2)).astype(float)
    assert np.array_equal(
        nondominated_ranks_2d(values),
        _brute_ranks(values),
    )


def _write_seed(path: Path, seed: int, *, duplicate_hash: bool = False) -> dict:
    count = 320
    frame = pd.DataFrame(
        {
            "terminal_population_index": np.arange(count),
            "physical_geometry_sha256": [
                _hash(index + (0 if duplicate_hash else seed * 1000) + 1)
                for index in range(count)
            ],
            "exterior_volume_mm3": np.arange(count, dtype=float) + seed,
            "total_loss_W": np.arange(count, 0, -1, dtype=float) + seed,
            "g_box": np.full(count, -1.0),
            "g_temp": np.full(count, -2.0),
            "gn_box": np.full(count, -0.1),
            "gn_temp": np.full(count, -0.2),
        }
    )
    table = path / f"seed-{seed}.csv"
    frame.to_csv(table, index=False)
    identities = {
        "dataset_sha256": _hash(101),
        "evaluation_model_sha256": _hash(102),
        "constraint_spec_sha256": _hash(103),
        "cooling_contract_sha256": _hash(104),
        "operating_point_sha256": _hash(105),
    }
    return {
        "seed": seed,
        "task_id": seed + 90000,
        "bundle_id": f"bundle-{seed}",
        "terminal_authenticated": True,
        "terminal_population_count": count,
        "table_path": table.name,
        "table_sha256": _sha256(table),
        "source_model_sha256": _hash(102),
        "identities": identities,
    }


def _manifest(path: Path, records: list[dict]) -> Path:
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "identities": records[0]["identities"],
        "objective_columns": ["exterior_volume_mm3", "total_loss_W"],
        "physical_constraint_columns": ["g_box", "g_temp"],
        "normalized_constraint_columns": ["gn_box", "gn_temp"],
        "records": records,
    }
    target = path / "manifest.json"
    target.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )
    return target


def test_global_pareto_uses_every_terminal_row_and_dedupes(tmp_path: Path) -> None:
    first = _write_seed(tmp_path, 1)
    second = _write_seed(tmp_path, 2)
    manifest = _manifest(tmp_path, [first, second])
    output = tmp_path / "output"

    summary = build_global_pareto(
        manifest_path=manifest,
        output_dir=output,
    )

    assert summary["authenticated_seed_count"] == 2
    assert summary["terminal_row_count"] == 640
    assert summary["unique_physical_candidate_count"] == 640
    assert summary["hard_feasible_count"] == 640
    ranked = pd.read_parquet(output / "ranked_unique_candidates.parquet")
    assert set(ranked["source_seed"]) == {1, 2}
    assert ranked["feasible_rank"].min() == 0
    assert (output / "feasible_front0.csv").is_file()
    selected = pd.read_csv(output / "standard_candidates.csv")
    assert len(selected) == 12
    roles = ",".join(selected["standard_selection_roles"])
    assert "minimum_volume" in roles
    assert "minimum_total_loss" in roles
    assert "pareto_knee" in roles
    assert "maximum_minimum_constraint_margin" in roles


def test_duplicate_physics_with_different_evaluation_fails_closed(
    tmp_path: Path,
) -> None:
    first = _write_seed(tmp_path, 1, duplicate_hash=True)
    second = _write_seed(tmp_path, 2, duplicate_hash=True)
    manifest = _manifest(tmp_path, [first, second])

    with pytest.raises(
        ParetoContractError,
        match="inconsistent final evaluation",
    ):
        build_global_pareto(
            manifest_path=manifest,
            output_dir=tmp_path / "output",
        )


def test_mixed_identity_fails_closed(tmp_path: Path) -> None:
    first = _write_seed(tmp_path, 1)
    second = _write_seed(tmp_path, 2)
    second["identities"] = dict(second["identities"])
    second["identities"]["constraint_spec_sha256"] = _hash(999)
    manifest = _manifest(tmp_path, [first, second])

    with pytest.raises(ParetoContractError, match="scientific identity mismatch"):
        build_global_pareto(
            manifest_path=manifest,
            output_dir=tmp_path / "output",
        )
