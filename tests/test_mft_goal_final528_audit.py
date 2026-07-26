from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools import mft_goal_final528_audit as audit


def _raw_row(
    task_id: int,
    seed: int,
    index: int,
    geometry: str,
    *,
    violation: float,
    volume: float,
    loss: float,
) -> dict[str, object]:
    return {
        "terminal_population_index": index,
        "physical_geometry_sha256": geometry,
        "objective_volume_L": volume,
        "objective_total_loss_W": loss,
        "decoded_physical_params_json": json.dumps(
            {"cw1": 5.0, "gap1": 1.6, "N1": 5}
        ),
        "source_seed": seed,
        "scheduler_task_id": task_id,
        "fixed_primary_turns_stratum": 5,
        "normalized_constraint_violation_l2": violation,
        "global_screening_feasible": False,
        "global_production_eligible": False,
    }


def test_raw_audit_recomputes_exact_geometry_representative(
    tmp_path: Path,
) -> None:
    shared = "a" * 64
    rows = [
        _raw_row(
            10,
            100,
            0,
            shared,
            violation=3.0,
            volume=100.0,
            loss=900.0,
        ),
        _raw_row(
            10,
            100,
            1,
            "b" * 64,
            violation=2.0,
            volume=120.0,
            loss=800.0,
        ),
        _raw_row(
            11,
            101,
            0,
            shared,
            violation=1.0,
            volume=110.0,
            loss=850.0,
        ),
        _raw_row(
            11,
            101,
            1,
            "c" * 64,
            violation=4.0,
            volume=90.0,
            loss=950.0,
        ),
    ]
    path = tmp_path / "raw.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coverage, representatives = audit._raw_coverage_and_representatives(
        path,
        collection_seed_by_task={10: 100, 11: 101},
        terminal_rows_per_seed=2,
    )

    assert coverage == {
        "row_count": 4,
        "logical_seed_count": 2,
        "selected_task_count": 2,
        "unique_geometry_count": 3,
        "cw1_unique_mm": [5.0],
        "gap1_unique_mm": [1.6],
        "N1_strata": [5],
    }
    assert representatives[shared]["scheduler_task_id"] == 11
    assert representatives[shared]["normalized_constraint_violation_l2"] == 1.0


def test_raw_audit_fails_closed_on_primary_thickness_drift(
    tmp_path: Path,
) -> None:
    row = _raw_row(
        10,
        100,
        0,
        "a" * 64,
        violation=1.0,
        volume=100.0,
        loss=900.0,
    )
    row["decoded_physical_params_json"] = json.dumps(
        {"cw1": 4.9, "gap1": 1.6, "N1": 5}
    )
    path = tmp_path / "raw.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(audit.AuditError, match="fixed 5T"):
        audit._raw_coverage_and_representatives(
            path,
            collection_seed_by_task={10: 100},
            terminal_rows_per_seed=1,
        )


def test_non_dominated_is_exact_and_deterministic() -> None:
    rows = [
        {
            "physical_geometry_sha256": "a" * 64,
            "objective_volume_L": 10.0,
            "objective_total_loss_W": 10.0,
        },
        {
            "physical_geometry_sha256": "b" * 64,
            "objective_volume_L": 11.0,
            "objective_total_loss_W": 11.0,
        },
        {
            "physical_geometry_sha256": "c" * 64,
            "objective_volume_L": 9.0,
            "objective_total_loss_W": 12.0,
        },
        {
            "physical_geometry_sha256": "d" * 64,
            "objective_volume_L": 12.0,
            "objective_total_loss_W": 9.0,
        },
    ]

    front = audit.non_dominated(
        rows, ("objective_volume_L", "objective_total_loss_W")
    )

    assert audit._front_hashes(front) == ["c" * 64, "a" * 64, "d" * 64]


def test_html_labels_diagnostic_fronts_as_nonproduction() -> None:
    row = {
        "physical_geometry_sha256": "a" * 64,
        "objective_volume_L": 100.0,
        "objective_total_loss_W": 900.0,
        "normalized_constraint_violation_l2": 1.0,
        "fixed_primary_turns_stratum": 5,
        "exterior_W_drawing_x_mm": 1190.0,
        "exterior_L_perpendicular_y_mm": 990.0,
        "exterior_H_mm": 740.0,
        "resonance_min_Hz_fixed_lm2mh": 14900.0,
        "primary_winding_robust_max_C_screening": 101.0,
        "secondary_winding_robust_max_C_screening": 119.0,
        "core_robust_max_C_screening": 118.0,
    }

    rendered = audit._render_html(
        audit_core_sha256="1" * 64,
        status_sha256="2" * 64,
        manifest_sha256="3" * 64,
        candidates=[row],
        production_front=[],
        minimum_violation_front=[row],
        objective_only_front=[row],
        acquisition=[row],
        generated_at="2026-07-27T07:00:00+09:00",
    )

    assert "생산 판정용 Front는 비어 있습니다" in rendered
    assert "진단용" in rendered
    assert "feasible 또는 production Pareto로 승격할 수 없습니다" in rendered
