from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tools import mft_goal_diagnostic_compact_collect as collector
from tools import mft_goal_diagnostic_compact_slurm as offload


def _hash(number: int) -> str:
    return f"{number:064x}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(
    root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    identities = {
        "dataset_sha256": _hash(1),
        "evaluation_model_sha256": _hash(2),
        "constraint_spec_sha256": _hash(3),
        "cooling_contract_sha256": _hash(4),
        "operating_point_sha256": _hash(5),
    }
    physical = [
        "physical_G:box",
        collector.RAW_CRX_PHYSICAL_COLUMN,
    ]
    normalized = [
        "normalized_G:box",
        collector.RAW_CRX_NORMALIZED_COLUMN,
    ]
    records = []
    for ordinal, seed in enumerate(seeds):
        task_id = 100_000 + ordinal
        task_dir = root / f"task-{task_id}"
        task_dir.mkdir(parents=True)
        count = collector.TERMINAL_POPULATION_COUNT
        frame = pd.DataFrame(
            {
                "terminal_population_index": np.arange(count),
                "physical_geometry_sha256": [
                    _hash(10_000 + ordinal * count + index)
                    for index in range(count)
                ],
                "objective_volume_L": (
                    np.arange(count, dtype=float) + ordinal
                ),
                "objective_total_loss_W": (
                    np.arange(count, 0, -1, dtype=float) + ordinal
                ),
                "physical_G:box": np.full(count, -1.0),
                collector.RAW_CRX_PHYSICAL_COLUMN: np.full(count, -0.2),
                "normalized_G:box": np.full(count, -0.1),
                collector.RAW_CRX_NORMALIZED_COLUMN: np.full(count, -0.02),
                "physical_feasible": np.full(count, True),
            }
        )
        table = task_dir / "terminal_physical_candidates.csv"
        frame.to_csv(table, index=False)
        record = collector._sealed(
            {
                "schema_version": collector.COLLECTION_RECORD_SCHEMA,
                "task_id": task_id,
                "seed": seed,
                "task_payload_sha256": _hash(20_000 + ordinal),
                "terminal_population_count": count,
                "identities": identities,
                "objective_columns": list(collector.OBJECTIVE_COLUMNS),
                "physical_constraint_columns": physical,
                "normalized_constraint_columns": normalized,
                "artifacts": {
                    "terminal_physical_candidates.csv": {
                        "sha256": _sha(table),
                        "size_bytes": table.stat().st_size,
                    }
                },
                "raw_same_metric_C_rx_rx_F_UCB_gate_active": True,
                "raw_same_metric_C_rx_rx_F_front_classification": (
                    "provisional_surrogate_screening_only"
                ),
                **collector.FAIL_CLOSED_FLAGS,
            }
        )
        records.append(record)
    return records


def _bounded_records(
    root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    records = _records(root, seeds)
    bounded = []
    for record in records:
        task_id = int(record["task_id"])
        table = root / f"task-{task_id}" / "terminal_physical_candidates.csv"
        frame = pd.read_csv(table).rename(
            columns={
                collector.RAW_CRX_PHYSICAL_COLUMN: (
                    collector.PROVISIONAL_CORRECTED_CRX_PHYSICAL_COLUMN
                ),
                collector.RAW_CRX_NORMALIZED_COLUMN: (
                    collector.PROVISIONAL_CORRECTED_CRX_NORMALIZED_COLUMN
                ),
            }
        )
        frame.to_csv(table, index=False)
        unsigned = {
            key: value
            for key, value in record.items()
            if key != "payload_sha256"
        }
        unsigned["physical_constraint_columns"] = [
            (
                collector.PROVISIONAL_CORRECTED_CRX_PHYSICAL_COLUMN
                if name == collector.RAW_CRX_PHYSICAL_COLUMN
                else name
            )
            for name in unsigned["physical_constraint_columns"]
        ]
        unsigned["normalized_constraint_columns"] = [
            (
                collector.PROVISIONAL_CORRECTED_CRX_NORMALIZED_COLUMN
                if name == collector.RAW_CRX_NORMALIZED_COLUMN
                else name
            )
            for name in unsigned["normalized_constraint_columns"]
        ]
        unsigned["artifacts"]["terminal_physical_candidates.csv"] = {
            "sha256": _sha(table),
            "size_bytes": table.stat().st_size,
        }
        unsigned["raw_same_metric_C_rx_rx_F_UCB_gate_active"] = False
        unsigned[
            "provisional_turn_graded_C_acquisition_gate_active"
        ] = True
        unsigned["capacitance_screening_constraint_name"] = (
            collector.scout
            .PROVISIONAL_TURN_GRADED_C_ACQUISITION_CONSTRAINT_NAME
        )
        unsigned["authenticated_turn_graded_transfer_ratio"] = (
            collector.scout.AUTHENTICATED_TURN_GRADED_TRANSFER_RATIO
        )
        unsigned["raw_two_net_C_physical_feasibility_authority"] = False
        unsigned["final_turn_graded_symmetric_FEA_required"] = True
        unsigned["raw_same_metric_C_rx_rx_F_front_classification"] = (
            "provisional_corrected_single_truth_screening_only"
        )
        bounded.append(collector._sealed(unsigned))
    return bounded


def test_read_only_scheduler_client_exposes_get_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    methods = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"id":123,"status":"completed"}'

    def urlopen(request: Any, timeout: float) -> Response:
        methods.append((request.get_method(), timeout))
        return Response()

    monkeypatch.setattr(collector.urllib.request, "urlopen", urlopen)
    client = collector.ReadOnlySchedulerClient(
        "http://scheduler.invalid", timeout=4
    )
    assert client.get_task(123)["id"] == 123
    assert methods == [("GET", 4.0)]
    assert not hasattr(client, "submit_task")
    assert not hasattr(client, "cancel_task")
    assert not hasattr(client, "preempt_task")


def test_terminal_manifest_threshold_and_exact100_gate(tmp_path: Path) -> None:
    nine = _records(tmp_path, list(offload.EXACT_SEEDS[:9]))
    with pytest.raises(RuntimeError, match="at least ten"):
        collector.terminal_population_manifest(
            records=nine,
            output_root=tmp_path,
            snapshot_class="provisional_first_threshold",
        )

    ten_root = tmp_path / "ten"
    ten = _records(ten_root, list(offload.EXACT_SEEDS[9:19]))
    with pytest.raises(RuntimeError, match="exact100 seed coverage"):
        collector.terminal_population_manifest(
            records=ten,
            output_root=ten_root,
            snapshot_class="final_integrated_exact100",
        )


def test_early_global_nds_pools_all_terminal_rows_and_stays_provisional(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, list(offload.EXACT_SEEDS[:10]))
    published = collector.publish_screening_nds(
        records=records,
        output_root=tmp_path,
        snapshot_class="provisional_first_threshold",
    )
    authority = collector._validate_sealed(
        collector._read_json(Path(published["screening_authority"])),
        schema=collector.SCREENING_AUTHORITY_SCHEMA,
    )
    summary = collector._read_json(
        Path(published["nds_output"]) / "summary.json"
    )
    assert summary["authenticated_seed_count"] == 10
    assert summary["terminal_row_count"] == 10 * 320
    assert authority["front_files_are_provisional_screening_only"] is True
    assert (
        authority["raw_same_metric_C_rx_rx_F_front_classification"]
        == "provisional_surrogate_screening_only"
    )
    assert authority["production_eligible"] is False
    assert authority["final_design_claim_allowed"] is False
    assert authority["scheduler_post_count"] == 0
    assert authority["integrated_seed_scope_complete"] is False


def test_final_integrated_nds_requires_and_pools_exact100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records(tmp_path, list(offload.EXACT_SEEDS))
    observed: dict[str, Any] = {}

    def fake_build(*, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
        manifest = collector._read_json(manifest_path)
        observed["manifest"] = manifest
        output_dir.mkdir(parents=True)
        summary = {
            "authenticated_seed_count": 100,
            "terminal_row_count": 100 * 320,
            "unique_physical_candidate_count": 100 * 320,
            "hard_feasible_count": 100 * 320,
            "objective_front0_count": 100,
            "feasible_front0_count": 100,
            "content_sha256": _hash(99),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary

    monkeypatch.setattr(
        collector.global_pareto, "build_global_pareto", fake_build
    )
    published = collector.publish_screening_nds(
        records=records,
        output_root=tmp_path,
        snapshot_class="final_integrated_exact100",
    )
    manifest = observed["manifest"]
    authority = collector._validate_sealed(
        collector._read_json(Path(published["screening_authority"])),
        schema=collector.SCREENING_AUTHORITY_SCHEMA,
    )
    assert manifest["seeds"] == list(offload.EXACT_SEEDS)
    assert len(manifest["records"]) == 100
    assert manifest["integrated_seed_scope_complete"] is True
    assert manifest["screening_only"] is True
    assert manifest["production_eligible"] is False
    assert authority["integrated_seed_scope_complete"] is True
    assert authority["front_files_are_provisional_screening_only"] is True
    assert authority["final_design_claim_allowed"] is False


def test_bounded_profile_manifest_accepts_dynamic_exact100_seed_authority(
    tmp_path: Path,
) -> None:
    seeds = list(range(2_607_264_300, 2_607_264_400))
    records = _bounded_records(tmp_path, seeds)
    manifest = collector.terminal_population_manifest(
        records=records,
        output_root=tmp_path,
        snapshot_class="final_integrated_exact100",
        expected_seeds=seeds,
    )
    contract = manifest["capacitance_screening_contract"]
    assert manifest["seeds"] == seeds
    assert manifest["raw_same_metric_C_rx_rx_F_UCB_gate_active"] is False
    assert (
        contract["constraint_name"]
        == collector.scout
        .PROVISIONAL_TURN_GRADED_C_ACQUISITION_CONSTRAINT_NAME
    )
    assert (
        contract["authenticated_turn_graded_transfer_ratio"]
        == collector.scout.AUTHENTICATED_TURN_GRADED_TRANSFER_RATIO
    )
    assert contract["physical_feasibility_authority"] is False
    assert contract["final_turn_graded_symmetric_FEA_required"] is True

    with pytest.raises(RuntimeError, match="exact100 seed coverage"):
        collector.terminal_population_manifest(
            records=records,
            output_root=tmp_path,
            snapshot_class="final_integrated_exact100",
        )
