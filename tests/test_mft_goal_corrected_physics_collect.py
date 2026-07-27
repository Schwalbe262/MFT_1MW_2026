from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from tools import mft_goal_corrected_physics_collect as collect
from tools import mft_goal_corrected_physics_nsga_lane as lane
from tools import tier1_corrected_generation_preflight as preflight


def _digest(number: int) -> str:
    return f"{number:064x}"


@pytest.fixture
def identities() -> dict[str, str]:
    return {
        "dataset_sha256": _digest(101),
        "evaluation_model_sha256": _digest(102),
        "constraint_spec_sha256": _digest(103),
        "cooling_contract_sha256": (
            preflight.GOAL_FIXED_COOLING_IDENTITY_SHA256
        ),
        "operating_point_sha256": (
            preflight.GOAL_FIXED_OPERATING_IDENTITY_SHA256
        ),
    }


def _task(
    seed: int,
    task_id: int,
    identities: dict[str, str],
) -> dict[str, Any]:
    temperature = _digest(106)
    hard = _digest(107)
    profile = {
        "payload_sha256": _digest(108),
        "geometry_constraint_profile_sha256": _digest(109),
        "authorized_seed_count": lane.SEED_COUNT,
        "physics_delta_rx_resonance_gate": {
            "model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
        },
    }
    return {
        "seed": seed,
        "campaign_id": lane.CAMPAIGN_ID,
        "payload_sha256": _digest(1_000_000 + seed),
        "stage_spec_sha256": identities["constraint_spec_sha256"],
        "temperature_contract_sha256": temperature,
        "hard_constraint_contract_sha256": hard,
        "manufacturing_search_profile": profile,
        "activation": {
            "manufacturing_search_profile": profile,
            "source_identity": {
                "dataset_sha256": identities["dataset_sha256"],
                "evaluation_model_sha256": identities[
                    "evaluation_model_sha256"
                ],
            },
            "fixed_lm2mh_resonance_contract_sha256": _digest(110),
        },
        "task_id": task_id,
    }


def _terminal_frame(
    *,
    seed: int,
    task_id: int,
    task: dict[str, Any],
    identities: dict[str, str],
    hash_offset: int = 0,
) -> pd.DataFrame:
    count = collect.TERMINAL_POPULATION_COUNT
    volume = 600.0 + np.arange(count, dtype=float) / 10.0
    loss = 12_000.0 - np.arange(count, dtype=float)
    resonance_g = np.full(count, -1000.0)
    width_g = np.full(count, -20.0)
    geometry = [
        _digest(hash_offset + index + 1) for index in range(count)
    ]
    return pd.DataFrame(
        {
            "terminal_population_index": np.arange(count),
            "decoder_valid": True,
            "surrogate_physical_valid": True,
            "surrogate_physicality_passed": True,
            "physical_geometry_sha256": geometry,
            "canonical_physical_params_sha256": [
                _digest(hash_offset + 50_000 + index)
                for index in range(count)
            ],
            "candidate_physics_sha": geometry,
            "objective_volume_L": volume,
            "objective_total_loss_W": loss,
            "physical_constraint_feasible": True,
            "physical_feasible": True,
            "physical_G_json": "{}",
            "normalized_G_json": "{}",
            "coordinate_unit_json": "[]",
            "decoded_physical_params_json": "{}",
            "source_seed": seed,
            "source_task_id": task_id,
            "source_bundle_id": task["payload_sha256"],
            "source_island_id": "diagnostic-n1-6-compact",
            **{
                name: value
                for name, value in identities.items()
            },
            "evaluation_model_artifacts_sha256": identities[
                "evaluation_model_sha256"
            ],
            "evaluation_model_generation_sha256": _digest(111),
            "evaluation_spec_sha256": identities[
                "constraint_spec_sha256"
            ],
            "evaluation_temperature_contract_sha256": task[
                "temperature_contract_sha256"
            ],
            "evaluation_hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
            collect.CORRECTED_PHYSICAL_COLUMN: resonance_g,
            "physical_G:exterior_width_limit": width_g,
            collect.CORRECTED_NORMALIZED_COLUMN: resonance_g / 150.0,
            "normalized_G:exterior_width_limit": width_g / 10.0,
        }
    )


def _write_table_fixture(
    root: Path,
    *,
    seed: int,
    task_id: int,
    identities: dict[str, str],
    hash_offset: int = 0,
) -> tuple[dict[str, Any], Path, Path, dict[str, dict[str, Any]]]:
    task = _task(seed, task_id, identities)
    frame = _terminal_frame(
        seed=seed,
        task_id=task_id,
        task=task,
        identities=identities,
        hash_offset=hash_offset,
    )
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "terminal_physical_candidates.csv"
    frame.to_csv(csv_path, index=False)
    csv_sha = collect.common._sha256_file(csv_path)
    manifest = {
        "schema_version": preflight.GOAL_TERMINAL_TABLE_SCHEMA,
        "row_count": len(frame),
        "terminal_population_index_min": 0,
        "terminal_population_index_max": len(frame) - 1,
        "columns": list(frame.columns),
        "csv": {
            "path": csv_path.name,
            "sha256": csv_sha,
            "size_bytes": csv_path.stat().st_size,
        },
        "source_identity": {
            "seed": seed,
            "task_id": str(task_id),
            "bundle_id": task["payload_sha256"],
            "island_id": "diagnostic-n1-6-compact",
            "dataset_sha256": identities["dataset_sha256"],
            "model_artifacts_sha256": identities[
                "evaluation_model_sha256"
            ],
            "temperature_contract_sha256": task[
                "temperature_contract_sha256"
            ],
            "hard_constraint_contract_sha256": task[
                "hard_constraint_contract_sha256"
            ],
        },
        "temperature_contract_sha256": task[
            "temperature_contract_sha256"
        ],
        "hard_constraint_contract_sha256": task[
            "hard_constraint_contract_sha256"
        ],
        "resonance_contract_schema": lane.RESONANCE_SCHEMA,
        "fixed_lm2mh_resonance_contract_sha256": task["activation"][
            "fixed_lm2mh_resonance_contract_sha256"
        ],
        "one_row_per_terminal_individual": True,
        "physical_deduplication_key": "physical_geometry_sha256",
        "global_pareto_provenance_ready": True,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    manifest_path = root / "terminal_physical_candidates.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    selected = {
        "terminal_physical_candidates": {
            "path": csv_path.name,
            "sha256": csv_sha,
            "size_bytes": csv_path.stat().st_size,
        },
        "terminal_physical_candidates_manifest": {
            "path": manifest_path.name,
            "sha256": collect.common._sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    return task, csv_path, manifest_path, selected


def _collection_record(
    *,
    root: Path,
    seed: int,
    task_id: int,
    identities: dict[str, str],
    hash_offset: int,
) -> dict[str, Any]:
    task, csv_path, _manifest_path, _selected = _write_table_fixture(
        root / f"task-{task_id}",
        seed=seed,
        task_id=task_id,
        identities=identities,
        hash_offset=hash_offset,
    )
    value = {
        "schema_version": collect.COLLECTION_RECORD_SCHEMA,
        "campaign_id": lane.CAMPAIGN_ID,
        "task_id": task_id,
        "seed": seed,
        "task_payload_sha256": task["payload_sha256"],
        "terminal_population_count": collect.TERMINAL_POPULATION_COUNT,
        "identities": identities,
        "objective_columns": list(collect.OBJECTIVE_COLUMNS),
        "physical_constraint_columns": [
            collect.CORRECTED_PHYSICAL_COLUMN,
            "physical_G:exterior_width_limit",
        ],
        "normalized_constraint_columns": [
            collect.CORRECTED_NORMALIZED_COLUMN,
            "normalized_G:exterior_width_limit",
        ],
        "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
        "physics_delta_model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
        "corrected_resonance_constraint_name": lane.PHYSICS_CONSTRAINT_NAME,
        "artifacts": {
            "terminal_physical_candidates.csv": {
                "path": csv_path.name,
                "sha256": collect.common._sha256_file(csv_path),
                "size_bytes": csv_path.stat().st_size,
            }
        },
        **collect.FAIL_CLOSED_FLAGS,
    }
    value["payload_sha256"] = canonical_sha256(value)
    return value


def test_exact60_plan_and_receipt_fixture_authenticates_read_only(
    tmp_path: Path,
    identities: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "offload_plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    receipt_path = tmp_path / "submission_receipt.json"
    tasks = [
        _task(seed, 97_000 + index, identities)
        for index, seed in enumerate(
            range(lane.SEED_START, lane.SEED_END + 1)
        )
    ]
    rows = [
        {
            "seed": task["seed"],
            "task_id": task["task_id"],
            "dedupe_key": f"{lane.DEDUPE_PREFIX}{task['seed']}",
        }
        for task in tasks
    ]
    receipt = {"tasks": rows, "remote_ready": {"runtime_verified": True}}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    plan = {"task_count": lane.SEED_COUNT, "bundle_id": "fixture"}
    monkeypatch.setattr(
        collect,
        "_configure_from_plan",
        lambda _path: {"payload_sha256": lane.MODEL_PAYLOAD_SHA256},
    )
    monkeypatch.setattr(
        collect.lane.offload,
        "authenticate_plan",
        lambda _path: (plan, {}, tasks, {"fixture": True}),
    )
    monkeypatch.setattr(
        collect.lane.offload,
        "_authorized_plan_seeds",
        lambda _plan: tuple(range(lane.SEED_START, lane.SEED_END + 1)),
    )
    monkeypatch.setattr(
        collect.lane,
        "_validate_search_profile",
        lambda value: value,
    )
    monkeypatch.setattr(
        collect.lane.offload,
        "scheduler_payload",
        lambda *, plan, task, priority: {
            "name": f"{lane.TASK_NAME_PREFIX}{task['seed']}",
            "dedupe_key": f"{lane.DEDUPE_PREFIX}{task['seed']}",
            "payload_json": {"seed": task["seed"]},
        },
    )
    monkeypatch.setattr(
        collect.lane.offload,
        "_validate_receipt",
        lambda value, **_kwargs: value,
    )

    context = collect.authenticate_context(
        plan_path=plan_path,
        receipt_path=receipt_path,
        scheduler_url="http://127.0.0.1:8002",
    )

    assert len(context["entries"]) == 60
    assert context["authorized_seeds"] == list(
        range(lane.SEED_START, lane.SEED_END + 1)
    )
    assert context["physics_delta_model_payload_sha256"] == (
        lane.MODEL_PAYLOAD_SHA256
    )


def test_corrected_terminal_table_requires_all_320_rows_and_physics_G(
    tmp_path: Path,
    identities: dict[str, str],
) -> None:
    seed = lane.SEED_START
    task_id = 97_500
    task, csv_path, manifest_path, selected = _write_table_fixture(
        tmp_path,
        seed=seed,
        task_id=task_id,
        identities=identities,
    )
    evidence = collect._validate_terminal_table(
        csv_path=csv_path,
        manifest_path=manifest_path,
        entry={"seed": seed, "task_id": task_id, "task": task},
        selected=selected,
    )
    assert evidence["corrected_resonance_constraint_name"] == (
        lane.PHYSICS_CONSTRAINT_NAME
    )
    assert collect.CORRECTED_PHYSICAL_COLUMN in evidence[
        "physical_constraint_columns"
    ]

    frame = pd.read_csv(csv_path).iloc[:-1]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        collect._validate_terminal_table(
            csv_path=csv_path,
            manifest_path=manifest_path,
            entry={"seed": seed, "task_id": task_id, "task": task},
            selected=selected,
        )


def test_corrected_result_rejects_legacy_capacitance_authority(
    tmp_path: Path,
    identities: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = lane.SEED_START
    task = _task(seed, 97_501, identities)
    profile = task["activation"]["manufacturing_search_profile"]
    monkeypatch.setattr(
        collect.lane,
        "_validate_search_profile",
        lambda _value: profile,
    )
    inventory = {
        "terminal_physical_candidates": {
            "path": "terminal_physical_candidates.csv",
            "sha256": _digest(201),
            "size_bytes": 100,
        },
        "terminal_physical_candidates_manifest": {
            "path": "terminal_physical_candidates.manifest.json",
            "sha256": _digest(202),
            "size_bytes": 100,
        },
    }
    result = {
        "schema_version": lane.RESULT_SCHEMA,
        "campaign_id": lane.CAMPAIGN_ID,
        "task_payload_sha256": task["payload_sha256"],
        "seed": seed,
        "fixed_primary_turns": 6,
        "population": collect.TERMINAL_POPULATION_COUNT,
        "generations": collect.scout.GENERATIONS,
        "terminal_population_count": collect.TERMINAL_POPULATION_COUNT,
        "manufacturing_search_profile_payload_sha256": profile[
            "payload_sha256"
        ],
        "geometry_constraint_profile_sha256": profile[
            "geometry_constraint_profile_sha256"
        ],
        "fixed_lm2mh_resonance_contract_sha256": task["activation"][
            "fixed_lm2mh_resonance_contract_sha256"
        ],
        "physics_delta_Crx_q90_ucb_gate_active": True,
        "physics_delta_fRx_q90_lcb_gate_active": True,
        "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
        "physics_delta_model_payload_sha256": lane.MODEL_PAYLOAD_SHA256,
        "raw_two_net_C_optimizer_objective_constraint_authority": False,
        "raw_two_net_C_terminal_eligibility_authority": False,
        "raw_two_net_C_physical_feasibility_authority": False,
        "single_0p759701_transfer_ratio_used": False,
        "legacy_half_magnetizing_resonance_G_present": False,
        "feature_extrapolation_penalty_active": True,
        "original_split_early_hard_rejection_allowed": False,
        "selected_split_Llt_G_remains_hard": True,
        "terminal_selected_and_neighbor_splits_in_decoded_params": True,
        "fixed20T_turn_graded_FEA_retraining_required": True,
        "approved_dielectric_stack_sensitivity_required": True,
        "final_turn_graded_symmetric_FEA_required": True,
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": canonical_sha256(inventory),
        "screening_only": True,
        "production_eligible": False,
        "final_design_claim_allowed": False,
        "fresh512_activation_evidence": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "symmetric_FEA_validation_still_required": True,
        "scheduler_write_performed": False,
        "scheduler_submission_performed": False,
    }
    result["payload_sha256"] = canonical_sha256(result)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    validated, selected = collect._validate_result(
        path,
        entry={"seed": seed, "task_id": 97_501, "task": task},
    )
    assert validated["physics_delta_fRx_q90_lcb_gate_active"] is True
    assert set(selected) == {
        "terminal_physical_candidates",
        "terminal_physical_candidates_manifest",
    }

    legacy = copy.deepcopy(result)
    legacy.pop("payload_sha256")
    legacy["authenticated_turn_graded_transfer_ratio"] = 0.759701
    legacy["payload_sha256"] = canonical_sha256(legacy)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        collect._validate_result(
            path,
            entry={"seed": seed, "task_id": 97_501, "task": task},
        )


def test_global_adapter_dedupes_full_rows_and_emits_html_csv_json(
    tmp_path: Path,
    identities: dict[str, str],
) -> None:
    first = _collection_record(
        root=tmp_path,
        seed=lane.SEED_START,
        task_id=97_600,
        identities=identities,
        hash_offset=0,
    )
    second = _collection_record(
        root=tmp_path,
        seed=lane.SEED_START + 1,
        task_id=97_601,
        identities=identities,
        hash_offset=0,
    )
    result = collect.publish_screening_nds(
        records=[first, second],
        output_root=tmp_path,
        snapshot_class="provisional_first_threshold",
        expected_seeds=range(lane.SEED_START, lane.SEED_END + 1),
    )
    nds = Path(result["nds_output"])
    summary = json.loads((nds / "summary.json").read_text(encoding="utf-8"))

    assert summary["terminal_row_count"] == 640
    assert summary["unique_physical_candidate_count"] == 320
    assert summary["duplicate_occurrence_count"] == 320
    assert summary["global_sort_scope"] == (
        "all_terminal_population_rows_after_physical_hash_dedupe"
    )
    assert (nds / "ranked_unique_candidates.csv").is_file()
    assert (nds / "feasible_front0.json").is_file()
    assert (nds / "global-pareto-audit.html").is_file()
    report = (nds / "global-pareto-audit.html").read_text(encoding="utf-8")
    assert "Seed-local Pareto front" in report
    authority = json.loads(
        (nds / "screening_authority.json").read_text(encoding="utf-8")
    )
    assert authority["seed_local_front_union_used"] is False


def test_final_manifest_requires_every_exact60_seed(
    tmp_path: Path,
    identities: dict[str, str],
) -> None:
    records = []
    for index, seed in enumerate(
        range(lane.SEED_START, lane.SEED_END + 1)
    ):
        task_id = 98_000 + index
        task_dir = tmp_path / f"task-{task_id}"
        task_dir.mkdir()
        table = task_dir / "terminal_physical_candidates.csv"
        table.write_text("fixture\n", encoding="utf-8")
        value = {
            "schema_version": collect.COLLECTION_RECORD_SCHEMA,
            "campaign_id": lane.CAMPAIGN_ID,
            "task_id": task_id,
            "seed": seed,
            "task_payload_sha256": _digest(500_000 + index),
            "terminal_population_count": collect.TERMINAL_POPULATION_COUNT,
            "identities": identities,
            "objective_columns": list(collect.OBJECTIVE_COLUMNS),
            "physical_constraint_columns": [
                collect.CORRECTED_PHYSICAL_COLUMN
            ],
            "normalized_constraint_columns": [
                collect.CORRECTED_NORMALIZED_COLUMN
            ],
            "physics_delta_model_file_sha256": lane.MODEL_FILE_SHA256,
            "physics_delta_model_payload_sha256": (
                lane.MODEL_PAYLOAD_SHA256
            ),
            "corrected_resonance_constraint_name": (
                lane.PHYSICS_CONSTRAINT_NAME
            ),
            "artifacts": {
                "terminal_physical_candidates.csv": {
                    "sha256": collect.common._sha256_file(table),
                    "size_bytes": table.stat().st_size,
                }
            },
            **collect.FAIL_CLOSED_FLAGS,
        }
        value["payload_sha256"] = canonical_sha256(value)
        records.append(value)

    with pytest.raises(RuntimeError, match="exact60 seed coverage"):
        collect.terminal_population_manifest(
            records=records[:-1],
            output_root=tmp_path,
            snapshot_class="final_integrated_exact60",
        )
    manifest = collect.terminal_population_manifest(
        records=records,
        output_root=tmp_path,
        snapshot_class="final_integrated_exact60",
    )
    assert manifest["authenticated_seed_count"] == 60
    assert manifest["integrated_seed_scope_complete"] is True
    assert manifest["seed_local_front_union_used"] is False
