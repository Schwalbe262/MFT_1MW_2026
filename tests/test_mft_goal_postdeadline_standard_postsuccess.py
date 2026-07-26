from __future__ import annotations

import ast
from pathlib import Path

import pytest

from module.mft_goal_20260726_contract import (
    FIXED_COOLING_IDENTITY,
    FIXED_OPERATING_IDENTITY,
    GOAL_TEMPERATURE_TARGETS,
    TEMPERATURE_TARGET_FAMILIES,
)
from tools import mft_goal_postdeadline_standard_collector as collector
from tools import mft_goal_postdeadline_standard_postsuccess as postsuccess
from tools import mft_goal_strict_al_ingest as strict_al


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(collector.canonical_bytes(value) + b"\n")
    return path


def _result(
    *,
    width_mm: float,
    length_mm: float,
    height_mm: float,
    resonance_hz: float,
    winding_c: float,
    core_c: float,
    loss_w: float,
) -> dict:
    l1 = 100.0
    result = {
        **FIXED_OPERATING_IDENTITY,
        **FIXED_COOLING_IDENTITY,
        "l1": l1,
        "l2": (width_mm - 4 * l1) / 2,
        "h1": height_mm - 2 * l1,
        "w1": length_mm,
        "N2_side": 0,
        "sl1_main_x": width_mm,
        "nwl1_main": 0,
        "sl1_main_y": length_mm,
        "nwb1_main_y": 0,
        "f_res_min_tx_rx_only_Hz": resonance_hz,
        "P_winding_total": loss_w,
        "P_core_total": 0.0,
        "P_core_plate_total": 0.0,
        "P_wcp_total": 0.0,
        "physics_data_revision": "physics-v1",
    }
    for target in GOAL_TEMPERATURE_TARGETS:
        if target in {
            "T_max_Rx_side",
            "Tprobe_Rx_side_leeward_max",
        }:
            continue
        result[target] = (
            winding_c
            if TEMPERATURE_TARGET_FAMILIES[target] == "winding"
            else core_c
        )
    return result


def _fake_view(
    tmp_path: Path,
    *,
    task_id: int,
    candidate: str,
    result: dict,
) -> tuple[dict, Path, Path]:
    source_root = tmp_path / f"source-{task_id}"
    receipt_path = _write_json(
        source_root / "collection_receipt.json",
        {"schema_version": collector.COLLECTION_SCHEMA, "task_id": task_id},
    )
    seal_path = _write_json(
        source_root / "collection_seal.json",
        {"schema_version": collector.COLLECTION_SEAL_SCHEMA, "task_id": task_id},
    )
    result_path = _write_json(source_root / "result.json", result)
    collection = postsuccess._sealed(
        {
            "schema_version": postsuccess.AUTHENTICATED_COLLECTION_SCHEMA,
            **postsuccess.SAFETY_FLAGS,
            "stage": "standard",
            "scheduler_status": "completed",
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "task_id": task_id,
            "task_name": f"task-{task_id}",
            "dedupe_key": f"dedupe-{task_id}",
            "candidate_physics_sha256": candidate,
            "source_collection_receipt": postsuccess._file_record(receipt_path),
            "source_collection_receipt_payload_sha256": "1" * 64,
            "source_collection_seal": postsuccess._file_record(seal_path),
            "source_collection_seal_payload_sha256": "2" * 64,
            "retained_symmetric_aedt": postsuccess._file_record(result_path),
            "result": result,
            "result_sha256": collector.payload_sha256(result),
            "result_json": postsuccess._file_record(result_path),
            "goal_physical_spec_reasons": [],
            "goal_physical_spec_passed": True,
            "fixed_identity_attestation": {"sha256": "3" * 64},
            "strict_al_adapter_authorized": True,
        }
    )
    return (
        {
            "schema_version": postsuccess.AUTHENTICATED_COLLECTION_SCHEMA,
            "collection": collection,
            "plan": {
                "solver_revision": "a" * 40,
                "library_revision": "b" * 40,
            },
            "params": {},
            "selected": {
                "task_identity": {
                    "seed": 2_607_260_000 + task_id,
                    "fixed_primary_turns": 6,
                    "payload_sha256": "4" * 64,
                }
            },
            "submission": {},
        },
        receipt_path,
        seal_path,
    )


def _aggregate_reference(tmp_path: Path) -> dict:
    aggregate = _write_json(tmp_path / "aggregate.json", {"aggregate": True})
    return {
        "manifest": postsuccess._file_record(aggregate),
        "manifest_payload_sha256": "5" * 64,
        "seed_count": 512,
        "input_terminal_row_count": 163_840,
        "deduplicated_physical_geometry_count": 133_563,
        "physical_feasible_count": 0,
        "global_pareto_count": 0,
        "global_objective_front_count": 22,
        "sorting_authority": "all_authenticated_terminal_rows",
        "artifacts": {},
        "manifest_and_published_artifact_bytes_authenticated": True,
        "all_512_seed_inputs_reauthenticated_by_this_cycle": False,
        "global_nds_recomputed_by_this_cycle": False,
    }


def test_pending_emits_no_scientific_or_production_claim(tmp_path: Path) -> None:
    state = postsuccess.process_cycle(
        lanes=[
            postsuccess.Lane(96325, tmp_path / "lane-a"),
            postsuccess.Lane(96327, tmp_path / "lane-b"),
        ],
        output_root=tmp_path / "output",
    )

    assert state["status"] == "pending_standard_collections"
    assert state["collection_count"] == 0
    assert state["pending_count"] == 2
    assert state["scientific_pass_claimed"] is False
    assert state["production_claimed"] is False
    assert state["production_pareto_emitted"] is False
    assert state["orchestrator_scheduler_methods_used"] == []
    assert state["scheduler_mutation_performed"] is False


def test_measured_constraints_use_requested_limits(tmp_path: Path) -> None:
    passing, _receipt, _seal = _fake_view(
        tmp_path,
        task_id=96325,
        candidate="a" * 64,
        result=_result(
            width_mm=1200,
            length_mm=1000,
            height_mm=750,
            resonance_hz=15000,
            winding_c=100,
            core_c=120,
            loss_w=100,
        ),
    )
    classified = postsuccess._measured_classification(passing)
    assert classified["measured_hard_constraints_passed"] is True
    assert classified["actual_winding_max_C"] == 100
    assert classified["actual_core_max_C"] == 120

    failing = dict(passing)
    failing["collection"] = dict(passing["collection"])
    failing["collection"]["result"] = _result(
        width_mm=1200.01,
        length_mm=1000,
        height_mm=750,
        resonance_hz=14999.99,
        winding_c=100.01,
        core_c=120.01,
        loss_w=100,
    )
    classified = postsuccess._measured_classification(failing)
    assert classified["measured_hard_constraints_passed"] is False
    assert classified["hard_constraint_evidence"]["width_mm"]["passed"] is False
    assert (
        classified["hard_constraint_evidence"]["resonance_Hz"]["passed"]
        is False
    )
    assert (
        classified["hard_constraint_evidence"]["winding_max_C"]["passed"]
        is False
    )
    assert (
        classified["hard_constraint_evidence"]["core_max_C"]["passed"]
        is False
    )


def test_successes_are_ranked_and_forwarded_without_production_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view_a, receipt_a, _seal_a = _fake_view(
        tmp_path,
        task_id=96325,
        candidate="a" * 64,
        result=_result(
            width_mm=1000,
            length_mm=900,
            height_mm=700,
            resonance_hz=16000,
            winding_c=90,
            core_c=110,
            loss_w=100,
        ),
    )
    view_b, receipt_b, _seal_b = _fake_view(
        tmp_path,
        task_id=96327,
        candidate="b" * 64,
        result=_result(
            width_mm=1100,
            length_mm=900,
            height_mm=700,
            resonance_hz=14000,
            winding_c=90,
            core_c=110,
            loss_w=90,
        ),
    )
    views = {str(receipt_a): view_a, str(receipt_b): view_b}
    monkeypatch.setattr(
        postsuccess,
        "authenticate_collection",
        lambda path: views[str(path.resolve(strict=True))],
    )
    monkeypatch.setattr(
        postsuccess,
        "authenticate_aggregate_reference",
        lambda _path: _aggregate_reference(tmp_path),
    )
    monkeypatch.setattr(
        postsuccess,
        "_strict_readiness",
        lambda paths, **_kwargs: (
            {
                "status": "rows_valid_admission_closed",
                "retraining_admission": {
                    "allowed": False,
                    "reasons": ["strict_new_rows<8"],
                },
                "dataset_write_performed": False,
                "surrogate_retraining_performed": False,
            },
            {96325: True, 96327: True},
        ),
    )

    state = postsuccess.process_cycle(
        lanes=[
            postsuccess.Lane(96325, receipt_a.parent),
            postsuccess.Lane(96327, receipt_b.parent),
        ],
        output_root=tmp_path / "output",
        aggregate_manifest=tmp_path / "unused-aggregate.json",
        base_dataset=tmp_path / "unused.parquet",
    )

    assert state["status"] == "all_standard_collections_processed"
    assert state["collection_count"] == 2
    snapshot_record = state["latest_snapshot"]["snapshot_manifest"]
    snapshot = postsuccess._validate_seal(
        postsuccess._read_json(
            Path(snapshot_record["path"]), "snapshot"
        ),
        postsuccess.SNAPSHOT_SCHEMA,
        "snapshot",
    )
    assert snapshot["measured_actual_nds_performed"] is True
    assert snapshot["measured_hard_feasible_count"] == 1
    assert snapshot["diagnostic_actual_rank0_count"] == 1
    assert {row["audit_non_dominated_rank"] for row in snapshot["ranked_rows"]} == {
        0
    }
    assert snapshot["production_pareto_emitted"] is False
    assert snapshot["production_claimed"] is False
    assert snapshot["strict_al_readiness"]["retraining_admission"]["allowed"] is False
    rerank = postsuccess._validate_seal(
        postsuccess._read_json(
            Path(snapshot["global_rerank_input"]["path"]), "rerank input"
        ),
        postsuccess.RERANK_INPUT_SCHEMA,
        "rerank input",
    )
    assert rerank["surrogate_and_measured_rows_directly_unioned"] is False
    assert rerank["handoff_to_next_surrogate_generation_allowed"] is False
    assert rerank["scheduler_mutation_performed"] is False

    replay = postsuccess.process_cycle(
        lanes=[
            postsuccess.Lane(96325, receipt_a.parent),
            postsuccess.Lane(96327, receipt_b.parent),
        ],
        output_root=tmp_path / "output",
        aggregate_manifest=tmp_path / "unused-aggregate.json",
        base_dataset=tmp_path / "unused.parquet",
    )
    assert replay["latest_snapshot"]["snapshot_id"] == snapshot["snapshot_id"]


def test_superseded_lifecycle_lane_is_authenticated_but_excluded_from_nds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_view, old_receipt, _old_seal = _fake_view(
        tmp_path,
        task_id=96332,
        candidate="5" * 64,
        result=_result(
            width_mm=1000,
            length_mm=900,
            height_mm=700,
            resonance_hz=16_000,
            winding_c=90,
            core_c=110,
            loss_w=80,
        ),
    )
    replacement_view, replacement_receipt, _replacement_seal = _fake_view(
        tmp_path,
        task_id=96338,
        candidate="9" * 64,
        result=_result(
            width_mm=1050,
            length_mm=900,
            height_mm=700,
            resonance_hz=16_000,
            winding_c=90,
            core_c=110,
            loss_w=100,
        ),
    )
    views = {
        str(old_receipt): old_view,
        str(replacement_receipt): replacement_view,
    }
    monkeypatch.setattr(
        postsuccess,
        "authenticate_collection",
        lambda path: views[str(path.resolve(strict=True))],
    )
    monkeypatch.setattr(
        postsuccess,
        "authenticate_aggregate_reference",
        lambda _path: _aggregate_reference(tmp_path),
    )
    monkeypatch.setattr(
        postsuccess,
        "_strict_readiness",
        lambda paths, **_kwargs: (
            {
                "status": "rows_valid_admission_closed",
                "retraining_admission": {
                    "allowed": False,
                    "reasons": ["strict_new_rows<8"],
                },
                "dataset_write_performed": False,
                "surrogate_retraining_performed": False,
            },
            {96338: True},
        ),
    )

    state = postsuccess.process_cycle(
        lanes=[
            postsuccess.Lane(
                96332,
                old_receipt.parent,
                selection_effective=False,
            ),
            postsuccess.Lane(
                96337,
                tmp_path / "failed-retry-lifecycle",
                selection_effective=False,
            ),
            postsuccess.Lane(96338, replacement_receipt.parent),
        ],
        output_root=tmp_path / "output",
        aggregate_manifest=tmp_path / "unused-aggregate.json",
        base_dataset=tmp_path / "unused.parquet",
    )

    assert state["expected_lane_count"] == 1
    assert state["lifecycle_lane_count"] == 3
    assert state["effective_task_ids"] == [96338]
    assert state["selection_superseded_task_ids"] == [96332, 96337]
    assert state["collection_count"] == 1
    assert state["lifecycle_collection_count"] == 2
    old_lane = next(
        lane for lane in state["lanes"] if lane["task_id"] == 96332
    )
    assert old_lane["selection_effective"] is False
    assert old_lane["selection_exclusion_reason"] == (
        "superseded_lifecycle_lane"
    )
    snapshot_record = state["latest_snapshot"]["snapshot_manifest"]
    snapshot = postsuccess._validate_seal(
        postsuccess._read_json(
            Path(snapshot_record["path"]), "snapshot"
        ),
        postsuccess.SNAPSHOT_SCHEMA,
        "snapshot",
    )
    assert snapshot["expected_lane_count"] == 1
    assert [
        row["task_id"] for row in snapshot["ranked_rows"]
    ] == [96338]


def test_strict_al_dispatches_only_through_named_custom_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_json(
        tmp_path / "collection_receipt.json",
        {"schema_version": postsuccess.COLLECTION_SCHEMA},
    )
    adapter_collection = postsuccess._sealed(
        {
            "schema_version": postsuccess.AUTHENTICATED_COLLECTION_SCHEMA,
            **postsuccess.SAFETY_FLAGS,
            "scheduler_get_only_collection": True,
            "scheduler_mutation_performed": False,
            "scientific_pass_claimed": False,
            "production_claimed": False,
            "task_id": 96325,
            "candidate_physics_sha256": "a" * 64,
            "result": {"placeholder": True},
            "result_sha256": "b" * 64,
        }
    )
    view = {
        "schema_version": postsuccess.AUTHENTICATED_COLLECTION_SCHEMA,
        "collection": adapter_collection,
        "plan": {
            "solver_revision": "c" * 40,
            "library_revision": "d" * 40,
        },
        "params": {},
        "selected": {
            "task_identity": {
                "payload_sha256": "e" * 64,
                "seed": 2_607_260_001,
                "fixed_primary_turns": 6,
            }
        },
        "submission": {},
    }
    monkeypatch.setattr(
        postsuccess, "authenticate_collection", lambda _path: view
    )

    truth = strict_al.authenticate_collection(path)

    assert truth.adapter_kind == "postdeadline_standard"
    assert truth.source_seed == 2_607_260_001
    assert truth.source_fixed_primary_turns == 6
    assert truth.collection["production_eligible"] is False


def test_module_contains_no_scheduler_mutator_call() -> None:
    source = Path(postsuccess.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "submit_verification",
        "submit",
        "cancel_task",
        "cancel",
        "post",
        "patch",
        "delete",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)
