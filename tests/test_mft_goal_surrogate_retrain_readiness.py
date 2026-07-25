from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import mft_goal_fea_handoff as production
from tools import mft_goal_surrogate_retrain_readiness as readiness
from tools import mft_goal_terminal_success_watcher as watcher


def _slot(logical: int, execution: int) -> dict[str, Any]:
    return {
        "logical_authority_task_id": logical,
        "execution_task_id": execution,
    }


def _receipt(
    collection: Path,
    *,
    logical: int,
    execution: int,
    feasible: bool,
) -> dict[str, Any]:
    return {
        "logical_authority_task_id": logical,
        "execution_task_id": execution,
        "collection": production._file_record(collection),
        "strict_authentication": {"adapter_kind": "diagnostic"},
        "eligible_for_global_truth_nds": feasible,
        "fixed_boundary_policy_preserved": True,
    }


def test_terminal_inventory_uses_all_successes_but_only_feasible_truth_for_nds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = [_slot(41, 91), _slot(42, 92), _slot(43, 93)]
    plan = {
        "output_root": str(tmp_path),
        "payload_sha256": "a" * 64,
        "slots": slots,
    }
    collections = []
    receipts = []
    state_slots = []
    for slot, feasible in zip(slots, (True, False, False), strict=True):
        paths = watcher._slot_paths(tmp_path, slot)
        paths["directory"].mkdir(parents=True)
        paths["collection"].write_text(
            json.dumps({"execution": slot["execution_task_id"]}),
            encoding="utf-8",
        )
        paths["receipt"].write_text("{}", encoding="utf-8")
        receipt = _receipt(
            paths["collection"],
            logical=slot["logical_authority_task_id"],
            execution=slot["execution_task_id"],
            feasible=feasible,
        )
        collections.append(paths["collection"].resolve())
        receipts.append(receipt)
        state_slots.append(
            {
                **slot,
                "state": (
                    "authenticated_feasible" if feasible else "authenticated_infeasible"
                ),
                "receipt": str(paths["receipt"]),
            }
        )
    state = {
        "slots": state_slots,
        "latest_truth_manifest": "unused-by-patched-authenticator",
    }
    monkeypatch.setattr(
        readiness,
        "_effective_watch_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        readiness,
        "_load_state",
        lambda *_args, **_kwargs: state,
    )
    by_execution = {item["execution_task_id"]: item for item in receipts}
    monkeypatch.setattr(
        readiness.watcher,
        "_load_success_receipt",
        lambda _path, slot: by_execution[slot["execution_task_id"]],
    )
    observed_feasible: list[dict[str, Any]] = []

    def authenticate_nds(
        *,
        root: Path,
        state: dict[str, Any],
        feasible_receipts: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        assert root == tmp_path
        observed_feasible.extend(feasible_receipts)
        return {"authenticated": True}

    monkeypatch.setattr(readiness, "_authenticate_truth_nds", authenticate_nds)

    inventory = readiness.authenticate_terminal_inventory(
        watch_plan_path=tmp_path / "watch_plan.json",
        state_path=tmp_path / "state.json",
    )

    assert inventory.collection_paths == tuple(collections)
    assert len(inventory.receipts) == 3
    assert len(inventory.feasible_receipts) == 1
    assert observed_feasible == [receipts[0]]


def test_global_truth_nds_binds_exact_feasible_receipt_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = tmp_path / "collection.json"
    collection.write_text("{}", encoding="utf-8")
    feasible = _receipt(
        collection,
        logical=41,
        execution=91,
        feasible=True,
    )
    key = watcher._nds_key([feasible])
    snapshot = tmp_path / "truth_snapshots" / f"n01-{key}"
    snapshot.mkdir(parents=True)
    manifest_path = snapshot / "truth_pareto_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    nds_receipt_path = snapshot.parent / f"{snapshot.name}.receipt.json"
    nds_receipt = watcher._sealed(
        {
            "schema_version": watcher.NDS_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "authority_key": key,
            "logical_authority_task_ids": [41],
            "input_collections": [feasible["collection"]],
            "truth_manifest": production._file_record(manifest_path),
            "global_nondominated_sort_performed": True,
            "scheduler_mutation_performed": False,
        }
    )
    nds_receipt_path.write_text(json.dumps(nds_receipt), encoding="utf-8")
    monkeypatch.setattr(
        readiness.promotion,
        "_load_truth_manifest",
        lambda _path: (
            {"payload_sha256": "b" * 64, "input_collection_count": 1},
            [{"truth_non_dominated_rank": 0}],
            {},
        ),
    )

    result = readiness._authenticate_truth_nds(
        root=tmp_path,
        state={"latest_truth_manifest": str(manifest_path)},
        feasible_receipts=[feasible],
    )

    assert result is not None
    assert result["authenticated"] is True
    assert result["input_collection_count"] == 1
    assert result["ranked_truth_count"] == 1

    tampered = dict(feasible)
    tampered["logical_authority_task_id"] = 42
    with pytest.raises(
        readiness.ReadinessContractError,
        match="inventory drifted",
    ):
        readiness._authenticate_truth_nds(
            root=tmp_path,
            state={"latest_truth_manifest": str(manifest_path)},
            feasible_receipts=[tampered],
        )


def test_closed_gate_emits_no_dataset_or_training_command(
    tmp_path: Path,
) -> None:
    result = readiness._command_plan(
        admitted=False,
        python_executable=Path("python"),
        code_root=tmp_path,
        expected_code_revision="a" * 40,
        base_dataset=tmp_path / "base.parquet",
        expected_base_sha256="b" * 64,
        expected_base_rows=6_151,
        collection_paths=[tmp_path / "collection.json"],
        derived_dataset_output=tmp_path / "dataset",
        training_plan_root=tmp_path / "training",
        remote_training_root="/gpfs/al",
    )

    assert result == {
        "emitted": False,
        "reason": "strict authenticated truth admission gate is closed",
        "dataset_build_argv": None,
        "training_plan_argv": None,
    }


def test_open_gate_emits_build_then_local_only_training_plan_argv(
    tmp_path: Path,
) -> None:
    collections = [tmp_path / f"collection-{index}.json" for index in range(8)]
    result = readiness._command_plan(
        admitted=True,
        python_executable=Path("python"),
        code_root=tmp_path / "code",
        expected_code_revision="a" * 40,
        base_dataset=tmp_path / "base.parquet",
        expected_base_sha256="b" * 64,
        expected_base_rows=6_151,
        collection_paths=collections,
        derived_dataset_output=tmp_path / "dataset",
        training_plan_root=tmp_path / "training",
        remote_training_root="/gpfs/al",
    )

    build = result["dataset_build_argv"]
    train = result["training_plan_argv"]
    assert result["emitted"] is True
    assert build[:4] == [
        "python",
        "-m",
        "tools.mft_goal_strict_al_ingest",
        "build",
    ]
    assert build.count("--collection") == 8
    assert "--require-retraining-ready" in build
    assert train[:4] == [
        "python",
        "-m",
        "tools.mft_goal_al_slurm_train",
        "plan",
    ]
    assert "--apply" not in train
    assert result["scheduler_post_authorized"] is False
    assert result["old_generation_result_mixing_allowed"] is False


def test_zero_truth_strict_readiness_audits_base_without_prepare_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(readiness.strict_al, "_profile_content", lambda: {})
    monkeypatch.setattr(
        readiness.strict_al,
        "_base_audit",
        lambda *_args, **_kwargs: (
            object(),
            {"sha256": "a" * 64, "row_count": 6_151},
            "physics-v3",
            {"Llt_phys": 6_151},
        ),
    )
    monkeypatch.setattr(
        readiness.strict_al,
        "prepare_ingest",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("zero truth must not call prepare_ingest")
        ),
    )

    summary, admission = readiness._strict_readiness(
        base_dataset=tmp_path / "base.parquet",
        expected_base_sha256="a" * 64,
        expected_base_rows=6_151,
        collection_paths=[],
    )

    assert summary["new_rows"] == 0
    assert admission["allowed"] is False
    assert admission["strict_new_rows"] == 0


def test_nonzero_truth_delegates_every_collection_to_strict_ingest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collections = [tmp_path / "one.json", tmp_path / "two.json"]
    observed: dict[str, Any] = {}
    prepared = SimpleNamespace(
        retraining_admission={"allowed": False, "strict_new_rows": 2}
    )

    def prepare_ingest(**kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return prepared

    monkeypatch.setattr(readiness.strict_al, "prepare_ingest", prepare_ingest)
    monkeypatch.setattr(
        readiness.strict_al,
        "_summary",
        lambda _prepared: {"new_rows": 2},
    )

    summary, admission = readiness._strict_readiness(
        base_dataset=tmp_path / "base.parquet",
        expected_base_sha256="a" * 64,
        expected_base_rows=6_151,
        collection_paths=collections,
    )

    assert observed["collection_paths"] == collections
    assert summary == {"new_rows": 2}
    assert admission["allowed"] is False


def test_state_drift_during_readiness_audit_fails_closed(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.json"
    state.write_text('{"generation":1}', encoding="utf-8")
    record = production._file_record(state)
    readiness._assert_state_record_current(state, record)

    state.write_text('{"generation":2}', encoding="utf-8")
    with pytest.raises(
        readiness.ReadinessContractError,
        match="changed during readiness audit",
    ):
        readiness._assert_state_record_current(state, record)
