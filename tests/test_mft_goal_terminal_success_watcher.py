from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from module.mft_goal_20260726_contract import (
    attest_fixed_identity,
    canonical_sha256,
    fixed_identity_expectations,
)
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_terminal_success_watcher as watcher
from tools import mft_goal_truth_promotion as promotion


def _slot() -> dict[str, Any]:
    return {
        "logical_authority_task_id": 41,
        "execution_task_id": 99,
        "task_name": "sealed-task",
        "dedupe_key": "sealed-dedupe",
        "retry_authority": "retry_of_timeout12h",
        "candidate_physics_sha256": "a" * 64,
        "plan": {"path": "plan.json", "sha256": "b" * 64, "size_bytes": 1},
        "plan_payload_sha256": "c" * 64,
        "submission": {
            "path": "submission.json",
            "sha256": "d" * 64,
            "size_bytes": 1,
        },
        "submission_payload_sha256": "e" * 64,
        "fixed_identity_attestation_sha256": "f" * 64,
    }


def _plan(root: Path) -> dict[str, Any]:
    return watcher._sealed(
        {
            "schema_version": watcher.WATCH_PLAN_SCHEMA,
            "scheduler_url": watcher.DEFAULT_SCHEDULER_URL,
            "scheduler_project": watcher.DEFAULT_PROJECT,
            "output_root": str(root),
            "poll_seconds": 45,
            "slots": [_slot()],
        }
    )


def _snapshot(*, success: bool) -> dict[str, Any]:
    return {
        "task_id": 99,
        "name": "sealed-task",
        "project": watcher.DEFAULT_PROJECT,
        "dedupe_key": "sealed-dedupe",
        "status": "completed" if success else "failed",
        "state": "succeeded" if success else "failed",
        "exit_code": 0 if success else 1,
        "failure_message": None if success else "native failure",
        "finished_at": "2026-07-25T10:00:00Z",
    }


def _fake_success_receipt(
    *, slot: dict[str, Any], collection_path: Path
) -> dict[str, Any]:
    return watcher._sealed(
        {
            "schema_version": watcher.SUCCESS_RECEIPT_SCHEMA,
            "logical_authority_task_id": slot["logical_authority_task_id"],
            "execution_task_id": slot["execution_task_id"],
            "collection": production._file_record(collection_path),
            "strict_authentication": {"adapter_kind": "diagnostic"},
            "hard_constraint_status": {"actual_truth_feasible": True},
            "actual_truth": {"truth": "authenticated"},
            "eligible_for_global_truth_nds": True,
            "scheduler_get_only": True,
            "scheduler_mutation_performed": False,
            "fixed_boundary_policy_preserved": True,
            "authenticated_at_utc": watcher._now(),
        }
    )


def test_terminal_kind_requires_exact_success_triple() -> None:
    assert watcher._terminal_kind(_snapshot(success=True)) == "success"
    wrong_exit = _snapshot(success=True)
    wrong_exit["exit_code"] = 1
    assert watcher._terminal_kind(wrong_exit) == "failure"
    assert watcher._terminal_kind(_snapshot(success=False)) == "failure"
    assert watcher._terminal_kind({"status": "running", "state": "running"}) == (
        "active"
    )


def test_failure_creates_ledger_only(tmp_path: Path) -> None:
    calls = {"collect": 0, "promote": 0}

    def collect(**_kwargs: Any) -> Path:
        calls["collect"] += 1
        raise AssertionError("failed task must not be collected")

    def promote(**_kwargs: Any) -> Path:
        calls["promote"] += 1
        raise AssertionError("failed task must not enter truth NDS")

    state = watcher.process_cycle(
        _plan(tmp_path),
        task_reader=lambda **_kwargs: _snapshot(success=False),
        collector=collect,
        promoter=promote,
    )
    paths = watcher._slot_paths(tmp_path, _slot())
    assert paths["failure"].is_file()
    assert not paths["collection"].exists()
    assert not paths["receipt"].exists()
    assert state["slots"][0]["state"] == "terminal_failure"
    assert calls == {"collect": 0, "promote": 0}


def test_success_collects_and_promotes_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"collect": 0, "promote": 0, "read": 0}

    def read(**_kwargs: Any) -> dict[str, Any]:
        calls["read"] += 1
        return _snapshot(success=True)

    def collect(*, output: Path, **_kwargs: Any) -> Path:
        calls["collect"] += 1
        return production._write_immutable_json(output, {"valid": True})

    def promote(*, output: Path, standard_collection_paths: Any) -> Path:
        calls["promote"] += 1
        output.mkdir(parents=True)
        return production._write_immutable_json(
            output / "truth_pareto_manifest.json",
            production._seal(
                {
                    "schema_version": promotion.TRUTH_MANIFEST_SCHEMA,
                    "input_collection_count": len(standard_collection_paths),
                }
            ),
        )

    monkeypatch.setattr(
        watcher, "_success_receipt", _fake_success_receipt
    )
    plan = _plan(tmp_path)
    first = watcher.process_cycle(
        plan, task_reader=read, collector=collect, promoter=promote
    )
    second = watcher.process_cycle(
        plan, task_reader=read, collector=collect, promoter=promote
    )
    assert first["slots"][0]["state"] == "authenticated_feasible"
    assert second["slots"][0]["state"] == "authenticated_feasible"
    assert calls == {"collect": 1, "promote": 1, "read": 1}


def test_restart_recovers_collection_before_scheduler_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = watcher._slot_paths(tmp_path, _slot())
    production._write_immutable_json(paths["collection"], {"valid": True})
    monkeypatch.setattr(
        watcher, "_success_receipt", _fake_success_receipt
    )

    def no_read(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("restart recovery must not depend on Scheduler")

    def promote(*, output: Path, standard_collection_paths: Any) -> Path:
        output.mkdir(parents=True)
        return production._write_immutable_json(
            output / "truth_pareto_manifest.json",
            production._seal(
                {
                    "schema_version": promotion.TRUTH_MANIFEST_SCHEMA,
                    "input_collection_count": len(standard_collection_paths),
                }
            ),
        )

    state = watcher.process_cycle(
        _plan(tmp_path), task_reader=no_read, promoter=promote
    )
    assert state["slots"][0]["recovered_after_restart"] is True
    assert paths["receipt"].is_file()


def _patch_authenticated_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    collection_path = production._write_immutable_json(
        tmp_path / "collection.json", {"authenticated": True}
    )
    fixed = attest_fixed_identity(fixed_identity_expectations())
    slot = _slot()
    slot["fixed_identity_attestation_sha256"] = fixed["sha256"]
    truth = {"fixed_identity_attestation": copy.deepcopy(fixed)}
    view = {
        "collection": {
            "task_id": slot["execution_task_id"],
            "plan": slot["plan"],
            "submission": slot["submission"],
            "truth_evidence": {
                "actual_fixed_identity_attestation": copy.deepcopy(fixed),
                "selected_fixed_identity_attestation": copy.deepcopy(fixed),
            },
        },
        "plan": {
            "retry_of_timeout12h": {
                "logical_authority_task_id": slot[
                    "logical_authority_task_id"
                ]
            }
        },
        "selected": {
            "row_contract": {
                "fixed_identity_attestation": copy.deepcopy(fixed)
            }
        },
    }
    strict = SimpleNamespace(
        adapter_kind="diagnostic",
        collection_file_sha256=production._sha256_file(collection_path),
        solver_revision="1" * 40,
        library_revision="2" * 40,
        source_task_payload_sha256="3" * 64,
        source_seed=2607262000,
        source_fixed_primary_turns=6,
    )
    monkeypatch.setattr(
        watcher.diagnostic, "authenticate_collection", lambda _path: view
    )
    monkeypatch.setattr(
        watcher.strict_al, "authenticate_collection", lambda _path: strict
    )
    monkeypatch.setattr(
        watcher.promotion,
        "_actual_standard_observation",
        lambda _view, *, collection_path: (
            truth,
            {"actual_truth_feasible": True},
        ),
    )
    monkeypatch.setattr(
        watcher.promotion,
        "_actual_standard_truth",
        lambda _view, *, collection_path: truth,
    )
    return collection_path, slot, view, truth


def test_slot_authentication_accepts_identical_canonical_fixed_identity_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_path, slot, _view, truth = _patch_authenticated_slot(
        tmp_path, monkeypatch
    )

    authenticated, status, strict = watcher._authenticate_slot_collection(
        collection_path, slot
    )

    assert authenticated == truth
    assert status["actual_truth_feasible"] is True
    assert strict["adapter_kind"] == "diagnostic"


@pytest.mark.parametrize(
    "source",
    ["collection_actual", "collection_selected", "plan", "actual_truth"],
)
def test_slot_authentication_rejects_fixed_identity_chain_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    collection_path, slot, view, truth = _patch_authenticated_slot(
        tmp_path, monkeypatch
    )
    drifted = copy.deepcopy(truth["fixed_identity_attestation"])
    drifted["observed"]["freq"] = 999.0
    unsigned = dict(drifted)
    unsigned.pop("sha256")
    drifted["sha256"] = canonical_sha256(unsigned)
    targets = {
        "collection_actual": view["collection"]["truth_evidence"],
        "collection_selected": view["collection"]["truth_evidence"],
        "plan": view["selected"]["row_contract"],
        "actual_truth": truth,
    }
    keys = {
        "collection_actual": "actual_fixed_identity_attestation",
        "collection_selected": "selected_fixed_identity_attestation",
        "plan": "fixed_identity_attestation",
        "actual_truth": "fixed_identity_attestation",
    }
    targets[source][keys[source]] = drifted

    with pytest.raises(
        watcher.WatcherContractError,
        match="authenticated fan/TIM/pad identity drifted",
    ):
        watcher._authenticate_slot_collection(collection_path, slot)


def test_single_instance_lock_rejects_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "watcher.lock"
    with watcher.SingleInstanceLock(lock):
        with pytest.raises(watcher.WatcherContractError):
            with watcher.SingleInstanceLock(lock):
                pass
