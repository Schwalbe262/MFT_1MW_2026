from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from regression_260707.verify import scheduler_client
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_full_postsuccess_orchestrator as postsuccess
from tools import mft_goal_truth_promotion as promotion


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _init(tmp_path: Path) -> tuple[Path, Path, Path]:
    fast_root = tmp_path / "fast"
    final_parent = tmp_path / "deliverables"
    fast_root.mkdir()
    final_parent.mkdir()
    fast_receipt = fast_root / "fast_lane_receipt.json"
    final_root = final_parent / "final_deliverable_20260726"
    plan_path = postsuccess.initialize_watch_plan(
        fast_lane_receipt_path=fast_receipt,
        output_root=tmp_path / "operational",
        final_output_root=final_root,
        poll_seconds=10,
    )
    return plan_path, fast_receipt, final_root


def _collection(task_id: int = 1234) -> dict:
    return postsuccess._sealed(
        {
            "candidate_physics_sha256": "a" * 64,
            "task_id": task_id,
            "goal_physical_spec_passed": True,
            "actual_truth_evidence": {
                "goal_physical_spec_passed": True,
            },
            "standard_retained_bundle_reauthenticated": True,
            "full_prune_protection_marker_verified": True,
            "standard_remote_aedt_bundle_receipt": {
                "artifact_size_bytes": 101,
                "results_size_bytes": 203,
                "retention_required": True,
                "prune_protection_required": True,
            },
            "remote_full_aedt_bundle_receipt": {
                "artifact_size_bytes": 307,
                "results_size_bytes": 401,
                "retention_required": True,
                "prune_protection_required": True,
            },
        }
    )


def _handoff(task_id: int = 1234) -> dict:
    return {
        "receipt_path": Path("fast_lane_receipt.json"),
        "receipt": {"payload_sha256": "b" * 64},
        "submission_path": Path("full_submission.json"),
        "submission": {"task_id": task_id, "payload_sha256": "c" * 64},
        "plan_path": Path("full_plan.json"),
        "plan": {
            "candidate_physics_sha256": "a" * 64,
            "stage": {
                "task_name": "goal-full-candidate-a",
                "retained_aedt_bundle": {"dedupe_key": "full-dedupe-a"},
            },
        },
    }


def _terminal(task_id: int = 1234, status: str = "completed") -> dict:
    task = {
        "task_id": task_id,
        "name": "goal-full-candidate-a",
        "dedupe_key": "full-dedupe-a",
        "project": scheduler_client.MFT_PROJECT,
        "status": status,
    }
    unsigned = {
        "candidate_physics_sha256": "a" * 64,
        "task_name": "goal-full-candidate-a",
        "dedupe_key": "full-dedupe-a",
        "matching_task_count": 1,
        "matching_tasks": [task],
    }
    return {
        "inventory": {
            **unsigned,
            "snapshot_sha256": canonical_sha256(unsigned),
        },
        "task": task,
        "status": status,
        "scheduler_methods_used": ["GET"],
        "scheduler_mutation_performed": False,
    }


def test_module_contains_no_scheduler_mutator_call() -> None:
    source = Path(postsuccess.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "submit_verification",
        "cancel_task",
        "cancel",
        "post",
        "patch",
        "delete",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert calls.isdisjoint(forbidden)


def _install_cycle_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    collection: dict,
) -> tuple[dict, Path]:
    collection_path = tmp_path / "operational" / "full_collection.json"
    _write_json(collection_path, collection)
    handoff = _handoff(int(collection["task_id"]))
    handoff["receipt_path"] = tmp_path / "fast" / "fast_lane_receipt.json"
    handoff["submission_path"] = tmp_path / "fast" / "full_submission.json"
    handoff["plan_path"] = tmp_path / "fast" / "full_plan.json"
    for path in (
        handoff["receipt_path"],
        handoff["submission_path"],
        handoff["plan_path"],
    ):
        _write_json(path, {"placeholder": path.name})
    handoff["receipt"] = {
        "payload_sha256": production._sha256_file(handoff["receipt_path"])
    }
    handoff["submission"]["payload_sha256"] = production._sha256_file(
        handoff["submission_path"]
    )
    monkeypatch.setattr(
        postsuccess,
        "_load_fast_lane_handoff",
        lambda _path: handoff,
    )
    monkeypatch.setattr(
        postsuccess,
        "_terminal_snapshot",
        lambda **_kwargs: _terminal(int(collection["task_id"])),
    )
    monkeypatch.setattr(
        postsuccess,
        "_load_or_collect",
        lambda **_kwargs: (collection_path, collection),
    )
    return handoff, collection_path


def test_init_waits_without_creating_stable_root(tmp_path: Path) -> None:
    plan_path, _fast_receipt, final_root = _init(tmp_path)

    assert not final_root.exists()
    state = postsuccess.process_cycle(watch_plan_path=plan_path)

    assert state["status"] == "waiting_for_full_fast_lane_receipt"
    assert state["scheduler_methods_used"] == ["GET"]
    assert state["scheduler_mutation_performed"] is False
    assert not final_root.exists()
    assert (plan_path.parent / "state.json").is_file()


def test_fast_lane_file_boundary_is_sealed_and_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_json(tmp_path / "full_plan.json", {"plan": 1})
    full_plan = {
        "candidate_physics_sha256": "a" * 64,
        "stage": {
            "task_name": "goal-full-candidate-a",
            "retained_aedt_bundle": {"dedupe_key": "full-dedupe-a"},
        },
    }
    submission = production._seal(
        {
            "schema_version": promotion.FULL_SUBMISSION_SCHEMA,
            "plan": production._file_record(plan_path),
            "task_id": 1234,
            "scheduler_url": postsuccess.DEFAULT_SCHEDULER_URL,
            "scheduler_project": scheduler_client.MFT_PROJECT,
            "resources": {"cpus": 16, "timeout_seconds": 43200},
            "payload": "test",
        }
    )
    submission_path = _write_json(tmp_path / "full_submission.json", submission)
    monkeypatch.setattr(
        promotion,
        "_load_full_plan",
        lambda _path: (full_plan, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        promotion,
        "_load_full_submission",
        lambda _path, **_kwargs: submission,
    )
    receipt = postsuccess._sealed(
        {
            "schema_version": postsuccess.FAST_LANE_RECEIPT_SCHEMA,
            "campaign_id": postsuccess.CAMPAIGN_ID,
            "full_submission": production._file_record(submission_path),
            "candidate_physics_sha256": "a" * 64,
            "task_id": 1234,
            "task_name": "goal-full-candidate-a",
            "dedupe_key": "full-dedupe-a",
            "resources": postsuccess.FULL_RESOURCES,
            "full_model": 1,
            "thermal_symmetry": "full",
            "fixed_boundary": postsuccess.FIXED_BOUNDARY,
            "exact_full_sibling_count": 1,
            "scheduler_submission_performed": True,
            "scheduler_repository_modified": False,
        }
    )
    receipt_path = _write_json(tmp_path / "fast_lane_receipt.json", receipt)

    view = postsuccess._load_fast_lane_handoff(receipt_path)
    assert view["submission"]["task_id"] == 1234

    tampered = dict(receipt)
    tampered["task_id"] = 9999
    _write_json(receipt_path, tampered)
    with pytest.raises(production.HandoffContractError, match="seal drifted"):
        postsuccess._load_fast_lane_handoff(receipt_path)


def test_live_inventory_requires_exactly_one_full_sibling() -> None:
    handoff = _handoff()
    row = {
        "id": 1234,
        "name": "goal-full-candidate-a",
        "dedupe_key": "full-dedupe-a",
        "project": scheduler_client.MFT_PROJECT,
        "status": "completed",
    }

    one = postsuccess._exact_sibling_inventory([row], handoff=handoff)
    assert one["matching_task_count"] == 1
    with pytest.raises(
        production.HandoffContractError,
        match="exactly one",
    ):
        postsuccess._exact_sibling_inventory(
            [row, {**row, "id": 1235}],
            handoff=handoff,
        )


def test_terminal_status_prefers_completed_over_execution_state() -> None:
    handoff = _handoff()
    row = {
        "id": 1234,
        "name": "goal-full-candidate-a",
        "dedupe_key": "full-dedupe-a",
        "project": scheduler_client.MFT_PROJECT,
        "status": "completed",
        "state": "succeeded",
    }

    snapshot = postsuccess._terminal_snapshot(
        handoff=handoff,
        scheduler_url=postsuccess.DEFAULT_SCHEDULER_URL,
        sibling_reader=lambda **_kwargs: [row],
        task_reader=lambda **_kwargs: row,
    )

    assert snapshot["status"] == "completed"


def test_disk_gate_uses_exact_receipt_sum_plus_ten_percent_and_five_gib(
    tmp_path: Path,
) -> None:
    plan_path, fast_receipt, _final_root = _init(tmp_path)
    _write_json(fast_receipt, {"placeholder": True})
    plan = postsuccess._load_watch_plan(plan_path)
    collection = _collection()
    collection_path = _write_json(
        plan_path.parent / "full_collection.json",
        collection,
    )
    retained = 101 + 203 + 307 + 401
    reserve = (retained * 10 + 99) // 100
    required = retained + reserve + postsuccess.LOCAL_FIXED_RESERVE_BYTES

    admission_path, admission = postsuccess._capture_disk_admission(
        plan=plan,
        collection_path=collection_path,
        collection=collection,
        disk_usage_reader=lambda _path: SimpleNamespace(
            total=required + 10,
            used=0,
            free=required + 10,
        ),
        device_reader=lambda _path: 77,
        observed_at_utc=postsuccess._now(),
    )

    assert admission["retained_receipt_bytes_total"] == retained
    assert admission["ten_percent_reserve_bytes"] == reserve
    assert admission["required_free_bytes"] == required
    assert admission["same_volume_verified"] is True
    assert admission["admission_passed"] is True
    assert postsuccess._validate_disk_admission(
        admission_path,
        plan=plan,
        collection_path=collection_path,
        collection=collection,
    ) == admission


def test_insufficient_same_volume_disk_never_creates_or_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _fast_receipt, final_root = _init(tmp_path)
    collection = _collection()
    _install_cycle_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        collection=collection,
    )
    package_calls: list[dict] = []

    state = postsuccess.process_cycle(
        watch_plan_path=plan_path,
        package_builder=lambda **kwargs: package_calls.append(kwargs),
        disk_usage_reader=lambda _path: SimpleNamespace(
            total=1024,
            used=0,
            free=1024,
        ),
        device_reader=lambda _path: 9,
        observed_at_utc=postsuccess._now(),
    )

    assert state["status"] == "waiting_for_same_volume_local_disk"
    assert state["disk_admission_passed"] is False
    assert package_calls == []
    assert not final_root.exists()
    assert not (plan_path.parent / "package_intent.json").exists()
    assert not (plan_path.parent / "success_receipt.json").exists()


def test_success_is_atomic_validated_rehashed_and_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _fast_receipt, final_root = _init(tmp_path)
    collection = _collection()
    _install_cycle_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        collection=collection,
    )
    events: list[str] = []
    package_box: dict[str, dict] = {}

    def disk_usage(_path: Path) -> SimpleNamespace:
        events.append("disk_gate")
        capacity = 20 * postsuccess.GIB
        return SimpleNamespace(total=capacity, used=0, free=capacity)

    def package_builder(**_kwargs: object) -> Path:
        events.append("package")
        assert not final_root.exists()
        final_root.mkdir()
        symmetric = final_root / "symmetric_model.aedt"
        full = final_root / "full_model.aedt"
        symmetric.write_bytes(b"symmetric")
        full.write_bytes(b"full")
        manifest = final_root / "package_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        package = {
            "payload_sha256": "d" * 64,
            "candidate_physics_sha256": collection[
                "candidate_physics_sha256"
            ],
            "result_identities": {
                "full": {"task_id": collection["task_id"]}
            },
            "aedt_artifacts": {
                "symmetric": {
                    "local_absolute_path": str(symmetric.resolve())
                },
                "full": {"local_absolute_path": str(full.resolve())},
            },
            "aedtresults_trees": {
                "symmetric": {"files": []},
                "full": {"files": []},
            },
            "evidence": {},
        }
        package_box["value"] = package
        return manifest

    validation_calls: list[Path] = []

    def validate_package(**kwargs: object) -> tuple[Path, dict]:
        events.append("validate")
        collection_path = Path(str(kwargs["collection_path"]))
        validation_calls.append(collection_path)
        return final_root / "package_manifest.json", package_box["value"]

    monkeypatch.setattr(
        postsuccess,
        "_validate_package_for_collection",
        validate_package,
    )
    state = postsuccess.process_cycle(
        watch_plan_path=plan_path,
        package_builder=package_builder,
        disk_usage_reader=disk_usage,
        device_reader=lambda _path: 11,
        observed_at_utc=postsuccess._now(),
    )

    assert state["status"] == "completed_validated_final_package"
    assert events == ["disk_gate", "package", "validate"]
    assert len(validation_calls) == 1
    success_path = plan_path.parent / "success_receipt.json"
    rehash_path = plan_path.parent / "package_rehash_manifest.json"
    assert success_path.is_file()
    assert rehash_path.is_file()
    success = postsuccess._validate_seal(
        postsuccess._read_json(success_path),
        postsuccess.SUCCESS_RECEIPT_SCHEMA,
        label="test success",
    )
    assert success["exactly_one_full_terminal_success"] is True
    assert success["full_validate_package_reauthenticated"] is True
    assert success["every_package_file_rehashed"] is True
    assert success["scheduler_methods_used"] == ["GET"]
    assert success["scheduler_mutation_performed"] is False
    assert success["standard_retained_remote_marker_pruned"] is False
    assert success["full_retained_remote_marker_pruned"] is False

    events.clear()
    state_again = postsuccess.process_cycle(
        watch_plan_path=plan_path,
        package_builder=lambda **_kwargs: pytest.fail(
            "restart must not package again"
        ),
        disk_usage_reader=lambda _path: pytest.fail(
            "restart must use the bound admission"
        ),
        device_reader=lambda _path: pytest.fail(
            "restart must use the bound volume proof"
        ),
    )
    assert state_again["status"] == "completed_validated_final_package"
    assert events == ["validate"]
    assert len(validation_calls) == 2
    assert production._file_record(success_path) == state_again[
        "success_receipt"
    ]


def test_preexisting_final_path_without_intent_is_not_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, _fast_receipt, final_root = _init(tmp_path)
    collection = _collection()
    _install_cycle_stubs(
        monkeypatch,
        tmp_path=tmp_path,
        collection=collection,
    )
    final_root.mkdir()
    (final_root / "package_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    state = postsuccess.process_cycle(watch_plan_path=plan_path)

    assert state["status"] == "blocked_fail_closed"
    assert "without a pre-package disk intent" in state["error"]
    assert state["completion_authority"] is None
    assert not (plan_path.parent / "success_receipt.json").exists()
