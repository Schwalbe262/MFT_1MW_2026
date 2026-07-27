from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from tools import mft_goal_corrected_replacement_strict_collect as collector
from tools import mft_goal_strict_al_ingest as strict_al


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40
PHYSICS_REVISION = "test-physics-v1"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _result(task_id: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "git_hash": SOLVER_REVISION,
        "pyaedt_library_git_hash": LIBRARY_REVISION,
        "physics_data_revision": PHYSICS_REVISION,
        "N1": 6,
        "N2": 60,
        "full_model": 0,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "matrix_skin_mesh": 0,
        "keep_project": 1,
        "round_corner": 0,
        "T_max_Tx_main": 99.0,
        "T_max_Rx_main": 119.0,
        "T_max_core": 119.0,
        "thermal_rx_block_interface_contract_version": (
            "thermal-rx-block-interface-coverage-v1"
        ),
        "thermal_rx_main_interface_coverage_passed": True,
        "thermal_rx_main_unpaired_interfaces": [],
        "thermal_temperature_limiter_triggered": False,
        "thermal_temperature_limiter_max_K": 392.15,
        "thermal_result_scientific_valid": True,
    }


def _authority(tmp_path: Path) -> dict[int, collector.AuthorityLane]:
    receipt_path = _write_json(tmp_path / "submission_receipt.json", {})
    source_plan_path = _write_json(tmp_path / "source_plan.json", {})
    replay_plan_path = _write_json(tmp_path / "replay_plan.json", {})
    plan = {
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "payload_sha256": "c" * 64,
        "lanes": [
            {
                "rank": offset + 1,
                "name": f"corrected-{task_id}",
                "resources": {
                    "aedt_backend": "standalone",
                    "cpus": 8,
                    "memory_mb": 65536,
                    "gpus": 0,
                    "priority": 99,
                    "timeout_seconds": 43200,
                    "max_workers_per_node": 1,
                },
            }
            for offset, task_id in enumerate(
                collector.EXPECTED_TASK_IDS
            )
        ],
    }
    authority = {}
    for offset, task_id in enumerate(collector.EXPECTED_TASK_IDS):
        campaign = "cooler" if offset < 16 else "lastmile"
        rank = offset + 1
        params_path = _write_json(
            tmp_path / "params" / f"{task_id}.json",
            {"physics_data_revision": PHYSICS_REVISION},
        )
        geometry = f"{offset + 1:064x}"
        authority[task_id] = collector.AuthorityLane(
            task_id=task_id,
            campaign=campaign,
            rank=rank,
            name=f"corrected-{task_id}",
            dedupe_key=f"dedupe-{task_id}",
            physical_geometry_sha256=geometry,
            old_task_id=96000 + offset,
            source_search_task_id=96425 + (offset % 5),
            source_seed=2_707_276_000 + offset,
            source_fixed_primary_turns=6,
            source_secondary_turns=60,
            effective_params_sha256=f"{1000 + offset:064x}",
            replacement_plan_path=replay_plan_path,
            replacement_plan=plan,
            source_plan_path=source_plan_path,
            source_plan={"payload_sha256": "e" * 64},
            source_params_path=params_path,
            source_params={"physics_data_revision": PHYSICS_REVISION},
            submission_receipt_path=receipt_path,
            submission_receipt={},
        )
    return authority


def _task(
    lane: collector.AuthorityLane,
    *,
    completed: bool,
    result: dict[str, object] | None,
) -> dict[str, object]:
    resources = next(
        row["resources"]
        for row in lane.replacement_plan["lanes"]
        if row["rank"] == lane.rank
    )
    state = "completed" if completed else "running"
    return {
        "id": lane.task_id,
        "task_id": lane.task_id,
        "name": lane.name,
        "dedupe_key": lane.dedupe_key,
        "project": collector.EXPECTED_PROJECT,
        "aedt_backend": resources["aedt_backend"],
        "cpus": resources["cpus"],
        "memory_mb": resources["memory_mb"],
        "gpus": resources["gpus"],
        "priority": resources["priority"],
        "timeout_seconds": resources["timeout_seconds"],
        "max_workers_per_node": resources["max_workers_per_node"],
        "status": state,
        "state": state,
        "exit_code": 0 if completed else None,
        "result_json": result,
    }


def _fake_getters(
    authority: dict[int, collector.AuthorityLane],
    *,
    valid_count: int,
    invalidate_task_id: int | None = None,
):
    valid_ids = set(tuple(collector.EXPECTED_TASK_IDS)[:valid_count])

    def json_getter(url: str) -> collector.FetchedResponse:
        task_id = int(re.search(r"/tasks/(\d+)$", url).group(1))
        lane = authority[task_id]
        result = _result(task_id) if task_id in valid_ids else None
        if result is not None and task_id == invalidate_task_id:
            result["thermal_rx_main_unpaired_interfaces"] = [
                "Rx_main_block_xn"
            ]
            result["thermal_result_scientific_valid"] = False
        value = _task(
            lane,
            completed=task_id in valid_ids,
            result=result,
        )
        raw = json.dumps(value, sort_keys=True).encode("utf-8")
        return collector.FetchedResponse(raw=raw, value=value)

    def stream_getter(url: str) -> collector.FetchedResponse:
        value = ""
        return collector.FetchedResponse(
            raw=json.dumps(value).encode("utf-8"),
            value=value,
        )

    return json_getter, stream_getter


def test_collect_emits_only_valid_unique_rows_and_direct_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority")
    monkeypatch.setattr(
        collector, "load_authority", lambda *args, **kwargs: authority
    )
    json_getter, stream_getter = _fake_getters(
        authority, valid_count=8
    )
    manifest_path = collector.collect(
        receipt_path=next(iter(authority.values())).submission_receipt_path,
        output_dir=tmp_path / "isolated",
        json_getter=json_getter,
        stream_getter=stream_getter,
    )

    authenticated = collector.authenticate_manifest(manifest_path)
    trigger = authenticated["retraining_trigger"]
    assert len(authenticated["collection_paths"]) == 8
    assert trigger["allowed"] is True
    assert trigger["scientific_valid_unique_rows"] == 8
    assert trigger["unique_scientific_geometries"] == 8
    assert trigger["unique_corrected_execution_tasks"] == 8
    assert trigger["unique_source_tasks"] == 8
    assert trigger["minimum_source_tasks"] == 4
    assert trigger["unique_upstream_search_tasks_informational"] == 5

    expanded = strict_al.collections_from_corrected_manifests(
        [manifest_path]
    )
    assert expanded == authenticated["collection_paths"]
    strict_truth = strict_al.authenticate_collection(expanded[0])
    assert strict_truth.adapter_kind == "corrected_replacement"
    assert strict_truth.source_fixed_primary_turns == 6
    assert strict_truth.result["thermal_result_scientific_valid"] is True
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["legacy_collection_reused"] is False
    assert raw["legacy_collection_rows_reused"] == 0
    assert raw["expected_task_ids"] == list(range(97116, 97140))
    assert not set(raw["expected_task_ids"]).intersection(
        collector.SOURCE_REPLACEMENT_TASK_IDS
    )
    assert len(raw["task_audit"]) == 24
    assert all(
        Path(path).name == "collection.json" for path in expanded
    )


def test_invalid_six_field_is_quarantined_and_trigger_stays_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority")
    monkeypatch.setattr(
        collector, "load_authority", lambda *args, **kwargs: authority
    )
    invalid_task = collector.EXPECTED_TASK_IDS[0]
    json_getter, stream_getter = _fake_getters(
        authority,
        valid_count=8,
        invalidate_task_id=invalid_task,
    )
    manifest_path = collector.collect(
        receipt_path=next(iter(authority.values())).submission_receipt_path,
        output_dir=tmp_path / "isolated",
        json_getter=json_getter,
        stream_getter=stream_getter,
    )

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["accepted_collection_count"] == 7
    assert raw["retraining_trigger"]["allowed"] is False
    rejected = next(
        row for row in raw["task_audit"] if row["task_id"] == invalid_task
    )
    assert rejected["scientific_valid"] is False
    assert (
        "scientific_invalid:thermal_rx_main_unpaired_interfaces_present"
        in rejected["reasons"]
    )
    assert not (
        manifest_path.parent
        / "rows"
        / f"task-{invalid_task}"
        / "collection.json"
    ).exists()


def test_manifest_authentication_rejects_drifted_result_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "authority")
    monkeypatch.setattr(
        collector, "load_authority", lambda *args, **kwargs: authority
    )
    json_getter, stream_getter = _fake_getters(
        authority, valid_count=8
    )
    manifest_path = collector.collect(
        receipt_path=next(iter(authority.values())).submission_receipt_path,
        output_dir=tmp_path / "isolated",
        json_getter=json_getter,
        stream_getter=stream_getter,
    )
    first_result = (
        manifest_path.parent
        / "rows"
        / f"task-{collector.EXPECTED_TASK_IDS[0]}"
        / "result.json"
    )
    first_result.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        collector.CorrectedReplacementCollectionError,
        match="bytes drifted",
    ):
        collector.authenticate_manifest(manifest_path)


def test_forbidden_legacy_collection_path_is_never_accepted(
    tmp_path: Path,
) -> None:
    legacy = _write_json(
        tmp_path / collector.FORBIDDEN_LEGACY_BASENAME, {}
    )
    with pytest.raises(
        collector.CorrectedReplacementCollectionError,
        match="forbidden legacy collection",
    ):
        collector.load_authority(legacy)


def test_superseded_97042_receipt_cannot_be_truth_authority(
    tmp_path: Path,
) -> None:
    old_receipt = collector._seal(
        {
            "schema": collector.SOURCE_SUBMISSION_RECEIPT_SCHEMA,
            "replacement_count": 24,
            "scheduler_post_calls": 24,
            "all_get_identities_valid": True,
            "all_tasks_accepted_active": True,
            "submissions": [],
        },
        schema=collector.SOURCE_SUBMISSION_RECEIPT_SCHEMA,
        schema_field="schema",
    )
    path = _write_json(tmp_path / "old_submission_receipt.json", old_receipt)
    with pytest.raises(
        collector.CorrectedReplacementCollectionError,
        match="clean-library",
    ):
        collector.load_authority(path)
