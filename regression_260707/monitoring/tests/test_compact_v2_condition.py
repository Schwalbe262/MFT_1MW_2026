import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import regression_260707.monitoring.readers as readers_module
from regression_260707.monitoring.app import create_app
from regression_260707.monitoring.readers import ArtifactService
from regression_260707.monitoring.tests.test_readers import (
    _final_goal_current7_contract,
    _isolate_continuous_nsga_sources,
    _publish_current7_runtime,
)
from tools.tier1_final1000_multiseed_contract import (
    PROTOCOL_VERSION,
)
from tools.tier1_final1000_multiseed_status import (
    build_compact_snapshot,
    publish_compact_snapshot,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _publish_compact_condition(
    root: Path,
    *,
    batch4: bool = False,
    hard_spec: dict[str, object] | None = None,
    constraint_names: list[str] | None = None,
    constraint_version: str = ("1000x1000x750-resmax20k-t100-core4-cw1-5-lmhalf"),
) -> dict[str, object]:
    if hard_spec is None or constraint_names is None:
        default_spec, default_names = _final_goal_current7_contract()
        hard_spec = hard_spec or default_spec
        constraint_names = constraint_names or default_names
    legacy = _publish_current7_runtime(
        root / "legacy-source",
        constraint_version_override=constraint_version,
        hard_spec_override=hard_spec,
        constraint_names_override=constraint_names,
        candidate_overrides={
            "size_W_mm": 950.0,
            "size_L_mm": 940.0,
            "size_H_mm": 700.0,
            "volume_L": 625.1,
            "pred_f_res_tx_screen_Hz": 18_200.0,
            "pred_f_res_rx_screen_Hz": 18_100.0,
            "pred_f_res_min_screen_Hz": 18_100.0,
            "physical_constraint_G": {name: -1.0 for name in constraint_names},
        },
    )
    status = _read(legacy["status_path"])
    records = copy.deepcopy(status["terminal_results"])
    task_id = 8_801 if batch4 else 7_001
    records[0]["task_ids"] = [task_id]
    if batch4:
        records[0].update(
            {
                "physical_parent_task_id": task_id,
                "batch_ordinal": 0,
            }
        )
        second = copy.deepcopy(records[0])
        second.update({"seed": 102, "batch_ordinal": 1})
        second["record_sha256"] = readers_module._canonical_json_sha256(
            {key: value for key, value in second.items() if key != "record_sha256"}
        )
        records.append(second)
        status["aggregate"]["authenticated_seed_count"] = 2
    records[0]["record_sha256"] = readers_module._canonical_json_sha256(
        {key: value for key, value in records[0].items() if key != "record_sha256"}
    )

    static = copy.deepcopy(status)
    for key in (
        "schema_version",
        "updated_at",
        "scheduler_task_count",
        "state_counts",
        "latest_tasks",
        "authenticated_terminal_seed_count",
        "terminal_results",
    ):
        static.pop(key, None)
    stage_id = "final-1000-t100"
    static["final_goal_stage_id"] = stage_id
    physical = [
        {
            "task_id": task_id,
            "protocol_version": (
                PROTOCOL_VERSION if batch4 else "final1000-single-seed-v1"
            ),
            "state": "running" if batch4 else "completed",
            "account_name": "synthetic-account",
            "batch_length": 4 if batch4 else 1,
            "current_seed": 102 if batch4 else 101,
            "sealed_child_count": 2 if batch4 else 1,
            "started_at": "2026-07-22T00:00:00+00:00",
            "updated_at": "2026-07-22T00:01:00+00:00",
        }
    ]
    snapshot = build_compact_snapshot(
        stage_id=stage_id,
        physical_lanes=physical,
        seed_records=records,
        frontend_static=static,
        shard_size=1,
        updated_at="2026-07-22T00:01:00+00:00",
    )
    compact_root = root / "compact" / "canonical"
    writes = publish_compact_snapshot(
        compact_root,
        *snapshot,
        pointer_name="current7-index.json",
    )
    assert writes > 0
    return {
        "index_path": compact_root / "current7-index.json",
        "root": compact_root,
        "snapshot": snapshot,
        "physical_task_id": task_id,
    }


def test_compact_condition_inventory_authenticates_without_hydrating_shards(
    tmp_path,
    monkeypatch,
):
    compact = _publish_compact_condition(tmp_path / "condition")
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(compact["index_path"]),
    )
    monkeypatch.setattr(
        readers_module,
        "adapt_compact_condition_index",
        lambda *_args, **_kwargs: pytest.fail(
            "dashboard inventory must not hydrate compact shards"
        ),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]
    records = service._current7_condition_generation_records([diagnostics])

    assert diagnostics["integrity_verified"] is True
    assert diagnostics["detail_hydrated"] is False
    assert diagnostics["condition_index_schema_version"] == (
        readers_module.CURRENT7_COMPACT_INDEX_SCHEMA
    )
    assert diagnostics["physical_lane_count"] == 1
    assert diagnostics["logical_seed_count"] == 1
    assert diagnostics["physical_task_ids"] == [7_001]
    assert diagnostics["virtual_scheduler_task_ids_created"] is False
    assert records[0]["selectable"] is True
    assert records[0]["detail_integrity_pending"] is True


def test_compact_condition_detail_pages_records_into_existing_renderer(
    tmp_path,
    monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "primary")
    compact = _publish_compact_condition(tmp_path / "condition")
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"]))
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(compact["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    calls: list[tuple[int, int | None]] = []
    adapter = readers_module.adapt_compact_condition_index

    def traced(path, *, offset=0, limit=None):
        calls.append((offset, limit))
        return adapter(path, offset=offset, limit=limit)

    monkeypatch.setattr(readers_module, "adapt_compact_condition_index", traced)
    payload = service._continuous_nsga2()
    generation = next(
        item
        for item in payload["pareto_generations"]
        if item.get("condition_index_schema_version")
        == readers_module.CURRENT7_COMPACT_INDEX_SCHEMA
    )

    detail = service.nsga2_generation(generation["id"])

    assert calls == [(0, readers_module.CURRENT7_COMPACT_DETAIL_PAGE_SIZE)]
    assert detail["available"] is True
    assert detail["integrity_verified"] is True
    assert detail["result_scope"] == "read_only_condition_generation"
    assert detail["valid_candidate_count"] == 1
    assert detail["candidates"][0]["spec_status"] == "pass"
    assert detail["candidates"][0]["gui_launch_eligible"] is False
    search = detail["tier1_current7_search"]
    assert search["detail_hydrated"] is True
    assert search["physical_task_ids"] == [7_001]
    assert search["virtual_scheduler_task_ids_created"] is False
    assert (
        len(json.dumps(detail, ensure_ascii=False).encode("utf-8")) < 32 * 1024 * 1024
    )


def test_compact_batch4_keeps_one_physical_parent_and_four_logical_seeds(
    tmp_path,
    monkeypatch,
):
    compact = _publish_compact_condition(tmp_path / "condition-batch4", batch4=True)
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(compact["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    metadata = service._tier1_current7_condition_diagnostics()[0]
    hydrated = service._tier1_compact_condition_diagnostics_locked(
        index_path=compact["index_path"],
        hydrate_candidates=True,
    )

    assert metadata["physical_lane_count"] == 1
    assert metadata["logical_seed_count"] == 4
    assert metadata["logical_sealed_seed_count"] == 2
    assert metadata["logical_unsealed_seed_count"] == 2
    assert metadata["physical_task_ids"] == [8_801]
    assert len(metadata["lanes"]) == 1
    assert metadata["lanes"][0]["task_id"] == 8_801
    assert hydrated["authenticated_terminal_seed_count"] == 2
    assert hydrated["virtual_scheduler_task_ids_created"] is False


def test_compact_condition_is_exposed_by_existing_nsga_api_flow(
    tmp_path,
    monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "primary-api")
    compact = _publish_compact_condition(tmp_path / "condition-api")
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"]))
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(compact["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)
    client = TestClient(create_app(service=service))

    inventory_response = client.get("/api/nsga2")
    assert inventory_response.status_code == 200
    inventory = inventory_response.json()
    generation = next(
        item
        for item in inventory["pareto_generations"]
        if item.get("condition_index_schema_version")
        == readers_module.CURRENT7_COMPACT_INDEX_SCHEMA
    )
    detail_response = client.get(generation["candidate_endpoint"])

    assert detail_response.status_code == 200
    assert len(detail_response.content) < 32 * 1024 * 1024
    detail = detail_response.json()
    assert detail["integrity_verified"] is True
    assert detail["valid_candidate_count"] == 1
    assert detail["generation"]["id"] == generation["id"]
    assert detail["tier1_current7_search"]["physical_task_ids"] == [7_001]


def test_compact_configured_index_corruption_fails_closed(
    tmp_path,
    monkeypatch,
):
    compact = _publish_compact_condition(tmp_path / "condition-corrupt-index")
    index_path = compact["index_path"]
    index = _read(index_path)
    index["stage_id"] = "tampered-stage"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(index_path),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()[0]

    assert diagnostics["available"] is True
    assert diagnostics["integrity_verified"] is False
    assert diagnostics["authority_eligible"] is False
    assert any("rejected" in warning for warning in diagnostics["warnings"])
    assert service._current7_condition_generation_records([diagnostics]) == []


def test_compact_shard_corruption_is_rejected_only_when_detail_is_selected(
    tmp_path,
    monkeypatch,
):
    primary = _publish_current7_runtime(tmp_path / "primary")
    compact = _publish_compact_condition(tmp_path / "condition-corrupt-shard")
    index = _read(compact["index_path"])
    manifest = _read(compact["root"] / index["seed_result_shards"]["path"])
    shard = compact["root"] / manifest["shards"][0]["path"]
    shard.chmod(0o644)
    shard.write_bytes(shard.read_bytes() + b" ")
    monkeypatch.setenv(readers_module.CURRENT7_INDEX_ENV, str(primary["index_path"]))
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        str(compact["index_path"]),
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)
    _isolate_continuous_nsga_sources(monkeypatch, service)

    metadata = service._tier1_current7_condition_diagnostics()[0]
    generation = service._current7_condition_generation_records([metadata])[0]
    detail = service.nsga2_generation(generation["id"])

    assert metadata["integrity_verified"] is True
    assert metadata["detail_hydrated"] is False
    assert detail["available"] is False
    assert detail["integrity_verified"] is False
    assert detail["candidates"] == []
    assert any(
        "file/reference seal mismatch" in warning for warning in detail["warnings"]
    )


def test_v1_and_compact_condition_indexes_coexist_without_identity_merge(
    tmp_path,
    monkeypatch,
):
    compact = _publish_compact_condition(tmp_path / "compact-final")
    entry_spec, entry_names = _final_goal_current7_contract()
    entry_spec.update(
        {
            "T_limit_C": 125.0,
            "size_W_max_mm": 1_200.0,
            "size_L_max_mm": 1_200.0,
        }
    )
    v1 = _publish_current7_runtime(
        tmp_path / "v1-entry",
        constraint_version_override=("1200x1200x750-resmax20k-t125-core4-cw1-5-lmhalf"),
        hard_spec_override=entry_spec,
        constraint_names_override=entry_names,
        candidate_overrides={
            "physical_constraint_G": {name: -1.0 for name in entry_names}
        },
    )
    monkeypatch.setenv(
        readers_module.CURRENT7_CONDITION_INDEXES_ENV,
        f"{v1['index_path']};{compact['index_path']}",
    )
    service = ArtifactService(tmp_path / "regression", record_runtime=False)

    diagnostics = service._tier1_current7_condition_diagnostics()
    records = service._current7_condition_generation_records(diagnostics)

    assert len(diagnostics) == 2
    assert all(item["integrity_verified"] is True for item in diagnostics)
    assert {record["condition_index_schema_version"] for record in records} == {
        readers_module.CURRENT7_INDEX_SCHEMA,
        readers_module.CURRENT7_COMPACT_INDEX_SCHEMA,
    }
    assert len(records) == 2
