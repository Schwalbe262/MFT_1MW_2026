from __future__ import annotations

import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from module.input_parameter_260706 import ALL_INPUT_KEYS
from tools import mft_goal_standard_full_continuation as continuation


CANDIDATE = "a" * 64
SOLVER = "b" * 40
LIBRARY = "c" * 40
TASK_ID = 96325
SOURCE_NODE = "n107"
TARGET_LANE = continuation.StrictLane("dw16", "n113")


def _json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _license(path: Path) -> Path:
    return _json(
        path,
        {
            "schema": "mft-aedt-license-headroom-snapshot-v1",
            "server_up": True,
            "server": "1055@172.16.10.81",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "features": {
                "anshpc": {"total": 100, "used": 0},
                "elec_solve_maxwell": {"total": 10, "used": 0},
                "electronics_desktop": {"total": 10, "used": 0},
                "electronics3d_gui": {"total": 10, "used": 0},
            },
        },
    )


def _row(
    *,
    candidate: str = CANDIDATE,
    task_id: int = TASK_ID,
    volume: float = 800.0,
    loss: float = 5000.0,
    passed: bool = True,
    feasible_rank: int = 0,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "candidate_physics_sha256": candidate,
        "actual_volume_L": volume,
        "actual_total_loss_W": loss,
        "actual_width_mm": 1100.0,
        "actual_length_mm": 900.0,
        "actual_height_mm": 700.0,
        "actual_resonance_Hz": 16000.0,
        "actual_winding_max_C": 90.0,
        "actual_core_max_C": 110.0,
        "measured_hard_constraints_passed": passed,
        "strict_al_row_eligible": True,
        "audit_non_dominated_rank": 0,
        "hard_feasible_non_dominated_rank": feasible_rank,
        "source_seed": 2607260001,
        "source_fixed_primary_turns": 6,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "physics_data_revision": "physics-v1",
    }


def _fixed() -> dict[str, Any]:
    return {
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
        "core_plate_on": 1,
        "wcp_on": 1,
    }


def _measured() -> dict[str, Any]:
    evidence = {
        "width_mm": {
            "actual": 1100.0,
            "limit": 1200.0,
            "relation": "<=",
            "margin": 100.0,
            "passed": True,
        },
        "length_mm": {
            "actual": 900.0,
            "limit": 1000.0,
            "relation": "<=",
            "margin": 100.0,
            "passed": True,
        },
        "height_mm": {
            "actual": 700.0,
            "limit": 750.0,
            "relation": "<=",
            "margin": 50.0,
            "passed": True,
        },
        "resonance_Hz": {
            "actual": 16000.0,
            "limit": 15000.0,
            "relation": ">=",
            "margin": 1000.0,
            "passed": True,
        },
        "winding_max_C": {
            "actual": 90.0,
            "limit": 100.0,
            "relation": "<=",
            "margin": 10.0,
            "passed": True,
        },
        "core_max_C": {
            "actual": 110.0,
            "limit": 120.0,
            "relation": "<=",
            "margin": 10.0,
            "passed": True,
        },
    }
    return {
        "task_id": TASK_ID,
        "candidate_physics_sha256": CANDIDATE,
        "source_seed": 2607260001,
        "source_fixed_primary_turns": 6,
        "solver_revision": SOLVER,
        "library_revision": LIBRARY,
        "physics_data_revision": "physics-v1",
        "actual_volume_L": 800.0,
        "actual_total_loss_W": 5000.0,
        "actual_loss_components_W": {
            "P_winding_total": 5000.0,
            "P_core_total": 0.0,
            "P_core_plate_total": 0.0,
            "P_wcp_total": 0.0,
        },
        "actual_dimensions_mm": {"W": 1100.0, "L": 900.0, "H": 700.0},
        "actual_resonance_Hz": 16000.0,
        "actual_winding_max_C": 90.0,
        "actual_core_max_C": 110.0,
        "active_temperature_targets": [
            "T_max_Tx",
            "T_max_Rx_main",
            "T_max_core",
        ],
        "actual_temperature_targets": {},
        "fixed_identity_attestation": {
            "contract_schema": "goal-contract-v1",
            "goal_spec_sha256": "9" * 64,
            "expected": _fixed(),
            "observed": _fixed(),
            "mismatches": [],
            "attested": True,
            "sha256": "8" * 64,
        },
        "hard_constraint_evidence": evidence,
        "measured_hard_constraints_passed": True,
        "campaign_full_physical_spec_reasons": [],
        "campaign_full_physical_spec_passed": True,
        "payload_sha256": "d" * 64,
    }


def _authority_fixture(tmp_path: Path) -> tuple[Any, ...]:
    collection = _json(tmp_path / "collection" / "collection_receipt.json", {})
    snapshot_manifest = _json(
        tmp_path / "snapshot" / "snapshot_manifest.json", {"snapshot": True}
    )
    observations = _json(
        snapshot_manifest.parent / "measured_actual_observations.csv",
        {"placeholder": True},
    )
    state = {
        "pending_count": 0,
        "collection_count": 1,
        "lanes": [
            {
                "task_id": TASK_ID,
                "status": "collection_ready",
                "collection_root": str(collection.parent),
            }
        ],
    }
    snapshot = {
        "payload_sha256": "e" * 64,
        "ranked_rows": [_row()],
        "measured_actual_observations_csv": {
            **continuation._file_record(observations),
            "row_count": 1,
            "columns": [],
        },
    }
    view = {
        "plan": {
            "solver_revision": SOLVER,
            "library_revision": LIBRARY,
        },
        "params": {key: index for index, key in enumerate(sorted(ALL_INPUT_KEYS))},
        "collection": {
            "source_collection_receipt_payload_sha256": "f" * 64,
        },
    }
    measured = _measured()
    return (
        state,
        snapshot,
        _row(),
        collection,
        view,
        measured,
        SOURCE_NODE,
    )


def _profile() -> tuple[dict[str, Any], dict[str, Any]]:
    profile = {
        "schema_version": "profile-v1",
        "stage": "full",
        "param_overrides": {"full_model": 1},
        "fixed_boundary_contract": _fixed(),
        "artifact_retention": {
            "stage": "full",
            "artifact_filename": "full_model.aedt",
        },
        "cpus": 16,
        "mem_mb": 98304,
        "timeout_seconds": 43200,
    }
    return profile, {"path": "reviewed", "sha256": "1" * 64, "size_bytes": 1}


def _payload(
    params: dict[str, Any],
    _profile_value: dict[str, Any],
    task_name: str,
    _workdir: str,
    _solver: str,
    _library: str,
    lane: continuation.StrictLane,
    _license_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    assert set(params) == set(ALL_INPUT_KEYS)
    payload = {
        "name": task_name,
        "project": continuation.PROJECT,
        "command": "bounded Full command",
        "cpus": 16,
        "memory_mb": 98304,
        "account_name": lane.account_name,
        "node_name": lane.node_name,
        "node_name_policy": "strict",
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
        "timeout_seconds": 86400,
        "dedupe_key": f"dedupe:{task_name}",
    }
    retained = {
        "stage": "full",
        "artifact_path": f"goal-fea-retained/{task_name}/full_model.aedt",
        "dedupe_key": payload["dedupe_key"],
    }
    return payload, retained, {"runtime_license_refresh_required": True}


def _patch_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authority: tuple[Any, ...],
) -> Path:
    license_path = _license(tmp_path / "licenses" / "one" / "snapshot.json")
    monkeypatch.setattr(
        continuation, "_candidate_authority", lambda _path: authority
    )
    monkeypatch.setattr(continuation.promotion, "_full_profile", _profile)
    monkeypatch.setattr(
        continuation.fastlane,
        "_latest_license_snapshot",
        lambda _directory: license_path,
    )
    monkeypatch.setattr(
        continuation,
        "_lane_preflight",
        lambda **_kwargs: {
            "schema_version": "preflight-v1",
            "collision_count": 0,
            "separate_strict_node_lane": True,
        },
    )
    return license_path


def test_best_measured_candidate_is_feasible_rank0_front_order() -> None:
    snapshot = {
        "ranked_rows": [
            _row(candidate="b" * 64, task_id=1, volume=900, loss=1),
            _row(candidate="c" * 64, task_id=2, volume=700, loss=9),
            _row(
                candidate="d" * 64,
                task_id=3,
                volume=600,
                loss=0,
                passed=False,
                feasible_rank=-1,
            ),
            _row(
                candidate="e" * 64,
                task_id=4,
                volume=500,
                loss=0,
                feasible_rank=1,
            ),
        ]
    }

    selected = continuation._best_measured_row(snapshot)

    assert selected is not None
    assert selected["candidate_physics_sha256"] == "c" * 64


@pytest.mark.parametrize(
    ("pending", "snapshot", "status"),
    [
        (3, None, "pending_standard_collections"),
        (0, {"ranked_rows": []}, "terminal_no_measured_pass"),
    ],
)
def test_pending_or_infeasible_cycle_performs_zero_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending: int,
    snapshot: dict[str, Any] | None,
    status: str,
) -> None:
    calls = {"post": 0, "payload": 0}
    authority = (
        {"pending_count": pending, "collection_count": 0, "lanes": []},
        snapshot,
        None,
        None,
        None,
        None,
        None,
    )
    monkeypatch.setattr(
        continuation, "_candidate_authority", lambda _path: authority
    )

    def no_payload(*_args: Any, **_kwargs: Any) -> Any:
        calls["payload"] += 1
        raise AssertionError("payload must not be built")

    def no_post(*_args: Any, **_kwargs: Any) -> Any:
        calls["post"] += 1
        raise AssertionError("POST must not occur")

    state = continuation.cycle(
        state_path=tmp_path / "upstream.json",
        output_root=tmp_path / "output",
        strict_lanes=[TARGET_LANE],
        license_snapshot_directory=tmp_path / "licenses",
        authorization=continuation.AUTHORIZATION_TOKEN,
        payload_builder=no_payload,
        post_once=no_post,
    )

    assert state["status"] == status
    assert state["scheduler_post_attempts_consumed"] == 0
    assert state["scientific_pass_claimed"] is False
    assert state["promotion_completed"] is False
    assert calls == {"post": 0, "payload": 0}
    assert not (tmp_path / "output" / "scheduler_post_attempt.json").exists()


def test_prepare_seals_same_candidate_full_payload_without_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority_fixture(tmp_path)
    _patch_prepare(monkeypatch, tmp_path, authority)

    status, plan_path, plan = continuation.prepare(
        state_path=tmp_path / "upstream.json",
        output_root=tmp_path / "output",
        strict_lanes=[TARGET_LANE],
        license_snapshot_directory=tmp_path / "licenses",
        payload_builder=_payload,
    )

    assert status == "prepared_full_continuation"
    assert plan_path is not None
    assert continuation.load_plan(plan_path) == plan
    assert plan["candidate_physics_sha256"] == CANDIDATE
    assert plan["standard_task_id"] == TASK_ID
    assert plan["standard_source_node"] == SOURCE_NODE
    assert plan["target_lane"]["node_name"] != SOURCE_NODE
    assert plan["scheduler_payload"]["cpus"] == 16
    assert plan["scheduler_payload"]["memory_mb"] == 98304
    assert plan["scheduler_payload"]["timeout_seconds"] == 86400
    assert plan["measured_standard_hard_constraints_passed"] is True
    assert plan["fixed_cooling_unchanged"] is True
    assert plan["full_result_available"] is False
    assert plan["scientific_pass_claimed"] is False
    assert plan["promotion_completed"] is False
    assert not (tmp_path / "output" / "scheduler_post_attempt.json").exists()


def test_one_shot_submit_writes_attempt_before_post_and_never_reposts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority_fixture(tmp_path)
    license_path = _patch_prepare(monkeypatch, tmp_path, authority)
    _status, plan_path, plan = continuation.prepare(
        state_path=tmp_path / "upstream.json",
        output_root=tmp_path / "output",
        strict_lanes=[TARGET_LANE],
        license_snapshot_directory=tmp_path / "licenses",
        payload_builder=_payload,
    )
    assert plan_path is not None
    monkeypatch.setattr(
        continuation.production,
        "_validate_license_snapshot",
        lambda _path: ("{}", "2" * 64),
    )
    calls = {"post": 0}

    def post_once(_url: str, _payload_value: Any) -> tuple[int, Any]:
        calls["post"] += 1
        assert continuation._attempt_path(tmp_path / "output").exists()
        return 201, {"task_id": 97001}

    def reader(path: str, _query: Any = None) -> dict[str, Any]:
        assert path == "/api/tasks/97001"
        return {
            "task_id": 97001,
            "name": plan["task_name"],
            "dedupe_key": plan["scheduler_payload"]["dedupe_key"],
            "project": continuation.PROJECT,
            "requested_node_name": TARGET_LANE.node_name,
            "requested_node_name_policy": "strict",
            "cpus": 16,
            "memory_mb": 98304,
        }

    receipt = continuation.submit_prepared(
        plan_path=plan_path,
        state_path=tmp_path / "upstream.json",
        license_snapshot_directory=license_path.parents[1],
        authorization=continuation.AUTHORIZATION_TOKEN,
        reader=reader,
        post_once=post_once,
        lock_factory=nullcontext,
    )
    replay = continuation.submit_prepared(
        plan_path=plan_path,
        state_path=tmp_path / "upstream.json",
        license_snapshot_directory=license_path.parents[1],
        authorization=continuation.AUTHORIZATION_TOKEN,
        reader=reader,
        post_once=post_once,
        lock_factory=nullcontext,
    )

    assert calls["post"] == 1
    assert receipt == replay
    assert receipt["status"] == "full_submitted_pending_actual_result"
    assert receipt["full_task_id"] == 97001
    assert receipt["scheduler_post_attempts_consumed"] == 1
    assert receipt["scientific_pass_claimed"] is False
    assert receipt["full_actual_constraints_passed"] is False
    assert receipt["promotion_completed"] is False


def test_ambiguous_post_consumes_attempt_and_second_call_is_forbidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority_fixture(tmp_path)
    license_path = _patch_prepare(monkeypatch, tmp_path, authority)
    _status, plan_path, _plan = continuation.prepare(
        state_path=tmp_path / "upstream.json",
        output_root=tmp_path / "output",
        strict_lanes=[TARGET_LANE],
        license_snapshot_directory=tmp_path / "licenses",
        payload_builder=_payload,
    )
    assert plan_path is not None
    monkeypatch.setattr(
        continuation.production,
        "_validate_license_snapshot",
        lambda _path: ("{}", "2" * 64),
    )
    calls = {"post": 0}

    def ambiguous(_url: str, _payload_value: Any) -> tuple[int, Any]:
        calls["post"] += 1
        raise TimeoutError("ambiguous network outcome")

    with pytest.raises(TimeoutError, match="ambiguous"):
        continuation.submit_prepared(
            plan_path=plan_path,
            state_path=tmp_path / "upstream.json",
            license_snapshot_directory=license_path.parents[1],
            authorization=continuation.AUTHORIZATION_TOKEN,
            reader=lambda *_args, **_kwargs: {},
            post_once=ambiguous,
            lock_factory=nullcontext,
        )
    with pytest.raises(continuation.ContinuationError, match="already consumed"):
        continuation.submit_prepared(
            plan_path=plan_path,
            state_path=tmp_path / "upstream.json",
            license_snapshot_directory=license_path.parents[1],
            authorization=continuation.AUTHORIZATION_TOKEN,
            reader=lambda *_args, **_kwargs: {},
            post_once=ambiguous,
            lock_factory=nullcontext,
        )

    assert calls["post"] == 1
    assert continuation._attempt_path(tmp_path / "output").exists()
    assert not continuation._receipt_path(tmp_path / "output").exists()


def test_source_collection_rejects_cooling_or_same_candidate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_root = tmp_path / "collection"
    terminal = _json(
        collection_root / "scheduler_terminal_task.json",
        {
            "task_id": TASK_ID,
            "strict_node_placement": True,
            "placement_contract_satisfied": True,
            "actual_node_name": SOURCE_NODE,
        },
    )
    receipt = _json(
        collection_root / "collection_receipt.json",
        {
            "source_files": {
                "scheduler_terminal_task.json": {
                    "path": terminal.name,
                    "sha256": continuation._file_record(terminal)["sha256"],
                    "size_bytes": terminal.stat().st_size,
                }
            }
        },
    )
    state = {
        "lanes": [
            {
                "task_id": TASK_ID,
                "collection_root": str(collection_root),
            }
        ]
    }
    view = {"collection": {}, "params": {}, "plan": {}}
    monkeypatch.setattr(
        continuation.postsuccess,
        "authenticate_collection",
        lambda _path: view,
    )
    measured = _measured()
    monkeypatch.setattr(
        continuation.postsuccess,
        "_measured_classification",
        lambda _view: measured,
    )

    selected = continuation._source_collection(state, _row())
    assert selected[0] == receipt.resolve()
    assert selected[-1] == SOURCE_NODE

    measured["fixed_identity_attestation"]["observed"][
        "fan_velocity_m_s"
    ] = 1.6
    with pytest.raises(continuation.ContinuationError, match="cooling"):
        continuation._source_collection(state, _row())


def test_full_lane_must_differ_from_standard_source_node() -> None:
    with pytest.raises(continuation.ContinuationError, match="equals"):
        continuation._lane_preflight(
            lane=continuation.StrictLane("dw16", SOURCE_NODE),
            source_node=SOURCE_NODE,
            task_name="task",
            dedupe_key="dedupe",
            source_task_id=TASK_ID,
            reader=lambda *_args, **_kwargs: {},
        )


def test_tool_keeps_scheduler_project_separate_and_only_one_post_boundary() -> None:
    source = Path(continuation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "urlopen"
    ]

    assert not any("slurm_scheduler" in name for name in imports)
    assert source.count('method="POST"') == 1
    assert len(calls) == 2  # one GET transport and one POST transport
    assert "scheduler_repository_modified" in source
