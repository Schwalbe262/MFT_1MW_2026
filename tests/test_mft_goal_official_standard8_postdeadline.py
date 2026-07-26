from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official_standard8_postdeadline as official8


OBSERVED = datetime(2026, 7, 26, 11, 15, tzinfo=timezone.utc)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(official8.canonical_bytes(value) + b"\n")
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _input_contract_source() -> bytes:
    return b"""
KEYS = [
    "fan_velocity", "k_ins", "core_plate_pad_t", "wcp_pad_t",
    "fan_config", "core_plate_on", "wcp_on", "full_model",
]
CORE_MATERIAL_INPUT_KEYS = ()
THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS = ()
EFFICIENCY_EXPERIMENT_INPUT_KEYS = ()
ELECTROSTATIC_STAGE_INPUT_KEYS = ()
PHYSICS_METADATA_INPUT_KEYS = ()
ALL_INPUT_KEYS = [
    *KEYS,
    *CORE_MATERIAL_INPUT_KEYS,
    *THERMAL_CORE_CONDUCTIVITY_INPUT_KEYS,
    *EFFICIENCY_EXPERIMENT_INPUT_KEYS,
    *ELECTROSTATIC_STAGE_INPUT_KEYS,
    *PHYSICS_METADATA_INPUT_KEYS,
]
"""


def _fixture_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fan_velocity: float = 1.5,
) -> tuple[Path, Path, Any]:
    source_root = tmp_path / "source"
    decoded = {
        "fan_velocity": fan_velocity,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "fan_config": "dual",
        "core_plate_on": 1,
        "wcp_on": 1,
        "full_model": 0,
    }
    common_row = {
        "terminal_population_index": "15",
        "decoder_valid": "True",
        "surrogate_physical_valid": "True",
        "surrogate_physicality_passed": "True",
        "physical_geometry_sha256": official8.CANDIDATE_SHA256,
        "canonical_physical_params_sha256": "1" * 64,
        "candidate_physics_sha": official8.CANDIDATE_SHA256,
        "objective_volume_L": "791.2974319548481",
        "objective_total_loss_W": "5416.970664662574",
        "physical_constraint_feasible": "False",
        "physical_feasible": "False",
        "physical_G_json": '{"Llt_robust_band":0.7065464273657436}',
        "normalized_G_json": '{"Llt_robust_band":1.2846298679377155}',
        "coordinate_unit_json": "[0.1,0.2]",
        "decoded_physical_params_json": official8.canonical_bytes(
            decoded
        ).decode("ascii"),
        "source_seed": str(official8.SOURCE_SEED),
        "source_task_id": str(official8.SOURCE_TASK_ID),
        "source_bundle_id": official8.SOURCE_BUNDLE_ID,
        "source_island_id": "n1-6",
        "dataset_sha256": "2" * 64,
        "evaluation_model_sha256": "3" * 64,
        "constraint_spec_sha256": "4" * 64,
        "cooling_contract_sha256": "5" * 64,
        "operating_point_sha256": "6" * 64,
        "evaluation_model_artifacts_sha256": "7" * 64,
        "evaluation_model_generation_sha256": "8" * 64,
        "evaluation_spec_sha256": "9" * 64,
        "evaluation_temperature_contract_sha256": "a" * 64,
        "evaluation_hard_constraint_contract_sha256": "b" * 64,
    }
    source_csv = _write_csv(
        source_root / "terminal_physical_candidates.csv", [common_row]
    )
    source_result = official8.sealed(
        {
            "schema_version": "source-result-v1",
            "campaign_id": official8.CAMPAIGN_ID,
            "seed": official8.SOURCE_SEED,
            "task_payload_sha256": official8.SOURCE_BUNDLE_ID,
            "search_only_proposal": True,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
            "hard_spec": {
                "fixed_cooling_identity": {
                    "core_k_thermal": 2.0,
                    "core_plate_on": 1,
                    "core_plate_pad_t": 2.0,
                    "fan_config": "dual",
                    "fan_velocity": 1.5,
                    "k_ins": 0.2,
                    "thermal_pad_conductivity_W_mK": 0.2,
                    "wcp_on": 1,
                    "wcp_pad_t": 2.0,
                }
            },
            "artifact_inventory": {
                "terminal_physical_candidates": {
                    "path": source_csv.name,
                    "sha256": official8.sha256_file(source_csv),
                    "size_bytes": source_csv.stat().st_size,
                }
            },
            "terminal_physical_candidates_manifest": {
                "source_identity": {
                    "seed": official8.SOURCE_SEED,
                    "task_id": str(official8.SOURCE_TASK_ID),
                    "bundle_id": official8.SOURCE_BUNDLE_ID,
                }
            },
        }
    )
    source_result_path = _write_json(
        source_root / "result.json", source_result
    )
    source_result_sha = official8.sha256_file(source_result_path)
    official_row = {
        **common_row,
        "source_result_path": str(source_result_path),
        "source_result_sha256": source_result_sha,
        "standard_selection_order": "8",
        "standard_selection_roles": "objective_space_maximin",
        "standard_selection_basis": "near_feasible_fallback",
        "hard_feasible": "False",
        "feasible_rank": "-1",
        "global_non_dominated_rank": "-1",
    }
    rows = [official_row]
    for index in range(11):
        rows.append(
            {
                **official_row,
                "candidate_physics_sha": f"{index + 1:064x}",
                "physical_geometry_sha256": f"{index + 1:064x}",
                "standard_selection_order": str(index + 1),
            }
        )
    standard_csv = _write_csv(tmp_path / "standard_candidates.csv", rows)
    standard_sha = official8.sha256_file(standard_csv)
    manifest = official8.sealed(
        {
            "schema_version": "mft-goal-20260726-global-pareto-v1",
            "campaign_id": official8.CAMPAIGN_ID,
            "seed_count": 512,
            "standard_candidate_count": 12,
            "production_eligible": False,
            "automatic_promotion_allowed": False,
            "search_only_proposal": True,
            "artifacts": {
                "standard_candidates": {
                    "path": "standard_candidates.csv",
                    "row_count": 12,
                    "sha256": standard_sha,
                }
            },
        }
    )
    manifest_path = _write_json(tmp_path / "aggregate_manifest.json", manifest)
    monkeypatch.setattr(
        official8, "STANDARD_CANDIDATES_SHA256", standard_sha
    )
    monkeypatch.setattr(
        official8,
        "AGGREGATE_MANIFEST_SHA256",
        official8.sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        official8, "AGGREGATE_PAYLOAD_SHA256", manifest["payload_sha256"]
    )
    monkeypatch.setattr(
        official8, "SOURCE_RESULT_SHA256", source_result_sha
    )

    def source_reader(path: str, revision: str) -> bytes:
        assert path == official8.INPUT_PARAMETER_PATH
        assert revision == official8.SOLVER_REVISION
        return _input_contract_source()

    return standard_csv, manifest_path, source_reader


def _fake_payload_builder(
    _params: dict[str, Any], _profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dedupe = "mft-al:official8:fresh"
    command = (
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json; simulation_rc=$?;"
    )
    payload = {
        "name": official8.TASK_NAME,
        "project": official8.PROJECT,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "command": command,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": official8.CPUS,
        "memory_mb": official8.MEMORY_MB,
        "gpus": 0,
        "account_name": official8.ACCOUNT_NAME,
        "node_name": official8.NODE_NAME,
        "node_name_policy": "strict",
        "max_workers_per_node": official8.MAX_WORKERS_PER_NODE,
        "priority": official8.PRIORITY,
        "timeout_seconds": official8.SCHEDULER_SECONDS,
        "dedupe_key": dedupe,
    }
    retained = {
        "dedupe_key": dedupe,
        "artifact_path": "goal-fea-retained/fresh/symmetric.aedt",
        "results_path": "goal-fea-retained/fresh/symmetric.aedtresults",
        "transport": {
            "chunk_directory": (
                "goal-fea-retained/fresh/symmetric.aedt.chunks"
            )
        },
        "retention_required": True,
        "prune_protection_required": True,
    }
    environment = {
        "MFT_STANDALONE_CORE_CONTRACT": "mft-standalone-core-optin-v1",
        "MFT_STANDALONE_CORE_COUNT": "8",
        "MFT_STANDALONE_CORE_AUTH_SHA256": official8.CORE_AUTH_SHA256,
    }
    return payload, environment, retained


class FakeScheduler:
    def __init__(
        self,
        *,
        queue_state: str = "opening",
        active_fea: bool = False,
        collision: bool = False,
        relaxed: bool = False,
        pressure: str = "ok",
        licenses: int = 395,
    ) -> None:
        self.queue_state = queue_state
        self.active_fea = active_fea
        self.collision = collision
        self.relaxed = relaxed
        self.pressure = pressure
        self.licenses = licenses
        self.calls: list[tuple[str, list[tuple[str, Any]]]] = []

    def __call__(
        self, path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        normalized = list(query or [])
        self.calls.append((path, normalized))
        if path == "/api/health":
            return {
                "ok": True,
                "scheduler_ok": True,
                "scheduler_thread_alive": True,
                "scheduler_stalled": False,
            }
        if path == f"/api/tasks/{official8.SOURCE_TASK_ID}":
            return {
                "task_id": official8.SOURCE_TASK_ID,
                "name": official8.SOURCE_TASK_NAME,
                "status": "completed",
                "exit_code": 0,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "dedupe_key": official8.SOURCE_TASK_DEDUPE_KEY,
            }
        if path == "/api/task-capacity":
            assert ("memory_mb", 98304) in normalized
            assert ("max_workers_per_node", 1) in normalized
            assert ("account_name", "jji0930") in normalized
            assert ("node_name", "n114") in normalized
            assert ("node_name_policy", "strict") in normalized
            opening = self.queue_state == "opening"
            return {
                "fit_slots": 0 if opening else 1,
                "ready_fit_slots": 0 if opening else 1,
                "pending_fit_slots": 0,
                "inflight_fit_slots": 0,
                "memory_pressure_state": self.pressure,
                "preferred_node_relaxed": self.relaxed,
                "allocations": [] if opening else [{"allocation_id": 55}],
                "queue_state": self.queue_state,
                "queue_reason": (
                    official8.OPENING_QUEUE_REASON if opening else "ready"
                ),
                "standalone_aedt_available": self.licenses,
            }
        if path == "/api/tasks":
            if ("name_prefix", official8.TASK_NAME) in normalized:
                if self.collision:
                    return [
                        {
                            "task_id": 88,
                            "name": official8.TASK_NAME,
                            "dedupe_key": "mft-al:official8:fresh",
                        }
                    ]
                return []
            if self.active_fea:
                return [
                    {
                        "task_id": 999,
                        "status": "running",
                        "requested_node_name": "n114",
                        "aedt_backend": "standalone",
                        "scheduling_profile": "fea_bursty",
                    }
                ]
            return []
        raise AssertionError(f"unexpected GET: {path} {normalized}")


def test_authenticates_candidate_eight_and_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path, monkeypatch
    )
    selected, params = official8.authenticate_official_candidate(
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        source_reader=source_reader,
    )
    official8.validate_seal(selected, official8.CANDIDATE_SCHEMA)
    auth = selected["authentication"]
    assert auth["candidate_standard_selection_order"] == 8
    assert auth["source_task_id"] == 96141
    assert auth["source_terminal_population_index"] == 15
    assert params["fan_velocity"] == 1.5
    assert params["k_ins"] == 0.2


def test_candidate_or_cooling_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path, monkeypatch, fan_velocity=2.0
    )
    with pytest.raises(
        official8.PostdeadlineContractError,
        match="fixed cooling physics drifted",
    ):
        official8.authenticate_official_candidate(
            standard_candidates_path=standard,
            aggregate_manifest_path=manifest,
            source_reader=source_reader,
        )


def test_exact_strict_opening_query_and_empty_lane_are_required() -> None:
    scheduler = FakeScheduler()
    preflight = official8.live_preflight(
        reader=scheduler,
        expected_dedupe_key="fresh",
        observed_at=OBSERVED,
    )
    assert preflight["queue_state"] == "opening"
    assert preflight["active_n114_fea_count"] == 0
    assert preflight["preferred_node_relaxed"] is False
    capacity_call = next(
        call for call in scheduler.calls if call[0] == "/api/task-capacity"
    )
    assert ("account_name", "jji0930") in capacity_call[1]
    assert ("node_name", "n114") in capacity_call[1]
    assert ("node_name_policy", "strict") in capacity_call[1]


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        (
            FakeScheduler(queue_state="ready"),
            "opening capacity gate failed",
        ),
        (
            FakeScheduler(active_fea=True),
            "not FEA-empty",
        ),
        (
            FakeScheduler(collision=True),
            "already exists",
        ),
        (
            FakeScheduler(relaxed=True),
            "opening capacity gate failed",
        ),
        (
            FakeScheduler(pressure="high"),
            "opening capacity gate failed",
        ),
        (
            FakeScheduler(licenses=0),
            "opening capacity gate failed",
        ),
    ],
)
def test_live_opening_drift_fails_closed(
    scheduler: FakeScheduler, message: str
) -> None:
    with pytest.raises(official8.PostdeadlineContractError, match=message):
        official8.live_preflight(
            reader=scheduler,
            expected_dedupe_key="mft-al:official8:fresh",
            observed_at=OBSERVED,
        )


def test_prepare_seals_fixed_root_zero_post_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path / "authority", monkeypatch
    )
    output = tmp_path / "prepared"
    monkeypatch.setattr(official8, "OUTPUT_ROOT", output)
    plan_path = official8.prepare(
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        reader=FakeScheduler(),
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
        observed_at=OBSERVED,
    )
    plan = official8.validate_seal(
        official8.read_json(plan_path), official8.PLAN_SCHEMA
    )
    receipt = official8.validate_seal(
        official8.read_json(output / "prepare_receipt.json"),
        official8.PREPARE_SCHEMA,
    )
    assert plan["scheduler_submission_performed"] is False
    assert plan["official_standard_selection_order"] == 8
    assert plan["placement"]["account_name"] == "jji0930"
    assert plan["placement"]["node_name"] == "n114"
    assert plan["placement"]["node_name_policy"] == "strict"
    assert plan["placement"]["allocation_state_at_prepare"] == "opening"
    assert plan["placement"]["allocation_id_at_prepare"] is None
    assert (
        plan["opening_demand_pool_contract"]["relaxed_allocation_allowed"]
        is False
    )
    assert (
        Path(
            plan["single_attempt_contract"]["attempt_ledger_path"]
        ).resolve()
        == output / official8.ATTEMPT_LEDGER_NAME
    )
    assert (
        Path(
            plan["single_attempt_contract"]["submission_output_path"]
        ).resolve()
        == output / official8.SUBMISSION_DIRECTORY_NAME
    )
    assert receipt["scheduler_post_calls"] == 0
    assert receipt["queue_state_at_prepare"] == "opening"


def test_prepare_rejects_any_output_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(official8, "OUTPUT_ROOT", tmp_path / "fixed")
    with pytest.raises(
        official8.PostdeadlineContractError, match="root is not fixed"
    ):
        official8.prepare(output=tmp_path / "different")


def test_load_plan_reauthenticates_candidate_and_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path / "authority", monkeypatch
    )
    output = tmp_path / "prepared"
    monkeypatch.setattr(official8, "OUTPUT_ROOT", output)
    plan_path = official8.prepare(
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        reader=FakeScheduler(),
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
        observed_at=OBSERVED,
    )
    plan, params, profile = official8.load_plan(
        plan_path,
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
    )
    assert plan["candidate_physics_sha256"] == official8.CANDIDATE_SHA256
    assert params["fan_velocity"] == 1.5
    assert profile["fixed_boundary_contract"] == official8.FIXED_BOUNDARY


def test_reviewed_payload_rejects_relaxation() -> None:
    payload, _environment, retained = _fake_payload_builder({}, {})
    payload["node_name_policy"] = "preferred"
    with pytest.raises(
        official8.PostdeadlineContractError,
        match="payload/retention drifted",
    ):
        official8.validate_scheduler_payload(payload, retained)


def test_reviewed_contract_patch_restores_candidate_six_globals() -> None:
    old_task = official8.reviewed.TASK_NAME
    old_node = official8.reviewed.NODE_NAME
    with official8._reviewed_contract_patch():
        assert official8.reviewed.TASK_NAME == official8.TASK_NAME
        assert official8.reviewed.NODE_NAME == "n114"
    assert official8.reviewed.TASK_NAME == old_task
    assert official8.reviewed.NODE_NAME == old_node


class _Lock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> _Lock:
        self.events.append("lock-enter")
        return self

    def __exit__(self, *_args: Any) -> None:
        self.events.append("lock-exit")


def _readback(payload: dict[str, Any], *, relaxed: bool = False) -> dict[str, Any]:
    return {
        "task_id": 12345,
        "name": official8.TASK_NAME,
        "dedupe_key": payload["dedupe_key"],
        "project": official8.PROJECT,
        "requested_account_name": official8.ACCOUNT_NAME,
        "requested_node_name": official8.NODE_NAME,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": official8.CPUS,
        "memory_mb": official8.MEMORY_MB,
        "timeout_seconds": official8.SCHEDULER_SECONDS,
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
        "status": "queued",
        "account_name": "",
        "node_name": "",
        "node_name_policy": "strict",
        "preferred_node_relaxed": relaxed,
    }


def test_submit_consumes_fixed_ledger_before_one_locked_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    monkeypatch.setattr(official8, "OUTPUT_ROOT", root)
    _write_json(root / official8.PLAN_NAME, {"plan": True})
    output = root / official8.SUBMISSION_DIRECTORY_NAME
    payload, _environment, _retained = _fake_payload_builder({}, {})
    plan = {
        "payload_sha256": "c" * 64,
        "scheduler_payload": payload,
        "scheduler_payload_sha256": official8.payload_sha256(payload),
        "dedupe_key": payload["dedupe_key"],
        "single_attempt_contract": {
            "attempt_ledger_path": str(
                root / official8.ATTEMPT_LEDGER_NAME
            ),
            "submission_output_path": str(output),
            "attempt_nonce": "d" * 64,
        },
    }
    events: list[str] = []
    scheduler = FakeScheduler()
    post_calls = 0

    def reader(
        path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        if path == "/api/tasks/12345":
            events.append("readback")
            return _readback(payload)
        result = scheduler(path, query)
        if path == "/api/task-capacity":
            events.append("locked-or-initial-get")
        return result

    def poster(
        _url: str, _payload: Mapping[str, Any]
    ) -> tuple[int, dict[str, int], None]:
        nonlocal post_calls
        post_calls += 1
        events.append("post")
        assert (root / official8.ATTEMPT_LEDGER_NAME).is_file()
        return 201, {"task_id": 12345}, None

    final_seal = official8.submit(
        authorize_post=official8.POST_AUTHORIZATION,
        reader=reader,
        poster=poster,
        observed_at=OBSERVED,
        lock_factory=lambda: _Lock(events),
        plan_loader=lambda _path: (plan, {}, {}),
    )
    assert final_seal.is_file()
    assert post_calls == 1
    assert events.count("locked-or-initial-get") == 2
    assert events.index("lock-enter") < events.index("post")
    assert events.index("post") < events.index("lock-exit")
    attempt = official8.read_json(
        root / official8.ATTEMPT_LEDGER_NAME
    )
    assert attempt["post_call_consumed_before_network"] is True
    assert attempt["campaign_mutation_lock_acquired"] is True

    with pytest.raises(
        official8.PostdeadlineContractError, match="already consumed"
    ):
        official8.submit(
            authorize_post=official8.POST_AUTHORIZATION,
            reader=reader,
            poster=poster,
            observed_at=OBSERVED,
            lock_factory=lambda: _Lock(events),
            plan_loader=lambda _path: (plan, {}, {}),
        )
    assert post_calls == 1


def test_submit_rejects_relaxed_durable_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    monkeypatch.setattr(official8, "OUTPUT_ROOT", root)
    _write_json(root / official8.PLAN_NAME, {"plan": True})
    payload, _environment, _retained = _fake_payload_builder({}, {})
    output = root / official8.SUBMISSION_DIRECTORY_NAME
    plan = {
        "payload_sha256": "c" * 64,
        "scheduler_payload": payload,
        "scheduler_payload_sha256": official8.payload_sha256(payload),
        "dedupe_key": payload["dedupe_key"],
        "single_attempt_contract": {
            "attempt_ledger_path": str(
                root / official8.ATTEMPT_LEDGER_NAME
            ),
            "submission_output_path": str(output),
            "attempt_nonce": "d" * 64,
        },
    }
    scheduler = FakeScheduler()

    def reader(
        path: str, query: Sequence[tuple[str, Any]] | None
    ) -> Any:
        if path == "/api/tasks/12345":
            return _readback(payload, relaxed=True)
        return scheduler(path, query)

    with pytest.raises(
        official8.PostdeadlineContractError,
        match="readback relaxed or drifted",
    ):
        official8.submit(
            authorize_post=official8.POST_AUTHORIZATION,
            reader=reader,
            poster=lambda *_args: (201, {"task_id": 12345}, None),
            observed_at=OBSERVED,
            lock_factory=lambda: _Lock([]),
            plan_loader=lambda _path: (plan, {}, {}),
        )


def test_submit_requires_exact_token_before_loading_or_posting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(official8, "OUTPUT_ROOT", tmp_path / "fixed")
    calls = 0

    def poster(
        _url: str, _payload: Mapping[str, Any]
    ) -> tuple[int, dict[str, int], None]:
        nonlocal calls
        calls += 1
        return 201, {"task_id": 12345}, None

    with pytest.raises(
        official8.PostdeadlineContractError, match="authorization is absent"
    ):
        official8.submit(
            authorize_post="wrong",
            poster=poster,
        )
    assert calls == 0
