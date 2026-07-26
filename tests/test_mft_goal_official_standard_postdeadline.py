from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_official_standard_postdeadline as official


OBSERVED = datetime(2026, 7, 26, 10, 30, tzinfo=timezone.utc)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(official.canonical_bytes(value) + b"\n")
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
        "terminal_population_index": "45",
        "decoder_valid": "True",
        "surrogate_physical_valid": "True",
        "surrogate_physicality_passed": "True",
        "physical_geometry_sha256": official.CANDIDATE_SHA256,
        "canonical_physical_params_sha256": "1" * 64,
        "candidate_physics_sha": official.CANDIDATE_SHA256,
        "objective_volume_L": "815.731705573816",
        "objective_total_loss_W": "5354.017853139445",
        "physical_constraint_feasible": "False",
        "physical_feasible": "False",
        "physical_G_json": '{"Llt_robust_band":0.7}',
        "normalized_G_json": '{"Llt_robust_band":1.2}',
        "coordinate_unit_json": "[0.1,0.2]",
        "decoded_physical_params_json": official.canonical_bytes(
            decoded
        ).decode("ascii"),
        "source_seed": str(official.SOURCE_SEED),
        "source_task_id": str(official.SOURCE_TASK_ID),
        "source_bundle_id": official.SOURCE_BUNDLE_ID,
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
    source_result = official.sealed(
        {
            "schema_version": "source-result-v1",
            "campaign_id": official.CAMPAIGN_ID,
            "seed": official.SOURCE_SEED,
            "task_payload_sha256": official.SOURCE_BUNDLE_ID,
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
                    "sha256": official.sha256_file(source_csv),
                    "size_bytes": source_csv.stat().st_size,
                }
            },
            "terminal_physical_candidates_manifest": {
                "source_identity": {
                    "seed": official.SOURCE_SEED,
                    "task_id": str(official.SOURCE_TASK_ID),
                    "bundle_id": official.SOURCE_BUNDLE_ID,
                }
            },
        }
    )
    source_result_path = _write_json(
        source_root / "result.json", source_result
    )
    source_result_sha = official.sha256_file(source_result_path)
    official_row = {
        **common_row,
        "source_result_path": str(source_result_path),
        "source_result_sha256": source_result_sha,
        "standard_selection_order": "6",
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
    standard_sha = official.sha256_file(standard_csv)
    manifest = official.sealed(
        {
            "schema_version": "mft-goal-20260726-global-pareto-v1",
            "campaign_id": official.CAMPAIGN_ID,
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
        official, "STANDARD_CANDIDATES_SHA256", standard_sha
    )
    monkeypatch.setattr(
        official,
        "AGGREGATE_MANIFEST_SHA256",
        official.sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        official, "AGGREGATE_PAYLOAD_SHA256", manifest["payload_sha256"]
    )
    monkeypatch.setattr(
        official, "SOURCE_RESULT_SHA256", source_result_sha
    )

    def source_reader(path: str, revision: str) -> bytes:
        assert path == official.INPUT_PARAMETER_PATH
        assert revision == official.SOLVER_REVISION
        return _input_contract_source()

    return standard_csv, manifest_path, source_reader


def _fake_payload_builder(
    _params: dict[str, Any], _profile: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dedupe = "mft-al:official6:fresh"
    command = (
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless "
        "--params cand.json; simulation_rc=$?;"
    )
    payload = {
        "name": official.TASK_NAME,
        "project": official.PROJECT,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "command": command,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": official.CPUS,
        "memory_mb": official.MEMORY_MB,
        "gpus": 0,
        "account_name": official.ACCOUNT_NAME,
        "node_name": official.NODE_NAME,
        "node_name_policy": "strict",
        "max_workers_per_node": official.MAX_WORKERS_PER_NODE,
        "priority": official.PRIORITY,
        "timeout_seconds": official.SCHEDULER_SECONDS,
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
        "MFT_STANDALONE_CORE_AUTH_SHA256": official.CORE_AUTH_SHA256,
    }
    return payload, environment, retained


class FakeScheduler:
    def __init__(self, *, active_fea: bool = False, ready: bool = True) -> None:
        self.active_fea = active_fea
        self.ready = ready
        self.calls: list[tuple[str, list[tuple[str, Any]]]] = []

    def __call__(
        self, path: str, query: list[tuple[str, Any]] | None
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
        if path == f"/api/tasks/{official.SOURCE_TASK_ID}":
            return {
                "task_id": official.SOURCE_TASK_ID,
                "name": official.SOURCE_TASK_NAME,
                "status": "completed",
                "exit_code": 0,
                "required_capability": "conda:pyaedt2026v1",
                "env_profile": "pyaedt2026v1",
                "dedupe_key": official.SOURCE_TASK_DEDUPE_KEY,
            }
        if path == "/api/task-capacity":
            assert ("max_workers_per_node", 1) in normalized
            assert ("account_name", "dw16") in normalized
            assert ("node_name", "n113") in normalized
            return {
                "queue_state": "ready" if self.ready else "opening",
                "ready_fit_slots": 8 if self.ready else 0,
                "fit_slots": 8 if self.ready else 0,
                "memory_pressure_state": "ok",
                "standalone_aedt_available": 396,
                "allocations": (
                    [
                        {
                            "allocation_id": 14620,
                            "account_name": "dw16",
                            "node_name": "n113",
                            "state": "active",
                            "fit_slots": 8,
                            "free_cpus": 64,
                            "free_memory_mb": 866173,
                        }
                    ]
                    if self.ready
                    else []
                ),
            }
        if path == "/api/allocations":
            return [
                {
                    "id": 14620,
                    "account_name": "dw16",
                    "node_name": "n113",
                    "state": "active",
                    "slurm_job_id": "829579",
                    "free_cpus": 64,
                    "free_memory_mb": 866173,
                    "node_fea_requested_cpus": 0,
                }
            ]
        if path == "/api/tasks":
            if ("name_prefix", official.TASK_NAME) in normalized:
                return []
            if self.active_fea:
                return [
                    {
                        "task_id": 999,
                        "status": "running",
                        "assigned_allocation": 14620,
                        "node_name": "n113",
                        "aedt_backend": "standalone",
                        "scheduling_profile": "fea_bursty",
                    }
                ]
            return []
        raise AssertionError(f"unexpected GET: {path} {normalized}")


def test_authenticates_official_and_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path, monkeypatch
    )
    selected, params = official.authenticate_official_candidate(
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        source_reader=source_reader,
    )
    official.validate_seal(selected, official.CANDIDATE_SCHEMA)
    assert selected["authentication"]["candidate_reauthenticated"] is True
    assert selected["authentication"]["source_task_id"] == 96185
    assert selected["official_row"]["standard_selection_order"] == "6"
    assert params["fan_velocity"] == 1.5
    assert params["k_ins"] == 0.2


def test_official_csv_byte_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path, monkeypatch
    )
    standard.write_bytes(standard.read_bytes() + b"\n")
    with pytest.raises(
        official.PostdeadlineContractError,
        match="Standard12 CSV bytes drifted",
    ):
        official.authenticate_official_candidate(
            standard_candidates_path=standard,
            aggregate_manifest_path=manifest,
            source_reader=source_reader,
        )


def test_candidate_cooling_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path, monkeypatch, fan_velocity=2.0
    )
    with pytest.raises(
        official.PostdeadlineContractError,
        match="fixed cooling physics drifted",
    ):
        official.authenticate_official_candidate(
            standard_candidates_path=standard,
            aggregate_manifest_path=manifest,
            source_reader=source_reader,
        )


def test_exact_capacity_query_and_empty_lane_are_required() -> None:
    scheduler = FakeScheduler()
    preflight = official.live_preflight(
        reader=scheduler,
        expected_dedupe_key="fresh",
        observed_at=OBSERVED,
    )
    assert preflight["selected_allocation_id"] == 14620
    assert preflight["selected_slurm_job_id"] == "829579"
    capacity_call = next(
        call for call in scheduler.calls if call[0] == "/api/task-capacity"
    )
    assert ("memory_mb", 98304) in capacity_call[1]
    assert ("max_workers_per_node", 1) in capacity_call[1]
    assert ("account_name", "dw16") in capacity_call[1]
    assert ("node_name", "n113") in capacity_call[1]


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        (FakeScheduler(active_fea=True), "no longer FEA-empty"),
        (FakeScheduler(ready=False), "capacity gate is not ready"),
    ],
)
def test_live_lane_drift_fails_closed(
    scheduler: FakeScheduler, message: str
) -> None:
    with pytest.raises(official.PostdeadlineContractError, match=message):
        official.live_preflight(
            reader=scheduler,
            expected_dedupe_key="fresh",
            observed_at=OBSERVED,
        )


def test_prepare_seals_zero_post_plan_and_fixed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path / "authority", monkeypatch
    )
    scheduler = FakeScheduler()
    output = tmp_path / "prepared"
    plan_path = official.prepare(
        output=output,
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        reader=scheduler,
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
        observed_at=OBSERVED,
    )
    plan = official.validate_seal(
        official.read_json(plan_path), official.PLAN_SCHEMA
    )
    receipt = official.validate_seal(
        official.read_json(output / "prepare_receipt.json"),
        official.PREPARE_SCHEMA,
    )
    assert plan["scheduler_submission_performed"] is False
    assert plan["diagnostic_only"] is True
    assert plan["search_only"] is True
    assert plan["noncanonical"] is True
    assert plan["original_deadline_missed"] is True
    assert plan["resources"]["memory_mb"] == 98304
    assert plan["resources"]["scheduler_timeout_seconds"] == 45300
    assert plan["placement"]["account_name"] == "dw16"
    assert plan["placement"]["node_name"] == "n113"
    assert (
        Path(
            plan["single_attempt_contract"]["attempt_ledger_path"]
        ).resolve()
        == output / official.ATTEMPT_LEDGER_NAME
    )
    assert (
        Path(
            plan["single_attempt_contract"]["submission_output_path"]
        ).resolve()
        == output / official.SUBMISSION_DIRECTORY_NAME
    )
    assert receipt["scheduler_post_calls"] == 0
    assert receipt["ready_for_explicit_submit"] is True


def test_load_plan_freshly_reauthenticates_every_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standard, manifest, source_reader = _fixture_authority(
        tmp_path / "authority", monkeypatch
    )
    output = tmp_path / "prepared"
    plan_path = official.prepare(
        output=output,
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        reader=FakeScheduler(),
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
        observed_at=OBSERVED,
    )
    plan, params, profile = official.load_plan(
        plan_path,
        standard_candidates_path=standard,
        aggregate_manifest_path=manifest,
        payload_builder=_fake_payload_builder,
        source_reader=source_reader,
    )
    assert plan["candidate_physics_sha256"] == official.CANDIDATE_SHA256
    assert params["fan_velocity"] == 1.5
    assert profile["fixed_boundary_contract"] == official.FIXED_BOUNDARY


def test_scheduler_payload_rejects_max_worker_drift() -> None:
    payload, _environment, retained = _fake_payload_builder({}, {})
    payload["max_workers_per_node"] = 2
    with pytest.raises(
        official.PostdeadlineContractError,
        match="payload/retention drifted",
    ):
        official.validate_scheduler_payload(payload, retained)


def test_submit_requires_exact_authorization_before_loading_or_posting(
    tmp_path: Path,
) -> None:
    calls = 0

    def poster(
        _url: str, _payload: dict[str, Any]
    ) -> tuple[int, dict[str, int], None]:
        nonlocal calls
        calls += 1
        return 201, {"task_id": 12345}, None

    with pytest.raises(
        official.PostdeadlineContractError, match="authorization is absent"
    ):
        official.submit(
            plan_path=tmp_path / "absent.json",
            output=tmp_path / "submission",
            authorize_post="wrong",
            poster=poster,
        )
    assert calls == 0


class _Lock:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> _Lock:
        self.events.append("lock-enter")
        return self

    def __exit__(self, *_args: Any) -> None:
        self.events.append("lock-exit")


def test_single_attempt_ledger_blocks_second_different_output_post(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    plan_path = _write_json(root / "plan.json", {"plan": True})
    output = root / official.SUBMISSION_DIRECTORY_NAME
    payload, _environment, _retained = _fake_payload_builder({}, {})
    plan = {
        "payload_sha256": "c" * 64,
        "scheduler_payload": payload,
        "scheduler_payload_sha256": official.payload_sha256(payload),
        "dedupe_key": payload["dedupe_key"],
        "single_attempt_contract": {
            "attempt_ledger_path": str(
                root / official.ATTEMPT_LEDGER_NAME
            ),
            "submission_output_path": str(output),
            "attempt_nonce": "d" * 64,
        },
    }
    events: list[str] = []
    scheduler = FakeScheduler()
    post_calls = 0

    def reader(
        path: str, query: list[tuple[str, Any]] | None
    ) -> Any:
        if path == "/api/tasks/12345":
            events.append("readback")
            return {
                "task_id": 12345,
                "name": official.TASK_NAME,
                "dedupe_key": payload["dedupe_key"],
                "project": official.PROJECT,
                "requested_account_name": official.ACCOUNT_NAME,
                "requested_node_name": official.NODE_NAME,
                "requested_node_name_policy": "strict",
                "same_node_as_task_id": 0,
                "cpus": official.CPUS,
                "memory_mb": official.MEMORY_MB,
                "timeout_seconds": official.SCHEDULER_SECONDS,
                "max_workers_per_node": 1,
                "aedt_backend": "standalone",
                "status": "running",
                "account_name": official.ACCOUNT_NAME,
                "node_name": official.NODE_NAME,
            }
        result = scheduler(path, query)
        if path == "/api/task-capacity":
            events.append("locked-or-initial-get")
        return result

    def poster(
        _url: str, _payload: dict[str, Any]
    ) -> tuple[int, dict[str, int], None]:
        nonlocal post_calls
        post_calls += 1
        events.append("post")
        assert (root / official.ATTEMPT_LEDGER_NAME).is_file()
        return 201, {"task_id": 12345}, None

    final_seal = official.submit(
        plan_path=plan_path,
        output=output,
        authorize_post=official.POST_AUTHORIZATION,
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
    attempt = official.read_json(root / official.ATTEMPT_LEDGER_NAME)
    assert attempt["post_call_consumed_before_network"] is True
    assert attempt["campaign_mutation_lock_acquired"] is True

    with pytest.raises(
        official.PostdeadlineContractError,
        match="submission output differs",
    ):
        official.submit(
            plan_path=plan_path,
            output=root / "different-output",
            authorize_post=official.POST_AUTHORIZATION,
            reader=reader,
            poster=poster,
            observed_at=OBSERVED,
            lock_factory=lambda: _Lock(events),
            plan_loader=lambda _path: (plan, {}, {}),
        )
    assert post_calls == 1
