from __future__ import annotations

import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.mft_goal_20260726_contract import canonical_sha256
from regression_260707.verify import scheduler_client
from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_fea_handoff as production
from tools import mft_goal_provisional_full_collector as collector
from tools import mft_goal_provisional_full_precompute as provisional


TASK_ID = 97001
SOLVER_REVISION = "1" * 40
LIBRARY_REVISION = "2" * 40
ACCOUNT = "dhj02"
NODE = "n116"
DEADLINE = "2026-07-26T09:00:00+00:00"
OBSERVED = datetime(2026, 7, 26, 8, 30, tzinfo=timezone.utc)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _admission(plan: dict, captured_at: str) -> dict:
    return production._seal(
        {
            "schema_version": provisional.GATE_SCHEMA,
            "plan_payload_sha256": plan["payload_sha256"],
            "captured_at_utc": captured_at,
            "selected_account_name": ACCOUNT,
            "requested_node_name": NODE,
            "license_snapshot_sha256": "3" * 64,
            "all_scheduler_reads_get_only": True,
            "fresh_capacity_passed": True,
            "fresh_license_passed": True,
            "fresh_storage_passed": True,
            "scheduler_repository_modified": False,
            "scheduler_submission_performed": False,
            **collector._flags(),
        }
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    hard_cap: int = 1024,
) -> dict:
    source_root = tmp_path / "source"
    source_root.mkdir()
    plan_path = _write_json(source_root / "plan.json", {"source": "plan"})
    profile, profile_record = production._profile_content("full")
    params = {name: 1 for name in production.ALL_INPUT_KEYS}
    task_name = "mft-goal-provisional-full-fixture"
    retained = scheduler_client.retained_aedt_identity(
        task_name,
        params,
        profile,
        SOLVER_REVISION,
        LIBRARY_REVISION,
        retained_aedt_max_bytes=hard_cap,
    )
    assert retained is not None
    plan = {
        "payload_sha256": "4" * 64,
        "output_root": str(source_root.resolve()),
        "campaign_id": provisional.CAMPAIGN_ID,
        "candidate_physics_sha256": "5" * 64,
        "logical_authority_task_id": 96230,
        "source_actual_standard_task_id": 96304,
        "selected_account_name": ACCOUNT,
        "requested_node_name": NODE,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "profile": profile_record,
        "profile_canonical_sha256": canonical_sha256(profile),
        "effective_full_params_sha256": canonical_sha256(
            production._effective_params(params, profile)
        ),
        "stage": {
            "name": "full",
            "task_name": task_name,
            "resources": dict(provisional.RESOURCES),
            "full_model": 1,
            "thermal_symmetry": "full",
            "aedt_backend": "standalone",
            "requested_account_name": ACCOUNT,
            "requested_node_name": NODE,
            "node_name_policy": "strict",
            "retained_aedt": retained,
        },
        "retention_storage_bound": {
            "expected_full_aedt_bytes": hard_cap,
        },
        "global_exact_once_scope": {
            "scope": "fixture",
            "task_name": task_name,
            "dedupe_key": retained["dedupe_key"],
        },
        "scheduler_url": provisional.SCHEDULER_URL,
        "scheduler_project": scheduler_client.MFT_PROJECT,
        "target_finish_kst": DEADLINE,
    }
    finalized = production._seal(
        {
            "schema_version": atomic_claim.FINALIZED_CLAIM_SCHEMA,
            "task_id": TASK_ID,
        }
    )
    submission = production._seal(
        {
            "schema_version": provisional.SUBMISSION_SCHEMA,
            "plan": production._file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_physics_sha256": plan["candidate_physics_sha256"],
            "logical_authority_task_id": 96230,
            "source_actual_standard_task_id": 96304,
            "selected_account_name": ACCOUNT,
            "requested_node_name": NODE,
            "global_exact_once_scope": plan["global_exact_once_scope"],
            "task_id": TASK_ID,
            "task_name": task_name,
            "dedupe_key": retained["dedupe_key"],
            "resources": dict(provisional.RESOURCES),
            "full_model": 1,
            "thermal_symmetry": "full",
            "profile": profile_record,
            "retained_aedt": retained,
            "retained_aedt_max_bytes": hard_cap,
            "retained_aedt_source_size_hard_cap_contract": (
                collector.SOURCE_SIZE_HARD_CAP_CONTRACT
            ),
            "retention_storage_bound": plan["retention_storage_bound"],
            "initial_fresh_admission": _admission(plan, "2026-07-25T19:00:00+00:00"),
            "locked_fresh_admission": _admission(plan, "2026-07-25T19:00:01+00:00"),
            "exact_sibling_count": 1,
            "atomic_claim_finalized": finalized,
            "maximum_scheduler_posts_lifetime": 1,
            "scheduler_submission_performed": True,
            "scheduler_cancel_performed": False,
            "scheduler_repository_modified": False,
            "canonical_truth_gate_modified": False,
            "canonical_claim_modified": False,
            **collector._flags(),
        }
    )
    submission_path = _write_json(source_root / "submission.json", submission)

    def load_plan(path: Path):
        assert path.resolve() == plan_path.resolve()
        return plan, params, profile

    monkeypatch.setattr(collector.provisional, "load_plan", load_plan)
    watch_plan_path = collector.initialize_watch_plan(
        provisional_plan_path=plan_path,
        submission_path=submission_path,
        output_root=tmp_path / "collector",
        poll_seconds=10,
    )
    return {
        "plan": plan,
        "params": params,
        "profile": profile,
        "submission": submission,
        "watch_plan_path": watch_plan_path,
        "retained": retained,
    }


def _task(fixture: dict, *, status: str = "completed") -> dict:
    completed = status == "completed"
    return {
        "id": TASK_ID,
        "name": fixture["submission"]["task_name"],
        "status": status,
        "state": "succeeded" if completed else "running",
        "exit_code": 0 if completed else None,
        "failure_message": None,
        "project": scheduler_client.MFT_PROJECT,
        "dedupe_key": fixture["submission"]["dedupe_key"],
        "cpus": 16,
        "memory_mb": 98304,
        "timeout_seconds": 43200,
        "aedt_backend": "standalone",
        "account_name": ACCOUNT,
        "requested_account_name": ACCOUNT,
        "actual_node_name": NODE,
        "allocation_node_name": NODE,
        "requested_node_name": NODE,
        "requested_node_name_policy": "strict",
        "allocation_id": 88,
        "slurm_job_id": "90001",
        "remote_cwd": "/gpfs/task/97001",
        "remote_dir": "/gpfs/task/97001",
        "created_at": "2026-07-25T19:00:00+00:00",
        "started_at": "2026-07-25T19:01:00+00:00",
        "finished_at": ("2026-07-26T08:00:00+00:00" if completed else None),
    }


def _result(fixture: dict) -> dict:
    runtime_license_sha = "6" * 64
    runtime_auth = production._core_auth(
        SOLVER_REVISION,
        16,
        license_contract=production.FULL_LICENSE_CONTRACT,
        license_snapshot_sha256=runtime_license_sha,
    )
    return {
        "git_hash": SOLVER_REVISION,
        "pyaedt_library_git_hash": LIBRARY_REVISION,
        "project_name": "fixture-full-project",
        "full_model": 1,
        "thermal_symmetry": "full",
        "solver_core_policy_schema": "mft-solver-core-policy-v1",
        "solver_core_contract_version": production.FULL_CORE_CONTRACT,
        "solver_core_opt_in": 1,
        "solver_core_backend": "standalone",
        "solver_num_cores_requested": 16,
        "solver_num_cores_effective": 16,
        "solver_num_tasks_effective": 1,
        "solver_core_affinity_count_readback": 16,
        "solver_core_slurm_cpus_per_task_readback": "16",
        "solver_core_scheduler_task_id_readback": str(TASK_ID),
        "solver_core_slurm_job_id_readback": "90001",
        "solver_core_auth_sha256": runtime_auth,
        "solver_matrix_hpc_num_cores_readback": 16,
        "solver_matrix_hpc_num_engines_readback": 1,
        "solver_matrix_hpc_acf_sha256": "7" * 64,
        "solver_core_license_contract": production.FULL_LICENSE_CONTRACT,
        "solver_core_license_snapshot_sha256": runtime_license_sha,
        "solver_core_license_checked_at_readback": ("2026-07-26T07:59:00+00:00"),
        "solver_core_license_snapshot_age_seconds_readback": 1.0,
        "solver_core_license_headroom_readback_json": json.dumps(
            {
                "anshpc": 16,
                "elec_solve_maxwell": 1,
                "electronics_desktop": 1,
                "electronics3d_gui": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


class _Scheduler:
    RESULT_VALID = scheduler_client.RESULT_VALID

    def __init__(self, result: dict, *, params_match: bool = True):
        self.result = result
        self.params_match = params_match
        self.fetch_calls = 0

    def fetch_result(self, *_args, **_kwargs):
        self.fetch_calls += 1
        return SimpleNamespace(state=self.RESULT_VALID, result=self.result)

    def result_matches_params(self, *_args, **_kwargs):
        return self.params_match


def _remote_evidence(
    fixture: dict,
    artifact: bytes,
) -> tuple[dict, dict, object]:
    retained = fixture["retained"]
    marker = {
        **retained["marker_contract"],
        "created_at": "2026-07-26T08:00:00+00:00",
    }
    marker_raw = production._json_bytes(marker)
    receipt = {
        "schema_version": production.REMOTE_RECEIPT_SCHEMA,
        "stage": "full",
        "dedupe_key": retained["dedupe_key"],
        "parameter_digest": retained["parameter_digest"],
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "profile_sha256": retained["profile_sha256"],
        "artifact_path": retained["artifact_path"],
        "marker_path": retained["marker_path"],
        "retention_required": True,
        "prune_protection_required": True,
        "scheduler_cleanup_exclusion_required": True,
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_size_bytes": len(artifact),
        "marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "marker_contract_sha256": retained["marker_contract_sha256"],
        "transport_schema_version": retained["transport"]["schema_version"],
        "transport_encoding": retained["transport"]["encoding"],
        "transport_chunk_directory": retained["transport"]["chunk_directory"],
        "transport_raw_chunk_bytes": retained["transport"]["raw_chunk_bytes"],
        "transport_max_encoded_chunk_bytes": retained["transport"][
            "max_encoded_chunk_bytes"
        ],
        "transport_chunk_count": math.ceil(
            len(artifact) / retained["transport"]["raw_chunk_bytes"]
        ),
        "source_project_filename": "fixture-full-project.aedt",
        "source_project_name": "fixture-full-project",
        "source_size_hard_cap_bytes": fixture["plan"]["retention_storage_bound"][
            "expected_full_aedt_bytes"
        ],
        "source_size_hard_cap_contract": (collector.SOURCE_SIZE_HARD_CAP_CONTRACT),
        "source_size_hard_cap_enforced_before_destination_create": True,
    }
    receipt_raw = production._json_bytes(receipt)

    def remote_reader(
        *,
        relative_path: str,
        **_kwargs,
    ) -> bytes:
        if relative_path == retained["receipt_path"]:
            return receipt_raw
        if relative_path == retained["marker_path"]:
            return marker_raw
        raise AssertionError(relative_path)

    return receipt, marker, remote_reader


def _success_cycle(
    fixture: dict,
    *,
    artifact: bytes = b"fixture-aedt",
    artifact_fetcher=None,
    params_match: bool = True,
):
    _receipt, _marker, remote_reader = _remote_evidence(fixture, artifact)
    scheduler = _Scheduler(_result(fixture), params_match=params_match)

    def default_fetcher(*, destination: Path, **_kwargs) -> None:
        destination.write_bytes(artifact)

    state = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=scheduler,
        status_reader=lambda **_kwargs: "completed",
        task_reader=lambda **_kwargs: _task(fixture),
        remote_reader=remote_reader,
        artifact_fetcher=artifact_fetcher or default_fetcher,
        now=OBSERVED,
    )
    return state, scheduler


def test_module_contains_no_scheduler_or_remote_mutator_call() -> None:
    source = Path(collector.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "submit_verification",
        "cancel_task",
        "cancel",
        "post",
        "patch",
        "delete",
        "unlink",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert calls.isdisjoint(forbidden)


def test_completed_task_collects_project_only_full_and_replays_immutably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    state, scheduler = _success_cycle(fixture)

    assert state["status"] == "completed_provisional_full_collected"
    assert scheduler.fetch_calls == 1
    output_root = fixture["watch_plan_path"].parent
    local = output_root / "full.aedt"
    success_path = output_root / "provisional_full_collection_receipt.json"
    assert local.read_bytes() == b"fixture-aedt"
    success_record = production._file_record(success_path)
    success = production._validate_seal(
        production._read_json(success_path), collector.SUCCESS_SCHEMA
    )
    assert success["production_eligible"] is False
    assert success["canonical_truth_authority_claimed"] is False
    assert success["source_artifact_size_within_sealed_hard_cap"] is True
    assert success["scheduler_methods_used"] == ["GET"]
    assert success["gpfs_write_performed"] is False

    replay = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: pytest.fail(
            "immutable replay must not query Scheduler"
        ),
        task_reader=lambda **_kwargs: pytest.fail(
            "immutable replay must not query task"
        ),
        now=OBSERVED,
    )
    assert replay["status"] == "completed_immutable_replay"
    assert production._file_record(success_path) == success_record


def test_sealed_source_hard_cap_blocks_before_chunk_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, hard_cap=4)
    calls = []

    state, _scheduler = _success_cycle(
        fixture,
        artifact=b"five!",
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(
        fixture,
        artifact=b"five!",
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(
        fixture,
        artifact=b"five!",
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )

    assert state["status"] == "blocked_collection_error"
    assert "sealed project-only cap" in state["error"]
    assert calls == []
    assert not (fixture["watch_plan_path"].parent / "full.aedt").exists()
    assert not (
        fixture["watch_plan_path"].parent / "provisional_full_collection_receipt.json"
    ).exists()


def test_partial_chunk_failure_is_fail_closed_without_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    def partial(**_kwargs) -> None:
        raise production.HandoffContractError("remote AEDT text chunk size drifted: 1")

    state, _scheduler = _success_cycle(fixture, artifact_fetcher=partial)
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(fixture, artifact_fetcher=partial)
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(fixture, artifact_fetcher=partial)

    assert state["status"] == "blocked_collection_error"
    root = fixture["watch_plan_path"].parent
    assert (root / "failure_ledger.json").is_file()
    assert not (root / "full.aedt").exists()
    assert not (root / "provisional_full_collection_receipt.json").exists()


def test_terminal_failure_writes_ledger_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    state = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: "failed",
        task_reader=lambda **_kwargs: _task(fixture, status="failed"),
        now=OBSERVED,
    )

    assert state["status"] == "blocked_terminal_failure"
    root = fixture["watch_plan_path"].parent
    assert (root / "failure_ledger.json").is_file()
    assert not (root / "full.aedt").exists()
    assert not (root / "provisional_full_collection_receipt.json").exists()


def test_cross_account_node_lineage_drift_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    task = _task(fixture)
    task["actual_node_name"] = "n114"

    for expected in (
        "retrying_after_collection_error",
        "retrying_after_collection_error",
        "blocked_collection_error",
    ):
        state = collector.process_cycle(
            watch_plan_path=fixture["watch_plan_path"],
            scheduler=SimpleNamespace(),
            status_reader=lambda **_kwargs: "completed",
            task_reader=lambda **_kwargs: task,
            now=OBSERVED,
        )
        assert state["status"] == expected

    assert "task lineage drifted" in state["error"]


def test_result_parameter_mismatch_is_blocked_before_artifact_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    calls = []

    state, _scheduler = _success_cycle(
        fixture,
        params_match=False,
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(
        fixture,
        params_match=False,
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )
    assert state["status"] == "retrying_after_collection_error"
    state, _scheduler = _success_cycle(
        fixture,
        params_match=False,
        artifact_fetcher=lambda **kwargs: calls.append(kwargs),
    )

    assert state["status"] == "blocked_collection_error"
    assert "RESULT_JSON identity drifted" in state["error"]
    assert calls == []


def test_active_task_waits_without_terminal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    state = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: "running",
        task_reader=lambda **_kwargs: _task(fixture, status="running"),
        now=OBSERVED,
    )

    assert state["status"] == "watching_active_task"
    root = fixture["watch_plan_path"].parent
    assert not (root / "failure_ledger.json").exists()
    assert not (root / "full.aedt").exists()


def test_active_task_reader_transient_once_recovers_and_resets_streak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    calls = 0

    def task_reader(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary task GET failure")
        return _task(fixture, status="running")

    first = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: "running",
        task_reader=task_reader,
        now=OBSERVED,
    )
    second = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: "running",
        task_reader=task_reader,
        now=OBSERVED,
    )

    assert first["status"] == "retrying_after_collection_error"
    assert first["consecutive_error_streak"] == 1
    assert second["status"] == "watching_active_task"
    assert second["consecutive_error_streak"] == 0
    assert not (fixture["watch_plan_path"].parent / "failure_ledger.json").exists()


def test_terminal_remote_metadata_transient_once_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    artifact = b"fixture-aedt"
    _receipt, _marker, durable_reader = _remote_evidence(fixture, artifact)
    reads = 0

    def transient_reader(**kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            raise OSError("temporary remote-file GET failure")
        return durable_reader(**kwargs)

    scheduler = _Scheduler(_result(fixture))

    def fetcher(*, destination: Path, **_kwargs) -> None:
        destination.write_bytes(artifact)

    first = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=scheduler,
        status_reader=lambda **_kwargs: "completed",
        task_reader=lambda **_kwargs: _task(fixture),
        remote_reader=transient_reader,
        artifact_fetcher=fetcher,
        now=OBSERVED,
    )
    second = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=scheduler,
        status_reader=lambda **_kwargs: "completed",
        task_reader=lambda **_kwargs: _task(fixture),
        remote_reader=transient_reader,
        artifact_fetcher=fetcher,
        now=OBSERVED,
    )

    assert first["status"] == "retrying_after_collection_error"
    assert first["consecutive_error_streak"] == 1
    assert second["status"] == "completed_provisional_full_collected"
    assert second["consecutive_error_streak"] == 0
    assert not (fixture["watch_plan_path"].parent / "failure_ledger.json").exists()


def test_repeated_transient_reaches_threshold_then_permanently_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    states = []

    def unavailable(**_kwargs):
        raise OSError("repeated task GET failure")

    for _index in range(collector.MAX_CONSECUTIVE_ERRORS):
        states.append(
            collector.process_cycle(
                watch_plan_path=fixture["watch_plan_path"],
                scheduler=SimpleNamespace(),
                status_reader=lambda **_kwargs: "running",
                task_reader=unavailable,
                now=OBSERVED,
            )
        )

    assert [state["status"] for state in states] == [
        "retrying_after_collection_error",
        "retrying_after_collection_error",
        "blocked_collection_error",
    ]
    assert [state["consecutive_error_streak"] for state in states] == [
        1,
        2,
        3,
    ]
    root = fixture["watch_plan_path"].parent
    failure_path = root / "failure_ledger.json"
    assert failure_path.is_file()
    failure_record = production._file_record(failure_path)

    replay = collector.process_cycle(
        watch_plan_path=fixture["watch_plan_path"],
        scheduler=SimpleNamespace(),
        status_reader=lambda **_kwargs: pytest.fail(
            "permanent failure replay must not use Scheduler"
        ),
        now=OBSERVED,
    )
    assert replay["status"] == "blocked_by_immutable_failure"
    assert production._file_record(failure_path) == failure_record


def test_remote_chunk_get_retries_before_returning_valid_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = b"abc"
    encoded = base64.b64encode(chunk)
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _maximum: int) -> bytes:
            return encoded

    def urlopen(_request, timeout: float):
        nonlocal calls
        calls += 1
        assert timeout == 120.0
        if calls == 1:
            raise OSError("one HTTP blip")
        return Response()

    monkeypatch.setattr(collector.urllib.request, "urlopen", urlopen)

    observed = collector._read_validated_remote_chunk(
        scheduler_url=provisional.SCHEDULER_URL,
        task_id=TASK_ID,
        relative_path="goal-fea-retained/a/full.aedt.chunks/00000000.b64",
        expected_encoded_size=len(encoded),
        expected_raw_size=len(chunk),
        maximum_encoded_size=(scheduler_client.RETAINED_AEDT_MAX_ENCODED_CHUNK_BYTES),
    )

    assert observed == chunk
    assert calls == 2
