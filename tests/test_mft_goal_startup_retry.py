from __future__ import annotations

import copy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from regression_260707.verify import scheduler_client
from tools import mft_campaign_atomic_claim as atomic_claim
from tools import mft_goal_safe_refill as refill
from tools import mft_goal_startup_retry as startup


def _source_submission() -> dict[str, Any]:
    return {
        "task_id": startup.FAILED_TASK_ID,
        "task_name": "source-task",
        "dedupe_key": "source-dedupe",
        "retained_aedt_bundle": {
            "relative_directory": "goal-fea-retained/source",
            "receipt_path": "goal-fea-retained/source/symmetric.aedt.receipt.json",
            "results_manifest_path": (
                "goal-fea-retained/source/symmetric.aedtresults.manifest.json"
            ),
        },
    }


def _source_snapshot() -> dict[str, Any]:
    return {
        "task_id": startup.FAILED_TASK_ID,
        "name": "source-task",
        "dedupe_key": "source-dedupe",
        "status": "failed",
        "state": "failed",
        "exit_code": 1,
        "failure_message": startup.EXPECTED_FAILURE_MESSAGE,
        "timeout_seconds": 43200,
        "cpus": 8,
        "memory_mb": 32768,
        "aedt_backend": "standalone",
        "project": startup.PROJECT,
        "account_name": startup.TARGET_ACCOUNT,
        "requested_account_name": startup.TARGET_ACCOUNT,
        "actual_node_name": startup.TARGET_NODE,
        "requested_node_name": startup.TARGET_NODE,
        "requested_node_name_policy": "strict",
        "placement_contract_satisfied": True,
        "slurm_job_id": "824575",
        "allocation_id": 14492,
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "remote_dir": "slurm_scheduler/runs/task-96303",
        "started_at": "2026-07-25 17:28:07",
        "finished_at": "2026-07-25 18:01:47",
    }


def _stdout() -> str:
    return (
        "MFT_WORKDIR /enroot/mft_source\n"
        "MFT_LIBRARY_GIT_HASH " + "a" * 40 + "\n"
    )


def _stderr() -> str:
    reason = (
        "AttributeError: 'NoneType' object has no attribute 'EnableAutoSave'"
    )
    return "\n".join(
        [
            f"WARNING:root:AEDT session startup attempt {index}/3 failed: {reason}"
            for index in (1, 2, 3)
        ]
        + [f"ERROR:root:run_one_loop failed: {startup.EXPECTED_FAILURE_MESSAGE}"]
    )


def _failure(**changes: Any) -> dict[str, Any]:
    values = {
        "snapshot": _source_snapshot(),
        "stdout": _stdout(),
        "stderr": _stderr(),
        "source_submission": _source_submission(),
        "retained_inventory": {
            "files": [],
            "base": "remote_cwd",
            "glob": "goal-fea-retained/source/**",
        },
        "collection_exists": False,
        "success_receipt_exists": False,
    }
    values.update(changes)
    return startup._source_failure_evidence(**values)


def test_exact_task96303_failure_is_accepted() -> None:
    evidence = _failure()
    assert evidence["startup_attempts"] == [1, 2, 3]
    assert evidence["result_json_absent"] is True
    assert evidence["selected_mft_workdir"] == "/enroot/mft_source"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "completed"),
        ("state", "succeeded"),
        ("exit_code", 0),
        ("failure_message", "another failure"),
        ("requested_node_name", "n116"),
        ("memory_mb", 65536),
    ],
)
def test_exact_failure_rejects_snapshot_variants(
    field: str, value: Any
) -> None:
    snapshot = _source_snapshot()
    snapshot[field] = value
    with pytest.raises(startup.StartupRetryError):
        _failure(snapshot=snapshot)


@pytest.mark.parametrize(
    "changes",
    [
        {"stderr": _stderr().replace("attempt 2/3", "attempt 1/3")},
        {"stdout": _stdout() + 'RESULT_JSON {"x":1}\n'},
        {"stderr": _stderr() + "\nRESULT_JSON: {}\n"},
        {"collection_exists": True},
        {"success_receipt_exists": True},
        {
            "retained_inventory": {
                "files": [
                    {
                        "path": (
                            "goal-fea-retained/source/"
                            "symmetric.aedt.receipt.json"
                        )
                    }
                ]
            }
        },
        {"stdout": "MFT_WORKDIR /gpfs/home1/r1/source\n"},
    ],
)
def test_exact_failure_rejects_evidence_variants(
    changes: dict[str, Any]
) -> None:
    with pytest.raises(startup.StartupRetryError):
        _failure(**changes)


def test_successor_environment_is_exact_and_physics_free() -> None:
    assert startup.TEMP_ENVIRONMENT == {
        "ANS_TEMP_PATH": "$MFT_WORKDIR",
        "TMPDIR": "$MFT_WORKDIR",
        "TMP": "$MFT_WORKDIR",
        "TEMP": "$MFT_WORKDIR",
    }
    assert "ANS_MW_INHERIT_TMP" not in startup.TEMP_ENVIRONMENT
    assert startup.RESOURCES == {
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
    }
    name, workdir = startup._task_identity(
        startup.EXPECTED_CANDIDATE_SHA256
    )
    assert "startup-r1-l96223" in name
    assert "startup_r1_l96223" in workdir


def test_scheduler_client_defers_temp_exports_until_enroot_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: dict[str, Any] = {}

    class Response:
        status_code = 201

        @staticmethod
        def json() -> dict[str, Any]:
            return {"id": 99001}

    monkeypatch.setattr(
        scheduler_client, "reconcile_task_id", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        scheduler_client,
        "live_project_submission_snapshot",
        lambda *_args, **_kwargs: {"project_submission_slots": 1},
    )

    def post(_url: str, *, json: dict[str, Any], timeout: int) -> Response:
        submitted.update(json)
        assert timeout == 20
        return Response()

    monkeypatch.setattr(scheduler_client.requests, "post", post)
    task_id = scheduler_client.submit_verification(
        "startup-command-test",
        "startup_workdir",
        {"x": 1},
        {},
        solver_revision="a" * 40,
        library_revision="b" * 40,
        aedt_backend="standalone",
        submission_env=copy.deepcopy(startup.TEMP_ENVIRONMENT),
        submission_env_after_workdir=True,
        required_workdir_prefix="/enroot/",
    )
    assert task_id == 99001
    command = submitted["command"]
    select_at = command.index("MFT_WORKDIR %s")
    assertion_at = command.index('case "$MFT_WORKDIR" in /enroot/*)')
    export_at = command.index('export ANS_TEMP_PATH="$MFT_WORKDIR"')
    mkdir_at = command.index('mkdir -p "${MFT_WORKDIR}"')
    assert select_at < assertion_at < export_at < mkdir_at
    assert "MFT_WORKDIR_ASSERTION_FAILED" in command
    assert "ANS_MW_INHERIT_TMP" not in command


def test_scheduler_client_rejects_unsafe_workdir_contract() -> None:
    with pytest.raises(ValueError):
        scheduler_client.submit_verification(
            "unsafe",
            "work",
            {},
            {},
            solver_revision="a" * 40,
            library_revision="b" * 40,
            submission_env={"TMPDIR": "$MFT_WORKDIR"},
            required_workdir_prefix="/enroot/",
        )


def _evaluation_plan() -> dict[str, Any]:
    return {
        "startup_successor": {
            "safe_refill_plan": {"path": "safe.json"},
            "source_plan": {"path": "source.json"},
            "source_submission": {"path": "submission.json"},
            "extension_authority": {"path": "authority.json"},
            "watcher_extension_target": "extension.json",
            "source_failure_evidence": {
                "watcher_state": "terminal_failure",
                "result_json_absent": True,
                "authenticated_collection_absent": True,
            },
            "submission_environment": copy.deepcopy(
                startup.TEMP_ENVIRONMENT
            ),
            "required_workdir_prefix": "/enroot/",
        },
        "stage": {
            "task_name": "successor",
            "retained_aedt_bundle": {"dedupe_key": "successor-dedupe"},
        },
        "scheduler_strict_node_contract": {},
    }


def _safe_plan() -> dict[str, Any]:
    return {
        "watcher_plan": {"path": "watch.json"},
        "candidates": [
            {
                "logical_authority_task_id": startup.LOGICAL_ID,
                "candidate_physics_sha256": (
                    startup.EXPECTED_CANDIDATE_SHA256
                ),
                "prospective_grid_gb": startup.PROSPECTIVE_GRID_GIB,
            },
            {
                "logical_authority_task_id": 96224,
                "candidate_physics_sha256": "4" * 64,
                "prospective_grid_gb": 27.094282487407327,
            },
            {
                "logical_authority_task_id": 96230,
                "candidate_physics_sha256": "b" * 64,
                "prospective_grid_gb": 24.507468160241842,
            },
        ],
    }


def _active(
    task_id: int, logical: int, candidate: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = {
        "task_id": task_id,
        "name": f"task-{logical}",
        "dedupe_key": f"dedupe-{logical}",
        "project": startup.PROJECT,
        "aedt_backend": "standalone",
        "account_name": startup.TARGET_ACCOUNT,
        "requested_account_name": startup.TARGET_ACCOUNT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
        "requested_node_name": startup.TARGET_NODE,
        "requested_node_name_policy": "strict",
    }
    slot = {
        "execution_task_id": task_id,
        "logical_authority_task_id": logical,
        "task_name": task["name"],
        "dedupe_key": task["dedupe_key"],
        "candidate_physics_sha256": candidate,
    }
    return task, slot


def test_evaluate_charges_cumulative_active_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_a, slot_a = _active(96304, 96230, "b" * 64)
    task_b, slot_b = _active(96305, 96224, "4" * 64)
    monkeypatch.setattr(startup.refill, "load_plan", lambda _path: _safe_plan())
    monkeypatch.setattr(
        startup.timeout12h,
        "_load_plan",
        lambda _path: ({"payload_sha256": "x"}, {}, {}, {}),
    )
    monkeypatch.setattr(
        startup.timeout12h,
        "load_submission_for_probe",
        lambda _path, plan: {"task_id": startup.FAILED_TASK_ID},
    )
    monkeypatch.setattr(
        startup,
        "_read_source_failure",
        lambda **_kwargs: _evaluation_plan()["startup_successor"][
            "source_failure_evidence"
        ],
    )
    monkeypatch.setattr(
        startup,
        "_effective_slots",
        lambda **_kwargs: {96304: slot_a, 96305: slot_b},
    )
    monkeypatch.setattr(
        startup, "_claim_state", lambda _plan: "unsubmitted"
    )
    monkeypatch.setattr(
        startup,
        "_read_json",
        lambda _path: {"schema_version": refill.EXTENSION_RECEIPT_SCHEMA},
    )
    capacity = {
        "ready_fit_slots": 2,
        "standalone_aedt_available": 10,
        "allocations": [
            {
                "account_name": startup.TARGET_ACCOUNT,
                "node_name": startup.TARGET_NODE,
                "state": "active",
                "fit_slots": 2,
            }
        ],
    }
    storage = {
        "limiting_quota": {
            "raw_free_gb": 110.0,
            "in_doubt_gb": 5.0,
        }
    }
    result = startup.evaluate(
        _evaluation_plan(),
        task_reader=lambda **_kwargs: [task_a, task_b],
        project_task_reader=lambda **_kwargs: [],
        capacity_reader=lambda **_kwargs: capacity,
        storage_reader=lambda _plan: storage,
        validate_cutover=False,
        deadline_reader=lambda: True,
    )
    active = 27.094282487407327 + 24.507468160241842
    expected = 105.0 - active - startup.PROSPECTIVE_GRID_GIB - 10.0
    assert result["eligible"] is True
    assert result["active_authenticated_storage_bound_gib"] == pytest.approx(
        active
    )
    assert result[
        "remaining_after_active_prospective_floor_gib"
    ] == pytest.approx(expected)


def test_evaluate_rejects_unapproved_active_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, slot = _active(96304, 96230, "b" * 64)
    unknown = copy.deepcopy(task)
    unknown.update(
        {"task_id": 99999, "name": "unknown", "dedupe_key": "unknown"}
    )
    monkeypatch.setattr(startup.refill, "load_plan", lambda _path: _safe_plan())
    monkeypatch.setattr(
        startup.timeout12h,
        "_load_plan",
        lambda _path: ({"payload_sha256": "x"}, {}, {}, {}),
    )
    monkeypatch.setattr(
        startup.timeout12h,
        "load_submission_for_probe",
        lambda _path, plan: {"task_id": startup.FAILED_TASK_ID},
    )
    monkeypatch.setattr(
        startup,
        "_read_source_failure",
        lambda **_kwargs: _evaluation_plan()["startup_successor"][
            "source_failure_evidence"
        ],
    )
    monkeypatch.setattr(
        startup, "_effective_slots", lambda **_kwargs: {96304: slot}
    )
    monkeypatch.setattr(
        startup, "_claim_state", lambda _plan: "unsubmitted"
    )
    monkeypatch.setattr(
        startup,
        "_read_json",
        lambda _path: {"schema_version": refill.EXTENSION_RECEIPT_SCHEMA},
    )
    result = startup.evaluate(
        _evaluation_plan(),
        task_reader=lambda **_kwargs: [task, unknown],
        project_task_reader=lambda **_kwargs: [],
        capacity_reader=lambda **_kwargs: {
            "ready_fit_slots": 1,
            "standalone_aedt_available": 1,
            "allocations": [
                {
                    "account_name": startup.TARGET_ACCOUNT,
                    "node_name": startup.TARGET_NODE,
                    "state": "active",
                    "fit_slots": 1,
                }
            ],
        },
        storage_reader=lambda _plan: {
            "limiting_quota": {
                "raw_free_gb": 200.0,
                "in_doubt_gb": 0.0,
            }
        },
        validate_cutover=False,
        deadline_reader=lambda: True,
    )
    assert result["eligible"] is False
    assert result["unapproved_active_mft_fea_task_ids"] == [99999]


def test_atomic_claim_allows_only_one_lifetime_fresh_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim_root = tmp_path / "claims"
    monkeypatch.setattr(startup, "CLAIM_ROOT", claim_root)
    startup.initialize_claim_root()
    reference = startup._claim_reference()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = {
        "payload_sha256": "a" * 64,
        "stage": {
            "profile_sha256": "b" * 64,
            "task_name": "successor",
            "retained_aedt_bundle": {"dedupe_key": "successor-dedupe"},
        },
    }
    winner = startup._claim_winner(plan_path, plan)
    first = atomic_claim.acquire_claim(claim_root, reference, winner)
    second = atomic_claim.acquire_claim(claim_root, reference, winner)
    assert first["status"] == "fresh_pending"
    assert second["status"] == "existing_pending"


def test_cutoff_is_exact_and_timezone_required() -> None:
    assert startup._deadline_open_at(
        startup.SUBMISSION_CUTOFF_UTC - timedelta(microseconds=1)
    )
    assert not startup._deadline_open_at(startup.SUBMISSION_CUTOFF_UTC)
    with pytest.raises(startup.StartupRetryError):
        startup._deadline_open_at(
            startup.SUBMISSION_CUTOFF_UTC.replace(tzinfo=None)
        )


def test_safe_refill_dispatches_replacement_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "l96223.json"
    path.write_text(
        json.dumps({"schema_version": startup.REPLACEMENT_SCHEMA}),
        encoding="utf-8",
    )
    expected = {"execution_task_id": 99001}
    monkeypatch.setattr(
        startup,
        "authenticate_watcher_replacement_receipt",
        lambda _path, *, authority: expected,
    )
    assert refill.authenticate_watcher_extension_receipt(
        path, authority={}
    ) == expected


def test_successor_sibling_is_singleton_and_collision_fails() -> None:
    plan = _evaluation_plan()
    exact = {
        "task_id": 99001,
        "name": "successor",
        "dedupe_key": "successor-dedupe",
        "project": startup.PROJECT,
        "cpus": 8,
        "memory_mb": 32768,
        "timeout_seconds": 43200,
        "aedt_backend": "standalone",
        "requested_node_name": startup.TARGET_NODE,
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
    }
    assert startup._successor_siblings([exact], plan=plan)[0][
        "task_id"
    ] == 99001
    duplicate = copy.deepcopy(exact)
    duplicate["task_id"] = 99002
    with pytest.raises(startup.StartupRetryError):
        startup._successor_siblings([exact, duplicate], plan=plan)
    collision = copy.deepcopy(exact)
    collision["dedupe_key"] = "wrong"
    with pytest.raises(startup.StartupRetryError):
        startup._successor_siblings([collision], plan=plan)


def test_watcher_extension_is_atomically_replaced_from_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "l96223.json"
    archive = tmp_path / "source.json"
    old = {
        "schema_version": refill.EXTENSION_RECEIPT_SCHEMA,
        "payload_sha256": "old",
    }
    target.write_text(json.dumps(old), encoding="utf-8")
    archive.write_text(json.dumps(old), encoding="utf-8")
    record = {
        "startup_successor": {
            "watcher_extension_target": str(target),
            "extension_authority": {"path": str(tmp_path / "authority.json")},
            "source_watcher_extension_archive": {
                "path": archive.name,
                "sha256": startup.production._sha256_file(archive),
            },
        }
    }
    replacement = startup._sealed(
        {
            "schema_version": startup.REPLACEMENT_SCHEMA,
            "successor_execution_task_id": 99001,
        }
    )
    monkeypatch.setattr(
        startup, "load_plan", lambda _path: (record, {}, {}, {})
    )
    monkeypatch.setattr(
        startup.refill,
        "authenticate_extension_authority",
        lambda _path: {},
    )
    monkeypatch.setattr(
        startup,
        "_replacement_receipt",
        lambda **_kwargs: replacement,
    )
    monkeypatch.setattr(
        startup,
        "authenticate_watcher_replacement_receipt",
        lambda _path, *, authority: {"execution_task_id": 99001},
    )
    result = startup._ensure_replacement(
        plan_path=tmp_path / "plan.json",
        submission_path=tmp_path / "submission.json",
    )
    assert result == target
    assert json.loads(target.read_text("utf-8"))["schema_version"] == (
        startup.REPLACEMENT_SCHEMA
    )
    assert json.loads(archive.read_text("utf-8")) == old


def test_direct_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/mft_goal_startup_retry.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "logical96223/task96303" in completed.stdout
