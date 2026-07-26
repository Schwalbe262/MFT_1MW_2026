from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tools import mft_goal_postdeadline_standard_collector as collector


SOLVER = "a" * 40
LIBRARY = "b" * 40
CANDIDATE = "c" * 64
TASK_NAME = "mft-goal-diag-standard-postdeadline-test-n107"


def _dump(path: Path, value: object) -> None:
    path.write_bytes(collector.canonical_bytes(value) + b"\n")


def _fixture(tmp_path: Path) -> dict[str, object]:
    params = {"N1_main": 6, "fan_velocity": 1.5}
    params_path = tmp_path / "fea_params.json"
    _dump(params_path, params)
    selected = collector.seal(
        {
            "schema_version": "selected-test-v1",
            "selected_row": {"candidate_physics_sha": CANDIDATE},
        }
    )
    selected_path = tmp_path / "selected_candidate.json"
    _dump(selected_path, selected)
    overrides = {
        "thermal_symmetry": "eighth",
        "full_model": 0,
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_on": 1,
        "core_plate_pad_t": 2.0,
        "wcp_on": 1,
        "wcp_pad_t": 2.0,
    }
    profile = {
        "fixed_boundary_contract": collector.FIXED_BOUNDARY,
        "param_overrides": overrides,
        "timeout_seconds": 43_200,
    }
    profile_path = tmp_path / "profile.json"
    _dump(profile_path, profile)
    persisted_params = json.loads(params_path.read_text())
    persisted_profile = json.loads(profile_path.read_text())
    merged = dict(persisted_params)
    merged.update(persisted_profile["param_overrides"])
    parameter_json = json.dumps(
        merged,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    parameter_digest = hashlib.sha256(parameter_json.encode()).hexdigest()[:16]
    dedupe = f"mft-al:{TASK_NAME}:{SOLVER}:{LIBRARY}:{parameter_digest}"
    command = (
        "timeout --signal=TERM --kill-after=300s 43200s "
        "python run_simulation_260706.py --fixed --thermal --headless"
    )
    scheduler_payload = {
        "name": TASK_NAME,
        "dedupe_key": dedupe,
        "project": collector.SCHEDULER_PROJECT,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "node_name": "n107",
        "node_name_policy": "strict",
        "aedt_backend": "standalone",
        "command": command,
    }
    plan = collector.seal(
        {
            "schema_version": collector.PLAN_SCHEMA,
            **collector.CLASSIFICATION,
            "fixed_physics_unchanged": True,
            "candidate_physics_sha256": CANDIDATE,
            "task_name": TASK_NAME,
            "dedupe_key": dedupe,
            "scheduler_url": "http://127.0.0.1:8002",
            "resources": {
                "cpus": 8,
                "memory_mb": 98_304,
                "same_node_as_task_id": 0,
                "dependency_task_id": 0,
                "node_name": "n107",
            },
            "timeout_envelope": {
                "solver_seconds": 43_200,
                "kill_grace_seconds": 300,
                "retention_seconds": 1_800,
                "scheduler_timeout_seconds": 45_300,
            },
            "scheduler_payload": scheduler_payload,
            "scheduler_payload_sha256": collector.payload_sha256(scheduler_payload),
            "preflight": {
                "fixed_boundary": collector.FIXED_BOUNDARY,
                "candidate_boundary_projection": collector.CANDIDATE_BOUNDARY,
                "candidate_physics_sha256": CANDIDATE,
                "candidate_reauthenticated": True,
                "fixed_physics_unchanged": True,
                "revision_reauthenticated": True,
            },
            "source_artifacts": {
                params_path.name: collector.file_record(params_path),
                selected_path.name: collector.file_record(selected_path),
            },
            "execution_profile": collector.file_record(profile_path),
            "execution_profile_canonical_sha256": collector.payload_sha256(profile),
            "solver_revision": SOLVER,
            "library_revision": LIBRARY,
        }
    )
    plan_path = tmp_path / "plan.json"
    _dump(plan_path, plan)
    submission = collector.seal(
        {
            "schema_version": collector.SUBMISSION_SCHEMA,
            **collector.CLASSIFICATION,
            "fixed_physics_unchanged": True,
            "candidate_physics_sha256": CANDIDATE,
            "task_id": 123,
            "task_name": TASK_NAME,
            "dedupe_key": dedupe,
            "plan": collector.file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "scheduler_payload_sha256": plan["scheduler_payload_sha256"],
        }
    )
    submission_path = tmp_path / "submission.json"
    _dump(submission_path, submission)
    return {
        "plan_path": plan_path,
        "submission_path": submission_path,
        "receipt_sha": collector.sha256_file(submission_path),
        "dedupe": dedupe,
    }


def _load(fixture: dict[str, object]) -> dict[str, object]:
    return collector.load_contract(
        plan_path=fixture["plan_path"],
        submission_path=fixture["submission_path"],
        expected_task_id=123,
        expected_task_name=TASK_NAME,
        expected_dedupe_key=fixture["dedupe"],
        expected_candidate_sha256=CANDIDATE,
        expected_receipt_sha256=fixture["receipt_sha"],
        scheduler_url="http://127.0.0.1:8002",
    )


def _task(contract: dict[str, object], state: str) -> dict[str, object]:
    return {
        "task_id": contract["task_id"],
        "name": contract["task_name"],
        "dedupe_key": contract["dedupe_key"],
        "project": collector.SCHEDULER_PROJECT,
        "requested_node_name": contract["node_name"],
        "requested_node_name_policy": "strict",
        "same_node_as_task_id": 0,
        "cpus": 8,
        "memory_mb": 98_304,
        "timeout_seconds": 45_300,
        "state": state,
        "status": state,
        "actual_node_name": "n107",
        "allocation_id": 99,
        "slurm_job_id": "100",
    }


def test_contract_and_task_identity_are_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract = _load(fixture)
    task = _task(contract, "running")

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return collector.canonical_bytes(task)

    assert collector.get_task(contract, getter=getter) == task
    drifted = copy.deepcopy(task)
    drifted["dedupe_key"] = "drifted"

    def bad_getter(*_args: object, **_kwargs: object) -> bytes:
        return collector.canonical_bytes(drifted)

    with pytest.raises(collector.CollectionError, match="dedupe_key"):
        collector.get_task(contract, getter=bad_getter)
    with pytest.raises(collector.CollectionError, match="expected watcher"):
        collector.load_contract(
            plan_path=fixture["plan_path"],
            submission_path=fixture["submission_path"],
            expected_task_id=123,
            expected_task_name=TASK_NAME,
            expected_dedupe_key=fixture["dedupe"],
            expected_candidate_sha256=CANDIDATE,
            expected_receipt_sha256="0" * 64,
            scheduler_url="http://127.0.0.1:8002",
        )


def test_terminal_failure_writes_no_scientific_claim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    contract = _load(fixture)
    task = _task(contract, "failed")

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return collector.canonical_bytes(task)

    output = tmp_path / "collection"
    terminal, event = collector.poll_once(
        contract=contract,
        output=output,
        getter=getter,
    )
    assert terminal is True
    assert event["event"] == "failure_ledger"
    ledger = json.loads((tmp_path / "collection.failure_ledger.json").read_text())
    assert ledger["scientific_pass_claimed"] is False
    assert ledger["scientific_infeasible_claimed"] is False
    assert ledger["collection_performed"] is False
    assert not output.exists()


def test_chunk_reconstruction_verifies_sha(tmp_path: Path) -> None:
    raw = b"test-aedt"
    encoded = base64.b64encode(raw)
    receipt = {
        "transport_chunk_count": 1,
        "transport_chunk_directory": "goal/chunks",
        "artifact_size_bytes": len(raw),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    contract = {"retained": {}, "scheduler_url": "http://test", "task_id": 1}

    def getter(*_args: object, **_kwargs: object) -> bytes:
        return encoded

    artifact, inventory = collector._fetch_aedt_and_chunks(  # noqa: SLF001
        contract=contract,
        receipt=receipt,
        staging=tmp_path,
        getter=getter,
    )
    assert artifact.read_bytes() == raw
    assert len(inventory) == 1
    assert inventory[0]["sha256"] == hashlib.sha256(encoded).hexdigest()


def test_results_manifest_tree_validation() -> None:
    files = [
        {
            "path": "a/file.bin",
            "sha256": "d" * 64,
            "size_bytes": 7,
        }
    ]
    tree = collector.payload_sha256(files)
    manifest = {
        "schema_version": collector.RESULTS_MANIFEST_SCHEMA,
        "source_project_name": "symmetric",
        "source_results_directory_name": "symmetric.aedtresults",
        "retained_results_directory_name": "symmetric.aedtresults",
        "file_count": 1,
        "size_bytes": 7,
        "tree_sha256": tree,
        "files": files,
    }
    raw = collector.canonical_bytes(manifest) + b"\n"
    receipt = {
        "results_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "results_file_count": 1,
        "results_size_bytes": 7,
        "results_tree_sha256": tree,
    }
    result = {"project_name": "symmetric"}
    assert (
        collector._validate_manifest(  # noqa: SLF001
            manifest, raw, receipt, result
        )
        == manifest
    )


def test_source_contains_no_scheduler_post() -> None:
    source = inspect.getsource(collector)
    assert 'method="POST"' not in source
    assert "requests.post" not in source
