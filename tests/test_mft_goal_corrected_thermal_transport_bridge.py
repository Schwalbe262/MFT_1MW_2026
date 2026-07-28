from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import mft_goal_corrected_thermal_transport_bridge as bridge
from tools import mft_goal_final_package_gate as final_gate


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _thermal_seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = bridge.thermal_canonical_sha256(result)
    return result


def _source_task() -> dict[str, Any]:
    return {
        **bridge._task_exact_fields(),
        "status": "completed",
        "state": "succeeded",
        "exit_code": 0,
        "started_at": "2026-07-26 09:00:00",
        "finished_at": "2026-07-26 10:00:00",
    }


def _plan_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    plan = {
        "plan_payload_sha256": bridge.SOURCE_PLAN_PAYLOAD_SHA256,
        "canonical_command_sha256": bridge.SOURCE_COMMAND_SHA256,
        "canonical": False,
        "diagnostic_only": True,
        "production_truth_eligible": False,
        "task_identity": {
            "name": bridge.SOURCE_TASK_NAME,
            "dedupe_key": bridge.SOURCE_DEDUPE_KEY,
            "candidate_sha256": bridge.SOURCE_CANDIDATE_SHA256,
        },
        "contract": {
            "fixed_physics": bridge.FIXED_PHYSICS,
            "library_revision": ("e6b9b9d20a832ff5c3f7ca97218737a0b8650781"),
        },
        "executor": {"revision": bridge.SOURCE_EXECUTOR_REVISION},
        "execution_contract": {
            "checkpoint_manifest_sha256": (bridge.SOURCE_CHECKPOINT_MANIFEST_SHA256),
            "executor_revision": bridge.SOURCE_EXECUTOR_REVISION,
            "plan_payload_sha256": (bridge.SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256),
            "submission_contract_sha256": (bridge.SOURCE_SUBMISSION_CONTRACT_SHA256),
        },
    }
    submission = {
        "payload_sha256": bridge.SOURCE_SUBMISSION_PAYLOAD_SHA256,
        "command_sha256": bridge.SOURCE_COMMAND_SHA256,
        "scheduler_post_calls": 1,
        "scheduler_submission_performed": True,
        "task_readback": {
            "task_id": bridge.SOURCE_TASK_ID,
            "name": bridge.SOURCE_TASK_NAME,
            "dedupe_key": bridge.SOURCE_DEDUPE_KEY,
        },
    }
    plan_path = tmp_path / "source-plan.json"
    submission_path = tmp_path / "source-submission.json"
    _json(plan_path, plan)
    _json(submission_path, submission)
    monkeypatch.setattr(
        bridge, "SOURCE_PLAN_FILE_SHA256", bridge.sha256_file(plan_path)
    )
    monkeypatch.setattr(
        bridge,
        "SOURCE_SUBMISSION_FILE_SHA256",
        bridge.sha256_file(submission_path),
    )
    return plan_path, submission_path


def _retained_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    retained = tmp_path / "retained" / "b7c-ce0bf14dbdb8-a0208331949f"
    retained.mkdir(parents=True)
    monkeypatch.setattr(bridge, "SOURCE_RETAINED_ROOT", retained)
    monkeypatch.setattr(
        bridge,
        "SOURCE_RETAINED_REMOTE",
        "fixture/retained/b7c-ce0bf14dbdb8-a0208331949f",
    )
    temperatures = {
        "T_max_Tx": 94.0,
        "T_max_Rx_main": 95.0,
        "T_max_Rx_side": 96.0,
        "T_max_core": 112.0,
    }
    observation = {
        "winding_max_c": 96.0,
        "winding_limit_c": 100.0,
        "winding_pass": True,
        "core_max_c": 112.0,
        "core_limit_c": 120.0,
        "core_pass": True,
        "all_temperature_constraints_pass": True,
        "diagnostic_only": True,
    }
    source = {
        "candidate_sha256": bridge.SOURCE_CANDIDATE_SHA256,
        "solver_revision": "a1e4f70cefa1af04673c73a6131bf490c0cc14b5",
        "library_revision": "e6b9b9d20a832ff5c3f7ca97218737a0b8650781",
    }
    executor = {
        "executor_solver_revision": bridge.SOURCE_EXECUTOR_REVISION,
        "execution_plan_sha256": (bridge.SOURCE_EXECUTION_PLAN_FILE_SHA256),
        "execution_plan_payload_sha256": (bridge.SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256),
        "imported": {
            "executor_solver_revision": bridge.SOURCE_EXECUTOR_REVISION,
            "executor_solver_dirty": 0,
            "pyaedt_library_revision": ("e6b9b9d20a832ff5c3f7ca97218737a0b8650781"),
            "pyaedt_library_dirty": 0,
        },
    }
    corrected = {
        "schema": "mft-corrected-thermal-diagnostic-result-v1",
        "diagnostic_only": True,
        "canonical": False,
        "source_provenance": source,
        "executor_provenance": executor,
        "constraint_observation": observation,
        "temperatures": temperatures,
        "temperature_extraction": {"passed": True},
        "convergence": {"thermal_converged": 1},
        "parallel_attestation": {"passed": True},
        "native_fixed_readback": {
            "schema": "mft-corrected-thermal-native-fixed-readback-v2",
            "passed": True,
        },
    }
    execution = {
        "schema": "mft-corrected-thermal-checkpoint-execution-v1",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "canonical": False,
        "production_truth_eligible": False,
        "source_checkpoint_manifest_sha256": (bridge.SOURCE_CHECKPOINT_MANIFEST_SHA256),
        "source_provenance": source,
        "executor_provenance": executor,
        "fixed_physics": bridge.FIXED_PHYSICS,
        "solver_core_contract": {
            "scheduler_task_id": bridge.SOURCE_TASK_ID,
            "slurm_job_id": int(bridge.SOURCE_SLURM_JOB_ID),
            "slurm_cpus_per_task": bridge.SOURCE_CPUS,
            "passed": True,
        },
        "constraint_observation": observation,
        "temperatures": temperatures,
    }
    symmetric = retained / "symmetric.aedt"
    symmetric.write_bytes((b"corrected-terminal-aedt-" * 60_000)[:1_200_123])
    corrected_path = retained / "corrected_result.json"
    execution_path = retained / "execution_receipt.json"
    convergence = retained / "convergence" / "monitor.csv"
    profile = retained / "profile" / "latest.profile"
    _json(corrected_path, corrected)
    _json(execution_path, execution)
    convergence.parent.mkdir()
    profile.parent.mkdir()
    convergence.write_text("iteration,temp\n1,96\n", encoding="utf-8")
    profile.write_text("profile", encoding="utf-8")
    files = []
    for path, relative in (
        (symmetric, "symmetric.aedt"),
        (corrected_path, "corrected_result.json"),
        (execution_path, "execution_receipt.json"),
        (convergence, "convergence/monitor.csv"),
        (profile, "profile/latest.profile"),
    ):
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": bridge.sha256_file(path),
            }
        )
    manifest = _thermal_seal(
        {
            "schema": ("mft-corrected-thermal-minimum-retained-package-v1"),
            "diagnostic_only": True,
            "canonical": False,
            "candidate_sha256": bridge.SOURCE_CANDIDATE_SHA256,
            "checkpoint_manifest_sha256": (bridge.SOURCE_CHECKPOINT_MANIFEST_SHA256),
            "source_solver_revision": ("a1e4f70cefa1af04673c73a6131bf490c0cc14b5"),
            "executor_solver_revision": bridge.SOURCE_EXECUTOR_REVISION,
            "executor_required_ancestor": "a" * 40,
            "aedt_mixed_provenance": "fixture",
            "files": files,
            "minimum_bundle_bytes": sum(int(row["size_bytes"]) for row in files),
            "optional_field_bundle": {"retained": False},
            "control_evidence": {},
        }
    )
    manifest_path = retained / "manifest.json"
    _json(manifest_path, manifest)
    marker = _thermal_seal(
        {
            "schema": "mft-corrected-thermal-retention-marker-v1",
            "diagnostic_only": True,
            "canonical": False,
            "production_truth_eligible": False,
            "task_name": bridge.SOURCE_TASK_NAME,
            "dedupe_key": bridge.SOURCE_DEDUPE_KEY,
            "submission_contract_sha256": (bridge.SOURCE_SUBMISSION_CONTRACT_SHA256),
            "execution_plan_payload_sha256": (
                bridge.SOURCE_EXECUTION_PLAN_PAYLOAD_SHA256
            ),
            "retained_destination": str(retained),
            "retention_manifest_path": str(manifest_path),
            "retention_manifest_payload_sha256": manifest["payload_sha256"],
            "retention_manifest_file_sha256": bridge.sha256_file(manifest_path),
            "corrected_receipt_path": str(execution_path),
            "corrected_receipt_sha256": bridge.sha256_file(execution_path),
            "execution_exit_code": 0,
            "verified_files": files,
            "minimum_bundle_bytes": manifest["minimum_bundle_bytes"],
            "optional_field_bundle": {"retained": False},
            "corrected_receipt": {},
        }
    )
    return {
        "retained": retained,
        "marker": marker,
        "files": files,
        "manifest": manifest,
        "corrected": corrected,
    }


def _task_files(
    tmp_path: Path,
    marker: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, bytes, bytes]:
    task = tmp_path / "workspace" / "2026-07-26" / "task-96324-1785056187"
    task.mkdir(parents=True)
    task_sh = (
        "#!/usr/bin/env bash\n"
        "export SLURM_SCHED_TASK_ID=96324\n"
        "echo 'strict node placement mismatch: expected n111'\n"
        "echo 'strict allocation mismatch: expected 839461'\n"
    ).encode()
    stdout = (
        b"solver complete\nCORRECTED_THERMAL_JSON "
        + json.dumps(marker, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    stderr = b""
    (task / "task.sh").write_bytes(task_sh)
    (task / "stdout.log").write_bytes(stdout)
    (task / "stderr.log").write_bytes(stderr)
    (task / "exit_code").write_text("0\n", encoding="ascii")
    monkeypatch.setattr(bridge, "SOURCE_TASK_SH_SIZE", len(task_sh))
    monkeypatch.setattr(bridge, "SOURCE_TASK_SH_SHA256", bridge.sha256_bytes(task_sh))
    return task, stdout, stderr


def _published_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    plan_path, submission_path = _plan_files(tmp_path, monkeypatch)
    retained = _retained_fixture(tmp_path, monkeypatch)
    task_directory, stdout, stderr = _task_files(
        tmp_path, retained["marker"], monkeypatch
    )
    authority = bridge.build_terminal_success_authority(
        task=_source_task(),
        stdout=stdout,
        stderr=stderr,
        plan_path=plan_path,
        submission_path=submission_path,
        observed_at_utc="2026-07-26T10:01:00+00:00",
    )
    authority_path = tmp_path / "source-authority.json"
    _json(authority_path, authority)
    workspace = tmp_path / "workspace"
    result = bridge.publish_terminal_transport(
        authority_path=authority_path,
        source_retained_root=retained["retained"],
        source_task_directory=task_directory,
        output_relative_root="transport/output",
        cwd=workspace,
        environ={
            "SLURM_SCHED_TASK_ID": "97001",
            "SLURM_JOB_ID": "900001",
            "SLURM_CPUS_PER_TASK": "1",
        },
        hostname="n111",
    )
    return {
        "plan": plan_path,
        "submission": submission_path,
        "authority": authority,
        "transport": Path(result["transport_directory"]),
        "result": result,
    }


class _PendingClient:
    get_count = 0

    def get_json(self, endpoint: str) -> dict[str, Any]:
        assert endpoint == "/api/tasks/96324"
        self.get_count += 1
        return {
            **_source_task(),
            "status": "running",
            "state": "running",
            "exit_code": None,
            "finished_at": None,
        }


def test_prepare_is_get_only_and_fail_closed_while_source_runs(
    tmp_path: Path,
):
    result = bridge.prepare_from_scheduler_get(
        client=_PendingClient(),  # type: ignore[arg-type]
        plan_path=tmp_path / "not-read.json",
        submission_path=tmp_path / "not-read.json",
        executor_revision="a" * 40,
        publisher_sha256="b" * 64,
        observed_at_utc="2026-07-26T09:30:00+00:00",
    )

    assert result["schema"] == bridge.PENDING_SCHEMA
    assert result["transport_authorized"] is False
    assert result["scheduler_get_calls"] == 1
    assert result["scheduler_mutations"] == 0
    assert result["scientific_pass_claimed"] is False


def test_terminal_publisher_chunks_more_than_one_mib_and_binds_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _published_fixture(tmp_path, monkeypatch)
    receipt = json.loads(
        (fixture["transport"] / "transport_receipt.json").read_text(encoding="utf-8")
    )

    assert fixture["result"]["status"] == "published"
    assert fixture["result"]["artifact_size_bytes"] > 1_048_576
    assert fixture["result"]["chunk_count"] == 4
    assert all(
        path.stat().st_size <= bridge.MAX_ENCODED_CHUNK_BYTES
        for path in (fixture["transport"] / "chunks").glob("*.b64")
    )
    assert receipt["source_task"]["task_id"] == bridge.SOURCE_TASK_ID
    assert receipt["source_task"]["actual_node_name"] == "n111"
    assert receipt["source_task"]["slurm_job_id"] == "839461"
    assert receipt["source_plan"]["file_sha256"] == bridge.SOURCE_PLAN_FILE_SHA256
    assert (
        receipt["corrected_result_sha256"]
        == fixture["result"]["corrected_result_sha256"]
    )
    assert receipt["transport_semantics"].startswith(
        "terminal-success retained GPFS package"
    )


def test_materializer_emits_final_gate_compatible_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _published_fixture(tmp_path, monkeypatch)
    materialized = bridge.materialize_collector_event(
        transport_directory=fixture["transport"],
        output_root=tmp_path / "thermal-materialized",
        source_plan_path=fixture["plan"],
        source_submission_path=fixture["submission"],
        observed_at_utc="2026-07-26T10:02:00+00:00",
    )
    event = json.loads(
        Path(materialized["thermal_event_path"]).read_text(encoding="utf-8")
    )
    unsigned = dict(event)
    claimed = unsigned.pop("event_sha256")

    assert materialized["final_package_gate_compatible"] is True
    assert event["target"] == "thermal96324"
    assert event["state"] == "success_collected_diagnostic"
    assert event["artifact_collection_attempted"] is True
    assert event["contract"]["terminal_success_retained_transport_verified"] is True
    assert event["contract"]["open_only_snapshot_used"] is False
    assert claimed == bridge.sha256_bytes(bridge.canonical_bytes(unsigned))
    assert {row["path"] for row in event["collected_files"]} == {
        "symmetric.aedt",
        "corrected_result.json",
    }
    assert Path(materialized["symmetric_aedt_path"]).stat().st_size > 1_048_576
    gate_event, pending = final_gate._validate_terminal_event(
        Path(materialized["thermal_event_path"]),
        target="thermal96324",
        task_id=96324,
    )
    assert pending == []
    gate_symmetric, gate_result, _result, _plan = final_gate._load_thermal_sources(
        Path(materialized["thermal_event_path"]),
        gate_event,
        authority_plan={"candidate_physics_sha256": bridge.SOURCE_CANDIDATE_SHA256},
    )
    assert gate_symmetric == Path(materialized["symmetric_aedt_path"])
    assert gate_result == Path(materialized["corrected_result_path"])


def test_materializer_rejects_one_tampered_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _published_fixture(tmp_path, monkeypatch)
    chunk = fixture["transport"] / "chunks" / "00000001.b64"
    chunk.chmod(0o600)
    data = bytearray(chunk.read_bytes())
    data[3] ^= 1
    chunk.write_bytes(data)

    with pytest.raises(bridge.BridgeError, match="encoded chunk drifted"):
        bridge.materialize_collector_event(
            transport_directory=fixture["transport"],
            output_root=tmp_path / "thermal-materialized",
            source_plan_path=fixture["plan"],
            source_submission_path=fixture["submission"],
        )


def test_ready_plan_is_unsubmitted_and_exactly_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _published_fixture(tmp_path, monkeypatch)
    plan = bridge.build_bridge_plan(
        authority=fixture["authority"],
        executor_revision="c" * 40,
        publisher_sha256="d" * 64,
        orchestration_root=tmp_path / "orchestration",
        source_plan_path=fixture["plan"],
        source_submission_path=fixture["submission"],
    )

    assert plan["scheduler"]["post_calls_performed"] == 0
    assert plan["scheduler"]["submission_performed"] is False
    assert plan["submission_profile"]["cpus"] == 1
    assert plan["submission_profile"]["memory_mb"] == 4096
    assert plan["submission_profile"]["timeout_seconds"] == 1800
    assert plan["transport"]["maximum_encoded_chunk_bytes"] == 512_000
    assert bridge.SOURCE_RETAINED_ROOT.as_posix() in plan["canonical_command"]
    assert "mft_goal_corrected_thermal_transport_bridge.py" in plan["canonical_command"]
    assert plan["remote_ref"] == bridge.BRIDGE_REMOTE_REF == "refs/heads/main"
    fetch_commands = [
        line
        for line in plan["canonical_command"].splitlines()
        if line.startswith('git -C "$toolroot" fetch ')
    ]
    assert fetch_commands == [bridge.BRIDGE_FETCH_COMMAND]
    assert "refs/heads/integration/mft-goal-20260726" not in plan["canonical_command"]

    with pytest.raises(bridge.BridgeError, match="bridge remote ref drifted"):
        bridge.build_bridge_plan(
            authority=fixture["authority"],
            executor_revision="c" * 40,
            publisher_sha256="d" * 64,
            orchestration_root=tmp_path / "legacy-orchestration",
            source_plan_path=fixture["plan"],
            source_submission_path=fixture["submission"],
            remote_ref="refs/heads/integration/mft-goal-20260726",
        )

    legacy_plan = dict(plan)
    legacy_plan.pop("payload_sha256")
    legacy_plan["remote_ref"] = "refs/heads/integration/mft-goal-20260726"
    with pytest.raises(bridge.BridgeError, match="bridge submission plan drifted"):
        bridge._verify_bridge_plan(bridge.sealed(legacy_plan))


class _BridgeGetClient:
    def __init__(
        self,
        *,
        plan: dict[str, Any],
        transport: Path,
        bridge_task_id: int,
    ):
        self.plan = plan
        self.transport = transport
        self.bridge_task_id = bridge_task_id
        self.get_count = 0

    def get_json(self, endpoint: str) -> dict[str, Any]:
        self.get_count += 1
        assert endpoint == f"/api/tasks/{self.bridge_task_id}"
        profile = self.plan["submission_profile"]
        return {
            "task_id": self.bridge_task_id,
            "name": profile["name"],
            "dedupe_key": profile["dedupe_key"],
            "account_name": bridge.SOURCE_ACCOUNT,
            "cpus": 1,
            "memory_mb": 4096,
            "timeout_seconds": 1800,
            "node_name": "n111",
            "actual_node_name": "n111",
            "status": "completed",
            "state": "succeeded",
            "exit_code": 0,
            "slurm_job_id": "900001",
        }

    def remote_file(
        self,
        task_id: int,
        path: str,
        *,
        base: str = "remote_cwd",
        maximum_bytes: int = bridge.MAX_JSON_BYTES,
    ) -> bytes:
        self.get_count += 1
        assert task_id == self.bridge_task_id
        if path == "task.sh":
            assert base == "remote_dir"
            return (
                f"export SLURM_SCHED_TASK_ID={task_id}\n"
                "echo 'strict node placement mismatch: expected n111'\n"
                "echo 'strict allocation mismatch: expected 900001'\n"
                f"{self.plan['canonical_command']}\n"
            ).encode()
        prefix = f"{bridge.OUTPUT_RELATIVE_ROOT}/{bridge.TRANSPORT_DIRECTORY_NAME}/"
        assert path.startswith(prefix)
        source = self.transport.joinpath(*Path(path[len(prefix) :]).parts)
        data = source.read_bytes()
        assert len(data) <= maximum_bytes
        return data


def _ready_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    fixture = _published_fixture(tmp_path, monkeypatch)
    root = tmp_path / "orchestration"
    plan = bridge.build_bridge_plan(
        authority=fixture["authority"],
        executor_revision="c" * 40,
        publisher_sha256="d" * 64,
        orchestration_root=root,
        source_plan_path=fixture["plan"],
        source_submission_path=fixture["submission"],
    )
    bridge._write_ready_outputs(
        root,
        {
            "transport_authorized": True,
            "authority": fixture["authority"],
            "plan": plan,
        },
    )
    return {**fixture, "root": root, "bridge_plan": plan}


class _FakeMutationLock:
    def __init__(self, client: "_SubmissionClient"):
        self.client = client

    def __enter__(self) -> "_FakeMutationLock":
        assert self.client.inside_lock is False
        self.client.inside_lock = True
        self.client.lock_entries += 1
        return self

    def __exit__(self, *_args: object) -> None:
        self.client.inside_lock = False


class _SubmissionClient:
    def __init__(self, *, plan: dict[str, Any], root: Path):
        self.plan = plan
        self.root = root
        self.get_count = 0
        self.post_count = 0
        self.inside_lock = False
        self.lock_entries = 0
        self.call_log: list[tuple[str, bool]] = []
        profile = plan["submission_profile"]
        self.task = {
            "task_id": 97001,
            "name": profile["name"],
            "dedupe_key": profile["dedupe_key"],
            "project": profile["project"],
            "cpus": profile["cpus"],
            "memory_mb": profile["memory_mb"],
            "timeout_seconds": profile["timeout_seconds"],
            "max_workers_per_node": profile["max_workers_per_node"],
            "aedt_backend": profile["aedt_backend"],
            "env_profile": profile["env_profile"],
            "required_capability": profile["required_capability"],
            "requested_account_name": bridge.SOURCE_ACCOUNT,
            "requested_node_name": bridge.SOURCE_NODE,
            "requested_node_name_policy": "strict",
            "account_name": bridge.SOURCE_ACCOUNT,
            "node_name": bridge.SOURCE_NODE,
            "status": "queued",
            "state": "queued",
        }

    def lock(self) -> _FakeMutationLock:
        return _FakeMutationLock(self)

    def _record(self, endpoint: str) -> None:
        self.get_count += 1
        self.call_log.append((endpoint, self.inside_lock))

    def get_json(self, endpoint: str) -> dict[str, Any]:
        self._record(endpoint)
        if endpoint == "/api/health":
            return {
                "ok": True,
                "scheduler_ok": True,
                "scheduler_thread_alive": True,
                "scheduler_stalled": False,
            }
        if endpoint == f"/api/tasks/{bridge.SOURCE_TASK_ID}":
            return _source_task()
        if endpoint == bridge._capacity_endpoint():
            return {
                "queue_state": "ready",
                "ready_fit_slots": 4,
                "memory_pressure_state": "ok",
                "standalone_aedt_available": 4,
                "allocations": [
                    {
                        "allocation_id": bridge.SOURCE_ALLOCATION_ID,
                        "account_name": bridge.SOURCE_ACCOUNT,
                        "node_name": bridge.SOURCE_NODE,
                        "state": "active",
                        "fit_slots": 4,
                        "free_cpus": 8,
                        "free_memory_mb": 32_768,
                        "memory_pressure_state": "ok",
                    }
                ],
            }
        if endpoint == "/api/tasks/97001":
            return dict(self.task)
        raise AssertionError(endpoint)

    def get_value(self, endpoint: str) -> Any:
        self._record(endpoint)
        if endpoint == "/api/allocations":
            return [
                {
                    "id": bridge.SOURCE_ALLOCATION_ID,
                    "account_name": bridge.SOURCE_ACCOUNT,
                    "node_name": bridge.SOURCE_NODE,
                    "state": "active",
                    "slurm_job_id": bridge.SOURCE_SLURM_JOB_ID,
                    "free_cpus": 8,
                    "free_memory_mb": 32_768,
                }
            ]
        if endpoint == bridge._collision_endpoint(self.plan):
            return [dict(self.task)] if self.post_count else []
        raise AssertionError(endpoint)

    def post(
        self, endpoint: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], None]:
        assert self.inside_lock is True
        assert endpoint == "/api/tasks"
        assert payload == self.plan["scheduler_payload"]
        assert (self.root / "bridge_post_attempt.json").is_file()
        self.post_count += 1
        assert self.post_count == 1
        return 201, {"task_id": 97001}, None


def test_get_only_collector_materializes_bridge_for_final_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _published_fixture(tmp_path, monkeypatch)
    plan = bridge.build_bridge_plan(
        authority=fixture["authority"],
        executor_revision="c" * 40,
        publisher_sha256="d" * 64,
        orchestration_root=tmp_path / "orchestration",
        source_plan_path=fixture["plan"],
        source_submission_path=fixture["submission"],
    )
    bridge_plan_path = tmp_path / "bridge-plan.json"
    _json(bridge_plan_path, plan)
    client = _BridgeGetClient(
        plan=plan,
        transport=fixture["transport"],
        bridge_task_id=97001,
    )

    result = bridge.collect_bridge_transport_get_only(
        client=client,  # type: ignore[arg-type]
        bridge_task_id=97001,
        bridge_plan_path=bridge_plan_path,
        output_root=tmp_path / "collected",
        source_plan_path=fixture["plan"],
        source_submission_path=fixture["submission"],
        observed_at_utc="2026-07-26T10:03:00+00:00",
    )

    assert result["status"] == "collected_and_materialized"
    assert result["scheduler_mutations"] == 0
    assert result["final_package_gate_compatible"] is True
    assert Path(result["thermal_event_path"]).is_file()
    assert Path(result["symmetric_aedt_path"]).stat().st_size > 1_048_576


def test_single_post_ledger_is_idempotent_and_output_cannot_be_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _ready_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    client = _SubmissionClient(plan=fixture["bridge_plan"], root=root)

    first = bridge.submit_ready_plan_once(
        client=client,  # type: ignore[arg-type]
        poster=client.post,  # type: ignore[arg-type]
        output_root=root,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        lock_factory=client.lock,
        required_root=root,
    )
    get_count_after_first = client.get_count
    second = bridge.submit_ready_plan_once(
        client=client,  # type: ignore[arg-type]
        poster=client.post,  # type: ignore[arg-type]
        output_root=root,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        lock_factory=client.lock,
        required_root=root,
    )

    assert first["status"] == "submitted"
    assert first["scheduler_post_calls_this_invocation"] == 1
    assert second["status"] == "already_submitted"
    assert second["scheduler_post_calls_this_invocation"] == 0
    assert client.post_count == 1
    assert client.get_count == get_count_after_first
    assert client.lock_entries == 1
    capacity_states = [
        inside
        for endpoint, inside in client.call_log
        if endpoint == bridge._capacity_endpoint()
    ]
    collision_states = [
        inside
        for endpoint, inside in client.call_log
        if endpoint == bridge._collision_endpoint(fixture["bridge_plan"])
    ]
    assert capacity_states == [False, True]
    assert collision_states == [False, True]

    with pytest.raises(bridge.BridgeError, match="campaign-fixed root"):
        bridge.submit_ready_plan_once(
            client=client,  # type: ignore[arg-type]
            poster=client.post,  # type: ignore[arg-type]
            output_root=tmp_path / "different-output",
            authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
            lock_factory=client.lock,
            required_root=root,
        )
    assert client.post_count == 1
    assert client.get_count == get_count_after_first


def test_running_and_terminal_failure_watch_paths_never_post(
    tmp_path: Path,
):
    calls = 0

    def forbidden_post(
        _endpoint: str, _payload: dict[str, Any]
    ) -> tuple[None, None, str]:
        nonlocal calls
        calls += 1
        return None, None, "forbidden"

    running_root = tmp_path / "running"
    running = bridge.watch_submit_orchestrator(
        client=_PendingClient(),  # type: ignore[arg-type]
        poster=forbidden_post,  # type: ignore[arg-type]
        output_root=running_root,
        source_plan_path=tmp_path / "not-read-plan.json",
        source_submission_path=tmp_path / "not-read-submission.json",
        executor_revision="a" * 40,
        publisher_sha256="b" * 64,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        watch=False,
        required_root=running_root,
    )
    assert running["status"] == "pending"
    assert running["scheduler_post_calls"] == 0
    assert not running_root.exists()

    class _FailedClient:
        get_count = 0

        def get_json(self, endpoint: str) -> dict[str, Any]:
            assert endpoint == f"/api/tasks/{bridge.SOURCE_TASK_ID}"
            self.get_count += 1
            return {
                **_source_task(),
                "state": "failed",
                "exit_code": 1,
            }

    failed_root = tmp_path / "failed"
    failed = bridge.watch_submit_orchestrator(
        client=_FailedClient(),  # type: ignore[arg-type]
        poster=forbidden_post,  # type: ignore[arg-type]
        output_root=failed_root,
        source_plan_path=tmp_path / "not-read-plan.json",
        source_submission_path=tmp_path / "not-read-submission.json",
        executor_revision="a" * 40,
        publisher_sha256="b" * 64,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        watch=False,
        required_root=failed_root,
    )
    assert failed["status"] == "source_terminal_failure"
    assert failed["scheduler_post_calls"] == 0
    assert calls == 0
    assert (failed_root / "source_terminal_failure.json").is_file()


def test_pending_watch_heartbeat_is_sealed_sibling_and_does_not_create_plan_root(
    tmp_path: Path,
):
    calls = 0

    def forbidden_post(
        _endpoint: str, _payload: dict[str, Any]
    ) -> tuple[None, None, str]:
        nonlocal calls
        calls += 1
        return None, None, "forbidden"

    root = tmp_path / "orchestration"
    state_path = tmp_path / bridge.WATCH_STATE_FILENAME
    result = bridge.watch_submit_orchestrator(
        client=_PendingClient(),  # type: ignore[arg-type]
        poster=forbidden_post,  # type: ignore[arg-type]
        output_root=root,
        source_plan_path=tmp_path / "not-read-plan.json",
        source_submission_path=tmp_path / "not-read-submission.json",
        executor_revision="a" * 40,
        publisher_sha256="b" * 64,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        watch=False,
        required_root=root,
        watch_state_file=state_path,
    )
    state = bridge._verify_seal(
        json.loads(state_path.read_text(encoding="utf-8")),
        schema=bridge.WATCH_STATE_SCHEMA,
    )

    assert result["status"] == "pending"
    assert calls == 0
    assert not root.exists()
    assert state["watcher_state"] == "running"
    assert state["stage"] == "waiting_source_terminal"
    assert state["source_task_id"] == bridge.SOURCE_TASK_ID
    assert state["source_task_state"] == "running"
    assert state["heartbeat_interval_seconds"] == 60
    assert state["scheduler_post_calls_total"] == 0
    assert state["scheduler_mutation_performed"] is False
    assert state["scientific_pass_claimed"] is False
    assert state["orchestration_root_exists"] is False
    assert state["orchestration_plan_exists"] is False


def test_watch_heartbeat_rejects_nonfixed_path(tmp_path: Path) -> None:
    with pytest.raises(bridge.BridgeError, match="campaign-fixed sibling"):
        bridge._write_watch_state(
            tmp_path / "wrong.json",
            required_root=tmp_path / "orchestration",
            watcher_state="armed",
            stage="watcher_started",
            source_task_state="unknown",
            scheduler_get_calls_total=0,
            scheduler_post_calls_total=0,
        )


def test_watch_collection_materializes_and_seals_latest_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _ready_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    submit_client = _SubmissionClient(
        plan=fixture["bridge_plan"],
        root=root,
    )
    bridge.submit_ready_plan_once(
        client=submit_client,  # type: ignore[arg-type]
        poster=submit_client.post,  # type: ignore[arg-type]
        output_root=root,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        lock_factory=submit_client.lock,
        required_root=root,
    )
    collect_client = _BridgeGetClient(
        plan=fixture["bridge_plan"],
        transport=fixture["transport"],
        bridge_task_id=97001,
    )

    first = bridge.watch_collect_materialize_once(
        client=collect_client,  # type: ignore[arg-type]
        output_root=root,
        required_root=root,
    )
    calls_after_first = collect_client.get_count
    second = bridge.watch_collect_materialize_once(
        client=collect_client,  # type: ignore[arg-type]
        output_root=root,
        required_root=root,
    )
    seal = bridge._verify_seal(
        json.loads((root / "collection_seal.json").read_text(encoding="utf-8")),
        schema=bridge.COLLECTION_SEAL_SCHEMA,
    )

    assert first["status"] == "collected_and_materialized"
    assert second["status"] == "already_materialized"
    assert collect_client.get_count == calls_after_first
    assert Path(first["thermal_event_path"]) == root / "collected" / "latest.json"
    assert seal["latest_event_path"] == str(
        (root / "collected" / "latest.json").resolve()
    )
    assert seal["final_package_gate_compatible"] is True


def test_watch_collection_writes_collected_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _ready_fixture(tmp_path, monkeypatch)
    root = fixture["root"]
    submit_client = _SubmissionClient(plan=fixture["bridge_plan"], root=root)
    bridge.submit_ready_plan_once(
        client=submit_client,  # type: ignore[arg-type]
        poster=submit_client.post,  # type: ignore[arg-type]
        output_root=root,
        authorize_post=bridge.POST_AUTHORIZATION_TOKEN,
        lock_factory=submit_client.lock,
        required_root=root,
    )
    collect_client = _BridgeGetClient(
        plan=fixture["bridge_plan"],
        transport=fixture["transport"],
        bridge_task_id=97001,
    )
    state_path = root.parent / bridge.WATCH_STATE_FILENAME

    result = bridge.watch_collect_materialize(
        client=collect_client,  # type: ignore[arg-type]
        output_root=root,
        watch=False,
        required_root=root,
        watch_state_file=state_path,
    )
    state = bridge._verify_seal(
        json.loads(state_path.read_text(encoding="utf-8")),
        schema=bridge.WATCH_STATE_SCHEMA,
    )

    assert result["status"] == "collected_and_materialized"
    assert state["watcher_state"] == "collected"
    assert state["artifact_collected"] is True
    assert state["bridge_task_id"] == 97001
    assert state["scheduler_post_calls_total"] == 1
    assert state["scheduler_mutation_performed"] is False
    assert state["scientific_pass_claimed"] is False
