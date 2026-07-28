import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from regression_260707.monitoring.app import create_app
from regression_260707.monitoring.pipeline_status import (
    ContinuousPipelineReader,
    JOB_STATES,
)


SOLVER_REVISION = "a" * 40
LIBRARY_REVISION = "b" * 40


def _write_role(root: Path, role: str, now: float, *, pid: int | None = None):
    path = root / "locks" / f"{role}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "role": role,
        "command": "control" if role == "controller" else "supervise",
        "pid": os.getpid() if pid is None else pid,
        "hostname": "test-host",
        "acquired_at": datetime.fromtimestamp(
            now, timezone.utc
        ).isoformat(),
    }
    if role == "controller":
        payload.update(
            solver_revision=SOLVER_REVISION,
            library_revision=LIBRARY_REVISION,
            verification_config_sha256="c" * 64,
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_queue(root: Path, now: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    database = root / "jobs.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE queue_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO queue_meta(key, value) VALUES('schema_version', '2');
        CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            job_type TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            input_generation TEXT,
            state TEXT NOT NULL,
            owner_lease TEXT,
            heartbeat_at REAL,
            lease_until REAL,
            attempt INTEGER NOT NULL,
            max_attempts INTEGER NOT NULL,
            next_retry_at REAL NOT NULL,
            terminal_reason TEXT,
            output_generation TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )
    rows = [
        (1, "collect", "collect-1", None, "running", "collector-1", now - 5,
         now + 100, 1, 5, now, None, None, now - 120, now - 5),
        (2, "train", "train-1", "dataset:g1", "succeeded", None, None,
         None, 1, 3, now, None, "models:g2", now - 400, now - 100),
        (3, "tune", "tune-1", "dataset:g2", "queued", None, None,
         None, 0, 3, now, None, None, now - 30, now - 30),
        (4, "optimize", "optimize-1", "models:g2", "running", "optimizer-1",
         now - 4, now + 100, 1, 3, now, None, None, now - 90, now - 4),
        (5, "verify_standard", "standard-1", "pareto:g3", "failed", None,
         None, None, 3, 3, now, "solver exploded", None, now - 500, now - 20),
        (6, "verify_fine", "fine-1", "verification:g4", "retry_wait", None,
         None, None, 2, 3, now + 60, "command_exit:1", None, now - 200, now - 10),
    ]
    connection.executemany(
        """
        INSERT INTO jobs(
            id, job_type, idempotency_key, input_generation, state,
            owner_lease, heartbeat_at, lease_until, attempt, max_attempts,
            next_retry_at, terminal_reason, output_generation, created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()
    connection.close()
    for job_id, attempt in ((1, 1), (2, 1), (4, 1), (5, 3), (6, 2)):
        log = root / "work" / f"job-{job_id:08d}" / f"attempt-{attempt:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("attempt log\n", encoding="utf-8")
    generation_id = "d" * 64
    generation = root / "artifacts" / "dataset" / generation_id
    generation.mkdir(parents=True)
    artifact = generation / "train.parquet"
    artifact.write_bytes(b"immutable parquet fixture")
    manifest = {
        "schema_version": 1,
        "kind": "dataset",
        "generation_id": generation_id,
        "created_at": datetime.fromtimestamp(now - 20, timezone.utc).isoformat(),
        "artifacts": {"train.parquet": {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size": artifact.stat().st_size,
        }},
        "metadata": {
            "strict_full_rows": 123,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
        },
    }
    manifest_path = generation / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (generation / "COMPLETED").write_text(json.dumps({
        "generation_id": generation_id,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }), encoding="utf-8")
    (root / "surrogate_status.json").write_text(json.dumps({
        "schema_version": 1,
        "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "pid": os.getpid(),
        "state": "waiting_for_next_dataset_check",
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "dataset_generation": f"dataset:{generation_id}",
        "raw_rows": 321,
        "strict_full_rows": 123,
    }), encoding="utf-8")
    return database


def _verified_contract() -> dict:
    return {
        "available": True,
        "verified": True,
        "quality_contract_path": "C:/fixture/quality_contract.py",
        "quality_contract_sha256": "1" * 64,
        "profile_path": "C:/fixture/standard.json",
        "profile_sha256": "2" * 64,
        "status": {},
        "error": None,
    }


def _audit_123(path, solver_revision, library_revision):
    return {
        "raw_rows": 321,
        "strict_em_rows": 150,
        "strict_full_rows": 123,
        "em_only_rows": 27,
    }


def test_pipeline_reader_reports_real_parallel_lanes_revisions_and_errors(tmp_path):
    now = time.time()
    root = tmp_path / "pipeline"
    database = _create_queue(root, now)
    _write_role(root, "controller", now)
    _write_role(root, "supervisor", now)
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "controller.stdout.log").write_text(
        "controller tick\n", encoding="utf-16"
    )
    (log_root / "controller.stderr.log").write_text("", encoding="utf-8")
    (log_root / "supervisor.stdout.log").write_text("", encoding="utf-8")
    (log_root / "supervisor.stderr.log").write_text("", encoding="utf-8")

    before_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    before_mtime = database.stat().st_mtime_ns
    payload = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_auditor=_audit_123,
        contract_provenance_provider=lambda revisions: _verified_contract(),
    ).snapshot()

    assert payload["health"] == "degraded"  # fine FEA is retrying.
    assert payload["roles"]["controller"]["status"] == "alive"
    assert payload["roles"]["supervisor"]["status"] == "alive"
    assert payload["roles"]["controller"]["logs"]["stdout"]["tail"] == [
        "controller tick"
    ]
    assert payload["revisions"] == {
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "solver_revision_exact": True,
        "library_revision_exact": True,
        "verification_config_sha256": "c" * 64,
        "exact": True,
    }
    assert payload["queue"]["available"] is True
    assert payload["queue"]["total_jobs"] == 6
    assert payload["queue"]["counts"] == {
        "queued": 1,
        "retry_wait": 1,
        "running": 2,
        "succeeded": 1,
        "failed": 1,
        "cancelled": 0,
    }
    assert payload["parallel"] == {
        "running_lane_count": 2,
        "running_lanes": ["collect", "optimize"],
        "active_lane_count": 4,
        "active_lanes": ["collect", "tune", "optimize", "verify_fine"],
        "durable_running_lane_count": 2,
        "durable_running_lanes": ["collect", "optimize"],
        "durable_active_lane_count": 4,
        "durable_active_lanes": [
            "collect", "tune", "optimize", "verify_fine"
        ],
        "external_running_lane_count": 0,
        "external_running_lanes": [],
        "parallel_work_confirmed": True,
    }
    lanes = {lane["job_type"]: lane for lane in payload["lanes"]}
    collect = lanes["collect"]["current_job"]
    assert collect["heartbeat_stale"] is False
    assert collect["started_at"]
    assert collect["heartbeat_at"]
    assert collect["elapsed_seconds"] >= 0
    assert lanes["train"]["current_job"]["input_generation"] == "dataset:g1"
    assert lanes["train"]["current_job"]["output_generation"] == "models:g2"
    assert payload["training"]["active_job"] is None
    assert lanes["verify_standard"]["last_error"]["reason"] == "solver exploded"
    assert lanes["verify_fine"]["health"] == "retrying"
    assert payload["cohort"]["current_strict_full_rows"] == 123
    assert lanes["train"]["prerequisite"]["threshold"] == 500
    assert lanes["tune"]["prerequisite"]["threshold"] == 4000
    assert lanes["optimize"]["prerequisite"]["threshold"] == 3000
    assert "NSGA-II output" in lanes["verify_standard"]["prerequisite"]["reason"]
    assert lanes["verify_fine"]["prerequisite"]["gate"] == "standard_fea_dependency"
    assert any("retrying" in warning for warning in payload["warnings"])
    # query_only plus mode=ro must leave the durable queue byte-for-byte alone.
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before_hash
    assert database.stat().st_mtime_ns == before_mtime


def test_pipeline_cohort_uses_fresh_authority_over_stale_controller_log(
    tmp_path,
):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    _write_role(root, "supervisor", now)
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    old_id = "d" * 64
    (log_root / "controller.stdout.log").write_text(
        json.dumps({
            "dataset_generation": f"dataset:{old_id}",
            "jobs": {"collect": 1},
            "blocked": {},
        }) + "\n",
        encoding="utf-8",
    )
    (log_root / "controller.stderr.log").write_text("", encoding="utf-8")
    (log_root / "supervisor.stdout.log").write_text("", encoding="utf-8")
    (log_root / "supervisor.stderr.log").write_text("", encoding="utf-8")

    old_path = root / "artifacts" / "dataset" / old_id
    os.utime(old_path, (now - 30, now - 30))
    new_id = "e" * 64
    new_path = root / "artifacts" / "dataset" / new_id
    new_path.mkdir(parents=True)
    artifact = new_path / "train.parquet"
    artifact.write_bytes(b"new immutable parquet fixture")
    manifest = {
        "schema_version": 1,
        "kind": "dataset",
        "generation_id": new_id,
        "created_at": datetime.fromtimestamp(now - 5, timezone.utc).isoformat(),
        "artifacts": {"train.parquet": {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size": artifact.stat().st_size,
        }},
        "metadata": {
            "strict_full_rows": 456,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
        },
    }
    manifest_path = new_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (new_path / "COMPLETED").write_text(json.dumps({
        "generation_id": new_id,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }), encoding="utf-8")
    (root / "surrogate_status.json").write_text(json.dumps({
        "schema_version": 1,
        "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "pid": os.getpid(),
        "state": "waiting_for_next_dataset_check",
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
        "dataset_generation": f"dataset:{new_id}",
        "raw_rows": 500,
        "strict_full_rows": 456,
    }), encoding="utf-8")
    os.utime(new_path, (now - 5, now - 5))

    payload = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_auditor=lambda path, solver, library: {
            "raw_rows": 500,
            "strict_em_rows": 480,
            "strict_full_rows": 456,
            "em_only_rows": 24,
        },
        contract_provenance_provider=lambda revisions: _verified_contract(),
    ).snapshot()

    assert payload["controller_cycle"]["dataset_generation"] == f"dataset:{old_id}"
    assert payload["cohort"]["generation"] == f"dataset:{new_id}"
    assert payload["cohort"]["strict_full_rows"] == 456
    assert payload["cohort"]["controller_generation"] == f"dataset:{new_id}"
    assert payload["cohort"]["is_controller_generation"] is True
    assert payload["cohort"]["authority_verified"] is True


def test_external_tuner_counts_only_after_recent_cpu_or_io_activity(
    tmp_path, monkeypatch
):
    now = [time.time()]
    root = tmp_path / "pipeline"
    _create_queue(root, now[0])
    _write_role(root, "controller", now[0])
    _write_role(root, "supervisor", now[0])
    dataset = tmp_path / "strict.parquet"
    dataset.write_bytes(b"immutable dataset identity")
    counters = {"cpu": 10.0, "read": 1_000.0, "write": 50.0}

    class AccessDenied(Exception):
        pass

    class NoSuchProcess(Exception):
        pass

    class FakeProcess:
        info = {
            "pid": 42,
            "name": "python.exe",
            "create_time": now[0] - 100,
        }

        @staticmethod
        def cmdline():
            return [
                "python.exe",
                "training/tune_optuna.py",
                "--all",
                "--trials",
                "200",
                "--dataset",
                str(dataset),
            ]

        @staticmethod
        def cpu_times():
            return SimpleNamespace(user=counters["cpu"], system=0.0)

        @staticmethod
        def io_counters():
            return SimpleNamespace(
                read_bytes=counters["read"],
                write_bytes=counters["write"],
            )

    fake_psutil = SimpleNamespace(
        AccessDenied=AccessDenied,
        NoSuchProcess=NoSuchProcess,
        process_iter=lambda attrs: [FakeProcess()],
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    reader = ContinuousPipelineReader(root, clock=lambda: now[0])

    first = reader.snapshot()
    assert first["external_tuners"]["validated_running_count"] == 0
    assert first["external_tuners"]["processes"][0]["command_validated"] is True
    assert first["parallel"]["running_lane_count"] == 2
    assert first["parallel"]["external_running_lanes"] == []

    now[0] += 20
    counters["cpu"] += 4.0
    counters["read"] += 2_048
    second = reader.snapshot()
    process = second["external_tuners"]["processes"][0]
    assert process["activity_confirmed"] is True
    assert process["cpu_seconds_delta"] == 4.0
    assert process["read_bytes_delta"] == 2_048
    assert process["validated_running"] is True
    assert second["parallel"]["running_lane_count"] == 3
    assert second["parallel"]["running_lanes"] == [
        "collect", "optimize", "external_tune"
    ]
    assert second["parallel"]["durable_running_lane_count"] == 2
    assert second["parallel"]["external_running_lane_count"] == 1


def test_pipeline_reader_audits_row_tiers_once_per_dataset_fingerprint(tmp_path):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    _write_role(root, "supervisor", now)
    artifact = (
        root / "artifacts" / "dataset" / ("d" * 64) / "train.parquet"
    )
    artifact.write_bytes(b"immutable parquet fixture")
    calls = []

    def audit(path, solver_revision, library_revision):
        calls.append((path, solver_revision, library_revision))
        return {
            "raw_rows": 321,
            "strict_em_rows": 150,
            "strict_full_rows": 123,
            "em_only_rows": 27,
        }

    reader = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_auditor=audit,
        contract_provenance_provider=lambda revisions: _verified_contract(),
    )
    first = reader.snapshot()
    second = reader.snapshot()

    assert len(calls) == 1
    assert calls[0] == (artifact, SOLVER_REVISION, LIBRARY_REVISION)
    assert first["cohort"]["counts_available"] is True
    assert first["cohort"]["raw_rows"] == 321
    assert first["cohort"]["strict_em_rows"] == 150
    assert first["cohort"]["strict_full_rows"] == 123
    assert first["cohort"]["em_only_rows"] == 27
    assert first["cohort"]["current_strict_full_rows"] == 123
    assert first["cohort"]["counts_source"] == (
        "authenticated_train.parquet_quality_contract"
    )
    assert first["cohort"]["manifest_matches_audit"] is True
    assert first["cohort"]["em_only_is_invalid"] is False
    assert "not invalid" in first["cohort"]["row_semantics"]["em_only_rows"]
    assert second["cohort"] == first["cohort"]


def test_pipeline_reader_bounds_dataset_audit_and_fails_closed(tmp_path):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    artifact = (
        root / "artifacts" / "dataset" / ("d" * 64) / "train.parquet"
    )
    artifact.write_bytes(b"too large for configured test bound")
    called = False

    def audit(path, solver_revision, library_revision):
        nonlocal called
        called = True
        raise AssertionError("bounded artifact must not be opened")

    payload = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_audit_max_bytes=8,
        dataset_auditor=audit,
        contract_provenance_provider=lambda revisions: _verified_contract(),
    ).snapshot()

    assert called is False
    assert payload["cohort"]["available"] is False
    assert payload["cohort"]["counts_available"] is False
    assert payload["cohort"]["strict_em_rows"] is None
    assert payload["cohort"]["strict_full_rows"] == 0
    assert payload["cohort"]["em_only_rows"] is None
    assert "exceeds" in payload["cohort"]["counts_error"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("stale", "stale"),
        ("pid", "PID"),
        ("revision", "revision mismatch"),
    ),
)
def test_pipeline_controller_authority_fails_closed(
    tmp_path, mutation, expected_error
):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    status_path = root / "surrogate_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if mutation == "stale":
        status["updated_at"] = datetime.fromtimestamp(
            now - 181, timezone.utc
        ).isoformat()
    elif mutation == "pid":
        status["pid"] = os.getpid() + 10_000
    else:
        status["solver_revision"] = "f" * 40
    status_path.write_text(json.dumps(status), encoding="utf-8")

    payload = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_auditor=_audit_123,
        contract_provenance_provider=lambda revisions: _verified_contract(),
    ).snapshot()

    assert payload["controller_authority"]["verified"] is False
    assert expected_error in payload["controller_authority"]["error"]
    assert payload["cohort"]["available"] is False
    assert payload["cohort"]["strict_full_rows"] == 0


@pytest.mark.parametrize("target", ("parquet", "completed"))
def test_pipeline_dataset_hash_chain_tamper_fails_closed(tmp_path, target):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    generation = root / "artifacts" / "dataset" / ("d" * 64)
    if target == "parquet":
        (generation / "train.parquet").write_bytes(b"tampered")
    else:
        completed = json.loads(
            (generation / "COMPLETED").read_text(encoding="utf-8")
        )
        completed["manifest_sha256"] = "0" * 64
        (generation / "COMPLETED").write_text(
            json.dumps(completed), encoding="utf-8"
        )

    payload = ContinuousPipelineReader(
        root,
        clock=lambda: now,
        inspect_external_processes=False,
        dataset_auditor=_audit_123,
        contract_provenance_provider=lambda revisions: _verified_contract(),
    ).snapshot()

    assert payload["cohort"]["available"] is False
    assert "mismatch" in payload["cohort"]["error"]


def test_contract_provenance_uses_checkpoint_deployment_not_shadow_import(
    tmp_path, monkeypatch
):
    root = tmp_path / "pipeline"
    deployment = tmp_path / "deployment" / "regression_260707"
    checkpoint_root = deployment / "training" / "checkpoint_runs" / "run"
    checkpoint_root.mkdir(parents=True)
    contract = deployment / "quality_contract.py"
    contract.write_text("DEPLOYMENT_SENTINEL = True\n", encoding="utf-8")
    profile = deployment / "verify" / "profiles" / "standard.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"param_overrides": {}}), encoding="utf-8")
    contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
    canonical_profile_sha = hashlib.sha256(json.dumps(
        {"param_overrides": {}}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    status = {
        "expected_solver_revision": SOLVER_REVISION,
        "expected_library_revision": LIBRARY_REVISION,
        "checkpoint_run_root": str(checkpoint_root),
        "state_identity": {
            "library_revision": LIBRARY_REVISION,
            "quality_contract_sha256": contract_sha,
            "profile_sha256": canonical_profile_sha,
            "profile_path": str(profile),
        },
    }
    checkpoint = root / "canonical_checkpoint" / "strict_data_status.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "quality_contract", object())

    provenance = ContinuousPipelineReader(
        root, inspect_external_processes=False
    )._contract_provenance({
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
    })

    assert provenance["verified"] is True
    assert Path(provenance["quality_contract_path"]) == contract.resolve()
    assert provenance["quality_contract_sha256"] == contract_sha


def test_active_model_awaiting_is_not_active_and_active_chain_is_authenticated(
    tmp_path,
):
    now = time.time()
    root = tmp_path / "pipeline"
    registry = root / "canonical_checkpoint" / "registry"
    registry.mkdir(parents=True)
    active_path = root / "active_surrogate.json"
    common = {
        "schema_version": 1,
        "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "registry": str(registry),
        "strict_full_rows": 5_670,
        "activation_minimum_strict_full_rows": 3_000,
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
    }
    active_path.write_text(json.dumps({
        **common, "state": "awaiting_activation"
    }), encoding="utf-8")
    reader = ContinuousPipelineReader(root, clock=lambda: now)
    revisions = {
        "solver_revision": SOLVER_REVISION,
        "library_revision": LIBRARY_REVISION,
    }
    awaiting = reader._active_model_status(revisions, now)
    assert awaiting["verified"] is True
    assert awaiting["production_active"] is False

    generation = registry / "generations" / "run-1"
    generation.mkdir(parents=True)
    report = {
        "training_run_id": "run-1",
        "dataset_sha256": "3" * 64,
        "profile_sha256": "4" * 64,
    }
    report_path = generation / "train_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    gate = {
        "passed": True,
        "training_run_id": "run-1",
        "dataset_sha256": "3" * 64,
        "profile_sha256": "4" * 64,
    }
    gate_path = generation / "quality_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    pointer = {
        "schema_version": 2,
        "training_run_id": "run-1",
        "generation": "generations/run-1",
        "dataset_sha256": "3" * 64,
        "profile_sha256": "4" * 64,
        "strict_full_rows": 5_670,
        "generation_report_sha256": report_sha,
        "quality_gate_sha256": gate_sha,
    }
    (registry / "current.json").write_text(
        json.dumps(pointer), encoding="utf-8"
    )
    active_path.write_text(json.dumps({
        **common, "state": "active"
    }), encoding="utf-8")
    active = reader._active_model_status(revisions, now)
    assert active["verified"] is True
    assert active["production_active"] is True

    report_path.write_text("{}", encoding="utf-8")
    tampered = reader._active_model_status(revisions, now)
    assert tampered["verified"] is False
    assert tampered["production_active"] is False


def test_pipeline_reader_is_bounded_and_fail_soft_for_missing_or_corrupt_state(tmp_path):
    now = time.time()
    missing = ContinuousPipelineReader(
        tmp_path / "missing", clock=lambda: now, inspect_external_processes=False
    ).snapshot()
    assert missing["available"] is False
    assert missing["health"] == "offline"
    assert missing["queue"]["available"] is False
    assert len(missing["lanes"]) == 6

    root = tmp_path / "corrupt"
    root.mkdir()
    (root / "jobs.sqlite3").write_bytes(b"not sqlite")
    _write_role(root, "controller", now, pid=2_147_000_000)
    payload = ContinuousPipelineReader(
        root, clock=lambda: now, inspect_external_processes=False
    ).snapshot()
    assert payload["available"] is True  # role metadata remains observable.
    assert payload["roles"]["controller"]["status"] in {"stale", "unknown"}
    assert payload["queue"]["available"] is False
    assert "DatabaseError" in payload["queue"]["error"]


def test_experimental_hpo_projection_exposes_bounded_live_progress(
        tmp_path, monkeypatch):
    now = time.time()
    root = tmp_path / "pipeline"
    status_path = root / "experimental_shadow_d7_continuous" / "status.json"
    status_path.parent.mkdir(parents=True)
    dataset = tmp_path / "train.parquet"
    dataset.write_bytes(b"parquet identity placeholder")
    status_path.write_text(json.dumps({
        "state": "wave_running",
        "wave_phase": "experimental_hpo",
        "active_wave": "wave-test",
        "pid": os.getpid(),
        "updated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "canonical_dataset": str(dataset),
        "observed_strict_full_rows": 5098,
        "selected_hpo_target_count": 8,
        "completed_hpo_target_count": 4,
        "active_hpo_processes": 4,
        "active_hpo_batch": 2,
        "hpo_batch_count": 2,
    }), encoding="utf-8")
    monkeypatch.setattr(
        "regression_260707.monitoring.pipeline_status._inspect_process",
        lambda pid: (True, now - 30.0),
    )

    payload = ContinuousPipelineReader(
        root, clock=lambda: now, inspect_external_processes=False
    )._experimental_shadow_training(now)

    assert payload["validated_running"] is True
    assert payload["observed_strict_full_rows"] == 5098
    assert payload["selected_hpo_target_count"] == 8
    assert payload["completed_hpo_target_count"] == 4
    assert payload["active_hpo_processes"] == 4
    assert payload["active_hpo_batch"] == payload["hpo_batch_count"] == 2


def test_pipeline_log_tail_is_bounded_and_current_role_error_is_visible(tmp_path):
    now = time.time()
    root = tmp_path / "pipeline"
    _create_queue(root, now)
    _write_role(root, "controller", now)
    _write_role(root, "supervisor", now)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stderr = logs / "controller.stderr.log"
    stderr.write_text(
        "UNIQUE-EARLY-MARKER\n" + ("padding line\n" * 10_000)
        + "fatal controller error\n",
        encoding="utf-8",
    )

    payload = ContinuousPipelineReader(
        root, clock=lambda: now, inspect_external_processes=False
    ).snapshot()
    log = payload["roles"]["controller"]["logs"]["stderr"]
    assert len(log["tail"]) <= 12
    assert "UNIQUE-EARLY-MARKER" not in log["tail"]
    assert log["tail"][-1] == "fatal controller error"
    assert payload["roles"]["controller"]["last_error"] == "fatal controller error"


def test_pipeline_api_and_static_panel_are_exposed(tmp_path):
    payload = {
        "schema_version": 1,
        "available": True,
        "health": "healthy",
        "roles": {},
        "revisions": {},
        "queue": {"available": True, "counts": {}},
        "lanes": [],
        "parallel": {"running_lane_count": 0},
        "warnings": [],
    }

    class StubReader:
        def snapshot(self):
            return payload

    class StubService:
        continuous_pipeline = StubReader()

    client = TestClient(create_app(regression_root=tmp_path, service=StubService()))
    assert client.get("/api/pipeline").json() == payload
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="continuous-pipeline-title"' in page.text
    assert 'id="continuous-pipeline-lanes"' in page.text
    assert 'id="pipeline-running-lanes"' in page.text
    assert 'id="pipeline-em-rows"' in page.text
    assert 'id="pipeline-em-only-rows"' in page.text
    assert 'id="pipeline-cohort-explanation"' in page.text
    script = client.get("/static/app.js").text
    assert "function renderContinuousPipeline" in script
    assert "parallel.parallel_work_confirmed" in script
    assert "job?.heartbeat_stale" in script
    assert "lane.prerequisite" in script
    assert "cohort.strict_em_rows" in script
    assert "cohort.em_only_rows" in script
    assert "무효 데이터가 아닙니다" in script
    assert "tier1_feedback_search" in script
    assert "#nsga-tier1-progress" in script
    stylesheet = client.get("/static/app.css").text
    assert ".continuous-pipeline-panel" in stylesheet
    assert ".continuous-lane-table" in stylesheet
    assert ".pipeline-heartbeat.stale" in stylesheet
    assert ".pipeline-cohort-explanation" in stylesheet


def test_job_state_fixture_covers_monitor_schema():
    assert set(JOB_STATES) == {
        "queued", "retry_wait", "running", "succeeded", "failed", "cancelled"
    }
