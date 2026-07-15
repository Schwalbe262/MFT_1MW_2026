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
    (generation / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "kind": "dataset",
        "generation_id": generation_id,
        "created_at": datetime.fromtimestamp(now - 20, timezone.utc).isoformat(),
        "metadata": {
            "strict_full_rows": 123,
            "solver_revision": SOLVER_REVISION,
            "library_revision": LIBRARY_REVISION,
        },
    }), encoding="utf-8")
    return database


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
        root, clock=lambda: now, inspect_external_processes=False
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
        "train.parquet_quality_contract"
    )
    assert first["cohort"]["manifest_matches_audit"] is True
    assert first["cohort"]["em_only_is_invalid"] is False
    assert "not invalid" in first["cohort"]["row_semantics"]["em_only_rows"]
    assert second["cohort"] == first["cohort"]


def test_pipeline_reader_bounds_dataset_audit_and_keeps_manifest_full_count(tmp_path):
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
    ).snapshot()

    assert called is False
    assert payload["cohort"]["available"] is True
    assert payload["cohort"]["counts_available"] is False
    assert payload["cohort"]["strict_em_rows"] is None
    assert payload["cohort"]["strict_full_rows"] == 123
    assert payload["cohort"]["em_only_rows"] is None
    assert "audit limit" in payload["cohort"]["counts_error"]


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
    stylesheet = client.get("/static/app.css").text
    assert ".continuous-pipeline-panel" in stylesheet
    assert ".continuous-lane-table" in stylesheet
    assert ".pipeline-heartbeat.stale" in stylesheet
    assert ".pipeline-cohort-explanation" in stylesheet


def test_job_state_fixture_covers_monitor_schema():
    assert set(JOB_STATES) == {
        "queued", "retry_wait", "running", "succeeded", "failed", "cancelled"
    }
