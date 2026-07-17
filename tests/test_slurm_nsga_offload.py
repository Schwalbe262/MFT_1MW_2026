import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "slurm_nsga_offload.py"
SPEC = importlib.util.spec_from_file_location("slurm_nsga_offload", MODULE_PATH)
offload = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(offload)


def _write(path, value=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remote_fixture(plan, task_id, state="completed", binary_suffix=b""):
    lane_id = f"lane-{task_id}"
    output = f"runs/task-{task_id}/{lane_id}"
    report = f"report-{task_id}".encode()
    status = {
        "schema_version": offload.LANE_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "task_id": str(task_id),
        "lane_id": lane_id,
        "seed_base": task_id,
        "workers": 1,
        "state": state,
        "optimizer_output": output,
    }
    if state == "infeasible":
        status["evidence"] = {
            "infeasibility_report_sha256": hashlib.sha256(report).hexdigest()
        }
    root = plan["remote_bundle"].rstrip("/")
    files = {
        f"{root}/runs/task-{task_id}/lane_status.json": json.dumps(
            status, sort_keys=True
        ).encode()
    }
    for filename in offload.HARVEST_FILENAMES:
        if filename == "infeasibility_report.json":
            value = report
        elif filename.endswith(".npy"):
            value = b"\x93NUMPY\x00\xff\xfe" + binary_suffix + filename.encode()
        else:
            value = f"{task_id}:{filename}".encode()
        files[f"{root}/{output}/round_00/{filename}"] = value
    submission = {
        "task_id": task_id,
        "lane": {"lane_id": lane_id, "seed_base": task_id, "workers": 1},
    }
    return submission, files


def _install_fake_sftp(
    monkeypatch,
    remote_files,
    *,
    corrupt_once=None,
    corrupt_always=None,
    download_delay=0.0,
):
    corrupt_once = set(corrupt_once or ())
    corrupt_always = set(corrupt_always or ())
    state = {
        "opened_by_account": {},
        "active": 0,
        "max_active": 0,
        "downloads": {},
    }
    lock = threading.Lock()

    class FakeSession:
        def __init__(self, account, default_timeout=300):
            del default_timeout
            self.account = account
            self.entered = False

        def __enter__(self):
            with lock:
                self.entered = True
                state["opened_by_account"][self.account.name] = (
                    state["opened_by_account"].get(self.account.name, 0) + 1
                )
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            with lock:
                if self.entered:
                    state["active"] -= 1
                    self.entered = False

        def download_file(self, remote_path, local_path):
            if download_delay:
                time.sleep(download_delay)
            key = (self.account.name, remote_path)
            with lock:
                count = state["downloads"].get(key, 0) + 1
                state["downloads"][key] = count
            value = remote_files[remote_path]
            if remote_path in corrupt_always or (
                remote_path in corrupt_once and count == 1
            ):
                value = b"corrupt-sftp-frame"
            Path(local_path).write_bytes(value)

    def fake_account(accounts_path, scheduler_source, name):
        del accounts_path, scheduler_source
        return SimpleNamespace(name=name), FakeSession

    def fake_identity(session, remote_path):
        del session
        if remote_path not in remote_files:
            raise offload.RemoteArtifactMissing(remote_path)
        value = remote_files[remote_path]
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    monkeypatch.setattr(offload, "_account", fake_account)
    monkeypatch.setattr(offload, "_remote_file_identity", fake_identity)
    return state


def _fixture(tmp_path):
    deployment = tmp_path / "deployment"
    required = [
        "regression_260707/optimization/run_nsga2.py",
        "regression_260707/optimization/nsga2_problem.py",
        "regression_260707/optimization/resonance.py",
        "regression_260707/training/predictor.py",
        "regression_260707/uncertainty_contract.py",
    ]
    for relative in required:
        _write(deployment / relative, relative.encode())
    _write(deployment / "__pycache__" / "ignored.pyc", b"ignored")

    dataset = _write(tmp_path / "strict.parquet", b"dataset")
    generation = tmp_path / "registry" / "generations" / "generation-1"
    model = _write(generation / "Llt_phys" / "models.pkl", b"model")
    meta = _write(generation / "Llt_phys" / "meta.json", b"{}")
    report = {
        "training_run_id": "run-1",
        "strict_full_rows": 2201,
        "artifacts": {
            "Llt_phys/models.pkl": _sha(model),
            "Llt_phys/meta.json": _sha(meta),
        },
    }
    _write(
        generation / "train_report.json",
        (json.dumps(report, sort_keys=True) + "\n").encode(),
    )
    quality = _write(
        tmp_path / "quality.json",
        json.dumps(
            {
                "solver_revision_pin": "a" * 40,
                "library_revision_pin": "b" * 40,
            }
        ).encode(),
    )
    runner = _write(tmp_path / "runner.py", b"print('runner')\n")
    return deployment, dataset, generation, quality, runner


def test_plan_is_content_addressed_and_pins_model_inventory(tmp_path):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    first, manifest1 = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    second, manifest2 = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    assert first["bundle_id"] == second["bundle_id"]
    assert first["bundle_manifest_sha256"] == second["bundle_manifest_sha256"]
    assert manifest1 == manifest2
    assert manifest1["model_artifacts_sha256"] == {
        "Llt_phys/models.pkl": _sha(generation / "Llt_phys" / "models.pkl"),
        "Llt_phys/meta.json": _sha(generation / "Llt_phys" / "meta.json"),
    }
    assert manifest1["execution_contract"]["slurm_safe_workers_per_task"] == 1
    assert not any("__pycache__" in path for path in manifest1["files"])


def test_plan_rejects_unreported_generation_bytes(tmp_path):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    _write(generation / "unsealed.bin", b"not in report")
    with pytest.raises(RuntimeError, match="generation inventory mismatch"):
        offload.build_plan(
            deployment,
            dataset,
            generation,
            quality,
            tmp_path / "plans",
            "/gpfs/tmp_cpu2/test",
            runner=runner,
        )


def test_cpu_task_is_standard_non_gpu_and_one_seed(tmp_path):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, manifest = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    payload = offload.build_task_payload(
        plan,
        manifest,
        seed_base=52000,
        restarts=1,
        population=320,
        max_generations=600,
        workers=1,
    )
    assert payload["scheduling_profile"] == "standard"
    assert payload["gpus"] == 0
    assert payload["cpus"] == 1
    assert payload["memory_mb"] == 32768
    assert payload["payload_json"]["seed_base"] == 52000
    assert payload["payload_json"]["production_eligible"] is False
    assert "SLURM_SCHEDULER_PAYLOAD_PATH" in payload["command"]
    assert 'payload_path="$HOME/$payload_path"' in payload["command"]
    assert (
        'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")'
        in payload["command"]
    )
    assert (
        '--payload "$payload_path" --payload-root "$payload_root"' in payload["command"]
    )
    expected_payload_sha = offload.sha256_bytes(
        offload.canonical_bytes(payload["payload_json"])
    )
    assert f"--payload-sha256 {expected_payload_sha}" in payload["command"]


def test_process_pool_shape_fails_closed(tmp_path):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, manifest = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    with pytest.raises(ValueError, match="one seed/one worker"):
        offload.build_task_payload(
            plan,
            manifest,
            seed_base=52000,
            restarts=4,
            population=320,
            max_generations=600,
            workers=4,
        )


def test_seed_ranges_must_not_overlap():
    offload.validate_seed_ranges([100, 104], 4)
    with pytest.raises(ValueError, match="overlap"):
        offload.validate_seed_ranges([100, 103], 4)


def test_harvest_is_binary_safe_ordered_bounded_and_persistent(tmp_path, monkeypatch):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    task_ids = [401, 402, 403, 404]
    submissions = []
    remote_files = {}
    for task_id in task_ids:
        submission, files = _remote_fixture(
            plan, task_id, binary_suffix=bytes([task_id % 256])
        )
        submissions.append(submission)
        remote_files.update(files)
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "scheduler_url": "http://scheduler",
            "submissions": submissions,
        },
    )
    barrier = threading.Barrier(len(task_ids))

    def fake_json(url, method="GET", payload=None, timeout=30):
        del method, payload, timeout
        task_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        barrier.wait(timeout=2)
        return {
            "task_id": task_id,
            "status": "completed",
            "account_name": f"account-{task_id % 2}",
            "actual_node_name": f"n{task_id}",
            "remote_cwd": plan["remote_bundle"],
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    sftp_state = _install_fake_sftp(monkeypatch, remote_files)
    summary = offload.harvest_status(
        Path(plan["local_plan_dir"]) / "offload_plan.json",
        scheduler_url="http://scheduler",
        max_workers=12,
    )
    assert summary["complete"] is True
    assert summary["transport"] == "direct_sftp"
    assert summary["api_concurrency_limit"] == 4
    assert summary["sftp_connection_limit"] == 4
    assert summary["expected_lane_count"] == 4
    assert summary["observed_lane_count"] == 4
    assert [row["task_id"] for row in summary["lanes"]] == task_ids
    assert summary["artifact_complete_count"] == 4
    assert all(
        len(row["downloaded"]) == len(offload.HARVEST_FILENAMES)
        for row in summary["lanes"]
    )
    assert sftp_state["opened_by_account"] == {"account-0": 1, "account-1": 1}
    assert sftp_state["max_active"] <= 4
    for row in summary["lanes"]:
        assert row["artifacts_complete"] is True
        assert all(
            item["verified"] is True
            and item["remote_sha256"] == item["local_sha256"] == item["sha256"]
            for item in row["downloaded"]
        )
        expected = remote_files[
            plan["remote_bundle"].rstrip("/")
            + f"/runs/task-{row['task_id']}/lane-{row['task_id']}"
            + "/round_00/pareto_X.npy"
        ]
        actual = (
            Path(plan["local_plan_dir"])
            / "harvest"
            / f"task-{row['task_id']}"
            / "pareto_X.npy"
        ).read_bytes()
        assert actual == expected
    persisted = json.loads(
        (Path(plan["local_plan_dir"]) / "harvest" / "status.json").read_text()
    )
    assert persisted == summary


def test_harvest_one_lane_failure_does_not_drop_other_lanes(tmp_path, monkeypatch):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "submissions": [{"task_id": 501}, {"task_id": 502}],
        },
    )

    def fake_json(url, method="GET", payload=None, timeout=30):
        del method, payload, timeout
        if url.endswith("/501"):
            raise TimeoutError("isolated timeout")
        return {
            "task_id": 502,
            "status": "queued",
            "account_name": "",
            "actual_node_name": "",
            "remote_cwd": plan["remote_bundle"],
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    summary = offload.harvest_status(
        Path(plan["local_plan_dir"]) / "offload_plan.json",
        scheduler_url="http://scheduler",
        max_workers=2,
    )
    assert summary["complete"] is True
    assert len(summary["lanes"]) == 2
    assert summary["lanes"][0]["error"].startswith(
        "scheduler_query_failed:TimeoutError"
    )
    assert summary["lanes"][1]["scheduler_status"] == "queued"
    assert summary["lanes"][1]["error"] is None


def test_harvest_retries_corrupt_sftp_and_isolates_permanent_failure(
    tmp_path, monkeypatch
):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    submissions = []
    remote_files = {}
    for task_id in (601, 602):
        submission, files = _remote_fixture(plan, task_id)
        submissions.append(submission)
        remote_files.update(files)
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "submissions": submissions,
        },
    )
    root = plan["remote_bundle"].rstrip("/")
    transient = root + "/runs/task-601/lane-601/round_00/pareto_X.npy"
    permanent = root + "/runs/task-602/lane-602/round_00/pareto_X.npy"
    sftp_state = _install_fake_sftp(
        monkeypatch,
        remote_files,
        corrupt_once={transient},
        corrupt_always={permanent},
    )

    def fake_json(url, method="GET", payload=None, timeout=30):
        del method, payload, timeout
        task_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        return {
            "task_id": task_id,
            "status": "completed",
            "account_name": "shared",
            "actual_node_name": "n1",
            "remote_cwd": plan["remote_bundle"],
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    summary = offload.harvest_status(
        Path(plan["local_plan_dir"]) / "offload_plan.json",
        scheduler_url="http://scheduler",
        max_workers=4,
        retries=3,
    )
    by_task = {row["task_id"]: row for row in summary["lanes"]}
    assert by_task[601]["artifacts_complete"] is True
    transient_record = next(
        item for item in by_task[601]["downloaded"] if item["file"] == "pareto_X.npy"
    )
    assert transient_record["attempts"] == 2
    assert sftp_state["downloads"][("shared", transient)] == 2
    assert by_task[602]["artifacts_complete"] is False
    assert by_task[602]["error"] == "artifact_harvest_incomplete"
    assert any(
        warning.startswith("pareto_X.npy:RuntimeError:")
        for warning in by_task[602]["download_warnings"]
    )
    assert sftp_state["downloads"][("shared", permanent)] == 3
    # Reconnects caused by one corrupt file stay within the one account worker
    # and do not prevent subsequent files or the sibling lane from harvesting.
    assert len(by_task[602]["downloaded"]) == len(offload.HARVEST_FILENAMES) - 1


def test_harvest_cache_is_remote_reverified_and_corruption_is_redownloaded(
    tmp_path, monkeypatch
):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    submission, remote_files = _remote_fixture(plan, 701)
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "submissions": [submission],
        },
    )
    sftp_state = _install_fake_sftp(monkeypatch, remote_files)

    def fake_json(url, method="GET", payload=None, timeout=30):
        del url, method, payload, timeout
        return {
            "task_id": 701,
            "status": "completed",
            "account_name": "cache-account",
            "actual_node_name": "n2",
            "remote_cwd": plan["remote_bundle"],
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    plan_path = Path(plan["local_plan_dir"]) / "offload_plan.json"
    first = offload.harvest_status(plan_path, scheduler_url="http://scheduler")
    assert first["lanes"][0]["cache_reused"] is False
    remote_x = (
        plan["remote_bundle"].rstrip("/")
        + "/runs/task-701/lane-701/round_00/pareto_X.npy"
    )
    first_downloads = sftp_state["downloads"][("cache-account", remote_x)]
    second = offload.harvest_status(plan_path, scheduler_url="http://scheduler")
    assert second["lanes"][0]["cache_reused"] is True
    assert sftp_state["downloads"][("cache-account", remote_x)] == first_downloads
    cached_x = Path(plan["local_plan_dir"]) / "harvest" / "task-701" / "pareto_X.npy"
    cached_x.write_bytes(b"locally-corrupted")
    third = offload.harvest_status(plan_path, scheduler_url="http://scheduler")
    assert third["lanes"][0]["cache_reused"] is False
    assert third["lanes"][0]["artifacts_complete"] is True
    assert cached_x.read_bytes() == remote_files[remote_x]
    assert sftp_state["downloads"][("cache-account", remote_x)] == first_downloads + 1
    sealed_bytes = cached_x.read_bytes()
    remote_files[remote_x] = b"mutated-after-terminal-seal"
    fourth = offload.harvest_status(plan_path, scheduler_url="http://scheduler")
    assert fourth["lanes"][0]["artifacts_complete"] is False
    assert (
        "terminal_artifact_changed:pareto_X.npy"
        in fourth["lanes"][0]["integrity_errors"]
    )
    assert cached_x.read_bytes() == sealed_bytes


def test_harvest_checkpoints_each_artifact_and_caps_api_at_four(tmp_path, monkeypatch):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    submissions = []
    remote_files = {}
    for task_id in range(801, 809):
        submission, files = _remote_fixture(plan, task_id)
        submissions.append(submission)
        remote_files.update(files)
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "submissions": submissions,
        },
    )
    _install_fake_sftp(monkeypatch, remote_files, download_delay=0.001)
    api_lock = threading.Lock()
    api_state = {"active": 0, "max_active": 0}

    def fake_json(url, method="GET", payload=None, timeout=30):
        del method, payload, timeout
        task_id = int(url.rstrip("/").rsplit("/", 1)[-1])
        with api_lock:
            api_state["active"] += 1
            api_state["max_active"] = max(api_state["max_active"], api_state["active"])
        time.sleep(0.005)
        with api_lock:
            api_state["active"] -= 1
        return {
            "task_id": task_id,
            "status": "completed",
            "account_name": f"account-{task_id}",
            "actual_node_name": f"n{task_id}",
            "remote_cwd": plan["remote_bundle"],
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    original_atomic = offload.atomic_json
    checkpoints = []
    status_path = Path(plan["local_plan_dir"]) / "harvest" / "status.json"

    def recording_atomic(path, payload):
        if Path(path) == status_path:
            checkpoints.append(json.loads(json.dumps(payload)))
        original_atomic(path, payload)

    monkeypatch.setattr(offload, "atomic_json", recording_atomic)
    summary = offload.harvest_status(
        Path(plan["local_plan_dir"]) / "offload_plan.json",
        scheduler_url="http://scheduler",
        max_workers=32,
    )
    assert api_state["max_active"] == 4
    assert summary["api_concurrency_limit"] == 4
    assert summary["sftp_connection_limit"] == 4
    assert summary["checkpoint_sequence"] == len(checkpoints)
    assert len(checkpoints) > len(submissions) * len(offload.HARVEST_FILENAMES)
    assert any(
        not snapshot["complete"]
        and any(
            (lane.get("artifact_progress") or {}).get("checked") == 1
            for lane in snapshot["lanes"]
        )
        for snapshot in checkpoints
    )


def test_harvest_rejects_task_remote_cwd_before_opening_ssh(tmp_path, monkeypatch):
    deployment, dataset, generation, quality, runner = _fixture(tmp_path)
    plan, _ = offload.build_plan(
        deployment,
        dataset,
        generation,
        quality,
        tmp_path / "plans",
        "/gpfs/tmp_cpu2/test",
        runner=runner,
    )
    offload.atomic_json(
        Path(plan["local_plan_dir"]) / "submissions.json",
        {
            "schema_version": offload.PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "submissions": [{"task_id": 901}],
        },
    )

    def fake_json(url, method="GET", payload=None, timeout=30):
        del url, method, payload, timeout
        return {
            "task_id": 901,
            "status": "completed",
            "account_name": "account",
            "actual_node_name": "n1",
            "remote_cwd": "/tmp/wrong-bundle",
        }

    monkeypatch.setattr(offload, "_api_json", fake_json)
    monkeypatch.setattr(
        offload,
        "_account",
        lambda *args: pytest.fail("SSH must not open for a mismatched remote_cwd"),
    )
    summary = offload.harvest_status(
        Path(plan["local_plan_dir"]) / "offload_plan.json",
        scheduler_url="http://scheduler",
    )
    assert summary["lanes"][0]["error"] == "remote_cwd_fingerprint_mismatch"
