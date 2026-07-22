from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse

import pytest

from tools import tier1_final1000_resource2_canary as canary


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "evidence"
    / "tier1_final1000_resource2_canary_v4_20260723.json"
)


def _config() -> dict[str, Any]:
    return canary.validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _baseline_terminal(config: dict[str, Any]) -> dict[str, Any]:
    metrics_unsigned = {
        "schema_version": canary.shape1.CPU_EVIDENCE_SCHEMA,
        "expected_shape": 1,
        "parent_requested_cpus": 4,
        "logical_seed_count": 1,
        "parent_elapsed_seconds": 101.0,
        "total_child_cpu_seconds": 135.0,
        "parent_average_cores_used": 135.0 / 101.0,
        "reserved_cpu_utilization_fraction": 135.0 / 404.0,
        "sum_logical_seed_elapsed_seconds": 100.0,
        "reported_child_cpu_seconds": 135.0,
        "weighted_average_cores_used_per_logical_seed": 1.35,
        "logical_seed_cpu_utilization_fraction": 1.35 / 4.0,
        "per_seed_average_cores": [
            {
                "ordinal": 0,
                "seed": config["baseline_seed"],
                "cpu_set": [8, 9, 10, 11],
                "wall_time_seconds": 100.0,
                "process_tree_cpu_seconds": 135.0,
                "average_cores_used": 1.35,
            }
        ],
        "baseline_weighted_average_cores_used_per_logical_seed": None,
        "per_seed_core_use_ratio_vs_baseline": None,
        "gate_results": {"all_required_baseline_gates": True},
        "low_cpu_utilization_detected": False,
        "low_cpu_diagnosis": None,
        "promotion_eligible": True,
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    metrics = {
        **metrics_unsigned,
        "terminal_cpu_evidence_sha256": canary.canonical_sha256(metrics_unsigned),
    }
    result_raw = b'{"result":"sealed"}\n'
    unsigned = {
        "schema_version": canary.shape1.REMOTE_TERMINAL_SCHEMA,
        "observed_at": "2026-07-23T05:00:00+09:00",
        "package_sha256": config["baseline_package_sha256"],
        "submission_receipt_sha256": config["baseline_submission_receipt_sha256"],
        "task_id": config["baseline_task_id"],
        "scheduler_status": "completed",
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": 1,
        "remote_files": {
            "result": {
                "relative_path": f"runs/task-{config['baseline_task_id']}/seed-{config['baseline_seed']}/result.json",
                "size": len(result_raw),
                "sha256": canary.hashlib.sha256(result_raw).hexdigest(),
            }
        },
        "batch_manifest_sha256": "1" * 64,
        "task_status_sha256": "2" * 64,
        "child_receipt_sha256": "3" * 64,
        "terminal_cpu_evidence": metrics,
        "terminal_cpu_evidence_sha256": metrics["terminal_cpu_evidence_sha256"],
        "promotion_eligible": True,
        "shape4_submission_allowed": True,
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_access": "read-only",
        "remote_write_count": 0,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {
        **unsigned,
        "remote_terminal_sha256": canary.canonical_sha256(unsigned),
    }


def _baseline_summary(config: dict[str, Any]) -> dict[str, Any]:
    return canary._baseline_summary(_baseline_terminal(config))


def _source_task(config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "seed": config["seed"],
        "inference_threads": 8,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    return {
        "name": f"mft-t1fg-entry-refill-{config['seed']}",
        "remote_cwd": "/gpfs/tmp_cpu2/mft_tier1_current7_bundles/current7-test",
        "command": "python seed_runner.py",
        "payload_json": payload,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": 4,
        "memory_mb": 28672,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": 1,
        "timeout_seconds": 86400,
        "dedupe_key": "mft-tier1-final1000:" + "a" * 64,
        "max_workers_per_node": 32,
    }


def _telemetry(package: dict[str, Any], task_id: int = 90001) -> dict[str, Any]:
    selected_before = {
        "depth_from_leaf": 1,
        "relative_path": "slurm/job",
        "memory_max_raw": str(canary.CANDIDATE_MEMORY_MB * 1024**2),
        "memory_limit_bytes": canary.CANDIDATE_MEMORY_MB * 1024**2,
        "memory_current_bytes": 1024,
        "memory_peak_bytes": 2048,
    }
    selected_after = {**selected_before, "memory_peak_bytes": 4096}
    leaf = {
        "depth_from_leaf": 0,
        "relative_path": "slurm/job/task",
        "memory_max_raw": "max",
        "memory_limit_bytes": None,
        "memory_current_bytes": 512,
        "memory_peak_bytes": 1024,
    }
    cgroup_text = "0::/slurm/job/task\n"
    mountinfo_text = (
        "30 29 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime "
        "- cgroup2 cgroup rw\n"
    )

    def diagnostic(path: str, text: str) -> dict[str, Any]:
        raw = text.encode("utf-8")
        return {
            "path": path,
            "maximum_bytes": canary.cgroup_memory.CGROUP_DIAGNOSTIC_MAX_BYTES,
            "bytes_captured": len(raw),
            "truncated": False,
            "utf8_valid": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": text,
            "error": None,
        }

    diagnostics = {
        "schema_version": canary.cgroup_memory.CGROUP_DIAGNOSTIC_SCHEMA,
        "maximum_bytes_per_file": (
            canary.cgroup_memory.CGROUP_DIAGNOSTIC_MAX_BYTES
        ),
        "sysfs_root": "/sys/fs/cgroup",
        "proc_self_cgroup": diagnostic("/proc/self/cgroup", cgroup_text),
        "proc_self_mountinfo": diagnostic(
            "/proc/self/mountinfo", mountinfo_text
        ),
    }

    def snapshot(selected: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": canary.cgroup_memory.CGROUP_SNAPSHOT_SCHEMA,
            "cgroup_version": "v2",
            "hierarchy_id": 0,
            "membership_controllers": [],
            "membership_path": "/slurm/job/task",
            "mount_root": "/",
            "mount_relative_path": ".",
            "leaf_relative_path": "slurm/job/task",
            "nearest_accounting_depth": 0,
            "nearest_accounting_limit_unbounded": True,
            "limit_filename": "memory.max",
            "current_filename": "memory.current",
            "peak_filename": "memory.peak",
            "ancestors": [leaf, selected],
            "selected_finite_ancestor": selected,
        }

    unsigned = {
        "schema_version": canary.TELEMETRY_SCHEMA,
        "started_at": "2026-07-23T05:00:00+09:00",
        "finished_at": "2026-07-23T05:01:40+09:00",
        "task_id": task_id,
        "seed": package["seed"],
        "payload_sha256": canary.canonical_sha256(
            package["candidate_task"]["payload_json"]
        ),
        "baseline_remote_terminal_sha256": package["baseline"][
            "remote_terminal_sha256"
        ],
        "requested_cpus": 2,
        "requested_memory_bytes": canary.CANDIDATE_MEMORY_MB * 1024**2,
        "rss_gate_bytes": canary.DEFAULT_PEAK_RSS_GATE_BYTES,
        "slurm_cpus_per_task": "2",
        "thread_environment": canary._thread_environment(),
        "cpu_set_before": [4, 5],
        "cpu_set_after": [4, 5],
        "wall_time_seconds": 100.0,
        "process_tree_cpu_seconds": 135.0,
        "cpu_capacity_seconds": 200.0,
        "average_cores_used": 1.35,
        "cpu_utilization_fraction": 0.675,
        "peak_rss_bytes": 2 * 1024**3,
        "cgroup_diagnostics": {
            "before": copy.deepcopy(diagnostics),
            "after": copy.deepcopy(diagnostics),
        },
        "cgroup_before": snapshot(selected_before),
        "cgroup_after": snapshot(selected_after),
        "seed_status": {
            "relative_path": f"runs/task-{task_id}/seed_status.json",
            "size": 100,
            "sha256": "4" * 64,
            "state": "completed",
            "terminal": True,
            "exit_code": 0,
        },
        "result": {
            "relative_path": f"runs/task-{task_id}/seed-{package['seed']}/result.json",
            "size": 200,
            "sha256": "5" * 64,
        },
        "exit_code": 0,
        "failure": None,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {**unsigned, "telemetry_sha256": canary.canonical_sha256(unsigned)}


def _minimal_package(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setattr(canary, "validate_task", lambda value, **_kwargs: dict(value))
    source = _source_task(config)
    baseline = _baseline_summary(config)
    candidate = canary._transform_source_task(source, config=config, baseline=baseline)
    return {
        "seed": config["seed"],
        "bundle_id": config["expected_bundle_id"],
        "bundle_manifest_sha256": config["expected_bundle_manifest_sha256"],
        "publication_receipt_sha256": "6" * 64,
        "package_sha256": "7" * 64,
        "source_4cpu_logical_dedupe_key": source["dedupe_key"],
        "candidate_task": candidate,
        "candidate_dedupe_key": candidate["dedupe_key"],
        "baseline": baseline,
    }


def test_sealed_config_is_exact_and_candidate_seed_is_distinct():
    config = _config()
    assert config["seed"] == 2_257_499_993
    assert config["baseline_task_id"] == 84_880
    assert config["candidate_cpus"] == 2
    assert config["candidate_memory_mb"] == 28_672


def test_v3_config_and_package_identity_are_launch_forbidden():
    old_config = json.loads(
        (
            CONFIG_PATH.parent
            / "tier1_final1000_resource2_canary_20260723.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(RuntimeError, match="config seal mismatch"):
        canary.validate_config(old_config)
    assert (
        canary.SUPERSEDED_V3_PACKAGE_SHA256
        == "98a16b3177a3f66f9644ee92c0bfac27304746b7e3ed3db74a59f5b8d2e3e303"
    )
    assert canary.PACKAGE_SCHEMA.endswith("-v2")
    assert canary.TELEMETRY_SCHEMA.endswith("-v2")


def test_render_fails_closed_when_baseline_files_are_absent(tmp_path: Path):
    missing = tmp_path / "absent.json"
    with pytest.raises(RuntimeError, match="baseline package is unavailable"):
        canary.build_package(
            config_path=CONFIG_PATH,
            offload_plan_path=missing,
            publication_receipt_path=missing,
            baseline_package_path=missing,
            baseline_submission_path=missing,
            baseline_terminal_path=missing,
            scheduler_url="http://127.0.0.1:8002",
        )


def test_baseline_must_be_promotion_eligible_and_exact_lineage():
    config = _config()
    package = {"package_sha256": config["baseline_package_sha256"]}
    submission = {"receipt_sha256": config["baseline_submission_receipt_sha256"]}
    terminal = _baseline_terminal(config)
    assert (
        canary.validate_baseline_terminal(
            terminal,
            config=config,
            baseline_package=package,
            baseline_submission=submission,
        )["task_id"]
        == 84_880
    )
    bad = copy.deepcopy(terminal)
    bad["promotion_eligible"] = False
    unsigned = {k: v for k, v in bad.items() if k != "remote_terminal_sha256"}
    bad["remote_terminal_sha256"] = canary.canonical_sha256(unsigned)
    with pytest.raises(
        RuntimeError, match="not an eligible GET-authenticated terminal"
    ):
        canary.validate_baseline_terminal(
            bad,
            config=config,
            baseline_package=package,
            baseline_submission=submission,
        )


def test_candidate_is_exact_2cpu_isolated_and_uses_only_pbd6_single_runner(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config()
    monkeypatch.setattr(canary, "validate_task", lambda value, **_kwargs: dict(value))
    source = _source_task(config)
    candidate = canary._transform_source_task(
        source, config=config, baseline=_baseline_summary(config)
    )
    assert candidate["cpus"] == 2
    assert candidate["memory_mb"] == 28_672
    assert candidate["payload_json"]["scheduler_cpus"] == 2
    assert candidate["payload_json"]["inference_threads"] == 2
    assert candidate["name"].startswith(canary.CANARY_PREFIX)
    assert not candidate["name"].startswith(canary.PRODUCTION_PREFIX)
    assert candidate["dedupe_key"].startswith(canary.CANARY_DEDUPE_PREFIX)
    assert "tier1_corrected_current7_slurm_seed_runner.py" in candidate["command"]
    assert "tier1_final1000_multiseed_phase_b_runner.py" not in candidate["command"]
    assert "unique finite cgroup memory hierarchy is unavailable" in candidate["command"]
    assert "memory.limit_in_bytes" in candidate["command"]
    assert "memory.max_usage_in_bytes" in candidate["command"]
    assert 'cgroup_diagnostics["before"] = capture_cgroup_diagnostics()' in candidate[
        "command"
    ]
    assert "SLURM_CPUS_PER_TASK" in candidate["command"]


def test_pbd6_runtime_hard_hash_rejects_any_runner_drift():
    manifest = {
        "files": {
            path: {"sha256": digest, "size": 1}
            for path, digest in canary.PBD6_RUNTIME_SHA256.items()
        }
    }
    assert canary._runtime_hard_hashes(manifest) == canary.PBD6_RUNTIME_SHA256
    manifest["files"][next(iter(canary.PBD6_RUNTIME_SHA256))]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="runtime hard hash drifted"):
        canary._runtime_hard_hashes(manifest)


def test_telemetry_accepts_unbounded_leaf_with_finite_parent(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config()
    package = _minimal_package(config, monkeypatch)
    telemetry = _telemetry(package)
    sealed = canary.validate_telemetry(telemetry, package=package, task_id=90001)
    assert sealed["cgroup_after"]["nearest_accounting_limit_unbounded"] is True
    assert sealed["cgroup_after"]["selected_finite_ancestor"]["depth_from_leaf"] == 1
    bad = copy.deepcopy(telemetry)
    bad["cgroup_after"]["selected_finite_ancestor"]["memory_limit_bytes"] = None
    unsigned = {k: v for k, v in bad.items() if k != "telemetry_sha256"}
    bad["telemetry_sha256"] = canary.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="cgroup"):
        canary.validate_telemetry(bad, package=package, task_id=90001)


def test_telemetry_accepts_actual_slurm_v1_memory_hierarchy(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config()
    package = _minimal_package(config, monkeypatch)
    telemetry = _telemetry(package)
    cgroup_text = "6:memory:/slurm_n012/system\n"
    mountinfo_text = (
        "41 32 0:34 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime "
        "shared:18 - cgroup cgroup rw,memory\n"
    )

    def record(path: str, text: str) -> dict[str, Any]:
        raw = text.encode("utf-8")
        return {
            "path": path,
            "maximum_bytes": canary.cgroup_memory.CGROUP_DIAGNOSTIC_MAX_BYTES,
            "bytes_captured": len(raw),
            "truncated": False,
            "utf8_valid": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": text,
            "error": None,
        }

    diagnostics = {
        "schema_version": canary.cgroup_memory.CGROUP_DIAGNOSTIC_SCHEMA,
        "maximum_bytes_per_file": (
            canary.cgroup_memory.CGROUP_DIAGNOSTIC_MAX_BYTES
        ),
        "sysfs_root": "/sys/fs/cgroup",
        "proc_self_cgroup": record("/proc/self/cgroup", cgroup_text),
        "proc_self_mountinfo": record("/proc/self/mountinfo", mountinfo_text),
    }

    def snapshot(current: int, peak: int) -> dict[str, Any]:
        selected = {
            "depth_from_leaf": 0,
            "relative_path": "memory/slurm_n012/system",
            "memory_max_raw": str(canary.CANDIDATE_MEMORY_MB * 1024**2),
            "memory_limit_bytes": canary.CANDIDATE_MEMORY_MB * 1024**2,
            "memory_current_bytes": current,
            "memory_peak_bytes": peak,
        }
        return {
            "schema_version": canary.cgroup_memory.CGROUP_SNAPSHOT_SCHEMA,
            "cgroup_version": "v1",
            "hierarchy_id": 6,
            "membership_controllers": ["memory"],
            "membership_path": "/slurm_n012/system",
            "mount_root": "/",
            "mount_relative_path": "memory",
            "leaf_relative_path": "memory/slurm_n012/system",
            "nearest_accounting_depth": 0,
            "nearest_accounting_limit_unbounded": False,
            "limit_filename": "memory.limit_in_bytes",
            "current_filename": "memory.usage_in_bytes",
            "peak_filename": "memory.max_usage_in_bytes",
            "ancestors": [selected],
            "selected_finite_ancestor": selected,
        }

    telemetry["cgroup_diagnostics"] = {
        "before": copy.deepcopy(diagnostics),
        "after": copy.deepcopy(diagnostics),
    }
    telemetry["cgroup_before"] = snapshot(1024, 2048)
    telemetry["cgroup_after"] = snapshot(2048, 4096)
    unsigned = {key: value for key, value in telemetry.items() if key != "telemetry_sha256"}
    telemetry["telemetry_sha256"] = canary.canonical_sha256(unsigned)
    sealed = canary.validate_telemetry(telemetry, package=package, task_id=90001)
    assert sealed["cgroup_after"]["cgroup_version"] == "v1"
    assert sealed["cgroup_after"]["mount_relative_path"] == "memory"


def test_terminal_calls_manifest_bound_result_validator_and_passes_throughput(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config()
    package = _minimal_package(config, monkeypatch)
    monkeypatch.setattr(canary, "validate_package", lambda value: dict(value))
    calls: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    monkeypatch.setattr(
        canary,
        "validate_result",
        lambda result, *, payload, manifest: calls.append(
            (dict(result), dict(payload), dict(manifest))
        ),
    )
    telemetry = _telemetry(package)
    status = {
        "schema_version": canary.STATUS_SCHEMA,
        "state": "completed",
        "terminal": True,
        "exit_code": 0,
        "task_id": "90001",
        "seed": package["seed"],
        "bundle_id": package["bundle_id"],
        "payload_sha256": canary.canonical_sha256(
            package["candidate_task"]["payload_json"]
        ),
        "inference_threads": 2,
        "result_sha256": "5" * 64,
    }
    result = {"inference_threads": 2, "scheduler_cpus": 2}
    terminal = canary.evaluate_terminal_evidence(
        package=package,
        manifest={"bundle_id": package["bundle_id"]},
        scheduler_status="completed",
        task_id=90001,
        telemetry=telemetry,
        seed_status=status,
        seed_status_file_sha256="4" * 64,
        seed_status_size=100,
        result=result,
        result_file_sha256="5" * 64,
        result_size=200,
    )
    assert len(calls) == 1
    assert terminal["promotion_eligible"] is True
    assert terminal["slot_weighted_throughput_ratio_vs_4cpu"] > 1.5


def test_remote_baseline_reverification_reads_and_hashes_actual_bytes(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config()
    terminal = _baseline_terminal(config)
    package = {"parent_task": {"sealed": True}}
    monkeypatch.setattr(
        canary,
        "_authenticate_task",
        lambda _row, _expected, *, label: (84_880, "completed"),
    )

    class Scheduler:
        def get_task(self, _task_id: int) -> dict[str, Any]:
            return {"id": 84_880}

    payload = b'{"result":"sealed"}\n'
    receipt = canary._reverify_baseline_remote(
        config=config,
        baseline_package=package,
        baseline_terminal=terminal,
        scheduler_url="http://127.0.0.1:8002",
        scheduler=Scheduler(),
        remote_file_reader=lambda **_kwargs: payload,
    )
    assert receipt["reverified"] is True
    assert receipt["scheduler_remote_file_get_count"] == 1
    assert receipt["scheduler_remote_file_attempt_count"] == 1
    assert receipt["remote_read_policy"] == canary._remote_read_policy()
    assert (
        receipt["remote_file_read_attempts"][0]["read_limit_bytes"] == len(payload) + 1
    )


def test_bounded_remote_reader_requests_expected_size_plus_one_and_checks_receipt(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"sealed remote evidence\n"
    observed: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            observed["read_limit"] = limit
            return payload

    def urlopen(request, *, timeout: float):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    result = canary._remote_file_bytes_bounded(
        scheduler_url="http://127.0.0.1:8002",
        task_id=84_880,
        relative_path="runs/task-84880/result.json",
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        attempt_audit=audit,
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(observed["url"]).query)
    assert result == payload
    assert query == {
        "base": ["remote_cwd"],
        "max_bytes": [str(len(payload) + 1)],
        "path": ["runs/task-84880/result.json"],
    }
    assert observed["read_limit"] == len(payload) + 1
    assert observed["timeout"] == 60.0
    assert audit == [
        {
            "relative_path": "runs/task-84880/result.json",
            "expected_size": len(payload),
            "request_max_bytes": len(payload) + 1,
            "read_limit_bytes": len(payload) + 1,
            "attempt_count": 1,
            "retry_count": 0,
        }
    ]


def test_bounded_remote_reader_accepts_sealed_empty_file(
    monkeypatch: pytest.MonkeyPatch,
):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return b""

    monkeypatch.setattr(
        canary.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    assert (
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="stderr.log",
            expected_size=0,
            expected_sha256=hashlib.sha256(b"").hexdigest(),
        )
        == b""
    )


def test_bounded_remote_reader_fails_closed_on_oversize_receipt():
    with pytest.raises(RuntimeError, match="reaches or exceeds the 16 MiB hard cap"):
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
            expected_size=canary.REMOTE_EVIDENCE_HARD_CAP_BYTES,
            expected_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    "mutated",
    [b"prefix-sealed", b"sealed-suffix"],
)
def test_bounded_remote_reader_rejects_extra_prefix_or_suffix(
    monkeypatch: pytest.MonkeyPatch, mutated: bytes
):
    expected = b"sealed"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            return mutated[:limit]

    monkeypatch.setattr(
        canary.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(RuntimeError, match="length differs"):
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
            expected_size=len(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
        )


def test_bounded_remote_reader_rejects_same_length_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = b"sealed"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return b"SEaled"

    monkeypatch.setattr(
        canary.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(RuntimeError, match="SHA differs"):
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
            expected_size=len(expected),
            expected_sha256=hashlib.sha256(expected).hexdigest(),
        )


def test_unsealed_remote_reader_rejects_ambiguous_live_hard_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, limit: int) -> bytes:
            assert limit == canary.REMOTE_EVIDENCE_HARD_CAP_BYTES
            return b"x" * limit

    monkeypatch.setattr(
        canary.urllib.request, "urlopen", lambda *_args, **_kwargs: Response()
    )
    with pytest.raises(RuntimeError, match="ambiguous hard bound"):
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
        )


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1:8002/api/tasks/84880/remote-file",
        code,
        "busy",
        {},
        io.BytesIO(body),
    )


def test_remote_reader_retries_429_and_503_with_sealed_increasing_backoff(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"sealed"
    outcomes: list[Any] = [_http_error(429, b"busy"), _http_error(503, b"busy")]
    sleeps: list[float] = []
    audit: list[dict[str, Any]] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return payload

    outcomes.append(Response())

    def urlopen(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(canary.time, "sleep", sleeps.append)
    assert (
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
            expected_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            attempt_audit=audit,
        )
        == payload
    )
    assert sleeps == [0.25, 0.5]
    assert audit[0]["attempt_count"] == 3
    assert audit[0]["retry_count"] == 2
    assert canary._remote_read_policy()["maximum_attempts"] == 4


def test_remote_reader_bounds_error_body_and_retry_attempts(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    sleeps: list[float] = []

    def urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (_ for _ in ()).throw(
            _http_error(
                429,
                b"A" * canary.HTTP_ERROR_BODY_LIMIT_BYTES + b"SECRET-SUFFIX",
            )
        )

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(canary.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError) as caught:
        canary._remote_file_bytes_bounded(
            scheduler_url="http://127.0.0.1:8002",
            task_id=84_880,
            relative_path="result.json",
        )
    assert calls == canary.REMOTE_READ_MAX_ATTEMPTS
    assert sleeps == list(canary.REMOTE_READ_BACKOFF_SECONDS)
    assert "after 4/4 attempts" in str(caught.value)
    assert "[truncated]" in str(caught.value)
    assert "SECRET-SUFFIX" not in str(caught.value)


def test_real_resource2_client_allows_exactly_one_exact_additive_post(
    monkeypatch: pytest.MonkeyPatch,
):
    package = _minimal_package(_config(), monkeypatch)
    candidate = package["candidate_task"]
    requests: list[Any] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps({"id": 90_001, **candidate}).encode("utf-8")

    def urlopen(request, *, timeout: float):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    client = canary.Resource2AdditiveSchedulerClient(
        "http://127.0.0.1:8002", expected_task=candidate
    )
    response = client.submit_task(candidate)
    assert response["id"] == 90_001
    assert client.post_count == 1
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.method == "POST"
    assert request.full_url == "http://127.0.0.1:8002/api/tasks"
    assert json.loads(request.data.decode("utf-8")) == candidate
    assert timeout == 30.0
    for forbidden in (
        "cancel_task",
        "preempt_task",
        "request_allocation",
        "pin_allocation",
        "_request",
    ):
        assert not hasattr(client, forbidden)
    with pytest.raises(RuntimeError, match="already attempted"):
        client.submit_task(candidate)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("field", "mutator"),
    [
        ("name", lambda value: "extra-" + value),
        ("name", lambda value: value + "-suffix"),
        ("dedupe_key", lambda value: "extra-" + value),
        ("dedupe_key", lambda value: value + "f"),
    ],
)
def test_resource2_client_rejects_extra_prefix_or_suffix_identity(
    monkeypatch: pytest.MonkeyPatch, field: str, mutator
):
    package = _minimal_package(_config(), monkeypatch)
    candidate = copy.deepcopy(package["candidate_task"])
    candidate[field] = mutator(candidate[field])
    with pytest.raises(RuntimeError, match="exact namespace"):
        canary.Resource2AdditiveSchedulerClient(
            "http://127.0.0.1:8002", expected_task=candidate
        )


def test_resource2_client_post_does_not_retry_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    package = _minimal_package(_config(), monkeypatch)
    candidate = package["candidate_task"]
    calls = 0

    def urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _http_error(429, b"busy")

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    client = canary.Resource2AdditiveSchedulerClient(
        "http://127.0.0.1:8002", expected_task=candidate
    )
    with pytest.raises(RuntimeError, match="without retry"):
        client.submit_task(candidate)
    assert calls == 1
    assert client.post_count == 1


def test_resource2_client_get_retries_503_with_bounded_policy(
    monkeypatch: pytest.MonkeyPatch,
):
    outcomes: list[Any] = [_http_error(503, b"busy")]
    sleeps: list[float] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit: int) -> bytes:
            return b'{"id":90001}'

    outcomes.append(Response())

    def urlopen(*_args, **_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(canary.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(canary.time, "sleep", sleeps.append)
    client = canary.Resource2AdditiveSchedulerClient("http://127.0.0.1:8002")
    assert client.get_task(90_001) == {"id": 90_001}
    assert client.get_attempt_count == 2
    assert client.get_retry_count == 1
    assert sleeps == [canary.REMOTE_READ_BACKOFF_SECONDS[0]]
    assert canary._scheduler_client_policy()["get_retry_maximum_attempts"] == 4


def test_complete_namespace_dry_run_never_posts(monkeypatch: pytest.MonkeyPatch):
    config = _config()
    package = _minimal_package(config, monkeypatch)
    monkeypatch.setattr(canary, "validate_package", lambda value: dict(value))
    monkeypatch.setattr(canary.shape1, "_live_ready", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        canary,
        "_authenticate_task",
        lambda _row, _expected, *, label: (84_880, "completed"),
    )

    class Scheduler:
        post_count = 0

        def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
            return []

        def get_task(self, _task_id: int) -> dict[str, Any]:
            return {}

    outcome = canary._submission_outcome(
        package=package,
        baseline_package={"parent_task": {}},
        publication={},
        transport=object(),
        scheduler=Scheduler(),
        apply=False,
        receipt_out=None,
    )
    assert outcome["scheduler_post_count"] == 0
    assert outcome["task_id"] is None


def test_post_submit_source_dedupe_race_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = _config()
    package = _minimal_package(config, monkeypatch)
    monkeypatch.setattr(canary, "validate_package", lambda value: dict(value))
    monkeypatch.setattr(canary.shape1, "_live_ready", lambda *_args, **_kwargs: {})

    class Scheduler:
        post_count = 0

        def __init__(self) -> None:
            self.posted = False

        def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
            if not self.posted:
                return []
            return [
                {
                    "id": 90001,
                    "name": package["candidate_task"]["name"],
                    "dedupe_key": package["candidate_dedupe_key"],
                    "status": "queued",
                },
                {
                    "id": 90002,
                    "name": f"{canary.PRODUCTION_PREFIX}entry-refill-{package['seed']}",
                    "dedupe_key": package["source_4cpu_logical_dedupe_key"],
                    "status": "queued",
                },
            ]

        def get_task(self, task_id: int) -> dict[str, Any]:
            return {"id": task_id}

        def submit_task(self, _task: dict[str, Any]) -> dict[str, Any]:
            self.post_count += 1
            self.posted = True
            return {"id": 90001}

    def authenticate(row, expected, *, label):
        if "baseline" in label:
            return 84_880, "completed"
        return int(row["id"]), "queued"

    monkeypatch.setattr(canary, "_authenticate_task", authenticate)
    with pytest.raises(RuntimeError, match="identity changed during Scheduler POST"):
        canary._submission_outcome(
            package=package,
            baseline_package={"parent_task": {}},
            publication={},
            transport=object(),
            scheduler=Scheduler(),
            apply=True,
            receipt_out=tmp_path / "receipt.json",
        )
