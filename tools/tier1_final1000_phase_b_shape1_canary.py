"""Render and fail-closed submit one entry Phase-B diagnostic parent.

The diagnostic is deliberately a one-parent/one-logical-seed execution of the
Phase-B concurrent-lane runner.  ``render`` performs no remote operation.
``submit`` re-authenticates the local publication, re-reads the live remote
READY over SSH, walks the complete Final1000 Scheduler namespace, and only
then permits its sole mutating operation: ``POST /api/tasks`` under ``--apply``.

There is no cancel, preempt, FEA, or AEDT surface in this command.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
import urllib.parse

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
    )
    from tier1_corrected_current7_slurm_bundle import sha256_file
    from tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tier1_corrected_current7_slurm_bundle import validate_required_runtime_code
    from tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
        build_concurrent_batch_task,
        validate_batch_task,
        validate_child_receipt,
        validate_task_status,
    )
    from tier1_final1000_slurm_controller import SchedulerApiClient
    from tier1_final1000_slurm_launch import (
        DEFAULT_MAX_WORKERS_PER_NODE,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        build_stage_task,
        validate_stage_binding,
    )
    from tier1_final1000_stage_profiles import BY_ID
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
    )
    from tools.tier1_corrected_current7_slurm_bundle import sha256_file
    from tools.tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tools.tier1_corrected_current7_slurm_bundle import (
        validate_required_runtime_code,
    )
    from tools.tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        INFERENCE_SAFETY,
        REMOTE_CODE_FILES,
        build_concurrent_batch_task,
        validate_batch_task,
        validate_child_receipt,
        validate_task_status,
    )
    from tools.tier1_final1000_slurm_controller import SchedulerApiClient
    from tools.tier1_final1000_slurm_launch import (
        DEFAULT_MAX_WORKERS_PER_NODE,
        DEFAULT_PEAK_RSS_GATE_BYTES,
        DEFAULT_PRIORITY,
        DEFAULT_TIMEOUT_SECONDS,
        build_stage_task,
        validate_stage_binding,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID


CONFIG_SCHEMA = "mft-tier1-final1000-phase-b-shape1-diagnostic-config-v1"
PACKAGE_SCHEMA = "mft-tier1-final1000-phase-b-shape1-diagnostic-package-v1"
SUBMISSION_SCHEMA = "mft-tier1-final1000-phase-b-shape1-submission-v1"
CPU_EVIDENCE_SCHEMA = "mft-tier1-final1000-phase-b-terminal-cpu-evidence-v1"
ENTRY_STAGE_ID = "entry-1200-t125"
FINAL1000_TASK_PREFIX = "mft-t1fg-"
PAGE_SIZE = 10_000
MAX_PAGES = 10_000
DIAGNOSTIC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable output already exists: {path}")
    staged = path.with_name(path.name + f".part.{os.getpid()}")
    try:
        staged.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    required = {
        "schema_version",
        "diagnostic_id",
        "stage_id",
        "seed",
        "seed_selection_policy",
        "expected_offload_plan_file_sha256",
        "expected_bundle_id",
        "expected_bundle_manifest_sha256",
        "expected_bundle_contract_sha256",
        "lane_shape",
        "logical_seed_count",
        "priority",
        "cpus",
        "memory_mb",
        "scheduling_profile",
        "gpus",
        "max_workers_per_node",
        "timeout_seconds",
        "scheduler_dedupe_policy",
        "remote_ready_reread_required",
        "phase_b_runner_required",
        "fea_submission_performed",
        "aedt_used",
        "config_sha256",
    }
    stage = BY_ID[ENTRY_STAGE_ID]
    if (
        set(value) != required
        or value.get("schema_version") != CONFIG_SCHEMA
        or not DIAGNOSTIC_ID_PATTERN.fullmatch(str(value.get("diagnostic_id") or ""))
        or value.get("stage_id") != ENTRY_STAGE_ID
        or value.get("seed") != stage.seed_window_end_exclusive - 1
        or value.get("seed_selection_policy")
        != "last-entry-window-seed-reserved-for-this-delta-shape1-diagnostic"
        or not _is_sha256(value.get("expected_offload_plan_file_sha256"))
        or not str(value.get("expected_bundle_id") or "").startswith("current7-")
        or not _is_sha256(value.get("expected_bundle_manifest_sha256"))
        or not _is_sha256(value.get("expected_bundle_contract_sha256"))
        or value.get("lane_shape") != 1
        or value.get("logical_seed_count") != 1
        or value.get("priority") != DEFAULT_PRIORITY
        or value.get("cpus") != CHILD_CPUS
        or value.get("memory_mb") != CHILD_MEMORY_MB
        or value.get("scheduling_profile") != "standard"
        or value.get("gpus") != 0
        or value.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or value.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or value.get("scheduler_dedupe_policy")
        != "parent-and-logical-dedupes-absent-from-complete-namespace-before-first-post"
        or value.get("remote_ready_reread_required") is not True
        or value.get("phase_b_runner_required") is not True
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or value.get("config_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B shape1 diagnostic config seal mismatch")
    return copy.deepcopy(dict(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _terminal_evidence(remote_cwd: str, seed: int) -> dict[str, Any]:
    root = PurePosixPath(remote_cwd) / "runs" / "task-{scheduler_task_id}"
    child = root / f"seed-{seed}"
    return {
        "task_root": root.as_posix(),
        "batch_manifest_path": (root / "batch_manifest.json").as_posix(),
        "task_status_path": (root / "task_status.json").as_posix(),
        "legacy_child_status_path": (child / "legacy_seed_status.json").as_posix(),
        "child_receipt_path": (child / "seed_status.json").as_posix(),
        "result_path": (child / "result.json").as_posix(),
        "semlock_evidence_json_pointer": (
            "/legacy_status/phase_b_semlock_stress_sha256"
        ),
        "affinity_evidence_json_pointer": "/cpu_set",
        "rss_evidence_json_pointer": "/legacy_status/observed_peak_rss_bytes",
        "dispatch_evidence_file": "task_status_path",
        "dispatch_evidence_json_pointer": "/resource_telemetry/dispatch_fill_seconds",
        "parent_elapsed_seconds_json_pointer": (
            "/resource_telemetry/parent_wall_time_seconds"
        ),
        "total_child_cpu_seconds_json_pointer": (
            "/resource_telemetry/aggregate_child_cpu_seconds"
        ),
        "reserved_cpu_utilization_json_pointer": (
            "/resource_telemetry/aggregate_cpu_utilization_fraction"
        ),
        "sum_logical_seed_elapsed_seconds_json_pointer": (
            "/resource_telemetry/sum_child_wall_time_seconds"
        ),
        "reported_child_cpu_seconds_json_pointer": (
            "/resource_telemetry/reported_child_process_tree_cpu_seconds"
        ),
        "reported_child_count_json_pointer": (
            "/resource_telemetry/reported_child_cpu_available_count"
        ),
        "derived_cpu_metrics": {
            "parent_average_cores_used": (
                "aggregate_child_cpu_seconds / parent_wall_time_seconds"
            ),
            "weighted_average_cores_used_per_logical_seed": (
                "reported_child_process_tree_cpu_seconds / "
                "sum_child_wall_time_seconds"
            ),
            "reserved_cpu_utilization_fraction": (
                "aggregate_child_cpu_seconds / "
                "requested_cpu_capacity_seconds"
            ),
            "logical_seed_cpu_utilization_fraction": (
                "reported_child_process_tree_cpu_seconds / "
                "reported_child_cpu_capacity_seconds"
            ),
        },
        "terminal_cpu_evaluator": (
            "tools/tier1_final1000_phase_b_shape1_canary.py::"
            "evaluate_terminal_cpu_evidence"
        ),
        "terminal_cpu_evidence_schema": CPU_EVIDENCE_SCHEMA,
    }


def _terminal_gates() -> dict[str, Any]:
    return {
        "scheduler_terminal_state": "completed",
        "parent_task_state": "completed",
        "completed_child_count": 1,
        "failed_child_count": 0,
        "sealed_child_count": 1,
        "semlock_stress_sha256_required": True,
        "semlock_or_enospc_count": 0,
        "exact_child_cpu_set_length": CHILD_CPUS,
        "affinity_escape_count": 0,
        "maximum_child_peak_rss_bytes": DEFAULT_PEAK_RSS_GATE_BYTES,
        "maximum_dispatch_fill_seconds": 35.0,
        "cpu_telemetry_available_required": True,
        "reported_child_cpu_available_count": 1,
        "minimum_weighted_average_cores_used_per_logical_seed": 0.75,
        "minimum_logical_seed_cpu_utilization_fraction": 0.1875,
        "low_cpu_utilization_blocks_shape_promotion": True,
        "affinity_thread_environment_or_lane_bottleneck_requires_diagnosis": True,
        "result_sha256_required": True,
        "child_terminal_receipt_required": True,
    }


def evaluate_terminal_cpu_evidence(
    task_status: Mapping[str, Any],
    child_receipts: Sequence[Mapping[str, Any]],
    *,
    expected_shape: int,
    baseline_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive CPU-seconds/elapsed utilization and a fail-closed promotion gate."""

    if isinstance(expected_shape, bool) or expected_shape not in {1, 4, 8}:
        raise RuntimeError("terminal CPU evidence shape must be one, four, or eight")
    status = validate_task_status(task_status)
    receipts = [validate_child_receipt(value) for value in child_receipts]
    telemetry = status.get("resource_telemetry") or {}
    if len(receipts) != expected_shape:
        raise RuntimeError("terminal CPU evidence child receipt count mismatch")

    parent_wall = float(telemetry.get("parent_wall_time_seconds") or 0.0)
    total_cpu_raw = telemetry.get("aggregate_child_cpu_seconds")
    total_cpu = float(total_cpu_raw) if total_cpu_raw is not None else 0.0
    child_wall = float(telemetry.get("sum_child_wall_time_seconds") or 0.0)
    reported_cpu = float(
        telemetry.get("reported_child_process_tree_cpu_seconds") or 0.0
    )
    parent_average_cores = total_cpu / parent_wall if parent_wall > 0 else 0.0
    weighted_seed_average_cores = (
        reported_cpu / child_wall if child_wall > 0 else 0.0
    )
    reserved_utilization = float(
        telemetry.get("aggregate_cpu_utilization_fraction") or 0.0
    )
    logical_utilization = float(
        telemetry.get("reported_child_cpu_utilization_fraction") or 0.0
    )
    per_seed_average_cores: list[dict[str, Any]] = []
    seen_cpus: set[int] = set()
    affinity_ok = True
    semlock_ok = True
    rss_ok = True
    completed_ok = True
    for receipt in receipts:
        child_telemetry = receipt["child_resource_telemetry"]
        wall = float(child_telemetry["wall_time_seconds"])
        cpu_raw = child_telemetry["process_tree_cpu_seconds"]
        cpu = float(cpu_raw) if cpu_raw is not None else 0.0
        cpuset = [int(value) for value in receipt["cpu_set"]]
        if len(cpuset) != CHILD_CPUS or seen_cpus.intersection(cpuset):
            affinity_ok = False
        seen_cpus.update(cpuset)
        legacy = receipt["legacy_status"]
        semlock_sha = legacy.get("phase_b_semlock_stress_sha256")
        semlock_ok = semlock_ok and _is_sha256(semlock_sha)
        peak_rss = legacy.get("observed_peak_rss_bytes")
        rss_ok = rss_ok and (
            isinstance(peak_rss, int)
            and not isinstance(peak_rss, bool)
            and 0 <= peak_rss <= DEFAULT_PEAK_RSS_GATE_BYTES
        )
        completed_ok = completed_ok and (
            receipt.get("state") == "completed"
            and receipt.get("exit_code") == 0
            and receipt.get("failure") is None
            and receipt.get("result_sha256") is not None
        )
        per_seed_average_cores.append(
            {
                "ordinal": int(receipt["ordinal"]),
                "seed": int(receipt["seed"]),
                "wall_time_seconds": wall,
                "process_tree_cpu_seconds": cpu,
                "average_cores_used": cpu / wall if wall > 0 else 0.0,
                "cpu_set": cpuset,
            }
        )
    per_seed_average_cores.sort(key=lambda item: int(item["ordinal"]))

    baseline_average = None
    baseline_ratio = None
    baseline_gate = True
    if baseline_metrics is not None:
        sealed_baseline = validate_terminal_cpu_evidence(baseline_metrics)
        baseline_average = float(
            sealed_baseline["weighted_average_cores_used_per_logical_seed"]
        )
        baseline_ratio = (
            weighted_seed_average_cores / baseline_average
            if baseline_average > 0
            else 0.0
        )
        baseline_gate = baseline_ratio >= 0.80

    gate_results = {
        "parent_completed_without_child_failure": (
            status.get("state") == "completed"
            and status.get("completed_child_count") == expected_shape
            and status.get("failed_child_count") == 0
            and completed_ok
        ),
        "cpu_telemetry_available": telemetry.get("available") is True,
        "every_child_cpu_telemetry_reported": (
            telemetry.get("reported_child_cpu_available_count") == expected_shape
            and all(
                receipt["child_resource_telemetry"].get("available") is True
                for receipt in receipts
            )
        ),
        "disjoint_exact_four_cpu_affinity": affinity_ok,
        "semlock_stress_attested": semlock_ok,
        "peak_rss_within_22_gib": rss_ok,
        "dispatch_fill_within_gate": (
            float(telemetry.get("dispatch_fill_seconds") or 0.0) <= 35.0
        ),
        "weighted_average_cores_per_seed_at_least_0p75": (
            weighted_seed_average_cores >= 0.75
        ),
        "logical_seed_cpu_utilization_at_least_0p1875": (
            logical_utilization >= 0.1875
        ),
        "per_seed_core_use_retains_80pct_of_baseline": baseline_gate,
    }
    low_cpu = not (
        gate_results["weighted_average_cores_per_seed_at_least_0p75"]
        and gate_results["logical_seed_cpu_utilization_at_least_0p1875"]
        and baseline_gate
    )
    unsigned = {
        "schema_version": CPU_EVIDENCE_SCHEMA,
        "expected_shape": expected_shape,
        "parent_requested_cpus": telemetry.get("parent_requested_cpus"),
        "logical_seed_count": telemetry.get("logical_child_count"),
        "parent_elapsed_seconds": parent_wall,
        "total_child_cpu_seconds": total_cpu,
        "parent_average_cores_used": parent_average_cores,
        "reserved_cpu_utilization_fraction": reserved_utilization,
        "sum_logical_seed_elapsed_seconds": child_wall,
        "reported_child_cpu_seconds": reported_cpu,
        "weighted_average_cores_used_per_logical_seed": (
            weighted_seed_average_cores
        ),
        "logical_seed_cpu_utilization_fraction": logical_utilization,
        "per_seed_average_cores": per_seed_average_cores,
        "baseline_weighted_average_cores_used_per_logical_seed": baseline_average,
        "per_seed_core_use_ratio_vs_baseline": baseline_ratio,
        "gate_results": gate_results,
        "low_cpu_utilization_detected": low_cpu,
        "low_cpu_diagnosis": (
            "affinity-thread-environment-or-lane-runner-bottleneck-investigation-required"
            if low_cpu
            else None
        ),
        "promotion_eligible": all(gate_results.values()),
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {
        **unsigned,
        "terminal_cpu_evidence_sha256": canonical_sha256(unsigned),
    }


def validate_terminal_cpu_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "terminal_cpu_evidence_sha256"
    }
    required = {
        "schema_version",
        "expected_shape",
        "parent_requested_cpus",
        "logical_seed_count",
        "parent_elapsed_seconds",
        "total_child_cpu_seconds",
        "parent_average_cores_used",
        "reserved_cpu_utilization_fraction",
        "sum_logical_seed_elapsed_seconds",
        "reported_child_cpu_seconds",
        "weighted_average_cores_used_per_logical_seed",
        "logical_seed_cpu_utilization_fraction",
        "per_seed_average_cores",
        "baseline_weighted_average_cores_used_per_logical_seed",
        "per_seed_core_use_ratio_vs_baseline",
        "gate_results",
        "low_cpu_utilization_detected",
        "low_cpu_diagnosis",
        "promotion_eligible",
        "automatic_promotion_performed",
        "fea_submission_performed",
        "aedt_used",
        "terminal_cpu_evidence_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != CPU_EVIDENCE_SCHEMA
        or value.get("expected_shape") not in {1, 4, 8}
        or value.get("terminal_cpu_evidence_sha256")
        != canonical_sha256(unsigned)
        or not isinstance(value.get("gate_results"), dict)
        or value.get("promotion_eligible")
        is not all(value["gate_results"].values())
        or value.get("automatic_promotion_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("Phase-B terminal CPU evidence seal mismatch")
    return copy.deepcopy(dict(value))


def _assert_exact_parent(
    task: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    remote_bundle: str,
) -> dict[str, Any]:
    parent = validate_batch_task(task)
    payload = parent["payload_json"]
    children = payload["children"]
    child = children[0] if len(children) == 1 else None
    logical_task = child.get("task") if isinstance(child, dict) else None
    command = str(parent.get("command") or "")
    if (
        parent.get("remote_cwd") != remote_bundle
        or parent.get("cpus") != CHILD_CPUS
        or parent.get("memory_mb") != CHILD_MEMORY_MB
        or parent.get("priority") != DEFAULT_PRIORITY
        or parent.get("scheduling_profile") != "standard"
        or parent.get("gpus") != 0
        or parent.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or parent.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or payload.get("stage_id") != ENTRY_STAGE_ID
        or payload.get("batch_length") != 1
        or payload.get("concurrent_children") != 1
        or payload.get("execution_mode") != "concurrent"
        or payload.get("bundle_id") != config["expected_bundle_id"]
        or payload.get("bundle_manifest_sha256")
        != config["expected_bundle_manifest_sha256"]
        or not isinstance(child, dict)
        or child.get("seed") != config["seed"]
        or not isinstance(logical_task, dict)
        or logical_task.get("cpus") != CHILD_CPUS
        or logical_task.get("memory_mb") != CHILD_MEMORY_MB
        or logical_task.get("priority") != DEFAULT_PRIORITY
        or "tier1_final1000_multiseed_phase_b_runner.py" not in command
        or "tier1_final1000_multiseed_lane_runner.py" in command
        or any(token in command.lower() for token in ("ansysedt", "pyaedt.desktop"))
        or payload.get("fea_submission_performed") is not False
        or payload.get("aedt_used") is not False
        or logical_task["payload_json"].get("fea_submission_performed") is not False
        or logical_task["payload_json"].get("aedt_used") is not False
    ):
        raise RuntimeError("Phase-B shape1 parent execution/resource seal mismatch")
    return parent


def _assemble_package(
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    publication: Mapping[str, Any],
    parent_task: Mapping[str, Any],
    offload_plan_file_sha256: str,
    publication_receipt_file_sha256: str,
    required_runtime_code_sha256: str,
) -> dict[str, Any]:
    config = validate_config(config)
    parent = _assert_exact_parent(
        parent_task,
        config=config,
        remote_bundle=str(plan["remote_bundle"]),
    )
    child = parent["payload_json"]["children"][0]
    unsigned = {
        "schema_version": PACKAGE_SCHEMA,
        "diagnostic_id": config["diagnostic_id"],
        "config_sha256": config["config_sha256"],
        "stage_id": ENTRY_STAGE_ID,
        "seed": config["seed"],
        "lane_shape": 1,
        "logical_seed_count": 1,
        "offload_plan_file_sha256": offload_plan_file_sha256,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "bundle_contract_sha256": plan["contract_sha256"],
        "remote_bundle": plan["remote_bundle"],
        "publication_receipt_file_sha256": publication_receipt_file_sha256,
        "publication_receipt_sha256": publication["receipt_sha256"],
        "ready_sha256": publication["ready_sha256"],
        "required_runtime_code_sha256": required_runtime_code_sha256,
        "parent_task": parent,
        "parent_task_sha256": canonical_sha256(parent),
        "parent_dedupe_key": parent["dedupe_key"],
        "logical_child_dedupe_key": child["logical_dedupe_key"],
        "logical_child_payload_sha256": child["payload_sha256"],
        "terminal_evidence": _terminal_evidence(
            str(parent["remote_cwd"]), int(config["seed"])
        ),
        "terminal_gates": _terminal_gates(),
        "complete_scheduler_namespace_required": True,
        "remote_ready_reread_required": True,
        "scheduler_endpoint": "POST /api/tasks",
        "explicit_apply_required": True,
        "scheduler_write_performed": False,
        "submission_performed": False,
        "cancellation_performed": False,
        "preemption_performed": False,
        "remote_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "production_eligible": False,
    }
    return {**unsigned, "package_sha256": canonical_sha256(unsigned)}


def validate_package(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "package_sha256"}
    required = {
        "schema_version",
        "diagnostic_id",
        "config_sha256",
        "stage_id",
        "seed",
        "lane_shape",
        "logical_seed_count",
        "offload_plan_file_sha256",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_contract_sha256",
        "remote_bundle",
        "publication_receipt_file_sha256",
        "publication_receipt_sha256",
        "ready_sha256",
        "required_runtime_code_sha256",
        "parent_task",
        "parent_task_sha256",
        "parent_dedupe_key",
        "logical_child_dedupe_key",
        "logical_child_payload_sha256",
        "terminal_evidence",
        "terminal_gates",
        "complete_scheduler_namespace_required",
        "remote_ready_reread_required",
        "scheduler_endpoint",
        "explicit_apply_required",
        "scheduler_write_performed",
        "submission_performed",
        "cancellation_performed",
        "preemption_performed",
        "remote_write_performed",
        "fea_submission_performed",
        "aedt_used",
        "production_eligible",
        "package_sha256",
    }
    parent = value.get("parent_task")
    if not isinstance(parent, dict):
        raise RuntimeError("Phase-B shape1 package lacks its parent task")
    validated_parent = validate_batch_task(parent)
    validated_parent = _assert_exact_parent(
        validated_parent,
        config={
            "expected_bundle_id": value.get("bundle_id"),
            "expected_bundle_manifest_sha256": value.get(
                "bundle_manifest_sha256"
            ),
            "seed": value.get("seed"),
        },
        remote_bundle=str(value.get("remote_bundle") or ""),
    )
    child = validated_parent["payload_json"]["children"][0]
    evidence = _terminal_evidence(
        str(validated_parent["remote_cwd"]), int(value.get("seed", -1))
    )
    if (
        set(value) != required
        or value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("package_sha256") != canonical_sha256(unsigned)
        or value.get("stage_id") != ENTRY_STAGE_ID
        or value.get("lane_shape") != 1
        or value.get("logical_seed_count") != 1
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "config_sha256",
                "offload_plan_file_sha256",
                "bundle_manifest_sha256",
                "bundle_contract_sha256",
                "publication_receipt_file_sha256",
                "publication_receipt_sha256",
                "ready_sha256",
                "required_runtime_code_sha256",
                "parent_task_sha256",
                "logical_child_payload_sha256",
            )
        )
        or value.get("parent_task_sha256") != canonical_sha256(validated_parent)
        or value.get("parent_dedupe_key") != validated_parent["dedupe_key"]
        or value.get("logical_child_dedupe_key") != child["logical_dedupe_key"]
        or value.get("logical_child_payload_sha256") != child["payload_sha256"]
        or value.get("parent_dedupe_key") == value.get("logical_child_dedupe_key")
        or value.get("terminal_evidence") != evidence
        or value.get("terminal_gates") != _terminal_gates()
        or value.get("complete_scheduler_namespace_required") is not True
        or value.get("remote_ready_reread_required") is not True
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("explicit_apply_required") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "submission_performed",
                "cancellation_performed",
                "preemption_performed",
                "remote_write_performed",
                "fea_submission_performed",
                "aedt_used",
                "production_eligible",
            )
        )
    ):
        raise RuntimeError("Phase-B shape1 diagnostic package seal mismatch")
    return copy.deepcopy(dict(value))


def build_package(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
) -> dict[str, Any]:
    config = validate_config(_read_json(config_path, "diagnostic config"))
    plan_file_sha = sha256_file(offload_plan_path.resolve(strict=True))
    if plan_file_sha != config["expected_offload_plan_file_sha256"]:
        raise RuntimeError("diagnostic offload plan file SHA mismatch")
    binding = validate_stage_binding(
        bindings_root=Path.cwd(),
        record={
            "stage_id": ENTRY_STAGE_ID,
            "offload_plan": str(offload_plan_path.resolve(strict=True)),
            "publication_receipt": str(
                publication_receipt_path.resolve(strict=True)
            ),
        },
        stage=BY_ID[ENTRY_STAGE_ID],
    )
    plan = binding["plan"]
    manifest = binding["manifest"]
    publication = binding["publication"]
    if (
        plan.get("bundle_id") != config["expected_bundle_id"]
        or plan.get("bundle_manifest_sha256")
        != config["expected_bundle_manifest_sha256"]
        or plan.get("contract_sha256")
        != config["expected_bundle_contract_sha256"]
    ):
        raise RuntimeError("diagnostic delta identity differs from sealed config")
    runtime = validate_required_runtime_code(
        manifest,
        REMOTE_CODE_FILES,
        required_code_sha256={
            str(INFERENCE_SAFETY["required_helper_path"]): str(
                INFERENCE_SAFETY["required_helper_sha256"]
            )
        },
    )
    base = build_current7_task_payload(
        plan,
        manifest,
        seed=int(config["seed"]),
        priority=DEFAULT_PRIORITY,
    )
    child = build_stage_task(base, stage=BY_ID[ENTRY_STAGE_ID], wave="refill")
    parent = build_concurrent_batch_task([child])
    value = _assemble_package(
        config=config,
        plan=plan,
        publication=publication,
        parent_task=parent,
        offload_plan_file_sha256=plan_file_sha,
        publication_receipt_file_sha256=sha256_file(
            publication_receipt_path.resolve(strict=True)
        ),
        required_runtime_code_sha256=runtime["sha256"],
    )
    return validate_package(value)


def _live_ready(
    package: Mapping[str, Any],
    publication: Mapping[str, Any],
    transport: Any,
) -> dict[str, Any]:
    remote_path = str(package["remote_bundle"]).rstrip("/") + "/READY.json"
    try:
        value = json.loads(transport.read_bytes(remote_path).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("entry diagnostic live READY is unavailable") from exc
    if (
        not isinstance(value, dict)
        or value != publication.get("ready")
        or canonical_sha256(value) != publication.get("ready_sha256")
        or canonical_sha256(value) != package.get("ready_sha256")
    ):
        raise RuntimeError("entry diagnostic live READY changed after publication")
    return copy.deepcopy(value)


def _complete_namespace(scheduler: Any) -> list[dict[str, Any]]:
    complete = getattr(scheduler, "list_complete_namespace_tasks", None)
    if callable(complete):
        rows = complete()
        if not isinstance(rows, list):
            raise RuntimeError("complete Scheduler namespace is not a list")
        return _validate_complete_rows(rows)

    request = getattr(scheduler, "_request", None)
    if not callable(request):
        raise RuntimeError("Scheduler client lacks complete namespace pagination")
    def read_page(*, page: int, before_id: int) -> dict[str, Any]:
        query: dict[str, Any] = {
            # Full paged rows retain dedupe_key.  The deployed ultra-compact
            # serializer intentionally omits it, while the legacy filtered
            # full-list branch does not apply before_id.  Paged+before_id is
            # therefore the only authenticated complete snapshot surface.
            "compact": "false",
            "paged": "true",
            "page": page,
            "page_size": PAGE_SIZE,
            "name_prefix": FINAL1000_TASK_PREFIX,
            "sort_by": "id",
            "sort_order": "desc",
        }
        if before_id:
            query["before_id"] = before_id
        payload = request("/api/tasks?" + urllib.parse.urlencode(query))
        items = payload.get("items") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or not isinstance(items, list)
            or len(items) > PAGE_SIZE
            or payload.get("page") != page
            or payload.get("page_size") != PAGE_SIZE
            or payload.get("sort_by") != "id"
            or payload.get("sort_order") != "desc"
            or (payload.get("filters") or {}).get("name_prefix")
            != FINAL1000_TASK_PREFIX
            or (payload.get("filters") or {}).get("before_id") != before_id
        ):
            raise RuntimeError("Scheduler returned an invalid paged namespace snapshot")
        return dict(payload)

    # Observe a high-watermark, then restart below it.  New submissions cannot
    # shift the offset pages of this snapshot.
    head = read_page(page=1, before_id=0)
    head_items = _validate_complete_rows(head["items"])
    high_watermark = max(
        (int(row.get("id", row.get("task_id"))) for row in head_items), default=0
    )
    snapshot_before = high_watermark + 1 if high_watermark else 0
    first = read_page(page=1, before_id=snapshot_before)
    page_count = first.get("page_count")
    filtered_total = first.get("filtered_total")
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or not 1 <= page_count <= MAX_PAGES
        or isinstance(filtered_total, bool)
        or not isinstance(filtered_total, int)
        or filtered_total < 0
    ):
        raise RuntimeError("Scheduler paged namespace metadata drifted")
    result = [dict(row) for row in first["items"]]
    for page in range(2, page_count + 1):
        payload = read_page(page=page, before_id=snapshot_before)
        if (
            payload.get("page_count") != page_count
            or payload.get("filtered_total") != filtered_total
        ):
            raise RuntimeError("Scheduler namespace snapshot changed between pages")
        result.extend(dict(row) for row in payload["items"])
    result = _validate_complete_rows(result)
    if len(result) != filtered_total:
        raise RuntimeError("Scheduler complete namespace snapshot is truncated")

    # Re-read the newest full page.  It must overlap the sealed high-watermark,
    # proving that every task admitted during the scan is included in this
    # bounded tail.  This permits continuous controllers to keep submitting
    # without weakening the all-history dedupe proof.
    tail = read_page(page=1, before_id=0)
    tail_items = _validate_complete_rows(tail["items"])
    if high_watermark and tail_items:
        tail_min = min(int(row.get("id", row.get("task_id"))) for row in tail_items)
        if len(tail_items) == PAGE_SIZE and tail_min > high_watermark:
            raise RuntimeError("Scheduler namespace tail no longer overlaps snapshot")
    new_rows = [
        row
        for row in tail_items
        if int(row.get("id", row.get("task_id"))) > high_watermark
    ]
    return _validate_complete_rows([*new_rows, *result])


def _validate_complete_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_id: dict[int, str] = {}
    by_dedupe: dict[str, int] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Scheduler namespace contains a non-object row")
        task_id = raw.get("id", raw.get("task_id"))
        dedupe = str(raw.get("dedupe_key") or "")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not str(raw.get("name") or "").startswith(FINAL1000_TASK_PREFIX)
            or not dedupe.startswith("mft-tier1-final1000:")
            or (task_id in by_id and by_id[task_id] != dedupe)
            or (dedupe in by_dedupe and by_dedupe[dedupe] != task_id)
        ):
            raise RuntimeError("Scheduler complete namespace identity drifted")
        if task_id in by_id:
            raise RuntimeError("Scheduler complete namespace repeated a task row")
        by_id[task_id] = dedupe
        by_dedupe[dedupe] = task_id
        result.append(dict(raw))
    return result


def _authenticate_task(
    row: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> tuple[int, str]:
    observation = scheduler_task_observation(row or {})
    if observation is None or not scheduler_task_identity_matches(row or {}, expected):
        raise RuntimeError(f"{label} changed the sealed Scheduler task identity")
    return observation


def _validate_submission_receipt(
    value: Mapping[str, Any], *, package: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "observed_at",
        "apply",
        "package_sha256",
        "bundle_id",
        "ready_sha256",
        "remote_ready_reread_count",
        "complete_namespace_row_count",
        "complete_namespace_sha256",
        "parent_dedupe_key",
        "logical_child_dedupe_key",
        "both_dedupes_absent_before_post",
        "task_id",
        "task_status",
        "submitted_count",
        "scheduler_endpoint",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "remote_write_count",
        "terminal_evidence",
        "terminal_gates",
        "fea_submission_performed",
        "aedt_used",
        "receipt_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("apply") is not True
        or value.get("package_sha256") != package["package_sha256"]
        or value.get("parent_dedupe_key") != package["parent_dedupe_key"]
        or value.get("logical_child_dedupe_key")
        != package["logical_child_dedupe_key"]
        or value.get("submitted_count") != 1
        or value.get("scheduler_post_count") != 1
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("remote_write_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("existing shape1 submission receipt identity/SHA mismatch")
    return copy.deepcopy(dict(value))


def _submission_outcome(
    *,
    package: Mapping[str, Any],
    publication: Mapping[str, Any],
    transport: Any,
    scheduler: Any,
    apply: bool,
    receipt_out: Path | None,
) -> dict[str, Any]:
    package = validate_package(package)
    live = _live_ready(package, publication, transport)
    rows = _complete_namespace(scheduler)
    inventory = {str(row.get("dedupe_key") or ""): row for row in rows}
    parent_dedupe = str(package["parent_dedupe_key"])
    child_dedupe = str(package["logical_child_dedupe_key"])
    existing_receipt = None
    if receipt_out is not None and receipt_out.is_file():
        existing_receipt = _validate_submission_receipt(
            _read_json(receipt_out, "submission receipt"), package=package
        )

    parent_row = inventory.get(parent_dedupe)
    child_row = inventory.get(child_dedupe)
    expected = package["parent_task"]
    submitted_count = 0
    if existing_receipt is not None:
        if parent_row is None or child_row is not None:
            raise RuntimeError("receipt task disappeared or logical dedupe was reused")
        task_id, status = _authenticate_task(
            parent_row, expected, label="receipt Scheduler inventory"
        )
        if task_id != existing_receipt["task_id"]:
            raise RuntimeError("receipt/Scheduler task id changed")
        detail_id, _detail_status = _authenticate_task(
            scheduler.get_task(task_id),
            expected,
            label="receipt Scheduler task detail",
        )
        if detail_id != task_id:
            raise RuntimeError("receipt Scheduler inventory/detail task id changed")
        return existing_receipt

    if parent_row is not None or child_row is not None:
        raise RuntimeError(
            "shape1 parent/logical dedupe is not distinct from all prior tasks"
        )
    task_id = None
    status = "absent"
    if apply:
        if receipt_out is None:
            raise RuntimeError("--apply requires --receipt-out")
        submitted = scheduler.submit_task(expected)
        submitted_count = 1
        task_id, status = _authenticate_task(
            submitted, expected, label="Scheduler POST response"
        )
        detail = scheduler.get_task(task_id)
        detail_id, status = _authenticate_task(
            detail, expected, label="Scheduler task detail"
        )
        if detail_id != task_id:
            raise RuntimeError("Scheduler POST/detail task id mismatch")

    inventory_projection = [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in rows
    ]
    unsigned = {
        "schema_version": SUBMISSION_SCHEMA,
        "observed_at": _now(),
        "apply": bool(apply),
        "package_sha256": package["package_sha256"],
        "bundle_id": package["bundle_id"],
        "ready_sha256": canonical_sha256(live),
        "remote_ready_reread_count": 1,
        "complete_namespace_row_count": len(rows),
        "complete_namespace_sha256": canonical_sha256(inventory_projection),
        "parent_dedupe_key": parent_dedupe,
        "logical_child_dedupe_key": child_dedupe,
        "both_dedupes_absent_before_post": True,
        "task_id": task_id,
        "task_status": status,
        "submitted_count": submitted_count,
        "scheduler_endpoint": "POST /api/tasks",
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_write_count": 0,
        "terminal_evidence": package["terminal_evidence"],
        "terminal_gates": package["terminal_gates"],
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    if apply:
        if submitted_count != 1 or getattr(scheduler, "post_count", 0) != 1:
            raise RuntimeError("shape1 Scheduler POST accounting mismatch")
        _atomic_json(receipt_out, value)  # type: ignore[arg-type]
    elif getattr(scheduler, "post_count", 0) != 0:
        raise RuntimeError("shape1 dry-run performed a Scheduler POST")
    return value


def submit(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    package_path: Path,
    scheduler_url: str,
    accounts_path: Path,
    scheduler_source: Path,
    publication_account: str,
    apply: bool,
    receipt_out: Path | None,
    dry_run_out: Path | None = None,
) -> dict[str, Any]:
    if apply and dry_run_out is not None:
        raise RuntimeError("--dry-run-out cannot be combined with --apply")
    rebuilt = build_package(
        config_path=config_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
    )
    package = validate_package(_read_json(package_path, "diagnostic package"))
    if package != rebuilt:
        raise RuntimeError("diagnostic package no longer reproduces from its sources")
    publication = _read_json(publication_receipt_path, "publication receipt")
    if (
        sha256_file(publication_receipt_path.resolve(strict=True))
        != package["publication_receipt_file_sha256"]
    ):
        raise RuntimeError("publication receipt changed after package reproduction")
    with scheduler_publication_transport(
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        account_name=publication_account,
    ) as transport:
        # Keep the transport open through the Scheduler inventory/POST boundary
        # so the observed READY is the immediate pre-admission identity.
        value = _submission_outcome(
            package=package,
            publication=publication,
            transport=transport,
            scheduler=SchedulerApiClient(scheduler_url),
            apply=apply,
            receipt_out=receipt_out,
        )
    if not apply and dry_run_out is not None:
        _atomic_json(dry_run_out, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    render = sub.add_parser("render", help="render one immutable local package")
    submit_parser = sub.add_parser(
        "submit", help="dry-run by default; POST only with --apply"
    )
    evaluate = sub.add_parser(
        "evaluate", help="seal terminal CPU/affinity/RSS evidence from local JSON"
    )
    for command in (render, submit_parser):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--offload-plan", type=Path, required=True)
        command.add_argument("--publication-receipt", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    submit_parser.add_argument("--package", type=Path, required=True)
    submit_parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    submit_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    submit_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    submit_parser.add_argument("--publication-account", default="harry261")
    submit_parser.add_argument("--apply", action="store_true")
    submit_parser.add_argument("--receipt-out", type=Path)
    submit_parser.add_argument("--dry-run-out", type=Path)
    evaluate.add_argument("--task-status", type=Path, required=True)
    evaluate.add_argument(
        "--child-receipt", type=Path, action="append", required=True
    )
    evaluate.add_argument("--shape", type=int, choices=(1, 4, 8), required=True)
    evaluate.add_argument("--baseline-metrics", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.operation == "render":
        value = build_package(
            config_path=args.config,
            offload_plan_path=args.offload_plan,
            publication_receipt_path=args.publication_receipt,
        )
        _atomic_json(args.out, value)
    elif args.operation == "submit":
        value = submit(
            config_path=args.config,
            offload_plan_path=args.offload_plan,
            publication_receipt_path=args.publication_receipt,
            package_path=args.package,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            publication_account=args.publication_account,
            apply=args.apply,
            receipt_out=args.receipt_out,
            dry_run_out=args.dry_run_out,
        )
    else:
        baseline = (
            _read_json(args.baseline_metrics, "baseline terminal CPU evidence")
            if args.baseline_metrics is not None
            else None
        )
        value = evaluate_terminal_cpu_evidence(
            _read_json(args.task_status, "Phase-B task status"),
            [
                _read_json(path, f"Phase-B child receipt {index}")
                for index, path in enumerate(args.child_receipt)
            ],
            expected_shape=args.shape,
            baseline_metrics=baseline,
        )
        _atomic_json(args.out, value)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
