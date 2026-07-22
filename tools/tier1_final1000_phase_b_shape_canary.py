"""Fail-closed Phase-B shape-1/shape-4/shape-8 canary admission and evidence.

This controller-side tool generalizes the proven shape-1 diagnostic without
changing its schemas or command line.  It renders exactly one concurrent
parent, authenticates the immutable bundle and live READY, proves every
parent and logical-child dedupe absent from the complete Final1000 namespace,
and permits one additive ``POST /api/tasks`` only under ``--apply``.

Terminal evaluation is GET-only.  It seals the parent status, every child
receipt, stdout/stderr/result bytes, CPU affinity and utilization, SemLock
fault scan, aggregate RSS, and parent cgroup memory evidence.  Shape 8 cannot
promote unless it is at least 1.7x the sealed shape-4 throughput while keeping
at least 0.75 weighted cores per logical seed and passing every safety gate.

There is no cancel, preempt, FEA, AEDT, publication, or automatic-promotion
surface in this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

try:
    import tier1_final1000_phase_b_shape1_canary as shape1
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
        sha256_file,
        validate_required_runtime_code,
    )
    from tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
        INFERENCE_SAFETY,
        PARENT_MEMORY_EVIDENCE_FILENAME,
        REMOTE_CODE_FILES,
        batch_manifest_from_payload,
        build_concurrent_batch_task,
        validate_batch_manifest,
        validate_batch_task,
        validate_child_receipt,
        validate_parent_memory_evidence,
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
    from tools import tier1_final1000_phase_b_shape1_canary as shape1
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        build_task_payload as build_current7_task_payload,
        sha256_file,
        validate_required_runtime_code,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        DEFAULT_MODEL_LOAD_STAGGER_SECONDS,
        INFERENCE_SAFETY,
        PARENT_MEMORY_EVIDENCE_FILENAME,
        REMOTE_CODE_FILES,
        batch_manifest_from_payload,
        build_concurrent_batch_task,
        validate_batch_manifest,
        validate_batch_task,
        validate_child_receipt,
        validate_parent_memory_evidence,
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


CONFIG_SCHEMA = "mft-tier1-final1000-phase-b-shape-canary-config-v1"
PACKAGE_SCHEMA = "mft-tier1-final1000-phase-b-shape-canary-package-v1"
SUBMISSION_SCHEMA = "mft-tier1-final1000-phase-b-shape-canary-submission-v1"
TERMINAL_EVIDENCE_SCHEMA = (
    "mft-tier1-final1000-phase-b-shape-terminal-evidence-v1"
)
REMOTE_TERMINAL_SCHEMA = (
    "mft-tier1-final1000-phase-b-shape-remote-terminal-evidence-v1"
)
ENTRY_STAGE_ID = shape1.ENTRY_STAGE_ID
SUPPORTED_SHAPES = frozenset({1, 4, 8})
SHAPE8_MINIMUM_THROUGHPUT_RATIO = 1.7
MINIMUM_WEIGHTED_CORES_PER_SEED = 0.75
SEED_SELECTION_POLICY = (
    "explicit-unique-entry-window-seeds-reserved-for-phase-b-shape-canary"
)
SCHEDULER_DEDUPE_POLICY = (
    "parent-and-all-logical-dedupes-absent-from-complete-namespace-before-first-post"
)
_FAULT_PATTERNS = (
    re.compile(rb"oserror:\s*\[errno\s+28\]"),
    re.compile(rb"no space left on device"),
    re.compile(rb"(?:oserror|runtimeerror|exception)[^\n]{0,160}semlock"),
    re.compile(rb"semlock[^\n]{0,160}(?:failed|error|exception)"),
    # multiprocessing.SemLock failures normally place the SemLock call in a
    # traceback frame and the concrete exception on a later line.  Keep the
    # scan bounded to one modest traceback-sized window so unrelated, benign
    # log statements cannot be joined across an arbitrarily large stream.
    re.compile(
        rb"traceback \(most recent call last\):"
        rb"(?=[\s\S]{0,12288}(?:_multiprocessing\.)?semlock)"
        rb"(?=[\s\S]{0,12288}"
        rb"(?:filenotfounderror|permissionerror|oserror|runtimeerror)\s*:)"
        rb"[\s\S]{0,12288}?"
        rb"(?:filenotfounderror|permissionerror|oserror|runtimeerror)\s*:"
    ),
)


def _shape(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in SUPPORTED_SHAPES:
        raise RuntimeError("Phase-B canary shape must be exactly one, four, or eight")
    return int(value)


def _next_shape(shape: int, eligible: bool) -> int | None:
    if not eligible:
        return None
    return {1: 4, 4: 8, 8: None}[_shape(shape)]


def _parent_memory_bytes(shape: int) -> int:
    return int(shape) * CHILD_MEMORY_MB * 1024 * 1024


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    required = {
        "schema_version",
        "diagnostic_id",
        "stage_id",
        "seeds",
        "seed_selection_policy",
        "expected_offload_plan_file_sha256",
        "expected_bundle_id",
        "expected_bundle_manifest_sha256",
        "expected_bundle_contract_sha256",
        "lane_shape",
        "logical_seed_count",
        "required_shape4_remote_terminal_sha256",
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
    shape = _shape(value.get("lane_shape"))
    seeds = value.get("seeds")
    stage = BY_ID[ENTRY_STAGE_ID]
    baseline = value.get("required_shape4_remote_terminal_sha256")
    if (
        set(value) != required
        or value.get("schema_version") != CONFIG_SCHEMA
        or not shape1.DIAGNOSTIC_ID_PATTERN.fullmatch(
            str(value.get("diagnostic_id") or "")
        )
        or value.get("stage_id") != ENTRY_STAGE_ID
        or not isinstance(seeds, list)
        or len(seeds) != shape
        or len(set(seeds)) != shape
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not stage.seed_start <= seed < stage.seed_window_end_exclusive
            for seed in seeds
        )
        or value.get("seed_selection_policy") != SEED_SELECTION_POLICY
        or not shape1._is_sha256(value.get("expected_offload_plan_file_sha256"))
        or not str(value.get("expected_bundle_id") or "").startswith("current7-")
        or not shape1._is_sha256(value.get("expected_bundle_manifest_sha256"))
        or not shape1._is_sha256(value.get("expected_bundle_contract_sha256"))
        or value.get("logical_seed_count") != shape
        or (shape in {1, 4} and baseline is not None)
        or (shape == 8 and not shape1._is_sha256(baseline))
        or value.get("priority") != DEFAULT_PRIORITY
        or value.get("cpus") != shape * CHILD_CPUS
        or value.get("memory_mb") != shape * CHILD_MEMORY_MB
        or value.get("scheduling_profile") != "standard"
        or value.get("gpus") != 0
        or value.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or value.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or value.get("scheduler_dedupe_policy") != SCHEDULER_DEDUPE_POLICY
        or value.get("remote_ready_reread_required") is not True
        or value.get("phase_b_runner_required") is not True
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or value.get("config_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B shape canary config seal mismatch")
    return copy.deepcopy(dict(value))


def _terminal_evidence(remote_cwd: str, seeds: Sequence[int]) -> dict[str, Any]:
    root = PurePosixPath(remote_cwd) / "runs" / "task-{scheduler_task_id}"
    children = []
    for ordinal, seed in enumerate(seeds):
        child = root / f"seed-{int(seed)}"
        children.append(
            {
                "ordinal": ordinal,
                "seed": int(seed),
                "legacy_child_status_path": (
                    child / "legacy_seed_status.json"
                ).as_posix(),
                "child_receipt_path": (child / "seed_status.json").as_posix(),
                "child_stdout_path": (child / "child_stdout.log").as_posix(),
                "child_stderr_path": (child / "child_stderr.log").as_posix(),
                "result_path": (child / "result.json").as_posix(),
            }
        )
    return {
        "task_root": root.as_posix(),
        "batch_manifest_path": (root / "batch_manifest.json").as_posix(),
        "task_status_path": (root / "task_status.json").as_posix(),
        "parent_memory_evidence_path": (
            root / PARENT_MEMORY_EVIDENCE_FILENAME
        ).as_posix(),
        "children": children,
        "raw_stdout_stderr_sha256_required": True,
        "child_receipt_sha256_required": True,
        "result_sha256_required": True,
        "semlock_or_enospc_log_scan_required": True,
        "cpu_affinity_evidence_json_pointer": "/cpu_set",
        "child_rss_evidence_json_pointer": (
            "/legacy_status/observed_peak_rss_bytes"
        ),
        "dispatch_evidence_json_pointer": (
            "/resource_telemetry/dispatch_fill_seconds"
        ),
        "weighted_core_evidence_json_pointer": (
            "/resource_telemetry/reported_child_cpu_utilization_fraction"
        ),
        "terminal_evaluator": (
            "tools/tier1_final1000_phase_b_shape_canary.py::"
            "evaluate_terminal_evidence"
        ),
        "terminal_evidence_schema": TERMINAL_EVIDENCE_SCHEMA,
    }


def _terminal_gates(shape: int) -> dict[str, Any]:
    shape = _shape(shape)
    return {
        "scheduler_terminal_state": "completed",
        "parent_task_state": "completed",
        "completed_child_count": shape,
        "failed_child_count": 0,
        "sealed_child_count": shape,
        "semlock_stress_sha256_required_for_every_child": True,
        "semlock_or_enospc_log_occurrences": 0,
        "exact_child_cpu_set_length": CHILD_CPUS,
        "disjoint_child_cpu_sets_required": True,
        "maximum_child_peak_rss_bytes": DEFAULT_PEAK_RSS_GATE_BYTES,
        "maximum_total_child_peak_rss_bytes": _parent_memory_bytes(shape),
        "cgroup_memory_peak_and_finite_limit_required": True,
        "cgroup_peak_must_not_exceed_limit": True,
        "cgroup_limit_must_cover_parent_request": True,
        "maximum_dispatch_fill_seconds": (
            (shape - 1) * DEFAULT_MODEL_LOAD_STAGGER_SECONDS
        ),
        "minimum_weighted_average_cores_used_per_logical_seed": (
            MINIMUM_WEIGHTED_CORES_PER_SEED
        ),
        "shape4_baseline_required": shape == 8,
        "minimum_throughput_ratio_vs_shape4": (
            SHAPE8_MINIMUM_THROUGHPUT_RATIO if shape == 8 else None
        ),
        "automatic_promotion_performed": False,
    }


def _assert_exact_parent(
    task: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    remote_bundle: str,
) -> dict[str, Any]:
    config = validate_config(config)
    shape = int(config["lane_shape"])
    parent = validate_batch_task(task)
    payload = parent["payload_json"]
    children = payload.get("children")
    command = str(parent.get("command") or "")
    if not isinstance(children, list) or len(children) != shape:
        raise RuntimeError("Phase-B shape parent child count drifted")
    for ordinal, (child, seed) in enumerate(zip(children, config["seeds"])):
        logical_task = child.get("task") if isinstance(child, dict) else None
        if (
            not isinstance(child, dict)
            or child.get("ordinal") != ordinal
            or child.get("seed") != seed
            or not isinstance(logical_task, dict)
            or logical_task.get("cpus") != CHILD_CPUS
            or logical_task.get("memory_mb") != CHILD_MEMORY_MB
            or logical_task.get("priority") != DEFAULT_PRIORITY
            or logical_task.get("payload_json", {}).get(
                "fea_submission_performed"
            )
            is not False
            or logical_task.get("payload_json", {}).get("aedt_used") is not False
        ):
            raise RuntimeError("Phase-B shape logical child seal mismatch")
    if (
        parent.get("remote_cwd") != remote_bundle
        or parent.get("cpus") != shape * CHILD_CPUS
        or parent.get("memory_mb") != shape * CHILD_MEMORY_MB
        or parent.get("priority") != DEFAULT_PRIORITY
        or parent.get("scheduling_profile") != "standard"
        or parent.get("gpus") != 0
        or parent.get("max_workers_per_node") != DEFAULT_MAX_WORKERS_PER_NODE
        or parent.get("timeout_seconds") != DEFAULT_TIMEOUT_SECONDS
        or payload.get("stage_id") != ENTRY_STAGE_ID
        or payload.get("batch_length") != shape
        or payload.get("concurrent_children") != shape
        or payload.get("execution_mode") != "concurrent"
        or payload.get("bundle_id") != config["expected_bundle_id"]
        or payload.get("bundle_manifest_sha256")
        != config["expected_bundle_manifest_sha256"]
        or "tier1_final1000_multiseed_phase_b_runner.py" not in command
        or "tier1_final1000_multiseed_lane_runner.py" in command
        or any(token in command.lower() for token in ("ansysedt", "pyaedt.desktop"))
        or payload.get("fea_submission_performed") is not False
        or payload.get("aedt_used") is not False
    ):
        raise RuntimeError("Phase-B shape parent execution/resource seal mismatch")
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
        parent_task, config=config, remote_bundle=str(plan["remote_bundle"])
    )
    children = parent["payload_json"]["children"]
    logical_dedupes = [str(child["logical_dedupe_key"]) for child in children]
    payload_hashes = [str(child["payload_sha256"]) for child in children]
    unsigned = {
        "schema_version": PACKAGE_SCHEMA,
        "diagnostic_id": config["diagnostic_id"],
        "config_sha256": config["config_sha256"],
        "stage_id": ENTRY_STAGE_ID,
        "seeds": list(config["seeds"]),
        "lane_shape": config["lane_shape"],
        "logical_seed_count": config["logical_seed_count"],
        "required_shape4_remote_terminal_sha256": config[
            "required_shape4_remote_terminal_sha256"
        ],
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
        "logical_child_dedupe_keys": logical_dedupes,
        "logical_child_payload_sha256s": payload_hashes,
        "terminal_evidence": _terminal_evidence(
            str(parent["remote_cwd"]), config["seeds"]
        ),
        "terminal_gates": _terminal_gates(int(config["lane_shape"])),
        "complete_scheduler_namespace_required": True,
        "remote_ready_reread_required": True,
        "scheduler_endpoint": "POST /api/tasks",
        "explicit_apply_required": True,
        "single_additive_parent_post_only": True,
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
        "seeds",
        "lane_shape",
        "logical_seed_count",
        "required_shape4_remote_terminal_sha256",
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
        "logical_child_dedupe_keys",
        "logical_child_payload_sha256s",
        "terminal_evidence",
        "terminal_gates",
        "complete_scheduler_namespace_required",
        "remote_ready_reread_required",
        "scheduler_endpoint",
        "explicit_apply_required",
        "single_additive_parent_post_only",
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
    shape = _shape(value.get("lane_shape"))
    baseline = value.get("required_shape4_remote_terminal_sha256")
    config_unsigned = {
        "schema_version": CONFIG_SCHEMA,
        "diagnostic_id": value.get("diagnostic_id"),
        "stage_id": value.get("stage_id"),
        "seeds": value.get("seeds"),
        "seed_selection_policy": SEED_SELECTION_POLICY,
        "expected_offload_plan_file_sha256": value.get(
            "offload_plan_file_sha256"
        ),
        "expected_bundle_id": value.get("bundle_id"),
        "expected_bundle_manifest_sha256": value.get("bundle_manifest_sha256"),
        "expected_bundle_contract_sha256": value.get("bundle_contract_sha256"),
        "lane_shape": shape,
        "logical_seed_count": shape,
        "required_shape4_remote_terminal_sha256": baseline,
        "priority": DEFAULT_PRIORITY,
        "cpus": shape * CHILD_CPUS,
        "memory_mb": shape * CHILD_MEMORY_MB,
        "scheduling_profile": "standard",
        "gpus": 0,
        "max_workers_per_node": DEFAULT_MAX_WORKERS_PER_NODE,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "scheduler_dedupe_policy": SCHEDULER_DEDUPE_POLICY,
        "remote_ready_reread_required": True,
        "phase_b_runner_required": True,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    config = validate_config(
        {**config_unsigned, "config_sha256": canonical_sha256(config_unsigned)}
    )
    if value.get("config_sha256") != config["config_sha256"]:
        raise RuntimeError("Phase-B shape package/config identity mismatch")
    parent_raw = value.get("parent_task")
    if not isinstance(parent_raw, Mapping):
        raise RuntimeError("Phase-B shape package lacks its parent task")
    parent = _assert_exact_parent(
        parent_raw, config=config, remote_bundle=str(value.get("remote_bundle") or "")
    )
    children = parent["payload_json"]["children"]
    dedupes = [str(child["logical_dedupe_key"]) for child in children]
    payload_hashes = [str(child["payload_sha256"]) for child in children]
    immutable_hash_fields = (
        "config_sha256",
        "offload_plan_file_sha256",
        "bundle_manifest_sha256",
        "bundle_contract_sha256",
        "publication_receipt_file_sha256",
        "publication_receipt_sha256",
        "ready_sha256",
        "required_runtime_code_sha256",
        "parent_task_sha256",
    )
    if (
        set(value) != required
        or value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("package_sha256") != canonical_sha256(unsigned)
        or value.get("stage_id") != ENTRY_STAGE_ID
        or value.get("logical_seed_count") != shape
        or any(not shape1._is_sha256(value.get(field)) for field in immutable_hash_fields)
        or (shape in {1, 4} and baseline is not None)
        or (shape == 8 and not shape1._is_sha256(baseline))
        or value.get("parent_task_sha256") != canonical_sha256(parent)
        or value.get("parent_dedupe_key") != parent["dedupe_key"]
        or value.get("logical_child_dedupe_keys") != dedupes
        or value.get("logical_child_payload_sha256s") != payload_hashes
        or len(set(dedupes)) != shape
        or value.get("parent_dedupe_key") in set(dedupes)
        or value.get("terminal_evidence")
        != _terminal_evidence(str(parent["remote_cwd"]), config["seeds"])
        or value.get("terminal_gates") != _terminal_gates(shape)
        or value.get("complete_scheduler_namespace_required") is not True
        or value.get("remote_ready_reread_required") is not True
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("explicit_apply_required") is not True
        or value.get("single_additive_parent_post_only") is not True
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
        raise RuntimeError("Phase-B shape canary package seal mismatch")
    return copy.deepcopy(dict(value))


def build_package(
    *,
    config_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
) -> dict[str, Any]:
    config = validate_config(shape1._read_json(config_path, "shape canary config"))
    plan_file_sha = sha256_file(offload_plan_path.resolve(strict=True))
    if plan_file_sha != config["expected_offload_plan_file_sha256"]:
        raise RuntimeError("shape canary offload plan file SHA mismatch")
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
        raise RuntimeError("shape canary bundle identity differs from sealed config")
    runtime = validate_required_runtime_code(
        manifest,
        REMOTE_CODE_FILES,
        required_code_sha256={
            str(INFERENCE_SAFETY["required_helper_path"]): str(
                INFERENCE_SAFETY["required_helper_sha256"]
            )
        },
    )
    stage = BY_ID[ENTRY_STAGE_ID]
    children = []
    for seed in config["seeds"]:
        base = build_current7_task_payload(
            plan, manifest, seed=int(seed), priority=DEFAULT_PRIORITY
        )
        children.append(build_stage_task(base, stage=stage, wave="refill"))
    parent = build_concurrent_batch_task(children)
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


def _validate_submission_receipt(
    value: Mapping[str, Any], *, package: Mapping[str, Any]
) -> dict[str, Any]:
    package = validate_package(package)
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "observed_at",
        "apply",
        "package_sha256",
        "bundle_id",
        "lane_shape",
        "ready_sha256",
        "remote_ready_reread_count",
        "complete_namespace_row_count",
        "complete_namespace_max_task_id",
        "complete_namespace_sha256",
        "post_submit_namespace_row_count",
        "post_submit_namespace_max_task_id",
        "post_submit_namespace_sha256",
        "post_submit_parent_match_count",
        "post_submit_logical_match_count",
        "all_dedupes_revalidated_after_post",
        "parent_dedupe_key",
        "logical_child_dedupe_keys",
        "all_dedupes_absent_before_post",
        "task_id",
        "task_status",
        "submitted_parent_count",
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
        or not str(value.get("observed_at") or "")
        or value.get("apply") is not True
        or value.get("package_sha256") != package["package_sha256"]
        or value.get("bundle_id") != package["bundle_id"]
        or value.get("lane_shape") != package["lane_shape"]
        or isinstance(value.get("complete_namespace_row_count"), bool)
        or not isinstance(value.get("complete_namespace_row_count"), int)
        or value.get("complete_namespace_row_count") < 0
        or isinstance(value.get("complete_namespace_max_task_id"), bool)
        or not isinstance(value.get("complete_namespace_max_task_id"), int)
        or value.get("complete_namespace_max_task_id") < 0
        or not shape1._is_sha256(value.get("complete_namespace_sha256"))
        or value.get("remote_ready_reread_count") != 1
        or value.get("ready_sha256") != package["ready_sha256"]
        or value.get("parent_dedupe_key") != package["parent_dedupe_key"]
        or value.get("logical_child_dedupe_keys")
        != package["logical_child_dedupe_keys"]
        or value.get("all_dedupes_absent_before_post") is not True
        or isinstance(value.get("task_id"), bool)
        or not isinstance(value.get("task_id"), int)
        or value.get("task_id") <= value.get("complete_namespace_max_task_id")
        or isinstance(value.get("post_submit_namespace_row_count"), bool)
        or not isinstance(value.get("post_submit_namespace_row_count"), int)
        or value.get("post_submit_namespace_row_count") < 1
        or isinstance(value.get("post_submit_namespace_max_task_id"), bool)
        or not isinstance(value.get("post_submit_namespace_max_task_id"), int)
        or value.get("post_submit_namespace_max_task_id") < value.get("task_id")
        or not shape1._is_sha256(value.get("post_submit_namespace_sha256"))
        or value.get("post_submit_parent_match_count") != 1
        or value.get("post_submit_logical_match_count") != 0
        or value.get("all_dedupes_revalidated_after_post") is not True
        or value.get("task_status")
        not in {
            "queued",
            "running",
            "attaching",
            "held",
            "pending",
            "completed",
            "failed",
            "cancelled",
            "timeout",
            "timed_out",
        }
        or value.get("submitted_parent_count") != 1
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("terminal_evidence") != package["terminal_evidence"]
        or value.get("terminal_gates") != package["terminal_gates"]
        or value.get("scheduler_post_count") != 1
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("remote_write_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("existing shape canary submission receipt drifted")
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
    live = shape1._live_ready(package, publication, transport)
    rows = shape1._complete_namespace(scheduler)
    inventory = {str(row.get("dedupe_key") or ""): row for row in rows}
    parent_dedupe = str(package["parent_dedupe_key"])
    child_dedupes = [str(value) for value in package["logical_child_dedupe_keys"]]
    all_dedupes = [parent_dedupe, *child_dedupes]
    existing_receipt = None
    if receipt_out is not None and receipt_out.is_file():
        existing_receipt = _validate_submission_receipt(
            shape1._read_json(receipt_out, "shape submission receipt"),
            package=package,
        )
    parent_row = inventory.get(parent_dedupe)
    logical_rows = [inventory.get(value) for value in child_dedupes]
    expected = package["parent_task"]
    submitted_count = 0
    if existing_receipt is not None:
        if parent_row is None or any(row is not None for row in logical_rows):
            raise RuntimeError("receipt task disappeared or logical dedupe was reused")
        task_id, _status = shape1._authenticate_task(
            parent_row, expected, label="receipt Scheduler inventory"
        )
        if task_id != existing_receipt["task_id"]:
            raise RuntimeError("receipt/Scheduler task id changed")
        detail_id, _detail_status = shape1._authenticate_task(
            scheduler.get_task(task_id),
            expected,
            label="receipt Scheduler task detail",
        )
        if detail_id != task_id:
            raise RuntimeError("receipt Scheduler inventory/detail task id changed")
        return existing_receipt
    if any(value in inventory for value in all_dedupes):
        raise RuntimeError(
            "shape parent/logical dedupe is not distinct from all prior tasks"
        )
    task_id = None
    status = "absent"
    post_rows: list[dict[str, Any]] = []
    post_parent_match_count = 0
    post_logical_match_count = 0
    post_revalidated = False
    if apply:
        if receipt_out is None:
            raise RuntimeError("--apply requires --receipt-out")
        submitted = scheduler.submit_task(expected)
        submitted_count = 1
        task_id, status = shape1._authenticate_task(
            submitted, expected, label="Scheduler POST response"
        )
        detail_id, status = shape1._authenticate_task(
            scheduler.get_task(task_id), expected, label="Scheduler task detail"
        )
        if detail_id != task_id:
            raise RuntimeError("Scheduler POST/detail task id mismatch")
        # Close the namespace-check/POST race with a second complete snapshot.
        # The submitted parent must be the sole authenticated parent-dedupe
        # row and no logical child dedupe may have appeared concurrently.
        post_rows = shape1._complete_namespace(scheduler)
        post_parent_rows = [
            row
            for row in post_rows
            if str(row.get("dedupe_key") or "") == parent_dedupe
        ]
        post_logical_rows = [
            row
            for row in post_rows
            if str(row.get("dedupe_key") or "") in child_dedupes
        ]
        post_parent_match_count = len(post_parent_rows)
        post_logical_match_count = len(post_logical_rows)
        if post_parent_match_count != 1 or post_logical_match_count != 0:
            raise RuntimeError(
                "shape dedupe identity changed during Scheduler POST"
            )
        post_task_id, _post_status = shape1._authenticate_task(
            post_parent_rows[0],
            expected,
            label="post-submit Scheduler inventory",
        )
        if post_task_id != task_id:
            raise RuntimeError("Scheduler POST/post-inventory task id mismatch")
        post_revalidated = True
    projection = [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in rows
    ]
    namespace_max_task_id = max(
        (int(row.get("id", row.get("task_id"))) for row in rows), default=0
    )
    if apply and (task_id is None or task_id <= namespace_max_task_id):
        raise RuntimeError("shape canary POST did not create a new additive task id")
    post_projection = [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in post_rows
    ]
    post_namespace_max_task_id = max(
        (int(row.get("id", row.get("task_id"))) for row in post_rows), default=0
    )
    if apply and post_namespace_max_task_id < int(task_id):
        raise RuntimeError("post-submit namespace omitted the submitted task id")
    unsigned = {
        "schema_version": SUBMISSION_SCHEMA,
        "observed_at": shape1._now(),
        "apply": bool(apply),
        "package_sha256": package["package_sha256"],
        "bundle_id": package["bundle_id"],
        "lane_shape": package["lane_shape"],
        "ready_sha256": canonical_sha256(live),
        "remote_ready_reread_count": 1,
        "complete_namespace_row_count": len(rows),
        "complete_namespace_max_task_id": namespace_max_task_id,
        "complete_namespace_sha256": canonical_sha256(projection),
        "post_submit_namespace_row_count": (
            len(post_rows) if apply else None
        ),
        "post_submit_namespace_max_task_id": (
            post_namespace_max_task_id if apply else None
        ),
        "post_submit_namespace_sha256": (
            canonical_sha256(post_projection) if apply else None
        ),
        "post_submit_parent_match_count": (
            post_parent_match_count if apply else 0
        ),
        "post_submit_logical_match_count": (
            post_logical_match_count if apply else 0
        ),
        "all_dedupes_revalidated_after_post": bool(post_revalidated),
        "parent_dedupe_key": parent_dedupe,
        "logical_child_dedupe_keys": child_dedupes,
        "all_dedupes_absent_before_post": True,
        "task_id": task_id,
        "task_status": status,
        "submitted_parent_count": submitted_count,
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
            raise RuntimeError("shape canary must perform exactly one additive POST")
        shape1._atomic_json(receipt_out, value)  # type: ignore[arg-type]
    elif getattr(scheduler, "post_count", 0) != 0:
        raise RuntimeError("shape canary dry-run performed a Scheduler POST")
    return value


def _fault_occurrences(streams: Sequence[bytes]) -> int:
    # Never join stdout/stderr: an error token at the end of one stream must
    # not be associated with a benign SemLock statement in another stream.
    return sum(
        len(pattern.findall(stream.lower()))
        for stream in streams
        for pattern in _FAULT_PATTERNS
    )


def _receipt_cpu_summary(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available = all(
        receipt["child_resource_telemetry"].get("available") is True
        for receipt in receipts
    )
    child_wall = sum(
        float(receipt["child_resource_telemetry"]["wall_time_seconds"])
        for receipt in receipts
    )
    child_cpu = sum(
        float(receipt["child_resource_telemetry"]["process_tree_cpu_seconds"])
        for receipt in receipts
        if receipt["child_resource_telemetry"].get("process_tree_cpu_seconds")
        is not None
    )
    weighted = child_cpu / child_wall if available and child_wall > 0 else 0.0
    return {
        "available": available,
        "sum_child_wall_time_seconds": child_wall,
        "reported_child_process_tree_cpu_seconds": child_cpu,
        "weighted_average_cores_used_per_logical_seed": weighted,
    }


def _bind_terminal_identity(
    status: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    *,
    shape: int,
) -> dict[str, Any]:
    """Bind status, every child receipt, and parent RSS aggregation."""

    shape = _shape(shape)
    task_id = status.get("task_id")
    manifest_sha = status.get("manifest_sha256")
    if (
        not isinstance(task_id, str)
        or not task_id
        or not task_id.isdigit()
        or int(task_id) <= 0
        or not shape1._is_sha256(manifest_sha)
        or memory.get("task_id") != task_id
        or memory.get("manifest_sha256") != manifest_sha
        or memory.get("task_status_sha256") != status.get("status_sha256")
        or len(receipts) != shape
        or [receipt.get("ordinal") for receipt in receipts] != list(range(shape))
        or len({receipt.get("seed") for receipt in receipts}) != shape
        or any(
            receipt.get("task_id") != task_id
            or receipt.get("manifest_sha256") != manifest_sha
            for receipt in receipts
        )
    ):
        raise RuntimeError("Phase-B terminal status/receipt/memory identity drifted")
    seen_cpus: set[int] = set()
    for receipt in receipts:
        cpu_set = receipt.get("cpu_set")
        legacy = receipt.get("legacy_status")
        applied_cpu_set = (
            legacy.get("phase_b_cpu_set") if isinstance(legacy, Mapping) else None
        )
        if (
            not isinstance(cpu_set, list)
            or len(cpu_set) != CHILD_CPUS
            or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                for cpu in cpu_set
            )
            or len(set(cpu_set)) != CHILD_CPUS
            or seen_cpus.intersection(cpu_set)
            or applied_cpu_set != cpu_set
        ):
            raise RuntimeError(
                "Phase-B receipt/applied CPU-set identity drifted"
            )
        seen_cpus.update(cpu_set)
    observations = [
        {
            "ordinal": int(receipt["ordinal"]),
            "seed": int(receipt["seed"]),
            "legacy_status_sha256": receipt["legacy_status_sha256"],
            "observed_peak_rss_bytes": receipt["legacy_status"][
                "observed_peak_rss_bytes"
            ],
        }
        for receipt in receipts
    ]
    observations_sha = canonical_sha256(observations)
    if memory.get("child_peak_rss_observations_sha256") != observations_sha:
        raise RuntimeError("Phase-B parent memory evidence is not bound to receipts")
    return {
        "task_id": task_id,
        "manifest_sha256": manifest_sha,
        "child_peak_rss_observations_sha256": observations_sha,
    }


def evaluate_terminal_evidence(
    task_status: Mapping[str, Any],
    child_receipts: Sequence[Mapping[str, Any]],
    parent_memory_evidence: Mapping[str, Any],
    *,
    expected_shape: int,
    semlock_or_enospc_log_occurrences: int,
    shape4_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shape = _shape(expected_shape)
    if (
        isinstance(semlock_or_enospc_log_occurrences, bool)
        or not isinstance(semlock_or_enospc_log_occurrences, int)
        or semlock_or_enospc_log_occurrences < 0
    ):
        raise RuntimeError("invalid SemLock/ENOSPC log occurrence count")
    status = validate_task_status(task_status)
    receipts = [validate_child_receipt(value) for value in child_receipts]
    memory = validate_parent_memory_evidence(parent_memory_evidence)
    identity = _bind_terminal_identity(status, receipts, memory, shape=shape)
    cpu = shape1.evaluate_terminal_cpu_evidence(
        status, receipts, expected_shape=shape
    )
    receipt_cpu = _receipt_cpu_summary(receipts)
    parent_elapsed = float(cpu["parent_elapsed_seconds"])
    throughput = shape / parent_elapsed if parent_elapsed > 0 else 0.0
    baseline_throughput = None
    baseline_sha = None
    throughput_ratio = None
    baseline_terminal = None
    baseline_gate = shape != 8
    if shape == 8:
        if shape4_baseline is None:
            raise RuntimeError("shape8 terminal evidence requires sealed shape4 baseline")
        baseline = validate_terminal_evidence(shape4_baseline)
        if baseline["lane_shape"] != 4 or baseline["promotion_eligible"] is not True:
            raise RuntimeError("shape8 baseline is not an eligible shape4 terminal")
        baseline_throughput = float(baseline["throughput_logical_seeds_per_second"])
        baseline_sha = baseline["terminal_evidence_sha256"]
        baseline_terminal = baseline
        throughput_ratio = (
            throughput / baseline_throughput if baseline_throughput > 0 else 0.0
        )
        baseline_gate = throughput_ratio >= SHAPE8_MINIMUM_THROUGHPUT_RATIO
    receipt_rss = [
        int(receipt["legacy_status"]["observed_peak_rss_bytes"])
        for receipt in receipts
    ]
    total_rss = sum(receipt_rss)
    requested_memory = _parent_memory_bytes(shape)
    dispatch_fill = float(status["resource_telemetry"]["dispatch_fill_seconds"])
    maximum_dispatch_fill = (
        (shape - 1) * DEFAULT_MODEL_LOAD_STAGGER_SECONDS
    )
    memory_binding_ok = (
        memory.get("logical_child_count") == shape
        and memory.get("parent_requested_memory_bytes") == requested_memory
        and memory.get("child_peak_rss_available_count") == shape
        and memory.get("child_peak_rss_sum_bytes") == total_rss
        and memory.get("child_peak_rss_max_bytes") == max(receipt_rss)
    )
    gate_results = {
        "cpu_affinity_rss_dispatch_and_child_completion": (
            cpu["promotion_eligible"] is True
        ),
        "no_child_failure": (
            status.get("state") == "completed"
            and status.get("completed_child_count") == shape
            and status.get("failed_child_count") == 0
            and all(
                receipt.get("state") == "completed"
                and receipt.get("exit_code") == 0
                and receipt.get("failure") is None
                for receipt in receipts
            )
        ),
        "weighted_cores_per_seed_at_least_0p75": (
            float(receipt_cpu["weighted_average_cores_used_per_logical_seed"])
            >= MINIMUM_WEIGHTED_CORES_PER_SEED
        ),
        "parent_and_child_cpu_telemetry_reconciled": (
            receipt_cpu["available"] is True
            and math.isclose(
                float(receipt_cpu["sum_child_wall_time_seconds"]),
                float(cpu["sum_logical_seed_elapsed_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(receipt_cpu["reported_child_process_tree_cpu_seconds"]),
                float(cpu["reported_child_cpu_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ),
        "dispatch_fill_within_shape_gate": (
            dispatch_fill <= maximum_dispatch_fill
        ),
        "semlock_attested_and_logs_clean": (
            semlock_or_enospc_log_occurrences == 0
            and all(
                shape1._is_sha256(
                    receipt["legacy_status"].get("phase_b_semlock_stress_sha256")
                )
                for receipt in receipts
            )
        ),
        "memory_evidence_matches_child_receipts": memory_binding_ok,
        "total_child_peak_rss_within_parent_request": (
            total_rss <= requested_memory
            and memory.get("total_child_peak_rss_within_parent_request") is True
        ),
        "cgroup_peak_within_finite_limit": (
            memory.get("cgroup_available") is True
            and memory.get("cgroup_limit_unbounded") is False
            and memory.get("cgroup_peak_within_limit") is True
            and memory.get("cgroup_limit_covers_parent_request") is True
        ),
        "parent_memory_safety_passed": memory.get("safety_passed") is True,
        "throughput_at_least_1p7x_shape4": baseline_gate,
    }
    unsigned = {
        "schema_version": TERMINAL_EVIDENCE_SCHEMA,
        "lane_shape": shape,
        "logical_seed_count": shape,
        "task_id": identity["task_id"],
        "manifest_sha256": identity["manifest_sha256"],
        "task_status": status,
        "task_status_sha256": status["status_sha256"],
        "child_receipts": receipts,
        "child_receipt_sha256s": [receipt["receipt_sha256"] for receipt in receipts],
        "parent_memory_evidence_sha256": memory["evidence_sha256"],
        "parent_memory_evidence": memory,
        "terminal_cpu_evidence": cpu,
        "terminal_cpu_evidence_sha256": cpu["terminal_cpu_evidence_sha256"],
        "parent_elapsed_seconds": parent_elapsed,
        "throughput_logical_seeds_per_second": throughput,
        "shape4_baseline_terminal_evidence_sha256": baseline_sha,
        "shape4_baseline_terminal_evidence": baseline_terminal,
        "shape4_throughput_logical_seeds_per_second": baseline_throughput,
        "throughput_ratio_vs_shape4": throughput_ratio,
        "minimum_required_throughput_ratio_vs_shape4": (
            SHAPE8_MINIMUM_THROUGHPUT_RATIO if shape == 8 else None
        ),
        "weighted_average_cores_used_per_logical_seed": receipt_cpu[
            "weighted_average_cores_used_per_logical_seed"
        ],
        "receipt_sum_child_wall_time_seconds": receipt_cpu[
            "sum_child_wall_time_seconds"
        ],
        "receipt_reported_child_process_tree_cpu_seconds": receipt_cpu[
            "reported_child_process_tree_cpu_seconds"
        ],
        "semlock_or_enospc_log_occurrences": semlock_or_enospc_log_occurrences,
        "dispatch_fill_seconds": dispatch_fill,
        "maximum_dispatch_fill_seconds": maximum_dispatch_fill,
        "parent_requested_memory_bytes": requested_memory,
        "child_peak_rss_bytes": receipt_rss,
        "child_peak_rss_observations_sha256": identity[
            "child_peak_rss_observations_sha256"
        ],
        "total_child_peak_rss_bytes": total_rss,
        "maximum_child_peak_rss_bytes": max(receipt_rss),
        "cgroup_peak_bytes": memory.get("cgroup_peak_bytes"),
        "cgroup_limit_bytes": memory.get("cgroup_limit_bytes"),
        "gate_results": gate_results,
        "promotion_eligible": all(gate_results.values()),
        "next_shape_allowed": _next_shape(shape, all(gate_results.values())),
        "production_shape8_promotion_eligible": (
            shape == 8 and all(gate_results.values())
        ),
        "automatic_promotion_performed": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return {
        **unsigned,
        "terminal_evidence_sha256": canonical_sha256(unsigned),
    }


def validate_terminal_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "terminal_evidence_sha256"
    }
    required = {
        "schema_version",
        "lane_shape",
        "logical_seed_count",
        "task_id",
        "manifest_sha256",
        "task_status",
        "task_status_sha256",
        "child_receipts",
        "child_receipt_sha256s",
        "parent_memory_evidence_sha256",
        "parent_memory_evidence",
        "terminal_cpu_evidence",
        "terminal_cpu_evidence_sha256",
        "parent_elapsed_seconds",
        "throughput_logical_seeds_per_second",
        "shape4_baseline_terminal_evidence_sha256",
        "shape4_baseline_terminal_evidence",
        "shape4_throughput_logical_seeds_per_second",
        "throughput_ratio_vs_shape4",
        "minimum_required_throughput_ratio_vs_shape4",
        "weighted_average_cores_used_per_logical_seed",
        "receipt_sum_child_wall_time_seconds",
        "receipt_reported_child_process_tree_cpu_seconds",
        "semlock_or_enospc_log_occurrences",
        "dispatch_fill_seconds",
        "maximum_dispatch_fill_seconds",
        "parent_requested_memory_bytes",
        "child_peak_rss_bytes",
        "child_peak_rss_observations_sha256",
        "total_child_peak_rss_bytes",
        "maximum_child_peak_rss_bytes",
        "cgroup_peak_bytes",
        "cgroup_limit_bytes",
        "gate_results",
        "promotion_eligible",
        "next_shape_allowed",
        "production_shape8_promotion_eligible",
        "automatic_promotion_performed",
        "scheduler_write_performed",
        "fea_submission_performed",
        "aedt_used",
        "terminal_evidence_sha256",
    }
    shape = _shape(value.get("lane_shape"))
    task_status = value.get("task_status")
    child_receipts = value.get("child_receipts")
    if not isinstance(task_status, Mapping) or not isinstance(child_receipts, list):
        raise RuntimeError("Phase-B shape terminal lacks bound runtime evidence")
    status = validate_task_status(task_status)
    receipts = [validate_child_receipt(receipt) for receipt in child_receipts]
    cpu = value.get("terminal_cpu_evidence")
    if not isinstance(cpu, Mapping):
        raise RuntimeError("Phase-B shape terminal lacks CPU evidence")
    sealed_cpu = shape1.validate_terminal_cpu_evidence(cpu)
    memory = value.get("parent_memory_evidence")
    if not isinstance(memory, Mapping):
        raise RuntimeError("Phase-B shape terminal lacks parent memory evidence")
    sealed_memory = validate_parent_memory_evidence(memory)
    identity = _bind_terminal_identity(status, receipts, sealed_memory, shape=shape)
    recomputed_cpu = shape1.evaluate_terminal_cpu_evidence(
        status, receipts, expected_shape=shape
    )
    gates = value.get("gate_results")
    parent_elapsed = value.get("parent_elapsed_seconds")
    throughput = value.get("throughput_logical_seeds_per_second")
    baseline_throughput = value.get("shape4_throughput_logical_seeds_per_second")
    throughput_ratio = value.get("throughput_ratio_vs_shape4")
    child_rss = value.get("child_peak_rss_bytes")
    receipt_cpu = _receipt_cpu_summary(receipts)
    parent_request = value.get("parent_requested_memory_bytes")
    semlock_occurrences = value.get("semlock_or_enospc_log_occurrences")
    dispatch_fill = value.get("dispatch_fill_seconds")
    maximum_dispatch_fill = value.get("maximum_dispatch_fill_seconds")
    if (
        isinstance(parent_elapsed, bool)
        or not isinstance(parent_elapsed, (int, float))
        or not math.isfinite(float(parent_elapsed))
        or float(parent_elapsed) <= 0
        or isinstance(throughput, bool)
        or not isinstance(throughput, (int, float))
        or not math.isfinite(float(throughput))
        or isinstance(parent_request, bool)
        or not isinstance(parent_request, int)
        or parent_request <= 0
        or isinstance(semlock_occurrences, bool)
        or not isinstance(semlock_occurrences, int)
        or semlock_occurrences < 0
        or isinstance(dispatch_fill, bool)
        or not isinstance(dispatch_fill, (int, float))
        or not math.isfinite(float(dispatch_fill))
        or float(dispatch_fill) < 0
        or isinstance(maximum_dispatch_fill, bool)
        or not isinstance(maximum_dispatch_fill, (int, float))
        or not math.isfinite(float(maximum_dispatch_fill))
        or float(maximum_dispatch_fill) < 0
        or not isinstance(child_rss, list)
        or len(child_rss) != shape
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in child_rss
        )
    ):
        raise RuntimeError("Phase-B shape terminal numeric evidence drifted")
    expected_throughput = shape / float(parent_elapsed)
    baseline_evidence = value.get("shape4_baseline_terminal_evidence")
    sealed_baseline = None
    if shape == 8:
        if (
            not isinstance(baseline_evidence, Mapping)
            or
            isinstance(baseline_throughput, bool)
            or not isinstance(baseline_throughput, (int, float))
            or not math.isfinite(float(baseline_throughput))
            or float(baseline_throughput) <= 0
            or isinstance(throughput_ratio, bool)
            or not isinstance(throughput_ratio, (int, float))
            or not math.isfinite(float(throughput_ratio))
        ):
            raise RuntimeError("Phase-B shape8 throughput baseline drifted")
        sealed_baseline = validate_terminal_evidence(baseline_evidence)
        if (
            sealed_baseline["lane_shape"] != 4
            or sealed_baseline["promotion_eligible"] is not True
        ):
            raise RuntimeError("Phase-B shape8 baseline is not promotion eligible")
        expected_ratio = expected_throughput / float(baseline_throughput)
    else:
        expected_ratio = None
    expected_gates = {
        "cpu_affinity_rss_dispatch_and_child_completion": (
            recomputed_cpu["promotion_eligible"] is True
        ),
        "no_child_failure": (
            status.get("state") == "completed"
            and status.get("completed_child_count") == shape
            and status.get("failed_child_count") == 0
            and all(
                receipt.get("state") == "completed"
                and receipt.get("exit_code") == 0
                and receipt.get("failure") is None
                for receipt in receipts
            )
        ),
        "weighted_cores_per_seed_at_least_0p75": (
            float(receipt_cpu["weighted_average_cores_used_per_logical_seed"])
            >= MINIMUM_WEIGHTED_CORES_PER_SEED
        ),
        "parent_and_child_cpu_telemetry_reconciled": (
            receipt_cpu["available"] is True
            and math.isclose(
                float(receipt_cpu["sum_child_wall_time_seconds"]),
                float(recomputed_cpu["sum_logical_seed_elapsed_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(receipt_cpu["reported_child_process_tree_cpu_seconds"]),
                float(recomputed_cpu["reported_child_cpu_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ),
        "dispatch_fill_within_shape_gate": (
            float(dispatch_fill) <= float(maximum_dispatch_fill)
        ),
        "semlock_attested_and_logs_clean": (
            semlock_occurrences == 0
            and recomputed_cpu["gate_results"].get("semlock_stress_attested")
            is True
        ),
        "memory_evidence_matches_child_receipts": (
            sealed_memory["logical_child_count"] == shape
            and sealed_memory["parent_requested_memory_bytes"]
            == value.get("parent_requested_memory_bytes")
            and sealed_memory["child_peak_rss_available_count"] == shape
            and sealed_memory["child_peak_rss_sum_bytes"] == sum(child_rss)
            and sealed_memory["child_peak_rss_max_bytes"] == max(child_rss)
        ),
        "total_child_peak_rss_within_parent_request": (
            sum(child_rss) <= parent_request
            and sealed_memory["total_child_peak_rss_within_parent_request"] is True
        ),
        "cgroup_peak_within_finite_limit": (
            sealed_memory["cgroup_available"] is True
            and sealed_memory["cgroup_limit_unbounded"] is False
            and sealed_memory["cgroup_peak_within_limit"] is True
            and sealed_memory["cgroup_limit_covers_parent_request"] is True
        ),
        "parent_memory_safety_passed": sealed_memory["safety_passed"] is True,
        "throughput_at_least_1p7x_shape4": (
            shape != 8 or expected_ratio >= SHAPE8_MINIMUM_THROUGHPUT_RATIO
        ),
    }
    if (
        set(value) != required
        or value.get("schema_version") != TERMINAL_EVIDENCE_SCHEMA
        or value.get("logical_seed_count") != shape
        or value.get("task_id") != identity["task_id"]
        or value.get("manifest_sha256") != identity["manifest_sha256"]
        or value.get("task_status_sha256") != status["status_sha256"]
        or value.get("child_receipt_sha256s")
        != [receipt["receipt_sha256"] for receipt in receipts]
        or value.get("terminal_evidence_sha256") != canonical_sha256(unsigned)
        or sealed_cpu != recomputed_cpu
        or value.get("terminal_cpu_evidence_sha256")
        != sealed_cpu["terminal_cpu_evidence_sha256"]
        or sealed_cpu["expected_shape"] != shape
        or value.get("parent_memory_evidence_sha256")
        != sealed_memory["evidence_sha256"]
        or float(parent_elapsed) != float(sealed_cpu["parent_elapsed_seconds"])
        or not abs(float(throughput) - expected_throughput) <= 1e-15
        or value.get("weighted_average_cores_used_per_logical_seed")
        != receipt_cpu["weighted_average_cores_used_per_logical_seed"]
        or value.get("receipt_sum_child_wall_time_seconds")
        != receipt_cpu["sum_child_wall_time_seconds"]
        or value.get("receipt_reported_child_process_tree_cpu_seconds")
        != receipt_cpu["reported_child_process_tree_cpu_seconds"]
        or float(dispatch_fill)
        != float(status["resource_telemetry"]["dispatch_fill_seconds"])
        or float(maximum_dispatch_fill)
        != (shape - 1) * DEFAULT_MODEL_LOAD_STAGGER_SECONDS
        or parent_request != _parent_memory_bytes(shape)
        or value.get("child_peak_rss_observations_sha256")
        != identity["child_peak_rss_observations_sha256"]
        or value.get("total_child_peak_rss_bytes") != sum(child_rss)
        or value.get("maximum_child_peak_rss_bytes") != max(child_rss)
        or value.get("cgroup_peak_bytes") != sealed_memory["cgroup_peak_bytes"]
        or value.get("cgroup_limit_bytes") != sealed_memory["cgroup_limit_bytes"]
        or not isinstance(gates, dict)
        or gates != expected_gates
        or value.get("promotion_eligible") is not all(gates.values())
        or value.get("next_shape_allowed")
        != _next_shape(shape, value.get("promotion_eligible") is True)
        or value.get("production_shape8_promotion_eligible")
        is not (shape == 8 and value.get("promotion_eligible") is True)
        or (
            shape in {1, 4}
            and any(
                value.get(field) is not None
                for field in (
                    "shape4_baseline_terminal_evidence_sha256",
                    "shape4_baseline_terminal_evidence",
                    "shape4_throughput_logical_seeds_per_second",
                    "throughput_ratio_vs_shape4",
                    "minimum_required_throughput_ratio_vs_shape4",
                )
            )
        )
        or (
            shape == 8
            and (
                not shape1._is_sha256(
                    value.get("shape4_baseline_terminal_evidence_sha256")
                )
                or value.get("shape4_baseline_terminal_evidence_sha256")
                != sealed_baseline["terminal_evidence_sha256"]
                or float(baseline_throughput)
                != float(
                    sealed_baseline["throughput_logical_seeds_per_second"]
                )
                or value.get("minimum_required_throughput_ratio_vs_shape4")
                != SHAPE8_MINIMUM_THROUGHPUT_RATIO
                or not abs(float(throughput_ratio) - expected_ratio) <= 1e-15
            )
        )
        or any(
            value.get(field) is not False
            for field in (
                "automatic_promotion_performed",
                "scheduler_write_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
    ):
        raise RuntimeError("Phase-B shape terminal evidence seal mismatch")
    return copy.deepcopy(dict(value))


def _json_remote(value: bytes, label: str) -> dict[str, Any]:
    return shape1._json_bytes_object(value, label)


def validate_remote_terminal(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "remote_terminal_sha256"
    }
    required = {
        "schema_version",
        "observed_at",
        "package_sha256",
        "submission_receipt_sha256",
        "bundle_id",
        "lane_shape",
        "logical_seed_count",
        "task_id",
        "scheduler_status",
        "scheduler_detail_get_count",
        "scheduler_remote_file_get_count",
        "remote_files",
        "batch_manifest",
        "batch_manifest_sha256",
        "task_status_sha256",
        "parent_memory_evidence_sha256",
        "child_receipt_sha256s",
        "shape4_baseline_remote_terminal_sha256",
        "terminal_evidence",
        "terminal_evidence_sha256",
        "promotion_eligible",
        "next_shape_allowed",
        "production_shape8_promotion_eligible",
        "scheduler_access",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "remote_access",
        "remote_write_count",
        "automatic_promotion_performed",
        "fea_submission_performed",
        "aedt_used",
        "remote_terminal_sha256",
    }
    terminal = value.get("terminal_evidence")
    if not isinstance(terminal, Mapping):
        raise RuntimeError("shape remote terminal lacks terminal evidence")
    sealed = validate_terminal_evidence(terminal)
    shape = int(sealed["lane_shape"])
    manifest_value = value.get("batch_manifest")
    if not isinstance(manifest_value, Mapping):
        raise RuntimeError("shape remote terminal lacks batch manifest")
    manifest = validate_batch_manifest(manifest_value)
    task_id = value.get("task_id")
    files = value.get("remote_files")
    receipts = sealed["child_receipts"]
    # Standalone receipt seals are insufficient here: bind every receipt back
    # to the fetched immutable manifest so ordinal/seed/payload/dedupe cannot
    # be changed and merely re-sealed as a self-consistent terminal artifact.
    manifest_bound_receipts = [
        validate_child_receipt(receipt, manifest=manifest)
        for receipt in receipts
    ]
    if manifest_bound_receipts != receipts:
        raise RuntimeError("shape child receipt/manifest binding drifted")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or not isinstance(files, dict)
    ):
        raise RuntimeError("shape remote terminal identity/files drifted")
    task_root = f"runs/task-{task_id}"
    expected_paths = {
        "batch_manifest": f"{task_root}/batch_manifest.json",
        "task_status": f"{task_root}/task_status.json",
        "parent_memory_evidence": (
            f"{task_root}/{PARENT_MEMORY_EVIDENCE_FILENAME}"
        ),
    }
    for ordinal, receipt in enumerate(receipts):
        seed = int(receipt["seed"])
        expected_paths.update(
            {
                f"child_{ordinal}_receipt": (
                    f"{task_root}/seed-{seed}/seed_status.json"
                ),
                f"child_{ordinal}_stdout": (
                    f"{task_root}/{receipt['stdout_relative_path']}"
                ),
                f"child_{ordinal}_stderr": (
                    f"{task_root}/{receipt['stderr_relative_path']}"
                ),
                f"child_{ordinal}_result": (
                    f"{task_root}/seed-{seed}/result.json"
                ),
            }
        )
    if set(files) != set(expected_paths):
        raise RuntimeError("shape remote terminal file inventory drifted")
    fetched_file_count = 0
    for label, expected_path in expected_paths.items():
        record = files.get(label)
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "size",
            "sha256",
        }:
            raise RuntimeError("shape remote terminal file record drifted")
        size = record.get("size")
        sha = record.get("sha256")
        if record.get("relative_path") != expected_path:
            raise RuntimeError("shape remote terminal file path drifted")
        ordinal = None
        stream_name = None
        match = re.fullmatch(r"child_(\d+)_(stdout|stderr|result)", label)
        if match is not None:
            ordinal = int(match.group(1))
            stream_name = match.group(2)
        expected_sha = None
        expected_size = None
        if ordinal is not None and stream_name in {"stdout", "stderr"}:
            receipt = receipts[ordinal]
            expected_sha = receipt[f"{stream_name}_sha256"]
            expected_size = receipt[f"{stream_name}_size_bytes"]
        elif ordinal is not None and stream_name == "result":
            expected_sha = receipts[ordinal].get("result_sha256")
        if expected_sha is None and stream_name == "result":
            if size is not None or sha is not None:
                raise RuntimeError("absent child result has a remote file seal")
            continue
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not shape1._is_sha256(sha)
            or (expected_sha is not None and sha != expected_sha)
            or (expected_size is not None and size != expected_size)
        ):
            raise RuntimeError("shape remote terminal file SHA/size drifted")
        fetched_file_count += 1
    if (
        set(value) != required
        or value.get("schema_version") != REMOTE_TERMINAL_SCHEMA
        or value.get("remote_terminal_sha256") != canonical_sha256(unsigned)
        or not str(value.get("observed_at") or "")
        or not shape1._is_sha256(value.get("package_sha256"))
        or not shape1._is_sha256(value.get("submission_receipt_sha256"))
        or not str(value.get("bundle_id") or "").startswith("current7-")
        or value.get("lane_shape") != shape
        or value.get("logical_seed_count") != shape
        or value.get("task_id") != int(sealed["task_id"])
        or value.get("scheduler_status")
        not in {"completed", "failed", "cancelled", "timeout", "timed_out"}
        or value.get("scheduler_detail_get_count") != 2
        or value.get("scheduler_remote_file_get_count") != fetched_file_count
        or manifest["manifest_sha256"] != sealed["manifest_sha256"]
        or value.get("batch_manifest_sha256") != manifest["manifest_sha256"]
        or value.get("task_status_sha256") != sealed["task_status_sha256"]
        or value.get("parent_memory_evidence_sha256")
        != sealed["parent_memory_evidence_sha256"]
        or value.get("child_receipt_sha256s")
        != sealed["child_receipt_sha256s"]
        or value.get("terminal_evidence_sha256")
        != sealed["terminal_evidence_sha256"]
        or value.get("promotion_eligible") is not (
            value.get("scheduler_status") == "completed"
            and sealed["promotion_eligible"] is True
        )
        or value.get("next_shape_allowed")
        != _next_shape(shape, value.get("promotion_eligible") is True)
        or value.get("production_shape8_promotion_eligible")
        is not (
            shape == 8
            and value.get("scheduler_status") == "completed"
            and sealed["promotion_eligible"] is True
        )
        or value.get("scheduler_access") != "GET-only"
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("remote_access") != "read-only"
        or value.get("remote_write_count") != 0
        or value.get("automatic_promotion_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or (
            shape in {1, 4}
            and value.get("shape4_baseline_remote_terminal_sha256") is not None
        )
        or (
            shape == 8
            and not shape1._is_sha256(
                value.get("shape4_baseline_remote_terminal_sha256")
            )
        )
    ):
        raise RuntimeError("Phase-B shape remote terminal seal mismatch")
    return copy.deepcopy(dict(value))


def evaluate_remote_terminal(
    *,
    package_path: Path,
    submission_receipt_path: Path,
    scheduler_url: str,
    task_id: int,
    output_path: Path,
    shape4_baseline_path: Path | None = None,
) -> dict[str, Any]:
    package = validate_package(shape1._read_json(package_path, "shape package"))
    submission = _validate_submission_receipt(
        shape1._read_json(submission_receipt_path, "shape submission receipt"),
        package=package,
    )
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
        or submission.get("task_id") != task_id
    ):
        raise RuntimeError("terminal evaluator task id/submission mismatch")
    shape = int(package["lane_shape"])
    shape4_remote = None
    shape4_terminal = None
    if shape == 8:
        if shape4_baseline_path is None:
            raise RuntimeError("shape8 remote evaluation requires shape4 baseline")
        shape4_remote = validate_remote_terminal(
            shape1._read_json(shape4_baseline_path, "shape4 remote terminal")
        )
        if (
            shape4_remote["lane_shape"] != 4
            or shape4_remote["bundle_id"] != package["bundle_id"]
            or shape4_remote["promotion_eligible"] is not True
            or shape4_remote["remote_terminal_sha256"]
            != package["required_shape4_remote_terminal_sha256"]
        ):
            raise RuntimeError("shape8 package/baseline lineage mismatch")
        shape4_terminal = shape4_remote["terminal_evidence"]
    elif shape4_baseline_path is not None:
        raise RuntimeError("shape4 remote evaluation cannot accept a shape4 baseline")

    scheduler = SchedulerApiClient(scheduler_url)
    detail = scheduler.get_task(task_id)
    observed_id, scheduler_status = shape1._authenticate_task(
        detail, package["parent_task"], label="terminal Scheduler task detail"
    )
    if observed_id != task_id:
        raise RuntimeError("terminal Scheduler detail task id mismatch")
    if scheduler_status not in {
        "completed",
        "failed",
        "cancelled",
        "timeout",
        "timed_out",
    }:
        raise RuntimeError(f"shape task {task_id} is not terminal: {scheduler_status}")

    task_root = f"runs/task-{task_id}"
    relative_paths: dict[str, str] = {
        "batch_manifest": f"{task_root}/batch_manifest.json",
        "task_status": f"{task_root}/task_status.json",
        "parent_memory_evidence": f"{task_root}/{PARENT_MEMORY_EVIDENCE_FILENAME}",
    }
    raw: dict[str, bytes] = {
        label: shape1._remote_file_bytes(
            scheduler_url=scheduler_url,
            task_id=task_id,
            relative_path=relative,
        )
        for label, relative in relative_paths.items()
    }
    manifest = validate_batch_manifest(
        _json_remote(raw["batch_manifest"], "batch manifest")
    )
    expected_manifest = batch_manifest_from_payload(package["parent_task"]["payload_json"])
    if manifest != expected_manifest:
        raise RuntimeError("remote terminal manifest differs from submitted parent")
    task_status = validate_task_status(
        _json_remote(raw["task_status"], "task status")
    )
    memory = validate_parent_memory_evidence(
        _json_remote(raw["parent_memory_evidence"], "parent memory evidence")
    )
    receipts = []
    stream_bytes: list[bytes] = []
    result_count = 0
    for ordinal, seed in enumerate(package["seeds"]):
        receipt_label = f"child_{ordinal}_receipt"
        receipt_relative = f"{task_root}/seed-{seed}/seed_status.json"
        relative_paths[receipt_label] = receipt_relative
        receipt_raw = shape1._remote_file_bytes(
            scheduler_url=scheduler_url,
            task_id=task_id,
            relative_path=receipt_relative,
        )
        raw[receipt_label] = receipt_raw
        receipt = validate_child_receipt(
            _json_remote(receipt_raw, f"child {ordinal} receipt"), manifest=manifest
        )
        if receipt["ordinal"] != ordinal or receipt["seed"] != seed:
            raise RuntimeError("remote child receipt order/seed drifted")
        for stream_name in ("stdout", "stderr"):
            label = f"child_{ordinal}_{stream_name}"
            relative = f"{task_root}/{receipt[f'{stream_name}_relative_path']}"
            relative_paths[label] = relative
            stream = shape1._remote_file_bytes(
                scheduler_url=scheduler_url,
                task_id=task_id,
                relative_path=relative,
            )
            if (
                len(stream) != int(receipt[f"{stream_name}_size_bytes"])
                or hashlib.sha256(stream).hexdigest()
                != receipt[f"{stream_name}_sha256"]
            ):
                raise RuntimeError(
                    f"remote child {ordinal} {stream_name} differs from receipt seal"
                )
            relative_paths[label] = relative
            raw[label] = stream
            stream_bytes.append(stream)
        result_label = f"child_{ordinal}_result"
        result_relative = f"{task_root}/seed-{seed}/result.json"
        relative_paths[result_label] = result_relative
        result_sha = receipt.get("result_sha256")
        result_raw = None
        if result_sha is not None:
            result_raw = shape1._remote_file_bytes(
                scheduler_url=scheduler_url,
                task_id=task_id,
                relative_path=result_relative,
            )
            if hashlib.sha256(result_raw).hexdigest() != result_sha:
                raise RuntimeError(
                    f"remote child {ordinal} result differs from receipt SHA"
                )
            raw[result_label] = result_raw
            result_count += 1
        receipts.append(receipt)

    fault_count = _fault_occurrences(stream_bytes)
    terminal = evaluate_terminal_evidence(
        task_status,
        receipts,
        memory,
        expected_shape=shape,
        semlock_or_enospc_log_occurrences=fault_count,
        shape4_baseline=shape4_terminal,
    )
    final_detail = scheduler.get_task(task_id)
    final_id, final_status = shape1._authenticate_task(
        final_detail,
        package["parent_task"],
        label="post-evidence Scheduler task detail",
    )
    if final_id != task_id or final_status != scheduler_status:
        raise RuntimeError("Scheduler identity/state changed during evidence read")
    file_records = {
        label: {
            "relative_path": relative_paths[label],
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for label, value in raw.items()
    }
    for ordinal, receipt in enumerate(receipts):
        if receipt.get("result_sha256") is None:
            label = f"child_{ordinal}_result"
            file_records[label] = {
                "relative_path": relative_paths[label],
                "size": None,
                "sha256": None,
            }
    promotion = scheduler_status == "completed" and terminal["promotion_eligible"] is True
    unsigned = {
        "schema_version": REMOTE_TERMINAL_SCHEMA,
        "observed_at": shape1._now(),
        "package_sha256": package["package_sha256"],
        "submission_receipt_sha256": submission["receipt_sha256"],
        "bundle_id": package["bundle_id"],
        "lane_shape": shape,
        "logical_seed_count": shape,
        "task_id": task_id,
        "scheduler_status": scheduler_status,
        "scheduler_detail_get_count": 2,
        "scheduler_remote_file_get_count": 3 + 3 * shape + result_count,
        "remote_files": file_records,
        "batch_manifest": manifest,
        "batch_manifest_sha256": manifest["manifest_sha256"],
        "task_status_sha256": task_status["status_sha256"],
        "parent_memory_evidence_sha256": memory["evidence_sha256"],
        "child_receipt_sha256s": [receipt["receipt_sha256"] for receipt in receipts],
        "shape4_baseline_remote_terminal_sha256": (
            shape4_remote["remote_terminal_sha256"]
            if shape4_remote is not None
            else None
        ),
        "terminal_evidence": terminal,
        "terminal_evidence_sha256": terminal["terminal_evidence_sha256"],
        "promotion_eligible": promotion,
        "next_shape_allowed": _next_shape(shape, promotion),
        "production_shape8_promotion_eligible": shape == 8 and promotion,
        "scheduler_access": "GET-only",
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "remote_access": "read-only",
        "remote_write_count": 0,
        "automatic_promotion_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "remote_terminal_sha256": canonical_sha256(unsigned)}
    value = validate_remote_terminal(value)
    shape1._atomic_json(output_path, value)
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
    package = validate_package(shape1._read_json(package_path, "shape package"))
    if package != rebuilt:
        raise RuntimeError("shape package no longer reproduces from its sources")
    publication = shape1._read_json(
        publication_receipt_path, "publication receipt"
    )
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
        value = _submission_outcome(
            package=package,
            publication=publication,
            transport=transport,
            scheduler=SchedulerApiClient(scheduler_url),
            apply=apply,
            receipt_out=receipt_out,
        )
    if not apply and dry_run_out is not None:
        shape1._atomic_json(dry_run_out, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    render = sub.add_parser("render", help="render one immutable shape package")
    submit_parser = sub.add_parser(
        "submit", help="complete-namespace dry-run; POST only with --apply"
    )
    evaluate = sub.add_parser(
        "evaluate", help="seal local terminal CPU/RSS/cgroup evidence"
    )
    evaluate_remote = sub.add_parser(
        "evaluate-remote", help="GET-only remote terminal evidence evaluator"
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
    evaluate.add_argument("--child-receipt", type=Path, action="append", required=True)
    evaluate.add_argument("--parent-memory-evidence", type=Path, required=True)
    evaluate.add_argument("--shape", type=int, choices=(1, 4, 8), required=True)
    evaluate.add_argument("--semlock-or-enospc-log-occurrences", type=int, default=0)
    evaluate.add_argument("--shape4-baseline", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate_remote.add_argument("--package", type=Path, required=True)
    evaluate_remote.add_argument("--submission-receipt", type=Path, required=True)
    evaluate_remote.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    evaluate_remote.add_argument("--task-id", type=int, required=True)
    evaluate_remote.add_argument("--shape4-baseline", type=Path)
    evaluate_remote.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.operation == "render":
        value = build_package(
            config_path=args.config,
            offload_plan_path=args.offload_plan,
            publication_receipt_path=args.publication_receipt,
        )
        shape1._atomic_json(args.out, value)
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
    elif args.operation == "evaluate":
        baseline = (
            validate_terminal_evidence(
                shape1._read_json(args.shape4_baseline, "shape4 baseline")
            )
            if args.shape4_baseline is not None
            else None
        )
        value = evaluate_terminal_evidence(
            shape1._read_json(args.task_status, "Phase-B task status"),
            [
                shape1._read_json(path, f"Phase-B child receipt {index}")
                for index, path in enumerate(args.child_receipt)
            ],
            shape1._read_json(
                args.parent_memory_evidence, "parent memory evidence"
            ),
            expected_shape=args.shape,
            semlock_or_enospc_log_occurrences=(
                args.semlock_or_enospc_log_occurrences
            ),
            shape4_baseline=baseline,
        )
        value = validate_terminal_evidence(value)
        shape1._atomic_json(args.out, value)
    else:
        value = evaluate_remote_terminal(
            package_path=args.package,
            submission_receipt_path=args.submission_receipt,
            scheduler_url=args.scheduler_url,
            task_id=args.task_id,
            output_path=args.out,
            shape4_baseline_path=args.shape4_baseline,
        )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
