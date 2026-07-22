"""Seal the task-87052 2/3/4-CPU paired resource comparator.

This module is intentionally narrow.  The 4-CPU anchor is the exact Phase-B
generic task 87052 and its one logical child (seed 2257498123).  Rendering
creates two direct single-seed companions whose science payload differs from
the anchor child only at ``inference_threads`` and ``scheduler_cpus``.  The
Scheduler envelope additionally changes only the execution command, CPU
request, name, and dedupe identity required to isolate each companion.

``submit`` is dry-run by default.  ``submit --apply`` requires a separately
sealed independent approval and a completed, promotion-eligible remote
terminal attestation for task 87052.  A client instance can issue at most one
additive POST for one selected companion and has no cancellation, preemption,
allocation, publication, FEA, or AEDT surface.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

try:
    import tier1_resource2_cgroup_memory as cgroup_memory
    import tier1_final1000_phase_b_shape_canary as shape_canary
    import tier1_final1000_resource2_canary as resource2
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import sha256_file
    from tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tier1_final1000_multiseed_phase_b_contract import (
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_child_receipt,
        validate_parent_memory_evidence,
        validate_task_status,
    )
    from tier1_final1000_slurm_launch import (
        DEFAULT_PEAK_RSS_GATE_BYTES,
        REQUIRED_SCHEDULER_FIELDS,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_resource2_cgroup_memory as cgroup_memory
    from tools import tier1_final1000_phase_b_shape_canary as shape_canary
    from tools import tier1_final1000_resource2_canary as resource2
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import sha256_file
    from tools.tier1_corrected_current7_slurm_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        scheduler_publication_transport,
    )
    from tools.tier1_final1000_multiseed_contract import (
        scheduler_task_identity_matches,
        scheduler_task_observation,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        batch_manifest_from_payload,
        validate_batch_manifest,
        validate_child_receipt,
        validate_parent_memory_evidence,
        validate_task_status,
    )
    from tools.tier1_final1000_slurm_launch import (
        DEFAULT_PEAK_RSS_GATE_BYTES,
        REQUIRED_SCHEDULER_FIELDS,
    )


CONFIG_SCHEMA = "mft-tier1-final1000-paired-resource-comparator-config-v1"
PACKAGE_SCHEMA = "mft-tier1-final1000-paired-resource-comparator-package-v1"
APPROVAL_SCHEMA = "mft-tier1-final1000-paired-resource-comparator-approval-v1"
SUBMISSION_SCHEMA = "mft-tier1-final1000-paired-resource-comparator-submission-v1"
TELEMETRY_SCHEMA = "mft-tier1-final1000-paired-resource-comparator-telemetry-v1"
COMPARISON_SCHEMA = "mft-tier1-final1000-paired-resource-comparison-v1"
CAPACITY_SCHEMA = "mft-tier1-final1000-paired-resource-capacity-v1"

ANCHOR_TASK_ID = 87_052
ANCHOR_SEED = 2_257_498_123
ANCHOR_CPUS = 4
COMPANION_CPUS = (2, 3)
ALL_CPU_SHAPES = (2, 3, 4)
MEMORY_MB = 28_672
MAX_WORKERS_PER_NODE = 32
STAGE_ID = "entry-1200-t125"
BUNDLE_ID = "current7-c2d90bf33d513c607f8f"
BUNDLE_MANIFEST_SHA256 = (
    "86bf540b8ac9fb4a86fecaf20064cd97f464d682b3f58ffd0f23ba85e45bbcd3"
)
BUNDLE_CONTRACT_SHA256 = (
    "c2d90bf33d513c607f8f7cea15c0b85bdeb7d1d640000f85e84b51f6d5808a02"
)
OFFLOAD_PLAN_FILE_SHA256 = (
    "84777d678736411db9179f921adda498e01f5888d1f2ba52864c7ba5ade3b4b2"
)
SOURCE_CONFIG_SHA256 = (
    "c7c51dd8f0767ef48a38158ad7276e9de4950e50969e12874b226425bc425ba0"
)
SOURCE_PACKAGE_SHA256 = (
    "2eb0618a0e12476cacee70c6966ce93cf5a90168e0e88893e2d82f9ac6a5dc2d"
)
SOURCE_SUBMISSION_RECEIPT_SHA256 = (
    "544e88f29a8618be4054cf54bd7b2ec5ef84780b6d0d04bbe28bb2da1c295464"
)
SOURCE_CONFIG_FILE_SHA256 = (
    "8d54a56059540206509b6460fab695b04cfe5f469383460f88019baa414c20fe"
)
SOURCE_PACKAGE_FILE_SHA256 = (
    "f2438490c00530131a73939be5cbfd2d831be04455d13e69f829630ea03180e4"
)
SOURCE_SUBMISSION_FILE_SHA256 = (
    "e61f16439c98f0f91627565e12996c36bb1600cbbde38d0bdd7038b846cc7a80"
)
SOURCE_PARENT_DEDUPE = (
    "mft-tier1-final1000:concurrent-lane:"
    "b8a3be8be8316aebf283a3a6a0f5ce1e38fc797027250b972215dadbcb9f37ad"
)
SOURCE_CHILD_DEDUPE = (
    "mft-tier1-final1000:"
    "b5aeb7c2a00a6770806e030a8825e55c3814a1a12e3ead96c468ad8c669c0757"
)
SOURCE_CHILD_PAYLOAD_SHA256 = (
    "6fb21e402dfac0e946748a0d3850c510a7e1d13f64733b0b3b5924bf9019003f"
)
SOURCE_CHILD_TASK_SHA256 = (
    "95f458f11a7cd2063908e304bc3fe04a182d5dff8fdfb2fe6a9aac4e9f66794c"
)
SOURCE_PARENT_TASK_SHA256 = (
    "69263c2ad94dfd32d212c25c7072c7b9c3a477facc753310f75271a41fb123bf"
)
PRIOR_UNPAIRED_TERMINAL_FILE_SHA256 = (
    "50da2ea68201fab06e8ff19f5261fde09645e1f3014be1eae5e884f4f280de45"
)
PRIOR_UNPAIRED_REMOTE_TERMINAL_SHA256 = (
    "5ca1c0cd9b32b39e7425c42c589b882788aaae8f1dd036eb76da9dbb91253217"
)
COMPARATOR_PREFIX = "mft-t1rc-87052-"
COMPARATOR_DEDUPE_PREFIX = "mft-tier1-resource-comparator:87052:"
PRODUCTION_PREFIX = "mft-t1fg-"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
TELEMETRY_FILENAME = "paired_resource_telemetry.json"
PAGE_SIZE = 10_000
MAX_PAGES = 10_000
SCHEDULER_RESPONSE_LIMIT = 64 * 1024**2
POST_RESPONSE_LIMIT = 1024**2
RETRYABLE_HTTP = (429, 503)
RETRY_BACKOFF_SECONDS = (0.25, 0.5, 1.0)

# These keys describe the permitted resource binding, not optimizer science.
# All other result fields, including every persisted artifact fingerprint,
# remain in the cross-resource science fingerprint.
SCIENCE_RESULT_RESOURCE_KEYS = frozenset(
    {
        "payload_sha256",
        "optimizer_pid",
        "inference_threads",
        "scheduler_cpus",
        "optimizer_inference_threads",
        "threads_per_model",
        "family_threads",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable or invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_bytes(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _file_sha(path: Path) -> str:
    return sha256_file(path.resolve(strict=True))


def _thread_environment(cpus: int) -> dict[str, str]:
    return {name: str(cpus) for name in THREAD_VARIABLES}


def _science_value(value: Any) -> Any:
    """Return the result projection after removing only resource metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _science_value(item)
            for key, item in value.items()
            if str(key) not in SCIENCE_RESULT_RESOURCE_KEYS
        }
    if isinstance(value, list):
        return [_science_value(item) for item in value]
    return copy.deepcopy(value)


def science_fingerprint(result: Mapping[str, Any]) -> dict[str, Any]:
    projection = _science_value(result)
    if not isinstance(projection, dict) or not projection:
        raise RuntimeError("science result projection is empty")
    return {
        "schema_version": "mft-tier1-final1000-resource-science-fingerprint-v1",
        "removed_resource_keys": sorted(SCIENCE_RESULT_RESOURCE_KEYS),
        "projection_sha256": canonical_sha256(projection),
    }


def _payload_without_resource_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    for key in ("inference_threads", "scheduler_cpus"):
        value.pop(key, None)
    return value


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    required = {
        "schema_version",
        "comparator_id",
        "anchor_task_id",
        "anchor_seed",
        "anchor_cpus",
        "companion_cpus",
        "stage_id",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_contract_sha256",
        "offload_plan_file_sha256",
        "source_config_file_sha256",
        "source_config_sha256",
        "source_package_file_sha256",
        "source_package_sha256",
        "source_submission_file_sha256",
        "source_submission_receipt_sha256",
        "source_parent_task_sha256",
        "source_child_task_sha256",
        "source_child_payload_sha256",
        "memory_mb",
        "max_workers_per_node",
        "peak_rss_gate_bytes",
        "prior_unpaired_evidence",
        "anchor_failure_observation",
        "launch_policy",
        "capacity_observed_at",
        "capacity_source",
        "eligible_allocations",
        "decision_policy",
        "batch_rollout_policy",
        "safety_policy",
        "config_sha256",
    }
    decision = value.get("decision_policy")
    safety = value.get("safety_policy")
    if (
        set(value) != required
        or value.get("schema_version") != CONFIG_SCHEMA
        or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{7,127}", str(value.get("comparator_id") or "")
        )
        or value.get("anchor_task_id") != ANCHOR_TASK_ID
        or value.get("anchor_seed") != ANCHOR_SEED
        or value.get("anchor_cpus") != ANCHOR_CPUS
        or value.get("companion_cpus") != list(COMPANION_CPUS)
        or value.get("stage_id") != STAGE_ID
        or value.get("bundle_id") != BUNDLE_ID
        or value.get("bundle_manifest_sha256") != BUNDLE_MANIFEST_SHA256
        or value.get("bundle_contract_sha256") != BUNDLE_CONTRACT_SHA256
        or value.get("offload_plan_file_sha256") != OFFLOAD_PLAN_FILE_SHA256
        or value.get("source_config_file_sha256") != SOURCE_CONFIG_FILE_SHA256
        or value.get("source_config_sha256") != SOURCE_CONFIG_SHA256
        or value.get("source_package_file_sha256") != SOURCE_PACKAGE_FILE_SHA256
        or value.get("source_package_sha256") != SOURCE_PACKAGE_SHA256
        or value.get("source_submission_file_sha256")
        != SOURCE_SUBMISSION_FILE_SHA256
        or value.get("source_submission_receipt_sha256")
        != SOURCE_SUBMISSION_RECEIPT_SHA256
        or value.get("source_parent_task_sha256") != SOURCE_PARENT_TASK_SHA256
        or value.get("source_child_task_sha256") != SOURCE_CHILD_TASK_SHA256
        or value.get("source_child_payload_sha256")
        != SOURCE_CHILD_PAYLOAD_SHA256
        or value.get("memory_mb") != MEMORY_MB
        or value.get("max_workers_per_node") != MAX_WORKERS_PER_NODE
        or value.get("peak_rss_gate_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or value.get("prior_unpaired_evidence")
        != {
            "task_id": 86501,
            "seed": 2257499993,
            "terminal_file_sha256": PRIOR_UNPAIRED_TERMINAL_FILE_SHA256,
            "remote_terminal_sha256": PRIOR_UNPAIRED_REMOTE_TERMINAL_SHA256,
            "promotion_eligible": False,
            "candidate_wall_time_seconds": 4600.7869332283735,
            "baseline_wall_time_seconds": 1871.5297520039603,
            "single_seed_throughput_ratio_vs_4cpu": 0.40678470426160496,
            "slot_weighted_throughput_ratio_vs_4cpu": 0.6324728623306833,
            "candidate_average_cores_used": 1.2403052070911338,
            "core_use_ratio_vs_4cpu": 0.8183366656776067,
            "candidate_peak_rss_bytes": 18983600128,
            "interpretation": "unpaired-seed-complexity-confounded-not-promotable",
        }
        or value.get("anchor_failure_observation")
        != {
            "observed_at": "2026-07-23T08:52:00+09:00",
            "scheduler_access": "GET-only",
            "scheduler_write_count": 0,
            "task_id": ANCHOR_TASK_ID,
            "status": "failed",
            "exit_code": 1,
            "failure_message": "RuntimeError: Phase B parent memory evidence seal mismatch",
            "started_at": "2026-07-22 23:16:04",
            "finished_at": "2026-07-22 23:50:37",
            "allocation_id": 14011,
        }
        or value.get("launch_policy")
        != {
            "launch_allowed": False,
            "block_reason": "task87052-failed-parent-memory-evidence-seal",
            "phase_b_parent_memory_fix_required": True,
            "replacement_exact_4cpu_anchor_required": True,
            "replacement_anchor_must_complete_and_pass_terminal_gates": True,
            "new_identity_package_and_independent_approval_required": True,
        }
        or not str(value.get("capacity_observed_at") or "")
        or value.get("capacity_source")
        != {
            "scheduler_endpoint": "GET /api/allocations",
            "scheduler_write_count": 0,
            "states_counted": ["active"],
            "resource_pool": "cpu",
            "exclusive_node": False,
        }
        or not isinstance(value.get("eligible_allocations"), list)
        or not value["eligible_allocations"]
        or not isinstance(decision, dict)
        or decision
        != {
            "metric": "allocation-local-exact-slots-per-second",
            "formula": "sum(min(floor(total_cpus/cpus),floor(total_memory_mb/28672),32))/wall_seconds",
            "minimum_ratio_vs_anchor_for_resource_change": 1.0,
            "strict_improvement_required": True,
            "per_seed_latency_gate": None,
            "per_seed_latency_override_allowed": False,
            "science_fingerprint_match_required": True,
            "automatic_promotion_allowed": False,
        }
        or not isinstance(safety, dict)
        or safety
        != {
            "independent_approval_required_before_each_apply": True,
            "anchor_terminal_attestation_required_before_each_apply": True,
            "maximum_scheduler_posts_per_companion": 1,
            "complete_production_and_comparator_namespace_scan_required": True,
            "post_submit_namespace_rescan_required": True,
            "cancellation_allowed": False,
            "preemption_allowed": False,
            "allocation_mutation_allowed": False,
            "publication_allowed": False,
            "fea_submission_allowed": False,
            "aedt_allowed": False,
        }
        or value.get("batch_rollout_policy")
        != {
            "objective": "maximize-completed-seeds-per-wall-clock",
            "target_formula": "exact_fit_slots[selected_cpus]",
            "allocation_states_counted": ["active"],
            "resource_pool": "cpu",
            "exclusive_allocations_counted": False,
            "allocation_local_cpu_and_memory_fit_required": True,
            "cross_allocation_resource_packing_allowed": False,
            "account_and_node_limits_are_hard": True,
            "max_workers_per_node": MAX_WORKERS_PER_NODE,
            "recompute_capacity_each_controller_cycle": True,
            "maintain_selected_active_target_until_stop": True,
            "one_durable_reservation_and_one_post_at_a_time": True,
            "per_seed_latency_priority": False,
            "resource2_rollout_currently_allowed": False,
            "resource2_unpaired_evidence_is_not_promotion_evidence": True,
            "promotion_requires_paired_comparison_and_independent_review": True,
            "fallback_cpus_before_promotion": ANCHOR_CPUS,
            "automatic_rollout_allowed": False,
        }
        or value.get("config_sha256") != canonical_sha256(unsigned)
    ):
        raise RuntimeError("paired resource comparator config seal mismatch")
    # Rebuild the capacity table to reject aggregate-only or duplicated inputs.
    capacity_snapshot(value["eligible_allocations"])
    return copy.deepcopy(dict(value))


def capacity_snapshot(
    allocations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in allocations:
        allocation_id = raw.get("allocation_id", raw.get("id"))
        total_cpus = raw.get("total_cpus")
        total_memory_mb = raw.get("total_memory_mb")
        if (
            isinstance(allocation_id, bool)
            or not isinstance(allocation_id, int)
            or allocation_id <= 0
            or allocation_id in seen
            or isinstance(total_cpus, bool)
            or not isinstance(total_cpus, int)
            or total_cpus <= 0
            or isinstance(total_memory_mb, bool)
            or not isinstance(total_memory_mb, int)
            or total_memory_mb < MEMORY_MB
        ):
            raise RuntimeError("eligible allocation identity/resource is invalid")
        seen.add(allocation_id)
        fits = {
            str(cpus): min(
                total_cpus // cpus,
                total_memory_mb // MEMORY_MB,
                MAX_WORKERS_PER_NODE,
            )
            for cpus in ALL_CPU_SHAPES
        }
        rows.append(
            {
                "allocation_id": allocation_id,
                "total_cpus": total_cpus,
                "total_memory_mb": total_memory_mb,
                "exact_fit_slots": fits,
            }
        )
    rows.sort(key=lambda item: int(item["allocation_id"]))
    totals = {
        str(cpus): sum(int(row["exact_fit_slots"][str(cpus)]) for row in rows)
        for cpus in ALL_CPU_SHAPES
    }
    unsigned = {
        "schema_version": CAPACITY_SCHEMA,
        "eligible_allocation_count": len(rows),
        "memory_mb_per_task": MEMORY_MB,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "formula": "min(floor(total_cpus/cpus),floor(total_memory_mb/28672),32)",
        "per_allocation": rows,
        "exact_fit_slots": totals,
        "allocation_local": True,
        "aggregate_cross_allocation_packing_allowed": False,
    }
    return {**unsigned, "capacity_sha256": canonical_sha256(unsigned)}


def validate_capacity_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    rows = value.get("per_allocation")
    if not isinstance(rows, list):
        raise RuntimeError("paired resource capacity rows are absent")
    rebuilt = capacity_snapshot(rows)
    if dict(value) != rebuilt:
        raise RuntimeError("paired resource capacity snapshot does not reproduce")
    return copy.deepcopy(dict(value))


def _authenticate_source(
    *,
    source_config_path: Path,
    source_package_path: Path,
    source_submission_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
) -> dict[str, Any]:
    expected_files = {
        "source config": (source_config_path, SOURCE_CONFIG_FILE_SHA256),
        "source package": (source_package_path, SOURCE_PACKAGE_FILE_SHA256),
        "source submission": (
            source_submission_path,
            SOURCE_SUBMISSION_FILE_SHA256,
        ),
        "offload plan": (offload_plan_path, OFFLOAD_PLAN_FILE_SHA256),
    }
    for label, (path, expected) in expected_files.items():
        if _file_sha(path) != expected:
            raise RuntimeError(f"task-87052 {label} file SHA drifted")
    source_config = _read_json(source_config_path, "task-87052 source config")
    config_unsigned = {
        key: item for key, item in source_config.items() if key != "config_sha256"
    }
    source_package = _read_json(source_package_path, "task-87052 source package")
    package_unsigned = {
        key: item for key, item in source_package.items() if key != "package_sha256"
    }
    submission = _read_json(source_submission_path, "task-87052 source submission")
    submission_unsigned = {
        key: item for key, item in submission.items() if key != "receipt_sha256"
    }
    publication = _read_json(publication_receipt_path, "bundle publication receipt")
    publication_unsigned = {
        key: item for key, item in publication.items() if key != "receipt_sha256"
    }
    if (
        source_config.get("schema_version")
        != "mft-tier1-final1000-phase-b-shape-canary-config-v2"
        or source_config.get("config_sha256") != canonical_sha256(config_unsigned)
        or source_config.get("config_sha256") != SOURCE_CONFIG_SHA256
        or source_package.get("schema_version")
        != "mft-tier1-final1000-phase-b-shape-canary-package-v2"
        or source_package.get("package_sha256") != canonical_sha256(package_unsigned)
        or source_package.get("package_sha256") != SOURCE_PACKAGE_SHA256
        or source_package.get("config_sha256") != SOURCE_CONFIG_SHA256
        or source_package.get("offload_plan_file_sha256")
        != OFFLOAD_PLAN_FILE_SHA256
        or source_package.get("publication_receipt_file_sha256")
        != _file_sha(publication_receipt_path)
        or source_package.get("publication_receipt_sha256")
        != publication.get("receipt_sha256")
        or publication.get("receipt_sha256") != canonical_sha256(publication_unsigned)
        or publication.get("bundle_id") != BUNDLE_ID
        or publication.get("bundle_manifest_sha256") != BUNDLE_MANIFEST_SHA256
        or publication.get("contract_sha256") != BUNDLE_CONTRACT_SHA256
        or publication.get("remote_bundle") != source_package.get("remote_bundle")
        or publication.get("ready_sha256") != source_package.get("ready_sha256")
        or submission.get("receipt_sha256") != canonical_sha256(submission_unsigned)
        or submission.get("package_sha256") != SOURCE_PACKAGE_SHA256
    ):
        raise RuntimeError("task-87052 source config/package/publication seal drifted")
    if (
        submission.get("receipt_sha256") != SOURCE_SUBMISSION_RECEIPT_SHA256
        or submission.get("task_id") != ANCHOR_TASK_ID
        or submission.get("scheduler_post_count") != 1
        or submission.get("parent_dedupe_key") != SOURCE_PARENT_DEDUPE
    ):
        raise RuntimeError("task-87052 source submission identity drifted")
    parent = copy.deepcopy(source_package["parent_task"])
    children = (parent.get("payload_json") or {}).get("children") or []
    if len(children) != 1 or not isinstance(children[0], Mapping):
        raise RuntimeError("task-87052 anchor is not exactly one logical child")
    child_record = dict(children[0])
    child = child_record.get("task")
    if not isinstance(child, Mapping):
        raise RuntimeError("task-87052 logical child task is absent")
    child = copy.deepcopy(dict(child))
    payload = child.get("payload_json")
    if (
        canonical_sha256(parent) != SOURCE_PARENT_TASK_SHA256
        or parent.get("dedupe_key") != SOURCE_PARENT_DEDUPE
        or canonical_sha256(child) != SOURCE_CHILD_TASK_SHA256
        or child_record.get("child_task_sha256") != SOURCE_CHILD_TASK_SHA256
        or child.get("dedupe_key") != SOURCE_CHILD_DEDUPE
        or not isinstance(payload, Mapping)
        or canonical_sha256(payload) != SOURCE_CHILD_PAYLOAD_SHA256
        or child_record.get("payload_sha256") != SOURCE_CHILD_PAYLOAD_SHA256
        or payload.get("seed") != ANCHOR_SEED
        or (payload.get("lane") or {}).get("stage_id") != STAGE_ID
        or payload.get("inference_threads") != ANCHOR_CPUS
        or payload.get("scheduler_cpus") != ANCHOR_CPUS
        or child.get("cpus") != ANCHOR_CPUS
        or child.get("memory_mb") != MEMORY_MB
        or child.get("max_workers_per_node") != MAX_WORKERS_PER_NODE
        or child.get("remote_cwd") != source_package.get("remote_bundle")
    ):
        raise RuntimeError("task-87052 anchor parent/child identity drifted")
    return {
        "source_package": source_package,
        "source_submission": submission,
        "parent_task": parent,
        "child_task": child,
        "publication_receipt_file_sha256": _file_sha(publication_receipt_path),
    }


_INLINE_WRAPPER_PREFIX = r"""import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEMA = "mft-tier1-final1000-paired-resource-comparator-telemetry-v1"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
REMOVED = frozenset(("payload_sha256", "optimizer_pid", "inference_threads", "scheduler_cpus", "optimizer_inference_threads", "threads_per_model", "family_threads"))

def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    try:
        staged.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()

def science_value(value):
    if isinstance(value, dict):
        return {str(key): science_value(item) for key, item in value.items() if str(key) not in REMOVED}
    if isinstance(value, list):
        return [science_value(item) for item in value]
    return value
"""


_INLINE_WRAPPER_SUFFIX = r"""
payload_path = Path(sys.argv[1]).resolve(strict=True)
payload_root = Path(sys.argv[2]).resolve(strict=True)
payload_sha = sys.argv[3]
seed = int(sys.argv[4])
requested_cpus = int(sys.argv[5])
requested_memory_bytes = int(sys.argv[6])
rss_gate_bytes = int(sys.argv[7])
comparator_identity_sha = sys.argv[8]
bundle = Path.cwd().resolve(strict=True)
task_id = str(os.environ.get("SLURM_SCHED_TASK_ID") or "")
if not task_id.isascii() or not task_id.isdigit() or int(task_id) <= 0:
    raise SystemExit("missing canonical SLURM_SCHED_TASK_ID")
task_root = bundle / "runs" / ("task-" + task_id)
telemetry_path = task_root / "paired_resource_telemetry.json"
started_at = now()
failure = None
exit_code = 79
cpu_set_before = []
cpu_set_after = []
cgroup_before = None
cgroup_after = None
cgroup_diagnostics = {"before": None, "after": None}
wall = None
cpu_seconds = None
peak_rss_bytes = None
science = None
status_record = {"relative_path": "runs/task-" + task_id + "/seed_status.json", "size": None, "sha256": None, "state": None, "terminal": None, "exit_code": None}
result_record = {"relative_path": "runs/task-" + task_id + "/seed-" + str(seed) + "/result.json", "size": None, "sha256": None}
cgroup_diagnostics["before"] = capture_cgroup_diagnostics()
try:
    if payload_path.name != "payload.json" or payload_root not in payload_path.parents:
        raise RuntimeError("scheduler payload escaped run root")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if canonical(payload) != payload_sha:
        raise RuntimeError("scheduler payload canonical SHA mismatch")
    if payload.get("seed") != seed or payload.get("inference_threads") != requested_cpus or payload.get("scheduler_cpus") != requested_cpus:
        raise RuntimeError("payload seed/CPU/thread binding drifted")
    cpu_set_before = sorted(os.sched_getaffinity(0))
    if len(cpu_set_before) != requested_cpus or len(set(cpu_set_before)) != requested_cpus:
        raise RuntimeError("inherited Scheduler cpuset does not match requested CPUs")
    if os.environ.get("SLURM_CPUS_PER_TASK") != str(requested_cpus):
        raise RuntimeError("SLURM_CPUS_PER_TASK does not match requested CPUs")
    if any(os.environ.get(name) != str(requested_cpus) for name in THREADS):
        raise RuntimeError("inference thread environment does not match requested CPUs")
    cgroup_before = cgroup_snapshot_from_diagnostics(cgroup_diagnostics["before"])
    selected_before = cgroup_before["selected_finite_ancestor"]
    if int(selected_before["memory_limit_bytes"]) < requested_memory_bytes:
        raise RuntimeError("finite cgroup ancestor does not cover requested memory")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_started = time.monotonic()
    command = [
        sys.executable, "-u",
        str(bundle / "artifacts/code/tools/tier1_corrected_current7_slurm_seed_runner.py"),
        "--bundle-root", str(bundle),
        "--payload", str(payload_path),
        "--payload-root", str(payload_root),
        "--payload-sha256", payload_sha,
    ]
    completed = subprocess.run(command, cwd=bundle / "artifacts/code", check=False)
    wall = max(0.0, time.monotonic() - wall_started)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(0.0, (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime))
    peak_rss_bytes = int(after.ru_maxrss) * 1024
    cpu_set_after = sorted(os.sched_getaffinity(0))
    cgroup_diagnostics["after"] = capture_cgroup_diagnostics()
    cgroup_after = cgroup_snapshot_from_diagnostics(cgroup_diagnostics["after"])
    selected_after = cgroup_after["selected_finite_ancestor"]
    if cgroup_diagnostics["after"]["proc_self_cgroup"]["sha256"] != cgroup_diagnostics["before"]["proc_self_cgroup"]["sha256"] or cgroup_diagnostics["after"]["proc_self_mountinfo"]["sha256"] != cgroup_diagnostics["before"]["proc_self_mountinfo"]["sha256"]:
        raise RuntimeError("cgroup membership or mount mapping changed")
    if cgroup_after["cgroup_version"] != cgroup_before["cgroup_version"] or cgroup_after["mount_relative_path"] != cgroup_before["mount_relative_path"] or cgroup_after["membership_path"] != cgroup_before["membership_path"] or selected_after["relative_path"] != selected_before["relative_path"] or selected_after["memory_limit_bytes"] != selected_before["memory_limit_bytes"]:
        raise RuntimeError("finite cgroup memory ancestor changed")
    if int(selected_after["memory_current_bytes"]) > int(selected_after["memory_limit_bytes"]) or int(selected_after["memory_peak_bytes"]) > int(selected_after["memory_limit_bytes"]):
        raise RuntimeError("cgroup memory usage/peak exceeds finite limit")
    status_path = task_root / "seed_status.json"
    if status_path.is_file():
        raw = status_path.read_bytes()
        status = json.loads(raw.decode("utf-8"))
        status_record = {"relative_path": status_record["relative_path"], "size": len(raw), "sha256": sha_bytes(raw), "state": status.get("state"), "terminal": status.get("terminal"), "exit_code": status.get("exit_code")}
    result_path = task_root / ("seed-" + str(seed)) / "result.json"
    if result_path.is_file():
        raw = result_path.read_bytes()
        result = json.loads(raw.decode("utf-8"))
        result_record = {"relative_path": result_record["relative_path"], "size": len(raw), "sha256": sha_bytes(raw)}
        science = {"schema_version": "mft-tier1-final1000-resource-science-fingerprint-v1", "removed_resource_keys": sorted(REMOVED), "projection_sha256": canonical(science_value(result))}
    exit_code = int(completed.returncode)
    if exit_code != 0 or status_record["state"] != "completed" or status_record["terminal"] is not True or status_record["exit_code"] != 0 or result_record["sha256"] is None or science is None:
        failure = "single-seed runner did not produce one completed sealed result"
        exit_code = 78 if exit_code == 0 else exit_code
    elif cpu_set_after != cpu_set_before:
        failure = "wrapper affinity changed during execution"
        exit_code = 78
    elif peak_rss_bytes <= 0 or peak_rss_bytes > rss_gate_bytes:
        failure = "per-seed peak RSS is outside its gate"
        exit_code = 78
except Exception as exc:
    failure = type(exc).__name__ + ":" + str(exc)
    exit_code = 79

capacity = wall * requested_cpus if wall is not None else None
average_cores = cpu_seconds / wall if cpu_seconds is not None and wall and wall > 0 else None
utilization = cpu_seconds / capacity if cpu_seconds is not None and capacity and capacity > 0 else None
unsigned = {
    "schema_version": SCHEMA,
    "started_at": started_at,
    "finished_at": now(),
    "task_id": int(task_id),
    "seed": seed,
    "comparator_identity_sha256": comparator_identity_sha,
    "payload_sha256": payload_sha,
    "requested_cpus": requested_cpus,
    "requested_memory_bytes": requested_memory_bytes,
    "rss_gate_bytes": rss_gate_bytes,
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "thread_environment": {name: os.environ.get(name) for name in THREADS},
    "cpu_set_before": cpu_set_before,
    "cpu_set_after": cpu_set_after,
    "wall_time_seconds": wall,
    "process_tree_cpu_seconds": cpu_seconds,
    "cpu_capacity_seconds": capacity,
    "average_cores_used": average_cores,
    "cpu_utilization_fraction": utilization,
    "peak_rss_bytes": peak_rss_bytes,
    "cgroup_diagnostics": cgroup_diagnostics,
    "cgroup_before": cgroup_before,
    "cgroup_after": cgroup_after,
    "seed_status": status_record,
    "result": result_record,
    "science_fingerprint": science,
    "exit_code": exit_code,
    "failure": failure,
    "fea_submission_performed": False,
    "aedt_used": False,
}
value = dict(unsigned)
value["telemetry_sha256"] = canonical(unsigned)
atomic_json(telemetry_path, value)
raise SystemExit(exit_code)
"""


_INLINE_WRAPPER = (
    _INLINE_WRAPPER_PREFIX
    + "\n"
    + cgroup_memory.embedded_runtime_source()
    + "\n"
    + _INLINE_WRAPPER_SUFFIX
)


def _companion_command(
    *, payload_sha256: str, cpus: int, package_identity_sha256: str
) -> str:
    if (
        not _is_sha256(payload_sha256)
        or cpus not in COMPANION_CPUS
        or not _is_sha256(package_identity_sha256)
    ):
        raise RuntimeError("companion command identity is invalid")
    exports = "\n".join(f"export {name}={cpus}" for name in THREAD_VARIABLES)
    return "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site:$PWD/artifacts/code'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            exports,
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            'python -u - "$payload_path" "$payload_root" '
            f"{payload_sha256} {ANCHOR_SEED} {cpus} {MEMORY_MB * 1024**2} "
            f"{DEFAULT_PEAK_RSS_GATE_BYTES} {package_identity_sha256} <<'PY'",
            _INLINE_WRAPPER,
            "PY",
        ]
    )


def _changed_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}/{key}"
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(_changed_paths(left[key], right[key], path))
        return result
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        result = set()
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            result.update(_changed_paths(a, b, f"{prefix}/{index}"))
        return result
    return set() if left == right else {prefix}


def _comparator_identity(
    *, config: Mapping[str, Any], source_child: Mapping[str, Any], capacity: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = {
        "namespace": "final1000-task87052-paired-resource-comparator-v1",
        "config_sha256": config["config_sha256"],
        "source_parent_task_sha256": SOURCE_PARENT_TASK_SHA256,
        "source_child_task_sha256": canonical_sha256(source_child),
        "source_child_payload_sha256": canonical_sha256(source_child["payload_json"]),
        "capacity_sha256": capacity["capacity_sha256"],
        "companion_cpus": list(COMPANION_CPUS),
    }
    return {**unsigned, "identity_sha256": canonical_sha256(unsigned)}


def _transform_child_task(
    source_child: Mapping[str, Any],
    *,
    cpus: int,
    comparator_id: str,
    comparator_identity_sha256: str,
) -> dict[str, Any]:
    if cpus not in COMPANION_CPUS or not _is_sha256(comparator_identity_sha256):
        raise RuntimeError("unsupported paired comparator CPU shape")
    if set(source_child) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError("task-87052 source child Scheduler envelope drifted")
    payload = copy.deepcopy(dict(source_child["payload_json"]))
    payload["inference_threads"] = cpus
    payload["scheduler_cpus"] = cpus
    if _changed_paths(source_child["payload_json"], payload) != {
        "/inference_threads",
        "/scheduler_cpus",
    } or _payload_without_resource_fields(payload) != _payload_without_resource_fields(
        source_child["payload_json"]
    ):
        raise RuntimeError("companion science payload differs outside resource fields")
    payload_sha = canonical_sha256(payload)
    resources = {
        "cpus": cpus,
        "memory_mb": MEMORY_MB,
        "scheduling_profile": source_child["scheduling_profile"],
        "aedt_backend": source_child["aedt_backend"],
        "gpus": source_child["gpus"],
        "priority": source_child["priority"],
        "timeout_seconds": source_child["timeout_seconds"],
        "max_workers_per_node": source_child["max_workers_per_node"],
    }
    dedupe_identity = {
        "namespace": "final1000-task87052-paired-resource-companion-v1",
        "comparator_id": comparator_id,
        "comparator_identity_sha256": comparator_identity_sha256,
        "source_child_task_sha256": SOURCE_CHILD_TASK_SHA256,
        "cpus": cpus,
        "payload_sha256": payload_sha,
        "resources": resources,
    }
    return {
        **copy.deepcopy(dict(source_child)),
        "name": f"{COMPARATOR_PREFIX}c{cpus}-s{ANCHOR_SEED}",
        "command": _companion_command(
            payload_sha256=payload_sha,
            cpus=cpus,
            package_identity_sha256=comparator_identity_sha256,
        ),
        "payload_json": payload,
        **resources,
        "dedupe_key": (
            f"{COMPARATOR_DEDUPE_PREFIX}c{cpus}:"
            f"{canonical_sha256(dedupe_identity)}"
        ),
    }


def _validate_companion_task(
    task: Mapping[str, Any],
    *,
    source_child: Mapping[str, Any],
    cpus: int,
    comparator_id: str,
    comparator_identity_sha256: str,
) -> dict[str, Any]:
    expected = _transform_child_task(
        source_child,
        cpus=cpus,
        comparator_id=comparator_id,
        comparator_identity_sha256=comparator_identity_sha256,
    )
    if dict(task) != expected or set(task) != REQUIRED_SCHEDULER_FIELDS:
        raise RuntimeError(f"{cpus}-CPU companion differs from sealed transform")
    allowed_envelope_changes = {
        "/name",
        "/command",
        "/cpus",
        "/dedupe_key",
        "/payload_json/inference_threads",
        "/payload_json/scheduler_cpus",
    }
    if _changed_paths(source_child, task) != allowed_envelope_changes:
        raise RuntimeError(f"{cpus}-CPU companion changed a science/envelope field")
    if (
        not str(task["name"]).startswith(COMPARATOR_PREFIX)
        or str(task["name"]).startswith(PRODUCTION_PREFIX)
        or not str(task["dedupe_key"]).startswith(
            f"{COMPARATOR_DEDUPE_PREFIX}c{cpus}:"
        )
        or SOURCE_CHILD_DEDUPE in str(task["dedupe_key"])
        or task["cpus"] != cpus
        or task["payload_json"]["inference_threads"] != cpus
        or task["payload_json"]["scheduler_cpus"] != cpus
        or task["memory_mb"] != MEMORY_MB
        or any(
            token in str(task["command"]).lower()
            for token in ("ansysedt", "pyaedt.desktop")
        )
    ):
        raise RuntimeError(f"{cpus}-CPU companion isolation/resource seal mismatch")
    return copy.deepcopy(dict(task))


def _terminal_evidence(remote_cwd: str, cpus: int) -> dict[str, Any]:
    root = PurePosixPath(remote_cwd) / "runs" / "task-{scheduler_task_id}"
    return {
        "task_root": root.as_posix(),
        "telemetry_path": (root / TELEMETRY_FILENAME).as_posix(),
        "seed_status_path": (root / "seed_status.json").as_posix(),
        "result_path": (root / f"seed-{ANCHOR_SEED}" / "result.json").as_posix(),
        "scheduler_detail_required": True,
        "exact_cpuset_length": cpus,
        "exact_cpuset_json_pointers": ["/cpu_set_before", "/cpu_set_after"],
        "thread_environment_json_pointer": "/thread_environment",
        "wall_json_pointer": "/wall_time_seconds",
        "process_tree_cpu_json_pointer": "/process_tree_cpu_seconds",
        "rss_json_pointer": "/peak_rss_bytes",
        "cgroup_diagnostics_json_pointer": "/cgroup_diagnostics",
        "finite_cgroup_ancestor_json_pointer": (
            "/cgroup_after/selected_finite_ancestor"
        ),
        "science_fingerprint_json_pointer": "/science_fingerprint",
    }


def _assemble_package(
    *, config: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    capacity = capacity_snapshot(config["eligible_allocations"])
    identity = _comparator_identity(
        config=config, source_child=source["child_task"], capacity=capacity
    )
    companions = {
        str(cpus): _transform_child_task(
            source["child_task"],
            cpus=cpus,
            comparator_id=str(config["comparator_id"]),
            comparator_identity_sha256=identity["identity_sha256"],
        )
        for cpus in COMPANION_CPUS
    }
    unsigned = {
        "schema_version": PACKAGE_SCHEMA,
        # Deterministic rebuild is mandatory at the future submission boundary.
        "rendered_at": str(config["capacity_observed_at"]),
        "config": copy.deepcopy(dict(config)),
        "comparator_identity": identity,
        "anchor": {
            "task_id": ANCHOR_TASK_ID,
            "seed": ANCHOR_SEED,
            "cpus": ANCHOR_CPUS,
            "source_package_sha256": SOURCE_PACKAGE_SHA256,
            "source_submission_receipt_sha256": SOURCE_SUBMISSION_RECEIPT_SHA256,
            "source_parent_dedupe_key": SOURCE_PARENT_DEDUPE,
            "source_child_logical_dedupe_key": SOURCE_CHILD_DEDUPE,
            "source_parent_task_sha256": SOURCE_PARENT_TASK_SHA256,
            "source_child_task_sha256": SOURCE_CHILD_TASK_SHA256,
            "source_child_payload_sha256": SOURCE_CHILD_PAYLOAD_SHA256,
            "parent_task": copy.deepcopy(source["parent_task"]),
            "child_task": copy.deepcopy(source["child_task"]),
        },
        "capacity": capacity,
        "companions": companions,
        "terminal_evidence": {
            str(cpus): _terminal_evidence(
                str(source["child_task"]["remote_cwd"]), cpus
            )
            for cpus in COMPANION_CPUS
        },
        "decision_policy": copy.deepcopy(config["decision_policy"]),
        "batch_rollout_policy": copy.deepcopy(config["batch_rollout_policy"]),
        "prior_unpaired_evidence": copy.deepcopy(
            config["prior_unpaired_evidence"]
        ),
        "anchor_failure_observation": copy.deepcopy(
            config["anchor_failure_observation"]
        ),
        "launch_policy": copy.deepcopy(config["launch_policy"]),
        "submission_policy": {
            "dry_run_default": True,
            "explicit_apply_required": True,
            "independent_approval_required": True,
            "anchor_task_must_be_exact_completed_and_promotion_eligible": True,
            "one_additive_post_maximum_per_companion": True,
            "companion_retries_without_new_package_and_approval_allowed": False,
            "complete_namespace_prefixes": [
                PRODUCTION_PREFIX,
                COMPARATOR_PREFIX,
            ],
            "post_submit_rescan_required": True,
            "automatic_production_promotion_allowed": False,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "allocation_mutation_count": 0,
            "publication_count": 0,
            "fea_submission_performed": False,
            "aedt_used": False,
        },
    }
    return {**unsigned, "package_sha256": canonical_sha256(unsigned)}


def validate_package(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "package_sha256"}
    required = {
        "schema_version",
        "rendered_at",
        "config",
        "comparator_identity",
        "anchor",
        "capacity",
        "companions",
        "terminal_evidence",
        "decision_policy",
        "batch_rollout_policy",
        "prior_unpaired_evidence",
        "anchor_failure_observation",
        "launch_policy",
        "submission_policy",
        "package_sha256",
    }
    config_raw = value.get("config")
    anchor = value.get("anchor")
    companions = value.get("companions")
    if (
        set(value) != required
        or value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("package_sha256") != canonical_sha256(unsigned)
        or not isinstance(config_raw, Mapping)
        or not isinstance(anchor, Mapping)
        or not isinstance(companions, Mapping)
        or set(companions) != {"2", "3"}
    ):
        raise RuntimeError("paired resource comparator package seal mismatch")
    config = validate_config(config_raw)
    capacity = validate_capacity_snapshot(value.get("capacity") or {})
    parent = anchor.get("parent_task")
    child = anchor.get("child_task")
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        raise RuntimeError("paired comparator anchor tasks are absent")
    if (
        canonical_sha256(parent) != SOURCE_PARENT_TASK_SHA256
        or canonical_sha256(child) != SOURCE_CHILD_TASK_SHA256
        or canonical_sha256(child.get("payload_json"))
        != SOURCE_CHILD_PAYLOAD_SHA256
        or anchor.get("task_id") != ANCHOR_TASK_ID
        or anchor.get("seed") != ANCHOR_SEED
        or anchor.get("cpus") != ANCHOR_CPUS
        or anchor.get("source_package_sha256") != SOURCE_PACKAGE_SHA256
        or anchor.get("source_submission_receipt_sha256")
        != SOURCE_SUBMISSION_RECEIPT_SHA256
        or anchor.get("source_parent_dedupe_key") != SOURCE_PARENT_DEDUPE
        or anchor.get("source_child_logical_dedupe_key") != SOURCE_CHILD_DEDUPE
    ):
        raise RuntimeError("paired comparator embedded anchor identity drifted")
    identity = _comparator_identity(
        config=config, source_child=child, capacity=capacity
    )
    if value.get("comparator_identity") != identity:
        raise RuntimeError("paired comparator identity does not reproduce")
    for cpus in COMPANION_CPUS:
        _validate_companion_task(
            companions[str(cpus)],
            source_child=child,
            cpus=cpus,
            comparator_id=config["comparator_id"],
            comparator_identity_sha256=identity["identity_sha256"],
        )
        if value["terminal_evidence"][str(cpus)] != _terminal_evidence(
            str(child["remote_cwd"]), cpus
        ):
            raise RuntimeError("paired comparator terminal evidence path drifted")
    if (
        value.get("decision_policy") != config["decision_policy"]
        or value.get("batch_rollout_policy")
        != config["batch_rollout_policy"]
        or value.get("prior_unpaired_evidence")
        != config["prior_unpaired_evidence"]
        or value.get("anchor_failure_observation")
        != config["anchor_failure_observation"]
        or value.get("launch_policy") != config["launch_policy"]
        or value.get("submission_policy")
        != {
            "dry_run_default": True,
            "explicit_apply_required": True,
            "independent_approval_required": True,
            "anchor_task_must_be_exact_completed_and_promotion_eligible": True,
            "one_additive_post_maximum_per_companion": True,
            "companion_retries_without_new_package_and_approval_allowed": False,
            "complete_namespace_prefixes": [
                PRODUCTION_PREFIX,
                COMPARATOR_PREFIX,
            ],
            "post_submit_rescan_required": True,
            "automatic_production_promotion_allowed": False,
            "scheduler_cancel_count": 0,
            "scheduler_preempt_count": 0,
            "allocation_mutation_count": 0,
            "publication_count": 0,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    ):
        raise RuntimeError("paired comparator policy drifted")
    return copy.deepcopy(dict(value))


def build_package(
    *,
    config_path: Path,
    source_config_path: Path,
    source_package_path: Path,
    source_submission_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
) -> dict[str, Any]:
    config = validate_config(_read_json(config_path, "paired comparator config"))
    source = _authenticate_source(
        source_config_path=source_config_path,
        source_package_path=source_package_path,
        source_submission_path=source_submission_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
    )
    return validate_package(_assemble_package(config=config, source=source))


def validate_telemetry(
    value: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
    cpus: int,
    task_id: int,
    result: Mapping[str, Any],
    result_raw_sha256: str,
) -> dict[str, Any]:
    package = validate_package(package)
    if cpus not in COMPANION_CPUS:
        raise RuntimeError("telemetry CPU shape is not a comparator companion")
    unsigned = {key: item for key, item in value.items() if key != "telemetry_sha256"}
    required = {
        "schema_version",
        "started_at",
        "finished_at",
        "task_id",
        "seed",
        "comparator_identity_sha256",
        "payload_sha256",
        "requested_cpus",
        "requested_memory_bytes",
        "rss_gate_bytes",
        "slurm_cpus_per_task",
        "thread_environment",
        "cpu_set_before",
        "cpu_set_after",
        "wall_time_seconds",
        "process_tree_cpu_seconds",
        "cpu_capacity_seconds",
        "average_cores_used",
        "cpu_utilization_fraction",
        "peak_rss_bytes",
        "cgroup_diagnostics",
        "cgroup_before",
        "cgroup_after",
        "seed_status",
        "result",
        "science_fingerprint",
        "exit_code",
        "failure",
        "fea_submission_performed",
        "aedt_used",
        "telemetry_sha256",
    }
    task = package["companions"][str(cpus)]
    wall = value.get("wall_time_seconds")
    cpu_seconds = value.get("process_tree_cpu_seconds")
    capacity = value.get("cpu_capacity_seconds")
    average = value.get("average_cores_used")
    utilization = value.get("cpu_utilization_fraction")
    peak = value.get("peak_rss_bytes")
    cpuset_before = value.get("cpu_set_before")
    cpuset_after = value.get("cpu_set_after")
    status = value.get("seed_status")
    result_record = value.get("result")
    if (
        set(value) != required
        or value.get("schema_version") != TELEMETRY_SCHEMA
        or value.get("telemetry_sha256") != canonical_sha256(unsigned)
        or value.get("task_id") != task_id
        or value.get("seed") != ANCHOR_SEED
        or value.get("comparator_identity_sha256")
        != package["comparator_identity"]["identity_sha256"]
        or value.get("payload_sha256") != canonical_sha256(task["payload_json"])
        or value.get("requested_cpus") != cpus
        or value.get("requested_memory_bytes") != MEMORY_MB * 1024**2
        or value.get("rss_gate_bytes") != DEFAULT_PEAK_RSS_GATE_BYTES
        or value.get("slurm_cpus_per_task") != str(cpus)
        or value.get("thread_environment") != _thread_environment(cpus)
        or not isinstance(cpuset_before, list)
        or not isinstance(cpuset_after, list)
        or cpuset_before != cpuset_after
        or len(cpuset_before) != cpus
        or len(set(cpuset_before)) != cpus
        or any(isinstance(item, bool) or not isinstance(item, int) for item in cpuset_before)
        or not isinstance(status, Mapping)
        or status.get("state") != "completed"
        or status.get("terminal") is not True
        or status.get("exit_code") != 0
        or not isinstance(result_record, Mapping)
        or result_record.get("sha256") != result_raw_sha256
        or value.get("science_fingerprint") != science_fingerprint(result)
        or value.get("exit_code") != 0
        or value.get("failure") is not None
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("paired comparator telemetry identity/terminal seal mismatch")
    numeric = (wall, cpu_seconds, capacity, average, utilization)
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) <= 0
        for item in numeric
    ) or (
        isinstance(peak, bool)
        or not isinstance(peak, int)
        or not 0 < peak <= DEFAULT_PEAK_RSS_GATE_BYTES
    ):
        raise RuntimeError("paired comparator telemetry resource metric is invalid")
    if (
        not math.isclose(float(capacity), float(wall) * cpus, rel_tol=1e-12)
        or not math.isclose(float(average), float(cpu_seconds) / float(wall), rel_tol=1e-12)
        or not math.isclose(
            float(utilization), float(cpu_seconds) / float(capacity), rel_tol=1e-12
        )
    ):
        raise RuntimeError("paired comparator telemetry derived CPU metric drifted")
    diagnostics = value.get("cgroup_diagnostics")
    before = value.get("cgroup_before")
    after = value.get("cgroup_after")
    if not isinstance(diagnostics, Mapping) or not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise RuntimeError("paired comparator cgroup evidence is absent")
    before_diag = cgroup_memory.validate_cgroup_diagnostics(diagnostics.get("before"))
    after_diag = cgroup_memory.validate_cgroup_diagnostics(diagnostics.get("after"))
    before_sealed = cgroup_memory.validate_cgroup_snapshot(before)
    after_sealed = cgroup_memory.validate_cgroup_snapshot(after)
    cgroup_memory.validate_cgroup_snapshot_diagnostics_binding(before_sealed, before_diag)
    cgroup_memory.validate_cgroup_snapshot_diagnostics_binding(after_sealed, after_diag)
    selected = after_sealed["selected_finite_ancestor"]
    if (
        before_diag["proc_self_cgroup"]["sha256"]
        != after_diag["proc_self_cgroup"]["sha256"]
        or before_diag["proc_self_mountinfo"]["sha256"]
        != after_diag["proc_self_mountinfo"]["sha256"]
        or before_sealed["cgroup_version"] != after_sealed["cgroup_version"]
        or before_sealed["membership_path"] != after_sealed["membership_path"]
        or before_sealed["mount_relative_path"]
        != after_sealed["mount_relative_path"]
        or before_sealed["selected_finite_ancestor"]["relative_path"]
        != selected["relative_path"]
        or int(selected["memory_limit_bytes"]) < MEMORY_MB * 1024**2
        or int(selected["memory_current_bytes"]) > int(selected["memory_limit_bytes"])
        or int(selected["memory_peak_bytes"]) > int(selected["memory_limit_bytes"])
    ):
        raise RuntimeError("paired comparator cgroup memory binding/gate failed")
    return copy.deepcopy(dict(value))


def validate_anchor_remote_terminal(
    value: Mapping[str, Any], *, package: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact v2 Phase-B task-87052 terminal without v1 coercion."""

    package = validate_package(package)
    unsigned = {
        key: item for key, item in value.items() if key != "remote_terminal_sha256"
    }
    terminal = value.get("terminal_evidence")
    remote_files = value.get("remote_files")
    batch_manifest = value.get("batch_manifest")
    if (
        value.get("schema_version")
        != "mft-tier1-final1000-phase-b-shape-remote-terminal-evidence-v2"
        or value.get("remote_terminal_sha256") != canonical_sha256(unsigned)
        or value.get("package_sha256") != SOURCE_PACKAGE_SHA256
        or value.get("submission_receipt_sha256")
        != SOURCE_SUBMISSION_RECEIPT_SHA256
        or value.get("bundle_id") != BUNDLE_ID
        or value.get("lane_shape") != 1
        or value.get("logical_seed_count") != 1
        or value.get("task_id") != ANCHOR_TASK_ID
        or value.get("scheduler_status") != "completed"
        or value.get("promotion_eligible") is not True
        or value.get("scheduler_access") != "GET-only"
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("remote_access") != "read-only"
        or value.get("remote_write_count") != 0
        or value.get("automatic_promotion_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or not isinstance(terminal, Mapping)
        or not isinstance(remote_files, Mapping)
        or not isinstance(batch_manifest, Mapping)
    ):
        raise RuntimeError("task-87052 v2 remote terminal top-level seal mismatch")
    terminal_unsigned = {
        key: item
        for key, item in terminal.items()
        if key != "terminal_evidence_sha256"
    }
    if (
        terminal.get("terminal_evidence_sha256")
        != canonical_sha256(terminal_unsigned)
        or value.get("terminal_evidence_sha256")
        != terminal.get("terminal_evidence_sha256")
        or terminal.get("task_id") != ANCHOR_TASK_ID
        or terminal.get("lane_shape") != 1
        or terminal.get("logical_seed_count") != 1
        or terminal.get("promotion_eligible") is not True
        or not isinstance(terminal.get("gate_results"), Mapping)
        or not all(terminal["gate_results"].values())
    ):
        raise RuntimeError("task-87052 terminal evidence/gates are not all sealed true")
    manifest = validate_batch_manifest(batch_manifest)
    expected_manifest = batch_manifest_from_payload(
        package["anchor"]["parent_task"]["payload_json"]
    )
    status = validate_task_status(terminal.get("task_status") or {})
    receipts_raw = terminal.get("child_receipts")
    memory = validate_parent_memory_evidence(
        terminal.get("parent_memory_evidence") or {}
    )
    if not isinstance(receipts_raw, list) or len(receipts_raw) != 1:
        raise RuntimeError("task-87052 terminal does not contain one child receipt")
    receipt = validate_child_receipt(receipts_raw[0], manifest=manifest)
    result_file = remote_files.get("child_0_result")
    if (
        manifest != expected_manifest
        or status.get("state") != "completed"
        or status.get("completed_child_count") != 1
        or status.get("failed_child_count") != 0
        or receipt.get("state") != "completed"
        or receipt.get("exit_code") != 0
        or receipt.get("failure") is not None
        or receipt.get("seed") != ANCHOR_SEED
        or receipt.get("logical_dedupe_key") != SOURCE_CHILD_DEDUPE
        or receipt.get("payload_sha256") != SOURCE_CHILD_PAYLOAD_SHA256
        or not isinstance(result_file, Mapping)
        or result_file.get("sha256") != receipt.get("result_sha256")
        or value.get("batch_manifest_sha256") != manifest["manifest_sha256"]
        or terminal.get("manifest_sha256") != manifest["manifest_sha256"]
        or value.get("task_status_sha256") != terminal.get("task_status_sha256")
        or value.get("parent_memory_evidence_sha256")
        != memory["evidence_sha256"]
        or terminal.get("parent_memory_evidence_sha256")
        != memory["evidence_sha256"]
    ):
        raise RuntimeError("task-87052 terminal manifest/status/receipt binding failed")
    return copy.deepcopy(dict(value))


def _anchor_summary(
    *,
    package: Mapping[str, Any],
    remote_terminal: Mapping[str, Any],
    result: Mapping[str, Any],
    result_raw_sha256: str,
) -> dict[str, Any]:
    package = validate_package(package)
    terminal = validate_anchor_remote_terminal(remote_terminal, package=package)
    evidence = terminal["terminal_evidence"]
    receipts = evidence.get("child_receipts") or []
    remote_result = (terminal.get("remote_files") or {}).get("child_0_result") or {}
    if (
        terminal.get("package_sha256") != SOURCE_PACKAGE_SHA256
        or terminal.get("submission_receipt_sha256")
        != SOURCE_SUBMISSION_RECEIPT_SHA256
        or terminal.get("task_id") != ANCHOR_TASK_ID
        or terminal.get("scheduler_status") != "completed"
        or terminal.get("promotion_eligible") is not True
        or evidence.get("lane_shape") != 1
        or len(receipts) != 1
        or receipts[0].get("seed") != ANCHOR_SEED
        or remote_result.get("sha256") != result_raw_sha256
    ):
        raise RuntimeError("task-87052 anchor terminal/result binding failed")
    child_cpu = receipts[0].get("child_resource_telemetry") or {}
    wall = child_cpu.get("wall_time_seconds")
    cpu_seconds = child_cpu.get("process_tree_cpu_seconds")
    cpuset = receipts[0].get("cpu_set")
    peak = (receipts[0].get("legacy_status") or {}).get("observed_peak_rss_bytes")
    if (
        child_cpu.get("available") is not True
        or child_cpu.get("child_cpus") != ANCHOR_CPUS
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) <= 0
        or isinstance(cpu_seconds, bool)
        or not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(float(cpu_seconds))
        or float(cpu_seconds) <= 0
        or not isinstance(cpuset, list)
        or len(cpuset) != ANCHOR_CPUS
        or len(set(cpuset)) != ANCHOR_CPUS
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or not 0 < peak <= DEFAULT_PEAK_RSS_GATE_BYTES
    ):
        raise RuntimeError("task-87052 anchor resource evidence failed")
    return {
        "cpus": ANCHOR_CPUS,
        "task_id": ANCHOR_TASK_ID,
        "wall_time_seconds": float(wall),
        "process_tree_cpu_seconds": float(cpu_seconds),
        "average_cores_used": float(cpu_seconds) / float(wall),
        "peak_rss_bytes": peak,
        "cpu_set": list(cpuset),
        "science_fingerprint": science_fingerprint(result),
        "remote_terminal_sha256": terminal["remote_terminal_sha256"],
        "all_resource_gates_passed": True,
    }


def decide_throughput(
    *, capacity: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    sealed_capacity = validate_capacity_snapshot(capacity)
    if set(summaries) != {"2", "3", "4"}:
        raise RuntimeError("paired throughput decision requires 2/3/4 summaries")
    fingerprints = {
        str(summary.get("science_fingerprint", {}).get("projection_sha256"))
        for summary in summaries.values()
    }
    science_match = len(fingerprints) == 1 and "None" not in fingerprints
    metrics: dict[str, dict[str, Any]] = {}
    anchor_rate = None
    for cpus in ALL_CPU_SHAPES:
        summary = summaries[str(cpus)]
        wall = summary.get("wall_time_seconds")
        if (
            isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) <= 0
        ):
            raise RuntimeError("paired throughput wall time is invalid")
        slots = int(sealed_capacity["exact_fit_slots"][str(cpus)])
        rate = slots / float(wall)
        metrics[str(cpus)] = {
            "cpus": cpus,
            "exact_fit_slots": slots,
            "wall_time_seconds": float(wall),
            "per_seed_throughput_per_second": 1.0 / float(wall),
            "allocation_local_slot_weighted_throughput_per_second": rate,
            "all_resource_gates_passed": summary.get("all_resource_gates_passed")
            is True,
        }
        if cpus == ANCHOR_CPUS:
            anchor_rate = rate
    assert anchor_rate is not None
    for record in metrics.values():
        record["slot_weighted_throughput_ratio_vs_4cpu"] = (
            record["allocation_local_slot_weighted_throughput_per_second"]
            / anchor_rate
        )
        record["science_fingerprint_matches_anchor"] = science_match
        record["eligible"] = (
            record["all_resource_gates_passed"] and science_match
        )
    eligible = [
        record
        for record in metrics.values()
        if record["eligible"]
        and (
            record["cpus"] == ANCHOR_CPUS
            or record["slot_weighted_throughput_ratio_vs_4cpu"] > 1.0
        )
    ]
    selected = max(
        eligible,
        key=lambda item: (
            float(item["allocation_local_slot_weighted_throughput_per_second"]),
            int(item["cpus"]),
        ),
        default=metrics["4"],
    )
    if selected["cpus"] != ANCHOR_CPUS and selected[
        "slot_weighted_throughput_ratio_vs_4cpu"
    ] <= 1.0:
        selected = metrics["4"]
    return {
        "metric": "allocation-local-exact-slots-per-second",
        "metrics": metrics,
        "science_fingerprints_all_match": science_match,
        "selected_cpus": int(selected["cpus"]),
        "selected_exact_active_target": int(selected["exact_fit_slots"]),
        "target_formula": "exact_fit_slots[selected_cpus]",
        "throughput_objective": "maximize-completed-seeds-per-wall-clock",
        "resource_change_recommended": int(selected["cpus"]) != ANCHOR_CPUS,
        "per_seed_latency_gate_applied": False,
        "per_seed_latency_override_applied": False,
        "automatic_promotion_performed": False,
    }


def _result_bytes(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.resolve(strict=True).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} result is unavailable/invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} result is not one JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _authenticate_scheduler_detail(
    detail: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> tuple[int, str]:
    observation = scheduler_task_observation(detail)
    if observation is None or not scheduler_task_identity_matches(detail, expected):
        raise RuntimeError(f"{label} Scheduler detail changed sealed identity")
    task_id, status = observation
    if status != "completed":
        raise RuntimeError(f"{label} Scheduler task is not completed")
    return task_id, status


def evaluate_comparison(
    *,
    package_path: Path,
    anchor_remote_terminal_path: Path,
    anchor_result_path: Path,
    companion2_detail_path: Path,
    companion2_telemetry_path: Path,
    companion2_result_path: Path,
    companion3_detail_path: Path,
    companion3_telemetry_path: Path,
    companion3_result_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    package = validate_package(_read_json(package_path, "paired comparator package"))
    anchor_result, anchor_result_sha = _result_bytes(
        anchor_result_path, "4-CPU anchor"
    )
    anchor = _anchor_summary(
        package=package,
        remote_terminal=_read_json(
            anchor_remote_terminal_path, "task-87052 remote terminal"
        ),
        result=anchor_result,
        result_raw_sha256=anchor_result_sha,
    )
    summaries: dict[str, dict[str, Any]] = {"4": anchor}
    task_ids = {ANCHOR_TASK_ID}
    evidence: dict[str, Any] = {
        "4": {
            "task_id": ANCHOR_TASK_ID,
            "remote_terminal_sha256": anchor["remote_terminal_sha256"],
            "result_file_sha256": anchor_result_sha,
            "science_fingerprint": anchor["science_fingerprint"],
        }
    }
    inputs = {
        2: (
            companion2_detail_path,
            companion2_telemetry_path,
            companion2_result_path,
        ),
        3: (
            companion3_detail_path,
            companion3_telemetry_path,
            companion3_result_path,
        ),
    }
    for cpus, (detail_path, telemetry_path, result_path) in inputs.items():
        detail = _read_json(detail_path, f"{cpus}-CPU Scheduler detail")
        task_id, _ = _authenticate_scheduler_detail(
            detail, package["companions"][str(cpus)], label=f"{cpus}-CPU companion"
        )
        if task_id in task_ids:
            raise RuntimeError("paired comparator Scheduler task id was reused")
        task_ids.add(task_id)
        result, result_sha = _result_bytes(result_path, f"{cpus}-CPU companion")
        telemetry = validate_telemetry(
            _read_json(telemetry_path, f"{cpus}-CPU telemetry"),
            package=package,
            cpus=cpus,
            task_id=task_id,
            result=result,
            result_raw_sha256=result_sha,
        )
        summaries[str(cpus)] = {
            "cpus": cpus,
            "task_id": task_id,
            "wall_time_seconds": float(telemetry["wall_time_seconds"]),
            "process_tree_cpu_seconds": float(
                telemetry["process_tree_cpu_seconds"]
            ),
            "average_cores_used": float(telemetry["average_cores_used"]),
            "peak_rss_bytes": int(telemetry["peak_rss_bytes"]),
            "cpu_set": list(telemetry["cpu_set_after"]),
            "science_fingerprint": copy.deepcopy(
                telemetry["science_fingerprint"]
            ),
            "telemetry_sha256": telemetry["telemetry_sha256"],
            "all_resource_gates_passed": True,
        }
        evidence[str(cpus)] = {
            "task_id": task_id,
            "telemetry_sha256": telemetry["telemetry_sha256"],
            "result_file_sha256": result_sha,
            "science_fingerprint": telemetry["science_fingerprint"],
        }
    decision = decide_throughput(
        capacity=package["capacity"], summaries=summaries
    )
    unsigned = {
        "schema_version": COMPARISON_SCHEMA,
        "observed_at": _now(),
        "package_sha256": package["package_sha256"],
        "comparator_identity_sha256": package["comparator_identity"][
            "identity_sha256"
        ],
        "anchor_task_id": ANCHOR_TASK_ID,
        "seed": ANCHOR_SEED,
        "capacity_sha256": package["capacity"]["capacity_sha256"],
        "summaries": summaries,
        "evidence": evidence,
        "decision": decision,
        "prior_unpaired_terminal_sha256": PRIOR_UNPAIRED_REMOTE_TERMINAL_SHA256,
        "paired_seed_complexity_controlled": True,
        "science_payload_diff_only_resource_fields": True,
        "science_fingerprints_all_match": decision[
            "science_fingerprints_all_match"
        ],
        "per_seed_latency_gate_applied": False,
        "per_seed_latency_override_applied": False,
        "independent_promotion_review_required": True,
        "automatic_promotion_performed": False,
        "scheduler_write_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "comparison_sha256": canonical_sha256(unsigned)}
    sealed = validate_comparison(value)
    if output_path is not None:
        _atomic_json(output_path, sealed)
    return sealed


def validate_comparison(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "comparison_sha256"}
    required = {
        "schema_version",
        "observed_at",
        "package_sha256",
        "comparator_identity_sha256",
        "anchor_task_id",
        "seed",
        "capacity_sha256",
        "summaries",
        "evidence",
        "decision",
        "prior_unpaired_terminal_sha256",
        "paired_seed_complexity_controlled",
        "science_payload_diff_only_resource_fields",
        "science_fingerprints_all_match",
        "per_seed_latency_gate_applied",
        "per_seed_latency_override_applied",
        "independent_promotion_review_required",
        "automatic_promotion_performed",
        "scheduler_write_performed",
        "fea_submission_performed",
        "aedt_used",
        "comparison_sha256",
    }
    summaries = value.get("summaries")
    decision = value.get("decision")
    if (
        set(value) != required
        or value.get("schema_version") != COMPARISON_SCHEMA
        or value.get("comparison_sha256") != canonical_sha256(unsigned)
        or not _is_sha256(value.get("package_sha256"))
        or not _is_sha256(value.get("comparator_identity_sha256"))
        or value.get("anchor_task_id") != ANCHOR_TASK_ID
        or value.get("seed") != ANCHOR_SEED
        or not _is_sha256(value.get("capacity_sha256"))
        or not isinstance(summaries, Mapping)
        or set(summaries) != {"2", "3", "4"}
        or not isinstance(decision, Mapping)
        or decision.get("per_seed_latency_gate_applied") is not False
        or decision.get("per_seed_latency_override_applied") is not False
        or decision.get("automatic_promotion_performed") is not False
        or value.get("prior_unpaired_terminal_sha256")
        != PRIOR_UNPAIRED_REMOTE_TERMINAL_SHA256
        or value.get("paired_seed_complexity_controlled") is not True
        or value.get("science_payload_diff_only_resource_fields") is not True
        or value.get("science_fingerprints_all_match")
        is not decision.get("science_fingerprints_all_match")
        or value.get("per_seed_latency_gate_applied") is not False
        or value.get("per_seed_latency_override_applied") is not False
        or value.get("independent_promotion_review_required") is not True
        or value.get("automatic_promotion_performed") is not False
        or value.get("scheduler_write_performed") is not False
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
    ):
        raise RuntimeError("paired resource comparison seal mismatch")
    return copy.deepcopy(dict(value))


def validate_approval(
    value: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
    anchor_remote_terminal: Mapping[str, Any],
    cpus: int,
) -> dict[str, Any]:
    package = validate_package(package)
    anchor = validate_anchor_remote_terminal(anchor_remote_terminal, package=package)
    unsigned = {key: item for key, item in value.items() if key != "approval_sha256"}
    required = {
        "schema_version",
        "reviewed_at",
        "reviewer",
        "package_sha256",
        "anchor_task_id",
        "anchor_remote_terminal_sha256",
        "approved_companion_cpus",
        "source_and_companion_identity_reviewed",
        "complete_namespace_policy_reviewed",
        "anchor_terminal_assumptions_accepted",
        "maximum_scheduler_posts_per_companion",
        "retry_requires_new_package_and_approval",
        "automatic_promotion_allowed",
        "approval_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != APPROVAL_SCHEMA
        or value.get("approval_sha256") != canonical_sha256(unsigned)
        or not str(value.get("reviewed_at") or "")
        or len(str(value.get("reviewer") or "").strip()) < 3
        or value.get("package_sha256") != package["package_sha256"]
        or value.get("anchor_task_id") != ANCHOR_TASK_ID
        or value.get("anchor_remote_terminal_sha256")
        != anchor["remote_terminal_sha256"]
        or value.get("approved_companion_cpus") != [cpus]
        or cpus not in COMPANION_CPUS
        or value.get("source_and_companion_identity_reviewed") is not True
        or value.get("complete_namespace_policy_reviewed") is not True
        or value.get("anchor_terminal_assumptions_accepted") is not True
        or value.get("maximum_scheduler_posts_per_companion") != 1
        or value.get("retry_requires_new_package_and_approval") is not True
        or value.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("paired resource independent approval seal mismatch")
    return copy.deepcopy(dict(value))


class ComparatorSchedulerClient:
    """Bounded GET inventory/detail client plus one exact companion POST."""

    __slots__ = ("base_url", "timeout", "expected", "get_count", "post_count")

    def __init__(
        self,
        base_url: str,
        *,
        expected: Mapping[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.expected = copy.deepcopy(dict(expected)) if expected is not None else None
        self.get_count = 0
        self.post_count = 0
        if self.expected is not None:
            self._validate_expected(self.expected)

    @staticmethod
    def _validate_expected(task: Mapping[str, Any]) -> None:
        if (
            set(task) != REQUIRED_SCHEDULER_FIELDS
            or not str(task.get("name") or "").startswith(COMPARATOR_PREFIX)
            or not str(task.get("dedupe_key") or "").startswith(
                COMPARATOR_DEDUPE_PREFIX
            )
            or task.get("cpus") not in COMPANION_CPUS
        ):
            raise RuntimeError("comparator Scheduler client candidate escaped namespace")

    def _get_json(self, path: str, *, allow_not_found: bool = False) -> Any:
        parsed = urllib.parse.urlsplit(path)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or not (
                parsed.path == "/api/tasks"
                or re.fullmatch(r"/api/tasks/[1-9][0-9]*", parsed.path)
            )
            or (parsed.path != "/api/tasks" and parsed.query)
        ):
            raise RuntimeError("comparator Scheduler client refused foreign GET")
        for attempt in range(4):
            self.get_count += 1
            request = urllib.request.Request(self.base_url + path, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read(SCHEDULER_RESPONSE_LIMIT)
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_not_found:
                    return None
                if exc.code in RETRYABLE_HTTP and attempt < 3:
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                raise RuntimeError(
                    f"comparator Scheduler GET failed with HTTP {exc.code}"
                ) from exc
            if len(raw) == SCHEDULER_RESPONSE_LIMIT:
                raise RuntimeError("comparator Scheduler GET reached ambiguous cap")
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("comparator Scheduler GET returned invalid JSON") from exc
        raise AssertionError("comparator Scheduler GET retry fell through")

    def read_inventory_path(self, path: str) -> Any:
        parsed = urllib.parse.urlsplit(path)
        if parsed.path != "/api/tasks" or not parsed.query:
            raise RuntimeError("comparator inventory path is invalid")
        return self._get_json(path)

    def list_complete_namespace_tasks(self) -> list[dict[str, Any]]:
        rows = [
            *resource2._paged_namespace(self, prefix=PRODUCTION_PREFIX),  # noqa: SLF001
            *resource2._paged_namespace(self, prefix=COMPARATOR_PREFIX),  # noqa: SLF001
        ]
        ids = [int(row.get("id", row.get("task_id"))) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("production/comparator namespace task id overlaps")
        return rows

    def get_task(self, task_id: int) -> Mapping[str, Any] | None:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("comparator Scheduler detail id is invalid")
        value = self._get_json(f"/api/tasks/{task_id}", allow_not_found=True)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("comparator Scheduler detail is not an object")
        return value

    def submit_task(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.post_count != 0:
            raise RuntimeError("comparator client already attempted its one POST")
        if self.expected is None or dict(task) != self.expected:
            raise RuntimeError("comparator POST differs from exact bound candidate")
        self._validate_expected(task)
        body = json.dumps(
            task,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.post_count += 1
        request = urllib.request.Request(
            self.base_url + "/api/tasks",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(POST_RESPONSE_LIMIT)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"comparator Scheduler POST failed with HTTP {exc.code}"
            ) from exc
        if len(raw) == POST_RESPONSE_LIMIT:
            raise RuntimeError("comparator Scheduler POST reached ambiguous cap")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("comparator Scheduler POST returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("comparator Scheduler POST response is not an object")
        return value


def _namespace_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id", row.get("task_id")),
            "name": row.get("name"),
            "dedupe_key": row.get("dedupe_key"),
            "status": row.get("status"),
        }
        for row in rows
    ]


def _anchor_terminal_for_submission(
    terminal: Mapping[str, Any], *, package: Mapping[str, Any]
) -> dict[str, Any]:
    sealed = validate_anchor_remote_terminal(terminal, package=package)
    if (
        sealed.get("package_sha256") != SOURCE_PACKAGE_SHA256
        or sealed.get("submission_receipt_sha256")
        != SOURCE_SUBMISSION_RECEIPT_SHA256
        or sealed.get("task_id") != ANCHOR_TASK_ID
        or sealed.get("scheduler_status") != "completed"
        or sealed.get("promotion_eligible") is not True
        or sealed.get("bundle_id") != BUNDLE_ID
        or sealed.get("lane_shape") != 1
        or sealed.get("logical_seed_count") != 1
        or package["anchor"]["source_parent_task_sha256"]
        != SOURCE_PARENT_TASK_SHA256
    ):
        raise RuntimeError("task-87052 terminal assumptions are not satisfied")
    return sealed


def _validate_submission_receipt(
    value: Mapping[str, Any], *, package: Mapping[str, Any], cpus: int
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    required = {
        "schema_version",
        "observed_at",
        "apply",
        "package_sha256",
        "approval_sha256",
        "anchor_remote_terminal_sha256",
        "companion_cpus",
        "candidate_dedupe_key",
        "complete_namespace_row_count",
        "complete_namespace_max_task_id",
        "complete_namespace_sha256",
        "post_namespace_row_count",
        "post_namespace_max_task_id",
        "post_namespace_sha256",
        "source_match_count",
        "selected_candidate_match_count_before",
        "selected_candidate_match_count_after",
        "unexpected_seed_claim_count_before",
        "unexpected_seed_claim_count_after",
        "all_identities_revalidated_after_post",
        "task_id",
        "task_status",
        "submitted_count",
        "scheduler_get_count",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
        "allocation_mutation_count",
        "publication_count",
        "fea_submission_performed",
        "aedt_used",
        "receipt_sha256",
    }
    apply = value.get("apply") is True
    if (
        set(value) != required
        or value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("package_sha256") != package["package_sha256"]
        or not _is_sha256(value.get("approval_sha256"))
        or not _is_sha256(value.get("anchor_remote_terminal_sha256"))
        or value.get("companion_cpus") != cpus
        or value.get("candidate_dedupe_key")
        != package["companions"][str(cpus)]["dedupe_key"]
        or value.get("source_match_count") != 1
        or value.get("selected_candidate_match_count_before") != 0
        or value.get("unexpected_seed_claim_count_before") != 0
        or value.get("submitted_count") != int(apply)
        or value.get("scheduler_post_count") != int(apply)
        or value.get("scheduler_cancel_count") != 0
        or value.get("scheduler_preempt_count") != 0
        or value.get("allocation_mutation_count") != 0
        or value.get("publication_count") != 0
        or value.get("fea_submission_performed") is not False
        or value.get("aedt_used") is not False
        or (
            apply
            and (
                value.get("selected_candidate_match_count_after") != 1
                or value.get("unexpected_seed_claim_count_after") != 0
                or value.get("all_identities_revalidated_after_post") is not True
                or not isinstance(value.get("task_id"), int)
            )
        )
        or (
            not apply
            and (
                value.get("selected_candidate_match_count_after") is not None
                or value.get("unexpected_seed_claim_count_after") is not None
                or value.get("all_identities_revalidated_after_post") is not False
                or value.get("task_id") is not None
                or value.get("task_status") != "absent"
            )
        )
    ):
        raise RuntimeError("paired comparator submission receipt seal mismatch")
    return copy.deepcopy(dict(value))


def _submission_outcome(
    *,
    package: Mapping[str, Any],
    approval: Mapping[str, Any],
    anchor_terminal: Mapping[str, Any],
    scheduler: Any,
    cpus: int,
    apply: bool,
    receipt_out: Path | None,
) -> dict[str, Any]:
    package = validate_package(package)
    if package["launch_policy"]["launch_allowed"] is not True:
        raise RuntimeError(
            "paired comparator launch forbidden: "
            + str(package["launch_policy"]["block_reason"])
        )
    sealed_anchor = _anchor_terminal_for_submission(anchor_terminal, package=package)
    sealed_approval = validate_approval(
        approval,
        package=package,
        anchor_remote_terminal=sealed_anchor,
        cpus=cpus,
    )
    if apply and receipt_out is None:
        raise RuntimeError("paired comparator --apply requires --receipt-out")
    if receipt_out is not None and receipt_out.exists():
        raise RuntimeError("paired comparator receipt already exists; retry is forbidden")
    source_detail = scheduler.get_task(ANCHOR_TASK_ID)
    source_id, source_status = _authenticate_scheduler_detail(
        source_detail or {}, package["anchor"]["parent_task"], label="4-CPU anchor"
    )
    if source_id != ANCHOR_TASK_ID or source_status != "completed":
        raise RuntimeError("task-87052 live Scheduler detail is not exact completed")
    rows = scheduler.list_complete_namespace_tasks()
    source_rows = [
        row for row in rows if row.get("dedupe_key") == SOURCE_PARENT_DEDUPE
    ]
    candidate = package["companions"][str(cpus)]
    candidate_dedupe = candidate["dedupe_key"]
    candidate_rows = [
        row for row in rows if row.get("dedupe_key") == candidate_dedupe
    ]
    allowed_dedupes = {
        SOURCE_PARENT_DEDUPE,
        *(task["dedupe_key"] for task in package["companions"].values()),
    }
    seed_rows = [row for row in rows if resource2._row_claims_seed(row, ANCHOR_SEED)]  # noqa: SLF001
    unexpected = [
        row for row in seed_rows if row.get("dedupe_key") not in allowed_dedupes
    ]
    if len(source_rows) != 1 or candidate_rows or unexpected:
        raise RuntimeError("paired comparator namespace/dedupe isolation failed")
    # Any already-launched opposite companion must still be the exact package
    # identity.  It is allowed but never touched by this selected submission.
    for other_cpu in COMPANION_CPUS:
        if other_cpu == cpus:
            continue
        other = package["companions"][str(other_cpu)]
        matches = [row for row in rows if row.get("dedupe_key") == other["dedupe_key"]]
        if len(matches) > 1:
            raise RuntimeError("opposite comparator companion dedupe is repeated")
        if matches:
            other_id = int(matches[0].get("id", matches[0].get("task_id")))
            detail = scheduler.get_task(other_id)
            observation = scheduler_task_observation(detail or {})
            if observation is None or not scheduler_task_identity_matches(detail or {}, other):
                raise RuntimeError("opposite comparator companion identity drifted")
    before_projection = _namespace_projection(rows)
    before_max = max((int(item["id"]) for item in before_projection), default=0)
    task_id = None
    task_status = "absent"
    post_rows: list[dict[str, Any]] = []
    selected_after = None
    unexpected_after = None
    revalidated = False
    if apply:
        posted = scheduler.submit_task(candidate)
        observation = scheduler_task_observation(posted)
        if observation is None or not scheduler_task_identity_matches(posted, candidate):
            raise RuntimeError("paired comparator POST response identity drifted")
        task_id, task_status = observation
        detail = scheduler.get_task(task_id)
        if not scheduler_task_identity_matches(detail or {}, candidate):
            raise RuntimeError("paired comparator POST detail identity drifted")
        post_rows = scheduler.list_complete_namespace_tasks()
        matches = [
            row for row in post_rows if row.get("dedupe_key") == candidate_dedupe
        ]
        post_seed_rows = [
            row for row in post_rows if resource2._row_claims_seed(row, ANCHOR_SEED)  # noqa: SLF001
        ]
        post_unexpected = [
            row
            for row in post_seed_rows
            if row.get("dedupe_key") not in allowed_dedupes
        ]
        selected_after = len(matches)
        unexpected_after = len(post_unexpected)
        post_max = max(
            (int(row.get("id", row.get("task_id"))) for row in post_rows),
            default=0,
        )
        if (
            selected_after != 1
            or unexpected_after != 0
            or task_id <= before_max
            or post_max < task_id
        ):
            raise RuntimeError("paired comparator POST was not one additive identity")
        match_id = int(matches[0].get("id", matches[0].get("task_id")))
        if match_id != task_id:
            raise RuntimeError("paired comparator post-scan task id drifted")
        revalidated = True
    post_projection = _namespace_projection(post_rows)
    post_max = max((int(item["id"]) for item in post_projection), default=0)
    unsigned = {
        "schema_version": SUBMISSION_SCHEMA,
        "observed_at": _now(),
        "apply": bool(apply),
        "package_sha256": package["package_sha256"],
        "approval_sha256": sealed_approval["approval_sha256"],
        "anchor_remote_terminal_sha256": sealed_anchor["remote_terminal_sha256"],
        "companion_cpus": cpus,
        "candidate_dedupe_key": candidate_dedupe,
        "complete_namespace_row_count": len(rows),
        "complete_namespace_max_task_id": before_max,
        "complete_namespace_sha256": canonical_sha256(before_projection),
        "post_namespace_row_count": len(post_rows) if apply else None,
        "post_namespace_max_task_id": post_max if apply else None,
        "post_namespace_sha256": canonical_sha256(post_projection) if apply else None,
        "source_match_count": len(source_rows),
        "selected_candidate_match_count_before": len(candidate_rows),
        "selected_candidate_match_count_after": selected_after,
        "unexpected_seed_claim_count_before": len(unexpected),
        "unexpected_seed_claim_count_after": unexpected_after,
        "all_identities_revalidated_after_post": revalidated,
        "task_id": task_id,
        "task_status": task_status,
        "submitted_count": int(apply),
        "scheduler_get_count": int(getattr(scheduler, "get_count", 0)),
        "scheduler_post_count": int(getattr(scheduler, "post_count", 0)),
        "scheduler_cancel_count": 0,
        "scheduler_preempt_count": 0,
        "allocation_mutation_count": 0,
        "publication_count": 0,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    value = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    sealed = _validate_submission_receipt(value, package=package, cpus=cpus)
    if apply:
        if getattr(scheduler, "post_count", 0) != 1:
            raise RuntimeError("paired comparator did not perform exactly one POST")
        _atomic_json(receipt_out, sealed)  # type: ignore[arg-type]
    elif getattr(scheduler, "post_count", 0) != 0:
        raise RuntimeError("paired comparator dry-run performed a POST")
    return sealed


def submit(
    *,
    config_path: Path,
    source_config_path: Path,
    source_package_path: Path,
    source_submission_path: Path,
    offload_plan_path: Path,
    publication_receipt_path: Path,
    package_path: Path,
    anchor_remote_terminal_path: Path,
    approval_path: Path,
    cpus: int,
    scheduler_url: str,
    accounts_path: Path,
    scheduler_source: Path,
    publication_account: str,
    apply: bool,
    receipt_out: Path | None,
    dry_run_out: Path | None,
) -> dict[str, Any]:
    if apply and dry_run_out is not None:
        raise RuntimeError("--dry-run-out cannot be combined with --apply")
    rebuilt = build_package(
        config_path=config_path,
        source_config_path=source_config_path,
        source_package_path=source_package_path,
        source_submission_path=source_submission_path,
        offload_plan_path=offload_plan_path,
        publication_receipt_path=publication_receipt_path,
    )
    package = validate_package(_read_json(package_path, "paired comparator package"))
    if package != rebuilt:
        raise RuntimeError("paired comparator package no longer reproduces")
    source_package = _read_json(source_package_path, "task-87052 source package")
    source_package_unsigned = {
        key: item
        for key, item in source_package.items()
        if key != "package_sha256"
    }
    if (
        source_package.get("package_sha256") != SOURCE_PACKAGE_SHA256
        or source_package.get("package_sha256")
        != canonical_sha256(source_package_unsigned)
    ):
        raise RuntimeError("task-87052 source package seal drifted before submit")
    publication = _read_json(publication_receipt_path, "publication receipt")
    anchor_terminal = _read_json(
        anchor_remote_terminal_path, "task-87052 remote terminal"
    )
    approval = _read_json(approval_path, "paired comparator independent approval")
    candidate = package["companions"][str(cpus)]
    with scheduler_publication_transport(
        accounts_path=accounts_path,
        scheduler_source=scheduler_source,
        account_name=publication_account,
    ) as transport:
        shape_canary._live_ready(source_package, publication, transport)  # noqa: SLF001
        if int(getattr(transport, "write_count", 0)) != 0:
            raise RuntimeError("paired comparator READY recheck performed a remote write")
        value = _submission_outcome(
            package=package,
            approval=approval,
            anchor_terminal=anchor_terminal,
            scheduler=ComparatorSchedulerClient(scheduler_url, expected=candidate),
            cpus=cpus,
            apply=apply,
            receipt_out=receipt_out,
        )
    if not apply and dry_run_out is not None:
        _atomic_json(dry_run_out, value)
    return value


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--source-submission", type=Path, required=True)
    parser.add_argument("--offload-plan", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    render = sub.add_parser("render", help="render the immutable offline pair")
    _add_source_arguments(render)
    render.add_argument("--out", type=Path, required=True)

    validate = sub.add_parser("validate", help="validate an immutable package")
    validate.add_argument("--package", type=Path, required=True)

    submit_parser = sub.add_parser(
        "submit", help="dry-run by default; one companion POST only with --apply"
    )
    _add_source_arguments(submit_parser)
    submit_parser.add_argument("--package", type=Path, required=True)
    submit_parser.add_argument("--anchor-remote-terminal", type=Path, required=True)
    submit_parser.add_argument("--approval", type=Path, required=True)
    submit_parser.add_argument("--cpus", type=int, choices=COMPANION_CPUS, required=True)
    submit_parser.add_argument("--scheduler-url", default="http://127.0.0.1:8002")
    submit_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    submit_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    submit_parser.add_argument("--publication-account", default="r1jae262")
    submit_parser.add_argument("--apply", action="store_true")
    submit_parser.add_argument("--receipt-out", type=Path)
    submit_parser.add_argument("--dry-run-out", type=Path)

    evaluate = sub.add_parser("evaluate", help="seal the paired 2/3/4 result")
    evaluate.add_argument("--package", type=Path, required=True)
    evaluate.add_argument("--anchor-remote-terminal", type=Path, required=True)
    evaluate.add_argument("--anchor-result", type=Path, required=True)
    evaluate.add_argument("--companion2-detail", type=Path, required=True)
    evaluate.add_argument("--companion2-telemetry", type=Path, required=True)
    evaluate.add_argument("--companion2-result", type=Path, required=True)
    evaluate.add_argument("--companion3-detail", type=Path, required=True)
    evaluate.add_argument("--companion3-telemetry", type=Path, required=True)
    evaluate.add_argument("--companion3-result", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.operation == "render":
        value = build_package(
            config_path=args.config,
            source_config_path=args.source_config,
            source_package_path=args.source_package,
            source_submission_path=args.source_submission,
            offload_plan_path=args.offload_plan,
            publication_receipt_path=args.publication_receipt,
        )
        _atomic_json(args.out, value)
    elif args.operation == "validate":
        value = validate_package(_read_json(args.package, "paired comparator package"))
    elif args.operation == "submit":
        value = submit(
            config_path=args.config,
            source_config_path=args.source_config,
            source_package_path=args.source_package,
            source_submission_path=args.source_submission,
            offload_plan_path=args.offload_plan,
            publication_receipt_path=args.publication_receipt,
            package_path=args.package,
            anchor_remote_terminal_path=args.anchor_remote_terminal,
            approval_path=args.approval,
            cpus=args.cpus,
            scheduler_url=args.scheduler_url,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            publication_account=args.publication_account,
            apply=args.apply,
            receipt_out=args.receipt_out,
            dry_run_out=args.dry_run_out,
        )
    else:
        value = evaluate_comparison(
            package_path=args.package,
            anchor_remote_terminal_path=args.anchor_remote_terminal,
            anchor_result_path=args.anchor_result,
            companion2_detail_path=args.companion2_detail,
            companion2_telemetry_path=args.companion2_telemetry,
            companion2_result_path=args.companion2_result,
            companion3_detail_path=args.companion3_detail,
            companion3_telemetry_path=args.companion3_telemetry,
            companion3_result_path=args.companion3_result,
            output_path=args.out,
        )
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
