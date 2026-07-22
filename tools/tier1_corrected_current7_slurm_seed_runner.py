"""Run one immutable corrected-current7 Tier-1 seed in a CPU Slurm task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

try:
    from tier1_corrected_current7_receipt import (
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        canonical_sha256,
        expected_generation_artifacts,
        training_profile_sha256,
        validate_adapter_receipt,
    )
    from tier1_corrected_current7_slurm_bundle import (
        BUNDLE_SCHEMA,
        READY_SCHEMA,
        RELOCATION_SCHEMA,
        REMOTE_PREFLIGHT_SCHEMA,
        RESULT_SCHEMA,
        SEARCH_INTERFACE_SCHEMA,
        STATUS_SCHEMA,
        TASK_SCHEMA,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import (
        CORRECTED_GENERATION_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS,
        CURRENT_REQUIRED_MODEL_TARGETS_SHA256,
        CURRENT_TEMPERATURE_TARGETS,
        canonical_sha256,
        expected_generation_artifacts,
        training_profile_sha256,
        validate_adapter_receipt,
    )
    from tools.tier1_corrected_current7_slurm_bundle import (
        BUNDLE_SCHEMA,
        READY_SCHEMA,
        RELOCATION_SCHEMA,
        REMOTE_PREFLIGHT_SCHEMA,
        RESULT_SCHEMA,
        SEARCH_INTERFACE_SCHEMA,
        STATUS_SCHEMA,
        TASK_SCHEMA,
    )


THREAD_LIMIT_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
PHASE_B_SEMLOCK_STRESS_SCHEMA = "mft-tier1-current7-semlock-free-repeated-predict-v1"
PHASE_B_INFERENCE_POLICY = "family_specific_semaphore_free_sklearn_forest_v1"
PHASE_B_SEMLOCK_SAFE_RELEASE_COMMIT = "6ea0e17e5e028ebb8d89d910c3cec1a0f014dae5"
PHASE_B_SEMLOCK_SAFE_HELPER_SHA256 = (
    "c25afd39752a990c31e1b934c77282d88ac11dab1cad763d4f07c95e4b87dd80"
)


def optimizer_environment(
    base: Mapping[str, str], payload: Mapping[str, Any]
) -> dict[str, str]:
    """Return one fail-closed CPU/thread environment for the child optimizer."""

    threads = payload.get("inference_threads")
    scheduler_cpus = payload.get("scheduler_cpus")
    if (
        isinstance(threads, bool)
        or not isinstance(threads, int)
        or threads <= 0
        or (
            scheduler_cpus is not None
            and (
                isinstance(scheduler_cpus, bool)
                or not isinstance(scheduler_cpus, int)
                or scheduler_cpus <= 0
                or scheduler_cpus != threads
            )
        )
    ):
        raise RuntimeError("task CPU/thread execution contract mismatch")
    environment = dict(base)
    for variable in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        environment[variable] = str(threads)
    return environment


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def contained(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    path = (root / str(relative)).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"bundle path escaped immutable root: {relative}") from exc
    return path


def _verify_file(path: Path, record: Mapping[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("size", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"{label} fingerprint mismatch")


def _require_fail_closed(value: Mapping[str, Any], label: str) -> None:
    if any(
        value.get(field) is not False
        for field in (
            "production_eligible",
            "fea_submission_approved",
            "fea_submission_performed",
            "aedt_used",
            "automatic_promotion_allowed",
        )
    ):
        raise RuntimeError(f"{label} fail-closed policy mismatch")


def _validate_repair_gate(
    identity: Mapping[str, Any], label: str, *, stratum_map: bool
) -> None:
    repair_contracts = identity.get(
        "optimizer_repair_contract_sha256_by_fixed_primary_turns"
    )
    repair_contract = identity.get("optimizer_repair_contract_sha256")
    if (
        identity.get("launch_eligible") is not True
        or identity.get("offspring_physics_repair") is not True
        or identity.get("fixed_primary_turns_supported") != [5, 6]
        or identity.get("initial_repair_attested") is not True
        or identity.get("warm_repair_attested") is not True
        or identity.get("every_offspring_decode_repair_attested") is not True
        or identity.get("terminal_physical_replay_attested") is not True
        or (
            stratum_map
            and (
                not isinstance(repair_contracts, dict)
                or set(repair_contracts) != {"5", "6"}
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in repair_contracts.values()
                )
            )
        )
        or (
            not stratum_map
            and (not isinstance(repair_contract, str) or len(repair_contract) != 64)
        )
    ):
        raise RuntimeError(f"{label} fixed-turn offspring repair/replay gate mismatch")


def verify_payload(
    bundle: Path,
    payload_path: Path,
    payload_root: Path,
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = bundle.resolve(strict=True)
    payload_root = payload_root.resolve(strict=True)
    payload_path = payload_path.resolve(strict=True)
    if payload_path.name != "payload.json" or payload_root not in payload_path.parents:
        raise RuntimeError("scheduler payload escaped the scheduler run root")
    payload = read_json(payload_path)
    if payload.get("schema_version") != TASK_SCHEMA:
        raise RuntimeError("unsupported current7 task payload schema")
    if canonical_sha256(payload) != expected_payload_sha256:
        raise RuntimeError("current7 task payload SHA-256 mismatch")
    _require_fail_closed(payload, "task payload")
    _validate_repair_gate(
        {
            "launch_eligible": True,
            **{
                key: payload.get(key)
                for key in (
                    "offspring_physics_repair",
                    "fixed_primary_turns_supported",
                    "initial_repair_attested",
                    "warm_repair_attested",
                    "every_offspring_decode_repair_attested",
                    "terminal_physical_replay_attested",
                    "optimizer_repair_contract_sha256",
                )
            },
        },
        "task payload",
        stratum_map=False,
    )

    manifest_path = bundle / "bundle_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != payload.get("bundle_manifest_sha256"):
        raise RuntimeError("bundle manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("bundle_id") != payload.get("bundle_id")
        or manifest.get("task_schema_version") != TASK_SCHEMA
        or manifest.get("status_schema_version") != STATUS_SCHEMA
    ):
        raise RuntimeError("bundle/task schema identity mismatch")
    _require_fail_closed(manifest, "bundle manifest")
    identity = (manifest.get("adapter_receipt") or {}).get("identity") or {}
    _validate_repair_gate(identity, "bundle adapter receipt", stratum_map=True)
    if (
        manifest.get("constraint_version") != identity.get("constraint_version")
        or manifest.get("hard_spec") != identity.get("hard_spec")
        or manifest.get("hard_spec_sha256") != identity.get("hard_spec_sha256")
        or manifest.get("constraint_names") != identity.get("constraint_names")
        or manifest.get("temperature_targets") != list(CURRENT_TEMPERATURE_TARGETS)
        or manifest.get("temperature_contract_sha256")
        != identity.get("temperature_contract_sha256")
        or manifest.get("hard_constraint_contract_sha256")
        != identity.get("hard_constraint_contract_sha256")
    ):
        raise RuntimeError("bundle hard-constraint identity is not pinned to receipt")
    if (
        payload.get("hard_spec") != manifest.get("hard_spec")
        or payload.get("hard_spec_sha256") != manifest.get("hard_spec_sha256")
        or canonical_sha256(payload.get("hard_spec")) != payload.get("hard_spec_sha256")
        or payload.get("constraint_version") != manifest.get("constraint_version")
        or payload.get("constraint_names") != manifest.get("constraint_names")
    ):
        raise RuntimeError("task staged hard-spec identity mismatch")
    for field in (
        "adapter_manifest_sha256",
        "train_report_sha256",
        "dataset_sha256",
        "profile_canonical_sha256",
        "required_model_targets_sha256",
        "temperature_contract_sha256",
        "hard_constraint_contract_sha256",
    ):
        payload_field = (
            "adapter_manifest_sha256" if field == "adapter_manifest_sha256" else field
        )
        if payload.get(payload_field) != identity.get(field):
            raise RuntimeError(f"task payload {field} is not pinned to receipt")
    lane = payload.get("lane") or {}
    repair_contracts = identity[
        "optimizer_repair_contract_sha256_by_fixed_primary_turns"
    ]
    if payload.get("optimizer_repair_contract_sha256") != repair_contracts.get(
        str(lane.get("fixed_primary_turns"))
    ):
        raise RuntimeError("task repair contract is not pinned to its N1 stratum")
    scheduler_cpus = payload.get("scheduler_cpus")
    execution_threads_match = (
        payload.get("inference_threads") == 8
        if scheduler_cpus is None
        else (
            not isinstance(scheduler_cpus, bool)
            and isinstance(scheduler_cpus, int)
            and scheduler_cpus > 0
            and payload.get("inference_threads") == scheduler_cpus
        )
    )
    if (
        payload.get("generation_artifact_inventory_sha256")
        != manifest.get("generation_artifact_inventory_sha256")
        or payload.get("relocation_contract_sha256")
        != (manifest.get("relocation") or {}).get("contract_sha256")
        or payload.get("search_interface_schema_version") != SEARCH_INTERFACE_SCHEMA
        or payload.get("population") != 320
        or payload.get("max_generations") != 300
        or not execution_threads_match
        or payload.get("optimizer_processes") != 1
    ):
        raise RuntimeError("task payload execution contract mismatch")

    ready = read_json(bundle / "READY.json")
    if (
        ready.get("schema_version") != READY_SCHEMA
        or ready.get("bundle_id") != manifest["bundle_id"]
        or ready.get("bundle_manifest_sha256") != manifest_sha
        or ready.get("every_file_sha256_verified") is not True
        or ready.get("runtime_verified") is not True
        or ready.get("code_inventory_sha256") != manifest.get("code_inventory_sha256")
        or ready.get("relocation_contract_sha256")
        != manifest["relocation"]["contract_sha256"]
        or ready.get("remote_git_checkout_performed") is not False
    ):
        raise RuntimeError("bundle has no matching atomic READY seal")

    actual_packages = {
        name: importlib.metadata.version(name)
        for name in manifest["runtime"]["critical_packages"]
    }
    if actual_packages != manifest["runtime"]["critical_packages"]:
        raise RuntimeError("remote runtime package versions differ from bundle")

    # Code is small enough to re-hash in every task.  Model artifacts are
    # authenticated exactly once by the current7 search process and attested
    # through remote_preflight.json below.
    for relative, record in manifest["code_inventory"].items():
        _verify_file(contained(bundle, relative), record, f"code file {relative}")
    marker = contained(bundle, "artifacts/code/.source-revision")
    if marker.read_text(encoding="ascii").strip() != manifest["bundle_code_revision"]:
        raise RuntimeError("bundle source revision marker mismatch")

    relocation_path = contained(bundle, manifest["relocation"]["path"])
    if sha256_file(relocation_path) != manifest["relocation"]["sha256"]:
        raise RuntimeError("relocation file SHA-256 mismatch")
    relocation = read_json(relocation_path)
    if (
        relocation.get("schema_version") != RELOCATION_SCHEMA
        or canonical_sha256(relocation) != manifest["relocation"]["contract_sha256"]
        or relocation.get("source_absolute_paths_are_documentary_only") is not True
        or relocation.get("local_adapter_authentication_replayed_remotely") is not False
        or relocation.get("generation_report_bytes_mutated") is not False
        or relocation.get("remote_git_checkout_required") is not False
    ):
        raise RuntimeError("relocation semantic contract mismatch")
    paths = relocation.get("bundle_paths") or {}
    relocated_identity = relocation.get("relocated_identity") or {}
    if relocated_identity != identity | {
        "generation_artifacts": manifest["generation_artifacts"],
        "generation_artifact_inventory_sha256": manifest[
            "generation_artifact_inventory_sha256"
        ],
    }:
        raise RuntimeError("relocation/manifest identity mismatch")

    receipt_path = contained(bundle, paths["adapter_receipt"])
    if (
        sha256_file(receipt_path) != manifest["adapter_receipt"]["file_sha256"]
        or sha256_file(receipt_path) != payload["adapter_receipt_file_sha256"]
    ):
        raise RuntimeError("relocated adapter receipt fingerprint mismatch")
    observed_receipt = validate_adapter_receipt(read_json(receipt_path)).to_dict()
    if observed_receipt != identity:
        raise RuntimeError("relocated adapter receipt identity mismatch")

    exact_evidence = {
        "train_report": identity["train_report_sha256"],
        "candidate": identity["candidate_sha256"],
        "quality_status": identity["quality_status_sha256"],
        "dataset": identity["dataset_sha256"],
    }
    for key, digest in exact_evidence.items():
        if sha256_file(contained(bundle, paths[key])) != digest:
            raise RuntimeError(f"relocated {key} fingerprint mismatch")
    profile = read_json(contained(bundle, paths["profile"]))
    if training_profile_sha256(profile) != identity["profile_canonical_sha256"]:
        raise RuntimeError("relocated profile canonical fingerprint mismatch")
    report = read_json(contained(bundle, paths["train_report"]))
    if report.get("targets") != list(CORRECTED_GENERATION_TARGETS) or report.get(
        "artifacts"
    ) != {
        relative: record["sha256"]
        for relative, record in manifest["generation_artifacts"].items()
    }:
        raise RuntimeError("relocated generation report inventory mismatch")
    generation = contained(bundle, paths["generation"])
    for relative in expected_generation_artifacts():
        artifact = contained(generation, relative)
        record = manifest["generation_artifacts"][relative]
        if not artifact.is_file() or artifact.stat().st_size != record["size"]:
            raise RuntimeError(f"generation artifact missing/wrong size: {relative}")

    lane = payload.get("lane") or {}
    island = (manifest.get("islands") or {}).get(lane.get("island_id")) or {}
    profile = island.get("current7_profile") or {}
    topology_contract = profile.get("topology_evolution_contract") or {}
    topology_niche = topology_contract.get(
        "final1000_topology_niche_contract"
    )
    if (
        profile.get("sha256") != payload.get("island_profile_sha256")
        or profile.get("fixed_primary_turns") != lane.get("fixed_primary_turns")
        or profile.get("fixed_primary_turns_repair_required") is not True
        or profile.get("offspring_physics_repair_required") is not True
        or profile.get("terminal_physical_replay_required") is not True
        or profile.get("temperature_targets") != list(CURRENT_TEMPERATURE_TARGETS)
        or payload.get("topology_evolution_contract_sha256")
        != topology_contract.get("sha256")
        or payload.get("topology_niche_contract_sha256")
        != (None if topology_niche is None else topology_niche.get("sha256"))
    ):
        raise RuntimeError("current7 fixed-turn island profile mismatch")
    warm = island.get("warm") or {}
    for kind in ("artifact", "contract"):
        _verify_file(
            contained(bundle, warm[kind]["path"]),
            warm[kind],
            f"selected warm {kind}",
        )
    if warm["artifact"]["sha256"] != payload.get("warm_artifact_sha256") or warm[
        "contract"
    ]["sha256"] != payload.get("warm_contract_sha256"):
        raise RuntimeError("task warm identity mismatch")
    return payload, manifest, relocation


def _process_rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    return None


PHASE_B_CHILD_MARKER = "final1000-finite-multiseed-phase-b-v1"
PHASE_B_CHILD_CPUSET_ENV = "MFT_FINAL1000_PHASE_B_CHILD_CPUSET"
PHASE_B_CHILD_MARKER_ENV = "MFT_FINAL1000_PHASE_B_CHILD_PROTOCOL"
PHASE_B_CHILD_CPUS = 4
PHASE_B_CHILD_RESOURCE_TELEMETRY_SCHEMA = (
    "mft-tier1-final1000-phase-b-child-cpu-telemetry-v1"
)


def _process_tree_cpu_seconds() -> float | None:
    try:
        import resource

        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, OSError, ValueError):
        return None
    value = (
        float(own.ru_utime)
        + float(own.ru_stime)
        + float(children.ru_utime)
        + float(children.ru_stime)
    )
    return value if math.isfinite(value) and value >= 0 else None


def phase_b_child_resource_telemetry(
    *,
    wall_started: float,
    cpu_started: float | None,
) -> dict[str, Any]:
    """Measure this seed-runner plus its optimizer descendants."""

    wall = max(0.0, time.monotonic() - wall_started)
    cpu_finished = _process_tree_cpu_seconds()
    available = cpu_started is not None and cpu_finished is not None
    cpu = max(0.0, cpu_finished - cpu_started) if available else None
    capacity = wall * PHASE_B_CHILD_CPUS
    utilization = (
        cpu / capacity
        if cpu is not None and capacity > 0
        else (0.0 if cpu is not None else None)
    )
    return {
        "schema_version": PHASE_B_CHILD_RESOURCE_TELEMETRY_SCHEMA,
        "available": available,
        "measurement": (
            "resource.getrusage(self+children)-delta"
            if available
            else "unavailable-on-platform"
        ),
        "child_cpus": PHASE_B_CHILD_CPUS,
        "wall_time_seconds": wall,
        "process_tree_cpu_seconds": cpu,
        "cpu_capacity_seconds": capacity,
        "cpu_utilization_fraction": utilization,
    }


def apply_phase_b_child_affinity(
    environment: Mapping[str, str],
    *,
    platform_name: str | None = None,
    affinity_getter: Callable[[int], set[int]] | None = None,
    affinity_setter: Callable[[int, set[int]], None] | None = None,
) -> tuple[int, ...] | None:
    """Apply one declared disjoint child cpuset before optimizer creation.

    The Scheduler already starts the concurrent parent inside one exact,
    exclusive 16/32-CPU step.  Starting nested ``srun --exclusive`` steps from
    inside that owning step can deadlock on the CPUs held by the outer step, so
    Phase B partitions the inherited cgroup/affinity mask in userspace.  The
    optimizer inherits this exact mask from its fresh seed-runner subprocess.
    """

    marker = environment.get(PHASE_B_CHILD_MARKER_ENV)
    encoded = environment.get(PHASE_B_CHILD_CPUSET_ENV)
    if marker is None and encoded is None:
        return None
    if marker != PHASE_B_CHILD_MARKER or encoded is None:
        raise RuntimeError("Phase B child affinity declaration is incomplete")
    fields = encoded.split(",")
    if len(fields) != PHASE_B_CHILD_CPUS or any(
        not field.isascii() or not field.isdigit() for field in fields
    ):
        raise RuntimeError("Phase B child cpuset is not four canonical CPU ids")
    cpus = tuple(int(field) for field in fields)
    if len(set(cpus)) != PHASE_B_CHILD_CPUS or encoded != ",".join(map(str, cpus)):
        raise RuntimeError("Phase B child cpuset is not canonical and unique")
    selected_platform = platform_name or os.name
    getter = affinity_getter or getattr(os, "sched_getaffinity", None)
    setter = affinity_setter or getattr(os, "sched_setaffinity", None)
    if selected_platform != "posix" or getter is None or setter is None:
        raise RuntimeError("Phase B child affinity requires Linux sched affinity")
    inherited = set(getter(0))
    if not set(cpus).issubset(inherited):
        raise RuntimeError("Phase B child cpuset escaped the Scheduler affinity mask")
    setter(0, set(cpus))
    if set(getter(0)) != set(cpus):
        raise RuntimeError("Phase B child cpuset could not be enforced exactly")
    return cpus


def _payload_sha_matches(value: Mapping[str, Any]) -> bool:
    expected = value.get("payload_sha256")
    unsigned = {key: item for key, item in value.items() if key != "payload_sha256"}
    return isinstance(expected, str) and expected == canonical_sha256(unsigned)


def validate_remote_preflight(
    value: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    optimizer_pid: int,
) -> int:
    lane = payload["lane"]
    reported_rss = value.get("observed_peak_rss_bytes")
    if isinstance(reported_rss, bool) or not isinstance(reported_rss, int):
        raise RuntimeError("remote preflight RSS observation is unavailable")
    live_rss = _process_rss_bytes(optimizer_pid)
    observed_rss = max(reported_rss, live_rss or 0)
    semlock = value.get("semlock_safe_prediction_stress") or {}
    semlock_unsigned = {
        key: item for key, item in semlock.items() if key != "sha256"
    }
    if (
        value.get("schema_version") != REMOTE_PREFLIGHT_SCHEMA
        or not _payload_sha_matches(value)
        or value.get("status") != "passed"
        or value.get("bundle_id") != payload["bundle_id"]
        or value.get("seed") != payload["seed"]
        or value.get("island_id") != lane["island_id"]
        or value.get("fixed_primary_turns") != lane["fixed_primary_turns"]
        or value.get("optimizer_pid") != optimizer_pid
        or value.get("optimizer_processes") != 1
        or value.get("model_mapping_instances") != 1
        or value.get("full_generation_authentication_passes") != 1
        or value.get("authenticated_artifact_count")
        != len(expected_generation_artifacts())
        or value.get("generation_artifact_inventory_sha256")
        != manifest["generation_artifact_inventory_sha256"]
        or value.get("adapter_manifest_sha256") != payload["adapter_manifest_sha256"]
        or value.get("train_report_sha256") != payload["train_report_sha256"]
        or value.get("dataset_sha256") != payload["dataset_sha256"]
        or value.get("profile_canonical_sha256") != payload["profile_canonical_sha256"]
        or value.get("temperature_contract_sha256")
        != payload["temperature_contract_sha256"]
        or value.get("hard_constraint_contract_sha256")
        != payload["hard_constraint_contract_sha256"]
        or value.get("stage_spec_sha256") != payload["hard_spec_sha256"]
        or value.get("island_profile_sha256") != payload["island_profile_sha256"]
        or value.get("warm_artifact_sha256") != payload["warm_artifact_sha256"]
        or value.get("warm_contract_sha256") != payload["warm_contract_sha256"]
        or value.get("loaded_model_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or value.get("loaded_model_targets_sha256")
        != CURRENT_REQUIRED_MODEL_TARGETS_SHA256
        or value.get("temperature_targets") != list(CURRENT_TEMPERATURE_TARGETS)
        or value.get("inference_threads") != payload["inference_threads"]
        or value.get("scheduler_cpus")
        != payload.get("scheduler_cpus", payload["inference_threads"])
        or value.get("optimizer_repair_contract_sha256")
        != payload["optimizer_repair_contract_sha256"]
        or (
            "topology_evolution_contract_sha256" in payload
            and value.get("topology_evolution_contract_sha256")
            != payload["topology_evolution_contract_sha256"]
        )
        or (
            "topology_niche_contract_sha256" in payload
            and value.get("topology_niche_contract_sha256")
            != payload["topology_niche_contract_sha256"]
        )
        or semlock.get("sha256") != canonical_sha256(semlock_unsigned)
        or semlock.get("rounds") != 8
        or semlock.get("prediction_call_count")
        != 8 * len(CURRENT_REQUIRED_MODEL_TARGETS)
        or semlock.get("sklearn_extratrees_n_jobs") != 1
        or semlock.get("semaphore_entry_growth_count") != 0
        or semlock.get("enospc_observed") is not False
        or semlock.get("stress_passed") is not True
        or value.get("offspring_physics_repair") is not True
        or value.get("initial_repair_attested") is not True
        or value.get("warm_repair_attested") is not True
        or value.get("offspring_repair_operator_installed") is not True
        or value.get("offspring_repair_operator_contract_sha256")
        != payload["optimizer_repair_contract_sha256"]
        or value.get("offspring_repair_execution_status")
        != "deferred_until_optimizer_execution"
        or value.get("every_offspring_decode_repair_attested") is not False
        or value.get("terminal_physical_replay_required") is not True
        or value.get("legacy_feedback_wrapper_used") is not False
        or value.get("local_adapter_authentication_replayed") is not False
        or observed_rss <= 0
        or observed_rss > payload["maximum_peak_rss_bytes"]
    ):
        raise RuntimeError("remote model-load/RSS/repair preflight gate mismatch")
    _require_fail_closed(value, "remote preflight")
    return observed_rss


def validate_phase_b_semlock_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require the repeated-predict SemLock gate for concurrent Phase B."""

    binding = value.get("inference_binding")
    stress = value.get("semlock_free_repeated_predict")
    release = value.get("semlock_safe_inference_release")
    if (
        not isinstance(binding, dict)
        or not isinstance(stress, dict)
        or not isinstance(release, dict)
    ):
        raise RuntimeError("Phase B SemLock preflight evidence is missing")
    stress_unsigned = {key: item for key, item in stress.items() if key != "sha256"}
    family_threads = binding.get("family_threads")
    families = binding.get("families")
    semaphore_free_families = binding.get("semaphore_free_families")
    release_unsigned = {key: item for key, item in release.items() if key != "sha256"}
    if (
        binding.get("target_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or not isinstance(binding.get("model_count"), int)
        or binding["model_count"] < len(CURRENT_REQUIRED_MODEL_TARGETS)
        or not isinstance(families, list)
        or not families
        or families != sorted(set(families))
        or not isinstance(family_threads, dict)
        or not isinstance(semaphore_free_families, list)
        or not semaphore_free_families
        or semaphore_free_families
        != sorted(set(families) & {"extratrees", "randomforest"})
        or any(family_threads.get(family) != 1 for family in semaphore_free_families)
        or binding.get("semaphore_free_sklearn_forest") is not True
        or binding.get("policy") != PHASE_B_INFERENCE_POLICY
        or stress.get("schema_version") != PHASE_B_SEMLOCK_STRESS_SCHEMA
        or stress.get("sha256") != canonical_sha256(stress_unsigned)
        or stress.get("status") != "passed"
        or not isinstance(stress.get("repeats"), int)
        or stress["repeats"] < 2
        or stress.get("target_count") != len(CURRENT_REQUIRED_MODEL_TARGETS)
        or stress.get("predict_call_count")
        != stress["repeats"] * len(CURRENT_REQUIRED_MODEL_TARGETS)
        or stress.get("covered_sklearn_forest_count", 0)
        < len(CURRENT_REQUIRED_MODEL_TARGETS)
        or stress.get("joblib_thread_pool_construction_attempt_count") != 0
        or stress.get("multiprocessing_semlock_construction_attempt_count") != 0
        or stress.get("sklearn_forest_n_jobs") != 1
        or stress.get("tmp_isolation_claimed_as_fix") is not False
        or stress.get("direct_enospc_cause")
        != "joblib-threadpool-simplequeue-semlock-churn"
        or release.get("sha256") != canonical_sha256(release_unsigned)
        or release.get("required_release_commit") != PHASE_B_SEMLOCK_SAFE_RELEASE_COMMIT
        or release.get("smoke_schema") != "mft-tier1-semlock-safe-inference-smoke-v1"
        or release.get("helper_relative_path")
        != "tools/tier1_semlock_safe_inference_smoke.py"
        or release.get("helper_sha256") != PHASE_B_SEMLOCK_SAFE_HELPER_SHA256
        or release.get("semaphore_free_sklearn_families")
        != ["extratrees", "randomforest"]
        or release.get("helper_code_inventory_authenticated") is not True
        or release.get("tmp_isolation_claimed_as_enospc_fix") is not False
    ):
        raise RuntimeError("Phase B repeated-predict SemLock gate mismatch")
    return dict(stress)


def validate_result(
    result: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    lane = payload["lane"]
    identity = manifest["adapter_receipt"]["identity"]
    thermal_constraints = [
        name
        for name in result.get("constraint_names", [])
        if str(name).startswith("temperature_robust_limit:")
    ]
    expected_constraints = [
        f"temperature_robust_limit:{target}" for target in CURRENT_TEMPERATURE_TARGETS
    ]
    topology = result.get("optimizer_topology_evolution_audit") or {}
    topology_unsigned = {
        key: item for key, item in topology.items() if key != "sha256"
    }
    topology_niche_required = (
        payload.get("topology_niche_contract_sha256") is not None
    )
    topology_contract = result.get("optimizer_topology_evolution_contract") or {}
    topology_contract_unsigned = {
        key: item for key, item in topology_contract.items() if key != "sha256"
    }
    result_niche_contract = topology_contract.get(
        "final1000_topology_niche_contract"
    )
    semlock = result.get("semlock_safe_prediction_stress") or {}
    semlock_unsigned = {
        key: item for key, item in semlock.items() if key != "sha256"
    }
    hard_spec = result.get("hard_spec")
    artifact_inventory = result.get("artifact_inventory")
    constraint_names = result.get("constraint_names")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or not _payload_sha_matches(result)
        or result.get("bundle_id") != payload["bundle_id"]
        or result.get("seed") != payload["seed"]
        or result.get("island_id") != lane["island_id"]
        or result.get("population") != payload["population"]
        or result.get("max_generations") != payload["max_generations"]
        or result.get("evaluated_generations") != payload["max_generations"]
        or result.get("completed_generations") != payload["max_generations"] + 1
        or result.get("optimizer_processes") != 1
        or result.get("optimizer_processes")
        != payload.get("optimizer_processes", 1)
        or result.get("inference_threads") != payload["inference_threads"]
        or result.get("scheduler_cpus")
        != payload.get("scheduler_cpus", payload["inference_threads"])
        or result.get("generation_artifact_inventory_sha256")
        != manifest["generation_artifact_inventory_sha256"]
        or result.get("adapter_manifest_sha256") != payload["adapter_manifest_sha256"]
        or result.get("train_report_sha256") != payload["train_report_sha256"]
        or result.get("dataset_sha256") != payload["dataset_sha256"]
        or result.get("profile_canonical_sha256") != payload["profile_canonical_sha256"]
        or result.get("temperature_contract_sha256")
        != payload["temperature_contract_sha256"]
        or result.get("hard_constraint_contract_sha256")
        != payload["hard_constraint_contract_sha256"]
        or not isinstance(hard_spec, dict)
        or not hard_spec
        or result.get("stage_spec_sha256") != canonical_sha256(hard_spec)
        or hard_spec != identity.get("hard_spec")
        or result.get("stage_spec_sha256") != identity.get("hard_spec_sha256")
        or result.get("constraint_version") != identity.get("constraint_version")
        or not isinstance(constraint_names, list)
        or not constraint_names
        or any(not isinstance(name, str) or not name for name in constraint_names)
        or len(constraint_names) != len(set(constraint_names))
        or constraint_names != identity.get("constraint_names")
        or not isinstance(artifact_inventory, dict)
        or not artifact_inventory
        or result.get("artifact_inventory_sha256")
        != canonical_sha256(artifact_inventory)
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record.get("path")
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or record["size_bytes"] <= 0
            for record in artifact_inventory.values()
        )
        or result.get("island_profile_sha256") != payload["island_profile_sha256"]
        or result.get("warm_artifact_sha256") != payload["warm_artifact_sha256"]
        or result.get("warm_contract_sha256") != payload["warm_contract_sha256"]
        or result.get("temperature_targets") != list(CURRENT_TEMPERATURE_TARGETS)
        or thermal_constraints != expected_constraints
        or result.get("fixed_primary_turns") != lane["fixed_primary_turns"]
        or result.get("terminal_population_primary_turn_values")
        != [lane["fixed_primary_turns"]]
        or result.get("terminal_population_fixed_primary_turns_verified") is not True
        or result.get("offspring_physics_repair") is not True
        or result.get("initial_population_repair_attested") is not True
        or result.get("warm_start_repair_attested") is not True
        or result.get("every_offspring_decode_repair_attested") is not True
        or isinstance(result.get("offspring_repair_operator_call_count"), bool)
        or not isinstance(result.get("offspring_repair_operator_call_count"), int)
        or result.get("offspring_repair_operator_call_count")
        < payload["max_generations"]
        or isinstance(result.get("offspring_repair_operator_row_count"), bool)
        or not isinstance(result.get("offspring_repair_operator_row_count"), int)
        or result.get("offspring_repair_operator_row_count") <= 0
        or result.get("terminal_physical_replay_attested") is not True
        or result.get("optimizer_repair_contract_sha256")
        != payload["optimizer_repair_contract_sha256"]
        or topology_contract.get("sha256")
        != canonical_sha256(topology_contract_unsigned)
        or result.get("topology_evolution_contract_sha256")
        != topology_contract.get("sha256")
        or result.get("topology_niche_contract_sha256")
        != (
            None
            if result_niche_contract is None
            else result_niche_contract.get("sha256")
        )
        or (
            "topology_evolution_contract_sha256" in payload
            and result.get("topology_evolution_contract_sha256")
            != payload["topology_evolution_contract_sha256"]
        )
        or (
            "topology_niche_contract_sha256" in payload
            and result.get("topology_niche_contract_sha256")
            != payload["topology_niche_contract_sha256"]
        )
        or result.get("semlock_safe_prediction_stress_sha256")
        != semlock.get("sha256")
        or result.get("semlock_safe_prediction_stress_attested") is not True
        or semlock.get("sha256") != canonical_sha256(semlock_unsigned)
        or semlock.get("rounds") != 8
        or semlock.get("prediction_call_count")
        != 8 * len(CURRENT_REQUIRED_MODEL_TARGETS)
        or semlock.get("sklearn_extratrees_n_jobs") != 1
        or semlock.get("semaphore_entry_growth_count") != 0
        or semlock.get("enospc_observed") is not False
        or semlock.get("stress_passed") is not True
        or topology.get("sha256") != canonical_sha256(topology_unsigned)
        or topology.get("migration_events", 0) < 1
        or topology.get("paired_parent_pairs_emitted", 0) < 1
        or topology.get("survival_calls", 0) < 2
        or topology.get("terminal_epsilon_zero") is not True
        or topology.get("all_required_topologies_preserved") is not True
        or (
            topology_niche_required
            and (
                topology.get("exact_topology_quota_every_generation_verified")
                is not True
                or topology.get("cross_36x37_pairs_emitted", 0) < 1
                or topology.get("cross_36x37_pair_fraction", 0.0) < 0.39
                or topology.get("cross_36x37_offspring_attributed", 0) < 1
                or topology.get("topology_niche_diversity_budget") is None
            )
        )
    ):
        raise RuntimeError("terminal current7 result/replay seal mismatch")
    _require_fail_closed(result, "terminal result")


def _optimizer_command(
    bundle: Path,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    relocation: Mapping[str, Any],
    output: Path,
    preflight_path: Path,
) -> list[str]:
    execution = manifest["search_execution"]
    if execution.get("schema_version") != SEARCH_INTERFACE_SCHEMA:
        raise RuntimeError("unsupported current7 search interface")
    entrypoint = contained(bundle, execution["entrypoint"])
    paths = relocation["bundle_paths"]
    lane = payload["lane"]
    island = manifest["islands"][lane["island_id"]]
    profile = island["current7_profile"]
    warm = island["warm"]
    command = [
        sys.executable,
        "-u",
        str(entrypoint),
        "search-seed",
        "--bundle-root",
        str(bundle),
        "--relocation",
        str(contained(bundle, manifest["relocation"]["path"])),
        "--adapter-receipt",
        str(contained(bundle, paths["adapter_receipt"])),
        "--registry",
        str(contained(bundle, paths["registry"])),
        "--generation",
        str(contained(bundle, paths["generation"])),
        "--dataset",
        str(contained(bundle, paths["dataset"])),
        "--profile",
        str(contained(bundle, paths["profile"])),
        "--warm-start",
        str(contained(bundle, warm["artifact"]["path"])),
        "--warm-contract",
        str(contained(bundle, warm["contract"]["path"])),
        "--output",
        str(output),
        "--remote-preflight",
        str(preflight_path),
        "--bundle-id",
        str(payload["bundle_id"]),
        "--island-id",
        str(lane["island_id"]),
        "--island-profile-sha256",
        str(payload["island_profile_sha256"]),
        "--seed",
        str(payload["seed"]),
        "--population",
        str(payload["population"]),
        "--max-generations",
        str(payload["max_generations"]),
        "--inference-threads",
        str(payload["inference_threads"]),
        "--scheduler-cpus",
        str(payload.get("scheduler_cpus", payload["inference_threads"])),
        "--fixed-primary-turns",
        str(lane["fixed_primary_turns"]),
        "--optimizer-termination-strategy",
        str(profile["optimizer_termination_strategy"]),
        "--optimizer-resonance-scale-hz",
        str(profile["optimizer_resonance_scale_Hz"]),
        "--optimizer-llt-scale-uh",
        str(profile["optimizer_llt_scale_uH"]),
        "--optimizer-all-thermal-scale-c",
        str(profile["optimizer_all_current7_thermal_scale_C"]),
        "--optimizer-repair-contract-sha256",
        str(payload["optimizer_repair_contract_sha256"]),
        "--stage-spec-json",
        json.dumps(
            payload["hard_spec"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        "--stage-spec-sha256",
        str(payload["hard_spec_sha256"]),
    ]
    resonance_allowance = profile.get("optimizer_resonance_allowance_Hz")
    llt_allowance = profile.get("optimizer_llt_allowance_uH")
    if (resonance_allowance is None) != (llt_allowance is None):
        raise RuntimeError("island optimizer allowance pair is incomplete")
    if resonance_allowance is not None:
        command.extend(
            [
                "--optimizer-resonance-allowance-hz",
                str(resonance_allowance),
                "--optimizer-llt-allowance-uh",
                str(llt_allowance),
            ]
        )
    return command


def run(
    bundle: Path,
    payload_path: Path,
    payload_root: Path,
    expected_payload_sha256: str,
    heartbeat_seconds: float = 5.0,
) -> int:
    bundle = bundle.resolve(strict=True)
    payload, manifest, relocation = verify_payload(
        bundle, payload_path, payload_root, expected_payload_sha256
    )
    phase_b_cpu_set = apply_phase_b_child_affinity(os.environ)
    phase_b_wall_started = time.monotonic()
    phase_b_cpu_started = (
        _process_tree_cpu_seconds() if phase_b_cpu_set is not None else None
    )
    task_id = str(os.environ.get("SLURM_SCHED_TASK_ID") or f"pid-{os.getpid()}")
    seed = int(payload["seed"])
    output = bundle / "runs" / f"task-{task_id}" / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    # Sequential Phase A retains its exact legacy journal path.  Concurrent
    # Phase B must never let sibling subprocesses overwrite one shared file,
    # so its opt-in marker places the mutable legacy status inside the seed's
    # already isolated output directory.  The parent later seals this status
    # into the immutable child receipt without changing task/payload identity.
    status_path = (
        output / "legacy_seed_status.json"
        if phase_b_cpu_set is not None
        else output.parent / "seed_status.json"
    )
    preflight_path = output / manifest["search_execution"]["remote_preflight_filename"]
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "state": "starting",
        "phase": "static_bundle_verified",
        "started_at": now(),
        "updated_at": now(),
        "terminal": False,
        "ramp_gate_passed": False,
        "task_id": task_id,
        "seed": seed,
        "island_id": payload["lane"]["island_id"],
        "bundle_id": payload["bundle_id"],
        "bundle_manifest_sha256": payload["bundle_manifest_sha256"],
        "payload_sha256": expected_payload_sha256,
        "optimizer_processes": 1,
        "inference_threads": payload["inference_threads"],
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    if phase_b_cpu_set is not None:
        status["phase_b_cpu_set"] = list(phase_b_cpu_set)
        status["phase_b_child_journal_isolated"] = True
    atomic_json(status_path, status)
    command = _optimizer_command(
        bundle, payload, manifest, relocation, output, preflight_path
    )
    environment = optimizer_environment(os.environ, payload)
    process = subprocess.Popen(
        command,
        cwd=contained(bundle, "artifacts/code"),
        env=environment,
    )
    stop = threading.Event()

    def terminate(_signum, _frame):
        stop.set()
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    status.update(
        state="running",
        phase="remote_model_load_pending",
        optimizer_pid=process.pid,
        optimizer_started=True,
        updated_at=now(),
    )
    atomic_json(status_path, status)
    deadline = time.monotonic() + float(payload["preflight_timeout_seconds"])
    try:
        while not preflight_path.is_file():
            if process.poll() is not None:
                raise RuntimeError(
                    "optimizer exited before remote model-load preflight"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("remote model-load preflight timed out")
            status["updated_at"] = now()
            status["runner_pid"] = os.getpid()
            atomic_json(status_path, status)
            stop.wait(min(heartbeat_seconds, 1.0))
        preflight = read_json(preflight_path)
        observed_rss = validate_remote_preflight(
            preflight,
            payload=payload,
            manifest=manifest,
            optimizer_pid=process.pid,
        )
        semlock_stress = (
            validate_phase_b_semlock_preflight(preflight)
            if phase_b_cpu_set is not None
            else None
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        failed = {
            **status,
            "state": "failed",
            "phase": "remote_model_load_failed",
            "terminal": True,
            "ramp_gate_passed": False,
            "failure": f"{type(exc).__name__}:{exc}",
            "exit_code": 72,
            "finished_at": now(),
            "updated_at": now(),
        }
        if phase_b_cpu_set is not None:
            failed["phase_b_child_resource_telemetry"] = (
                phase_b_child_resource_telemetry(
                    wall_started=phase_b_wall_started,
                    cpu_started=phase_b_cpu_started,
                )
            )
        atomic_json(status_path, failed)
        return 72
    status.update(
        phase="remote_model_load_passed_optimizer_running",
        ramp_gate_passed=True,
        ramp_gate_passed_at=now(),
        loaded_model_count=preflight["loaded_model_count"],
        full_generation_authentication_passes=preflight[
            "full_generation_authentication_passes"
        ],
        authenticated_artifact_count=preflight["authenticated_artifact_count"],
        phase_b_semlock_stress_sha256=(
            semlock_stress["sha256"] if semlock_stress is not None else None
        ),
        observed_peak_rss_bytes=observed_rss,
        updated_at=now(),
    )
    atomic_json(status_path, status)

    while process.poll() is None:
        status["updated_at"] = now()
        live_rss = _process_rss_bytes(process.pid)
        if live_rss is not None:
            status["observed_peak_rss_bytes"] = max(
                int(status["observed_peak_rss_bytes"]), live_rss
            )
        atomic_json(status_path, status)
        stop.wait(heartbeat_seconds)
    exit_code = int(process.returncode)
    terminal = dict(status)
    terminal.update(
        terminal=True,
        ramp_gate_passed=False,
        finished_at=now(),
        updated_at=now(),
        state="failed",
        phase="terminal",
        exit_code=exit_code,
    )
    result_path = output / manifest["search_execution"]["result_filename"]
    if exit_code == 0 and result_path.is_file():
        result = read_json(result_path)
        try:
            validate_result(result, payload=payload, manifest=manifest)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            terminal["failure"] = f"{type(exc).__name__}:{exc}"
            exit_code = 70
        else:
            terminal.update(
                state="completed",
                result_path=str(result_path),
                result_sha256=sha256_file(result_path),
                completed_generations=result["completed_generations"],
                evaluated_generations=result["evaluated_generations"],
                feasible_pareto_count=int(result.get("feasible_pareto_count", 0)),
            )
    elif exit_code == 0:
        terminal["failure"] = "optimizer exited zero without result.json"
        exit_code = 71
    else:
        terminal["failure"] = f"optimizer exit code {exit_code}"
    terminal["exit_code"] = exit_code
    if phase_b_cpu_set is not None:
        terminal["phase_b_child_resource_telemetry"] = phase_b_child_resource_telemetry(
            wall_started=phase_b_wall_started,
            cpu_started=phase_b_cpu_started,
        )
    atomic_json(status_path, terminal)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.heartbeat_seconds) or args.heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be finite and positive")
    return run(
        args.bundle_root,
        args.payload,
        args.payload_root,
        args.payload_sha256,
        args.heartbeat_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
