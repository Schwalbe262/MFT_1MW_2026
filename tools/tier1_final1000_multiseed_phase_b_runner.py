"""Run 1..8 final1000 seeds concurrently inside one exact parent step.

Each seed is a fresh authenticated single-seed subprocess with a disjoint
four-CPU Linux affinity mask and its own payload, mutable legacy journal,
result directory, and immutable terminal receipt.  Child completions may
arrive in any order, but receipts become consumer-visible only as a durable
contiguous ordinal prefix.  No Scheduler, AEDT, FEA, or remote API is used.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    import tier1_final1000_multiseed_lane_runner as phase_a_runner
    from tier1_corrected_current7_slurm_bundle import STATUS_SCHEMA
    from tier1_corrected_current7_slurm_seed_runner import (
        PHASE_B_CHILD_CPUSET_ENV,
        PHASE_B_CHILD_MARKER,
        PHASE_B_CHILD_MARKER_ENV,
    )
    from tier1_final1000_multiseed_contract import canonical_sha256, now
    from tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        CHILD_RECEIPT_SCHEMA,
        CHILD_RESOURCE_TELEMETRY_SCHEMA,
        CPU_ISOLATION,
        PARENT_MEMORY_EVIDENCE_FILENAME,
        PARENT_MEMORY_EVIDENCE_SCHEMA,
        PROTOCOL_VERSION,
        RUNTIME_ISOLATION,
        TASK_STATUS_SCHEMA,
        batch_manifest_from_payload,
        seal_child_receipt,
        seal_parent_memory_evidence,
        seal_task_status,
        validate_batch_manifest,
        validate_batch_payload,
        validate_child_resource_telemetry,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_final1000_multiseed_lane_runner as phase_a_runner
    from tools.tier1_corrected_current7_slurm_bundle import STATUS_SCHEMA
    from tools.tier1_corrected_current7_slurm_seed_runner import (
        PHASE_B_CHILD_CPUSET_ENV,
        PHASE_B_CHILD_MARKER,
        PHASE_B_CHILD_MARKER_ENV,
    )
    from tools.tier1_final1000_multiseed_contract import (
        canonical_sha256,
        now,
    )
    from tools.tier1_final1000_multiseed_phase_b_contract import (
        CHILD_CPUS,
        CHILD_MEMORY_MB,
        CHILD_RECEIPT_SCHEMA,
        CHILD_RESOURCE_TELEMETRY_SCHEMA,
        CPU_ISOLATION,
        PARENT_MEMORY_EVIDENCE_FILENAME,
        PARENT_MEMORY_EVIDENCE_SCHEMA,
        PROTOCOL_VERSION,
        RUNTIME_ISOLATION,
        TASK_STATUS_SCHEMA,
        batch_manifest_from_payload,
        seal_child_receipt,
        seal_parent_memory_evidence,
        seal_task_status,
        validate_batch_manifest,
        validate_batch_payload,
        validate_child_resource_telemetry,
    )


DEFAULT_HEARTBEAT_SECONDS = 5.0
DEFAULT_CHILD_TERMINATION_GRACE_SECONDS = 30.0
LANE_FATAL_EXIT_CODE = 74
STOPPED_EXIT_CODE = 75
CHILD_LAUNCH_FAILURE_EXIT_CODE = 76
CHILD_FAILURE_EXIT_CODE = 77
FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGABRT)
LEGACY_STATUS_FILENAME = "legacy_seed_status.json"
CHILD_STDOUT_FILENAME = "child_stdout.log"
CHILD_STDERR_FILENAME = "child_stderr.log"
CHILD_STDOUT_ACTIVE_FILENAME = CHILD_STDOUT_FILENAME + ".active"
CHILD_STDERR_ACTIVE_FILENAME = CHILD_STDERR_FILENAME + ".active"
_CHILD_STDOUT_ACTIVE_ENV = "MFT_PHASE_B_CHILD_STDOUT_ACTIVE_PATH"
_CHILD_STDERR_ACTIVE_ENV = "MFT_PHASE_B_CHILD_STDERR_ACTIVE_PATH"
CPU_TELEMETRY_SCHEMA = "mft-tier1-final1000-phase-b-cpu-telemetry-v1"
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_CGROUP_V1_UNBOUNDED_MIN_BYTES = 1 << 60


def _canonical_cgroup_path(value: str) -> str | None:
    if not value.startswith("/") or "\x00" in value:
        return None
    candidate = PurePosixPath(value)
    if ".." in candidate.parts:
        return None
    return candidate.as_posix()


def _safe_cgroup_directory(root: Path, membership: str) -> Path | None:
    canonical = _canonical_cgroup_path(membership)
    if canonical is None:
        return None
    try:
        resolved_root = root.resolve(strict=True)
        relative = PurePosixPath(canonical).relative_to("/")
        candidate = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate != resolved_root and not candidate.is_relative_to(resolved_root):
        return None
    return candidate if candidate.is_dir() else None


def _read_cgroup_scalar(path: Path, *, allow_max: bool = False) -> int | str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > 128:
        return None
    try:
        value = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if allow_max and value == "max":
        return "max"
    if not value or not value.isascii() or not value.isdigit():
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _unavailable_cgroup_memory(
    reason: str,
    *,
    version: int | None = None,
    cgroup_path: str | None = None,
    current: int | None = None,
    peak: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    return {
        "cgroup_available": False,
        "cgroup_version": version,
        "cgroup_path": cgroup_path,
        "cgroup_measurement": (
            "unavailable-on-platform"
            if version is None
            else "unavailable-or-incomplete"
        ),
        "cgroup_unavailable_reason": reason,
        "cgroup_current_bytes": current,
        "cgroup_peak_bytes": peak,
        "cgroup_limit_bytes": limit,
        "cgroup_limit_unbounded": False,
    }


def _collect_parent_cgroup_memory(
    *,
    platform_name: str | None = None,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
    cgroup_root: Path = _CGROUP_ROOT,
) -> dict[str, Any]:
    """Read the owning Linux memory cgroup without following an escape path."""

    selected_platform = platform_name or sys.platform
    if not selected_platform.startswith("linux"):
        return _unavailable_cgroup_memory("non-linux-platform")
    try:
        raw = proc_self_cgroup.read_bytes()
    except OSError:
        return _unavailable_cgroup_memory("proc-self-cgroup-unavailable")
    if len(raw) > 1024 * 1024:
        return _unavailable_cgroup_memory("proc-self-cgroup-oversized")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return _unavailable_cgroup_memory("proc-self-cgroup-non-ascii")

    v2_path: str | None = None
    v1_path: str | None = None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, membership = fields
        canonical = _canonical_cgroup_path(membership)
        if canonical is None:
            continue
        if hierarchy == "0" and controllers == "":
            v2_path = canonical
            break
        if "memory" in controllers.split(","):
            v1_path = canonical

    if v2_path is not None:
        directory = _safe_cgroup_directory(cgroup_root, v2_path)
        if directory is None:
            return _unavailable_cgroup_memory(
                "cgroup-v2-directory-unavailable", version=2, cgroup_path=v2_path
            )
        current_raw = _read_cgroup_scalar(directory / "memory.current")
        peak_raw = _read_cgroup_scalar(directory / "memory.peak")
        limit_raw = _read_cgroup_scalar(directory / "memory.max", allow_max=True)
        current = current_raw if isinstance(current_raw, int) else None
        peak = peak_raw if isinstance(peak_raw, int) else None
        unbounded = limit_raw == "max"
        limit = limit_raw if isinstance(limit_raw, int) and limit_raw > 0 else None
        if current is None or peak is None or (limit is None and not unbounded):
            return _unavailable_cgroup_memory(
                "cgroup-v2-memory-files-incomplete",
                version=2,
                cgroup_path=v2_path,
                current=current,
                peak=peak,
                limit=limit,
            )
        return {
            "cgroup_available": True,
            "cgroup_version": 2,
            "cgroup_path": v2_path,
            "cgroup_measurement": "linux-cgroup-v2-memory-files",
            "cgroup_unavailable_reason": None,
            "cgroup_current_bytes": current,
            "cgroup_peak_bytes": peak,
            "cgroup_limit_bytes": limit,
            "cgroup_limit_unbounded": unbounded,
        }

    if v1_path is not None:
        directory = _safe_cgroup_directory(cgroup_root / "memory", v1_path)
        if directory is None:
            directory = _safe_cgroup_directory(cgroup_root, v1_path)
        if directory is None:
            return _unavailable_cgroup_memory(
                "cgroup-v1-directory-unavailable", version=1, cgroup_path=v1_path
            )
        current_raw = _read_cgroup_scalar(directory / "memory.usage_in_bytes")
        peak_raw = _read_cgroup_scalar(directory / "memory.max_usage_in_bytes")
        limit_raw = _read_cgroup_scalar(directory / "memory.limit_in_bytes")
        current = current_raw if isinstance(current_raw, int) else None
        peak = peak_raw if isinstance(peak_raw, int) else None
        parsed_limit = limit_raw if isinstance(limit_raw, int) else None
        unbounded = bool(
            parsed_limit is not None
            and parsed_limit >= _CGROUP_V1_UNBOUNDED_MIN_BYTES
        )
        limit = (
            parsed_limit
            if parsed_limit is not None and parsed_limit > 0 and not unbounded
            else None
        )
        if current is None or peak is None or (limit is None and not unbounded):
            return _unavailable_cgroup_memory(
                "cgroup-v1-memory-files-incomplete",
                version=1,
                cgroup_path=v1_path,
                current=current,
                peak=peak,
                limit=limit,
            )
        return {
            "cgroup_available": True,
            "cgroup_version": 1,
            "cgroup_path": v1_path,
            "cgroup_measurement": "linux-cgroup-v1-memory-files",
            "cgroup_unavailable_reason": None,
            "cgroup_current_bytes": current,
            "cgroup_peak_bytes": peak,
            "cgroup_limit_bytes": limit,
            "cgroup_limit_unbounded": unbounded,
        }

    return _unavailable_cgroup_memory("memory-cgroup-membership-unavailable")


def _build_parent_memory_evidence(
    *,
    task_id: str,
    manifest_sha256: str,
    terminal_status: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    finished: Mapping[int, Mapping[str, Any]],
    cgroup_memory: Mapping[str, Any],
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    peaks: list[int] = []
    for child in children:
        ordinal = int(child["ordinal"])
        record = finished.get(ordinal)
        legacy = record.get("legacy") if isinstance(record, Mapping) else None
        raw_peak = (
            legacy.get("observed_peak_rss_bytes")
            if isinstance(legacy, Mapping)
            else None
        )
        peak = (
            int(raw_peak)
            if isinstance(raw_peak, int)
            and not isinstance(raw_peak, bool)
            and raw_peak > 0
            else None
        )
        if peak is not None:
            peaks.append(peak)
        observations.append(
            {
                "ordinal": ordinal,
                "seed": int(child["seed"]),
                "legacy_status_sha256": (
                    canonical_sha256(legacy)
                    if isinstance(legacy, Mapping)
                    else None
                ),
                "observed_peak_rss_bytes": peak,
            }
        )

    logical_children = len(children)
    child_requested = CHILD_MEMORY_MB * 1024**2
    parent_requested = logical_children * child_requested
    rss_sum = sum(peaks)
    child_within = len(peaks) == logical_children and rss_sum <= parent_requested
    cgroup_available = cgroup_memory.get("cgroup_available") is True
    cgroup_unbounded = cgroup_memory.get("cgroup_limit_unbounded") is True
    cgroup_peak = cgroup_memory.get("cgroup_peak_bytes")
    cgroup_limit = cgroup_memory.get("cgroup_limit_bytes")
    peak_within = bool(
        cgroup_available
        and not cgroup_unbounded
        and isinstance(cgroup_peak, int)
        and not isinstance(cgroup_peak, bool)
        and isinstance(cgroup_limit, int)
        and not isinstance(cgroup_limit, bool)
        and cgroup_peak <= cgroup_limit
    )
    limit_covers = bool(
        cgroup_available
        and not cgroup_unbounded
        and isinstance(cgroup_limit, int)
        and not isinstance(cgroup_limit, bool)
        and cgroup_limit >= parent_requested
    )
    return seal_parent_memory_evidence(
        {
            "schema_version": PARENT_MEMORY_EVIDENCE_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": str(task_id),
            "manifest_sha256": str(manifest_sha256),
            "task_status_sha256": str(terminal_status["status_sha256"]),
            "logical_child_count": logical_children,
            "child_requested_memory_bytes": child_requested,
            "parent_requested_memory_bytes": parent_requested,
            "child_peak_rss_available_count": len(peaks),
            "child_peak_rss_sum_bytes": rss_sum,
            "child_peak_rss_max_bytes": max(peaks) if peaks else None,
            "child_peak_rss_observations_sha256": canonical_sha256(observations),
            "cgroup_available": cgroup_memory.get("cgroup_available"),
            "cgroup_version": cgroup_memory.get("cgroup_version"),
            "cgroup_path": cgroup_memory.get("cgroup_path"),
            "cgroup_measurement": cgroup_memory.get("cgroup_measurement"),
            "cgroup_unavailable_reason": cgroup_memory.get(
                "cgroup_unavailable_reason"
            ),
            "cgroup_current_bytes": cgroup_memory.get("cgroup_current_bytes"),
            "cgroup_peak_bytes": cgroup_memory.get("cgroup_peak_bytes"),
            "cgroup_limit_bytes": cgroup_memory.get("cgroup_limit_bytes"),
            "cgroup_limit_unbounded": cgroup_memory.get(
                "cgroup_limit_unbounded"
            ),
            "total_child_peak_rss_within_parent_request": child_within,
            "cgroup_peak_within_limit": peak_within,
            "cgroup_limit_covers_parent_request": limit_covers,
            "safety_passed": child_within and peak_within and limit_covers,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )


def _aggregate_reaped_child_cpu_seconds() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (ImportError, OSError, ValueError):
        return None
    value = float(usage.ru_utime) + float(usage.ru_stime)
    return value if math.isfinite(value) and value >= 0 else None


def _resource_telemetry(
    *,
    concurrency: int,
    wall_time_seconds: float,
    child_cpu_start: float | None,
    child_cpu_finish: float | None,
    child_wall_times: Sequence[float],
    child_resource_telemetry: Sequence[Mapping[str, Any]],
    model_load_stagger_seconds: float,
    dispatch_fill_seconds: float,
) -> dict[str, Any]:
    wall = max(0.0, float(wall_time_seconds))
    requested_cpus = concurrency * CHILD_CPUS
    capacity_seconds = wall * requested_cpus
    available = child_cpu_start is not None and child_cpu_finish is not None
    child_cpu = (
        max(0.0, float(child_cpu_finish) - float(child_cpu_start))
        if available
        else None
    )
    utilization = (
        child_cpu / capacity_seconds
        if child_cpu is not None and capacity_seconds > 0
        else (0.0 if child_cpu is not None else None)
    )
    child_wall = sum(max(0.0, float(value)) for value in child_wall_times)
    logical_child_capacity = child_wall * CHILD_CPUS
    logical_child_utilization = (
        child_cpu / logical_child_capacity
        if child_cpu is not None and logical_child_capacity > 0
        else (0.0 if child_cpu is not None else None)
    )
    reported = [
        validate_child_resource_telemetry(value) for value in child_resource_telemetry
    ]
    reported_available = [value for value in reported if value["available"]]
    reported_cpu = sum(
        float(value["process_tree_cpu_seconds"]) for value in reported_available
    )
    reported_capacity = sum(
        float(value["cpu_capacity_seconds"]) for value in reported_available
    )
    reported_utilization = (
        reported_cpu / reported_capacity if reported_capacity > 0 else 0.0
    )
    reaped_reported_delta = (
        child_cpu - reported_cpu
        if child_cpu is not None and len(reported_available) == concurrency
        else None
    )
    return {
        "schema_version": CPU_TELEMETRY_SCHEMA,
        "available": available,
        "measurement": (
            "resource.getrusage(RUSAGE_CHILDREN)-delta"
            if available
            else "unavailable-on-platform"
        ),
        "parent_requested_cpus": requested_cpus,
        "logical_child_cpus": CHILD_CPUS,
        "logical_child_count": concurrency,
        "parent_wall_time_seconds": wall,
        "aggregate_child_cpu_seconds": child_cpu,
        "requested_cpu_capacity_seconds": capacity_seconds,
        "aggregate_cpu_utilization_fraction": utilization,
        "sum_child_wall_time_seconds": child_wall,
        "logical_child_cpu_capacity_seconds": logical_child_capacity,
        "logical_child_cpu_utilization_fraction": logical_child_utilization,
        "reported_child_cpu_available_count": len(reported_available),
        "reported_child_process_tree_cpu_seconds": reported_cpu,
        "reported_child_cpu_capacity_seconds": reported_capacity,
        "reported_child_cpu_utilization_fraction": reported_utilization,
        "reaped_minus_reported_child_cpu_seconds": reaped_reported_delta,
        "scheduler_envelope_idle_capacity_seconds": max(
            0.0, capacity_seconds - logical_child_capacity
        ),
        "configured_model_load_stagger_seconds": float(model_load_stagger_seconds),
        "dispatch_fill_seconds": max(0.0, float(dispatch_fill_seconds)),
        "maximum_planned_dispatch_fill_seconds": max(
            0.0, (concurrency - 1) * float(model_load_stagger_seconds)
        ),
        "scheduler_ready_lane_reserve": 0,
        "scheduler_admission_policy": "natural-terminal-vacancy-only-v1",
        "future_comparison_child_cpu_counts": [1, 2, 4],
        "automatic_child_cpu_reduction_allowed": False,
    }


class ChildProcess(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ProcessLauncher(Protocol):
    def __call__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> ChildProcess: ...


def _default_process_launcher(
    *, command: Sequence[str], cwd: Path, environment: Mapping[str, str]
) -> ChildProcess:
    child_environment = dict(environment)
    try:
        stdout_path = Path(child_environment.pop(_CHILD_STDOUT_ACTIVE_ENV))
        stderr_path = Path(child_environment.pop(_CHILD_STDERR_ACTIVE_ENV))
    except KeyError as exc:
        raise RuntimeError("Phase B child stdio capture path is missing") from exc
    options: dict[str, Any] = {}
    if os.name == "posix":
        options["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        return subprocess.Popen(
            list(command),
            cwd=cwd,
            env=child_environment,
            stdout=stdout,
            stderr=stderr,
            **options,
        )


class ProcessSetStopLatch:
    """One-way signal/deadline latch that owns every active child group."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._reason: str | None = None
        self._processes: dict[int, ChildProcess] = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def bind(self, process: ChildProcess) -> None:
        with self._lock:
            pid = int(process.pid)
            if pid <= 0 or pid in self._processes:
                raise RuntimeError("Phase B child process identity drifted")
            self._processes[pid] = process
            requested = self._event.is_set()
        if requested and process.poll() is None:
            phase_a_runner._signal_process_group(process, signal.SIGTERM)

    def unbind(self, process: ChildProcess) -> None:
        with self._lock:
            if self._processes.pop(int(process.pid), None) is not process:
                raise RuntimeError("Phase B child process binding drifted")

    def active(self) -> list[ChildProcess]:
        with self._lock:
            return list(self._processes.values())

    def terminate_processes(self) -> None:
        for process in self.active():
            if process.poll() is None:
                phase_a_runner._signal_process_group(process, signal.SIGTERM)

    def kill_processes(self) -> None:
        for process in self.active():
            if process.poll() is None:
                phase_a_runner._signal_process_group(process, FORCE_KILL_SIGNAL)

    def request(self, reason: str) -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = str(reason)
                self._event.set()
        self.terminate_processes()


def partition_inherited_cpus(
    concurrent_children: int,
    *,
    available_cpus: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic disjoint four-CPU partitions of the parent mask."""

    required = int(concurrent_children) * CHILD_CPUS
    if available_cpus is None:
        if os.name != "posix" or not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("Phase B requires Linux Scheduler CPU affinity")
        available_cpus = sorted(os.sched_getaffinity(0))
    cpus = [int(cpu) for cpu in available_cpus]
    if (
        len(cpus) < required
        or len(set(cpus)) != len(cpus)
        or any(cpu < 0 for cpu in cpus)
    ):
        raise RuntimeError("Phase B parent does not own its declared CPU envelope")
    selected = sorted(cpus)[:required]
    return tuple(
        tuple(selected[offset : offset + CHILD_CPUS])
        for offset in range(0, required, CHILD_CPUS)
    )


def prepare_child_runtime(child_dir: Path) -> tuple[Path, dict[str, str]]:
    """Create one fresh, seed-owned temp/joblib/cache namespace."""

    child_dir = child_dir.resolve(strict=True)
    runtime = child_dir / "runtime"
    if runtime.exists() or runtime.is_symlink():
        raise RuntimeError("Phase B child runtime scratch already exists")
    runtime.mkdir()
    paths = {
        "tmp": runtime / str(RUNTIME_ISOLATION["tmp_relative_path"]),
        "joblib": runtime / str(RUNTIME_ISOLATION["joblib_relative_path"]),
        "cache": runtime / str(RUNTIME_ISOLATION["cache_relative_path"]),
    }
    for path in paths.values():
        path.mkdir()
    environment = {
        name: str(paths[str(relative)])
        for name, relative in RUNTIME_ISOLATION["environment_variables"].items()
    }
    return runtime, environment


def prepare_child_stdio(child_dir: Path) -> dict[str, str]:
    """Create seed-owned active streams before the child can emit output."""

    child_dir = child_dir.resolve(strict=True)
    paths = {
        _CHILD_STDOUT_ACTIVE_ENV: child_dir / CHILD_STDOUT_ACTIVE_FILENAME,
        _CHILD_STDERR_ACTIVE_ENV: child_dir / CHILD_STDERR_ACTIVE_FILENAME,
    }
    final_paths = (
        child_dir / CHILD_STDOUT_FILENAME,
        child_dir / CHILD_STDERR_FILENAME,
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (*paths.values(), *final_paths)
    ):
        raise RuntimeError("Phase B child stdio capture already exists")
    for path in paths.values():
        path.open("xb").close()
    return {name: str(path) for name, path in paths.items()}


def _append_child_stderr(child_dir: Path, message: str) -> None:
    path = child_dir.resolve(strict=True) / CHILD_STDERR_ACTIVE_FILENAME
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase B active child stderr capture drifted")
    with path.open("ab", buffering=0) as stream:
        stream.write(
            (str(message).rstrip("\n") + "\n").encode(
                "utf-8", errors="replace"
            )
        )


def finalize_child_stdio(child_dir: Path, *, seed: int) -> dict[str, Any]:
    """Atomically publish and hash-bind reaped-child stdout and stderr."""

    child_dir = child_dir.resolve(strict=True)
    if child_dir.name != f"seed-{int(seed)}":
        raise RuntimeError("Phase B child stdio seed directory drifted")
    evidence: dict[str, Any] = {"stdio_capture_complete": True}
    for label, active_name, final_name in (
        ("stdout", CHILD_STDOUT_ACTIVE_FILENAME, CHILD_STDOUT_FILENAME),
        ("stderr", CHILD_STDERR_ACTIVE_FILENAME, CHILD_STDERR_FILENAME),
    ):
        active = child_dir / active_name
        final = child_dir / final_name
        if (
            active.is_symlink()
            or not active.is_file()
            or final.exists()
            or final.is_symlink()
        ):
            raise RuntimeError(f"Phase B child {label} capture drifted")
        active.replace(final)
        size = final.stat().st_size
        evidence[f"{label}_relative_path"] = f"seed-{int(seed)}/{final_name}"
        evidence[f"{label}_sha256"] = phase_a_runner._sha256_file(final)
        evidence[f"{label}_size_bytes"] = int(size)
    return evidence


def cleanup_child_runtime(child_dir: Path, runtime: Path) -> None:
    """Delete only the reaped child's owned runtime root, never shared tmp."""

    child_dir = child_dir.resolve(strict=True)
    expected = child_dir / "runtime"
    # Compare lexical absolute paths before resolving the candidate.  Resolving
    # a hostile replacement symlink would point at shared storage; in that case
    # unlink only the child-owned directory entry and never follow its target.
    runtime_absolute = runtime.absolute()
    if runtime_absolute != expected.absolute():
        raise RuntimeError("Phase B cleanup escaped the child runtime root")
    if runtime.is_symlink():
        runtime.unlink()
    elif runtime.exists():
        shutil.rmtree(runtime)
    if runtime.exists() or runtime.is_symlink():
        raise RuntimeError("Phase B child runtime scratch cleanup did not finish")


def _initial_status(
    task_id: str, manifest_sha256: str, concurrent_children: int
) -> dict[str, Any]:
    timestamp = now()
    return seal_task_status(
        {
            "schema_version": TASK_STATUS_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": task_id,
            "manifest_sha256": manifest_sha256,
            "state": "starting",
            "stop_requested": False,
            "stop_reason": None,
            "active_children": [],
            "launched_child_count": 0,
            "finished_child_count": 0,
            "sealed_child_count": 0,
            "completed_child_count": 0,
            "failed_child_count": 0,
            "started_at": timestamp,
            "updated_at": timestamp,
            "finished_at": None,
            "execution_mode": "concurrent",
            "concurrent_children": int(concurrent_children),
            "subprocess_per_seed": True,
            "model_context_reuse": False,
            "rng_context_reuse": False,
            "cpu_isolation": copy.deepcopy(CPU_ISOLATION),
            "resource_telemetry": None,
            "physical_scheduler_task_count": 1,
            "virtual_scheduler_task_ids_created": False,
            "scheduler_mutation_performed": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )


def _update_status(status: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    value = {
        key: copy.deepcopy(item)
        for key, item in status.items()
        if key != "status_sha256"
    }
    value.update(copy.deepcopy(changes))
    value["updated_at"] = now()
    return seal_task_status(value)


def _unavailable_child_resource_telemetry(
    wall_time_seconds: float,
) -> dict[str, Any]:
    wall = max(0.0, float(wall_time_seconds))
    return validate_child_resource_telemetry(
        {
            "schema_version": CHILD_RESOURCE_TELEMETRY_SCHEMA,
            "available": False,
            "measurement": "unavailable-on-platform",
            "child_cpus": CHILD_CPUS,
            "wall_time_seconds": wall,
            "process_tree_cpu_seconds": None,
            "cpu_capacity_seconds": wall * CHILD_CPUS,
            "cpu_utilization_fraction": None,
        }
    )


def _receipt_child_resource_telemetry(
    legacy: Mapping[str, Any], *, wall_time_seconds: float
) -> dict[str, Any]:
    candidate = legacy.get("phase_b_child_resource_telemetry")
    if isinstance(candidate, Mapping):
        try:
            return validate_child_resource_telemetry(candidate)
        except RuntimeError:
            pass
    return _unavailable_child_resource_telemetry(wall_time_seconds)


def _legacy_terminal(
    path: Path,
    *,
    task_id: str,
    child: Mapping[str, Any],
    exit_code: int,
    stop_requested: bool,
    stop_reason: str | None,
) -> tuple[dict[str, Any], str, bool, str | None, str | None]:
    """Authenticate one seed-local legacy journal and result binding."""

    try:
        legacy = phase_a_runner._read_json(path)
    except RuntimeError as exc:
        legacy = {
            "schema_version": STATUS_SCHEMA,
            "state": "failed",
            "phase": "missing_or_unreadable_child_status",
            "terminal": True,
            "task_id": task_id,
            "seed": int(child["seed"]),
            "payload_sha256": child["payload_sha256"],
            "exit_code": int(exit_code),
            "failure": f"{type(exc).__name__}:{exc}",
        }
        if stop_requested:
            return legacy, "stopped", False, None, stop_reason
        return legacy, "refused", True, None, legacy["failure"]
    identity_matches = (
        legacy.get("schema_version") == STATUS_SCHEMA
        and legacy.get("task_id") == task_id
        and legacy.get("seed") == int(child["seed"])
        and legacy.get("payload_sha256") == child["payload_sha256"]
        and legacy.get("terminal") is True
    )
    if stop_requested:
        if not identity_matches:
            return legacy, "stopped", False, None, stop_reason
        return legacy, "stopped", False, None, stop_reason
    if not identity_matches:
        return (
            legacy,
            "refused",
            True,
            None,
            "legacy child status identity/terminal seal mismatch",
        )
    recorded_exit_code = legacy.get("exit_code")
    if (
        isinstance(recorded_exit_code, bool)
        or not isinstance(recorded_exit_code, int)
        or recorded_exit_code != int(exit_code)
    ):
        return (
            legacy,
            "refused",
            True,
            None,
            "legacy child status/process exit-code mismatch",
        )
    try:
        validate_child_resource_telemetry(
            legacy.get("phase_b_child_resource_telemetry") or {}
        )
    except RuntimeError as exc:
        return (
            legacy,
            "refused",
            True,
            None,
            f"completed child CPU telemetry mismatch: {exc}",
        )
    semlock_stress_sha = legacy.get("phase_b_semlock_stress_sha256")
    semlock_gate_passed = (
        isinstance(semlock_stress_sha, str)
        and len(semlock_stress_sha) == 64
        and all(character in "0123456789abcdef" for character in semlock_stress_sha)
    )
    if legacy.get("state") == "completed":
        if exit_code != 0:
            return (
                legacy,
                "refused",
                True,
                None,
                "completed child process exited nonzero",
            )
        if not semlock_gate_passed:
            return (
                legacy,
                "refused",
                True,
                None,
                "completed Phase B child lacks SemLock stress attestation",
            )
        result_path = path.parent / "result.json"
        result_sha = legacy.get("result_sha256")
        if (
            not result_path.is_file()
            or not isinstance(result_sha, str)
            or phase_a_runner._sha256_file(result_path) != result_sha
        ):
            return (
                legacy,
                "refused",
                True,
                None,
                "completed child result/status SHA binding mismatch",
            )
        return legacy, "completed", False, result_sha, None
    failure = str(legacy.get("failure") or f"child exit code {exit_code}")
    if legacy.get("phase") == "remote_model_load_failed":
        return legacy, "failed", True, None, failure
    if legacy.get("phase") == "terminal" and "loaded_model_count" in legacy:
        if not semlock_gate_passed:
            return (
                legacy,
                "refused",
                True,
                None,
                "terminal Phase B child lacks SemLock stress attestation",
            )
        return legacy, "failed", False, None, failure
    return legacy, "refused", True, None, failure


def _child_command(
    bundle: Path,
    child_payload_path: Path,
    child: Mapping[str, Any],
    heartbeat_seconds: float,
) -> list[str]:
    runner = (
        bundle
        / "artifacts"
        / "code"
        / "tools"
        / "tier1_corrected_current7_slurm_seed_runner.py"
    )
    return [
        sys.executable,
        "-u",
        str(runner),
        "--bundle-root",
        str(bundle),
        "--payload",
        str(child_payload_path),
        "--payload-root",
        str(bundle),
        "--payload-sha256",
        str(child["payload_sha256"]),
        "--heartbeat-seconds",
        str(heartbeat_seconds),
    ]


def run(
    bundle: Path,
    payload_path: Path,
    payload_root: Path,
    expected_payload_sha256: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    termination_grace_seconds: float = DEFAULT_CHILD_TERMINATION_GRACE_SECONDS,
    *,
    process_launcher: ProcessLauncher | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    duration_clock: Callable[[], float] = time.monotonic,
    available_cpus: Sequence[int] | None = None,
) -> int:
    """Execute one sealed concurrent parent without external mutations."""

    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be finite and positive")
    if not math.isfinite(termination_grace_seconds) or termination_grace_seconds <= 0:
        raise ValueError("termination grace must be finite and positive")
    bundle = bundle.resolve(strict=True)
    payload_root = payload_root.resolve(strict=True)
    payload_path = phase_a_runner._contained(
        payload_root, payload_path, "Phase B payload path"
    )
    if payload_path.name != "payload.json":
        raise RuntimeError("Phase B parent payload basename is unsafe")
    payload = validate_batch_payload(phase_a_runner._read_json(payload_path))
    if canonical_sha256(payload) != expected_payload_sha256:
        raise RuntimeError("Phase B parent payload SHA-256 mismatch")
    manifest = validate_batch_manifest(batch_manifest_from_payload(payload))
    scheduler_task_id = os.environ.get("SLURM_SCHED_TASK_ID")
    if scheduler_task_id is not None and (
        not scheduler_task_id.isascii()
        or not scheduler_task_id.isdigit()
        or len(scheduler_task_id) > 20
        or int(scheduler_task_id) <= 0
        or str(int(scheduler_task_id)) != scheduler_task_id
    ):
        raise RuntimeError("Scheduler task id must be a positive decimal integer")
    task_id = str(scheduler_task_id or f"local-pid-{os.getpid()}")
    runs_root = (bundle / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    run_root = (runs_root / f"task-{task_id}").resolve()
    if run_root.parent != runs_root:
        raise RuntimeError("Phase B run directory escaped its bundle root")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "batch_manifest.json"
    status_path = run_root / "task_status.json"
    memory_evidence_path = run_root / PARENT_MEMORY_EVIDENCE_FILENAME
    # Same-parent restart is fail-closed.  The immutable receipt prefix remains
    # harvestable, while recovery proceeds only through a new physical parent
    # identity; an unsealed child is never silently resumed or duplicated.
    if (
        status_path.exists()
        or memory_evidence_path.exists()
        or any(run_root.glob(f"seed-*/{LEGACY_STATUS_FILENAME}"))
        or any(run_root.glob("seed-*/seed_status.json"))
    ):
        raise RuntimeError(
            "existing Phase B journal requires a new physical parent identity"
        )
    phase_a_runner._immutable_json(manifest_path, manifest)
    concurrency = int(payload["concurrent_children"])
    cpu_sets = partition_inherited_cpus(concurrency, available_cpus=available_cpus)
    status = _initial_status(task_id, manifest["manifest_sha256"], concurrency)
    phase_a_runner._atomic_json(status_path, status)
    telemetry_wall_started = duration_clock()
    telemetry_cpu_started = _aggregate_reaped_child_cpu_seconds()
    launcher = process_launcher or _default_process_launcher
    stop_latch = ProcessSetStopLatch()
    prior_handlers: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        stop_latch.request(f"signal:{signal.Signals(signum).name}")

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, on_signal)

    started = monotonic()
    deadline = started + float(payload["internal_deadline_seconds"])
    next_launch_at = started
    next_ordinal = 0
    active: dict[int, dict[str, Any]] = {}
    finished: dict[int, dict[str, Any]] = {}
    next_commit = 0
    lane_fatal = False
    termination_started: float | None = None
    launch_clocks: list[float] = []

    def active_journal() -> list[dict[str, Any]]:
        return [
            {
                "ordinal": ordinal,
                "seed": int(record["seed"]),
                "pid": int(record["process"].pid),
                "cpu_set": list(record["cpu_set"]),
                "launched_at": str(record["launched_at"]),
            }
            for ordinal, record in sorted(active.items())
        ]

    def persist_running() -> None:
        nonlocal status
        status = _update_status(
            status,
            state="stopping" if stop_latch.requested else "running",
            stop_requested=stop_latch.requested,
            stop_reason=stop_latch.reason,
            active_children=active_journal(),
            launched_child_count=next_ordinal,
            finished_child_count=len(finished),
        )
        phase_a_runner._atomic_json(status_path, status)

    try:
        while active or (
            next_ordinal < len(payload["children"])
            and not stop_latch.requested
            and not lane_fatal
        ):
            current = monotonic()
            if current >= deadline and not stop_latch.requested:
                stop_latch.request("internal_deadline_elapsed")
                termination_started = current

            if (
                not stop_latch.requested
                and not lane_fatal
                and next_ordinal < len(payload["children"])
                and current >= next_launch_at
            ):
                remaining = deadline - current
                if remaining < float(payload["minimum_child_start_budget_seconds"]):
                    stop_latch.request("internal_deadline_before_next_seed")
                    termination_started = current
                else:
                    child = payload["children"][next_ordinal]
                    ordinal = int(child["ordinal"])
                    seed = int(child["seed"])
                    child_dir = run_root / f"seed-{seed}"
                    child_dir.mkdir(parents=True, exist_ok=True)
                    child_payload_path = child_dir / "payload.json"
                    phase_a_runner._immutable_json(
                        child_payload_path, child["task"]["payload_json"]
                    )
                    runtime_scratch, runtime_environment = prepare_child_runtime(
                        child_dir
                    )
                    stdio_environment = prepare_child_stdio(child_dir)
                    environment = os.environ.copy()
                    environment[PHASE_B_CHILD_MARKER_ENV] = PHASE_B_CHILD_MARKER
                    environment[PHASE_B_CHILD_CPUSET_ENV] = ",".join(
                        str(cpu) for cpu in cpu_sets[ordinal]
                    )
                    environment.update(runtime_environment)
                    environment.update(stdio_environment)
                    launched_at = now()
                    launched_clock = duration_clock()
                    launch_clocks.append(launched_clock)
                    try:
                        process = launcher(
                            command=_child_command(
                                bundle, child_payload_path, child, heartbeat_seconds
                            ),
                            cwd=bundle / "artifacts" / "code",
                            environment=environment,
                        )
                        stop_latch.bind(process)
                    except Exception as exc:
                        exit_code = CHILD_LAUNCH_FAILURE_EXIT_CODE
                        _append_child_stderr(
                            child_dir,
                            f"Phase B child launcher failed: {type(exc).__name__}:{exc}",
                        )
                        child_wall = max(0.0, duration_clock() - launched_clock)
                        unavailable_telemetry = _unavailable_child_resource_telemetry(
                            child_wall
                        )
                        phase_a_runner._atomic_json(
                            child_dir / LEGACY_STATUS_FILENAME,
                            {
                                "schema_version": STATUS_SCHEMA,
                                "state": "failed",
                                "phase": "child_executor_failed",
                                "terminal": True,
                                "task_id": task_id,
                                "seed": seed,
                                "payload_sha256": child["payload_sha256"],
                                "exit_code": exit_code,
                                "failure": f"{type(exc).__name__}:{exc}",
                                "phase_b_child_resource_telemetry": (
                                    unavailable_telemetry
                                ),
                            },
                        )
                        legacy, state, child_fatal, result_sha, failure = (
                            _legacy_terminal(
                                child_dir / LEGACY_STATUS_FILENAME,
                                task_id=task_id,
                                child=child,
                                exit_code=exit_code,
                                stop_requested=False,
                                stop_reason=None,
                            )
                        )
                        cleanup_child_runtime(child_dir, runtime_scratch)
                        stdio_evidence = finalize_child_stdio(child_dir, seed=seed)
                        finished[ordinal] = {
                            "child": child,
                            "legacy": legacy,
                            "state": state,
                            "lane_fatal": child_fatal,
                            "result_sha256": result_sha,
                            "failure": failure,
                            "exit_code": exit_code,
                            "started_at": launched_at,
                            "finished_at": now(),
                            "wall_time_seconds": child_wall,
                            "child_resource_telemetry": unavailable_telemetry,
                            "cpu_set": cpu_sets[ordinal],
                            "runtime_scratch_cleanup_performed": True,
                            "stdio_evidence": stdio_evidence,
                        }
                        lane_fatal = True
                    else:
                        active[ordinal] = {
                            "child": child,
                            "seed": seed,
                            "process": process,
                            "started_at": launched_at,
                            "started_clock": launched_clock,
                            "cpu_set": cpu_sets[ordinal],
                            "launched_at": launched_at,
                            "child_dir": child_dir,
                            "runtime_scratch": runtime_scratch,
                        }
                    next_ordinal += 1
                    next_launch_at = current + float(
                        payload["model_load_stagger_seconds"]
                    )

            for ordinal, record in list(active.items()):
                process = record["process"]
                exit_code = process.poll()
                if exit_code is None:
                    continue
                stop_latch.unbind(process)
                child = record["child"]
                child_wall = max(0.0, duration_clock() - record["started_clock"])
                legacy, state, child_fatal, result_sha, failure = _legacy_terminal(
                    run_root / f"seed-{record['seed']}" / LEGACY_STATUS_FILENAME,
                    task_id=task_id,
                    child=child,
                    exit_code=int(exit_code),
                    stop_requested=stop_latch.requested,
                    stop_reason=stop_latch.reason,
                )
                cleanup_child_runtime(record["child_dir"], record["runtime_scratch"])
                stdio_evidence = finalize_child_stdio(
                    record["child_dir"], seed=int(record["seed"])
                )
                finished[ordinal] = {
                    "child": child,
                    "legacy": legacy,
                    "state": state,
                    "lane_fatal": child_fatal,
                    "result_sha256": result_sha,
                    "failure": failure,
                    "exit_code": int(exit_code),
                    "started_at": record["started_at"],
                    "finished_at": now(),
                    "wall_time_seconds": child_wall,
                    "child_resource_telemetry": (
                        _receipt_child_resource_telemetry(
                            legacy, wall_time_seconds=child_wall
                        )
                    ),
                    "cpu_set": record["cpu_set"],
                    "runtime_scratch_cleanup_performed": True,
                    "stdio_evidence": stdio_evidence,
                }
                del active[ordinal]
                if child_fatal and not stop_latch.requested:
                    lane_fatal = True

            if lane_fatal and active:
                if termination_started is None:
                    termination_started = monotonic()
                    stop_latch.terminate_processes()
            if stop_latch.requested and active and termination_started is None:
                termination_started = monotonic()
                stop_latch.terminate_processes()
            if (
                termination_started is not None
                and active
                and monotonic() - termination_started >= termination_grace_seconds
            ):
                stop_latch.kill_processes()

            while next_commit in finished:
                record = finished[next_commit]
                child = record["child"]
                seed = int(child["seed"])
                receipt = seal_child_receipt(
                    {
                        "schema_version": CHILD_RECEIPT_SCHEMA,
                        "protocol_version": PROTOCOL_VERSION,
                        "task_id": task_id,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "ordinal": next_commit,
                        "seed": seed,
                        "payload_sha256": child["payload_sha256"],
                        "logical_dedupe_key": child["logical_dedupe_key"],
                        "state": record["state"],
                        "terminal": True,
                        "lane_fatal": bool(record["lane_fatal"]),
                        "exit_code": int(record["exit_code"]),
                        "legacy_status": record["legacy"],
                        "legacy_status_sha256": canonical_sha256(record["legacy"]),
                        "result_sha256": record["result_sha256"],
                        "started_at": record["started_at"],
                        "finished_at": record["finished_at"],
                        "wall_time_seconds": record["wall_time_seconds"],
                        "child_resource_telemetry": record["child_resource_telemetry"],
                        "failure": record["failure"],
                        "cpu_set": list(record["cpu_set"]),
                        "child_cpus": CHILD_CPUS,
                        "child_memory_mb": CHILD_MEMORY_MB,
                        "runtime_scratch_relative_path": f"seed-{seed}/runtime",
                        "runtime_scratch_cleanup_performed": bool(
                            record["runtime_scratch_cleanup_performed"]
                        ),
                        "shared_tmp_deleted": False,
                        **record["stdio_evidence"],
                        "production_eligible": False,
                        "fea_submission_performed": False,
                        "aedt_used": False,
                    }
                )
                phase_a_runner._immutable_json(
                    run_root / f"seed-{seed}" / "seed_status.json", receipt
                )
                status = _update_status(
                    status,
                    active_children=active_journal(),
                    launched_child_count=next_ordinal,
                    finished_child_count=len(finished),
                    sealed_child_count=int(status["sealed_child_count"]) + 1,
                    completed_child_count=int(status["completed_child_count"])
                    + (1 if record["state"] == "completed" else 0),
                    failed_child_count=int(status["failed_child_count"])
                    + (0 if record["state"] == "completed" else 1),
                )
                next_commit += 1
            persist_running()
            if active or (
                next_ordinal < len(payload["children"])
                and not stop_latch.requested
                and not lane_fatal
            ):
                stop_latch.wait(min(heartbeat_seconds, 0.25))

        if stop_latch.requested:
            parent_state = (
                "deadline"
                if str(stop_latch.reason or "").startswith("internal_deadline")
                else "stopped"
            )
        elif lane_fatal:
            parent_state = "failed"
        elif int(status["sealed_child_count"]) != len(payload["children"]):
            parent_state = "failed"
            lane_fatal = True
        elif int(status["failed_child_count"]):
            parent_state = "completed_with_failures"
        else:
            parent_state = "completed"
        status = _update_status(
            status,
            state=parent_state,
            stop_requested=stop_latch.requested,
            stop_reason=stop_latch.reason,
            active_children=[],
            launched_child_count=next_ordinal,
            finished_child_count=len(finished),
            finished_at=now(),
            resource_telemetry=_resource_telemetry(
                concurrency=concurrency,
                wall_time_seconds=max(0.0, duration_clock() - telemetry_wall_started),
                child_cpu_start=telemetry_cpu_started,
                child_cpu_finish=_aggregate_reaped_child_cpu_seconds(),
                child_wall_times=[
                    float(record["wall_time_seconds"]) for record in finished.values()
                ],
                child_resource_telemetry=[
                    record["child_resource_telemetry"] for record in finished.values()
                ],
                model_load_stagger_seconds=float(payload["model_load_stagger_seconds"]),
                dispatch_fill_seconds=(
                    max(launch_clocks) - min(launch_clocks) if launch_clocks else 0.0
                ),
            ),
        )
        phase_a_runner._atomic_json(status_path, status)
        memory_evidence = _build_parent_memory_evidence(
            task_id=task_id,
            manifest_sha256=manifest["manifest_sha256"],
            terminal_status=status,
            children=payload["children"],
            finished=finished,
            cgroup_memory=_collect_parent_cgroup_memory(),
        )
        phase_a_runner._immutable_json(memory_evidence_path, memory_evidence)
        if stop_latch.requested:
            return STOPPED_EXIT_CODE
        if lane_fatal:
            return LANE_FATAL_EXIT_CODE
        if int(status["failed_child_count"]):
            return CHILD_FAILURE_EXIT_CODE
        return 0
    finally:
        stop_latch.terminate_processes()
        for process in stop_latch.active():
            try:
                process.wait(timeout=termination_grace_seconds)
            except subprocess.TimeoutExpired:
                phase_a_runner._signal_process_group(process, FORCE_KILL_SIGNAL)
                process.wait(timeout=termination_grace_seconds)
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument(
        "--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_CHILD_TERMINATION_GRACE_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run(
        args.bundle_root,
        args.payload,
        args.payload_root,
        args.payload_sha256,
        args.heartbeat_seconds,
        args.termination_grace_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
