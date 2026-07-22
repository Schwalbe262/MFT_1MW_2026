"""Execute a finite final1000 seed batch with one fresh subprocess per seed.

Phase A intentionally reuses the authenticated single-seed runner without an
in-process model cache.  The only optimization is that one Scheduler parent
owns a finite ordered seed block.  Every terminal child is sealed before the
next subprocess can start, and a stop/deadline latch permanently prohibits a
next child from being launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import STATUS_SCHEMA
    from tier1_final1000_multiseed_contract import (
        CHILD_RECEIPT_SCHEMA,
        PROTOCOL_VERSION,
        TASK_STATUS_SCHEMA,
        batch_manifest_from_payload,
        json_bytes,
        now,
        seal_child_receipt,
        seal_task_status,
        validate_batch_manifest,
        validate_batch_payload,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import STATUS_SCHEMA
    from tools.tier1_final1000_multiseed_contract import (
        CHILD_RECEIPT_SCHEMA,
        PROTOCOL_VERSION,
        TASK_STATUS_SCHEMA,
        batch_manifest_from_payload,
        json_bytes,
        now,
        seal_child_receipt,
        seal_task_status,
        validate_batch_manifest,
        validate_batch_payload,
    )


DEFAULT_HEARTBEAT_SECONDS = 5.0
DEFAULT_CHILD_TERMINATION_GRACE_SECONDS = 30.0
LANE_FATAL_EXIT_CODE = 74
STOPPED_EXIT_CODE = 75
CHILD_EXECUTOR_FAILURE_EXIT_CODE = 76
FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGABRT)


def _signal_process_group(
    process: subprocess.Popen[Any], signum: signal.Signals
) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signum)
        except ProcessLookupError:
            return
    elif signum == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


class ChildExecutor(Protocol):
    def __call__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
        stop_latch: "StopLatch",
        heartbeat_seconds: float,
        termination_grace_seconds: float,
        deadline_monotonic: float,
        monotonic: Callable[[], float],
        on_heartbeat: Callable[[int], None],
    ) -> int: ...


class StopLatch:
    """Thread-safe one-way stop latch shared by signal and child wait paths."""

    def __init__(self) -> None:
        self._event = threading.Event()
        # Python signal handlers run on the main thread between bytecodes.  An
        # RLock prevents a SIGTERM handler from deadlocking if it interrupts a
        # bind/unbind critical section on that same thread.
        self._lock = threading.RLock()
        self._reason: str | None = None
        self._process: subprocess.Popen[Any] | None = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def request(self, reason: str) -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = str(reason)
                self._event.set()
            process = self._process
        if process is not None and process.poll() is None:
            _signal_process_group(process, signal.SIGTERM)

    def bind(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError(
                    "multi-seed lane attempted overlapping child processes"
                )
            self._process = process
            requested = self._event.is_set()
        if requested and process.poll() is None:
            _signal_process_group(process, signal.SIGTERM)

    def unbind(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            if self._process is not process:
                raise RuntimeError("multi-seed lane child process binding drifted")
            self._process = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(json_bytes(value))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except PermissionError:  # Windows read-only hard-link cleanup
                os.chmod(temporary, 0o600)
                temporary.unlink()
                if path.exists():
                    os.chmod(path, 0o444)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Create once; an identical retry is accepted without an overwrite."""

    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable multi-seed evidence collision: {path}")
        return False
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}."
        f"{time.time_ns()}.immutable.tmp"
    )
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
            published = True
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"immutable multi-seed evidence collision: {path}")
            return False
        return True
    finally:
        if temporary.exists():
            if os.name == "nt":
                os.chmod(temporary, 0o600)
            temporary.unlink()
            if published and os.name == "nt":
                os.chmod(path, 0o444)


def _contained(root: Path, value: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    path = value.resolve(strict=True)
    if path == root or not path.is_relative_to(root):
        raise RuntimeError(f"unsafe {label}: {path}")
    return path


def _default_child_executor(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    stop_latch: StopLatch,
    heartbeat_seconds: float,
    termination_grace_seconds: float,
    deadline_monotonic: float,
    monotonic: Callable[[], float],
    on_heartbeat: Callable[[int], None],
) -> int:
    popen_options: dict[str, Any] = {}
    if os.name == "posix":
        popen_options["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        list(command), cwd=cwd, env=dict(environment), **popen_options
    )
    stop_latch.bind(process)
    try:
        while process.poll() is None:
            if monotonic() >= deadline_monotonic:
                stop_latch.request("internal_deadline_elapsed")
            on_heartbeat(int(process.pid))
            stop_latch.wait(heartbeat_seconds)
            if stop_latch.requested and process.poll() is None:
                _signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=termination_grace_seconds)
                except subprocess.TimeoutExpired:
                    _signal_process_group(process, FORCE_KILL_SIGNAL)
                    process.wait(timeout=termination_grace_seconds)
        return int(process.returncode)
    finally:
        try:
            if process.poll() is None:
                _signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=termination_grace_seconds)
                except subprocess.TimeoutExpired:
                    _signal_process_group(process, FORCE_KILL_SIGNAL)
                    process.wait(timeout=termination_grace_seconds)
        finally:
            stop_latch.unbind(process)


def _initial_status(task_id: str, manifest_sha256: str) -> dict[str, Any]:
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
            "current_ordinal": None,
            "current_seed": None,
            "sealed_child_count": 0,
            "completed_child_count": 0,
            "failed_child_count": 0,
            "started_at": timestamp,
            "updated_at": timestamp,
            "finished_at": None,
            "subprocess_per_seed": True,
            "model_context_reuse": False,
            "scheduler_mutation_performed": False,
            "fea_submission_performed": False,
            "aedt_used": False,
        }
    )


def _update_status(status: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    value = {key: item for key, item in status.items() if key != "status_sha256"}
    value.update(changes)
    value["updated_at"] = now()
    return seal_task_status(value)


def _legacy_status_or_refusal(
    path: Path,
    *,
    task_id: str,
    child: Mapping[str, Any],
    exit_code: int,
    stop_latch: StopLatch,
) -> tuple[dict[str, Any], str, bool, str | None, str | None]:
    """Return legacy status, child state, lane-fatal, result SHA, failure."""

    try:
        status = _read_json(path)
    except RuntimeError as exc:
        status = {
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
        state = "stopped" if stop_latch.requested else "refused"
        return status, state, not stop_latch.requested, None, status["failure"]
    failure: str | None = None
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("task_id") != task_id
        or status.get("seed") != int(child["seed"])
        or status.get("payload_sha256") != child["payload_sha256"]
        or status.get("terminal") is not True
    ):
        failure = "legacy child status identity/terminal seal mismatch"
        return status, "refused", True, None, failure
    if stop_latch.requested:
        return status, "stopped", False, None, stop_latch.reason
    if status.get("state") == "completed":
        result_path = path.parent / f"seed-{int(child['seed'])}" / "result.json"
        result_sha = status.get("result_sha256")
        if (
            not result_path.is_file()
            or not isinstance(result_sha, str)
            or _sha256_file(result_path) != result_sha
        ):
            failure = "completed child result/status SHA binding mismatch"
            return status, "refused", True, None, failure
        return status, "completed", False, result_sha, None
    failure = str(status.get("failure") or f"child exit code {exit_code}")
    if status.get("phase") == "remote_model_load_failed":
        return status, "failed", True, None, failure
    # A terminal optimizer failure after authenticated model load is seed-local.
    if status.get("phase") == "terminal" and "loaded_model_count" in status:
        return status, "failed", False, None, failure
    return status, "refused", True, None, failure


def run(
    bundle: Path,
    payload_path: Path,
    payload_root: Path,
    expected_payload_sha256: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    termination_grace_seconds: float = DEFAULT_CHILD_TERMINATION_GRACE_SECONDS,
    *,
    child_executor: ChildExecutor | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    duration_clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run a finite batch.  No Scheduler, AEDT, FEA, or remote API is used."""

    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat seconds must be finite and positive")
    if not math.isfinite(termination_grace_seconds) or termination_grace_seconds <= 0:
        raise ValueError("termination grace must be finite and positive")
    bundle = bundle.resolve(strict=True)
    payload_root = payload_root.resolve(strict=True)
    payload_path = _contained(payload_root, payload_path, "batch payload path")
    if payload_path.name != "payload.json":
        raise RuntimeError("multi-seed parent payload basename is unsafe")
    payload = validate_batch_payload(_read_json(payload_path))
    if canonical_sha256(payload) != expected_payload_sha256:
        raise RuntimeError("multi-seed parent payload SHA-256 mismatch")
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
        raise RuntimeError("multi-seed run directory escaped its bundle root")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "batch_manifest.json"
    status_path = run_root / "task_status.json"
    # A process restart may harvest prior immutable receipts, but Phase A never
    # resumes an unsealed seed or silently advances a crashed parent.
    if status_path.exists() or any(run_root.glob("seed-*/seed_status.json")):
        raise RuntimeError(
            "existing multi-seed lane journal requires a new parent identity"
        )
    _immutable_json(manifest_path, manifest)
    status = _initial_status(task_id, manifest["manifest_sha256"])
    _atomic_json(status_path, status)
    stop_latch = StopLatch()
    prior_handlers: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        stop_latch.request(f"signal:{signal.Signals(signum).name}")

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, on_signal)
    started = monotonic()
    deadline_monotonic = started + float(payload["internal_deadline_seconds"])
    child_executor = child_executor or _default_child_executor
    lane_fatal = False
    try:
        for child in payload["children"]:
            ordinal = int(child["ordinal"])
            seed = int(child["seed"])
            elapsed = monotonic() - started
            if stop_latch.requested:
                break
            if elapsed + float(payload["minimum_child_start_budget_seconds"]) > float(
                payload["internal_deadline_seconds"]
            ):
                stop_latch.request("internal_deadline_before_next_seed")
                break
            child_dir = run_root / f"seed-{seed}"
            child_dir.mkdir(parents=True, exist_ok=True)
            # The authenticated v1 runner deliberately requires this basename.
            child_payload_path = child_dir / "payload.json"
            _immutable_json(child_payload_path, child["task"]["payload_json"])
            legacy_status_path = run_root / "seed_status.json"
            if legacy_status_path.exists():
                legacy_status_path.unlink()
            status = _update_status(
                status,
                state="running",
                current_ordinal=ordinal,
                current_seed=seed,
                stop_requested=False,
                stop_reason=None,
            )
            _atomic_json(status_path, status)
            child_started_at = now()
            child_started_clock = duration_clock()

            def heartbeat(pid: int) -> None:
                nonlocal status
                status = _update_status(
                    status,
                    state="stopping" if stop_latch.requested else "running",
                    stop_requested=stop_latch.requested,
                    stop_reason=stop_latch.reason,
                )
                value = {
                    key: item for key, item in status.items() if key != "status_sha256"
                }
                value["child_pid"] = int(pid)
                # child_pid is intentionally not part of the bounded wire schema.
                value.pop("child_pid", None)
                status = seal_task_status(value)
                _atomic_json(status_path, status)

            runner = (
                bundle
                / "artifacts"
                / "code"
                / "tools"
                / "tier1_corrected_current7_slurm_seed_runner.py"
            )
            command = [
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
            environment = os.environ.copy()
            try:
                exit_code = child_executor(
                    command=command,
                    cwd=bundle / "artifacts" / "code",
                    environment=environment,
                    stop_latch=stop_latch,
                    heartbeat_seconds=heartbeat_seconds,
                    termination_grace_seconds=termination_grace_seconds,
                    deadline_monotonic=deadline_monotonic,
                    monotonic=monotonic,
                    on_heartbeat=heartbeat,
                )
            except Exception as exc:
                exit_code = CHILD_EXECUTOR_FAILURE_EXIT_CODE
                _atomic_json(
                    legacy_status_path,
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
                    },
                )
            legacy, child_state, child_lane_fatal, result_sha, failure = (
                _legacy_status_or_refusal(
                    legacy_status_path,
                    task_id=task_id,
                    child=child,
                    exit_code=exit_code,
                    stop_latch=stop_latch,
                )
            )
            child_wall_time_seconds = max(
                0.0, float(duration_clock()) - float(child_started_clock)
            )
            receipt = seal_child_receipt(
                {
                    "schema_version": CHILD_RECEIPT_SCHEMA,
                    "protocol_version": PROTOCOL_VERSION,
                    "task_id": task_id,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "ordinal": ordinal,
                    "seed": seed,
                    "payload_sha256": child["payload_sha256"],
                    "logical_dedupe_key": child["logical_dedupe_key"],
                    "state": child_state,
                    "terminal": True,
                    "lane_fatal": bool(child_lane_fatal),
                    "exit_code": int(exit_code),
                    "legacy_status": legacy,
                    "legacy_status_sha256": canonical_sha256(legacy),
                    "result_sha256": result_sha,
                    "started_at": child_started_at,
                    "finished_at": now(),
                    "wall_time_seconds": child_wall_time_seconds,
                    "failure": failure,
                    "production_eligible": False,
                    "fea_submission_performed": False,
                    "aedt_used": False,
                }
            )
            _immutable_json(child_dir / "seed_status.json", receipt)
            completed_increment = 1 if child_state == "completed" else 0
            failed_increment = 0 if child_state == "completed" else 1
            status = _update_status(
                status,
                sealed_child_count=int(status["sealed_child_count"]) + 1,
                completed_child_count=int(status["completed_child_count"])
                + completed_increment,
                failed_child_count=int(status["failed_child_count"]) + failed_increment,
                stop_requested=stop_latch.requested,
                stop_reason=stop_latch.reason,
            )
            _atomic_json(status_path, status)
            lane_fatal = bool(child_lane_fatal)
            if stop_latch.requested or lane_fatal:
                break
        if stop_latch.requested:
            parent_state = (
                "deadline"
                if str(stop_latch.reason or "").startswith("internal_deadline")
                else "stopped"
            )
        elif lane_fatal:
            parent_state = "failed"
        elif int(status["sealed_child_count"]) != int(payload["batch_length"]):
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
            current_ordinal=None,
            current_seed=None,
            finished_at=now(),
        )
        _atomic_json(status_path, status)
        if stop_latch.requested:
            return STOPPED_EXIT_CODE
        if lane_fatal:
            return LANE_FATAL_EXIT_CODE
        return 0
    finally:
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
