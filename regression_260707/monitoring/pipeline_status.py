"""Bounded, read-only status reader for the continuous MFT pipeline.

The pipeline runs outside the repository so that controller/supervisor restarts
do not mix mutable queue state with source code.  This reader deliberately does
not import :class:`DurableJobQueue`: constructing that class initializes and
may migrate the database.  Monitoring must never mutate operational state.
"""

from __future__ import annotations

from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Callable
from urllib.parse import quote


DEFAULT_PIPELINE_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_pipeline"
)
PIPELINE_ROOT_ENV = "MFT_PIPELINE_ROOT"
DATABASE_MAX_BYTES = 256 * 1024 * 1024
LOCK_MAX_BYTES = 32 * 1024
LOG_TAIL_BYTES = 64 * 1024
LOG_TAIL_LINES = 12
RECENT_JOBS_PER_LANE = 4
QUERY_DEADLINE_SECONDS = 1.0
ARTIFACT_SCAN_LIMIT = 2_048
ARTIFACT_CANDIDATE_LIMIT = 64
DATASET_AUDIT_MAX_BYTES = 128 * 1024 * 1024
ROLE_ACTIVITY_STALE_SECONDS = 20 * 60
RUNNING_HEARTBEAT_STALE_SECONDS = 180
EXTERNAL_ACTIVITY_CONFIRMATION_SECONDS = 120
FIRST_TRAINING_ROWS = 500
MODEL_ACTIVATION_ROWS = 3_000
FIRST_TUNING_ROWS = 4_000
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

LANES: tuple[tuple[str, str, str], ...] = (
    ("collect", "collector", "Data collect"),
    ("train", "trainer", "Surrogate train"),
    ("tune", "tuner", "Surrogate tune"),
    ("optimize", "optimizer", "NSGA-II search"),
    ("verify_standard", "standard-verifier", "Standard FEA"),
    ("verify_fine", "fine-verifier", "Fine/full FEA"),
)
JOB_STATES = (
    "queued",
    "retry_wait",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)
EXPECTED_JOB_COLUMNS = frozenset({
    "id",
    "job_type",
    "idempotency_key",
    "input_generation",
    "state",
    "owner_lease",
    "heartbeat_at",
    "lease_until",
    "attempt",
    "max_attempts",
    "next_retry_at",
    "terminal_reason",
    "output_generation",
    "created_at",
    "updated_at",
})


def _bounded_text(value: Any, limit: int = 1000) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\x00", "").lstrip("\ufeff").strip()
    if not text:
        return None
    # Control characters make both JSON and the browser diagnostics hard to
    # audit.  Preserve tabs/newlines and flatten everything else.
    text = "".join(
        char if char in "\t\n\r" or ord(char) >= 32 else "?"
        for char in text
    )
    return text[:limit]


def _timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _iso_timestamp(value: Any) -> str | None:
    timestamp = _timestamp(value)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
            timespec="seconds"
        )
    except (OSError, OverflowError, ValueError):
        return None


def _parse_iso(value: Any) -> float | None:
    text = _bounded_text(value, 100)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError):
        return None


def _age(now: float, value: Any) -> float | None:
    timestamp = _timestamp(value)
    return max(0.0, now - timestamp) if timestamp is not None else None


def _read_bounded_json(
    path: Path, max_bytes: int = LOCK_MAX_BYTES
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        stat = path.stat()
        if stat.st_size <= 0:
            return None, "metadata file is empty"
        if stat.st_size > max_bytes:
            return None, f"metadata exceeds {max_bytes} byte limit"
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return None, f"metadata exceeds {max_bytes} byte limit"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            return None, "metadata is not a JSON object"
        return value, None
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {_bounded_text(exc, 300)}"


def _read_log_tail(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "size_bytes": 0,
        "updated_at": None,
        "tail": [],
    }
    try:
        stat = path.stat()
        result.update(
            exists=True,
            size_bytes=int(stat.st_size),
            updated_at=_iso_timestamp(stat.st_mtime),
        )
        if stat.st_size <= 0:
            return result
        with path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - LOG_TAIL_BYTES))
            raw = handle.read(LOG_TAIL_BYTES)
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            decoded = raw.decode("utf-16", errors="replace")
        elif raw and raw.count(b"\x00") > len(raw) // 4:
            # PowerShell Start-Process redirection commonly produces UTF-16LE.
            # A bounded seek may omit the BOM, so infer byte order from where
            # the NUL bytes occur in an ASCII-heavy operational log.
            odd_nuls = raw[1::2].count(b"\x00")
            even_nuls = raw[0::2].count(b"\x00")
            encoding = "utf-16-le" if odd_nuls >= even_nuls else "utf-16-be"
            decoded = raw.decode(encoding, errors="replace")
        else:
            decoded = raw.decode("utf-8-sig", errors="replace")
        if stat.st_size > LOG_TAIL_BYTES:
            # The first bytes can be a partial line after a bounded seek.
            decoded = decoded.split("\n", 1)[-1]
        lines = [
            _bounded_text(line, 2000)
            for line in decoded.splitlines()
            if line.strip()
        ]
        result["tail"] = [line for line in lines if line][-LOG_TAIL_LINES:]
    except FileNotFoundError:
        pass
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {_bounded_text(exc, 300)}"
    return result


def _windows_process(pid: int) -> tuple[bool | None, float | None]:
    """Return active state and creation time without third-party packages."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    )
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    process = kernel32.OpenProcess(0x1000, False, pid)  # query limited info
    if not process:
        error = ctypes.get_last_error()
        # ERROR_INVALID_PARAMETER means that the PID does not exist.  Access
        # denied is unknown rather than dead.
        return (False, None) if error == 87 else (None, None)
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            return None, None
        active = int(exit_code.value) == 259  # STILL_ACTIVE
        creation = ctypes.c_ulonglong()
        exit_time = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        created_at = None
        if kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            # Windows FILETIME is 100 ns since 1601-01-01.
            created_at = creation.value / 10_000_000 - 11_644_473_600
        return active, created_at
    finally:
        kernel32.CloseHandle(process)


def _inspect_process(pid: Any) -> tuple[bool | None, float | None]:
    if isinstance(pid, bool):
        return False, None
    try:
        normalized = int(pid)
    except (TypeError, ValueError):
        return False, None
    if normalized <= 0:
        return False, None
    if os.name == "nt":
        try:
            return _windows_process(normalized)
        except (AttributeError, OSError):
            return None, None
    try:
        os.kill(normalized, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None
    except OSError:
        return None, None
    return True, None


class ContinuousPipelineReader:
    """Read pipeline roles and jobs without initializing or reconciling them."""

    def __init__(
        self,
        pipeline_root: str | os.PathLike[str] | None = None,
        *,
        clock: Callable[[], float] = time.time,
        busy_timeout_seconds: float = 0.15,
        database_max_bytes: int = DATABASE_MAX_BYTES,
        inspect_external_processes: bool = True,
        dataset_audit_max_bytes: int = DATASET_AUDIT_MAX_BYTES,
        dataset_auditor: Callable[
            [Path, str | None, str | None], dict[str, int]
        ] | None = None,
    ) -> None:
        configured = pipeline_root or os.environ.get(
            PIPELINE_ROOT_ENV, str(DEFAULT_PIPELINE_ROOT)
        )
        self.root = Path(configured).expanduser().resolve()
        self.clock = clock
        self.busy_timeout_seconds = max(0.01, min(float(busy_timeout_seconds), 2.0))
        self.database_max_bytes = max(1024, int(database_max_bytes))
        self.inspect_external_processes = bool(inspect_external_processes)
        self.dataset_audit_max_bytes = max(
            1, int(dataset_audit_max_bytes)
        )
        self.dataset_auditor = dataset_auditor or self._audit_dataset_artifact
        self._dataset_audit_lock = threading.Lock()
        self._dataset_audit_key: tuple[Any, ...] | None = None
        self._dataset_audit_result: dict[str, Any] | None = None
        self._external_process_lock = threading.Lock()
        self._external_process_samples: dict[
            tuple[int, float], dict[str, float | None]
        ] = {}

    @staticmethod
    def _audit_dataset_artifact(
        path: Path,
        solver_revision: str | None,
        library_revision: str | None,
    ) -> dict[str, int]:
        """Recompute row tiers from an immutable dataset generation.

        This is intentionally imported lazily: a missing Parquet engine must
        degrade only the optional row classification, not the monitoring API.
        The caller caches the small result by artifact fingerprint so the
        quality contract is not rerun on every browser refresh.
        """
        import pandas as pd

        from ..quality_contract import annotate_validity

        raw = pd.read_parquet(path)
        audited = annotate_validity(
            raw,
            expected_solver_revision=solver_revision,
            expected_library_revision=library_revision,
        )
        raw_rows = int(len(audited))
        strict_em_rows = int(
            audited["_strict_valid_em"].fillna(False).astype(bool).sum()
        )
        strict_full_rows = int(
            audited["_strict_valid_full"].fillna(False).astype(bool).sum()
        )
        if not 0 <= strict_full_rows <= strict_em_rows <= raw_rows:
            raise ValueError(
                "dataset row tiers violate full <= EM <= raw invariant"
            )
        return {
            "raw_rows": raw_rows,
            "strict_em_rows": strict_em_rows,
            "strict_full_rows": strict_full_rows,
            "em_only_rows": strict_em_rows - strict_full_rows,
        }

    def _cached_dataset_audit(
        self,
        path: Path,
        solver_revision: str | None,
        library_revision: str | None,
    ) -> dict[str, Any]:
        """Return a bounded, read-only and fingerprint-cached row audit."""
        try:
            if path.is_symlink():
                raise OSError("dataset artifact symlinks are not audited")
            before = path.stat()
            if not path.is_file():
                raise OSError("dataset artifact is not a regular file")
            if before.st_size <= 0:
                raise OSError("dataset artifact is empty")
            if before.st_size > self.dataset_audit_max_bytes:
                raise OSError(
                    "dataset artifact exceeds read-only audit limit "
                    f"({before.st_size} > {self.dataset_audit_max_bytes} bytes)"
                )
        except OSError as exc:
            return {
                "available": False,
                "source": "manifest",
                "error": f"{type(exc).__name__}: {_bounded_text(exc, 400)}",
            }

        key = (
            str(path),
            int(before.st_size),
            int(before.st_mtime_ns),
            solver_revision,
            library_revision,
        )
        with self._dataset_audit_lock:
            if key == self._dataset_audit_key and self._dataset_audit_result:
                return dict(self._dataset_audit_result)
            try:
                counts = self.dataset_auditor(
                    path, solver_revision, library_revision
                )
                after = path.stat()
                if (
                    after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns
                ):
                    raise OSError("dataset artifact changed during row audit")
                normalized = {
                    name: int(counts[name])
                    for name in (
                        "raw_rows",
                        "strict_em_rows",
                        "strict_full_rows",
                        "em_only_rows",
                    )
                }
                if not (
                    0
                    <= normalized["strict_full_rows"]
                    <= normalized["strict_em_rows"]
                    <= normalized["raw_rows"]
                    and normalized["em_only_rows"]
                    == normalized["strict_em_rows"]
                    - normalized["strict_full_rows"]
                ):
                    raise ValueError("dataset auditor returned incoherent row tiers")
                result: dict[str, Any] = {
                    "available": True,
                    "source": "train.parquet_quality_contract",
                    "artifact_size_bytes": int(after.st_size),
                    "error": None,
                    **normalized,
                }
            except Exception as exc:
                result = {
                    "available": False,
                    "source": "manifest",
                    "artifact_size_bytes": int(before.st_size),
                    "error": f"{type(exc).__name__}: {_bounded_text(exc, 500)}",
                }
            self._dataset_audit_key = key
            self._dataset_audit_result = result
            return dict(result)

    @staticmethod
    def _controller_cycle(role: dict[str, Any]) -> dict[str, Any]:
        stdout = (role.get("logs") or {}).get("stdout") or {}
        for line in reversed(stdout.get("tail") or []):
            try:
                value = json.loads(str(line).lstrip("\ufeff"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            if not any(key in value for key in ("jobs", "blocked", "dataset_generation")):
                continue
            jobs = value.get("jobs") if isinstance(value.get("jobs"), dict) else {}
            blocked = (
                value.get("blocked") if isinstance(value.get("blocked"), dict) else {}
            )
            return {
                "available": True,
                "dataset_generation": _bounded_text(
                    value.get("dataset_generation"), 200
                ),
                "jobs": {
                    str(key): int(item)
                    for key, item in jobs.items()
                    if isinstance(item, int) and not isinstance(item, bool)
                },
                "blocked": {
                    str(key): _bounded_text(item, 300)
                    for key, item in blocked.items()
                    if _bounded_text(item, 300)
                },
            }
        return {"available": False, "dataset_generation": None, "jobs": {}, "blocked": {}}

    @staticmethod
    def _unavailable_dataset_cohort(
        error: str,
        *,
        scan_truncated: bool,
    ) -> dict[str, Any]:
        return {
            "available": False,
            "counts_available": False,
            "raw_rows": None,
            "strict_em_rows": None,
            "strict_full_rows": 0,
            "em_only_rows": None,
            "current_raw_rows": None,
            "current_strict_em_rows": None,
            "current_strict_full_rows": 0,
            "current_em_only_rows": None,
            "counts_source": None,
            "counts_error": error,
            "error": error,
            "scan_truncated": scan_truncated,
        }

    def _dataset_cohort(
        self,
        revisions: dict[str, Any],
        controller_cycle: dict[str, Any],
    ) -> dict[str, Any]:
        dataset_root = self.root / "artifacts" / "dataset"
        solver = revisions.get("solver_revision")
        library = revisions.get("library_revision")
        candidates: list[tuple[float, Path]] = []
        generation = controller_cycle.get("dataset_generation")
        if isinstance(generation, str) and generation.startswith("dataset:"):
            generation_id = generation.split(":", 1)[1]
            if re.fullmatch(r"[0-9a-f]{64}", generation_id):
                candidates.append((float("inf"), dataset_root / generation_id))
        truncated = False
        try:
            with os.scandir(dataset_root) as entries:
                for index, entry in enumerate(entries):
                    if index >= ARTIFACT_SCAN_LIMIT:
                        truncated = True
                        break
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        candidates.append((entry.stat(follow_symlinks=False).st_mtime, Path(entry.path)))
                    except OSError:
                        continue
        except (FileNotFoundError, NotADirectoryError):
            return self._unavailable_dataset_cohort(
                "exact-SHA dataset generation is unavailable",
                scan_truncated=False,
            )
        except OSError as exc:
            return self._unavailable_dataset_cohort(
                f"{type(exc).__name__}: {_bounded_text(exc, 400)}",
                scan_truncated=False,
            )
        seen: set[Path] = set()
        for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
            if path in seen:
                continue
            seen.add(path)
            if len(seen) > ARTIFACT_CANDIDATE_LIMIT:
                break
            manifest, _ = _read_bounded_json(path / "manifest.json", 64 * 1024)
            if not manifest:
                continue
            metadata = manifest.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if solver and metadata.get("solver_revision") != solver:
                continue
            if library and metadata.get("library_revision") != library:
                continue
            manifest_full_rows = _nonnegative_integer(
                metadata.get("strict_full_rows")
            )
            if manifest_full_rows is None:
                continue
            audit = self._cached_dataset_audit(
                path / "train.parquet", solver, library
            )
            if audit.get("available"):
                raw_rows = audit["raw_rows"]
                strict_em_rows = audit["strict_em_rows"]
                strict_full_rows = audit["strict_full_rows"]
                em_only_rows = audit["em_only_rows"]
                counts_available = True
            else:
                raw_rows = _nonnegative_integer(metadata.get("raw_rows"))
                strict_em_rows = _nonnegative_integer(
                    metadata.get("strict_em_rows")
                )
                strict_full_rows = manifest_full_rows
                em_only_rows = (
                    strict_em_rows - strict_full_rows
                    if strict_em_rows is not None
                    and strict_em_rows >= strict_full_rows
                    else None
                )
                counts_available = (
                    raw_rows is not None
                    and strict_em_rows is not None
                    and em_only_rows is not None
                    and strict_em_rows <= raw_rows
                )
            manifest_consistent = (
                strict_full_rows == manifest_full_rows
                if audit.get("available")
                else None
            )
            counts_warning = None
            if manifest_consistent is False:
                counts_warning = (
                    "manifest strict_full_rows differs from the read-only "
                    "quality-contract audit"
                )
            return {
                "available": True,
                "counts_available": counts_available,
                "raw_rows": raw_rows,
                "strict_em_rows": strict_em_rows,
                "strict_full_rows": strict_full_rows,
                "em_only_rows": em_only_rows,
                # Keep the original current_* keys for older UI clients.
                "current_raw_rows": raw_rows,
                "current_strict_em_rows": strict_em_rows,
                "current_strict_full_rows": strict_full_rows,
                "current_em_only_rows": em_only_rows,
                "manifest_strict_full_rows": manifest_full_rows,
                "manifest_matches_audit": manifest_consistent,
                "counts_source": audit.get("source"),
                "counts_error": audit.get("error"),
                "counts_warning": counts_warning,
                "generation_id": _bounded_text(manifest.get("generation_id"), 100),
                "generation": f"dataset:{manifest.get('generation_id')}",
                "created_at": _bounded_text(manifest.get("created_at"), 100),
                "solver_revision": _bounded_text(metadata.get("solver_revision"), 80),
                "library_revision": _bounded_text(metadata.get("library_revision"), 80),
                "scan_truncated": truncated,
                "error": None,
            }
        return self._unavailable_dataset_cohort(
            "no dataset generation matches the exact controller revisions",
            scan_truncated=truncated,
        )

    def _external_tuners(self, now: float) -> dict[str, Any]:
        if not self.inspect_external_processes:
            return {
                "available": False,
                "processes": [],
                "validated_running_count": 0,
                "validated_lane_active": False,
                "error": None,
            }
        try:
            import psutil
        except ImportError:
            return {
                "available": False,
                "processes": [],
                "validated_running_count": 0,
                "validated_lane_active": False,
                "error": "psutil unavailable; external Optuna processes are not observable",
            }
        processes = []
        samples: dict[tuple[int, float], dict[str, float | None]] = {}
        deadline = time.monotonic() + 1.0
        try:
            iterator = psutil.process_iter(
                attrs=("pid", "name", "create_time")
            )
            for index, process in enumerate(iterator):
                if index >= 4_096 or time.monotonic() > deadline:
                    break
                try:
                    info = process.info
                    if "python" not in str(info.get("name") or "").lower():
                        continue
                    arguments = [str(value) for value in (process.cmdline() or [])]
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                    continue
                joined = " ".join(arguments).replace("\\", "/").lower()
                if "tune_optuna.py" not in joined:
                    continue
                managed_root = str(self.root).replace("\\", "/").lower()
                managed = managed_root in joined
                if managed:
                    continue

                def argument_after(flag: str) -> str | None:
                    try:
                        position = arguments.index(flag)
                    except ValueError:
                        return None
                    return (
                        _bounded_text(arguments[position + 1], 500)
                        if position + 1 < len(arguments)
                        else None
                    )

                created_at = _timestamp(info.get("create_time"))
                if created_at is None:
                    continue
                trials_text = argument_after("--trials")
                dataset = argument_after("--dataset")
                try:
                    trials = int(trials_text) if trials_text is not None else None
                except (TypeError, ValueError):
                    trials = None
                selection_valid = (
                    "--all" in arguments
                    or (
                        argument_after("--target") is not None
                        and argument_after("--family") is not None
                    )
                )
                command_validated = bool(
                    dataset and trials is not None and trials > 0 and selection_valid
                )
                cpu_seconds = None
                read_bytes = None
                write_bytes = None
                try:
                    cpu_times = process.cpu_times()
                    cpu_seconds = float(cpu_times.user + cpu_times.system)
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                    pass
                try:
                    io_counters = process.io_counters()
                    read_bytes = float(io_counters.read_bytes)
                    write_bytes = float(io_counters.write_bytes)
                except (
                    AttributeError,
                    psutil.AccessDenied,
                    psutil.NoSuchProcess,
                    OSError,
                ):
                    pass
                identity = (int(info["pid"]), float(int(created_at)))
                sample = {
                    "sampled_at": now,
                    "cpu_seconds": cpu_seconds,
                    "read_bytes": read_bytes,
                    "write_bytes": write_bytes,
                    "activity_confirmed_at": None,
                }
                samples[identity] = sample
                processes.append({
                    "pid": int(info["pid"]),
                    "name": _bounded_text(info.get("name"), 100),
                    "started_at": _iso_timestamp(created_at),
                    "elapsed_seconds": (
                        max(0.0, now - created_at)
                        if created_at is not None
                        else None
                    ),
                    "trials": trials_text,
                    "dataset": dataset,
                    "managed_by_durable_pipeline": False,
                    "command_validated": command_validated,
                    "activity_confirmed": False,
                    "activity_age_seconds": None,
                    "activity_window_seconds": None,
                    "cpu_seconds_delta": None,
                    "read_bytes_delta": None,
                    "write_bytes_delta": None,
                    "validated_running": False,
                })
        except Exception as exc:
            return {
                "available": False,
                "processes": [],
                "validated_running_count": 0,
                "validated_lane_active": False,
                "error": f"{type(exc).__name__}: {_bounded_text(exc, 400)}",
            }
        with self._external_process_lock:
            previous_samples = self._external_process_samples
            for process in processes:
                started_at = _parse_iso(process["started_at"])
                identity = (
                    int(process["pid"]),
                    float(int(started_at)) if started_at is not None else -1.0,
                )
                current = samples.get(identity)
                previous = previous_samples.get(identity)
                if current is None:
                    continue
                confirmed_at = (
                    previous.get("activity_confirmed_at")
                    if previous is not None else None
                )
                if previous is not None:
                    window = max(
                        0.0,
                        now - float(previous.get("sampled_at") or now),
                    )
                    process["activity_window_seconds"] = window
                    deltas: dict[str, float | None] = {}
                    for source, destination in (
                        ("cpu_seconds", "cpu_seconds_delta"),
                        ("read_bytes", "read_bytes_delta"),
                        ("write_bytes", "write_bytes_delta"),
                    ):
                        before = previous.get(source)
                        after = current.get(source)
                        delta = (
                            max(0.0, float(after) - float(before))
                            if before is not None and after is not None
                            else None
                        )
                        process[destination] = delta
                        deltas[source] = delta
                    if (
                        (deltas["cpu_seconds"] or 0.0) >= 0.01
                        or (deltas["read_bytes"] or 0.0) > 0
                        or (deltas["write_bytes"] or 0.0) > 0
                    ):
                        confirmed_at = now
                current["activity_confirmed_at"] = confirmed_at
                activity_age = (
                    max(0.0, now - float(confirmed_at))
                    if confirmed_at is not None else None
                )
                activity_confirmed = bool(
                    activity_age is not None
                    and activity_age <= EXTERNAL_ACTIVITY_CONFIRMATION_SECONDS
                )
                process["activity_confirmed"] = activity_confirmed
                process["activity_age_seconds"] = activity_age
                process["validated_running"] = bool(
                    process["command_validated"] and activity_confirmed
                )
            self._external_process_samples = samples
        validated = [
            process for process in processes if process["validated_running"]
        ]
        return {
            "available": True,
            "processes": sorted(processes, key=lambda item: item["pid"]),
            "validated_running_count": len(validated),
            "validated_lane_active": bool(validated),
            "error": None,
        }

    def _role(self, role: str, now: float) -> dict[str, Any]:
        lock_path = self.root / "locks" / f"{role}.lock"
        metadata, metadata_error = _read_bounded_json(lock_path)
        logs = {
            kind: _read_log_tail(
                self.root / "logs" / f"{role}.{kind}.log"
            )
            for kind in ("stdout", "stderr")
        }
        logs["launcher"] = _read_log_tail(
            self.root / "logs" / f"{role}.launcher.jsonl"
        )
        result: dict[str, Any] = {
            "role": role,
            "status": "missing",
            "alive": False,
            "pid": None,
            "hostname": None,
            "started_at": None,
            "elapsed_seconds": None,
            "last_activity_at": None,
            "activity_age_seconds": None,
            "last_error": None,
            "metadata": {},
            "logs": logs,
        }
        if metadata_error:
            result.update(status="invalid", error=metadata_error)
        if metadata is None:
            last_error_lines = logs["stderr"].get("tail") or []
            result["last_error"] = last_error_lines[-1] if last_error_lines else None
            return result

        safe_metadata = {
            key: metadata.get(key)
            for key in (
                "schema_version",
                "role",
                "command",
                "pid",
                "hostname",
                "acquired_at",
                "solver_revision",
                "library_revision",
                "verification_config_sha256",
            )
            if key in metadata
        }
        pid = metadata.get("pid")
        alive, process_created_at = _inspect_process(pid)
        acquired_at = _parse_iso(metadata.get("acquired_at"))
        # A reused PID must not make an old lock record look healthy.
        if (
            alive is True
            and process_created_at is not None
            and acquired_at is not None
            and process_created_at > acquired_at + 5
        ):
            alive = False
            result["pid_reused"] = True
        if alive is True:
            role_status = "alive"
        elif alive is False:
            role_status = "stale"
        else:
            role_status = "unknown"

        activities = [
            _parse_iso(item.get("updated_at"))
            for item in logs.values()
            if isinstance(item, dict)
        ]
        activities.append(acquired_at)
        latest_activity = max(
            (value for value in activities if value is not None), default=None
        )
        stderr_updated = _parse_iso(logs["stderr"].get("updated_at"))
        error_lines = logs["stderr"].get("tail") or []
        if (
            acquired_at is not None
            and stderr_updated is not None
            and stderr_updated + 2 < acquired_at
        ):
            error_lines = []
        result.update(
            status=role_status,
            alive=alive is True,
            pid=int(pid) if isinstance(pid, int) and not isinstance(pid, bool) else None,
            hostname=_bounded_text(metadata.get("hostname"), 200),
            started_at=(
                datetime.fromtimestamp(acquired_at, timezone.utc).isoformat(
                    timespec="seconds"
                )
                if acquired_at is not None
                else None
            ),
            elapsed_seconds=(
                max(0.0, now - acquired_at) if acquired_at is not None else None
            ),
            last_activity_at=_iso_timestamp(latest_activity),
            activity_age_seconds=_age(now, latest_activity),
            activity_stale=(
                latest_activity is not None
                and now - latest_activity > ROLE_ACTIVITY_STALE_SECONDS
            ),
            last_error=error_lines[-1] if error_lines else None,
            metadata=safe_metadata,
        )
        return result

    def _connect(self, database: Path, deadline: float) -> sqlite3.Connection:
        uri_path = quote(database.resolve().as_posix(), safe="/:")
        connection = sqlite3.connect(
            f"file:{uri_path}?mode=ro",
            uri=True,
            timeout=self.busy_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute(
            f"PRAGMA busy_timeout={int(self.busy_timeout_seconds * 1000)}"
        )
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0,
            1000,
        )
        return connection

    def _job(self, row: sqlite3.Row, now: float) -> dict[str, Any]:
        created_at = _timestamp(row["created_at"])
        updated_at = _timestamp(row["updated_at"])
        heartbeat_at = _timestamp(row["heartbeat_at"])
        attempt = int(row["attempt"])
        job_id = int(row["id"])
        attempt_log = (
            self.root / "work" / f"job-{job_id:08d}" /
            f"attempt-{attempt:03d}.log"
        )
        started_at = None
        start_source = "created_at_fallback"
        try:
            stat = attempt_log.stat()
            if os.name == "nt":
                started_at = float(stat.st_ctime)
                start_source = "attempt_log_created_at"
            elif hasattr(stat, "st_birthtime"):
                started_at = float(stat.st_birthtime)
                start_source = "attempt_log_created_at"
        except OSError:
            pass
        if started_at is None:
            started_at = created_at
        state = str(row["state"])
        elapsed_end = now if state in {"queued", "retry_wait", "running"} else updated_at
        elapsed_seconds = (
            max(0.0, elapsed_end - started_at)
            if elapsed_end is not None and started_at is not None
            else None
        )
        heartbeat_age = _age(now, heartbeat_at)
        lease_until = _timestamp(row["lease_until"])
        return {
            "id": job_id,
            "job_type": str(row["job_type"]),
            "idempotency_key": _bounded_text(row["idempotency_key"], 512),
            "state": state,
            "attempt": attempt,
            "max_attempts": int(row["max_attempts"]),
            "owner_lease": _bounded_text(row["owner_lease"], 512),
            "created_at": _iso_timestamp(created_at),
            "started_at": _iso_timestamp(started_at),
            "started_at_source": start_source,
            "heartbeat_at": _iso_timestamp(heartbeat_at),
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_stale": (
                state == "running"
                and (
                    heartbeat_age is None
                    or heartbeat_age > RUNNING_HEARTBEAT_STALE_SECONDS
                    or (lease_until is not None and lease_until < now)
                )
            ),
            "lease_until": _iso_timestamp(lease_until),
            "elapsed_seconds": elapsed_seconds,
            "updated_at": _iso_timestamp(updated_at),
            "terminal_reason": _bounded_text(row["terminal_reason"], 1200),
            "input_generation": _bounded_text(row["input_generation"], 1200),
            "output_generation": _bounded_text(row["output_generation"], 1200),
            "attempt_log": str(attempt_log) if attempt > 0 else None,
        }

    def _queue(self, now: float) -> dict[str, Any]:
        database = self.root / "jobs.sqlite3"
        base: dict[str, Any] = {
            "available": False,
            "path": str(database),
            "schema_version": None,
            "size_bytes": None,
            "total_jobs": 0,
            "counts": {state: 0 for state in JOB_STATES},
            "last_updated_at": None,
            "error": None,
        }
        try:
            size = database.stat().st_size
            base["size_bytes"] = int(size)
            if size > self.database_max_bytes:
                raise ValueError(
                    f"queue database exceeds {self.database_max_bytes} byte limit"
                )
            wal = database.with_name(database.name + "-wal")
            if wal.exists() and wal.stat().st_size > self.database_max_bytes:
                raise ValueError(
                    f"queue WAL exceeds {self.database_max_bytes} byte limit"
                )
        except FileNotFoundError:
            base["error"] = "queue database is missing"
            return {**base, "lanes": self._empty_lanes()}
        except OSError as exc:
            base["error"] = f"{type(exc).__name__}: {_bounded_text(exc, 500)}"
            return {**base, "lanes": self._empty_lanes()}

        deadline = time.monotonic() + QUERY_DEADLINE_SECONDS
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect(database, deadline)
            connection.execute("BEGIN")
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)")
            }
            missing = sorted(EXPECTED_JOB_COLUMNS - columns)
            if missing:
                raise ValueError("queue schema is missing columns: " + ", ".join(missing))
            meta = connection.execute(
                "SELECT value FROM queue_meta WHERE key='schema_version'"
            ).fetchone()
            base["schema_version"] = (
                int(meta["value"]) if meta is not None else None
            )
            placeholders = ",".join("?" for _ in LANES)
            job_types = [item[0] for item in LANES]
            counts = connection.execute(
                f"SELECT job_type, state, COUNT(*) AS n FROM jobs "
                f"WHERE job_type IN ({placeholders}) GROUP BY job_type, state",
                job_types,
            ).fetchall()
            per_lane: dict[str, dict[str, int]] = {
                job_type: {state: 0 for state in JOB_STATES}
                for job_type, _, _ in LANES
            }
            for row in counts:
                state = str(row["state"])
                job_type = str(row["job_type"])
                if job_type in per_lane and state in per_lane[job_type]:
                    per_lane[job_type][state] = int(row["n"])
                    base["counts"][state] += int(row["n"])
            base["total_jobs"] = sum(base["counts"].values())

            select_columns = (
                "id, job_type, idempotency_key, input_generation, state, "
                "owner_lease, heartbeat_at, lease_until, attempt, max_attempts, "
                "next_retry_at, terminal_reason, output_generation, created_at, "
                "updated_at"
            )
            lanes = []
            all_updated: list[float] = []
            for job_type, lane_key, label in LANES:
                recent_rows = connection.execute(
                    f"SELECT {select_columns} FROM jobs WHERE job_type=? "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?",
                    (job_type, RECENT_JOBS_PER_LANE),
                ).fetchall()
                recent = [self._job(row, now) for row in recent_rows]
                active = next(
                    (
                        job for job in recent
                        if job["state"] in {"running", "queued", "retry_wait"}
                    ),
                    None,
                )
                current = active or (recent[0] if recent else None)
                error_row = connection.execute(
                    f"SELECT {select_columns} FROM jobs WHERE job_type=? "
                    "AND terminal_reason IS NOT NULL "
                    "AND TRIM(terminal_reason) != '' "
                    "ORDER BY updated_at DESC, id DESC LIMIT 1",
                    (job_type,),
                ).fetchone()
                last_error_job = self._job(error_row, now) if error_row else None
                lane_counts = per_lane[job_type]
                if current and current.get("heartbeat_stale"):
                    health = "stale"
                elif lane_counts["running"]:
                    health = "running"
                elif (
                    current
                    and current["state"] == "retry_wait"
                    and current.get("terminal_reason")
                ):
                    health = "retrying"
                elif lane_counts["queued"] or lane_counts["retry_wait"]:
                    health = "waiting"
                elif current and current["state"] == "failed":
                    health = "failed"
                elif current and current["state"] == "succeeded":
                    health = "succeeded"
                else:
                    health = "idle"
                lanes.append({
                    "job_type": job_type,
                    "lane": lane_key,
                    "label": label,
                    "health": health,
                    "counts": lane_counts,
                    "current_job": current,
                    "last_error": (
                        {
                            "job_id": last_error_job["id"],
                            "at": last_error_job["updated_at"],
                            "reason": last_error_job["terminal_reason"],
                        }
                        if last_error_job
                        else None
                    ),
                    "recent_jobs": recent,
                })
                all_updated.extend(
                    timestamp
                    for timestamp in (
                        _parse_iso(job.get("updated_at")) for job in recent
                    )
                    if timestamp is not None
                )
            base.update(
                available=True,
                lanes=lanes,
                last_updated_at=_iso_timestamp(max(all_updated, default=None)),
            )
            return base
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            base["error"] = f"{type(exc).__name__}: {_bounded_text(exc, 700)}"
            return {**base, "lanes": self._empty_lanes()}
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _empty_lanes() -> list[dict[str, Any]]:
        return [
            {
                "job_type": job_type,
                "lane": lane,
                "label": label,
                "health": "unavailable",
                "counts": {state: 0 for state in JOB_STATES},
                "current_job": None,
                "last_error": None,
                "recent_jobs": [],
            }
            for job_type, lane, label in LANES
        ]

    @staticmethod
    def _lane_prerequisites(
        lanes: list[dict[str, Any]],
        cohort: dict[str, Any],
        controller_cycle: dict[str, Any],
    ) -> None:
        rows = int(cohort.get("current_strict_full_rows") or 0)
        cycle_blocked = controller_cycle.get("blocked") or {}
        reasons = {
            "collect": {
                "gate": "always_eligible",
                "ready": True,
                "reason": "scheduled every controller cycle",
                "current": rows,
                "threshold": None,
            },
            "train": {
                "gate": "first_training_checkpoint",
                "ready": rows >= FIRST_TRAINING_ROWS,
                "reason": (
                    "first checkpoint is eligible"
                    if rows >= FIRST_TRAINING_ROWS
                    else f"waiting for exact-SHA strict rows: {rows}/{FIRST_TRAINING_ROWS}"
                ),
                "current": rows,
                "threshold": FIRST_TRAINING_ROWS,
            },
            "tune": {
                "gate": "first_optuna_generation",
                "ready": rows >= FIRST_TUNING_ROWS,
                "reason": (
                    "first durable tuning generation is eligible"
                    if rows >= FIRST_TUNING_ROWS
                    else f"waiting for exact-SHA strict rows: {rows}/{FIRST_TUNING_ROWS}"
                ),
                "current": rows,
                "threshold": FIRST_TUNING_ROWS,
            },
            "optimize": {
                "gate": "promoted_exact_model",
                "ready": False,
                "reason": (
                    "waiting for promoted exact-SHA model: activation requires "
                    f">={MODEL_ACTIVATION_ROWS} strict rows (current {rows})"
                ),
                "current": rows,
                "threshold": MODEL_ACTIVATION_ROWS,
            },
            "verify_standard": {
                "gate": "nsga_output_dependency",
                "ready": False,
                "reason": "waiting for NSGA-II output dependency (33 standard FEA candidates)",
                "current": None,
                "threshold": 33,
            },
            "verify_fine": {
                "gate": "standard_fea_dependency",
                "ready": False,
                "reason": "waiting for successful standard FEA dependency (3 fine/full candidates)",
                "current": None,
                "threshold": 3,
            },
        }
        if cycle_blocked.get("verification_standard"):
            reasons["verify_standard"]["reason"] = str(
                cycle_blocked["verification_standard"]
            )
        if cycle_blocked.get("verification_fine"):
            reasons["verify_fine"]["reason"] = str(
                cycle_blocked["verification_fine"]
            )
        for lane in lanes:
            job_type = lane["job_type"]
            prerequisite = dict(reasons[job_type])
            current = lane.get("current_job")
            if isinstance(current, dict):
                state = current.get("state")
                if state in {"queued", "retry_wait", "running"}:
                    prerequisite.update(
                        ready=True,
                        reason=f"durable queue job #{current['id']} is {state}",
                    )
                elif state == "succeeded":
                    prerequisite.update(
                        ready=True,
                        reason=f"latest durable job #{current['id']} succeeded",
                    )
            lane["prerequisite"] = prerequisite

    def snapshot(self) -> dict[str, Any]:
        now = float(self.clock())
        roles = {
            role: self._role(role, now)
            for role in ("controller", "supervisor")
        }
        queue = self._queue(now)
        lanes = queue.pop("lanes")
        controller_metadata = roles["controller"].get("metadata") or {}
        solver_revision = _bounded_text(
            controller_metadata.get("solver_revision"), 80
        )
        library_revision = _bounded_text(
            controller_metadata.get("library_revision"), 80
        )
        revisions = {
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "solver_revision_exact": bool(
                solver_revision and FULL_SHA.fullmatch(solver_revision)
            ),
            "library_revision_exact": bool(
                library_revision and FULL_SHA.fullmatch(library_revision)
            ),
            "verification_config_sha256": _bounded_text(
                controller_metadata.get("verification_config_sha256"), 80
            ),
        }
        revisions["exact"] = (
            revisions["solver_revision_exact"]
            and revisions["library_revision_exact"]
        )
        controller_cycle = self._controller_cycle(roles["controller"])
        cohort = self._dataset_cohort(revisions, controller_cycle)
        self._lane_prerequisites(lanes, cohort, controller_cycle)
        external_tuners = self._external_tuners(now)
        durable_running_lanes = [
            lane["job_type"]
            for lane in lanes
            if lane["counts"].get("running", 0) > 0
        ]
        durable_active_lanes = [
            lane["job_type"]
            for lane in lanes
            if sum(
                lane["counts"].get(state, 0)
                for state in ("queued", "retry_wait", "running")
            ) > 0
        ]
        external_running_lanes = (
            ["external_tune"]
            if external_tuners.get("validated_lane_active") else []
        )
        running_lanes = durable_running_lanes + external_running_lanes
        active_lanes = durable_active_lanes + external_running_lanes
        stale_jobs = [
            lane["job_type"]
            for lane in lanes
            if isinstance(lane.get("current_job"), dict)
            and lane["current_job"].get("heartbeat_stale")
        ]
        warnings: list[str] = []
        if not queue["available"]:
            warnings.append(f"continuous pipeline queue unavailable: {queue['error']}")
        for name, role in roles.items():
            if role["status"] != "alive":
                warnings.append(
                    f"continuous pipeline {name} is {role['status']}"
                )
            if role.get("last_error"):
                warnings.append(
                    f"continuous pipeline {name} stderr: {role['last_error']}"
                )
        if stale_jobs:
            warnings.append(
                "continuous pipeline heartbeat stale: " + ", ".join(stale_jobs)
            )
        retrying_jobs = [
            lane["job_type"]
            for lane in lanes
            if lane.get("health") == "retrying"
        ]
        if retrying_jobs:
            warnings.append(
                "continuous pipeline jobs are retrying after errors: "
                + ", ".join(retrying_jobs)
            )
        if external_tuners.get("error"):
            warnings.append(str(external_tuners["error"]))
        if external_tuners.get("processes"):
            pids = ", ".join(
                str(item["pid"]) for item in external_tuners["processes"]
            )
            if external_tuners.get("validated_lane_active"):
                warnings.append(
                    "external Optuna tuner has recent CPU/I/O activity "
                    f"(PID {pids}); included as external_tune in observed lane "
                    "counts but excluded from durable queue counts"
                )
            else:
                warnings.append(
                    "external Optuna tuner process is present but recent CPU/I/O "
                    f"activity is not yet confirmed (PID {pids}); excluded from "
                    "observed lane counts"
                )
        both_roles_alive = all(role["alive"] for role in roles.values())
        if (
            both_roles_alive
            and queue["available"]
            and not stale_jobs
            and not retrying_jobs
        ):
            health = "healthy"
        elif any(role["alive"] for role in roles.values()) or queue["available"]:
            health = "degraded"
        else:
            health = "offline"
        return {
            "schema_version": 1,
            "generated_at": _iso_timestamp(now),
            "available": queue["available"] or any(
                role["status"] != "missing" for role in roles.values()
            ),
            "root": str(self.root),
            "health": health,
            "roles": roles,
            "revisions": revisions,
            "controller_cycle": controller_cycle,
            "cohort": {
                **cohort,
                "first_training_rows": FIRST_TRAINING_ROWS,
                "model_activation_rows": MODEL_ACTIVATION_ROWS,
                "first_tuning_rows": FIRST_TUNING_ROWS,
                "em_only_is_invalid": False,
                "row_semantics": {
                    "strict_em_rows": (
                        "EM quality and provenance passed"
                    ),
                    "strict_full_rows": (
                        "EM and thermal quality passed; used by full-pipeline gates"
                    ),
                    "em_only_rows": (
                        "EM-usable with incomplete thermal results; not invalid"
                    ),
                },
            },
            "external_tuners": external_tuners,
            "queue": queue,
            "lanes": lanes,
            "parallel": {
                "running_lane_count": len(running_lanes),
                "running_lanes": running_lanes,
                "active_lane_count": len(active_lanes),
                "active_lanes": active_lanes,
                "durable_running_lane_count": len(durable_running_lanes),
                "durable_running_lanes": durable_running_lanes,
                "durable_active_lane_count": len(durable_active_lanes),
                "durable_active_lanes": durable_active_lanes,
                "external_running_lane_count": len(external_running_lanes),
                "external_running_lanes": external_running_lanes,
                "parallel_work_confirmed": len(running_lanes) >= 2,
            },
            "warnings": warnings,
        }
