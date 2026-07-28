"""Bounded local-AEDT GUI launcher for an authenticated Pareto candidate.

This module deliberately has no scheduler client.  A launch always creates a
new standalone Desktop in the interactive Windows session and gives the runner
only a server-resolved, validated design payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable

from filelock import FileLock


LOCAL_GUI_DESIGN_PARAMETER_KEYS = (
    "N1_main", "N1_side", "N2_main", "N2_side",
    "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t",
    "cw1", "gap1", "cw2", "gap2", "nwh1", "nwh2",
    "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y",
    "w1c_w2s_space_x", "w2s_w1s_space_x", "w1s_w2s_space_y",
    "w1s_cs_space_x", "cs_w1s_space_y",
    "wcp_t", "wcp_pad_t", "wcp_len_x",
)
DEADLINE_GUI_NUMERIC_PARAMETER_KEYS = (
    "fan_velocity", "plate_temp", "air_temp",
)
DEADLINE_GUI_STRING_PARAMETER_KEYS = ("fan_config",)
DEADLINE_GUI_EXECUTION_PROFILE_VERSION = (
    "deadline-tim-k3-local-gui-exact-fea-profile-v1"
)
DEADLINE_GUI_COMMON_SOLVER_PARAMETERS = {
    "freq": 1000.0,
    "V1_rms": 1000.0,
    "I1_rated": 1000.0,
    "I2_rated": 100.0,
    "I2_phase_deg": 0.0,
    "P_target": 1_000_000.0,
    "V2_rms": 10_000.0,
    "core_cm": 1.377,
    "core_x": 1.51,
    "core_y": 1.74,
    "core_k_thermal": 2.0,
    "k_ins": 0.2,
    "conductor_temp_C": 80.0,
    "core_plate_on": 1,
    "wcp_on": 1,
    "round_corner": 0,
    "loss_from_copy": 1,
    "loss_on": 1,
    "matrix_on": 1,
    "thermal_on": 1,
    "cap_on": 1,
    "thermal_max_iterations": 250,
    "rx_mesh_mode": "skin",
    "keep_project": 1,
}
DEADLINE_GUI_MODE_SOLVER_PARAMETERS = {
    "symmetry": {
        "full_model": 0,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "matrix_percent_error": 1.5,
        "matrix_max_passes": 20,
        "matrix_min_converged": 1,
        "matrix_skin_mesh": 0,
        "percent_error": 1.5,
        "max_passes": 10,
        "min_converged": 2,
    },
    "full": {
        "full_model": 1,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "n_explicit_turns": 2,
        "matrix_percent_error": 0.5,
        "matrix_max_passes": 24,
        "matrix_min_converged": 2,
        "matrix_skin_mesh": 1,
        "percent_error": 0.5,
        "max_passes": 18,
        "min_converged": 2,
    },
}
INTEGER_DESIGN_PARAMETERS = {
    "N1_main", "N1_side", "N2_main", "N2_side", "n_core_group",
}
MODE_VALUES = frozenset({"symmetry", "full"})
ACTION_VALUES = frozenset({"build", "solve"})
MANIFEST_SCHEMA_VERSION = 1
LOG_TAIL_BYTES = 24 * 1024
RESULT_MAX_BYTES = 8 * 1024 * 1024
GUI_RESULT_SCHEMA = "mft-local-aedt-gui-result-v1"
DEADLINE_HALF_LM_RESONANCE_MINIMUM_HZ = 15_000.0
DEFAULT_SOLVER_REVISION = "7251f407d66a157f83aa95f9be838baebccb8067"
DEFAULT_RUNNER_SHA256 = (
    "dd6ea5b59f855d671b80b08e57cb2cda27e46cede5b6880239db48bc406bdcfb"
)
DEFAULT_SOLVER_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_local_gui_solver_deployments/"
    "7251f40"
)


class LocalAedtGuiError(RuntimeError):
    """Base class for a rejected or failed local GUI launch."""


class CandidateLaunchError(LocalAedtGuiError):
    """The selected candidate is stale, incomplete, or non-physical."""


class DuplicateLaunchError(LocalAedtGuiError):
    """The same design/mode is already building or held open."""

    def __init__(self, message: str, launch: dict[str, Any]):
        super().__init__(message)
        self.launch = launch


class LaunchCapacityError(LocalAedtGuiError):
    """The bounded number of local interactive Desktops is already active."""


class LocalAedtGuiLauncher:
    """Create and report standalone model-only/hold AEDT GUI processes."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        solver_root: str | Path | None = None,
        runner_path: str | Path | None = None,
        expected_solver_revision: str | None = None,
        expected_solver_branch: str | None = None,
        expected_runner_sha256: str | None = None,
        expected_solver_source_runner_sha256: str | None = None,
        expected_solver_thermal_module_sha256: str | None = None,
        library_root: str | Path | None = None,
        expected_library_revision: str | None = None,
        extra_environment: dict[str, str] | None = None,
        runtime_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        max_active: int | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        process_probe: Callable[[int], bool] | None = None,
        aedt_process_probe: Callable[[dict[str, Any]], Any] | None = None,
        parameter_validator: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        solver_identity_provider: Callable[[Path], dict[str, Any]] | None = None,
        workspace_validator: Callable[[Path], None] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        configured_solver_root = os.environ.get(
            "MFT_LOCAL_AEDT_SOLVER_ROOT", ""
        ).strip()
        self.solver_root = Path(
            solver_root or configured_solver_root or DEFAULT_SOLVER_ROOT
        ).resolve()
        self.runner_path = Path(
            runner_path or self.solver_root / "run_simulation_260706.py"
        ).resolve()
        self.expected_solver_revision = str(
            expected_solver_revision
            or os.environ.get("MFT_LOCAL_AEDT_SOLVER_REVISION", "").strip()
            or DEFAULT_SOLVER_REVISION
        ).lower()
        self.expected_runner_sha256 = str(
            expected_runner_sha256
            or os.environ.get("MFT_LOCAL_AEDT_RUNNER_SHA256", "").strip()
            or DEFAULT_RUNNER_SHA256
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{40}", self.expected_solver_revision):
            raise ValueError("local AEDT solver revision must be an exact SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_runner_sha256):
            raise ValueError("local AEDT runner hash must be SHA-256")
        self.expected_solver_source_runner_sha256 = str(
            expected_solver_source_runner_sha256 or ""
        ).lower()
        if (
            self.expected_solver_source_runner_sha256
            and not re.fullmatch(
                r"[0-9a-f]{64}",
                self.expected_solver_source_runner_sha256,
            )
        ):
            raise ValueError("local AEDT source runner hash must be SHA-256")
        self.expected_solver_thermal_module_sha256 = str(
            expected_solver_thermal_module_sha256 or ""
        ).lower()
        if (
            self.expected_solver_thermal_module_sha256
            and not re.fullmatch(
                r"[0-9a-f]{64}",
                self.expected_solver_thermal_module_sha256,
            )
        ):
            raise ValueError("local AEDT thermal module hash must be SHA-256")
        self.expected_solver_branch = str(
            expected_solver_branch or ""
        ).strip()
        self.library_root = (
            Path(library_root).resolve() if library_root is not None else None
        )
        self.expected_library_revision = str(
            expected_library_revision or ""
        ).lower()
        if self.expected_library_revision and not re.fullmatch(
            r"[0-9a-f]{40}", self.expected_library_revision
        ):
            raise ValueError("local AEDT library revision must be an exact SHA")
        if bool(self.library_root) != bool(self.expected_library_revision):
            raise ValueError(
                "local AEDT library root and revision must be configured together"
            )
        self.extra_environment = {
            str(key): str(value)
            for key, value in (extra_environment or {}).items()
        }
        configured_runtime = os.environ.get("MFT_LOCAL_AEDT_GUI_RUNTIME", "").strip()
        if runtime_root is None:
            runtime_root = configured_runtime or (
                Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                / "MFT_1MW_2026" / "local-aedt-gui"
            )
        self.runtime_root = Path(runtime_root).resolve()
        self.manifest_root = self.runtime_root / "launches"
        self.parameter_root = self.runtime_root / "parameters"
        self.log_root = self.runtime_root / "logs"
        self.workspace_root = self.runtime_root / "workspaces"
        self.lock_path = self.runtime_root / "launcher.lock"
        self.python_executable = str(
            python_executable
            or os.environ.get("MFT_LOCAL_AEDT_PYTHON", "").strip()
            or sys.executable
        )
        configured_limit = os.environ.get("MFT_LOCAL_AEDT_GUI_MAX_ACTIVE", "2")
        self.max_active = int(max_active if max_active is not None else configured_limit)
        if not 1 <= self.max_active <= 4:
            raise ValueError("local AEDT GUI max_active must be between 1 and 4")
        self._popen_factory = popen_factory
        self._process_probe = process_probe or self._default_process_probe
        self._aedt_process_probe = (
            aedt_process_probe or self._default_aedt_process_probe
        )
        self._parameter_validator = parameter_validator or self._validate_with_solver
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._solver_identity_provider = (
            solver_identity_provider or self._read_solver_identity
        )
        self._workspace_validator = (
            workspace_validator or self._require_local_fixed_drive
        )
        self._solver_identity_cache: tuple[float, dict[str, Any]] | None = None
        self._children: dict[str, Any] = {}
        self._thread_lock = threading.RLock()

    @staticmethod
    def _default_process_probe(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            import psutil  # type: ignore

            process = psutil.Process(pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except ImportError:
            pass
        except Exception:
            return False
        if os.name == "nt":
            try:
                import ctypes

                query_limited_information = 0x1000
                still_active = 259
                handle = ctypes.windll.kernel32.OpenProcess(
                    query_limited_information, False, int(pid)
                )
                if not handle:
                    return False
                try:
                    exit_code = ctypes.c_ulong()
                    if not ctypes.windll.kernel32.GetExitCodeProcess(
                        handle, ctypes.byref(exit_code)
                    ):
                        return False
                    return int(exit_code.value) == still_active
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _require_local_fixed_drive(path: Path) -> None:
        """Reject UNC, mapped-network, removable, and RAM-disk workspaces."""
        resolved = path.resolve()
        if os.name != "nt":
            return
        if str(resolved).startswith("\\\\") or not resolved.drive:
            raise LocalAedtGuiError(
                "local AEDT launch workspace must be on a fixed local drive"
            )
        try:
            import ctypes

            drive_type = int(
                ctypes.windll.kernel32.GetDriveTypeW(resolved.anchor)
            )
        except Exception as exc:
            raise LocalAedtGuiError(
                "could not verify the local AEDT workspace drive type"
            ) from exc
        # DRIVE_FIXED == 3.  A mapped network drive reports DRIVE_REMOTE == 4.
        if drive_type != 3:
            raise LocalAedtGuiError(
                "local AEDT launch workspace must be on a fixed local drive"
            )

    @staticmethod
    def _default_aedt_process_probe(identity: dict[str, Any]) -> dict[str, Any]:
        """Verify an AEDT process without accepting a PID-reuse false positive."""
        pid = identity.get("pid")
        captured_create_time = identity.get("create_time")
        try:
            pid = int(pid)
            captured_create_time = float(captured_create_time)
        except (TypeError, ValueError, OverflowError):
            return {
                "verified": False,
                "alive": False,
                "error": "invalid_pid_or_create_time",
            }
        if pid <= 0 or not math.isfinite(captured_create_time):
            return {
                "verified": False,
                "alive": False,
                "pid": pid,
                "error": "invalid_pid_or_create_time",
            }
        try:
            import psutil  # type: ignore

            process = psutil.Process(pid)
            current_create_time = float(process.create_time())
            create_time_match = (
                abs(current_create_time - captured_create_time) <= 0.05
            )
            current_name = str(process.name() or "")
            current_executable = str(process.exe() or "")
            expected_name = str(identity.get("name") or "")
            expected_executable = str(identity.get("executable") or "")
            name_match = bool(
                not expected_name
                or current_name.casefold() == expected_name.casefold()
            )
            executable_match = bool(
                not expected_executable
                or os.path.normcase(os.path.normpath(current_executable))
                == os.path.normcase(os.path.normpath(expected_executable))
            )
            alive = bool(
                process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            )
            verified = bool(
                alive and create_time_match and name_match and executable_match
            )
            return {
                "verified": verified,
                "alive": alive,
                "pid": pid,
                "captured_create_time": captured_create_time,
                "current_create_time": current_create_time,
                "create_time_match": create_time_match,
                "name_match": name_match,
                "executable_match": executable_match,
                "current_name": current_name,
                "current_executable": current_executable,
                "error": None if verified else "aedt_process_identity_mismatch",
            }
        except ImportError:
            return {
                "verified": False,
                "alive": False,
                "pid": pid,
                "error": "psutil_unavailable",
            }
        except Exception as exc:
            return {
                "verified": False,
                "alive": False,
                "pid": pid,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _probe_retained_aedt(
        self, identity: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not isinstance(identity, dict):
            return {
                "verified": False,
                "alive": False,
                "error": "aedt_process_identity_unavailable",
            }
        try:
            raw = self._aedt_process_probe(dict(identity))
        except Exception as exc:
            return {
                "verified": False,
                "alive": False,
                "pid": identity.get("pid"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        if isinstance(raw, bool):
            return {
                "verified": raw,
                "alive": raw,
                "pid": identity.get("pid"),
                "error": None if raw else "aedt_process_probe_rejected",
            }
        if not isinstance(raw, dict):
            return {
                "verified": False,
                "alive": False,
                "pid": identity.get("pid"),
                "error": "aedt_process_probe_returned_invalid_payload",
            }
        verified = raw.get("verified") is True and raw.get("alive") is True
        return {
            **raw,
            "verified": verified,
            "alive": raw.get("alive") is True,
            "pid": raw.get("pid", identity.get("pid")),
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise OSError("runner is not a regular immutable file")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise OSError("runner changed while hashing")
        return digest.hexdigest()

    def _read_solver_identity(self, root: Path) -> dict[str, Any]:
        try:
            def git_value(checkout: Path, *arguments: str) -> str:
                return subprocess.run(
                    [
                        "git", "-c", "safe.directory=*", "-C", str(checkout),
                        *arguments,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()

            revision = git_value(root, "rev-parse", "HEAD").lower()
            branch = git_value(root, "branch", "--show-current")
            dirty = bool(git_value(
                root, "status", "--porcelain", "--untracked-files=no"
            ))
            runner_sha = self._file_sha256(self.runner_path)
            source_runner_sha = self._file_sha256(
                root / "run_simulation_260706.py"
            )
            thermal_module_sha = None
            if self.expected_solver_thermal_module_sha256:
                thermal_module_sha = self._file_sha256(
                    root / "module" / "thermal_260706.py"
                )
            library_revision = None
            library_dirty = None
            if self.library_root is not None:
                library_revision = git_value(
                    self.library_root, "rev-parse", "HEAD"
                ).lower()
                library_dirty = bool(git_value(
                    self.library_root,
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    "src",
                ))
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "available": False,
                "verified": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        verified = bool(
            revision == self.expected_solver_revision
            and runner_sha == self.expected_runner_sha256
            and (
                not self.expected_solver_source_runner_sha256
                or source_runner_sha
                == self.expected_solver_source_runner_sha256
            )
            and (
                not self.expected_solver_thermal_module_sha256
                or thermal_module_sha
                == self.expected_solver_thermal_module_sha256
            )
            and (
                not self.expected_solver_branch
                or branch == self.expected_solver_branch
            )
            and (
                self.library_root is None
                or (
                    library_revision == self.expected_library_revision
                    and library_dirty is False
                )
            )
            and not dirty
        )
        return {
            "available": True,
            "verified": verified,
            "root": str(root),
            "revision": revision,
            "expected_revision": self.expected_solver_revision,
            "branch": branch,
            "expected_branch": self.expected_solver_branch or None,
            "runner_sha256": runner_sha,
            "expected_runner_sha256": self.expected_runner_sha256,
            "source_runner_sha256": source_runner_sha,
            "expected_source_runner_sha256": (
                self.expected_solver_source_runner_sha256 or None
            ),
            "thermal_module_sha256": thermal_module_sha,
            "expected_thermal_module_sha256": (
                self.expected_solver_thermal_module_sha256 or None
            ),
            "library_root": (
                str(self.library_root) if self.library_root is not None else None
            ),
            "library_revision": library_revision,
            "expected_library_revision": (
                self.expected_library_revision or None
            ),
            "library_dirty": library_dirty,
            "dirty": dirty,
            "error": (
                None if verified else
                "local GUI solver/library branch/revision/hash/cleanliness mismatch"
            ),
        }

    def _solver_identity(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._solver_identity_cache
        if not force and cached is not None and now - cached[0] < 10:
            return dict(cached[1])
        try:
            identity = self._solver_identity_provider(self.solver_root)
        except Exception as exc:
            identity = {
                "available": False,
                "verified": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(identity, dict):
            identity = {
                "available": False,
                "verified": False,
                "error": "solver identity provider returned a non-object",
            }
        self._solver_identity_cache = (now, dict(identity))
        return dict(identity)

    def _validate_with_solver(self, parameters: dict[str, Any]) -> None:
        script = (
            "import json,sys; "
            "from module.input_parameter_260706 import "
            "create_input_parameter,validation_check; "
            "value=json.load(sys.stdin); "
            "validation_check(create_input_parameter(value),strict=True)"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(self.solver_root)
        completed = subprocess.run(
            [self.python_executable, "-c", script],
            cwd=str(self.solver_root),
            env=environment,
            input=json.dumps(parameters, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise ValueError(
                "immutable solver parameter validation failed: " + detail
            )

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _tail(path_text: Any) -> str:
        try:
            path = Path(str(path_text))
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - LOG_TAIL_BYTES))
                data = handle.read(LOG_TAIL_BYTES)
        except (OSError, TypeError, ValueError):
            return ""
        if not data:
            return ""
        for encoding in ("utf-8", "utf-16-le", "cp949"):
            try:
                decoded = data.decode(encoding)
                if encoding != "utf-16-le" or "\x00" not in decoded:
                    return decoded[-12000:]
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")[-12000:]

    def _manifest_path(self, launch_id: str) -> Path:
        return self.manifest_root / f"{launch_id}.json"

    def _read_manifests(self) -> list[dict[str, Any]]:
        if not self.manifest_root.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for path in self.manifest_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("schema_version") == MANIFEST_SCHEMA_VERSION
                and isinstance(payload.get("launch_id"), str)
            ):
                records.append(payload)
        return records

    @staticmethod
    def _validated_aedt_process_identity(
        value: Any, *, required: bool
    ) -> dict[str, Any] | None:
        if value is None and not required:
            return None
        if not isinstance(value, dict):
            raise ValueError("GUI result AEDT process identity is missing")
        expected_keys = {
            "pid", "create_time", "name", "executable",
            "grpc_port", "alive", "error",
        }
        if not expected_keys.issubset(value):
            raise ValueError("GUI result AEDT process identity is incomplete")
        pid = value.get("pid")
        create_time = value.get("create_time")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(create_time, bool)
            or not isinstance(create_time, (int, float))
            or not math.isfinite(float(create_time))
        ):
            raise ValueError("GUI result AEDT PID identity is invalid")
        for key in ("name", "executable"):
            if not isinstance(value.get(key), str):
                raise ValueError(
                    f"GUI result AEDT process {key} is invalid"
                )
        grpc_port = value.get("grpc_port")
        if grpc_port is not None and (
            isinstance(grpc_port, bool)
            or not isinstance(grpc_port, int)
            or grpc_port <= 0
            or grpc_port > 65_535
        ):
            raise ValueError("GUI result AEDT gRPC port is invalid")
        if not isinstance(value.get("alive"), bool):
            raise ValueError("GUI result AEDT liveness evidence is invalid")
        error = value.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("GUI result AEDT process error is invalid")
        if required and (value.get("alive") is not True or error is not None):
            raise ValueError("GUI result does not attest a live held AEDT")
        return {
            "pid": pid,
            "create_time": float(create_time),
            "name": value.get("name"),
            "executable": value.get("executable"),
            "grpc_port": grpc_port,
            "alive": value.get("alive"),
            "error": error,
        }

    @staticmethod
    def _echo_matches(actual: Any, expected: Any) -> bool:
        if isinstance(expected, str):
            return isinstance(actual, str) and actual == expected
        if isinstance(expected, bool) or not isinstance(
            expected, (int, float)
        ):
            return actual == expected
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return (
            math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    def _validate_required_result_echo(
        self,
        record: dict[str, Any],
        payload: dict[str, Any],
        status: str,
    ) -> None:
        expected = record.get("required_result_echo")
        if expected in (None, {}):
            return
        if not isinstance(expected, dict):
            raise ValueError("GUI result echo manifest is invalid")
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("GUI result has no parameter echo")
        launched_parameter_keys = set(
            record.get("launched_parameter_keys") or []
        )
        for key, value in expected.items():
            if key in launched_parameter_keys and not self._echo_matches(
                parameters.get(key), value
            ):
                raise ValueError(
                    f"GUI parameter echo mismatch for {key}"
                )
        if status != "completed":
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("completed GUI solve has no result echo")
        for key, value in expected.items():
            if not self._echo_matches(result.get(key), value):
                raise ValueError(f"GUI result echo mismatch for {key}")

    def _read_result_artifact(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        result_text = record.get("result_path")
        workspace_text = record.get("workspace")
        if not result_text or not workspace_text:
            return None, None
        try:
            workspace = Path(str(workspace_text)).resolve()
            path = Path(str(result_text)).resolve()
            if path.parent != workspace or path.is_symlink():
                raise ValueError("GUI result artifact escapes its launch workspace")
            stat = path.stat()
            if not path.is_file() or stat.st_size <= 0:
                raise ValueError("GUI result artifact is not a regular file")
            if stat.st_size > RESULT_MAX_BYTES:
                raise ValueError("GUI result artifact exceeds safety limit")
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("GUI result artifact is not an object")
            status = payload.get("status")
            if (
                payload.get("schema") != GUI_RESULT_SCHEMA
                or payload.get("launch_id") != record.get("launch_id")
                or payload.get("candidate_id") != record.get("candidate_id")
                or str(payload.get("solver_revision") or "").lower()
                != self.expected_solver_revision
                or payload.get("solver_dirty") != 0
                or status not in {
                    "completed", "failed_held", "failed", "model_ready"
                }
            ):
                raise ValueError("GUI result artifact identity is invalid")
            self._validate_required_result_echo(record, payload, str(status))
            held_open = payload.get("held_open") is True
            if status in {"completed", "failed_held", "model_ready"} and not held_open:
                raise ValueError("GUI result status requires held_open evidence")
            detach_confirmed = payload.get("detach_confirmed")
            detach_error = payload.get("detach_error")
            if detach_confirmed not in {True, False, None}:
                raise ValueError("GUI result detach confirmation is invalid")
            if detach_error is not None and not isinstance(detach_error, str):
                raise ValueError("GUI result detach error is invalid")
            if held_open != (detach_confirmed is True):
                raise ValueError("GUI result held/detach evidence disagrees")
            if status in {"completed", "failed_held", "model_ready"} and (
                detach_confirmed is not True or detach_error is not None
            ):
                raise ValueError(
                    "GUI result held status requires confirmed clean detach"
                )
            if detach_confirmed is False and not detach_error:
                raise ValueError("GUI result failed detach has no error evidence")
            aedt_process = self._validated_aedt_process_identity(
                payload.get("aedt_process"),
                required=status in {"completed", "failed_held", "model_ready"},
            )
            project_text = payload.get("project_path")
            project_path: str | None = None
            if project_text is not None:
                if not isinstance(project_text, str) or not project_text.strip():
                    raise ValueError("GUI result project path is invalid")
                resolved_project = Path(project_text).resolve()
                if not resolved_project.is_relative_to(workspace):
                    raise ValueError(
                        "GUI result project escapes its launch workspace"
                    )
                project_path = str(resolved_project)
            elif held_open:
                raise ValueError("GUI held result has no project path")
            error_payload = payload.get("error")
            error_payload = error_payload if isinstance(error_payload, dict) else None
            reports = payload.get("inspection_reports")
            reports = reports if isinstance(reports, list) else []
            return {
                "schema": GUI_RESULT_SCHEMA,
                "status": status,
                "success": status in {"completed", "model_ready"},
                "held_open": held_open,
                "detach_confirmed": detach_confirmed,
                "detach_error": detach_error,
                "created_at": payload.get("created_at_utc"),
                "launch_id": payload.get("launch_id"),
                "candidate_id": payload.get("candidate_id"),
                "solver_revision": payload.get("solver_revision"),
                "project_name": payload.get("project_name"),
                "project_path": project_path,
                "aedt_process": aedt_process,
                "parameters": (
                    payload.get("parameters")
                    if isinstance(payload.get("parameters"), dict) else None
                ),
                "result": (
                    payload.get("result")
                    if isinstance(payload.get("result"), dict) else None
                ),
                "error": error_payload,
                "inspection_reports": reports[:64],
                "artifact": {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                },
            }, None
        except FileNotFoundError:
            return None, None
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _diagnose_logs(stdout_tail: str, stderr_tail: str) -> dict[str, Any] | None:
        combined = "\n".join((stdout_tail, stderr_tail))
        lowered = combined.lower()
        if "winerror 5" in lowered or "access is denied" in lowered or "액세스가 거부" in combined:
            return {
                "code": "workspace_access_denied",
                "summary": (
                    "기존 실행이 네트워크 solver checkout 아래에 .pyaedt 결과를 "
                    "기록해 접근 거부가 발생했습니다. 새 실행은 로컬 launch "
                    "workspace를 사용합니다."
                ),
            }
        if "result extraction failed" in lowered:
            return {
                "code": "result_extraction_failed",
                "summary": "FEA solve 뒤 결과 추출 단계에서 실패했습니다.",
            }
        if "desktop has been released and closed" in lowered:
            return {
                "code": "legacy_aedt_closed_on_error",
                "summary": "구버전 예외 경로가 AEDT를 닫았습니다.",
            }
        return None

    def _render_status(self, record: dict[str, Any]) -> dict[str, Any]:
        launch_id = str(record.get("launch_id") or "")
        pid = int(record.get("pid") or 0)
        child = self._children.get(launch_id)
        exit_code = None
        if child is not None:
            try:
                exit_code = child.poll()
            except Exception:
                exit_code = None
            launcher_active = exit_code is None
        else:
            launcher_active = self._process_probe(pid)
        stdout_tail = self._tail(record.get("stdout_path"))
        stderr_tail = self._tail(record.get("stderr_path"))
        result_artifact, result_error = self._read_result_artifact(record)
        result_status = (
            result_artifact.get("status") if result_artifact else None
        )
        held_attested = bool(
            result_artifact and result_artifact.get("held_open") is True
        )
        retained_probe = self._probe_retained_aedt(
            result_artifact.get("aedt_process")
            if held_attested and result_artifact else None
        )
        retained_aedt = bool(
            held_attested
            and retained_probe.get("verified") is True
            and retained_probe.get("alive") is True
        )
        if result_status == "completed" and retained_aedt:
            state = "completed_held"
        elif result_status == "completed" and launcher_active:
            state = "finalizing"
        elif result_status == "completed":
            state = "completed_detached"
        elif result_status == "model_ready" and retained_aedt:
            state = "ready"
        elif result_status == "model_ready" and launcher_active:
            state = "finalizing"
        elif result_status == "model_ready":
            state = "ready_detached"
        elif result_status == "failed_held" and retained_aedt:
            state = "failed_held"
        elif result_status == "failed_held" and launcher_active:
            state = "finalizing_failed"
        elif result_status == "failed_held":
            state = "failed_detached"
        elif result_status == "failed":
            state = "failed"
        elif result_error and not launcher_active:
            state = "failed_artifact"
        elif launcher_active and (
            "HOLD model-only mode" in stdout_tail
            or "=== HOLD:" in stdout_tail
        ):
            state = "building" if record.get("action") == "build" else "solving"
        elif launcher_active:
            state = "solving" if record.get("action") == "solve" else "building"
        elif exit_code == 0:
            state = "exited"
        else:
            state = "failed" if (exit_code is not None or stderr_tail) else "exited"
        diagnosis = self._diagnose_logs(stdout_tail, stderr_tail)
        result_error_payload = (
            result_artifact.get("error") if result_artifact else None
        )
        error_summary = (
            str(result_error_payload.get("message"))[:2000]
            if isinstance(result_error_payload, dict)
            and result_error_payload.get("message")
            else diagnosis.get("summary") if diagnosis else None
        )
        if error_summary is None and result_error:
            error_summary = result_error[:2000]
        if (
            error_summary is None
            and held_attested
            and not retained_aedt
            and not launcher_active
        ):
            error_summary = (
                "AEDT process retention could not be verified: "
                + str(retained_probe.get("error") or "identity mismatch")
            )
        return {
            **record,
            "active": launcher_active,
            "launcher_active": launcher_active,
            "retained_aedt": retained_aedt,
            "occupied": bool(launcher_active or retained_aedt),
            "state": state,
            "exit_code": exit_code,
            "held_open": retained_aedt,
            "held_open_attested": held_attested,
            "aedt_process_probe": retained_probe,
            "fea_validation": result_artifact,
            "result_artifact_error": result_error,
            "error_summary": error_summary,
            "diagnosis": diagnosis,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

    def _refresh_record(self, record: dict[str, Any]) -> dict[str, Any]:
        status = self._render_status(record)
        updates = {
            "last_observed_state": status.get("state"),
            "error_summary": status.get("error_summary"),
            "retained_aedt": status.get("retained_aedt") is True,
        }
        if record.get("last_observed_state") != status.get("state"):
            updates["last_observed_at"] = self._clock().isoformat(
                timespec="seconds"
            )
        if status.get("exit_code") is not None:
            updates["exit_code"] = int(status["exit_code"])
        validation = status.get("fea_validation")
        if isinstance(validation, dict):
            updates["result_status"] = validation.get("status")
            artifact = validation.get("artifact")
            if isinstance(artifact, dict):
                updates["result_sha256"] = artifact.get("sha256")
            if validation.get("created_at"):
                updates["finished_at"] = validation.get("created_at")
        elif status.get("exit_code") is not None:
            updates.setdefault(
                "finished_at", self._clock().isoformat(timespec="seconds")
            )
        changed = any(record.get(key) != value for key, value in updates.items())
        if changed:
            updated_record = {**record, **updates}
            self._atomic_json(
                self._manifest_path(str(record.get("launch_id"))),
                updated_record,
            )
            status.update(updated_record)
        return status

    @staticmethod
    def _candidate_id(candidate: dict[str, Any]) -> str:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str):
            raise CandidateLaunchError("candidate id is missing")
        candidate_id = candidate_id.strip()
        if not candidate_id or len(candidate_id) > 200 or any(
            ord(character) < 32 for character in candidate_id
        ):
            raise CandidateLaunchError("candidate id is invalid")
        return candidate_id

    def _deadline_contract(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any] | None:
        contract = candidate.get("local_gui_solver_contract")
        if contract is None:
            return None
        if not isinstance(contract, dict):
            raise CandidateLaunchError(
                "candidate local GUI solver contract is invalid"
            )
        required = {
            "solver_variant": "deadline-tim-k3",
            "solver_revision": self.expected_solver_revision,
            "solver_branch": self.expected_solver_branch,
            "solver_source_runner_sha256": (
                self.expected_solver_source_runner_sha256
            ),
            "solver_thermal_module_sha256": (
                self.expected_solver_thermal_module_sha256
            ),
            "gui_runner_sha256": self.expected_runner_sha256,
            "library_revision": self.expected_library_revision,
            "backend": "standalone",
            "keep_project": 1,
            "thermal_on": 1,
            "half_magnetizing_resonance_minimum_Hz": (
                DEADLINE_HALF_LM_RESONANCE_MINIMUM_HZ
            ),
            "thermal_pad_native_readback_contract_version": (
                "thermal-pad-native-material-readback-v1"
            ),
            "thermal_pad_native_readback_required": True,
            "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
            "thermal_pad_native_electrical_conductivity_S_m": 0.0,
            "execution_profile_contract_version": (
                DEADLINE_GUI_EXECUTION_PROFILE_VERSION
            ),
            "common_solver_parameters": (
                DEADLINE_GUI_COMMON_SOLVER_PARAMETERS
            ),
            "mode_solver_parameters": DEADLINE_GUI_MODE_SOLVER_PARAMETERS,
            "result_persistence_required": True,
            "retained_gui_required": True,
        }
        if any(contract.get(key) != value for key, value in required.items()):
            raise CandidateLaunchError(
                "candidate local GUI solver identity/lifecycle contract drifted"
            )
        echo_keys = contract.get("required_result_echo_keys")
        mandatory_echoes = {
            "fan_config", "fan_velocity", "plate_temp", "air_temp",
            "core_plate_pad_t", "wcp_pad_t",
            "thermal_pad_conductivity_W_mK",
            "thermal_pad_material_policy",
            "thermal_pad_native_readback_contract_version",
            "thermal_pad_native_readback_attested",
            "thermal_pad_native_thermal_conductivity_W_mK",
            "thermal_pad_native_electrical_conductivity_S_m",
            "P_target", "freq", "V1_rms", "V2_rms",
            "I1_rated", "I2_rated", "full_model", "loss_sym_on",
            "thermal_symmetry", "n_explicit_turns",
            "matrix_percent_error", "matrix_max_passes",
            "matrix_min_converged", "matrix_skin_mesh",
            "percent_error", "max_passes", "min_converged",
        }
        if (
            not isinstance(echo_keys, list)
            or not mandatory_echoes.issubset(
                {str(value) for value in echo_keys}
            )
        ):
            raise CandidateLaunchError(
                "candidate local GUI result-echo contract is incomplete"
            )
        resonance_constraint = candidate.get("constraints")
        resonance_constraint = (
            resonance_constraint.get("resonance")
            if isinstance(resonance_constraint, dict) else None
        )
        try:
            resonance = float(candidate.get("pred_f_res_min_screen_Hz"))
            minimum = float(
                candidate.get("resonance_minimum_required_Hz")
            )
            margin = float(candidate.get("resonance_margin_Hz"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CandidateLaunchError(
                "candidate half-Lm resonance contract is incomplete"
            ) from exc
        if (
            not all(math.isfinite(value) for value in (
                resonance, minimum, margin
            ))
            or minimum != DEADLINE_HALF_LM_RESONANCE_MINIMUM_HZ
            or resonance < minimum
            or not math.isclose(
                margin, resonance - minimum, rel_tol=0.0, abs_tol=1e-9
            )
            or not isinstance(resonance_constraint, dict)
            or resonance_constraint.get("pass") is not True
            or resonance_constraint.get("direction") != "minimum"
            or resonance_constraint.get("operator") != ">="
            or resonance_constraint.get("minimum_Hz") != minimum
            or resonance_constraint.get("limit") != minimum
            or resonance_constraint.get("value") != resonance
        ):
            raise CandidateLaunchError(
                "candidate half-Lm resonance must be >= 15 kHz"
            )
        return contract

    def _build_parameters(
        self, candidate: dict[str, Any], mode: str, action: str = "build"
    ) -> dict[str, Any]:
        if mode not in MODE_VALUES:
            raise CandidateLaunchError("model must be 'symmetry' or 'full'")
        if candidate.get("artifact_hydrated") is not True:
            raise CandidateLaunchError(
                "the Pareto artifact is not fully hydrated; refresh after artifact indexing"
            )
        raw_source = candidate.get("parameters")
        if not isinstance(raw_source, dict):
            raise CandidateLaunchError("candidate design parameters are missing")
        source = dict(raw_source)
        deadline_contract = self._deadline_contract(candidate)

        # Historical Pareto artifacts omit three validation/schema inputs.
        # They do not change the selected geometry in the currently supported
        # N1_side == 0 contract: ``w1c_w2s_space_x`` is only the minimum
        # clearance assertion, while the two side-to-side spacings apply only
        # when a primary side winding exists.  Recover those values from the
        # authenticated candidate contract instead of making every currently
        # displayed point unlaunchable.  Never guess them for a side-winding
        # design where they would affect geometry.
        if "w1c_w2s_space_x" not in source:
            insulation = candidate.get("min_insulation_mm")
            if insulation is None:
                constraints = candidate.get("constraints")
                if isinstance(constraints, dict):
                    insulation_check = constraints.get("insulation")
                    if isinstance(insulation_check, dict):
                        insulation = insulation_check.get("value")
            try:
                insulation_value = float(insulation)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandidateLaunchError(
                    "candidate is missing its authenticated insulation requirement"
                ) from exc
            if not math.isfinite(insulation_value) or insulation_value <= 0:
                raise CandidateLaunchError(
                    "candidate insulation requirement must be finite and positive"
                )
            source["w1c_w2s_space_x"] = insulation_value
        missing_side_spacings = [
            key for key in ("w2s_w1s_space_x", "w1s_w2s_space_y")
            if key not in source
        ]
        if missing_side_spacings:
            try:
                primary_side_turns = float(source.get("N1_side"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandidateLaunchError(
                    "candidate primary side winding contract is invalid"
                ) from exc
            if not math.isfinite(primary_side_turns) or primary_side_turns != 0:
                raise CandidateLaunchError(
                    "candidate is missing side-winding geometry parameters: "
                    + ", ".join(missing_side_spacings)
                )
            for key in missing_side_spacings:
                source[key] = 0.0
        numeric_keys = LOCAL_GUI_DESIGN_PARAMETER_KEYS + (
            DEADLINE_GUI_NUMERIC_PARAMETER_KEYS
            if deadline_contract is not None else ()
        )
        string_keys = (
            DEADLINE_GUI_STRING_PARAMETER_KEYS
            if deadline_contract is not None else ()
        )
        missing = [
            key for key in (*numeric_keys, *string_keys)
            if key not in source
        ]
        if missing:
            raise CandidateLaunchError(
                "candidate is missing exact GUI design parameters: " + ", ".join(missing)
            )
        parameters: dict[str, Any] = {}
        for key in numeric_keys:
            raw = source[key]
            if isinstance(raw, bool):
                raise CandidateLaunchError(f"candidate parameter {key} is not numeric")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandidateLaunchError(
                    f"candidate parameter {key} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise CandidateLaunchError(f"candidate parameter {key} is not finite")
            if key in INTEGER_DESIGN_PARAMETERS:
                if not value.is_integer():
                    raise CandidateLaunchError(f"candidate parameter {key} must be an integer")
                parameters[key] = int(value)
            else:
                parameters[key] = value
        for key in string_keys:
            value = source[key]
            if not isinstance(value, str) or not value.strip():
                raise CandidateLaunchError(
                    f"candidate parameter {key} is not a non-empty string"
                )
            parameters[key] = value.strip()
        if deadline_contract is not None:
            parameters.update(DEADLINE_GUI_COMMON_SOLVER_PARAMETERS)
            parameters.update(DEADLINE_GUI_MODE_SOLVER_PARAMETERS[mode])
        else:
            parameters.update({
                "core_plate_on": 1,
                "wcp_on": 1,
                "round_corner": 0,
                "full_model": 1 if mode == "full" else 0,
                "matrix_on": 1,
                "loss_on": 1,
                "thermal_on": 0,
                "keep_project": 1,
                "cap_on": 1,
            })
        try:
            self._parameter_validator(parameters)
        except CandidateLaunchError:
            raise
        except Exception as exc:
            raise CandidateLaunchError(
                f"selected design failed solver input validation: {type(exc).__name__}: {exc}"
            ) from exc
        return parameters

    def _launch_environment(
        self, launch_id: str, candidate_id: str
    ) -> dict[str, str]:
        environment = dict(os.environ)
        for key in list(environment):
            if (
                key.startswith("SLURM_")
                or key == "SIMULATION_ID"
                or key.startswith("MFT_AEDT_")
                or key.startswith("MFT_GUI_")
                or key == "MFT_SLURM_SCHEDULER_ROOT"
            ):
                environment.pop(key, None)
        environment["MFT_AEDT_BACKEND"] = "standalone"
        environment["MFT_GUI_LAUNCH_ID"] = launch_id
        environment["MFT_GUI_CANDIDATE_ID"] = candidate_id
        environment["PYTHONPATH"] = str(self.solver_root)
        if self.library_root is not None:
            environment["MFT_PYAEDT_LIBRARY_ROOT"] = str(self.library_root)
        environment.update(self.extra_environment)
        return environment

    def launch(
        self,
        candidate: dict[str, Any],
        mode: str,
        action: str = "build",
    ) -> dict[str, Any]:
        candidate_id = self._candidate_id(candidate)
        if action not in ACTION_VALUES:
            raise CandidateLaunchError("action must be 'build' or 'solve'")
        solver_identity = self._solver_identity(force=True)
        if solver_identity.get("verified") is not True:
            raise LocalAedtGuiError(
                "immutable local AEDT solver deployment is unavailable: "
                + str(solver_identity.get("error") or "identity mismatch")
            )
        deadline_contract = self._deadline_contract(candidate)
        parameters = self._build_parameters(candidate, mode, action)
        canonical = json.dumps(
            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        parameter_sha256 = hashlib.sha256(canonical).hexdigest()

        self._workspace_validator(self.workspace_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, FileLock(str(self.lock_path), timeout=30):
            current = [self._refresh_record(item) for item in self._read_manifests()]
            duplicate = next((
                item for item in current
                if item.get("occupied") is True
                and item.get("mode") == mode
                and item.get("parameter_sha256") == parameter_sha256
            ), None)
            if duplicate is not None:
                raise DuplicateLaunchError(
                    "the selected candidate/model is already open or building",
                    duplicate,
                )
            occupied = [
                item for item in current if item.get("occupied") is True
            ]
            if len(occupied) >= self.max_active:
                raise LaunchCapacityError(
                    f"{len(occupied)} local AEDT GUIs are running or retained; "
                    "close one before launching another"
                )

            launch_id = uuid.uuid4().hex
            workspace = self.workspace_root / launch_id
            workspace.mkdir(parents=True, exist_ok=False)
            if (
                workspace.is_symlink()
                or workspace.resolve().parent != self.workspace_root.resolve()
            ):
                raise LocalAedtGuiError(
                    "local AEDT launch workspace identity is invalid"
                )
            parameter_path = workspace / "parameters.json"
            result_path = workspace / "gui_result.json"
            stdout_path = workspace / "stdout.log"
            stderr_path = workspace / "stderr.log"
            self._atomic_json(parameter_path, parameters)
            command = [
                self.python_executable,
                "-u",
                str(self.runner_path),
                "--fixed",
                "--params",
                str(parameter_path),
                "--hold",
                "--result-json",
                str(result_path),
            ]
            if action == "build":
                command.append("--model-only")
            if mode == "full":
                command.append("--full")
            required_result_echo: dict[str, Any] = {}
            if deadline_contract is not None:
                for key in deadline_contract["required_result_echo_keys"]:
                    if key in parameters:
                        required_result_echo[key] = parameters[key]
                    elif key in candidate:
                        required_result_echo[key] = candidate[key]
                    else:
                        raise CandidateLaunchError(
                            f"candidate is missing required result echo {key}"
                        )
            creationflags = (
                int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                if os.name == "nt" else 0
            )
            try:
                with stdout_path.open("ab", buffering=0) as stdout_handle, \
                        stderr_path.open("ab", buffering=0) as stderr_handle:
                    process = self._popen_factory(
                        command,
                        cwd=str(workspace),
                        env=self._launch_environment(launch_id, candidate_id),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        creationflags=creationflags,
                    )
            except Exception as exc:
                raise LocalAedtGuiError(
                    f"could not start local AEDT GUI: {type(exc).__name__}: {exc}"
                ) from exc
            record = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "launch_id": launch_id,
                "candidate_id": candidate_id,
                "mode": mode,
                "action": action,
                "pid": int(process.pid),
                "started_at": self._clock().isoformat(timespec="seconds"),
                "parameter_sha256": parameter_sha256,
                "pareto_front_sha256": candidate.get("pareto_front_sha256"),
                "pareto_row_number": candidate.get("pareto_row_number"),
                "model_id": candidate.get("model_id"),
                "parameter_path": str(parameter_path),
                "workspace": str(workspace),
                "result_path": str(result_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "runner_path": str(self.runner_path),
                "solver_root": str(self.solver_root),
                "solver_revision": solver_identity.get("revision"),
                "runner_sha256": solver_identity.get("runner_sha256"),
                "solver_branch": solver_identity.get("branch"),
                "solver_source_runner_sha256": solver_identity.get(
                    "source_runner_sha256"
                ),
                "library_revision": solver_identity.get("library_revision"),
                "required_result_echo_keys": (
                    list(deadline_contract["required_result_echo_keys"])
                    if deadline_contract is not None else []
                ),
                "required_result_echo": required_result_echo,
                "launched_parameter_keys": sorted(parameters),
                "backend": "standalone",
                "headless": False,
                "model_only": action == "build",
                "solve_started": action == "solve",
                "command": command,
            }
            self._children[launch_id] = process
            self._atomic_json(self._manifest_path(launch_id), record)
            return self._refresh_record(record)

    def snapshot(self, candidate_id: str | None = None) -> dict[str, Any]:
        if candidate_id is not None and (
            not isinstance(candidate_id, str)
            or len(candidate_id) > 200
            or any(ord(character) < 32 for character in candidate_id)
        ):
            raise CandidateLaunchError("candidate id filter is invalid")
        self._workspace_validator(self.workspace_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, FileLock(str(self.lock_path), timeout=30):
            launches = [self._refresh_record(item) for item in self._read_manifests()]
        if candidate_id is not None:
            launches = [
                item for item in launches if item.get("candidate_id") == candidate_id
            ]
        launches.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        active_count = sum(item.get("active") is True for item in launches)
        retained_aedt_count = sum(
            item.get("retained_aedt") is True for item in launches
        )
        occupied_count = sum(
            item.get("occupied") is True for item in launches
        )
        solver_identity = self._solver_identity()
        return {
            "schema_version": 1,
            "available": solver_identity.get("verified") is True,
            "backend": "standalone",
            "runner_path": str(self.runner_path),
            "solver_identity": solver_identity,
            "result_schema": GUI_RESULT_SCHEMA,
            "runtime_root": str(self.runtime_root),
            "max_active": self.max_active,
            "active_count": active_count,
            "retained_aedt_count": retained_aedt_count,
            "occupied_count": occupied_count,
            "launches": launches[:20],
            "generated_at": self._clock().isoformat(timespec="seconds"),
        }

    def validation_index(self, limit: int = 1_000) -> dict[str, dict[str, Any]]:
        """Return one compact latest validation per Pareto candidate."""
        if not 1 <= int(limit) <= 10_000:
            raise ValueError("validation index limit is out of bounds")
        self._workspace_validator(self.workspace_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        with self._thread_lock, FileLock(str(self.lock_path), timeout=30):
            launches = [
                self._refresh_record(item) for item in self._read_manifests()
            ]
        launches.sort(
            key=lambda item: str(item.get("started_at") or ""), reverse=True
        )
        index: dict[str, dict[str, Any]] = {}
        for launch in launches[: int(limit)]:
            candidate_id = launch.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in index:
                continue
            validation = launch.get("fea_validation")
            index[candidate_id] = {
                key: launch.get(key) for key in (
                    "launch_id", "candidate_id", "mode", "action", "pid",
                    "started_at", "finished_at", "state", "active",
                    "launcher_active", "retained_aedt", "occupied",
                    "held_open", "held_open_attested", "exit_code",
                    "error_summary", "diagnosis", "aedt_process_probe",
                    "parameter_sha256", "pareto_front_sha256",
                    "pareto_row_number", "model_id", "workspace",
                    "result_path", "stdout_path", "stderr_path",
                    "solver_revision", "runner_sha256",
                )
            }
            index[candidate_id]["fea_validation"] = validation
            index[candidate_id]["result_artifact_error"] = launch.get(
                "result_artifact_error"
            )
        return index


class RoutedLocalAedtGuiLauncher:
    """Route deadline TIM-k3 candidates to their isolated exact solver."""

    def __init__(
        self,
        default_launcher: LocalAedtGuiLauncher,
        deadline_launcher: LocalAedtGuiLauncher | None,
    ) -> None:
        self.default_launcher = default_launcher
        self.deadline_launcher = deadline_launcher

    def _launcher(
        self, candidate: dict[str, Any]
    ) -> LocalAedtGuiLauncher:
        contract = candidate.get("local_gui_solver_contract")
        if isinstance(contract, dict) and contract.get(
            "solver_variant"
        ) == "deadline-tim-k3":
            if self.deadline_launcher is None:
                raise LocalAedtGuiError(
                    "exact deadline TIM-k3 local GUI deployment is not configured"
                )
            return self.deadline_launcher
        return self.default_launcher

    def launch(
        self,
        candidate: dict[str, Any],
        mode: str,
        action: str = "build",
    ) -> dict[str, Any]:
        return self._launcher(candidate).launch(candidate, mode, action)

    def _launchers(self) -> list[LocalAedtGuiLauncher]:
        values = [self.default_launcher]
        if self.deadline_launcher is not None:
            values.append(self.deadline_launcher)
        return values

    def snapshot(self, candidate_id: str | None = None) -> dict[str, Any]:
        snapshots = [
            launcher.snapshot(candidate_id=candidate_id)
            for launcher in self._launchers()
        ]
        launches = [
            launch
            for snapshot in snapshots
            for launch in snapshot.get("launches", [])
            if isinstance(launch, dict)
        ]
        launches.sort(
            key=lambda item: str(item.get("started_at") or ""), reverse=True
        )
        return {
            "schema_version": 1,
            "available": any(
                snapshot.get("available") is True for snapshot in snapshots
            ),
            "backend": "standalone",
            "routed": True,
            "solver_identities": [
                snapshot.get("solver_identity") for snapshot in snapshots
            ],
            "result_schema": GUI_RESULT_SCHEMA,
            "max_active": sum(
                int(snapshot.get("max_active") or 0)
                for snapshot in snapshots
            ),
            "active_count": sum(
                int(snapshot.get("active_count") or 0)
                for snapshot in snapshots
            ),
            "retained_aedt_count": sum(
                int(snapshot.get("retained_aedt_count") or 0)
                for snapshot in snapshots
            ),
            "occupied_count": sum(
                int(snapshot.get("occupied_count") or 0)
                for snapshot in snapshots
            ),
            "launches": launches[:20],
            "generated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }

    def validation_index(self, limit: int = 1_000) -> dict[str, dict[str, Any]]:
        if not 1 <= int(limit) <= 10_000:
            raise ValueError("validation index limit is out of bounds")
        merged: dict[str, dict[str, Any]] = {}
        for launcher in self._launchers():
            for candidate_id, value in launcher.validation_index(limit).items():
                current = merged.get(candidate_id)
                if current is None or str(value.get("started_at") or "") > str(
                    current.get("started_at") or ""
                ):
                    merged[candidate_id] = value
        ordered = sorted(
            merged.items(),
            key=lambda item: str(item[1].get("started_at") or ""),
            reverse=True,
        )
        return dict(ordered[: int(limit)])
