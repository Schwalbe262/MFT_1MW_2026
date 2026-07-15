"""Opt-in bridge from the MFT runner to the scheduler AEDT session host.

The production default remains one runner-owned Desktop per process.  Pooled
mode requires an explicit exclusive 1:1, disposable shared 1:2 pilot, or
bounded shared 1:2 canary acknowledgement.  Standalone remains the default.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


STANDALONE_BACKEND = "standalone"
POOLED_BACKEND = "pooled"
EXCLUSIVE_1TO1_ACK = "MFT_AEDT_EXCLUSIVE_1TO1"
SHARED_1TO2_PILOT_ACK = "MFT_AEDT_SHARED_1TO2_PILOT"
SHARED_CANARY_ACK = "MFT_AEDT_SHARED_CANARY"
ISOLATION_POLICY_ENV = "MFT_AEDT_ISOLATION_POLICY"
SESSION_VERSION_ENV = "MFT_AEDT_SESSION_VERSION"
DEFAULT_SESSION_VERSION = "2025.2"
POOL_HPC_CORES = 4
TERMINAL_LEASE_STATES = {
    "released",
    "failed",
    "cancelled",
    "expired",
}


def aedt_backend() -> str:
    value = os.environ.get("MFT_AEDT_BACKEND", STANDALONE_BACKEND).strip().lower()
    if value not in {STANDALONE_BACKEND, POOLED_BACKEND}:
        raise RuntimeError(
            "MFT_AEDT_BACKEND must be 'standalone' or 'pooled'"
        )
    if value == POOLED_BACKEND:
        acknowledgements = {
            EXCLUSIVE_1TO1_ACK: os.environ.get(EXCLUSIVE_1TO1_ACK, "").strip() == "1",
            SHARED_1TO2_PILOT_ACK: os.environ.get(SHARED_1TO2_PILOT_ACK, "").strip() == "1",
            SHARED_CANARY_ACK: os.environ.get(SHARED_CANARY_ACK, "").strip() == "1",
        }
        if sum(acknowledgements.values()) != 1:
            raise RuntimeError(
                "pooled AEDT requires exactly one explicit acknowledgement: "
                "MFT_AEDT_EXCLUSIVE_1TO1=1 or "
                "MFT_AEDT_SHARED_1TO2_PILOT=1 or "
                "MFT_AEDT_SHARED_CANARY=1"
            )
    return value


def pooled_backend_enabled() -> bool:
    return aedt_backend() == POOLED_BACKEND


def shared_1to2_pilot_enabled() -> bool:
    return (
        aedt_backend() == POOLED_BACKEND
        and os.environ.get(SHARED_1TO2_PILOT_ACK, "").strip() == "1"
    )


def shared_canary_enabled() -> bool:
    return (
        aedt_backend() == POOLED_BACKEND
        and os.environ.get(SHARED_CANARY_ACK, "").strip() == "1"
    )


def shared_1to2_enabled() -> bool:
    return shared_1to2_pilot_enabled() or shared_canary_enabled()


def _scheduler_attach_module() -> Any:
    root_text = os.environ.get("MFT_SLURM_SCHEDULER_ROOT", "").strip()
    if not root_text:
        raise RuntimeError(
            "MFT_SLURM_SCHEDULER_ROOT is required for pooled AEDT"
        )
    root = Path(root_text).expanduser().resolve()
    expected = root / "slurm_scheduler" / "aedt_attach_client.py"
    if not expected.is_file():
        raise RuntimeError(
            f"scheduler attach client is missing: {expected}"
        )
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    module = importlib.import_module("slurm_scheduler.aedt_attach_client")
    loaded = Path(module.__file__).resolve()
    if loaded != expected:
        raise RuntimeError(
            "loaded scheduler attach client does not match the configured root: "
            f"expected={expected}, actual={loaded}"
        )
    return module


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def pooled_session_profile() -> dict[str, Any]:
    """Desktop-global settings that MFT and IPMSM may safely share."""
    version = os.environ.get(SESSION_VERSION_ENV, DEFAULT_SESSION_VERSION).strip()
    if not version:
        raise RuntimeError(f"{SESSION_VERSION_ENV} must not be blank")
    return {
        "profile_version": 2,
        "aedt_version": version,
        "python_environment": "pyaedt2026v1",
        "filesystem": "gpfs-shared-v1",
        "desktop_dso": {
            "config_name": "pyaedt_config",
            # Every solver used by the full MFT workflow is Desktop-global.
            # Icepak explicitly disables auto settings in PyAEDT while both
            # Maxwell design types use it, so encode each read-back contract.
            "designs": {
                "Icepak": {
                    "cores": POOL_HPC_CORES,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": False,
                },
                "Maxwell 2D": {
                    "cores": POOL_HPC_CORES,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": True,
                },
                "Maxwell 3D": {
                    "cores": POOL_HPC_CORES,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": True,
                },
            },
        },
    }


def pooled_isolation_policy(*, exclusive: bool) -> str:
    if exclusive:
        return "exclusive"
    policy = os.environ.get(ISOLATION_POLICY_ENV, "family").strip().lower()
    if policy not in {"family", "shared_if_compatible"}:
        raise RuntimeError(
            f"{ISOLATION_POLICY_ENV} must be family or shared_if_compatible"
        )
    return policy


def pooled_workspace_path() -> Path:
    configured = os.environ.get("MFT_AEDT_POOL_WORKSPACE", "").strip()
    workspace = (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.cwd() / "simulation").resolve()
    )
    workspace.mkdir(parents=True, exist_ok=True, mode=0o777)
    try:
        workspace.chmod(0o777)
    except OSError as exc:
        raise RuntimeError(
            f"pooled AEDT workspace is not cross-account writable: {workspace}"
        ) from exc
    return workspace


def acquire_pooled_desktop(
    *,
    desktop_factory: Any,
    non_graphical: bool,
) -> tuple[Any, Any]:
    """Acquire a pilot lease and attach without Desktop ownership."""
    if not pooled_backend_enabled():
        raise RuntimeError("pooled Desktop acquisition requested while disabled")
    scheduler_url = os.environ.get("MFT_AEDT_SCHEDULER_URL", "").strip()
    if not scheduler_url.startswith(("http://", "https://")):
        raise RuntimeError(
            "MFT_AEDT_SCHEDULER_URL must be an http(s) URL"
        )
    client = _scheduler_attach_module()
    task_text = os.environ.get("SLURM_SCHED_TASK_ID", "").strip()
    task_id = int(task_text) if task_text.isdigit() else 0
    pending_project = (
        f"mft-pending-{task_id or os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    shared = shared_1to2_enabled()
    shared_mode = "canary" if shared_canary_enabled() else "pilot"
    workspace = pooled_workspace_path()
    request_mode = "1to2-" + shared_mode if shared else "1to1"
    lease = client.acquire_project_lease(
        scheduler_url,
        pending_project,
        # One process owns one durable intent. A lost HTTP response must not
        # turn a retry into a second queued/ghost lease.
        request_key=f"mft-{request_mode}:{task_id or os.getpid()}",
        task_id=task_id,
        exclusive_session=not shared,
        workload_family="mft",
        session_profile=pooled_session_profile(),
        project_namespace="mft",
        isolation_policy=pooled_isolation_policy(exclusive=not shared),
        workspace_path=str(workspace),
        protocol_version=2,
        heartbeat_seconds=_positive_int_env(
            "MFT_AEDT_LEASE_HEARTBEAT_SECONDS", 30
        ),
    )
    try:
        lease.wait_until_leased(
            timeout_seconds=_positive_int_env(
                "MFT_AEDT_LEASE_WAIT_SECONDS", 1800
            ),
            heartbeat_seconds=_positive_int_env(
                "MFT_AEDT_LEASE_HEARTBEAT_SECONDS", 30
            ),
        )
        lease_workspace = Path(str(getattr(lease, "workspace_path", "") or "")).resolve()
        if lease_workspace != workspace:
            raise RuntimeError(
                "pooled AEDT lease workspace readback mismatch: "
                f"requested={workspace}, actual={lease_workspace}"
            )
        # acquire_project_lease(protocol_version=2) owns one spawn-based child
        # keepalive from queue admission through release. Do not add a second
        # in-process heartbeat: native AEDT calls can hold the parent GIL.
        desktop = lease.connect_desktop(
            non_graphical=non_graphical,
            desktop_factory=desktop_factory,
        )
    except Exception as error:
        try:
            report_failure(
                lease,
                error,
                solver_may_run=False,
            )
            lease.release(wait_seconds=120)
        except Exception:
            pass
        raise
    return desktop, lease


def pilot_pre_solve_barrier(project_name: str) -> None:
    """Optional disposable 1:2 hook for a client-abort isolation test.

    It runs only under the explicit shared-pilot acknowledgement.  Production
    and exclusive 1:1 runs ignore the marker/hang variables entirely.
    """
    if not shared_1to2_pilot_enabled():
        return
    marker_text = os.environ.get("MFT_AEDT_PILOT_PRE_SOLVE_READY_FILE", "").strip()
    hang_text = os.environ.get("MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS", "0").strip()
    try:
        hang_seconds = int(hang_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS must be an integer"
        ) from exc
    if not 0 <= hang_seconds <= 3600:
        raise RuntimeError(
            "MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS must be between 0 and 3600"
        )
    if marker_text:
        marker = Path(marker_text).expanduser().resolve()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"project_name": project_name, "pid": os.getpid()}),
            encoding="utf-8",
        )
    if hang_seconds:
        time.sleep(hang_seconds)


def bind_project_name(lease: Any, project_name: str) -> None:
    if not project_name or not str(project_name).strip():
        raise RuntimeError("MFT project name is empty before pooled bind")
    status = lease.bind_project_name(str(project_name).strip())
    if str(status.get("project_name") or "") != str(project_name).strip():
        raise RuntimeError("scheduler lease project-name readback mismatch")


def activate_project(lease: Any, project_name: str) -> dict:
    """Declare a pooled lease active only after AEDT created the project."""
    if not project_name or not str(project_name).strip():
        raise RuntimeError("MFT project name is empty before pooled activation")
    status = lease.activate(project_name=str(project_name).strip())
    if str(status.get("state") or "") != "active":
        raise RuntimeError(
            "scheduler lease activation was not acknowledged: "
            f"state={status.get('state')!r}"
        )
    if str(status.get("project_name") or str(project_name).strip()) != str(
        project_name
    ).strip():
        raise RuntimeError("scheduler lease activation project-name mismatch")
    return status


def release_project(lease: Any, *, wait_seconds: int | None = None) -> dict:
    status = lease.release(
        wait_seconds=(
            _positive_int_env("MFT_AEDT_RELEASE_WAIT_SECONDS", 300)
            if wait_seconds is None else int(wait_seconds)
        )
    )
    state = str(status.get("state") or "")
    if state != "released":
        raise RuntimeError(
            f"pooled AEDT project close was not acknowledged: state={state!r}"
        )
    return status


def report_failure(lease: Any, error: BaseException, *, solver_may_run: bool) -> dict:
    text = f"{type(error).__name__}: {error}"[:4000]
    lower = text.lower()
    if solver_may_run:
        kind = "solver_timeout"
        phase = "solve"
    elif "timeout" in lower or "timed out" in lower:
        kind = "admission_timeout"
        phase = "admission"
    elif any(token in lower for token in ("grpc", "desktop died", "connection reset")):
        kind = "aedt_transport_death"
        phase = "attach_or_transport"
    else:
        kind = "script_error"
        phase = "pre_solve"
    return lease.report_fault(
        kind,
        phase=phase,
        evidence={
            "exception_type": type(error).__name__,
            "solver_may_run": bool(solver_may_run),
        },
        failure_message=text,
        sibling_grace_seconds=60,
    )
