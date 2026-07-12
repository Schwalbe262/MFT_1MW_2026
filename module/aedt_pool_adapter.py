"""Opt-in bridge from the MFT runner to the scheduler AEDT session host.

The production default remains one runner-owned Desktop per process.  Pooled
mode is accepted only with an explicit 1:1 acknowledgement and always requests
an exclusive scheduler session; this module has no 1:2 code path.
"""

from __future__ import annotations

import importlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any


STANDALONE_BACKEND = "standalone"
POOLED_BACKEND = "pooled"
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
    if value == POOLED_BACKEND and os.environ.get(
        "MFT_AEDT_EXCLUSIVE_1TO1", ""
    ).strip() != "1":
        raise RuntimeError(
            "pooled AEDT is experimental and requires "
            "MFT_AEDT_EXCLUSIVE_1TO1=1"
        )
    return value


def pooled_backend_enabled() -> bool:
    return aedt_backend() == POOLED_BACKEND


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


def acquire_pooled_desktop(
    *,
    desktop_factory: Any,
    non_graphical: bool,
) -> tuple[Any, Any]:
    """Acquire one exclusive lease and attach without Desktop ownership."""
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
    lease = client.acquire_project_lease(
        scheduler_url,
        pending_project,
        request_key=(
            f"mft-1to1:{task_id or os.getpid()}:{uuid.uuid4().hex}"
        ),
        task_id=task_id,
        exclusive_session=True,
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
        desktop = lease.connect_desktop(
            non_graphical=non_graphical,
            desktop_factory=desktop_factory,
        )
    except Exception:
        try:
            lease.report_fault(
                "script_error",
                failure_message="MFT failed before project creation",
            )
            lease.release(wait_seconds=120)
        except Exception:
            pass
        raise
    return desktop, lease


def bind_project_name(lease: Any, project_name: str) -> None:
    if not project_name or not str(project_name).strip():
        raise RuntimeError("MFT project name is empty before pooled bind")
    status = lease.bind_project_name(str(project_name).strip())
    if str(status.get("project_name") or "") != str(project_name).strip():
        raise RuntimeError("scheduler lease project-name readback mismatch")


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
    if solver_may_run or "timeout" in lower or "timed out" in lower:
        kind = "solver_timeout"
    elif any(token in lower for token in ("grpc", "desktop died", "connection reset")):
        kind = "aedt_death"
    else:
        kind = "script_error"
    return lease.report_fault(
        kind,
        failure_message=text,
        sibling_grace_seconds=60,
    )
