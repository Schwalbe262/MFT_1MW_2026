"""Opt-in bridge from the MFT runner to the scheduler AEDT session host.

The production default remains one runner-owned Desktop per process.  Pooled
mode requires an explicit exclusive 1:1, disposable shared 1:2 pilot, or
bounded shared 1:2 canary acknowledgement.  Standalone remains the default.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
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


# The solver's long native calls can hold the GIL for 10+ minutes, which
# starves any in-process heartbeat thread and expires the lease mid-solve.
# A child process is immune to the parent's GIL.  It exits by itself when
# the parent dies (orphaned to init) or when the lease goes terminal on the
# server (persistent HTTP errors), so a crashed client cannot pin its slot.
_HEARTBEAT_PROC_SRC = """
import os, time, urllib.request
url = os.environ["MFT_HB_URL"]
headers = {"Content-Type": "application/json"}
if os.environ.get("MFT_HB_BOOTSTRAP", ""):
    headers["X-AEDT-Bootstrap-Token"] = os.environ["MFT_HB_BOOTSTRAP"]
if os.environ.get("MFT_HB_TOKEN", ""):
    headers["X-AEDT-Lease-Token"] = os.environ["MFT_HB_TOKEN"]
interval = max(5, int(os.environ.get("MFT_HB_INTERVAL", "30")))
failures = 0
while failures < 20:
    if hasattr(os, "getppid") and os.getppid() == 1:
        break
    try:
        request = urllib.request.Request(
            url, data=b"{}", headers=headers, method="POST"
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        opener.open(request, timeout=25).read()
        failures = 0
    except Exception:
        failures += 1
    time.sleep(interval)
"""


def _spawn_heartbeat_process(lease: Any) -> Any:
    env = dict(os.environ)
    env["MFT_HB_URL"] = (
        f"{lease.http.scheduler_url}/api/aedt-pool/leases/{lease.lease_id}/heartbeat"
    )
    env["MFT_HB_BOOTSTRAP"] = str(getattr(lease.http, "bootstrap_token", "") or "")
    env["MFT_HB_TOKEN"] = str(lease.client_token or "")
    env["MFT_HB_INTERVAL"] = os.environ.get(
        "MFT_AEDT_LEASE_HEARTBEAT_SECONDS", "30"
    )
    return subprocess.Popen(
        [sys.executable, "-c", _HEARTBEAT_PROC_SRC],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_lease_keepalive(lease: Any) -> None:
    if getattr(lease, "_mft_hb_proc", None) is not None:
        return
    lease._mft_hb_proc = _spawn_heartbeat_process(lease)


def stop_lease_keepalive(lease: Any) -> None:
    proc = getattr(lease, "_mft_hb_proc", None)
    lease._mft_hb_proc = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    except Exception:
        pass


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
    lease = client.acquire_project_lease(
        scheduler_url,
        pending_project,
        request_key=(
            f"mft-{'1to2-' + shared_mode if shared else '1to1'}:"
            f"{task_id or os.getpid()}:{uuid.uuid4().hex}"
        ),
        task_id=task_id,
        exclusive_session=not shared,
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
        # wait_until_leased only heartbeats while queued.  The lease TTL is
        # far shorter than a Desktop attach or a solve, so ownership must be
        # kept alive from the moment the lease is granted until release/fault.
        # The in-process thread alone is not enough: the solver's native calls
        # hold the GIL for many minutes, so a child process keeps beating too.
        lease.start_heartbeat(
            heartbeat_seconds=_positive_int_env(
                "MFT_AEDT_LEASE_HEARTBEAT_SECONDS", 30
            )
        )
        start_lease_keepalive(lease)
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
        finally:
            stop_lease_keepalive(lease)
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


def release_project(lease: Any, *, wait_seconds: int | None = None) -> dict:
    stop_lease_keepalive(lease)
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
    stop_lease_keepalive(lease)
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
