from __future__ import annotations

from types import SimpleNamespace

import pytest

from module import aedt_pool_adapter as adapter


class FakeLease:
    lease_id = 17
    exclusive_session = True

    def __init__(self):
        self.calls = []

    def wait_until_leased(self, **kwargs):
        self.calls.append(("wait", kwargs))
        return {"state": "leased", "endpoint": "node:50001"}

    def connect_desktop(self, **kwargs):
        self.calls.append(("connect", kwargs))
        return "desktop"

    def bind_project_name(self, name):
        self.calls.append(("bind", name))
        return {"project_name": name}

    def release(self, **kwargs):
        self.calls.append(("release", kwargs))
        return {"state": "released"}

    def report_fault(self, kind, **kwargs):
        self.calls.append(("fault", kind, kwargs))
        return {"state": "releasing"}


def test_default_backend_is_standalone_and_does_not_require_scheduler(
    monkeypatch,
):
    monkeypatch.delenv("MFT_AEDT_BACKEND", raising=False)
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    assert adapter.aedt_backend() == "standalone"
    assert adapter.pooled_backend_enabled() is False


def test_pooled_backend_requires_explicit_exclusive_ack(monkeypatch):
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    with pytest.raises(RuntimeError, match="exactly one explicit acknowledgement"):
        adapter.aedt_backend()


def test_pooled_backend_rejects_ambiguous_dual_ack(monkeypatch):
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.setenv("MFT_AEDT_EXCLUSIVE_1TO1", "1")
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    with pytest.raises(RuntimeError, match="exactly one explicit acknowledgement"):
        adapter.aedt_backend()


def test_pooled_acquire_always_requests_exclusive_session(monkeypatch):
    lease = FakeLease()
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        return lease

    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.setenv("MFT_AEDT_EXCLUSIVE_1TO1", "1")
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", "123")
    monkeypatch.setattr(
        adapter,
        "_scheduler_attach_module",
        lambda: SimpleNamespace(acquire_project_lease=acquire),
    )

    desktop, acquired = adapter.acquire_pooled_desktop(
        desktop_factory="factory",
        non_graphical=True,
    )

    assert desktop == "desktop"
    assert acquired is lease
    assert requests[0][2]["exclusive_session"] is True
    assert requests[0][2]["task_id"] == 123
    assert lease.calls[0][0] == "wait"
    assert lease.calls[1][0] == "connect"
    assert lease.calls[1][1]["desktop_factory"] == "factory"


def test_shared_pilot_requests_nonexclusive_session(monkeypatch):
    lease = FakeLease()
    lease.exclusive_session = False
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        return lease

    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.setattr(
        adapter,
        "_scheduler_attach_module",
        lambda: SimpleNamespace(acquire_project_lease=acquire),
    )

    adapter.acquire_pooled_desktop(
        desktop_factory="factory",
        non_graphical=True,
    )

    assert requests[0][2]["exclusive_session"] is False
    assert requests[0][2]["request_key"].startswith("mft-1to2:")


def test_shared_pilot_barrier_writes_marker_without_hanging(tmp_path, monkeypatch):
    marker = tmp_path / "ready.json"
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_READY_FILE", str(marker))
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS", "0")

    adapter.pilot_pre_solve_barrier("simulation_a")

    assert marker.exists()
    assert marker.read_text(encoding="utf-8").find("simulation_a") >= 0


def test_bind_release_and_failure_classification(monkeypatch):
    lease = FakeLease()
    monkeypatch.setenv("MFT_AEDT_RELEASE_WAIT_SECONDS", "9")

    adapter.bind_project_name(lease, "simulation_pilot")
    assert adapter.release_project(lease)["state"] == "released"
    adapter.report_failure(
        lease,
        TimeoutError("solver timed out"),
        solver_may_run=False,
    )
    adapter.report_failure(
        lease,
        RuntimeError("bad input"),
        solver_may_run=False,
    )

    assert ("bind", "simulation_pilot") in lease.calls
    assert ("release", {"wait_seconds": 9}) in lease.calls
    fault_kinds = [call[1] for call in lease.calls if call[0] == "fault"]
    assert fault_kinds == ["solver_timeout", "script_error"]


def test_nonreleased_host_ack_fails_closed():
    lease = FakeLease()
    lease.release = lambda **_kwargs: {"state": "releasing"}
    with pytest.raises(RuntimeError, match="not acknowledged"):
        adapter.release_project(lease, wait_seconds=1)
