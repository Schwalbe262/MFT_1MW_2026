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

    def start_heartbeat(self, **kwargs):
        self.calls.append(("start_heartbeat", kwargs))

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
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
    with pytest.raises(RuntimeError, match="exactly one explicit acknowledgement"):
        adapter.aedt_backend()


def test_pooled_backend_rejects_ambiguous_dual_ack(monkeypatch):
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.setenv("MFT_AEDT_EXCLUSIVE_1TO1", "1")
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
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
    assert lease.calls[1][0] == "start_heartbeat"
    assert lease.calls[1][1]["heartbeat_seconds"] == 30
    assert lease.calls[2][0] == "connect"
    assert lease.calls[2][1]["desktop_factory"] == "factory"


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
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
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
    assert requests[0][2]["request_key"].startswith("mft-1to2-pilot:")


def test_shared_canary_requests_nonexclusive_session_without_pilot_barrier(
    tmp_path, monkeypatch
):
    lease = FakeLease()
    lease.exclusive_session = False
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        return lease

    marker = tmp_path / "must-not-exist.json"
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_CANARY", "1")
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_READY_FILE", str(marker))
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS", "3600")
    monkeypatch.setattr(
        adapter,
        "_scheduler_attach_module",
        lambda: SimpleNamespace(acquire_project_lease=acquire),
    )

    adapter.acquire_pooled_desktop(desktop_factory="factory", non_graphical=True)
    adapter.pilot_pre_solve_barrier("simulation_canary")

    assert requests[0][2]["exclusive_session"] is False
    assert requests[0][2]["request_key"].startswith("mft-1to2-canary:")
    assert not marker.exists()


def test_shared_pilot_barrier_writes_marker_without_hanging(tmp_path, monkeypatch):
    marker = tmp_path / "ready.json"
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
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


class FakeSharedDesktop:
    def __init__(self, sibling_running=True):
        self.sibling_running = sibling_running
        self.running_calls = 0
        self.active_project_calls = []
        self.active_config = "Local"
        self.registry_loads = []
        self.registry_sets = []

    def AreThereSimulationsRunning(self):
        self.running_calls += 1
        return self.sibling_running

    def SetActiveProject(self, name):
        self.active_project_calls.append(name)
        raise AssertionError("pooled preflight must not use Desktop active state")

    def GetRegistryString(self, _key):
        return self.active_config

    def SetRegistryFromFile(self, path):
        self.registry_loads.append(path)

    def SetRegistryString(self, key, value):
        self.registry_sets.append((key, value))
        self.active_config = value


class FakeOwnedDesign:
    def GetName(self):
        return "simulation_own;maxwell_loss"

    def GetDesignType(self):
        return "Maxwell 3D"

    def GetSolutionType(self):
        return "AC Magnetic"

    def GetModule(self, name):
        assert name == "AnalysisSetup"
        return SimpleNamespace(GetSetups=lambda: ["Setup1"])


class FakeOwnedProject:
    def __init__(self):
        self.design = FakeOwnedDesign()
        self.active_design_calls = []

    def GetName(self):
        return "simulation_own"

    def GetDesigns(self):
        return [self.design]

    def SetActiveDesign(self, name):
        self.active_design_calls.append(name)
        raise AssertionError("pooled preflight must enumerate its owned design")


def _pooled_preflight_harness(monkeypatch, *, own_running):
    from run_simulation_260706 import Simulation

    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_CANARY", "1")
    assert adapter.pooled_backend_enabled() is True

    desktop = FakeSharedDesktop(sibling_running=True)
    project = FakeOwnedProject()
    desktop_wrapper = SimpleNamespace(odesktop=desktop)
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "pooled"
    simulation.PROJECT_NAME = "simulation_own"
    simulation.solver_may_be_running = own_running
    simulation.desktop = desktop_wrapper
    simulation.project = SimpleNamespace(
        project=project,
        proj=project,
        desktop=desktop_wrapper,
    )
    simulation.design1 = SimpleNamespace(design_name="maxwell_loss")
    simulation._matrix_hpc_acf_path = "owned-matrix.acf"
    simulation._validated_matrix_hpc_acf = lambda path: path
    return simulation, desktop, project


def test_pooled_preflight_ignores_solving_sibling_when_owned_project_is_idle(
    monkeypatch,
):
    simulation, desktop, project = _pooled_preflight_harness(
        monkeypatch, own_running=False
    )

    context = simulation._prepare_copied_loss_native_analysis(
        max_attempts=1,
        timeout_s=0,
        sleeper=lambda _seconds: None,
    )

    assert context["odesign"] is project.design
    assert context["odesktop"] is desktop
    assert context["original_config"] == "Local"
    assert desktop.registry_loads == ["owned-matrix.acf"]
    assert desktop.running_calls == 0
    assert desktop.active_project_calls == []
    assert project.active_design_calls == []


def test_pooled_preflight_fails_when_owned_project_is_solving(monkeypatch):
    simulation, desktop, _project = _pooled_preflight_harness(
        monkeypatch, own_running=True
    )

    with pytest.raises(RuntimeError, match="this client's project: True"):
        simulation._prepare_copied_loss_native_analysis(
            max_attempts=1,
            timeout_s=0,
            sleeper=lambda _seconds: None,
        )

    assert desktop.running_calls == 0
    assert desktop.active_project_calls == []
    assert desktop.registry_loads == []
