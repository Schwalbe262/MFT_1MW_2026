from __future__ import annotations

import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from module import aedt_pool_adapter as adapter


class FakeLease:
    lease_id = 17
    exclusive_session = True

    def __init__(self):
        self.calls = []
        self.http = SimpleNamespace(
            scheduler_url="http://scheduler:8000", bootstrap_token="boot"
        )
        self.client_token = "lease-token"
        self.workspace_path = ""

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

    def activate(self, *, project_name=""):
        self.calls.append(("activate", project_name))
        return {"state": "active", "project_name": project_name}

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


def test_activation_is_explicit_and_requires_project_creation_identity():
    lease = FakeLease()

    status = adapter.activate_project(lease, "simulation17")

    assert status == {"state": "active", "project_name": "simulation17"}
    assert lease.calls == [("activate", "simulation17")]


@pytest.mark.parametrize(
    ("raw_timeout", "expected"),
    (("0", 0.0), ("0.25", 0.25), ("900", 900.0)),
)
def test_pooled_fill_timeout_accepts_closed_interval_boundaries(
    monkeypatch, raw_timeout, expected,
):
    monkeypatch.setenv(adapter.POOL_FILL_TIMEOUT_ENV, raw_timeout)

    assert adapter.validate_pooled_fill_timeout() == expected


@pytest.mark.parametrize(
    ("raw_timeout", "message"),
    (
        ("", "must be numeric"),
        ("not-a-number", "must be numeric"),
        ("nan", "between 0 and 900"),
        ("inf", "between 0 and 900"),
        ("-0.001", "between 0 and 900"),
        ("900.001", "between 0 and 900"),
    ),
)
def test_invalid_fill_timeout_fails_before_workspace_or_lease_request(
    monkeypatch, raw_timeout, message,
):
    events = []
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.setenv("MFT_AEDT_EXCLUSIVE_1TO1", "1")
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.setenv(adapter.POOL_FILL_TIMEOUT_ENV, raw_timeout)
    monkeypatch.setattr(
        adapter,
        "_scheduler_attach_module",
        lambda: events.append("scheduler-client-import"),
    )
    monkeypatch.setattr(
        adapter,
        "pooled_workspace_path",
        lambda: events.append("workspace-create"),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter.acquire_pooled_desktop(
            desktop_factory="must-not-attach",
            non_graphical=True,
        )

    assert events == []


def test_run_loop_rejects_fill_timeout_before_session_or_modeling(monkeypatch):
    import run_simulation_260706 as runner

    events = []
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.setenv("MFT_AEDT_EXCLUSIVE_1TO1", "1")
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
    monkeypatch.setenv(adapter.POOL_FILL_TIMEOUT_ENV, "901")
    monkeypatch.setattr(
        runner,
        "_snapshot_descendants",
        lambda: events.append("snapshot"),
    )
    monkeypatch.setattr(
        runner,
        "_create_simulation_session",
        lambda: events.append("session-and-modeling"),
    )

    with pytest.raises(RuntimeError, match="between 0 and 900"):
        runner.run_one_loop()

    assert events == []


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


def test_pooled_acquire_always_requests_exclusive_session(monkeypatch, tmp_path):
    lease = FakeLease()
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        lease.workspace_path = kwargs["workspace_path"]
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
    monkeypatch.chdir(tmp_path)

    desktop, acquired = adapter.acquire_pooled_desktop(
        desktop_factory="factory",
        non_graphical=True,
    )

    assert desktop == "desktop"
    assert acquired is lease
    assert requests[0][2]["exclusive_session"] is True
    assert requests[0][2]["task_id"] == 123
    assert requests[0][2]["workload_family"] == "mft"
    assert requests[0][2]["project_namespace"] == "mft"
    assert requests[0][2]["isolation_policy"] == "exclusive"
    assert requests[0][2]["protocol_version"] == 2
    assert requests[0][2]["admission_timeout_seconds"] == 1800
    assert requests[0][2]["session_profile"] == adapter.pooled_session_profile()
    assert lease.calls[0][0] == "wait"
    assert lease.calls[1][0] == "connect"
    assert lease.calls[1][1]["desktop_factory"] == "factory"
    adapter.release_project(lease, wait_seconds=1)


def test_shared_pilot_requests_nonexclusive_session(monkeypatch, tmp_path):
    lease = FakeLease()
    lease.exclusive_session = False
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        lease.workspace_path = kwargs["workspace_path"]
        return lease

    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_1TO2_PILOT", "1")
    monkeypatch.delenv("MFT_AEDT_SHARED_CANARY", raising=False)
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.chdir(tmp_path)
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
    assert requests[0][2]["request_key"] == f"mft-1to2-pilot:{os.getpid()}"
    assert requests[0][2]["isolation_policy"] == "family"


def test_shared_canary_requests_nonexclusive_session_without_pilot_barrier(
    tmp_path, monkeypatch
):
    lease = FakeLease()
    lease.exclusive_session = False
    requests = []

    def acquire(url, project, **kwargs):
        requests.append((url, project, kwargs))
        lease.workspace_path = kwargs["workspace_path"]
        return lease

    marker = tmp_path / "must-not-exist.json"
    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_CANARY", "1")
    monkeypatch.setenv("MFT_AEDT_SCHEDULER_URL", "http://scheduler:8000")
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_READY_FILE", str(marker))
    monkeypatch.setenv("MFT_AEDT_PILOT_PRE_SOLVE_HANG_SECONDS", "3600")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        adapter,
        "_scheduler_attach_module",
        lambda: SimpleNamespace(acquire_project_lease=acquire),
    )

    adapter.acquire_pooled_desktop(desktop_factory="factory", non_graphical=True)
    adapter.pilot_pre_solve_barrier("simulation_canary")

    assert requests[0][2]["exclusive_session"] is False
    assert requests[0][2]["request_key"] == f"mft-1to2-canary:{os.getpid()}"
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
    assert fault_kinds == ["admission_timeout", "script_error"]


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
        self.active_config = "pyaedt_config"
        self.registry_loads = []
        self.registry_sets = []
        self.projects = []

    def AreThereSimulationsRunning(self):
        self.running_calls += 1
        return self.sibling_running

    def SetActiveProject(self, name):
        self.active_project_calls.append(name)
        raise AssertionError("pooled preflight must not use Desktop active state")

    def GetProjects(self):
        return list(self.projects)

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
    desktop.projects = [project]
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
    simulation._matrix_hpc_acf_path = None
    simulation._validated_matrix_hpc_acf = lambda _path: (_ for _ in ()).throw(
        AssertionError("pooled preflight must not require a client ACF")
    )
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
    assert context["original_config"] is None
    assert desktop.registry_loads == []
    assert desktop.registry_sets == []
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


def test_pooled_preflight_rejects_session_dso_profile_drift(monkeypatch):
    simulation, desktop, _project = _pooled_preflight_harness(
        monkeypatch, own_running=False
    )
    desktop.active_config = "Local"

    with pytest.raises(RuntimeError, match="pooled session DSO profile mismatch"):
        simulation._prepare_copied_loss_native_analysis(
            max_attempts=1,
            timeout_s=0,
            sleeper=lambda _seconds: None,
        )

    assert desktop.registry_loads == []
    assert desktop.registry_sets == []


def test_pooled_project_name_uses_task_and_pool_wide_lease_identity(
    monkeypatch, tmp_path
):
    from run_simulation_260706 import Simulation

    workspace = tmp_path / "pool-workspace"
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "pooled"
    simulation.aedt_lease = SimpleNamespace(
        lease_id=4321,
        workspace_path=str(workspace),
    )
    monkeypatch.setenv("SLURM_SCHED_TASK_ID", "9876")

    simulation.create_simulation_name()

    assert re.fullmatch(r"mft-9876-4321-[0-9a-f]{12}", simulation.PROJECT_NAME)
    assert Path(simulation.project_path).parent == workspace.resolve()


def _pooled_results_harness(tmp_path):
    from run_simulation_260706 import Simulation

    workspace = tmp_path / "lease-workspace"
    workspace.mkdir()
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "pooled"
    simulation.PROJECT_NAME = "mft-own"
    simulation.aedt_lease = SimpleNamespace(workspace_path=str(workspace))
    simulation.project_path = str(workspace / simulation.PROJECT_NAME)
    return simulation, workspace


def test_pooled_project_preclaims_world_writable_results_before_aedt_create(
    tmp_path,
):
    simulation, workspace = _pooled_results_harness(tmp_path)
    observed = []

    class Desktop:
        def create_project(self, *, path, name):
            results = Path(path) / f"{name}.aedtresults"
            observed.append((results.is_dir(), results.stat().st_mode & 0o777))
            return SimpleNamespace(name=name)

    simulation.desktop = Desktop()

    simulation._create_project_locked()

    results = workspace / "mft-own" / "mft-own.aedtresults"
    assert observed == [(True, 0o777)]
    assert results.is_dir()
    assert results.stat().st_mode & 0o777 == 0o777


def test_pooled_project_postcheck_rejects_aedt_replacing_results_with_file(
    tmp_path,
):
    simulation, _workspace = _pooled_results_harness(tmp_path)

    class Desktop:
        def create_project(self, *, path, name):
            results = Path(path) / f"{name}.aedtresults"
            assert results.is_dir()
            results.rmdir()
            results.write_text("replaced", encoding="utf-8")
            return SimpleNamespace(name=name)

    simulation.desktop = Desktop()

    with pytest.raises(RuntimeError, match="not a plain directory"):
        simulation._create_project_locked()


def test_pooled_project_reclaims_empty_aedt_replaced_results_root(
    tmp_path, monkeypatch,
):
    simulation, workspace = _pooled_results_harness(tmp_path)
    foreign_inode = [None]

    class Desktop:
        def create_project(self, *, path, name):
            results = Path(path) / f"{name}.aedtresults"
            results.rmdir()
            results.mkdir(mode=0o755)
            foreign_inode[0] = results.stat().st_ino
            return SimpleNamespace(name=name)

    original_chmod = os.chmod

    def reject_foreign_inode(path, mode):
        if (
                foreign_inode[0] is not None
                and Path(path).exists()
                and Path(path).stat().st_ino == foreign_inode[0]):
            raise PermissionError("AEDT host owns this inode")
        return original_chmod(path, mode)

    simulation.desktop = Desktop()
    monkeypatch.setattr(os, "chmod", reject_foreign_inode)

    simulation._create_project_locked()

    results = workspace / "mft-own" / "mft-own.aedtresults"
    assert results.stat().st_ino != foreign_inode[0]
    assert results.stat().st_mode & 0o777 == 0o777


def test_pooled_project_never_removes_nonempty_foreign_results_root(
    tmp_path, monkeypatch,
):
    simulation, workspace = _pooled_results_harness(tmp_path)
    foreign_inode = [None]

    class Desktop:
        def create_project(self, *, path, name):
            results = Path(path) / f"{name}.aedtresults"
            results.rmdir()
            results.mkdir(mode=0o755)
            (results / "solver-data").write_text("preserve", encoding="utf-8")
            foreign_inode[0] = results.stat().st_ino
            return SimpleNamespace(name=name)

    original_chmod = os.chmod

    def reject_foreign_inode(path, mode):
        if (
                foreign_inode[0] is not None
                and Path(path).exists()
                and Path(path).stat().st_ino == foreign_inode[0]):
            raise PermissionError("AEDT host owns this inode")
        return original_chmod(path, mode)

    simulation.desktop = Desktop()
    monkeypatch.setattr(os, "chmod", reject_foreign_inode)

    with pytest.raises(RuntimeError, match="non-empty foreign directory"):
        simulation._create_project_locked()

    results = workspace / "mft-own" / "mft-own.aedtresults"
    assert results.stat().st_ino == foreign_inode[0]
    assert (results / "solver-data").read_text(encoding="utf-8") == "preserve"


def test_pooled_results_reject_project_outside_lease_workspace(tmp_path):
    simulation, _workspace = _pooled_results_harness(tmp_path)
    outside = tmp_path / "outside" / simulation.PROJECT_NAME
    simulation.project_path = str(outside)

    with pytest.raises(RuntimeError, match="outside its lease workspace"):
        simulation._ensure_pooled_shared_results_directory()

    assert not outside.exists()


def test_standalone_results_helper_has_no_filesystem_side_effect(tmp_path):
    from run_simulation_260706 import Simulation

    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "standalone"
    simulation.PROJECT_NAME = "standalone-own"
    simulation.project_path = str(tmp_path / "must-not-exist")
    simulation.aedt_lease = None

    assert simulation._ensure_pooled_shared_results_directory(
        "maxwell_matrix"
    ) is None
    assert not Path(simulation.project_path).exists()


def test_pooled_results_permission_failure_is_fail_closed(tmp_path, monkeypatch):
    simulation, _workspace = _pooled_results_harness(tmp_path)
    original_chmod = os.chmod

    def deny_results_chmod(path, mode):
        if str(path).endswith(".aedtresults"):
            raise PermissionError("cross-account owner mismatch")
        return original_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", deny_results_chmod)

    with pytest.raises(
        RuntimeError, match="failed to prepare cross-account pooled AEDT results"
    ):
        simulation._ensure_pooled_shared_results_directory()


def test_pooled_results_preclaim_exact_design_alias_and_reject_traversal(
    tmp_path,
):
    simulation, workspace = _pooled_results_harness(tmp_path)

    alias = simulation._ensure_pooled_shared_results_directory(
        "maxwell_matrix"
    )

    expected = (
        workspace
        / "mft-own"
        / "mft-own.aedtresults"
        / "maxwell_matrix"
    )
    assert Path(alias) == expected
    assert expected.is_dir()
    assert expected.stat().st_mode & 0o777 == 0o777
    with pytest.raises(RuntimeError, match="unsafe pooled shared-results"):
        simulation._ensure_pooled_shared_results_directory("../sibling")
    assert not (workspace / "mft-own" / "sibling").exists()


@pytest.mark.parametrize(
    ("label", "solution_type"),
    (("matrix", "AC Magnetic"), ("cap", "Electrostatic"), ("loss", "AC Magnetic")),
)
def test_pooled_native_analyze_is_exact_and_runs_outside_automation_lock(
    monkeypatch, tmp_path, label, solution_type,
):
    from run_simulation_260706 import Simulation

    monkeypatch.setenv("MFT_AEDT_BACKEND", "pooled")
    monkeypatch.delenv("MFT_AEDT_EXCLUSIVE_1TO1", raising=False)
    monkeypatch.delenv("MFT_AEDT_SHARED_1TO2_PILOT", raising=False)
    monkeypatch.setenv("MFT_AEDT_SHARED_CANARY", "1")
    events = []
    lock_depth = [0]

    class Guard:
        def __enter__(self):
            lock_depth[0] += 1
            events.append("lock-enter")
            return self

        def __exit__(self, *_args):
            events.append("lock-exit")
            lock_depth[0] -= 1

    workspace = tmp_path / "lease-workspace"
    workspace.mkdir()

    class Lease:
        protocol_version = 2
        workspace_path = str(workspace)

        def automation_guard(self):
            return Guard()

    class Project:
        def Save(self):
            assert lock_depth[0] > 0
            events.append("save")
            return None

    class Design:
        def GetName(self):
            return f"maxwell_{label}"

        def Analyze(self, setup_name, blocking):
            assert lock_depth[0] == 0
            events.append(("analyze", setup_name, blocking))
            return 0

    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "pooled"
    simulation.aedt_lease = Lease()
    simulation.PROJECT_NAME = "mft-own"
    simulation.project_path = str(workspace / simulation.PROJECT_NAME)
    simulation.design1 = SimpleNamespace(design_name=f"maxwell_{label}")
    simulation.solve_attempts = {"matrix": 0, "cap": 0, "loss": 0}
    simulation.solver_may_be_running = False
    simulation.pooled_activation_done = False
    simulation.activate_pooled_for_solve = lambda: events.append("activate")
    project = Project()
    design = Design()
    contracts = []

    def verify(**kwargs):
        assert lock_depth[0] > 0
        contracts.append(kwargs)
        return project, design

    simulation._verified_pooled_native_setup = verify
    simulation._native_desktop_handle = lambda: SimpleNamespace(
        GetRegistryString=lambda _key: "pyaedt_config"
    )
    simulation.save_project = lambda strict=False: events.append(
        ("save-wrapper", strict)
    )
    simulation._log_recent_aedt_messages = lambda _label: None

    elapsed = simulation._analyze_exact_pooled_design(label)

    assert elapsed >= 0
    assert events[0] == "activate"
    analyze_event = ("analyze", "Setup1", True)
    assert analyze_event in events
    assert events.index("lock-exit") < events.index(analyze_event)
    assert events.count(analyze_event) == 1
    assert events.count("lock-enter") == 2
    assert contracts == [
        {
            "setup_name": "Setup1",
            "expected_design_type": "Maxwell 3D",
            "expected_solution_type": solution_type,
        },
        {
            "setup_name": "Setup1",
            "expected_design_type": "Maxwell 3D",
            "expected_solution_type": solution_type,
            "project_refresh_max_attempts": 3,
            "project_refresh_retry_delay": 0.5,
        },
    ]
    assert simulation.solve_attempts[label] == 1
    assert simulation.solver_may_be_running is False
    design_results = (
        workspace
        / "mft-own"
        / "mft-own.aedtresults"
        / f"maxwell_{label}"
    )
    assert design_results.is_dir()
    assert design_results.stat().st_mode & 0o777 == 0o777
