from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from ansys.aedt.core.internal.errors import GrpcApiError

from run_simulation_260706 import Simulation


class _FakeNativeDesign:
    def __init__(
            self, project_name, design_name,
            design_type="Maxwell 3D", solution_type="AC Magnetic"):
        self._name = f"{project_name};{design_name}"
        self._design_type = design_type
        self._solution_type = solution_type

    def GetName(self):
        return self._name

    def GetDesignType(self):
        return self._design_type

    def GetSolutionType(self):
        return self._solution_type


class _FakeNativeProject:
    def __init__(self, name, design):
        self._name = name
        self._design = design
        self.save_calls = 0
        self.save_effects = []

    def GetName(self):
        return self._name

    def GetPath(self):
        return None

    def GetDesigns(self):
        return [self._design]

    def SetActiveDesign(self, _name):
        return self._design

    def Save(self):
        self.save_calls += 1
        effect = self.save_effects.pop(0) if self.save_effects else True
        if isinstance(effect, BaseException):
            raise effect
        return effect


class _FakeNativeDesktop:
    def __init__(self, projects):
        self.projects = list(projects)
        self.set_active_project_calls = []

    def GetProjects(self):
        return list(self.projects)

    def SetActiveProject(self, name):
        self.set_active_project_calls.append(name)
        for project in self.projects:
            if project.GetName() == name:
                return project
        return None

    def GetDefaultUnit(self, _unit_system):
        return None


class _FakeDesignSolutions:
    def __init__(self, native_design, design_type, solution_type):
        self._odesign = native_design
        self._design_type = design_type
        self._solution_type = solution_type

    @property
    def solution_type(self):
        return self._odesign.GetSolutionType()


class _FakeSolution:
    units_data = {"Matrix.L(Tx_winding,Tx_winding)": "H"}

    def data_real(self, _expression):
        return [2.5]


class _FakePost:
    def __init__(self, app):
        # PyAEDT PostProcessorCommon performs this lookup in __init__.
        self.scratch = app.working_directory
        self.calls = []

    def get_solution_data(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSolution()


class _FakePooledApp:
    def __init__(self, desktop, native_project, native_design, design_name):
        self._desktop_class = desktop
        self._desktop = desktop.odesktop
        self._odesktop = desktop.odesktop
        self._oproject = native_project
        self._odesign = native_design
        self._design_name = design_name
        self._design_type = "HFSS"
        self._temp_solution_type = "Modal"
        self._project_name = None
        self._project_path = None
        self._post = None
        self._modeler = object()
        self._mesh = object()
        self._materials = object()
        self._available_variations = object()
        self._setups = [object()]
        self._setup = "stale_setup"
        self._boundaries = {"stale": object()}
        self._oanalysis = object()
        self._oboundary = object()
        self._osolution = object()
        self._ofieldsreporter = object()
        self._oreportsetup = object()
        self._oeditor = object()
        self._aedt_version = "2026.1"
        self._logger = SimpleNamespace(oproject=native_project, odesign=native_design)
        self._global_logger = self._logger
        self.design_solutions = _FakeDesignSolutions(
            native_design, "HFSS", "Modal"
        )
        self.save_calls = 0

    @property
    def design_type(self):
        return self._design_type

    @property
    def solution_type(self):
        return self.design_solutions.solution_type

    @property
    def project_name(self):
        if self._project_name:
            return self._project_name
        self._project_name = self._oproject.GetName()
        # PyAEDT invalidates the path while refreshing the project name.
        self._project_path = None
        return self._project_name

    @property
    def project_path(self):
        if not self._project_path and self._oproject:
            self._project_path = self._oproject.GetPath()
        return self._project_path

    @property
    def working_directory(self):
        project_name = self.project_name.replace(" ", "_")
        toolkit_directory = Path(self.project_path) / f"{project_name}.pyaedt"
        return str(toolkit_directory / self._design_name)

    @property
    def post(self):
        if self._post is None:
            self._post = _FakePost(self)
        return self._post

    def save_project(self):
        self.save_calls += 1
        result = self._oproject.Save()
        if result is False:
            return False
        # Match PyAEDT Design.save_project's post-save cache invalidation.
        self._project_name = None
        self._project_path = None
        return True


class _FakeDesign:
    def __init__(self, app, design_name):
        self.solver_instance = app
        self.design_name = design_name

    @property
    def post(self):
        return self.solver_instance.post

    def save_project(self):
        return self.solver_instance.save_project()

    def export_rl_matrix(self, **kwargs):
        identity = (
            self.solver_instance.design_type,
            self.solver_instance.solution_type,
        )
        if identity != ("Maxwell 3D", "AC Magnetic"):
            raise RuntimeError(f"wrong RL export identity: {identity!r}")
        self.export_identity = identity
        output_file = kwargs.get("output_file")
        if output_file:
            Path(output_file).write_text(
                "Inductance Unit: uH\n\n"
                "1000Hz\n"
                "R,L\n"
                "Tx_winding  1.0E-03, 10  2.0E-03, 2\n"
                "Rx_winding  2.0E-03, 2  8.0E-02, 20\n",
                encoding="utf-8",
            )
        return True


def _simulation(tmp_path, backend):
    project_name = "simulation_own"
    design_name = "maxwell_matrix"
    native_design = _FakeNativeDesign(project_name, design_name)
    cached_project = _FakeNativeProject(project_name, native_design)
    fresh_project = _FakeNativeProject(project_name, native_design)
    stale_design = _FakeNativeDesign(
        "simulation_sibling", design_name,
        design_type="HFSS", solution_type="Modal",
    )
    stale_project = _FakeNativeProject("simulation_sibling", stale_design)
    native_desktop = _FakeNativeDesktop([fresh_project])
    desktop = SimpleNamespace(odesktop=native_desktop)
    stale_desktop = SimpleNamespace(
        odesktop=_FakeNativeDesktop([stale_project])
    )
    app = _FakePooledApp(
        stale_desktop, stale_project, stale_design, design_name
    )
    design = _FakeDesign(app, design_name)

    project_path = tmp_path / project_name
    project_path.mkdir()
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = backend
    simulation.aedt_lease = SimpleNamespace(protocol_version=1)
    simulation.PROJECT_NAME = project_name
    simulation.project_path = str(project_path)
    simulation.desktop = desktop
    simulation.project = SimpleNamespace(
        project=cached_project,
        proj=cached_project,
        desktop=desktop,
    )
    simulation.design1 = design
    return simulation, app, design, project_path


def test_pooled_project_refresh_retries_only_transient_getprojects_grpc(
        tmp_path, caplog):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    native_desktop = simulation.desktop.odesktop
    expected_project = native_desktop.projects[0]
    effects = [
        GrpcApiError("transient enumeration one"),
        GrpcApiError("transient enumeration two"),
        [expected_project],
    ]
    calls = []
    sleeps = []

    def get_projects():
        calls.append(True)
        effect = effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect

    native_desktop.GetProjects = get_projects

    with caplog.at_level("WARNING"):
        refreshed = simulation._refresh_native_project_handle(
            get_projects_max_attempts=3,
            get_projects_retry_delay=0.5,
            sleeper=sleeps.append,
        )

    assert refreshed is expected_project
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert simulation.project.project is expected_project
    assert simulation.project.proj is expected_project
    assert caplog.text.count("GetProjects transient gRPC failure") == 2


def test_native_desktop_handle_uses_only_endpoint_project_attested_cache(
        tmp_path, caplog):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    native_desktop = simulation.desktop.odesktop
    native_desktop.GetProcessID = lambda: 8123
    native_desktop.GetGrpcServerPort = lambda: 57387
    simulation.aedt_lease = SimpleNamespace(
        protocol_version=2,
        state="active",
        session_process_id="8123",
        endpoint="nib110.hpc:57387",
    )

    assert simulation._native_desktop_handle() is native_desktop
    simulation.desktop.odesktop = None
    simulation.project.desktop.odesktop = None

    with caplog.at_level("WARNING"):
        recovered = simulation._native_desktop_handle()

    assert recovered is native_desktop
    assert "endpoint/project-attested cached native proxy" in caplog.text


def test_native_desktop_handle_rejects_dead_or_replaced_cached_endpoint(
        tmp_path):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    native_desktop = simulation.desktop.odesktop
    native_desktop.GetProcessID = lambda: 9999
    native_desktop.GetGrpcServerPort = lambda: 57387
    simulation.aedt_lease = SimpleNamespace(
        protocol_version=2,
        state="active",
        session_process_id="8123",
        endpoint="nib110.hpc:57387",
    )

    assert simulation._native_desktop_handle() is native_desktop
    simulation.desktop.odesktop = None
    simulation.project.desktop.odesktop = None

    with pytest.raises(
            RuntimeError, match="cached native Desktop PID mismatch"):
        simulation._native_desktop_handle()


def test_native_desktop_handle_reattaches_exact_lease_once_after_stale_cache(
        tmp_path, caplog):
    simulation, app, _design, _project_path = _simulation(tmp_path, "pooled")
    stale_desktop = simulation.desktop.odesktop
    stale_desktop.GetProcessID = lambda: 9999
    fresh_project = simulation.desktop.odesktop.projects[0]
    fresh_desktop = _FakeNativeDesktop([fresh_project])
    fresh_desktop.GetProcessID = lambda: 8123
    fresh_desktop.GetGrpcServerPort = lambda: 57387
    fresh_wrapper = SimpleNamespace(
        odesktop=fresh_desktop,
        aedt_process_id=8123,
        port=57387,
    )
    connect_calls = []

    def connect_desktop(**kwargs):
        connect_calls.append(kwargs)
        return fresh_wrapper

    simulation.aedt_lease = SimpleNamespace(
        protocol_version=2,
        state="active",
        session_process_id="8123",
        endpoint="nib110.hpc:57387",
        connect_desktop=connect_desktop,
    )

    assert simulation._native_desktop_handle() is stale_desktop
    simulation.desktop.odesktop = None
    simulation.project.desktop.odesktop = None

    with caplog.at_level("WARNING"):
        recovered = simulation._native_desktop_handle()

    assert recovered is fresh_desktop
    assert len(connect_calls) == 1
    assert isinstance(connect_calls[0]["non_graphical"], bool)
    assert connect_calls[0]["desktop_factory"].__name__ == "_PooledDesktop"
    assert simulation.desktop is fresh_wrapper
    assert simulation.project.desktop is fresh_wrapper
    assert app._desktop_class is fresh_wrapper
    assert app._desktop is fresh_desktop
    assert app._odesktop is fresh_desktop
    assert "one non-owning same-lease reattach" in caplog.text

    # The recovered wrapper can itself enter another failed recreation gap,
    # but one task must never create an unbounded reconnect loop.
    fresh_desktop.GetProcessID = lambda: 7777
    fresh_wrapper.odesktop = None
    simulation.project.desktop.odesktop = None
    with pytest.raises(RuntimeError, match="already attempted"):
        simulation._native_desktop_handle()
    assert len(connect_calls) == 1


def test_native_desktop_handle_rejects_cached_proxy_after_lease_release(
        tmp_path):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    native_desktop = simulation.desktop.odesktop
    simulation.aedt_lease = SimpleNamespace(
        protocol_version=2,
        state="released",
        session_process_id="",
        endpoint="",
    )

    assert simulation._native_desktop_handle() is native_desktop
    simulation.desktop.odesktop = None
    simulation.project.desktop.odesktop = None

    with pytest.raises(RuntimeError, match="lease is no longer active"):
        simulation._native_desktop_handle()


def test_pooled_project_refresh_does_not_retry_non_grpc_getprojects_error(
        tmp_path):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    calls = []
    sleeps = []

    def get_projects():
        calls.append(True)
        raise RuntimeError("non-gRPC enumeration failure")

    simulation.desktop.odesktop.GetProjects = get_projects

    with pytest.raises(RuntimeError, match="non-gRPC enumeration failure"):
        simulation._refresh_native_project_handle(
            get_projects_max_attempts=3,
            sleeper=sleeps.append,
        )

    assert len(calls) == 1
    assert sleeps == []


def test_pooled_project_refresh_does_not_retry_identity_absence(tmp_path):
    simulation, _app, _design, _project_path = _simulation(tmp_path, "pooled")
    sibling_design = _FakeNativeDesign("simulation_sibling", "maxwell_matrix")
    sibling_project = _FakeNativeProject("simulation_sibling", sibling_design)
    calls = []
    sleeps = []

    def get_projects():
        calls.append(True)
        return [sibling_project]

    simulation.desktop.odesktop.GetProjects = get_projects

    with pytest.raises(
            RuntimeError,
            match="expected one live pooled AEDT project named 'simulation_own', found 0",
    ):
        simulation._refresh_native_project_handle(
            get_projects_max_attempts=3,
            sleeper=sleeps.append,
        )

    assert len(calls) == 1
    assert sleeps == []


def test_pooled_close_skips_stale_cleanup_and_releases_exact_lease(
        monkeypatch, caplog):
    """Reproduce q8: a sibling changed the Desktop active project at close."""

    sibling = {"solution_deleted": False}
    lease = object()
    release_calls = []

    def stale_cleanup_solution():
        # PyAEDT 0.22 resolves ``odesign`` through its cached ``oproject``.
        # In q8 that SetActiveDesign lookup saw a sibling project and logged an
        # AEDT API Error.  A mutation-capable fake makes sibling safety explicit.
        sibling["solution_deleted"] = True
        raise RuntimeError("stale wrapper resolved the sibling active project")

    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "pooled"
    simulation.aedt_lease = lease
    simulation.pooled_release_done = False
    simulation.spawned_descendants = set()
    simulation.df_plus = pd.DataFrame([{"keep_project": 0}])
    simulation.design1 = SimpleNamespace(
        cleanup_solution=stale_cleanup_solution
    )

    def forbidden_client_transaction():
        raise AssertionError(
            "disposable pooled close must not enter a client AEDT transaction"
        )

    simulation.aedt_automation_transaction = forbidden_client_transaction
    monkeypatch.setattr(
        "run_simulation_260706._snapshot_descendants", lambda: {}
    )
    monkeypatch.setattr(
        "run_simulation_260706.release_pooled_project",
        lambda candidate: release_calls.append(candidate),
    )

    with caplog.at_level("INFO"):
        simulation.close_project()

    assert sibling["solution_deleted"] is False
    assert release_calls == [lease]
    assert simulation.pooled_release_done is True
    assert "Skipping cleanup_solution for disposable pooled project" in caplog.text


def test_standalone_close_keeps_existing_solution_cleanup(monkeypatch):
    cleanup_calls = []
    close_calls = []
    release_calls = []
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = "standalone"
    simulation.spawned_descendants = set()
    simulation.df_plus = pd.DataFrame([{"keep_project": 0}])
    simulation.design1 = SimpleNamespace(
        cleanup_solution=lambda: cleanup_calls.append(True),
        close_project=lambda: close_calls.append(True),
    )
    simulation.desktop = SimpleNamespace(
        release_desktop=lambda **kwargs: release_calls.append(kwargs)
    )
    simulation.aedt_automation_transaction = nullcontext
    monkeypatch.setattr(
        "run_simulation_260706._snapshot_descendants", lambda: {}
    )

    simulation.close_project()

    assert cleanup_calls == [True]
    assert close_calls == [True]
    assert release_calls == [{
        "close_projects": True,
        "close_on_exit": True,
    }]


def test_pooled_solution_data_hydrates_none_project_path_before_post(tmp_path):
    simulation, app, design, project_path = _simulation(tmp_path, "pooled")

    with pytest.raises(
        TypeError,
        match="expected str, bytes or os.PathLike object, not NoneType",
    ):
        _ = design.post
    # Start the fixed path from PyAEDT's post-save state, where both caches
    # have been invalidated.
    app._project_name = None
    app._project_path = None

    frame = simulation._solution_data_frame(
        ["Matrix.L(Tx_winding,Tx_winding)"],
        aliases=["Ltx"],
        target_units={"Matrix.L(Tx_winding,Tx_winding)": "uH"},
        report_category="AC Magnetic",
        report_context="Matrix",
        extraction_key="matrix",
        max_attempts=1,
        retry_delay=0,
    )

    assert frame["Ltx"].iloc[0] == 2_500_000.0
    assert app._oproject is simulation.project.project
    assert app._odesign is simulation.project.project._design
    assert app.design_solutions._odesign is app._odesign
    assert app._design_type == "Maxwell 3D"
    assert app.design_solutions._design_type == "Maxwell 3D"
    assert app.solution_type == "AC Magnetic"
    assert app._oreportsetup is None
    assert app._project_name == simulation.PROJECT_NAME
    assert app._project_path == str(project_path.resolve())
    assert app._post.scratch == str(
        project_path / "simulation_own.pyaedt" / "maxwell_matrix"
    )
    assert app._post.calls[0]["context"] == "Matrix"


def test_pooled_hydration_repoints_full_solution_identity_for_rl_export(tmp_path):
    simulation, app, design, project_path = _simulation(tmp_path, "pooled")
    stale_solutions = app.design_solutions
    stale_desktop = app._desktop
    cached_project = simulation.project.project
    simulation.df_plus = pd.DataFrame([{"freq": 1000.0}])
    simulation.extraction_attempts = {}
    simulation.extraction_backends = {}

    frame = simulation.get_magnetic_parameter()

    assert design.export_identity == ("Maxwell 3D", "AC Magnetic")
    assert frame["Ltx"].iloc[0] == 10.0
    assert simulation.extraction_backends["matrix"] == "export_rl_matrix"
    assert app._oproject is simulation.project.project
    assert app._oproject is not cached_project
    assert simulation.project.proj is app._oproject
    assert app._odesign is simulation.project.project._design
    assert app._desktop is simulation.desktop.odesktop
    assert app._desktop is not stale_desktop
    assert app._odesktop is simulation.desktop.odesktop
    assert app._design_name == "maxwell_matrix"
    assert app._design_type == "Maxwell 3D"
    assert app._temp_solution_type == "AC Magnetic"
    assert app.design_solutions is not stale_solutions
    assert app.design_solutions._odesign is app._odesign
    assert app.design_solutions._design_type == "Maxwell 3D"
    assert app.design_solutions._solution_type == "AC Magnetic"
    assert app.design_solutions.model_name == "Maxwell3DModel"
    assert "AC Magnetic" in app.design_solutions._solution_options
    assert app._project_name == "simulation_own"
    assert app._project_path == str(project_path.resolve())
    assert app._logger.oproject is app._oproject
    assert app._logger.odesign is app._odesign
    assert simulation._fields_reporter_project is app._oproject
    assert simulation.desktop.odesktop.set_active_project_calls == []
    assert app._oanalysis is None
    assert app._oboundary is None
    assert app._osolution is None
    assert app._ofieldsreporter is None
    assert app._oreportsetup is None
    assert app._oeditor is None
    assert app._post is None
    assert app._modeler is None
    assert app._mesh is None
    assert app._materials is None
    assert app._available_variations is None
    assert app._setups == []
    assert app._setup is None
    assert app._boundaries == {}


def test_pooled_save_rehydrates_and_retries_wrapper_once(
        tmp_path, caplog):
    simulation, app, _design, project_path = _simulation(tmp_path, "pooled")
    stale_project = app._oproject
    native_project = simulation.desktop.odesktop.projects[0]
    native_project.save_effects = [RuntimeError("transient Save"), True]

    assert simulation.save_project() is True

    assert app.save_calls == 2
    assert stale_project.save_calls == 0
    assert native_project.save_calls == 2
    assert app._oproject is native_project
    assert app._project_name == simulation.PROJECT_NAME
    assert app._project_path == str(project_path.resolve())
    assert simulation.stage_timings["stage_count_project_save"] == 1
    assert "pooled wrapper save failed; rehydrating" in caplog.text


def test_pooled_save_does_not_swallow_persistent_retry_failure(
        tmp_path, caplog):
    simulation, app, _design, _project_path = _simulation(tmp_path, "pooled")
    stale_project = app._oproject
    native_project = simulation.desktop.odesktop.projects[0]
    native_project.save_effects = [
        RuntimeError("first fresh Save"),
        RuntimeError("second fresh Save"),
    ]

    with pytest.raises(
        RuntimeError,
        match=r"Failed to save pooled project: .*first fresh Save.*second fresh Save",
    ):
        simulation.save_project()

    assert app.save_calls == 2
    assert stale_project.save_calls == 0
    assert native_project.save_calls == 2
    assert simulation.stage_timings["stage_count_project_save"] == 1
    assert "pooled wrapper save failed; rehydrating" in caplog.text


def test_solution_data_path_hydration_is_pooled_only(tmp_path):
    simulation, app, _design, _project_path = _simulation(
        tmp_path, "standalone"
    )
    original_project = app._oproject
    original_design = app._odesign
    original_solutions = app.design_solutions
    original_design_type = app._design_type
    original_analysis = app._oanalysis

    assert simulation._prepare_pooled_solution_data_app() is None
    assert app._oproject is original_project
    assert app._odesign is original_design
    assert app.design_solutions is original_solutions
    assert app._design_type == original_design_type
    assert app._oanalysis is original_analysis
    assert app._project_name is None
    assert app._project_path is None


def test_standalone_save_keeps_existing_wrapper_then_native_fallback(tmp_path):
    simulation, app, _design, _project_path = _simulation(
        tmp_path, "standalone"
    )
    stale_project = app._oproject
    native_project = simulation.project.project
    stale_project.save_effects = [RuntimeError("wrapper Save")]

    assert simulation.save_project(strict=True) is True

    assert app.save_calls == 1
    assert stale_project.save_calls == 1
    assert native_project.save_calls == 1
    assert app._oproject is stale_project
    assert app._design_type == "HFSS"


def test_standalone_non_strict_save_still_returns_false_on_persistent_failure(
        tmp_path):
    simulation, app, _design, _project_path = _simulation(
        tmp_path, "standalone"
    )
    stale_project = app._oproject
    native_project = simulation.project.project
    stale_project.save_effects = [RuntimeError("wrapper Save")]
    native_project.save_effects = [RuntimeError("native Save")]

    assert simulation.save_project() is False

    assert app.save_calls == 1
    assert stale_project.save_calls == 1
    assert native_project.save_calls == 1
    assert app._oproject is stale_project
