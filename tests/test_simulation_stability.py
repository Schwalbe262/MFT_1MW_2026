from pathlib import Path
from types import SimpleNamespace

import pytest

from run_simulation_260706 import Simulation


class _FakeNativeDesign:
    def __init__(self, project_name, design_name):
        self._name = f"{project_name};{design_name}"

    def GetName(self):
        return self._name


class _FakeNativeProject:
    def __init__(self, name, design):
        self._name = name
        self._design = design

    def GetName(self):
        return self._name

    def GetPath(self):
        return None

    def GetDesigns(self):
        return [self._design]

    def SetActiveDesign(self, _name):
        return self._design


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
    def __init__(self, native_project, native_design, design_name):
        self._oproject = native_project
        self._odesign = native_design
        self._design_name = design_name
        self._project_name = None
        self._project_path = None
        self._post = None
        self._oreportsetup = object()
        self.design_solutions = SimpleNamespace(_odesign=native_design)

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


class _FakeDesign:
    def __init__(self, app, design_name):
        self.solver_instance = app
        self.design_name = design_name

    @property
    def post(self):
        return self.solver_instance.post


def _simulation(tmp_path, backend):
    project_name = "simulation_own"
    design_name = "maxwell_matrix"
    native_design = _FakeNativeDesign(project_name, design_name)
    native_project = _FakeNativeProject(project_name, native_design)
    stale_design = _FakeNativeDesign("simulation_sibling", design_name)
    stale_project = _FakeNativeProject("simulation_sibling", stale_design)
    app = _FakePooledApp(stale_project, stale_design, design_name)
    design = _FakeDesign(app, design_name)

    project_path = tmp_path / project_name
    project_path.mkdir()
    simulation = Simulation.__new__(Simulation)
    simulation.aedt_backend = backend
    simulation.PROJECT_NAME = project_name
    simulation.project_path = str(project_path)
    simulation.project = SimpleNamespace(
        project=native_project,
        proj=native_project,
    )
    simulation.design1 = design
    return simulation, app, design, project_path


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
    assert app._oreportsetup is None
    assert app._project_name == simulation.PROJECT_NAME
    assert app._project_path == str(project_path.resolve())
    assert app._post.scratch == str(
        project_path / "simulation_own.pyaedt" / "maxwell_matrix"
    )
    assert app._post.calls[0]["context"] == "Matrix"


def test_solution_data_path_hydration_is_pooled_only(tmp_path):
    simulation, app, _design, _project_path = _simulation(
        tmp_path, "standalone"
    )

    assert simulation._prepare_pooled_solution_data_app() is None
    assert app._project_name is None
    assert app._project_path is None
