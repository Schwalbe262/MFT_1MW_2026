import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "mft_goal_direct_full_cap_gui.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mft_goal_direct_full_cap_gui", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeNativeProject:
    def __init__(self):
        self.design_names = []
        self.active_designs = []

    def GetTopDesignList(self):
        return list(self.design_names)

    def SetActiveDesign(self, name):
        self.active_designs.append(name)
        return name


class FakeODesktop:
    def __init__(self):
        self.project = FakeNativeProject()
        self.active_projects = []

    def SetActiveProject(self, name):
        self.active_projects.append(name)
        return self.project


class FakeDesktop:
    def __init__(self):
        self.odesktop = FakeODesktop()
        self.release_calls = []

    def release_desktop(self, *, close_projects, close_on_exit):
        self.release_calls.append({
            "close_projects": close_projects,
            "close_on_exit": close_on_exit,
        })


class FakeSimulation:
    def __init__(self, desktop, *, fail_cap=False):
        self.desktop = desktop
        self.project = SimpleNamespace(desktop=desktop)
        self.fail_cap = fail_cap
        self.events = []
        self.design1 = None
        self.save_count = 0

    def create_project(self):
        self.events.append(("create_project",))

    def create_design(self, *, name, solution):
        self.events.append(("create_design", name, solution))
        self.design1 = "geometry-source"

    def create_core(self):
        self.events.append(("create_core",))

    def create_coil(self):
        self.events.append(("create_coil",))

    def create_capacitance_design(self, *, name):
        self.events.append(("create_capacitance_design", name))
        self.desktop.odesktop.project.design_names.append(
            "Maxwell 3D;maxwell_cap"
        )
        return "cap-design"

    def save_project(self):
        self.save_count += 1
        self.events.append(("save_project",))

    def get_capacitance_parameter(self):
        raise AssertionError("the fake extractor is populated by the dispatcher")

    def analyze_and_extract(self, label, extractor):
        self.events.append(("analyze_and_extract", label, extractor))
        if label != "cap":
            raise AssertionError(f"unexpected native solver stage: {label}")
        if self.fail_cap:
            raise RuntimeError("cap solve failed")
        self.df_cap = pd.DataFrame([{
            "C_tx_tx_F": 7.8045e-9,
            "C_rx_rx_F": 7.7127e-10,
            "C_tx_rx_F": 2.757e-10,
        }])
        return 63.5


class FakeRunner:
    def __init__(self, *, fail_cap=False):
        self.fail_cap = fail_cap
        self.desktop = FakeDesktop()
        self.sim = None
        self.loaded_parameters = None
        self.desktop_kwargs = None
        self.variable_bindings = []

    def _load_fixed_input_parameter(self, parameters):
        self.loaded_parameters = dict(parameters)
        return pd.DataFrame([parameters]), "physics-revision-test"

    @staticmethod
    def validation_check(frame, *, strict):
        assert strict is True
        return True, frame.copy()

    def pyDesktop(self, **kwargs):
        self.desktop_kwargs = dict(kwargs)
        return self.desktop

    def Simulation(self, *, desktop):
        assert desktop is self.desktop
        self.sim = FakeSimulation(desktop, fail_cap=self.fail_cap)
        return self.sim

    def set_design_variables(self, design, frame):
        self.variable_bindings.append((design, frame.copy()))


def _args(tmp_path, parameter_path):
    return argparse.Namespace(
        params=parameter_path,
        output_root=tmp_path / "out",
        project_name="reference-cap-test",
        cores=4,
        prior_ltx_uh=7745.74713934634,
        prior_lrx_uh=773038.495371942,
        prior_llt_uh=63.3481121527661,
    )


def test_load_parameters_supports_json_and_final_jsonl_record(tmp_path):
    json_path = tmp_path / "parameters.json"
    json_path.write_text(
        json.dumps({"parameters": {"cw1": 5.0}}),
        encoding="utf-8",
    )
    assert MODULE._load_parameters(json_path) == {"cw1": 5.0}

    jsonl_path = tmp_path / "parameters.jsonl"
    jsonl_path.write_text(
        "\n".join([
            json.dumps({"parameters": {"cw1": 1.0}}),
            "",
            json.dumps({"parameters": {"cw1": 5.0, "gap1": 1.6}}),
        ]),
        encoding="utf-8",
    )
    assert MODULE._load_parameters(jsonl_path) == {
        "cw1": 5.0,
        "gap1": 1.6,
    }


def test_write_status_atomically_replaces_document(tmp_path):
    status_path = tmp_path / "status.json"
    MODULE._write_status(status_path, stage="first", value=1)
    MODULE._write_status(status_path, stage="second", value=2)
    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "stage": "second",
        "value": 2,
    }
    assert not status_path.with_suffix(".json.tmp").exists()


def test_windows_aedt_path_budget_rejects_long_result_tree(tmp_path):
    MODULE._validate_project_path_budget(
        Path(r"C:\w\rc5\project\rc5"),
        "rc5",
        platform_name="nt",
    )
    long_name = "reference260706_full_nonrounded_cap_gui_v4"
    long_root = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
        r"\local_reference_drawing260706_full_cap_gui_v4\project"
    )
    with pytest.raises(ValueError, match="result path is too long"):
        MODULE._validate_project_path_budget(
            long_root / long_name,
            long_name,
            platform_name="nt",
        )


def test_direct_workflow_dispatches_cap_only_and_holds_gui(tmp_path):
    parameter_path = tmp_path / "parameters.json"
    parameter_path.write_text(
        json.dumps({
            "cw1": 5.0,
            "gap1": 1.6,
            "full_model": 0,
            "round_corner": 1,
            "loss_on": 1,
            "thermal_on": 1,
        }),
        encoding="utf-8",
    )
    runner = FakeRunner()

    exit_code = MODULE.run_direct_cap_gui(
        _args(tmp_path, parameter_path), runner
    )

    assert exit_code == 0
    assert runner.loaded_parameters["full_model"] == 1
    assert runner.loaded_parameters["round_corner"] == 0
    assert runner.loaded_parameters["matrix_on"] == 1
    assert runner.loaded_parameters["cap_on"] == 1
    assert runner.loaded_parameters["loss_on"] == 0
    assert runner.loaded_parameters["thermal_on"] == 0
    analyze_events = [
        event for event in runner.sim.events
        if event[0] == "analyze_and_extract"
    ]
    assert len(analyze_events) == 1
    assert analyze_events[0][1] == "cap"
    assert all(event[1] != "matrix" for event in analyze_events)
    assert runner.desktop.release_calls == [{
        "close_projects": False,
        "close_on_exit": False,
    }]
    assert runner.desktop.odesktop.project.active_designs[-1] == "maxwell_cap"

    status = json.loads(
        (tmp_path / "out" / "direct_cap_gui_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["stage"] == "complete_gui_held"
    assert status["active_design"] == "maxwell_cap"
    assert status["solver_dispatch"] == "cap_only"
    result = json.loads(
        (tmp_path / "out" / "direct_cap_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["solver_stage"] == "maxwell_cap"
    assert result["result"]["C_tx_tx_F"] == 7.8045e-9


def test_cap_failure_still_saves_reactivates_and_holds_gui(tmp_path):
    parameter_path = tmp_path / "parameters.json"
    parameter_path.write_text(
        json.dumps({"cw1": 5.0, "gap1": 1.6}),
        encoding="utf-8",
    )
    runner = FakeRunner(fail_cap=True)

    exit_code = MODULE.run_direct_cap_gui(
        _args(tmp_path, parameter_path), runner
    )

    assert exit_code == 1
    status = json.loads(
        (tmp_path / "out" / "direct_cap_gui_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["stage"] == "failed:maxwell_cap_solve"
    assert status["active_design"] == "maxwell_cap"
    assert status["error_type"] == "RuntimeError"
    assert runner.sim.save_count >= 2
    assert runner.desktop.odesktop.project.active_designs[-1] == "maxwell_cap"
    assert runner.desktop.release_calls == [{
        "close_projects": False,
        "close_on_exit": False,
    }]
