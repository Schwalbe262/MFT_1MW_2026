import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from regression_260707.monitoring.local_aedt_gui import (
    DEFAULT_RUNNER_SHA256,
    DEFAULT_SOLVER_REVISION,
    DEFAULT_SOLVER_ROOT,
    DEADLINE_GUI_COMMON_SOLVER_PARAMETERS,
    DEADLINE_GUI_EXECUTION_PROFILE_VERSION,
    DEADLINE_GUI_MODE_SOLVER_PARAMETERS,
    DEADLINE_GUI_NUMERIC_PARAMETER_KEYS,
    GUI_RESULT_SCHEMA,
    LOCAL_GUI_DESIGN_PARAMETER_KEYS,
    CandidateLaunchError,
    LocalAedtGuiError,
    LocalAedtGuiLauncher,
    RoutedLocalAedtGuiLauncher,
)


SOLVER_REVISION = "a" * 40
RUNNER_SHA256 = "b" * 64
SOURCE_RUNNER_SHA256 = "c" * 64
THERMAL_MODULE_SHA256 = "e" * 64
LIBRARY_REVISION = "d" * 40
SOLVER_BRANCH = "deadline-test"


def test_default_solver_contract_is_latest_immutable_detach_deployment():
    assert DEFAULT_SOLVER_REVISION == "7251f407d66a157f83aa95f9be838baebccb8067"
    assert DEFAULT_RUNNER_SHA256 == (
        "dd6ea5b59f855d671b80b08e57cb2cda27e46cede5b6880239db48bc406bdcfb"
    )
    assert DEFAULT_SOLVER_ROOT.name == "7251f40"


def _candidate():
    parameters = {key: 1.0 for key in LOCAL_GUI_DESIGN_PARAMETER_KEYS}
    parameters.update({
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 37,
        "N2_side": 23,
        "n_core_group": 4,
    })
    return {
        "id": "candidate-1",
        "artifact_hydrated": True,
        "parameters": parameters,
        "min_insulation_mm": 40.0,
        "pareto_front_sha256": "c" * 64,
        "pareto_row_number": 8,
        "model_id": "model-1",
    }


def _deadline_candidate():
    candidate = _candidate()
    candidate["parameters"].update({
        key: 50.0 for key in DEADLINE_GUI_NUMERIC_PARAMETER_KEYS
    })
    candidate["parameters"].update({
        "fan_velocity": 3.0,
        "plate_temp": 50.0,
        "air_temp": 50.0,
        "fan_config": "dual",
    })
    candidate.update({
        "thermal_pad_conductivity_W_mK": 3.0,
        "thermal_pad_material_policy": "deadline-k3",
        "thermal_pad_native_readback_contract_version": (
            "thermal-pad-native-material-readback-v1"
        ),
        "thermal_pad_native_readback_attested": 1,
        "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
        "thermal_pad_native_electrical_conductivity_S_m": 0.0,
        "pred_f_res_min_screen_Hz": 16_000.0,
        "resonance_minimum_required_Hz": 15_000.0,
        "resonance_margin_Hz": 1_000.0,
        "constraints": {
            "resonance": {
                "value": 16_000.0,
                "limit": 15_000.0,
                "pass": True,
                "direction": "minimum",
                "operator": ">=",
                "minimum_Hz": 15_000.0,
            },
        },
        "local_gui_solver_contract": {
            "solver_variant": "deadline-tim-k3",
            "solver_revision": SOLVER_REVISION,
            "solver_branch": SOLVER_BRANCH,
            "solver_source_runner_sha256": SOURCE_RUNNER_SHA256,
            "solver_thermal_module_sha256": THERMAL_MODULE_SHA256,
            "gui_runner_sha256": RUNNER_SHA256,
            "library_revision": LIBRARY_REVISION,
            "backend": "standalone",
            "keep_project": 1,
            "thermal_on": 1,
            "half_magnetizing_resonance_minimum_Hz": 15_000.0,
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
            "mode_solver_parameters": (
                DEADLINE_GUI_MODE_SOLVER_PARAMETERS
            ),
            "result_persistence_required": True,
            "retained_gui_required": True,
            "required_result_echo_keys": [
                "fan_config",
                "fan_velocity",
                "plate_temp",
                "air_temp",
                "core_plate_pad_t",
                "wcp_pad_t",
                "thermal_pad_conductivity_W_mK",
                "thermal_pad_material_policy",
                "thermal_pad_native_readback_contract_version",
                "thermal_pad_native_readback_attested",
                "thermal_pad_native_thermal_conductivity_W_mK",
                "thermal_pad_native_electrical_conductivity_S_m",
                *DEADLINE_GUI_COMMON_SOLVER_PARAMETERS,
                *DEADLINE_GUI_MODE_SOLVER_PARAMETERS["symmetry"],
                *DEADLINE_GUI_MODE_SOLVER_PARAMETERS["full"],
            ],
        },
    })
    return candidate


class FakeProcess:
    pid = 12345

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


def _launcher(
    tmp_path, *, identity_verified=True, retained=True,
    workspace_validator=None, deadline=False,
):
    solver = tmp_path / "solver"
    solver.mkdir()
    (solver / "run_simulation_260706.py").write_text(
        "# immutable runner fixture\n", encoding="utf-8"
    )
    library = tmp_path / "library"
    if deadline:
        library.mkdir()
    process = FakeProcess()
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    def identity(root):
        return {
            "available": True,
            "verified": identity_verified,
            "root": str(root),
            "revision": SOLVER_REVISION,
            "runner_sha256": RUNNER_SHA256,
            "dirty": False,
            "error": None if identity_verified else "identity mismatch",
        }

    launcher = LocalAedtGuiLauncher(
        tmp_path / "monitor-repo",
        solver_root=solver,
        expected_solver_revision=SOLVER_REVISION,
        expected_solver_branch=SOLVER_BRANCH if deadline else None,
        expected_runner_sha256=RUNNER_SHA256,
        expected_solver_source_runner_sha256=(
            SOURCE_RUNNER_SHA256 if deadline else None
        ),
        expected_solver_thermal_module_sha256=(
            THERMAL_MODULE_SHA256 if deadline else None
        ),
        library_root=library if deadline else None,
        expected_library_revision=(
            LIBRARY_REVISION if deadline else None
        ),
        runtime_root=tmp_path / "runtime",
        python_executable="python.exe",
        popen_factory=popen,
        process_probe=lambda pid: False,
        aedt_process_probe=lambda identity: {
            "verified": retained,
            "alive": retained,
            "pid": identity.get("pid"),
            "create_time_match": retained,
            "error": None if retained else "aedt_process_identity_mismatch",
        },
        parameter_validator=lambda parameters: None,
        clock=lambda: datetime(2026, 7, 18, 5, 0, tzinfo=timezone.utc),
        solver_identity_provider=identity,
        workspace_validator=workspace_validator,
    )
    return launcher, process, calls


def test_launch_uses_immutable_solver_local_workspace_and_result_contract(
    tmp_path,
):
    launcher, process, calls = _launcher(tmp_path)
    launched = launcher.launch(_candidate(), "full", "solve")

    command, kwargs = calls[0]
    workspace = Path(launched["workspace"])
    result_path = Path(launched["result_path"])
    assert kwargs["cwd"] == str(workspace)
    assert workspace.is_relative_to(tmp_path / "runtime" / "workspaces")
    assert command[command.index("--result-json") + 1] == str(result_path)
    assert kwargs["env"]["PYTHONPATH"] == str(launcher.solver_root)
    assert kwargs["env"]["MFT_GUI_LAUNCH_ID"] == launched["launch_id"]
    assert kwargs["env"]["MFT_GUI_CANDIDATE_ID"] == "candidate-1"
    assert launched["solver_revision"] == SOLVER_REVISION
    assert launched["runner_sha256"] == RUNNER_SHA256

    result_path.write_text(json.dumps({
        "schema": GUI_RESULT_SCHEMA,
        "status": "completed",
        "created_at_utc": "2026-07-18T05:10:00+00:00",
        "launch_id": launched["launch_id"],
        "candidate_id": "candidate-1",
        "held_open": True,
        "detach_confirmed": True,
        "detach_error": None,
        "solver_revision": SOLVER_REVISION,
        "solver_dirty": 0,
        "project_name": "simulation1",
        "project_path": str(workspace / "simulation1.aedt"),
        "aedt_process": {
            "pid": 54321,
            "create_time": 1_784_351_400.125,
            "name": "ansysedt.exe",
            "executable": "C:\\Program Files\\AnsysEM\\ansysedt.exe",
            "grpc_port": 50052,
            "alive": True,
            "error": None,
        },
        "parameters": {"N1_main": 6},
        "result": {
            "Llt_phys": 27.5,
            "P_core_total": 1234.0,
            "P_winding_total": 2345.0,
            "B_max_core": 1.1,
        },
        "inspection_reports": ["MFT_Matrix_Inspection", "MFT_Loss_Inspection"],
        "error": None,
    }), encoding="utf-8")
    process.returncode = 0

    snapshot = launcher.snapshot(candidate_id="candidate-1")
    status = snapshot["launches"][0]
    assert status["state"] == "completed_held"
    assert status["held_open"] is True
    assert status["retained_aedt"] is True
    assert status["occupied"] is True
    assert status["exit_code"] == 0
    assert status["finished_at"] == "2026-07-18T05:10:00+00:00"
    assert status["fea_validation"]["success"] is True
    assert status["fea_validation"]["result"]["Llt_phys"] == 27.5
    assert status["fea_validation"]["inspection_reports"] == [
        "MFT_Matrix_Inspection", "MFT_Loss_Inspection"
    ]
    compact = launcher.validation_index()["candidate-1"]
    assert compact["fea_validation"]["artifact"]["sha256"]
    assert "stdout_tail" not in compact


def test_completed_artifact_does_not_claim_hold_after_pid_identity_loss(
    tmp_path,
):
    launcher, process, _ = _launcher(tmp_path, retained=False)
    launched = launcher.launch(_candidate(), "full", "solve")
    Path(launched["result_path"]).write_text(json.dumps({
        "schema": GUI_RESULT_SCHEMA,
        "status": "completed",
        "created_at_utc": "2026-07-18T05:10:00+00:00",
        "launch_id": launched["launch_id"],
        "candidate_id": "candidate-1",
        "held_open": True,
        "detach_confirmed": True,
        "detach_error": None,
        "solver_revision": SOLVER_REVISION,
        "solver_dirty": 0,
        "project_name": "simulation1",
        "project_path": str(Path(launched["workspace"]) / "simulation1.aedt"),
        "aedt_process": {
            "pid": 54321,
            "create_time": 1_784_351_400.125,
            "name": "ansysedt.exe",
            "executable": "C:\\Program Files\\AnsysEM\\ansysedt.exe",
            "grpc_port": 50052,
            "alive": True,
            "error": None,
        },
        "parameters": {},
        "result": {"Llt_phys": 27.5},
        "inspection_reports": [],
        "error": None,
    }), encoding="utf-8")
    process.returncode = 0

    snapshot = launcher.snapshot(candidate_id="candidate-1")
    status = snapshot["launches"][0]
    assert status["state"] == "completed_detached"
    assert status["held_open_attested"] is True
    assert status["retained_aedt"] is False
    assert status["occupied"] is False
    assert snapshot["retained_aedt_count"] == 0
    assert snapshot["occupied_count"] == 0
    assert "retention could not be verified" in status["error_summary"]


def test_held_result_requires_confirmed_clean_desktop_detach(tmp_path):
    launcher, process, _ = _launcher(tmp_path)
    launched = launcher.launch(_candidate(), "symmetry", "solve")
    Path(launched["result_path"]).write_text(json.dumps({
        "schema": GUI_RESULT_SCHEMA,
        "status": "completed",
        "created_at_utc": "2026-07-18T05:10:00+00:00",
        "launch_id": launched["launch_id"],
        "candidate_id": "candidate-1",
        "held_open": False,
        "detach_confirmed": False,
        "detach_error": "release_desktop_returned:False",
        "solver_revision": SOLVER_REVISION,
        "solver_dirty": 0,
        "project_name": "simulation1",
        "project_path": str(Path(launched["workspace"]) / "simulation1.aedt"),
        "aedt_process": {
            "pid": 54321,
            "create_time": 1_784_351_400.125,
            "name": "ansysedt.exe",
            "executable": "C:\\Program Files\\AnsysEM\\ansysedt.exe",
            "grpc_port": 50052,
            "alive": True,
            "error": None,
        },
        "parameters": {},
        "result": {"Llt_phys": 27.5},
        "inspection_reports": [],
        "error": None,
    }), encoding="utf-8")
    process.returncode = 1

    status = launcher.snapshot(candidate_id="candidate-1")["launches"][0]
    assert status["fea_validation"] is None
    assert "requires held_open evidence" in status["result_artifact_error"]
    assert status["retained_aedt"] is False
    assert status["occupied"] is False


def test_gui_result_tamper_and_legacy_access_error_are_fail_closed(tmp_path):
    launcher, process, _ = _launcher(tmp_path)
    launched = launcher.launch(_candidate(), "symmetry", "solve")
    Path(launched["result_path"]).write_text(json.dumps({
        "schema": GUI_RESULT_SCHEMA,
        "status": "completed",
        "launch_id": "wrong-launch",
        "candidate_id": "candidate-1",
        "held_open": True,
        "solver_revision": SOLVER_REVISION,
        "solver_dirty": 0,
    }), encoding="utf-8")
    Path(launched["stderr_path"]).write_text(
        "[WinError 5] access is denied while reading simulation1.pyaedt",
        encoding="utf-8",
    )
    process.returncode = 1

    status = launcher.snapshot(candidate_id="candidate-1")["launches"][0]
    assert status["fea_validation"] is None
    assert "identity is invalid" in status["result_artifact_error"]
    assert status["diagnosis"]["code"] == "workspace_access_denied"
    assert status["error_summary"]
    assert status["exit_code"] == 1


def test_launch_rejects_noncanonical_solver_identity(tmp_path):
    launcher, _, calls = _launcher(tmp_path, identity_verified=False)
    with pytest.raises(LocalAedtGuiError, match="immutable local AEDT solver"):
        launcher.launch(_candidate(), "full", "solve")
    assert calls == []


def test_launch_rejects_nonlocal_workspace_before_process_start(tmp_path):
    def reject_workspace(path):
        raise LocalAedtGuiError("fixed local drive required")

    launcher, _, calls = _launcher(
        tmp_path, workspace_validator=reject_workspace
    )
    with pytest.raises(LocalAedtGuiError, match="fixed local drive"):
        launcher.launch(_candidate(), "full", "solve")
    assert calls == []


def test_deadline_launch_preserves_exact_cooling_and_thermal_contract(
    tmp_path,
):
    launcher, _, calls = _launcher(tmp_path, deadline=True)

    launched = launcher.launch(_deadline_candidate(), "full", "solve")

    parameters = json.loads(
        Path(launched["parameter_path"]).read_text(encoding="utf-8")
    )
    _, kwargs = calls[0]
    assert parameters["fan_config"] == "dual"
    assert parameters["fan_velocity"] == 3.0
    assert parameters["plate_temp"] == 50.0
    assert parameters["air_temp"] == 50.0
    assert parameters["thermal_on"] == 1
    assert parameters["keep_project"] == 1
    assert parameters["P_target"] == 1_000_000.0
    assert parameters["full_model"] == 1
    assert parameters["loss_sym_on"] == 0
    assert parameters["thermal_symmetry"] == "full"
    assert parameters["n_explicit_turns"] == 2
    assert parameters["matrix_percent_error"] == 0.5
    assert parameters["matrix_max_passes"] == 24
    assert parameters["matrix_skin_mesh"] == 1
    assert parameters["percent_error"] == 0.5
    assert parameters["max_passes"] == 18
    assert kwargs["env"]["MFT_PYAEDT_LIBRARY_ROOT"].endswith("library")
    assert launched["required_result_echo"]["fan_velocity"] == 3.0
    assert launched["required_result_echo"]["P_target"] == 1_000_000.0
    assert launched["required_result_echo"]["thermal_symmetry"] == "full"
    assert (
        launched["required_result_echo"]["thermal_pad_conductivity_W_mK"]
        == 3.0
    )


def test_deadline_symmetry_launch_uses_exact_standard_fea_profile(tmp_path):
    launcher, _, _ = _launcher(tmp_path, deadline=True)

    launched = launcher.launch(
        _deadline_candidate(), "symmetry", "solve"
    )

    parameters = json.loads(
        Path(launched["parameter_path"]).read_text(encoding="utf-8")
    )
    assert parameters["P_target"] == 1_000_000.0
    assert parameters["full_model"] == 0
    assert parameters["loss_sym_on"] == 1
    assert parameters["thermal_symmetry"] == "eighth"
    assert parameters["n_explicit_turns"] == 0
    assert parameters["matrix_percent_error"] == 1.5
    assert parameters["matrix_max_passes"] == 20
    assert parameters["matrix_min_converged"] == 1
    assert parameters["matrix_skin_mesh"] == 0
    assert parameters["percent_error"] == 1.5
    assert parameters["max_passes"] == 10
    assert parameters["min_converged"] == 2


@pytest.mark.parametrize(
    ("resonance_Hz", "direction", "operator"),
    (
        (14_999.0, "minimum", ">="),
        (16_000.0, "maximum", "<"),
    ),
)
def test_deadline_launch_rejects_wrong_resonance_contract_before_start(
    tmp_path, resonance_Hz, direction, operator
):
    launcher, _, calls = _launcher(tmp_path, deadline=True)
    candidate = _deadline_candidate()
    candidate["pred_f_res_min_screen_Hz"] = resonance_Hz
    candidate["resonance_margin_Hz"] = resonance_Hz - 15_000.0
    candidate["constraints"]["resonance"].update({
        "value": resonance_Hz,
        "direction": direction,
        "operator": operator,
    })

    with pytest.raises(
        CandidateLaunchError, match="must be >= 15 kHz"
    ):
        launcher.launch(candidate, "full", "solve")

    assert calls == []


def test_deadline_completed_result_requires_exact_cooling_echo(tmp_path):
    launcher, process, _ = _launcher(tmp_path, deadline=True)
    launched = launcher.launch(_deadline_candidate(), "symmetry", "solve")
    workspace = Path(launched["workspace"])
    parameters = json.loads(
        Path(launched["parameter_path"]).read_text(encoding="utf-8")
    )
    result = {
        **parameters,
        "thermal_pad_conductivity_W_mK": 3.0,
        "thermal_pad_material_policy": "deadline-k3",
    }
    result["fan_velocity"] = 1.5
    Path(launched["result_path"]).write_text(json.dumps({
        "schema": GUI_RESULT_SCHEMA,
        "status": "completed",
        "created_at_utc": "2026-07-18T05:10:00+00:00",
        "launch_id": launched["launch_id"],
        "candidate_id": "candidate-1",
        "held_open": True,
        "detach_confirmed": True,
        "detach_error": None,
        "solver_revision": SOLVER_REVISION,
        "solver_dirty": 0,
        "project_name": "simulation1",
        "project_path": str(workspace / "simulation1.aedt"),
        "aedt_process": {
            "pid": 54321,
            "create_time": 1_784_351_400.125,
            "name": "ansysedt.exe",
            "executable": "C:\\Program Files\\AnsysEM\\ansysedt.exe",
            "grpc_port": 50052,
            "alive": True,
            "error": None,
        },
        "parameters": parameters,
        "result": result,
        "inspection_reports": [],
        "error": None,
    }), encoding="utf-8")
    process.returncode = 0

    status = launcher.snapshot(candidate_id="candidate-1")["launches"][0]

    assert status["fea_validation"] is None
    assert "result echo mismatch for fan_velocity" in (
        status["result_artifact_error"]
    )


def test_router_uses_isolated_deadline_launcher():
    class Stub:
        def __init__(self, label):
            self.label = label

        def launch(self, candidate, mode, action):
            return {"label": self.label}

        def snapshot(self, candidate_id=None):
            return {
                "available": True,
                "solver_identity": {"label": self.label},
                "max_active": 1,
                "active_count": 0,
                "retained_aedt_count": 0,
                "occupied_count": 0,
                "launches": [],
            }

        def validation_index(self, limit=1_000):
            return {}

    router = RoutedLocalAedtGuiLauncher(Stub("default"), Stub("deadline"))

    assert router.launch(_candidate(), "symmetry")["label"] == "default"
    assert (
        router.launch(_deadline_candidate(), "symmetry")["label"]
        == "deadline"
    )
