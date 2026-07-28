from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import mft_goal_h390_visible_gui as launch


def _base() -> dict:
    return {
        "N1_main": 6,
        "N1_side": 0,
        "N2_main": 35,
        "N2_side": 25,
        "h1": 470.0,
        "nwh1": 390.0,
        "nwh2": 390.0,
        "cw1": 5.0,
        "cw2": 0.3,
        "gap1": 1.6,
        "gap2": 0.85,
        "core_center_gap_mm": 0.4448,
        "core_equal_three_leg_air_gap": 1,
        "core_plate_on": 1,
        "core_plate_t": 20.0,
        "wcp_on": 1,
        "wcp_t": 20.0,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "wcp_len_x": 324.6,
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "l1": 90.0,
        "corrected_leakage_marker": "preserve-me",
    }


def _args(tmp_path: Path, mode: str) -> SimpleNamespace:
    source = tmp_path / "approved.json"
    source.write_text(json.dumps(_base()), encoding="utf-8")
    return SimpleNamespace(
        mode=mode,
        params=source,
        params_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output_root=tmp_path / "fresh-output",
        simulation_id="h390_corrected_test",
        cores=16,
        preflight_only=True,
    )


def test_symmetric_contract_is_eighth_nonrounded_and_manual_hpc(
    tmp_path: Path,
) -> None:
    parameters, evidence = launch._preflight(
        _args(tmp_path, launch.SYMMETRIC_MODE)
    )

    assert parameters["h1"] == 470.0
    assert parameters["nwh1"] == parameters["nwh2"] == 390.0
    assert parameters["N2_main"] == 35
    assert parameters["N2_side"] == 25
    assert parameters["corrected_leakage_marker"] == "preserve-me"
    assert parameters["full_model"] == 0
    assert parameters["round_corner"] == 0
    assert parameters["thermal_symmetry"] == "eighth"
    assert parameters["matrix_on"] == 1
    assert parameters["cap_on"] == 0
    assert parameters["cap_turn_graded_active_winding"] == "Rx"
    assert parameters["loss_on"] == 1
    assert parameters["thermal_on"] == 1
    for prefix in ("matrix_", ""):
        assert parameters[f"{prefix}max_passes"] == 7
        assert parameters[f"{prefix}min_converged"] == 1
        assert parameters[f"{prefix}percent_error"] == 1.5
    assert parameters["cap_max_passes"] == 7
    assert parameters["cap_percent_error"] == 1.5
    assert evidence["solver_contract"]["cores"] == 16
    assert evidence["solver_contract"]["tasks"] == 1
    assert evidence["solver_contract"]["maxwell_tasks"] == 1
    assert evidence["solver_contract"]["icepak_tasks"] == 16
    assert evidence["solver_contract"]["gpus"] == 0
    assert evidence["solver_contract"]["use_auto_settings"] is False
    assert evidence["solver_contract"]["analysis_sequence"] == [
        "matrix",
        "cap_turn_graded_rx",
        "loss",
        "thermal",
    ]
    assert (
        evidence["solver_contract"]["expected_maxwell_analysis_calls"] == 3
    )
    assert (
        evidence["solver_contract"]["expected_thermal_analysis_calls"] == 1
    )
    assert evidence["scheduler_project_touched"] is False
    assert evidence["compact_geometry"] == {
        "h1_mm": 470.0,
        "equal_winding_height_mm": 390.0,
        "top_clearance_mm": 40.0,
        "bottom_clearance_mm": 40.0,
        "total_height_mm": 650.0,
        "equal_three_leg_gap_mm": 0.4448,
    }


def test_full_curved_contract_preserves_geometry_and_forbids_analysis(
    tmp_path: Path,
) -> None:
    parameters, evidence = launch._preflight(
        _args(tmp_path, launch.FULL_CURVED_MODE)
    )

    assert parameters["h1"] == 470.0
    assert parameters["nwh1"] == parameters["nwh2"] == 390.0
    assert parameters["corrected_leakage_marker"] == "preserve-me"
    assert parameters["full_model"] == 1
    assert parameters["round_corner"] == 1
    assert parameters["corner_radius"] == 10.0
    assert parameters["corner_segments"] == 4
    assert parameters["wcp_len_x"] == 319.0
    assert parameters["matrix_on"] == 1
    assert parameters["cap_on"] == 0
    assert parameters["cap_turn_graded_active_winding"] == "off"
    assert parameters["loss_on"] == 0
    assert parameters["thermal_on"] == 0
    assert evidence["solver_contract"]["analysis_sequence"] == []
    assert (
        evidence["solver_contract"]["expected_maxwell_analysis_calls"] == 0
    )
    assert (
        evidence["solver_contract"]["expected_thermal_analysis_calls"] == 0
    )
    assert evidence["solver_contract"]["new_visible_desktop"] is True
    assert evidence["solver_contract"]["reuse_existing_project"] is False
    assert evidence["curved_geometry_adaptation"] == {
        "reason": "R10_shortens_Tx_straight_span",
        "source_wcp_len_x_mm": 324.6,
        "curved_wcp_len_x_mm": 319.0,
        "curved_reference_span_mm": 398.8,
        "curved_wcp_length_percent": pytest.approx(79.98996990972919),
        "plate_thickness_unchanged_mm": 20.0,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("core_plate_t", 19.9),
        ("wcp_t", 19.9),
        ("core_plate_pad_t", 1.0),
        ("wcp_pad_t", 1.0),
        ("wcp_len_x", 319.0),
        ("fan_velocity", 2.0),
        ("k_ins", 0.3),
        ("h1", 469.9),
        ("nwh1", 389.9),
        ("nwh2", 389.9),
        ("N2_main", 36),
        ("N2_side", 24),
        ("core_center_gap_mm", 0.3),
        ("core_equal_three_leg_air_gap", 0),
    ],
)
def test_fixed_source_contract_fails_closed(name: str, value: float) -> None:
    parameters = _base()
    parameters[name] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        launch._seal_parameters(parameters, launch.SYMMETRIC_MODE)


def test_preflight_rejects_parameter_hash_drift(tmp_path: Path) -> None:
    args = _args(tmp_path, launch.SYMMETRIC_MODE)
    args.params_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        launch._preflight(args)


def test_preflight_rejects_nonfresh_output(tmp_path: Path) -> None:
    args = _args(tmp_path, launch.FULL_CURVED_MODE)
    args.output_root.mkdir()
    with pytest.raises(RuntimeError, match="fresh isolated path"):
        launch._preflight(args)


def test_zero_analysis_guard_fails_on_any_dispatch() -> None:
    class Simulation:
        @staticmethod
        def analyze_and_extract(*_args, **_kwargs):
            return True

    runner = SimpleNamespace(Simulation=Simulation)
    launch._install_zero_analysis_guard(runner)
    with pytest.raises(RuntimeError, match="forbids every analysis"):
        runner.Simulation.analyze_and_extract("matrix", lambda: None)


def test_full_curved_run_is_model_only_held_and_analysis_guarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, launch.FULL_CURVED_MODE)
    args.preflight_only = False
    calls: list[dict] = []
    core_evidence: list[tuple[str, dict]] = []

    class Simulation:
        def __init__(self, desktop=None):
            self.desktop = desktop
            self.solver_core_policy = {"backend": "standalone"}

        @staticmethod
        def analyze_and_extract(*_args, **_kwargs):
            raise AssertionError("analysis guard was not installed")

    runner = ModuleType("run_simulation_260706")
    runner.GUI = True
    runner.Simulation = Simulation

    def emit_solver_core_evidence(marker: str, evidence: dict) -> None:
        core_evidence.append((marker, dict(evidence)))

    def run_one_loop(*, param: dict, model_only: bool, hold: bool) -> bool:
        simulation = runner.Simulation()
        analysis_blocked = False
        try:
            simulation.analyze_and_extract("matrix", lambda: None)
        except RuntimeError as error:
            analysis_blocked = "forbids every analysis" in str(error)
        calls.append({
            "param": dict(param),
            "model_only": model_only,
            "hold": hold,
            "analysis_blocked": analysis_blocked,
            "num_core": simulation.NUM_CORE,
            "num_task": simulation.NUM_TASK,
        })
        return True

    runner._emit_solver_core_evidence = emit_solver_core_evidence
    runner.run_one_loop = run_one_loop

    thermal = ModuleType("module.thermal_260706")
    thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV = (
        "MFT_TEST_SYMMETRY_DIRECT_ANALYZE"
    )
    thermal.SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN = "unused-in-model-only"
    monkeypatch.setitem(sys.modules, "run_simulation_260706", runner)
    monkeypatch.setitem(sys.modules, "module.thermal_260706", thermal)
    monkeypatch.setenv("SIMULATION_ID", "preexisting-user-value")

    assert launch._run(args) == 0

    assert runner.GUI is False
    assert calls == [{
        "param": launch._seal_parameters(_base(), launch.FULL_CURVED_MODE),
        "model_only": True,
        "hold": True,
        "analysis_blocked": True,
        "num_core": 16,
        "num_task": 1,
    }]
    assert core_evidence[0][1]["scheduler_policy_mutated"] is False
    assert core_evidence[0][1]["effective_num_cores"] == 16
    assert os.environ["SIMULATION_ID"] == "preexisting-user-value"
    status = json.loads((args.output_root / "status.json").read_text())
    assert status["state"] == "full_curved_model_gui_held_analysis_zero"
    assert status["completed"] is True


def test_launcher_contains_no_direct_process_or_desktop_shutdown_calls() -> None:
    tree = ast.parse(Path(launch.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "kill",
        "terminate",
        "release_desktop",
        "close_desktop",
        "close_project",
    }
    invoked = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    }
    invoked.update({
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in forbidden
    })
    assert invoked == set()
