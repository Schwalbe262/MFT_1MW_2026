"""Launch an approved compact 6/60 candidate in two isolated visible AEDT modes.

This local-only entry point is deliberately parameter-sha-bound.  It does not
select a candidate and it never reuses an existing AEDT project:

* ``symmetric-validation`` builds and solves the non-rounded eighth-symmetry
  Matrix + turn-graded Rx capacitance + Loss + Thermal chain.
* ``full-curved-model`` builds the same source geometry as a full rounded
  drawing model and performs zero analysis calls.

The scheduler and its resource policy are not imported or modified here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mft-compact-selected-visible-gui-launch-v2"
SYMMETRIC_MODE = "symmetric-validation"
FULL_CURVED_MODE = "full-curved-model"
MODES = (SYMMETRIC_MODE, FULL_CURVED_MODE)
FIXED_TIM_CONDUCTIVITY_W_MK = 0.2
SOURCE_WCP_LENGTH_MM = 324.6
CURVED_WCP_LENGTH_MM = 319.0
FIXED_SOURCE_CONTRACT = {
    "N1_main": 6,
    "N1_side": 0,
    "N2_main": 35,
    "N2_side": 25,
    "nwh1": 390.0,
    "nwh2": 390.0,
    "cw1": 5.0,
    "cw2": 0.3,
    "gap1": 1.6,
    "gap2": 0.85,
    "core_equal_three_leg_air_gap": 1,
    "core_plate_on": 1,
    "core_plate_t": 20.0,
    "wcp_on": 1,
    "wcp_t": 20.0,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
    "wcp_len_x": SOURCE_WCP_LENGTH_MM,
    "fan_velocity": 1.5,
    "k_ins": FIXED_TIM_CONDUCTIVITY_W_MK,
}
MIN_VERTICAL_CLEARANCE_MM = 40.0
MAX_VERTICAL_CLEARANCE_MM = 50.0
MIN_COMPACT_H1_MM = 470.0
MAX_COMPACT_H1_MM = 490.0
MAX_COMPACT_TOTAL_HEIGHT_MM = 670.0
MIN_EQUAL_LEG_GAP_MM = 0.35
MAX_EQUAL_LEG_GAP_MM = 0.55
SYMMETRIC_OVERRIDES = {
    "matrix_on": 1,
    "matrix_max_passes": 7,
    "matrix_min_converged": 1,
    "matrix_percent_error": 1.5,
    # The legacy two-equipotential capacitance solve is not part of the final
    # truth chain.  Keep it disabled so the Rx turn-graded stage below is the
    # only electrostatic solve.
    "cap_on": 0,
    "cap_max_passes": 7,
    "cap_percent_error": 1.5,
    "cap_turn_graded_active_winding": "Rx",
    "cap_turn_graded_voltage_policy": "turn_midpoint",
    "cap_turn_graded_section_order": "main,side",
    "cap_turn_graded_reverse_sections": "none",
    "cap_turn_graded_reverse_terminal_polarity": 0,
    "cap_turn_graded_side_polarity": 1,
    "cap_turn_graded_side2_polarity": 1,
    "loss_on": 1,
    "loss_sym_on": 1,
    "max_passes": 7,
    "min_converged": 1,
    "percent_error": 1.5,
    "thermal_on": 1,
    "thermal_symmetry": "eighth",
    "full_model": 0,
    "round_corner": 0,
    "keep_project": 1,
}
FULL_CURVED_OVERRIDES = {
    "matrix_on": 1,
    "cap_on": 0,
    "cap_turn_graded_active_winding": "off",
    "loss_on": 0,
    "thermal_on": 0,
    "thermal_symmetry": "full",
    "full_model": 1,
    "round_corner": 1,
    "corner_radius": 10.0,
    "corner_segments": 4,
    # R10 shortens the Tx straight reference span from 418.8 to 398.8 mm.
    # Keep the plate inside the audited 80% straight-span construction limit.
    "wcp_len_x": CURVED_WCP_LENGTH_MM,
    "keep_project": 1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_parameter_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("compact parameter payload must be a JSON object")
    parameters = payload.get("parameters", payload)
    if not isinstance(parameters, dict):
        raise TypeError("compact parameters must be a JSON object")
    return dict(parameters)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_source_contract(parameters: dict[str, Any]) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    for name, expected in FIXED_SOURCE_CONTRACT.items():
        actual = parameters.get(name)
        if isinstance(expected, float):
            try:
                actual = _finite_number(actual, name)
            except ValueError:
                pass
        if actual != expected:
            mismatches[name] = {
                "expected": expected,
                "actual": parameters.get(name),
            }
    if mismatches:
        raise ValueError(
            "compact fixed cooling/equal-gap source contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    h1 = _finite_number(parameters.get("h1"), "h1")
    nwh1 = _finite_number(parameters.get("nwh1"), "nwh1")
    nwh2 = _finite_number(parameters.get("nwh2"), "nwh2")
    l1 = _finite_number(parameters.get("l1"), "l1")
    equal_gap = _finite_number(
        parameters.get("core_center_gap_mm"),
        "core_center_gap_mm",
    )
    compact_mismatches: dict[str, Any] = {}
    if not MIN_COMPACT_H1_MM <= h1 <= MAX_COMPACT_H1_MM:
        compact_mismatches["h1_mm"] = {
            "allowed": [MIN_COMPACT_H1_MM, MAX_COMPACT_H1_MM],
            "actual": h1,
        }
    if nwh1 != nwh2:
        compact_mismatches["equal_winding_height"] = {
            "nwh1_mm": nwh1,
            "nwh2_mm": nwh2,
        }
    vertical_clearance = (h1 - max(nwh1, nwh2)) / 2.0
    if not (
        MIN_VERTICAL_CLEARANCE_MM
        <= vertical_clearance
        <= MAX_VERTICAL_CLEARANCE_MM
    ):
        compact_mismatches["vertical_clearance_mm"] = {
            "allowed": [
                MIN_VERTICAL_CLEARANCE_MM,
                MAX_VERTICAL_CLEARANCE_MM,
            ],
            "actual": vertical_clearance,
        }
    total_height = h1 + 2.0 * l1
    if total_height > MAX_COMPACT_TOTAL_HEIGHT_MM:
        compact_mismatches["total_height_mm"] = {
            "maximum": MAX_COMPACT_TOTAL_HEIGHT_MM,
            "actual": total_height,
        }
    if not MIN_EQUAL_LEG_GAP_MM <= equal_gap <= MAX_EQUAL_LEG_GAP_MM:
        compact_mismatches["equal_three_leg_gap_mm"] = {
            "allowed": [MIN_EQUAL_LEG_GAP_MM, MAX_EQUAL_LEG_GAP_MM],
            "actual": equal_gap,
        }
    if compact_mismatches:
        raise ValueError(
            "compact selected geometry contract mismatch: "
            + json.dumps(compact_mismatches, sort_keys=True)
        )
    for name in ("N1_main", "N2_main"):
        try:
            value = int(parameters[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a positive integer") from error
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _seal_parameters(
    parameters: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported compact GUI mode: {mode!r}")
    _validate_source_contract(parameters)
    sealed = dict(parameters)
    sealed.update(
        SYMMETRIC_OVERRIDES
        if mode == SYMMETRIC_MODE
        else FULL_CURVED_OVERRIDES
    )
    return sealed


def _solver_contract(mode: str) -> dict[str, Any]:
    common = {
        "cores": 16,
        "maxwell_tasks": 1,
        "icepak_tasks": 16,
        "icepak_expected_fluent_command": "-t16",
        "icepak_runtime_attestation": "external_os_watch_required",
        "tasks": 1,
        "gpus": 0,
        "use_auto_settings": False,
        "new_visible_desktop": True,
        "reuse_existing_project": False,
    }
    if mode == SYMMETRIC_MODE:
        common.update({
            "maxwell_max_passes": 7,
            "minimum_converged_passes": 1,
            "percent_error": 1.5,
            "analysis_sequence": [
                "matrix",
                "cap_turn_graded_rx",
                "loss",
                "thermal",
            ],
            "expected_maxwell_analysis_calls": 3,
            "expected_thermal_analysis_calls": 1,
        })
    else:
        common.update({
            "analysis_sequence": [],
            "expected_maxwell_analysis_calls": 0,
            "expected_thermal_analysis_calls": 0,
        })
    return common


def _install_local_manual_hpc(runner: Any, cores: int) -> None:
    """Install a process-local GUI policy without changing scheduler policy."""
    original_init = runner.Simulation.__init__

    def local_init(self, desktop=None):
        original_init(self, desktop=desktop)
        self.NUM_CORE = int(cores)
        self.NUM_TASK = 1
        self.local_gui_manual_maxwell_hpc = True
        # The local Icepak call is independently proven as cores=16/tasks=16.
        # Keep in-process attestation disabled because AEDT's blocking COM call
        # can hold the GIL long enough for a Python watcher thread to miss the
        # short-lived Fluent command.  An external OS watcher performs the
        # runtime -t16 readback without blocking the visible validation launch.
        self.local_gui_strict_thermal_hpc = False
        policy = dict(self.solver_core_policy)
        policy.update({
            "schema": "mft-compact-local-visible-core-policy-v2",
            "contract_version": "compact-local-visible-manual-hpc-v2",
            "opt_in": False,
            "requested_num_cores": int(cores),
            "effective_num_cores": int(cores),
            "num_tasks": 1,
            "num_gpus": 0,
            "use_auto_settings": False,
            "diagnostic_only": True,
            "thermal_runtime_attestation": "external_os_watch_required",
            "scheduler_policy_mutated": False,
        })
        self.solver_core_policy = policy
        runner._emit_solver_core_evidence(
            "COMPACT_LOCAL_GUI_SOLVER_CORE_CONTRACT_JSON", policy
        )

    runner.Simulation.__init__ = local_init


def _install_zero_analysis_guard(runner: Any) -> None:
    def forbidden_analyze(*_args, **_kwargs):
        raise RuntimeError(
            "full-curved-model contract forbids every analysis call"
        )

    runner.Simulation.analyze_and_extract = forbidden_analyze


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--params-sha256",
        required=True,
        help="Exact SHA-256 supplied with the approved compact payload.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--simulation-id", required=True)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and print the launch contract without creating files/AEDT.",
    )
    args = parser.parse_args(argv)
    if args.cores != 16:
        parser.error("compact visible GUI workflows are fixed to 16 CPU cores")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.params_sha256):
        parser.error("--params-sha256 must be exactly 64 hexadecimal characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.simulation_id):
        parser.error("--simulation-id contains unsupported characters")
    return args


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source = args.params.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"approved compact parameter file is missing: {source}"
        )
    actual_sha = _sha256(source)
    expected_sha = str(args.params_sha256).lower()
    if actual_sha != expected_sha:
        raise RuntimeError(
            "approved compact parameter SHA-256 mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(
            "compact GUI output must be a fresh isolated path: "
            f"{output_root}"
        )
    parameters = _seal_parameters(
        _load_parameter_payload(source),
        args.mode,
    )
    evidence = {
        "schema": "mft-compact-selected-visible-gui-preflight-v2",
        "passed": True,
        "mode": args.mode,
        "source_parameter_path": str(source),
        "source_parameter_sha256": actual_sha,
        "output_root": str(output_root),
        "simulation_id": args.simulation_id,
        "project_name": f"simulation_job_{args.simulation_id}",
        "model": (
            "nonrounded_symmetric_eighth"
            if args.mode == SYMMETRIC_MODE
            else "full_curved_rounded_R10_S4"
        ),
        "fixed_cooling": {
            "core_plate_t_mm": 20.0,
            "wcp_t_mm": 20.0,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "fan_velocity_m_per_s": 1.5,
            "thermal_pad_conductivity_W_mK": (
                FIXED_TIM_CONDUCTIVITY_W_MK
            ),
        },
        "compact_geometry": {
            "h1_mm": float(parameters["h1"]),
            "equal_winding_height_mm": float(parameters["nwh1"]),
            "top_clearance_mm": (
                float(parameters["h1"]) - float(parameters["nwh1"])
            )
            / 2.0,
            "bottom_clearance_mm": (
                float(parameters["h1"]) - float(parameters["nwh1"])
            )
            / 2.0,
            "total_height_mm": (
                float(parameters["h1"]) + 2.0 * float(parameters["l1"])
            ),
            "equal_three_leg_gap_mm": float(
                parameters["core_center_gap_mm"]
            ),
        },
        "solver_contract": _solver_contract(args.mode),
        "scheduler_project_touched": False,
    }
    if args.mode == FULL_CURVED_MODE:
        evidence["curved_geometry_adaptation"] = {
            "reason": "R10_shortens_Tx_straight_span",
            "source_wcp_len_x_mm": SOURCE_WCP_LENGTH_MM,
            "curved_wcp_len_x_mm": CURVED_WCP_LENGTH_MM,
            "curved_reference_span_mm": 398.8,
            "curved_wcp_length_percent": (
                100.0 * CURVED_WCP_LENGTH_MM / 398.8
            ),
            "plate_thickness_unchanged_mm": 20.0,
        }
    return parameters, evidence


def _run(args: argparse.Namespace) -> int:
    parameters, evidence = _preflight(args)
    if args.preflight_only:
        print(
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    status_path = output_root / "status.json"
    parameter_path = output_root / "approved_effective_params.json"
    status = {
        **evidence,
        "schema": SCHEMA,
        "state": "preparing_new_visible_aedt",
        "controller_pid": os.getpid(),
        "updated_at_utc": _utc_now(),
    }
    _atomic_json(status_path, status)
    _atomic_json(parameter_path, parameters)

    previous_cwd = Path.cwd()
    previous_simulation_id = os.environ.get("SIMULATION_ID")
    try:
        os.chdir(output_root)
        os.environ["SIMULATION_ID"] = args.simulation_id
        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        import run_simulation_260706 as runner
        from module.thermal_260706 import (
            SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV,
            SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN,
        )

        runner.GUI = False
        _install_local_manual_hpc(runner, 16)
        model_only = args.mode == FULL_CURVED_MODE
        if model_only:
            _install_zero_analysis_guard(runner)
        else:
            os.environ[SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV] = (
                SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN
            )

        status.update({
            "state": (
                "building_full_curved_model_zero_analysis"
                if model_only
                else "running_symmetric_validation"
            ),
            "updated_at_utc": _utc_now(),
            "effective_parameter_path": str(parameter_path),
            "effective_parameter_sha256": _sha256(parameter_path),
        })
        _atomic_json(status_path, status)
        completed = bool(
            runner.run_one_loop(
                param=parameters,
                model_only=model_only,
                hold=True,
            )
        )
        status.update({
            "state": (
                "full_curved_model_gui_held_analysis_zero"
                if model_only and completed
                else (
                    "symmetric_validation_gui_held"
                    if completed
                    else "completed_invalid_gui_held"
                )
            ),
            "completed": completed,
            "updated_at_utc": _utc_now(),
            "project_glob": str(
                output_root
                / "simulation"
                / f"simulation_job_{args.simulation_id}*"
            ),
        })
        _atomic_json(status_path, status)
        return 0 if completed else 2
    except Exception as error:
        status.update({
            "state": "failed_gui_preserved_when_available",
            "updated_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        _atomic_json(status_path, status)
        return 1
    finally:
        os.chdir(previous_cwd)
        if previous_simulation_id is None:
            os.environ.pop("SIMULATION_ID", None)
        else:
            os.environ["SIMULATION_ID"] = previous_simulation_id


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
