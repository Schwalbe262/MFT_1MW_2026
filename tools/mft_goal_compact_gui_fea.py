"""Run the selected compact candidate in a visible 1/8 AEDT GUI.

This launcher is intentionally a local, inspectable diagnostic companion to
the authenticated Slurm solve.  It preserves the selected geometry and fixed
cooling contract, disables the separately-running capacitance stage, and
leaves the solved AEDT project open for inspection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


SCHEMA = "mft-compact-visible-gui-fea-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CANDIDATE_SHA256 = (
    "50f849eb3c0054d21bc7af8259fd3daff2223d37d2458a6ba4f0301d58e689b9"
)
EXPECTED_PARAMETER_FILE_SHA256 = (
    "96d1d83215ffe0af297f5543899e1060ca4a27c5b999f5f753873c22395c0074"
)
REQUIRED_FIXED_VALUES = {
    "N1_main": 6,
    "N1_side": 0,
    "N2_main": 32,
    "N2_side": 28,
    "core_center_gap_mm": 0.402982779173143,
    "core_equal_three_leg_air_gap": 1,
    "core_plate_t": 20.0,
    "wcp_t": 20.0,
    "core_plate_pad_t": 2.0,
    "wcp_pad_t": 2.0,
    "fan_velocity": 1.5,
    "full_model": 0,
    "round_corner": 0,
    "thermal_symmetry": "eighth",
}
GUI_OVERRIDES = {
    "matrix_on": 1,
    "cap_on": 0,
    "loss_on": 1,
    "thermal_on": 1,
    "loss_sym_on": 1,
    "full_model": 0,
    "round_corner": 0,
    "thermal_symmetry": "eighth",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_and_seal(source: Path) -> dict:
    actual_sha256 = _sha256(source)
    if actual_sha256 != EXPECTED_PARAMETER_FILE_SHA256:
        raise RuntimeError(
            "selected compact parameter file SHA-256 mismatch: "
            f"expected={EXPECTED_PARAMETER_FILE_SHA256}, actual={actual_sha256}"
        )
    parameters = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(parameters, dict):
        raise TypeError("selected compact parameter payload is not an object")
    mismatches = {
        key: {"expected": expected, "actual": parameters.get(key)}
        for key, expected in REQUIRED_FIXED_VALUES.items()
        if parameters.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "selected compact parameter contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    sealed = dict(parameters)
    sealed.update(GUI_OVERRIDES)
    return sealed


def _install_local_gui_core_override(runner, cores: int) -> None:
    """Keep the production policy untouched while sizing this GUI diagnostic."""
    original_init = runner.Simulation.__init__

    def diagnostic_init(self, desktop=None):
        original_init(self, desktop=desktop)
        self.NUM_CORE = int(cores)
        self.NUM_TASK = 1
        policy = dict(self.solver_core_policy)
        policy.update(
            {
                "schema": "mft-local-visible-gui-core-policy-v1",
                "contract_version": "local-visible-gui-diagnostic-v1",
                "opt_in": False,
                "requested_num_cores": int(cores),
                "effective_num_cores": int(cores),
                "num_tasks": 1,
                "diagnostic_only": True,
                "production_truth_eligible": False,
            }
        )
        self.solver_core_policy = policy
        runner._emit_solver_core_evidence(
            "LOCAL_GUI_SOLVER_CORE_CONTRACT_JSON", policy
        )

    runner.Simulation.__init__ = diagnostic_init


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument(
        "--candidate-sha256",
        default=EXPECTED_CANDIDATE_SHA256,
    )
    args = parser.parse_args(argv)
    if args.cores != 16:
        parser.error("this visible GUI diagnostic is sealed to exactly 16 cores")
    if args.candidate_sha256 != EXPECTED_CANDIDATE_SHA256:
        parser.error("candidate SHA-256 does not identify the selected compact case")
    return args


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    sealed_path = output_root / "compact_exact_gap_gui_params.json"
    source = args.params.resolve()
    status = {
        "schema": SCHEMA,
        "state": "preparing",
        "updated_at_utc": _utc_now(),
        "controller_pid": os.getpid(),
        "candidate_sha256": args.candidate_sha256,
        "source_parameter_path": str(source),
        "source_parameter_sha256": _sha256(source),
        "output_root": str(output_root),
        "requested_cores": int(args.cores),
        "model": "symmetric_eighth_nonrounded",
        "solver_sequence": ["matrix", "loss", "thermal"],
        "capacitance_stage": "disabled_parallel_slurm_lane_is_authoritative",
        "diagnostic_only": True,
        "production_truth_eligible": False,
    }
    _write_json(status_path, status)

    previous_cwd = Path.cwd()
    previous_simulation_id = os.environ.get("SIMULATION_ID")
    try:
        parameters = _load_and_seal(source)
        _write_json(sealed_path, parameters)
        os.chdir(output_root)
        os.environ["SIMULATION_ID"] = "compact50f849-exactgap-gui"
        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))

        import run_simulation_260706 as runner
        from module.thermal_260706 import (
            SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV,
            SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN,
        )

        runner.GUI = False
        os.environ[SYMMETRY_THERMAL_DIRECT_ANALYZE_ENV] = (
            SYMMETRY_THERMAL_DIRECT_ANALYZE_TOKEN
        )
        _install_local_gui_core_override(runner, int(args.cores))
        status.update(
            {
                "state": "running_visible_aedt",
                "updated_at_utc": _utc_now(),
                "sealed_parameter_path": str(sealed_path),
                "sealed_parameter_sha256": _sha256(sealed_path),
                "thermal_mesh_policy": "standard_eighth_direct_analyze_v1",
            }
        )
        _write_json(status_path, status)

        completed = bool(
            runner.run_one_loop(
                param=parameters,
                model_only=False,
                hold=True,
            )
        )
        status.update(
            {
                "state": (
                    "solved_gui_held"
                    if completed
                    else "completed_with_invalid_result_gui_held"
                ),
                "updated_at_utc": _utc_now(),
                "completed": completed,
                "project_glob": str(
                    output_root
                    / "simulation"
                    / "simulation_job_compact50f849-exactgap-gui*"
                ),
            }
        )
        _write_json(status_path, status)
        return 0 if completed else 2
    except Exception as error:
        status.update(
            {
                "state": "failed_gui_preserved_when_available",
                "updated_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(status_path, status)
        return 1
    finally:
        os.chdir(previous_cwd)
        if previous_simulation_id is None:
            os.environ.pop("SIMULATION_ID", None)
        else:
            os.environ["SIMULATION_ID"] = previous_simulation_id


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
