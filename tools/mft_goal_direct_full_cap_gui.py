"""Build and solve an exact full-model electrostatic capacitance design.

The magnetic design is used only as an unsolved geometry source.  The only
native solver dispatch is the ``maxwell_cap`` Electrostatic setup.  The AEDT
project remains open after success or failure so the GUI and result tree can be
inspected.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import traceback

import pandas as pd


EXACT_CAP_ONLY_OVERRIDES = {
    "full_model": 1,
    # The input schema requires matrix_on when cap_on is enabled.  No magnetic
    # setup is created or analyzed by this launcher.
    "matrix_on": 1,
    "cap_on": 1,
    "loss_on": 0,
    "thermal_on": 0,
    "keep_project": 1,
}
WINDOWS_AEDT_RESULT_PATH_LIMIT = 240


def _load_parameters(path: Path) -> dict:
    """Load a JSON object or the final non-empty JSONL record."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        if not records:
            raise RuntimeError(f"parameter JSONL is empty: {path}")
        record = records[-1]
    else:
        record = json.loads(text)
    parameters = record.get("parameters", record)
    if not isinstance(parameters, dict):
        raise TypeError("parameter payload must be a JSON object")
    return dict(parameters)


def _write_status(path: Path, **payload) -> None:
    """Atomically replace the live status document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sealed_cap_only_parameters(
    parameters: dict,
    *,
    rounded: bool = False,
    core_center_gap_mm: float | None = None,
    wcp_len_x_mm: float | None = None,
) -> dict:
    """Return an exact full-model, cap-only parameter payload."""
    sealed = dict(parameters)
    sealed.update(EXACT_CAP_ONLY_OVERRIDES)
    sealed["round_corner"] = int(bool(rounded))
    if core_center_gap_mm is not None:
        sealed["core_center_gap_mm"] = float(core_center_gap_mm)
    if wcp_len_x_mm is not None:
        sealed["wcp_len_x"] = float(wcp_len_x_mm)
    return sealed


def _validate_project_path_budget(
    project_path: Path,
    project_name: str,
    *,
    platform_name: str | None = None,
) -> None:
    """Fail early when a Windows AEDT result path would exceed its safe budget."""
    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt":
        return
    representative_result = (
        project_path
        / f"{project_name}.aedtresults"
        / "maxwell_cap.results"
        / "DV000_SOL000_V000.Field000"
        / "fields.hdr"
    )
    if len(str(representative_result)) >= WINDOWS_AEDT_RESULT_PATH_LIMIT:
        raise ValueError(
            "AEDT result path is too long for a reliable Windows solve "
            f"({len(str(representative_result))} characters); choose a short "
            "output root such as C:\\w\\refcap"
        )


def _status_base(args: argparse.Namespace, sim, stage: str) -> dict:
    rounded = bool(getattr(args, "rounded", False))
    return {
        "schema": "mft-direct-full-cap-gui-status-v1",
        "stage": stage,
        "project_name": getattr(sim, "PROJECT_NAME", str(args.project_name)),
        "project_path": getattr(sim, "project_path", ""),
        "controller_pid": os.getpid(),
        "full_model": True,
        "rounded_winding": rounded,
        "solver_dispatch": "cap_only",
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--project-name",
        default="reference260706_full_nonrounded_cap_gui",
    )
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument(
        "--rounded",
        action="store_true",
        help="Build the full model with rounded/bent winding corners.",
    )
    parser.add_argument(
        "--core-center-gap-mm",
        type=float,
        default=None,
        help="Override the physical center-leg air gap for geometry parity.",
    )
    parser.add_argument(
        "--wcp-len-x-mm",
        type=float,
        default=None,
        help="Override the winding cooling plate length for rounded geometry.",
    )
    parser.add_argument(
        "--prior-ltx-uh",
        type=float,
        default=7745.74713934634,
        help="Prior full-reference Ltx, used only for derived LC fields.",
    )
    parser.add_argument(
        "--prior-lrx-uh",
        type=float,
        default=773038.495371942,
        help="Prior full-reference Lrx, used only for derived LC fields.",
    )
    parser.add_argument(
        "--prior-llt-uh",
        type=float,
        default=63.3481121527661,
        help="Prior full-reference Llt, used only for derived LC fields.",
    )
    args = parser.parse_args(argv)
    if args.cores <= 0:
        parser.error("--cores must be positive")
    return args


def run_direct_cap_gui(args: argparse.Namespace, runner) -> int:
    """Execute the direct cap-only workflow with an injectable solver module."""
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "direct_cap_gui_status.json"
    project_root = output_root / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    _validate_project_path_budget(
        (project_root / args.project_name).resolve(),
        str(args.project_name),
    )

    desktop = None
    sim = None
    stage = "startup"
    exit_code = 1
    try:
        parameters = _sealed_cap_only_parameters(
            _load_parameters(args.params.resolve()),
            rounded=bool(getattr(args, "rounded", False)),
            core_center_gap_mm=getattr(args, "core_center_gap_mm", None),
            wcp_len_x_mm=getattr(args, "wcp_len_x_mm", None),
        )
        input_df, physics_revision = runner._load_fixed_input_parameter(
            parameters
        )
        _, df_plus = runner.validation_check(input_df, strict=True)
        df_plus["physics_data_revision"] = physics_revision

        stage = "desktop"
        desktop = runner.pyDesktop(
            version=None,
            non_graphical=False,
            close_on_exit=False,
            new_desktop=True,
        )
        sim = runner.Simulation(desktop=desktop)
        sim.NUM_CORE = int(args.cores)
        sim.input_df = input_df
        sim.df_plus = df_plus
        sim.full_model = True
        sim.PROJECT_NAME = str(args.project_name)
        sim.project_path = str((project_root / args.project_name).resolve())
        _write_status(status_path, **_status_base(args, sim, stage))

        # Geometry source only: intentionally no matrix boundary, setup, native
        # analyze, extraction, or convergence call is made in this workflow.
        stage = "geometry_source"
        sim.create_project()
        sim.create_design(
            name="geometry_source_no_solve",
            solution="AC Magnetic",
        )
        runner.set_design_variables(sim.design1, sim.input_df)
        sim.create_core()
        sim.create_coil()
        sim.design_matrix = sim.design1

        stage = "maxwell_cap_model"
        cap_design = sim.create_capacitance_design(name="maxwell_cap")
        sim.design_cap = cap_design
        sim.design1 = cap_design
        # Raw C extraction is independent of L.  These sealed prior magnetic
        # values are used only for the explicitly derived resonance columns.
        sim.df1 = pd.DataFrame([{
            "Ltx": float(args.prior_ltx_uh),
            "Lrx": float(args.prior_lrx_uh),
            "Llt": float(args.prior_llt_uh),
        }])
        sim.save_project()
        _write_status(
            status_path,
            **_status_base(args, sim, stage),
            active_design="maxwell_cap",
            capacitance_model="two_net_screening",
            inductance_use="prior_reference_only_for_derived_lc_fields",
        )

        stage = "maxwell_cap_solve"
        elapsed = sim.analyze_and_extract(
            "cap", sim.get_capacitance_parameter
        )
        cap_payload = sim.df_cap.iloc[0].to_dict()
        result_path = output_root / "direct_cap_result.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema": "mft-direct-full-cap-result-v1",
                    "full_model": True,
                    "rounded_winding": bool(
                        getattr(args, "rounded", False)
                    ),
                    "capacitance_model": "two_net_screening",
                    "solver_stage": "maxwell_cap",
                    "elapsed_s": float(elapsed),
                    "inductance_use": (
                        "prior full-reference values only for derived LC "
                        "fields; raw capacitance values are independent"
                    ),
                    "result": cap_payload,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        sim.save_project()
        sim.project.desktop.odesktop.SetActiveProject(
            sim.PROJECT_NAME
        ).SetActiveDesign("maxwell_cap")
        stage = "complete_gui_held"
        exit_code = 0
        _write_status(
            status_path,
            **_status_base(args, sim, stage),
            active_design="maxwell_cap",
            capacitance_model="two_net_screening",
            elapsed_s=float(elapsed),
            result_path=str(result_path),
        )
    except Exception as error:
        logging.exception("direct full capacitance GUI run failed")
        if sim is not None:
            try:
                sim.save_project()
            except Exception:
                logging.exception("failed to save inspection project")
            try:
                project = sim.project.desktop.odesktop.SetActiveProject(
                    sim.PROJECT_NAME
                )
                design_names = [
                    str(name) for name in (project.GetTopDesignList() or [])
                ]
                if any("maxwell_cap" in name for name in design_names):
                    project.SetActiveDesign("maxwell_cap")
            except Exception:
                logging.exception(
                    "failed to reactivate maxwell_cap after error"
                )
        _write_status(
            status_path,
            **_status_base(args, sim, f"failed:{stage}"),
            active_design=(
                "maxwell_cap"
                if stage.startswith("maxwell_cap")
                or stage == "complete_gui_held"
                else "geometry_source_no_solve"
            ),
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(
                    close_projects=False,
                    close_on_exit=False,
                )
            except Exception:
                logging.exception(
                    "failed to detach while preserving the AEDT GUI"
                )
    return exit_code


def main(argv=None) -> int:
    args = parse_args(argv)
    # Import after the caller has selected the repository and PyAEDT library
    # roots.  This also keeps helper-only unit tests independent of AEDT.
    import run_simulation_260706 as runner

    return run_direct_cap_gui(args, runner)


if __name__ == "__main__":
    sys.exit(main())
