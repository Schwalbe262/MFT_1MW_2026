#!/usr/bin/env python3
"""Reopen the healthy sibling from job 732549 in a fresh AEDT Desktop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import traceback


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_project).resolve()
    source_results = Path(str(source) + "results")
    artifact = Path(args.artifact_dir).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    clone_dir = artifact / "clone"
    clone_dir.mkdir()
    clone = clone_dir / source.name
    clone_results = Path(str(clone) + "results")
    shutil.copy2(source, clone)
    if source_results.is_dir():
        shutil.copytree(source_results, clone_results)
    result = {
        "schema": 1,
        "source_project": str(source),
        "source_project_sha256": digest(source),
        "source_results_exists": source_results.is_dir(),
        "clone_project": str(clone),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "priority_intent": os.environ.get("PILOT_PRIORITY_INTENT"),
    }
    desktop = None
    try:
        import ansys.aedt.core as pyaedt
        from ansys.aedt.core import Desktop, Maxwell3d

        result["pyaedt"] = pyaedt.__version__
        desktop = Desktop(
            version="2025.2",
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        result["desktop_pid"] = int(desktop.odesktop.GetProcessID())
        result["grpc_port"] = int(desktop.port)
        app = Maxwell3d(
            project=str(clone),
            design="DesignB",
            solution_type="ElectroDCConduction",
            version="2025.2",
            non_graphical=True,
            new_desktop=False,
            close_on_exit=False,
            port=int(desktop.port),
        )
        objects = sorted(app.modeler.object_names)
        setups = sorted(app.setup_names)
        sweeps = sorted(str(value) for value in app.existing_analysis_sweeps)
        nominal = str(app.nominal_adaptive)
        result.update(
            {
                "project": app.project_name,
                "design": app.design_name,
                "objects": objects,
                "setups": setups,
                "existing_analysis_sweeps": sweeps,
                "nominal_adaptive": nominal,
                "project_list": sorted(str(value) for value in desktop.project_list),
            }
        )
        native_solutions = []
        try:
            module = app.odesign.GetModule("Solutions")
            for method_name in (
                "GetAllSolutionNames",
                "GetSolutionNames",
                "GetAvailableSolutionNames",
            ):
                method = getattr(module, method_name, None)
                if callable(method):
                    try:
                        native_solutions = [str(value) for value in (method() or [])]
                        if native_solutions:
                            break
                    except Exception:
                        continue
        except Exception as exc:
            result["native_solution_error"] = f"{type(exc).__name__}: {exc}"
        result["native_solutions"] = native_solutions

        convergence = artifact / "B_convergence_fresh.prop"
        try:
            exported = app.export_convergence("Setup1", output_file=str(convergence))
            result["convergence_call_result"] = str(exported)
        except Exception as exc:
            result["convergence_error"] = f"{type(exc).__name__}: {exc}"
        result["convergence_exists"] = (
            convergence.exists() and convergence.stat().st_size > 0
        )
        result["convergence_size"] = (
            convergence.stat().st_size if convergence.exists() else 0
        )
        convergence_text = (
            convergence.read_text(encoding="utf-8", errors="replace")
            if convergence.exists()
            else ""
        )
        completed_match = re.search(
            r"^Completed\s*:\s*(\S+)", convergence_text, re.MULTILINE
        )
        completed_value = completed_match.group(1) if completed_match else ""
        convergence_rows = [
            line
            for line in convergence_text.splitlines()
            if re.match(r"^\s*\d+\|", line)
        ]
        result["convergence_completed"] = completed_value
        result["convergence_data_rows"] = len(convergence_rows)
        result["convergence_numeric_complete"] = bool(
            completed_value.isdigit()
            and int(completed_value) > 0
            and convergence_rows
        )

        quantities = []
        quantity_error = ""
        for solution in [nominal, *sweeps]:
            if not solution:
                continue
            try:
                quantities = [
                    str(value)
                    for value in app.post.available_report_quantities(solution=solution)
                ]
                if quantities:
                    result["quantity_solution"] = solution
                    break
            except Exception as exc:
                quantity_error = f"{type(exc).__name__}: {exc}"
        result["available_quantity_count"] = len(quantities)
        result["available_quantities_sample"] = quantities[:50]
        if quantity_error:
            result["quantity_error"] = quantity_error

        asol = clone_results / "DesignB.asol"
        asol_text = asol.read_text(encoding="utf-8", errors="replace") if asol.exists() else ""
        result["asol_exists"] = asol.exists()
        result["asol_sha256"] = digest(asol) if asol.exists() else None
        result["asol_has_last_adaptive"] = "LastAdaptive" in asol_text
        result["identity_intact"] = (
            result["project"] == "pilot_B"
            and result["design"] == "DesignB"
            and setups == ["Setup1"]
            and objects == [f"B_conductor_{index:02d}" for index in range(12)]
        )
        result["solution_index_intact"] = bool(
            result["asol_has_last_adaptive"]
            and any("LastAdaptive" in value for value in [nominal, *sweeps])
        )
        result["fresh_reopen_pass"] = bool(
            result["identity_intact"]
            and result["solution_index_intact"]
            and result["convergence_numeric_complete"]
        )
        return_code = 0 if result["fresh_reopen_pass"] else 4
    except Exception as exc:
        result["fresh_reopen_pass"] = False
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return_code = 2
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=True, close_on_exit=True)
            except Exception as exc:
                result["release_error"] = f"{type(exc).__name__}: {exc}"
        (artifact / "reopen_verdict.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
