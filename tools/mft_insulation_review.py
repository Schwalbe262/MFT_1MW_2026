"""Offline insulation assessment or native fields from a preserved project copy.

Examples (use the Python environment with PyAEDT installed)::

    python tools/mft_insulation_review.py assess --input evidence.json --output assessment.json
    python tools/mft_insulation_review.py extract --project model.aedt --design maxwell_cap \
        --case secondary --output-dir review_secondary --rerun

Native extraction is diagnostic. It never establishes AC withstand or PD pass,
and adaptive energy convergence does not establish convergence of local Emax.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.insulation_assessment import TEST_NAMES, assess_insulation

SOLUTION = "Setup1 : LastAdaptive"
REQUIRED_BOUNDARIES = ("CapTx", "CapRx", "CapGroundSolids", "CapGroundRegion")


def voltage_peak_v(expression):
    """Parse numeric AEDT voltage expressions without eval or variable lookup.

    The result is the applied electrostatic amplitude in volts. No RMS/peak
    conversion is inferred from the expression itself.
    """
    match = re.fullmatch(r"\s*(.*?)\s*(kV|V)\s*", str(expression), re.IGNORECASE)
    if not match or not match.group(1):
        raise ValueError(f"voltage must be numeric with V/kV suffix: {expression!r}")
    try:
        tree = ast.parse(match.group(1), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid voltage expression: {expression!r}") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("voltage expression is too complex")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                result = left / right
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords):
            result = math.sqrt(evaluate(node.args[0]))
        else:
            raise ValueError("voltage expression permits only numbers, + - * / and sqrt")
        if not math.isfinite(result):
            raise ValueError("voltage expression is nonfinite")
        return result

    try:
        result = evaluate(tree.body) * (1000 if match.group(2).lower() == "kv" else 1)
    except (ArithmeticError, OverflowError) as error:
        raise ValueError(f"invalid voltage expression: {expression!r}") from error
    if not math.isfinite(result):
        raise ValueError("voltage expression is nonfinite")
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_project_snapshot(project, output_dir):
    source = Path(project).resolve(strict=True)
    if source.suffix.lower() != ".aedt" or not source.is_file():
        raise ValueError("--project must identify an existing .aedt file")
    results = source.with_suffix(".aedtresults")
    output = Path(output_dir).resolve()
    if output.is_relative_to(results.resolve()):
        raise ValueError("output directory must be outside source .aedtresults")
    output.mkdir(parents=True, exist_ok=True)
    working = output / "working_project"
    working.mkdir(exist_ok=False)
    before = file_sha256(source)
    copied = working / source.name
    shutil.copy2(source, copied)
    if results.exists():
        shutil.copytree(results, working / results.name)
    if file_sha256(source) != before or file_sha256(copied) != before:
        raise RuntimeError("source project changed while making the inspection snapshot")
    return copied, {
        "source_project": str(source), "source_project_sha256": before,
        "snapshot_project": str(copied), "snapshot_project_initial_sha256": before,
        "source_results_copied": results.exists(),
    }


def _names(value):
    if isinstance(value, str):
        return [value]
    return list(value or [])


def _conductivity(app, obj):
    material = app.materials[obj.material_name]
    value = getattr(material.conductivity, "value", material.conductivity)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"cannot classify conductivity of material {obj.material_name!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid conductivity of material {obj.material_name!r}")
    return result


def _boundary_object_names(app, assignment):
    """Native voltage props use integer object IDs; some wrappers use names."""
    names = set()
    for value in _names(assignment):
        if isinstance(value, int) or (isinstance(value, str) and value.isdecimal()):
            try:
                names.add(app.modeler.objects[int(value)].name)
            except KeyError as error:
                raise ValueError(f"boundary references missing native object ID {value}") from error
        elif isinstance(value, str):
            names.add(value)
        else:
            raise ValueError(f"unsupported boundary object assignment {value!r}")
    return names


def boundary_contract(app):
    boundaries = {boundary.name: boundary for boundary in app.boundaries}
    missing = set(REQUIRED_BOUNDARIES) - boundaries.keys()
    if missing:
        raise ValueError(f"required voltage boundaries missing: {sorted(missing)}")
    selected = {name: boundaries[name] for name in REQUIRED_BOUNDARIES}
    for name, boundary in selected.items():
        if str(boundary.type).lower() != "voltage":
            raise ValueError(f"{name} is not a Voltage boundary")
    objects = {name: _boundary_object_names(app, selected[name].props.get("Objects"))
               for name in REQUIRED_BOUNDARIES[:3]}
    tx, rx, ground = (objects[name] for name in REQUIRED_BOUNDARIES[:3])
    if not tx or not rx or not ground or tx & rx or (tx | rx) & ground:
        raise ValueError("primary, secondary and hardware voltage groups must be nonempty and disjoint")
    actual = set(app.modeler.object_names)
    if not (tx | rx | ground) <= actual:
        raise ValueError("voltage boundary references a missing solid")
    for prefix, grouped in (("Tx_", tx), ("Rx_", rx)):
        # Cooling plates and dielectric pads also have Tx_/Rx_ prefixes.
        # Grounded hardware is not part of the shorted winding electrode.
        turns = {name for name in actual - ground if name.startswith(prefix)
                 and _conductivity(app, app.modeler[name]) > 1}
        if not turns or turns != grouped:
            raise ValueError(f"{prefix} turns are not all in their own equipotential boundary")
    if not _names(selected["CapGroundRegion"].props.get("Faces")):
        raise ValueError("CapGroundRegion must ground the modeled remote region faces")
    if any(str(boundary.type).lower() == "voltage" and name not in selected
           for name, boundary in boundaries.items()):
        raise ValueError("additional voltage boundaries make the requested two-net test ambiguous")
    for name in ("CapGroundSolids", "CapGroundRegion"):
        if voltage_peak_v(selected[name].props["Voltage"]) != 0:
            raise ValueError(f"{name} must be at 0 V")

    dielectric = []
    for name in sorted(actual - tx - rx - ground):
        obj = app.modeler[name]
        if _conductivity(app, obj) > 1:
            raise ValueError(f"modeled conductive hardware is not grounded: {name}")
        dielectric.append(obj)
    if "Region" not in {obj.name for obj in dielectric}:
        raise ValueError("air Region is missing from the dielectric extraction domain")
    return selected, objects, dielectric


def native_variation(app):
    variation = []
    for name, value in app.available_variations.get_independent_nominal_values().items():
        if app.variable_manager.variables[name].sweep:
            variation.extend([name + ":=", value])
    return variation


def last_numeric_line(path):
    for line in reversed(Path(path).read_text(encoding="utf-8-sig").splitlines()):
        try:
            value = float(line.strip())
        except ValueError:
            continue
        if math.isfinite(value):
            return value
        raise ValueError(f"nonfinite calculator export: {path}")
    raise ValueError(f"calculator export contains no numeric result: {path}")


def maximum_location_mm(app, object_name, output_dir):
    reporter = app.post.ofieldsreporter
    coordinates = []
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", object_name)
    for axis in "XYZ":
        target = Path(output_dir) / f"{safe_name}_maxpos_{axis}.fld"
        reporter.CalcStack("clear")
        try:
            reporter.CopyNamedExprToStack("Mag_E")
            reporter.EnterVol(object_name)
            reporter.CalcOp("VolumeValue")
            reporter.CalcOp("MaxPos")
            reporter.CalcOp("Scalar" + axis)
            result = reporter.CalculatorWrite(str(target), ["Solution:=", SOLUTION], native_variation(app))
            if result is False:
                raise RuntimeError(f"CalculatorWrite failed for {object_name}/{axis}")
            coordinates.append(last_numeric_line(target) * 1000.0)
        finally:
            reporter.CalcStack("clear")
    return coordinates


def verify_solution_potentials(app, objects, excited, peak, *, rerun):
    probes = []
    try:
        for boundary, names in objects.items():
            expected = peak if boundary == excited else 0.0
            for name in sorted(names):
                for operation in ("Minimum", "Maximum"):
                    raw = app.post.get_scalar_field_value(
                        "Voltage", scalar_function=operation, solution=SOLUTION,
                        object_name=name, object_type="volume",
                    )
                    if raw is None or isinstance(raw, bool):
                        raise RuntimeError("native Voltage quantity is unavailable")
                    actual = float(raw)
                    if not math.isfinite(actual):
                        raise RuntimeError("native Voltage quantity is unavailable")
                    probes.append({"object": name, "operation": operation,
                                   "expected_V": expected, "actual_V": actual})
                    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=max(abs(peak) * 1e-6, 1e-7)):
                        raise ValueError("native solved potential disagrees with boundary voltage; rerun required")
    except (RuntimeError, TypeError, AttributeError) as error:
        if not rerun:
            raise ValueError("saved field voltage cannot be verified; use --rerun on the copied project") from error
        return {"status": "unavailable", "reason": str(error), "probes": probes,
                "normalization_evidence": "fresh_rerun_at_prescribed_boundary_voltage"}
    return {"status": "verified", "probes": probes}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract(args, app_factory=None):
    copied, provenance = copy_project_snapshot(args.project, args.output_dir)
    output = Path(args.output_dir).resolve()
    app = None
    try:
        if app_factory is None:
            from ansys.aedt.core import Maxwell3d
            app_factory = Maxwell3d
        app = app_factory(project=str(copied), design=args.design, version="2025.2",
                          non_graphical=True, new_desktop=True, close_on_exit=True)
        if Path(app.project_file).resolve() != copied.resolve():
            raise RuntimeError("AEDT opened a different project path; refusing to edit or extract")
        if app.design_name != args.design or app.solution_type != "Electrostatic":
            raise ValueError("requested design is not the exact Electrostatic design")
        setup = app.get_setup("Setup1")
        provenance["original_boundaries"] = {name: deepcopy(dict(boundary.props))
                                              for name, boundary in ((b.name, b) for b in app.boundaries)}
        provenance["original_setup"] = deepcopy(dict(setup.props))
        provenance["design"] = app.design_name
        provenance["solution"] = SOLUTION
        provenance["aedt_version"] = str(app.aedt_version_id)
        provenance["pyaedt_version"] = importlib.metadata.version("pyaedt")
        _write_json(output / "native_provenance.json", provenance)
        boundaries, objects, dielectric = boundary_contract(app)
        excited, opposite = ("CapTx", "CapRx") if args.case == "primary" else ("CapRx", "CapTx")
        if args.rerun:
            for name, voltage in ((excited, "1V"), (opposite, "0V")):
                boundaries[name].props["Voltage"] = voltage
                if boundaries[name].update() is False:
                    raise RuntimeError(f"failed to update {name} voltage on copied project")
            setup.props.update({"MaximumPasses": 10, "MinimumPasses": 2,
                                "MinimumConvergedPasses": 1, "PercentError": 1.0,
                                "SolveFieldOnly": True, "SolveMatrixAtLast": False})
            if setup.update() is False or setup.analyze(cores=4) is False:
                raise RuntimeError("copied-project electrostatic rerun failed")
            if app.save_project() is False:
                raise RuntimeError("failed to save copied project")
        amplitude = voltage_peak_v(boundaries[excited].props["Voltage"])
        if amplitude == 0 or voltage_peak_v(boundaries[opposite].props["Voltage"]) != 0:
            raise ValueError("selected case requires nonzero excited and zero opposite winding; use --rerun")
        potentials = verify_solution_potentials(app, objects, excited, amplitude, rerun=args.rerun)
        hotspots = []
        for obj in dielectric:
            raw = app.post.get_scalar_field_value(
                "Mag_E", scalar_function="Maximum", solution=SOLUTION,
                object_name=obj.name, object_type="volume",
            )
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"native E maximum missing for {obj.name}")
            maximum = float(raw)
            if not math.isfinite(maximum) or maximum < 0:
                raise ValueError(f"invalid native E maximum for {obj.name}")
            hotspots.append({"object_name": obj.name, "material": obj.material_name,
                             "E_actual_V_per_m": maximum,
                             "E_1V_V_per_m": maximum / abs(amplitude),
                             "location_mm": maximum_location_mm(app, obj.name, output)})
        if not any(item["E_actual_V_per_m"] > 0 for item in hotspots):
            raise ValueError("all native E maxima are zero despite nonzero excitation")
        provenance.update({"actual_excitation_voltage_V": amplitude, "rerun": bool(args.rerun),
                           "snapshot_project_final_sha256": file_sha256(copied),
                           "working_boundaries": {name: dict(boundary.props) for name, boundary in boundaries.items()},
                           "working_setup": dict(setup.props), "potential_readback": potentials})
        test_name = TEST_NAMES[0 if args.case == "primary" else 1]
        payload = {
            "schema_version": "mft-native-insulation-field-extraction-v1",
            "provenance": provenance,
            "model": {"source_geometry_kind": "existing_aedt_not_verified_final_cad"},
            "materials": [{"name": name} for name in sorted({obj.material_name for obj in dielectric})],
            "convergence": {"field_local_max_converged": False},
            "tests": {test_name: {
                "boundary_conditions": {"terminals_within_each_winding_shorted": True,
                                        "primary_secondary_connected": False,
                                        "other_winding_grounded": True,
                                        "all_core_cooling_hardware_grounded": True,
                                        "excited_winding": args.case},
                "dielectric_regions_covered": True,
                "native_field_evidence": {"source": "native Maxwell volume Maximum and MaxPos exports",
                                          "solution": SOLUTION, "normalization_voltage_V": 1.0, "unit": "V/m"},
                "hotspots": hotspots,
            }},
        }
        _write_json(output / "native_insulation_fields.json", payload)
        _write_json(output / "insulation_assessment.json", assess_insulation(payload))
        return payload
    except Exception as error:
        _write_json(output / "extraction_failure.json", {"status": "failed", "error": str(error),
                                                       "provenance": provenance})
        raise
    finally:
        try:
            if app is not None:
                app.release_desktop(close_projects=True, close_desktop=True)
        finally:
            if file_sha256(Path(provenance["source_project"])) != provenance["source_project_sha256"]:
                raise RuntimeError("source project changed during extraction")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("assess", help="assess saved evidence without AEDT")
    offline.add_argument("--input", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    native = commands.add_parser("extract", help="copy project/results and extract one native test case")
    native.add_argument("--project", type=Path, required=True)
    native.add_argument("--design", required=True)
    native.add_argument("--case", choices=("primary", "secondary"), required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    native.add_argument("--rerun", action="store_true", help="solve prescribed 1V test case on copied project with four cores")
    args = parser.parse_args(argv)
    try:
        if args.command == "assess":
            _write_json(args.output, assess_insulation(args.input))
        else:
            extract(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Insulation review failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
