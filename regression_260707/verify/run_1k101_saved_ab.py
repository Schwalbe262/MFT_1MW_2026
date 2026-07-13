"""Evaluate the 1K101 native-lamination gate from a solved AEDT project.

This runner is intentionally postprocess-only.  It opens an existing Maxwell
project, verifies that ``Setup1 : LastAdaptive`` is present, performs ordinary
field/result extraction, and closes the project without saving it.  It never
dispatches an AEDT analysis.

The voltage and material-reference inputs are not all AEDT design variables.
Consequently a scheduler ``cand.json`` (or an explicit ``--params-path``) is
required and is checked against the geometry variables retained in the
project.  Missing or inconsistent provenance fails closed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_SPEC = Path(__file__).with_name("1k101_native_ab_gate.json")
LIBRARY_ROOT = Path(os.environ.get(
    "MFT_PYAEDT_LIBRARY_ROOT", REPO_ROOT.parent / "pyaedt_library"
)).resolve()
LIBRARY_SRC = LIBRARY_ROOT if LIBRARY_ROOT.name == "src" else LIBRARY_ROOT / "src"
if str(LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(LIBRARY_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EVIDENCE_SCHEMA = "mft-1k101-saved-native-ab-evidence-v1"
CORE_OBJECT_RE = re.compile(
    r"^core_(\d+)_(leg_(?:left|center|right)|yoke_(?:top|bottom))$"
)
CORE_REGIONS = (
    "leg_left", "leg_center", "leg_right", "yoke_bottom", "yoke_top",
)
SYMMETRY_CORE_REGIONS = ("leg_left", "leg_center", "yoke_top")
REQUIRED_PARAM_KEYS = {
    "V1_rms", "freq", "N1_main", "N1_side", "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t", "core_cm",
    "core_x", "core_y", "core_loss_margin", "full_model", "loss_sym_on",
}
PROJECT_VARIABLE_KEYS = (
    "N1_main", "N1_side", "l1", "l2", "h1", "w1", "n_core_group",
    "core_plate_t", "core_plate_pad_t", "full_model",
)
REQUIRED_NUMERICAL_GATE_KEYS = {
    "material_readback_exact",
    "geometry_relative_error_max",
    "faraday_relative_error_max",
    "induced_vs_source_peak_relative_error_max",
    "standard_Bavg_vs_Faraday_Bmaterial_relative_error_max",
    "native_loss_vs_Faraday_POWERLITE_mass_relative_error_max",
    "thermal_native_power_balance_relative_error_max",
    "B_material_kf0p85_ratio_expected",
    "B_material_kf0p70_ratio_expected",
    "B_material_kf_ratio_relative_tolerance",
}


class AedtOperationTimeout(TimeoutError):
    """Raised when one bounded AEDT operation exceeds its time budget."""


@contextmanager
def operation_timeout(seconds: float, label: str):
    """Bound one AEDT call using the established Linux SIGALRM pattern."""
    seconds = float(seconds)
    if seconds <= 0 or os.name == "nt":
        yield
        return

    def _expired(_signum, _frame):
        raise AedtOperationTimeout(
            f"AEDT operation {label!r} exceeded {seconds:g} seconds"
        )

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _aedt_call(label: str, timeout_seconds: float, function: Callable, *args):
    started = time.monotonic()
    print(f"SAVED_AB_AEDT_START {label}", flush=True)
    with operation_timeout(timeout_seconds, label):
        value = function(*args)
    print(
        f"SAVED_AB_AEDT_PASS {label} "
        f"elapsed_seconds={time.monotonic() - started:.6f}",
        flush=True,
    )
    return value


def _snapshot_descendants() -> dict[int, float]:
    """Capture only descendants of this runner, with PID-reuse protection."""
    try:
        import psutil

        parent = psutil.Process(os.getpid())
        return {
            process.pid: process.create_time()
            for process in parent.children(recursive=True)
        }
    except Exception:
        return {}


def _terminate_new_descendants(
    baseline: Mapping[int, float], captured: Mapping[int, float], wait_seconds=5.0,
) -> int:
    """Terminate only processes created below this runner after its baseline."""
    import psutil

    processes = []
    for pid, create_time in captured.items():
        if pid in baseline and baseline[pid] == create_time:
            continue
        try:
            process = psutil.Process(pid)
            if process.create_time() != create_time:
                continue
            process.terminate()
            processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _gone, alive = psutil.wait_procs(processes, timeout=float(wait_seconds))
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=float(wait_seconds))
    return len(processes)


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def relative_error(
    value: Any, reference: Any, *, denominator: Any | None = None,
) -> float | None:
    """Return a finite relative error, or ``None`` for unusable inputs."""
    actual = _number(value)
    expected = _number(reference)
    scale = _number(reference if denominator is None else denominator)
    if actual is None or expected is None or scale is None:
        return None
    return abs(actual - expected) / max(abs(scale), 1.0e-12)


def _limit_metric(
    value: Any,
    limit: Any,
    *,
    spec_key: str,
    applicable: bool = True,
    unavailable_reason: str = "required evidence is unavailable",
) -> dict:
    numeric_value = _number(value)
    numeric_limit = _number(limit)
    if not applicable:
        return {
            "value": numeric_value,
            "limit": numeric_limit,
            "comparison": "<=",
            "spec_key": spec_key,
            "applicable": False,
            "available": numeric_value is not None,
            "passed": True,
            "reason": "not applicable to this saved case",
        }
    available = numeric_value is not None and numeric_limit is not None
    return {
        "value": numeric_value,
        "limit": numeric_limit,
        "comparison": "<=",
        "spec_key": spec_key,
        "applicable": True,
        "available": available,
        "passed": bool(
            available and numeric_value >= 0.0 and numeric_value <= numeric_limit
        ),
        **({} if available else {"reason": unavailable_reason}),
    }


def _exact_metric(value: Any, expected: bool, *, spec_key: str) -> dict:
    available = isinstance(value, bool)
    return {
        "value": value if available else None,
        "expected": bool(expected),
        "comparison": "==",
        "spec_key": spec_key,
        "applicable": True,
        "available": available,
        "passed": bool(available and value is bool(expected)),
        **({} if available else {"reason": "required evidence is unavailable"}),
    }


def _ratio_expected(gates: Mapping[str, Any], kf: float) -> tuple[float, str]:
    if math.isclose(kf, 0.85, rel_tol=0.0, abs_tol=1.0e-12):
        key = "B_material_kf0p85_ratio_expected"
    elif math.isclose(kf, 0.70, rel_tol=0.0, abs_tol=1.0e-12):
        key = "B_material_kf0p70_ratio_expected"
    else:
        return 1.0 / kf, "derived_1_over_kf"
    expected = _number(gates.get(key))
    if expected is None:
        raise ValueError(f"gate specification has invalid {key}")
    return expected, key


def _validate_gate_configuration(
    gate_spec: Mapping[str, Any], gates: Mapping[str, Any], kf: float,
) -> None:
    if gates.get("material_readback_exact") is not True:
        raise ValueError("material_readback_exact must be true")
    nonnegative = (
        "geometry_relative_error_max",
        "faraday_relative_error_max",
        "induced_vs_source_peak_relative_error_max",
        "standard_Bavg_vs_Faraday_Bmaterial_relative_error_max",
        "native_loss_vs_Faraday_POWERLITE_mass_relative_error_max",
        "thermal_native_power_balance_relative_error_max",
        "B_material_kf_ratio_relative_tolerance",
    )
    for key in nonnegative:
        value = _number(gates.get(key))
        if value is None or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in (
        "B_material_kf0p85_ratio_expected",
        "B_material_kf0p70_ratio_expected",
    ):
        value = _number(gates.get(key))
        if value is None or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    cases = gate_spec.get("ab_cases")
    if not isinstance(cases, list):
        raise ValueError("gate specification has no ab_cases list")
    candidates = [
        _number(item.get("stacking_factor"))
        for item in cases if isinstance(item, Mapping)
    ]
    if not any(
        value is not None
        and math.isclose(value, kf, rel_tol=0.0, abs_tol=1.0e-12)
        for value in candidates
    ):
        raise ValueError(f"kf={kf:g} is not an ab_cases stacking factor")


def evaluate_numerical_gates(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
    kf: float,
) -> dict:
    """Pure-Python evaluation of every numerical gate relevant to one case."""
    kf_number = _number(kf)
    if kf_number is None or not 0 < kf_number <= 1:
        raise ValueError("kf must satisfy 0 < kf <= 1")
    gates = gate_spec.get("numerical_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("gate specification has no numerical_gates object")
    missing = sorted(REQUIRED_NUMERICAL_GATE_KEYS - set(gates))
    if missing:
        raise ValueError(f"gate specification is missing keys: {missing}")
    _validate_gate_configuration(gate_spec, gates, kf_number)

    geometry_errors = observed.get("geometry_relative_errors")
    geometry_value = None
    if isinstance(geometry_errors, Mapping):
        required_geometry = ("area", "volume", "mass")
        values = [_number(geometry_errors.get(key)) for key in required_geometry]
        if all(value is not None for value in values):
            geometry_value = max(values)  # type: ignore[arg-type]

    b_error = relative_error(
        observed.get("native_B_material_mean_T"), reference.get("B_material_T")
    )
    loss_error = relative_error(
        observed.get("native_core_loss_raw_W"),
        reference.get("native_raw_loss_W"),
    )
    frequency = _number(reference.get("frequency_hz"))
    induced = _number(observed.get("induced_voltage_peak_V"))
    linkage = _number(observed.get("flux_linkage_peak_Wb_turn"))
    faraday_voltage = (
        2.0 * math.pi * frequency * linkage
        if frequency is not None and linkage is not None else None
    )
    faraday_error = relative_error(
        induced, faraday_voltage, denominator=induced
    )
    source_peak = _number(reference.get("source_voltage_peak_V"))
    source_error = relative_error(induced, source_peak)

    b_pack = _number(observed.get("native_B_pack_mean_T"))
    b_material = _number(observed.get("native_B_material_mean_T"))
    ratio = (
        b_material / b_pack
        if b_material is not None and b_pack is not None and abs(b_pack) > 1.0e-12
        else None
    )
    ratio_expected, ratio_expected_key = _ratio_expected(gates, kf_number)
    ratio_error = relative_error(ratio, ratio_expected)

    thermal_required = bool(observed.get("thermal_required", False))
    thermal_error = relative_error(
        observed.get("thermal_native_power_W"),
        observed.get("thermal_expected_power_W"),
    )

    metrics = {
        "material_readback_exact": _exact_metric(
            observed.get("material_readback_exact"),
            bool(gates["material_readback_exact"]),
            spec_key="material_readback_exact",
        ),
        "geometry_relative_error": _limit_metric(
            geometry_value,
            gates["geometry_relative_error_max"],
            spec_key="geometry_relative_error_max",
        ),
        "faraday_relative_error": _limit_metric(
            faraday_error,
            gates["faraday_relative_error_max"],
            spec_key="faraday_relative_error_max",
        ),
        "induced_vs_source_peak_relative_error": _limit_metric(
            source_error,
            gates["induced_vs_source_peak_relative_error_max"],
            spec_key="induced_vs_source_peak_relative_error_max",
        ),
        "standard_Bavg_vs_Faraday_Bmaterial_relative_error": _limit_metric(
            b_error,
            gates["standard_Bavg_vs_Faraday_Bmaterial_relative_error_max"],
            spec_key="standard_Bavg_vs_Faraday_Bmaterial_relative_error_max",
        ),
        "native_loss_vs_Faraday_POWERLITE_mass_relative_error": _limit_metric(
            loss_error,
            gates["native_loss_vs_Faraday_POWERLITE_mass_relative_error_max"],
            spec_key="native_loss_vs_Faraday_POWERLITE_mass_relative_error_max",
        ),
        "thermal_native_power_balance_relative_error": _limit_metric(
            thermal_error,
            gates["thermal_native_power_balance_relative_error_max"],
            spec_key="thermal_native_power_balance_relative_error_max",
            applicable=thermal_required,
            unavailable_reason="thermal evidence was required but unavailable",
        ),
        "B_material_to_B_pack_ratio": {
            "value": ratio,
            "expected": ratio_expected,
            "relative_error": ratio_error,
            "relative_tolerance": _number(
                gates["B_material_kf_ratio_relative_tolerance"]
            ),
            "comparison": "relative_error <= relative_tolerance",
            "expected_spec_key": ratio_expected_key,
            "tolerance_spec_key": "B_material_kf_ratio_relative_tolerance",
            "attestation_scope": (
                "conversion_contract_B_material_equals_extracted_B_pack_div_kf"
            ),
            "applicable": True,
            "available": ratio_error is not None,
            "passed": bool(
                ratio_error is not None
                and ratio_error <= float(
                    gates["B_material_kf_ratio_relative_tolerance"]
                )
            ),
            **({} if ratio_error is not None else {
                "reason": "required evidence is unavailable"
            }),
        },
    }
    applicable = [item for item in metrics.values() if item["applicable"]]
    return {
        "passed": all(item["passed"] for item in applicable),
        "metrics": metrics,
        "applicable_metric_count": len(applicable),
        "passed_metric_count": sum(bool(item["passed"]) for item in applicable),
    }


def _resolve_project_path(value: str) -> tuple[Path, Path]:
    supplied = Path(value).expanduser().resolve()
    if supplied.is_file():
        if supplied.suffix.lower() != ".aedt":
            raise ValueError(f"--project-path is not an AEDT project: {supplied}")
        return supplied, supplied.parent
    if not supplied.is_dir():
        raise FileNotFoundError(f"saved project path is unavailable: {supplied}")
    projects = sorted(path for path in supplied.rglob("*.aedt") if path.is_file())
    if not projects:
        raise FileNotFoundError(f"no .aedt project exists under {supplied}")
    solved_layout = [
        path for path in projects
        if path.with_suffix(path.suffix + "results").is_dir()
    ]
    candidates = solved_layout or projects
    if len(candidates) != 1:
        raise RuntimeError(
            "saved project path is ambiguous; candidates="
            + json.dumps([str(path) for path in candidates])
        )
    return candidates[0], supplied


def _json_object(path: Path) -> dict:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        if not rows:
            raise ValueError(f"no JSON object exists in {path}")
        return rows[-1]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"parameter sidecar is not a JSON object: {path}")
    return value


def _candidate_from_object(value: Mapping[str, Any], kf: float) -> dict | None:
    for key in ("parameters", "params"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidate = _candidate_from_object(nested, kf)
            if candidate is not None:
                return candidate
    if REQUIRED_PARAM_KEYS.issubset(value):
        return dict(value)
    common = value.get("common_params")
    candidates = value.get("candidates")
    if isinstance(common, Mapping) and isinstance(candidates, list):
        matches = []
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            factor = _number(
                item.get("lamination_factor", item.get("core_lamination_factor"))
            )
            if factor is not None and math.isclose(
                factor, kf, rel_tol=0.0, abs_tol=1.0e-12
            ):
                matches.append(item)
        if len(matches) == 1:
            merged = dict(common)
            merged.update(matches[0])
            merged["core_lamination_factor"] = kf
            return merged
    for key in ("result", "payload", "evidence"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            candidate = _candidate_from_object(nested, kf)
            if candidate is not None:
                return candidate
    return None


def _discover_params_path(search_root: Path, project_path: Path) -> Path:
    priority = []
    for directory in (project_path.parent, search_root):
        for name in ("cand.json", "params.json", "parameters.json"):
            priority.append(directory / name)
    priority.extend(sorted(search_root.rglob("cand.json")))
    priority.extend(sorted(search_root.rglob("params.json")))
    priority.extend(sorted(search_root.rglob("failed_samples_260706.jsonl")))
    seen = set()
    valid = []
    for path in priority:
        try:
            key = path.resolve()
        except OSError:
            continue
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        valid.append(path)
    if not valid:
        raise FileNotFoundError(
            "no scheduler parameter sidecar was found; copy cand.json into the "
            "snapshot or pass --params-path"
        )
    return valid[0]


def _load_parameters(
    params_path: str | None, search_root: Path, project_path: Path, kf: float,
) -> tuple[dict, dict, Path]:
    path = (
        Path(params_path).expanduser().resolve()
        if params_path else _discover_params_path(search_root, project_path)
    )
    if not path.is_file():
        raise FileNotFoundError(f"parameter sidecar is unavailable: {path}")
    source = _json_object(path)
    candidate = _candidate_from_object(source, kf)
    if candidate is None:
        raise ValueError(f"no complete parameter payload exists in {path}")
    missing = sorted(REQUIRED_PARAM_KEYS - set(candidate))
    if missing:
        raise ValueError(f"parameter payload is missing required keys: {missing}")

    factor_in_payload = candidate.get("core_lamination_factor")
    if factor_in_payload is None:
        candidate["core_lamination_factor"] = kf
        kf_source = "cli_authenticated_later_by_native_material_readback"
    else:
        factor = _number(factor_in_payload)
        if factor is None or not math.isclose(
            factor, kf, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"sidecar kf={factor_in_payload!r} does not match --kf={kf:g}"
            )
        kf_source = "sidecar_and_cli_exact_match"

    from module.input_parameter_260706 import (
        ALL_INPUT_KEYS, create_input_parameter, validation_check,
    )

    selected = {key: candidate[key] for key in ALL_INPUT_KEYS if key in candidate}
    input_frame = create_input_parameter(selected)
    _, validated = validation_check(input_frame, strict=True)
    row = validated.iloc[0].to_dict()
    return row, {
        "path": str(path),
        "kf_source": kf_source,
        "required_keys_authenticated": sorted(REQUIRED_PARAM_KEYS),
    }, path


def _parse_aedt_number(value: Any, label: str) -> float:
    if isinstance(value, (int, float)):
        number = _number(value)
    else:
        match = re.match(
            r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            str(value),
        )
        number = _number(match.group(1)) if match else None
    if number is None:
        raise RuntimeError(f"cannot parse AEDT number for {label}: {value!r}")
    return number


def _design_name(value: Any) -> str:
    try:
        value = value.GetName()
    except AttributeError:
        pass
    return str(value or "").split(";")[-1].strip()


def _aedt_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "on", "yes"}:
        return True
    if text in {"false", "0", "off", "no", ""}:
        return False
    raise RuntimeError(f"cannot parse AEDT boolean readback: {value!r}")


def _is_ac_magnetic_solution(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    return normalized in {"acmagnetic", "eddycurrent"}


def _core_objects(editor) -> dict[int, list[str]]:
    names = editor.GetMatchedObjectName("core_*") or []
    groups: dict[int, list[str]] = {}
    for item in names:
        name = str(item)
        match = CORE_OBJECT_RE.fullmatch(name)
        if match:
            groups.setdefault(int(match.group(1)), []).append(name)
    for group_index in groups:
        groups[group_index].sort(
            key=lambda name: CORE_REGIONS.index(CORE_OBJECT_RE.fullmatch(name).group(2))
        )
    return dict(sorted(groups.items()))


def _expected_core_group_indices(
    n_core_group: int, full_model: Any,
) -> list[int]:
    group_count = int(n_core_group)
    if group_count < 1:
        raise ValueError(f"n_core_group must be positive, got {group_count}")
    if bool(int(full_model)):
        return list(range(1, group_count + 1))

    # create_core places group i about
    # (i - (n + 1) / 2) * (core_depth + plate_stack).  The XZ
    # PositiveOnly split therefore deletes the negative-y groups; for odd n it
    # halves the central group.  The retained IDs start at floor(n / 2) + 1.
    return list(range(group_count // 2 + 1, group_count + 1))


def _validate_core_topology(
    core_groups: Mapping[int, list[str]], params: Mapping[str, Any],
) -> None:
    full_model = bool(int(params["full_model"]))
    expected_groups = _expected_core_group_indices(
        int(params["n_core_group"]), full_model
    )
    actual_groups = sorted(core_groups)
    if actual_groups != expected_groups:
        raise RuntimeError(
            f"core group coverage mismatch: actual={actual_groups}, "
            f"expected={expected_groups}"
        )

    expected_regions = CORE_REGIONS if full_model else SYMMETRY_CORE_REGIONS
    topology = "full" if full_model else "eighth-symmetry retained"
    for group_index in expected_groups:
        expected_names = [
            f"core_{group_index}_{region}" for region in expected_regions
        ]
        actual_names = core_groups[group_index]
        if actual_names != expected_names:
            raise RuntimeError(
                f"{topology} core topology mismatch for group {group_index}: "
                f"actual={actual_names}, expected={expected_names}"
            )


def _solution_marker(raw_design) -> tuple[str, ...]:
    solutions = raw_design.GetModule("Solutions")
    values = solutions.GetAvailableVariations("Setup1 : LastAdaptive") or []
    return tuple(sorted(str(item) for item in values))


def _select_solved_loss_design(
    project, preferred_name: str | None, timeout_seconds: float,
) -> tuple[Any, dict[int, list[str]], list[dict]]:
    designs = _aedt_call(
        "list-project-designs", timeout_seconds, project.GetDesigns
    ) or []
    inspected = []
    passing = []
    for item in designs:
        name = _design_name(item)
        record = {"name": name, "eligible": False}
        inspected.append(record)
        try:
            raw = item if hasattr(item, "GetName") else _aedt_call(
                f"bind-{name}", timeout_seconds, project.SetActiveDesign, name
            )
            if raw is None or raw is False or _design_name(raw) != name:
                raise RuntimeError(f"failed to bind native design {name!r}")
            if str(_aedt_call(
                f"{name}-design-type", timeout_seconds, raw.GetDesignType
            )) != "Maxwell 3D":
                record["reason"] = "not Maxwell 3D"
                continue
            solution_type = str(_aedt_call(
                f"{name}-solution-type", timeout_seconds, raw.GetSolutionType
            ))
            if not _is_ac_magnetic_solution(solution_type):
                record["reason"] = f"not AC Magnetic: {solution_type!r}"
                continue
            def _setups():
                return raw.GetModule("AnalysisSetup").GetSetups()

            setups = tuple(str(item) for item in (_aedt_call(
                f"{name}-setups", timeout_seconds, _setups
            ) or []))
            if "Setup1" not in setups:
                record["reason"] = "Setup1 absent"
                continue
            marker = _aedt_call(
                f"{name}-solution-marker", timeout_seconds, _solution_marker, raw
            )
            if not marker:
                record["reason"] = "Setup1 : LastAdaptive has no variation"
                continue
            _aedt_call(
                f"activate-{name}", timeout_seconds, project.SetActiveDesign, name
            )
            editor = _aedt_call(
                f"{name}-3d-editor",
                timeout_seconds,
                raw.SetActiveEditor,
                "3D Modeler",
            )
            groups = _aedt_call(
                f"{name}-core-objects", timeout_seconds, _core_objects, editor
            )
            if not groups:
                record["reason"] = "no canonical native core objects"
                continue
            boundary = _aedt_call(
                f"{name}-boundary-module",
                timeout_seconds,
                raw.GetModule,
                "BoundarySetup",
            )
            missing_loss = []
            for object_name in [name for names in groups.values() for name in names]:
                enabled = _aedt_call(
                    f"coreloss-readback-{object_name}",
                    timeout_seconds,
                    boundary.GetCoreLossEffect,
                    object_name,
                )
                if not _aedt_bool(enabled):
                    missing_loss.append(object_name)
            if missing_loss:
                record["reason"] = f"CoreLoss disabled: {missing_loss}"
                continue
            record.update({
                "eligible": True,
                "solution_type": solution_type,
                "solution_marker": list(marker),
                "core_object_count": sum(len(items) for items in groups.values()),
            })
            passing.append((raw, groups, record))
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"

    if preferred_name:
        preferred = [item for item in passing if item[2]["name"] == preferred_name]
        if len(preferred) != 1:
            raise RuntimeError(
                f"requested solved loss design {preferred_name!r} is unavailable; "
                f"inspected={inspected}"
            )
        return preferred[0][0], preferred[0][1], inspected
    if len(passing) != 1:
        raise RuntimeError(
            "expected exactly one solved CoreLoss design; eligible="
            + json.dumps([item[2]["name"] for item in passing])
        )
    return passing[0][0], passing[0][1], inspected


def _register_expression(
    reporter, name: str, builder: Callable, timeout_seconds: float,
) -> str:
    def _register():
        try:
            if reporter.DoesNamedExpressionExists(name):
                return name
        except Exception:
            pass
        reporter.CalcStack("clear")
        builder(reporter)
        result = reporter.AddNamedExpression(name, "Fields")
        if result is False:
            raise RuntimeError(f"AddNamedExpression returned False for {name}")
        if not reporter.DoesNamedExpressionExists(name):
            raise RuntimeError(f"named expression readback failed for {name}")
        return name

    return _aedt_call(f"register-{name}", timeout_seconds, _register)


def _b_expression_builder(object_name: str, operation: str) -> Callable:
    def _build(reporter):
        reporter.EnterQty("B")
        reporter.CalcOp("CmplxMag")
        reporter.CalcOp("Mag")
        reporter.EnterVol(object_name)
        reporter.CalcOp(operation)
    return _build


def _loss_expression_builder(object_names: list[str]) -> Callable:
    def _build(reporter):
        for index, object_name in enumerate(object_names):
            reporter.EnterQty("CoreLoss")
            reporter.EnterVol(object_name)
            reporter.CalcOp("Integrate")
            if index:
                reporter.CalcOp("+")
    return _build


UNIT_FACTORS = {
    "T": ("B", 1.0), "mT": ("B", 1.0e-3), "uT": ("B", 1.0e-6),
    "W": ("P", 1.0), "mW": ("P", 1.0e-3), "kW": ("P", 1.0e3),
    "V": ("V", 1.0), "mV": ("V", 1.0e-3), "kV": ("V", 1.0e3),
    "Wb": ("F", 1.0), "mWb": ("F", 1.0e-3), "uWb": ("F", 1.0e-6),
}


def _convert_unit(value: float, source: Any, target: str) -> float:
    source_unit = str(source or target).strip().replace("µ", "u").replace("μ", "u")
    target_unit = str(target).strip()
    aliases = {"tesla": "T", "watt": "W", "volt": "V", "weber": "Wb"}
    source_unit = aliases.get(source_unit.lower(), source_unit)
    if not source_unit or source_unit == target_unit:
        return value
    source_info = UNIT_FACTORS.get(source_unit)
    target_info = UNIT_FACTORS.get(target_unit)
    if not source_info or not target_info or source_info[0] != target_info[0]:
        raise RuntimeError(
            f"cannot convert AEDT result unit {source_unit!r} to {target_unit!r}"
        )
    return value * source_info[1] / target_info[1]


def _solution_values(solution, expressions: list[str], targets: Mapping[str, str]) -> dict:
    if solution is None or solution is False:
        raise RuntimeError("AEDT returned no solution data")
    units = getattr(solution, "units_data", {}) or {}
    values = {}
    for expression in expressions:
        if hasattr(solution, "get_expression_data"):
            _sweep, data = solution.get_expression_data(expression, formula="real")
        else:
            data = solution.data_real(expression)
        if data is None or data is False or len(data) == 0:
            raise RuntimeError(f"AEDT returned no value for {expression}")
        number = _number(data[0])
        if number is None:
            raise RuntimeError(f"AEDT returned non-finite value for {expression}")
        values[expression] = _convert_unit(
            number, units.get(expression, ""), targets[expression]
        )
    return values


def _query_fields(
    wrapped_design, expressions: list[str], targets: Mapping[str, str],
    timeout_seconds: float,
) -> dict:
    post = _aedt_call(
        "bind-fields-postprocessor",
        timeout_seconds,
        lambda: wrapped_design.post,
    )
    if callable(post) and not hasattr(post, "get_solution_data_per_variation"):
        post = post()

    def _query():
        return post.get_solution_data_per_variation(
            solution_type="Fields",
            setup_sweep_name="Setup1 : LastAdaptive",
            context=[],
            sweeps={"Freq": ["All"], "Phase": ["0deg"]},
            expressions=expressions,
        )

    solution = _aedt_call(
        "query-fields-solution-data", timeout_seconds, _query
    )
    return _aedt_call(
        "read-fields-solution-values",
        timeout_seconds,
        _solution_values,
        solution,
        expressions,
        targets,
    )


def _query_ac_magnetic(
    wrapped_design, expressions: list[str], targets: Mapping[str, str],
    timeout_seconds: float,
) -> dict:
    post = _aedt_call(
        "bind-ac-postprocessor",
        timeout_seconds,
        lambda: wrapped_design.post,
    )
    if callable(post) and not hasattr(post, "get_solution_data"):
        post = post()

    def _query():
        return post.get_solution_data(
            expressions=expressions,
            setup_sweep_name="Setup1 : LastAdaptive",
            report_category="AC Magnetic",
            context=None,
        )

    solution = _aedt_call("query-ac-magnetic-solution-data", timeout_seconds, _query)
    return _aedt_call(
        "read-ac-solution-values",
        timeout_seconds,
        _solution_values,
        solution,
        expressions,
        targets,
    )


def _raw_material_properties(material_manager, material_name: str) -> dict:
    from ansys.aedt.core.generic.data_handlers import _arg2dict

    raw = list(material_manager.GetData(material_name))
    parsed = {}
    _arg2dict(raw, parsed)
    if len(parsed) != 1:
        raise RuntimeError(
            f"unexpected native material payload for {material_name!r}"
        )
    properties = next(iter(parsed.values()))
    if not isinstance(properties, dict) or not properties:
        raise RuntimeError(f"empty native material payload for {material_name!r}")
    return properties


def _symmetry_factors(object_name: str, params: Mapping[str, Any]) -> tuple[int, float, float]:
    from module.input_parameter_260706 import sym_cut_count
    import pandas as pd

    if bool(int(params["full_model"])):
        return 0, 1.0, 1.0
    # sym_cut_count only needs raw input columns; preserve the established helper.
    frame = pd.DataFrame([dict(params)])
    cut_count = int(sym_cut_count(object_name, frame))
    mirror_factor = 1.0 if cut_count == 3 else 2.0
    # Core volume is summed only over retained y>=0 groups.  A group wholly on
    # that side has c=2 (x/z) and mirror_factor=2 to restore its deleted y twin;
    # an odd central group has c=3 (x/y/z) and no distinct twin.  Both paths
    # restore the retained-volume basis by 8 to the complete physical core.
    geometry_factor = (2.0 ** cut_count) * mirror_factor
    loss_amplitude_factor = (
        2.0 ** float(params["core_y"])
        if bool(int(params["loss_sym_on"])) else 1.0
    )
    core_loss_factor = geometry_factor / loss_amplitude_factor
    return cut_count, core_loss_factor, geometry_factor


def _material_and_geometry_readback(
    native_project,
    raw_design,
    editor,
    core_groups: Mapping[int, list[str]],
    params: Mapping[str, Any],
    kf: float,
    volumes_model_units3: Mapping[str, float],
    timeout_seconds: float,
) -> tuple[dict, dict]:
    from module.core_material_contract import (
        LEG_STACKING_DIRECTION, YOKE_STACKING_DIRECTION,
        validate_native_lamination_readback,
    )

    assigned = {}
    orientations = {}
    for object_name in [name for names in core_groups.values() for name in names]:
        value = _aedt_call(
            f"material-assignment-{object_name}",
            timeout_seconds,
            editor.GetPropertyValue,
            "Geometry3DAttributeTab",
            object_name,
            "Material",
        )
        assigned[object_name] = str(value).strip().strip('"')
        orientation = _aedt_call(
            f"orientation-{object_name}",
            timeout_seconds,
            editor.GetPropertyValue,
            "Geometry3DAttributeTab",
            object_name,
            "Orientation",
        )
        orientations[object_name] = str(orientation).strip().strip('"')
    leg_materials = sorted({
        assigned[name] for name in assigned if "_leg_" in name
    })
    yoke_materials = sorted({
        assigned[name] for name in assigned if "_yoke_" in name
    })
    material_details = {
        "leg_material_names": leg_materials,
        "yoke_material_names": yoke_materials,
        "object_orientations": orientations,
        "readbacks": {},
        "errors": [],
    }
    exact = len(leg_materials) == 1 and len(yoke_materials) == 1
    if not exact:
        material_details["errors"].append(
            "core pieces do not use exactly one leg and one yoke material"
        )
    non_global = sorted(
        name for name, orientation in orientations.items()
        if orientation != "Global"
    )
    if non_global:
        exact = False
        material_details["errors"].append(
            f"core pieces do not use Global orientation: {non_global}"
        )
    if exact:
        def _material_manager():
            definition_manager = native_project.GetDefinitionManager()
            return definition_manager.GetManager("Material")

        manager = _aedt_call(
            "native-material-manager",
            timeout_seconds,
            _material_manager,
        )
        for region, name, direction in (
            ("leg", leg_materials[0], LEG_STACKING_DIRECTION),
            ("yoke", yoke_materials[0], YOKE_STACKING_DIRECTION),
        ):
            try:
                props = _aedt_call(
                    f"native-material-data-{region}",
                    timeout_seconds,
                    _raw_material_properties,
                    manager,
                    name,
                )
                material_details["readbacks"][region] = (
                    validate_native_lamination_readback(
                        props,
                        lamination_factor=kf,
                        stacking_direction=direction,
                        cm_base=float(params["core_cm"]),
                        core_x=float(params["core_x"]),
                        core_y=float(params["core_y"]),
                    )
                )
            except Exception as error:
                exact = False
                material_details["errors"].append(
                    f"{region}: {type(error).__name__}: {error}"
                )
    material_details["exact"] = exact

    model_units = str(_aedt_call(
        "model-units", timeout_seconds, editor.GetModelUnits
    )).strip().lower()
    length_to_m = {"mm": 1.0e-3, "cm": 1.0e-2, "m": 1.0}.get(model_units)
    if length_to_m is None:
        raise RuntimeError(f"unsupported saved-project model units: {model_units!r}")
    total_volume_m3 = 0.0
    center_volume_m3 = 0.0
    object_geometry = {}
    for object_name, retained_volume in volumes_model_units3.items():
        cut_count, _loss_factor, geometry_factor = _symmetry_factors(
            object_name, params
        )
        physical_volume_m3 = retained_volume * length_to_m ** 3 * geometry_factor
        total_volume_m3 += physical_volume_m3
        if object_name.endswith("_leg_center"):
            center_volume_m3 += physical_volume_m3
        object_geometry[object_name] = {
            "retained_volume_model_units3": retained_volume,
            "symmetry_cut_count": cut_count,
            "full_geometry_restore_factor": geometry_factor,
            "physical_volume_m3": physical_volume_m3,
        }
    actual_area_m2 = center_volume_m3 / (float(params["h1"]) * 1.0e-3)
    actual_mass_kg = (
        total_volume_m3 * kf * float(params["core_mass_density_kg_m3"])
    )
    expected = {
        "area_m2": float(params["Ae_gross_m2"]),
        "volume_m3": float(params["core_vol_gross_m3"]),
        "mass_kg": float(params["core_mass_effective_kg"]),
    }
    actual = {
        "area_m2": actual_area_m2,
        "volume_m3": total_volume_m3,
        "mass_kg": actual_mass_kg,
    }
    errors = {
        key: relative_error(actual[f"{key}_m2"], expected[f"{key}_m2"])
        if key == "area" else
        relative_error(actual[f"{key}_m3"], expected[f"{key}_m3"])
        if key == "volume" else
        relative_error(actual[f"{key}_kg"], expected[f"{key}_kg"])
        for key in ("area", "volume", "mass")
    }
    geometry = {
        "model_units": model_units,
        "expected": expected,
        "actual": actual,
        "relative_errors": errors,
        "objects": object_geometry,
    }
    return material_details, geometry


def _extract_saved_solution(
    project_path: Path,
    params: Mapping[str, Any],
    kf: float,
    *,
    design_name: str | None,
    timeout_seconds: float,
) -> dict:
    from ansys.aedt.core import settings
    from pyaedt_module.core import pyDesktop

    settings.skip_license_check = True
    settings.wait_for_license = False
    desktop = None
    project = None
    project_name = None
    result = None
    baseline_descendants = _snapshot_descendants()
    cleanup = {
        "errors": [],
        "forced_descendant_cleanup_count": 0,
        "attested": False,
    }
    try:
        desktop = _aedt_call(
            "start-desktop-2025.2",
            timeout_seconds,
            pyDesktop,
            "2025.2",
            True,
            True,
            True,
        )
        project = _aedt_call(
            "open-saved-project",
            timeout_seconds,
            desktop.load_project,
            str(project_path),
        )
        project_name = str(_aedt_call(
            "project-name", timeout_seconds, project.GetName
        ))
        raw_design, core_groups, inspected = _select_solved_loss_design(
            project, design_name, timeout_seconds
        )
        selected_design_name = _design_name(raw_design)
        _aedt_call(
            f"activate-selected-{selected_design_name}",
            timeout_seconds,
            project.SetActiveDesign,
            selected_design_name,
        )
        editor = _aedt_call(
            "selected-3d-editor",
            timeout_seconds,
            raw_design.SetActiveEditor,
            "3D Modeler",
        )
        _validate_core_topology(core_groups, params)
        loss_sym = bool(int(params["loss_sym_on"])) and not bool(
            int(params["full_model"])
        )

        variable_checks = {}
        for key in PROJECT_VARIABLE_KEYS:
            actual_text = _aedt_call(
                f"project-variable-{key}",
                timeout_seconds,
                raw_design.GetVariableValue,
                key,
            )
            actual = _parse_aedt_number(actual_text, key)
            expected = float(params[key])
            error = relative_error(actual, expected)
            passed = error is not None and error <= 1.0e-12
            variable_checks[key] = {
                "project_value": actual,
                "sidecar_value": expected,
                "relative_error": error,
                "passed": passed,
            }
            if not passed:
                raise RuntimeError(
                    f"project/sidecar variable mismatch for {key}: "
                    f"project={actual_text!r}, sidecar={expected!r}"
                )

        volumes = {}
        for object_name in [name for names in core_groups.values() for name in names]:
            volume = _aedt_call(
                f"object-volume-{object_name}",
                timeout_seconds,
                editor.GetObjectVolume,
                object_name,
            )
            number = _parse_aedt_number(volume, f"volume of {object_name}")
            if number <= 0:
                raise RuntimeError(f"non-positive volume for {object_name}: {number}")
            volumes[object_name] = number

        reporter = _aedt_call(
            "fields-reporter",
            timeout_seconds,
            raw_design.GetModule,
            "FieldsReporter",
        )
        field_expressions = []
        targets = {}
        b_mean_names = {}
        b_max_names = {}
        loss_names = {}
        expression_prefix = f"savedab_{uuid.uuid4().hex[:10]}"
        for group_index, object_names in core_groups.items():
            loss_name = f"{expression_prefix}_P_core_{group_index}"
            _register_expression(
                reporter,
                loss_name,
                _loss_expression_builder(object_names),
                timeout_seconds,
            )
            field_expressions.append(loss_name)
            targets[loss_name] = "W"
            loss_names[group_index] = loss_name
            for object_name in object_names:
                mean_name = f"{expression_prefix}_Bm_{object_name}"
                max_name = f"{expression_prefix}_Bx_{object_name}"
                _register_expression(
                    reporter,
                    mean_name,
                    _b_expression_builder(object_name, "Mean"),
                    timeout_seconds,
                )
                _register_expression(
                    reporter,
                    max_name,
                    _b_expression_builder(object_name, "Maximum"),
                    timeout_seconds,
                )
                field_expressions.extend((mean_name, max_name))
                targets[mean_name] = "T"
                targets[max_name] = "T"
                b_mean_names[object_name] = mean_name
                b_max_names[object_name] = max_name

        solution_type = str(_aedt_call(
            "selected-solution-type",
            timeout_seconds,
            raw_design.GetSolutionType,
        ))
        wrapped_design = _aedt_call(
            "attach-pyaedt-solved-design",
            timeout_seconds,
            project.create_design,
            selected_design_name,
            "maxwell3d",
            solution_type,
        )
        field_values = _query_fields(
            wrapped_design, field_expressions, targets, timeout_seconds
        )
        ac_expressions = [
            "mag(InducedVoltage(Tx_winding))",
            "mag(FluxLinkage(Tx_winding))",
        ]
        ac_values = _query_ac_magnetic(
            wrapped_design,
            ac_expressions,
            {ac_expressions[0]: "V", ac_expressions[1]: "Wb"},
            timeout_seconds,
        )

        native_project = getattr(project, "project", project)
        material, geometry = _material_and_geometry_readback(
            native_project,
            raw_design,
            editor,
            core_groups,
            params,
            kf,
            volumes,
            timeout_seconds,
        )

        b_amplitude_factor = 0.5 if loss_sym else 1.0
        flux_amplitude_factor = 2.0 if loss_sym else 1.0
        weighted_b = 0.0
        physical_volume = 0.0
        b_max_pack = 0.0
        per_object_b = {}
        for object_name, volume_record in geometry["objects"].items():
            pack_mean = field_values[b_mean_names[object_name]] * b_amplitude_factor
            pack_max = field_values[b_max_names[object_name]] * b_amplitude_factor
            weight = volume_record["physical_volume_m3"]
            weighted_b += pack_mean * weight
            physical_volume += weight
            b_max_pack = max(b_max_pack, pack_max)
            per_object_b[object_name] = {
                "reported_mean_T": field_values[b_mean_names[object_name]],
                "reported_max_T": field_values[b_max_names[object_name]],
                "physical_pack_mean_T": pack_mean,
                "physical_pack_max_T": pack_max,
                "physical_material_mean_T": pack_mean / kf,
                "physical_material_max_T": pack_max / kf,
            }
        if physical_volume <= 0:
            raise RuntimeError("zero physical core volume during B averaging")
        b_pack_mean = weighted_b / physical_volume

        native_loss = 0.0
        per_group_loss = {}
        for group_index, object_names in core_groups.items():
            cuts = {
                _symmetry_factors(name, params)[0] for name in object_names
            }
            if len(cuts) != 1:
                raise RuntimeError(
                    f"core group {group_index} spans symmetry cuts {sorted(cuts)}"
                )
            cut_count, loss_factor, _geometry_factor = _symmetry_factors(
                object_names[0], params
            )
            reported = field_values[loss_names[group_index]]
            physical = reported * loss_factor
            native_loss += physical
            per_group_loss[str(group_index)] = {
                "reported_native_CoreLoss_W": reported,
                "symmetry_cut_count": cut_count,
                "full_raw_loss_restore_factor": loss_factor,
                "physical_native_raw_CoreLoss_W": physical,
            }

        induced_reported = ac_values[ac_expressions[0]]
        linkage_reported = ac_values[ac_expressions[1]]
        observed = {
            "material_readback_exact": bool(material["exact"]),
            "geometry_relative_errors": geometry["relative_errors"],
            "native_B_pack_mean_T": b_pack_mean,
            "native_B_material_mean_T": b_pack_mean / kf,
            "native_B_pack_max_T": b_max_pack,
            "native_B_material_max_T": b_max_pack / kf,
            "native_core_loss_raw_W": native_loss,
            "induced_voltage_peak_V": induced_reported * flux_amplitude_factor,
            "flux_linkage_peak_Wb_turn": linkage_reported * flux_amplitude_factor,
            "thermal_required": bool(int(params.get("thermal_on", 0))),
        }
        result = {
            "project_name": project_name,
            "design_name": selected_design_name,
            "designs_inspected": inspected,
            "solution_type": solution_type,
            "analyze_dispatched": False,
            "extraction_operator_B": "CmplxMag_then_Mag",
            "extraction_operator_CoreLoss": "native_CoreLoss_volume_integral",
            "named_expression_run_prefix": expression_prefix,
            "loss_symmetry_mode": loss_sym,
            "B_report_to_physical_amplitude_factor": b_amplitude_factor,
            "flux_report_to_physical_amplitude_factor": flux_amplitude_factor,
            "project_variable_checks": variable_checks,
            "core_groups": {str(key): value for key, value in core_groups.items()},
            "material_readback": material,
            "geometry_readback": geometry,
            "per_object_B": per_object_b,
            "per_group_native_CoreLoss": per_group_loss,
            "reported_induced_voltage_peak_V": induced_reported,
            "reported_flux_linkage_peak_Wb_turn": linkage_reported,
            "observed": observed,
            "cleanup": cleanup,
        }
    finally:
        captured_descendants = _snapshot_descendants()
        if desktop is not None and project_name:
            try:
                closed = _aedt_call(
                    "close-project-without-save",
                    timeout_seconds,
                    desktop.odesktop.CloseProject,
                    project_name,
                )
                if closed is False:
                    raise RuntimeError("CloseProject returned False")
            except Exception as error:
                cleanup["errors"].append(
                    f"close project: {type(error).__name__}: {error}"
                )
        if desktop is not None:
            try:
                released = _aedt_call(
                    "release-desktop",
                    timeout_seconds,
                    desktop.release_desktop,
                    True,
                    True,
                )
                if released is False:
                    raise RuntimeError("release_desktop returned False")
            except Exception as error:
                cleanup["errors"].append(
                    f"release desktop: {type(error).__name__}: {error}"
                )
        captured_descendants.update(_snapshot_descendants())
        try:
            cleanup["forced_descendant_cleanup_count"] = (
                _terminate_new_descendants(
                    baseline_descendants, captured_descendants
                )
            )
        except Exception as error:
            cleanup["errors"].append(
                f"descendant cleanup: {type(error).__name__}: {error}"
            )
        cleanup["attested"] = not cleanup["errors"]

    if cleanup["errors"]:
        raise RuntimeError("AEDT cleanup failed: " + "; ".join(cleanup["errors"]))
    if result is None:
        raise RuntimeError("saved solution extraction produced no result")
    return result


def _load_gate_spec(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"gate specification is not an object: {path}")
    return value


def run_saved_gate(args) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "started_at": started,
        "status": "running",
        "passed": False,
        "pass_scope": "saved_case_applicable_numerical_metrics_only",
        "analyze_dispatched": False,
        "requested_project_path": str(Path(args.project_path).expanduser()),
        "requested_kf": _number(args.kf),
        "aedt_version": "2025.2",
        "op_timeout_seconds": _number(args.op_timeout_seconds),
    }
    stage = "validate_cli"
    try:
        kf = float(args.kf)
        if not math.isfinite(kf) or not 0 < kf <= 1:
            raise ValueError("--kf must satisfy 0 < kf <= 1")
        stage = "resolve_project"
        project_path, search_root = _resolve_project_path(args.project_path)
        evidence["project_path"] = str(project_path)
        evidence["snapshot_search_root"] = str(search_root)

        stage = "load_gate_spec"
        gate_path = Path(args.gate_spec).expanduser().resolve()
        gate_spec = _load_gate_spec(gate_path)
        evidence["gate_spec_path"] = str(gate_path)
        evidence["gate_spec_schema"] = gate_spec.get("schema")
        evidence["physics_data_revision"] = gate_spec.get("physics_data_revision")

        stage = "load_parameters"
        params, param_provenance, _param_path = _load_parameters(
            args.params_path, search_root, project_path, kf
        )
        evidence["parameter_provenance"] = param_provenance
        evidence["parameters"] = {
            key: params[key] for key in (
                "V1_rms", "freq", "N1_main", "N1_side", "N1", "l1", "l2",
                "h1", "w1", "n_core_group", "core_lamination_factor",
                "core_loss_margin", "core_cm", "core_x", "core_y",
                "Ae_gross_m2", "Ae_effective_m2", "core_vol_gross_m3",
                "core_mass_effective_kg", "full_model", "loss_sym_on",
                "thermal_on",
            )
        }

        stage = "compute_python_reference"
        from module.core_material_contract import faraday_lumped_core_reference

        reference = faraday_lumped_core_reference(
            voltage_rms_v=float(params["V1_rms"]),
            frequency_hz=float(params["freq"]),
            turns=int(params["N1_main"]) + int(params["N1_side"]),
            gross_area_m2=float(params["Ae_gross_m2"]),
            lamination_factor=kf,
            effective_mass_kg=float(params["core_mass_effective_kg"]),
            loss_margin=float(params["core_loss_margin"]),
            coefficient=6.5,
            x=float(params["core_x"]),
            y=float(params["core_y"]),
        )
        reference.update({
            "frequency_hz": float(params["freq"]),
            "source_voltage_peak_V": math.sqrt(2.0) * float(params["V1_rms"]),
        })
        evidence["python_reference"] = reference

        stage = "extract_saved_aedt_solution"
        extraction = _extract_saved_solution(
            project_path,
            params,
            kf,
            design_name=args.design,
            timeout_seconds=float(args.op_timeout_seconds),
        )
        evidence["aedt_extraction"] = extraction

        stage = "evaluate_numerical_gates"
        evaluation = evaluate_numerical_gates(
            extraction["observed"], reference, gate_spec, kf
        )
        evidence["gate_evaluation"] = evaluation
        evidence["saved_em_case_passed"] = bool(evaluation["passed"])
        evidence["full_gate_passed"] = False
        evidence["coverage"] = {
            "saved_case_applicable_numerical_gate_complete": True,
            "all_gate_spec_metrics_available": all(
                metric["available"]
                for metric in evaluation["metrics"].values()
            ),
            "full_gate_spec_complete": False,
            "full_gate_spec_reason": (
                "one saved project supplies one EM execution mode; the gate spec "
                "also requires the complementary full/eighth mode, all kf cases, "
                "and thermal native-power evidence"
            ),
            "execution_mode": (
                "eighth_current_driven_loss_sym"
                if extraction["loss_symmetry_mode"]
                else "full_saved_loss_design"
            ),
        }
        evidence["passed"] = bool(evaluation["passed"])
        evidence["status"] = (
            "passed_saved_em_case_full_gate_incomplete"
            if evidence["passed"] else "failed_saved_em_case_gate"
        )
    except Exception as error:
        evidence.update({
            "status": "error",
            "passed": False,
            "failure": {
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        })
    evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
    return evidence


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Postprocess a saved solved 1K101 Maxwell project without Analyze"
    )
    parser.add_argument(
        "--project-path", required=True,
        help="saved .aedt file or snapshot directory containing exactly one project",
    )
    parser.add_argument("--kf", required=True, type=float)
    parser.add_argument("--out", required=True, help="machine-readable evidence JSON")
    parser.add_argument(
        "--params-path", default=None,
        help="scheduler cand.json; auto-discovered inside the snapshot when omitted",
    )
    parser.add_argument(
        "--design", default=None,
        help="solved loss design name; auto-selected when exactly one is eligible",
    )
    parser.add_argument("--gate-spec", default=str(DEFAULT_GATE_SPEC))
    parser.add_argument("--op-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.kf) or not 0 < args.kf <= 1:
        parser.error("--kf must satisfy 0 < kf <= 1")
    if (
        not math.isfinite(args.op_timeout_seconds)
        or args.op_timeout_seconds <= 0
    ):
        parser.error("--op-timeout-seconds must be finite and positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    evidence = run_saved_gate(args)
    try:
        _atomic_json(Path(args.out), evidence)
    except Exception as error:
        evidence.update({
            "status": "error",
            "passed": False,
            "output_failure": {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        })
    compact = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    print("===RESULT_JSON===", flush=True)
    print(json.dumps(
        evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ), flush=True)
    print("===END_RESULT_JSON===", flush=True)
    # Existing scheduler collectors scan this single-line compatibility form.
    print("RESULT_JSON " + compact, flush=True)
    return 0 if evidence.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
