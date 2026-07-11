"""
설계도면260706.pdf 반영 MFT 시뮬레이션 스크립트 (run_simulation_260514.py 기반)

도면 대비 추가/변경 사항:
  1. 코어 y방향 3분할 + 콜드플레이트(20T, 알루미늄) 4장
  2. 1차 권선 냉각 플레이트(20T): 턴1-2 사이, 턴(N-1)-N 사이 (y측면만, 양측 대칭)
  3. 권선 모서리 라운드 처리 on/off (반경은 안쪽 턴 기준 파라미터)
  4. 파라미터 직접 입력 모드 (--fixed / --params) + 기존 랜덤 스윕 모드

실행 예:
  python run_simulation_260706.py --fixed                  # 도면 치수 1회 (라운드 off)
  python run_simulation_260706.py --fixed --round          # 라운드 on
  python run_simulation_260706.py --fixed --params my.json # 일부 값 변경
  python run_simulation_260706.py --fixed --model-only     # 모델링만 하고 해석 생략
  python run_simulation_260706.py --fixed --full           # 대칭 미적용 풀모델
  python run_simulation_260706.py                          # 랜덤 스윕 (무한루프)
"""

import sys
import traceback
import logging
import portalocker
import os
import re
import json
import argparse
import uuid
import tempfile

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# 경로 설정 - 플랫폼에 따라 다르게 처리
library_override = os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", "").strip()
if library_override and os.path.basename(os.path.normpath(library_override)).lower() != "src":
    library_override = os.path.join(library_override, "src")
possible_paths = [library_override] if library_override else []
if os.name == 'nt':  # Windows
    possible_paths.append(r"Y:/git/pyaedt_library/src/")
else:  # Linux/Unix
    possible_paths += [
        r"../pyaedt_library/src/",
        os.path.abspath(os.path.join(BASE_DIR, "../git/pyaedt_library/src/")),
        "/home1/r1jae262/jupyter/git/pyaedt_library/src/",
        "/home1/dhj02/NEC/git/pyaedt_library/src/",
        "/home1/dw16/NEC/git/pyaedt_library/src/",
        "/home1/harry261/NEC/git/pyaedt_library/src/",
        "/home1/hmlee31/NEC/git/pyaedt_library/src/",
        "/home1/jji0930/NEC/git/pyaedt_library/src/",
        "/home1/wjddn5916/NEC/git/pyaedt_library/src/"
    ]
PYAEDT_LIBRARY_SRC = ""
for path in possible_paths:
    if path and os.path.isdir(path):
        PYAEDT_LIBRARY_SRC = os.path.abspath(path)
        sys.path.insert(0, PYAEDT_LIBRARY_SRC)
        break


# FlexNet 클라이언트 타임아웃 상향 (기본 0.1초): 바쁜 라이선스 데몬(lmgrd)의 느린 응답을
# 연결 리셋으로 판정하지 않게 함. AEDT 기동 전에 설정돼야 하므로 임포트 시점에 적용.
os.environ.setdefault("FLEXLM_TIMEOUT", "3000000")

import pyaedt_module
from pyaedt_module.core import pyDesktop
import os
import time
from datetime import datetime

import math
import copy

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import platform
import csv

from module.input_parameter_260706 import (
    create_input_parameter,
    set_design_variables,
    validation_check,
    get_tx_y_gaps,
    get_drawing_default_params,
    sym_cut_count,
)
from module.modeling_260706 import (
    create_core,
    create_coil,
    create_winding_cooling_plates,
    create_coil_section,
)

from ansys.aedt.core import settings

settings.skip_license_check = True
settings.wait_for_license = False

if os.name == 'nt':  # Windows
    GUI = False
else:  # Linux/Unix
    GUI = True

from filelock import FileLock
import shutil
from module.source_contract import SOLVER_REVISION_PATHS


PLATE_COLOR = [144, 190, 144]
PAD_COLOR = [200, 160, 200]


def _git_provenance():
    """Return the full solver revision and tracked-worktree dirty flag."""
    try:
        import subprocess
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
            stderr=subprocess.DEVNULL, text=True).strip().lower()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--",
             *SOLVER_REVISION_PATHS],
            cwd=BASE_DIR, stderr=subprocess.DEVNULL, text=True).strip())
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"invalid solver revision {revision!r}")
        return revision, int(dirty)
    except Exception:
        return "unknown", 1


GIT_HASH, GIT_DIRTY = _git_provenance()


def _library_git_provenance():
    """Return the imported pyaedt_library full revision and tracked-src dirty flag."""
    try:
        import subprocess
        root = os.path.abspath(os.path.join(PYAEDT_LIBRARY_SRC, os.pardir))
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL, text=True).strip().lower()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
            cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"invalid library revision {revision!r}")
        return revision, int(dirty)
    except Exception:
        return "unknown", 1


PYAEDT_LIBRARY_GIT_HASH, PYAEDT_LIBRARY_GIT_DIRTY = _library_git_provenance()


class SolutionDataUnavailableError(RuntimeError):
    """Legacy extraction error retained for compatibility with external callers."""


class _AedtIdentityMismatch(RuntimeError):
    """A fresh native AEDT handle resolved to a project or design we did not request."""


_SOLUTION_UNIT_FACTORS = {
    "fa": 1e-15, "pa": 1e-12, "na": 1e-9, "ua": 1e-6,
    "ma": 1e-3, "a": 1.0, "ka": 1e3,
    "ph": 1e-12, "nh": 1e-9, "uh": 1e-6, "mh": 1e-3, "h": 1.0,
    "nw": 1e-9, "uw": 1e-6, "mw": 1e-3, "w": 1.0, "kw": 1e3, "megaw": 1e6,
    "ut": 1e-6, "mt": 1e-3, "t": 1.0, "tesla": 1.0,
    "rad": 180.0 / math.pi, "deg": 1.0,
}


def _convert_solution_unit(value, source_unit, target_unit):
    """Convert a scalar returned by SolutionData while tolerating omitted AEDT units."""
    source_raw = str(source_unit or "").strip().replace("µ", "u").replace("μ", "u")
    target_raw = str(target_unit or "").strip().replace("µ", "u").replace("μ", "u")
    source = source_raw.lower()
    target = target_raw.lower()
    if not source or not target or source_raw == target_raw:
        return float(value)
    # Lower-casing would make megawatts (MW) indistinguishable from milliwatts (mW).
    source_factor = 1e6 if source_raw == "MW" else _SOLUTION_UNIT_FACTORS.get(source)
    target_factor = 1e6 if target_raw == "MW" else _SOLUTION_UNIT_FACTORS.get(target)
    if source_factor is None or target_factor is None:
        logging.warning(f"unknown SolutionData unit conversion '{source_unit}' -> '{target_unit}'")
        return float(value)
    return float(value) * source_factor / target_factor


_RL_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def _parse_rl_matrix_export(text, frequency_hz, tx_name="Tx_winding", rx_name="Rx_winding"):
    """Parse one validated 2x2 R/L block exported by Maxwell ExportSolnData."""
    unit_match = re.search(r"(?im)^Inductance Unit:\s*([^\s]+)\s*$", text)
    if not unit_match:
        raise RuntimeError("RL matrix export has no inductance unit")
    source_unit = unit_match.group(1)
    if source_unit.lower() not in {"ph", "nh", "uh", "mh", "h"}:
        raise RuntimeError(f"RL matrix export has unsupported inductance unit: {source_unit}")
    lines = text.splitlines()
    target_index = None
    for index, line in enumerate(lines):
        match = re.fullmatch(rf"\s*({_RL_NUMBER})Hz\s*", line)
        if match and math.isclose(
                float(match.group(1)), float(frequency_hz), rel_tol=1e-9, abs_tol=1e-9):
            target_index = index
            break
    if target_index is None:
        raise RuntimeError(f"RL matrix export has no {float(frequency_hz):g}Hz block")

    block_end = next(
        (index for index in range(target_index + 1, len(lines))
         if re.fullmatch(rf"\s*{_RL_NUMBER}Hz\s*", lines[index])),
        len(lines),
    )
    rl_index = next(
        (index for index in range(target_index + 1, block_end)
         if lines[index].strip() == "R,L"),
        None,
    )
    if rl_index is None:
        raise RuntimeError("RL matrix export has no R,L section")

    pair_pattern = re.compile(rf"({_RL_NUMBER})\s*,\s*({_RL_NUMBER})")
    rows = {}
    for line in lines[rl_index + 1:block_end]:
        stripped = line.strip()
        name = stripped.split(None, 1)[0] if stripped else ""
        if name not in (tx_name, rx_name):
            continue
        pairs = [(float(r), float(l)) for r, l in pair_pattern.findall(stripped)]
        if len(pairs) == 2:
            rows[name] = pairs
        if len(rows) == 2:
            break
    if set(rows) != {tx_name, rx_name}:
        raise RuntimeError("RL matrix export is missing Tx/Rx matrix rows")

    ltx_raw = rows[tx_name][0][1]
    l12_raw = rows[tx_name][1][1]
    l21_raw = rows[rx_name][0][1]
    lrx_raw = rows[rx_name][1][1]
    symmetry_scale = max(abs(l12_raw), abs(l21_raw), 1.0)
    if abs(l12_raw - l21_raw) > 1e-9 * symmetry_scale:
        raise RuntimeError("RL inductance matrix is not symmetric")
    mutual_raw = 0.5 * (l12_raw + l21_raw)
    if not all(math.isfinite(value) for value in (ltx_raw, lrx_raw, mutual_raw)):
        raise RuntimeError("RL inductance matrix contains non-finite values")
    determinant = ltx_raw * lrx_raw - mutual_raw * mutual_raw
    if ltx_raw <= 0 or lrx_raw <= 0 or determinant <= 0:
        raise RuntimeError("RL inductance matrix is not positive definite")
    k = abs(mutual_raw) / math.sqrt(ltx_raw * lrx_raw)
    if not math.isfinite(k) or k > 1.0 + 1e-9:
        raise RuntimeError(f"RL coupling coefficient is invalid: {k}")

    ltx = _convert_solution_unit(ltx_raw, source_unit, "uH")
    lrx = _convert_solution_unit(lrx_raw, source_unit, "uH")
    mutual = abs(_convert_solution_unit(mutual_raw, source_unit, "uH"))
    k2 = min(k, 1.0) ** 2
    return {
        "Ltx": ltx,
        "Lrx": lrx,
        "M": mutual,
        "k": min(k, 1.0),
        "Lmt": ltx * k2,
        "Lmr": lrx * k2,
        "Llt": ltx * (1.0 - k2),
        "Llr": lrx * (1.0 - k2),
        # Matrix-design solid loss is diagnostic only; the production loss
        # design supplies authoritative component losses.
        "Tx_loss": float("nan"),
        "Rx_loss": float("nan"),
    }


MATRIX_REQUIRED_RESULT_COLUMNS = (
    "Ltx", "Lrx", "M", "k", "Lmt", "Lmr", "Llt", "Llr",
)
LOSS_REQUIRED_RESULT_COLUMNS = (
    "P_core_total", "P_core_plate_total", "P_wcp_total",
    "P_winding_total", "B_mean_core", "B_max_core",
)


def _finite_result_value(frame, column):
    """Return one finite numeric result value, or None for a missing/bad value."""
    try:
        value = float(frame[column].iloc[0])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return None
    return value if math.isfinite(value) else None


def _parse_convergence_history(lines, tolerance):
    """Parse a complete AEDT convergence export and count trailing good passes."""
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("invalid convergence tolerance") from error
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise RuntimeError("invalid convergence tolerance")

    if isinstance(lines, str):
        lines = lines.splitlines()

    completed = None
    rows = []
    for raw_line in lines:
        line = str(raw_line).strip()
        completed_match = re.match(r"^Completed\s*:\s*(\d+)\s*$", line, re.IGNORECASE)
        if completed_match:
            declared = int(completed_match.group(1))
            if completed is not None and completed != declared:
                raise RuntimeError("conflicting completed-pass counts")
            completed = declared
            continue

        if not re.match(r"^\d+\s*\|", line):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 5:
            raise RuntimeError(f"malformed convergence row: {line}")
        try:
            pass_index = int(parts[0])
            tetrahedra = float(parts[1].replace(",", ""))
            total_energy = float(parts[2])
            energy_error = float(parts[3])
            delta_token = parts[4].upper()
            delta_energy = (
                None if delta_token in {"N/A", "NA"} else float(parts[4])
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"malformed convergence row: {line}") from error
        numeric_values = [tetrahedra, total_energy, energy_error]
        if delta_energy is not None:
            numeric_values.append(delta_energy)
        if not all(math.isfinite(value) for value in numeric_values):
            raise RuntimeError(f"non-finite convergence row: {line}")
        if pass_index < 1 or tetrahedra <= 0:
            raise RuntimeError(f"invalid convergence row: {line}")
        rows.append((pass_index, tetrahedra, energy_error, delta_energy))

    if completed is None or completed < 1:
        raise RuntimeError("completed-pass count is missing")
    expected_indices = list(range(1, completed + 1))
    actual_indices = [row[0] for row in rows]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"incomplete convergence history: completed={completed}, rows={actual_indices}"
        )

    consecutive = 0
    for _, _, energy_error, delta_energy in reversed(rows):
        if (
            delta_energy is not None
            and 0 <= energy_error <= tolerance
            and 0 <= delta_energy <= tolerance
        ):
            consecutive += 1
        else:
            break

    last = rows[-1]
    return {
        "passes": float(completed),
        "mesh_tets": float(last[1]),
        "error_pct": float(last[2]),
        "delta_pct": (
            float(last[3]) if last[3] is not None else float("nan")
        ),
        "consecutive": float(consecutive),
    }


def _em_result_validation(frame, matrix_on=True, loss_on=True):
    """Validate enabled EM stages against the configured adaptive criteria."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return False, "result frame is missing"

    enabled = []
    if matrix_on:
        enabled.append((
            "matrix", "matrix_percent_error", "matrix_min_converged",
            MATRIX_REQUIRED_RESULT_COLUMNS,
        ))
    if loss_on:
        enabled.append((
            "loss", "percent_error", "min_converged", LOSS_REQUIRED_RESULT_COLUMNS,
        ))
    if not enabled:
        return False, "no EM stage is enabled"

    failures = []
    for label, tolerance_column, minimum_column, required_columns in enabled:
        tolerance = _finite_result_value(frame, tolerance_column)
        minimum_passes = _finite_result_value(frame, minimum_column)
        passes = _finite_result_value(frame, f"conv_passes_{label}")
        consecutive = _finite_result_value(frame, f"conv_consecutive_{label}")
        error = _finite_result_value(frame, f"conv_error_pct_{label}")
        delta = _finite_result_value(frame, f"conv_delta_pct_{label}")
        if tolerance is None or tolerance <= 0:
            failures.append(f"{label}: invalid {tolerance_column}")
        if minimum_passes is None or minimum_passes < 1:
            failures.append(f"{label}: invalid {minimum_column}")
        if passes is None or passes < 1 or not passes.is_integer():
            failures.append(f"{label}: convergence pass count is missing")
        if consecutive is None or consecutive < 0 or not consecutive.is_integer():
            failures.append(f"{label}: consecutive converged pass count is missing")
        elif minimum_passes is not None and consecutive < minimum_passes:
            failures.append(
                f"{label}: consecutive converged pass count {consecutive:g} "
                f"is below {minimum_passes:g}"
            )
        elif passes is not None and consecutive > passes:
            failures.append(f"{label}: inconsistent convergence pass counts")
        if error is None or error < 0:
            failures.append(f"{label}: energy error is missing")
        elif tolerance is not None and tolerance > 0 and error > tolerance:
            failures.append(
                f"{label}: energy error {error:g}% exceeds {tolerance:g}%"
            )
        if delta is None or delta < 0:
            failures.append(f"{label}: delta energy is missing")
        elif tolerance is not None and tolerance > 0 and delta > tolerance:
            failures.append(
                f"{label}: delta energy {delta:g}% exceeds {tolerance:g}%"
            )

        missing_outputs = [
            column for column in required_columns
            if _finite_result_value(frame, column) is None
        ]
        if missing_outputs:
            failures.append(
                f"{label}: non-finite required outputs {missing_outputs}"
            )

    return not failures, "; ".join(failures) if failures else "valid"


def _em_result_is_valid(frame, matrix_on=True, loss_on=True):
    """Return True only for finite, adaptively converged enabled EM stages."""
    return _em_result_validation(frame, matrix_on=matrix_on, loss_on=loss_on)[0]


MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15
MANDATORY_THERMAL_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
)
SIDE_THERMAL_TEMPERATURE_COLUMNS = (
    "T_max_Rx_side",
    "Tprobe_Rx_side_leeward_max",
)


def _thermal_result_is_valid(frame):
    """Return True only when every required thermal group passed extraction."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return False
    try:
        if int(frame["thermal_solved"].iloc[0]) != 1:
            return False
        if int(frame["thermal_convergence_available"].iloc[0]) != 1:
            return False
        if int(frame["thermal_converged"].iloc[0]) != 1:
            return False
        if float(frame["thermal_iterations"].iloc[0]) <= 0:
            return False
        if int(frame["thermal_extraction_complete"].iloc[0]) != 1:
            return False
        if int(frame["thermal_rx_power_balance_ok"].iloc[0]) != 1:
            return False
        if float(frame["thermal_rx_power_balance_group_count"].iloc[0]) < 1:
            return False
        rx_expected = float(frame["thermal_rx_expected_power_w"].iloc[0])
        rx_assigned = float(frame["thermal_rx_assigned_power_w"].iloc[0])
        rx_balance_error = float(frame["thermal_rx_power_balance_max_abs_w"].iloc[0])
        if not (
            str(frame["thermal_rx_model"].iloc[0])
            in {"homogenized_blocks", "hybrid_explicit"}
            and math.isfinite(rx_expected)
            and rx_expected >= 0
            and math.isfinite(rx_assigned)
            and rx_assigned >= 0
            and math.isclose(rx_assigned, rx_expected, rel_tol=1e-12, abs_tol=1e-9)
            and math.isfinite(rx_balance_error)
            and 0 <= rx_balance_error <= 1e-9
        ):
            return False
        flow_limit = float(frame["thermal_residual_flow_limit"].iloc[0])
        energy_limit = float(frame["thermal_residual_energy_limit"].iloc[0])
        flow_residuals = [
            float(frame[column].iloc[0])
            for column in (
                "thermal_residual_continuity",
                "thermal_residual_x_velocity",
                "thermal_residual_y_velocity",
                "thermal_residual_z_velocity",
            )
        ]
        energy_residual = float(frame["thermal_residual_energy"].iloc[0])
        if not (
            math.isfinite(flow_limit)
            and 0 < flow_limit <= 1e-3
            and math.isfinite(energy_limit)
            and 0 < energy_limit <= 1e-7
            and all(math.isfinite(value) and 0 <= value <= flow_limit for value in flow_residuals)
            and math.isfinite(energy_residual)
            and 0 <= energy_residual <= energy_limit
        ):
            return False
        group_bits = {
            "T_max_Tx": 1,
            "T_max_Rx_main": 2,
            "T_max_Rx_side": 4,
            "T_max_core": 8,
        }
        required_mask = int(frame["thermal_required_group_mask"].iloc[0])
        if required_mask & 11 != 11 or required_mask & ~15:
            return False
        required = list(MANDATORY_THERMAL_TEMPERATURE_COLUMNS)
        if required_mask & group_bits["T_max_Rx_side"]:
            required.extend(SIDE_THERMAL_TEMPERATURE_COLUMNS)
        temperatures = [float(frame[column].iloc[0]) for column in required]
        return all(
            math.isfinite(value)
            and MIN_TRUSTED_TEMPERATURE_C < value < MAX_TRUSTED_TEMPERATURE_C
            for value in temperatures
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _thermal_failure_frame(error):
    """Build a harvestable EM row marker for a hard thermal-stage failure."""
    message = str(error).strip() or repr(error)
    return pd.DataFrame({
        "thermal_solved": [0],
        "thermal_convergence_available": [0],
        "thermal_converged": [0],
        "thermal_extraction_complete": [0],
        "thermal_required_missing_count": [4],
        "thermal_required_group_mask": [15],
        "thermal_required_group_count": [4],
        "thermal_rx_model": ["unknown"],
        "thermal_rx_power_balance_ok": [0],
        "thermal_rx_power_balance_group_count": [0],
        "thermal_rx_power_balance_max_abs_w": [float("nan")],
        "thermal_rx_expected_power_w": [float("nan")],
        "thermal_rx_assigned_power_w": [float("nan")],
        "thermal_error_type": [type(error).__name__],
        "thermal_error_message": [message[:2000]],
    })


def _completion_exit_code(successes, requested):
    """A bounded batch succeeds only after every requested valid row completes."""
    if requested is None:
        return 0
    return 0 if successes >= requested else 1


def _project_delete_policy(input_frame, fixed_mode=False, hold=False, model_only=False):
    """Return whether this run's project is disposable, before validation can fail."""
    default_keep = 1 if fixed_mode else 0
    keep_values = input_frame.get("keep_project", pd.Series([default_keep]))
    keep_project = int(keep_values.iloc[0]) != 0
    return not (keep_project or hold or model_only)


def _lightweight_matrix_enabled(sim):
    return int(sim.df_plus["matrix_skin_mesh"].iloc[0]) == 0


def _aedt_bool(value):
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "on", "solid"}:
        return True
    if token in {"false", "0", "no", "off", "stranded"}:
        return False
    raise RuntimeError(f"unrecognized AEDT boolean readback: {value!r}")


def _set_winding_solid_state(winding, is_solid, label):
    """Update a copied winding and fail before solve if AEDT rejects the edit."""
    if winding is None or winding is False:
        raise RuntimeError(f"{label} winding is unavailable")
    props = winding.props
    setter = getattr(props, "_setitem_without_update", None)
    updates = {
        "IsSolid": bool(is_solid),
        "Resistance": "0ohm",
        "Inductance": "0H",
        "Voltage": "0V",
        "ParallelBranchesNum": "1",
    }
    for key, value in updates.items():
        if callable(setter):
            setter(key, value)
        else:
            props[key] = value
    if winding.update() is not True:
        raise RuntimeError(f"failed to set {label} winding IsSolid={is_solid}")
    if _aedt_bool(props.get("IsSolid")) != bool(is_solid):
        raise RuntimeError(f"{label} winding IsSolid readback mismatch")
    if not str(props.get("Resistance", "")).strip().lower().startswith("0"):
        raise RuntimeError(f"{label} winding resistance readback is not zero")


def _configure_em_conductor_mesh(sim, mode):
    """Use no skin operations in lightweight matrix and full detail in loss."""
    if mode == "matrix" and _lightweight_matrix_enabled(sim):
        logging.info(
            "matrix design: stranded windings, plate eddy effects off, "
            "and no skin-depth mesh operations"
        )
        plate_count = sim.assign_plate_settings(
            enable_eddy_effects=False, assign_skin_mesh=False
        )
        sim.matrix_conductor_policy = "stranded_no_eddy_no_skin"
        sim.matrix_winding_stranded_count = 2
        sim.matrix_conductor_mesh_operation_count = 0
        sim.matrix_plate_eddy_off_readback_count = int(plate_count)
        return False
    winding_mesh_count = sim.assign_skin_depth()
    plate_count = sim.assign_plate_settings(
        enable_eddy_effects=True, assign_skin_mesh=True
    )
    if mode == "matrix":
        sim.matrix_conductor_policy = "solid_skin"
        sim.matrix_winding_stranded_count = 0
        sim.matrix_conductor_mesh_operation_count = (
            int(winding_mesh_count) + int(int(plate_count) > 0)
        )
        sim.matrix_plate_eddy_off_readback_count = 0
    return True


def _configure_loss_copy_skin_mesh(sim):
    """Restore precise conductor physics when the matrix used stranded conductors."""
    if not _lightweight_matrix_enabled(sim):
        logging.info("loss copy: reusing inherited solid windings and mesh operations")
        return False
    _set_winding_solid_state(sim.tx_winding, True, "Tx")
    _set_winding_solid_state(sim.rx_winding, True, "Rx")
    winding_mesh_count = sim.assign_skin_depth()
    plate_count = sim.assign_plate_settings(
        enable_eddy_effects=True, assign_skin_mesh=True
    )
    sim.loss_winding_solid_update_count = 2
    sim.loss_winding_mesh_operation_count = int(winding_mesh_count)
    sim.loss_plate_eddy_on_readback_count = int(plate_count)
    sim.loss_conductor_mesh_operation_count = (
        int(winding_mesh_count) + int(int(plate_count) > 0)
    )
    return True


def _normalized_aedt_token(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _is_ac_magnetic_solution(value):
    return _normalized_aedt_token(value) in {"acmagnetic", "eddycurrent"}


_LOSS_SETUP_PROPERTY_KEYS = (
    "Max. Number of Passes",
    "Min. Converged Passes",
    "Percent Error",
)


def _ready_loss_setup_properties(setup):
    if setup is None or setup is False:
        return None
    child = getattr(setup, "_child_object", None)
    if child is None or child is False:
        return None
    properties = getattr(setup, "properties", None)
    if not isinstance(properties, dict):
        return None
    if not all(key in properties for key in _LOSS_SETUP_PROPERTY_KEYS):
        return None
    return properties


def _configure_copied_loss_setup(setup, max_passes, min_converged, percent_error):
    expected = {
        "Max. Number of Passes": int(max_passes),
        "Min. Converged Passes": int(min_converged),
        "Percent Error": float(percent_error),
    }
    properties = _ready_loss_setup_properties(setup)
    if properties is None:
        raise RuntimeError("copied loss Setup1 has no live COM property object")
    for key, value in expected.items():
        properties[key] = value

    readback = _ready_loss_setup_properties(setup)
    mismatches = {}
    if readback is None:
        mismatches["properties"] = "unavailable after update"
    else:
        for key, value in expected.items():
            actual = readback.get(key)
            try:
                matches = float(actual) == float(value)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                mismatches[key] = {"expected": value, "actual": actual}
    if mismatches:
        raise RuntimeError(f"copied loss Setup1 property read-back failed: {mismatches}")
    return setup


def _aedt_design_name(value):
    try:
        value = value.GetName()
    except AttributeError:
        pass
    return str(value or "").split(";")[-1].strip()


def _project_design_entries(project):
    entries = []
    for item in project.GetDesigns() or []:
        entries.append((_aedt_design_name(item), item if hasattr(item, "GetName") else None))
    return [(name, design) for name, design in entries if name]


def _wait_for_ready_copied_loss_design(
        project, before_names, wrapper_factory, timeout_s=60.0, poll_s=0.25,
        clock=time.monotonic, sleeper=time.sleep):
    """Bind a pasted Maxwell design only after its COM and PyAEDT state is stable."""
    deadline = clock() + max(0.0, float(timeout_s))
    target_name = None
    stable_name = None
    stable_signature = None
    stable_count = 0
    last = {
        "new_names": [], "design_type": None, "solution_type": None,
        "setups": [], "wrapper": None,
    }

    while True:
        try:
            entries = _project_design_entries(project)
            new_entries = [(name, raw) for name, raw in entries if name not in before_names]
            last["new_names"] = [name for name, _raw in new_entries]
            if target_name is None and len(new_entries) == 1:
                target_name = new_entries[0][0]
            candidates = [item for item in new_entries if item[0] == target_name]
            for name, raw in candidates:
                if raw is None:
                    raw = project.SetActiveDesign(name)
                design_type = str(raw.GetDesignType() or "")
                solution_type = str(raw.GetSolutionType() or "")
                setups = tuple(str(item) for item in (
                    raw.GetModule("AnalysisSetup").GetSetups() or []
                ))
                last.update({
                    "design_type": design_type,
                    "solution_type": solution_type,
                    "setups": list(setups),
                })
                signature = (name, design_type, solution_type, setups)
                ready = (
                    design_type == "Maxwell 3D"
                    and _is_ac_magnetic_solution(solution_type)
                    and "Setup1" in setups
                )
                if not ready:
                    if stable_name == name:
                        stable_name = None
                        stable_signature = None
                        stable_count = 0
                    continue
                if stable_name == name and stable_signature == signature:
                    stable_count += 1
                else:
                    stable_name = name
                    stable_signature = signature
                    stable_count = 1
                if stable_count < 2:
                    continue

                project.SetActiveDesign(name)
                wrapper = wrapper_factory(name, solution_type)
                wrapper_name = _aedt_design_name(getattr(wrapper, "design_name", ""))
                wrapper_solution = str(getattr(wrapper, "solution_type", "") or "")
                wrapper_exists = wrapper is not None and wrapper is not False
                setup = wrapper.get_setup(name="Setup1") if wrapper_exists else None
                setup_exists = setup is not None and setup is not False
                setup_properties = _ready_loss_setup_properties(setup)
                wrapper_ready = (
                    wrapper_exists
                    and wrapper_name == name
                    and _is_ac_magnetic_solution(wrapper_solution)
                    and setup_exists
                    and setup_properties is not None
                )
                last["wrapper"] = {
                    "name": wrapper_name,
                    "solution_type": wrapper_solution,
                    "setup_ready": bool(wrapper_ready),
                    "setup_properties": (
                        sorted(setup_properties) if setup_properties is not None else []
                    ),
                }
                if wrapper_ready:
                    return wrapper, setup
        except Exception as error:
            last["error"] = f"{type(error).__name__}: {error}"

        now = clock()
        if now >= deadline:
            raise RuntimeError(
                "copied loss design did not become ready within "
                f"{float(timeout_s):g}s; last={last}"
            )
        sleeper(min(max(0.0, float(poll_s)), max(0.0, deadline - now)))


def _named_object_sequence(value):
    """Return object names without allowing an unnameable remap entry."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    names = []
    for item in values:
        if isinstance(item, (list, tuple)):
            names.extend(_named_object_sequence(item))
            continue
        name = item if isinstance(item, str) else getattr(item, "name", None)
        if not name:
            raise RuntimeError(f"copied-object remap contains no name: {item!r}")
        names.append(str(name))
    return names


def _remap_copied_design_objects(old_design, new_design, attributes):
    """Remap copied objects and reject every missing or reordered object."""
    for attribute in attributes:
        if not hasattr(old_design, attribute):
            continue
        source = getattr(old_design, attribute)
        expected_names = _named_object_sequence(source)
        try:
            mapped = new_design.model3d.find_object(source)
        except Exception as error:
            raise RuntimeError(f"copied-object remap failed for {attribute}") from error
        actual_names = _named_object_sequence(mapped) if mapped is not None else []
        if actual_names != expected_names:
            raise RuntimeError(
                f"copied-object remap mismatch for {attribute}: "
                f"expected={expected_names}, actual={actual_names}"
            )
        setattr(new_design, attribute, mapped)


def _delete_copied_solution_or_raise(primary_design, fallback_design):
    """Delete inherited fields before loss solve, using one of two COM paths."""
    errors = []
    for label, design in (("wrapper", primary_design), ("native", fallback_design)):
        try:
            design.DeleteFullVariation("All", False)
            return label
        except Exception as error:
            errors.append(f"{label}={type(error).__name__}: {error}")
    raise RuntimeError("copied solution deletion failed: " + "; ".join(errors))


class Simulation():

    def __init__(self, desktop=None):

        # 실제 사용가능 코어(cgroup affinity)에 맞춤. SLURM_CPUS_PER_TASK는 packed
        # 잡에서 잡 전체 값(예: 64)이라 4코어 cgroup에 64스레드를 요청하는 사고 유발
        # (2026-07-09 심야 전면 저속의 원인). 상한 4 = 검증된 캠페인 구성.
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:
            avail = 4  # Windows
        self.NUM_CORE = max(1, min(avail, 4))
        self.NUM_TASK = 1
        self.desktop = desktop
        self.full_model = False
        self.project_path = None
        self.solve_attempts = {"matrix": 0, "loss": 0}
        self.extraction_attempts = {}
        self.extraction_backends = {}
        self.spawned_descendants = {}

    def create_simulation_name(self):

        # slurm_scheduler dynamic_packed_srun 모드: SIMULATION_ID 환경변수 기반 이름
        # (공유 파일시스템에서 카운터 파일 락 경합 없이 고유 이름 보장)
        sim_id = os.environ.get("SIMULATION_ID")
        if sim_id:
            job_id = os.environ.get("SLURM_JOB_ID", "job")
            self.num = sim_id
            self.PROJECT_NAME = f"simulation_{job_id}_{sim_id}"
            os.makedirs("./simulation", exist_ok=True)
            self.project_path = os.path.abspath(os.path.join("simulation", self.PROJECT_NAME))
            return

        # 공유 프로젝트 폴더(MFT_1MW_2026v1) 동시 실행: 카운터 파일은 같은 계정의
        # 동시 태스크끼리 레이스 -> SLURM 환경이면 job+pid 기반 고유명 사용
        slurm_job = os.environ.get("SLURM_JOB_ID")
        if slurm_job:
            self.num = str(os.getpid())
            self.PROJECT_NAME = f"simulation_{slurm_job}_{os.getpid()}"
            os.makedirs("./simulation", exist_ok=True)
            self.project_path = os.path.abspath(os.path.join("simulation", self.PROJECT_NAME))
            return

        file_path = "./simulation_num.txt"
        simulation_dir = "./simulation"
        os.makedirs(simulation_dir, exist_ok=True)

        with open(file_path, "a+", encoding="utf-8") as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.seek(0)
            raw = file.read().strip()

            if raw.isdigit():
                current_num = int(raw)
            else:
                current_num = 1
                try:
                    existing_nums = []
                    for name in os.listdir(simulation_dir):
                        m = re.match(r"^simulation(\d+)$", name)
                        if m:
                            existing_nums.append(int(m.group(1)))
                    if existing_nums:
                        current_num = max(existing_nums) + 1
                except Exception:
                    pass

            self.num = current_num
            self.PROJECT_NAME = f"simulation{current_num}"
            next_num = current_num + 1

            file.seek(0)
            file.truncate()
            file.write(str(next_num))
            file.flush()

        self.project_path = os.path.abspath(os.path.join(simulation_dir, self.PROJECT_NAME))

    def create_project(self):

        simulation_dir = "./simulation"
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir, exist_ok=True)

        if self.project_path is None:
            self.project_path = os.path.abspath(os.path.join(simulation_dir, self.PROJECT_NAME))

        if self.desktop is None:
            raise RuntimeError("Desktop instance is None. Cannot create project.")

        try:
            self.project = self.desktop.create_project(path=self.project_path, name=self.PROJECT_NAME)
        except Exception as e:
            error_msg = f"Failed to create project '{self.PROJECT_NAME}' at path '{self.project_path}': {e}\n"
            print(error_msg, file=sys.stderr)
            sys.stderr.flush()
            raise

    def _native_project_handle(self):
        """Return the native AEDT project without probing pyProject dynamic attributes."""
        project_wrapper = getattr(self, "project", None)
        try:
            project_state = vars(project_wrapper)
        except TypeError:
            project_state = {}

        for attribute in ("project", "proj"):
            native = project_state.get(attribute)
            if native is not None and native is not False and callable(
                getattr(native, "SetActiveDesign", None)
            ):
                return native

        design = getattr(self, "design1", None)
        solver_instance = getattr(design, "solver_instance", None)
        if solver_instance is not None:
            native = getattr(solver_instance, "oproject", None)
            if native is not None and native is not False and callable(
                getattr(native, "SetActiveDesign", None)
            ):
                return native

        raise RuntimeError("native AEDT project handle is unavailable")

    def create_design(self, name="maxwell_design"):
        self.design1 = self.project.create_design(name=name, solver="maxwell3d", solution="AC Magnetic")

        # skip mesh setting
        # pyaedt 0.22: GetActiveDesign이 None을 주면 디자인 삽입 경로가 bool 오류로 무너져
        # odesign 핸들을 못 받는 케이스 실측 (AEDT에는 디자인이 실제로 생성됨).
        # -> 짧은 재시도 후, 네이티브 SetActiveDesign으로 생성된 디자인의 핸들을 직접 회수
        oDesign = self.design1.odesign
        for _ in range(3):
            if oDesign is not None and oDesign is not False:
                break
            time.sleep(5)
            oDesign = self.design1.odesign
        if oDesign is None or oDesign is False:
            try:
                native = self._native_project_handle().SetActiveDesign(name)
                if native is not None and native is not False:
                    solver_instance = self.design1.solver_instance
                    solver_instance._odesign = native
                    design_solutions = getattr(solver_instance, "design_solutions", None)
                    if design_solutions is not None:
                        design_solutions._odesign = native
                    oDesign = native
                    logging.warning(f"odesign recovered via native SetActiveDesign ({name})")
            except Exception as e:
                logging.warning(f"native SetActiveDesign fallback failed: {e}")
        if oDesign is None or oDesign is False:
            raise RuntimeError(f"odesign handle is None after design creation ({name}) - desktop unstable")
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=", False,
                "Perform Minimal validation:=", False,
                "EnabledObjects:=", [],
                "PerfectConductorThreshold:=", 1E+30,
                "InsulatorThreshold:=", 1,
                "SolveFraction:=", False,
                "Multiplier:=", "1",
                "SkipMeshChecks:=", True
            ],
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:=", "Strict",
                "IgnoreUnclassifiedObjects:=", False,
                "SkipIntersectionChecks:=", False
            ])

    def create_thermal_pad_material(self):
        # 서멀패드(실리콘 패드): 비도전성 (AC Magnetic 해석에서는 절연체로 동작)
        if "thermal_pad" not in self.design1.materials.material_keys:
            mat = self.design1.materials.add_material("thermal_pad")
            mat.conductivity = 0
            mat.permittivity = 4
            mat.permeability = 1
            mat.thermal_conductivity = 0.2  # W/(m*K)

    def create_core(self):
        # 2605SA1/1K101 코어손실 계수 [W/m^3, Hz 기준] (데이터시트 kHz 계수에서 변환됨)
        # 재질은 프로젝트 스코프이므로 두 번째 디자인에서는 재사용
        if "power_ferrite" not in self.design1.materials.material_keys:
            self.design1.set_power_ferrite(
                cm=float(self.df_plus["core_cm"].iloc[0]),
                x=float(self.df_plus["core_x"].iloc[0]),
                y=float(self.df_plus["core_y"].iloc[0])
            )
        self.power_ferrite_mat = self.design1.materials["power_ferrite"]
        self.power_ferrite_mat.permeability = "3000"

        self.create_thermal_pad_material()

        n_group = int(self.df_plus["n_core_group"].iloc[0])
        plate_on = int(self.df_plus["core_plate_on"].iloc[0]) != 0
        pad_on = float(self.df_plus["core_plate_pad_t"].iloc[0]) > 0

        core_objs, plate_objs, pad_objs = create_core(
            design=self.design1,
            name="core",
            core_material="power_ferrite",
            n_group=n_group,
            plate_material="aluminum",
            pad_material="thermal_pad",
            plate_on=plate_on,
            pad_on=pad_on,
            plate_color=PLATE_COLOR,
            pad_color=PAD_COLOR
        )
        self.design1.core_objs = core_objs
        self.design1.core_plates = plate_objs
        self.design1.core_pads = pad_objs

    def _op_temp_conductor_material(self):
        """운전 온도 기준 도전율의 구리 재질 생성 (기본 80C).
        20C 구리(5.8e7 S/m) 기준이면 실물(~80-100C) 권선손실을 ~25% 과소평가한다.
        sigma(T) = sigma20 / (1 + 0.00393*(T-20))"""
        T = float(self.df_plus["conductor_temp_C"].iloc[0])
        name = f"copper_{int(round(T))}C"
        mats = self.design1.materials
        if name not in mats.material_keys:
            m = mats.add_material(name)
            m.conductivity = 5.8e7 / (1.0 + 0.00393 * (T - 20.0))
            m.permeability = 0.999991
        return name

    def create_coil(self):

        l1 = self.df_plus["l1"].iloc[0]
        l2 = self.df_plus["l2"].iloc[0]

        conductor_mat = self._op_temp_conductor_material()
        # Keep the authoritative material name without consulting Object3d
        # properties later.  The latter forces a full PyAEDT modeler refresh,
        # which is not reliable immediately after CopyDesign/Paste.
        self.winding_conductor_material = conductor_mat

        round_corner = int(self.df_plus["round_corner"].iloc[0]) != 0
        corner_radius = float(self.df_plus["corner_radius"].iloc[0]) if round_corner else None
        corner_segments = int(self.df_plus["corner_segments"].iloc[0])

        # 1차 중심 권선: y방향은 냉각판 슬롯 간격으로 벌어짐
        tx_y_gaps, tx_slot_indices = get_tx_y_gaps(self.df_plus)

        self.design1.Tx_windings_main, self.N_Tx_main, self.Tx_coil_width_main, self.Tx_coil_height_main, self.Tx_coil_gap_x_main, self.Tx_coil_gap_z_main = create_coil(
            design=self.design1,
            name="Tx_main",
            window_height=self.df_plus["nwh1"].iloc[0],
            window_length=self.df_plus["nwl1_main"].iloc[0],
            window_layer=self.df_plus["N1_main"].iloc[0],
            N_input=1,
            width_fill_factor=self.df_plus["wff1_main"].iloc[0],
            space_length=self.df_plus["sl1_main_x"].iloc[0],
            space_width=self.df_plus["sl1_main_y"].iloc[0],
            shape="rectangle",
            offset=[0, 0, 0],
            color=[255, 10, 10],
            y_slot_gaps=tx_y_gaps,
            round_corner=round_corner,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            material=conductor_mat
        )

        self.design1.Rx_windings_main, self.N_Rx_main, self.Rx_coil_width_main, self.Rx_coil_height_main, self.Rx_coil_gap_x_main, self.Rx_coil_gap_z_main = create_coil(
            design=self.design1,
            name="Rx_main",
            window_height=self.df_plus["nwh2"].iloc[0],
            window_length=self.df_plus["nwl2_main"].iloc[0],
            window_layer=self.df_plus["N2_main"].iloc[0],
            N_input=1,
            width_fill_factor=self.df_plus["wff2_main"].iloc[0],
            space_length=self.df_plus["sl2_main_x"].iloc[0],
            space_width=self.df_plus["sl2_main_y"].iloc[0],
            shape="rectangle",
            offset=[0, 0, 0],
            color=[10, 10, 255],
            round_corner=round_corner,
            corner_radius=corner_radius,
            corner_segments=corner_segments,
            material=conductor_mat
        )

        if self.df_plus["N1_side"].iloc[0] != 0:
            self.design1.Tx_windings_side, self.N_Tx_side, self.Tx_coil_width_side, self.Tx_coil_height_side, self.Tx_coil_gap_x_side, self.Tx_coil_gap_z_side = create_coil(
                design=self.design1,
                name="Tx_side",
                window_height=self.df_plus["nwh1"].iloc[0],
                window_length=self.df_plus["nwl1_side"].iloc[0],
                window_layer=self.df_plus["N1_side"].iloc[0],
                N_input=1,
                width_fill_factor=self.df_plus["wff1_side"].iloc[0],
                space_length=self.df_plus["sl1_side_x"].iloc[0],
                space_width=self.df_plus["sl1_side_y"].iloc[0],
                shape="rectangle",
                offset=[(-l1 - l2 - l1 / 2), 0, 0],
                color=[255, 10, 10],
                round_corner=round_corner,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                material=conductor_mat
            )

        if self.df_plus["N2_side"].iloc[0] != 0:
            self.design1.Rx_windings_side, self.N_Rx_side, self.Rx_coil_width_side, self.Rx_coil_height_side, self.Rx_coil_gap_x_side, self.Rx_coil_gap_z_side = create_coil(
                design=self.design1,
                name="Rx_side",
                window_height=self.df_plus["nwh2"].iloc[0],
                window_length=self.df_plus["nwl2_side"].iloc[0],
                window_layer=self.df_plus["N2_side"].iloc[0],
                N_input=1,
                width_fill_factor=self.df_plus["wff2_side"].iloc[0],
                space_length=self.df_plus["sl2_side_x"].iloc[0],
                space_width=self.df_plus["sl2_side_y"].iloc[0],
                shape="rectangle",
                offset=[(-l1 - l2 - l1 / 2), 0, 0],
                color=[10, 10, 255],
                round_corner=round_corner,
                corner_radius=corner_radius,
                corner_segments=corner_segments,
                material=conductor_mat
            )

        if self.df_plus["N1_side"].iloc[0] == 0:
            self.design1.Tx_windings_side = []
            self.N_Tx_side = 0
            self.Tx_coil_width_side = 0
            self.Tx_coil_height_side = 0
            self.Tx_coil_gap_x_side = 0
            self.Tx_coil_gap_z_side = 0

        if self.df_plus["N2_side"].iloc[0] == 0:
            self.design1.Rx_windings_side = []
            self.N_Rx_side = 0
            self.Rx_coil_width_side = 0
            self.Rx_coil_height_side = 0
            self.Rx_coil_gap_x_side = 0
            self.Rx_coil_gap_z_side = 0

        # 풀모델: 대칭이 없으므로 반대쪽(+x) 측면 레그의 측면 권선도 실제로 생성
        self.design1.Tx_windings_side2 = []
        self.design1.Rx_windings_side2 = []
        if self.full_model:
            if self.df_plus["N1_side"].iloc[0] != 0:
                self.design1.Tx_windings_side2, _, _, _, _, _ = create_coil(
                    design=self.design1,
                    name="Tx_side2",
                    window_height=self.df_plus["nwh1"].iloc[0],
                    window_length=self.df_plus["nwl1_side"].iloc[0],
                    window_layer=self.df_plus["N1_side"].iloc[0],
                    N_input=1,
                    width_fill_factor=self.df_plus["wff1_side"].iloc[0],
                    space_length=self.df_plus["sl1_side_x"].iloc[0],
                    space_width=self.df_plus["sl1_side_y"].iloc[0],
                    shape="rectangle",
                    offset=[(l1 + l2 + l1 / 2), 0, 0],
                    color=[255, 10, 10],
                    round_corner=round_corner,
                    corner_radius=corner_radius,
                    corner_segments=corner_segments,
                    material=conductor_mat
                )
            if self.df_plus["N2_side"].iloc[0] != 0:
                self.design1.Rx_windings_side2, _, _, _, _, _ = create_coil(
                    design=self.design1,
                    name="Rx_side2",
                    window_height=self.df_plus["nwh2"].iloc[0],
                    window_length=self.df_plus["nwl2_side"].iloc[0],
                    window_layer=self.df_plus["N2_side"].iloc[0],
                    N_input=1,
                    width_fill_factor=self.df_plus["wff2_side"].iloc[0],
                    space_length=self.df_plus["sl2_side_x"].iloc[0],
                    space_width=self.df_plus["sl2_side_y"].iloc[0],
                    shape="rectangle",
                    offset=[(l1 + l2 + l1 / 2), 0, 0],
                    color=[10, 10, 255],
                    round_corner=round_corner,
                    corner_radius=corner_radius,
                    corner_segments=corner_segments,
                    material=conductor_mat
                )

        # 1차 권선 냉각 플레이트 (y측면 슬롯, 양측 대칭, 서멀패드|알루미늄|서멀패드)
        wcp_on = int(self.df_plus["wcp_on"].iloc[0]) != 0
        if wcp_on and len(tx_slot_indices) > 0:
            self.design1.wcp_plates, self.design1.wcp_pads = create_winding_cooling_plates(
                design=self.design1,
                name="Tx_main_wcp",
                space_width=self.df_plus["sl1_main_y"].iloc[0],
                coil_width=self.Tx_coil_width_main,
                y_gaps=tx_y_gaps,
                slot_indices=tx_slot_indices,
                wcp_len_x=float(self.df_plus["wcp_len_x"].iloc[0]),
                wcp_t=float(self.df_plus["wcp_t"].iloc[0]),
                pad_t=float(self.df_plus["wcp_pad_t"].iloc[0]),
                height=float(self.df_plus["nwh1"].iloc[0]),
                plate_material="aluminum",
                pad_material="thermal_pad",
                plate_color=PLATE_COLOR,
                pad_color=PAD_COLOR,
                offset=[0, 0, 0]
            )
        else:
            self.design1.wcp_plates = []
            self.design1.wcp_pads = []

        self.Tx_windings = self.design1.Tx_windings_main + self.design1.Tx_windings_side + self.design1.Tx_windings_side2
        self.Rx_windings = self.design1.Rx_windings_main + self.design1.Rx_windings_side + self.design1.Rx_windings_side2
        self.design1.Tx_windings = self.Tx_windings
        self.design1.Rx_windings = self.Rx_windings

    def split_geometry(self):

        # 풀모델: 대칭 분할 없이 전체 지오메트리 유지
        if self.full_model:
            return

        geometrys = (self.design1.core_objs + self.design1.core_plates + self.design1.core_pads
                     + self.design1.wcp_plates + self.design1.wcp_pads
                     + self.design1.Tx_windings_main + self.design1.Rx_windings_main
                     + self.design1.Tx_windings_side + self.design1.Rx_windings_side)

        # 분할 순서대로 진행하되, 앞 분할에서 통째로 삭제된 오브젝트를 다음 호출에 넘기지 않음
        # (넘기면 AEDT가 'Part not found' 경고를 배치로 뿜음 - 무해하지만 소음)
        def _alive(objs):
            existing = set(self.design1.modeler.object_names)
            return [o for o in objs if o.name in existing]

        self.design1.modeler.split(assignment=geometrys, plane="XY", sides="PositiveOnly")
        geometrys = _alive(geometrys)
        self.design1.modeler.split(assignment=geometrys, plane="XZ", sides="PositiveOnly")
        geometrys = _alive(geometrys)
        self.design1.modeler.split(assignment=geometrys, plane="YZ", sides="NegativeOnly")

        # 대칭 분할로 완전히 잘려나간 오브젝트(y<0 쪽 콜드플레이트/냉각판 등)를 리스트에서 제거
        # (이후 eddy 설정/손실 계산이 존재하지 않는 오브젝트를 참조하지 않도록)
        existing = set(self.design1.modeler.object_names)
        self.design1.core_objs = [o for o in self.design1.core_objs if o.name in existing]
        self.design1.core_plates = [o for o in self.design1.core_plates if o.name in existing]
        self.design1.core_pads = [o for o in self.design1.core_pads if o.name in existing]
        self.design1.wcp_plates = [o for o in self.design1.wcp_plates if o.name in existing]
        self.design1.wcp_pads = [o for o in self.design1.wcp_pads if o.name in existing]

    def create_coil_section(self):

        if self.full_model:
            self._create_coil_section_full()
            return

        self.Tx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix=None, plane="YZ", rename_faces=False, mod="single")
        self.Tx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_main, sheet_prefix=None, plane="ZX", rename_faces=False, mod="single")

        self.Rx_main_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix=None, plane="ZX", rename_faces=False, mod="single")
        self.Rx_main_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_main, sheet_prefix=None, plane="YZ", rename_faces=False, mod="single")

        if self.df_plus["N1_side"].iloc[0] != 0:
            self.Tx_side_sheets_in, self.Tx_side_sheets_out = create_coil_section(design=self.design1, winding_obj=self.design1.Tx_windings_side, sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")
        if self.df_plus["N2_side"].iloc[0] != 0:
            self.Rx_side_sheets_out, self.Rx_side_sheets_in = create_coil_section(design=self.design1, winding_obj=self.design1.Rx_windings_side, sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")

    def _create_coil_section_full(self):
        """
        풀모델용 단면 생성: 닫힌 링 도체는 자르지 않고 ZX 평면 단면 시트를
        턴당 1개만 남겨 터미널로 사용한다. (ZX 단면은 링당 2개 생기므로
        한쪽을 삭제. 남기는 다리/극성은 하프모델의 전류 방향 관례와 일치시킴)
        """
        def _pick(winding_objs, keep):
            x_neg, x_pos = create_coil_section(design=self.design1, winding_obj=winding_objs,
                                               sheet_prefix=None, plane="ZX", rename_faces=False, mod="both")
            kept, drop = (x_neg, x_pos) if keep == "neg" else (x_pos, x_neg)
            if drop:
                self.design1.modeler.delete(drop)
            return kept

        # 중심 권선: x- 다리 시트 사용 (Tx는 Negative, Rx는 Positive 극성 -> 상호 반대 방향)
        self.Tx_main_sheets_full = _pick(self.design1.Tx_windings_main, keep="neg")
        self.Rx_main_sheets_full = _pick(self.design1.Rx_windings_main, keep="neg")

        # 측면 권선 (-x 레그): 하프모델과 동일한 다리 선택
        self.Tx_side_sheets_full = []
        self.Rx_side_sheets_full = []
        self.Tx_side2_sheets_full = []
        self.Rx_side2_sheets_full = []
        if self.df_plus["N1_side"].iloc[0] != 0:
            self.Tx_side_sheets_full = _pick(self.design1.Tx_windings_side, keep="neg")   # 바깥 다리
            self.Tx_side2_sheets_full = _pick(self.design1.Tx_windings_side2, keep="pos")  # 미러: 바깥 다리
        if self.df_plus["N2_side"].iloc[0] != 0:
            self.Rx_side_sheets_full = _pick(self.design1.Rx_windings_side, keep="pos")   # 안쪽 다리
            self.Rx_side2_sheets_full = _pick(self.design1.Rx_windings_side2, keep="neg")  # 미러: 안쪽 다리

    def _assign_coil_full(self):
        """풀모델: 턴당 터미널 1개. 미러(+x) 측 권선은 반사 대칭으로 순환 방향이
        반전되므로 극성을 반대로 지정한다."""
        self.Tx_coil = []
        self.Rx_coil = []

        for idx, sheet in enumerate(self.Tx_main_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_center_coil_{idx}"))
        for idx, sheet in enumerate(self.Rx_main_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_center_coil_{idx}"))

        for idx, sheet in enumerate(self.Tx_side_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_side_coil_{idx}"))
        for idx, sheet in enumerate(self.Tx_side2_sheets_full, start=1):
            self.Tx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_side2_coil_{idx}"))

        for idx, sheet in enumerate(self.Rx_side_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil_{idx}"))
        for idx, sheet in enumerate(self.Rx_side2_sheets_full, start=1):
            self.Rx_coil.append(self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_side2_coil_{idx}"))

        self.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in self.Tx_coil])
        self.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in self.Rx_coil])

    def assign_winding(self, mode="matrix"):
        """
        mode="matrix": Tx/Rx 모두 정격 전류원 (L/k 매트릭스용, 기존 방식)
        mode="loss"  : Tx 전압원(V1) + Rx 정격 전류원 -> 코어 자속이 전압으로 결정되어
                       권선손실+코어손실을 한 번에 해석 (전류 강제로 인한 비물리적 자속 방지)
        """
        I1 = float(self.df_plus["I1_rated"].iloc[0])
        I2 = float(self.df_plus["I2_rated"].iloc[0])
        matrix_is_solid = not (mode == "matrix" and _lightweight_matrix_enabled(self))

        if mode == "loss":
            # 손실 원샷 여자 (풀모델): Tx 전압원(V1) + Rx 정격 전류원.
            # Maxwell이 1차 전류(부하분 + 자화분)를 스스로 풀어 코어 자속이 실제 운전 수준이 됨.
            # 검증: 무부하 케이스에서 코어손실/턴손실이 자화전류(Im=V1/wLm) 주입 방식과 1% 이내 일치.
            # 주의: 전압 권선의 InputCurrent 리포트는 0으로 표시될 수 있으나 (표시 아티팩트)
            #       실제 해는 유효함 (InducedVoltage ~= V1, 손실/자속 정상).
            V1 = float(self.df_plus["V1_rms"].iloc[0])
            # P_target 자동 위상이 계산되어 있으면 우선 사용
            phase2 = getattr(self, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(self.df_plus["I2_phase_deg"].iloc[0])

            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Voltage",
                is_solid=True,
                voltage=f"{V1 * math.sqrt(2)}V",
                resistance=0,
                inductance=0,
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I2 * math.sqrt(2)}A",
                phase=f"{phase2}deg",
                name="Rx_winding"
            )
        elif mode == "loss_sym":
            # 손실 원샷 여자 (대칭 1/8, 캠페인용): 전압원이 대칭 터미널 구조에서 무효이므로
            # Tx 전류 = 부하분(N2/N1 x I2, Rx와 역상) + 자화분(Im = sqrt(2)V1/(w Lm_true), -90deg)
            # 페이저 합을 직접 주입. 선형 해석이므로 올바른 복소 전류 = 실제 운전 자속/전류 재현.
            # (Lm은 design1 매트릭스에서 자동 취득 - run_one_loop에서 self.loss_I1_* 설정)
            phase2 = getattr(self, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(self.df_plus["I2_phase_deg"].iloc[0])

            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{self.loss_I1_peak}A",
                phase=f"{self.loss_I1_phase_deg}deg",
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=True,
                current=f"{I2 * math.sqrt(2)}A",
                phase=f"{phase2}deg",
                name="Rx_winding"
            )
        else:
            self.tx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=matrix_is_solid,
                current=f"{I1 * math.sqrt(2)}A",
                name="Tx_winding"
            )

            self.rx_winding = self.design1.assign_winding(
                assignment=[],
                winding_type="Current",
                is_solid=matrix_is_solid,
                current=f"{I2 * math.sqrt(2)}A",
                name="Rx_winding"
            )

    def assign_coil(self):

        if self.full_model:
            self._assign_coil_full()
            return

        self.Tx_coil = []
        self.Rx_coil = []

        for idx, sheet in enumerate(self.Tx_main_sheets_in, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_center_coil_in_{idx}")
            self.Tx_coil.append(coil)
        for idx, sheet in enumerate(self.Tx_main_sheets_out, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_center_coil_out_{idx}")
            self.Tx_coil.append(coil)

        for idx, sheet in enumerate(self.Rx_main_sheets_in, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_center_coil_in_{idx}")
            self.Rx_coil.append(coil)
        for idx, sheet in enumerate(self.Rx_main_sheets_out, start=1):
            coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_center_coil_out_{idx}")
            self.Rx_coil.append(coil)

        if self.df_plus["N1_side"].iloc[0] != 0:
            for idx, sheet in enumerate(self.Tx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Tx_side_coil_in_{idx}")
                self.Tx_coil.append(coil)
            for idx, sheet in enumerate(self.Tx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Tx_side_coil_out_{idx}")
                self.Tx_coil.append(coil)

        if self.df_plus["N2_side"].iloc[0] != 0:
            for idx, sheet in enumerate(self.Rx_side_sheets_in, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Positive", name=f"Rx_side_coil_in_{idx}")
                self.Rx_coil.append(coil)
            for idx, sheet in enumerate(self.Rx_side_sheets_out, start=1):
                coil = self.design1.assign_coil(sheet, conductors_number=1, polarity="Negative", name=f"Rx_side_coil_out_{idx}")
                self.Rx_coil.append(coil)

        self.design1.add_winding_coils(assignment="Tx_winding", coils=[coil.name for coil in self.Tx_coil])
        self.design1.add_winding_coils(assignment="Rx_winding", coils=[coil.name for coil in self.Rx_coil])

    def assign_matrix(self):
        self.design1.assign_matrix(matrix_name="Matrix", assignment=["Tx_winding", "Rx_winding"])

    def assign_core_loss(self):
        """loss 디자인: 코어 그룹에 코어손실 계산 활성화 (Power Ferrite 계수는 create_core에서 설정)"""
        assigned = self.design1.set_core_losses(
            assignment=[c.name for c in self.design1.core_objs],
            core_loss_on_field=False
        )
        if assigned is False:
            raise RuntimeError(
                "set_core_losses returned False; copied loss design is not ready"
            )

    def assign_skin_depth(self):

        freq = float(self.df_plus["freq"].iloc[0])

        mu0 = 4 * math.pi * 1e-7
        mu_copper = mu0
        sigma_copper = 58000000
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu_copper * sigma_copper)) * 1e3  # in mm

        self.Tx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment=self.design1.Tx_windings,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="2",
            name="Tx_winding_skin_depth"
        )
        if self.Tx_skin_depth_mesh is False or self.Tx_skin_depth_mesh is None:
            raise RuntimeError("failed to assign Tx winding skin-depth mesh")

        rx_mode = str(self.df_plus["rx_mesh_mode"].iloc[0])
        cw2 = float(self.df_plus["cw2"].iloc[0])

        if rx_mode == "length":
            # 실험적: foil 두께 방향 2요소 수준의 length 기반 메시 (벤치마크용)
            self.Rx_length_mesh = self.design1.mesh.assign_length_mesh(
                assignment=self.design1.Rx_windings,
                maximum_length=f"{max(cw2 / 2, 0.1)}mm",
                maximum_elements=None,
                name="Rx_winding_length_mesh"
            )
            if self.Rx_length_mesh is False or self.Rx_length_mesh is None:
                raise RuntimeError("failed to assign Rx winding length mesh")
        elif rx_mode == "length-coarse":
            # 실험적: foil 두께 1요소 (최대 가속 후보, 벤치마크용)
            self.Rx_length_mesh = self.design1.mesh.assign_length_mesh(
                assignment=self.design1.Rx_windings,
                maximum_length=f"{cw2}mm",
                maximum_elements=None,
                name="Rx_winding_length_mesh"
            )
            if self.Rx_length_mesh is False or self.Rx_length_mesh is None:
                raise RuntimeError("failed to assign Rx winding coarse length mesh")
        else:
            # 기본: 기존 skin-depth op (proximity effect 반영 검증된 설정)
            self.Rx_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
                assignment=self.design1.Rx_windings,
                skin_depth=f'{skin_depth}mm',
                triangulation_max_length='50mm',
                layers_number="1",
                name="Rx_winding_skin_depth"
            )
            if self.Rx_skin_depth_mesh is False or self.Rx_skin_depth_mesh is None:
                raise RuntimeError("failed to assign Rx winding skin-depth mesh")
        return 2

    @staticmethod
    def _native_object_names(value, label):
        """Normalize a native AEDT name sequence without accepting False/strings."""
        if value is None or value is False or isinstance(value, (str, bytes, bool)):
            raise RuntimeError(f"{label} returned no object-name sequence: {value!r}")
        try:
            names = [str(item).strip() for item in value]
        except TypeError as error:
            raise RuntimeError(
                f"{label} returned a non-iterable object-name sequence: {value!r}"
            ) from error
        if any(not name for name in names):
            raise RuntimeError(f"{label} returned an empty object name: {names!r}")
        if len(set(names)) != len(names):
            raise RuntimeError(f"{label} returned duplicate object names: {names!r}")
        return names

    def _set_plate_eddy_effects_native(
            self, enable_eddy_effects, max_attempts=5, timeout_s=30.0,
            initial_retry_delay=0.5, clock=time.monotonic, sleeper=time.sleep):
        """Set the complete conductor eddy vector through fresh native handles.

        PyAEDT ``eddy_effects_on`` first discovers every conductor by walking the
        complete cached modeler/Object3d inventory.  A copied design can already
        be valid in AEDT while that cache still contains a stale 3D editor.  This
        path instead performs two native bulk material queries, validates the
        exact campaign conductor universe, writes one full vector, and reads the
        complete vector back.  Retrying this transaction is idempotent and never
        repeats Copy/Paste, mesh creation, or a solve.
        """
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be at least one")

        expected_project_name = str(getattr(self, "PROJECT_NAME", "") or "").strip()
        expected_design_name = _aedt_design_name(
            getattr(self.design1, "design_name", "")
        )
        winding_material = str(
            getattr(self, "winding_conductor_material", "") or ""
        ).strip()
        if not expected_project_name:
            raise RuntimeError("native eddy project identity is unavailable")
        if not expected_design_name:
            raise RuntimeError("native eddy design identity is unavailable")
        if not winding_material:
            raise RuntimeError("authoritative winding conductor material is unavailable")

        winding_names = (
            _named_object_sequence(getattr(self.design1, "Tx_windings", None))
            + _named_object_sequence(getattr(self.design1, "Rx_windings", None))
        )
        plate_names = (
            _named_object_sequence(getattr(self.design1, "core_plates", None))
            + _named_object_sequence(getattr(self.design1, "wcp_plates", None))
        )
        if not winding_names:
            raise RuntimeError("native eddy contract has no winding conductors")
        if not plate_names:
            raise RuntimeError("native eddy contract has no plate conductors")
        expected_names = winding_names + plate_names
        if len(set(expected_names)) != len(expected_names):
            raise RuntimeError(
                "native eddy contract contains duplicate/overlapping conductor names"
            )

        desired_eddy = {
            name: False for name in winding_names
        }
        desired_eddy.update({
            name: bool(enable_eddy_effects) for name in plate_names
        })
        desired_displacement = {name: False for name in expected_names}

        deadline = clock() + max(0.0, float(timeout_s))
        attempts = []
        for attempt in range(1, int(max_attempts) + 1):
            if attempt > 1 and clock() >= deadline:
                break
            observed = {"copper": None, "aluminum": None}
            try:
                oproject = self._refresh_native_project_handle()
                actual_project_name = str(oproject.GetName() or "").strip()
                if actual_project_name != expected_project_name:
                    raise _AedtIdentityMismatch(
                        "project identity mismatch: "
                        f"expected={expected_project_name}, actual={actual_project_name or '<empty>'}"
                    )

                odesign = oproject.SetActiveDesign(expected_design_name)
                if odesign is None or odesign is False:
                    raise RuntimeError(
                        f"SetActiveDesign returned no design ({expected_design_name})"
                    )
                actual_design_name = _aedt_design_name(odesign)
                if actual_design_name != expected_design_name:
                    raise _AedtIdentityMismatch(
                        "design identity mismatch: "
                        f"expected={expected_design_name}, actual={actual_design_name or '<empty>'}"
                    )
                design_type = str(odesign.GetDesignType() or "")
                solution_type = str(odesign.GetSolutionType() or "")
                if design_type != "Maxwell 3D" or not _is_ac_magnetic_solution(
                        solution_type):
                    raise _AedtIdentityMismatch(
                        "design physics mismatch: "
                        f"type={design_type!r}, solution={solution_type!r}"
                    )

                oeditor = odesign.SetActiveEditor("3D Modeler")
                if oeditor is None or oeditor is False:
                    raise RuntimeError("SetActiveEditor returned no 3D Modeler")
                oboundary = odesign.GetModule("BoundarySetup")
                if oboundary is None or oboundary is False:
                    raise RuntimeError("active design returned no BoundarySetup module")

                observed_windings = self._native_object_names(
                    oeditor.GetObjectsByMaterial(winding_material),
                    f"GetObjectsByMaterial({winding_material})",
                )
                observed_plates = self._native_object_names(
                    oeditor.GetObjectsByMaterial("aluminum"),
                    "GetObjectsByMaterial(aluminum)",
                )
                observed = {
                    "copper": observed_windings,
                    "aluminum": observed_plates,
                }
                if (
                        set(observed_windings) != set(winding_names)
                        or len(observed_windings) != len(winding_names)):
                    raise RuntimeError(
                        "winding conductor universe mismatch: "
                        f"expected={winding_names}, actual={observed_windings}"
                    )
                if (
                        set(observed_plates) != set(plate_names)
                        or len(observed_plates) != len(plate_names)):
                    raise RuntimeError(
                        "plate conductor universe mismatch: "
                        f"expected={plate_names}, actual={observed_plates}"
                    )
                observed_all = observed_windings + observed_plates
                if (
                        len(set(observed_all)) != len(observed_all)
                        or set(observed_all) != set(expected_names)
                        or len(observed_all) != len(expected_names)):
                    raise RuntimeError(
                        "combined conductor universe mismatch: "
                        f"expected={expected_names}, actual={observed_all}"
                    )

                eddy_vector = ["NAME:EddyEffectVector"]
                for name in expected_names:
                    eddy_vector.append([
                        "NAME:Data",
                        "Object Name:=", name,
                        "Eddy Effect:=", desired_eddy[name],
                        "Displacement Current:=", desired_displacement[name],
                    ])
                result = oboundary.SetEddyEffect([
                    "NAME:Eddy Effect Setting", eddy_vector
                ])
                if result is False:
                    raise RuntimeError("BoundarySetup.SetEddyEffect returned False")

                mismatches = {}
                for name in expected_names:
                    actual_eddy = _aedt_bool(oboundary.GetEddyEffect(name))
                    actual_displacement = _aedt_bool(
                        oboundary.GetDisplacementCurrent(name)
                    )
                    if (
                            actual_eddy != desired_eddy[name]
                            or actual_displacement != desired_displacement[name]):
                        mismatches[name] = {
                            "eddy": {
                                "expected": desired_eddy[name],
                                "actual": actual_eddy,
                            },
                            "displacement": {
                                "expected": desired_displacement[name],
                                "actual": actual_displacement,
                            },
                        }
                if mismatches:
                    raise RuntimeError(
                        f"native eddy-effect readback mismatch: {mismatches}"
                    )
                return len(plate_names)
            except _AedtIdentityMismatch:
                # Never mutate, or even retry against, an unexpected project/design.
                raise
            except Exception as error:
                attempts.append({
                    "attempt": attempt,
                    "error": f"{type(error).__name__}: {error}",
                    "observed": observed,
                })
                now = clock()
                if attempt >= int(max_attempts) or now >= deadline:
                    break
                delay = min(
                    max(0.0, float(initial_retry_delay)) * (2 ** (attempt - 1)),
                    max(0.0, deadline - now),
                )
                logging.warning(
                    "native conductor eddy transaction failed "
                    f"(attempt {attempt}/{int(max_attempts)}): {error}"
                )
                sleeper(delay)
        raise RuntimeError(
            "native conductor eddy transaction failed closed; "
            f"project={expected_project_name}, design={expected_design_name}, "
            f"attempts={attempts}"
        )

    def assign_plate_settings(self, enable_eddy_effects=True, assign_skin_mesh=True):
        """콜드플레이트/권선 냉각판 (알루미늄) 와전류 설정 + 메시"""

        plates = self.design1.core_plates + self.design1.wcp_plates
        if not plates:
            return 0

        plate_names = [p.name for p in plates]

        plate_count = self._set_plate_eddy_effects_native(enable_eddy_effects)
        if int(plate_count) != len(plate_names):
            raise RuntimeError(
                "native plate eddy-effect count mismatch: "
                f"expected={len(plate_names)}, actual={plate_count}"
            )

        if not assign_skin_mesh:
            logging.info("plate skin-depth mesh skipped")
            return len(plates)

        freq = float(self.df_plus["freq"].iloc[0])
        mu0 = 4 * math.pi * 1e-7
        sigma_al = 3.8e+7
        omega = 2 * math.pi * freq
        skin_depth = math.sqrt(2 / (omega * mu0 * sigma_al)) * 1e3  # in mm (~2.6mm @1kHz)

        self.plate_skin_depth_mesh = self.design1.mesh.assign_skin_depth(
            assignment=plate_names,
            skin_depth=f'{skin_depth}mm',
            triangulation_max_length='50mm',
            layers_number="1",
            name="plate_skin_depth"
        )
        if self.plate_skin_depth_mesh is False or self.plate_skin_depth_mesh is None:
            raise RuntimeError("failed to assign plate skin-depth mesh")
        return int(plate_count)

    def assign_boundary(self):

        if self.full_model:
            # 풀모델: 대칭 경계 없이 전방향 air region + 전면 radiation
            self.air_region = self.design1.modeler.create_air_region(x_pos=100.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=100.0, z_neg=100.0, is_percentage=True)
            self.design1.assign_radiation(
                assignment=[
                    self.air_region.top_face_x, self.air_region.bottom_face_x,
                    self.air_region.top_face_y, self.air_region.bottom_face_y,
                    self.air_region.top_face_z, self.air_region.bottom_face_z
                ],
                radiation="Radiation"
            )
            return

        self.air_region = self.design1.modeler.create_air_region(x_pos=0.0, y_pos=100.0, z_pos=100.0, x_neg=100.0, y_neg=0.0, z_neg=0.0, is_percentage=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_z, symmetry_name="Symmetry1", is_odd=False)
        self.design1.assign_symmetry(assignment=self.air_region.top_face_x, symmetry_name="Symmetry2", is_odd=True)
        self.design1.assign_symmetry(assignment=self.air_region.bottom_face_y, symmetry_name="Symmetry3", is_odd=True)
        self.design1.assign_radiation(assignment=[self.air_region.top_face_z, self.air_region.bottom_face_x, self.air_region.top_face_y], radiation="Radiation")

    def create_setup(self, mode="loss"):
        """mode="matrix": 인덕턴스 전용 경량 수렴 (skin 없음 + 완화된 pe)
        mode="loss": 정밀 수렴 (손실/근접효과 - 기존 설정 유지)"""
        pfx = "matrix_" if mode == "matrix" else ""

        def _p(key, default):
            col = pfx + key
            if col in self.df_plus.columns and pd.notna(self.df_plus[col].iloc[0]):
                return self.df_plus[col].iloc[0]
            return self.df_plus[key].iloc[0] if key in self.df_plus.columns else default

        self.design1.setup = self.design1.create_setup(name="Setup1")
        self.design1.setup.properties["Max. Number of Passes"] = int(_p("max_passes", 10))
        self.design1.setup.properties["Min. Number of Passes"] = 1
        self.design1.setup.properties["Min. Converged Passes"] = int(_p("min_converged", 2))
        self.design1.setup.properties["Percent Error"] = float(_p("percent_error", 2.0))
        self.design1.setup.properties["Frequency Setup"] = f"{float(self.df_plus['freq'].iloc[0])}Hz"

    def _solution_data_frame(self, expressions, aliases=None, target_units=None,
                             report_category=None, report_context=None,
                             extraction_key="result",
                             max_attempts=3, retry_delay=5):
        """Read finite scalar results without creating or exporting an AEDT report file.

        Fields calculator expressions bypass PyAEDT's high-level report object. That
        object adds every design variable to the sweep selection and returns an
        unusable SolutionData lookup for scalar Maxwell fields in PyAEDT 0.22.
        """
        expressions = list(expressions)
        aliases = list(aliases or expressions)
        if len(expressions) != len(aliases):
            raise ValueError("expressions and aliases must have the same length")
        target_units = target_units or {}
        last_error = None
        backend = (
            "get_solution_data_per_variation"
            if report_category == "Fields"
            else "get_solution_data"
        )

        for attempt in range(1, max_attempts + 1):
            self.extraction_attempts[extraction_key] = self.extraction_attempts.get(extraction_key, 0) + 1
            try:
                post = self.design1.post
                if callable(post) and not hasattr(post, "get_solution_data"):
                    post = post()
                if report_category == "Fields":
                    solution = post.get_solution_data_per_variation(
                        solution_type="Fields",
                        setup_sweep_name="Setup1 : LastAdaptive",
                        context=[],
                        sweeps={"Freq": ["All"], "Phase": ["0deg"]},
                        expressions=expressions,
                    )
                else:
                    solution = post.get_solution_data(
                        expressions=expressions,
                        setup_sweep_name="Setup1 : LastAdaptive",
                        report_category=report_category,
                        context=report_context,
                    )
                if solution is None or solution is False:
                    last_error = RuntimeError(f"{backend} returned no usable response")
                else:
                    units = getattr(solution, "units_data", {}) or {}
                    row = {}
                    missing = []
                    for expression, alias in zip(expressions, aliases):
                        if hasattr(solution, "get_expression_data"):
                            _, values = solution.get_expression_data(expression, formula="real")
                        else:
                            values = solution.data_real(expression)
                        if values is None or values is False or len(values) == 0:
                            missing.append(expression)
                            continue
                        try:
                            value = float(values[0])
                            value = _convert_solution_unit(
                                value, units.get(expression, ""), target_units.get(expression, "")
                            )
                            if not math.isfinite(value):
                                missing.append(expression)
                                continue
                            row[alias] = value
                        except (TypeError, ValueError, OverflowError) as e:
                            missing.append(expression)
                            last_error = e
                    if not missing:
                        self.extraction_backends[extraction_key] = backend
                        return pd.DataFrame([row], columns=aliases)
                    last_error = RuntimeError("missing/non-finite expressions: " + ", ".join(missing))
            except Exception as e:
                last_error = e
                logging.warning(
                    f"[{extraction_key}] get_solution_data failed "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
            if attempt < max_attempts:
                time.sleep(retry_delay)

        message = f"[{extraction_key}] result extraction failed after {max_attempts} attempts: {last_error}"
        raise RuntimeError(message)

    def _refresh_native_project_handle(self):
        """Rebind the current project through Desktop without touching any solve."""
        project_wrapper = getattr(self, "project", None)
        try:
            project_state = vars(project_wrapper)
        except TypeError:
            project_state = {}

        desktop_wrapper = project_state.get("desktop") or getattr(self, "desktop", None)
        odesktop = getattr(desktop_wrapper, "odesktop", None)
        set_active_project = getattr(odesktop, "SetActiveProject", None)
        project_name = project_state.get("name") or getattr(self, "PROJECT_NAME", None)
        if not callable(set_active_project) or not project_name:
            raise RuntimeError("native Desktop/project identity is unavailable")
        native_project = set_active_project(project_name)
        if native_project is None or native_project is False:
            raise RuntimeError(f"SetActiveProject returned no project ({project_name})")
        return native_project

    @staticmethod
    def _fields_reporter_from_project(oproject, design_name):
        """Get a reporter only from the verified requested native design."""
        route_errors = []
        routes = (
            ("GetActiveDesign", getattr(oproject, "GetActiveDesign", None), ()),
            ("SetActiveDesign", getattr(oproject, "SetActiveDesign", None), (design_name,)),
        )
        for route_name, route, args in routes:
            if not callable(route):
                route_errors.append(f"{route_name}=unavailable")
                continue
            try:
                odesign = route(*args)
                if odesign is None or odesign is False:
                    raise RuntimeError("returned no design")
                actual_name = _aedt_design_name(odesign)
                if actual_name != design_name:
                    raise RuntimeError(
                        f"design mismatch: expected={design_name}, actual={actual_name or '<empty>'}"
                    )
                reporter = odesign.GetModule("FieldsReporter")
                if reporter is None or reporter is False:
                    raise RuntimeError("returned no FieldsReporter")
                return reporter
            except Exception as error:
                route_errors.append(f"{route_name}={type(error).__name__}: {error}")
        raise RuntimeError("; ".join(route_errors))

    def _fresh_fields_reporter(self, max_attempts=3, retry_delay=2):
        """Reacquire FieldsReporter without re-running the completed EM solve."""
        last_error = None
        design_name = _aedt_design_name(getattr(self.design1, "design_name", ""))
        if not design_name:
            raise RuntimeError("FieldsReporter design identity is unavailable")

        for attempt in range(1, max_attempts + 1):
            candidates = []
            preferred = getattr(self, "_fields_reporter_project", None)
            if preferred is not None and preferred is not False:
                candidates.append(("preferred", preferred))
            try:
                native_project = self._native_project_handle()
                if all(native_project is not item[1] for item in candidates):
                    candidates.append(("cached", native_project))
            except Exception as error:
                last_error = error

            candidate_errors = []
            for label, native_project in candidates:
                try:
                    reporter = self._fields_reporter_from_project(
                        native_project, design_name
                    )
                    self._fields_reporter_project = native_project
                    return reporter
                except Exception as error:
                    candidate_errors.append(
                        f"{label}={type(error).__name__}: {error}"
                    )

            try:
                refreshed_project = self._refresh_native_project_handle()
                reporter = self._fields_reporter_from_project(
                    refreshed_project, design_name
                )
                self._fields_reporter_project = refreshed_project
                return reporter
            except Exception as error:
                candidate_errors.append(
                    f"refresh={type(error).__name__}: {error}"
                )

            last_error = RuntimeError("; ".join(candidate_errors))
            logging.warning(
                f"FieldsReporter reacquire failed (attempt {attempt}/{max_attempts}): "
                f"{last_error}"
            )
            if attempt < max_attempts:
                time.sleep(retry_delay)
        raise RuntimeError(
            f"FieldsReporter unavailable after {max_attempts} attempts: {last_error}"
        )

    def _add_field_expression(self, expr_name, stack_builder, max_attempts=3, retry_delay=2):
        """Build one named expression with a freshly acquired calculator handle."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                reporter = self._fresh_fields_reporter(max_attempts=1, retry_delay=0)
                try:
                    if reporter.DoesNamedExpressionExists(expr_name):
                        return expr_name
                except Exception:
                    pass
                reporter.CalcStack("clear")
                stack_builder(reporter)
                result = reporter.AddNamedExpression(expr_name, "Fields")
                if result is False:
                    try:
                        if reporter.DoesNamedExpressionExists(expr_name):
                            return expr_name
                    except Exception:
                        pass
                    raise RuntimeError(f"AddNamedExpression returned False ({expr_name})")
                return expr_name
            except Exception as e:
                last_error = e
                logging.warning(
                    f"field expression '{expr_name}' failed (attempt {attempt}/{max_attempts}): {e}"
                )
                if attempt < max_attempts:
                    time.sleep(retry_delay)
        raise RuntimeError(f"failed to register field expression '{expr_name}': {last_error}")

    def get_magnetic_parameter(self):
        params = [
            ["Matrix.L(Tx_winding,Tx_winding)", "Ltx", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)", "Lrx", "uH"],
            ["Matrix.L(Tx_winding,Rx_winding)", "M", "uH"],
            ["abs(Matrix.CplCoef(Tx_winding,Rx_winding))", "k", ""],
            ["Matrix.L(Tx_winding,Tx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", "Lmt", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)*(abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", "Lmr", "uH"],
            ["Matrix.L(Tx_winding,Tx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", "Llt", "uH"],
            ["Matrix.L(Rx_winding,Rx_winding)*(1-abs(Matrix.CplCoef(Tx_winding,Rx_winding))^2)", "Llr", "uH"],
            ["PerWindingSolidLoss(Tx_winding)", "Tx_loss", "W"],
            ["PerWindingSolidLoss(Rx_winding)", "Rx_loss", "W"],
        ]
        expressions = [p[0] for p in params]
        self.report1 = None
        export_path = None
        self.extraction_attempts["matrix"] = self.extraction_attempts.get("matrix", 0) + 1
        try:
            fd, export_path = tempfile.mkstemp(prefix="mft_rl_", suffix=".txt")
            os.close(fd)
            os.remove(export_path)
            started = time.time()
            exported = self.design1.export_rl_matrix(
                matrix_name="Matrix",
                output_file=export_path,
                is_format_default=False,
                width=24,
                precision=15,
                is_exponential=True,
                setup="Setup1",
                default_adaptive="LastAdaptive",
            )
            if exported is False:
                raise RuntimeError("export_rl_matrix returned False")
            if not os.path.isfile(export_path) or os.path.getsize(export_path) <= 0:
                raise RuntimeError("export_rl_matrix did not create a non-empty file")
            if os.path.getmtime(export_path) < started - 2:
                raise RuntimeError("export_rl_matrix returned a stale file")
            with open(export_path, encoding="utf-8", errors="strict") as exported_file:
                row = _parse_rl_matrix_export(
                    exported_file.read(), float(self.df_plus["freq"].iloc[0]))
            self.df1 = pd.DataFrame([row])
            self.extraction_backends["matrix"] = "export_rl_matrix"
        except Exception as export_error:
            logging.warning(
                f"[matrix] export_rl_matrix failed; trying SolutionData: {export_error}"
            )
            self.df1 = self._solution_data_frame(
                expressions,
                aliases=[p[1] for p in params],
                target_units={p[0]: p[2] for p in params if p[2]},
                report_category="AC Magnetic",
                report_context="Matrix",
                extraction_key="matrix",
            )
        finally:
            if export_path and os.path.isfile(export_path):
                try:
                    os.remove(export_path)
                except OSError:
                    pass
        # The historical pyaedt_library report path applied abs() to magnetic
        # parameters. Preserve the established dataset convention for mutual M.
        self.df1["M"] = self.df1["M"].abs()
        return self.df1

    def _export_field_report(self, report_name, Y_components):
        # Kept as a compatibility-shaped helper for callers; no AEDT report/file is created.
        target_units = {
            expression: ("T" if expression.startswith("B_") else "W")
            for expression in Y_components
        }
        return self._solution_data_frame(
            Y_components,
            target_units=target_units,
            report_category="Fields",
            extraction_key="loss",
        )

    def save_calculation(self):

        def _get_calculator_loss(obj, loss, name):
            assignment = obj if isinstance(obj, str) else obj.name
            name = f"P_{name}"

            def _build(reporter):
                reporter.EnterQty(loss)
                reporter.EnterVol(assignment)
                reporter.CalcOp("Integrate")

            return self._add_field_expression(name, _build)

        # ---- 1차 권선 손실 ----
        _get_calculator_loss(self.design1.Tx_windings_main[0].name, "EMLoss", "Tx_main_winding_inner")
        _get_calculator_loss(self.design1.Tx_windings_main[-1].name, "EMLoss", "Tx_main_winding_outer")
        if self.df_plus["N1_side"].iloc[0] > 0:
            _get_calculator_loss(self.design1.Tx_windings_side[0].name, "EMLoss", "Tx_side_winding_inner")
            _get_calculator_loss(self.design1.Tx_windings_side[-1].name, "EMLoss", "Tx_side_winding_outer")

        # ---- 2차 권선 손실 ----
        _get_calculator_loss(self.design1.Rx_windings_main[0].name, "EMLoss", "Rx_main_winding_inner")
        _get_calculator_loss(self.design1.Rx_windings_main[-1].name, "EMLoss", "Rx_main_winding_outer")
        if self.df_plus["N2_side"].iloc[0] > 0:
            _get_calculator_loss(self.design1.Rx_windings_side[0].name, "EMLoss", "Rx_side_winding_inner")
            _get_calculator_loss(self.design1.Rx_windings_side[-1].name, "EMLoss", "Rx_side_winding_outer")

        # ---- 플레이트 손실 (콜드플레이트 / 권선 냉각판) ----
        core_plate_exprs = []
        for p in self.design1.core_plates:
            core_plate_exprs.append(_get_calculator_loss(p.name, "EMLoss", p.name))
        wcp_exprs = []
        for p in self.design1.wcp_plates:
            wcp_exprs.append(_get_calculator_loss(p.name, "EMLoss", p.name))

        # ---- report1: Tx 권선 손실 ----
        Y_components = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer"]
        if self.df_plus["N1_side"].iloc[0] > 0:
            Y_components.append("P_Tx_side_winding_inner")
            Y_components.append("P_Tx_side_winding_outer")

        tx_components = list(Y_components)

        # ---- report2: Rx 권선 손실 ----
        rx_components = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer"]
        if self.df_plus["N2_side"].iloc[0] > 0:
            rx_components.append("P_Rx_side_winding_inner")
            rx_components.append("P_Rx_side_winding_outer")

        # Tx/Rx/plate를 한 번에 읽어 gRPC 결과 조회 횟수를 최소화한다.
        plate_exprs = core_plate_exprs + wcp_exprs
        all_components = tx_components + rx_components + plate_exprs
        df_all = self._export_field_report("calculator_report", all_components)
        df_original1 = df_all[tx_components]

        if self.df_plus["N1_side"].iloc[0] > 0:
            df = df_original1.iloc[:, -4:].copy()
            df.columns = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer", "P_Tx_side_winding_inner", "P_Tx_side_winding_outer"]
            self.df_calculator1 = df
        else:
            df = df_original1.iloc[:, -2:].copy()
            df.columns = ["P_Tx_main_winding_inner", "P_Tx_main_winding_outer"]
            df["P_Tx_side_winding_inner"] = 0
            df["P_Tx_side_winding_outer"] = 0
            self.df_calculator1 = df[["P_Tx_main_winding_inner", "P_Tx_main_winding_outer", "P_Tx_side_winding_inner", "P_Tx_side_winding_outer"]]

        df_original2 = df_all[rx_components]

        if self.df_plus["N2_side"].iloc[0] > 0:
            df = df_original2.iloc[:, -4:].copy()
            df.columns = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer", "P_Rx_side_winding_inner", "P_Rx_side_winding_outer"]
            self.df_calculator2 = df
        else:
            df = df_original2.iloc[:, -2:].copy()
            df.columns = ["P_Rx_main_winding_inner", "P_Rx_main_winding_outer"]
            df["P_Rx_side_winding_inner"] = 0
            df["P_Rx_side_winding_outer"] = 0
            self.df_calculator2 = df[["P_Rx_main_winding_inner", "P_Rx_main_winding_outer", "P_Rx_side_winding_inner", "P_Rx_side_winding_outer"]]

        # ---- report3: 플레이트 손실 ----
        if plate_exprs:
            df3 = df_all[plate_exprs].copy()
            df3.columns = plate_exprs
            P_core_plate = df3[core_plate_exprs].sum(axis=1) if core_plate_exprs else 0
            P_winding_plate = df3[wcp_exprs].sum(axis=1) if wcp_exprs else 0
            self.df_calculator3 = pd.DataFrame({
                "P_core_plate": P_core_plate if core_plate_exprs else [0],
                "P_winding_plate": P_winding_plate if wcp_exprs else [0],
            })
        else:
            self.df_calculator3 = pd.DataFrame({"P_core_plate": [0], "P_winding_plate": [0]})

    def _sym_cut_count(self, obj_name):
        """대칭 1/8 분할 절단면 수 (공용 로직 위임)"""
        return sym_cut_count(obj_name, self.df_plus)

    def _mirror_mult(self, obj_name):
        """대칭 loss 디자인에서 삭제된 미러 오브젝트 몫을 총계에 반영하는 배수.
        (y=0에 걸치지 않는 코어/플레이트/냉각판은 y<0 쪽 미러가 삭제되어 있으므로 x2)
        풀모델이면 항상 1 (모든 오브젝트가 실존)."""
        if not getattr(self, "loss_is_sym", False):
            return 1.0
        name = obj_name
        if name.startswith("Tx_main_wcp"):
            return 2.0  # _p만 잔존 (_n 미러 삭제)
        if name.startswith("core_plate") or (name.startswith("core_") and not name.startswith("core_plate")):
            return 1.0 if self._sym_cut_count(name) == 3 else 2.0  # y=0 스팬이면 미러 없음
        return 1.0

    def _phys_factor(self, expr_name, is_core_loss):
        """대칭 loss 디자인의 적분값 -> 실물값 환산 계수 (풀모델이면 1)"""
        if not getattr(self, "loss_is_sym", False):
            return 1.0
        # 표현식 이름에서 오브젝트 이름 추출: P_core_3 / P_turn_Rx_main_0_0 / P_Tx_main_group ...
        name = expr_name
        for prefix in ("P_turn_", "P_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        name = name.replace("_group", "")
        c = self._sym_cut_count(name)
        if is_core_loss:
            core_y = float(self.df_plus["core_y"].iloc[0])
            return (2 ** c) / (2 ** core_y)
        return (2 ** c) / 4.0

    def _calc_field_expr(self, obj_name, quantity, op, expr_name):
        """계산기: quantity를 오브젝트 볼륨에 대해 op(Integrate/Mean/Maximum) 후 named expression 등록.
        quantity="B_peak"는 위상 무관한 자속밀도 페이저 크기 (Mag_B는 Phase=0 순간값이라 부적합)."""
        def _build(reporter):
            if quantity == "B_peak":
                reporter.EnterQty("B")
                reporter.CalcOp("CmplxMag")
                reporter.CalcOp("Mag")
            else:
                reporter.EnterQty(quantity)
            reporter.EnterVol(obj_name)
            reporter.CalcOp(op)

        return self._add_field_expression(expr_name, _build)

    def _calc_group_loss(self, objs, expr_name, quantity="EMLoss"):
        """여러 오브젝트의 손실 적분 합을 하나의 named expression으로 등록"""
        def _build(reporter):
            for i, obj in enumerate(objs):
                name = obj if isinstance(obj, str) else obj.name
                reporter.EnterQty(quantity)
                reporter.EnterVol(name)
                reporter.CalcOp("Integrate")
                if i > 0:
                    reporter.CalcOp("+")

        return self._add_field_expression(expr_name, _build)

    @staticmethod
    def _select_explicit_turns(turns, count):
        """Select inner/outer turns once, preserving their original order."""
        turns = list(turns)
        if count < 0:
            candidates = turns
        elif count == 0:
            candidates = []
        else:
            candidates = turns[:count] + turns[-count:]

        selected = []
        seen_names = set()
        for turn in candidates:
            name = turn if isinstance(turn, str) else turn.name
            if name not in seen_names:
                seen_names.add(name)
                selected.append(turn)
        return selected

    def save_loss_reports(self):
        """
        loss 디자인 전용 추출:
          - 코어 그룹별 CoreLoss 적분 (P_core_i, P_core_total)
          - 코어 그룹별 B 평균/최대 (B_mean_core, B_max_core) - 자속밀도 sanity check
          - Tx 해석 전류 I1 (전압원이므로 해석 결과, 정격+자화 성분 검증용)
          - 권선 그룹 총손실 + explicit 턴별 손실 (열해석 배분용) -> self.loss_map
        """
        n_exp = int(self.df_plus["n_explicit_turns"].iloc[0])

        # ---- 코어손실 + B ----
        core_exprs = []
        b_mean_exprs = []
        b_max_exprs = []
        for c in self.design1.core_objs:
            core_exprs.append(self._calc_field_expr(c.name, "CoreLoss", "Integrate", f"P_{c.name}"))
            b_mean_exprs.append(self._calc_field_expr(c.name, "B_peak", "Mean", f"B_mean_{c.name}"))
            b_max_exprs.append(self._calc_field_expr(c.name, "B_peak", "Maximum", f"B_max_{c.name}"))

        # ---- 권선 그룹 총손실 + explicit 턴 손실 (열해석용) ----
        group_exprs = []
        turn_exprs = []
        plate_exprs = []
        group_exprs.append(self._calc_group_loss(self.design1.Tx_windings_main, "P_Tx_main_group"))
        group_exprs.append(self._calc_group_loss(self.design1.Rx_windings_main, "P_Rx_main_group"))
        if self.design1.Rx_windings_side:
            group_exprs.append(self._calc_group_loss(self.design1.Rx_windings_side, "P_Rx_side_group"))
        # 플레이트류 개별 손실: save_calculation이 이미 P_<name> 표현식을 만들었으므로 재사용
        for p in self.design1.core_plates + self.design1.wcp_plates:
            plate_exprs.append(f"P_{p.name}")
        # Tx는 전 턴 explicit (열모델에서 foil 그대로) -> 턴별 손실
        for w in self.design1.Tx_windings_main:
            turn_exprs.append(self._calc_field_expr(w.name, "EMLoss", "Integrate", f"P_turn_{w.name}"))
        # Rx explicit 턴 (안쪽 n개 / 바깥 n개)
        for grp in [self.design1.Rx_windings_main, self.design1.Rx_windings_side]:
            if not grp:
                continue
            explicit = self._select_explicit_turns(grp, n_exp)
            for w in explicit:
                turn_exprs.append(self._calc_field_expr(w.name, "EMLoss", "Integrate", f"P_turn_{w.name}"))

        all_exprs = core_exprs + b_mean_exprs + b_max_exprs + group_exprs + turn_exprs + plate_exprs
        df_loss = self._export_field_report("calculator_report_loss", all_exprs)
        vals = df_loss.iloc[0, -len(all_exprs):]
        vals.index = all_exprs
        self.loss_map = {k: float(v) for k, v in vals.items()}

        # 실물 기준(_phys) 환산: 대칭 loss 디자인이면 오브젝트별 절단면 수로 보정, 풀모델이면 x1
        b_factor = 0.5 if getattr(self, "loss_is_sym", False) else 1.0
        self.loss_map_phys = {}
        for e in core_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * self._phys_factor(e, is_core_loss=True)
        for e in group_exprs + turn_exprs + plate_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * self._phys_factor(e, is_core_loss=False)
        for e in b_mean_exprs + b_max_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * b_factor

        def _obj_of(expr):
            n = expr
            for pref in ("P_turn_", "P_"):
                if n.startswith(pref):
                    return n[len(pref):].replace("_group", "")
            return n

        # 총계 (대칭 모델의 삭제된 미러 몫 포함 - 실물 전체 기준)
        core_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e)) for e in core_exprs)
        cplate_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                           for e in plate_exprs if "core_plate" in e)
        wcp_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                        for e in plate_exprs if "wcp" in e)
        p_tx = self.loss_map_phys.get("P_Tx_main_group", 0.0)
        p_rxm = self.loss_map_phys.get("P_Rx_main_group", 0.0)
        p_rxs_one = self.loss_map_phys.get("P_Rx_side_group", 0.0)
        winding_total = p_tx + p_rxm + 2 * p_rxs_one  # 측면 링 2개 (좌우 대칭)

        b_mean = sum(self.loss_map_phys[e] for e in b_mean_exprs) / max(len(b_mean_exprs), 1)
        b_max = max((self.loss_map_phys[e] for e in b_max_exprs), default=0)

        # CSV에는 실물 기준 값을 기본으로 기록 (raw 대칭 적분값은 _raw 접미사)
        summary = {
            "P_core_total": [core_total],
            "P_core_plate_total": [cplate_total],
            "P_wcp_total": [wcp_total],
            "P_winding_total": [winding_total],
            "P_Rx_side_total": [2 * p_rxs_one],
            "B_mean_core": [b_mean], "B_max_core": [b_max],
        }
        for e in core_exprs + group_exprs + turn_exprs + plate_exprs:
            summary[e] = [self.loss_map_phys[e]]
            if getattr(self, "loss_is_sym", False):
                summary[f"{e}_raw"] = [self.loss_map[e]]

        # ---- Tx 해석 전류 (전압원 여자 검증용) ----
        I1_mag = float("nan")
        I1_phase = float("nan")
        try:
            current_mag = "mag(InputCurrent(Tx_winding))"
            current_phase = "ang_deg(InputCurrent(Tx_winding))"
            df_i = self._solution_data_frame(
                [current_mag, current_phase],
                aliases=["I1_mag_peak", "I1_phase_deg"],
                target_units={current_mag: "A", current_phase: "deg"},
                report_category="AC Magnetic",
                extraction_key="winding_current",
            )
            I1_mag = float(df_i["I1_mag_peak"].iloc[0])
            I1_phase = float(df_i["I1_phase_deg"].iloc[0])
        except Exception as e:
            logging.warning(f"Failed to extract Tx winding current: {e}")

        summary["I1_mag_peak"] = [I1_mag]
        summary["I1_phase_deg"] = [I1_phase]
        summary["phi_deg"] = [getattr(self, "phi_deg", float("nan"))]
        phase_used = getattr(self, "I2_phase_auto", None)
        summary["I2_phase_used_deg"] = [phase_used if phase_used is not None
                                        else float(self.df_plus["I2_phase_deg"].iloc[0])]

        self.df_loss_summary = pd.DataFrame(summary)
        return self.df_loss_summary

    def get_convergence_info(self, label):
        """Export full pass history and derive fail-closed convergence telemetry."""
        cols = {
            f"conv_passes_{label}": float("nan"),
            f"conv_consecutive_{label}": float("nan"),
            f"conv_error_pct_{label}": float("nan"),
            f"conv_delta_pct_{label}": float("nan"),
            f"mesh_tets_{label}": float("nan"),
        }
        try:
            if not self.project_path:
                raise RuntimeError("deterministic project path is unavailable")
            tolerance_columns = {
                "matrix": "matrix_percent_error",
                "loss": "percent_error",
            }
            tolerance_column = tolerance_columns.get(label)
            if tolerance_column is None:
                raise RuntimeError(f"unknown convergence stage: {label}")
            tolerance = float(self.df_plus[tolerance_column].iloc[0])
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise RuntimeError(f"invalid configured tolerance: {tolerance_column}")

            path = os.path.join(self.project_path, f"convergence_{label}.txt")
            if os.path.exists(path):
                os.remove(path)
            try:
                variation = self.design1.available_variations.nominal_w_values
                if isinstance(variation, (list, tuple)):
                    variation = " ".join(str(v) for v in variation)
            except Exception:
                variation = ""
            exported = self.design1.odesign.ExportConvergence("Setup1", variation, path)
            if exported is False:
                raise RuntimeError("ExportConvergence returned False")
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                raise RuntimeError("ExportConvergence produced no history file")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                metrics = _parse_convergence_history(f, tolerance)
            cols[f"conv_passes_{label}"] = metrics["passes"]
            cols[f"conv_consecutive_{label}"] = metrics["consecutive"]
            cols[f"conv_error_pct_{label}"] = metrics["error_pct"]
            cols[f"conv_delta_pct_{label}"] = metrics["delta_pct"]
            cols[f"mesh_tets_{label}"] = metrics["mesh_tets"]
        except Exception as e:
            logging.warning(f"convergence info extraction failed ({label}): {e}")
        return pd.DataFrame({k: [v] for k, v in cols.items()})

    def _log_recent_aedt_messages(self, label):
        try:
            messages = self.design1.odesktop.GetMessages(
                self.PROJECT_NAME, self.design1.design_name, 0
            )
            for message in list(messages)[-10:]:
                logging.warning(f"[{label}][AEDT] {message}")
        except Exception as message_error:
            logging.warning(f"[{label}] AEDT messages unavailable: {message_error}")

    def _native_desktop_handle(self):
        """Return the original Desktop handle, never a copied-design proxy."""
        candidates = [getattr(self, "desktop", None)]
        project_wrapper = getattr(self, "project", None)
        try:
            project_state = vars(project_wrapper)
        except TypeError:
            project_state = {}
        candidates.append(project_state.get("desktop"))
        for candidate in candidates:
            odesktop = getattr(candidate, "odesktop", None)
            if odesktop is not None and odesktop is not False and callable(
                    getattr(odesktop, "SetActiveProject", None)):
                return odesktop
        raise RuntimeError("original native AEDT Desktop handle is unavailable")

    def _verified_native_maxwell_setup(self, odesktop, setup_name="Setup1"):
        """Resolve the exact active Maxwell setup through fresh native handles."""
        expected_project = str(getattr(self, "PROJECT_NAME", "") or "").strip()
        expected_design = _aedt_design_name(
            getattr(self.design1, "design_name", "")
        )
        if not expected_project or not expected_design:
            raise RuntimeError("native analysis project/design identity is unavailable")

        oproject = odesktop.SetActiveProject(expected_project)
        if oproject is None or oproject is False:
            raise RuntimeError(f"SetActiveProject returned no project ({expected_project})")
        actual_project = str(oproject.GetName() or "").strip()
        if actual_project != expected_project:
            raise _AedtIdentityMismatch(
                "analysis project identity mismatch: "
                f"expected={expected_project}, actual={actual_project or '<empty>'}"
            )

        odesign = oproject.SetActiveDesign(expected_design)
        if odesign is None or odesign is False:
            raise RuntimeError(f"SetActiveDesign returned no design ({expected_design})")
        actual_design = _aedt_design_name(odesign)
        if actual_design != expected_design:
            raise _AedtIdentityMismatch(
                "analysis design identity mismatch: "
                f"expected={expected_design}, actual={actual_design or '<empty>'}"
            )
        design_type = str(odesign.GetDesignType() or "")
        solution_type = str(odesign.GetSolutionType() or "")
        if design_type != "Maxwell 3D" or not _is_ac_magnetic_solution(solution_type):
            raise _AedtIdentityMismatch(
                "analysis design physics mismatch: "
                f"type={design_type!r}, solution={solution_type!r}"
            )
        analysis = odesign.GetModule("AnalysisSetup")
        if analysis is None or analysis is False:
            raise RuntimeError("active design returned no AnalysisSetup module")
        setups = tuple(str(name) for name in (analysis.GetSetups() or []))
        if setups != (setup_name,):
            raise RuntimeError(
                f"native analysis setup mismatch: expected={(setup_name,)}, actual={setups}"
            )
        return oproject, odesign

    def _validated_matrix_hpc_acf(self):
        """Return the matrix solve's exact 4-core/one-engine DSO configuration."""
        matrix_design = getattr(self, "design_matrix", None)
        solver = getattr(matrix_design, "solver_instance", None)
        working_directory = str(getattr(solver, "working_directory", "") or "").strip()
        if not working_directory:
            raise RuntimeError("authoritative matrix HPC working directory is unavailable")
        path = os.path.abspath(os.path.join(working_directory, "pyaedt_config.acf"))
        if not os.path.isfile(path):
            raise RuntimeError(f"authoritative matrix HPC ACF is missing: {path}")
        if os.path.getsize(path) <= 0 or os.path.getsize(path) > 65536:
            raise RuntimeError(f"authoritative matrix HPC ACF has invalid size: {path}")
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            text = stream.read()

        expected = {
            "ConfigName": "'pyaedt_config'",
            "DesignType": "'Maxwell 3D'",
            "MachineName": "'localhost'",
            "NumEngines": str(int(self.NUM_TASK)),
            "NumCores": str(int(self.NUM_CORE)),
            "NumGPUs": "0",
            "UseAutoSettings": "True",
        }
        mismatches = {}
        for key, value in expected.items():
            matches = re.findall(
                rf"(?m)^\s*{re.escape(key)}\s*=\s*([^\r\n]+?)\s*$", text
            )
            if matches != [value]:
                mismatches[key] = {"expected": value, "actual": matches}
        dso_begin_count = text.count("$begin 'DSOConfig'")
        dso_end_count = text.count("$end 'DSOConfig'")
        if dso_begin_count != 1 or dso_end_count != 1:
            mismatches["DSOConfig"] = {
                "expected": {"begin": 1, "end": 1},
                "actual": {"begin": dso_begin_count, "end": dso_end_count},
            }
        if mismatches:
            raise RuntimeError(
                f"authoritative matrix HPC ACF contract mismatch: {mismatches}"
            )
        return path

    def _restore_native_maxwell_dso(
            self, registry_key, original_config, max_attempts=5,
            retry_delay=0.5, sleeper=time.sleep):
        """Restore the pre-solve Maxwell DSO without ever touching Analyze."""
        if not original_config:
            return
        errors = []
        for attempt in range(1, int(max_attempts) + 1):
            try:
                odesktop = self._native_desktop_handle()
                result = odesktop.SetRegistryString(registry_key, original_config)
                if result is False:
                    raise RuntimeError("SetRegistryString returned False")
                actual = str(odesktop.GetRegistryString(registry_key) or "").strip()
                if actual != original_config:
                    raise RuntimeError(
                        "DSO restore readback mismatch: "
                        f"expected={original_config!r}, actual={actual!r}"
                    )
                return
            except Exception as error:
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                if attempt < int(max_attempts):
                    sleeper(max(0.0, float(retry_delay)) * (2 ** (attempt - 1)))
        raise RuntimeError("native Maxwell DSO restore failed: " + "; ".join(errors))

    def _prepare_copied_loss_native_analysis(
            self, setup_name="Setup1", max_attempts=5, timeout_s=30.0,
            initial_retry_delay=0.5, clock=time.monotonic, sleeper=time.sleep):
        """Retry only copied-loss solve preflight; never dispatch a solve here."""
        acf_path = self._validated_matrix_hpc_acf()
        registry_key = r"Desktop/ActiveDSOConfigurations/Maxwell 3D"
        deadline = clock() + max(0.0, float(timeout_s))
        original_config = None
        config_may_be_active = False
        attempts = []

        for attempt in range(1, int(max_attempts) + 1):
            if attempt > 1 and clock() >= deadline:
                break
            try:
                odesktop = self._native_desktop_handle()
                _oproject, odesign = self._verified_native_maxwell_setup(
                    odesktop, setup_name=setup_name
                )
                running = odesktop.AreThereSimulationsRunning()
                if running is not False:
                    raise RuntimeError(
                        f"AEDT reports an overlapping simulation: {running!r}"
                    )
                active = odesktop.GetRegistryString(registry_key)
                if active is None or active is False:
                    raise RuntimeError("GetRegistryString returned no active DSO")
                active = str(active).strip()
                if not active:
                    raise RuntimeError("GetRegistryString returned an empty active DSO")
                if original_config is None:
                    original_config = active

                loaded = odesktop.SetRegistryFromFile(acf_path)
                if loaded is False:
                    raise RuntimeError("SetRegistryFromFile returned False")
                config_may_be_active = True
                selected = odesktop.SetRegistryString(registry_key, "pyaedt_config")
                if selected is False:
                    raise RuntimeError("SetRegistryString returned False")
                actual = str(odesktop.GetRegistryString(registry_key) or "").strip()
                if actual != "pyaedt_config":
                    raise RuntimeError(
                        "native HPC DSO readback mismatch: "
                        f"expected='pyaedt_config', actual={actual!r}"
                    )
                _oproject, odesign = self._verified_native_maxwell_setup(
                    odesktop, setup_name=setup_name
                )
                return {
                    "odesktop": odesktop,
                    "odesign": odesign,
                    "registry_key": registry_key,
                    "original_config": original_config,
                    "acf_path": acf_path,
                }
            except _AedtIdentityMismatch:
                if config_may_be_active and original_config:
                    self._restore_native_maxwell_dso(
                        registry_key, original_config, sleeper=sleeper
                    )
                raise
            except Exception as error:
                attempts.append(
                    f"attempt {attempt}: {type(error).__name__}: {error}"
                )
                now = clock()
                if attempt >= int(max_attempts) or now >= deadline:
                    break
                logging.warning(
                    "copied-loss native analysis preflight failed "
                    f"(attempt {attempt}/{int(max_attempts)}): {error}"
                )
                sleeper(min(
                    max(0.0, float(initial_retry_delay)) * (2 ** (attempt - 1)),
                    max(0.0, deadline - now),
                ))

        restore_error = None
        if config_may_be_active and original_config:
            try:
                self._restore_native_maxwell_dso(
                    registry_key, original_config, sleeper=sleeper
                )
            except Exception as error:
                restore_error = f"{type(error).__name__}: {error}"
        raise RuntimeError(
            "copied-loss native analysis preflight failed closed; "
            f"attempts={attempts}, restore_error={restore_error}"
        )

    def _postcheck_copied_loss_native_analysis(
            self, setup_name="Setup1", max_attempts=5,
            retry_delay=0.5, sleeper=time.sleep):
        """Reacquire exact identities after dispatch without any solve retry."""
        errors = []
        for attempt in range(1, int(max_attempts) + 1):
            try:
                odesktop = self._native_desktop_handle()
                self._verified_native_maxwell_setup(
                    odesktop, setup_name=setup_name
                )
                running = odesktop.AreThereSimulationsRunning()
                if running is not False:
                    raise RuntimeError(
                        f"AEDT still reports a running simulation: {running!r}"
                    )
                return
            except _AedtIdentityMismatch:
                raise
            except Exception as error:
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                if attempt < int(max_attempts):
                    sleeper(max(0.0, float(retry_delay)) * (2 ** (attempt - 1)))
        raise RuntimeError(
            "copied-loss post-dispatch identity check failed: " + "; ".join(errors)
        )

    def analyze_and_extract(self, label, extractor):
        """Analyze exactly once; result-query failures never justify another solve."""
        def _analyze_once():
            if label != "loss" or not bool(getattr(
                    self, "loss_native_analyze_required", False)):
                self.solve_attempts[label] = self.solve_attempts.get(label, 0) + 1
                t0 = time.time()
                try:
                    # The original, non-copied design retains PyAEDT's supported
                    # high-level path. Setup.analyze() itself returns None.
                    analyze_result = self.design1.setup.analyze(cores=self.NUM_CORE)
                    if analyze_result is False:
                        raise RuntimeError(f"[{label}] Setup1 analyze returned False")
                except Exception:
                    self._log_recent_aedt_messages(label)
                    raise
                elapsed = time.time() - t0
                self.save_project()
                return elapsed

            context = self._prepare_copied_loss_native_analysis()
            try:
                # Preserve app.analyze() semantics: save before the only dispatch.
                self.save_project(strict=True)
                _oproject, dispatch_design = self._verified_native_maxwell_setup(
                    context["odesktop"], setup_name="Setup1"
                )
                active = str(context["odesktop"].GetRegistryString(
                    context["registry_key"]
                ) or "").strip()
                if active != "pyaedt_config":
                    raise RuntimeError(
                        f"[{label}] native HPC DSO changed before dispatch: {active!r}"
                    )
            except Exception:
                self._restore_native_maxwell_dso(
                    context["registry_key"], context["original_config"]
                )
                raise

            self.solve_attempts[label] = self.solve_attempts.get(label, 0) + 1
            t0 = time.time()
            dispatch_error = None
            analyze_result = None
            try:
                logging.info(
                    f"[{label}] native Analyze dispatch Setup1 blocking=True"
                )
                analyze_result = dispatch_design.Analyze("Setup1", True)
            except Exception as error:
                dispatch_error = error
            elapsed = time.time() - t0

            restore_error = None
            try:
                self._restore_native_maxwell_dso(
                    context["registry_key"], context["original_config"]
                )
            except Exception as error:
                restore_error = error

            if dispatch_error is not None:
                self._log_recent_aedt_messages(label)
                if restore_error is not None:
                    raise RuntimeError(
                        f"[{label}] native Analyze dispatch and DSO restore failed: "
                        f"dispatch={dispatch_error}; restore={restore_error}"
                    ) from dispatch_error
                raise dispatch_error
            if restore_error is not None:
                raise RuntimeError(
                    f"[{label}] native Analyze completed but DSO restore failed: "
                    f"{restore_error}"
                ) from restore_error
            if analyze_result is not None and (
                    type(analyze_result) is not int or analyze_result != 0):
                raise RuntimeError(
                    f"[{label}] native Analyze returned invalid status: "
                    f"{analyze_result!r}"
                )
            self._postcheck_copied_loss_native_analysis()
            self.save_project(strict=True)
            return elapsed

        elapsed = _analyze_once()
        extractor()
        return elapsed

    def get_execution_telemetry(self):
        """Return solve/extraction provenance alongside each training row."""
        row = {}
        for label in ("matrix", "loss"):
            row[f"{label}_solve_attempts"] = int(self.solve_attempts.get(label, 0))
            row[f"{label}_solution_queries"] = int(self.extraction_attempts.get(label, 0))
            row[f"{label}_extraction_backend"] = self.extraction_backends.get(label, "not_run")
        row["winding_current_solution_queries"] = int(
            self.extraction_attempts.get("winding_current", 0)
        )
        row["winding_current_extraction_backend"] = self.extraction_backends.get(
            "winding_current", "not_run"
        )
        row["matrix_conductor_policy"] = getattr(
            self, "matrix_conductor_policy", "not_recorded"
        )
        for key in (
            "matrix_winding_stranded_count",
            "matrix_conductor_mesh_operation_count",
            "matrix_plate_eddy_off_readback_count",
            "loss_winding_solid_update_count",
            "loss_winding_mesh_operation_count",
            "loss_conductor_mesh_operation_count",
            "loss_plate_eddy_on_readback_count",
        ):
            row[key] = int(getattr(self, key, -1))
        return pd.DataFrame([row])

    def save_results_to_csv(self, results_df, filename="simulation_results_260706.csv"):
        """Atomically save a per-run parquet part, then append a compatible legacy CSV."""
        results_df = results_df.copy()
        results_df["git_hash"] = GIT_HASH
        results_df["git_dirty"] = GIT_DIRTY
        results_df["pyaedt_library_git_hash"] = PYAEDT_LIBRARY_GIT_HASH
        results_df["pyaedt_library_git_dirty"] = PYAEDT_LIBRARY_GIT_DIRTY
        results_df["project_name"] = getattr(self, "PROJECT_NAME", "")
        results_df["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # RESULT_JSON 스트리밍이 동일 메타(특히 dedup 키)를 쓰도록 보관
        self.last_save_meta = {"git_hash": GIT_HASH,
                               "git_dirty": GIT_DIRTY,
                               "pyaedt_library_git_hash": PYAEDT_LIBRARY_GIT_HASH,
                               "pyaedt_library_git_dirty": PYAEDT_LIBRARY_GIT_DIRTY,
                               "project_name": results_df["project_name"].iloc[0],
                               "saved_at": results_df["saved_at"].iloc[0]}

        # Write the authoritative per-run part before touching the legacy CSV.
        # os.replace keeps collectors from observing a partially written parquet file.
        part = None
        part_tmp = None
        part_saved = False
        try:
            parts_dir = "results_parts_260706"
            os.makedirs(parts_dir, exist_ok=True)
            part_nonce = uuid.uuid4().hex
            part = os.path.join(parts_dir,
                                f"part_{datetime.now().strftime('%y%m%d_%H%M%S_%f')}_"
                                f"{part_nonce}_{os.getpid()}_{self.PROJECT_NAME}.parquet")
            part_tmp = part + f".tmp-{part_nonce}"
            results_df.to_parquet(part_tmp, index=False)
            os.replace(part_tmp, part)
            part_saved = True
        except Exception as e:
            logging.warning(f"parquet part write failed; legacy CSV fallback remains available: {e}")
            if part_tmp:
                try:
                    os.remove(part_tmp)
                except OSError:
                    pass

        lock_path = filename + ".lock"
        csv_saved = False
        with FileLock(lock_path):
            file_exists = os.path.isfile(filename)
            schema_matches = True
            if file_exists:
                with open(filename, "r", encoding="utf-8", newline="") as stream:
                    header = next(csv.reader(stream), [])
                schema_matches = header == list(results_df.columns)
                if not schema_matches:
                    logging.warning(
                        f"CSV schema mismatch; preserving {filename} and skipping append "
                        f"(existing={len(header)} columns, current={len(results_df.columns)} columns)"
                    )
            if schema_matches:
                results_df.to_csv(filename, mode="a", header=not file_exists, index=False)
                csv_saved = True

        if not part_saved and not csv_saved:
            fallback_path = "results_fallback_260706.jsonl"
            with FileLock(fallback_path + ".lock"):
                with open(fallback_path, "a", encoding="utf-8", newline="\n") as stream:
                    for _, row in results_df.iterrows():
                        stream.write(row.to_json(date_format="iso") + "\n")
            logging.warning(
                f"primary result sinks unavailable; saved JSONL fallback to {fallback_path}"
            )

        if csv_saved:
            logging.info(f"Results saved to {filename}")
        if part is not None and os.path.isfile(part):
            logging.info(f"Result part saved to {part}")

    def save_project(self, strict=False):
        errors = []
        try:
            result = self.design1.save_project()
            if result is False:
                raise RuntimeError("wrapper save_project returned False")
            return True
        except Exception as error:
            errors.append(f"wrapper={type(error).__name__}: {error}")
        try:
            result = self._native_project_handle().Save()
            if result is False:
                raise RuntimeError("native Project.Save returned False")
            return True
        except Exception as error:
            errors.append(f"native={type(error).__name__}: {error}")
        message = "Failed to save project: " + "; ".join(errors)
        if strict:
            raise RuntimeError(message)
        logging.warning(message)
        return False

    def close_project(self):
        # Capture solver/session descendants before AEDT release can orphan them.
        self.spawned_descendants.update(_snapshot_descendants())
        # keep_project=1 이면 솔루션 데이터를 보존한 채 닫는다
        # (cleanup_solution은 저장 프로젝트의 Results를 지워버림 - 삭제 예정일 때만 수행)
        try:
            keep = int(self.df_plus["keep_project"].iloc[0]) != 0
        except Exception:
            keep = False
        if not keep:
            try:
                self.design1.cleanup_solution()
            except Exception:
                pass
        else:
            try:
                self.save_project()
            except Exception:
                pass
        self.design1.close_project()
        self.desktop.release_desktop(close_projects=True, close_on_exit=True)

    def delete_project_folder(self, max_attempts=6, wait_s=10):
        """
        완료된 시뮬레이션 파일 삭제 (슈퍼컴퓨터 저장공간 확보용 - 반드시 지워져야 함).
        AEDT가 파일 핸들을 늦게 놓는 경우가 있어 재시도하며, .lock 등 부산물도 제거한다.
        """
        project_folder = self.project_path or os.path.join(os.getcwd(), "simulation", self.PROJECT_NAME)

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(wait_s)
            try:
                if os.path.isdir(project_folder):
                    shutil.rmtree(project_folder)
                # 폴더 밖에 생기는 부산물 (.lock, .auto 등)
                sim_dir = os.path.dirname(project_folder)
                for name in os.listdir(sim_dir):
                    if name.startswith(self.PROJECT_NAME + ".") and (
                            name.endswith(".lock") or name.endswith(".auto")
                            or name.endswith(".lock.txt") or name.endswith(".asol.lock")):
                        try:
                            os.remove(os.path.join(sim_dir, name))
                        except OSError:
                            pass
                if not os.path.isdir(project_folder):
                    logging.info(f"Successfully deleted project folder: {project_folder}")
                    return True
            except Exception as e:
                logging.warning(f"Delete attempt {attempt}/{max_attempts} failed for {project_folder}: {e}")

        logging.error(f"FAILED to delete project folder after {max_attempts} attempts: {project_folder}")
        return False


def _snapshot_descendants():
    """Map recursive child PIDs to (depth, create_time) for PID-reuse-safe cleanup."""
    try:
        import psutil

        root = psutil.Process()
        root_pid = root.pid
        snapshot = {}
        for child in root.children(recursive=True):
            depth = 1
            try:
                parent = child.parent()
                while parent is not None and parent.pid != root_pid:
                    depth += 1
                    parent = parent.parent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            try:
                create_time = child.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            snapshot[child.pid] = (depth, create_time)
        return snapshot
    except Exception:
        return {}


def _terminate_spawned_descendants(baseline_descendants, captured_descendants=None, wait_s=5):
    """Terminate every descendant created by this dedicated simulation run, deepest first."""
    try:
        import psutil

        captured = dict(captured_descendants or {})
        captured.update(_snapshot_descendants())
        spawned = {
            pid: metadata for pid, metadata in captured.items()
            if (
                pid != os.getpid()
                and (
                    pid not in baseline_descendants
                    or abs(baseline_descendants[pid][1] - metadata[1]) > 0.01
                )
            )
        }
        processes = []
        ordered = sorted(spawned.items(), key=lambda item: item[1][0], reverse=True)
        for pid, (depth, captured_create_time) in ordered:
            try:
                process = psutil.Process(pid)
                if abs(process.create_time() - captured_create_time) > 0.01:
                    logging.warning(f"skipping reused child pid={pid}")
                    continue
                logging.warning(
                    f"terminating leaked child pid={pid} name={process.name()} depth={depth}"
                )
                process.terminate()
                processes.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(processes, timeout=wait_s)
        for process in alive:
            try:
                logging.warning(f"killing leaked child pid={process.pid} name={process.name()}")
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if alive:
            psutil.wait_procs(alive, timeout=wait_s)
    except Exception as e:
        logging.warning(f"descendant cleanup failed: {e}")


def _finalize_run_cleanup(
        baseline_descendants, captured_descendants, sim=None,
        held=False, delete_project=False):
    """Stop this run's solver children before removing its disposable project."""
    if held:
        return
    _terminate_spawned_descendants(baseline_descendants, captured_descendants)
    if delete_project and sim is not None:
        try:
            sim.delete_project_folder(max_attempts=3, wait_s=1)
        except Exception as error:
            logging.exception(f"Error deleting project folder after descendant cleanup: {error}")


def log_failed_sample(input_df, reason, filename="failed_samples_260706.jsonl"):
    """Append one schema-independent failure record with the complete input row."""
    try:
        if isinstance(input_df, pd.DataFrame):
            if input_df.empty:
                parameters = {}
            else:
                parameters = json.loads(input_df.iloc[0].to_json(date_format="iso"))
        elif isinstance(input_df, dict):
            parameters = dict(input_df)
        else:
            parameters = {"value": str(input_df)}
        reason_text = str(reason)
        record = {
            "parameters": parameters,
            "fail_reason": reason_text,
            "failure_stage": reason_text.split(":", 1)[0],
            "fail_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_hash": GIT_HASH,
            "git_dirty": GIT_DIRTY,
            "pyaedt_library_git_hash": PYAEDT_LIBRARY_GIT_HASH,
            "pyaedt_library_git_dirty": PYAEDT_LIBRARY_GIT_DIRTY,
        }
        lock_path = filename + ".lock"
        with FileLock(lock_path):
            with open(filename, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    except Exception as e:
        logging.warning(f"failed-sample logging failed: {e}")


def run_one_loop(param=None, model_only=False, hold=False, golden=False, overrides=None):
    """
    param 이 None  -> 랜덤 파라미터 1회 (검증 실패 시 재추첨), 완료 후 프로젝트 삭제
    param 이 dict 등 -> 해당 값으로 1회 (fixed 모드), 프로젝트 폴더 보존
    model_only=True -> 모델링/셋업까지만 하고 해석은 생략 (지오메트리 확인용)
    """
    fixed_mode = param is not None
    sim = None
    desktop = None
    held = [False]  # hold 성공 시 finally에서 desktop을 닫지 않기 위한 플래그
    delete_project_on_exit = not (fixed_mode or hold or model_only)
    baseline_descendants = _snapshot_descendants()
    try:
        # pyDesktop을 context manager로 쓰면 release_desktop 이후 __exit__에서
        # close_on_exit 속성 오류가 발생하므로 직접 생성하고 finally에서 해제한다.
        desktop = pyDesktop(version=None, non_graphical=GUI, close_on_exit=True, new_desktop=True)

        sim = Simulation(desktop=desktop)

        sim.create_simulation_name()
        sim.create_project()

        if fixed_mode:
            sim.input_df = create_input_parameter(param)
            delete_project_on_exit = _project_delete_policy(
                sim.input_df, fixed_mode=True, hold=hold, model_only=model_only
            )
            # 위반 시 이유를 담아 ValueError raise
            _, sim.df_plus = validation_check(sim.input_df, strict=True)
        else:
            while True:
                sim.input_df = create_input_parameter(None)
                # CLI 오버라이드 (랜덤 모드에서도 --thermal/--loss 등 플래그 적용)
                if overrides:
                    for k, v in overrides.items():
                        sim.input_df[k] = v
                delete_project_on_exit = _project_delete_policy(
                    sim.input_df, fixed_mode=False, hold=hold, model_only=model_only
                )
                result, sim.df_plus, errors = validation_check(sim.input_df, return_errors=True)
                if result:
                    break
                # 기각 샘플 기록 (설계공간 경계 데이터)
                log_failed_sample(sim.input_df, "validation: " + " / ".join(errors))

        sim.full_model = int(sim.df_plus["full_model"].iloc[0]) != 0
        matrix_on = int(sim.df_plus["matrix_on"].iloc[0]) != 0
        loss_on = int(sim.df_plus["loss_on"].iloc[0]) != 0
        thermal_on = int(sim.df_plus["thermal_on"].iloc[0]) != 0

        def _build_em_design(design_name, mode):
            """EM 디자인 1개 생성: 지오메트리 + 여자 + 메시 + 경계 + 셋업"""
            sim.create_design(name=design_name)
            set_design_variables(sim.design1, sim.input_df)
            sim.create_core()
            sim.create_coil()
            sim.split_geometry()
            sim.create_coil_section()
            sim.assign_winding(mode=mode)
            sim.assign_coil()
            if mode == "matrix":
                sim.assign_matrix()
            else:
                sim.assign_core_loss()
            _configure_em_conductor_mesh(sim, mode)
            sim.assign_boundary()
            sim.create_setup(mode=mode)

        def _build_loss_by_copy():
            """maxwell_matrix를 복제해 loss_sym 디자인으로 전환 (모델링 절반 절약).
            레퍼런스: pyaedt_library/example/MFT_TAB second_simulation()"""
            import math as _m
            old_design = sim.design1  # 객체 핸들 리매핑용
            op = sim.project.desktop.odesktop.SetActiveProject(sim.project.name)
            before_names = {name for name, _raw in _project_design_entries(op)}
            # The reference implementation gives AEDT five seconds to commit
            # the solved source design before CopyDesign. Shorter matrix runs
            # exposed a copied design with no solution type or Setup1.
            old_design.save_project()
            time.sleep(5)
            op.CopyDesign("maxwell_matrix")
            op.Paste()
            new_design, copied_setup = _wait_for_ready_copied_loss_design(
                op, before_names,
                lambda name, solution: sim.project.create_design(
                    name=name, solver="maxwell3d", solution=solution,
                ),
            )
            sim.design1 = new_design

            # 모델링 때 래퍼에 저장된 객체 핸들들을 복제 디자인으로 리매핑
            # (save_calculation/save_loss_reports가 소비 - MFT_TAB 레퍼런스 패턴)
            _remap_copied_design_objects(
                old_design,
                new_design,
                (
                    "Tx_windings_main", "Tx_windings_side", "Tx_windings_side2",
                    "Tx_windings", "Rx_windings_main", "Rx_windings_side",
                    "Rx_windings_side2", "Rx_windings", "core_objs", "core_plates",
                    "core_pads", "wcp_plates", "wcp_pads",
                ),
            )

            # matrix 파라미터 제거 (loss 디자인에는 불필요한 연산)
            od = op.GetActiveDesign()
            try:
                od.GetModule("MaxwellParameterSetup").DeleteParameters(["Matrix"])
            except Exception as error:
                raise RuntimeError("matrix parameter deletion failed on loss copy") from error

            # 여자 전류를 loss_sym 페이저로 제자리 수정 (타입 동일: Current)
            I2 = float(sim.df_plus["I2_rated"].iloc[0])
            phase2 = getattr(sim, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(sim.df_plus["I2_phase_deg"].iloc[0])
            tx, rx = sim.design1.get_excitation(excitation_name=["Tx_winding", "Rx_winding"])
            tx["Current"] = f"{sim.loss_I1_peak}A"
            tx["Phase"] = f"{sim.loss_I1_phase_deg}deg"
            rx["Current"] = f"{I2 * _m.sqrt(2)}A"
            rx["Phase"] = f"{phase2}deg"
            sim.tx_winding, sim.rx_winding = tx, rx

            # 복제 디자인이 물려받은 matrix 해를 삭제 - 안 지우면 여자를 바꿔도
            # 솔버가 재해석 없이 '해 없음 완료'로 끝남 (로컬 랜덤 검증에서 3/3 재현)
            _delete_copied_solution_or_raise(
                sim.design1.design.odesign,
                op.GetActiveDesign(),
            )

            # 코어손실 + skin 메시(손실 정밀용) + 셋업 정밀값
            sim.assign_core_loss()
            _configure_loss_copy_skin_mesh(sim)
            sim.design1.setup = _configure_copied_loss_setup(
                copied_setup,
                max_passes=sim.df_plus["max_passes"].iloc[0],
                min_converged=sim.df_plus["min_converged"].iloc[0],
                percent_error=sim.df_plus["percent_error"].iloc[0],
            )
            # A copied pyDesign owns a separately constructed PyAEDT application
            # wrapper.  Its cached Desktop proxy is not trusted for solve dispatch;
            # analyze_and_extract uses the original Desktop and native design once.
            sim.loss_native_analyze_required = True

        result_parts = [sim.df_plus]
        total_time = 0.0

        # ---- design1: L/k 매트릭스 (전류원, 기존 방식) ----
        if matrix_on:
            _build_em_design("maxwell_matrix", "matrix")
            sim.design_matrix = sim.design1
            if not model_only:
                t_matrix = sim.analyze_and_extract("matrix", sim.get_magnetic_parameter)
                total_time += t_matrix
                result_parts.append(sim.df1)
                result_parts.append(sim.get_convergence_info("matrix"))
                result_parts.append(pd.DataFrame({"time_matrix": [t_matrix]}))

        # ---- design2: 손실 원샷 ----
        # loss_sym_on=1 (캠페인 기본): 대칭 1/8 + 전류 여자 (Tx = 부하+자화 페이저 합)
        #   -> 추출 시 오브젝트별 상수 보정으로 실물(_phys) 기록. 시간 ~4x 단축.
        # loss_sym_on=0 (최종 검증): 풀모델 + Tx 전압원 (검증된 물리 기준 경로)
        if loss_on:
            loss_sym = int(sim.df_plus["loss_sym_on"].iloc[0]) != 0 and not sim.full_model

            # P_target > 0 이면 design1의 누설(Lk = Llt_true)로 DAB 운전 위상을 역산해
            # I2 위상(-phi/2)을 자동 주입: phi = asin(P w Lk / (V1 V2'))
            P_t = float(sim.df_plus["P_target"].iloc[0])
            if P_t > 0 and not model_only:
                if not matrix_on:
                    raise RuntimeError("P_target>0 requires matrix_on=1 (Lk needed for phase calculation).")
                freq = float(sim.df_plus["freq"].iloc[0])
                V1 = float(sim.df_plus["V1_rms"].iloc[0])
                V2p = float(sim.df_plus["V2_rms"].iloc[0]) * int(sim.df_plus["N1"].iloc[0]) / int(sim.df_plus["N2"].iloc[0])
                Llt_true = float(sim.df1["Llt"].iloc[0]) * 1e-6 * (1.0 if sim.full_model else 2.0)
                omega = 2 * math.pi * freq
                arg = P_t * omega * Llt_true / (V1 * V2p) if V1 * V2p > 0 else 2.0
                if arg >= 1.0:
                    logging.warning(f"P_target unreachable with Lk={Llt_true*1e6:.1f}uH (sin(phi)={arg:.2f}>1) - phi=90deg capped")
                    phi_deg = 90.0
                else:
                    phi_deg = math.degrees(math.asin(arg))
                sim.I2_phase_auto = -phi_deg / 2.0
                sim.phi_deg = phi_deg
                logging.info(f"auto phase: Lk={Llt_true*1e6:.2f}uH, phi={phi_deg:.2f}deg -> I2 phase {sim.I2_phase_auto:.2f}deg")

            if loss_sym and not model_only:
                if not matrix_on:
                    raise RuntimeError("loss_sym_on=1 requires matrix_on=1 (Lm needed for magnetizing current).")
                # Tx 합성 전류: I1 = I_load∠phase2 + Im∠-90 (복소 합)
                freq = float(sim.df_plus["freq"].iloc[0])
                V1 = float(sim.df_plus["V1_rms"].iloc[0])
                I2 = float(sim.df_plus["I2_rated"].iloc[0])
                N1 = int(sim.df_plus["N1"].iloc[0])
                N2 = int(sim.df_plus["N2"].iloc[0])
                phase2 = getattr(sim, "I2_phase_auto", None)
                if phase2 is None:
                    phase2 = float(sim.df_plus["I2_phase_deg"].iloc[0])
                Lm_true = float(sim.df1["Lmt"].iloc[0]) * 1e-6 * 2.0  # 대칭 매트릭스 L은 실물의 1/2
                omega = 2 * math.pi * freq
                Im_peak = math.sqrt(2) * V1 / (omega * Lm_true) if Lm_true > 0 else 0.0
                I_load_peak = math.sqrt(2) * I2 * N2 / N1
                z = I_load_peak * complex(math.cos(math.radians(phase2)), math.sin(math.radians(phase2))) \
                    + Im_peak * complex(0, -1)
                sim.loss_I1_peak = abs(z)
                sim.loss_I1_phase_deg = math.degrees(math.atan2(z.imag, z.real))
                logging.info(f"loss_sym excitation: I_load={I_load_peak:.2f}A + Im={Im_peak:.2f}A "
                             f"-> I1={sim.loss_I1_peak:.2f}A ang {sim.loss_I1_phase_deg:.2f}deg")
            elif loss_sym and model_only:
                sim.loss_I1_peak = math.sqrt(2) * float(sim.df_plus["I1_rated"].iloc[0])
                sim.loss_I1_phase_deg = 0.0

            prev_full = sim.full_model
            if loss_sym:
                sim.loss_em_full = False
                sim.loss_is_sym = True
                if int(sim.df_plus.get("loss_from_copy", pd.Series([1])).iloc[0]):
                    _build_loss_by_copy()
                else:
                    _build_em_design("maxwell_loss", "loss_sym")
            else:
                sim.full_model = True
                sim.loss_em_full = True
                sim.loss_is_sym = False
                _build_em_design("maxwell_loss", "loss")
            sim.design_loss = sim.design1
            if not model_only:
                def _extract_loss_results():
                    sim.save_calculation()
                    sim.save_loss_reports()

                t_loss = sim.analyze_and_extract("loss", _extract_loss_results)
                total_time += t_loss
                result_parts += [sim.df_calculator1, sim.df_calculator2, sim.df_calculator3,
                                 sim.df_loss_summary, sim.get_convergence_info("loss"),
                                 pd.DataFrame({"time_loss": [t_loss]})]
            sim.full_model = prev_full

        # ---- design3: Icepak 열해석 (풀 지오메트리, EM 손실 주입) ----
        thermal_result_valid = not thermal_on
        if thermal_on and loss_on and not model_only:
            from module.thermal_260706 import run_thermal_analysis
            t0 = time.time()
            try:
                df_thermal = run_thermal_analysis(sim)
            except Exception as thermal_error:
                logging.exception(f"thermal stage failed: {thermal_error}")
                log_failed_sample(
                    sim.input_df,
                    f"thermal: {type(thermal_error).__name__}: {thermal_error}",
                )
                df_thermal = _thermal_failure_frame(thermal_error)
            thermal_result_valid = _thermal_result_is_valid(df_thermal)
            t_thermal = time.time() - t0
            total_time += t_thermal
            result_parts += [df_thermal, pd.DataFrame({"time_thermal": [t_thermal]})]

        if model_only:
            print(sim.df_plus)
            sim.save_project()
            logging.info(f"Model-only mode: project '{sim.PROJECT_NAME}' saved, skipping analysis.")
            if hold:
                held[0] = True
                logging.info(
                    f"HOLD model-only mode: project '{sim.PROJECT_NAME}' left open for inspection."
                )
                print(
                    f"\n=== HOLD: AEDT에 '{sim.PROJECT_NAME}' 모델이 열린 채 유지됩니다. "
                    "확인 후 직접 닫으세요. ==="
                )
                return True
            try:
                sim.close_project()
            except Exception as e:
                logging.exception(f"Error closing project: {e}")
            return

        simulation_time = pd.DataFrame({"time": [total_time]})
        result = pd.concat(
            result_parts + [sim.get_execution_telemetry(), simulation_time], axis=1
        )
        em_result_valid, em_validity_reason = _em_result_validation(
            result, matrix_on=matrix_on, loss_on=loss_on
        )
        result["result_valid_em"] = int(em_result_valid)
        result["em_validity_reason"] = em_validity_reason
        result["result_valid_thermal"] = (
            int(thermal_result_valid) if thermal_on else float("nan")
        )
        if not em_result_valid:
            logging.error(f"EM result rejected: {em_validity_reason}")
            log_failed_sample(sim.input_df, f"em_validation: {em_validity_reason}")

        persistence_error = None
        try:
            sim.save_results_to_csv(result)
            if golden:
                # golden case: 동일 기준 케이스를 주기적으로 재해석해 결과 표류(드리프트) 감지
                sim.save_results_to_csv(result, filename="golden_history_260706.csv")
        except Exception as e:
            logging.exception(f"Error saving results to CSV: {e}")
            persistence_error = e

        if fixed_mode or hold:
            print(result)
            sim.save_project()
        # 스케줄러 stdout 회수용: 결과 1행을 JSON 한 줄로 즉시 스트리밍
        # (랜덤 모드도 포함 - 태스크 완주를 기다리지 않고 샘플 단위로 데이터 회수 가능)
        try:
            d = json.loads(result.iloc[0].to_json())
            d.update(getattr(sim, "last_save_meta", {}))  # git_hash/project_name/saved_at (dedup 키)
            print("RESULT_JSON " + json.dumps(d), flush=True)
        except Exception as e:
            logging.warning(f"RESULT_JSON print failed: {e}")

        if persistence_error is not None:
            raise RuntimeError(f"result persistence failed: {persistence_error}") from persistence_error

        if hold:
            # 결과 확인용: AEDT와 프로젝트를 연 채로 종료 (사용자가 직접 닫을 때까지 유지)
            held[0] = True
            logging.info(f"HOLD mode: project '{sim.PROJECT_NAME}' left open in AEDT for inspection.")
            print(f"\n=== HOLD: AEDT에 '{sim.PROJECT_NAME}' 프로젝트가 열린 채 유지됩니다. 확인 후 직접 닫으세요. ===")
            return bool(em_result_valid and thermal_result_valid)

        try:
            sim.close_project()
        except Exception as e:
            logging.exception(f"Error closing project: {e}")

        # Partial thermal rows remain streamed and are useful for EM surrogates, but
        # --thermal --count N advances only on thermally valid rows.
        return bool(em_result_valid and thermal_result_valid)
    except Exception as e:
        logging.exception(f"run_one_loop failed: {e}")
        if sim is not None and getattr(sim, "input_df", None) is not None:
            log_failed_sample(sim.input_df, f"runtime: {e}")
        if fixed_mode:
            # fixed 모드에서는 실패를 조용히 넘기지 않는다
            raise
        if sim is not None:
            try:
                sim.close_project()
                time.sleep(1)
            except Exception:
                pass
        return False
    finally:
        spawned_descendants = _snapshot_descendants()
        if sim is not None:
            spawned_descendants.update(getattr(sim, "spawned_descendants", {}))
        if desktop is not None:
            try:
                if held[0]:
                    # HOLD: 프로젝트/AEDT는 열어둔 채 python의 gRPC 세션만 해제
                    # (이걸 안 하면 python 프로세스가 AEDT를 붙잡은 채 종료되지 않음)
                    desktop.release_desktop(close_projects=False, close_on_exit=False)
                else:
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                time.sleep(1)
            except Exception:
                pass
        # release가 조용히 실패하거나 solver가 re-parent되기 전에 PID를 확보해 둔다.
        # 이 프로세스가 이번 run에서 만든 자식만 회수하므로 다른 태스크에는 손대지 않는다.
        _finalize_run_cleanup(
            baseline_descendants,
            spawned_descendants,
            sim=sim,
            held=held[0],
            delete_project=delete_project_on_exit,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="MFT simulation (design 260706)")
    parser.add_argument("--fixed", action="store_true",
                        help="도면 기본값으로 1회 실행 (랜덤 루프 대신)")
    parser.add_argument("--params", type=str, default=None,
                        help="기본값 위에 덮어쓸 파라미터 JSON 파일 경로 (지정 시 fixed 모드)")
    parser.add_argument("--round", dest="round_corner", action="store_true", default=None,
                        help="권선 모서리 라운드 처리 on")
    parser.add_argument("--no-round", dest="round_corner", action="store_false",
                        help="권선 모서리 라운드 처리 off")
    parser.add_argument("--model-only", action="store_true",
                        help="모델링/셋업까지만 수행하고 해석은 생략 (지오메트리 확인용)")
    parser.add_argument("--full", action="store_true",
                        help="대칭(1/8 분할) 미적용 풀모델로 모델링/해석")
    parser.add_argument("--headless", action="store_true",
                        help="AEDT 창 없이 실행 (해석 중 GUI 조작으로 인한 블로킹 방지)")
    parser.add_argument("--count", type=int, default=None,
                        help="랜덤 모드에서 N회 성공 후 종료 (slurm_scheduler fea_bursty/packed 태스크용; 미지정 시 무한루프)")
    parser.add_argument(
        "--require-consecutive", action="store_true",
        help="abort a bounded random batch on its first unsuccessful sample",
    )
    parser.add_argument("--no-matrix", dest="matrix_on", action="store_false", default=None,
                        help="design1(L/k 매트릭스) 생략")
    parser.add_argument("--no-loss", dest="loss_on", action="store_false", default=None,
                        help="design2(손실 원샷) 생략")
    parser.add_argument("--thermal", dest="thermal_on", action="store_true", default=None,
                        help="design3(Icepak 열해석)까지 수행")
    parser.add_argument("--hold", action="store_true",
                        help="해석 완료 후 AEDT/프로젝트를 닫지 않고 유지 (결과 직접 확인용, 1회 실행)")
    parser.add_argument("--golden", action="store_true",
                        help="golden case(고정 기준 케이스) 1회 해석 후 golden_history CSV에 기록 (드리프트 감지용)")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="파라미터 오버라이드 (반복 가능, fixed/random 공용). 예: --set P_target=1e6 --set percent_error=1.0")
    return parser.parse_args()


def _parse_set_overrides(pairs):
    """--set KEY=VALUE 목록을 타입 변환된 dict로"""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--set 형식 오류 (KEY=VALUE 필요): {p}")
        k, v = p.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    global GUI

    args = parse_args()
    require_consecutive = bool(getattr(args, "require_consecutive", False))

    if require_consecutive and (args.count is None or args.count <= 0):
        raise ValueError("--require-consecutive requires a positive --count")

    if args.headless:
        GUI = True  # non_graphical=True

    if args.golden:
        # 고정 기준 케이스: 배치마다 같이 돌려 결과 표류를 시계열로 감시
        golden_path = os.path.join(BASE_DIR, "verification_params", "golden_case.json")
        with open(golden_path, "r", encoding="utf-8") as f:
            param = json.load(f)
        run_one_loop(param=param, golden=True)
        return

    fixed_mode = args.fixed or (args.params is not None)

    # model-only는 지오메트리 확인용이므로 항상 1회 실행
    if args.model_only and not fixed_mode:
        run_one_loop(param=None, model_only=True, hold=args.hold)
        return

    if fixed_mode:
        param = {}
        if args.params is not None:
            with open(args.params, "r", encoding="utf-8") as f:
                param.update(json.load(f))
        if args.round_corner is not None:
            param["round_corner"] = 1 if args.round_corner else 0
        if args.full:
            param["full_model"] = 1
        if args.matrix_on is not None:
            param["matrix_on"] = 1 if args.matrix_on else 0
        if args.loss_on is not None:
            param["loss_on"] = 1 if args.loss_on else 0
        if args.thermal_on is not None:
            param["thermal_on"] = 1 if args.thermal_on else 0
        param.update(_parse_set_overrides(args.set_overrides))

        # 데스크톱 불안정(라이선스 폭풍 중 pyaedt 핸들 유실)은 새 데스크톱으로 재시도 가치가 있음
        for attempt in range(1, 4):
            try:
                completed = run_one_loop(
                    param=param, model_only=args.model_only, hold=args.hold
                )
                # 성공 시 pyaedt atexit teardown 크래시가 exit 1로 둔갑시키는 것 방지
                if not args.hold:
                    sys.stdout.flush(); sys.stderr.flush()
                    exit_code = 0 if args.model_only else _completion_exit_code(
                        int(bool(completed)), 1
                    )
                    os._exit(exit_code)
                return
            except RuntimeError as e:
                if "desktop unstable" in str(e) and attempt < 3:
                    print(f"\n=== 데스크톱 불안정으로 실패 -> 새 세션으로 재시도 ({attempt + 1}/3) ===\n",
                          flush=True)
                    time.sleep(30)
                    continue
                raise
        return

    # 랜덤 스윕: --count N 이면 N회 성공 후 종료 (slurm_scheduler 태스크의 완료 감지용),
    # 미지정 시 기존처럼 무한루프.
    # CLI 플래그는 랜덤 모드에도 적용 (--thermal 은 손실 해석이 선행돼야 하므로 loss도 자동 활성화)
    overrides = {}
    if args.matrix_on is not None:
        overrides["matrix_on"] = 1 if args.matrix_on else 0
    if args.loss_on is not None:
        overrides["loss_on"] = 1 if args.loss_on else 0
    if args.thermal_on:
        overrides["thermal_on"] = 1
        overrides.setdefault("loss_on", 1)
        overrides.setdefault("matrix_on", 1)
    if args.round_corner is not None:
        overrides["round_corner"] = 1 if args.round_corner else 0
    if args.full:
        overrides["full_model"] = 1
    if args.hold:
        overrides["keep_project"] = 1
    overrides.update(_parse_set_overrides(args.set_overrides))

    successes = 0
    attempts = 0
    max_attempts = args.count * 3 if args.count else None

    while True:

        ok = False
        try:
            ok = run_one_loop(param=None, model_only=args.model_only, hold=args.hold,
                              overrides=overrides or None)
            if ok:
                successes += 1
                if args.hold:
                    # 확인용 1회 실행 - AEDT를 연 채로 종료
                    return
            elif args.hold:
                print(f"\n=== HOLD 모드: 이번 샘플 실패 (라이선스/해석 오류 등) -> "
                      f"새 랜덤 샘플로 재시도합니다 (시도 {attempts + 1}) ===\n", flush=True)
        except Exception as e:
            logging.exception(f"Error running simulation: {e}")

        finally:
            time.sleep(10)

        attempts += 1
        if args.count is not None:
            if require_consecutive and not ok:
                logging.error(
                    f"Consecutive gate failed on attempt {attempts}; "
                    f"completed {successes}/{args.count} simulations.")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            if successes >= args.count:
                logging.info(f"Completed {successes}/{args.count} simulations.")
                # os._exit: pyaedt atexit 핸들러의 간헐적 teardown 크래시가
                # 성공한 런을 실패(exit 1)로 둔갑시키는 것 방지 (파일은 이미 flush됨)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            if attempts >= max_attempts:
                logging.error(f"Reached max attempts ({attempts}) with only {successes}/{args.count} successes.")
                sys.stdout.flush()
                sys.stderr.flush()
                # RESULT_JSON rows already emitted by partial batches remain harvestable,
                # but the scheduler must not count a short batch as fully completed.
                os._exit(_completion_exit_code(successes, args.count))


if __name__ == "__main__":
    main()
