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
import hashlib
import argparse
import uuid
import tempfile
import time

_PROCESS_STARTED_MONOTONIC = time.monotonic()
_PROCESS_STARTED_EPOCH_S = time.time()

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
from datetime import datetime, timezone

_PROCESS_STARTED_AT_UTC = datetime.fromtimestamp(
    _PROCESS_STARTED_EPOCH_S, tz=timezone.utc
).isoformat()

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
from module.core_material_contract import (
    LEG_STACKING_DIRECTION,
    PHYSICS_DATA_REVISION,
    YOKE_STACKING_DIRECTION,
    faraday_lumped_core_reference,
    material_flux_density_t,
    native_lamination_material_specs,
    sinusoidal_b_peak_material_t,
    square_wave_b_material_t,
    validate_native_lamination_readback,
)
from module.thermal_probe_contract import (
    RX_SIDE_FACE_MAX_RULE,
    RX_SIDE_FACE_MEAN_RULE,
    RX_SIDE_FACE_PROBE_CONTRACT_VERSION,
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
B_POWER_REFERENCE_VARIABLE = "B_power_reference"
B_POWER_REFERENCE_T = 1.0


def _grpc_calcop_unavailability_reason(error):
    """Return the structured fail-soft reason for a CalcOp-class failure."""
    detail = " ".join(str(error).split()) or type(error).__name__
    normalized = detail.casefold()
    if "grpc" not in normalized or "calcop" not in normalized:
        return ""
    return f"grpc_calcop_unavailable:{detail}"


def _raw_aedt_material_props(materials, material_name):
    """Return fresh native material data, bypassing PyAEDT's assigned cache."""
    manager = getattr(materials, "omaterial_manager", None)
    if manager is None or not callable(getattr(manager, "GetData", None)):
        raise RuntimeError("AEDT material manager cannot provide native readback")
    try:
        raw = list(manager.GetData(str(material_name)))
        from ansys.aedt.core.generic.data_handlers import _arg2dict

        parsed = {}
        _arg2dict(raw, parsed)
    except Exception as exc:
        raise RuntimeError(
            f"native AEDT material readback failed for {material_name!r}"
        ) from exc
    if len(parsed) != 1:
        raise RuntimeError(
            f"unexpected native material payload for {material_name!r}: "
            f"top-level keys={list(parsed)}"
        )
    props = next(iter(parsed.values()))
    if not isinstance(props, dict) or not props:
        raise RuntimeError(
            f"native AEDT material payload is empty for {material_name!r}"
        )
    return props


def _core_group_index(object_name):
    """Extract the depth-group index from legacy or segmented core names."""
    match = re.fullmatch(r"core_(\d+)(?:_(?:leg_(?:left|center|right)|yoke_(?:top|bottom)))?", str(object_name))
    if not match:
        raise RuntimeError(f"unrecognized core object name: {object_name!r}")
    return int(match.group(1))


_NATIVE_CORE_REGION_ORDER = (
    "leg_left",
    "leg_center",
    "leg_right",
    "yoke_bottom",
    "yoke_top",
)


def _native_core_report_plan(
        core_groups, sym_cut_counter, *, require_complete_groups=False):
    """Validate native core coverage and topology by symmetry-cut count.

    Full models require all five canonical pieces; symmetry models cover the
    exact surviving subset after the geometry split.  The returned batches are
    retained for diagnostic provenance only; production loss validation uses
    the independent Python Faraday/POWERLITE/mass reference.
    """
    ordered_groups = []
    all_names = []
    batches = {}
    for group_index, pieces in sorted(core_groups.items()):
        by_name = {}
        for piece in pieces:
            name = str(piece if isinstance(piece, str) else piece.name)
            if name in by_name:
                raise RuntimeError(
                    f"duplicate native core report object: {name!r}"
                )
            by_name[name] = piece

        expected_names = tuple(
            f"core_{group_index}_{region}"
            for region in _NATIVE_CORE_REGION_ORDER
        )
        unexpected = sorted(set(by_name) - set(expected_names))
        missing = sorted(set(expected_names) - set(by_name))
        if unexpected or (require_complete_groups and missing):
            raise RuntimeError(
                "native core report coverage mismatch for group "
                f"{group_index}: missing={missing}, unexpected={unexpected}"
            )

        retained_names = tuple(name for name in expected_names if name in by_name)
        if not retained_names:
            raise RuntimeError(
                f"native core report group {group_index} has no retained objects"
            )
        ordered_pieces = tuple(by_name[name] for name in retained_names)
        cut_counts = {int(sym_cut_counter(name)) for name in retained_names}
        if len(cut_counts) != 1:
            raise RuntimeError(
                "native core group spans multiple symmetry-cut counts: "
                f"group={group_index}, cuts={sorted(cut_counts)}"
            )
        cut_count = cut_counts.pop()
        ordered_groups.append((int(group_index), ordered_pieces))
        all_names.extend(retained_names)
        batches.setdefault(cut_count, []).extend(ordered_pieces)

    if len(all_names) != len(set(all_names)):
        raise RuntimeError("native core report coverage contains duplicate objects")
    if not all_names:
        raise RuntimeError("native core report coverage is empty")

    membership_text = "\n".join(all_names)
    return {
        "groups": tuple(ordered_groups),
        "batches": tuple(
            (cut_count, tuple(pieces))
            for cut_count, pieces in sorted(batches.items())
        ),
        "object_names": tuple(all_names),
        "membership_sha256": hashlib.sha256(
            membership_text.encode("utf-8")
        ).hexdigest(),
    }


def _native_b_power_restore_factor(cut_count, core_y, loss_is_sym):
    """Return the exact full-model volume/amplitude factor for one cut class."""
    if not loss_is_sym:
        return 1.0
    cut_count = int(cut_count)
    core_y = float(core_y)
    if cut_count not in (0, 1, 2, 3):
        raise ValueError(f"invalid native symmetry-cut count: {cut_count!r}")
    mirror_factor = 1.0 if cut_count == 3 else 2.0
    return (2.0 ** cut_count) / (2.0 ** core_y) * mirror_factor


def _raw_aedt_object_attribute(obj, property_name):
    """Read one geometry attribute directly from AEDT's native editor."""
    editor = getattr(obj, "_oeditor", None)
    getter = getattr(editor, "GetPropertyValue", None)
    if not callable(getter):
        raise RuntimeError(
            f"native editor readback unavailable for {getattr(obj, 'name', obj)!r}"
        )
    try:
        value = getter(
            "Geometry3DAttributeTab", str(obj.name), str(property_name)
        )
    except Exception as exc:
        raise RuntimeError(
            f"native object {property_name} readback failed for {obj.name!r}"
        ) from exc
    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f"native object {property_name} readback empty for {obj.name!r}"
        )
    return str(value).strip().strip('"')


def _sheet_area_model_units(sheet):
    """Read one sheet area through PyAEDT's FacePrimitive API."""
    name = getattr(sheet, "name", sheet)
    try:
        faces = list(sheet.faces)
    except Exception as exc:
        raise RuntimeError(
            f"cannot enumerate faces for sheet {name!r}"
        ) from exc
    if len(faces) != 1:
        raise RuntimeError(
            f"sheet {name!r} must expose exactly one face, got {len(faces)}"
        )
    try:
        area = abs(float(faces[0].area))
    except Exception as exc:
        raise RuntimeError(
            f"cannot read face area for sheet {name!r}"
        ) from exc
    if not math.isfinite(area) or area <= 0:
        raise RuntimeError(f"invalid sheet area for {name!r}: {area!r}")
    return area


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
    "uv": 1e-6, "mv": 1e-3, "v": 1.0, "kv": 1e3,
    "uwb": 1e-6, "mwb": 1e-3, "wb": 1.0,
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


def _b_power_volume_integral_si(
        value, unit, exponent, *, normalized_by_one_tesla=False):
    """Normalize a B-power integral to the numeric T**y*m**3 moment.

    When the calculator expression uses ``(B / 1 tesla)**y``, its reported
    unit is volume only while its numerical value is exactly the desired
    moment with B expressed in tesla.
    """
    number = float(value)
    exponent = float(exponent)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"invalid B-power volume integral: {value!r}")
    raw = str(unit or "").strip().lower().replace(" ", "")
    raw = raw.replace("³", "^3").replace("tesla", "t")
    if (not normalized_by_one_tesla) and (not raw or "t" not in raw):
        raise RuntimeError(
            f"B-power integral unit lacks tesla basis: {unit!r}"
        )
    if (not normalized_by_one_tesla) and not any(
            token in raw for token in (str(exponent), f"^{exponent:g}")):
        # Some AEDT builds omit a fractional exponent from the display unit.
        # Accept a generic powered-T marker, but never silently accept plain T.
        if "t^" not in raw and "pow" not in raw:
            raise RuntimeError(
                f"B-power integral unit lacks exponent {exponent:g}: {unit!r}"
            )
    if "mm^3" in raw or "mm3" in raw:
        volume_factor = 1e-9
    elif "cm^3" in raw or "cm3" in raw:
        volume_factor = 1e-6
    elif "meter^3" in raw or "m^3" in raw or re.search(r"(?<![a-z])m3(?![a-z])", raw):
        volume_factor = 1.0
    else:
        raise RuntimeError(
            f"unsupported or ambiguous B-power volume unit: {unit!r}"
        )
    return number * volume_factor


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
STACKING_REVISION_SIDE_THERMAL_TEMPERATURE_COLUMNS = (
    "Tprobe_Rx_side_leeward_mean",
    "Tprobe_Rx_side_outer_max",
    "Tprobe_Rx_side_outer_mean",
    "Tprobe_Rx_side_inner_max",
    "Tprobe_Rx_side_inner_mean",
)


def _thermal_result_is_valid(frame, physics_data_revision=None):
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
        physics_revision = str(
            physics_data_revision
            if physics_data_revision is not None
            else frame.get(
                "physics_data_revision", pd.Series([""])
            ).iloc[0]
        ).strip()
        if (
            required_mask & group_bits["T_max_Rx_side"]
            and physics_revision == PHYSICS_DATA_REVISION
        ):
            required.extend(
                STACKING_REVISION_SIDE_THERMAL_TEMPERATURE_COLUMNS
            )
            if str(frame["thermal_rx_side_probe_contract_version"].iloc[0]) != (
                RX_SIDE_FACE_PROBE_CONTRACT_VERSION
            ):
                return False
            if str(frame["thermal_rx_side_probe_max_rule"].iloc[0]) != (
                RX_SIDE_FACE_MAX_RULE
            ):
                return False
            if str(frame["thermal_rx_side_probe_mean_rule"].iloc[0]) != (
                RX_SIDE_FACE_MEAN_RULE
            ):
                return False
            selected_face = str(
                frame["thermal_rx_side_probe_selected_face"].iloc[0]
            )
            if selected_face not in {
                "Tprobe_Rx_side_side", "Tprobe_Rx_side1_inner",
                "Tprobe_Rx_side2_side", "Tprobe_Rx_side2_inner",
            }:
                return False
            mode = str(frame.get(
                "thermal_symmetry", pd.Series(["eighth"])
            ).iloc[0]).strip().lower()
            expected_face_count = 4 if mode == "full" else 2
            if int(frame["thermal_rx_side_probe_face_count"].iloc[0]) != (
                expected_face_count
            ):
                return False
        temperatures = [float(frame[column].iloc[0]) for column in required]
        return all(
            math.isfinite(value)
            and MIN_TRUSTED_TEMPERATURE_C < value < MAX_TRUSTED_TEMPERATURE_C
            for value in temperatures
        )
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def _thermal_failure_frame(error, core_conductivity=None):
    """Build a harvestable EM row marker for a hard thermal-stage failure."""
    message = str(error).strip() or repr(error)
    core_conductivity = core_conductivity or {}
    return pd.DataFrame({
        "thermal_solved": [0],
        "thermal_convergence_available": [0],
        "thermal_converged": [0],
        "thermal_extraction_complete": [0],
        "thermal_required_missing_count": [4],
        "thermal_required_group_mask": [15],
        "thermal_required_group_count": [4],
        "thermal_rx_model": ["unknown"],
        "thermal_core_conductivity_model": [
            core_conductivity.get(
                "thermal_core_conductivity_model", "unknown"
            )
        ],
        "thermal_core_k_inplane": [core_conductivity.get(
            "thermal_core_k_inplane", float("nan")
        )],
        "thermal_core_k_throughstack": [core_conductivity.get(
            "thermal_core_k_throughstack", float("nan")
        )],
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


def _load_fixed_input_parameter(param):
    """Load one fixed payload and bind it to this solver's physics revision."""
    input_frame = create_input_parameter(param)
    revision = input_frame["physics_data_revision"].iloc[0]
    if not isinstance(revision, str):
        raise ValueError("physics_data_revision must be a string")
    if revision != PHYSICS_DATA_REVISION:
        raise ValueError(
            "physics_data_revision mismatch: payload has "
            f"{revision!r}, but this solver requires "
            f"{PHYSICS_DATA_REVISION!r}; refusing to run"
        )
    return input_frame, revision


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


def _configure_loss_copy_skin_mesh(sim, native_windings_solid=False):
    """Restore precise conductor physics when the matrix used stranded conductors."""
    if not _lightweight_matrix_enabled(sim):
        logging.info("loss copy: reusing inherited solid windings and mesh operations")
        return False
    if native_windings_solid:
        logging.info(
            "loss copy: native winding edit already established solid Tx/Rx; "
            "assigning conductor meshes only"
        )
    else:
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


def _find_raw_design(project, design_name):
    matches = [
        raw for name, raw in _project_design_entries(project)
        if name == design_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one AEDT design named {design_name!r}, found {len(matches)}"
        )
    raw = matches[0]
    if raw is None:
        raw = project.SetActiveDesign(design_name)
    if _aedt_design_name(raw) != design_name:
        raise RuntimeError(
            f"AEDT returned the wrong design for {design_name!r}: "
            f"{_aedt_design_name(raw)!r}"
        )
    return raw


def _validate_raw_copied_loss_design(raw_design, expected_name):
    actual_name = _aedt_design_name(raw_design)
    if actual_name != expected_name:
        raise RuntimeError(
            f"copied native design identity mismatch: "
            f"expected={expected_name!r}, actual={actual_name!r}"
        )
    design_type = str(raw_design.GetDesignType() or "")
    solution_type = str(raw_design.GetSolutionType() or "")
    setups = tuple(str(item) for item in (
        raw_design.GetModule("AnalysisSetup").GetSetups() or []
    ))
    if design_type != "Maxwell 3D":
        raise RuntimeError(f"copied design type is not Maxwell 3D: {design_type!r}")
    if not _is_ac_magnetic_solution(solution_type):
        raise RuntimeError(
            f"copied design solution is not AC magnetic: {solution_type!r}"
        )
    if "Setup1" not in setups:
        raise RuntimeError(f"copied design has no Setup1: {setups!r}")
    return {
        "name": actual_name,
        "design_type": design_type,
        "solution_type": solution_type,
        "setups": setups,
    }


def _matrix_source_signature(project, source_name, require_solved=True):
    """Attest the source design and any native Matrix/solution evidence available."""
    raw = _find_raw_design(project, source_name)
    contract = _validate_raw_copied_loss_design(raw, source_name)
    parameter_names = None
    try:
        parameter_names = tuple(sorted(str(item) for item in (
            raw.GetChildObject("Parameters").GetChildNames() or []
        )))
    except Exception:
        parameter_names = None
    if parameter_names is not None and "Matrix" not in parameter_names:
        raise RuntimeError(
            f"solved source lost Matrix parameter: {parameter_names!r}"
        )

    solution_marker = None
    try:
        solution_module = raw.GetModule("Solutions")
        variations = []
        queried = False
        for sweep_name in ("Setup1 : LastAdaptive", "Setup1 : Last Adaptive"):
            try:
                values = solution_module.GetAvailableVariations(sweep_name) or []
                queried = True
                variations.extend(str(item) for item in values)
                # AEDT logs an invalid sweep spelling as a GUI macro error even
                # when Python catches the exception.  Do not probe the legacy
                # fallback after one spelling has already been accepted.
                break
            except Exception:
                continue
        solution_marker = tuple(sorted(set(variations))) if queried else None
    except Exception:
        solution_marker = None
    if require_solved and solution_marker == ():
        raise RuntimeError("solved matrix source has no available solution variation")
    source_windings = {}
    for winding_name in ("Tx_winding", "Rx_winding"):
        source_windings[winding_name] = tuple(
            str(_native_child_property(
                _native_winding_child(raw, winding_name), property_name
            ))
            for property_name in (
                "Type", "Winding Type", "Current", "Phase", "IsSolid"
            )
        )
    return {
        **contract,
        "parameter_names": parameter_names,
        "solution_marker": solution_marker,
        "windings": source_windings,
    }


def _assert_matrix_source_preserved(
        project, source_name, expected_signature, require_solved=True):
    actual = _matrix_source_signature(
        project, source_name, require_solved=require_solved
    )
    if actual != expected_signature:
        raise RuntimeError(
            "solved matrix source changed during copied-loss preparation: "
            f"expected={expected_signature!r}, actual={actual!r}"
        )


def _native_winding_child(raw_design, winding_name):
    """Return the native winding using PyAEDT's Boundaries-first precedence."""
    errors = []
    for root_name in ("Boundaries", "Excitations"):
        try:
            root = raw_design.GetChildObject(root_name)
            child_names = tuple(str(item) for item in (root.GetChildNames() or []))
            if winding_name not in child_names:
                errors.append(f"{root_name} names={child_names!r}")
                continue
            child = root.GetChildObject(winding_name)
            if child is None or child is False:
                errors.append(f"{root_name} child unavailable")
                continue
            return child
        except Exception as error:
            errors.append(f"{root_name}={type(error).__name__}: {error}")
    raise RuntimeError(
        f"native winding child {winding_name!r} is unavailable: {errors}"
    )


def _native_child_property(child, property_name):
    names = tuple(str(item) for item in (child.GetPropNames() or []))
    wanted = re.sub(r"[^a-z0-9]", "", property_name.lower())
    matches = [
        name for name in names
        if re.sub(r"[^a-z0-9]", "", name.lower()) == wanted
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"native property {property_name!r} is not unique in {names!r}"
        )
    return child.GetPropValue(matches[0])


def _aedt_quantity_parts(value):
    token = str(value).strip().replace("°", "deg")
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)",
        token,
    )
    if not match:
        raise RuntimeError(f"invalid native AEDT quantity: {value!r}")
    return float(match.group(1)), match.group(2).lower()


def _assert_native_quantity(actual, expected, label):
    actual_value, actual_unit = _aedt_quantity_parts(actual)
    expected_value, expected_unit = _aedt_quantity_parts(expected)
    if actual_unit != expected_unit:
        raise RuntimeError(
            f"{label} native unit mismatch: expected={expected!r}, actual={actual!r}"
        )
    if not math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            f"{label} native value mismatch: expected={expected!r}, actual={actual!r}"
        )


def _assert_native_copied_loss_windings(
        raw_design, expected_design_name, tx_current, tx_phase,
        rx_current, rx_phase, require_solid):
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    expected = {
        "Tx_winding": (tx_current, tx_phase),
        "Rx_winding": (rx_current, rx_phase),
    }
    for winding_name, (current, phase) in expected.items():
        child = _native_winding_child(raw_design, winding_name)
        object_kind = str(_native_child_property(
            child, "Type"
        )).strip().lower()
        if object_kind != "winding group":
            raise RuntimeError(
                f"{winding_name} native object kind is not Winding Group: "
                f"{object_kind!r}"
            )
        winding_type = str(_native_child_property(
            child, "Winding Type"
        )).strip().lower()
        if winding_type != "current":
            raise RuntimeError(
                f"{winding_name} native Winding Type is not Current: "
                f"{winding_type!r}"
            )
        _assert_native_quantity(
            _native_child_property(
                child, "Current"
            ), current,
            f"{winding_name} Current",
        )
        _assert_native_quantity(
            _native_child_property(
                child, "Phase"
            ), phase,
            f"{winding_name} Phase",
        )
        solid = _aedt_bool(_native_child_property(child, "IsSolid"))
        if solid != bool(require_solid):
            raise RuntimeError(
                f"{winding_name} native IsSolid mismatch: "
                f"expected={bool(require_solid)}, actual={solid}"
            )


def _assert_native_core_loss_assignment(raw_design, core_names):
    boundary = raw_design.GetModule("BoundarySetup")
    missing = []
    for name in core_names:
        try:
            enabled = _aedt_bool(boundary.GetCoreLossEffect(name))
        except Exception as error:
            raise RuntimeError(
                f"native core-loss readback failed for {name!r}"
            ) from error
        if not enabled:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"native copied design has core loss disabled for {missing!r}"
        )


def _edit_native_copied_loss_winding(
        raw_design, expected_design_name, winding_name, current, phase):
    """Edit one exact copied winding without trusting PyAEDT's cached props."""
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    child = _native_winding_child(raw_design, winding_name)
    object_kind = str(_native_child_property(child, "Type")).strip().lower()
    if object_kind != "winding group":
        raise RuntimeError(
            f"refusing to edit non-winding-group native object {winding_name!r}: "
            f"{object_kind!r}"
        )
    winding_type = str(
        _native_child_property(child, "Winding Type")
    ).strip().lower()
    if winding_type != "current":
        raise RuntimeError(
            f"refusing to edit non-current native winding {winding_name!r}: "
            f"{winding_type!r}"
        )
    arguments = [
        f"NAME:{winding_name}",
        "Type:=", "Current",
        "IsSolid:=", True,
        "Current:=", str(current),
        "Resistance:=", "0ohm",
        "Inductance:=", "0H",
        "Voltage:=", "0V",
        "ParallelBranchesNum:=", "1",
        "Phase:=", str(phase),
    ]
    boundary = raw_design.GetModule("BoundarySetup")
    boundary.EditWindingGroup(winding_name, arguments)
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    return arguments


def _assign_native_copied_core_loss(
        raw_design, expected_design_name, core_names):
    """Assign and prove core loss through the exact copied native design."""
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    core_names = [str(name) for name in core_names]
    if not core_names or any(not name for name in core_names):
        raise RuntimeError(f"invalid copied core-loss assignment: {core_names!r}")
    boundary = raw_design.GetModule("BoundarySetup")
    boundary.SetCoreLoss(core_names, False)
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    _assert_native_core_loss_assignment(raw_design, core_names)
    return len(core_names)


def _validate_saved_copied_loss_preparation(
        aedt_path, source_name, copied_name,
        tx_current, tx_phase, rx_current, rx_phase, core_count,
        max_passes, min_converged, percent_error, frequency,
        require_source_solved=True):
    """Validate the authoritative saved AEDT snapshot before the loss solve."""
    from ansys.aedt.core.internal.load_aedt_file import load_entire_aedt_file

    if not os.path.isfile(aedt_path):
        raise RuntimeError(f"copied-loss prepare snapshot is missing: {aedt_path}")
    project = load_entire_aedt_file(aedt_path)
    try:
        models = project["AnsoftProject"]["Maxwell3DModel"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("saved AEDT has no Maxwell3DModel list") from error
    if not isinstance(models, list):
        raise RuntimeError("saved AEDT Maxwell3DModel is not a list")

    def _one_model(name):
        matches = [model for model in models if model.get("Name") == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"saved AEDT expected one model {name!r}, found {len(matches)}"
            )
        return matches[0]

    source_model = _one_model(source_name)
    copied_model = _one_model(copied_name)
    try:
        source_parameters = source_model[
            "MaxwellParameterSetup"
        ]["MaxwellParameters"]
        copied_parameters = copied_model[
            "MaxwellParameterSetup"
        ]["MaxwellParameters"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("saved AEDT has malformed Maxwell parameter data") from error
    if not isinstance(source_parameters.get("Matrix"), dict):
        raise RuntimeError("saved matrix source lost its Matrix parameter")
    if "Matrix" in copied_parameters:
        raise RuntimeError("saved copied loss design still contains Matrix parameter")

    boundaries = copied_model.get("BoundarySetup", {}).get("Boundaries", {})
    expected_windings = {
        "Tx_winding": (tx_current, tx_phase),
        "Rx_winding": (rx_current, rx_phase),
    }
    for winding_name, (current, phase) in expected_windings.items():
        winding = boundaries.get(winding_name)
        if not isinstance(winding, dict):
            raise RuntimeError(
                f"saved copied design has no {winding_name!r} boundary"
            )
        if str(winding.get("Type", "")).strip().lower() != "current":
            raise RuntimeError(
                f"saved {winding_name} Type is not Current: {winding.get('Type')!r}"
            )
        if not _aedt_bool(winding.get("IsSolid")):
            raise RuntimeError(f"saved {winding_name} is not solid")
        _assert_native_quantity(
            winding.get("Current"), current, f"saved {winding_name} Current"
        )
        _assert_native_quantity(
            winding.get("Phase"), phase, f"saved {winding_name} Phase"
        )

    global_data = copied_model.get("BoundarySetup", {}).get("GlobalBoundData", {})
    core_ids = global_data.get("CoreLossObjectIDs")
    if core_ids is None:
        core_ids = []
    elif not isinstance(core_ids, (list, tuple)):
        core_ids = [core_ids]
    if len(core_ids) != int(core_count):
        raise RuntimeError(
            "saved copied design core-loss ID count mismatch: "
            f"expected={int(core_count)}, actual={len(core_ids)}, ids={core_ids!r}"
        )

    try:
        setup = copied_model["AnalysisSetup"]["SolveSetups"]["Setup1"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("saved copied loss design has no Setup1") from error
    expected_setup = {
        "MaximumPasses": int(max_passes),
        "MinimumConvergedPasses": int(min_converged),
    }
    if str(setup.get("SetupType", "")).strip() != "AC Magnetic":
        raise RuntimeError(
            f"saved copied Setup1 type mismatch: {setup.get('SetupType')!r}"
        )
    if not _aedt_bool(setup.get("Enabled")):
        raise RuntimeError("saved copied Setup1 is disabled")
    for key, expected in expected_setup.items():
        if int(setup.get(key, -1)) != expected:
            raise RuntimeError(
                f"saved copied Setup1 {key} mismatch: "
                f"expected={expected}, actual={setup.get(key)!r}"
            )
    if not math.isclose(
            float(setup.get("PercentError", float("nan"))),
            float(percent_error), rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(
            "saved copied Setup1 PercentError mismatch: "
            f"expected={percent_error!r}, actual={setup.get('PercentError')!r}"
        )
    _assert_native_quantity(
        setup.get("Frequency"), f"{float(frequency):g}Hz",
        "saved copied Setup1 Frequency",
    )

    preview = project.get("ProjectPreview", {}).get("DesignInfo")
    if not isinstance(preview, list):
        raise RuntimeError("saved AEDT has no ProjectPreview DesignInfo list")

    def _one_preview(name):
        matches = [item for item in preview if item.get("DesignName") == name]
        if len(matches) != 1:
            raise RuntimeError(
                f"saved AEDT expected one preview {name!r}, found {len(matches)}"
            )
        return matches[0]

    source_solved = _aedt_bool(_one_preview(source_name).get("IsSolved"))
    if require_source_solved and not source_solved:
        raise RuntimeError("saved matrix source is no longer solved")
    if _aedt_bool(_one_preview(copied_name).get("IsSolved")):
        raise RuntimeError("saved copied design still owns inherited solution data")
    return {
        "source": source_name,
        "copied": copied_name,
        "core_loss_ids": len(core_ids),
        "source_solved": source_solved,
        "copied_solved": False,
    }


def _set_copied_loss_winding_excitation(
        winding, expected_name, current, phase, label):
    """Edit one copied winding only after its boundary DataModel is usable."""
    if winding is None or winding is False:
        raise RuntimeError(f"copied {label} winding is unavailable")
    resolved_name = str(winding.name or "").strip()
    if resolved_name != expected_name:
        raise RuntimeError(
            f"copied {label} winding identity mismatch: "
            f"expected={expected_name!r}, actual={resolved_name!r}"
        )
    props = winding.props
    if not isinstance(props, dict) or not props:
        raise RuntimeError(f"copied {label} winding properties are unavailable")
    updates = {"Current": str(current), "Phase": str(phase)}
    setter = getattr(props, "_setitem_without_update", None)
    previous_auto_update = getattr(winding, "auto_update", None)
    try:
        if previous_auto_update is not None:
            winding.auto_update = False
        for key, value in updates.items():
            if callable(setter):
                setter(key, value)
            else:
                props[key] = value
        if winding.update() is not True:
            raise RuntimeError(
                f"copied {label} winding Current/Phase update returned False"
            )
    finally:
        if previous_auto_update is not None:
            winding.auto_update = previous_auto_update
    readback_name = str(winding.name or "").strip()
    if readback_name != expected_name:
        raise RuntimeError(
            f"copied {label} winding identity changed after update: "
            f"expected={expected_name!r}, actual={readback_name!r}"
        )


def _configure_copied_loss_excitations(
        design, raw_design, expected_design_name,
        tx_current, tx_phase, rx_current, rx_phase):
    """Edit both windings natively and prove the complete copied state."""
    wrapper_raw = getattr(getattr(design, "solver_instance", None), "odesign", None)
    if wrapper_raw is None:
        raise RuntimeError("copied wrapper has no native odesign before mutation")
    _validate_raw_copied_loss_design(wrapper_raw, expected_design_name)
    wrapper_name = _aedt_design_name(getattr(design, "design_name", ""))
    if wrapper_name != expected_design_name:
        raise RuntimeError(
            f"copied wrapper identity mismatch: expected={expected_design_name!r}, "
            f"actual={wrapper_name!r}"
        )
    _validate_raw_copied_loss_design(raw_design, expected_design_name)
    tx = _native_winding_child(raw_design, "Tx_winding")
    rx = _native_winding_child(raw_design, "Rx_winding")
    _edit_native_copied_loss_winding(
        raw_design, expected_design_name, "Tx_winding", tx_current, tx_phase
    )
    _edit_native_copied_loss_winding(
        raw_design, expected_design_name, "Rx_winding", rx_current, rx_phase
    )
    _assert_native_copied_loss_windings(
        raw_design, expected_design_name,
        tx_current, tx_phase, rx_current, rx_phase,
        require_solid=True,
    )
    return tx, rx


def _validate_prepared_copied_loss_design(
        project, prepared, before_names, source_name):
    current_names = {
        name for name, _raw in _project_design_entries(project)
    }
    new_names = sorted(current_names - set(before_names))
    if len(new_names) != 1:
        raise RuntimeError(
            f"copied preparation introduced {len(new_names)} designs: {new_names!r}"
        )
    copied_name = new_names[0]
    if copied_name == source_name:
        raise RuntimeError("copied design resolved to the matrix source")
    wrapper_name = _aedt_design_name(getattr(prepared, "design_name", ""))
    wrapper_solution = str(getattr(prepared, "solution_type", "") or "")
    setup = prepared.get_setup(name="Setup1")
    if (
            wrapper_name != copied_name
            or not _is_ac_magnetic_solution(wrapper_solution)
            or setup is None or setup is False
            or _ready_loss_setup_properties(setup) is None):
        raise RuntimeError(
            "returned copied wrapper is not fully ready: "
            f"name={wrapper_name!r}, solution={wrapper_solution!r}, "
            f"copied_name={copied_name!r}"
        )
    raw = _find_raw_design(project, copied_name)
    _validate_raw_copied_loss_design(raw, copied_name)
    wrapper_raw = getattr(getattr(prepared, "solver_instance", None), "odesign", None)
    if wrapper_raw is None:
        raise RuntimeError("returned copied wrapper has no native odesign")
    _validate_raw_copied_loss_design(wrapper_raw, copied_name)
    return copied_name, raw


def _cleanup_bad_copied_loss_design(
        project, before_names, source_name, source_signature,
        require_source_solved=True,
        max_attempts=3, poll_s=0.25, sleeper=time.sleep):
    """Delete the one exact new design and prove the solved source survived."""
    before_names = set(before_names)
    active_source = project.SetActiveDesign(source_name)
    _validate_raw_copied_loss_design(active_source, source_name)
    current_names = {
        name for name, _raw in _project_design_entries(project)
    }
    new_names = sorted(current_names - before_names)
    if not new_names:
        for _poll in range(max(1, int(max_attempts))):
            if poll_s:
                sleeper(max(0.0, float(poll_s)))
            current_names = {
                name for name, _raw in _project_design_entries(project)
            }
            new_names = sorted(current_names - before_names)
            if new_names:
                break
            if current_names != before_names:
                raise RuntimeError(
                    "copied-design cleanup baseline changed while waiting: "
                    f"before={sorted(before_names)!r}, "
                    f"current={sorted(current_names)!r}"
                )
        _assert_matrix_source_preserved(
            project, source_name, source_signature,
            require_solved=require_source_solved,
        )
        if not new_names:
            return None
    if len(new_names) != 1:
        raise RuntimeError(
            f"refusing ambiguous copied-design cleanup: new_names={new_names!r}"
        )
    bad_name = new_names[0]
    if bad_name == source_name:
        raise RuntimeError("refusing to delete solved matrix source")
    errors = []
    for _attempt in range(1, max(1, int(max_attempts)) + 1):
        active_source = project.SetActiveDesign(source_name)
        _validate_raw_copied_loss_design(active_source, source_name)
        try:
            project.DeleteDesign(bad_name)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        if poll_s:
            sleeper(max(0.0, float(poll_s)))
        remaining = {
            name for name, _raw in _project_design_entries(project)
        }
        if bad_name not in remaining:
            if remaining != before_names:
                raise RuntimeError(
                    "copied-design cleanup did not restore the exact baseline: "
                    f"before={sorted(before_names)!r}, after={sorted(remaining)!r}"
                )
            _assert_matrix_source_preserved(
                project, source_name, source_signature,
                require_solved=require_source_solved,
            )
            return bad_name
    raise RuntimeError(
        f"failed to delete exact copied design {bad_name!r}: {errors}"
    )


def _retry_copied_loss_preparation(
        project, source_name, prepare_attempt, max_attempts=3,
        retry_delay_s=5.0, sleeper=time.sleep, require_source_solved=True):
    """Retry only pre-solve copy preparation while preserving the solved source."""
    max_attempts = max(1, int(max_attempts))
    source_signature = _matrix_source_signature(
        project, source_name, require_solved=require_source_solved
    )
    baseline_names = {
        name for name, _raw in _project_design_entries(project)
    }
    failures = []
    for attempt in range(1, max_attempts + 1):
        active_source = project.SetActiveDesign(source_name)
        _validate_raw_copied_loss_design(active_source, source_name)
        _assert_matrix_source_preserved(
            project, source_name, source_signature,
            require_solved=require_source_solved,
        )
        current_names = {
            name for name, _raw in _project_design_entries(project)
        }
        if current_names != baseline_names:
            raise RuntimeError(
                "copied-loss retry baseline is not stable: "
                f"expected={sorted(baseline_names)!r}, "
                f"actual={sorted(current_names)!r}"
            )
        before_names = set(baseline_names)
        try:
            prepared = prepare_attempt(before_names, attempt)
            if prepared is None or prepared is False:
                raise RuntimeError("copied loss preparation returned no design")
            _validate_prepared_copied_loss_design(
                project, prepared, before_names, source_name
            )
            _assert_matrix_source_preserved(
                project, source_name, source_signature,
                require_solved=require_source_solved,
            )
            return prepared, attempt
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            try:
                _cleanup_bad_copied_loss_design(
                    project, before_names, source_name, source_signature,
                    require_source_solved=require_source_solved,
                    sleeper=sleeper,
                )
            except Exception as cleanup_error:
                raise RuntimeError(
                    "copied loss preparation failed and exact bad-design cleanup "
                    f"also failed: prepare={error}; cleanup={cleanup_error}"
                ) from cleanup_error
            if attempt >= max_attempts:
                raise RuntimeError(
                    "copied loss design preparation failed after "
                    f"{max_attempts} fresh copies; " + "; ".join(failures)
                ) from error
            logging.warning(
                "copied loss preparation attempt %s/%s failed before solve; "
                "deleted the exact bad copy and will re-copy solved %s: %s",
                attempt, max_attempts, source_name, error,
            )
            if retry_delay_s:
                sleeper(max(0.0, float(retry_delay_s)))


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


def _delete_copied_solution_or_raise(
        primary_design, fallback_design, expected_design_name):
    """Delete inherited fields only through an exact copied-design identity."""
    errors = []
    for label, design in (("wrapper", primary_design), ("native", fallback_design)):
        try:
            _validate_raw_copied_loss_design(design, expected_design_name)
        except Exception as error:
            errors.append(f"{label}-identity={type(error).__name__}: {error}")
            continue
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
        self.extraction_units = {}
        self.spawned_descendants = {}
        self.stage_timings = {}

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

    def _rebind_native_project_for_design_creation(
            self, max_attempts=3, retry_delay=0.5, sleeper=time.sleep):
        """Rebind a stale pyProject handle before creating the next design."""
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError("project rebind max_attempts must be positive")
        expected_project = str(getattr(self, "PROJECT_NAME", "") or "").strip()
        project_wrapper = getattr(self, "project", None)
        try:
            project_state = vars(project_wrapper)
        except TypeError:
            project_state = {}
        if not expected_project or not project_state:
            raise RuntimeError("project wrapper identity is unavailable for native rebind")

        odesktop = self._native_desktop_handle()
        set_active_project = getattr(odesktop, "SetActiveProject", None)
        if not callable(set_active_project):
            raise RuntimeError("native Desktop cannot activate a project")

        errors = []
        missing = object()
        for attempt in range(1, max_attempts + 1):
            previous_project = project_state.get("project", missing)
            previous_proj = project_state.get("proj", missing)
            rebound = False

            def restore_previous_binding():
                for key, previous in (
                        ("project", previous_project), ("proj", previous_proj)):
                    if previous is missing:
                        project_state.pop(key, None)
                    else:
                        project_state[key] = previous

            try:
                native_project = set_active_project(expected_project)
                if native_project is None or native_project is False:
                    raise RuntimeError(
                        f"SetActiveProject returned no project ({expected_project})"
                    )
                actual_project = str(native_project.GetName() or "").strip()
                if actual_project != expected_project:
                    raise _AedtIdentityMismatch(
                        "thermal project identity mismatch: "
                        f"expected={expected_project}, "
                        f"actual={actual_project or '<empty>'}"
                    )

                project_state["project"] = native_project
                project_state["proj"] = native_project
                rebound = True
                rebound_name = str(project_wrapper.name or "").strip()
                if rebound_name != expected_project:
                    raise _AedtIdentityMismatch(
                        "rebound project identity mismatch: "
                        f"expected={expected_project}, "
                        f"actual={rebound_name or '<empty>'}"
                    )
                return native_project
            except _AedtIdentityMismatch:
                if rebound:
                    restore_previous_binding()
                raise
            except Exception as error:
                if rebound:
                    restore_previous_binding()
                errors.append(
                    f"attempt {attempt}: {type(error).__name__}: {error}"
                )
                if attempt < max_attempts:
                    logging.warning(
                        "native project rebind failed "
                        f"(attempt {attempt}/{max_attempts}): {error}"
                    )
                    sleeper(max(0.0, float(retry_delay)) * (2 ** (attempt - 1)))
        raise RuntimeError(
            "native project rebind failed before design creation: "
            + "; ".join(errors)
        )

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

    def _configure_1k101_native_material(self, material_name, direction):
        """Create/update one wound-core orientation and attest native AEDT data."""
        cm = float(self.df_plus["core_cm_assigned"].iloc[0])
        core_x = float(self.df_plus["core_x"].iloc[0])
        core_y = float(self.df_plus["core_y"].iloc[0])
        kf = float(self.df_plus["core_lamination_factor"].iloc[0])
        specs = native_lamination_material_specs(kf)
        region = "leg" if direction == LEG_STACKING_DIRECTION else "yoke"
        if specs[region]["stacking_direction"] != direction:
            raise RuntimeError(f"internal stacking-direction mismatch for {region}")

        materials = self.design1.materials
        base_name = "power_ferrite"
        if base_name not in materials.material_keys:
            created = self.design1.set_power_ferrite(
                cm=cm, x=core_x, y=core_y
            )
            if created is False:
                raise RuntimeError("failed to create base POWERLITE material")
        base = materials[base_name]
        if base is None:
            raise RuntimeError("base POWERLITE material is unavailable")
        base.permeability = "3000"
        base.conductivity = 0
        set_loss = getattr(base, "set_power_ferrite_coreloss", None)
        if not callable(set_loss) or set_loss(
            cm=cm, x=core_x, y=core_y, kdc=0, cut_depth=0.0
        ) is False:
            raise RuntimeError("failed to apply base POWERLITE core-loss law")

        if material_name not in materials.material_keys:
            material = materials.duplicate_material(base_name, material_name)
        else:
            material = materials[material_name]
        if material is None or material is False:
            raise RuntimeError(f"failed to create native material {material_name!r}")

        # Reapply every solver-relevant property because project materials are
        # shared by copied designs and may otherwise retain stale values.
        material.permeability = "3000"
        material.conductivity = 0
        set_loss = getattr(material, "set_power_ferrite_coreloss", None)
        if not callable(set_loss) or set_loss(
            cm=cm, x=core_x, y=core_y, kdc=0, cut_depth=0.0
        ) is False:
            raise RuntimeError(
                f"failed to apply POWERLITE law to {material_name!r}"
            )
        material.stacking_type = specs[region]["stacking_type"]
        material.stacking_factor = specs[region]["stacking_factor"]
        material.stacking_direction = specs[region]["stacking_direction"]

        raw_props = _raw_aedt_material_props(materials, material_name)
        attestation = validate_native_lamination_readback(
            raw_props,
            lamination_factor=kf,
            stacking_direction=direction,
            cm_base=cm,
            core_x=core_x,
            core_y=core_y,
            permeability=3000.0,
        )
        return material, attestation

    def create_core(self):
        # Keep the traceable POWERLITE base law in Maxwell.  The new physics
        # revision uses AEDT's native Lamination model and splits the XZ frame
        # into V(1) legs and V(3) yokes.  Legacy revisions keep the old single
        # solid so sealed b171 candidates remain interpretable.
        cm_assigned = float(self.df_plus["core_cm_assigned"].iloc[0])
        core_x = float(self.df_plus["core_x"].iloc[0])
        core_y = float(self.df_plus["core_y"].iloc[0])
        revision = (
            str(self.df_plus["physics_data_revision"].iloc[0]).strip()
            if "physics_data_revision" in self.df_plus.columns else ""
        )
        native_stacking = revision == PHYSICS_DATA_REVISION

        if native_stacking:
            leg_name = "power_ferrite_1k101_leg_v1"
            yoke_name = "power_ferrite_1k101_yoke_v3"
            leg_mat, leg_attestation = self._configure_1k101_native_material(
                leg_name, LEG_STACKING_DIRECTION
            )
            yoke_mat, yoke_attestation = self._configure_1k101_native_material(
                yoke_name, YOKE_STACKING_DIRECTION
            )
            self.power_ferrite_mat = leg_mat
            self.core_material_native_attestation = {
                "physics_data_revision": revision,
                "leg": leg_attestation,
                "yoke": yoke_attestation,
            }
            self.df_plus["core_native_material_readback_attested"] = [1]
            self.df_plus["core_native_leg_material_name"] = [leg_name]
            self.df_plus["core_native_yoke_material_name"] = [yoke_name]
            self.df_plus["core_native_leg_stacking_direction_readback"] = [
                leg_attestation["stacking_direction"]
            ]
            self.df_plus["core_native_yoke_stacking_direction_readback"] = [
                yoke_attestation["stacking_direction"]
            ]
            self.df_plus["core_native_leg_stacking_factor_readback"] = [
                leg_attestation["stacking_factor"]
            ]
            self.df_plus["core_native_yoke_stacking_factor_readback"] = [
                yoke_attestation["stacking_factor"]
            ]
            self.df_plus["core_native_cm_readback"] = [
                leg_attestation["core_loss_cm"]
            ]
        else:
            leg_name = yoke_name = "power_ferrite"
            if "power_ferrite" not in self.design1.materials.material_keys:
                self.design1.set_power_ferrite(
                    cm=cm_assigned, x=core_x, y=core_y
                )
            self.power_ferrite_mat = self.design1.materials["power_ferrite"]
            update_core_loss = getattr(
                self.power_ferrite_mat, "set_power_ferrite_coreloss", None
            )
            if not callable(update_core_loss) or update_core_loss(
                cm=cm_assigned, x=core_x, y=core_y
            ) is False:
                raise RuntimeError("failed to apply legacy core-loss contract")
            self.power_ferrite_mat.permeability = "3000"
            self.core_material_native_attestation = {
                "physics_data_revision": revision or "legacy_unspecified",
                "native_lamination": False,
            }
            self.df_plus["core_native_material_readback_attested"] = [0]

        self.create_thermal_pad_material()

        n_group = int(self.df_plus["n_core_group"].iloc[0])
        plate_on = int(self.df_plus["core_plate_on"].iloc[0]) != 0
        pad_on = float(self.df_plus["core_plate_pad_t"].iloc[0]) > 0

        core_objs, plate_objs, pad_objs = create_core(
            design=self.design1,
            name="core",
            core_material=leg_name,
            n_group=n_group,
            plate_material="aluminum",
            pad_material="thermal_pad",
            plate_on=plate_on,
            pad_on=pad_on,
            plate_color=PLATE_COLOR,
            pad_color=PAD_COLOR,
            segmented_lamination=native_stacking,
            core_material_leg=leg_name,
            core_material_yoke=yoke_name,
        )
        if native_stacking:
            expected_regions = {
                "leg_left", "leg_center", "leg_right",
                "yoke_bottom", "yoke_top",
            }
            by_group = {index: set() for index in range(1, n_group + 1)}
            for obj in core_objs:
                index = _core_group_index(obj.name)
                prefix = f"core_{index}_"
                by_group.setdefault(index, set()).add(obj.name[len(prefix):])
                expected_material = (
                    leg_name if "_leg_" in obj.name else yoke_name
                ).casefold()
                actual_material = _raw_aedt_object_attribute(
                    obj, "Material"
                ).casefold()
                if actual_material != expected_material:
                    raise RuntimeError(
                        f"core material assignment mismatch for {obj.name}: "
                        f"{actual_material!r} != {expected_material!r}"
                    )
                orientation = _raw_aedt_object_attribute(obj, "Orientation")
                if orientation != "Global":
                    raise RuntimeError(
                        f"core object orientation mismatch for {obj.name}: "
                        f"{orientation!r} != 'Global'"
                    )
            drift = {
                index: sorted(regions)
                for index, regions in by_group.items()
                if regions != expected_regions
            }
            if drift or len(core_objs) != 5 * n_group:
                raise RuntimeError(
                    "segmented core topology mismatch: "
                    f"count={len(core_objs)}, groups={drift or by_group}"
                )
            expected_total_mm3 = float(
                self.df_plus["core_vol_gross_m3"].iloc[0]
            ) * 1e9
            actual_total_mm3 = sum(abs(float(obj.volume)) for obj in core_objs)
            volume_rel_error = abs(
                actual_total_mm3 - expected_total_mm3
            ) / max(abs(expected_total_mm3), 1e-12)
            if not math.isfinite(volume_rel_error) or volume_rel_error > 1e-9:
                raise RuntimeError(
                    "segmented core gross-volume preservation failed: "
                    f"actual={actual_total_mm3:.12g}mm3, "
                    f"expected={expected_total_mm3:.12g}mm3, "
                    f"relative_error={volume_rel_error:.6g}"
                )
            center_area_mm2 = sum(
                abs(float(obj.volume))
                for obj in core_objs if obj.name.endswith("_leg_center")
            ) / float(self.df_plus["h1"].iloc[0])
            expected_center_area_mm2 = float(
                self.df_plus["Ae_gross_m2"].iloc[0]
            ) * 1e6
            area_rel_error = abs(
                center_area_mm2 - expected_center_area_mm2
            ) / max(abs(expected_center_area_mm2), 1e-12)
            if area_rel_error > 1e-9:
                raise RuntimeError(
                    "segmented center-leg gross-area preservation failed: "
                    f"actual={center_area_mm2:.12g}mm2, "
                    f"expected={expected_center_area_mm2:.12g}mm2"
                )
            density = float(self.df_plus["core_mass_density_kg_m3"].iloc[0])
            actual_mass_kg = actual_total_mm3 * 1e-9 * density
            expected_mass_kg = float(
                self.df_plus["core_mass_gross_kg"].iloc[0]
            )
            mass_rel_error = abs(actual_mass_kg - expected_mass_kg) / max(
                abs(expected_mass_kg), 1e-12
            )
            if mass_rel_error > 1e-9:
                raise RuntimeError(
                    "segmented core gross-mass preservation failed: "
                    f"actual={actual_mass_kg:.12g}kg, "
                    f"expected={expected_mass_kg:.12g}kg"
                )
            self.df_plus["core_segmented_piece_count"] = [len(core_objs)]
            self.df_plus["core_segmented_area_rel_error"] = [area_rel_error]
            self.df_plus["core_segmented_volume_rel_error"] = [volume_rel_error]
            self.df_plus["core_segmented_mass_rel_error"] = [mass_rel_error]

        stack_expr = (
            "(core_plate_t + 2*core_plate_pad_t)" if pad_on
            else "core_plate_t"
        )
        depth_expr = f"((w1 - {n_group + 1}*{stack_expr})/{n_group})"
        core_flux_sheets = []
        for index in range(n_group):
            y0 = (
                f"(-w1/2 + {index + 1}*{stack_expr} "
                f"+ {index}*{depth_expr})"
            )
            sheet = self.design1.modeler.create_rectangle(
                orientation="XY",
                origin=["-l1", y0, "0mm"],
                sizes=["2*l1", depth_expr],
                name=f"core_flux_section_{index + 1}",
            )
            if sheet is None or sheet is False:
                raise RuntimeError(
                    f"failed to create core flux section {index + 1}"
                )
            sheet.model = False
            core_flux_sheets.append(sheet)
        full_flux_area_m2 = sum(
            _sheet_area_model_units(sheet) for sheet in core_flux_sheets
        ) * 1e-6
        expected_full_flux_area_m2 = float(
            self.df_plus["Ae_gross_m2"].iloc[0]
        )
        full_flux_area_rel_error = abs(
            full_flux_area_m2 - expected_full_flux_area_m2
        ) / max(abs(expected_full_flux_area_m2), 1e-12)
        if full_flux_area_rel_error > 1e-9:
            raise RuntimeError(
                "full core flux-section area mismatch: "
                f"actual={full_flux_area_m2:.12g}m2, "
                f"expected={expected_full_flux_area_m2:.12g}m2"
            )
        self.df_plus["core_flux_section_full_area_rel_error"] = [
            full_flux_area_rel_error
        ]
        self.design1.core_objs = core_objs
        self.design1.core_flux_sheets = core_flux_sheets
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

        geometrys = (self.design1.core_objs
                     + self.design1.core_plates + self.design1.core_pads
                     + self.design1.wcp_plates + self.design1.wcp_pads
                     + self.design1.Tx_windings_main + self.design1.Rx_windings_main
                     + self.design1.Tx_windings_side + self.design1.Rx_windings_side)
        flux_sheets = list(self.design1.core_flux_sheets)

        # 분할 순서대로 진행하되, 앞 분할에서 통째로 삭제된 오브젝트를 다음 호출에 넘기지 않음
        # (넘기면 AEDT가 'Part not found' 경고를 배치로 뿜음 - 무해하지만 소음)
        def _alive(objs):
            existing = set(self.design1.modeler.object_names)
            return [o for o in objs if o.name in existing]

        self.design1.modeler.split(assignment=geometrys, plane="XY", sides="PositiveOnly")
        geometrys = _alive(geometrys)
        self.design1.modeler.split(assignment=geometrys, plane="XZ", sides="PositiveOnly")
        # Flux sheets are exactly coplanar with XY (z=0), so do not submit them
        # to the ambiguous z split. They only need the y>=0 and x<=0 cuts.
        self.design1.modeler.split(
            assignment=flux_sheets, plane="XZ", sides="PositiveOnly"
        )
        geometrys = _alive(geometrys)
        flux_sheets = _alive(flux_sheets)
        self.design1.modeler.split(assignment=geometrys, plane="YZ", sides="NegativeOnly")
        self.design1.modeler.split(
            assignment=flux_sheets, plane="YZ", sides="NegativeOnly"
        )

        # 대칭 분할로 완전히 잘려나간 오브젝트(y<0 쪽 콜드플레이트/냉각판 등)를 리스트에서 제거
        # (이후 eddy 설정/손실 계산이 존재하지 않는 오브젝트를 참조하지 않도록)
        existing = set(self.design1.modeler.object_names)
        self.design1.core_objs = [o for o in self.design1.core_objs if o.name in existing]
        self.design1.core_flux_sheets = [
            o for o in self.design1.core_flux_sheets if o.name in existing
        ]
        if not self.design1.core_flux_sheets:
            raise RuntimeError("symmetry split removed every core flux section")
        retained_flux_area_m2 = sum(
            _sheet_area_model_units(sheet)
            for sheet in self.design1.core_flux_sheets
        ) * 1e-6
        expected_flux_area_m2 = float(
            self.df_plus["Ae_gross_m2"].iloc[0]
        ) / 4.0
        flux_area_rel_error = abs(
            retained_flux_area_m2 - expected_flux_area_m2
        ) / max(abs(expected_flux_area_m2), 1e-12)
        if flux_area_rel_error > 1e-9:
            raise RuntimeError(
                "symmetry core flux-section area mismatch: "
                f"retained={retained_flux_area_m2:.12g}m2, "
                f"expected={expected_flux_area_m2:.12g}m2, "
                f"relative_error={flux_area_rel_error:.6g}"
            )
        self.df_plus["core_flux_section_symmetry_area_rel_error"] = [
            flux_area_rel_error
        ]
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
        # Some verification/replay callers construct a lightweight Simulation
        # shell without invoking __init__.  Lazily create telemetry maps so
        # extraction correctness does not depend on constructor side effects.
        for attribute in (
            "extraction_attempts", "extraction_backends", "extraction_units"
        ):
            if not isinstance(getattr(self, attribute, None), dict):
                setattr(self, attribute, {})
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
                    self.extraction_units[extraction_key] = {
                        expression: str(units.get(expression, "") or "")
                        for expression in expressions
                    }
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
            reporter = None
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
            finally:
                # The calculator stack is shared by the design.  A failed CalcOp
                # can otherwise leave a partial expression behind and poison the
                # next, unrelated loss expression.  Cleanup is best-effort and
                # must never mask the original gRPC failure.
                if reporter is not None:
                    try:
                        reporter.CalcStack("clear")
                    except Exception as cleanup_error:
                        logging.warning(
                            f"field calculator stack cleanup failed ({expr_name}): "
                            f"{cleanup_error}"
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

    def _export_field_report(
            self, report_name, Y_components, extraction_key="loss"):
        # Kept as a compatibility-shaped helper for callers; no AEDT report/file is created.
        target_units = {}
        for expression in Y_components:
            if expression.startswith(("B_mean_", "B_max_")):
                target_units[expression] = "T"
            elif expression.startswith("Phi_"):
                target_units[expression] = "Wb"
            elif expression.startswith("P_"):
                target_units[expression] = "W"
            # Bpow_* deliberately retains its compound T**y*volume unit.  It
            # is validated and normalized explicitly before use.
        return self._solution_data_frame(
            Y_components,
            target_units=target_units,
            report_category="Fields",
            extraction_key=extraction_key,
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
        if (name.startswith("core_plate")
                and name.endswith(("_left", "_right"))):
            # The retained I plate has a discrete x-side twin. A plate stack
            # that does not cross y=0 also has a discrete y-side twin.
            return 2.0 if self._sym_cut_count(name) == 2 else 4.0
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
                # Maxwell AC Magnetic 2025.2 rejects CalcOp("ComplxPeak") at
                # the gRPC calculator boundary.  Retain the established,
                # supported standard extraction path and label its semantics
                # explicitly in every result row.  It is a diagnostic complex
                # vector norm and is not used for the independent loss formula.
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

    def _calc_group_b_power_integral(self, objs, exponent, expr_name):
        """Legacy postprocess diagnostic; never used by the production gate."""
        exponent = float(exponent)
        if not math.isfinite(exponent) or exponent <= 0:
            raise ValueError(f"invalid B exponent: {exponent!r}")

        def _build(reporter):
            for i, obj in enumerate(objs):
                name = obj if isinstance(obj, str) else obj.name
                reporter.EnterQty("B")
                reporter.CalcOp("ComplxPeak")
                # Fractional powers of a dimensional field are rejected by
                # AEDT 2025.2's scalar Pow operation.  Normalize by the exact
                # 1-tesla design variable first; the resulting dimensionless
                # field has the same numerical value as B expressed in tesla.
                reporter.EnterScalarFunc(B_POWER_REFERENCE_VARIABLE)
                reporter.CalcOp("/")
                reporter.EnterScalar(exponent)
                reporter.CalcOp("Pow")
                reporter.EnterVol(name)
                reporter.CalcOp("Integrate")
                if i > 0:
                    reporter.CalcOp("+")

        return self._add_field_expression(expr_name, _build)

    def _calc_core_flux_integral(self, sheets, expr_name):
        """Integrate complex +Z B through center-leg XY section sheets.

        The +Z component is explicit because Maxwell warns that a standalone
        sheet's implicit normal can be ill-defined.  All section sheets use the
        same XY orientation and the complex phasors are summed before taking a
        magnitude, so no per-sheet sign cancellation is hidden.
        """
        sheets = list(sheets)
        if not sheets:
            raise RuntimeError("no center-leg section sheets for flux integral")

        def _build(reporter):
            for index, sheet in enumerate(sheets):
                name = sheet if isinstance(sheet, str) else sheet.name
                reporter.EnterQty("B")
                reporter.CalcOp("ScalarZ")
                reporter.EnterSurf(name)
                reporter.CalcOp("SurfaceValue")
                reporter.CalcOp("Integrate")
                if index > 0:
                    reporter.CalcOp("+")
            # Take magnitude only after summing complex flux phasors.
            reporter.CalcOp("CmplxMag")

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
        revision = (
            str(self.df_plus["physics_data_revision"].iloc[0]).strip()
            if "physics_data_revision" in self.df_plus.columns else ""
        )
        native_contract = revision == PHYSICS_DATA_REVISION

        # ---- 코어손실 + B ----
        # Native wound-core orientation splits one former core_i solid into
        # three legs and two yokes. Preserve the public P_core_i contract by
        # integrating all retained pieces of each physical depth group.
        core_groups = {}
        for core_obj in self.design1.core_objs:
            core_groups.setdefault(
                _core_group_index(core_obj.name), []
            ).append(core_obj)
        if not core_groups:
            raise RuntimeError("no core objects available for loss reporting")

        core_exprs = []
        b_mean_exprs = []
        b_max_exprs = []
        b_expr_group = {}
        b_expr_volume = {}
        native_report_plan = None

        for group_index, pieces in sorted(core_groups.items()):
            group_name = f"core_{group_index}"
            core_exprs.append(self._calc_group_loss(
                pieces, f"P_{group_name}", quantity="CoreLoss"
            ))

        if native_contract:
            native_report_plan = _native_core_report_plan(
                core_groups,
                self._sym_cut_count,
                require_complete_groups=bool(self.full_model),
            )

        # Per-piece B diagnostics retain the established CmplxMag->Mag
        # extraction.  The independent core-loss reference is evaluated in
        # Python from Faraday B and effective mass; no calculator B**y moment
        # participates in the production gate.
        for group_index, pieces in sorted(core_groups.items()):
            for piece in pieces:
                mean_expr = self._calc_field_expr(
                    piece.name, "B_peak", "Mean", f"B_mean_{piece.name}"
                )
                max_expr = self._calc_field_expr(
                    piece.name, "B_peak", "Maximum", f"B_max_{piece.name}"
                )
                try:
                    volume = abs(float(piece.volume))
                except Exception as exc:
                    raise RuntimeError(
                        f"cannot read core component volume for {piece.name!r}"
                    ) from exc
                if not math.isfinite(volume) or volume <= 0:
                    raise RuntimeError(
                        f"invalid core component volume for {piece.name!r}: {volume!r}"
                    )
                b_mean_exprs.append(mean_expr)
                b_max_exprs.append(max_expr)
                b_expr_group[mean_expr] = group_index
                b_expr_group[max_expr] = group_index
                b_expr_volume[mean_expr] = volume
        flux_expr = None
        flux_unavailable_reason = ""
        try:
            flux_expr = self._calc_core_flux_integral(
                self.design1.core_flux_sheets, "Phi_center_leg_B_normal"
            )
        except Exception as exc:
            detail = " ".join(str(exc).split()) or type(exc).__name__
            flux_unavailable_reason = f"grpc_calcop_unavailable:{detail}"
            logging.warning(
                "Center-leg surface-flux expression is unavailable; "
                f"continuing loss extraction: {flux_unavailable_reason}"
            )
        flux_section_area_retained_m2 = float("nan")
        if flux_expr is not None:
            try:
                flux_section_area_retained_m2 = sum(
                    _sheet_area_model_units(sheet)
                    for sheet in self.design1.core_flux_sheets
                ) * 1e-6
            except Exception as exc:
                raise RuntimeError(
                    "cannot read retained core flux-section area"
                ) from exc
            if not math.isfinite(flux_section_area_retained_m2) or (
                flux_section_area_retained_m2 <= 0
            ):
                raise RuntimeError(
                    "invalid retained core flux-section area: "
                    f"{flux_section_area_retained_m2!r}"
                )

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

        # Keep the optional CalcOp-based surface integral out of the mandatory
        # batch.  Its evaluation can fail independently without discarding the
        # already solved core/winding loss and standard B evidence.
        all_exprs = (
            core_exprs + b_mean_exprs + b_max_exprs
            + group_exprs + turn_exprs + plate_exprs
        )
        df_loss = self._export_field_report("calculator_report_loss", all_exprs)
        vals = df_loss.iloc[0, -len(all_exprs):]
        vals.index = all_exprs
        self.loss_map = {k: float(v) for k, v in vals.items()}

        flux_integral_reported_wb = float("nan")
        if flux_expr is not None:
            try:
                df_flux_integral = self._export_field_report(
                    "calculator_report_center_leg_flux",
                    [flux_expr],
                    extraction_key="center_leg_surface_flux_integral",
                )
                flux_integral_reported_wb = float(
                    df_flux_integral[flux_expr].iloc[0]
                )
                if not math.isfinite(flux_integral_reported_wb):
                    raise RuntimeError(
                        f"non-finite field result for {flux_expr}: "
                        f"{flux_integral_reported_wb!r}"
                    )
                self.loss_map[flux_expr] = flux_integral_reported_wb
            except Exception as exc:
                detail = " ".join(str(exc).split()) or type(exc).__name__
                flux_unavailable_reason = f"grpc_calcop_unavailable:{detail}"
                flux_expr = None
                flux_integral_reported_wb = float("nan")
                logging.warning(
                    "Center-leg surface-flux evaluation is unavailable; "
                    f"continuing loss extraction: {flux_unavailable_reason}"
                )

        # 실물 기준(_phys) 환산: 대칭 loss 디자인이면 오브젝트별 절단면 수로 보정, 풀모델이면 x1
        b_factor = 0.5 if getattr(self, "loss_is_sym", False) else 1.0
        self.loss_map_phys = {}
        self.loss_map_native_raw_phys = {}
        self.loss_map_average_phys = {}
        self.loss_map_macro_phys = {}
        loss_margin = (
            float(self.df_plus["core_loss_margin"].iloc[0])
            if native_contract else 1.0
        )
        for e in core_exprs:
            native_raw = self.loss_map[e] * self._phys_factor(
                e, is_core_loss=True
            )
            self.loss_map_native_raw_phys[e] = native_raw
            self.loss_map_phys[e] = native_raw * loss_margin
        for e in group_exprs + turn_exprs + plate_exprs:
            self.loss_map_phys[e] = self.loss_map[e] * self._phys_factor(e, is_core_loss=False)
        kf = float(self.df_plus["core_lamination_factor"].iloc[0])
        core_area_basis = str(
            self.df_plus["core_geometry_material_basis"].iloc[0]
        )
        for e in b_mean_exprs + b_max_exprs:
            # Symmetry conversion first recovers the macroscopic flux density
            # of the gross homogeneous model.  Only then convert to ribbon-
            # material flux density using the UU137 lamination factor.
            b_average = self.loss_map[e] * b_factor
            self.loss_map_average_phys[e] = b_average
            self.loss_map_macro_phys[e] = b_average
            self.loss_map_phys[e] = material_flux_density_t(
                b_average, kf, area_basis=core_area_basis
            )

        ae_gross_m2 = float(self.df_plus["Ae_gross_m2"].iloc[0])
        if flux_expr is not None:
            b_section_average = (
                flux_integral_reported_wb
                / flux_section_area_retained_m2
                * b_factor
            )
            b_section_material = material_flux_density_t(
                b_section_average, kf, area_basis=core_area_basis
            )
            flux_integral_physical_wb = b_section_average * ae_gross_m2
        else:
            b_section_average = float("nan")
            b_section_material = float("nan")
            flux_integral_physical_wb = float("nan")

        def _obj_of(expr):
            n = expr
            for pref in ("P_turn_", "P_"):
                if n.startswith(pref):
                    return n[len(pref):].replace("_group", "")
            return n

        # 총계 (대칭 모델의 삭제된 미러 몫 포함 - 실물 전체 기준)
        core_total_native_raw = sum(
            self.loss_map_native_raw_phys[e] * self._mirror_mult(_obj_of(e))
            for e in core_exprs
        )
        core_total = sum(
            self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
            for e in core_exprs
        )
        cplate_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                           for e in plate_exprs if "core_plate" in e)
        wcp_total = sum(self.loss_map_phys[e] * self._mirror_mult(_obj_of(e))
                        for e in plate_exprs if "wcp" in e)
        p_tx = self.loss_map_phys.get("P_Tx_main_group", 0.0)
        p_rxm = self.loss_map_phys.get("P_Rx_main_group", 0.0)
        p_rxs_one = self.loss_map_phys.get("P_Rx_side_group", 0.0)
        winding_total = p_tx + p_rxm + 2 * p_rxs_one  # 측면 링 2개 (좌우 대칭)

        group_b_average = {}
        group_volume = {}
        for group_index in core_groups:
            means = [
                e for e in b_mean_exprs if b_expr_group[e] == group_index
            ]
            volume = sum(b_expr_volume[e] for e in means)
            if volume <= 0:
                raise RuntimeError(f"zero retained core volume for group {group_index}")
            group_b_average[group_index] = sum(
                self.loss_map_average_phys[e] * b_expr_volume[e]
                for e in means
            ) / volume
            group_volume[group_index] = volume

        physical_volume_weights = {
            group_index: group_volume[group_index]
            * (2 ** self._sym_cut_count(f"core_{group_index}"))
            * self._mirror_mult(f"core_{group_index}")
            for group_index in group_b_average
        }
        total_volume_weight = sum(physical_volume_weights.values())
        if total_volume_weight <= 0:
            raise RuntimeError("zero physical core volume weight")
        b_mean_average = sum(
            group_b_average[index] * physical_volume_weights[index]
            for index in group_b_average
        ) / total_volume_weight
        b_max_average = max(
            (self.loss_map_average_phys[e] for e in b_max_exprs), default=0
        )
        b_mean_material = material_flux_density_t(
            b_mean_average, kf, area_basis=core_area_basis
        )
        b_max_material = max(
            (self.loss_map_phys[e] for e in b_max_exprs), default=0
        )

        primary_turns = int(self.df_plus["N1_main"].iloc[0]) + int(
            self.df_plus["N1_side"].iloc[0]
        )
        ae_effective_m2 = float(self.df_plus["Ae_effective_m2"].iloc[0])
        b_design_square_material = square_wave_b_material_t(
            float(self.df_plus["V1_rms"].iloc[0]),
            float(self.df_plus["freq"].iloc[0]),
            primary_turns,
            ae_effective_m2,
        )
        faraday_reference = faraday_lumped_core_reference(
            voltage_rms_v=float(self.df_plus["V1_rms"].iloc[0]),
            frequency_hz=float(self.df_plus["freq"].iloc[0]),
            turns=primary_turns,
            gross_area_m2=ae_gross_m2,
            lamination_factor=kf,
            effective_mass_kg=float(
                self.df_plus["core_mass_effective_kg"].iloc[0]
            ),
            loss_margin=loss_margin,
            coefficient=6.5,
            x=float(self.df_plus["core_x"].iloc[0]),
            y=float(self.df_plus["core_y"].iloc[0]),
        )
        b_ac_sine_material = faraday_reference["B_material_T"]
        b_ac_sine_average = faraday_reference["B_pack_T"]
        # Cross-check the algebraic gross-area route against the existing
        # effective-area helper.  This is deterministic Python math, not an
        # AEDT field-calculator identity.
        b_ac_sine_material_effective_area = sinusoidal_b_peak_material_t(
            float(self.df_plus["V1_rms"].iloc[0]),
            float(self.df_plus["freq"].iloc[0]),
            primary_turns,
            ae_effective_m2,
        )
        if not math.isclose(
            b_ac_sine_material,
            b_ac_sine_material_effective_area,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Faraday gross/effective area references disagree")

        core_loss_expected = faraday_reference["margin_adjusted_loss_W"]
        core_loss_expected_native_raw = faraday_reference["native_raw_loss_W"]
        core_loss_native_rel_error = float("nan")
        core_loss_native_attested = 0
        core_loss_native_tolerance_rel = 0.30
        b_mean_faraday_rel_error = abs(
            b_mean_material - b_ac_sine_material
        ) / max(abs(b_ac_sine_material), 1e-12)
        b_mean_faraday_tolerance_rel = 0.15
        b_mean_faraday_attested = int(
            math.isfinite(b_mean_faraday_rel_error)
            and b_mean_faraday_rel_error <= b_mean_faraday_tolerance_rel
        )
        if native_contract:
            denominator = max(abs(core_loss_expected_native_raw), 1e-12)
            core_loss_native_rel_error = abs(
                core_total_native_raw - core_loss_expected_native_raw
            ) / denominator
            if not b_mean_faraday_attested:
                raise RuntimeError(
                    "standard AEDT B average failed Faraday lumped reference: "
                    f"standard_material={b_mean_material:.12g}T, "
                    f"faraday_material={b_ac_sine_material:.12g}T, "
                    f"relative_error={b_mean_faraday_rel_error:.6g}, "
                    f"tolerance={b_mean_faraday_tolerance_rel:.6g}"
                )
            if not math.isfinite(core_loss_native_rel_error) or (
                core_loss_native_rel_error > core_loss_native_tolerance_rel
            ):
                raise RuntimeError(
                    "AEDT native lamination CoreLoss failed lumped Faraday/"
                    "POWERLITE/mass reference: "
                    f"solver_raw={core_total_native_raw:.12g}W, "
                    f"expected_raw={core_loss_expected_native_raw:.12g}W, "
                    f"relative_error={core_loss_native_rel_error:.6g}, "
                    f"tolerance={core_loss_native_tolerance_rel:.6g}"
                )
            core_loss_native_attested = 1

        if native_contract:
            native_group_count = len(native_report_plan["groups"])
            native_piece_count = len(native_report_plan["object_names"])
            native_expected_piece_count = sum(
                len(pieces) for pieces in core_groups.values()
            )
            native_pre_split_piece_count = int(
                self.df_plus["core_segmented_piece_count"].iloc[0]
            )
            native_coverage_attested = int(
                native_piece_count == native_expected_piece_count
                and len(set(native_report_plan["object_names"]))
                == native_piece_count
            )
            if not native_coverage_attested:
                raise RuntimeError(
                    "native core report coverage attestation failed after planning"
                )
            native_membership_sha256 = native_report_plan[
                "membership_sha256"
            ]
        else:
            native_group_count = 0
            native_piece_count = 0
            native_expected_piece_count = 0
            native_pre_split_piece_count = 0
            native_coverage_attested = 0
            native_membership_sha256 = ""

        # CSV에는 실물 기준 값을 기본으로 기록 (raw 대칭 적분값은 _raw 접미사)
        summary = {
            "P_core_total": [core_total],
            "P_core_total_native_raw_W": [core_total_native_raw],
            "P_core_total_expected_from_Bavg_integral": [core_loss_expected],
            "P_core_total_expected_from_faraday_mass": [core_loss_expected],
            "P_core_total_expected_native_raw_W": [core_loss_expected_native_raw],
            "P_core_total_expected_faraday_native_raw_W": [
                core_loss_expected_native_raw
            ],
            "core_loss_native_rel_error": [core_loss_native_rel_error],
            "core_loss_native_tolerance_rel": [core_loss_native_tolerance_rel],
            "core_loss_native_attested": [core_loss_native_attested],
            "core_loss_margin_applied": [loss_margin],
            "core_loss_reference_basis": [faraday_reference["basis"]],
            "core_loss_reference_specific_W_kg": [
                faraday_reference["specific_loss_W_kg"]
            ],
            "core_loss_reference_effective_mass_kg": [
                faraday_reference["effective_mass_kg"]
            ],
            "Bavg_power_volume_integral": [float("nan")],
            "Bavg_power_integral_normalized_by_one_tesla": [0],
            "Bavg_power_integral_status": [
                "disabled_by_user_use_independent_faraday_lumped_gate"
            ],
            "native_core_report_group_count": [native_group_count],
            "native_core_report_piece_count": [native_piece_count],
            "native_core_report_expected_piece_count": [
                native_expected_piece_count
            ],
            "native_core_pre_split_piece_count": [
                native_pre_split_piece_count
            ],
            "native_core_report_coverage_basis": [
                "all_retained_post_symmetry_core_objects"
                if native_contract else "not_applicable"
            ],
            "native_core_report_coverage_attested": [
                native_coverage_attested
            ],
            "native_core_report_membership_sha256": [
                native_membership_sha256
            ],
            "native_core_b_power_batch_count": [0],
            "native_core_b_power_expression_count": [0],
            "native_core_b_power_batches_json": ["{}"],
            "native_core_b_power_restore_factors_json": ["{}"],
            "core_flux_section_retained_area_m2": [
                flux_section_area_retained_m2
            ],
            "core_flux_integral_reported_Wb": [flux_integral_reported_wb],
            "core_flux_integral_physical_Wb": [flux_integral_physical_wb],
            "B_core_section_average": [b_section_average],
            "B_core_section_material": [b_section_material],
            "B_core_section_vs_volume_mean_rel_error": [
                abs(b_section_average - b_mean_average)
                / max(abs(b_mean_average), 1e-12)
                if math.isfinite(b_section_average) else float("nan")
            ],
            "B_design_square_material_analytic": [b_design_square_material],
            "B_ac_sine_material_analytic": [b_ac_sine_material],
            "B_ac_sine_average_analytic": [b_ac_sine_average],
            "B_faraday_pack_T": [faraday_reference["B_pack_T"]],
            "B_faraday_material_T": [faraday_reference["B_material_T"]],
            "B_standard_extraction_operator": ["CmplxMag_then_Mag"],
            "B_standard_extraction_semantics": [
                "complex_vector_norm_diagnostic_not_phase_peak"
            ],
            "B_mean_material_vs_sine_analytic_rel_error": [
                b_mean_faraday_rel_error
            ],
            "B_mean_faraday_tolerance_rel": [
                b_mean_faraday_tolerance_rel
            ],
            "B_mean_faraday_attested": [b_mean_faraday_attested],
            "P_core_plate_total": [cplate_total],
            "P_wcp_total": [wcp_total],
            "P_winding_total": [winding_total],
            "P_Rx_side_total": [2 * p_rxs_one],
            # Compatibility aliases now deliberately mean ribbon-material B.
            # The explicit macro/material columns prevent cross-revision
            # consumers from guessing which area basis was used.
            "B_mean_core": [b_mean_material],
            "B_max_core": [b_max_material],
            "B_mean_core_average": [b_mean_average],
            "B_max_core_average_diagnostic": [b_max_average],
            "B_mean_core_macro": [b_mean_average],
            "B_max_core_macro": [b_max_average],
            "B_mean_core_material": [b_mean_material],
            "B_max_core_material": [b_max_material],
            "B_max_core_usage": [
                "diagnostic_only_edge_and_segment_interface_spikes"
            ],
        }
        for e in core_exprs + group_exprs + turn_exprs + plate_exprs:
            summary[e] = [self.loss_map_phys[e]]
            if e in core_exprs:
                summary[f"{e}_native_raw"] = [
                    self.loss_map_native_raw_phys[e]
                ]
            if getattr(self, "loss_is_sym", False):
                summary[f"{e}_raw"] = [self.loss_map[e]]
        for e in b_mean_exprs + b_max_exprs:
            summary[f"{e}_average"] = [self.loss_map_average_phys[e]]
            summary[f"{e}_material"] = [self.loss_map_phys[e]]

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

        report_to_physical_flux_factor = (
            2.0 if getattr(self, "loss_is_sym", False) else 1.0
        )
        induced_voltage_reported_peak = float("nan")
        flux_linkage_reported_peak = float("nan")
        induced_voltage_peak = float("nan")
        flux_linkage_peak = float("nan")
        faraday_rel_error = float("nan")
        source_voltage_rel_error = float("nan")
        b_flux_linkage_material = float("nan")
        surface_flux_linkage_rel_error = float("nan")
        surface_flux_induced_voltage_rel_error = float("nan")
        flux_linkage_attested = 0
        winding_readback_available = False
        winding_readback_passed = False
        winding_readback_unavailable_reason = ""
        try:
            induced_expr = "mag(InducedVoltage(Tx_winding))"
            linkage_expr = "mag(FluxLinkage(Tx_winding))"
            df_flux = self._solution_data_frame(
                [induced_expr, linkage_expr],
                aliases=["induced_voltage_peak", "flux_linkage_peak"],
                target_units={induced_expr: "V", linkage_expr: "Wb"},
                report_category="AC Magnetic",
                extraction_key="flux_linkage",
            )
            induced_voltage_reported_peak = float(
                df_flux["induced_voltage_peak"].iloc[0]
            )
            flux_linkage_reported_peak = float(
                df_flux["flux_linkage_peak"].iloc[0]
            )
            induced_voltage_peak = (
                induced_voltage_reported_peak
                * report_to_physical_flux_factor
            )
            flux_linkage_peak = (
                flux_linkage_reported_peak
                * report_to_physical_flux_factor
            )
            omega = 2.0 * math.pi * float(self.df_plus["freq"].iloc[0])
            faraday_voltage = omega * flux_linkage_peak
            source_voltage_peak = (
                math.sqrt(2.0) * float(self.df_plus["V1_rms"].iloc[0])
            )
            faraday_rel_error = abs(
                induced_voltage_peak - faraday_voltage
            ) / max(abs(induced_voltage_peak), 1e-12)
            source_voltage_rel_error = abs(
                induced_voltage_peak - source_voltage_peak
            ) / max(abs(source_voltage_peak), 1e-12)
            b_flux_linkage_material = flux_linkage_peak / (
                primary_turns * ae_effective_m2
            )
            if flux_expr is not None:
                surface_linkage = primary_turns * flux_integral_physical_wb
                surface_flux_linkage_rel_error = abs(
                    surface_linkage - flux_linkage_peak
                ) / max(abs(flux_linkage_peak), 1e-12)
                surface_induced_voltage = omega * surface_linkage
                surface_flux_induced_voltage_rel_error = abs(
                    surface_induced_voltage - induced_voltage_peak
                ) / max(abs(induced_voltage_peak), 1e-12)
            winding_readback_passed = bool(
                math.isfinite(faraday_rel_error)
                and faraday_rel_error <= 0.01
                and math.isfinite(source_voltage_rel_error)
                and source_voltage_rel_error <= 0.05
            )
            if native_contract and not winding_readback_passed:
                raise RuntimeError(
                    "native stacking flux-linkage attestation failed: "
                    f"Faraday relative error={faraday_rel_error:.6g}, "
                    f"source relative error={source_voltage_rel_error:.6g}"
                )
            winding_readback_available = True
            flux_linkage_attested = int(native_contract)
        except Exception as exc:
            calcop_reason = _grpc_calcop_unavailability_reason(exc)
            if calcop_reason:
                winding_readback_unavailable_reason = calcop_reason
                logging.warning(
                    "Winding flux-linkage/induced-voltage readback is "
                    "unavailable; continuing loss extraction: "
                    f"{calcop_reason}"
                )
            elif native_contract:
                raise RuntimeError(
                    "new physics revision requires induced-voltage/flux-linkage "
                    "Faraday readback"
                ) from exc
            else:
                detail = " ".join(str(exc).split()) or type(exc).__name__
                winding_readback_unavailable_reason = (
                    f"readback_unavailable:{detail}"
                )
                logging.warning(f"Failed to extract flux-linkage audit: {exc}")

        winding_readback_evidence = {
            "status": (
                "available" if winding_readback_available else "unavailable"
            ),
            "applicable": winding_readback_available,
            "available": winding_readback_available,
            "passed": (
                winding_readback_passed
                if winding_readback_available
                else winding_readback_unavailable_reason.startswith(
                    "grpc_calcop_unavailable:"
                )
            ),
            "reason": (
                "" if winding_readback_available
                else winding_readback_unavailable_reason
            ),
            "reported_induced_voltage_peak_V": (
                induced_voltage_reported_peak
                if winding_readback_available else None
            ),
            "reported_flux_linkage_peak_Wb_turn": (
                flux_linkage_reported_peak
                if winding_readback_available else None
            ),
            "physical_induced_voltage_peak_V": (
                induced_voltage_peak if winding_readback_available else None
            ),
            "physical_flux_linkage_peak_Wb_turn": (
                flux_linkage_peak if winding_readback_available else None
            ),
            "required_physics_evidence": ["B_mean_faraday_attested"],
        }

        summary.update({
            "loss_report_to_physical_flux_factor": [
                report_to_physical_flux_factor
            ],
            "loss_report_flux_basis": [
                "eighth_current_driven_half_amplitude"
                if getattr(self, "loss_is_sym", False)
                else "full_voltage_driven_physical_amplitude"
            ],
            "Tx_induced_voltage_reported_peak_V": [
                induced_voltage_reported_peak
            ],
            "Tx_induced_voltage_peak_V": [induced_voltage_peak],
            "Tx_flux_linkage_reported_peak_Wb_turn": [
                flux_linkage_reported_peak
            ],
            "Tx_flux_linkage_peak_Wb_turn": [flux_linkage_peak],
            "Tx_flux_linkage_faraday_rel_error": [faraday_rel_error],
            "Tx_induced_vs_source_peak_rel_error": [source_voltage_rel_error],
            "B_flux_linkage_material": [b_flux_linkage_material],
            "core_surface_flux_vs_linkage_rel_error": [
                surface_flux_linkage_rel_error
            ],
            "core_surface_flux_vs_induced_voltage_rel_error": [
                surface_flux_induced_voltage_rel_error
            ],
            "B_flux_linkage_vs_sine_analytic_rel_error": [
                abs(b_flux_linkage_material - b_ac_sine_material)
                / max(abs(b_ac_sine_material), 1e-12)
                if math.isfinite(b_flux_linkage_material) else float("nan")
            ],
            "flux_linkage_attested": [flux_linkage_attested],
        })

        flux_available = flux_expr is not None
        surface_metric_passed = bool(
            not flux_available
            or (
                math.isfinite(surface_flux_linkage_rel_error)
                and surface_flux_linkage_rel_error <= 0.05
                and math.isfinite(surface_flux_induced_voltage_rel_error)
                and surface_flux_induced_voltage_rel_error <= 0.05
            )
        )
        center_leg_flux_evidence = {
            "status": "available" if flux_available else "unavailable",
            "applicable": flux_available,
            "available": flux_available,
            "passed": surface_metric_passed,
            "reason": "" if flux_available else flux_unavailable_reason,
            "reported_value_Wb": (
                flux_integral_reported_wb if flux_available else None
            ),
            "physical_value_Wb": (
                flux_integral_physical_wb if flux_available else None
            ),
            "equivalent_flux_evidence": [
                "Tx_flux_linkage_faraday_rel_error",
                "Tx_induced_vs_source_peak_rel_error",
            ],
        }
        self.loss_gate_evidence = {
            "center_leg_surface_flux_integral": center_leg_flux_evidence,
            "winding_flux_linkage_readback": winding_readback_evidence,
        }
        summary.update({
            "center_leg_surface_flux_integral_status": [
                center_leg_flux_evidence["status"]
            ],
            "center_leg_surface_flux_integral_applicable": [
                int(center_leg_flux_evidence["applicable"])
            ],
            "center_leg_surface_flux_integral_available": [
                int(center_leg_flux_evidence["available"])
            ],
            "center_leg_surface_flux_integral_passed": [
                int(center_leg_flux_evidence["passed"])
            ],
            "center_leg_surface_flux_integral_reason": [
                center_leg_flux_evidence["reason"]
            ],
            "center_leg_surface_flux_integral_evidence_json": [
                json.dumps(
                    {
                        "center_leg_surface_flux_integral": (
                            center_leg_flux_evidence
                        )
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ],
            "winding_flux_linkage_readback_status": [
                winding_readback_evidence["status"]
            ],
            "winding_flux_linkage_readback_applicable": [
                int(winding_readback_evidence["applicable"])
            ],
            "winding_flux_linkage_readback_available": [
                int(winding_readback_evidence["available"])
            ],
            "winding_flux_linkage_readback_passed": [
                int(winding_readback_evidence["passed"])
            ],
            "winding_flux_linkage_readback_reason": [
                winding_readback_evidence["reason"]
            ],
            "winding_flux_linkage_readback_evidence_json": [
                json.dumps(
                    {
                        "winding_flux_linkage_readback": (
                            winding_readback_evidence
                        )
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ],
        })

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

    def _validated_matrix_hpc_acf(self, acf_path=None):
        """Return the matrix solve's exact 4-core/one-engine DSO configuration."""
        if acf_path is None:
            matrix_design = getattr(self, "design_matrix", None)
            solver = getattr(matrix_design, "solver_instance", None)
            working_directory = str(
                getattr(solver, "working_directory", "") or ""
            ).strip()
            if not working_directory:
                raise RuntimeError(
                    "authoritative matrix HPC working directory is unavailable"
                )
            path = os.path.abspath(
                os.path.join(working_directory, "pyaedt_config.acf")
            )
        else:
            path = os.path.abspath(os.fspath(acf_path))
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

    def _capture_matrix_hpc_acf(
            self, max_attempts=5, retry_delay=0.5, sleeper=time.sleep):
        """Capture the matrix ACF before CopyDesign can stale its PyAEDT wrapper."""
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError("matrix HPC ACF capture max_attempts must be positive")
        errors = []
        for attempt in range(1, max_attempts + 1):
            try:
                path = self._validated_matrix_hpc_acf()
                self._matrix_hpc_acf_path = path
                return path
            except Exception as error:
                errors.append(
                    f"attempt {attempt}: {type(error).__name__}: {error}"
                )
                if attempt < max_attempts:
                    logging.warning(
                        "matrix HPC ACF capture failed "
                        f"(attempt {attempt}/{max_attempts}): {error}"
                    )
                    sleeper(
                        max(0.0, float(retry_delay)) * (2 ** (attempt - 1))
                    )
        raise RuntimeError(
            "matrix HPC ACF capture failed before copied-loss design creation: "
            + "; ".join(errors)
        )

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
        captured_acf = getattr(self, "_matrix_hpc_acf_path", None)
        if not captured_acf:
            raise RuntimeError(
                "captured matrix HPC ACF is unavailable before copied-loss solve"
            )
        # Revalidate the exact captured file and its full DSO contract without
        # calling the source design's stale oproject.GetPath after CopyDesign.
        acf_path = self._validated_matrix_hpc_acf(captured_acf)
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
        if not hasattr(self, "stage_timings"):
            self.stage_timings = {}

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

            # Preserve app.analyze() semantics, then make the bounded native
            # preflight the final operation before the only solve dispatch.
            # Repeating SetActiveDesign/GetRegistryString after a successful
            # preflight can itself wedge an otherwise healthy copied design.
            self.save_project(strict=True)
            logging.info(
                f"[{label}] preparing native Analyze Setup1 blocking=True"
            )
            context = self._prepare_copied_loss_native_analysis()
            dispatch_error = None
            restore_error = None
            analyze_result = None
            try:
                dispatch_design = context["odesign"]
                self.solve_attempts[label] = self.solve_attempts.get(label, 0) + 1
                t0 = time.time()
                try:
                    analyze_result = dispatch_design.Analyze("Setup1", True)
                except Exception as error:
                    dispatch_error = error
                elapsed = time.time() - t0
            finally:
                pending_error = sys.exc_info()[1]
                try:
                    self._restore_native_maxwell_dso(
                        context["registry_key"], context["original_config"]
                    )
                except Exception as error:
                    restore_error = error
                    if pending_error is not None and hasattr(pending_error, "add_note"):
                        pending_error.add_note(
                            f"native Maxwell DSO restore also failed: {error}"
                        )

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

        analyze_started = time.monotonic()
        elapsed = _analyze_once()
        solve_finished = time.monotonic()
        self.stage_timings[f"stage_time_{label}_solve_s"] = elapsed
        self.stage_timings[f"stage_time_{label}_analyze_overhead_s"] = max(
            0.0, solve_finished - analyze_started - elapsed
        )
        extraction_started = time.monotonic()
        try:
            extractor()
        finally:
            extraction_finished = time.monotonic()
            self.stage_timings[f"stage_time_{label}_extract_s"] = (
                extraction_finished - extraction_started
            )
            self.stage_timings[f"stage_time_{label}_analyze_total_s"] = (
                extraction_finished - analyze_started
            )
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
        row["loss_copy_prepare_attempts"] = int(getattr(
            self, "loss_copy_prepare_attempts", 0
        ))
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
        save_started = time.monotonic()

        def _record_save_timing():
            if not hasattr(self, "stage_timings"):
                self.stage_timings = {}
            self.stage_timings["stage_count_project_save"] = int(
                self.stage_timings.get("stage_count_project_save", 0)
            ) + 1
            self.stage_timings["stage_time_project_save_s"] = float(
                self.stage_timings.get("stage_time_project_save_s", 0.0)
            ) + (time.monotonic() - save_started)

        errors = []
        try:
            result = self.design1.save_project()
            if result is False:
                raise RuntimeError("wrapper save_project returned False")
            _record_save_timing()
            return True
        except Exception as error:
            errors.append(f"wrapper={type(error).__name__}: {error}")
        try:
            result = self._native_project_handle().Save()
            if result is False:
                raise RuntimeError("native Project.Save returned False")
            _record_save_timing()
            return True
        except Exception as error:
            errors.append(f"native={type(error).__name__}: {error}")
        message = "Failed to save project: " + "; ".join(errors)
        _record_save_timing()
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


def _create_simulation_session(max_attempts=3, retry_delay_s=30):
    """Create one healthy AEDT Desktop/Simulation pair with clean retries.

    High-concurrency nodes can occasionally expose a half-created gRPC Desktop
    (for example ``odesktop is None`` during ``EnableAutoSave``).  No candidate
    has been generated at this point, so retrying the session is safe and does
    not change the sampled design.  Every failed attempt releases its wrapper
    and terminates only descendants created after this helper started.
    """
    max_attempts = int(max_attempts)
    retry_delay_s = float(retry_delay_s)
    if max_attempts < 1 or retry_delay_s < 0:
        raise ValueError("invalid AEDT session retry policy")

    baseline_descendants = _snapshot_descendants()
    failures = []
    last_error = None
    for attempt in range(1, max_attempts + 1):
        desktop = None
        try:
            desktop = pyDesktop(
                version=None,
                non_graphical=GUI,
                close_on_exit=True,
                new_desktop=True,
            )
            simulation = Simulation(desktop=desktop)
            if simulation is None:
                raise RuntimeError("Simulation construction returned None")
            return desktop, simulation
        except Exception as error:
            last_error = error
            failures.append(f"{type(error).__name__}: {error}")
            logging.warning(
                "AEDT session startup attempt %d/%d failed: %s",
                attempt,
                max_attempts,
                failures[-1],
            )
            captured_descendants = _snapshot_descendants()
            if desktop is not None:
                try:
                    desktop.release_desktop(
                        close_projects=True,
                        close_on_exit=True,
                    )
                except Exception:
                    pass
            _terminate_spawned_descendants(
                baseline_descendants,
                captured_descendants,
                wait_s=5,
            )
            if attempt < max_attempts:
                time.sleep(retry_delay_s)

    raise RuntimeError(
        "AEDT desktop startup failed after "
        f"{max_attempts} attempts: {'; '.join(failures)}"
    ) from last_error


def run_one_loop(param=None, model_only=False, hold=False, golden=False, overrides=None):
    """
    param 이 None  -> 랜덤 파라미터 1회 (검증 실패 시 재추첨), 완료 후 프로젝트 삭제
    param 이 dict 등 -> 해당 값으로 1회 (fixed 모드), 프로젝트 폴더 보존
    model_only=True -> 모델링/셋업까지만 하고 해석은 생략 (지오메트리 확인용)
    """
    run_started = time.monotonic()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    fixed_mode = param is not None
    sim = None
    desktop = None
    held = [False]  # hold 성공 시 finally에서 desktop을 닫지 않기 위한 플래그
    delete_project_on_exit = not (fixed_mode or hold or model_only)
    baseline_descendants = _snapshot_descendants()
    try:
        fixed_input_df = None
        validated_physics_data_revision = None
        if fixed_mode:
            # Reject foreign physics payloads before starting AEDT. The
            # material-contract expansion below intentionally emits the code
            # pin, so validating after it would silently hide a mismatch.
            fixed_input_df, validated_physics_data_revision = (
                _load_fixed_input_parameter(param)
            )

        # pyDesktop을 context manager로 쓰면 release_desktop 이후 __exit__에서
        # close_on_exit 속성 오류가 발생하므로 직접 생성하고 finally에서 해제한다.
        aedt_startup_started = time.monotonic()
        desktop, sim = _create_simulation_session()
        sim.stage_timings.update({
            "stage_time_process_pre_run_s": max(
                0.0, run_started - _PROCESS_STARTED_MONOTONIC
            ),
            "stage_time_aedt_startup_s": (
                time.monotonic() - aedt_startup_started
            ),
        })

        project_input_started = time.monotonic()
        sim.create_simulation_name()
        sim.create_project()

        if fixed_mode:
            sim.input_df = fixed_input_df
            delete_project_on_exit = _project_delete_policy(
                sim.input_df, fixed_mode=True, hold=hold, model_only=model_only
            )
            # 위반 시 이유를 담아 ValueError raise
            _, sim.df_plus = validation_check(sim.input_df, strict=True)
            # Echo the value that passed the exact payload check, rather than
            # relying on a downstream contract builder to replace it.
            sim.df_plus["physics_data_revision"] = (
                validated_physics_data_revision
            )
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

        sim.stage_timings["stage_time_project_input_s"] = (
            time.monotonic() - project_input_started
        )

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

        def _prepare_loss_copy_once(before_names, _attempt):
            """maxwell_matrix를 복제해 loss_sym 디자인으로 전환 (모델링 절반 절약).
            레퍼런스: pyaedt_library/example/MFT_TAB second_simulation()"""
            import math as _m
            op = sim.project.desktop.odesktop.SetActiveProject(sim.project.name)
            old_design = sim.design_matrix
            source_name = _aedt_design_name(
                getattr(old_design, "design_name", "")
            )
            if source_name != "maxwell_matrix":
                raise RuntimeError(
                    f"loss copy source is not maxwell_matrix: {source_name!r}"
                )
            # The reference implementation gives AEDT five seconds to commit
            # the solved source design before CopyDesign. Shorter matrix runs
            # exposed a copied design with no solution type or Setup1.
            sim.save_project(strict=True)
            time.sleep(5)
            op.CopyDesign(source_name)
            op.Paste()
            new_design, copied_setup = _wait_for_ready_copied_loss_design(
                op, before_names,
                lambda name, solution: sim.project.create_design(
                    name=name, solver="maxwell3d", solution=solution,
                ),
            )
            sim.design1 = new_design
            copied_name = _aedt_design_name(
                getattr(new_design, "design_name", "")
            )
            if copied_name == source_name or copied_name in before_names:
                raise RuntimeError(
                    f"fresh copied design identity is invalid: {copied_name!r}"
                )
            wrapper_raw = getattr(
                getattr(new_design, "solver_instance", None), "odesign", None
            )
            if wrapper_raw is None:
                raise RuntimeError("fresh copied wrapper has no native odesign")
            _validate_raw_copied_loss_design(wrapper_raw, copied_name)
            active_raw = op.GetActiveDesign()
            _validate_raw_copied_loss_design(active_raw, copied_name)

            # 모델링 때 래퍼에 저장된 객체 핸들들을 복제 디자인으로 리매핑
            # (save_calculation/save_loss_reports가 소비 - MFT_TAB 레퍼런스 패턴)
            _remap_copied_design_objects(
                old_design,
                new_design,
                (
                    "Tx_windings_main", "Tx_windings_side", "Tx_windings_side2",
                    "Tx_windings", "Rx_windings_main", "Rx_windings_side",
                    "Rx_windings_side2", "Rx_windings", "core_objs",
                    "core_flux_sheets", "core_plates", "core_pads",
                    "wcp_plates", "wcp_pads",
                ),
            )

            # matrix 파라미터 제거 (loss 디자인에는 불필요한 연산)
            od = op.GetActiveDesign()
            _validate_raw_copied_loss_design(od, copied_name)
            _validate_raw_copied_loss_design(wrapper_raw, copied_name)
            try:
                od.GetModule("MaxwellParameterSetup").DeleteParameters(["Matrix"])
            except Exception as error:
                raise RuntimeError("matrix parameter deletion failed on loss copy") from error

            # 여자 전류를 loss_sym 페이저로 제자리 수정 (타입 동일: Current)
            I2 = float(sim.df_plus["I2_rated"].iloc[0])
            phase2 = getattr(sim, "I2_phase_auto", None)
            if phase2 is None:
                phase2 = float(sim.df_plus["I2_phase_deg"].iloc[0])
            tx_current = f"{sim.loss_I1_peak}A"
            tx_phase = f"{sim.loss_I1_phase_deg}deg"
            rx_current = f"{I2 * _m.sqrt(2)}A"
            rx_phase = f"{phase2}deg"
            tx, rx = _configure_copied_loss_excitations(
                new_design, wrapper_raw, copied_name,
                tx_current, tx_phase, rx_current, rx_phase,
            )
            sim.tx_winding, sim.rx_winding = tx, rx

            # 복제 디자인이 물려받은 matrix 해를 삭제 - 안 지우면 여자를 바꿔도
            # 솔버가 재해석 없이 '해 없음 완료'로 끝남 (로컬 랜덤 검증에서 3/3 재현)
            _delete_copied_solution_or_raise(
                wrapper_raw,
                op.GetActiveDesign(),
                copied_name,
            )

            # 코어손실 + skin 메시(손실 정밀용) + 셋업 정밀값
            _validate_raw_copied_loss_design(op.GetActiveDesign(), copied_name)
            _validate_raw_copied_loss_design(wrapper_raw, copied_name)
            core_names = [item.name for item in new_design.core_objs]
            _assign_native_copied_core_loss(
                wrapper_raw, copied_name, core_names
            )
            _configure_loss_copy_skin_mesh(
                sim, native_windings_solid=True
            )
            _validate_raw_copied_loss_design(op.GetActiveDesign(), copied_name)
            _validate_raw_copied_loss_design(wrapper_raw, copied_name)
            _assert_native_copied_loss_windings(
                wrapper_raw, copied_name,
                tx_current, tx_phase, rx_current, rx_phase,
                require_solid=True,
            )
            _assert_native_core_loss_assignment(wrapper_raw, core_names)
            new_design.setup = _configure_copied_loss_setup(
                copied_setup,
                max_passes=sim.df_plus["max_passes"].iloc[0],
                min_converged=sim.df_plus["min_converged"].iloc[0],
                percent_error=sim.df_plus["percent_error"].iloc[0],
            )
            _validate_raw_copied_loss_design(op.GetActiveDesign(), copied_name)
            _validate_raw_copied_loss_design(wrapper_raw, copied_name)
            sim.save_project(strict=True)
            _validate_saved_copied_loss_preparation(
                os.path.join(
                    sim.project_path, f"{sim.PROJECT_NAME}.aedt"
                ),
                source_name, copied_name,
                tx_current, tx_phase, rx_current, rx_phase,
                len(core_names),
                sim.df_plus["max_passes"].iloc[0],
                sim.df_plus["min_converged"].iloc[0],
                sim.df_plus["percent_error"].iloc[0],
                sim.df_plus["freq"].iloc[0],
                require_source_solved=not model_only,
            )
            # A copied pyDesign owns a separately constructed PyAEDT application
            # wrapper.  Its cached Desktop proxy is not trusted for solve dispatch;
            # analyze_and_extract uses the original Desktop and native design once.
            return new_design

        def _build_loss_by_copy():
            """Prepare a fresh copied loss design without ever re-solving Matrix."""
            source_design = sim.design_matrix
            source_tx_winding = sim.tx_winding
            source_rx_winding = sim.rx_winding
            temporary_names = (
                "Tx_skin_depth_mesh", "Rx_skin_depth_mesh", "Rx_length_mesh",
                "plate_skin_depth_mesh", "loss_winding_solid_update_count",
                "loss_winding_mesh_operation_count",
                "loss_conductor_mesh_operation_count",
                "loss_plate_eddy_on_readback_count",
                "loss_native_analyze_required", "loss_copy_prepare_attempts",
                "design_loss",
            )
            baseline = {
                name: (name in sim.__dict__, sim.__dict__.get(name))
                for name in temporary_names
            }

            def _rollback_failed_prepare():
                sim.design1 = source_design
                sim.tx_winding = source_tx_winding
                sim.rx_winding = source_rx_winding
                for name, (existed, value) in baseline.items():
                    if existed:
                        sim.__dict__[name] = value
                    else:
                        sim.__dict__.pop(name, None)

            if not model_only:
                sim._capture_matrix_hpc_acf()

            def _attempt(before_names, attempt):
                _rollback_failed_prepare()
                try:
                    return _prepare_loss_copy_once(before_names, attempt)
                except Exception:
                    _rollback_failed_prepare()
                    raise

            op = sim.project.desktop.odesktop.SetActiveProject(sim.project.name)
            try:
                new_design, attempts = _retry_copied_loss_preparation(
                    op, "maxwell_matrix", _attempt,
                    max_attempts=3, retry_delay_s=5.0,
                    require_source_solved=not model_only,
                )
            except Exception:
                _rollback_failed_prepare()
                raise
            sim.design1 = new_design
            sim.loss_copy_prepare_attempts = attempts
            sim.loss_native_analyze_required = True
            logging.info(
                "copied loss design became fully ready on prepare attempt %s/3; "
                "the exact solved Matrix source was reused",
                attempts,
            )

        result_parts = [sim.df_plus]
        total_time = 0.0

        # ---- design1: L/k 매트릭스 (전류원, 기존 방식) ----
        if matrix_on:
            matrix_model_started = time.monotonic()
            _build_em_design("maxwell_matrix", "matrix")
            sim.stage_timings["stage_time_matrix_model_s"] = (
                time.monotonic() - matrix_model_started
            )
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
            loss_prepare_started = time.monotonic()
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
            sim.stage_timings["stage_time_loss_prepare_s"] = (
                time.monotonic() - loss_prepare_started
            )
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
            from module.thermal_260706 import (
                _core_thermal_conductivity_contract,
                run_thermal_analysis,
            )
            core_conductivity = _core_thermal_conductivity_contract(
                sim.df_plus
            )
            t0 = time.monotonic()
            try:
                df_thermal = run_thermal_analysis(sim)
            except Exception as thermal_error:
                logging.exception(f"thermal stage failed: {thermal_error}")
                log_failed_sample(
                    sim.input_df,
                    f"thermal: {type(thermal_error).__name__}: {thermal_error}",
                )
                df_thermal = _thermal_failure_frame(
                    thermal_error, core_conductivity=core_conductivity
                )
            thermal_result_valid = _thermal_result_is_valid(
                df_thermal,
                physics_data_revision=sim.df_plus.get(
                    "physics_data_revision", pd.Series([""])
                ).iloc[0],
            )
            t_thermal = time.monotonic() - t0
            sim.stage_timings["stage_time_thermal_total_s"] = t_thermal
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

        pre_result_finished = time.monotonic()
        sim.stage_timings["stage_time_pre_result_s"] = (
            pre_result_finished - run_started
        )
        sim.stage_timings["stage_time_process_to_result_s"] = (
            pre_result_finished - _PROCESS_STARTED_MONOTONIC
        )
        nonoverlapping_keys = (
            "stage_time_aedt_startup_s",
            "stage_time_project_input_s",
            "stage_time_matrix_model_s",
            "stage_time_matrix_analyze_total_s",
            "stage_time_loss_prepare_s",
            "stage_time_loss_analyze_total_s",
            "stage_time_thermal_total_s",
        )
        accounted = sum(
            float(sim.stage_timings.get(key, 0.0))
            for key in nonoverlapping_keys
        )
        sim.stage_timings["stage_time_unattributed_s"] = max(
            0.0, sim.stage_timings["stage_time_pre_result_s"] - accounted
        )
        timing_frame = pd.DataFrame([{
            "timing_schema": "mft-stage-timing-v1",
            "timing_process_started_at_utc": _PROCESS_STARTED_AT_UTC,
            "timing_run_started_at_utc": run_started_at_utc,
            "timing_pre_result_at_utc": datetime.now(timezone.utc).isoformat(),
            **sim.stage_timings,
        }])
        simulation_time = pd.DataFrame({"time": [total_time]})
        result = pd.concat(
            result_parts + [sim.get_execution_telemetry(), timing_frame,
                            simulation_time], axis=1
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
