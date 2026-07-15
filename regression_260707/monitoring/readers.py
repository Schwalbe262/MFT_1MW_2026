"""Safe readers and view-model builders for MFT campaign artifacts.

Most readers intentionally depend only on the Python standard library.  The
campaign reader loads the lossless Parquet audit dataset lazily so the
electrostatic fields that are not present in ``train_io.csv`` remain visible.
A missing Parquet dependency, partially written file, or corrupt artifact is
still isolated from the rest of the dashboard.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    from ..model_targets import (
        SURROGATE_TEMPERATURE_TARGETS,
        SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
    )
except ImportError:  # Direct execution with regression_260707 on sys.path.
    from model_targets import (
        SURROGATE_TEMPERATURE_TARGETS,
        SURROGATE_WINDING_COMPONENT_LOSS_TARGETS,
    )

try:
    from module.core_material_contract import (
        PHYSICS_DATA_REVISION as CURRENT_PHYSICS_DATA_REVISION,
        solver_revision_matches_physics_cohort,
    )
except Exception as exc:  # The dashboard must remain available if repo imports fail.
    CURRENT_PHYSICS_DATA_REVISION: str | None = None
    solver_revision_matches_physics_cohort = None
    PHYSICS_DATA_REVISION_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    PHYSICS_DATA_REVISION_IMPORT_ERROR = None

try:
    try:
        from ..training.checkpoint_contract import (
            checkpoint_status_revision_identity_matches,
        )
    except ImportError:
        from training.checkpoint_contract import (
            checkpoint_status_revision_identity_matches,
        )
except Exception:  # Keep the dashboard fail-soft when repo imports are unavailable.
    checkpoint_status_revision_identity_matches = None


SCHEMA_VERSION = 1
DATA_GOAL = 3_000
STRETCH_GOAL = 10_000
LEGACY_PHYSICS_DATA_REVISION = "legacy_unspecified"
THERMAL_MODEL_TAGS = (
    "isotropic_legacy",
    "anisotropic_wound_rule_of_mixtures_v1",
)
CAPACITANCE_FIELDS = {
    "tx_tx": "C_tx_tx_F",
    "rx_rx": "C_rx_rx_F",
    "tx_rx": "C_tx_rx_F",
}
RESONANCE_FIELDS = {
    "tx_self": "f_res_tx_self_Hz",
    "rx_self": "f_res_rx_self_Hz",
    "interwinding": "f_res_interwinding_Hz",
}
# The scheduler owns the authoritative bounds.  These values are used only
# while an older scheduler has no simulation-policy capability to advertise.
# Keep the fallback in one place so the UI never grows its own competing cap.
DEFAULT_PARALLEL_TARGET_MIN = 0
DEFAULT_PARALLEL_TARGET_MAX = 500
PARALLEL_TARGET_SAFETY_CEILING = 600
NODE_LOCAL_AEDT_PROJECT = "_aedt_pool_hosts"
NODE_LOCAL_AEDT_ENTRYPOINT = "aedt_node_canary_host"
NODE_LOCAL_AEDT_ACTIVE_STATES = ("queued", "attaching", "running")
NODE_LOCAL_AEDT_TASK_LIMIT = 1_000
DEFAULT_CONTROLLER_STATE_PATH = (
    "C:/Users/peets/slurm_scheduler_runtime/mft_controller/"
    "restart_v3_7_controller_state.json"
)
DEFAULT_CONTROLLER_LOG_PATH = (
    "C:/Users/peets/slurm_scheduler_runtime/mft_controller/"
    "controller_restart_v3_7.log"
)
CONTROLLER_LOG_TAIL_BYTES = 128 * 1024
CONTROLLER_STATE_MAX_BYTES = 8 * 1024 * 1024
SIMULATION_TIMING_WINDOW_ROWS = 100
MAPE_ZERO_ABS_TOLERANCE = 1e-9
SIMULATION_TIMING_FIELDS = (
    ("matrix", "time_matrix"),
    ("loss", "time_loss"),
    ("icepak", "time_thermal"),
    ("total", "time"),
)
CAP_TIMING_FIELDS = ("cap_solve_time_s", "cap_extraction_time_s")
CHECKPOINT_STATE_SCHEMA_VERSION = 2
CHECKPOINT_METRICS_SCHEMA_VERSION = 1
CHECKPOINT_PARITY_SCHEMA_VERSION = 1
CHECKPOINT_PARITY_ARTIFACT_TYPE = "checkpoint_cv_oof_parity"
CHECKPOINT_PARITY_PAIR_LIMIT = 2_000
TARGETS: tuple[dict[str, str], ...] = (
    {"name": "Llt_phys", "label": "누설 인덕턴스 (Llt)", "unit": "µH"},
    {"name": "P_winding_total", "label": "권선 손실", "unit": "W"},
    {"name": "P_core_total", "label": "코어 손실", "unit": "W"},
    {"name": "P_core_plate_total", "label": "코어 플레이트 손실", "unit": "W"},
    {"name": "P_wcp_total", "label": "권선 냉각판 손실", "unit": "W"},
    {"name": "P_Tx_main_group", "label": "1차 권선 손실", "unit": "W"},
    {"name": "P_Rx_main_group", "label": "2차 중앙 권선 손실", "unit": "W"},
    {"name": "P_Rx_side_total", "label": "2차 측면 권선 손실", "unit": "W"},
    {"name": "Tprobe_Tx_leeward_max", "label": "Tx 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_main_leeward_max", "label": "Rx main 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_side_leeward_max", "label": "Rx side 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_center_max", "label": "코어 최대 온도(3영역 최대)", "unit": "°C"},
    {"name": "Tprobe_core_center_leg_max", "label": "코어 중앙 레그 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_side_leg_max", "label": "코어 사이드 레그 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_top_yoke_max", "label": "코어 상부 요크 최대 온도", "unit": "°C"},
    {"name": "k", "label": "결합계수 (k)", "unit": ""},
    {"name": "B_mean_core", "label": "코어 평균 자속밀도", "unit": "T"},
)
TARGET_META = {item["name"]: item for item in TARGETS}
TEMPERATURE_TARGETS = tuple(SURROGATE_TEMPERATURE_TARGETS)
DESIGN_PARAMETER_KEYS = (
    "N1_main", "N1_side", "N2_main", "N2_side", "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t",
    "cw1", "gap1", "cw2", "gap2",
    "nwh1", "nwh2", "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y", "w1c_w2s_gap_x_actual",
    "w1s_cs_space_x", "cs_w1s_space_y", "h_gap2", "wcp_t", "wcp_pad_t",
    "wcp_len_pct", "wcp_len_x",
)
CANDIDATE_REPORT_FIELDS = (
    "size_W_mm", "size_L_mm", "size_H_mm", "size_WxLxH_mm",
    "volume_L", "footprint_cm2",
    "turns_primary", "turns_secondary_center", "turns_secondary_side",
    "cw1_conductor_thickness_mm", "cw2_conductor_thickness_mm",
    "gap1_mm", "gap2_mm",
    "nwl1_main_pack_width_mm", "nwl1_side_pack_width_mm",
    "nwl2_main_pack_width_mm", "nwl2_side_pack_width_mm",
    "nwh1_winding_height_mm", "nwh2_winding_height_mm",
    "core_depth_each_mm", "n_core_group",
    "core_cold_plate_thickness_mm", "core_thermal_pad_thickness_mm",
    "winding_cold_plate_thickness_mm", "winding_thermal_pad_thickness_mm",
    "wcp_len_pct", "wcp_len_x_mm",
    "leakage_target_uH", "pred_leakage_inductance_uH",
    "B_design_analytic_T", "B_legacy_0p7_T", "B_design_waveform",
    "B_denominator_coefficient", "Ae_m2", "Ae_gross_m2",
    "Ae_effective_m2", "core_lamination_factor", "B_area_basis",
    "pred_core_loss_W", "pred_core_cold_plate_loss_W",
    "pred_winding_cold_plate_loss_W", "pred_primary_winding_loss_W",
    "pred_secondary_center_winding_loss_W",
    "pred_secondary_side_winding_loss_W", "pred_secondary_winding_loss_W",
    "pred_component_winding_loss_sum_W", "pred_total_winding_loss_W",
    "pred_total_loss_W", "rated_power_W", "pred_efficiency_pct",
    "surrogate_output_basis",
)
INSULATION_KEYS = (
    "cc_w2c_space_x", "cc_w2c_space_y", "w2c_w1c_space_x",
    "w2c_w1c_space_y", "w1c_w2s_gap_x_actual", "w1s_cs_space_x",
    "cs_w1s_space_y", "h_gap2",
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _duration_seconds(value: Any) -> float | None:
    """Return a non-negative finite duration without inventing missing timing."""
    if isinstance(value, bool):
        return None
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _simulation_timing_summary(
    frame: Any,
    local_tz=None,
    limit: int = SIMULATION_TIMING_WINDOW_ROWS,
    active_cohort: tuple[str, str] | None = None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    """Summarize recent timing fields for the dynamically active cohort."""
    columns = (
        "git_hash", "physics_data_revision", "saved_at",
        *(source_field for _, source_field in SIMULATION_TIMING_FIELDS),
        "cap_on", *CAP_TIMING_FIELDS,
    )
    rows = _frame_records(frame, columns)
    if active_cohort is None:
        active_cohort = _active_cohort_identity(
            rows, local_tz, current_physics_revision
        )
    active = _active_cohort_view(
        active_cohort, current_physics_revision
    )
    cohort_rows = [
        row for row in rows
        if _is_active_cohort(_cohort_identity(row), active_cohort)
    ]
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(cohort_rows):
        stamp = _parse_time(row.get("saved_at"), local_tz)
        ranked.append((stamp.timestamp() if stamp else float("-inf"), index, row))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    recent = [item[2] for item in ranked[:max(0, int(limit))]]
    stages = {}
    for name, source_field in SIMULATION_TIMING_FIELDS:
        values = [
            duration for row in recent
            if (duration := _duration_seconds(row.get(source_field))) is not None
        ]
        stages[name] = {
            "source_field": source_field,
            "sample_count": len(values),
            "mean_seconds": sum(values) / len(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
        }
    electrostatic_values = []
    for row in recent:
        if _optional_flag(row.get("cap_on")) is not True:
            continue
        cap_parts = [
            _duration_seconds(row.get(source_field))
            for source_field in CAP_TIMING_FIELDS
        ]
        if all(value is not None for value in cap_parts):
            electrostatic_values.append(sum(cap_parts))
    stages["electrostatic"] = {
        "source_fields": list(CAP_TIMING_FIELDS),
        "sample_count": len(electrostatic_values),
        "mean_seconds": (
            sum(electrostatic_values) / len(electrostatic_values)
            if electrostatic_values else None
        ),
        "median_seconds": (
            statistics.median(electrostatic_values)
            if electrostatic_values else None
        ),
    }
    return {
        "available": any(stage["sample_count"] for stage in stages.values()),
        "cohort_basis": "active_identity",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "cohort_rows": len(cohort_rows),
        "unit": "seconds",
        "window_limit_rows": max(0, int(limit)),
        "window_rows": len(recent),
        "stages": stages,
    }


def _zero_aware_percentage_metrics(
    actual_values: list[Any],
    predicted_values: list[Any],
    zero_abs_tolerance: float = MAPE_ZERO_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Calculate APE metrics without dividing by structural zero targets."""
    tolerance = _finite_number(zero_abs_tolerance)
    if tolerance is None or tolerance < 0:
        raise ValueError("MAPE zero tolerance must be finite and non-negative")
    if len(actual_values) != len(predicted_values):
        raise ValueError("MAPE actual/predicted lengths differ")
    relative_errors: list[float] = []
    valid_pair_count = 0
    excluded_zero_count = 0
    for raw_actual, raw_predicted in zip(actual_values, predicted_values):
        actual = _finite_number(raw_actual)
        predicted = _finite_number(raw_predicted)
        if actual is None or predicted is None:
            continue
        valid_pair_count += 1
        if abs(actual) <= tolerance:
            excluded_zero_count += 1
            continue
        relative_errors.append(abs(predicted - actual) / abs(actual))
    return {
        "mape_pct": (
            statistics.mean(relative_errors) * 100
            if relative_errors else None
        ),
        "p90_ape_pct": (
            statistics.quantiles(relative_errors, n=10, method="inclusive")[8] * 100
            if len(relative_errors) >= 2
            else relative_errors[0] * 100 if relative_errors else None
        ),
        "mape_n": len(relative_errors),
        "mape_excluded_zero_count": excluded_zero_count,
        "mape_valid_pair_count": valid_pair_count,
        "mape_zero_abs_tolerance": tolerance,
    }


def _integer(value: Any, default: int = 0) -> int:
    number = _finite_number(value)
    return int(number) if number is not None else default


def _flag(value: Any) -> bool:
    number = _finite_number(value)
    if number is not None:
        return number == 1.0
    return str(value).strip().lower() in {"true", "yes", "pass", "passed"}


def _safe_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _parse_time(value: Any, local_tz=None) -> datetime | None:
    text = _safe_text(value, 80)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%y%m%d_%H%M%S_%f", "%y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz or _now().tzinfo)
    return parsed


def _coerce(value: Any) -> Any:
    number = _finite_number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return _safe_text(value)


def _optional_text(value: Any, limit: int = 500) -> str | None:
    """Normalize schema-union nulls without turning NaN into a cohort tag."""
    if value is None:
        return None
    number = _finite_number(value)
    if number is None and isinstance(value, (float, int)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "<na>", "none", "null"}:
        return None
    return text[:limit]


def _cohort_identity(row: dict[str, Any]) -> tuple[str, str]:
    """Return the normalized identity used by every campaign cohort reader."""
    revision = (_optional_text(row.get("git_hash"), 40) or "").lower()
    physics_revision = (
        _optional_text(row.get("physics_data_revision"), 160)
        or LEGACY_PHYSICS_DATA_REVISION
    )
    return revision, physics_revision


def _active_cohort_identity(
    frame: Any,
    local_tz=None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> tuple[str, str] | None:
    """Select the newest solver identity for the deployed physics revision.

    Rows without a comparable ``saved_at`` cannot establish recency and are
    ignored.  If the newest comparable row lacks a solver hash, no active pair
    is claimed instead of silently selecting an older solver cohort.
    """
    expected = _optional_text(current_physics_revision, 160)
    if expected is None:
        return None
    ranked: list[tuple[float, int, tuple[str, str]]] = []
    for index, row in enumerate(_frame_records(
        frame, ("git_hash", "physics_data_revision", "saved_at")
    )):
        identity = _cohort_identity(row)
        if identity[1] != expected:
            continue
        stamp = _parse_time(row.get("saved_at"), local_tz)
        if stamp is None:
            continue
        ranked.append((
            stamp.timestamp(),
            index,
            identity,
        ))
    if not ranked:
        return None
    identity = max(ranked, key=lambda item: (item[0], item[1]))[2]
    return identity if identity[0] else None


def _is_active_cohort(
    identity: tuple[str, str],
    active_cohort: tuple[str, str] | None,
) -> bool:
    if active_cohort is None or identity[1] != active_cohort[1]:
        return False
    if solver_revision_matches_physics_cohort is None:
        return identity == active_cohort
    return solver_revision_matches_physics_cohort(
        identity[0], active_cohort[0], identity[1]
    )


def _active_cohort_view(
    active_cohort: tuple[str, str] | None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    expected = _optional_text(current_physics_revision, 160)
    if active_cohort is not None:
        git_hash, physics_revision = active_cohort
        return {
            "available": True,
            "status": "active",
            "git_hash": git_hash,
            "git_hash_short": git_hash[:10],
            "physics_data_revision": physics_revision,
            "expected_physics_data_revision": expected,
            "label": f"활성 코호트 {git_hash[:10]}",
        }
    if expected is not None:
        label = f"현재 revision 데이터 없음 · 기대 revision: {expected}"
        status = "no_current_revision_rows"
    else:
        label = "현재 revision 확인 불가 · PHYSICS_DATA_REVISION import 실패"
        status = "physics_revision_unavailable"
    return {
        "available": False,
        "status": status,
        "git_hash": None,
        "git_hash_short": None,
        "physics_data_revision": None,
        "expected_physics_data_revision": expected,
        "label": label,
    }


def _optional_flag(value: Any) -> bool | None:
    if value is None:
        return None
    number = _finite_number(value)
    if number is not None:
        if number == 1.0:
            return True
        if number == 0.0:
            return False
        return None
    text = str(value).strip().casefold()
    if text in {"true", "yes", "pass", "passed"}:
        return True
    if text in {"false", "no", "fail", "failed"}:
        return False
    return None


def _frame_records(frame: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Project a pandas-like frame or record iterable onto bounded columns."""
    if frame is None:
        return []
    if isinstance(frame, dict):
        return [{key: frame.get(key) for key in columns if key in frame}]
    if isinstance(frame, (list, tuple)):
        return [
            {key: row.get(key) for key in columns if key in row}
            for row in frame if isinstance(row, dict)
        ]
    available = getattr(frame, "columns", ())
    try:
        selected = [key for key in columns if key in available]
        if not selected:
            return [{} for _ in range(len(frame))]
        projected = frame.loc[:, selected]
        records = projected.to_dict(orient="records")
    except (AttributeError, KeyError, TypeError, ValueError):
        return []
    return [dict(row) for row in records if isinstance(row, dict)]


def _scaled_stats(
    rows: list[dict[str, Any]],
    column: str,
    scale: float,
    suffix: str,
) -> dict[str, Any]:
    values = [
        value * scale
        for row in rows
        if (value := _finite_number(row.get(column))) is not None
    ]
    return {
        "source_column": column,
        "sample_count": len(values),
        f"min_{suffix}": min(values) if values else None,
        f"median_{suffix}": statistics.median(values) if values else None,
        f"max_{suffix}": max(values) if values else None,
    }


def _plain_stats(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    values = [
        value for row in rows
        if (value := _finite_number(row.get(column))) is not None
    ]
    return {
        "source_column": column,
        "sample_count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _invalid_reasons(row: dict[str, Any]) -> list[str]:
    raw = _optional_text(row.get("_strict_invalid_reasons"), 20_000)
    if not raw:
        raw = _optional_text(row.get("em_validity_reason"), 20_000)
    if not raw:
        return []
    return list(dict.fromkeys(
        reason.strip() for reason in raw.split(";") if reason.strip()
    ))


def _campaign_frame_summary(
    frame: Any,
    now: datetime,
    active_cohort: tuple[str, str] | None = None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    """Summarize the campaign audit frame while tolerating missing columns.

    Canonically audited frames carry ``_strict_*`` columns.  Their per-row
    classifications are aggregated by physics revision without imposing an
    additional active-SHA gate.  Synthetic and CSV fallback rows may only
    carry stored result flags; those remain fail-closed outside the active
    pair because they were not recomputed by the quality contract.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    columns = (
        "git_hash", "physics_data_revision", "saved_at",
        "result_valid_em", "result_valid_thermal",
        "_strict_valid_em", "_strict_valid_full",
        "_strict_invalid_reasons", "em_validity_reason",
        "cap_on", *CAPACITANCE_FIELDS.values(), *RESONANCE_FIELDS.values(),
        "thermal_core_conductivity_model", "thermal_core_k_inplane",
        "thermal_core_k_throughstack", "core_lamination_factor",
        "winding_flux_linkage_readback_status",
        "winding_flux_linkage_readback_applicable",
        "winding_flux_linkage_readback_available",
        "winding_flux_linkage_readback_passed",
        "winding_flux_linkage_readback_reason",
    )
    records = _frame_records(frame, columns)
    if active_cohort is None:
        active_cohort = _active_cohort_identity(
            records, now.tzinfo, current_physics_revision
        )
    active = _active_cohort_view(
        active_cohort, current_physics_revision
    )
    prepared: list[dict[str, Any]] = []
    for row in records:
        revision, physics_revision = _cohort_identity(row)
        current = _is_active_cohort(
            (revision, physics_revision), active_cohort
        )
        strict_em_flag = _optional_flag(row.get("_strict_valid_em"))
        strict_full_flag = _optional_flag(row.get("_strict_valid_full"))
        strict_flags_recomputed = (
            strict_em_flag is not None and strict_full_flag is not None
        )
        if not strict_flags_recomputed:
            if strict_em_flag is None:
                strict_em_flag = (
                    _optional_flag(row.get("result_valid_em")) is True
                )
        if strict_full_flag is None:
            strict_full_flag = (
                bool(strict_em_flag)
                and _optional_flag(row.get("result_valid_thermal")) is True
            )
        prepared.append({
            **row,
            "_monitor_git_hash": revision,
            "_monitor_physics_revision": physics_revision,
            "_monitor_current": current,
            "_monitor_strict_recomputed": strict_flags_recomputed,
            "_monitor_strict_em": bool(
                strict_em_flag and (strict_flags_recomputed or current)
            ),
            "_monitor_strict_full": bool(
                strict_em_flag and strict_full_flag
                and (strict_flags_recomputed or current)
            ),
        })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        grouped[(
            row["_monitor_git_hash"], row["_monitor_physics_revision"]
        )].append(row)

    cutoff = now - timedelta(hours=1)
    cohorts: list[dict[str, Any]] = []
    for (revision, physics_revision), cohort_rows in grouped.items():
        saved_stamps = [
            stamp
            for row in cohort_rows
            if (
                stamp := _parse_time(row.get("saved_at"), now.tzinfo)
            ) is not None
        ]
        latest_saved_at = max(
            saved_stamps, key=lambda stamp: stamp.timestamp(), default=None
        )
        strict_em_rows = sum(row["_monitor_strict_em"] for row in cohort_rows)
        strict_full_rows = sum(row["_monitor_strict_full"] for row in cohort_rows)
        recent_growth = sum(
            1 for row in cohort_rows
            if row["_monitor_strict_full"]
            and (
                stamp := _parse_time(row.get("saved_at"), now.tzinfo)
            ) is not None
            and cutoff <= stamp <= now
        )
        current = _is_active_cohort(
            (revision, physics_revision), active_cohort
        )
        cohorts.append({
            "git_hash": revision or None,
            "git_hash_short": revision[:10] if revision else "unknown",
            "physics_data_revision": physics_revision,
            "latest_saved_at": (
                latest_saved_at.isoformat() if latest_saved_at else None
            ),
            "active": current,
            "current": current,
            "raw_rows": len(cohort_rows),
            "strict_em_rows": strict_em_rows,
            "strict_full_rows": strict_full_rows,
            "growth_rate_per_hour": float(recent_growth),
        })

    def _cohort_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        latest = _parse_time(item.get("latest_saved_at"), now.tzinfo)
        latest_rank = latest.timestamp() if latest is not None else float("-inf")
        return (
            not item["active"], -latest_rank, -item["raw_rows"],
            item["git_hash_short"], item["physics_data_revision"],
        )

    cohorts.sort(key=_cohort_sort_key)

    expected_physics_revision = _optional_text(
        current_physics_revision, 160
    )
    revision_rows = [
        row for row in prepared
        if expected_physics_revision is not None
        and row["_monitor_physics_revision"] == expected_physics_revision
    ]
    revision_strict_em = [
        row for row in revision_rows if row["_monitor_strict_em"]
    ]
    revision_strict_full = [
        row for row in revision_rows if row["_monitor_strict_full"]
    ]
    revision_strict_stamps = sorted(
        stamp
        for row in revision_strict_full
        if (
            stamp := _parse_time(row.get("saved_at"), now.tzinfo)
        ) is not None
    )
    revision_history: list[dict[str, Any]] = []
    cumulative = 0
    for stamp, count in Counter(revision_strict_stamps).items():
        cumulative += count
        revision_history.append({
            "time": stamp.isoformat(),
            "added": count,
            "total": cumulative,
        })
    if len(revision_history) > 240:
        stride = math.ceil(len(revision_history) / 240)
        sampled_history = revision_history[::stride]
        if sampled_history[-1] != revision_history[-1]:
            sampled_history.append(revision_history[-1])
        revision_history = sampled_history
    member_cohorts = [
        cohort for cohort in cohorts
        if cohort["physics_data_revision"] == expected_physics_revision
        and cohort.get("git_hash")
    ]
    physics_revision_aggregate = {
        "available": expected_physics_revision is not None,
        "physics_data_revision": expected_physics_revision,
        "has_rows": bool(revision_rows),
        "raw_rows": len(revision_rows),
        "strict_em_rows": len(revision_strict_em),
        "strict_full_rows": len(revision_strict_full),
        "growth_rate_per_hour": float(sum(
            cutoff <= stamp <= now for stamp in revision_strict_stamps
        )),
        "added_24h": sum(
            now - timedelta(hours=24) <= stamp <= now
            for stamp in revision_strict_stamps
        ),
        "member_git_hashes": [
            cohort["git_hash"] for cohort in member_cohorts
        ],
        "member_git_hash_shorts": [
            str(cohort["git_hash"])[:7] for cohort in member_cohorts
        ],
        "first_strict_saved_at": (
            revision_strict_stamps[0].isoformat()
            if revision_strict_stamps else None
        ),
        "latest_strict_saved_at": (
            revision_strict_stamps[-1].isoformat()
            if revision_strict_stamps else None
        ),
        "history": revision_history,
    }

    current_rows = [row for row in prepared if row["_monitor_current"]]
    current_strict = [
        row for row in current_rows if row["_monitor_strict_full"]
    ]
    present_rows: list[dict[str, Any]] = []
    absent_rows = 0
    unknown_cap_rows = 0
    for row in current_strict:
        cap_flag = _optional_flag(row.get("cap_on"))
        if cap_flag is True:
            present_rows.append(row)
        elif cap_flag is False:
            absent_rows += 1
        else:
            unknown_cap_rows += 1
    electrostatic = {
        "available": bool(current_strict and present_rows),
        "cohort_basis": "active_strict_full",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "cohort_rows": len(current_strict),
        "cap_stage_present_rows": len(present_rows),
        "cap_stage_absent_rows": absent_rows,
        "cap_stage_unknown_rows": unknown_cap_rows,
        "capacitance": {
            key: _scaled_stats(present_rows, column, 1e9, "nF")
            for key, column in CAPACITANCE_FIELDS.items()
        },
        "resonance": {
            key: _scaled_stats(present_rows, column, 1e-3, "kHz")
            for key, column in RESONANCE_FIELDS.items()
        },
    }

    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current_rows:
        model = _optional_text(row.get("thermal_core_conductivity_model"), 160)
        if model:
            model_rows[model].append(row)
    ordered_models = [
        model for model in THERMAL_MODEL_TAGS if model in model_rows
    ] + sorted(model for model in model_rows if model not in THERMAL_MODEL_TAGS)
    thermal_models = {
        "available": bool(model_rows),
        "cohort_basis": "active_identity",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "total_rows": len(current_rows),
        "tagged_rows": sum(len(rows) for rows in model_rows.values()),
        "missing_rows": len(current_rows) - sum(
            len(rows) for rows in model_rows.values()
        ),
        "models": [
            {
                "model": model,
                "count": len(model_rows[model]),
                "percent": (
                    len(model_rows[model]) / len(current_rows) * 100.0
                    if current_rows else 0.0
                ),
                "thermal_core_k_inplane": _plain_stats(
                    model_rows[model], "thermal_core_k_inplane"
                ),
                "thermal_core_k_throughstack": _plain_stats(
                    model_rows[model], "thermal_core_k_throughstack"
                ),
            }
            for model in ordered_models
        ],
    }

    current_reason_counts: Counter[str] = Counter()
    legacy_reason_counts: Counter[str] = Counter()
    current_quarantined = 0
    legacy_quarantined = 0
    for row in prepared:
        reasons = _invalid_reasons(row)
        if row["_monitor_current"]:
            if row["_monitor_strict_full"]:
                continue
            current_quarantined += 1
            if not reasons:
                if not row["_monitor_strict_em"]:
                    reasons.append("stored_flag:result_valid_em")
                else:
                    reasons.append("stored_flag:result_valid_thermal")
            current_reason_counts.update(reasons)
        else:
            if row["_monitor_strict_full"]:
                continue
            legacy_quarantined += 1
            if (
                active_cohort is None
                or row["_monitor_git_hash"] != active_cohort[0]
            ):
                reasons.append(
                    "untrusted_provenance:solver_revision_mismatch"
                )
            if (
                current_physics_revision is not None
                and
                row["_monitor_physics_revision"]
                != current_physics_revision
            ):
                reasons.append("cohort:physics_data_revision_mismatch")
            legacy_reason_counts.update(dict.fromkeys(reasons, 1))

    def _reason_items(counts: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    statuses = Counter(
        _optional_text(
            row.get("winding_flux_linkage_readback_status"), 80
        ) or "missing"
        for row in current_rows
    )
    readback_available = 0
    readback_unavailable = 0
    readback_missing = 0
    for row in current_rows:
        available = _optional_flag(
            row.get("winding_flux_linkage_readback_available")
        )
        status = _optional_text(
            row.get("winding_flux_linkage_readback_status"), 80
        )
        if available is True or status == "available":
            readback_available += 1
        elif available is False or status == "unavailable":
            readback_unavailable += 1
        else:
            readback_missing += 1

    return {
        "active_cohort": active,
        "cohorts": cohorts,
        "physics_revision_aggregate": physics_revision_aggregate,
        "electrostatic": electrostatic,
        "thermal_models": thermal_models,
        "quarantine": {
            "current": {
                "label": (
                    f"활성 코호트 {active['git_hash_short']}"
                    if active["available"] else active["label"]
                ),
                "rows": current_quarantined,
                "reasons": _reason_items(current_reason_counts),
            },
            "legacy": {
                "label": "레거시 cohort 잡음",
                "rows": legacy_quarantined,
                "reasons": _reason_items(legacy_reason_counts),
            },
        },
        "current_cohort_metadata": {
            "core_lamination_factor": _plain_stats(
                current_rows, "core_lamination_factor"
            ),
            "winding_flux_linkage_readback": {
                "cohort_rows": len(current_rows),
                "available_rows": readback_available,
                "unavailable_rows": readback_unavailable,
                "missing_rows": readback_missing,
                "statuses": [
                    {"status": status, "count": count}
                    for status, count in sorted(statuses.items())
                ],
            },
        },
    }


@dataclass(frozen=True)
class ReadResult:
    value: Any
    path: str
    exists: bool
    mtime: datetime | None = None
    warning: str | None = None


@dataclass
class _CacheEntry:
    signature: tuple[int, int]
    value: Any
    mtime: datetime


class SafeArtifactCache:
    """Caches the last good parse of each artifact by size and mtime."""

    def __init__(self) -> None:
        self._good: dict[tuple[str, str], _CacheEntry] = {}
        self._failed: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _read(
        self,
        path: Path,
        kind: str,
        parser: Callable[[Path], Any],
        default: Any,
    ) -> ReadResult:
        key = (str(path.resolve()), kind)
        try:
            signature = self._signature(path)
        except FileNotFoundError:
            return ReadResult(default, str(path), False)
        except OSError as exc:
            return ReadResult(default, str(path), False, warning=f"{path.name} 상태 확인 실패: {exc}")

        with self._lock:
            cached = self._good.get(key)
            if cached and cached.signature == signature:
                return ReadResult(cached.value, str(path), True, cached.mtime)
            if self._failed.get(key) == signature:
                previous = cached.value if cached else default
                previous_time = cached.mtime if cached else None
                return ReadResult(
                    previous,
                    str(path),
                    True,
                    previous_time,
                    f"{path.name} 손상/작성 중: 마지막 정상 데이터를 표시합니다.",
                )

        try:
            value = parser(path)
            mtime = datetime.fromtimestamp(signature[0] / 1_000_000_000, tz=_now().tzinfo)
        except (
            OSError, UnicodeError, ValueError, TypeError, ImportError,
            csv.Error, json.JSONDecodeError,
        ) as exc:
            with self._lock:
                self._failed[key] = signature
                cached = self._good.get(key)
            previous = cached.value if cached else default
            previous_time = cached.mtime if cached else None
            return ReadResult(
                previous,
                str(path),
                True,
                previous_time,
                f"{path.name} 읽기 실패: {type(exc).__name__}: {exc}",
            )

        with self._lock:
            self._good[key] = _CacheEntry(signature, value, mtime)
            self._failed.pop(key, None)
        return ReadResult(value, str(path), True, mtime)

    def json(self, path: Path, default: Any = None, max_bytes: int = 16 * 1024 * 1024) -> ReadResult:
        def parser(source: Path) -> Any:
            if source.stat().st_size > max_bytes:
                raise ValueError(f"file exceeds {max_bytes} byte safety limit")
            with source.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)

        return self._read(path, "json", parser, default)

    def csv(self, path: Path, max_rows: int = 100_000) -> ReadResult:
        def parser(source: Path) -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("CSV header is missing")
                for index, row in enumerate(reader):
                    if index >= max_rows:
                        raise ValueError(f"CSV exceeds {max_rows} row safety limit")
                    rows.append(dict(row))
            return rows

        return self._read(path, f"csv:{max_rows}", parser, [],)

    def parquet(
        self,
        path: Path,
        max_rows: int = 100_000,
        max_columns: int = 2_000,
    ) -> ReadResult:
        """Read a bounded Parquet frame lazily and retain the last good frame."""
        def parser(source: Path) -> Any:
            try:
                import pandas as pd
                import pyarrow.parquet as parquet

                metadata = parquet.ParquetFile(source).metadata
                if metadata.num_rows > max_rows:
                    raise ValueError(
                        f"Parquet exceeds {max_rows} row safety limit"
                    )
                if metadata.num_columns > max_columns:
                    raise ValueError(
                        f"Parquet exceeds {max_columns} column safety limit"
                    )
                return pd.read_parquet(source)
            except (ImportError, OSError, TypeError, ValueError):
                raise
            except Exception as exc:
                # Arrow exception classes are intentionally not imported at
                # module load time; normalize parser failures for _read().
                raise ValueError(
                    f"Parquet parse failed: {type(exc).__name__}: {exc}"
                ) from exc

        return self._read(
            path,
            f"parquet:{max_rows}:{max_columns}",
            parser,
            None,
        )


class RefillControllerReader:
    """Read the external refill controller's last durable status tick."""

    def __init__(
        self,
        state_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        configured_state_path = state_path or os.environ.get(
            "MFT_CONTROLLER_STATE_PATH", DEFAULT_CONTROLLER_STATE_PATH
        )
        configured_log_path = log_path or os.environ.get(
            "MFT_CONTROLLER_LOG_PATH", DEFAULT_CONTROLLER_LOG_PATH
        )
        self.state_path = Path(configured_state_path)
        self.log_path = Path(configured_log_path)

    @staticmethod
    def _count(value: Any) -> int | None:
        if type(value) is int and value >= 0:
            return value
        if isinstance(value, float) and math.isfinite(value) and value >= 0:
            integer = int(value)
            return integer if integer == value else None
        return None

    @classmethod
    def _concurrency_target(cls, state: dict[str, Any]) -> int | None:
        paths = (
            ("policy", "target"),
            ("policy", "concurrency_target"),
            ("policy", "project_concurrency_target"),
            ("target",),
            ("concurrency_target",),
            ("project_concurrency_target",),
            ("generation", "policy", "target"),
            ("generation", "identity", "project_concurrency_target"),
            ("generation", "identity", "concurrency_target"),
        )
        for path in paths:
            value: Any = state
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            target = cls._count(value)
            if target is not None:
                return target
        return cls._count(state.get("policy/target"))

    def _read_state(self) -> dict[str, Any] | None:
        stat = self.state_path.stat()
        if stat.st_size <= 0 or stat.st_size > CONTROLLER_STATE_MAX_BYTES:
            return None
        with self.state_path.open("rb") as handle:
            raw = handle.read(CONTROLLER_STATE_MAX_BYTES + 1)
        if len(raw) > CONTROLLER_STATE_MAX_BYTES:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    def _read_last_tick(self) -> tuple[dict[str, Any] | None, datetime | None]:
        stat = self.log_path.stat()
        if stat.st_size <= 0:
            return None, None
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - CONTROLLER_LOG_TAIL_BYTES))
            tail = handle.read(CONTROLLER_LOG_TAIL_BYTES)
        last_line = next((line for line in reversed(tail.splitlines()) if line.strip()), None)
        if last_line is None:
            return None, None
        value = json.loads(last_line)
        if not isinstance(value, dict):
            return None, None
        tick_at = datetime.fromtimestamp(stat.st_mtime, tz=_now().tzinfo)
        return value, tick_at

    def snapshot(self) -> dict[str, Any]:
        try:
            state = self._read_state()
            tick, tick_at = self._read_last_tick()
            if state is None or tick is None or tick_at is None:
                return {"available": False}

            action = tick.get("action")
            if not isinstance(action, str) or not action.strip():
                return {"available": False}

            result: dict[str, Any] = {
                "available": True,
                "last_tick_at": _iso(tick_at),
                "action": action.strip(),
            }
            for key in (
                "active_project_tasks_before",
                "accepted_or_reconciled_count",
            ):
                count = self._count(tick.get(key))
                if count is not None:
                    result[key] = count
            generation = tick.get("generation")
            generation_id = generation.get("id") if isinstance(generation, dict) else None
            if isinstance(generation_id, str) and generation_id.strip():
                result["generation_id"] = generation_id.strip()
            concurrency_target = self._concurrency_target(state)
            if concurrency_target is not None:
                result["concurrency_target"] = concurrency_target
            return result
        except Exception:
            return {"available": False}


class SimulationPolicyConflict(RuntimeError):
    """The scheduler rejected a stale simulation-policy revision."""


class SchedulerReader:
    """Adapter for MFT scheduler status and durable simulation policy."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        task_prefix: str = "mft",
        project_name: str = "MFT_1MW_2026v1",
        timeout: float = 2.0,
        optional_timeout: float | None = None,
        ttl: float = 10.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_prefix = task_prefix
        self.project_name = project_name.strip()
        self.timeout = timeout
        self.optional_timeout = (
            timeout if optional_timeout is None else max(0.1, optional_timeout)
        )
        self.ttl = ttl
        self._opener = opener
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        body = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "mft-monitor/1",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        with self._opener(
            request,
            timeout=self.timeout if timeout is None else timeout,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _nonnegative_integer(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        number = _finite_number(value)
        if number is None or number < 0 or not number.is_integer():
            return None
        return int(number)

    @staticmethod
    def _boolean(value: Any) -> bool | None:
        return value if type(value) is bool else None

    @classmethod
    def _pool_snapshot(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("AEDT pool response is invalid")
        config = payload.get("config")
        plan = payload.get("plan")
        sessions = payload.get("sessions")
        leases = payload.get("leases")
        if (
            not isinstance(config, dict)
            or not isinstance(plan, dict)
            or not isinstance(sessions, list)
            or not isinstance(leases, list)
        ):
            raise ValueError("AEDT pool response is missing config/plan/sessions/leases")
        session_states = {
            str(state): max(0, _integer(count, 0))
            for state, count in (
                plan.get("state_counts")
                if isinstance(plan.get("state_counts"), dict) else {}
            ).items()
        }
        observed_session_states = Counter(
            state
            for item in sessions if isinstance(item, dict)
            if (state := _optional_text(item.get("state"), 80))
        )
        if not session_states:
            session_states = dict(observed_session_states)
        lease_states = {
            str(state): max(0, _integer(count, 0))
            for state, count in (
                plan.get("lease_counts")
                if isinstance(plan.get("lease_counts"), dict) else {}
            ).items()
        }
        observed_lease_states = Counter(
            state
            for item in leases if isinstance(item, dict)
            if (state := _optional_text(item.get("state"), 80))
        )
        if not lease_states:
            lease_states = dict(observed_lease_states)
        live_leases = cls._nonnegative_integer(plan.get("live_projects"))
        if live_leases is None:
            # Match the scheduler's LEASE_LIVE_STATES contract.  Queued is
            # also reported separately below so the UI can expose pressure.
            live_leases = sum(
                lease_states.get(state, 0)
                for state in ("queued", "leased", "active", "releasing")
            )
        return {
            "available": True,
            "enabled": cls._boolean(config.get("enabled")),
            "adapter_ready": cls._boolean(config.get("adapter_ready")),
            "validation_passed": cls._boolean(
                config.get("validation_passed")
            ),
            "operational": cls._boolean(config.get("operational")),
            "max_sessions": cls._nonnegative_integer(
                config.get("max_aedt_sessions")
            ),
            "min_idle_sessions": cls._nonnegative_integer(
                config.get("min_idle_aedt_sessions")
            ),
            "idle_sessions": cls._nonnegative_integer(
                plan.get("idle_session_count")
            ),
            "hard_sessions": cls._nonnegative_integer(
                plan.get("hard_session_count")
            ),
            "warm_spare_deficit": cls._nonnegative_integer(
                plan.get("warm_spare_deficit")
            ),
            "warm_spare_start_needed": cls._nonnegative_integer(
                plan.get("warm_spare_start_needed")
            ),
            # These arrays are capped history/diagnostic records, not live
            # capacity.  Keep their sizes explicitly named and never use
            # them as the denominator for pool utilization.
            "session_record_count": len(sessions),
            "lease_record_count": len(leases),
            "live_leases": live_leases,
            "queued_leases": lease_states.get("queued", 0),
            "ready_sessions": session_states.get("ready", 0),
            "busy_sessions": session_states.get("busy", 0),
            "session_states": dict(sorted(session_states.items())),
            "lease_states": dict(sorted(lease_states.items())),
            "warm_spare_reason": _safe_text(
                plan.get("warm_spare_status_reason"), 500
            ),
            "error": None,
        }

    @classmethod
    def _license_snapshot(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("license response is invalid")
        candidates: list[dict[str, Any]] = []
        for key in ("display", "features", "in_use"):
            values = payload.get(key)
            if isinstance(values, list):
                candidates.extend(
                    item for item in values if isinstance(item, dict)
                )
        selected = next((
            item for item in candidates
            if str(item.get("feature") or "").strip().casefold()
            == "electronics_desktop"
        ), None)
        admission = payload.get("admission")
        admission = admission if isinstance(admission, dict) else {}
        admission_features = (
            admission.get("features")
            if isinstance(admission.get("features"), dict) else {}
        )
        if selected is None:
            fallback = admission_features.get("electronics_desktop")
            selected = fallback if isinstance(fallback, dict) else None
        if selected is None:
            raise ValueError("electronics_desktop license feature is unavailable")
        used = cls._nonnegative_integer(selected.get("used"))
        total = cls._nonnegative_integer(selected.get("total"))
        if used is None or total is None or used > total:
            raise ValueError("electronics_desktop license counts are invalid")
        error = _safe_text(payload.get("error"), 500)
        snapshot_valid = admission.get("snapshot_valid")
        if type(snapshot_valid) is not bool:
            snapshot_valid = payload.get("server_up") is True and not error
        return {
            "available": True,
            "feature": "electronics_desktop",
            "label": _safe_text(selected.get("label"), 100)
            or "AnsysElectronicsDesktop",
            "used": used,
            "total": total,
            "snapshot_valid": snapshot_valid,
            "checked_at": _safe_text(payload.get("checked_at"), 80),
            "error": error,
        }

    @classmethod
    def _node_local_snapshot(cls, payload: Any) -> dict[str, Any]:
        """Normalize active node-local AEDT host tasks and bundle identities."""
        if not isinstance(payload, list):
            raise ValueError("node-local AEDT host response is invalid")
        hosts: list[dict[str, Any]] = []
        seen_task_ids: set[int] = set()
        statuses: Counter[str] = Counter()
        expected_projects_by_bundle: dict[str, int] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            project = _optional_text(item.get("project"), 160)
            if project not in (None, NODE_LOCAL_AEDT_PROJECT):
                continue
            name = _optional_text(item.get("name"), 500) or ""
            entrypoint = _optional_text(item.get("entrypoint"), 500)
            if entrypoint is not None:
                entrypoint_leaf = entrypoint.replace("\\", "/").rsplit(
                    "/", 1
                )[-1]
                entrypoint_leaf = entrypoint_leaf.removesuffix(".py").rsplit(
                    ".", 1
                )[-1]
                if entrypoint_leaf != NODE_LOCAL_AEDT_ENTRYPOINT:
                    continue
            elif not name.endswith("-host"):
                # Compatibility for older task-list responses that omitted
                # entrypoint.  Canary host names are ``{bundle_id}-host``.
                continue
            status = (
                _optional_text(item.get("status"), 40) or ""
            ).lower()
            if status not in NODE_LOCAL_AEDT_ACTIVE_STATES:
                continue
            task_id = cls._nonnegative_integer(
                item.get("task_id", item.get("id"))
            )
            if task_id is None or task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            raw_payload = item.get("payload_json")
            if isinstance(raw_payload, str):
                try:
                    raw_payload = json.loads(raw_payload)
                except (TypeError, ValueError):
                    raw_payload = None
            task_payload = raw_payload if isinstance(raw_payload, dict) else {}
            bundle_id = _optional_text(
                task_payload.get("aedt_canary_bundle_id"), 500
            )
            if bundle_id is None and name.endswith("-host"):
                bundle_id = name[:-5] or None
            expected_projects = cls._nonnegative_integer(
                task_payload.get("aedt_canary_expected_projects")
            )
            if bundle_id and expected_projects is not None:
                expected_projects_by_bundle[bundle_id] = expected_projects
            statuses[status] += 1
            hosts.append({
                "task_id": task_id,
                "name": name or None,
                "status": status,
                "bundle_id": bundle_id,
            })
        bundle_ids = list(dict.fromkeys(
            host["bundle_id"] for host in hosts if host["bundle_id"]
        ))
        return {
            "available": True,
            "project": NODE_LOCAL_AEDT_PROJECT,
            "active_host_tasks": len(hosts),
            "statuses": dict(sorted(statuses.items())),
            "bundle_count": len(bundle_ids),
            "bundle_ids": bundle_ids,
            "expected_projects": (
                sum(expected_projects_by_bundle[bundle_id]
                    for bundle_id in bundle_ids)
                if bundle_ids and all(
                    bundle_id in expected_projects_by_bundle
                    for bundle_id in bundle_ids
                ) else None
            ),
            "hosts": hosts,
            "error": None,
        }

    @staticmethod
    def _unavailable_pool(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "enabled": None,
            "adapter_ready": None,
            "validation_passed": None,
            "operational": None,
            "max_sessions": None,
            "min_idle_sessions": None,
            "idle_sessions": None,
            "hard_sessions": None,
            "warm_spare_deficit": None,
            "warm_spare_start_needed": None,
            "session_record_count": None,
            "lease_record_count": None,
            "live_leases": None,
            "queued_leases": None,
            "ready_sessions": None,
            "busy_sessions": None,
            "session_states": {},
            "lease_states": {},
            "warm_spare_reason": None,
            "error": error,
        }

    @staticmethod
    def _unavailable_license(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "feature": "electronics_desktop",
            "label": "AnsysElectronicsDesktop",
            "used": None,
            "total": None,
            "snapshot_valid": None,
            "checked_at": None,
            "error": error,
        }

    @staticmethod
    def _unavailable_node_local(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "project": NODE_LOCAL_AEDT_PROJECT,
            "active_host_tasks": None,
            "statuses": {},
            "bundle_count": None,
            "bundle_ids": [],
            "expected_projects": None,
            "hosts": [],
            "error": error,
        }

    def _aedt_attach_snapshot(self) -> dict[str, Any]:
        """Fetch optional attach diagnostics concurrently within one deadline."""
        pool = self._unavailable_pool()
        license_status = self._unavailable_license()
        node_local = self._unavailable_node_local()
        node_local_query = urlencode({
            "project": NODE_LOCAL_AEDT_PROJECT,
            "status": ",".join(NODE_LOCAL_AEDT_ACTIVE_STATES),
            "limit": NODE_LOCAL_AEDT_TASK_LIMIT,
        })
        requests = {
            "pool": "/api/aedt-pool",
            "license": "/api/licenses",
            "node_local": f"/api/tasks?{node_local_query}",
        }
        executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="mft-monitor-aedt"
        )
        futures = {
            name: executor.submit(
                self._request_json,
                path,
                timeout=self.optional_timeout,
            )
            for name, path in requests.items()
        }
        deadline = time.monotonic() + self.optional_timeout + 0.25
        try:
            for name, future in futures.items():
                try:
                    remaining = max(0.01, deadline - time.monotonic())
                    payload = future.result(timeout=remaining)
                    if name == "pool":
                        pool = self._pool_snapshot(payload)
                    elif name == "license":
                        license_status = self._license_snapshot(payload)
                    else:
                        node_local = self._node_local_snapshot(payload)
                except Exception as exc:
                    message = (
                        f"{requests[name]} 조회 실패: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if name == "pool":
                        pool = self._unavailable_pool(message)
                    elif name == "license":
                        license_status = self._unavailable_license(message)
                    else:
                        # Older scheduler deployments do not expose the
                        # filtered host-task view.  Keep that optional absence
                        # isolated from central pool/license health.
                        node_local = self._unavailable_node_local(message)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        errors = [
            error for error in (pool.get("error"), license_status.get("error"))
            if error
        ]
        if pool["available"]:
            if pool["enabled"] is False:
                state = "disabled"
            elif pool["operational"] is not True:
                state = "gated"
            else:
                idle = pool.get("idle_sessions")
                minimum_idle = pool.get("min_idle_sessions")
                if idle is None or minimum_idle is None:
                    state = "partial"
                elif idle < minimum_idle:
                    state = (
                        "warming"
                        if (pool.get("warm_spare_start_needed") or 0) > 0
                        else "shortfall"
                    )
                elif not license_status["available"]:
                    state = "partial"
                elif (
                    license_status.get("snapshot_valid") is not True
                    or bool(license_status.get("error"))
                ):
                    state = "degraded"
                else:
                    state = "operational"
        elif license_status["available"]:
            state = "pool_unavailable"
        else:
            state = "unavailable"
        return {
            "available": bool(pool["available"] or license_status["available"]),
            "state": state,
            "license": license_status,
            "pool": pool,
            "node_local": node_local,
            "errors": errors,
        }

    @classmethod
    def _first_count(cls, *values: Any) -> int | None:
        for value in values:
            parsed = cls._nonnegative_integer(value)
            if parsed is not None:
                return parsed
        return None

    def _validate_project_identity(self, project: Any) -> dict[str, Any]:
        if not isinstance(project, dict):
            raise ValueError("scheduler project response is invalid")
        name = str(project.get("project") or project.get("name") or "").strip()
        if name != self.project_name:
            raise ValueError("scheduler returned a different project")
        return project

    def _project_status(self, project: Any) -> dict[str, Any]:
        """Normalize project counters without treating its safety cap as demand."""
        project = self._validate_project_identity(project)
        queued = self._first_count(project.get("queued_count"))
        attaching = self._first_count(project.get("attaching_count"))
        active = self._first_count(
            project.get("active_count"), project.get("executing_count")
        )
        solving = self._first_count(project.get("solving_count"), active)
        if any(value is None for value in (queued, attaching, active, solving)):
            raise ValueError("scheduler project live counts are unavailable or invalid")
        return {
            "project": self.project_name,
            "live_queued": queued,
            "live_attaching": attaching,
            "live_active": active,
            "live_solving": solving,
            # Compatibility for older dashboard consumers.
            "live_running": active,
            # Desired concurrency excludes work that is merely queued.
            "logical_active": attaching + active,
            "legacy_project_cap": self._first_count(project.get("max_active_tasks")),
            "project_updated_at": project.get("updated_at"),
        }

    def _simulation_policy_control(
        self,
        policy: Any,
        *,
        project: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize the scheduler's versioned, durable desired-concurrency policy."""
        if not isinstance(policy, dict):
            raise ValueError("scheduler simulation policy response is invalid")
        combined = {**(project or {}), **policy}
        self._validate_project_identity(combined)
        limits = combined.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        desired_min = self._first_count(
            combined.get("min_desired_simulations"),
            combined.get("desired_simulations_min"),
            limits.get("min_desired_simulations"),
            limits.get("minimum"),
            DEFAULT_PARALLEL_TARGET_MIN,
        )
        desired_max = self._first_count(
            combined.get("max_desired_simulations"),
            combined.get("desired_simulations_max"),
            limits.get("max_desired_simulations"),
            limits.get("maximum"),
            DEFAULT_PARALLEL_TARGET_MAX,
        )
        validated = self._first_count(
            combined.get("validated_concurrency_limit"),
            limits.get("validated_concurrency_limit"),
        )
        desired = self._first_count(combined.get("desired_simulations"))
        effective = self._first_count(combined.get("effective_simulations"))
        if desired_min is None or desired_max is None or desired_min > desired_max:
            raise ValueError("scheduler simulation policy bounds are invalid")
        if desired_max > PARALLEL_TARGET_SAFETY_CEILING:
            raise ValueError("scheduler simulation policy exceeds the MFT safety ceiling")
        if (
            validated is None
            or not desired_min <= validated <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError("scheduler validated concurrency limit is invalid")
        # Accept a legacy desired value above a newly lowered capability so an
        # operator can repair it through the bounded UI.  The next value is
        # still limited to parallel_target_max below.
        if (
            desired is None
            or not desired_min <= desired <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError("scheduler desired simulation count is invalid")
        if effective is not None and effective > desired_max:
            raise ValueError("scheduler effective simulation count is invalid")
        revision = combined.get("policy_revision")
        if isinstance(revision, bool) or not isinstance(revision, (int, str)):
            raise ValueError("scheduler simulation policy revision is invalid")
        if (isinstance(revision, int) and revision < 0) or not str(revision).strip():
            raise ValueError("scheduler simulation policy revision is invalid")

        counts = combined.get("counts")
        counts = counts if isinstance(counts, dict) else {}
        queued = self._first_count(
            combined.get("queued_simulations"), combined.get("queued_count"),
            counts.get("queued"),
        )
        attaching = self._first_count(
            combined.get("attaching_simulations"), combined.get("attaching_count"),
            counts.get("attaching"),
        )
        active = self._first_count(
            combined.get("active_simulations"), combined.get("active_count"),
            combined.get("executing_count"), counts.get("active"),
        )
        solving = self._first_count(
            combined.get("solving_simulations"), combined.get("solving_count"),
            counts.get("solving"), active,
        )
        reported_logical = self._first_count(
            combined.get("logical_active_simulations"),
            combined.get("logical_active_count"),
            counts.get("logical_active"),
        )
        if project is not None:
            status = self._project_status(project)
            queued = status["live_queued"] if queued is None else queued
            attaching = status["live_attaching"] if attaching is None else attaching
            active = status["live_active"] if active is None else active
            solving = status["live_solving"] if solving is None else solving
        if any(value is None for value in (queued, attaching, active, solving)):
            raise ValueError("scheduler simulation policy live counts are unavailable")
        # Compatibility with the first policy API revision, where active_count
        # was accidentally emitted as attaching+solving.  The explicit logical
        # total makes that wire shape unambiguous and keeps UI semantics stable.
        if reported_logical == active and attaching > 0 and solving <= active:
            active = max(solving, active - attaching)
        if solving > active:
            raise ValueError("scheduler solving count exceeds active count")

        raw_constraint = combined.get("resource_constraint")
        if isinstance(raw_constraint, dict):
            constraint = {
                str(key): value for key, value in raw_constraint.items()
                if isinstance(key, str) and value is not None
            }
        elif raw_constraint is None:
            constraint = None
        else:
            constraint = {"reason": _safe_text(raw_constraint, 500)}
        gate_reason = (
            _optional_text(combined.get("control_gate_reason"), 500)
            or _optional_text(combined.get("gate_reason"), 500)
            or (
                _optional_text(constraint.get("reason"), 500)
                if isinstance(constraint, dict) else None
            )
        )
        scheduler_control_enabled = combined.get("control_enabled")
        control_enabled = (
            scheduler_control_enabled
            if type(scheduler_control_enabled) is bool else True
        )
        if not control_enabled and not gate_reason:
            gate_reason = "scheduler가 simulation-policy 변경을 잠갔습니다"
        return {
            "project": self.project_name,
            "policy_supported": True,
            "control_enabled": control_enabled,
            "read_only": not control_enabled,
            "parallel_target": desired,
            "desired_simulations": desired,
            "effective_simulations": effective,
            "validated_concurrency_limit": validated,
            "parallel_target_min": desired_min,
            # An operator may only select a concurrency level that has passed
            # the scheduler rollout gate, even if the configured safety cap is higher.
            "parallel_target_max": min(desired_max, validated),
            "configured_target_max": desired_max,
            "policy_revision": revision,
            "scale_down_mode": str(
                combined.get("scale_down_mode") or "drain"
            ).strip().lower(),
            "live_queued": queued,
            "live_attaching": attaching,
            "live_active": active,
            "live_solving": solving,
            "live_running": active,
            "logical_active": attaching + active,
            "resource_constraint": constraint,
            "control_gate_reason": gate_reason,
            "project_updated_at": combined.get("updated_at"),
        }

    def set_simulation_policy(
        self,
        desired_simulations: int,
        *,
        expected_revision: int | str,
    ) -> dict[str, Any]:
        """CAS-update durable MFT demand; lowering always uses graceful drain."""
        if (
            type(desired_simulations) is not int
            or not DEFAULT_PARALLEL_TARGET_MIN
            <= desired_simulations
            <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError(
                "desired simulations must be an integer between "
                f"{DEFAULT_PARALLEL_TARGET_MIN} and "
                f"{PARALLEL_TARGET_SAFETY_CEILING}"
            )
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (int, str))
            or not str(expected_revision).strip()
        ):
            raise ValueError("expected policy revision is required")
        if not self.project_name:
            raise ValueError("scheduler project control is not configured")
        project_path = quote(self.project_name, safe="")
        try:
            policy = self._request_json(
                f"/api/projects/{project_path}/simulation-policy",
                method="PATCH",
                payload={
                    "desired_simulations": desired_simulations,
                    "expected_revision": expected_revision,
                    "scale_down_mode": "drain",
                },
            )
        except HTTPError as exc:
            if exc.code == 409:
                raise SimulationPolicyConflict(
                    "simulation policy changed; refresh and retry"
                ) from exc
            raise
        control = self._simulation_policy_control(policy)
        if control["desired_simulations"] != desired_simulations:
            raise ValueError("scheduler simulation policy readback mismatch")
        with self._lock:
            self._cached = None
            self._cached_at = 0.0
        return control

    def set_parallel_target(
        self, target: int, *, expected_revision: int | str | None = None
    ) -> dict[str, Any]:
        """Compatibility name for callers migrated to the versioned policy API."""
        if expected_revision is None:
            raise ValueError("expected policy revision is required")
        return self.set_simulation_policy(
            target, expected_revision=expected_revision
        )

    def snapshot(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with self._lock:
            if self._cached is not None and now_monotonic - self._cached_at < self.ttl:
                return self._cached
        # The campaign has more than ten thousand historical tasks.  Reading
        # /api/tasks would make a dashboard refresh take many seconds, so use
        # the aggregate summary plus the single MFT project record.
        query = urlencode({"name_prefix": self.task_prefix})
        try:
            payload = self._request_json(f"/api/tasks/summary?{query}")
            if not isinstance(payload, dict) or not isinstance(payload.get("statuses"), dict):
                raise ValueError("scheduler summary response is invalid")
            statuses = Counter({
                str(name).strip().lower(): _integer(count, 0)
                for name, count in payload["statuses"].items()
            })
            running = sum(statuses.get(name, 0) for name in ("running", "executing"))
            pending = sum(statuses.get(name, 0) for name in ("pending", "queued", "submitted"))
            completed = sum(statuses.get(name, 0) for name in ("completed", "complete", "succeeded", "success"))
            failed = sum(statuses.get(name, 0) for name in ("failed", "error", "timed_out", "timeout"))
            cancelled = sum(statuses.get(name, 0) for name in ("cancelled", "canceled"))
            result = {
                "connected": True,
                "url": self.base_url,
                "read_only": True,
                "control_enabled": False,
                "policy_supported": False,
                "task_prefix": self.task_prefix,
                "total": _integer(payload.get("total"), sum(statuses.values())),
                "running": running,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "other": max(0, _integer(payload.get("total"), sum(statuses.values())) - running - pending - completed - failed - cancelled),
                "statuses": dict(sorted(statuses.items())),
                "error": None,
                "updated_at": _iso(_now()),
            }
            if self.project_name:
                try:
                    project_path = quote(self.project_name, safe="")
                    project = self._request_json(f"/api/projects/{project_path}")
                    result.update(self._project_status(project))
                    embedded = project.get("simulation_policy")
                    if isinstance(embedded, dict):
                        policy = {**project, **embedded}
                    elif (
                        "desired_simulations" in project
                        or "policy_revision" in project
                        or "validated_concurrency_limit" in project
                    ):
                        # Transitional schedulers advertise the core policy on
                        # the project record and expose effective/gate/count
                        # fields at the dedicated GET endpoint.
                        policy = self._request_json(
                            f"/api/projects/{project_path}/simulation-policy"
                        )
                    else:
                        policy = None
                    if policy is not None:
                        result.update(
                            self._simulation_policy_control(policy, project=project)
                        )
                    else:
                        result.update({
                            "parallel_target": None,
                            "desired_simulations": None,
                            "effective_simulations": None,
                            "validated_concurrency_limit": None,
                            "parallel_target_min": DEFAULT_PARALLEL_TARGET_MIN,
                            "parallel_target_max": DEFAULT_PARALLEL_TARGET_MAX,
                            "policy_revision": None,
                            "resource_constraint": None,
                            "control_gate_reason": (
                                "scheduler가 durable simulation-policy "
                                "capability를 아직 제공하지 않습니다"
                            ),
                        })
                    result["project_error"] = None
                except (
                    HTTPError,
                    URLError,
                    OSError,
                    ValueError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as exc:
                    result["project"] = self.project_name
                    result["parallel_target"] = None
                    result["desired_simulations"] = None
                    result["effective_simulations"] = None
                    result["validated_concurrency_limit"] = None
                    result["parallel_target_min"] = DEFAULT_PARALLEL_TARGET_MIN
                    result["parallel_target_max"] = DEFAULT_PARALLEL_TARGET_MAX
                    result["policy_revision"] = None
                    result["live_queued"] = max(0, statuses.get("queued", 0))
                    result["live_attaching"] = max(0, statuses.get("attaching", 0))
                    result["live_active"] = max(0, statuses.get("running", 0))
                    result["live_solving"] = result["live_active"]
                    result["live_running"] = result["live_active"]
                    result["logical_active"] = (
                        result["live_attaching"]
                        + result["live_running"]
                    )
                    result["resource_constraint"] = None
                    result["control_gate_reason"] = (
                        "scheduler project 상태를 검증할 수 없습니다"
                    )
                    result["project_error"] = (
                        f"scheduler project 조회 실패: {type(exc).__name__}: {exc}"
                    )
            result["aedt_attach"] = self._aedt_attach_snapshot()
        except (HTTPError, URLError, OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            result = {
                "connected": False,
                "url": self.base_url,
                "read_only": True,
                "control_enabled": False,
                "policy_supported": False,
                "task_prefix": self.task_prefix,
                "project": self.project_name,
                "parallel_target": None,
                "desired_simulations": None,
                "effective_simulations": None,
                "validated_concurrency_limit": None,
                "parallel_target_min": DEFAULT_PARALLEL_TARGET_MIN,
                "parallel_target_max": DEFAULT_PARALLEL_TARGET_MAX,
                "policy_revision": None,
                "live_queued": 0,
                "live_attaching": 0,
                "live_active": 0,
                "live_solving": 0,
                "live_running": 0,
                "logical_active": 0,
                "project_error": None,
                "resource_constraint": None,
                "control_gate_reason": "scheduler에 연결할 수 없습니다",
                "total": 0,
                "running": 0,
                "pending": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "other": 0,
                "statuses": {},
                "error": f"scheduler 조회 실패: {type(exc).__name__}: {exc}",
                "updated_at": _iso(_now()),
                "aedt_attach": {
                    "available": False,
                    "state": "unavailable",
                    "license": self._unavailable_license(),
                    "pool": self._unavailable_pool(),
                    "node_local": self._unavailable_node_local(),
                    "errors": [],
                },
            }
        with self._lock:
            self._cached = result
            self._cached_at = now_monotonic
        return result


class RuntimeRecorder:
    """Persists a compact current snapshot and low-frequency history."""

    def __init__(self, directory: Path, min_interval_seconds: int = 60) -> None:
        self.directory = directory
        self.snapshot_path = directory / "monitor_snapshot.json"
        self.history_path = directory / "monitor_history.jsonl"
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._last_signature: tuple[Any, ...] | None = None
        self._last_write = 0.0
        self._snapshot_error: str | None = None

    def _write_snapshot(self, payload: dict[str, Any]) -> None:
        """Write one durable snapshot with a RaiDrive-safe bounded fallback."""
        serialized = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.snapshot_path.name}.",
            suffix=".tmp",
            dir=self.directory,
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

            replace_error: OSError | None = None
            for attempt in range(5):
                try:
                    os.replace(temp, self.snapshot_path)
                    return
                except OSError as exc:
                    replace_error = exc
                    if attempt < 4:
                        time.sleep(0.05 * (2 ** attempt))

            # Some Windows network filesystems deny rename/replace even when
            # creating or overwriting the target directly is allowed.  The
            # recorder lock serializes writers while this fallback writes,
            # fsyncs, and verifies the complete serialized payload.
            try:
                with self.snapshot_path.open("wb") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                if self.snapshot_path.read_bytes() != serialized:
                    raise OSError("snapshot direct-write readback mismatch")
                return
            except OSError as fallback_error:
                raise OSError(
                    f"snapshot replace failed ({replace_error}); "
                    f"direct-write fallback failed ({fallback_error})"
                ) from fallback_error
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def _append_history(self, summary: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _summary(dashboard: dict[str, Any]) -> dict[str, Any]:
        data = dashboard.get("data", {})
        models = dashboard.get("models", {})
        nsga = dashboard.get("nsga2", {})
        verification = dashboard.get("verification", {})
        scheduler = dashboard.get("scheduler", {})
        return {
            "schema_version": SCHEMA_VERSION,
            "time": dashboard.get("generated_at"),
            "overall": dashboard.get("status", {}).get("overall"),
            "data": {
                "total_rows": data.get("total_rows"),
                "complete_rows": data.get("complete_rows"),
                "throughput_1h": data.get("throughput_1h"),
                "count_basis": data.get("count_basis"),
                "physics_data_revision": data.get(
                    "current_physics_data_revision"
                ),
                "revision_raw_rows": data.get("revision_raw_rows"),
                "member_git_hashes": data.get("member_git_hashes"),
                "pinned_solver_revision": data.get("pinned_revision"),
                "pinned_library_revision": data.get("pinned_library_revision"),
                "cohorts": [
                    {
                        key: cohort.get(key)
                        for key in (
                            "git_hash", "physics_data_revision", "raw_rows",
                            "strict_em_rows", "strict_full_rows",
                        )
                    }
                    for cohort in data.get("cohorts", [])
                    if isinstance(cohort, dict)
                ],
            },
            "models": {
                "trained": models.get("trained_count"),
                "planned": models.get("target_count"),
            },
            "nsga2": {
                "round": nsga.get("round"),
                "candidate_count": nsga.get("candidate_count"),
                "min_volume_L": (nsga.get("summary") or {}).get("min_volume_L"),
            },
            "verification": {
                "stage": verification.get("stage"),
                "valid": (verification.get("counts") or {}).get("valid"),
                "total": (verification.get("counts") or {}).get("total"),
                "final_status": (verification.get("final") or {}).get("status"),
            },
            "scheduler": {
                "connected": scheduler.get("connected"),
                "running": scheduler.get("running"),
                "pending": scheduler.get("pending"),
                "failed": scheduler.get("failed"),
            },
        }

    @staticmethod
    def _snapshot(dashboard: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Keep the durable snapshot small; detailed points remain in source artifacts."""
        return {
            **summary,
            "project": dashboard.get("project"),
            "status": dashboard.get("status"),
            "data": {
                **summary["data"],
                "em_valid_rows": dashboard.get("data", {}).get("em_valid_rows"),
                "thermal_valid_rows": dashboard.get("data", {}).get("thermal_valid_rows"),
                "eta_3000": dashboard.get("data", {}).get("eta_3000"),
                "latest_data_at": dashboard.get("data", {}).get("latest_data_at"),
            },
            "models": {
                **summary["models"],
                "items": [
                    {
                        key: model.get(key)
                        for key in ("target", "status", "n_used", "r2", "rmse", "mape_pct", "p90_ape_pct")
                    }
                    for model in dashboard.get("models", {}).get("models", [])
                ],
            },
            "nsga2": {**summary["nsga2"], "summary": dashboard.get("nsga2", {}).get("summary")},
            "verification": {
                **summary["verification"],
                "counts": dashboard.get("verification", {}).get("counts"),
                "agreement": dashboard.get("verification", {}).get("agreement"),
            },
            "scheduler": {**summary["scheduler"], "total": dashboard.get("scheduler", {}).get("total")},
        }

    def record(self, dashboard: dict[str, Any]) -> None:
        summary = self._summary(dashboard)
        signature = (
            summary["overall"],
            summary["data"]["total_rows"],
            summary["data"]["complete_rows"],
            summary["data"]["physics_data_revision"],
            tuple(summary["data"].get("member_git_hashes") or []),
            tuple(
                (
                    cohort.get("git_hash"),
                    cohort.get("physics_data_revision"),
                    cohort.get("strict_full_rows"),
                )
                for cohort in summary["data"]["cohorts"]
            ),
            summary["models"]["trained"],
            summary["nsga2"]["round"],
            summary["nsga2"]["candidate_count"],
            summary["verification"]["stage"],
            summary["verification"]["final_status"],
            summary["scheduler"]["running"],
            summary["scheduler"]["pending"],
        )
        now_monotonic = time.monotonic()
        with self._lock:
            if signature == self._last_signature and now_monotonic - self._last_write < self.min_interval_seconds:
                if self._snapshot_error:
                    raise OSError(self._snapshot_error)
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            errors = []
            try:
                self._write_snapshot(self._snapshot(dashboard, summary))
                self._snapshot_error = None
            except (OSError, TypeError, ValueError) as exc:
                self._snapshot_error = f"snapshot write failed: {type(exc).__name__}: {exc}"
                errors.append(self._snapshot_error)

            history_written = False
            try:
                self._append_history(summary)
                history_written = True
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"history append failed: {type(exc).__name__}: {exc}")

            if history_written:
                self._last_signature = signature
                self._last_write = now_monotonic
            if errors:
                raise OSError("; ".join(errors))

    def history(self, limit: int = 2_000) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"entries": [], "warning": None}
        entries: list[dict[str, Any]] = []
        bad_lines = 0
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            entries.append(value)
                    except json.JSONDecodeError:
                        bad_lines += 1
        except OSError as exc:
            return {"entries": [], "warning": f"모니터링 이력 읽기 실패: {exc}"}
        entries = entries[-limit:]
        warning = f"손상된 이력 {bad_lines}줄을 건너뛰었습니다." if bad_lines else None
        return {"entries": entries, "warning": warning}


class ArtifactService:
    """Builds stable JSON view models from live campaign files."""

    def __init__(
        self,
        regression_root: Path,
        scheduler: SchedulerReader | None = None,
        clock: Callable[[], datetime] = _now,
        record_runtime: bool = True,
        refill_controller: RefillControllerReader | None = None,
    ) -> None:
        self.root = Path(regression_root).resolve()
        self.cache = SafeArtifactCache()
        self.scheduler = scheduler or SchedulerReader()
        self.refill_controller = refill_controller or RefillControllerReader()
        self.clock = clock
        self.recorder = RuntimeRecorder(self.root / "monitoring" / "runtime") if record_runtime else None
        self._campaign_audit_lock = threading.RLock()
        self._campaign_audit_key: tuple[Any, ...] | None = None
        self._campaign_audit_frame: Any = None
        self._campaign_audit_warning: str | None = None

    @staticmethod
    def _warnings(*results: ReadResult) -> list[str]:
        return [result.warning for result in results if result.warning]

    def _audited_campaign_frame(
        self,
        result: ReadResult,
        expected_solver_revision: str | None,
        expected_library_revision: str | None,
    ) -> tuple[Any, str | None]:
        """Cache canonical per-SHA strict annotations for one Parquet read.

        Every well-formed solver SHA is passed back into the quality contract
        as the exact expected revision for that subgroup.  This keeps strict
        evidence recomputed per row while allowing one physics revision to
        span multiple clean solver rolls.  Missing or malformed identities
        are still audited, without an expected SHA, and fail provenance.
        """
        if not result.exists or result.value is None:
            return None, result.warning
        key = (
            result.path,
            result.mtime,
            id(result.value),
            expected_solver_revision,
            expected_library_revision,
        )
        with self._campaign_audit_lock:
            if key == self._campaign_audit_key:
                return self._campaign_audit_frame, self._campaign_audit_warning
            warning = None
            audit_fields = (
                "_strict_valid_em",
                "_strict_valid_thermal",
                "_strict_valid_full",
                "_strict_invalid_reasons",
            )
            try:
                from ..quality_contract import annotate_validity

                source = result.value
                if "git_hash" not in source.columns:
                    audited = annotate_validity(
                        source,
                        expected_solver_revision=None,
                        expected_library_revision=None,
                    )
                else:
                    audited = source.copy()
                    audited["_strict_valid_em"] = False
                    audited["_strict_valid_thermal"] = False
                    audited["_strict_valid_full"] = False
                    audited["_strict_invalid_reasons"] = ""
                    positions_by_revision: dict[str, list[int]] = defaultdict(list)
                    for position, value in enumerate(source["git_hash"].tolist()):
                        revision = (_optional_text(value, 160) or "").lower()
                        positions_by_revision[revision].append(position)
                    active_revision = (
                        _optional_text(expected_solver_revision, 160) or ""
                    ).lower()
                    for revision, positions in positions_by_revision.items():
                        exact_revision = (
                            revision
                            if re.fullmatch(r"[0-9a-f]{40}", revision)
                            else None
                        )
                        cohort = source.iloc[positions].copy()
                        cohort_audited = annotate_validity(
                            cohort,
                            expected_solver_revision=exact_revision,
                            expected_library_revision=(
                                expected_library_revision
                                if exact_revision == active_revision
                                else None
                            ),
                        )
                        for field in audit_fields:
                            column_position = audited.columns.get_loc(field)
                            audited.iloc[positions, column_position] = (
                                cohort_audited[field].to_numpy()
                            )
            except Exception as exc:
                # Stored validity flags remain a fail-soft operational view;
                # _campaign_frame_summary still pins them to the dynamically
                # selected active identity instead of trusting other cohorts.
                audited = result.value
                if hasattr(audited, "drop"):
                    # Never mistake persisted/stale underscore columns for a
                    # successful recomputation after the canonical audit
                    # itself failed.
                    audited = audited.drop(
                        columns=list(audit_fields), errors="ignore"
                    )
                warning = (
                    "train.parquet strict audit failed; stored validity flags "
                    "are shown for the active cohort: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._campaign_audit_key = key
            self._campaign_audit_frame = audited
            self._campaign_audit_warning = warning
            return audited, warning

    def data(self) -> dict[str, Any]:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        dataset_dir = self.root / "data" / "dataset"
        manifest_result = self.cache.json(dataset_dir / "manifest.json", {})
        rows_result = self.cache.csv(dataset_dir / "train_io.csv")
        parquet_result = self.cache.parquet(dataset_dir / "train.parquet")
        cache_result = self.cache.json(dataset_dir / "collect_cache.json", {})
        strict_result = self.cache.json(
            self.root / "training" / "strict_data_status.json", {}
        )
        manifest = manifest_result.value if isinstance(manifest_result.value, dict) else {}
        rows = rows_result.value if isinstance(rows_result.value, list) else []
        collector_cache = cache_result.value if isinstance(cache_result.value, dict) else {}
        strict_status = (
            strict_result.value if isinstance(strict_result.value, dict) else {}
        )
        warnings = self._warnings(
            manifest_result, rows_result, parquet_result, cache_result,
            strict_result,
        )
        if PHYSICS_DATA_REVISION_IMPORT_ERROR:
            warnings.append(
                "PHYSICS_DATA_REVISION import failed; active cohort panels "
                f"are unavailable: {PHYSICS_DATA_REVISION_IMPORT_ERROR}"
            )

        manifest_total = _integer(manifest.get("total_rows"), -1)
        parquet_rows = (
            len(parquet_result.value)
            if parquet_result.value is not None
            and hasattr(parquet_result.value, "__len__") else -1
        )
        observed_rows = parquet_rows if parquet_rows >= 0 else len(rows)
        raw_total = manifest_total if manifest_total >= 0 else observed_rows
        if rows and manifest_total >= 0 and len(rows) != manifest_total:
            warnings.append(f"manifest({manifest_total})와 train_io.csv({len(rows)}) 행 수가 다릅니다.")
        if parquet_rows >= 0 and manifest_total >= 0 and parquet_rows != manifest_total:
            warnings.append(
                f"manifest({manifest_total})와 train.parquet({parquet_rows}) 행 수가 다릅니다."
            )

        identity = strict_status.get("state_identity")
        identity = identity if isinstance(identity, dict) else {}
        status_solver_revision = (
            strict_status.get("expected_solver_revision")
            or identity.get("solver_revision")
        )
        status_library_revision = (
            strict_status.get("expected_library_revision")
            or identity.get("library_revision")
        )
        pinned_revision = str(status_solver_revision or "").strip().lower()
        pinned_library_revision = str(
            status_library_revision or ""
        ).strip().lower()
        full_pins = all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (
            pinned_revision, pinned_library_revision
        ))
        pins_valid = bool(
            full_pins
            and checkpoint_status_revision_identity_matches is not None
            and checkpoint_status_revision_identity_matches(
                strict_status, pinned_revision, pinned_library_revision
            )
        )
        strict_available = bool(strict_result.exists and pins_valid)
        if not strict_available:
            pinned_revision = None
            pinned_library_revision = None
        raw_campaign_frame = (
            parquet_result.value
            if parquet_result.value is not None else rows
        )
        active_cohort = _active_cohort_identity(
            raw_campaign_frame, now.tzinfo, CURRENT_PHYSICS_DATA_REVISION
        )
        active_solver_revision = (
            active_cohort[0] if active_cohort is not None else None
        )
        audit_library_revision = (
            pinned_library_revision
            if (
                pinned_revision is not None
                and active_cohort is not None
                and solver_revision_matches_physics_cohort is not None
                and solver_revision_matches_physics_cohort(
                    active_solver_revision,
                    pinned_revision,
                    active_cohort[1],
                )
            ) else None
        )
        campaign_frame, audit_warning = self._audited_campaign_frame(
            parquet_result,
            active_solver_revision,
            audit_library_revision,
        )
        if campaign_frame is None:
            campaign_frame = rows
        if audit_warning and audit_warning not in warnings:
            warnings.append(audit_warning)
        if not strict_available:
            warnings.append(
                "pinned strict-data status is unavailable; the physics-revision "
                "aggregate uses row-level validity evidence instead."
            )

        simulation_timing = _simulation_timing_summary(
            campaign_frame,
            now.tzinfo,
            active_cohort=active_cohort,
            current_physics_revision=CURRENT_PHYSICS_DATA_REVISION,
        )
        campaign_summary = _campaign_frame_summary(
            campaign_frame,
            now,
            active_cohort=active_cohort,
            current_physics_revision=CURRENT_PHYSICS_DATA_REVISION,
        )
        physics_aggregate = campaign_summary["physics_revision_aggregate"]
        revision_raw_rows = _integer(physics_aggregate.get("raw_rows"), 0)
        total = _integer(physics_aggregate.get("strict_full_rows"), 0)
        em_valid = _integer(physics_aggregate.get("strict_em_rows"), 0)
        thermal_valid = complete = total
        throughput_1h = _integer(
            physics_aggregate.get("growth_rate_per_hour"), 0
        )
        added_24h = _integer(physics_aggregate.get("added_24h"), 0)
        latest_data = _parse_time(
            physics_aggregate.get("latest_strict_saved_at"), now.tzinfo
        )
        first_data = _parse_time(
            physics_aggregate.get("first_strict_saved_at"), now.tzinfo
        )
        stalled_minutes = (
            max(0.0, (now - latest_data).total_seconds() / 60.0)
            if latest_data else None
        )
        hourly_rate = float(throughput_1h)
        if hourly_rate <= 0 and added_24h:
            hourly_rate = added_24h / 24.0
        remaining = max(0, DATA_GOAL - total)
        eta_hours = remaining / hourly_rate if hourly_rate > 0 else None
        eta = now + timedelta(hours=eta_hours) if eta_hours is not None else None
        history = list(physics_aggregate.get("history") or [])

        campaign_identity_rows = _frame_records(
            campaign_frame, ("git_hash", "saved_at")
        )
        revision_rows = campaign_identity_rows or rows
        revisions = Counter(
            revision.lower()
            for row in revision_rows
            if (revision := _optional_text(row.get("git_hash"), 40))
        )
        latest_revision = None
        timed_revisions = [
            (stamp, revision)
            for row in revision_rows
            if (stamp := _parse_time(row.get("saved_at"), now.tzinfo)) is not None
            if (revision := _optional_text(row.get("git_hash"), 40))
        ]
        if timed_revisions:
            latest_revision = max(timed_revisions, key=lambda item: item[0])[1].lower()
        if not latest_revision and revisions:
            latest_revision = revisions.most_common(1)[0][0]
        revision_mismatch = (
            sum(count for revision, count in revisions.items() if revision != latest_revision)
            if latest_revision else 0
        )
        hashes = manifest.get("git_hashes") if isinstance(manifest.get("git_hashes"), list) else []
        harvested = collector_cache.get("harvested")
        nodata = collector_cache.get("nodata")
        local_parts = collector_cache.get("local_parts")

        return {
            "schema_version": SCHEMA_VERSION,
            "available": (
                manifest_result.exists or rows_result.exists
                or parquet_result.exists
            ),
            "count_basis": "physics_revision_strict_full",
            "strict_status_available": strict_available,
            "raw_total_rows": raw_total,
            "revision_raw_rows": revision_raw_rows,
            "total_rows": total,
            "em_valid_rows": em_valid,
            "thermal_valid_rows": thermal_valid,
            "complete_rows": complete,
            "em_only_rows": max(0, em_valid - complete),
            "invalid_em_rows": max(0, revision_raw_rows - em_valid),
            "manifest_new_rows": _integer(manifest.get("new_rows"), 0),
            "manifest_new_unique_rows": _integer(manifest.get("new_unique_rows"), 0),
            "goal": DATA_GOAL,
            "stretch_goal": STRETCH_GOAL,
            "goal_progress_pct": min(100.0, total / DATA_GOAL * 100.0),
            "stretch_progress_pct": min(100.0, total / STRETCH_GOAL * 100.0),
            "remaining_to_goal": remaining,
            "throughput_1h": throughput_1h,
            "added_24h": added_24h,
            "effective_hourly_rate": hourly_rate,
            "eta_3000": _iso(eta),
            "eta_hours": eta_hours,
            "first_data_at": _iso(first_data),
            "latest_data_at": _iso(latest_data),
            "stalled_minutes": stalled_minutes,
            "stalled": bool(stalled_minutes is not None and stalled_minutes >= 90 and total < DATA_GOAL),
            "latest_revision": latest_revision,
            "pinned_revision": pinned_revision,
            "pinned_library_revision": pinned_library_revision,
            "current_solver_revision": active_solver_revision,
            "current_physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "member_git_hashes": physics_aggregate["member_git_hashes"],
            "member_git_hash_shorts": (
                physics_aggregate["member_git_hash_shorts"]
            ),
            "revision_count": len(revisions) or len(hashes),
            "rows_not_latest_revision": revision_mismatch,
            "rows_not_current_physics_revision": max(
                0, raw_total - revision_raw_rows
            ),
            **campaign_summary,
            "collector": {
                "harvested_tasks": len(harvested) if isinstance(harvested, list) else None,
                "no_data_tasks": len(nodata) if isinstance(nodata, list) else None,
                "local_parts": len(local_parts) if isinstance(local_parts, list) else None,
            },
            "simulation_timing": simulation_timing,
            "history": history,
            "source": {
                "manifest": str(manifest_result.path),
                "rows": str(rows_result.path),
                "parquet": str(parquet_result.path),
                "campaign_rows": (
                    str(parquet_result.path)
                    if parquet_result.exists
                    and parquet_result.value is not None
                    else str(rows_result.path)
                ),
                "strict_status": str(strict_result.path),
                "updated_at": _iso(
                    parquet_result.mtime or manifest_result.mtime
                    or rows_result.mtime or strict_result.mtime
                ),
            },
            "warnings": warnings,
        }

    @staticmethod
    def _exact_integer(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        number = _finite_number(value)
        if number is None or not number.is_integer():
            return None
        return int(number)

    def _latest_checkpoint_evidence(self) -> dict[str, Any]:
        """Return the newest SHA-authorized metrics-only checkpoint.

        Checkpoint CV is evaluation evidence, not a deployed model.  The state
        item authorizes the immutable metrics result by SHA-256; learning-curve
        rows are deliberately not considered here.
        """
        training_root = self.root / "training"
        strict_result = self.cache.json(training_root / "strict_data_status.json", {})
        warnings = self._warnings(strict_result)
        if strict_result.exists and strict_result.warning:
            return {"candidate": None, "warnings": warnings}
        strict_payload = strict_result.value if isinstance(strict_result.value, dict) else {}
        strict_identity = (
            strict_payload.get("state_identity")
            if isinstance(strict_payload.get("state_identity"), dict)
            else {}
        )
        identity_keys = (
            (
                "solver_revision_cohort",
                "physics_data_revision",
                "library_revision",
            )
            if strict_identity.get("solver_revision_cohort")
            else ("solver_revision", "library_revision")
        )
        required_identity = {
            key: str(strict_identity[key]).strip().lower()
            for key in identity_keys
            if _safe_text(strict_identity.get(key), 160)
        }

        candidates: list[tuple[int, float, dict[str, Any]]] = []
        runs_root = training_root / "checkpoint_runs"
        try:
            state_paths = sorted(runs_root.glob("*/checkpoint_state.json"))
        except OSError as exc:
            return {
                "candidate": None,
                "warnings": warnings + [f"checkpoint state scan failed: {exc}"],
            }

        for state_path in state_paths:
            state_result = self.cache.json(state_path, {})
            if state_result.warning:
                warnings.append(state_result.warning)
                continue
            state = state_result.value if isinstance(state_result.value, dict) else {}
            if state.get("schema_version") != CHECKPOINT_STATE_SCHEMA_VERSION:
                continue
            identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
            if any(
                str(identity.get(key, "")).strip().lower() != expected
                for key, expected in required_identity.items()
            ):
                # Other solver/library runs are retained as archives and are
                # expected to coexist with the currently pinned campaign.
                continue
            completed = state.get("completed")
            if not isinstance(completed, list):
                continue
            try:
                run_root = state_path.parent.resolve()
            except OSError as exc:
                warnings.append(f"checkpoint run path resolution failed: {exc}")
                continue

            for item in completed:
                if not isinstance(item, dict) or item.get("kind") != "metrics_only":
                    continue
                threshold = self._exact_integer(item.get("threshold"))
                strict_rows = self._exact_integer(item.get("actual_strict_full_rows"))
                completed_text = _safe_text(item.get("completed_at"), 80)
                completed_at = _parse_time(completed_text, self.clock().tzinfo)
                if threshold is None or threshold < 0 or strict_rows is None or completed_at is None:
                    warnings.append(f"checkpoint completion is malformed: {state_path}")
                    continue

                metrics_text = _safe_text(item.get("metrics_result"), 4_096)
                if not metrics_text:
                    warnings.append(f"checkpoint metrics path is missing: {state_path}")
                    continue
                metrics_path = Path(metrics_text)
                if not metrics_path.is_absolute():
                    metrics_path = run_root / metrics_path
                try:
                    metrics_path = metrics_path.resolve()
                    if not metrics_path.is_relative_to(run_root):
                        raise ValueError("metrics result escapes checkpoint run root")
                except (OSError, ValueError) as exc:
                    warnings.append(f"checkpoint metrics path rejected: {exc}")
                    continue

                expected_hash = str(item.get("metrics_result_sha256", "")).strip().lower()
                actual_hash = _sha256_file(metrics_path)
                if not expected_hash or actual_hash != expected_hash:
                    warnings.append(f"checkpoint metrics hash mismatch: {metrics_path}")
                    continue
                metrics_result = self.cache.json(metrics_path, {})
                if metrics_result.warning or not metrics_result.exists:
                    warnings.extend(self._warnings(metrics_result))
                    continue
                metrics_payload = (
                    metrics_result.value if isinstance(metrics_result.value, dict) else {}
                )
                snapshot_sha = str(item.get("snapshot_sha256", "")).strip().lower()
                profile_sha = str(item.get("profile_sha256", "")).strip().lower()
                payload_checkpoint = self._exact_integer(metrics_payload.get("checkpoint"))
                payload_rows = self._exact_integer(metrics_payload.get("strict_full_rows"))
                evidence_matches = (
                    metrics_payload.get("schema_version") == CHECKPOINT_METRICS_SCHEMA_VERSION
                    and payload_checkpoint == threshold
                    and payload_rows == strict_rows
                    and str(metrics_payload.get("dataset_sha256", "")).strip().lower() == snapshot_sha
                    and str(metrics_payload.get("profile_sha256", "")).strip().lower() == profile_sha
                    and bool(snapshot_sha)
                    and bool(profile_sha)
                )
                if not evidence_matches:
                    warnings.append(f"checkpoint metrics identity mismatch: {metrics_path}")
                    continue
                state_profile_sha = str(identity.get("profile_sha256", "")).strip().lower()
                if state_profile_sha and state_profile_sha != profile_sha:
                    warnings.append(f"checkpoint state/profile identity mismatch: {state_path}")
                    continue

                metrics_rows = metrics_payload.get("metrics")
                if not isinstance(metrics_rows, list):
                    warnings.append(f"checkpoint metrics rows are malformed: {metrics_path}")
                    continue
                global_metrics: dict[str, dict[str, Any]] = {}
                duplicate_target = False
                for row in metrics_rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("slice", "global")).strip().lower() != "global":
                        continue
                    target = _safe_text(row.get("target"), 120)
                    if not target:
                        continue
                    if target in global_metrics:
                        duplicate_target = True
                        break
                    global_metrics[target] = row
                if duplicate_target or not global_metrics:
                    warnings.append(f"checkpoint global metrics are ambiguous or empty: {metrics_path}")
                    continue

                evaluated_at = (
                    _safe_text(metrics_payload.get("completed_at"), 80)
                    or completed_text
                )
                activation_minimum = self._exact_integer(
                    identity.get("activation_minimum_strict_full_rows")
                )
                if activation_minimum is None:
                    activation_minimum = self._exact_integer(
                        item.get("activation_minimum_strict_full_rows")
                    )
                candidate = {
                    "checkpoint": threshold,
                    "completed_at": completed_text,
                    "evaluated_at": evaluated_at,
                    "strict_full_rows": strict_rows,
                    "activation_minimum_strict_full_rows": activation_minimum,
                    "metrics": global_metrics,
                    "metrics_payload": metrics_payload,
                    "metrics_path": metrics_path,
                    "state_path": state_path,
                    "parity_path": metrics_path.with_suffix(".parity.json"),
                    "parity_targets": {},
                    "parity_metadata": {},
                }
                candidates.append((threshold, completed_at.timestamp(), candidate))

        if not candidates:
            return {"candidate": None, "warnings": list(dict.fromkeys(warnings))}
        _, _, selected = max(candidates, key=lambda entry: (entry[0], entry[1]))
        parity_path = selected["parity_path"]
        parity_result = self.cache.json(parity_path, {}, max_bytes=64 * 1024 * 1024)
        if parity_result.warning:
            warnings.append(parity_result.warning)
        elif parity_result.exists:
            parity = parity_result.value if isinstance(parity_result.value, dict) else {}
            metrics_payload = selected["metrics_payload"]
            parity_matches = (
                parity.get("schema_version") == CHECKPOINT_PARITY_SCHEMA_VERSION
                and parity.get("artifact_type") == CHECKPOINT_PARITY_ARTIFACT_TYPE
                and self._exact_integer(parity.get("checkpoint")) == selected["checkpoint"]
                and self._exact_integer(parity.get("strict_full_rows")) == selected["strict_full_rows"]
                and str(parity.get("dataset_sha256", "")).strip().lower()
                == str(metrics_payload.get("dataset_sha256", "")).strip().lower()
                and str(parity.get("profile_sha256", "")).strip().lower()
                == str(metrics_payload.get("profile_sha256", "")).strip().lower()
            )
            raw_targets = parity.get("targets")
            if not parity_matches or not isinstance(raw_targets, dict):
                warnings.append(f"checkpoint parity identity mismatch: {parity_path}")
            else:
                parity_targets: dict[str, dict[str, Any]] = {}
                parity_error: str | None = None
                for target, target_payload in raw_targets.items():
                    if not isinstance(target, str) or not isinstance(target_payload, dict):
                        parity_error = "target payload is malformed"
                        break
                    pairs = target_payload.get("pairs")
                    sample_count = self._exact_integer(target_payload.get("sample_count"))
                    n_rows = self._exact_integer(target_payload.get("n"))
                    metric = selected["metrics"].get(target)
                    metric_n = self._exact_integer(metric.get("n")) if metric else None
                    if (
                        not isinstance(pairs, list)
                        or sample_count != len(pairs)
                        or sample_count is None
                        or sample_count > CHECKPOINT_PARITY_PAIR_LIMIT
                        or n_rows is None
                        or sample_count > n_rows
                        or (metric_n is not None and metric_n != n_rows)
                    ):
                        parity_error = f"{target} sample metadata is malformed"
                        break
                    clean_pairs = []
                    for pair in pairs:
                        if not isinstance(pair, dict):
                            parity_error = f"{target} pair is malformed"
                            break
                        actual = _finite_number(pair.get("actual"))
                        predicted = _finite_number(pair.get("predicted"))
                        if actual is None or predicted is None:
                            parity_error = f"{target} pair is non-finite"
                            break
                        clean_pairs.append({
                            "row_position": self._exact_integer(pair.get("row_position")),
                            "row_index": _coerce(pair.get("row_index")),
                            "actual": actual,
                            "predicted": predicted,
                        })
                    if parity_error:
                        break
                    parity_targets[target] = {
                        "n": n_rows,
                        "sample_count": sample_count,
                        "sampling": (
                            target_payload.get("sampling")
                            if isinstance(target_payload.get("sampling"), dict)
                            else {}
                        ),
                        "pairs": clean_pairs,
                    }
                if parity_error:
                    warnings.append(f"checkpoint parity malformed ({parity_error}): {parity_path}")
                else:
                    selected["parity_targets"] = parity_targets
                    selected["parity_metadata"] = {
                        "artifact_type": parity.get("artifact_type"),
                        "prediction_kind": _safe_text(parity.get("prediction_kind"), 80),
                        "cv": parity.get("cv") if isinstance(parity.get("cv"), dict) else {},
                        "max_pairs_per_target": self._exact_integer(
                            parity.get("max_pairs_per_target")
                        ),
                    }

        return {"candidate": selected, "warnings": list(dict.fromkeys(warnings))}

    def models(self, current_data_count: int | None = None) -> dict[str, Any]:
        registry = self.root / "training" / "registry"
        pointer_result = self.cache.json(registry / "current.json", {})
        pointer = (
            pointer_result.value
            if isinstance(pointer_result.value, dict)
            else {}
        )
        generation = registry / "generations" / "__unavailable__"
        pointer_warning = pointer_result.warning
        relative = pointer.get("generation")
        try:
            if pointer_warning:
                raise ValueError("active model pointer is unreadable")
            candidate = (registry / relative).resolve() if relative else None
            generations_root = (registry / "generations").resolve()
            if pointer.get("schema_version") != 2:
                raise ValueError("accepted schema-v2 model pointer is unavailable")
            if candidate is None or not candidate.is_relative_to(generations_root):
                raise ValueError("model pointer escapes generations root")
            generation = candidate
        except (OSError, TypeError, ValueError) as exc:
            pointer_warning = str(exc)
        report_result = self.cache.json(generation / "train_report.json", {})
        gate_result = self.cache.json(generation / "quality_gate.json", {})
        curve_result = self.cache.csv(self.root / "training" / "learning_curve.csv", max_rows=200_000)
        report_payload = report_result.value if isinstance(report_result.value, dict) else {}
        gate_payload = gate_result.value if isinstance(gate_result.value, dict) else {}
        report = report_payload.get("report") if isinstance(report_payload.get("report"), dict) else {}
        curve_rows = curve_result.value if isinstance(curve_result.value, list) else []
        warnings = self._warnings(pointer_result, report_result, gate_result, curve_result)
        evidence_error = pointer_warning or report_result.warning or gate_result.warning
        if not evidence_error and (
            _sha256_file(Path(report_result.path))
            != pointer.get("generation_report_sha256")
        ):
            evidence_error = "active generation report fingerprint mismatch"
        if not evidence_error and (
            _sha256_file(Path(gate_result.path))
            != pointer.get("quality_gate_sha256")
        ):
            evidence_error = "active generation gate fingerprint mismatch"
        if pointer_warning:
            warnings.append(pointer_warning)
            report = {}
        elif evidence_error:
            warnings.append(str(evidence_error))
            report = {}
        elif (
            gate_payload.get("passed") is not True
            or gate_payload.get("training_run_id") != pointer.get("training_run_id")
            or report_payload.get("training_run_id") != pointer.get("training_run_id")
            or gate_payload.get("generation") != pointer.get("generation")
            or gate_payload.get("generation_report_sha256")
            != pointer.get("generation_report_sha256")
            or gate_payload.get("dataset_sha256") != pointer.get("dataset_sha256")
            or gate_payload.get("profile_sha256") != pointer.get("profile_sha256")
            or report_payload.get("dataset_sha256") != pointer.get("dataset_sha256")
            or report_payload.get("profile_sha256") != pointer.get("profile_sha256")
            or report_payload.get("strict_full_rows")
            != pointer.get("strict_full_rows")
        ):
            warnings.append("active model generation has no matching passing gate")
            report = {}
        if current_data_count is None:
            current_data_count = self.data()["total_rows"]

        checkpoint_result = self._latest_checkpoint_evidence()
        warnings.extend(checkpoint_result["warnings"])
        checkpoint = checkpoint_result["candidate"]
        checkpoint_metrics = checkpoint["metrics"] if checkpoint else {}
        activation_minimum = (
            checkpoint["activation_minimum_strict_full_rows"] if checkpoint else None
        )
        preactivation_checkpoint = bool(
            checkpoint
            and not pointer_result.exists
            and activation_minimum is not None
            and current_data_count < activation_minimum
        )
        if preactivation_checkpoint and pointer_warning:
            # Before the activation floor, an absent pointer is the expected
            # state: checkpoint CV has run, but no deployable generation may be
            # promoted yet.  Existing/corrupt pointers and post-floor absence
            # remain warnings.
            warnings = [warning for warning in warnings if warning != pointer_warning]

        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in curve_rows:
            target = _safe_text(row.get("target"), 120)
            if not target or str(row.get("slice", "global")).strip().lower() != "global":
                continue
            item = {
                "time": _safe_text(row.get("time"), 80),
                "n": _integer(row.get("n"), 0),
                "r2": _finite_number(row.get("r2")),
                "rmse": _finite_number(row.get("rmse")),
                "mape_pct": _finite_number(row.get("mape_pct")),
                "p90_ape_pct": _finite_number(row.get("p90_ape_pct")),
            }
            histories[target].append(item)

        ordered_targets = [item["name"] for item in TARGETS]
        ordered_targets.extend(
            sorted((set(report) | set(checkpoint_metrics)) - set(ordered_targets))
        )
        models: list[dict[str, Any]] = []
        latest_times: list[datetime] = []
        for target in ordered_targets:
            active_metrics = report.get(target) if isinstance(report.get(target), dict) else None
            meta_result = self.cache.json(generation / target / "meta.json", {})
            if meta_result.warning:
                warnings.append(meta_result.warning)
            meta = meta_result.value if isinstance(meta_result.value, dict) else {}
            if active_metrics is None and isinstance(meta.get("metrics"), dict):
                active_metrics = meta["metrics"]
            active_metrics = active_metrics or {}
            trained = bool(active_metrics)
            checkpoint_metric = (
                checkpoint_metrics.get(target)
                if not trained and isinstance(checkpoint_metrics.get(target), dict)
                else None
            )
            evaluated = trained or checkpoint_metric is not None
            metrics = active_metrics if trained else (checkpoint_metric or {})
            trained_at = (
                _safe_text(meta.get("trained_at") or report_payload.get("time"), 80)
                if trained else None
            )
            parsed_trained_at = _parse_time(trained_at, self.clock().tzinfo)
            if parsed_trained_at:
                latest_times.append(parsed_trained_at)
            n_train = _integer(metrics.get("n_train"), 0) if trained else 0
            n_holdout = _integer(metrics.get("n_holdout"), 0) if trained else 0
            n_used = (
                n_train + n_holdout
                if trained else max(0, self._exact_integer(metrics.get("n")) or 0)
            )
            stale = bool(trained and current_data_count and n_used and current_data_count >= n_used + max(100, int(n_used * 0.25)))
            history = [dict(item) for item in histories.get(target, [])[-100:]]
            r2 = _finite_number(metrics.get("r2"))
            mape = _finite_number(metrics.get("mape_pct"))
            p90_ape = _finite_number(metrics.get("p90_ape_pct"))
            mape_n = self._exact_integer(metrics.get("mape_n"))
            mape_excluded_zero_count = self._exact_integer(
                metrics.get("mape_excluded_zero_count")
            )
            mape_zero_abs_tolerance = _finite_number(
                metrics.get("mape_zero_abs_tolerance")
            )
            percentage_metric_source = (
                "artifact_zero_aware" if mape_n is not None
                else "legacy_all_rows"
            )
            # Legacy checkpoints used max(|actual|, 1e-9) as the denominator,
            # so a structural zero could turn one ordinary prediction error
            # into a multi-million-percent MAPE.  The parity artifact contains
            # every OOF pair while n <= the artifact limit, allowing an exact,
            # non-mutating display correction until the next zero-aware
            # checkpoint is produced.
            parity_metric = (
                checkpoint["parity_targets"].get(target, {})
                if checkpoint_metric is not None else {}
            )
            parity_pairs = parity_metric.get("pairs", [])
            if (
                mape_n is None
                and isinstance(parity_pairs, list)
                and parity_metric.get("sample_count") == parity_metric.get("n")
                and parity_metric.get("n") == n_used
            ):
                corrected = _zero_aware_percentage_metrics(
                    [pair.get("actual") for pair in parity_pairs],
                    [pair.get("predicted") for pair in parity_pairs],
                )
                if corrected["mape_n"] > 0:
                    mape = corrected["mape_pct"]
                    p90_ape = corrected["p90_ape_pct"]
                    mape_n = corrected["mape_n"]
                    mape_excluded_zero_count = corrected[
                        "mape_excluded_zero_count"
                    ]
                    mape_zero_abs_tolerance = corrected[
                        "mape_zero_abs_tolerance"
                    ]
                    percentage_metric_source = "parity_recomputed_zero_aware"
                    for item in reversed(history):
                        if item.get("n") == n_used:
                            item.update({
                                "mape_pct": mape,
                                "p90_ape_pct": p90_ape,
                                "mape_n": mape_n,
                                "mape_excluded_zero_count": (
                                    mape_excluded_zero_count
                                ),
                                "mape_zero_abs_tolerance": (
                                    mape_zero_abs_tolerance
                                ),
                            })
                            break
            # learning_curve.csv remains historical display data only.  It is
            # never used as the numeric source for checkpoint fallback rows.
            previous = history[-1] if trained and history else None
            attention = bool(trained and ((r2 is not None and r2 < 0.8) or (mape is not None and mape > 20.0)))
            if checkpoint_metric is not None:
                status = "checkpoint"
            elif not trained:
                status = "not_trained"
            elif stale:
                status = "stale"
            elif attention:
                status = "attention"
            else:
                status = "trained"
            models.append({
                "target": target,
                "label": TARGET_META.get(target, {}).get("label", target),
                "unit": TARGET_META.get(target, {}).get("unit", ""),
                "status": status,
                "trained": trained,
                "evaluated": evaluated,
                "deployable": trained,
                "evaluation_kind": (
                    "active_registry" if trained
                    else ("checkpoint_cv" if checkpoint_metric is not None else None)
                ),
                "checkpoint": checkpoint["checkpoint"] if checkpoint_metric is not None else None,
                "stale": stale,
                "n_train": n_train,
                "n_holdout": n_holdout,
                "n_used": n_used,
                "r2": r2,
                "rmse": _finite_number(metrics.get("rmse")),
                "mape_pct": mape,
                "p90_ape_pct": p90_ape,
                "mape_n": mape_n,
                "mape_excluded_zero_count": mape_excluded_zero_count,
                "mape_zero_abs_tolerance": mape_zero_abs_tolerance,
                "percentage_metric_source": percentage_metric_source,
                "q90_conformal": _finite_number(metrics.get("q90_conformal") or meta.get("q90")),
                "trained_at": trained_at,
                "evaluated_at": (
                    trained_at
                    if trained else (checkpoint["evaluated_at"] if checkpoint_metric is not None else None)
                ),
                "delta_r2": (r2 - previous["r2"] if r2 is not None and previous and previous["r2"] is not None else None),
                "delta_mape_pct": (mape - previous["mape_pct"] if mape is not None and previous and previous["mape_pct"] is not None else None),
                "parity_available": bool(
                    checkpoint_metric is not None
                    and checkpoint["parity_targets"].get(target, {}).get("pairs")
                ),
                "parity_sample_count": (
                    checkpoint["parity_targets"].get(target, {}).get("sample_count", 0)
                    if checkpoint_metric is not None else 0
                ),
                "parity_source": (
                    str(checkpoint["parity_path"])
                    if checkpoint_metric is not None
                    and target in checkpoint["parity_targets"] else None
                ),
                "source_kind": (
                    "active_registry" if trained
                    else ("checkpoint_cv" if checkpoint_metric is not None else None)
                ),
                "source": (
                    str(report_result.path) if trained
                    else (str(checkpoint["metrics_path"]) if checkpoint_metric is not None else None)
                ),
                "history": history,
            })

        trained_count = sum(model["trained"] for model in models)
        evaluated_count = sum(model["evaluated"] for model in models)
        primary_source = (
            str(report_result.path) if trained_count
            else (str(checkpoint["metrics_path"]) if checkpoint else str(report_result.path))
        )
        quality_note = (
            f"현재 strict 데이터 {current_data_count}개는 활성화 기준 {activation_minimum}개 미만이므로 "
            "검증된 checkpoint CV 평가만 표시하며 배포 모델로 취급하지 않습니다."
            if preactivation_checkpoint else
            "주의 표시는 탐색용 기준(R² < 0.8 또는 MAPE > 20%)이며, 최종 합격은 독립 FEA 검증으로 판정합니다."
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "available": report_result.exists or curve_result.exists or checkpoint is not None,
            "target_count": len(models),
            "trained_count": trained_count,
            "evaluated_count": evaluated_count,
            "missing_count": len(models) - trained_count,
            "current_data_count": current_data_count,
            "latest_trained_at": _iso(max(latest_times)) if latest_times else _safe_text(report_payload.get("time"), 80),
            "latest_checkpoint": checkpoint["checkpoint"] if checkpoint else None,
            "checkpoint_evaluated_at": checkpoint["evaluated_at"] if checkpoint else None,
            "activation_minimum_strict_full_rows": activation_minimum,
            "activation_state": (
                "active_registry" if trained_count
                else ("preactivation_checkpoint" if preactivation_checkpoint
                      else ("activation_due" if checkpoint and activation_minimum is not None
                            and current_data_count >= activation_minimum else "unavailable"))
            ),
            "quality_note": quality_note,
            "models": models,
            "warnings": list(dict.fromkeys(warnings)),
            "source": primary_source,
            "source_kind": (
                "active_registry" if trained_count
                else ("checkpoint_cv" if checkpoint else "unavailable")
            ),
            "active_source": str(report_result.path),
            "checkpoint_source": str(checkpoint["metrics_path"]) if checkpoint else None,
            "checkpoint_state_source": str(checkpoint["state_path"]) if checkpoint else None,
        }

    def model_history(self, target: str) -> dict[str, Any] | None:
        if target not in TARGET_META:
            return None
        models = self.models()
        model = next((item for item in models["models"] if item["target"] == target), None)
        return {"target": target, "label": TARGET_META[target]["label"], "history": model["history"] if model else []}

    def model_parity(self, target: str) -> dict[str, Any] | None:
        if target not in TARGET_META:
            return None
        models = self.models()
        model = next((item for item in models["models"] if item["target"] == target), None)
        base = {
            "schema_version": SCHEMA_VERSION,
            "target": target,
            "label": TARGET_META[target]["label"],
            "unit": TARGET_META[target]["unit"],
            "available": False,
            "evaluation_kind": model.get("evaluation_kind") if model else None,
            "checkpoint": model.get("checkpoint") if model else None,
            "evaluated_at": model.get("evaluated_at") if model else None,
            "n": model.get("n_used", 0) if model else 0,
            "sample_count": 0,
            "sampling": {},
            "pairs": [],
            "metadata": {},
            "source": None,
            "warnings": models.get("warnings", []),
        }
        # A passing active registry has priority.  A checkpoint sidecar is not
        # presented as parity evidence for a different, deployed generation.
        if not model or model.get("evaluation_kind") != "checkpoint_cv":
            return base

        checkpoint_result = self._latest_checkpoint_evidence()
        candidate = checkpoint_result["candidate"]
        warnings = list(dict.fromkeys(
            [*base["warnings"], *checkpoint_result["warnings"]]
        ))
        if not candidate or candidate["checkpoint"] != model.get("checkpoint"):
            base["warnings"] = warnings
            return base
        parity = candidate["parity_targets"].get(target)
        if not isinstance(parity, dict) or not parity.get("pairs"):
            base["warnings"] = warnings
            return base
        return {
            **base,
            "available": True,
            "checkpoint": candidate["checkpoint"],
            "evaluated_at": candidate["evaluated_at"],
            "n": parity["n"],
            "sample_count": parity["sample_count"],
            "sampling": parity["sampling"],
            "pairs": parity["pairs"],
            "metadata": candidate["parity_metadata"],
            "source": str(candidate["parity_path"]),
            "warnings": warnings,
        }

    @staticmethod
    def _constraint(value: float | None, limit: float | tuple[float, float], mode: str) -> dict[str, Any]:
        if value is None:
            return {"value": None, "limit": limit, "margin": None, "pass": None}
        if mode == "max":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": limit_value - value, "pass": value <= limit_value}
        if mode == "min":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": value - limit_value, "pass": value >= limit_value}
        low, high = limit
        return {
            "value": value,
            "limit": [low, high],
            "margin": min(value - low, high - value),
            "pass": low <= value <= high,
        }

    def _candidate(self, row: dict[str, Any], round_number: int, index: int) -> dict[str, Any]:
        volume = _finite_number(row.get("volume_L"))
        loss = _finite_number(row.get("total_loss_W"))
        llt = _finite_number(row.get("pred_Llt_phys"))
        b_design = _finite_number(row.get("B_design_analytic_T"))
        b_mean = _finite_number(row.get("pred_B_mean_core"))
        bmax_diagnostic = _finite_number(row.get("pred_B_max_core"))
        temperatures = {
            target: _finite_number(row.get(f"pred_{target}")) for target in TEMPERATURE_TARGETS
        }
        available_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(available_temperatures) if available_temperatures else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(row.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        constraints = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 100.0, "max"),
            "bfield": self._constraint(b_design, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
        }
        passes = [item["pass"] for item in constraints.values()]
        spec_status = "fail" if False in passes else ("pass" if passes and all(value is True for value in passes) else "unknown")
        sigmas = {
            key.removeprefix("sigma_"): value
            for key, raw in row.items()
            if key.startswith("sigma_") and (value := _finite_number(raw)) is not None
        }
        parameters = {
            key: _coerce(row.get(key)) for key in DESIGN_PARAMETER_KEYS if row.get(key) not in (None, "")
        }
        report = {
            key: _coerce(row.get(key))
            for key in CANDIDATE_REPORT_FIELDS
            if row.get(key) not in (None, "")
        }
        return {
            "id": f"r{round_number:02d}-{index:04d}",
            "index": index,
            "round": round_number,
            "volume_L": volume,
            "total_loss_W": loss,
            "pred_Llt_phys": llt,
            "B_design_analytic_T": b_design,
            "pred_B_mean_core": b_mean,
            "diagnostic_pred_B_max_core": bmax_diagnostic,
            "pred_max_temperature_C": max_temperature,
            "pred_temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "constraints": constraints,
            "spec_status": spec_status,
            "uncertainty": sigmas,
            "report": report,
            "parameters": parameters,
        }

    def _round_directories(self) -> list[tuple[int, Path]]:
        base = self.root / "al_rounds"
        if not base.is_dir():
            return []
        found = []
        for child in base.iterdir():
            match = re.fullmatch(r"round_(\d+)", child.name)
            if match and child.is_dir() and (child / "pareto_front.csv").exists():
                found.append((int(match.group(1)), child))
        return sorted(found)

    def nsga2(self) -> dict[str, Any]:
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        rounds = self._round_directories()
        warnings = self._warnings(state_result)
        if not rounds:
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "waiting",
                "round": _integer(state.get("round"), 0) if state else None,
                "al_stage": _safe_text(state.get("stage"), 30),
                "candidate_count": 0,
                "configured_restarts": 16,
                "completed_restarts": None,
                "candidates": [],
                "summary": {},
                "rounds": [],
                "warnings": warnings,
            }

        round_summaries: list[dict[str, Any]] = []
        parsed_rounds: dict[int, tuple[ReadResult, list[dict[str, str]]]] = {}
        for number, directory in rounds:
            result = self.cache.csv(directory / "pareto_front.csv", max_rows=20_000)
            rows = result.value if isinstance(result.value, list) else []
            parsed_rounds[number] = (result, rows)
            if result.warning:
                warnings.append(result.warning)
            volumes = [value for row in rows if (value := _finite_number(row.get("volume_L"))) is not None]
            losses = [value for row in rows if (value := _finite_number(row.get("total_loss_W"))) is not None]
            round_summaries.append({
                "round": number,
                "candidate_count": len(rows),
                "min_volume_L": min(volumes) if volumes else None,
                "min_loss_W": min(losses) if losses else None,
                "updated_at": _iso(result.mtime),
            })

        latest_round, latest_dir = rounds[-1]
        latest_result, latest_rows = parsed_rounds[latest_round]
        candidates = [self._candidate(row, latest_round, index) for index, row in enumerate(latest_rows)]
        volume_candidates = [item for item in candidates if item["volume_L"] is not None]
        loss_candidates = [item for item in candidates if item["total_loss_W"] is not None]
        minimum_volume = min(volume_candidates, key=lambda item: item["volume_L"]) if volume_candidates else None
        minimum_loss = min(loss_candidates, key=lambda item: item["total_loss_W"]) if loss_candidates else None
        for candidate in candidates:
            candidate["is_min_volume"] = bool(minimum_volume and candidate["id"] == minimum_volume["id"])
            candidate["is_min_loss"] = bool(minimum_loss and candidate["id"] == minimum_loss["id"])
        candidates.sort(key=lambda item: (item["volume_L"] is None, item["volume_L"] or 0.0))

        latest_summary = round_summaries[-1]
        previous_summary = round_summaries[-2] if len(round_summaries) > 1 else None
        comparison = None
        if previous_summary:
            comparison = {
                "previous_round": previous_summary["round"],
                "min_volume_change_L": (
                    latest_summary["min_volume_L"] - previous_summary["min_volume_L"]
                    if latest_summary["min_volume_L"] is not None and previous_summary["min_volume_L"] is not None else None
                ),
                "min_loss_change_W": (
                    latest_summary["min_loss_W"] - previous_summary["min_loss_W"]
                    if latest_summary["min_loss_W"] is not None and previous_summary["min_loss_W"] is not None else None
                ),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            "status": "running" if str(state.get("stage", "")).upper() == "OPTIMIZE" else "completed",
            "round": latest_round,
            "al_round": _integer(state.get("round"), latest_round) if state else latest_round,
            "al_stage": _safe_text(state.get("stage"), 30),
            "candidate_count": len(candidates),
            "configured_restarts": 16,
            "completed_restarts": state.get("nsga2_restarts_completed") if state else None,
            "candidates": candidates,
            "summary": {
                **latest_summary,
                "min_volume_candidate_id": minimum_volume["id"] if minimum_volume else None,
                "min_loss_candidate_id": minimum_loss["id"] if minimum_loss else None,
                "known_spec_pass_count": sum(item["spec_status"] == "pass" for item in candidates),
                "known_spec_fail_count": sum(item["spec_status"] == "fail" for item in candidates),
                "unknown_spec_count": sum(item["spec_status"] == "unknown" for item in candidates),
            },
            "comparison": comparison,
            "rounds": round_summaries,
            "source": str(latest_dir / "pareto_front.csv"),
            "updated_at": _iso(latest_result.mtime),
            "note": "Pareto 파일의 해는 최적화기가 feasible로 반환한 후보입니다. 아직 학습되지 않은 출력의 명목 사양 판정은 ‘확인 불가’로 표시합니다.",
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _evaluate_fea(self, result: dict[str, Any], require_full_model: bool = False) -> dict[str, Any]:
        full_model_value = _finite_number(result.get("full_model"))
        llt = _finite_number(result.get("Llt_phys"))
        if llt is None:
            raw_llt = _finite_number(result.get("Llt"))
            if raw_llt is not None:
                llt = raw_llt * (1.0 if full_model_value == 1.0 else 2.0)
        bmax = _finite_number(result.get("B_max_core"))
        n2_side = _finite_number(result.get("N2_side"))
        temperature_keys = ["T_max_Tx", "T_max_Rx_main", "T_max_core"]
        if n2_side is not None and n2_side > 0:
            temperature_keys.insert(2, "T_max_Rx_side")
        temperatures = {key: _finite_number(result.get(key)) for key in temperature_keys}
        finite_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(finite_temperatures) if len(finite_temperatures) == len(temperature_keys) else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(result.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        matrix_error = _finite_number(result.get("conv_error_pct_matrix"))
        loss_error = _finite_number(result.get("conv_error_pct_loss"))
        convergence_value = max(matrix_error, loss_error) if matrix_error is not None and loss_error is not None else None
        loss_components = [
            _finite_number(result.get(key)) for key in
            ("P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total")
        ]
        losses_complete = all(value is not None for value in loss_components)
        losses_nonnegative = (
            all(value >= 0 for value in loss_components) if losses_complete else None
        )
        total_loss = sum(loss_components) if losses_complete else None
        checks = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 100.0, "max"),
            "bmax": self._constraint(bmax, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
            "convergence": self._constraint(convergence_value, 1.5, "max"),
            "loss_components": {
                "value": total_loss,
                "limit": "all four finite and >= 0 W",
                "margin": None,
                "pass": losses_nonnegative,
            },
        }
        if require_full_model:
            checks["full_model"] = {
                "value": full_model_value,
                "limit": 1,
                "margin": None,
                "pass": full_model_value == 1.0 if full_model_value is not None else None,
            }
        states = [item["pass"] for item in checks.values()]
        computed_status = "fail" if False in states else ("pass" if states and all(value is True for value in states) else "unknown")
        return {
            "computed_status": computed_status,
            "checks": checks,
            "Llt_phys_uH": llt,
            "B_max_core_T": bmax,
            "max_temperature_C": max_temperature,
            "temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "total_loss_W": total_loss,
            "volume_L": _finite_number(result.get("volume_L")),
            "conv_error_pct_matrix": matrix_error,
            "conv_error_pct_loss": loss_error,
            "solver_revision": _safe_text(result.get("git_hash"), 40),
            "library_revision": _safe_text(result.get("pyaedt_library_git_hash"), 40),
            "timing_seconds": {
                "matrix": _duration_seconds(result.get("time_matrix")),
                "loss": _duration_seconds(result.get("time_loss")),
                "icepak": _duration_seconds(result.get("time_thermal")),
                "total": _duration_seconds(result.get("time")),
            },
            "parameters": {
                key: _coerce(result.get(key)) for key in DESIGN_PARAMETER_KEYS if result.get(key) not in (None, "")
            },
        }

    def _final_artifact(self) -> tuple[ReadResult, dict[str, Any]]:
        paths = (
            self.root / "verify" / "results" / "final_verification.json",
            self.root / "verify" / "final_verification.json",
            self.root / "monitoring" / "runtime" / "final_verification.json",
        )
        for path in paths:
            if path.exists():
                result = self.cache.json(path, {})
                value = result.value if isinstance(result.value, dict) else {}
                return result, value
        return ReadResult({}, str(paths[0]), False), {}

    def verification(self, nsga: dict[str, Any] | None = None) -> dict[str, Any]:
        nsga = nsga or self.nsga2()
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        warnings = self._warnings(state_result)
        records = state.get("task_records") if isinstance(state.get("task_records"), dict) else {}
        candidates_by_index = {item["index"]: item for item in nsga.get("candidates", [])}
        standard: list[dict[str, Any]] = []
        for index_text, record_value in records.items():
            if not isinstance(record_value, dict):
                continue
            index = _integer(index_text, -1)
            result = record_value.get("result") if isinstance(record_value.get("result"), dict) else None
            evaluation = self._evaluate_fea(result) if result else None
            predicted = candidates_by_index.get(index)
            standard.append({
                "candidate_id": predicted.get("id") if predicted else f"r{_integer(state.get('round'), 0):02d}-{max(index, 0):04d}",
                "index": index,
                "profile": "standard",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": evaluation,
                "predicted": predicted,
                "error": _safe_text(record_value.get("fetch_error") or record_value.get("error"), 500),
            })

        fine_records = state.get("fine_task_records") \
            if isinstance(state.get("fine_task_records"), dict) else {}
        fine_queue = state.get("final_candidates") \
            if isinstance(state.get("final_candidates"), list) else []
        fine_candidates: list[dict[str, Any]] = []
        for rank_text, record_value in fine_records.items():
            if not isinstance(record_value, dict):
                continue
            rank = _integer(rank_text, -1)
            candidate = fine_queue[rank] if 0 <= rank < len(fine_queue) \
                and isinstance(fine_queue[rank], dict) else {}
            result = record_value.get("result") \
                if isinstance(record_value.get("result"), dict) else None
            fine_candidates.append({
                "rank": rank,
                "candidate_id": _safe_text(candidate.get("candidate_digest"), 100),
                "volume_L": _finite_number(candidate.get("volume_L")),
                "profile": "fine",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": self._evaluate_fea(result, require_full_model=True) if result else None,
                "error": _safe_text(
                    record_value.get("unverified_reason")
                    or record_value.get("fetch_error")
                    or record_value.get("error"), 500,
                ),
            })

        verification_counts = state.get("verification_counts") if isinstance(state.get("verification_counts"), dict) else {}
        if verification_counts:
            counts = {
                "total": _integer(verification_counts.get("total"), len(records)),
                "valid": _integer(verification_counts.get("valid"), 0),
                "pending": _integer(verification_counts.get("pending"), 0),
                "exhausted": _integer(verification_counts.get("exhausted"), 0),
                "ingested": _integer(verification_counts.get("ingested"), 0),
            }
        else:
            counts = {
                "total": len(records),
                "valid": sum(item.get("outcome") == "valid" for item in records.values() if isinstance(item, dict)),
                "pending": sum(item.get("outcome") in {None, "pending", "fetch_error", "submission_unknown"} for item in records.values() if isinstance(item, dict)),
                "exhausted": sum(item.get("outcome") == "exhausted" for item in records.values() if isinstance(item, dict)),
                "ingested": 0,
            }
        counts["coverage"] = counts["valid"] / counts["total"] if counts["total"] else None

        error_files = sorted((self.root / "al_rounds").glob("round_*/verification_errors.csv"))
        errors: list[dict[str, Any]] = []
        error_source = None
        if error_files:
            error_result = self.cache.csv(error_files[-1], max_rows=10_000)
            error_source = str(error_files[-1])
            errors = [
                {key: _coerce(value) for key, value in row.items()}
                for row in (error_result.value if isinstance(error_result.value, list) else [])
            ]
            if error_result.warning:
                warnings.append(error_result.warning)

        final_result, final_payload = self._final_artifact()
        if final_result.warning:
            warnings.append(final_result.warning)
        raw_final_result = final_payload.get("result") if isinstance(final_payload.get("result"), dict) else final_payload
        final_evaluation = self._evaluate_fea(raw_final_result, require_full_model=True) if final_payload else None
        declared = _safe_text(final_payload.get("status"), 30) if final_payload else None
        declared_pass = final_payload.get("passed", final_payload.get("overall_pass")) if final_payload else None
        declared_success = declared_pass is True or (
            declared and declared.lower() in {"pass", "passed", "complete", "completed"}
        )
        declared_failure = declared_pass is False or (
            declared and declared.lower() in {"fail", "failed", "error"}
        )
        # Never let a manually declared PASS override a physical check.  A
        # partial result remains unknown until every required value is present.
        if declared_failure or (final_evaluation and final_evaluation["computed_status"] == "fail"):
            final_status = "fail"
        elif final_evaluation and final_evaluation["computed_status"] == "pass":
            final_status = "pass"
        elif declared_success or final_payload:
            final_status = "unknown"
        elif state.get("stage") == "FINE_BLOCKED":
            final_status = "blocked"
        else:
            final_status = "waiting"
        final = {
            "available": bool(final_payload),
            "status": final_status,
            "candidate_id": _safe_text(final_payload.get("candidate_id"), 100) if final_payload else None,
            "profile": _safe_text(final_payload.get("profile"), 30) or ("fine" if final_payload else None),
            "task_id": (
                final_payload.get("fine_task_id") or final_payload.get("task_id")
            ) if final_payload else None,
            "task_status": _safe_text(
                final_payload.get("fine_task_status") or final_payload.get("task_status"), 30
            ) if final_payload else None,
            "evaluation": final_evaluation,
            "declared_status": declared,
            "error": _safe_text(
                (final_payload.get("error") or final_payload.get("failure_reason"))
                if final_payload else state.get("fine_block_reason"), 1000,
            ),
            "updated_at": _safe_text(
                final_payload.get("generated_at") or final_payload.get("updated_at")
                or final_payload.get("time"), 80,
            ) if final_payload else None,
            "source": final_result.path if final_result.exists else None,
        }

        history = state.get("history") if isinstance(state.get("history"), list) else []
        agreement = history[-1] if history and isinstance(history[-1], dict) else None
        return {
            "schema_version": SCHEMA_VERSION,
            "available": bool(state or standard or errors or final_payload),
            "stage": _safe_text(state.get("stage"), 30) or "NOT_STARTED",
            "round": _integer(state.get("round"), 0) if state else None,
            "counts": counts,
            "standard_candidates": standard,
            "fine_candidates": fine_candidates,
            "verification_errors": errors,
            "agreement": agreement,
            "final": final,
            "sources": {"state": state_result.path if state_result.exists else None, "errors": error_source},
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _status(
        self,
        data: dict[str, Any],
        models: dict[str, Any],
        nsga: dict[str, Any],
        verification: dict[str, Any],
        scheduler: dict[str, Any],
    ) -> dict[str, Any]:
        stages = []
        simulation_active = scheduler.get("running", 0) + scheduler.get("pending", 0) > 0
        if simulation_active:
            simulation_state = "active"
            simulation_detail = f"실행 {scheduler.get('running', 0)} · 대기 {scheduler.get('pending', 0)}"
        elif not scheduler.get("connected"):
            simulation_state = "warning"
            simulation_detail = "스케줄러 상태 확인 불가"
        else:
            simulation_state = "waiting"
            simulation_detail = "실행 중 작업 없음"
        stages.append({"key": "simulation", "label": "시뮬레이션", "state": simulation_state, "detail": simulation_detail})

        if data["total_rows"] >= DATA_GOAL:
            data_state, data_detail = "complete", f"목표 달성 · {data['total_rows']:,}개"
        elif data["throughput_1h"] > 0:
            data_state, data_detail = "active", f"최근 1시간 +{data['throughput_1h']}개"
        elif data["stalled"]:
            data_state, data_detail = "warning", "90분 이상 데이터 증가 없음"
        else:
            data_state, data_detail = "waiting", f"{data['total_rows']:,}개 확보"
        stages.append({"key": "data", "label": "데이터 적재", "state": data_state, "detail": data_detail})

        if models.get("activation_state") == "preactivation_checkpoint":
            model_state = "waiting"
            model_detail = (
                f"checkpoint {models.get('latest_checkpoint')} CV "
                f"{models.get('evaluated_count', 0)}/{models.get('target_count', 0)} · "
                f"활성화 {models.get('current_data_count', 0):,}/"
                f"{models.get('activation_minimum_strict_full_rows', 0):,}"
            )
        elif models["trained_count"] == 0:
            model_state, model_detail = "waiting", "학습 모델 없음"
        elif models["missing_count"]:
            model_state, model_detail = "warning", f"{models['trained_count']}/{models['target_count']} 모델 학습"
        else:
            model_state, model_detail = "complete", f"{models['trained_count']}개 모델 준비"
        stages.append({"key": "models", "label": "모델 학습", "state": model_state, "detail": model_detail})

        nsga_state = "active" if nsga["status"] == "running" else ("complete" if nsga["available"] else "waiting")
        nsga_detail = f"round {nsga.get('round')} · {nsga['candidate_count']}개" if nsga["available"] else "실행 전"
        stages.append({"key": "nsga2", "label": "NSGA-II", "state": nsga_state, "detail": nsga_detail})

        verification_stage = str(verification.get("stage", "NOT_STARTED")).upper()
        if verification["counts"]["total"]:
            verify_state = "active" if verification["counts"]["pending"] else "complete"
            verify_detail = f"유효 {verification['counts']['valid']}/{verification['counts']['total']}"
        elif verification_stage in {"SUBMIT", "WAIT", "INGEST", "CHECK"}:
            verify_state, verify_detail = "active", verification_stage
        else:
            verify_state, verify_detail = "waiting", "검증 전"
        stages.append({"key": "verification", "label": "후보 FEA", "state": verify_state, "detail": verify_detail})

        final_status = verification["final"]["status"]
        final_state = {"pass": "complete", "fail": "error"}.get(final_status, "waiting")
        final_detail = {"pass": "최종 설계 확정", "fail": "fine FEA 실패"}.get(final_status, "검증 전")
        stages.append({"key": "final", "label": "최종 설계", "state": final_state, "detail": final_detail})

        warnings: list[str] = []
        for payload in (data, models, nsga, verification):
            warnings.extend(payload.get("warnings", []))
        if not scheduler.get("connected") and scheduler.get("error"):
            warnings.append(scheduler["error"])
        if data["stalled"]:
            warnings.append(f"유효 데이터가 약 {data['stalled_minutes']:.0f}분 동안 증가하지 않았습니다.")
        if (
            models["missing_count"]
            and models.get("activation_state") != "preactivation_checkpoint"
        ):
            missing = [model["label"] for model in models["models"] if not model["trained"]]
            warnings.append("미학습 모델: " + ", ".join(missing))
        if final_status == "fail":
            warnings.append("최종 fine FEA가 사양을 통과하지 못했습니다.")
        warnings = list(dict.fromkeys(warnings))

        if final_status == "fail" or any(stage["state"] == "error" for stage in stages):
            overall = "error"
        elif warnings:
            overall = "warning"
        elif any(stage["state"] == "active" for stage in stages):
            overall = "active"
        else:
            overall = "idle"
        current = next((stage for stage in reversed(stages) if stage["state"] == "active"), None)
        if current is None:
            current = next((stage for stage in stages if stage["state"] in {"warning", "error"}), stages[0])
        return {
            "overall": overall,
            "current_stage": current["key"],
            "current_stage_label": current["label"],
            "stages": stages,
            "warnings": warnings,
        }

    def dashboard(self, record: bool = True) -> dict[str, Any]:
        generated_at = _iso(self.clock())
        data = self.data()
        models = self.models(data["total_rows"])
        nsga = self.nsga2()
        verification = self.verification(nsga)
        scheduler = self.scheduler.snapshot()
        refill_controller = self.refill_controller.snapshot()
        dashboard = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "project": "MFT_1MW_2026",
            "status": self._status(data, models, nsga, verification, scheduler),
            "data": data,
            "models": models,
            "nsga2": nsga,
            "verification": verification,
            "scheduler": scheduler,
            "refill_controller": refill_controller,
        }
        if record and self.recorder:
            try:
                self.recorder.record(dashboard)
            except (OSError, TypeError, ValueError) as exc:
                dashboard["status"]["warnings"].append(f"모니터 이력 기록 실패: {exc}")
                if dashboard["status"]["overall"] not in {"error"}:
                    dashboard["status"]["overall"] = "warning"
        return dashboard

    def status(self) -> dict[str, Any]:
        dashboard = self.dashboard(record=False)
        return {
            "schema_version": dashboard["schema_version"],
            "generated_at": dashboard["generated_at"],
            "project": dashboard["project"],
            **dashboard["status"],
            "scheduler": dashboard["scheduler"],
            "refill_controller": dashboard["refill_controller"],
        }

    def history(self) -> dict[str, Any]:
        if not self.recorder:
            return {"entries": [], "warning": None}
        return self.recorder.history()
