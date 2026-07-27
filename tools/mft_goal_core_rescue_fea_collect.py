"""Collect and attest the compact core-rescue symmetric FEA campaign.

The collector is deliberately read-only with respect to Scheduler.  It binds
every task to its sealed feeder plan and submission receipt, extracts the
single ``RESULT_JSON`` record from stdout, restores eighth-model inductances
to the full physical transformer with the campaign scale of 2.0, and writes a
deterministic CSV plus a sealed JSON evidence package.

Capacitance values emitted by the solver already have their independent
eighth-to-full energy restoration applied.  They must therefore *not* be
multiplied by the inductance factor in this collector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "mft-goal-core-rescue-fea-collection-v1"
PLAN_SCHEMA = "mft-goal-core-rescue-direct-fea-plan-v1"
RECEIPT_SCHEMA = "mft-goal-core-rescue-direct-fea-submission-v1"
NATIVE_EIGHTH_TO_FULL_INDUCTANCE_SCALE = 2.0
TARGET_LM_PRIMARY_MH = 2.0
TARGET_TURN_RATIO = 10.0
LM_RELATIVE_TOLERANCE = 0.01
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
FULL_MODE = "matrix_turngraded_cap_loss_thermal"
MATRIX_MODE = "matrix_only_lm_bracket"


class CollectionError(RuntimeError):
    """Raised when sealed campaign or result evidence is inconsistent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CollectionError(f"JSON object required: {path}")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str, path: Path) -> None:
    if value.get("schema_version") != schema:
        raise CollectionError(f"schema mismatch in {path}")
    expected = value.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise CollectionError(f"payload seal missing in {path}")
    payload = dict(value)
    payload.pop("payload_sha256", None)
    if _sha(payload) != expected:
        raise CollectionError(f"payload seal mismatch in {path}")


def _http_json(base_url: str, suffix: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{suffix}"
    try:
        with urllib.request.urlopen(url, timeout=30.0) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise CollectionError(f"Scheduler GET failed: {url}: {error}") from error
    if not isinstance(value, dict):
        raise CollectionError(f"Scheduler JSON object required: {url}")
    return value


def _http_text(base_url: str, suffix: str) -> str:
    url = f"{base_url.rstrip('/')}{suffix}"
    try:
        with urllib.request.urlopen(url, timeout=30.0) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, UnicodeDecodeError) as error:
        raise CollectionError(f"Scheduler GET failed: {url}: {error}") from error


def _result_json(stdout: str, task_id: int) -> dict[str, Any]:
    prefix = "RESULT_JSON "
    matches = [
        line[len(prefix) :]
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise CollectionError(
            f"task {task_id} must contain exactly one RESULT_JSON; "
            f"observed {len(matches)}"
        )
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise CollectionError(
            f"task {task_id} RESULT_JSON is malformed"
        ) from error
    if not isinstance(value, dict):
        raise CollectionError(f"task {task_id} RESULT_JSON must be an object")
    return value


def _close(actual: Any, expected: float, tolerance: float = 1e-9) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, expected, rel_tol=0.0, abs_tol=tolerance
    )


def _finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _empty_json_list(value: Any) -> bool:
    if value == []:
        return True
    if not isinstance(value, str):
        return False
    try:
        return json.loads(value) == []
    except json.JSONDecodeError:
        return False


def _candidate_table(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = row.get("candidate_sha256", "")
        if len(identity) != 64 or identity in result:
            raise CollectionError(f"candidate table identity is invalid: {path}")
        result[identity] = row
    return result


def _plan_bindings(
    plan_paths: Sequence[Path],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    bindings: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for raw_path in plan_paths:
        plan_path = raw_path.resolve(strict=True)
        plan = _load_json(plan_path)
        _validate_seal(plan, PLAN_SCHEMA, plan_path)
        receipt_path = plan_path.with_name("submission_receipt.json")
        receipt = _load_json(receipt_path)
        _validate_seal(receipt, RECEIPT_SCHEMA, receipt_path)
        if (
            receipt.get("plan_payload_sha256") != plan["payload_sha256"]
            or receipt.get("complete") is not True
            or receipt.get("scheduler_POST_performed") is not True
        ):
            raise CollectionError(f"submission receipt drifted: {receipt_path}")
        lanes = {
            int(lane["lane_index"]): lane
            for lane in plan.get("lanes", [])
            if isinstance(lane, dict)
        }
        for submission in receipt.get("submissions", []):
            task_id = int(submission["task_id"])
            lane_index = int(submission["lane_index"])
            lane = lanes.get(lane_index)
            if lane is None:
                raise CollectionError(
                    f"receipt lane {lane_index} absent from {plan_path}"
                )
            if task_id in bindings:
                raise CollectionError(f"duplicate task binding: {task_id}")
            if (
                submission.get("candidate_sha256")
                != lane.get("candidate_sha256")
                or submission.get("mode") != lane.get("mode")
                or submission.get("dedupe_key")
                != lane.get("scheduler", {}).get("dedupe_key")
            ):
                raise CollectionError(
                    f"task {task_id} receipt-to-plan binding drifted"
                )
            bindings[task_id] = {
                "plan": plan,
                "plan_path": plan_path,
                "receipt": receipt,
                "receipt_path": receipt_path,
                "lane": lane,
                "submission": submission,
            }
        sources.append(
            {
                "plan_path": str(plan_path),
                "plan_file_sha256": _file_sha(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "receipt_path": str(receipt_path),
                "receipt_file_sha256": _file_sha(receipt_path),
                "receipt_payload_sha256": receipt["payload_sha256"],
                "solver_revision": plan["solver_revision"],
                "library_revision": plan["library_revision"],
                "task_ids": sorted(
                    int(item["task_id"])
                    for item in receipt.get("submissions", [])
                ),
            }
        )
    return bindings, sources


def _assert_contract(
    *,
    task_id: int,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, bool]:
    lane = binding["lane"]
    plan = binding["plan"]
    expected_gap = float(lane["core_center_gap_mm"])
    expected_thermal = lane["mode"] == FULL_MODE
    checks = {
        "scheduler_succeeded": (
            task.get("state") == "succeeded"
            and task.get("status") == "completed"
            and int(task.get("exit_code")) == 0
        ),
        "solver_revision": (
            result.get("git_hash") == plan["solver_revision"]
            and int(result.get("git_dirty", -1)) == 0
        ),
        "library_revision": (
            result.get("pyaedt_library_git_hash")
            == plan["library_revision"]
            and int(result.get("pyaedt_library_git_dirty", -1)) == 0
        ),
        "scheduler_task_readback": (
            str(result.get("solver_core_scheduler_task_id_readback"))
            == str(task_id)
        ),
        "solver_8cpu_contract": (
            int(result.get("solver_num_cores_requested", -1)) == 8
            and int(result.get("solver_num_cores_effective", -1)) == 8
            and str(result.get("solver_core_slurm_cpus_per_task_readback"))
            == "8"
            and result.get("solver_core_backend") == "standalone"
        ),
        "nonrounded_eighth_model": (
            int(result.get("round_corner", -1)) == 0
            and int(result.get("full_model", -1)) == 0
            and result.get("thermal_symmetry") == "eighth"
        ),
        "equal_winding_heights": _close(
            result.get("nwh1"), float(result.get("nwh2")), 1e-8
        ),
        "equal_three_leg_gap_input": (
            int(result.get("core_equal_three_leg_air_gap", -1)) == 1
            and int(result.get("core_air_gap_gapped_leg_count", -1)) == 3
            and result.get("core_air_gap_topology")
            == "equal_center_and_both_side_legs"
        ),
        "equal_three_leg_gap_geometry": (
            int(
                result.get(
                    "core_equal_three_leg_air_gap_geometry_attested", -1
                )
            )
            == 1
            and int(
                result.get(
                    "core_air_gap_identical_all_gapped_legs_attested", -1
                )
            )
            == 1
            and int(result.get("core_center_gap_geometry_attested", -1)) == 1
            and int(
                result.get(
                    "core_equal_three_leg_air_gap_symmetry_geometry_attested",
                    -1,
                )
            )
            == 1
            and _close(
                result.get("core_center_gap_readback_mm"),
                expected_gap,
                1e-8,
            )
            and _close(
                result.get("core_center_gap_symmetry_half_gap_readback_mm"),
                expected_gap,
                1e-8,
            )
            and _close(
                result.get("core_center_gap_group_spread_mm"), 0.0, 1e-8
            )
        ),
        "fixed_cooling_geometry": (
            int(result.get("core_plate_on", -1)) == 1
            and _close(result.get("core_plate_t"), 20.0)
            and _close(result.get("core_plate_pad_t"), 2.0)
            and int(result.get("wcp_on", -1)) == 1
            and _close(result.get("wcp_t"), 20.0)
            and _close(result.get("wcp_pad_t"), 2.0)
        ),
        "fixed_boundary_contract": (
            int(result.get("fixed_boundary_authoritative_attested", -1)) == 1
            and result.get("fixed_boundary_authority_class")
            == "authoritative_fixed_boundary"
            and _close(result.get("fixed_boundary_fan_velocity_m_s"), 1.5)
            and _close(result.get("fixed_boundary_core_plate_pad_t_mm"), 2.0)
            and _close(result.get("fixed_boundary_wcp_pad_t_mm"), 2.0)
            and _close(
                result.get(
                    "fixed_boundary_thermal_pad_conductivity_W_mK"
                ),
                0.2,
            )
            and _empty_json_list(
                result.get("fixed_boundary_mismatches_json")
            )
        ),
        "em_valid": (
            int(result.get("result_valid_em", -1)) == 1
            and result.get("em_validity_reason") == "valid"
        ),
        "thermal_valid_when_requested": (
            not expected_thermal
            or int(result.get("result_valid_thermal", -1)) == 1
        ),
        "turn_graded_rx_when_requested": (
            not expected_thermal
            or (
                result.get("cap_turn_graded_active_winding") == "Rx"
                and result.get("active_winding") == "Rx"
                and result.get("cap_turn_graded_schema_version")
                == "mft-turn-graded-energy-capacitance-v2"
                and result.get("output_basis") == "full_physical"
                and _close(result.get("capacitance_restoration_factor"), 8.0)
                and _close(
                    result.get(
                        "cap_turn_graded_inductance_restoration_factor"
                    ),
                    NATIVE_EIGHTH_TO_FULL_INDUCTANCE_SCALE,
                )
            )
        ),
    }
    return checks


def _result_dimensions(
    result: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    """Reproduce the sealed optimizer exterior-box formula from decoded FEA."""
    l1 = float(result["l1"])
    l2 = float(result["l2"])
    h1 = float(result["h1"])
    w1 = float(result["w1"])
    core_x = 4.0 * l1 + 2.0 * l2
    center_x = (
        float(result["sl1_main_x"]) + 2.0 * float(result["nwl1_main"])
    )
    center_y = (
        float(result["sl1_main_y"]) + 2.0 * float(result["nwb1_main_y"])
    )
    x_candidates = [core_x, center_x]
    y_candidates = [w1, center_y]
    if int(result["N2_side"]) > 0:
        offset = l1 + l2 + l1 / 2.0
        side_x_out = (
            offset
            + float(result["sl2_side_x"]) / 2.0
            + float(result["nwl2_side"])
        )
        side_y_out = (
            float(result["sl2_side_y"]) / 2.0
            + float(result["nwl2_side"])
        )
        x_candidates.append(2.0 * side_x_out)
        y_candidates.append(2.0 * side_y_out)
    width = max(x_candidates)
    length = max(y_candidates)
    height = h1 + 2.0 * l1
    return width, length, height, width * length * height * 1e-6


def _make_record(
    *,
    task_id: int,
    task: Mapping[str, Any],
    stdout: str,
    result: Mapping[str, Any],
    binding: Mapping[str, Any],
    candidate: Mapping[str, str] | None,
) -> dict[str, Any]:
    lane = binding["lane"]
    candidate_sha = str(lane["candidate_sha256"])
    checks = _assert_contract(
        task_id=task_id, task=task, result=result, binding=binding
    )
    contract_valid = all(checks.values())
    scale = NATIVE_EIGHTH_TO_FULL_INDUCTANCE_SCALE
    lmt_native = _finite(result.get("Lmt"))
    lmr_native = _finite(result.get("Lmr"))
    lm_primary_mh = None if lmt_native is None else lmt_native * scale / 1000.0
    lm_secondary_mh = (
        None if lmr_native is None else lmr_native * scale / 1000.0
    )
    lm_error = (
        None
        if lm_primary_mh is None
        else abs(lm_primary_mh - TARGET_LM_PRIMARY_MH)
        / TARGET_LM_PRIMARY_MH
    )
    lm_ratio_error = (
        None
        if lmt_native in (None, 0.0) or lmr_native is None
        else abs(
            lmr_native / lmt_native - TARGET_TURN_RATIO * TARGET_TURN_RATIO
        )
        / (TARGET_TURN_RATIO * TARGET_TURN_RATIO)
    )
    w_mm, length_mm, h_mm, volume_l = _result_dimensions(result)
    candidate_dimension_match: bool | None = None
    if candidate is not None:
        expected_dimensions = (
            _finite(candidate.get("W_mm")),
            _finite(candidate.get("L_mm")),
            _finite(candidate.get("H_mm")),
            _finite(candidate.get("volume_L")),
        )
        candidate_dimension_match = all(
            expected is not None and _close(actual, expected, 1e-7)
            for actual, expected in zip(
                (w_mm, length_mm, h_mm, volume_l), expected_dimensions
            )
        )
    geometry_pass = (
        w_mm is not None
        and length_mm is not None
        and h_mm is not None
        and w_mm <= 1200.0
        and length_mm <= 900.0
        and h_mm <= 750.0
    )
    tx_temp = _finite(result.get("T_max_Tx"))
    rx_main_temp = _finite(result.get("T_max_Rx_main"))
    rx_side_temp = _finite(result.get("T_max_Rx_side"))
    core_temp = _finite(result.get("T_max_core"))
    rx_temp = (
        None
        if rx_main_temp is None
        else max(
            rx_main_temp,
            rx_side_temp if rx_side_temp is not None else rx_main_temp,
        )
    )
    thermal_pass = (
        tx_temp is not None
        and rx_temp is not None
        and core_temp is not None
        and tx_temp <= 110.0
        and rx_temp <= 130.0
        and core_temp <= 130.0
    )
    f_tx = _finite(result.get("f_res_tx_self_Hz"))
    f_rx_turn_graded = _finite(result.get("f_res_rx_turn_graded_Hz"))
    resonance_pass = (
        f_tx is not None
        and f_rx_turn_graded is not None
        and f_tx >= 15000.0
        and f_rx_turn_graded >= 15000.0
    )
    exact_lm_pass = lm_error is not None and lm_error <= LM_RELATIVE_TOLERANCE
    full_physics = lane["mode"] == FULL_MODE
    exact_gap_task = bool(
        binding["plan"].get("regression_contract", {}).get(
            "full_direct_physics_at_interpolated_gap"
        )
    )
    feasible = (
        full_physics
        and contract_valid
        and geometry_pass
        and thermal_pass
        and resonance_pass
        and exact_lm_pass
    )
    fields = {
        "task_id": task_id,
        "name": task.get("name"),
        "mode": lane["mode"],
        "candidate_sha256": candidate_sha,
        "candidate_short": candidate_sha[:10],
        "candidate_index": lane.get("candidate_index"),
        "gap_mm": _finite(result.get("core_center_gap_mm")),
        "N1": result.get("N1"),
        "N2": result.get("N2"),
        "N2_main": result.get("N2_main"),
        "N2_side": result.get("N2_side"),
        "n_core_group": result.get("n_core_group"),
        "core_depth_each_mm": _finite(result.get("core_depth_each")),
        "W_mm": w_mm,
        "L_mm": length_mm,
        "H_mm": h_mm,
        "volume_L": volume_l,
        "candidate_table_dimension_match": candidate_dimension_match,
        "exact_gap_task": exact_gap_task,
        "Ltx_full_uH": (
            None
            if _finite(result.get("Ltx")) is None
            else float(result["Ltx"]) * scale
        ),
        "Lrx_full_uH": (
            None
            if _finite(result.get("Lrx")) is None
            else float(result["Lrx"]) * scale
        ),
        "M_full_uH": (
            None
            if _finite(result.get("M")) is None
            else float(result["M"]) * scale
        ),
        "Lm_primary_full_mH": lm_primary_mh,
        "Lm_secondary_full_mH": lm_secondary_mh,
        "Lm_primary_relative_error": lm_error,
        "Lm_turn_ratio_squared_relative_error": lm_ratio_error,
        "Llt_full_uH": (
            None
            if _finite(result.get("Llt")) is None
            else float(result["Llt"]) * scale
        ),
        "Llr_full_uH": (
            None
            if _finite(result.get("Llr")) is None
            else float(result["Llr"]) * scale
        ),
        "k": _finite(result.get("k")),
        "C_rx_turn_graded_full_F": _finite(
            result.get("C_rx_rx_turn_graded_F")
        ),
        "f_rx_turn_graded_Hz": f_rx_turn_graded,
        "C_tx_grounded_other_full_F": _finite(result.get("C_tx_tx_F")),
        "f_tx_grounded_other_Hz": f_tx,
        "C_rx_grounded_other_full_F": _finite(result.get("C_rx_rx_F")),
        "f_rx_grounded_other_Hz": _finite(result.get("f_res_rx_self_Hz")),
        "C_interwinding_full_F": _finite(result.get("C_tx_rx_F")),
        "f_interwinding_Hz": _finite(
            result.get("f_res_interwinding_Hz")
        ),
        "Tx_loss_W": _finite(result.get("Tx_loss")),
        "Rx_loss_W": _finite(result.get("Rx_loss")),
        "core_loss_W": _finite(result.get("P_core_total")),
        "T_max_Tx_C": tx_temp,
        "T_max_Rx_main_C": rx_main_temp,
        "T_max_Rx_side_C": rx_side_temp,
        "T_max_Rx_C": rx_temp,
        "T_max_core_C": core_temp,
        "geometry_pass": geometry_pass,
        "thermal_pass": thermal_pass,
        "resonance_pass": resonance_pass,
        "exact_Lm_pass": exact_lm_pass,
        "contract_valid": contract_valid,
        "feasible": feasible,
        "solver_revision": result.get("git_hash"),
        "library_revision": result.get("pyaedt_library_git_hash"),
        "scheduler_state": task.get("state"),
        "finished_at": task.get("finished_at"),
        "result_json_sha256": _sha(result),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
    }
    return {
        **fields,
        "contract_checks": checks,
        "contract_failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
    }


CSV_FIELDS = [
    "rank",
    "task_id",
    "name",
    "mode",
    "candidate_sha256",
    "candidate_short",
    "candidate_index",
    "gap_mm",
    "N1",
    "N2",
    "N2_main",
    "N2_side",
    "n_core_group",
    "core_depth_each_mm",
    "W_mm",
    "L_mm",
    "H_mm",
    "volume_L",
    "candidate_table_dimension_match",
    "exact_gap_task",
    "Ltx_full_uH",
    "Lrx_full_uH",
    "M_full_uH",
    "Lm_primary_full_mH",
    "Lm_secondary_full_mH",
    "Lm_primary_relative_error",
    "Lm_turn_ratio_squared_relative_error",
    "Llt_full_uH",
    "Llr_full_uH",
    "k",
    "C_rx_turn_graded_full_F",
    "f_rx_turn_graded_Hz",
    "C_tx_grounded_other_full_F",
    "f_tx_grounded_other_Hz",
    "C_rx_grounded_other_full_F",
    "f_rx_grounded_other_Hz",
    "C_interwinding_full_F",
    "f_interwinding_Hz",
    "Tx_loss_W",
    "Rx_loss_W",
    "core_loss_W",
    "T_max_Tx_C",
    "T_max_Rx_main_C",
    "T_max_Rx_side_C",
    "T_max_Rx_C",
    "T_max_core_C",
    "geometry_pass",
    "thermal_pass",
    "resonance_pass",
    "exact_Lm_pass",
    "contract_valid",
    "feasible",
    "solver_revision",
    "library_revision",
    "scheduler_state",
    "finished_at",
    "result_json_sha256",
    "stdout_sha256",
    "contract_failed_checks",
]


def _ranking_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    full_mode = record.get("mode") == FULL_MODE
    volume = record.get("volume_L")
    lm_error = record.get("Lm_primary_relative_error")
    max_temp = max(
        [
            float(value)
            for value in (
                record.get("T_max_Tx_C"),
                record.get("T_max_Rx_C"),
                record.get("T_max_core_C"),
            )
            if value is not None
        ]
        or [math.inf]
    )
    return (
        not bool(record.get("feasible")),
        not full_mode,
        not bool(record.get("contract_valid")),
        math.inf if lm_error is None else float(lm_error),
        math.inf if volume is None else float(volume),
        max_temp,
        int(record["task_id"]),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _csv_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for rank, record in enumerate(records, 1):
        row = dict(record)
        row["rank"] = rank
        row["contract_failed_checks"] = json.dumps(
            record.get("contract_failed_checks", []),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def collect(
    *,
    scheduler_base_url: str,
    plan_paths: Sequence[Path],
    candidate_csv: Path | None,
    diagnostic_task_ids: Sequence[int],
    output_dir: Path,
    allow_nonterminal: bool,
    allow_result_json_failures: bool,
) -> tuple[Path, Path]:
    bindings, sources = _plan_bindings(plan_paths)
    candidates = _candidate_table(candidate_csv)
    records: list[dict[str, Any]] = []
    task_evidence: list[dict[str, Any]] = []
    for task_id in sorted(bindings):
        task = _http_json(scheduler_base_url, f"/api/tasks/{task_id}")
        state = str(task.get("state"))
        if state not in TERMINAL_STATES:
            if allow_nonterminal:
                task_evidence.append(
                    {
                        "task_id": task_id,
                        "name": task.get("name"),
                        "state": state,
                        "terminal": False,
                    }
                )
                continue
            raise CollectionError(f"task {task_id} is nonterminal: {state}")
        if state != "succeeded" and not allow_result_json_failures:
            raise CollectionError(
                f"authoritative task {task_id} did not succeed: "
                f"{task.get('failure_message')}"
            )
        stdout = _http_text(scheduler_base_url, f"/api/tasks/{task_id}/stdout")
        result = _result_json(stdout, task_id)
        candidate_sha = str(bindings[task_id]["lane"]["candidate_sha256"])
        record = _make_record(
            task_id=task_id,
            task=task,
            stdout=stdout,
            result=result,
            binding=bindings[task_id],
            candidate=candidates.get(candidate_sha),
        )
        records.append(record)
        task_evidence.append(
            {
                "task_id": task_id,
                "name": task.get("name"),
                "state": state,
                "terminal": True,
                "task_snapshot_sha256": _sha(task),
                "result_json_sha256": record["result_json_sha256"],
                "stdout_sha256": record["stdout_sha256"],
            }
        )
    diagnostics: list[dict[str, Any]] = []
    for task_id in sorted(set(diagnostic_task_ids)):
        task = _http_json(scheduler_base_url, f"/api/tasks/{task_id}")
        diagnostics.append(
            {
                "task_id": task_id,
                "name": task.get("name"),
                "state": task.get("state"),
                "status": task.get("status"),
                "exit_code": task.get("exit_code"),
                "failure_message": task.get("failure_message"),
                "terminal": task.get("state") in TERMINAL_STATES,
                "task_snapshot_sha256": _sha(task),
            }
        )
    ranked = sorted(records, key=_ranking_key)
    csv_payload = _csv_bytes(ranked)
    csv_path = output_dir / "core_rescue_fea_results.csv"
    _atomic_bytes(csv_path, csv_payload)
    full_records = [row for row in ranked if row["mode"] == FULL_MODE]
    exact_records = [
        row
        for row in full_records
        if row["exact_gap_task"]
    ]
    package: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scheduler_base_url": scheduler_base_url.rstrip("/"),
        "read_only_scheduler_collection": True,
        "native_eighth_to_full_inductance_scale": (
            NATIVE_EIGHTH_TO_FULL_INDUCTANCE_SCALE
        ),
        "capacitance_values_already_full_physical": True,
        "target_contract": {
            "W_max_mm": 1200.0,
            "L_max_mm": 900.0,
            "H_max_mm": 750.0,
            "T_max_Tx_C": 110.0,
            "T_max_Rx_C": 130.0,
            "T_max_core_C": 130.0,
            "f_res_tx_min_Hz": 15000.0,
            "f_res_rx_min_Hz": 15000.0,
            "Lm_primary_target_mH": TARGET_LM_PRIMARY_MH,
            "Lm_relative_tolerance": LM_RELATIVE_TOLERANCE,
            "turn_ratio": TARGET_TURN_RATIO,
            "equal_winding_heights": True,
            "equal_physical_gap_center_and_both_side_legs": True,
            "core_cold_plate_t_mm": 20.0,
            "winding_cold_plate_t_mm": 20.0,
            "fan_velocity_m_s": 1.5,
        },
        "source_evidence": sources,
        "candidate_csv": (
            None
            if candidate_csv is None
            else {
                "path": str(candidate_csv.resolve(strict=True)),
                "sha256": _file_sha(candidate_csv.resolve(strict=True)),
            }
        ),
        "expected_authoritative_task_count": len(bindings),
        "collected_authoritative_task_count": len(records),
        "terminal_task_evidence": task_evidence,
        "diagnostic_excluded_tasks": diagnostics,
        "matrix_record_count": sum(
            row["mode"] == MATRIX_MODE for row in ranked
        ),
        "full_physics_record_count": len(full_records),
        "exact_gap_full_physics_record_count": len(exact_records),
        "contract_valid_full_physics_count": sum(
            bool(row["contract_valid"]) for row in full_records
        ),
        "exact_Lm_within_1pct_count": sum(
            bool(row["exact_Lm_pass"]) for row in exact_records
        ),
        "feasible_count": sum(bool(row["feasible"]) for row in ranked),
        "ranked_records": ranked,
        "csv": {
            "path": str(csv_path),
            "sha256": hashlib.sha256(csv_payload).hexdigest(),
            "row_count": len(ranked),
        },
        "complete": len(records) == len(bindings),
    }
    package["payload_sha256"] = _sha(package)
    json_path = output_dir / "collection.json"
    _atomic_bytes(json_path, json.dumps(
        package,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n")
    return json_path, csv_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler-base-url", default="http://127.0.0.1:8002"
    )
    parser.add_argument(
        "--plan", type=Path, action="append", required=True
    )
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument(
        "--diagnostic-task-id", type=int, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nonterminal", action="store_true")
    parser.add_argument(
        "--allow-result-json-failures",
        action="store_true",
        help=(
            "harvest failed tasks only when stdout still contains one "
            "RESULT_JSON; contract/thermal gates remain failed"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    json_path, csv_path = collect(
        scheduler_base_url=args.scheduler_base_url,
        plan_paths=args.plan,
        candidate_csv=args.candidate_csv,
        diagnostic_task_ids=args.diagnostic_task_id,
        output_dir=args.output_dir,
        allow_nonterminal=args.allow_nonterminal,
        allow_result_json_failures=args.allow_result_json_failures,
    )
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
