"""Watch and collect the targeted fixed-5T/fixed-Lm2mH NSGA-II campaign.

The final Pareto artifacts are written only after all 16 authenticated tasks
finish successfully and exactly 5,120 terminal rows have been collected.
Thermal and resonance results remain screening-only; this collector never
promotes a candidate to production truth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping
import urllib.parse
import urllib.request

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import slurm_nsga_offload as transport
from tools.mft_goal_fixed_primary_5t_collect import (
    CORE_TARGETS,
    WINDING_TARGETS,
    _atomic_json,
    _read_json,
    _sha,
    _sha_file,
    _truth,
    _validate_seal,
    _write_csv,
)


SCHEMA = "mft-goal-fixed-lm2mh-targeted-global-nds-v1"
PARETO_SCHEMA = "mft-goal-fixed-lm2mh-targeted-pareto-manifest-v1"
SMOKE_SCHEMA = "mft-goal-fixed-lm2mh-official5-projected-smoke-v1"
FIRST_TERMINAL_SMOKE_SCHEMA = (
    "mft-goal-fixed-lm2mh-first-terminal-smoke-v1"
)
CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-targeted-v1"
)
RESULT_SCHEMA = "mft-goal-fixed-primary-5t-nsga-result-v1"
SUBMISSION_SCHEMA = "mft-goal-fixed-primary-5t-nsga-submission-v1"
FIXED_TERMINAL_SCHEMA = "mft-goal-fixed-lm2mh-terminal-candidates-v1"
PREOPT_SCHEMA = "mft-goal-fixed-primary-5t-pre-optimization-v1"
INITIALIZATION_SCHEMA = "mft-goal-fixed-lm2mh-targeted-initialization-v1"
EXPECTED_HARD_SPEC_SHA256 = (
    "227cdc0db3b8dae490275d549e8aea93b295a75ee97ea98cdaa601bb96591298"
)
EXPECTED_RUNNER_SOURCE_SHA256 = (
    "d82458cbc9793b199163653e0e74df99226d82d563f87811a13ec9f0cddd8673"
)
EXPECTED_RESONANCE_CONTRACT_SHA256 = (
    "9858b78f3084d344077fe0d90f7e390f7485d18248d1d711d2fde22c36c01ed1"
)
EXPECTED_TASK_IDS = tuple(range(96416, 96432))
TASK_PREFIX = "mft-5t-lm2-"
REMOTE_BUNDLE = (
    "/gpfs/tmp_cpu2/mft_goal_20260726/"
    "mft-goal-763dbb46ae74e1722969d2c2"
)
DEFAULT_MANIFEST = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_targeted_w1200_l1000_v1"
    r"\submission_manifest.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_targeted_w1200_l1000_v1_global_nds"
)
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
SCHEDULER_URL = "http://127.0.0.1:8002"
TERMINAL_ROWS_PER_TASK = 320
EXPECTED_RAW_ROWS = 16 * TERMINAL_ROWS_PER_TASK
FINAL_FILES = (
    "global_terminal_all_seeds_raw.csv",
    "global_terminal_candidates.csv",
    "global_pareto_front.csv",
    "global_conditional_nonthermal_pareto_front.csv",
    "global_minimum_violation_objective_front.csv",
    "fea_acquisition_candidates.csv",
)


def _api_any(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _submission(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _validate_seal(
        _read_json(path.resolve(strict=True)), SUBMISSION_SCHEMA
    )
    entries = list(manifest.get("submissions") or [])
    task_ids = tuple(sorted(int(item["task_id"]) for item in entries))
    seeds = tuple(sorted(int(item["seed"]) for item in entries))
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("hard_spec_sha256") != EXPECTED_HARD_SPEC_SHA256
        or manifest.get("runner_source_sha256")
        != EXPECTED_RUNNER_SOURCE_SHA256
        or manifest.get("population") != TERMINAL_ROWS_PER_TASK
        or manifest.get("generations") != 80
        or manifest.get("campaign_total_seed_count") != 16
        or manifest.get("seed_count") != 16
        or task_ids != EXPECTED_TASK_IDS
        or len(set(seeds)) != 16
    ):
        raise RuntimeError("targeted fixed-Lm submission coverage drifted")
    hard_spec = manifest.get("hard_spec") or {}
    if (
        hard_spec.get("primary_conductor_thickness_mm") != 5.0
        or hard_spec.get("primary_interturn_gap_mm") != 1.6
        or hard_spec.get("magnetizing_inductance_H") != 0.002
        or (hard_spec.get("size_limits_mm") or {})
        != {"H": 750.0, "L": 1000.0, "W": 1200.0}
        or (
            hard_spec.get("axis_contract") or {}
        ).get("rotation_or_axis_swap_allowed")
        is not False
        or (
            hard_spec.get("resonance_contract") or {}
        ).get("classification")
        != "screening-only"
    ):
        raise RuntimeError("targeted fixed-Lm hard-spec content drifted")
    entries.sort(key=lambda item: int(item["task_id"]))
    return entries, manifest


def _status_inventory(
    entries: Iterable[Mapping[str, Any]], scheduler_url: str
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "compact": "true",
            "limit": "100",
            "name_prefix": TASK_PREFIX,
        }
    )
    payload = _api_any(
        scheduler_url.rstrip("/") + "/api/tasks?" + query
    )
    tasks = payload if isinstance(payload, list) else payload.get("tasks")
    if not isinstance(tasks, list):
        raise RuntimeError("Scheduler compact task inventory is invalid")
    by_id = {
        int(task["id"]): task
        for task in tasks
        if isinstance(task, dict)
        and int(task.get("id") or -1) in EXPECTED_TASK_IDS
    }
    statuses = []
    for entry in entries:
        task_id = int(entry["task_id"])
        task = by_id.get(task_id)
        if task is None or task.get("name") != entry["name"]:
            raise RuntimeError(f"Scheduler task {task_id} is missing/drifted")
        statuses.append(task)
    return statuses


def _status_detail(
    entry: Mapping[str, Any], scheduler_url: str
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    url = (
        scheduler_url.rstrip("/")
        + f"/api/tasks/{task_id}?include_output=true&output_limit=30000"
    )
    value = _api_any(url)
    if not isinstance(value, dict):
        raise RuntimeError(f"Scheduler task {task_id} detail is invalid")
    if (
        int(value.get("id", value.get("task_id", -1))) != task_id
        or value.get("name") != entry["name"]
        or int(value.get("cpus") or 0) != 8
        or int(value.get("memory_mb") or 0) != 65_536
        or value.get("remote_cwd") != REMOTE_BUNDLE
    ):
        raise RuntimeError(f"Scheduler task {task_id} identity drifted")
    return value


def _terminal_details(
    entries: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    scheduler_url: str,
) -> dict[int, dict[str, Any]]:
    terminal = {
        int(status["id"])
        for status in statuses
        if status.get("status") in {"completed", "failed", "cancelled"}
    }
    selected = [
        entry for entry in entries if int(entry["task_id"]) in terminal
    ]
    if not selected:
        return {}
    workers = min(8, len(selected))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(
            executor.map(
                lambda entry: _status_detail(entry, scheduler_url),
                selected,
            )
        )
    return {int(value["id"]): value for value in values}


def _cached_or_download(
    connection: Any,
    *,
    remote: str,
    destination: Path,
    expected: Mapping[str, Any] | None,
    max_bytes: int,
) -> dict[str, Any]:
    normalized = None
    if expected is not None:
        normalized = {
            "sha256": str(expected["sha256"]),
            "bytes": int(
                expected.get("size_bytes", expected.get("bytes"))
            ),
        }
    if destination.is_file():
        observed = {
            "sha256": _sha_file(destination),
            "bytes": destination.stat().st_size,
        }
        if normalized is None or observed == normalized:
            return {
                **observed,
                "local_sha256": observed["sha256"],
                "remote_sha256": (
                    observed["sha256"]
                    if normalized is None
                    else normalized["sha256"]
                ),
                "remote_bytes": (
                    observed["bytes"]
                    if normalized is None
                    else normalized["bytes"]
                ),
                "verified": True,
                "transport": "authenticated_local_cache",
                "attempts": 0,
            }
        raise RuntimeError(f"cached artifact identity drifted: {destination}")
    return transport._download_verified_sftp(
        connection,
        remote,
        destination,
        retries=3,
        expected_identity=normalized,
        max_bytes=max_bytes,
    )


def _validate_result(
    entry: Mapping[str, Any],
    detail: Mapping[str, Any],
    *,
    authoritative_file_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    result = (
        dict(authoritative_file_result)
        if authoritative_file_result is not None
        else _result_from_detail(detail)
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"task {task_id} has no Scheduler RESULT_JSON")
    result = _validate_seal(result, RESULT_SCHEMA)
    if (
        result.get("campaign_id") != CAMPAIGN_ID
        or result.get("lane") != "fixed-lm-targeted"
        or result.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or result.get("physics_sha256") != entry["physics_sha256"]
        or result.get("hard_spec_sha256")
        != EXPECTED_HARD_SPEC_SHA256
        or result.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or result.get("terminal_population_count")
        != TERMINAL_ROWS_PER_TASK
        or result.get("terminal_primary_controls_attested") is not True
        or result.get("terminal_gap1_hard_fixed") is not True
        or result.get("terminal_primary_control_escape_count") != 0
        or result.get("global_nds_ready") is not True
        or result.get("screening_only") is not True
        or result.get("production_eligible") is not False
        or result.get(
            "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current"
        )
        is not False
    ):
        raise RuntimeError(f"task {task_id} result identity drifted")
    return result


def _result_from_detail(detail: Mapping[str, Any]) -> dict[str, Any] | None:
    """Use Scheduler RESULT_JSON, with its stdout copy as a sealed fallback."""

    direct = detail.get("result_json")
    if isinstance(direct, dict):
        return direct
    stdout = detail.get("stdout")
    if isinstance(stdout, list):
        stdout = "\n".join(str(value) for value in stdout)
    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith("RESULT_JSON="):
            try:
                value = json.loads(line[len("RESULT_JSON=") :])
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, dict) else None
    return None


def _download_task(
    *,
    connection: Any,
    entry: Mapping[str, Any],
    detail: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    task_root = output / f"task-{task_id}"
    remote_root = (
        f"{REMOTE_BUNDLE}/runs/fixed-lm2mh-targeted-task-{task_id}"
    )
    records = {
        "result.json": _cached_or_download(
            connection,
            remote=f"{remote_root}/result.json",
            destination=task_root / "result.json",
            expected=None,
            max_bytes=16 * 1024 * 1024,
        )
    }
    local_result = _validate_seal(
        _read_json(task_root / "result.json"), RESULT_SCHEMA
    )
    scheduler_result = _result_from_detail(detail)
    if scheduler_result is not None and local_result != scheduler_result:
        raise RuntimeError(f"task {task_id} RESULT_JSON/file mismatch")
    result = _validate_result(
        entry, detail, authoritative_file_result=local_result
    )
    inventory = result.get("artifact_inventory") or {}
    names = (
        "terminal_fixed_lm2mh_candidates.csv",
        "terminal_fixed_lm2mh_candidates.manifest.json",
    )
    if any(name not in inventory for name in names):
        raise RuntimeError(f"task {task_id} fixed-Lm inventory is incomplete")
    for name in names:
        records[name] = _cached_or_download(
            connection,
            remote=f"{remote_root}/{name}",
            destination=task_root / name,
            expected=inventory[name],
            max_bytes=128 * 1024 * 1024,
        )
    fixed_manifest = _validate_seal(
        _read_json(
            task_root
            / "terminal_fixed_lm2mh_candidates.manifest.json"
        ),
        FIXED_TERMINAL_SCHEMA,
    )
    csv_path = task_root / "terminal_fixed_lm2mh_candidates.csv"
    if (
        fixed_manifest.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or fixed_manifest.get("row_count") != TERMINAL_ROWS_PER_TASK
        or fixed_manifest.get("resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or fixed_manifest.get("csv_sha256") != _sha_file(csv_path)
        or fixed_manifest.get("thermal_screening_only") is not True
        or fixed_manifest.get("production_eligible") is not False
    ):
        raise RuntimeError(f"task {task_id} fixed terminal drifted")
    return {
        "task_id": task_id,
        "seed": int(entry["seed"]),
        "fixed_primary_turns": int(entry["fixed_primary_turns"]),
        "result_payload_sha256": result["payload_sha256"],
        "fixed_terminal_manifest_payload_sha256": fixed_manifest[
            "payload_sha256"
        ],
        "screening_feasible_count": int(
            fixed_manifest["screening_feasible_count"]
        ),
        "csv": str(csv_path),
        "csv_sha256": _sha_file(csv_path),
        "download_records": records,
    }


def _official5_smoke(
    *,
    connection: Any,
    entry: Mapping[str, Any],
    status: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    local = output / "smoke" / f"task-{task_id}" / "pre_optimization.json"
    remote = (
        f"{REMOTE_BUNDLE}/runs/fixed-lm2mh-targeted-task-{task_id}/"
        "pre_optimization.json"
    )
    try:
        record = _cached_or_download(
            connection,
            remote=remote,
            destination=local,
            expected=None,
            max_bytes=16 * 1024 * 1024,
        )
    except Exception as exc:
        if status.get("status") not in {"completed", "failed", "cancelled"}:
            return {
                "verified": False,
                "state": "worker_preoptimization_pending",
                "task_id": task_id,
                "detail": f"{type(exc).__name__}:{exc}",
            }
        raise
    preopt = _validate_seal(_read_json(local), PREOPT_SCHEMA)
    initialization = _validate_seal(
        dict(preopt.get("initialization") or {}), INITIALIZATION_SCHEMA
    )
    smoke = initialization.get("projected_anchor_smoke") or {}
    physical_g = smoke.get("physical_G") or {}
    required_g = (
        "exterior_width_limit",
        "exterior_length_limit",
        "exterior_height_limit",
        "half_magnetizing_resonance_minimum",
    )
    if (
        preopt.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or preopt.get("hard_spec_sha256")
        != EXPECTED_HARD_SPEC_SHA256
        or preopt.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or preopt.get("screening_only") is not True
        or (preopt.get("axis_contract") or {}).get(
            "rotation_or_axis_swap_allowed"
        )
        is not False
        or initialization.get("official5_projected_anchor_included")
        is not True
        or initialization.get("official5_projected_anchor_first_row")
        is not True
        or float(smoke.get("cw1_mm", math.nan)) != 5.0
        or float(smoke.get("gap1_mm", math.nan)) != 1.6
        or int(smoke.get("N1", -1)) != 6
        or any(name not in physical_g for name in required_g)
    ):
        raise RuntimeError("official#5 projected worker smoke drifted")
    observed = {
        "W_drawing_x_mm": 1200.0
        + float(physical_g["exterior_width_limit"]),
        "L_perpendicular_y_mm": 1000.0
        + float(physical_g["exterior_length_limit"]),
        "H_mm": 750.0 + float(physical_g["exterior_height_limit"]),
        "resonance_min_Hz_fixed_lm2mh": 15_000.0
        - float(physical_g["half_magnetizing_resonance_minimum"]),
    }
    if any(not math.isfinite(value) for value in observed.values()):
        raise RuntimeError("official#5 projected smoke is non-finite")
    result = {
        "schema_version": SMOKE_SCHEMA,
        "verified": True,
        "state": "verified_worker_preoptimization",
        "task_id": task_id,
        "seed": int(entry["seed"]),
        "official5_projected_anchor": True,
        "cw1_mm": 5.0,
        "gap1_mm": 1.6,
        "N1": 6,
        "axis_contract": {
            "W": "drawing_x_original_973mm_direction",
            "L": "drawing_y_perpendicular_direction",
            "rotation_or_axis_swap_allowed": False,
        },
        **observed,
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "hard_spec_sha256": EXPECTED_HARD_SPEC_SHA256,
        "pre_optimization_sha256": _sha_file(local),
        "download_record": record,
        "screening_only": True,
        "production_eligible": False,
    }
    result["payload_sha256"] = _sha(result)
    _atomic_json(output / "official5_projected_smoke.json", result)
    return result


def _temperatures(
    physical_g: Mapping[str, Any],
) -> tuple[float, float]:
    winding = [
        100.0 + float(physical_g[f"temperature_robust_limit:{name}"])
        for name in WINDING_TARGETS
        if f"temperature_robust_limit:{name}" in physical_g
    ]
    core = [
        120.0 + float(physical_g[f"temperature_robust_limit:{name}"])
        for name in CORE_TARGETS
        if f"temperature_robust_limit:{name}" in physical_g
    ]
    if not winding or not core:
        raise RuntimeError("terminal thermal constraints are incomplete")
    return max(winding), max(core)


def _load_rows(
    collections: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = []
    deduplicated: dict[str, dict[str, Any]] = {}
    for collection in collections:
        task_id = int(collection["task_id"])
        expected_turns = int(collection["fixed_primary_turns"])
        with Path(collection["csv"]).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "decoded_physical_params_json",
                "physical_geometry_sha256",
                "physical_G_fixed_lm2mh_json",
                "normalized_G_json",
                "screening_feasible_fixed_lm2mh",
                "fixed_lm2mh_resonance_contract_sha256",
                "fTx_Hz_fixed_lm2mh",
                "fRx_Hz_fixed_lm2mh",
                "fInter_Hz_fixed_lm2mh_diagnostic",
                "resonance_min_Hz_fixed_lm2mh",
                "objective_volume_L",
                "objective_total_loss_W",
            }
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise RuntimeError(
                    f"task {task_id} fixed terminal columns drifted"
                )
            task_rows = 0
            for csv_row_index, row in enumerate(reader, start=2):
                task_rows += 1
                params = json.loads(row["decoded_physical_params_json"])
                physical_g = json.loads(
                    row["physical_G_fixed_lm2mh_json"]
                )
                normalized_g = json.loads(row["normalized_G_json"])
                f_tx = float(row["fTx_Hz_fixed_lm2mh"])
                f_rx = float(row["fRx_Hz_fixed_lm2mh"])
                f_inter = float(
                    row["fInter_Hz_fixed_lm2mh_diagnostic"]
                )
                f_min = float(row["resonance_min_Hz_fixed_lm2mh"])
                expected_feasible = (
                    _truth(row["decoder_valid"])
                    and _truth(row["surrogate_physical_valid"])
                    and all(float(value) <= 0.0 for value in physical_g.values())
                )
                if (
                    int(params["N1"]) != expected_turns
                    or float(params["cw1"]) != 5.0
                    or float(params["gap1"]) != 1.6
                    or row["fixed_lm2mh_resonance_contract_sha256"]
                    != EXPECTED_RESONANCE_CONTRACT_SHA256
                    or not all(
                        math.isfinite(value) and value > 0.0
                        for value in (f_tx, f_rx, f_inter, f_min)
                    )
                    or not math.isclose(
                        f_min,
                        min(f_tx, f_rx),
                        rel_tol=1e-12,
                        abs_tol=1e-6,
                    )
                    or not math.isclose(
                        float(
                            physical_g[
                                "fixed_Lm2mH_self_resonance_minimum"
                            ]
                        ),
                        15_000.0 - f_min,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    or _truth(
                        row["screening_feasible_fixed_lm2mh"]
                    )
                    != expected_feasible
                    or not _truth(row["screening_only_fixed_lm2mh"])
                    or _truth(row["production_eligible_fixed_lm2mh"])
                ):
                    raise RuntimeError(
                        "fixed-Lm terminal row contract drifted: "
                        f"task={task_id} csv_row={csv_row_index}"
                    )
                winding_max, core_max = _temperatures(physical_g)
                violation = math.sqrt(
                    sum(
                        max(float(value), 0.0) ** 2
                        for value in normalized_g.values()
                    )
                )
                conditional_violation = math.sqrt(
                    sum(
                        max(float(value), 0.0) ** 2
                        for name, value in normalized_g.items()
                        if not name.startswith("temperature_robust_limit:")
                    )
                )
                conditional = (
                    _truth(row["decoder_valid"])
                    and _truth(row["surrogate_physical_valid"])
                    and all(
                        float(value) <= 0.0
                        for name, value in physical_g.items()
                        if not name.startswith("temperature_robust_limit:")
                    )
                )
                enriched = {
                    **row,
                    "scheduler_task_id": task_id,
                    "fixed_primary_turns_stratum": expected_turns,
                    "exterior_W_drawing_x_mm": 1200.0
                    + float(physical_g["exterior_width_limit"]),
                    "exterior_L_perpendicular_y_mm": 1000.0
                    + float(physical_g["exterior_length_limit"]),
                    "exterior_H_mm": 750.0
                    + float(physical_g["exterior_height_limit"]),
                    "winding_robust_max_C_screening": winding_max,
                    "core_robust_max_C_screening": core_max,
                    "normalized_constraint_violation_l2": violation,
                    "normalized_nonthermal_violation_l2": (
                        conditional_violation
                    ),
                    "conditional_nonthermal_feasible_diagnostic": (
                        conditional
                    ),
                    "global_screening_feasible": expected_feasible,
                    "global_production_eligible": False,
                }
                if any(
                    not math.isfinite(float(enriched[name]))
                    for name in (
                        "exterior_W_drawing_x_mm",
                        "exterior_L_perpendicular_y_mm",
                        "exterior_H_mm",
                        "winding_robust_max_C_screening",
                        "core_robust_max_C_screening",
                        "normalized_constraint_violation_l2",
                    )
                ):
                    raise RuntimeError(
                        f"task {task_id} terminal row is non-finite"
                    )
                raw_rows.append(enriched)
                identity = row["physical_geometry_sha256"]
                prior = deduplicated.get(identity)
                rank = (
                    violation,
                    float(row["objective_volume_L"]),
                    float(row["objective_total_loss_W"]),
                    task_id,
                )
                if prior is None or rank < (
                    float(prior["normalized_constraint_violation_l2"]),
                    float(prior["objective_volume_L"]),
                    float(prior["objective_total_loss_W"]),
                    int(prior["scheduler_task_id"]),
                ):
                    deduplicated[identity] = enriched
            if task_rows != TERMINAL_ROWS_PER_TASK:
                raise RuntimeError(
                    f"task {task_id} has {task_rows} terminal rows, expected 320"
                )
    raw_rows.sort(
        key=lambda row: (
            int(row["scheduler_task_id"]),
            int(row["terminal_population_index"]),
        )
    )
    unique_rows = sorted(
        deduplicated.values(),
        key=lambda row: (
            float(row["normalized_constraint_violation_l2"]),
            float(row["objective_volume_L"]),
            float(row["objective_total_loss_W"]),
            row["physical_geometry_sha256"],
        ),
    )
    return raw_rows, unique_rows


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    keys: tuple[str, ...],
) -> bool:
    observed = [
        (float(left[key]), float(right[key])) for key in keys
    ]
    return all(a <= b + 1e-12 for a, b in observed) and any(
        a < b - 1e-12 for a, b in observed
    )


def _non_dominated(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            *(float(row[key]) for key in keys),
            row["physical_geometry_sha256"],
        ),
    )
    front: list[dict[str, Any]] = []
    for row in ordered:
        if any(_dominates(prior, row, keys) for prior in front):
            continue
        front = [
            prior
            for prior in front
            if not _dominates(row, prior, keys)
        ]
        front.append(row)
    return sorted(
        front,
        key=lambda row: (
            float(row["objective_volume_L"]),
            float(row["objective_total_loss_W"]),
            row["physical_geometry_sha256"],
        ),
    )


def _acquisition_candidates(
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    count: int = 12,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    pool_map: dict[str, dict[str, Any]] = {}
    for row in (
        feasible_front
        + conditional_front
        + constraint_front
        + sorted(
            rows,
            key=lambda item: (
                float(item["normalized_constraint_violation_l2"]),
                float(item["objective_volume_L"]),
            ),
        )[:256]
    ):
        pool_map[row["physical_geometry_sha256"]] = row
    pool = list(pool_map.values())
    selected: list[tuple[str, dict[str, Any]]] = []
    seen = set()

    def add(role: str, row: dict[str, Any] | None) -> None:
        if row is None:
            return
        identity = row["physical_geometry_sha256"]
        if identity not in seen and len(selected) < count:
            selected.append((role, row))
            seen.add(identity)

    add(
        "least_normalized_constraint_violation",
        min(
            rows,
            key=lambda row: float(
                row["normalized_constraint_violation_l2"]
            ),
        ),
    )
    add(
        "minimum_volume",
        min(pool, key=lambda row: float(row["objective_volume_L"])),
    )
    add(
        "minimum_loss",
        min(pool, key=lambda row: float(row["objective_total_loss_W"])),
    )
    add(
        "maximum_fixed_lm2mh_resonance_margin",
        max(
            pool,
            key=lambda row: float(
                row["resonance_min_Hz_fixed_lm2mh"]
            ),
        ),
    )
    for column, role in (
        ("pred_C_tx_tx_F_fixed_lm2mh", "minimum_predicted_C_tx_tx"),
        ("pred_C_rx_rx_F_fixed_lm2mh", "minimum_predicted_C_rx_rx"),
        ("pred_C_tx_rx_F_fixed_lm2mh", "minimum_predicted_C_tx_rx"),
        (
            "winding_robust_max_C_screening",
            "minimum_screened_winding_temperature",
        ),
        (
            "core_robust_max_C_screening",
            "minimum_screened_core_temperature",
        ),
    ):
        add(role, min(pool, key=lambda row: float(row[column])))
    official_pool = [
        row
        for row in rows
        if int(row["fixed_primary_turns_stratum"]) == 6
    ]
    add(
        "nearest_official5_drawing_baseline_N1_6",
        (
            min(
                official_pool,
                key=lambda row: (
                    (
                        (
                            float(row["exterior_W_drawing_x_mm"])
                            - 1194.38
                        )
                        / 1200.0
                    )
                    ** 2
                    + (
                        (
                            float(
                                row[
                                    "exterior_L_perpendicular_y_mm"
                                ]
                            )
                            - 961.4
                        )
                        / 1000.0
                    )
                    ** 2
                    + (
                        (float(row["exterior_H_mm"]) - 707.0) / 750.0
                    )
                    ** 2
                ),
            )
            if official_pool
            else None
        ),
    )
    for turns in (5, 6, 7, 8):
        stratum = [
            row
            for row in rows
            if int(row["fixed_primary_turns_stratum"]) == turns
        ]
        add(
            f"N1_{turns}_least_violation",
            (
                min(
                    stratum,
                    key=lambda row: (
                        float(
                            row[
                                "normalized_constraint_violation_l2"
                            ]
                        ),
                        float(row["objective_volume_L"]),
                    ),
                )
                if stratum
                else None
            ),
        )

    dimensions = (
        "normalized_constraint_violation_l2",
        "objective_volume_L",
        "objective_total_loss_W",
        "resonance_min_Hz_fixed_lm2mh",
        "pred_C_tx_tx_F_fixed_lm2mh",
        "pred_C_rx_rx_F_fixed_lm2mh",
        "pred_C_tx_rx_F_fixed_lm2mh",
        "exterior_W_drawing_x_mm",
        "exterior_L_perpendicular_y_mm",
        "exterior_H_mm",
    )
    minima = {
        key: min(float(row[key]) for row in pool) for key in dimensions
    }
    spans = {
        key: max(float(row[key]) for row in pool) - minima[key]
        for key in dimensions
    }

    def vector(row: Mapping[str, Any]) -> tuple[float, ...]:
        return tuple(
            0.0
            if spans[key] <= 0.0
            else (float(row[key]) - minima[key]) / spans[key]
            for key in dimensions
        )

    def distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        return math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(vector(left), vector(right))
            )
        )

    while len(selected) < min(count, len(pool)):
        remaining = [
            row
            for row in pool
            if row["physical_geometry_sha256"] not in seen
        ]
        if not remaining:
            break
        choice = max(
            remaining,
            key=lambda row: min(
                distance(row, prior) for _, prior in selected
            ),
        )
        add("farthest_point_diversity_fill", choice)
    return [
        {
            **row,
            "acquisition_rank": rank,
            "acquisition_selection_role": role,
            "acquisition_topology": "symmetric_unrounded",
            "acquisition_rounded_FEA_allowed": False,
            "acquisition_classification": (
                "screening-only-requires-symmetric-FEA"
            ),
            "acquisition_production_eligible": False,
        }
        for rank, (role, row) in enumerate(selected, start=1)
    ]


def _first_terminal_smoke(
    collection: Mapping[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    task_id = int(collection["task_id"])
    subset = [
        row for row in rows if int(row["scheduler_task_id"]) == task_id
    ]
    if len(subset) != TERMINAL_ROWS_PER_TASK:
        raise RuntimeError("first terminal smoke row coverage drifted")
    value = {
        "schema_version": FIRST_TERMINAL_SMOKE_SCHEMA,
        "verified": True,
        "task_id": task_id,
        "seed": int(collection["seed"]),
        "row_count": len(subset),
        "cw1_unique_mm": sorted(
            {float(json.loads(row["decoded_physical_params_json"])["cw1"]) for row in subset}
        ),
        "gap1_unique_mm": sorted(
            {float(json.loads(row["decoded_physical_params_json"])["gap1"]) for row in subset}
        ),
        "axis_contract": {
            "W": "drawing_x_original_973mm_direction",
            "L": "drawing_y_perpendicular_direction",
            "rotation_or_axis_swap_allowed": False,
        },
        "W_range_mm": [
            min(float(row["exterior_W_drawing_x_mm"]) for row in subset),
            max(float(row["exterior_W_drawing_x_mm"]) for row in subset),
        ],
        "L_range_mm": [
            min(
                float(row["exterior_L_perpendicular_y_mm"])
                for row in subset
            ),
            max(
                float(row["exterior_L_perpendicular_y_mm"])
                for row in subset
            ),
        ],
        "H_range_mm": [
            min(float(row["exterior_H_mm"]) for row in subset),
            max(float(row["exterior_H_mm"]) for row in subset),
        ],
        "fixed_lm2mh_resonance_min_range_Hz": [
            min(
                float(row["resonance_min_Hz_fixed_lm2mh"])
                for row in subset
            ),
            max(
                float(row["resonance_min_Hz_fixed_lm2mh"])
                for row in subset
            ),
        ],
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "screening_feasible_count": sum(
            bool(row["global_screening_feasible"]) for row in subset
        ),
        "screening_only": True,
        "production_eligible": False,
    }
    if (
        value["cw1_unique_mm"] != [5.0]
        or value["gap1_unique_mm"] != [1.6]
    ):
        raise RuntimeError("first terminal smoke fixed controls drifted")
    value["payload_sha256"] = _sha(value)
    _atomic_json(output / "first_completed_terminal_smoke.json", value)
    return value


def _write_partial(
    output: Path,
    raw_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
) -> None:
    _write_csv(output / "partial_global_terminal_all_seeds_raw.csv", raw_rows)
    _write_csv(output / "partial_global_terminal_candidates.csv", rows)
    _write_csv(
        output / "partial_global_screening_pareto_front.csv",
        feasible_front,
    )
    _write_csv(
        output / "partial_global_conditional_nonthermal_pareto_front.csv",
        conditional_front,
    )
    _write_csv(
        output / "partial_global_constraint_objective_front.csv",
        constraint_front,
    )
    _write_csv(
        output / "partial_fea_acquisition_candidates.csv",
        acquisition,
    )


def _write_final(
    output: Path,
    raw_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    feasible_front: list[dict[str, Any]],
    conditional_front: list[dict[str, Any]],
    constraint_front: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = {
        "global_terminal_all_seeds_raw.csv": raw_rows,
        "global_terminal_candidates.csv": rows,
        "global_pareto_front.csv": feasible_front,
        "global_conditional_nonthermal_pareto_front.csv": (
            conditional_front
        ),
        "global_minimum_violation_objective_front.csv": constraint_front,
        "fea_acquisition_candidates.csv": acquisition,
    }
    for name, content in targets.items():
        _write_csv(output / name, content)
    manifest = {
        "schema_version": PARETO_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_seed_count": 16,
        "source_raw_terminal_row_count": len(raw_rows),
        "geometry_deduplicated_candidate_count": len(rows),
        "global_screening_feasible_count": sum(
            bool(row["global_screening_feasible"]) for row in rows
        ),
        "global_pareto_count": len(feasible_front),
        "conditional_nonthermal_pareto_count": len(conditional_front),
        "minimum_violation_objective_front_count": len(constraint_front),
        "fea_acquisition_candidate_count": len(acquisition),
        "global_non_dominated_sorting_complete": True,
        "pareto_definition": (
            "cross-seed geometry-deduplicated non-dominated sorting on "
            "(objective_volume_L, objective_total_loss_W) after every fixed-"
            "Lm2mH physical screening constraint is <=0"
        ),
        "conditional_front_definition": (
            "diagnostic-only non-dominated sorting after excluding thermal "
            "constraints; never a substitute for global_pareto_front.csv"
        ),
        "minimum_violation_front_definition": (
            "diagnostic-only 3-objective non-dominated sorting on normalized "
            "constraint violation L2, volume, and loss"
        ),
        "hard_spec_sha256": EXPECTED_HARD_SPEC_SHA256,
        "runner_source_sha256": EXPECTED_RUNNER_SOURCE_SHA256,
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": False,
        "screening_only": True,
        "production_eligible": False,
        "files": {
            name: {
                "path": str(output / name),
                "sha256": _sha_file(output / name),
                "row_count": len(targets[name]),
            }
            for name in FINAL_FILES
        },
    }
    manifest["payload_sha256"] = _sha(manifest)
    _atomic_json(output / "global_pareto_manifest.json", manifest)
    return manifest


def collect(
    *,
    manifest_path: Path,
    output: Path,
    scheduler_url: str,
    accounts: Path,
    scheduler_source: Path,
) -> dict[str, Any]:
    entries, submission = _submission(manifest_path)
    statuses = _status_inventory(entries, scheduler_url)
    details = _terminal_details(entries, statuses, scheduler_url)
    by_id = {int(entry["task_id"]): entry for entry in entries}
    scheduler_success_details = [
        detail
        for detail in details.values()
        if detail.get("status") == "completed"
        and int(detail.get("exit_code") or 0) == 0
    ]
    completed_details = list(scheduler_success_details)
    artifact_pending_details = [
        detail
        for detail in scheduler_success_details
        if not isinstance(_result_from_detail(detail), dict)
    ]
    failed_details = [
        detail
        for detail in details.values()
        if detail.get("status") in {"failed", "cancelled"}
        or (
            detail.get("status") == "completed"
            and int(detail.get("exit_code") or 0) != 0
        )
    ]
    account, ssh_session = transport._account(
        accounts.resolve(strict=True),
        scheduler_source.resolve(strict=True),
        "harry261",
    )
    connection = transport._PersistentAccountConnection(account, ssh_session)
    collections = []
    official_entry = next(
        entry for entry in entries if int(entry["fixed_primary_turns"]) == 6
    )
    official_status = next(
        status
        for status in statuses
        if int(status["id"]) == int(official_entry["task_id"])
    )
    try:
        official_smoke = _official5_smoke(
            connection=connection,
            entry=official_entry,
            status=official_status,
            output=output,
        )
        for detail in sorted(
            completed_details, key=lambda value: int(value["id"])
        ):
            task_id = int(detail["id"])
            collections.append(
                _download_task(
                    connection=connection,
                    entry=by_id[task_id],
                    detail=detail,
                    output=output / "collected",
                )
            )
    finally:
        connection.close()
    raw_rows, rows = _load_rows(collections)
    feasible = [
        row for row in rows if bool(row["global_screening_feasible"])
    ]
    conditional = [
        row
        for row in rows
        if bool(row["conditional_nonthermal_feasible_diagnostic"])
    ]
    objective_keys = ("objective_volume_L", "objective_total_loss_W")
    feasible_front = _non_dominated(feasible, objective_keys)
    conditional_front = _non_dominated(conditional, objective_keys)
    constraint_front = _non_dominated(
        rows,
        (
            "normalized_constraint_violation_l2",
            "objective_volume_L",
            "objective_total_loss_W",
        ),
    )
    acquisition = _acquisition_candidates(
        rows,
        feasible_front,
        conditional_front,
        constraint_front,
        count=12,
    )
    _write_partial(
        output,
        raw_rows,
        rows,
        feasible_front,
        conditional_front,
        constraint_front,
        acquisition,
    )
    first_terminal_smoke = (
        _first_terminal_smoke(collections[0], raw_rows, output)
        if collections
        else {
            "verified": False,
            "state": "awaiting_first_successful_terminal_task",
        }
    )
    status_counts: dict[str, int] = {}
    for status in statuses:
        name = str(status.get("status") or "unknown")
        status_counts[name] = status_counts.get(name, 0) + 1
    final_ready = (
        len(collections) == 16
        and not failed_details
        and len(raw_rows) == EXPECTED_RAW_ROWS
        and status_counts.get("completed", 0) == 16
    )
    pareto_manifest = (
        _write_final(
            output,
            raw_rows,
            rows,
            feasible_front,
            conditional_front,
            constraint_front,
            acquisition,
        )
        if final_ready
        else None
    )
    result = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "submission_manifest": str(manifest_path),
        "submission_manifest_payload_sha256": submission["payload_sha256"],
        "hard_spec_sha256": EXPECTED_HARD_SPEC_SHA256,
        "runner_source_sha256": EXPECTED_RUNNER_SOURCE_SHA256,
        "fixed_lm2mh_resonance_contract_sha256": (
            EXPECTED_RESONANCE_CONTRACT_SHA256
        ),
        "expected_seed_count": 16,
        "successful_terminal_seed_count": len(collections),
        "status_counts": dict(sorted(status_counts.items())),
        "raw_terminal_row_count": len(raw_rows),
        "expected_raw_terminal_row_count": EXPECTED_RAW_ROWS,
        "geometry_deduplicated_candidate_count": len(rows),
        "global_screening_feasible_count": len(feasible),
        "partial_screening_pareto_count": len(feasible_front),
        "partial_conditional_nonthermal_pareto_count": len(
            conditional_front
        ),
        "partial_minimum_violation_objective_front_count": len(
            constraint_front
        ),
        "fea_acquisition_candidate_count": len(acquisition),
        "global_nds_final": final_ready,
        "final_files_written": bool(pareto_manifest),
        "official5_projected_smoke": official_smoke,
        "first_completed_terminal_smoke": first_terminal_smoke,
        "failed_terminal_tasks": [
            {
                "task_id": int(detail["id"]),
                "status": detail.get("status"),
                "exit_code": detail.get("exit_code"),
                "failure_message": detail.get("failure_message"),
                "stderr_tail": str(detail.get("stderr") or "")[-4000:],
            }
            for detail in failed_details
        ],
        "successful_terminal_artifact_pending_tasks": [
            {
                "task_id": int(detail["id"]),
                "status": detail.get("status"),
                "exit_code": detail.get("exit_code"),
                "state": "awaiting_scheduler_RESULT_JSON_visibility",
            }
            for detail in artifact_pending_details
        ],
        "collections": collections,
        "task_status": [
            {
                "task_id": int(status["id"]),
                "name": status["name"],
                "status": status["status"],
                "started_at": status.get("started_at"),
            }
            for status in statuses
        ],
        "classification": "screening-only",
        "thermal_surrogate_retrained_for_Lm2mH_magnetizing_current": False,
        "production_eligible": False,
        "pareto_manifest_payload_sha256": (
            None
            if pareto_manifest is None
            else pareto_manifest["payload_sha256"]
        ),
    }
    result["payload_sha256"] = _sha(result)
    _atomic_json(output / "collector_status.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.poll_seconds < 10.0 or args.poll_seconds > 300.0:
        raise RuntimeError("poll-seconds must be within 10..300")
    while True:
        result = collect(
            manifest_path=args.manifest,
            output=args.output.resolve(),
            scheduler_url=args.scheduler_url,
            accounts=args.accounts,
            scheduler_source=args.scheduler_source,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if (
            not args.watch
            or result["global_nds_final"]
            or result["failed_terminal_tasks"]
        ):
            return (
                0
                if result["global_nds_final"] or not args.watch
                else 2
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
