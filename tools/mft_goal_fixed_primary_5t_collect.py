"""Collect axis-v6 5T search results and rebuild one cross-seed Pareto front."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
import urllib.request

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import slurm_nsga_offload as transport


SCHEMA = "mft-goal-fixed-primary-5t-axis-global-nds-v1"
CAMPAIGN_ID = "mft-goal-fixed-primary-5t-gap1-1p6-axis-v6"
RESULT_SCHEMA = "mft-goal-fixed-primary-5t-nsga-result-v1"
SUBMISSION_SCHEMA = "mft-goal-fixed-primary-5t-nsga-submission-v1"
EXPECTED_HARD_SPEC_SHA256 = (
    "bb05c758dab06a802627681c0c19be8e7062f423dda54089c0a539e61b0f7d5c"
)
REMOTE_BUNDLE = (
    "/gpfs/tmp_cpu2/mft_goal_20260726/"
    "mft-goal-763dbb46ae74e1722969d2c2"
)
DEFAULT_MANIFESTS = (
    Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
        r"\fixed_primary_5t_gap1_1p6_axis_nsga_v6_canary1"
        r"\submission_manifest.json"
    ),
    Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
        r"\fixed_primary_5t_gap1_1p6_axis_nsga_v6_wave15"
        r"\submission_manifest.json"
    ),
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_primary_5t_gap1_1p6_axis_nsga_v6_global_nds"
)
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
SCHEDULER_URL = "http://127.0.0.1:8002"
WINDING_TARGETS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
)
CORE_TARGETS = (
    "T_max_core",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _sha(unsigned):
        raise RuntimeError(f"{schema} seal mismatch")
    return dict(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _api_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Scheduler returned a non-object")
    return value


def _submission_entries(
    manifests: Iterable[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = []
    common = None
    for path in manifests:
        manifest = _validate_seal(
            _read_json(path.resolve(strict=True)), SUBMISSION_SCHEMA
        )
        stable = {
            "campaign_id": manifest["campaign_id"],
            "hard_spec_sha256": manifest["hard_spec_sha256"],
            "runner_source_sha256": manifest["runner_source_sha256"],
            "population": manifest["population"],
            "generations": manifest["generations"],
            "campaign_total_seed_count": manifest[
                "campaign_total_seed_count"
            ],
        }
        if common is None:
            common = stable
        elif stable != common:
            raise RuntimeError("submission manifests mix campaign identity")
        entries.extend(manifest["submissions"])
    assert common is not None
    if (
        common["campaign_id"] != CAMPAIGN_ID
        or common["hard_spec_sha256"] != EXPECTED_HARD_SPEC_SHA256
        or common["population"] != 320
        or common["generations"] != 80
        or common["campaign_total_seed_count"] != 16
        or len(entries) != 16
        or len({int(item["task_id"]) for item in entries}) != 16
        or len({int(item["seed"]) for item in entries}) != 16
    ):
        raise RuntimeError("axis-v6 submission coverage mismatch")
    entries.sort(key=lambda item: int(item["seed"]))
    return entries, common


def _status(entry: Mapping[str, Any], scheduler_url: str) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    value = _api_json(
        scheduler_url.rstrip("/")
        + f"/api/tasks/{task_id}?include_output=true&output_limit=20000"
    )
    expected_name = (
        f"mft-5t-g1p6-s{int(entry['seed'])}-"
        f"n1-{int(entry['fixed_primary_turns'])}"
    )
    if (
        int(value.get("id", value.get("task_id", -1))) != task_id
        or value.get("name") != expected_name
        or int(value.get("cpus") or 0) != 8
        or int(value.get("memory_mb") or 0) != 65_536
        or value.get("remote_cwd") != REMOTE_BUNDLE
    ):
        raise RuntimeError(f"Scheduler task {task_id} identity drifted")
    return value


def _download_task(
    *,
    connection: Any,
    entry: Mapping[str, Any],
    status: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    task_id = int(entry["task_id"])
    result_json = status.get("result_json")
    if not isinstance(result_json, dict):
        raise RuntimeError(f"task {task_id} has no Scheduler RESULT_JSON")
    if (
        result_json.get("schema_version") != RESULT_SCHEMA
        or result_json.get("task_payload_sha256")
        != entry["task_payload_sha256"]
        or result_json.get("physics_sha256") != entry["physics_sha256"]
        or result_json.get("hard_spec_sha256")
        != EXPECTED_HARD_SPEC_SHA256
        or result_json.get("terminal_primary_controls_attested") is not True
        or result_json.get("terminal_primary_control_escape_count") != 0
        or result_json.get("global_nds_ready") is not True
    ):
        raise RuntimeError(f"task {task_id} result identity drifted")
    inventory = result_json.get("artifact_inventory") or {}
    names = (
        "result.json",
        "terminal_physical_candidates.csv",
        "terminal_physical_candidates.manifest.json",
    )
    task_root = output / f"task-{task_id}"
    remote_root = (
        f"{REMOTE_BUNDLE}/runs/fixed-primary-5t-task-{task_id}"
    )
    records: dict[str, Any] = {}
    for name in names:
        destination = task_root / name
        remote = f"{remote_root}/{name}"
        expected = (
            None if name == "result.json" else inventory.get(name)
        )
        record = transport._download_verified_sftp(
            connection,
            remote,
            destination,
            retries=3,
            expected_identity=(
                None
                if expected is None
                else {
                    "sha256": expected["sha256"],
                    "bytes": expected["size_bytes"],
                }
            ),
            max_bytes=64 * 1024 * 1024,
        )
        records[name] = record
    result = _validate_seal(
        _read_json(task_root / "result.json"), RESULT_SCHEMA
    )
    if result != result_json:
        raise RuntimeError(f"task {task_id} RESULT_JSON/file mismatch")
    manifest = _validate_seal(
        _read_json(
            task_root / "terminal_physical_candidates.manifest.json"
        ),
        "mft-goal-20260726-terminal-physical-candidates-v1",
    )
    csv_path = task_root / "terminal_physical_candidates.csv"
    if (
        manifest.get("row_count") != 320
        or (manifest.get("csv") or {}).get("sha256") != _sha_file(csv_path)
        or (manifest.get("source_identity") or {}).get("task_id")
        != str(task_id)
        or manifest.get("stage_spec_sha256")
        != result["artifact_inventory"][
            "terminal_physical_candidates.manifest.json"
        ].get("stage_spec_sha256", manifest.get("stage_spec_sha256"))
    ):
        raise RuntimeError(f"task {task_id} terminal manifest mismatch")
    return {
        "task_id": task_id,
        "seed": int(entry["seed"]),
        "fixed_primary_turns": int(entry["fixed_primary_turns"]),
        "result_payload_sha256": result["payload_sha256"],
        "csv": str(csv_path),
        "csv_sha256": _sha_file(csv_path),
        "download_records": records,
    }


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _objective_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["objective_volume_L"]),
            float(row["objective_total_loss_W"]),
            row["physical_geometry_sha256"],
        ),
    )
    result = []
    best_loss = math.inf
    for row in ordered:
        loss = float(row["objective_total_loss_W"])
        if loss < best_loss - 1e-12:
            result.append(row)
            best_loss = loss
    return result


def _row_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    g = json.loads(row["physical_G_json"])
    normalized = json.loads(row["normalized_G_json"])
    params = json.loads(row["decoded_physical_params_json"])
    violation = sum(max(float(value), 0.0) ** 2 for value in normalized.values())
    winding = max(
        100.0 + float(g.get(f"temperature_robust_limit:{name}", 1e9))
        for name in WINDING_TARGETS
        if f"temperature_robust_limit:{name}" in g
    )
    core = max(
        120.0 + float(g.get(f"temperature_robust_limit:{name}", 1e9))
        for name in CORE_TARGETS
        if f"temperature_robust_limit:{name}" in g
    )
    resonance = 15_000.0 - float(
        g["half_magnetizing_resonance_minimum"]
    )
    return {
        "constraint_violation_l2": violation,
        "exterior_W_mm": 1000.0 + float(g["exterior_width_limit"]),
        "exterior_L_mm": 1200.0 + float(g["exterior_length_limit"]),
        "exterior_H_mm": 750.0 + float(g["exterior_height_limit"]),
        "resonance_Hz": resonance,
        "winding_robust_max_C": winding,
        "core_robust_max_C": core,
        "N1": int(params["N1"]),
        "N2": int(params["N2"]),
        "cw1_mm": float(params["cw1"]),
        "gap1_mm": float(params["gap1"]),
        "cw2_mm": float(params["cw2"]),
        "gap2_mm": float(params["gap2"]),
        "nwl2_main_mm": float(params["nwl2_main"]),
        "nwl2_side_mm": float(params["nwl2_side"]),
    }


def _load_rows(collections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for collection in collections:
        with Path(collection["csv"]).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            for csv_row_index, row in enumerate(reader, start=2):
                metrics = _row_metrics(row)
                if (
                    not _truth(row["decoder_valid"])
                    or not math.isclose(
                        metrics["cw1_mm"], 5.0, rel_tol=0.0, abs_tol=1e-12
                    )
                    or not math.isclose(
                        metrics["gap1_mm"], 1.6, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise RuntimeError(
                        "collected terminal row violates fixed controls: "
                        f"task={collection['task_id']} csv_row={csv_row_index} "
                        f"terminal_index={row.get('terminal_population_index')} "
                        f"decoder_valid={row.get('decoder_valid')!r} "
                        f"cw1_mm={metrics['cw1_mm']!r} "
                        f"gap1_mm={metrics['gap1_mm']!r}"
                    )
                enriched = {
                    **row,
                    **metrics,
                    "scheduler_task_id": int(collection["task_id"]),
                }
                key = row["physical_geometry_sha256"]
                prior = deduplicated.get(key)
                candidate_key = (
                    metrics["constraint_violation_l2"],
                    float(row["objective_volume_L"]),
                    float(row["objective_total_loss_W"]),
                )
                if prior is None or candidate_key < (
                    float(prior["constraint_violation_l2"]),
                    float(prior["objective_volume_L"]),
                    float(prior["objective_total_loss_W"]),
                ):
                    deduplicated[key] = enriched
    return list(deduplicated.values())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compact_summary(
    feasible_front: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pool = feasible_front or sorted(
        all_rows,
        key=lambda row: (
            float(row["constraint_violation_l2"]),
            float(row["exterior_W_mm"]),
            float(row["objective_volume_L"]),
        ),
    )[:50]
    if not pool:
        return {"candidate_count": 0, "selection": {}}
    dimensions = (
        "exterior_W_mm",
        "objective_volume_L",
        "objective_total_loss_W",
    )
    minima = {name: min(float(row[name]) for row in pool) for name in dimensions}
    maxima = {name: max(float(row[name]) for row in pool) for name in dimensions}

    def balanced(row: Mapping[str, Any]) -> float:
        total = 0.0
        for name in dimensions:
            span = maxima[name] - minima[name]
            total += (
                0.0
                if span <= 0.0
                else ((float(row[name]) - minima[name]) / span) ** 2
            )
        return math.sqrt(total)

    choices = {
        "minimum_width": min(pool, key=lambda row: float(row["exterior_W_mm"])),
        "minimum_volume": min(
            pool, key=lambda row: float(row["objective_volume_L"])
        ),
        "minimum_loss": min(
            pool, key=lambda row: float(row["objective_total_loss_W"])
        ),
        "balanced_width_volume_loss_knee": min(pool, key=balanced),
        "least_constraint_violation": min(
            pool, key=lambda row: float(row["constraint_violation_l2"])
        ),
    }
    fields = (
        "physical_geometry_sha256",
        "scheduler_task_id",
        "source_seed",
        "N1",
        "N2",
        "objective_volume_L",
        "objective_total_loss_W",
        "exterior_W_mm",
        "exterior_L_mm",
        "exterior_H_mm",
        "resonance_Hz",
        "winding_robust_max_C",
        "core_robust_max_C",
        "cw1_mm",
        "gap1_mm",
        "cw2_mm",
        "gap2_mm",
        "nwl2_main_mm",
        "nwl2_side_mm",
        "constraint_violation_l2",
    )
    return {
        "candidate_count": len(pool),
        "selection_basis": (
            "hard_feasible_global_pareto"
            if feasible_front
            else "least_violation_fallback_not_production"
        ),
        "selection": {
            name: {field: row[field] for field in fields}
            for name, row in choices.items()
        },
    }


def collect(
    *,
    manifests: Iterable[Path],
    output: Path,
    scheduler_url: str,
    accounts: Path,
    scheduler_source: Path,
) -> dict[str, Any]:
    entries, common = _submission_entries(manifests)
    statuses = [_status(entry, scheduler_url) for entry in entries]
    by_id = {int(entry["task_id"]): entry for entry in entries}
    completed = [
        status
        for status in statuses
        if status.get("status") == "completed"
        and int(status.get("exit_code") or 0) == 0
    ]
    account, ssh_session = transport._account(
        accounts.resolve(strict=True),
        scheduler_source.resolve(strict=True),
        "harry261",
    )
    connection = transport._PersistentAccountConnection(account, ssh_session)
    collections = []
    try:
        for status in completed:
            task_id = int(status["id"])
            collections.append(
                _download_task(
                    connection=connection,
                    entry=by_id[task_id],
                    status=status,
                    output=output / "collected",
                )
            )
    finally:
        connection.close()
    rows = _load_rows(collections)
    feasible = []
    for row in rows:
        physical_g = json.loads(row["physical_G_json"])
        if (
            _truth(row["physical_feasible"])
            and _truth(row["surrogate_physical_valid"])
            and all(float(value) <= 1e-9 for value in physical_g.values())
        ):
            feasible.append(row)
    feasible_front = _objective_front(feasible)
    objective_front = _objective_front(rows)
    _write_csv(output / "global_terminal_candidates.csv", rows)
    _write_csv(output / "global_pareto_front.csv", feasible_front)
    _write_csv(output / "global_objective_front.csv", objective_front)
    compact = _compact_summary(feasible_front, rows)
    _atomic_json(output / "compact_width_knee_summary.json", compact)
    status_counts: dict[str, int] = {}
    for status in statuses:
        key = str(status.get("status") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    result = {
        "schema_version": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "hard_spec_sha256": EXPECTED_HARD_SPEC_SHA256,
        "runner_source_sha256": common["runner_source_sha256"],
        "expected_seed_count": 16,
        "terminal_success_seed_count": len(collections),
        "status_counts": dict(sorted(status_counts.items())),
        "deduplicated_terminal_candidate_count": len(rows),
        "hard_feasible_candidate_count": len(feasible),
        "global_pareto_count": len(feasible_front),
        "global_objective_front_count": len(objective_front),
        "global_nds_final": len(collections) == 16,
        "all_scheduler_tasks_terminal": sum(
            status_counts.get(name, 0)
            for name in ("completed", "failed", "cancelled")
        )
        == 16,
        "files": {
            name: {
                "path": str(output / name),
                "sha256": _sha_file(output / name),
            }
            for name in (
                "global_terminal_candidates.csv",
                "global_pareto_front.csv",
                "global_objective_front.csv",
                "compact_width_knee_summary.json",
            )
        },
        "collections": collections,
        "task_status": [
            {
                "task_id": int(status["id"]),
                "status": status["status"],
                "node": status.get("actual_node_name") or None,
                "started_at": status.get("started_at"),
                "finished_at": status.get("finished_at"),
                "failure_message": status.get("failure_message") or None,
            }
            for status in statuses
        ],
    }
    result["payload_sha256"] = _sha(result)
    _atomic_json(output / "collector_status.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, action="append", dest="manifests"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=45.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.poll_seconds < 10.0 or args.poll_seconds > 300.0:
        raise RuntimeError("poll-seconds must be within 10..300")
    while True:
        result = collect(
            manifests=args.manifests or DEFAULT_MANIFESTS,
            output=args.output.resolve(),
            scheduler_url=args.scheduler_url,
            accounts=args.accounts,
            scheduler_source=args.scheduler_source,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if (
            not args.watch
            or result["global_nds_final"]
            or result["all_scheduler_tasks_terminal"]
        ):
            return 0 if result["global_nds_final"] or not args.watch else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
