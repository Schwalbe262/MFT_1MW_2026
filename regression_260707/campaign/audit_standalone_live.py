"""Read-only heartbeat for the standalone MFT production cohort.

The report intentionally performs no scheduler or dataset mutation.  It is a
small operator-side companion to the Web UI: summarize the logical cohort,
sample live stdout to prove solver-stage progress, and show whether terminal
tasks have been judged by the canonical collector.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_SCHEDULER = os.environ.get(
    "MFT_SCHEDULER_URL", "http://127.0.0.1:8002"
).rstrip("/")
DEFAULT_DATASET_DIR = Path(
    r"Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset"
)
TERMINAL = frozenset({"completed", "failed", "timed_out", "canceled"})


def _get_json(url: str, timeout: float = 30.0) -> Any:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local API
        return json.load(response)


def _get_text(url: str, timeout: float = 30.0) -> str:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local API
        return response.read().decode("utf-8", errors="replace")


def classify_stdout(text: str) -> str:
    """Return the furthest unambiguous production stage in one stdout tail."""

    if "RESULT_JSON " in text:
        return "result_emitted"
    if "Solving design setup ThermalSetup" in text:
        return "thermal_solving"
    if "Added design 'icepak_thermal'" in text:
        return "thermal_modeling"
    if "Active Design set to maxwell_matrix1" in text:
        return "loss_copy_or_solve"
    if "Added design 'maxwell_cap'" in text:
        if "Error in Solving Setup Setup1" in text:
            return "cap_solve_error"
        if text.count("Design setup Setup1 solved correctly") >= 2:
            return "cap_solved"
        return "cap_modeling"
    if "Solving design setup Setup1" in text:
        return "matrix_solving"
    if "Added design 'maxwell_matrix'" in text:
        return "matrix_modeling"
    if "Electronics Desktop started" in text:
        return "aedt_started"
    return "startup_or_no_recent_marker"


def _load_collector_cache(path: Path) -> dict[str, set[int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, set[int]] = {}
    for key in ("harvested", "nodata"):
        values = raw.get(key, [])
        if isinstance(values, list):
            result[key] = {
                value for value in values
                if isinstance(value, int) and not isinstance(value, bool)
            }
    return result


def _dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    train = dataset_dir / "train.parquet"
    cache_path = dataset_dir / "collect_cache.json"
    result: dict[str, Any] = {
        "directory": str(dataset_dir),
        "train_exists": train.is_file(),
        "collector_cache_exists": cache_path.is_file(),
    }
    for path, prefix in ((train, "train"), (cache_path, "collector_cache")):
        if path.is_file():
            stat = path.stat()
            result[f"{prefix}_size_bytes"] = stat.st_size
            result[f"{prefix}_mtime_ns"] = stat.st_mtime_ns
    if train.is_file():
        try:
            import pyarrow.parquet as parquet

            result["train_rows"] = parquet.ParquetFile(train).metadata.num_rows
        except Exception as exc:  # report degraded evidence; never mutate/retry
            result["train_rows_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    query = urlencode({
        "project": args.project,
        "name_prefix": args.name_prefix,
        "compact": "true",
        "limit": args.limit,
    })
    tasks = _get_json(f"{args.scheduler_url}/api/tasks?{query}")
    if not isinstance(tasks, list):
        raise RuntimeError("scheduler returned a non-list task inventory")
    cohort = [
        task for task in tasks
        if isinstance(task, dict)
        and isinstance(task.get("id"), int)
        and task["id"] >= args.min_task_id
    ]
    cohort.sort(key=lambda item: item["id"], reverse=True)
    status_counts = Counter(str(task.get("status") or "unknown") for task in cohort)
    terminal = [task for task in cohort if task.get("status") in TERMINAL]

    details: list[dict[str, Any]] = []
    for task in terminal[: args.terminal_detail_limit]:
        detail = _get_json(f"{args.scheduler_url}/api/tasks/{task['id']}")
        if isinstance(detail, dict):
            details.append({
                key: detail.get(key)
                for key in (
                    "id", "name", "status", "account_name", "node_name",
                    "allocation_id", "exit_code", "failure_message",
                    "started_at", "finished_at",
                )
            })

    active = [
        task for task in reversed(cohort)
        if task.get("status") in {"attaching", "running"}
    ]
    samples: list[dict[str, Any]] = []
    for task in active[: args.active_log_samples]:
        text = _get_text(
            f"{args.scheduler_url}/api/tasks/{task['id']}/stdout?"
            "tail_lines=160&max_bytes=65536",
            timeout=45,
        )
        samples.append({
            "id": task["id"],
            "status": task.get("status"),
            "stage": classify_stdout(text),
            "cap_solve_error": "Error in Solving Setup Setup1" in text,
            "result_emitted": "RESULT_JSON " in text,
        })

    cache = _load_collector_cache(args.dataset_dir / "collect_cache.json")
    judged = cache.get("harvested", set()) | cache.get("nodata", set())
    terminal_ids = {task["id"] for task in terminal}
    dataset = _dataset_summary(args.dataset_dir)
    dataset.update({
        "collector_harvested_total": len(cache.get("harvested", set())),
        "collector_nodata_total": len(cache.get("nodata", set())),
        "cohort_terminal_judged": len(terminal_ids & judged),
        "cohort_terminal_unjudged_ids": sorted(terminal_ids - judged),
    })
    return {
        "read_only": True,
        "scheduler_url": args.scheduler_url,
        "project": args.project,
        "name_prefix": args.name_prefix,
        "min_task_id": args.min_task_id,
        "cohort_total": len(cohort),
        "status_counts": dict(sorted(status_counts.items())),
        "logical_active": sum(
            status_counts.get(status, 0)
            for status in ("queued", "attaching", "running")
        ),
        "terminal_details": details,
        "active_stage_samples": samples,
        "dataset": dataset,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER)
    result.add_argument("--project", default="MFT_1MW_2026v1")
    result.add_argument("--name-prefix", default="mft-camp")
    result.add_argument("--min-task-id", type=int, required=True)
    result.add_argument("--limit", type=int, default=2000)
    result.add_argument("--terminal-detail-limit", type=int, default=20)
    result.add_argument("--active-log-samples", type=int, default=8)
    result.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return result


def main() -> None:
    args = parser().parse_args()
    args.scheduler_url = args.scheduler_url.rstrip("/")
    if args.min_task_id <= 0 or args.limit <= 0:
        raise SystemExit("--min-task-id and --limit must be positive")
    print(json.dumps(build_report(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
