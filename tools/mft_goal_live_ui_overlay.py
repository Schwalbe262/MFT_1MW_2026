"""Overlay the active corrected NSGA and core-rescue FEA lanes on the web UI.

The updater is deliberately read-only with respect to Scheduler.  It performs
only ``GET /api/tasks`` and atomically rewrites the existing Codex status JSON.
Existing completed/attention history is preserved.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping
import urllib.parse
import urllib.request


PROJECT = "MFT_1MW_2026v1"
NSGA_PREFIX = "mft-goal-physics-delta-nsga-"
FEA_PREFIX = "mft-core-rescue"
ACTIVE = frozenset({"queued", "attaching", "running"})
TERMINAL = frozenset({"completed", "failed", "cancelled"})


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staged_path = Path(staged)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_path, path)
    finally:
        staged_path.unlink(missing_ok=True)


def _get_tasks(scheduler_url: str, prefix: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "limit": 10000,
            "project": PROJECT,
            "name_prefix": prefix,
        }
    )
    request = urllib.request.Request(
        f"{scheduler_url.rstrip('/')}/api/tasks?{query}",
        method="GET",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, list) or len(value) >= 10000:
        raise RuntimeError(f"Scheduler task inventory is malformed: {prefix}")
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or row.get("state") or "unknown")
        result[status] = result.get(status, 0) + 1
    return result


def _count_text(counts: Mapping[str, int]) -> str:
    return (
        f"제출 {sum(counts.values())} · 실행 {counts.get('running', 0)} · "
        f"연결 {counts.get('attaching', 0)} · 대기 {counts.get('queued', 0)} · "
        f"완료 {counts.get('completed', 0)} · 실패 {counts.get('failed', 0)}"
    )


def _replace_section(
    sections: list[dict[str, Any]], section: Mapping[str, Any]
) -> list[dict[str, Any]]:
    target = str(section["id"])
    return [
        copy.deepcopy(dict(section)),
        *[
            copy.deepcopy(dict(row))
            for row in sections
            if str(row.get("id") or "") != target
        ],
    ]


def _candidate_evidence(plan: Mapping[str, Any]) -> list[str]:
    selection = plan.get("selection") or {}
    candidates = (
        selection.get("candidates")
        or selection.get("selected_candidates")
        or []
    )
    if not isinstance(candidates, list):
        return []
    evidence = []
    for index, raw in enumerate(candidates, start=1):
        if not isinstance(raw, Mapping):
            continue
        dimensions = raw.get("dimensions_mm") or {}
        params = raw.get("params") or {}
        thermal = raw.get("temperature_surrogate") or {}
        cap = raw.get("physics_capacitance") or {}
        split = raw.get("split_repair") or {}
        evidence.append(
            "C{index}: {W:.1f}×{L:.1f}×{H:.1f} mm | "
            "N2={main}/{side} | cw2={cw2:.2f} | gap2={gap2:.2f} | "
            "B={B:.3f} T | fRx(mean)={fkhz:.2f} kHz | "
            "Llt(mean)={llt:.2f} µH | Tx/core(q90)={tx:.1f}/{core:.1f} °C".format(
                index=index,
                W=float(dimensions.get("W", 0.0)),
                L=float(dimensions.get("L", 0.0)),
                H=float(dimensions.get("H", 0.0)),
                main=int(params.get("N2_main", 0)),
                side=int(params.get("N2_side", 0)),
                cw2=float(params.get("cw2", 0.0)),
                gap2=float(params.get("gap2", 0.0)),
                B=float(raw.get("B_design_square_material_analytic_T", 0.0)),
                fkhz=float(cap.get("physics_delta_fRx_mean_Hz", 0.0)) / 1000.0,
                llt=float(split.get("Llt_mean_uH", 0.0)),
                tx=float(
                    (thermal.get("T_max_Tx") or {}).get("q90_upper_C", 0.0)
                ),
                core=float(
                    (thermal.get("T_max_core") or {}).get(
                        "q90_upper_C", 0.0
                    )
                ),
            )
        )
    return evidence


def update(
    *,
    status_file: Path,
    scheduler_url: str,
    core_rescue_plan: Path,
    fea_prefix: str = FEA_PREFIX,
    extra_fea_prefixes: Iterable[str] = (),
) -> dict[str, Any]:
    status = _read_json(status_file)
    nsga_rows = _get_tasks(scheduler_url, NSGA_PREFIX)
    fea_rows_by_id: dict[int, dict[str, Any]] = {}
    for prefix in (fea_prefix, *extra_fea_prefixes):
        for row in _get_tasks(scheduler_url, prefix):
            fea_rows_by_id[int(row["id"])] = row
    fea_rows = list(fea_rows_by_id.values())
    nsga_counts = _counts(nsga_rows)
    fea_counts = _counts(fea_rows)
    plan = _read_json(core_rescue_plan) if core_rescue_plan.is_file() else {}
    planned_lanes = max(len(plan.get("lanes") or []), len(fea_rows))
    candidates = _candidate_evidence(plan)
    now = _now()

    nsga_state = (
        "completed"
        if nsga_counts.get("completed", 0) == 60
        else "in_progress"
    )
    nsga_detail = (
        "보정된 60-turn 물리 커패시턴스와 Lm=2 mH 공진 계약을 사용하는 "
        "exact-60 다중 시드 NSGA-II입니다. 모든 시드의 terminal 320행을 "
        "통합한 뒤 전역 non-dominated sorting을 수행합니다."
    )
    nsga_evidence = [
        _count_text(nsga_counts),
        "seeds=2607264400..2607264459 / priority=100",
        "population=320 / generations=300",
        "raw two-net C authority=false",
        "physics-delta mean acquisition + final direct turn-graded FEA",
        "Scheduler access by this UI watcher=GET only",
    ]
    if not nsga_rows:
        nsga_evidence.insert(0, "상태=sealed prepare/stage/POST 진행 중")

    fea_state = (
        "completed"
        if planned_lanes > 0
        and fea_counts.get("completed", 0) == planned_lanes
        else "in_progress"
    )
    fea_detail = (
        "n_core_group=4 후보와 compact n_core_group=5 후보에 대해 동일 "
        "3-leg air-gap의 0.20/0.35/0.50 mm bracket을 병렬 실행합니다. "
        "0.35 mm lane은 Matrix·turn-graded Rx Cap·Loss·Thermal 전체, "
        "양쪽 lane은 Lm bracket입니다."
    )
    fea_evidence = [
        f"계획 lane={planned_lanes} / {_count_text(fea_counts)}",
        "1/8 symmetric / non-rounded / 6/60 turns",
        "core groups=4 or 5 / core plate stacks=5 or 6",
        "pure core depth=68..90 mm",
        "core plate=20T / winding cold plate=20T / pad=2T",
        "fan=1.5 m/s / TIM unchanged",
        "equal winding heights / cw1=5T / gap1=1.6 mm / cw2≤1T",
        "final Lm target=2.00 mH with identical physical gap on all 3 legs",
        *candidates,
    ]

    current = [
        copy.deepcopy(dict(row))
        for row in status.get("current", [])
        if isinstance(row, Mapping)
    ]
    current = _replace_section(
        current,
        {
            "id": "active-corrected-physics-exact60",
            "title": f"보정 NSGA-II exact-60 | {_count_text(nsga_counts)}",
            "detail": nsga_detail,
            "state": nsga_state,
            "updated_at": now,
            "evidence": nsga_evidence,
            "progress_pct": round(
                100.0 * nsga_counts.get("completed", 0) / 60.0, 1
            ),
        },
    )
    current = _replace_section(
        current,
        {
            "id": "active-core-rescue-symmetric-fea",
            "title": (
                "4·5그룹 core-rescue symmetric FEA | "
                f"{_count_text(fea_counts)}"
            ),
            "detail": fea_detail,
            "state": fea_state,
            "updated_at": now,
            "evidence": fea_evidence,
            "progress_pct": (
                round(
                    100.0
                    * fea_counts.get("completed", 0)
                    / float(planned_lanes),
                    1,
                )
                if planned_lanes
                else 0.0
            ),
        },
    )

    status["generated_at"] = now
    status["summary"] = (
        f"보정 NSGA exact-60: {_count_text(nsga_counts)}. "
        f"4·5그룹 symmetric FEA: 계획 {planned_lanes}, "
        f"{_count_text(fea_counts)}. "
        "후보 확정은 동일 3-leg gap의 Lm=2 mH, turn-graded C, loss/thermal "
        "직접 FEA를 통과한 뒤에만 수행합니다."
    )
    status["current"] = current
    status["live_overlay"] = {
        "schema_version": "mft-goal-live-ui-overlay-v1",
        "updated_at": now,
        "scheduler_access": "GET_only",
        "scheduler_mutation_performed": False,
        "nsga_task_count": len(nsga_rows),
        "fea_task_count": len(fea_rows),
        "fea_task_prefix": fea_prefix,
        "extra_fea_task_prefixes": list(extra_fea_prefixes),
        "core_rescue_plan": str(core_rescue_plan),
    }
    _atomic_json(status_file, status)
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument(
        "--scheduler-url", default="http://127.0.0.1:8002"
    )
    parser.add_argument("--core-rescue-plan", type=Path, required=True)
    parser.add_argument("--fea-prefix", default=FEA_PREFIX)
    parser.add_argument("--extra-fea-prefix", action="append", default=[])
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.pid_file is not None:
        _atomic_json(
            args.pid_file,
            {
                "schema_version": "mft-goal-live-ui-overlay-pid-v1",
                "pid": os.getpid(),
                "started_at": _now(),
                "status_file": str(args.status_file),
                "scheduler_access": "GET_only",
                "scheduler_mutation_performed": False,
            },
        )
    while True:
        update(
            status_file=args.status_file,
            scheduler_url=args.scheduler_url,
            core_rescue_plan=args.core_rescue_plan,
            fea_prefix=args.fea_prefix,
            extra_fea_prefixes=args.extra_fea_prefix,
        )
        if not args.watch:
            return 0
        time.sleep(max(5.0, float(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
