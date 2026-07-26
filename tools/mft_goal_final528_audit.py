"""Independently audit and visualize the authoritative final-528 NSGA result.

The collector is the source of record.  This tool deliberately recomputes its
coverage, geometry representative selection, and non-dominated fronts from the
sealed final files.  Diagnostic fronts are kept visibly separate from the
hard-feasible production front.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


STATUS_SCHEMA = "mft-goal-fixed-lm2mh-targeted-global-nds-v3"
MANIFEST_SCHEMA = "mft-goal-fixed-lm2mh-targeted-pareto-manifest-v3"
AUDIT_SCHEMA = "mft-goal-final528-independent-audit-v1"
CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-"
    "old16-plus-splittemp512-global-v3"
)
HARD_SPEC_SHA256 = (
    "486418c731af63007915c9dd2034df1c65df80a5543546915aa25334c89d164f"
)
EXPECTED_FILES = (
    "global_terminal_all_seeds_raw.csv",
    "global_terminal_candidates.csv",
    "global_pareto_front.csv",
    "global_conditional_nonthermal_pareto_front.csv",
    "global_minimum_violation_objective_front.csv",
    "fea_acquisition_candidates.csv",
)
DEFAULT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_old16_plus_splittemp512_global_nds_v3"
)
DEFAULT_VISUALIZATION_COPY = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "visualizations"
    / "2026"
    / "07"
    / "26"
    / "019f93ab-d9b6-7ca1-b513-1e26f1f8d489"
    / "global-pareto-audit.html"
)
FLOAT_TOLERANCE = 1e-12
KST = timezone(timedelta(hours=9), "KST")


class AuditError(RuntimeError):
    """Raised when final evidence is incomplete or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha(value: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sealed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON evidence is not an object: {path}")
    if (
        value.get("schema_version") != schema
        or value.get("payload_sha256") != _payload_sha(value)
    ):
        raise AuditError(f"sealed evidence drifted: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
        Path(staged).replace(path)
    finally:
        candidate = Path(staged)
        if candidate.exists():
            candidate.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AuditError(f"invalid boolean value: {value!r}")


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"invalid numeric column {key}") from exc
    if not math.isfinite(value):
        raise AuditError(f"non-finite numeric column {key}")
    return value


def _resolve_manifest_file(
    root: Path, name: str, record: Mapping[str, Any]
) -> Path:
    if set(record) != {"path", "row_count", "sha256"}:
        raise AuditError(f"manifest file record drifted: {name}")
    expected = (root / name).resolve()
    try:
        observed = Path(str(record["path"])).resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"manifest file is missing: {name}") from exc
    if observed != expected or observed.parent != root:
        raise AuditError(f"manifest file escapes authoritative root: {name}")
    if _file_sha(observed) != str(record["sha256"]):
        raise AuditError(f"manifest file SHA256 mismatch: {name}")
    return observed


def _dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    keys: Sequence[str],
) -> bool:
    values = [
        (_finite_float(left, key), _finite_float(right, key)) for key in keys
    ]
    return all(a <= b + FLOAT_TOLERANCE for a, b in values) and any(
        a < b - FLOAT_TOLERANCE for a, b in values
    )


def non_dominated(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            *(_finite_float(row, key) for key in keys),
            str(row["physical_geometry_sha256"]),
        ),
    )
    front: list[dict[str, Any]] = []
    for row in ordered:
        if any(_dominates(prior, row, keys) for prior in front):
            continue
        front = [
            prior for prior in front if not _dominates(row, prior, keys)
        ]
        front.append(row)
    return sorted(
        front,
        key=lambda row: (
            _finite_float(row, "objective_volume_L"),
            _finite_float(row, "objective_total_loss_W"),
            str(row["physical_geometry_sha256"]),
        ),
    )


def _read_small_csv(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _front_hashes(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(row["physical_geometry_sha256"]) for row in rows]


def _require_exact_front(
    label: str,
    observed: list[dict[str, str]],
    expected: list[dict[str, Any]],
) -> None:
    if _front_hashes(observed) != _front_hashes(expected):
        raise AuditError(f"{label} is not the exact recomputed front")


def _raw_coverage_and_representatives(
    path: Path,
    *,
    collection_seed_by_task: Mapping[int, int],
    terminal_rows_per_seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    required = {
        "terminal_population_index",
        "physical_geometry_sha256",
        "objective_volume_L",
        "objective_total_loss_W",
        "decoded_physical_params_json",
        "source_seed",
        "scheduler_task_id",
        "fixed_primary_turns_stratum",
        "normalized_constraint_violation_l2",
        "global_screening_feasible",
        "global_production_eligible",
    }
    task_indices: dict[int, set[int]] = {}
    task_seeds: dict[int, set[int]] = {}
    seed_counts: dict[int, int] = {}
    representatives: dict[str, dict[str, Any]] = {}
    row_count = 0
    cw1_values: set[float] = set()
    gap1_values: set[float] = set()
    n1_values: set[int] = set()
    last_order: tuple[int, int] | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise AuditError("raw terminal table columns drifted")
        for row in reader:
            row_count += 1
            try:
                task_id = int(row["scheduler_task_id"])
                seed = int(row["source_seed"])
                population_index = int(row["terminal_population_index"])
                turns = int(row["fixed_primary_turns_stratum"])
                params = json.loads(row["decoded_physical_params_json"])
                cw1 = float(params["cw1"])
                gap1 = float(params["gap1"])
                decoded_turns = int(params["N1"])
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise AuditError("raw terminal identity/control drifted") from exc
            order = (task_id, population_index)
            if last_order is not None and order <= last_order:
                raise AuditError("raw terminal rows are not strictly task/index sorted")
            last_order = order
            if collection_seed_by_task.get(task_id) != seed:
                raise AuditError(
                    f"raw task/seed is not selected collection: {task_id}/{seed}"
                )
            if turns != decoded_turns or turns not in {5, 6, 7, 8}:
                raise AuditError(f"raw primary-turn stratum drifted: task {task_id}")
            if cw1 != 5.0 or gap1 != 1.6:
                raise AuditError(f"fixed 5T/1.6mm control drifted: task {task_id}")
            if _truth(row["global_production_eligible"]):
                raise AuditError("screening row was marked production eligible")
            physical_hash = str(row["physical_geometry_sha256"]).lower()
            if (
                len(physical_hash) != 64
                or any(character not in "0123456789abcdef" for character in physical_hash)
            ):
                raise AuditError("invalid physical geometry SHA256")
            task_indices.setdefault(task_id, set()).add(population_index)
            task_seeds.setdefault(task_id, set()).add(seed)
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
            cw1_values.add(cw1)
            gap1_values.add(gap1)
            n1_values.add(turns)
            compact = {
                "physical_geometry_sha256": physical_hash,
                "objective_volume_L": _finite_float(
                    row, "objective_volume_L"
                ),
                "objective_total_loss_W": _finite_float(
                    row, "objective_total_loss_W"
                ),
                "normalized_constraint_violation_l2": _finite_float(
                    row, "normalized_constraint_violation_l2"
                ),
                "scheduler_task_id": task_id,
                "terminal_population_index": population_index,
                "global_screening_feasible": _truth(
                    row["global_screening_feasible"]
                ),
            }
            prior = representatives.get(physical_hash)
            rank = (
                compact["normalized_constraint_violation_l2"],
                compact["objective_volume_L"],
                compact["objective_total_loss_W"],
                task_id,
            )
            if prior is None or rank < (
                prior["normalized_constraint_violation_l2"],
                prior["objective_volume_L"],
                prior["objective_total_loss_W"],
                prior["scheduler_task_id"],
            ):
                representatives[physical_hash] = compact
    expected_indices = set(range(terminal_rows_per_seed))
    if (
        set(task_indices) != set(collection_seed_by_task)
        or any(indices != expected_indices for indices in task_indices.values())
        or any(len(seeds) != 1 for seeds in task_seeds.values())
        or any(count != terminal_rows_per_seed for count in seed_counts.values())
        or set(seed_counts) != set(collection_seed_by_task.values())
    ):
        raise AuditError("raw terminal per-task/per-seed coverage drifted")
    return (
        {
            "row_count": row_count,
            "logical_seed_count": len(seed_counts),
            "selected_task_count": len(task_indices),
            "unique_geometry_count": len(representatives),
            "cw1_unique_mm": sorted(cw1_values),
            "gap1_unique_mm": sorted(gap1_values),
            "N1_strata": sorted(n1_values),
        },
        representatives,
    )


def _candidate_rows(
    path: Path,
    representatives: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows = _read_small_csv(path)
    required = {
        "physical_geometry_sha256",
        "objective_volume_L",
        "objective_total_loss_W",
        "normalized_constraint_violation_l2",
        "scheduler_task_id",
        "terminal_population_index",
        "global_screening_feasible",
        "conditional_nonthermal_feasible_diagnostic",
        "global_production_eligible",
        "fixed_primary_turns_stratum",
        "exterior_W_drawing_x_mm",
        "exterior_L_perpendicular_y_mm",
        "exterior_H_mm",
        "resonance_min_Hz_fixed_lm2mh",
        "primary_winding_robust_max_C_screening",
        "secondary_winding_robust_max_C_screening",
        "core_robust_max_C_screening",
    }
    if rows and not required.issubset(rows[0]):
        raise AuditError("global candidate columns drifted")
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        physical_hash = str(row["physical_geometry_sha256"])
        if physical_hash in by_hash:
            raise AuditError("global candidates contain duplicate geometry")
        by_hash[physical_hash] = row
        if _truth(row["global_production_eligible"]):
            raise AuditError("global candidate was marked production eligible")
    if set(by_hash) != set(representatives):
        raise AuditError("geometry deduplication coverage does not match raw rows")
    for physical_hash, row in by_hash.items():
        expected = representatives[physical_hash]
        for key in (
            "objective_volume_L",
            "objective_total_loss_W",
            "normalized_constraint_violation_l2",
        ):
            if not math.isclose(
                _finite_float(row, key),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AuditError(
                    f"geometry representative numeric drifted: {physical_hash}"
                )
        if (
            int(row["scheduler_task_id"])
            != int(expected["scheduler_task_id"])
            or int(row["terminal_population_index"])
            != int(expected["terminal_population_index"])
            or _truth(row["global_screening_feasible"])
            != bool(expected["global_screening_feasible"])
        ):
            raise AuditError(
                f"geometry representative identity drifted: {physical_hash}"
            )
    return rows


def _row_count(path: Path) -> int:
    if path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _format_point(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "h": str(row["physical_geometry_sha256"])[:12],
        "v": round(_finite_float(row, "objective_volume_L"), 6),
        "l": round(_finite_float(row, "objective_total_loss_W"), 6),
        "c": round(
            _finite_float(row, "normalized_constraint_violation_l2"), 9
        ),
        "n1": int(row["fixed_primary_turns_stratum"]),
        "w": round(_finite_float(row, "exterior_W_drawing_x_mm"), 3),
        "d": round(
            _finite_float(row, "exterior_L_perpendicular_y_mm"), 3
        ),
        "z": round(_finite_float(row, "exterior_H_mm"), 3),
        "f": round(
            _finite_float(row, "resonance_min_Hz_fixed_lm2mh"), 3
        ),
        "tp": round(
            _finite_float(
                row, "primary_winding_robust_max_C_screening"
            ),
            3,
        ),
        "ts": round(
            _finite_float(
                row, "secondary_winding_robust_max_C_screening"
            ),
            3,
        ),
        "tc": round(
            _finite_float(row, "core_robust_max_C_screening"), 3
        ),
    }


def _render_html(
    *,
    audit_core_sha256: str,
    status_sha256: str,
    manifest_sha256: str,
    candidates: list[dict[str, str]],
    production_front: list[dict[str, str]],
    minimum_violation_front: list[dict[str, str]],
    objective_only_front: list[dict[str, Any]],
    acquisition: list[dict[str, str]],
    generated_at: str,
) -> str:
    plot = [_format_point(row) for row in candidates]
    production = [_format_point(row) for row in production_front]
    minimum = [_format_point(row) for row in minimum_violation_front]
    objective = [_format_point(row) for row in objective_only_front]
    acquisition_points = [_format_point(row) for row in acquisition]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{index}</td><td><code>{html.escape(point['h'])}</code></td>"
        f"<td>{point['n1']}</td><td>{point['w']:.1f}×{point['d']:.1f}×{point['z']:.1f}</td>"
        f"<td>{point['v']:.2f}</td><td>{point['l']:.1f}</td>"
        f"<td>{point['f']:.1f}</td><td>{point['tp']:.1f}/{point['ts']:.1f}/{point['tc']:.1f}</td>"
        f"<td>{point['c']:.5f}</td></tr>"
        for index, point in enumerate(minimum[:25], start=1)
    )
    if not table_rows:
        table_rows = '<tr><td colspan="9">최소위반 Front가 비어 있습니다.</td></tr>'
    data = json.dumps(
        {
            "all": plot,
            "production": production,
            "minimum": minimum,
            "objective": objective,
            "acquisition": acquisition_points,
        },
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    production_message = (
        "하드 제약을 모두 통과한 설계가 없어 생산 판정용 Front는 비어 있습니다."
        if not production
        else "하드 제약 통과 후보만으로 계산한 생산 판정용 Front입니다."
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFT final528 global Pareto audit</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2e;--line:#243854;--text:#eef5ff;--muted:#9eb0c8;--cyan:#22d3ee;--amber:#f59e0b;--rose:#fb7185;--green:#34d399}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#143358 0,#07111f 42%);color:var(--text);font:14px/1.5 Inter,Segoe UI,sans-serif}}
main{{max-width:1480px;margin:auto;padding:28px}}h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:18px;margin:0 0 12px}}.sub{{color:var(--muted);margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px}}.metric,.panel{{background:rgba(13,27,46,.94);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 14px 40px #0005}}
.metric b{{font-size:25px;display:block}}.metric span{{color:var(--muted)}}.danger{{border-color:#7f1d1d;background:#2b1119}}.danger b{{color:var(--rose)}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(290px,1fr);gap:14px;margin-top:14px}}canvas{{width:100%;height:520px;background:#081524;border-radius:10px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin-top:8px}}.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}}
.notice{{border-left:4px solid var(--amber);padding:11px 12px;background:#241b0b;margin:0 0 10px}}.ok{{border-color:var(--green);background:#0c241d}}
code{{color:#b8e7ff}}.seal{{font-size:11px;word-break:break-all;color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:nth-child(2),td:nth-child(2){{text-align:left}}.table-wrap{{overflow:auto;max-height:520px}}
@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}.layout{{grid-template-columns:1fr}}}}@media(max-width:560px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>final528 전역 Pareto 독립 감사</h1>
<div class="sub">16 legacy + 512 fresh seed · 5T / gap 1.6 mm 고정 · 생성 {html.escape(generated_at)}</div>
<section class="grid">
<div class="metric"><b>528 / 528</b><span>인증 logical seeds</span></div>
<div class="metric"><b>168,960</b><span>전체 terminal rows</span></div>
<div class="metric"><b>{len(candidates):,}</b><span>geometry-deduplicated</span></div>
<div class="metric danger"><b>{len(production):,}</b><span>hard-feasible production Front</span></div>
<div class="metric"><b>{len(minimum):,}</b><span>최소위반 3목적 감사 Front</span></div>
</section>
<section class="layout">
<div class="panel"><h2>Volume–Loss 전역 지도</h2><canvas id="plot"></canvas>
<div class="legend"><span><i class="dot" style="background:#64748b"></i>전체 dedup 후보</span><span><i class="dot" style="background:#22d3ee"></i>제약 무시 2목적 NDS ({len(objective)})</span><span><i class="dot" style="background:#f59e0b"></i>최소위반 3목적 Front ({len(minimum)})</span><span><i class="dot" style="background:#fb7185"></i>FEA acquisition ({len(acquisition_points)})</span></div></div>
<aside class="panel"><h2>판정 경계</h2>
<p class="notice">{html.escape(production_message)}</p>
<p class="notice">주황색 최소위반 Front와 청록색 제약 무시 NDS는 <b>진단용</b>입니다. feasible 또는 production Pareto로 승격할 수 없습니다.</p>
<p>생산 Front는 크기·15 kHz·1차 100°C·2차 120°C·코어 120°C를 포함한 모든 screening 제약을 통과한 후보에 대해서만 volume/loss NDS를 계산합니다.</p>
<p>열 값은 surrogate screening이며 최종 생산 판정에는 대칭 FEA gate가 별도로 필요합니다.</p>
<h2>Seal</h2>
<div class="seal">collector<br>{html.escape(status_sha256)}<br><br>manifest<br>{html.escape(manifest_sha256)}<br><br>independent audit core<br>{html.escape(audit_core_sha256)}</div>
</aside></section>
<section class="panel" style="margin-top:14px"><h2>최소위반 3목적 Front 상위 25개</h2><div class="table-wrap"><table><thead><tr><th>#</th><th>geometry</th><th>N1</th><th>W×L×H mm</th><th>Volume L</th><th>Loss W</th><th>fmin Hz</th><th>Tpri/Tsec/Tcore °C</th><th>violation L2</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
</main>
<script>
const D={data};const c=document.getElementById("plot"),x=c.getContext("2d");
function draw(){{const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;x.setTransform(d,0,0,d,0,0);const W=r.width,H=r.height,p={{l:65,r:24,t:22,b:48}};const all=D.all;if(!all.length)return;const xs=all.map(q=>q.v),ys=all.map(q=>q.l),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const X=v=>p.l+(v-xmin)/(xmax-xmin||1)*(W-p.l-p.r),Y=v=>H-p.b-(v-ymin)/(ymax-ymin||1)*(H-p.t-p.b);x.strokeStyle="#243854";x.fillStyle="#9eb0c8";x.font="12px Segoe UI";for(let i=0;i<=5;i++){{let xx=p.l+i*(W-p.l-p.r)/5,yy=p.t+i*(H-p.t-p.b)/5;x.beginPath();x.moveTo(xx,p.t);x.lineTo(xx,H-p.b);x.moveTo(p.l,yy);x.lineTo(W-p.r,yy);x.stroke();x.fillText((xmin+i*(xmax-xmin)/5).toFixed(0),xx-12,H-20);x.fillText((ymax-i*(ymax-ymin)/5).toFixed(0),8,yy+4)}}function dots(a,color,size,alpha){{x.globalAlpha=alpha;x.fillStyle=color;for(const q of a){{x.beginPath();x.arc(X(q.v),Y(q.l),size,0,Math.PI*2);x.fill()}}x.globalAlpha=1}}dots(D.all,"#64748b",2,.35);const line=[...D.objective].sort((a,b)=>a.v-b.v);x.strokeStyle="#22d3ee";x.lineWidth=2;x.beginPath();line.forEach((q,i)=>i?x.lineTo(X(q.v),Y(q.l)):x.moveTo(X(q.v),Y(q.l)));x.stroke();dots(D.objective,"#22d3ee",3,1);dots(D.minimum,"#f59e0b",4,1);dots(D.acquisition,"#fb7185",5,1);x.fillStyle="#eef5ff";x.fillText("Objective volume [L]",W/2-55,H-5);x.save();x.translate(14,H/2+45);x.rotate(-Math.PI/2);x.fillText("Objective total loss [W]",0,0);x.restore()}}addEventListener("resize",draw);draw();
</script></body></html>
"""


def audit_final528(
    root: Path,
    *,
    html_output: Path | None = None,
    audit_output: Path | None = None,
    visualization_copy: Path | None = None,
    expected_seed_count: int = 528,
    expected_legacy_seed_count: int = 16,
    expected_fresh_seed_count: int = 512,
    terminal_rows_per_seed: int = 320,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    status_path = root / "collector_status.json"
    manifest_path = root / "global_pareto_manifest.json"
    status = _read_sealed(status_path, STATUS_SCHEMA)
    manifest = _read_sealed(manifest_path, MANIFEST_SCHEMA)
    expected_raw_rows = expected_seed_count * terminal_rows_per_seed
    if (
        status.get("campaign_id") != CAMPAIGN_ID
        or status.get("aggregate_hard_spec_sha256") != HARD_SPEC_SHA256
        or status.get("global_nds_final") is not True
        or status.get("final_files_written") is not True
        or status.get("production_eligible") is not False
        or status.get("successful_terminal_seed_count") != expected_seed_count
        or status.get("raw_terminal_row_count") != expected_raw_rows
        or (status.get("status_counts") or {}).get("completed")
        != expected_seed_count
        or status.get("failed_terminal_tasks")
    ):
        raise AuditError("collector is not authoritative final coverage")
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("aggregate_hard_spec_sha256") != HARD_SPEC_SHA256
        or manifest.get("source_seed_count") != expected_seed_count
        or manifest.get("legacy_source_seed_count")
        != expected_legacy_seed_count
        or manifest.get("fresh_source_seed_count")
        != expected_fresh_seed_count
        or manifest.get("source_raw_terminal_row_count") != expected_raw_rows
        or manifest.get("global_non_dominated_sorting_complete") is not True
        or manifest.get("screening_only") is not True
        or manifest.get("production_eligible") is not False
        or status.get("pareto_manifest_payload_sha256")
        != manifest.get("payload_sha256")
    ):
        raise AuditError("final Pareto manifest identity/coverage drifted")
    collections = list(status.get("collections") or [])
    task_status = list(status.get("task_status") or [])
    if len(collections) != expected_seed_count or len(task_status) != expected_seed_count:
        raise AuditError("selected collection/task coverage drifted")
    collection_seed_by_task: dict[int, int] = {}
    collection_campaign_counts: dict[str, int] = {}
    for collection in collections:
        task_id = int(collection["task_id"])
        seed = int(collection["seed"])
        if task_id in collection_seed_by_task or seed in collection_seed_by_task.values():
            raise AuditError("selected task/seed is duplicated")
        collection_seed_by_task[task_id] = seed
        campaign = str(collection["source_campaign_id"])
        collection_campaign_counts[campaign] = (
            collection_campaign_counts.get(campaign, 0) + 1
        )
        if (
            int(collection["fixed_primary_turns"]) not in {5, 6, 7, 8}
            or not collection.get("csv_sha256")
        ):
            raise AuditError("selected collection controls drifted")
    status_tasks = {
        int(record["task_id"]): str(record["status"]) for record in task_status
    }
    if (
        set(status_tasks) != set(collection_seed_by_task)
        or set(status_tasks.values()) != {"completed"}
    ):
        raise AuditError("selected task status is not exactly completed528")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise AuditError("final manifest file inventory drifted")
    paths = {
        name: _resolve_manifest_file(root, name, files[name])
        for name in EXPECTED_FILES
    }
    raw_coverage, representatives = _raw_coverage_and_representatives(
        paths["global_terminal_all_seeds_raw.csv"],
        collection_seed_by_task=collection_seed_by_task,
        terminal_rows_per_seed=terminal_rows_per_seed,
    )
    if (
        raw_coverage["row_count"] != expected_raw_rows
        or raw_coverage["logical_seed_count"] != expected_seed_count
        or raw_coverage["selected_task_count"] != expected_seed_count
        or raw_coverage["cw1_unique_mm"] != [5.0]
        or raw_coverage["gap1_unique_mm"] != [1.6]
        or raw_coverage["N1_strata"] != [5, 6, 7, 8]
    ):
        raise AuditError("independent raw coverage/control audit failed")
    candidates = _candidate_rows(
        paths["global_terminal_candidates.csv"], representatives
    )
    feasible = [
        row for row in candidates if _truth(row["global_screening_feasible"])
    ]
    conditional = [
        row
        for row in candidates
        if _truth(row["conditional_nonthermal_feasible_diagnostic"])
    ]
    production_expected = non_dominated(
        feasible, ("objective_volume_L", "objective_total_loss_W")
    )
    conditional_expected = non_dominated(
        conditional, ("objective_volume_L", "objective_total_loss_W")
    )
    minimum_expected = non_dominated(
        candidates,
        (
            "normalized_constraint_violation_l2",
            "objective_volume_L",
            "objective_total_loss_W",
        ),
    )
    objective_only = non_dominated(
        candidates, ("objective_volume_L", "objective_total_loss_W")
    )
    production = _read_small_csv(paths["global_pareto_front.csv"])
    conditional_front = _read_small_csv(
        paths["global_conditional_nonthermal_pareto_front.csv"]
    )
    minimum = _read_small_csv(
        paths["global_minimum_violation_objective_front.csv"]
    )
    acquisition = _read_small_csv(paths["fea_acquisition_candidates.csv"])
    _require_exact_front("hard-feasible production front", production, production_expected)
    _require_exact_front("conditional diagnostic front", conditional_front, conditional_expected)
    _require_exact_front("minimum-violation diagnostic front", minimum, minimum_expected)
    if (
        len(acquisition) != int(manifest["fea_acquisition_candidate_count"])
        or len(set(_front_hashes(acquisition))) != len(acquisition)
        or not set(_front_hashes(acquisition)).issubset(
            set(_front_hashes(candidates))
        )
    ):
        raise AuditError("FEA acquisition candidate inventory drifted")
    expected_counts = {
        "global_terminal_all_seeds_raw.csv": expected_raw_rows,
        "global_terminal_candidates.csv": len(candidates),
        "global_pareto_front.csv": len(production),
        "global_conditional_nonthermal_pareto_front.csv": len(conditional_front),
        "global_minimum_violation_objective_front.csv": len(minimum),
        "fea_acquisition_candidates.csv": len(acquisition),
    }
    for name, path in paths.items():
        observed_rows = (
            raw_coverage["row_count"]
            if name == "global_terminal_all_seeds_raw.csv"
            else _row_count(path)
        )
        if (
            observed_rows != expected_counts[name]
            or observed_rows != int(files[name]["row_count"])
        ):
            raise AuditError(f"final file row count drifted: {name}")
    if (
        len(candidates) != int(manifest["geometry_deduplicated_candidate_count"])
        or len(feasible) != int(manifest["global_screening_feasible_count"])
        or len(production) != int(manifest["global_pareto_count"])
        or len(conditional_front)
        != int(manifest["conditional_nonthermal_pareto_count"])
        or len(minimum)
        != int(manifest["minimum_violation_objective_front_count"])
    ):
        raise AuditError("manifest/result counts drifted")
    generated_at = datetime.now(KST).isoformat(timespec="seconds")
    core = {
        "schema_version": AUDIT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at_kst": generated_at,
        "classification": "independent-final528-audit",
        "global_all_row_aggregation_verified": True,
        "global_non_dominated_sorting_recomputed": True,
        "selected_logical_seed_count": expected_seed_count,
        "legacy_seed_count": expected_legacy_seed_count,
        "fresh_seed_count": expected_fresh_seed_count,
        "selected_task_count": expected_seed_count,
        "raw_terminal_row_count": expected_raw_rows,
        "terminal_rows_per_seed": terminal_rows_per_seed,
        "geometry_deduplicated_candidate_count": len(candidates),
        "hard_feasible_candidate_count": len(feasible),
        "production_hard_feasible_pareto_count": len(production),
        "conditional_nonthermal_diagnostic_front_count": len(conditional_front),
        "minimum_violation_diagnostic_front_count": len(minimum),
        "objective_only_all_candidate_diagnostic_front_count": len(objective_only),
        "fea_acquisition_candidate_count": len(acquisition),
        "fixed_controls": {
            "primary_conductor_thickness_mm": 5.0,
            "primary_interturn_gap_mm": 1.6,
            "N1_strata": raw_coverage["N1_strata"],
        },
        "front_truth_boundary": {
            "production_front": (
                "hard-feasible rows only; exact volume/loss NDS"
            ),
            "minimum_violation_front": (
                "diagnostic only; not feasible and not production"
            ),
            "objective_only_all_candidate_front": (
                "diagnostic only; constraints intentionally ignored"
            ),
            "screening_thermal_requires_symmetric_fea_gate": True,
        },
        "collector_status_payload_sha256": status["payload_sha256"],
        "pareto_manifest_payload_sha256": manifest["payload_sha256"],
        "authoritative_files": {
            name: {
                "path": str(paths[name]),
                "sha256": files[name]["sha256"],
                "row_count": expected_counts[name],
            }
            for name in EXPECTED_FILES
        },
        "collection_campaign_counts": dict(
            sorted(collection_campaign_counts.items())
        ),
        "superseded_task_count": len(status.get("superseded_tasks") or []),
        "scheduler_result_json_visibility_pending_count": len(
            status.get("successful_terminal_artifact_pending_tasks") or []
        ),
        "checks": {
            "collector_seal": "PASS",
            "manifest_seal": "PASS",
            "all_file_sha256": "PASS",
            "all_file_row_counts": "PASS",
            "selected_seed_task_uniqueness": "PASS",
            "raw_per_seed_0_to_319": "PASS",
            "raw_fixed_5T_gap1p6": "PASS",
            "geometry_representative_selection": "PASS",
            "production_front_exact": "PASS",
            "conditional_front_exact": "PASS",
            "minimum_violation_front_exact": "PASS",
        },
    }
    audit_core_sha = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    html_output = (
        html_output.resolve()
        if html_output is not None
        else root / "global-pareto-audit.html"
    )
    rendered = _render_html(
        audit_core_sha256=audit_core_sha,
        status_sha256=status["payload_sha256"],
        manifest_sha256=manifest["payload_sha256"],
        candidates=candidates,
        production_front=production,
        minimum_violation_front=minimum,
        objective_only_front=objective_only,
        acquisition=acquisition,
        generated_at=generated_at,
    )
    _atomic_write(html_output, rendered.encode("utf-8"))
    if visualization_copy is not None:
        destination = visualization_copy.resolve()
        _atomic_write(destination, rendered.encode("utf-8"))
    audit = {
        **core,
        "audit_core_sha256": audit_core_sha,
        "html": {
            "path": str(html_output),
            "sha256": _file_sha(html_output),
            "visualization_copy": (
                str(visualization_copy.resolve())
                if visualization_copy is not None
                else None
            ),
        },
    }
    audit["payload_sha256"] = _payload_sha(audit)
    audit_output = (
        audit_output.resolve()
        if audit_output is not None
        else root / "global_pareto_independent_audit.json"
    )
    _atomic_json(audit_output, audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--visualization-copy",
        type=Path,
        default=DEFAULT_VISUALIZATION_COPY,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    audit = audit_final528(
        args.root,
        html_output=args.html_output,
        audit_output=args.audit_output,
        visualization_copy=args.visualization_copy,
    )
    print(
        json.dumps(
            {
                "payload_sha256": audit["payload_sha256"],
                "raw_terminal_row_count": audit["raw_terminal_row_count"],
                "geometry_deduplicated_candidate_count": audit[
                    "geometry_deduplicated_candidate_count"
                ],
                "production_hard_feasible_pareto_count": audit[
                    "production_hard_feasible_pareto_count"
                ],
                "minimum_violation_diagnostic_front_count": audit[
                    "minimum_violation_diagnostic_front_count"
                ],
                "objective_only_all_candidate_diagnostic_front_count": audit[
                    "objective_only_all_candidate_diagnostic_front_count"
                ],
                "html": audit["html"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
