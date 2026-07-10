"""Final candidate ranking, physical gates, and exact-geometry report output."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd


REGRESSION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = REGRESSION_ROOT.parent
import sys
for path in (str(REGRESSION_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from module.input_parameter_260706 import KEYS  # noqa: E402
from optimization.geometry_metrics import bounding_box_lit  # noqa: E402
from quality_contract import validate_record  # noqa: E402


SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "B_limit_T": 1.2,
    "T_limit_C": 100.0,
    "insulation_min_mm": 40.0,
}
TEMPERATURE_COLUMNS = (
    "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
)
INSULATION_COLUMNS = (
    "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y",
    "w1c_w2s_gap_x_actual",
    "w1s_cs_space_x", "cs_w1s_space_y", "h_gap2",
)
SIDE_INSULATION_COLUMNS = ("w2s_w1s_space_x", "w1s_w2s_space_y")


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def physical_spec_reasons(result, candidate=None):
    reasons = []
    try:
        llt = float(result["Llt"])
        if int(float(result["full_model"])) == 0:
            llt *= 2.0
    except (KeyError, TypeError, ValueError, OverflowError):
        llt = math.nan
    if not _finite(llt) or abs(llt - SPEC["Llt_target_uH"]) > SPEC["Llt_tol_uH"]:
        reasons.append("Llt_out_of_spec")
    bmax = result.get("B_max_core")
    if not _finite(bmax) or not 0 <= float(bmax) <= SPEC["B_limit_T"]:
        reasons.append("B_max_out_of_spec")
    for column in TEMPERATURE_COLUMNS:
        # Rx-side is required whenever the candidate has side turns.
        if column == "T_max_Rx_side" and _finite(result.get("N2_side")) \
                and float(result["N2_side"]) <= 0:
            continue
        value = result.get(column)
        if not _finite(value) or float(value) > SPEC["T_limit_C"]:
            reasons.append(f"temperature_out_of_spec:{column}")
    for column in ("P_winding_total", "P_core_total", "P_core_plate_total"):
        value = result.get(column)
        if not _finite(value) or float(value) < 0:
            reasons.append(f"invalid_loss:{column}")
    if candidate is not None:
        columns = list(INSULATION_COLUMNS)
        if _finite(candidate.get("N1_side")) and float(candidate["N1_side"]) > 0:
            columns.extend(SIDE_INSULATION_COLUMNS)
        for column in columns:
            if column not in candidate:
                reasons.append(f"missing_insulation:{column}")
            elif not _finite(candidate[column]) or float(candidate[column]) < SPEC["insulation_min_mm"]:
                reasons.append(f"insulation_below_minimum:{column}")
    return reasons


def _candidate_digest(candidate):
    payload = {
        key: candidate[key]
        for key in KEYS if key in candidate and _finite(candidate[key])
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def collect_standard_pass_candidates(
    al_root, dataset, limit=None, expected_solver_revision=None,
    expected_library_revision=None,
):
    """Return strict standard-FEA pass candidates ordered by recomputed volume."""
    master = pd.read_parquet(dataset)
    records = []
    for round_dir in sorted(Path(al_root).glob("round_[0-9][0-9]")):
        front_path = round_dir / "pareto_front.csv"
        errors_path = round_dir / "verification_errors.csv"
        if not front_path.is_file() or not errors_path.is_file():
            continue
        front = pd.read_csv(front_path)
        errors = pd.read_csv(errors_path)
        for error in errors.to_dict("records"):
            if not _finite(error.get("idx")) or int(float(error["idx"])) >= len(front):
                continue
            index = int(float(error["idx"]))
            task_id = error.get("task_id")
            matches = master[
                pd.to_numeric(master.get("task_id"), errors="coerce").eq(
                    pd.to_numeric(pd.Series([task_id]), errors="coerce").iloc[0]
                )
            ] if "task_id" in master.columns else master.iloc[0:0]
            if matches.empty:
                continue
            result = matches.iloc[-1].to_dict()
            validity = validate_record(
                result,
                expected_solver_revision=expected_solver_revision,
                expected_library_revision=expected_library_revision,
            )
            candidate = front.iloc[index].to_dict()
            spec_reasons = physical_spec_reasons(result, candidate)
            if not validity.full_valid or spec_reasons:
                continue
            try:
                volume = float(bounding_box_lit(front.iloc[index])[0])
            except Exception:
                continue
            if not _finite(volume) or volume <= 0:
                continue
            params = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in candidate.items()
                if key in KEYS and _finite(value)
            }
            records.append({
                "round": int(round_dir.name.split("_")[-1]),
                "index": index,
                "task_id": task_id,
                "volume_L": volume,
                "total_loss_W": float(candidate.get("total_loss_W", math.nan)),
                "params": params,
                "geometry_evidence": {
                    column: candidate.get(column)
                    for column in (*INSULATION_COLUMNS, *SIDE_INSULATION_COLUMNS, "N1_side")
                },
                "candidate_digest": _candidate_digest(params),
                "standard_result": result,
            })
    records.sort(key=lambda item: (item["volume_L"], item["total_loss_W"]))
    unique = []
    seen = set()
    for record in records:
        if record["candidate_digest"] in seen:
            continue
        seen.add(record["candidate_digest"])
        unique.append(record)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _atomic_text(text, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def write_final_artifacts(
    output_root, selected, fine_results, model_quality, solver_revision,
    library_revision, prior_attempts=None,
):
    """Choose the smallest fine pass and atomically write UI/report artifacts."""
    from verify import scheduler_client

    with open(
        REGRESSION_ROOT / "verify" / "profiles" / "fine.json", encoding="utf-8"
    ) as handle:
        fine_profile = json.load(handle)["param_overrides"]
    attempts = list(prior_attempts or [])
    passed = []
    for candidate, result in zip(selected, fine_results):
        reasons = physical_spec_reasons(
            result, candidate.get("geometry_evidence", candidate["params"])
        )
        if not scheduler_client.is_valid_result(
            result,
            expected_revision=solver_revision,
            expected_library_revision=library_revision,
            expected_profile=fine_profile,
        ):
            reasons.insert(0, "strict_fine_contract_failed")
        item = {
            "candidate_digest": candidate["candidate_digest"],
            "volume_L": candidate["volume_L"],
            "passed": not reasons,
            "reasons": reasons,
            "fine_result": result,
            "fine_task_id": candidate.get("fine_task_id"),
            "fine_task_status": candidate.get("fine_task_status"),
            "fine_attempt": candidate.get("fine_attempt", 0),
        }
        attempts.append(item)
        if not reasons:
            passed.append((candidate, result))
    passed.sort(key=lambda pair: pair[0]["volume_L"])
    winner = passed[0] if passed else None
    compatibility_result = None
    if winner:
        compatibility_result = dict(winner[1])
        compatibility_result.update(winner[0].get("geometry_evidence", {}))
        compatibility_result["volume_L"] = winner[0]["volume_L"]
    artifact = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS" if winner else "FAIL",
        "passed": bool(winner),
        "solver_git_revision": solver_revision,
        "pyaedt_library_git_revision": library_revision,
        "model_quality": model_quality,
        "manufacturing_tolerance_policy": (
            "excluded; exact-as-FEA geometry is assumed"
        ),
        "fine_attempts": attempts,
        "candidate_id": winner[0]["candidate_digest"] if winner else None,
        "fine_task_id": winner[0].get("fine_task_id") if winner else None,
        "fine_task_status": winner[0].get("fine_task_status") if winner else None,
        "candidate": winner[0] if winner else None,
        "fine_result": winner[1] if winner else None,
        # Compatibility key consumed by the standalone monitor.
        "result": compatibility_result,
    }
    results_dir = os.path.join(output_root, "verify", "results")
    _atomic_text(
        json.dumps(artifact, indent=1, ensure_ascii=False, default=str),
        os.path.join(results_dir, "final_verification.json"),
    )
    if winner:
        candidate, result = winner
        fine_loss = sum(
            float(result[key]) for key in
            ("P_winding_total", "P_core_total", "P_core_plate_total")
        )
        parameter_rows = "\n".join(
            f"| {key} | {value} |" for key, value in sorted(candidate["params"].items())
        )
        target_quality = model_quality.get("targets", {})
        quality_rows = "\n".join(
            "| {target} | {r2} | {rmse} | {coverage} |".format(
                target=target,
                r2=(details.get("metrics") or {}).get("r2"),
                rmse=(details.get("metrics") or {}).get("rmse"),
                coverage=(details.get("metrics") or {}).get("interval_coverage"),
            )
            for target, details in sorted(target_quality.items())
        )
        report = f"""# MFT 1 MW 최종 검증 보고서

- 판정: PASS
- 최종 부피: {candidate['volume_L']:.6g} L
- fine 총손실: {fine_loss:.6g} W
- NSGA/표준 FEA 출처: round {candidate.get('round')}, Pareto index {candidate.get('index')}, standard task {candidate.get('task_id')}
- fine FEA: task {candidate.get('fine_task_id')}, project {result.get('project_name')}
- Llt(물리): {(float(result['Llt']) if int(float(result['full_model'])) == 1 else 2 * float(result['Llt'])):.6g} uH
- Bmax: {float(result['B_max_core']):.6g} T
- 온도(Tx/Rx-main/Rx-side/core): {result.get('T_max_Tx')} / {result.get('T_max_Rx_main')} / {result.get('T_max_Rx_side')} / {result.get('T_max_core')} °C
- EM 수렴(matrix/loss): pass {result.get('conv_passes_matrix')}/{result.get('conv_passes_loss')}, error {result.get('conv_error_pct_matrix')}/{result.get('conv_error_pct_loss')} %, delta {result.get('conv_delta_pct_matrix')}/{result.get('conv_delta_pct_loss')} %
- thermal 수렴: iteration {result.get('thermal_iterations')}, continuity {result.get('thermal_residual_continuity')}, energy {result.get('thermal_residual_energy')}
- surrogate generation: {model_quality.get('training_run_id')} (strict-full {model_quality.get('strict_full_rows')} rows)
- solver revision: {solver_revision}
- pyaedt_library revision: {library_revision}
- 제작공차는 적용하지 않았으며, 형상은 FEA와 정확히 동일하게 제작된다고 가정했다.

## 최종 설계 파라미터

| 파라미터 | 값 |
|---|---:|
{parameter_rows}

## Surrogate 독립 평가/불확실성

| target | R² | RMSE | 90% interval coverage |
|---|---:|---:|---:|
{quality_rows}
"""
    else:
        report = """# MFT 1 MW 최종 검증 보고서

- 판정: FAIL
- fine FEA에서 사양을 만족한 후보가 없어 최종 설계를 확정하지 않았다.
- 제작공차는 적용하지 않았으며, 형상은 FEA와 정확히 동일하게 제작된다고 가정했다.
"""
    _atomic_text(report, os.path.join(output_root, "final_report.md"))
    return artifact
