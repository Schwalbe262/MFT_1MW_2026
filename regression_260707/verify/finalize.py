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
MIN_STRICT_FULL_ROWS = 3000
QUALITY_THRESHOLDS_PATH = REGRESSION_ROOT / "training" / "model_quality_thresholds.json"
REQUIRED_OPTIMIZATION_MODELS = frozenset({
    "Llt_phys", "k",
    "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
    "B_max_core", "B_mean_core",
    "Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max",
})


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optimization_manifest_reasons(
    round_dir, expected_solver_revision, expected_library_revision,
):
    reasons = []
    manifest_path = Path(round_dir) / "optimization_manifest.json"
    front_path = Path(round_dir) / "pareto_front.csv"
    quality_path = Path(round_dir) / "model_quality_snapshot.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ["optimization_manifest_unavailable"]
    if not manifest.get("quality_gate_passed"):
        reasons.append("optimization_quality_gate_not_passed")
    if int(manifest.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS:
        reasons.append("optimization_below_3000_strict_rows")
    if manifest.get("solver_revision") != expected_solver_revision:
        reasons.append("optimization_solver_revision_mismatch")
    if manifest.get("library_revision") != expected_library_revision:
        reasons.append("optimization_library_revision_mismatch")
    try:
        vetted_thresholds = _sha256(QUALITY_THRESHOLDS_PATH)
    except Exception:
        vetted_thresholds = None
    if (vetted_thresholds is None
            or manifest.get("quality_thresholds_sha256") != vetted_thresholds):
        reasons.append("optimization_quality_thresholds_mismatch")
    for path, key in (
        (front_path, "pareto_front_sha256"),
        (Path(round_dir) / "pareto_X.npy", "pareto_X_sha256"),
        (Path(round_dir) / "pareto_F.npy", "pareto_F_sha256"),
        (quality_path, "quality_status_sha256"),
    ):
        try:
            if _sha256(path) != manifest.get(key):
                reasons.append(f"optimization_artifact_mismatch:{key}")
        except Exception:
            reasons.append(f"optimization_artifact_unavailable:{key}")
    required = set(manifest.get("required_models") or [])
    if required != REQUIRED_OPTIMIZATION_MODELS:
        reasons.append("optimization_required_models_incomplete")
    generation = manifest.get("registry_generation")
    try:
        generation_path = Path(generation).resolve()
        registry_root = (REGRESSION_ROOT / "training" / "registry").resolve()
        if os.path.commonpath([str(generation_path), str(registry_root)]) != str(registry_root):
            raise RuntimeError("generation escapes registry")
        report_path = generation_path / "train_report.json"
        if _sha256(report_path) != manifest.get("generation_report_sha256"):
            reasons.append("optimization_generation_report_mismatch")
        generation_report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_hashes = manifest.get("model_artifacts_sha256") or {}
        for target in required:
            for name in ("models.pkl", "meta.json"):
                if _sha256(generation_path / target / name) != (
                        artifact_hashes.get(target) or {}).get(name):
                    reasons.append(f"optimization_model_artifact_mismatch:{target}:{name}")
    except Exception:
        reasons.append("optimization_generation_unavailable")
        generation_report = {}
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not quality.get("passed"):
            reasons.append("optimization_quality_snapshot_failed")
        if int(quality.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS:
            reasons.append("optimization_quality_snapshot_below_3000")
        for key in (
            "training_run_id", "dataset_sha256", "solver_revision",
            "library_revision", "quality_thresholds_sha256",
        ):
            if quality.get(key) != manifest.get(key):
                reasons.append(f"optimization_quality_identity_mismatch:{key}")
        if set((quality.get("targets") or {}).keys()) != required:
            reasons.append("optimization_quality_targets_incomplete")
        for key in ("training_run_id", "dataset_sha256", "strict_full_rows"):
            if generation_report.get(key) != manifest.get(key):
                reasons.append(f"optimization_generation_identity_mismatch:{key}")
        if generation_report.get("profile_sha256") != manifest.get("profile_sha256"):
            reasons.append("optimization_profile_identity_mismatch")
    except Exception:
        reasons.append("optimization_quality_snapshot_unavailable")
    return list(dict.fromkeys(reasons))


def physical_spec_reasons(result, candidate=None):
    """Evaluate physical specifications from the authoritative FEA result.

    ``candidate`` remains accepted for API compatibility, but candidate/front
    columns are never evidence for an FEA pass.
    """
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
    for column in (
        "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
    ):
        value = result.get(column)
        if not _finite(value) or float(value) < 0:
            reasons.append(f"invalid_loss:{column}")
    columns = list(INSULATION_COLUMNS)
    n1_side = result.get("N1_side")
    if not _finite(n1_side):
        reasons.append("missing_geometry:N1_side")
    elif float(n1_side) > 0:
        columns.extend(SIDE_INSULATION_COLUMNS)
    for column in columns:
        if column not in result:
            reasons.append(f"missing_insulation:{column}")
        elif (not _finite(result[column])
              or float(result[column]) < SPEC["insulation_min_mm"]):
            reasons.append(f"insulation_below_minimum:{column}")
    return reasons


def _candidate_digest(candidate):
    payload = {}
    for key in KEYS:
        if key not in candidate:
            continue
        value = candidate[key]
        if hasattr(value, "item"):
            value = value.item()
        payload[key] = value
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def candidate_identity_reasons(result, candidate, expected_params_key="params"):
    """Authenticate the candidate record and its exact submitted FEA inputs."""
    from verify import scheduler_client

    if not isinstance(candidate, dict):
        return ["candidate_record_missing"]
    params = candidate.get("params")
    if not isinstance(params, dict) or set(params) != set(KEYS):
        return ["candidate_params_missing"]
    reasons = []
    if candidate.get("candidate_digest") != _candidate_digest(params):
        reasons.append("candidate_digest_mismatch")
    expected_params = candidate.get(expected_params_key)
    if not isinstance(expected_params, dict) or set(expected_params) != set(KEYS):
        reasons.append(f"candidate_{expected_params_key}_missing")
    elif not scheduler_client.result_matches_params(
            result, expected_params, required_keys=KEYS):
        reasons.append("candidate_result_identity_mismatch")
    return reasons


def collect_standard_pass_candidates(
    al_root, dataset, limit=None, expected_solver_revision=None,
    expected_library_revision=None,
):
    """Return strict standard-FEA pass candidates ordered by recomputed volume."""
    from verify import scheduler_client

    standard_profile = json.loads(
        (REGRESSION_ROOT / "verify" / "profiles" / "standard.json").read_text(
            encoding="utf-8"
        )
    )
    master = pd.read_parquet(dataset)
    records = []
    for round_dir in sorted(Path(al_root).glob("round_[0-9][0-9]")):
        front_path = round_dir / "pareto_front.csv"
        errors_path = round_dir / "verification_errors.csv"
        if not front_path.is_file() or not errors_path.is_file():
            continue
        if _optimization_manifest_reasons(
                round_dir, expected_solver_revision, expected_library_revision):
            continue
        manifest_path = round_dir / "optimization_manifest.json"
        optimization_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
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
            candidate = front.iloc[index].to_dict()
            params = {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in candidate.items()
                if key in KEYS and _finite(value)
            }
            if set(params) != set(KEYS):
                continue
            standard_params = scheduler_client.effective_verification_params(
                params, standard_profile
            )
            authenticated = {
                "params": params,
                "standard_params": standard_params,
                "candidate_digest": _candidate_digest(params),
            }
            validity = validate_record(
                result,
                expected_solver_revision=expected_solver_revision,
                expected_library_revision=expected_library_revision,
            )
            spec_reasons = physical_spec_reasons(result)
            identity_reasons = candidate_identity_reasons(
                result, authenticated, expected_params_key="standard_params"
            )
            if not validity.full_valid or spec_reasons or identity_reasons:
                continue
            try:
                # The result, not a stale/spoofed Pareto row, is authoritative
                # for the minimum-volume ranking.
                volume = float(bounding_box_lit(result)[0])
            except Exception:
                continue
            if not _finite(volume) or volume <= 0:
                continue
            records.append({
                "round": int(round_dir.name.split("_")[-1]),
                "index": index,
                "task_id": task_id,
                "volume_L": volume,
                "total_loss_W": float(candidate.get("total_loss_W", math.nan)),
                "params": params,
                "standard_params": standard_params,
                "geometry_evidence": {
                    column: result.get(column)
                    for column in (*INSULATION_COLUMNS, *SIDE_INSULATION_COLUMNS, "N1_side")
                },
                "candidate_digest": _candidate_digest(params),
                "standard_result": result,
                "optimization_manifest": optimization_manifest,
                "optimization_manifest_path": str(manifest_path),
                "optimization_manifest_sha256": _sha256(manifest_path),
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


def _model_quality_reasons(model_quality, solver_revision, library_revision):
    reasons = []
    if not isinstance(model_quality, dict) or not model_quality.get("passed"):
        return ["model_quality_gate_not_passed"]
    if int(model_quality.get("strict_full_rows") or 0) < MIN_STRICT_FULL_ROWS:
        reasons.append("model_quality_below_3000_strict_rows")
    if model_quality.get("solver_revision") != solver_revision:
        reasons.append("model_quality_solver_revision_mismatch")
    if model_quality.get("library_revision") != library_revision:
        reasons.append("model_quality_library_revision_mismatch")
    try:
        if model_quality.get("quality_thresholds_sha256") != _sha256(
                QUALITY_THRESHOLDS_PATH):
            reasons.append("model_quality_thresholds_mismatch")
    except Exception:
        reasons.append("model_quality_thresholds_unavailable")
    if set((model_quality.get("targets") or {}).keys()) != REQUIRED_OPTIMIZATION_MODELS:
        reasons.append("model_quality_targets_incomplete")
    for key in ("training_run_id", "dataset_sha256"):
        value = str(model_quality.get(key) or "")
        if not value or (key.endswith("sha256") and (
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value.lower()))):
            reasons.append(f"model_quality_{key}_missing")
    return reasons


def _final_candidate_provenance_reasons(
        candidate, solver_revision, library_revision):
    """Revalidate standard FEA and the complete optimization evidence chain."""
    from verify import scheduler_client

    reasons = []
    if not isinstance(candidate, dict):
        return ["candidate_record_missing"]
    manifest_path_value = candidate.get("optimization_manifest_path")
    try:
        manifest_path = Path(manifest_path_value).resolve()
        if _sha256(manifest_path) != candidate.get("optimization_manifest_sha256"):
            reasons.append("candidate_optimization_manifest_hash_mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != candidate.get("optimization_manifest"):
            reasons.append("candidate_optimization_manifest_content_mismatch")
        reasons.extend(_optimization_manifest_reasons(
            manifest_path.parent, solver_revision, library_revision
        ))
    except Exception:
        reasons.append("candidate_optimization_manifest_unavailable")

    standard = candidate.get("standard_result")
    standard_profile = json.loads(
        (REGRESSION_ROOT / "verify" / "profiles" / "standard.json").read_text(
            encoding="utf-8"
        )
    )["param_overrides"]
    if not scheduler_client.is_valid_result(
            standard,
            expected_revision=solver_revision,
            expected_library_revision=library_revision,
            expected_profile=standard_profile):
        reasons.append("strict_standard_contract_failed")
    reasons.extend(physical_spec_reasons(standard or {}))
    reasons.extend(candidate_identity_reasons(
        standard or {}, candidate, expected_params_key="standard_params"
    ))
    try:
        standard_volume = float(bounding_box_lit(standard)[0])
        recorded_volume = float(candidate["volume_L"])
        if (not _finite(standard_volume) or standard_volume <= 0
                or not math.isclose(
                    standard_volume, recorded_volume, rel_tol=1e-12, abs_tol=1e-9
                )):
            reasons.append("standard_volume_identity_mismatch")
    except Exception:
        reasons.append("standard_volume_unavailable")
    return list(dict.fromkeys(reasons))


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
    quality_reasons = _model_quality_reasons(
        model_quality, solver_revision, library_revision
    )
    passed = []
    for candidate, result in zip(selected, fine_results):
        reasons = list(quality_reasons)
        reasons.extend(_final_candidate_provenance_reasons(
            candidate, solver_revision, library_revision
        ))
        reasons.extend(physical_spec_reasons(result))
        reasons.extend(candidate_identity_reasons(
            result, candidate, expected_params_key="fine_params"
        ))
        if not scheduler_client.is_valid_result(
            result,
            expected_revision=solver_revision,
            expected_library_revision=library_revision,
            expected_profile=fine_profile,
        ):
            reasons.insert(0, "strict_fine_contract_failed")
        try:
            actual_volume = float(bounding_box_lit(result)[0])
            if not _finite(actual_volume) or actual_volume <= 0:
                raise ValueError("nonpositive fine volume")
        except Exception:
            actual_volume = None
            reasons.append("fine_volume_unavailable")
        reasons = list(dict.fromkeys(reasons))
        item = {
            "candidate_digest": candidate["candidate_digest"],
            "volume_L": actual_volume,
            "passed": not reasons,
            "reasons": reasons,
            "fine_result": result,
            "fine_task_id": candidate.get("fine_task_id"),
            "fine_task_status": candidate.get("fine_task_status"),
            "fine_attempt": candidate.get("fine_attempt", 0),
        }
        attempts.append(item)
        if not reasons:
            authoritative_candidate = dict(candidate)
            authoritative_candidate["standard_volume_L"] = candidate.get("volume_L")
            authoritative_candidate["volume_L"] = actual_volume
            passed.append((authoritative_candidate, result))
    passed.sort(key=lambda pair: pair[0]["volume_L"])
    winner = passed[0] if passed else None
    compatibility_result = None
    if winner:
        compatibility_result = dict(winner[1])
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
        "standard_result": winner[0].get("standard_result") if winner else None,
        "optimization_manifest": (
            winner[0].get("optimization_manifest") if winner else None
        ),
        "fine_result": winner[1] if winner else None,
        # Compatibility key consumed by the standalone monitor.
        "result": compatibility_result,
    }
    results_dir = os.path.join(output_root, "verify", "results")
    if winner:
        candidate, result = winner
        standard = candidate.get("standard_result") or {}
        optimization_manifest = candidate.get("optimization_manifest") or {}
        fine_loss = sum(
            float(result[key]) for key in
            ("P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total")
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
- EM 수렴(matrix/loss): total pass {result.get('conv_passes_matrix')}/{result.get('conv_passes_loss')}, consecutive pass {result.get('conv_consecutive_matrix')}/{result.get('conv_consecutive_loss')}, error {result.get('conv_error_pct_matrix')}/{result.get('conv_error_pct_loss')} %, delta {result.get('conv_delta_pct_matrix')}/{result.get('conv_delta_pct_loss')} %
- thermal 수렴: iteration {result.get('thermal_iterations')}, continuity {result.get('thermal_residual_continuity')}, energy {result.get('thermal_residual_energy')}
- surrogate generation: {model_quality.get('training_run_id')} (strict-full {model_quality.get('strict_full_rows')} rows)
- training dataset SHA-256: {model_quality.get('dataset_sha256')}
- quality contract SHA-256: {model_quality.get('quality_thresholds_sha256')}
- NSGA manifest SHA-256: {candidate.get('optimization_manifest_sha256')}
- NSGA restarts/population: {optimization_manifest.get('restarts')} / {optimization_manifest.get('population')}
- standard FEA task/project: {candidate.get('task_id')} / {standard.get('project_name')}
- standard EM total/consecutive pass (matrix/loss): {standard.get('conv_passes_matrix')}/{standard.get('conv_consecutive_matrix')} · {standard.get('conv_passes_loss')}/{standard.get('conv_consecutive_loss')}
- standard extraction (matrix/loss): {standard.get('matrix_extraction_backend')} / {standard.get('loss_extraction_backend')}
- standard thermal power (expected/assigned/max error): {standard.get('thermal_rx_expected_power_w')} / {standard.get('thermal_rx_assigned_power_w')} / {standard.get('thermal_rx_power_balance_max_abs_w')} W
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
    # The JSON is the monitor/automation commit marker.  Publish it only after
    # the human-readable report so a crash cannot expose PASS with an old or
    # missing report.
    _atomic_text(
        json.dumps(artifact, indent=1, ensure_ascii=False, default=str),
        os.path.join(results_dir, "final_verification.json"),
    )
    return artifact
