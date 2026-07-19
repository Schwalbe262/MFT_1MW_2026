"""Prepare a sealed, non-executing loss/thermal full-FEA acquisition lane.

This lane is intentionally separate from target-only Tier-1 feedback.  It may
propose geometry-valid near candidates for a strict matrix/capacitance/loss/
thermal measurement, but it never changes their production validity and never
submits scheduler work.  A later executor must require separate authorization
and preserve the proposal/dedupe identity.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tier1_resonance_feedback import (  # noqa: E402
    MANIFEST_SCHEMA,
    MODEL_SCHEMA,
    _atomic_json,
    _json_sha,
    _now,
    _read_json,
    _sha256,
)
from tools.tier1_terminal_followup import (  # noqa: E402
    CONSTRAINT_TOLERANCE,
    MEASUREMENT_GEOMETRY_CONSTRAINTS,
    TERMINAL_ANALYSIS_SCHEMA,
)


THERMAL_ACQUISITION_SCHEMA = "mft-tier1-loss-thermal-acquisition-proposal-v1"
THERMAL_PREFIX = "temperature_robust_limit:"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{label} is not finite") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def _authenticate_model(model_dir: Path, expected_manifest_sha: str) -> dict:
    model_dir = model_dir.resolve()
    manifest_path = model_dir / "manifest.json"
    report_path = model_dir / "training_report.json"
    if _sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("terminal analysis/model manifest SHA mismatch")
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or report.get("schema_version") != MODEL_SCHEMA
        or manifest.get("files", {}).get("training_report.json")
        != _sha256(report_path)
        or manifest.get("production_eligible") is not False
        or report.get("production_eligible") is not False
        or report.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("experimental model authentication failed")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha,
        "training_report_path": str(report_path),
        "training_report_sha256": _sha256(report_path),
        "solver_revision": report["solver_revision"],
        "library_revision": report["library_revision"],
    }


def _authenticate_terminal_analysis(path: Path) -> dict:
    path = path.resolve()
    analysis = _read_json(path)
    if (
        analysis.get("schema_version") != TERMINAL_ANALYSIS_SCHEMA
        or analysis.get("production_eligible") is not False
        or analysis.get("automatic_promotion_allowed") is not False
        or analysis.get("fea_submission_approved") is not False
    ):
        raise RuntimeError("terminal analysis fail-closed contract drifted")
    for reference in analysis.get("result_references") or []:
        result_path = Path(str(reference.get("path") or "")).resolve()
        if _sha256(result_path) != reference.get("sha256"):
            raise RuntimeError("terminal result reference SHA mismatch")
    return analysis


def prepare_thermal_acquisition(
    *, terminal_analysis: Path, model_dir: Path, output: Path,
    candidate_limit: int = 4,
) -> dict:
    """Seal a diverse proposal for strict loss/thermal measurements."""

    if not 1 <= int(candidate_limit) <= 32:
        raise RuntimeError("candidate_limit must be between 1 and 32")
    terminal_analysis = terminal_analysis.resolve()
    analysis = _authenticate_terminal_analysis(terminal_analysis)
    model = _authenticate_model(
        model_dir, str(analysis.get("model_manifest_sha256") or "")
    )

    eligible: list[dict] = []
    seen_digests: set[str] = set()
    for row in analysis.get("near_candidates") or []:
        params = row.get("decoded_params")
        digest = str(row.get("decoded_params_sha256") or "")
        if not isinstance(params, dict) or digest != _json_sha(params):
            raise RuntimeError("near-candidate decoded parameter SHA mismatch")
        if digest in seen_digests:
            raise RuntimeError("duplicate near-candidate geometry")
        seen_digests.add(digest)
        constraints = row.get("constraint_G")
        if not isinstance(constraints, dict):
            raise RuntimeError("near-candidate constraint evidence is missing")
        missing = [
            name for name in MEASUREMENT_GEOMETRY_CONSTRAINTS
            if name not in constraints
        ]
        if missing:
            raise RuntimeError(f"near-candidate geometry constraints missing: {missing}")
        geometry_max = max(
            max(_finite(constraints[name], name), 0.0)
            for name in MEASUREMENT_GEOMETRY_CONSTRAINTS
        )
        thermal = {
            name: _finite(value, name)
            for name, value in constraints.items()
            if name.startswith(THERMAL_PREFIX)
        }
        if geometry_max > CONSTRAINT_TOLERANCE or not thermal:
            continue
        positive = {name: max(value, 0.0) for name, value in thermal.items()}
        if not any(value > CONSTRAINT_TOLERANCE for value in positive.values()):
            continue
        eligible.append({
            "row": row,
            "thermal_positive_sum_G": sum(positive.values()),
            "thermal_positive_max_G": max(positive.values()),
            "thermal_positive_count": sum(
                value > CONSTRAINT_TOLERANCE for value in positive.values()
            ),
            "thermal_constraint_G": thermal,
        })

    # Prefer measurements nearest the thermal boundary, then keep distinct
    # optimizer seeds before taking a second point from one seed.
    eligible.sort(key=lambda item: (
        item["thermal_positive_sum_G"],
        item["thermal_positive_count"],
        item["row"]["decoded_params_sha256"],
    ))
    selected: list[dict] = []
    selected_seeds: set[int] = set()
    for prefer_new_seed in (True, False):
        for item in eligible:
            if len(selected) >= int(candidate_limit):
                break
            if item in selected:
                continue
            seed = int(item["row"]["source_seed"])
            if prefer_new_seed and seed in selected_seeds:
                continue
            selected.append(item)
            selected_seeds.add(seed)

    proposals = []
    for rank, item in enumerate(selected, start=1):
        row = item["row"]
        params = copy.deepcopy(row["decoded_params"])
        params.update({
            "matrix_on": 1,
            "cap_on": 1,
            "loss_on": 1,
            "thermal_on": 1,
            "keep_project": 0,
        })
        acquisition_digest = _json_sha({
            "source_decoded_params_sha256": row["decoded_params_sha256"],
            "effective_params": params,
            "solver_revision": model["solver_revision"],
            "library_revision": model["library_revision"],
            "lane": THERMAL_ACQUISITION_SCHEMA,
        })
        name = f"mft-nsgafea-al-{acquisition_digest[:16]}"
        proposals.append({
            "rank": rank,
            "source_seed": int(row["source_seed"]),
            "source_decoded_params_sha256": row["decoded_params_sha256"],
            "acquisition_digest": acquisition_digest,
            "task_name": name,
            "scheduler_dedupe_key": (
                f"mft-thermal-al:{name}:{model['solver_revision']}:"
                f"{model['library_revision']}:{acquisition_digest[:16]}"
            ),
            "effective_params": params,
            "required_solve_families": [
                "matrix", "capacitance", "loss", "thermal"
            ],
            "strict_full_result_contract_required": True,
            "thermal_constraint_G": item["thermal_constraint_G"],
            "thermal_positive_sum_G": item["thermal_positive_sum_G"],
            "thermal_positive_max_G": item["thermal_positive_max_G"],
            "production_eligible": False,
            "automatic_promotion_allowed": False,
        })

    proposal = {
        "schema_version": THERMAL_ACQUISITION_SCHEMA,
        "created_at": _now(),
        "source_terminal_analysis": {
            "path": str(terminal_analysis),
            "sha256": _sha256(terminal_analysis),
        },
        "source_model": model,
        "selection_policy": (
            "geometry-hard-pass_then_nearest-positive-thermal-boundary_"
            "with-source-seed-diversity-v1"
        ),
        "candidate_limit": int(candidate_limit),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_count": len(proposals),
        "candidates": proposals,
        "lane_is_separate_from_target_only_tier1": True,
        "execution_authorized": False,
        "submission_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "proposal.json", proposal)
    return proposal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-analysis", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = prepare_thermal_acquisition(
        terminal_analysis=args.terminal_analysis,
        model_dir=args.model_dir,
        output=args.output,
        candidate_limit=args.candidate_limit,
    )
    print(result["selected_candidate_count"])


if __name__ == "__main__":
    main()
