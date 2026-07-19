"""Authenticated one-candidate Tier-1 active-learning measurement canary.

Production design validity and measurement eligibility are intentionally
separate.  This adapter may submit one geometry-valid Maxwell matrix plus
electrostatic measurement whose surrogate robust constraints are not yet
valid.  It never runs loss or thermal solves and never promotes a model or
candidate.  Scheduler mutation requires the explicit execute token and a
plan-bound authorization file generated from the terminal evidence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.tier1_resonance_feedback import (  # noqa: E402
    EXPECTED_NSGA_REVISION,
    _atomic_json,
    _json_sha,
    _now,
    _read_json,
    _sha256,
)
from tools.tier1_terminal_followup import TERMINAL_ANALYSIS_SCHEMA  # noqa: E402


EXPECTED_ADAPTER_REVISION = "926d8f95e29c252857002e25370be0fba0a194cd"
PREPARED_SCHEMA = "mft-tier1-active-learning-canary-prepared-v1"
CANDIDATE_SCHEMA = "mft-tier1-active-learning-measurement-candidate-v1"
EXECUTE_TOKEN = "ACTIVE_LEARNING_MEASUREMENT_ONLY"


def _git_revision(root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return process.stdout.strip()


def _git_dirty(root: Path) -> bool:
    process = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1"],
        check=True, capture_output=True, text=True,
    )
    return bool(process.stdout.strip())


def _file_reference(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size}


def _assert_reference(reference: dict, label: str) -> Path:
    if not isinstance(reference, dict):
        raise RuntimeError(f"{label} reference is missing")
    path = Path(str(reference.get("path") or "")).resolve()
    if (
        not path.is_file()
        or reference.get("sha256") != _sha256(path)
        or int(reference.get("size", -1)) != path.stat().st_size
    ):
        raise RuntimeError(f"{label} reference authentication failed")
    return path


def _load_adapter(adapter_root: Path):
    adapter_root = adapter_root.resolve()
    if _git_revision(adapter_root) != EXPECTED_ADAPTER_REVISION:
        raise RuntimeError("Tier-1 adapter revision drifted")
    if _git_dirty(adapter_root):
        raise RuntimeError("Tier-1 adapter checkout is dirty")
    regression = adapter_root / "regression_260707"
    for path in (str(adapter_root), str(regression)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(
        "regression_260707.verify.resonance_tier1_submission"
    )


def _load_sidecar_rows(audit_path: Path) -> tuple[dict, list[dict]]:
    audit_path = audit_path.resolve()
    audit = _read_json(audit_path)
    reference = audit.get("sidecar")
    sidecar_path = _assert_reference(reference, "base sidecar")
    rows = []
    with open(sidecar_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError("base sidecar row is not an object")
                rows.append(row)
    if len(rows) != int(reference.get("row_count") or -1):
        raise RuntimeError("base sidecar row count drifted")
    return audit, rows


def _select_measurement_candidate(analysis: dict) -> dict:
    if (
        analysis.get("schema_version") != TERMINAL_ANALYSIS_SCHEMA
        or analysis.get("production_eligible") is not False
        or analysis.get("automatic_promotion_allowed") is not False
        or analysis.get("fea_submission_approved") is not False
        or analysis.get("active_learning_measurement_submission_performed")
        is not False
        or analysis.get(
            "production_validity_and_measurement_eligibility_are_separate"
        ) is not True
    ):
        raise RuntimeError("terminal active-learning analysis is not fail-closed")
    candidate = analysis.get("first_active_learning_measurement_candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("measurement_eligible") is not True
        or candidate.get("measurement_geometry_passed") is not True
        or candidate.get("production_eligible") is not False
        or candidate.get("fea_submission_approved") is not False
        or candidate.get("decoded_params_sha256")
        != _json_sha(candidate.get("decoded_params"))
    ):
        raise RuntimeError("active-learning measurement candidate is invalid")
    return candidate


def _next_candidate_ordinal(rows: list[dict]) -> int:
    ordinals = []
    for row in rows:
        match = re.fullmatch(r"res20-([0-9]{3})", str(row.get("candidate_id") or ""))
        if not match:
            raise RuntimeError("base sidecar candidate ID is invalid")
        ordinals.append(int(match.group(1)))
    ordinal = max(ordinals, default=0) + 1
    if ordinal > 999:
        raise RuntimeError("Tier-1 candidate ID space is exhausted")
    return ordinal


def _hard_candidate_payload(adapter, candidate: dict, profile: dict, ordinal: int):
    decoded = candidate["decoded_params"]
    missing = [key for key in adapter.KEYS if key not in decoded]
    if missing:
        raise RuntimeError(f"decoded candidate is missing Tier-1 inputs: {missing}")
    raw = {key: decoded[key] for key in adapter.KEYS}
    effective = adapter.sc.effective_verification_params(raw, profile)
    valid, frame, errors = adapter.validation_check(
        adapter.create_input_parameter(effective),
        strict=False,
        return_errors=True,
    )
    if not bool(valid) or len(frame) != 1:
        raise RuntimeError(f"measurement candidate geometry is invalid: {errors}")
    row = frame.iloc[0]
    volume_l, dimensions = adapter.bounding_box_lit(row)
    dimensions = [float(value) for value in dimensions]
    b_field = float(adapter.design_analytical_b_field_t(
        row,
        core_lamination_factor=0.85,
        area_basis=adapter.B_AREA_BASIS_GROSS_WITH_LAMINATION,
    ))
    if not math.isfinite(b_field):
        raise RuntimeError("measurement candidate analytical B is not finite")
    payload = {
        "candidate_id": f"res20-{ordinal:03d}",
        "rank": ordinal,
        "decoded_parameters": raw,
        "hard_contract": {
            "cw1_mm": float(row["cw1"]),
            "n_core_group": int(row["n_core_group"]),
            "N1": int(row["N1"]),
            "N2": int(row["N2"]),
            "N2_side": int(row["N2_side"]),
            "analytical_B_T": b_field,
            "box_mm": dimensions,
            "volume_L": float(volume_l),
        },
    }
    return payload


def _assert_no_base_overlap(record: dict, rows: list[dict]) -> None:
    fields = (
        "candidate_id", "candidate_digest", "task_name", "scheduler_dedupe_key",
    )
    for field in fields:
        values = {str(row.get(field) or "") for row in rows}
        if str(record[field]) in values:
            raise RuntimeError(f"active-learning canary overlaps base {field}")


def _build_plan(
    adapter, *, record: dict, terminal_reference: dict,
    candidate_reference: dict, model_reference: dict,
    upstream_reference: dict, profile_reference: dict,
    solver_revision: str, library_revision: str, scheduler_url: str,
    candidate: dict,
) -> dict:
    plan = {
        "schema_version": adapter.PLAN_SCHEMA,
        "created_at": _now(),
        "classification": [
            "EXPERIMENTAL", "ACTIVE-LEARNING-MEASUREMENT",
            "PRODUCTION-INELIGIBLE",
        ],
        "production_eligible": False,
        "source_fea_submission_approved": False,
        "submission_performed": False,
        "automatic_model_or_candidate_promotion": False,
        "source_bundle": {
            "analysis": terminal_reference,
            "candidate_manifest": candidate_reference,
            "model_source": model_reference,
            "upstream_model_source": upstream_reference,
            "analysis_source_revision": EXPECTED_NSGA_REVISION,
            "model_id": model_reference["sha256"],
            "model_lane": "experimental-target-specific",
        },
        "solver_contract": {
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "aedt_release": "2025.2",
            "backend": "standalone",
            "profile": profile_reference,
            "physics": "Maxwell AC matrix plus Maxwell Electrostatic only",
            "output_basis": "full_physical",
            "required_outputs": list(adapter.REQUIRED_OUTPUTS),
        },
        "adapter_contract": adapter._adapter_contract(profile_reference),
        "scheduler_contract": {
            "scheduler_url": adapter.normalize_scheduler_url(scheduler_url),
            "priority": 5,
            "lane_active_cap": 1,
            "project_active_hard_cap": 500,
            "project_config_ceiling": 600,
            "cpus_per_task": 4,
            "memory_mb_per_task": 32768,
            "task_prefix": adapter.TASK_PREFIX,
        },
        "hard_contract": dict(adapter.EXPECTED_HARD_CONTRACT),
        "eligibility_contract": {
            "production_design_valid": False,
            "active_learning_measurement_eligible": True,
            "measurement_geometry_passed": True,
            "surrogate_robust_Llt_passed": False,
            "loss_thermal_validated": False,
            "thermal_acquisition_lane": "separate-full-FEA",
            "measurement_purpose": candidate["measurement_purpose"],
            "source_decoded_params_sha256": candidate["decoded_params_sha256"],
            "source_seed": int(candidate["source_seed"]),
        },
        "authorization_contract": {
            "schema_version": adapter.AUTHORIZATION_SCHEMA,
            "execute_requires_separate_plan_bound_authorization": True,
            "production_promotion_authorized": False,
        },
        "result_contract": {
            "measurement_can_be_verified": True,
            "verification_does_not_imply_fea_or_production_approval": True,
            "matrix_solve_attempts": 1,
            "cap_solve_attempts": 1,
            "loss_solve_attempts": 0,
            "thermal_on": 0,
            "independent_pole_recalculation_required": True,
        },
        "candidate_count": 1,
        "candidates": [record],
    }
    plan["plan_id"] = adapter._canonical_sha256(plan)
    return plan


def prepare_canary(
    *, terminal_analysis_path: Path, base_audit_path: Path,
    model_manifest_path: Path, adapter_root: Path, profile_path: Path,
    runtime: Path, scheduler_url: str, operator: str,
) -> dict:
    runtime = runtime.resolve()
    if runtime.exists():
        raise RuntimeError("active-learning canary runtime already exists")
    terminal_analysis_path = terminal_analysis_path.resolve()
    model_manifest_path = model_manifest_path.resolve()
    terminal = _read_json(terminal_analysis_path)
    candidate = _select_measurement_candidate(terminal)
    model_manifest = _read_json(model_manifest_path)
    if (
        _sha256(model_manifest_path) != terminal.get("model_manifest_sha256")
        or model_manifest.get("production_eligible") is not False
        or model_manifest.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("terminal analysis/model manifest binding failed")
    audit, base_rows = _load_sidecar_rows(base_audit_path)
    solver_revision = str(audit["solver_revision"])
    library_revision = str(audit["library_revision"])
    adapter = _load_adapter(adapter_root)
    profile, profile_reference = adapter.load_profile(profile_path)
    ordinal = _next_candidate_ordinal(base_rows)
    payload = _hard_candidate_payload(adapter, candidate, profile, ordinal)
    record = adapter.verify_candidate(
        payload, profile, solver_revision, library_revision,
    )
    _assert_no_base_overlap(record, base_rows)
    runtime.mkdir(parents=True)
    terminal_reference = _file_reference(terminal_analysis_path)
    model_reference = _file_reference(model_manifest_path)
    upstream_path = model_manifest_path.parent / "training_report.json"
    upstream_reference = _file_reference(upstream_path)
    candidate_manifest = {
        "schema_version": CANDIDATE_SCHEMA,
        "created_at": _now(),
        "classification": [
            "EXPERIMENTAL", "ACTIVE-LEARNING-MEASUREMENT",
            "PRODUCTION-INELIGIBLE",
        ],
        "source_terminal_analysis": terminal_reference,
        "source_model_manifest": model_reference,
        "source_decoded_params_sha256": candidate["decoded_params_sha256"],
        "candidate_count": 1,
        "candidate": payload,
        "scheduler_record": record,
        "measurement_eligible": True,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "automatic_promotion_allowed": False,
    }
    candidate_path = runtime / "measurement_candidate.json"
    _atomic_json(candidate_path, candidate_manifest)
    candidate_reference = _file_reference(candidate_path)
    plan = _build_plan(
        adapter, record=record, terminal_reference=terminal_reference,
        candidate_reference=candidate_reference, model_reference=model_reference,
        upstream_reference=upstream_reference, profile_reference=profile_reference,
        solver_revision=solver_revision, library_revision=library_revision,
        scheduler_url=scheduler_url, candidate=candidate,
    )
    prepared_runtime = adapter.prepare_runtime(runtime, plan)
    plan_reference = prepared_runtime["plan"]
    authorization = {
        "schema_version": adapter.AUTHORIZATION_SCHEMA,
        "authorized_at": _now(),
        "operator": str(operator).strip(),
        "authorization_basis": (
            "parent directive: active-learning measurement eligibility is "
            "separate from production design validity"
        ),
        "measurement_submission_authorized": True,
        "production_promotion_authorized": False,
        "plan_sha256": plan_reference["sha256"],
        "plan_id": plan["plan_id"],
        "candidate_manifest_sha256": candidate_reference["sha256"],
        "analysis_sha256": terminal_reference["sha256"],
        "model_source_sha256": model_reference["sha256"],
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "scheduler_url": adapter.normalize_scheduler_url(scheduler_url),
    }
    if not authorization["operator"]:
        raise RuntimeError("operator identity is required")
    authorization_path = runtime / "OPERATOR_AUTHORIZATION.json"
    _atomic_json(authorization_path, authorization)
    # Instantiate once without contacting the scheduler.  Constructor checks
    # serialized parameter/dedupe identity and plan-bound authorization.
    adapter.Tier1Controller(runtime, authorization_path, scheduler_url)
    seal = {
        "schema_version": PREPARED_SCHEMA,
        "prepared_at": _now(),
        "runtime": str(runtime),
        "terminal_analysis": terminal_reference,
        "base_audit": _file_reference(base_audit_path.resolve()),
        "base_sidecar": audit["sidecar"],
        "model_manifest": model_reference,
        "candidate_manifest": candidate_reference,
        "plan": _file_reference(runtime / "plan.json"),
        "state": _file_reference(runtime / "state.json"),
        "authorization": _file_reference(authorization_path),
        "candidate_id": record["candidate_id"],
        "candidate_digest": record["candidate_digest"],
        "task_name": record["task_name"],
        "scheduler_dedupe_key": record["scheduler_dedupe_key"],
        "base_overlap_count": 0,
        "active_learning_measurement_eligible": True,
        "production_design_valid": False,
        "submission_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }
    _atomic_json(runtime / "ACTIVE_LEARNING_CANARY_PREPARED.json", seal)
    return seal


def run_canary(
    *, runtime: Path, adapter_root: Path, scheduler_url: str,
    execute_token: str, poll_seconds: int, watch: bool,
) -> dict:
    if execute_token != EXECUTE_TOKEN:
        raise RuntimeError("active-learning measurement execute token is absent")
    runtime = runtime.resolve()
    seal = _read_json(runtime / "ACTIVE_LEARNING_CANARY_PREPARED.json")
    if (
        seal.get("schema_version") != PREPARED_SCHEMA
        or seal.get("active_learning_measurement_eligible") is not True
        or seal.get("production_design_valid") is not False
        or seal.get("submission_performed") is not False
        or seal.get("production_eligible") is not False
        or seal.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("active-learning canary prepared seal is invalid")
    for key in (
        "terminal_analysis", "base_audit", "base_sidecar", "model_manifest",
        "candidate_manifest", "plan", "authorization",
    ):
        _assert_reference(seal[key], key)
    adapter = _load_adapter(adapter_root)
    state_path = Path(str(seal["state"].get("path") or "")).resolve()
    if state_path != (runtime / "state.json").resolve() or not state_path.is_file():
        raise RuntimeError("active-learning mutable state path drifted")
    state = _read_json(state_path)
    plan = _read_json(runtime / "plan.json")
    if (
        state.get("schema_version") != adapter.STATE_SCHEMA
        or state.get("plan_sha256") != _sha256(runtime / "plan.json")
        or state.get("plan_id") != plan.get("plan_id")
        or state.get("production_eligible") is not False
        or state.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("active-learning mutable state binding failed")
    controller = adapter.Tier1Controller(
        runtime, seal["authorization"]["path"], scheduler_url,
    )
    result = (
        controller.run_watch(poll_seconds)
        if watch else controller.run_once()
    )
    return {
        "result": result,
        "runtime": str(runtime),
        "production_eligible": False,
        "automatic_promotion_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--terminal-analysis", type=Path, required=True)
    prepare.add_argument("--base-audit", type=Path, required=True)
    prepare.add_argument("--model-manifest", type=Path, required=True)
    prepare.add_argument("--adapter-root", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--runtime", type=Path, required=True)
    prepare.add_argument("--scheduler-url", required=True)
    prepare.add_argument("--operator", required=True)
    run = commands.add_parser("run")
    run.add_argument("--runtime", type=Path, required=True)
    run.add_argument("--adapter-root", type=Path, required=True)
    run.add_argument("--scheduler-url", required=True)
    run.add_argument("--execute-token", required=True)
    run.add_argument("--poll-seconds", type=int, default=15)
    run.add_argument("--watch", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_canary(
            terminal_analysis_path=args.terminal_analysis,
            base_audit_path=args.base_audit,
            model_manifest_path=args.model_manifest,
            adapter_root=args.adapter_root,
            profile_path=args.profile,
            runtime=args.runtime,
            scheduler_url=args.scheduler_url,
            operator=args.operator,
        )
    else:
        if not 5 <= args.poll_seconds <= 300:
            raise SystemExit("--poll-seconds must be between 5 and 300")
        result = run_canary(
            runtime=args.runtime, adapter_root=args.adapter_root,
            scheduler_url=args.scheduler_url,
            execute_token=args.execute_token,
            poll_seconds=args.poll_seconds, watch=args.watch,
        )
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
