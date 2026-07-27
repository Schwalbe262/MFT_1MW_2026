"""Submit the tuned 2b213 rank-1 symmetric final verification.

This is a narrow post-gap bridge.  It accepts only the sealed physical-gap
tuning result for geometry ``2b2138a99445...`` and submits one non-rounded
eighth-symmetry Matrix+Cap+Loss+Thermal run under the existing MFT Scheduler
project.  Scheduler mutation requires ``--apply``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import (  # noqa: E402
    create_input_parameter,
    validation_check,
)
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_lm2mh_gap_tuner as tuner  # noqa: E402
from module.core_air_gap_contract import validate_air_gap_contract  # noqa: E402


PLAN_SCHEMA = "mft-goal-tuned-symmetric-final-plan-v1"
RECEIPT_SCHEMA = "mft-goal-tuned-symmetric-final-submission-v1"
EXPECTED_GEOMETRY_SHA256 = tuner.EXPECTED_NEIGHBORHOOD_GEOMETRY_SHA256
EXPECTED_CAMPAIGN_ID = tuner.NEIGHBORHOOD_CAMPAIGN_ID
DEFAULT_ROOT = tuner.DEFAULT_NEIGHBORHOOD_OUTPUT
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_standard.json"
)
SCHEDULER_URL = tuner.SCHEDULER_URL
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
PRIORITY = 100
MAX_WORKERS_PER_NODE = 1
PROJECT_ACTIVE_TASK_CAP = 500


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise tuner.GapTuningError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise tuner.GapTuningError(f"{label} must be finite")
    return number


def _exact(value: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not math.isclose(
            _finite(value, label), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise tuner.GapTuningError(f"{label} drifted")
    elif value != expected:
        raise tuner.GapTuningError(f"{label} drifted")


def _load_inputs(
    campaign_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = campaign_root.resolve(strict=True)
    campaign, loaded_root = tuner._load_campaign(
        root / "campaign_manifest.json"
    )
    if loaded_root != root:
        raise tuner.GapTuningError("campaign root readback drifted")
    final_path = root / "tuned_gap_manifest.json"
    final = tuner._validate_seal(
        tuner._read_json(final_path), tuner.FINAL_SCHEMA
    )
    try:
        validate_air_gap_contract(
            final.get("equal_three_leg_air_gap_contract"),
            require_equal_three_leg=True,
            require_verified=True,
        )
    except ValueError as exc:
        raise tuner.GapTuningError(
            "legacy center-only tuned artifacts are diagnostic-only"
        ) from exc
    if (
        campaign.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or campaign.get("candidate", {}).get("physical_geometry_sha256")
        != EXPECTED_GEOMETRY_SHA256
        or final.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or final.get("campaign_payload_sha256")
        != campaign["payload_sha256"]
        or final.get("source_candidate", {}).get(
            "physical_geometry_sha256"
        )
        != EXPECTED_GEOMETRY_SHA256
        or final.get("physical_gap_geometry_attested") is not True
        or final.get("symmetric_matrix_convergence_attested") is not True
        or final.get("native_L11_L22_M_k_Lm_readback_attested") is not True
        or final.get("core_equal_three_leg_air_gap") != 1
        or final.get("equal_three_leg_air_gap_FEA_required") is not True
        or final.get("physical_Lm_2mH_symmetric_FEA_verified") is not True
    ):
        raise tuner.GapTuningError("tuned 2b213 authority drifted")
    gap = _finite(final.get("tuned_core_center_gap_mm"), "tuned gap")
    lm_h = _finite(final.get("tuned_Lm_primary_referred_H"), "tuned Lm")
    if (
        not 0.0 < gap <= tuner.MAX_CENTER_GAP_MM
        or abs(lm_h - tuner.TARGET_LM_H) > tuner.LM_ABS_TOLERANCE_H
    ):
        raise tuner.GapTuningError("tuned gap is outside the terminal contract")

    record = final.get("downstream_params", {}).get(
        "symmetric_loss_thermal"
    )
    if not isinstance(record, dict):
        raise tuner.GapTuningError("symmetric downstream params are absent")
    params_path = (root / str(record.get("path") or "")).resolve(strict=True)
    if tuner._file_record(params_path, relative_to=root) != record:
        raise tuner.GapTuningError("symmetric downstream params bytes drifted")
    params = tuner._read_json(params_path)
    required = {
        "core_center_gap_mm": gap,
        "core_equal_three_leg_air_gap": 1,
        "full_model": 0,
        "round_corner": 0,
        "matrix_on": 1,
        "cap_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
        "keep_project": 1,
    }
    for key, expected in required.items():
        _exact(params.get(key), expected, f"params.{key}")
    ok, _validated = validation_check(
        create_input_parameter(params), strict=True
    )
    if not ok:
        raise tuner.GapTuningError("symmetric final params failed validation")

    profile = tuner._read_json(PROFILE_PATH)
    overrides = profile.get("param_overrides")
    boundary = profile.get("fixed_boundary_contract")
    retention = profile.get("artifact_retention")
    if (
        not isinstance(overrides, dict)
        or not isinstance(boundary, dict)
        or not isinstance(retention, dict)
        or profile.get("stage") != "standard"
        or profile.get("cli_flags") != "--thermal --headless"
        or retention.get("artifact_filename") != "symmetric.aedt"
        or retention.get("retention_required") is not True
    ):
        raise tuner.GapTuningError("standard symmetric profile drifted")
    for key, expected in {
        "full_model": 0,
        "round_corner": 0,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "k_ins": 0.2,
        "keep_project": 1,
    }.items():
        _exact(overrides.get(key), expected, f"profile.{key}")
    for key, expected in {
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "fan_velocity_m_s": 1.5,
        "fan_config": "dual",
    }.items():
        _exact(boundary.get(key), expected, f"boundary.{key}")
    return campaign, final, params, profile


def prepare(*, campaign_root: Path, solver_revision: str) -> Path:
    root = campaign_root.resolve(strict=True)
    solver = str(solver_revision).lower()
    if not tuner.HEX40.fullmatch(solver):
        raise tuner.GapTuningError(
            "full 40-character final solver revision required"
        )
    output = root / "final_symmetric_submission_plan.json"
    if output.exists():
        return output
    campaign, final, params, profile = _load_inputs(root)
    gap = float(final["tuned_core_center_gap_mm"])
    token = tuner._gap_token(gap)
    stem = EXPECTED_GEOMETRY_SHA256[:12]
    name = f"mft-final-sym-gap-{stem}-g{token}"
    workdir = f"mft_final_sym_gap_{stem}_g{token}"
    identity = scheduler_client.verification_submission_identity(
        name,
        params,
        profile,
        solver,
        campaign["library_revision"],
    )
    retained = scheduler_client.retained_aedt_identity(
        name,
        params,
        profile,
        solver,
        campaign["library_revision"],
    )
    value = tuner._seal(
        {
            "schema_version": PLAN_SCHEMA,
            "created_at_utc": tuner._now(),
            "campaign": tuner._file_record(
                root / "campaign_manifest.json", relative_to=root
            ),
            "campaign_payload_sha256": campaign["payload_sha256"],
            "tuned_manifest": tuner._file_record(
                root / "tuned_gap_manifest.json", relative_to=root
            ),
            "tuned_manifest_payload_sha256": final["payload_sha256"],
            "physical_geometry_sha256": EXPECTED_GEOMETRY_SHA256,
            "tuned_core_center_gap_mm": gap,
            "tuned_matrix_readback": copy.deepcopy(
                final["selected_observation"]["full_physical_matrix_readback"]
            ),
            "equal_three_leg_air_gap_contract": copy.deepcopy(
                final["equal_three_leg_air_gap_contract"]
            ),
            "equal_three_leg_air_gap_FEA_required": True,
            "physical_Lm_2mH_symmetric_FEA_verified": True,
            "final_promotion_allowed": False,
            "params": copy.deepcopy(
                final["downstream_params"]["symmetric_loss_thermal"]
            ),
            "params_sha256": tuner._sha(params),
            "profile": tuner._file_record(PROFILE_PATH),
            "profile_sha256": tuner._sha(profile),
            "model_contract": {
                "symmetric_eighth": True,
                "rounded_winding": False,
                "matrix": True,
                "capacitance": True,
                "loss": True,
                "thermal": True,
            },
            "fixed_boundary_contract": {
                "fan_velocity_m_s": 1.5,
                "thermal_pad_thickness_mm": 2.0,
                "thermal_pad_conductivity_W_mK": 0.2,
                "TIM_mutated": False,
            },
            "acceptance_gate": {
                "primary_winding_max_C": 100.0,
                "secondary_winding_max_C": 120.0,
                "core_max_C": 120.0,
                "resonance_min_Hz": 15_000.0,
            },
            "tuning_solver_revision": campaign["solver_revision"],
            "solver_revision": solver,
            "library_revision": campaign["library_revision"],
            "scheduler": {
                "project": scheduler_client.MFT_PROJECT,
                "name": name,
                "workdir": workdir,
                "dedupe_key": identity["dedupe_key"],
                "parameter_digest": identity["parameter_digest"],
                "effective_params_sha256": tuner._sha(identity["merged"]),
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "timeout_seconds": TIMEOUT_SECONDS,
                "max_workers_per_node": MAX_WORKERS_PER_NODE,
                "priority": PRIORITY,
                "aedt_backend": "standalone",
                "environment": tuner._core_environment(
                    solver
                ),
                "retained_aedt": retained,
            },
            "scheduler_repository_modified": False,
            "scheduler_project_configuration_modified": False,
            "scheduler_submission_performed": False,
        }
    )
    return tuner._atomic_json(output, value)


def submit(
    *,
    campaign_root: Path,
    scheduler_url: str,
    apply: bool,
    solver_revision: str,
) -> Path:
    root = campaign_root.resolve(strict=True)
    plan_path = prepare(
        campaign_root=root, solver_revision=solver_revision
    )
    plan = tuner._validate_seal(tuner._read_json(plan_path), PLAN_SCHEMA)
    output = root / "final_symmetric_submission_receipt.json"
    if output.exists():
        return output
    campaign, final, params, profile = _load_inputs(root)
    scheduler = plan["scheduler"]
    if not apply:
        value = tuner._seal(
            {
                "schema_version": RECEIPT_SCHEMA,
                "created_at_utc": tuner._now(),
                "plan_payload_sha256": plan["payload_sha256"],
                "physical_geometry_sha256": EXPECTED_GEOMETRY_SHA256,
                "tuned_core_center_gap_mm": final[
                    "tuned_core_center_gap_mm"
                ],
                "scheduler_POST_performed": False,
                "dry_run": True,
            }
        )
        return tuner._atomic_json(output, value)
    evidence = scheduler_client.submit_verification(
        scheduler["name"],
        scheduler["workdir"],
        params,
        profile,
        mem_mb=scheduler["memory_mb"],
        cpus=scheduler["cpus"],
        solver_revision=plan["solver_revision"],
        library_revision=campaign["library_revision"],
        priority=scheduler["priority"],
        max_workers_per_node=scheduler["max_workers_per_node"],
        aedt_backend="standalone",
        submission_env=scheduler["environment"],
        required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
        return_submission_evidence=True,
        max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
        scheduler_url=scheduler_url,
    )
    if not isinstance(evidence, dict):
        raise tuner.GapTuningError("final Scheduler evidence is absent")
    readback = (
        evidence.get("api_pre_submission_readback")
        or evidence.get("api_post_submission_response")
        or evidence
    )
    if isinstance(readback, dict) and isinstance(readback.get("task"), dict):
        readback = readback["task"]
    if not isinstance(readback, dict):
        raise tuner.GapTuningError("final Scheduler readback is absent")
    task_id = int(
        evidence.get(
            "task_id",
            evidence.get("id", readback.get("id", readback.get("task_id", 0))),
        )
    )
    if (
        task_id <= 0
        or readback.get("name") != scheduler["name"]
        or readback.get("dedupe_key") != scheduler["dedupe_key"]
        or readback.get("project") != scheduler_client.MFT_PROJECT
        or int(readback.get("cpus") or 0) != scheduler["cpus"]
        or int(readback.get("memory_mb") or 0) != scheduler["memory_mb"]
        or int(readback.get("timeout_seconds") or 0)
        != scheduler["timeout_seconds"]
        or int(readback.get("priority") or -1) != scheduler["priority"]
    ):
        raise tuner.GapTuningError("final Scheduler readback drifted")
    receipt = tuner._seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "created_at_utc": tuner._now(),
            "plan": tuner._file_record(plan_path, relative_to=root),
            "plan_payload_sha256": plan["payload_sha256"],
            "campaign_payload_sha256": campaign["payload_sha256"],
            "tuned_manifest_payload_sha256": final["payload_sha256"],
            "physical_geometry_sha256": EXPECTED_GEOMETRY_SHA256,
            "tuned_core_center_gap_mm": final["tuned_core_center_gap_mm"],
            "task_id": task_id,
            "name": scheduler["name"],
            "dedupe_key": scheduler["dedupe_key"],
            "readback": readback,
            "submission_evidence": evidence,
            "scheduler_POST_performed": True,
            "dry_run": False,
            "complete": True,
            "scheduler_repository_modified": False,
            "scheduler_project_configuration_modified": False,
        }
    )
    return tuner._atomic_json(output, receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.poll_seconds < 5.0 or args.poll_seconds > 60.0:
        raise tuner.GapTuningError("poll-seconds must be within 5..60")
    if args.wait:
        while not (
            args.campaign_root.resolve()
            / "tuned_gap_manifest.json"
        ).exists():
            time.sleep(args.poll_seconds)
    result = submit(
        campaign_root=args.campaign_root,
        scheduler_url=args.scheduler_url,
        apply=args.apply,
        solver_revision=args.solver_revision,
    )
    print(json.dumps(str(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
