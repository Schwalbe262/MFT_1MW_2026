"""Run the final tuned-gap, turn-graded, symmetric truth confirmation.

The pre-gap thermal acquisition, the physical-gap Matrix tuner, and the
turn-graded capacitance solves are selection evidence.  They cannot be spliced
together into a final winner.  This tool permits exactly one final 1/8,
non-rounded Matrix + actual-Rx-turn-graded-Cap + Loss + Thermal rerun after the
same candidate has passed both upstream gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping
import urllib.parse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import mft_goal_lm2mh_gap_tuner as gap_tuner  # noqa: E402
from tools import mft_goal_targeted_symmetric_fea_batch as targeted  # noqa: E402
from tools import mft_goal_turn_graded_cap_batch as graded_cap  # noqa: E402


PLAN_SCHEMA = "mft-goal-final-symmetric-truth-chain-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-final-symmetric-truth-chain-submission-v1"
COLLECTION_SCHEMA = "mft-goal-final-symmetric-truth-chain-collection-v1"
PROFILE_SCHEMA = "mft-goal-final-symmetric-truth-chain-profile-v1"
SCHEDULER_URL = "http://127.0.0.1:8002"
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
MAX_WORKERS_PER_NODE = 1
PRIORITY = 100
PROJECT_ACTIVE_TASK_CAP = 500
SIZE_LIMITS_MM = {"W": 1200.0, "L": 1000.0, "H": 750.0}
TERMINAL = {"completed", "failed", "cancelled", "timeout"}
HEX40 = re.compile(r"[0-9a-f]{40}")


class FinalTruthError(RuntimeError):
    """An upstream truth gate, scheduler identity, or result drifted."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalTruthError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FinalTruthError(f"JSON object required: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalTruthError(f"{label} is not finite") from exc
    if not math.isfinite(parsed):
        raise FinalTruthError(f"{label} is not finite")
    return parsed


def _actual_variant(active: str) -> dict[str, Any]:
    variant_id = graded_cap.ACTUAL_CONNECTION_VARIANT_IDS[active]
    matches = [
        dict(item)
        for item in graded_cap.VARIANTS
        if item["id"] == variant_id
    ]
    if len(matches) != 1:
        raise FinalTruthError(f"actual {active} graded variant is not unique")
    return matches[0]


def _validate_actual_cap_gate(
    *,
    tuned: Mapping[str, Any],
    campaign: Mapping[str, Any],
    cap_plan: Mapping[str, Any],
    cap_collection: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless actual Tx/Rx graded Cap matches gap and geometry."""
    actual = cap_collection.get("actual_connection_results") or {}
    source = cap_plan.get("source") or {}
    geometry = (campaign.get("candidate") or {}).get(
        "physical_geometry_sha256"
    )
    gap = _finite(tuned.get("tuned_core_center_gap_mm"), "tuned gap")
    if (
        cap_collection.get("plan_payload_sha256")
        != cap_plan.get("payload_sha256")
        or cap_collection.get("all_terminal") is not True
        or cap_collection.get("actual_connection_resonance_spec_pass")
        is not True
        or not {
            graded_cap.ACTUAL_CONNECTION_VARIANT_IDS["Tx"],
            graded_cap.ACTUAL_CONNECTION_VARIANT_IDS["Rx"],
        }.issubset(set(cap_plan.get("selected_variant_ids") or []))
        or source.get("physical_geometry_sha256") != geometry
        or not math.isclose(
            _finite(source.get("core_center_gap_mm"), "Cap source gap"),
            gap,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise FinalTruthError(
            "actual turn-graded Cap evidence does not match the tuned candidate"
        )
    for active in ("Tx", "Rx"):
        row = actual.get(active) or {}
        if (
            row.get("variant_id")
            != graded_cap.ACTUAL_CONNECTION_VARIANT_IDS[active]
            or row.get("contract_valid") is not True
            or row.get("resonance_spec_pass_15kHz") is not True
            or _finite(row.get("C_terminal_F"), f"{active} Cap") <= 0.0
            or _finite(row.get("f_res_Hz"), f"{active} resonance")
            < graded_cap.RESONANCE_MIN_HZ
        ):
            raise FinalTruthError(f"actual turn-graded {active} gate failed")
    return dict(actual)


def _source_batch(
    campaign: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    candidate = campaign.get("candidate") or {}
    authority = candidate.get("authority") or {}
    record = authority.get("batch_plan") or {}
    path = Path(str(record.get("path") or ""))
    if not path.is_absolute():
        raise FinalTruthError("source targeted batch path is not absolute")
    plan, root, profile = targeted._load_plan(path)  # noqa: SLF001
    if (
        authority.get("batch_plan_payload_sha256") != plan["payload_sha256"]
        or targeted._file_record(path) != record  # noqa: SLF001
    ):
        raise FinalTruthError("source targeted batch authority drifted")
    rank = int(candidate.get("source_batch_rank") or 0)
    lanes = [
        lane
        for lane in plan.get("lanes") or []
        if int(lane.get("rank") or 0) == rank
        and (lane.get("candidate") or {}).get("physical_geometry_sha256")
        == candidate.get("physical_geometry_sha256")
    ]
    if len(lanes) != 1:
        raise FinalTruthError("source targeted candidate is not unique")
    return plan, root, profile


def _load_upstream(
    *,
    gap_manifest_path: Path,
    cap_plan_path: Path,
    cap_collection_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    tuned_path = gap_manifest_path.resolve(strict=True)
    gap_root = tuned_path.parent
    tuned = gap_tuner._validate_seal(  # noqa: SLF001
        gap_tuner._read_json(tuned_path),  # noqa: SLF001
        gap_tuner.FINAL_SCHEMA,
    )
    campaign_path = gap_root / "campaign_manifest.json"
    campaign, observed_root = gap_tuner._load_campaign(  # noqa: SLF001
        campaign_path
    )
    candidate = campaign.get("candidate") or {}
    acquisition_temperatures = (
        candidate.get("authenticated_temperatures_C") or {}
    )
    gap = _finite(tuned.get("tuned_core_center_gap_mm"), "tuned gap")
    tuned_lm = _finite(
        tuned.get("tuned_Lm_primary_referred_H"), "tuned Lm"
    )
    if (
        observed_root != gap_root
        or tuned.get("campaign_payload_sha256")
        != campaign["payload_sha256"]
        or gap <= 0.0
        or tuned.get("physical_gap_geometry_attested") is not True
        or tuned.get("symmetric_matrix_convergence_attested") is not True
        or tuned.get("native_L11_L22_M_k_Lm_readback_attested") is not True
        or abs(tuned_lm - gap_tuner.TARGET_LM_H)
        > gap_tuner.LM_ABS_TOLERANCE_H
        or candidate.get("source_kind")
        != "authenticated_n1_6_targeted_symmetric_split_temperature_pass"
        or _finite(
            acquisition_temperatures.get("primary_winding_max_C"),
            "acquisition primary temperature",
        )
        > 100.0
        or _finite(
            acquisition_temperatures.get("secondary_winding_max_C"),
            "acquisition secondary temperature",
        )
        > 120.0
        or _finite(
            acquisition_temperatures.get("core_max_C"),
            "acquisition core temperature",
        )
        > 120.0
    ):
        raise FinalTruthError("physical-gap tuning is not a terminal PASS")

    cap_plan, _cap_root, _cap_profile = graded_cap._load_plan(  # noqa: SLF001
        cap_plan_path
    )
    cap_collection = gap_tuner._validate_seal(  # noqa: SLF001
        _read(cap_collection_path),
        graded_cap.COLLECTION_SCHEMA,
    )
    actual = _validate_actual_cap_gate(
        tuned=tuned,
        campaign=campaign,
        cap_plan=cap_plan,
        cap_collection=cap_collection,
    )
    return (
        tuned,
        gap_root,
        campaign,
        cap_plan,
        cap_collection,
        actual,
    )


def _final_profile(
    base_profile: Mapping[str, Any], rx_variant: Mapping[str, Any]
) -> dict[str, Any]:
    profile = copy.deepcopy(dict(base_profile))
    overrides = copy.deepcopy(dict(profile.get("param_overrides") or {}))
    overrides.update(
        {
            "full_model": 0,
            "round_corner": 0,
            "matrix_on": 1,
            "cap_on": 0,
            "loss_on": 1,
            "thermal_on": 1,
            "loss_sym_on": 1,
            "thermal_symmetry": "eighth",
            "keep_project": 1,
            "cap_turn_graded_active_winding": "Rx",
            "cap_turn_graded_voltage_policy": rx_variant["policy"],
            "cap_turn_graded_section_order": rx_variant["order"],
            "cap_turn_graded_reverse_sections": rx_variant["reverse"],
            "cap_turn_graded_reverse_terminal_polarity": rx_variant[
                "terminal_reverse"
            ],
            "cap_turn_graded_side_polarity": rx_variant["side_polarity"],
            "cap_turn_graded_side2_polarity": 1,
        }
    )
    profile.update(
        {
            "schema_version": PROFILE_SCHEMA,
            "comment": (
                "Final authority: tuned physical gap + actual Rx graded Cap "
                "+ symmetric Matrix/Loss/Thermal rerun"
            ),
            "param_overrides": overrides,
            "cpus": CPUS,
            "mem_mb": MEMORY_MB,
            "timeout_seconds": TIMEOUT_SECONDS,
        }
    )
    profile.pop("artifact_retention", None)
    return profile


def prepare(
    *,
    gap_manifest_path: Path,
    cap_plan_path: Path,
    cap_collection_path: Path,
    output: Path,
    priority: int = PRIORITY,
) -> Path:
    if (
        isinstance(priority, bool)
        or not isinstance(priority, int)
        or not 0 <= priority <= 100
    ):
        raise FinalTruthError("priority must be an integer in 0..100")
    tuned, gap_root, campaign, cap_plan, cap_collection, actual = (
        _load_upstream(
            gap_manifest_path=gap_manifest_path,
            cap_plan_path=cap_plan_path,
            cap_collection_path=cap_collection_path,
        )
    )
    source_plan, _source_root, base_profile = _source_batch(campaign)
    if (
        source_plan["solver_revision"] != cap_plan["solver_revision"]
        or source_plan["library_revision"] != cap_plan["library_revision"]
    ):
        raise FinalTruthError("solver/library revisions differ upstream")
    params_record = (tuned.get("downstream_params") or {}).get(
        "symmetric_loss_thermal"
    ) or {}
    params_path = (gap_root / str(params_record.get("path") or "")).resolve(
        strict=True
    )
    if gap_tuner._file_record(  # noqa: SLF001
        params_path, relative_to=gap_root
    ) != params_record:
        raise FinalTruthError("tuned downstream parameter bytes drifted")
    params = _read(params_path)
    rx_variant = _actual_variant("Rx")
    profile = _final_profile(base_profile, rx_variant)
    params.update(profile["param_overrides"])
    gap = _finite(tuned["tuned_core_center_gap_mm"], "tuned gap")
    params["core_center_gap_mm"] = gap
    if (
        _finite(params.get("fan_velocity"), "fan velocity") != 1.5
        or params.get("fan_config") != "dual"
        or _finite(params.get("core_plate_pad_t"), "core TIM") != 2.0
        or _finite(params.get("wcp_pad_t"), "winding TIM") != 2.0
        or _finite(params.get("k_ins"), "TIM conductivity") != 0.2
        or int(params.get("full_model")) != 0
        or int(params.get("round_corner")) != 0
        or params.get("thermal_symmetry") != "eighth"
        or params.get("cap_turn_graded_active_winding") != "Rx"
        or _finite(params.get("cw1"), "primary conductor thickness") != 5.0
        or _finite(params.get("gap1"), "primary interturn gap") != 1.6
        or int(params.get("N1_main") or 0)
        + int(params.get("N1_side") or 0)
        != 6
        or int(params.get("N2_main") or 0)
        + int(params.get("N2_side") or 0)
        != 60
    ):
        raise FinalTruthError("fixed final boundary/topology contract drifted")
    from module.input_parameter_260706 import (  # local: cheaper CLI import
        create_input_parameter,
        validation_check,
    )

    valid, _decoded = validation_check(
        create_input_parameter(params), strict=True
    )
    if not valid:
        raise FinalTruthError("final parameters failed validation")
    destination = output.resolve()
    if destination.exists():
        raise FinalTruthError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    profile_path = gap_tuner._atomic_json(  # noqa: SLF001
        destination / "profile.json", profile
    )
    final_params_path = gap_tuner._atomic_json(  # noqa: SLF001
        destination / "params.json", params
    )
    # Scheduler parameter digests are order-sensitive.  Replay the exact
    # persisted, canonically sorted JSON objects before deriving identity.
    profile = _read(profile_path)
    params = _read(final_params_path)
    geometry = campaign["candidate"]["physical_geometry_sha256"]
    name = f"mft-final-symtruth-{geometry[:12]}"
    identity = scheduler_client.verification_submission_identity(
        name,
        params,
        profile,
        source_plan["solver_revision"],
        source_plan["library_revision"],
    )
    plan = gap_tuner._seal(  # noqa: SLF001
        {
            "schema_version": PLAN_SCHEMA,
            "created_at_utc": gap_tuner._now(),  # noqa: SLF001
            "classification": (
                "final-symmetric-nonrounded-tuned-gap-actual-graded-truth"
            ),
            "source": {
                "gap_manifest": gap_tuner._file_record(gap_manifest_path),  # noqa: SLF001
                "gap_manifest_payload_sha256": tuned["payload_sha256"],
                "gap_campaign_payload_sha256": campaign["payload_sha256"],
                "cap_plan": gap_tuner._file_record(cap_plan_path),  # noqa: SLF001
                "cap_plan_payload_sha256": cap_plan["payload_sha256"],
                "cap_collection": gap_tuner._file_record(cap_collection_path),  # noqa: SLF001
                "cap_collection_payload_sha256": cap_collection[
                    "payload_sha256"
                ],
                "source_targeted_batch_payload_sha256": source_plan[
                    "payload_sha256"
                ],
            },
            "physical_geometry_sha256": geometry,
            "tuned_core_center_gap_mm": gap,
            "tuned_Lm_primary_referred_H": tuned[
                "tuned_Lm_primary_referred_H"
            ],
            "upstream_actual_connection_results": copy.deepcopy(actual),
            "actual_graded_variant_in_chain": copy.deepcopy(rx_variant),
            "profile": gap_tuner._file_record(  # noqa: SLF001
                profile_path, relative_to=destination
            ),
            "profile_sha256": gap_tuner._sha(profile),  # noqa: SLF001
            "params": gap_tuner._file_record(  # noqa: SLF001
                final_params_path, relative_to=destination
            ),
            "params_sha256": gap_tuner._sha(params),  # noqa: SLF001
            "solver_revision": source_plan["solver_revision"],
            "library_revision": source_plan["library_revision"],
            "scheduler_priority": priority,
            "scheduler": {
                "project": scheduler_client.MFT_PROJECT,
                "name": name,
                "workdir": f"mft_final_symtruth_{geometry[:12]}",
                "dedupe_key": identity["dedupe_key"],
                "parameter_digest": identity["parameter_digest"],
                "effective_params_sha256": gap_tuner._sha(  # noqa: SLF001
                    identity["merged"]
                ),
                "cpus": CPUS,
                "memory_mb": MEMORY_MB,
                "timeout_seconds": TIMEOUT_SECONDS,
                "max_workers_per_node": MAX_WORKERS_PER_NODE,
                "priority": priority,
                "environment": gap_tuner._core_environment(  # noqa: SLF001
                    source_plan["solver_revision"]
                ),
            },
            "fixed_boundary_contract": copy.deepcopy(
                profile.get("fixed_boundary_contract")
            ),
            "rounded_FEA_used": False,
            "symmetric_model": True,
            "single_final_authority_task": True,
            "scheduler_submission_performed": False,
            "final_design_pass": False,
        }
    )
    return gap_tuner._atomic_json(  # noqa: SLF001
        destination / "plan.json", plan
    )


def _load_plan(
    path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    plan_path = path.resolve(strict=True)
    root = plan_path.parent
    plan = gap_tuner._validate_seal(  # noqa: SLF001
        _read(plan_path), PLAN_SCHEMA
    )
    profile_path = (root / plan["profile"]["path"]).resolve(strict=True)
    params_path = (root / plan["params"]["path"]).resolve(strict=True)
    profile = _read(profile_path)
    params = _read(params_path)
    scheduler = plan.get("scheduler") or {}
    priority = plan.get("scheduler_priority")
    if (
        plan.get("rounded_FEA_used") is not False
        or plan.get("symmetric_model") is not True
        or plan.get("single_final_authority_task") is not True
        or plan.get("final_design_pass") is not False
        or not HEX40.fullmatch(str(plan.get("solver_revision") or ""))
        or not HEX40.fullmatch(str(plan.get("library_revision") or ""))
        or profile.get("schema_version") != PROFILE_SCHEMA
        or gap_tuner._file_record(  # noqa: SLF001
            profile_path, relative_to=root
        )
        != plan["profile"]
        or gap_tuner._sha(profile) != plan["profile_sha256"]  # noqa: SLF001
        or gap_tuner._file_record(  # noqa: SLF001
            params_path, relative_to=root
        )
        != plan["params"]
        or gap_tuner._sha(params) != plan["params_sha256"]  # noqa: SLF001
        or isinstance(priority, bool)
        or not isinstance(priority, int)
        or not 0 <= priority <= 100
    ):
        raise FinalTruthError("final truth plan contract drifted")
    identity = scheduler_client.verification_submission_identity(
        scheduler.get("name"),
        params,
        profile,
        plan["solver_revision"],
        plan["library_revision"],
    )
    if (
        scheduler.get("project") != scheduler_client.MFT_PROJECT
        or scheduler.get("dedupe_key") != identity["dedupe_key"]
        or scheduler.get("parameter_digest") != identity["parameter_digest"]
        or scheduler.get("effective_params_sha256")
        != gap_tuner._sha(identity["merged"])  # noqa: SLF001
        or scheduler.get("cpus") != CPUS
        or scheduler.get("memory_mb") != MEMORY_MB
        or scheduler.get("timeout_seconds") != TIMEOUT_SECONDS
        or scheduler.get("max_workers_per_node") != MAX_WORKERS_PER_NODE
        or scheduler.get("priority") != priority
        or scheduler.get("environment")
        != gap_tuner._core_environment(plan["solver_revision"])  # noqa: SLF001
    ):
        raise FinalTruthError("final scheduler identity drifted")
    return plan, root, profile, params


def submit(
    *,
    plan_path: Path,
    output: Path,
    scheduler_url: str,
    apply: bool,
) -> Path:
    plan, _root, profile, params = _load_plan(plan_path)
    scheduler = plan["scheduler"]
    if not apply:
        return gap_tuner._atomic_json(  # noqa: SLF001
            output,
            gap_tuner._seal(  # noqa: SLF001
                {
                    "schema_version": SUBMISSION_SCHEMA,
                    "created_at_utc": gap_tuner._now(),  # noqa: SLF001
                    "classification": "dry-run",
                    "plan_payload_sha256": plan["payload_sha256"],
                    "scheduler_POST_performed": False,
                    "ready_for_explicit_apply": True,
                }
            ),
        )
    evidence = scheduler_client.submit_verification(
        scheduler["name"],
        scheduler["workdir"],
        params,
        profile,
        mem_mb=MEMORY_MB,
        cpus=CPUS,
        solver_revision=plan["solver_revision"],
        library_revision=plan["library_revision"],
        priority=scheduler["priority"],
        max_workers_per_node=MAX_WORKERS_PER_NODE,
        aedt_backend="standalone",
        submission_env=scheduler["environment"],
        required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
        max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
        return_submission_evidence=True,
        scheduler_url=scheduler_url,
    )
    if not isinstance(evidence, dict):
        raise FinalTruthError("scheduler submission evidence is absent")
    readback = (
        evidence.get("api_pre_submission_readback")
        or evidence.get("api_post_submission_response")
        or evidence
    )
    if isinstance(readback, dict) and isinstance(readback.get("task"), dict):
        readback = readback["task"]
    task_id = int(
        evidence.get("task_id")
        or (readback or {}).get("id")
        or (readback or {}).get("task_id")
        or 0
    )
    if (
        not isinstance(readback, dict)
        or task_id <= 0
        or readback.get("name") != scheduler["name"]
        or readback.get("dedupe_key") != scheduler["dedupe_key"]
        or int(readback.get("priority") or -1) != scheduler["priority"]
    ):
        raise FinalTruthError("scheduler submission readback drifted")
    return gap_tuner._atomic_json(  # noqa: SLF001
        output,
        gap_tuner._seal(  # noqa: SLF001
            {
                "schema_version": SUBMISSION_SCHEMA,
                "created_at_utc": gap_tuner._now(),  # noqa: SLF001
                "plan": gap_tuner._file_record(plan_path),  # noqa: SLF001
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_url": scheduler_url.rstrip("/"),
                "task_id": task_id,
                "readback": readback,
                "submission_evidence": evidence,
                "complete": True,
                "scheduler_POST_performed": True,
            }
        ),
    )


def _task(base: str, task_id: int) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/api/tasks/{task_id}"
    task = targeted._api_json(url)  # noqa: SLF001
    if not isinstance(task, dict):
        raise FinalTruthError("scheduler task readback is absent")
    if (
        str(task.get("status") or task.get("state") or "").lower()
        in TERMINAL
        and targeted._result_json(task) is None  # noqa: SLF001
    ):
        query = urllib.parse.urlencode(
            {"tail_lines": 24, "max_bytes": 5_000_000}
        )
        stdout = targeted._api_json(  # noqa: SLF001
            f"{base.rstrip('/')}/api/tasks/{task_id}/stdout?{query}"
        )
        task = dict(task)
        task["stdout"] = stdout.get("stdout", "")
    return task


def _temperature_group(
    result: Mapping[str, Any], names: tuple[str, ...], *, secondary: bool
) -> dict[str, float]:
    values = {}
    for name in names:
        if secondary and name in {
            "T_max_Rx_side",
            "Tprobe_Rx_side_leeward_max",
        } and int(result.get("N2_side") or 0) <= 0:
            continue
        values[name] = _finite(result.get(name), f"result {name}")
    return values


def collect(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str | None = None,
) -> Path:
    plan, _root, profile, params = _load_plan(plan_path)
    submission = gap_tuner._validate_seal(  # noqa: SLF001
        _read(submission_path), SUBMISSION_SCHEMA
    )
    if (
        submission.get("complete") is not True
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
        or int(submission.get("task_id") or 0) <= 0
    ):
        raise FinalTruthError("final submission contract drifted")
    base = str(
        scheduler_url
        or submission.get("scheduler_url")
        or SCHEDULER_URL
    ).rstrip("/")
    task = _task(base, int(submission["task_id"]))
    status = str(task.get("status") or task.get("state") or "").lower()
    result = targeted._result_json(task)  # noqa: SLF001
    row: dict[str, Any] = {
        "task_id": int(submission["task_id"]),
        "status": status,
        "terminal": status in TERMINAL,
        "result_available": result is not None,
        "final_design_pass": False,
    }
    if result is not None:
        reasons: list[str] = []

        def require(condition: bool, reason: str) -> None:
            if not condition:
                reasons.append(reason)

        strict = scheduler_client.is_valid_result(
            dict(result),
            expected_revision=plan["solver_revision"],
            expected_library_revision=plan["library_revision"],
            expected_profile=dict(profile["param_overrides"]),
        )
        exact = scheduler_client.result_matches_params(
            dict(result),
            scheduler_client.effective_verification_params(params, profile),
        )
        require(strict, "strict_solver_result_contract_invalid")
        require(exact, "effective_params_echo_mismatch")
        for name, expected in (
            ("full_model", 0),
            ("round_corner", 0),
            ("matrix_on", 1),
            ("cap_on", 0),
            ("loss_on", 1),
            ("thermal_on", 1),
            ("loss_sym_on", 1),
            ("thermal_symmetry", "eighth"),
            ("cap_turn_graded_active_winding", "Rx"),
            ("fan_config", "dual"),
        ):
            require(result.get(name) == expected, f"identity_mismatch:{name}")
        for name, expected in (
            ("fan_velocity", 1.5),
            ("core_plate_pad_t", 2.0),
            ("wcp_pad_t", 2.0),
            ("k_ins", 0.2),
        ):
            require(
                math.isclose(
                    _finite(result.get(name), f"result {name}"),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"fixed_boundary_mismatch:{name}",
            )
        gap = _finite(plan["tuned_core_center_gap_mm"], "plan tuned gap")
        require(
            math.isclose(
                _finite(result.get("core_center_gap_mm"), "input gap"),
                gap,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "physical_gap_input_mismatch",
        )
        require(
            int(_finite(
                result.get("core_center_gap_geometry_attested"),
                "gap geometry attestation",
            ))
            == 1,
            "physical_gap_geometry_not_attested",
        )
        require(
            int(_finite(
                result.get("core_center_gap_symmetry_geometry_attested"),
                "gap symmetry attestation",
            ))
            == 1,
            "physical_gap_symmetry_not_attested",
        )
        require(
            result.get("core_center_gap_topology")
            == "center_leg_bottom_top_physical_air_interval",
            "physical_gap_topology_mismatch",
        )
        require(
            math.isclose(
                _finite(
                    result.get("core_center_gap_readback_mm"),
                    "gap readback",
                ),
                gap,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "physical_gap_readback_mismatch",
        )

        l11 = _finite(result.get("Ltx"), "native L11")
        l22 = _finite(result.get("Lrx"), "native L22")
        mutual = _finite(result.get("M"), "native mutual")
        coupling = _finite(result.get("k"), "native coupling")
        native_lm = _finite(result.get("Lmt"), "native Lm")
        physical_lm_h = 2.0 * native_lm * 1e-6
        require(min(l11, l22, mutual, native_lm) > 0.0, "nonpositive_matrix")
        require(0.0 < coupling <= 1.0, "invalid_coupling")
        require(
            math.isclose(
                native_lm,
                l11 * coupling * coupling,
                rel_tol=1e-8,
                abs_tol=1e-6,
            ),
            "Lm_formula_mismatch",
        )
        require(
            abs(physical_lm_h - gap_tuner.TARGET_LM_H)
            <= gap_tuner.LM_ABS_TOLERANCE_H,
            "physical_Lm_not_within_2mH_tolerance",
        )
        matrix_tol = _finite(
            result.get("matrix_percent_error"), "matrix tolerance"
        )
        require(
            _finite(result.get("conv_passes_matrix"), "matrix passes") >= 1
            and _finite(
                result.get("conv_consecutive_matrix"),
                "matrix consecutive",
            )
            >= _finite(
                result.get("matrix_min_converged"), "matrix min converged"
            )
            and _finite(
                result.get("conv_error_pct_matrix"), "matrix energy error"
            )
            <= matrix_tol
            and _finite(
                result.get("conv_delta_pct_matrix"),
                "matrix delta energy",
            )
            <= matrix_tol,
            "matrix_not_converged",
        )

        rx_cap = _finite(
            result.get("C_rx_rx_turn_graded_F"),
            "actual graded Rx capacitance",
        )
        rx_frequency = _finite(
            result.get("f_res_rx_turn_graded_Hz"),
            "actual graded Rx resonance",
        )
        source_actual = plan["upstream_actual_connection_results"]
        source_rx_cap = _finite(
            source_actual["Rx"]["C_terminal_F"], "source Rx capacitance"
        )
        tx_frequency = _finite(
            source_actual["Tx"]["f_res_Hz"], "source Tx resonance"
        )
        require(rx_cap > 0.0, "nonpositive_actual_graded_Rx_capacitance")
        require(
            math.isclose(rx_cap, source_rx_cap, rel_tol=0.05, abs_tol=0.0),
            "actual_graded_Rx_capacitance_not_reproduced_within_5pct",
        )
        final_fmin = min(tx_frequency, rx_frequency)
        require(
            final_fmin >= graded_cap.RESONANCE_MIN_HZ,
            "actual_turn_graded_resonance_below_15kHz",
        )

        primary = _temperature_group(
            result,
            targeted.PRIMARY_WINDING_TEMPERATURES,
            secondary=False,
        )
        secondary = _temperature_group(
            result,
            targeted.SECONDARY_WINDING_TEMPERATURES,
            secondary=True,
        )
        core = _temperature_group(
            result,
            targeted.CORE_TEMPERATURES,
            secondary=False,
        )
        primary_max = max(primary.values())
        secondary_max = max(secondary.values())
        core_max = max(core.values())
        require(primary_max <= 100.0, "primary_winding_above_100C")
        require(secondary_max <= 120.0, "secondary_winding_above_120C")
        require(core_max <= 120.0, "core_above_120C")
        require(
            _finite(result.get("P_winding_total"), "winding loss") > 0.0
            and _finite(result.get("P_core_total"), "core loss") > 0.0,
            "loss_stage_not_attested",
        )
        require(
            _finite(result.get("thermal_iterations"), "thermal iterations")
            > 0.0,
            "thermal_stage_not_attested",
        )

        volume, dimensions_raw = geometry_metrics.bounding_box_lit(
            scheduler_client.effective_verification_params(params, profile)
        )
        dimensions = {
            "W_drawing_x": float(dimensions_raw[0]),
            "L_perpendicular_y": float(dimensions_raw[1]),
            "H": float(dimensions_raw[2]),
        }
        require(
            dimensions["W_drawing_x"] <= SIZE_LIMITS_MM["W"] + 1e-9
            and dimensions["L_perpendicular_y"]
            <= SIZE_LIMITS_MM["L"] + 1e-9
            and dimensions["H"] <= SIZE_LIMITS_MM["H"] + 1e-9,
            "size_envelope_failed",
        )
        final_pass = not reasons and status == "completed"
        row.update(
            {
                "result_sha256": gap_tuner._sha(result),  # noqa: SLF001
                "strict_solver_result_valid": strict,
                "exact_effective_params_echo_valid": exact,
                "physical_gap_geometry_attested": (
                    "physical_gap_geometry_not_attested" not in reasons
                    and "physical_gap_symmetry_not_attested" not in reasons
                    and "physical_gap_topology_mismatch" not in reasons
                    and "physical_gap_readback_mismatch" not in reasons
                ),
                "Lm_primary_referred_H": physical_lm_h,
                "Lm_primary_referred_mH": physical_lm_h * 1e3,
                "actual_graded_Rx_C_F": rx_cap,
                "actual_graded_Rx_f_Hz": rx_frequency,
                "authenticated_actual_graded_Tx_f_Hz": tx_frequency,
                "actual_turn_graded_fmin_Hz": final_fmin,
                "temperatures_C": {
                    "primary_winding": primary,
                    "secondary_winding": secondary,
                    "core": core,
                },
                "measured_primary_winding_max_C": primary_max,
                "measured_secondary_winding_max_C": secondary_max,
                "measured_core_max_C": core_max,
                "dimensions_mm": dimensions,
                "volume_L": float(volume),
                "reasons": reasons,
                "contract_valid": not reasons,
                "final_design_pass": final_pass,
            }
        )
    value = gap_tuner._seal(  # noqa: SLF001
        {
            "schema_version": COLLECTION_SCHEMA,
            "created_at_utc": gap_tuner._now(),  # noqa: SLF001
            "plan": gap_tuner._file_record(plan_path),  # noqa: SLF001
            "plan_payload_sha256": plan["payload_sha256"],
            "submission": gap_tuner._file_record(submission_path),  # noqa: SLF001
            "submission_payload_sha256": submission["payload_sha256"],
            "row": row,
            "all_terminal": row["terminal"],
            "final_design_pass": row["final_design_pass"],
            "authority": (
                "single same-candidate same-tuned-gap actual-graded "
                "symmetric Matrix+Cap+Loss+Thermal result"
            ),
            "ungapped_acquisition_used_as_final_authority": False,
            "rounded_FEA_used": False,
            "production_eligible": row["final_design_pass"],
        }
    )
    return gap_tuner._atomic_json(  # noqa: SLF001
        output, value, replace=output.exists()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--gap-manifest", type=Path, required=True)
    prepare_parser.add_argument("--cap-plan", type=Path, required=True)
    prepare_parser.add_argument("--cap-collection", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--priority", type=int, default=PRIORITY)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--output", type=Path, required=True)
    submit_parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    submit_parser.add_argument("--apply", action="store_true")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--submission", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--scheduler-url", default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            gap_manifest_path=args.gap_manifest,
            cap_plan_path=args.cap_plan,
            cap_collection_path=args.cap_collection,
            output=args.output,
            priority=args.priority,
        )
    elif args.command == "submit":
        result = submit(
            plan_path=args.plan,
            output=args.output,
            scheduler_url=args.scheduler_url,
            apply=args.apply,
        )
    else:
        result = collect(
            plan_path=args.plan,
            submission_path=args.submission,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
    print(json.dumps({"status": "ok", "path": str(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
